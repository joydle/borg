//! Fallback ladders — try rungs in order until one ships.
//!
//! Two surfaces over one rule set:
//!
//! - [`Ladder`]: a typed ladder of plain async work (`Result<T, E>`), for
//!   provider calls that are not agents — a model and its fallbacks, a
//!   primary API and a mirror of it. Errors stay the app's own type `E`.
//! - [`FallbackLadder`] (DSL: [`crate::dsl::fallback_ladder`]): the same
//!   ladder over [`Composition`] rungs, where errors are
//!   [`crate::BorgError`].
//!
//! The rules, for both:
//!
//! 1. **Fall through, or stop.** A rung's error falls through to the next
//!    rung only when its predicate says so (the rung's own, else the
//!    ladder's; a typed ladder falls through on every error by default, a
//!    composition ladder on [`BorgError::is_retryable`]). Any other error
//!    *stops* the ladder and is handed back as-is — a content-policy
//!    refusal, a user Stop, a request no rung can take — never wrapped in
//!    "all fallbacks exhausted".
//! 2. **Rungs choose themselves.** A rung with a `when` predicate runs
//!    only if it returns true, given the failures so far (and, for a
//!    composition, the handoff): skip a rung that cannot take this
//!    request, or that the previous failure says is down too.
//! 3. **Nothing is lost.** Every failure is kept, in order, with the rung
//!    that produced it: on [`Shipped::failures`] when a later rung ships,
//!    on [`LadderError`] / [`BorgError::FallbackExhausted`] when none does.
//!    [`LadderError::merge`] folds them into one error (summing cost,
//!    keeping the first id, …).
//! 4. **Observe each fall.** `on_fall` sees every failure that falls
//!    through, with the name of the rung it falls to — for a metric or a
//!    log line, or (composition ladders) to record a per-rung outcome on
//!    the handoff.
//!
//! ```
//! use borg::ladder::{Ladder, LadderError, Rung};
//!
//! #[derive(Debug, PartialEq)]
//! enum Fail { Down { cost: u32 }, Refused }
//!
//! # tokio_test_block(async {
//! let shipped = Ladder::new()
//!     .rung("primary", async { Err::<&str, _>(Fail::Down { cost: 3 }) })
//!     .rung("backup", async { Ok("answer") })
//!     .falls_through(|e: &Fail| matches!(e, Fail::Down { .. }))
//!     .run_shipped()
//!     .await
//!     .unwrap();
//! assert_eq!(shipped.value, "answer");
//! assert_eq!(&*shipped.rung, "backup");
//! assert_eq!(shipped.failures.len(), 1);
//!
//! // A refusal stops the ladder and comes back as-is.
//! let err = Ladder::new()
//!     .rung("primary", async { Err::<(), _>(Fail::Refused) })
//!     .rung("backup", async { Ok(()) })
//!     .falls_through(|e: &Fail| matches!(e, Fail::Down { .. }))
//!     .run()
//!     .await
//!     .unwrap_err();
//! assert!(matches!(err, LadderError::Stopped { .. }));
//! # });
//! # fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
//! # }
//! ```

use crate::composition::Composition;
use crate::error::BorgError;
use crate::handoff::HandoffState;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// One rung's failure, kept in the order the rungs ran.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RungFailure<E> {
    /// The rung's name (a composition rung: its [`Composition::label`]).
    pub rung: Arc<str>,
    /// The rung's position in the ladder (0 = the first rung), counting
    /// rungs that were skipped.
    pub index: usize,
    /// What the rung returned.
    pub error: E,
}

/// A successful run: the value, the rung that shipped it, and every
/// failure before it.
#[derive(Debug)]
#[non_exhaustive]
pub struct Shipped<T, E> {
    /// What the shipping rung returned.
    pub value: T,
    /// The shipping rung's name.
    pub rung: Arc<str>,
    /// The shipping rung's position (0 = the first rung).
    pub index: usize,
    /// The rungs that failed and fell through before it, in order.
    pub failures: Vec<RungFailure<E>>,
}

impl<T, E> Shipped<T, E> {
    /// The value alone.
    pub fn into_value(self) -> T {
        self.value
    }

    /// Whether the first rung shipped (no fall happened).
    pub fn first_try(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Why a [`Ladder`] shipped nothing.
#[derive(Debug)]
#[non_exhaustive]
pub enum LadderError<E> {
    /// A rung's error did not fall through: it is handed back as-is
    /// (`failure.error`), with the failures that fell through before it.
    Stopped {
        failure: RungFailure<E>,
        earlier: Vec<RungFailure<E>>,
    },
    /// Every rung that applied fell through, or no rung applied at all
    /// (`failures` empty, or holding only the rungs that ran).
    Exhausted { failures: Vec<RungFailure<E>> },
    /// The ladder's cancel token fired between rungs.
    Cancelled { failures: Vec<RungFailure<E>> },
}

impl<E> LadderError<E> {
    /// Every failure, in the order the rungs ran (a stopping one last).
    pub fn failures(&self) -> impl Iterator<Item = &RungFailure<E>> {
        let (earlier, last): (&[RungFailure<E>], Option<&RungFailure<E>>) = match self {
            LadderError::Stopped { failure, earlier } => (earlier, Some(failure)),
            LadderError::Exhausted { failures } | LadderError::Cancelled { failures } => {
                (failures, None)
            }
        };
        earlier.iter().chain(last)
    }

    /// The last failure (the stopping one, or the last rung that ran).
    pub fn last(&self) -> Option<&RungFailure<E>> {
        self.failures().last()
    }

    /// Every failure, owned, in order.
    pub fn into_failures(self) -> Vec<RungFailure<E>> {
        match self {
            LadderError::Stopped {
                failure,
                mut earlier,
            } => {
                earlier.push(failure);
                earlier
            }
            LadderError::Exhausted { failures } | LadderError::Cancelled { failures } => failures,
        }
    }

    /// Fold every failure into the LAST one: `fold(later, earlier)` runs
    /// from the last failure back to the first, so the app decides what
    /// an earlier rung adds (cost summed, the first request id kept, …).
    /// `None` when no rung ran.
    pub fn merge(self, mut fold: impl FnMut(E, E) -> E) -> Option<E> {
        let mut failures = self.into_failures();
        let mut acc = failures.pop()?.error;
        while let Some(earlier) = failures.pop() {
            acc = fold(acc, earlier.error);
        }
        Some(acc)
    }

    /// True for [`LadderError::Stopped`].
    pub fn is_stopped(&self) -> bool {
        matches!(self, LadderError::Stopped { .. })
    }
}

impl<E: fmt::Display> fmt::Display for LadderError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LadderError::Stopped { failure, .. } => {
                write!(f, "rung `{}` failed: {}", failure.rung, failure.error)
            }
            LadderError::Exhausted { failures } if failures.is_empty() => {
                write!(f, "no fallback rung applied")
            }
            LadderError::Exhausted { failures } => {
                write!(f, "every fallback rung failed")?;
                write_failures(f, failures)
            }
            LadderError::Cancelled { .. } => write!(f, "fallback ladder cancelled"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for LadderError<E> {}

pub(crate) fn write_failures<E: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    failures: &[RungFailure<E>],
) -> fmt::Result {
    for (i, r) in failures.iter().enumerate() {
        let sep = if i == 0 { ": " } else { "; " };
        write!(f, "{sep}`{}`: {}", r.rung, r.error)?;
    }
    Ok(())
}

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type ErrPred<'a, E> = Box<dyn Fn(&E) -> bool + Send + Sync + 'a>;
type WhenPred<'a, E> = Box<dyn Fn(&[RungFailure<E>]) -> bool + Send + Sync + 'a>;
type FallHook<'a, E> = Box<dyn FnMut(&RungFailure<E>, Option<&str>) + Send + 'a>;

/// One rung of a typed [`Ladder`]: a name, the work, and optional
/// `when` / `falls_through` predicates of its own.
#[must_use]
pub struct Rung<'a, T, E> {
    name: Arc<str>,
    work: BoxFut<'a, Result<T, E>>,
    when: Option<WhenPred<'a, E>>,
    falls_through: Option<ErrPred<'a, E>>,
}

impl<'a, T, E> Rung<'a, T, E> {
    /// A rung named `name` running `work`. The future is only polled if
    /// the rung runs; a skipped rung's future is dropped unpolled.
    pub fn new(
        name: impl Into<Arc<str>>,
        work: impl Future<Output = Result<T, E>> + Send + 'a,
    ) -> Self {
        Self {
            name: name.into(),
            work: Box::pin(work),
            when: None,
            falls_through: None,
        }
    }

    /// Run this rung only when `pred(failures so far)` is true (the first
    /// rung sees an empty slice). A rung without `when` always applies.
    pub fn when(mut self, pred: impl Fn(&[RungFailure<E>]) -> bool + Send + Sync + 'a) -> Self {
        self.when = Some(Box::new(pred));
        self
    }

    /// This rung's own fall-through rule, in place of the ladder's.
    pub fn falls_through(mut self, pred: impl Fn(&E) -> bool + Send + Sync + 'a) -> Self {
        self.falls_through = Some(Box::new(pred));
        self
    }
}

/// A typed fallback ladder over async work. Build it per call (the rungs
/// borrow for `'a`), then [`Ladder::run`] or [`Ladder::run_shipped`]. See
/// the [module docs](self) for the rules.
#[must_use = "a ladder does nothing until `run` is awaited"]
pub struct Ladder<'a, T, E> {
    rungs: Vec<Rung<'a, T, E>>,
    falls_through: Option<ErrPred<'a, E>>,
    on_fall: Option<FallHook<'a, E>>,
    cancel: Option<CancellationToken>,
}

impl<T, E> Default for Ladder<'_, T, E> {
    fn default() -> Self {
        Self {
            rungs: Vec::new(),
            falls_through: None,
            on_fall: None,
            cancel: None,
        }
    }
}

impl<'a, T, E> Ladder<'a, T, E> {
    /// An empty ladder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rung named `name` running `work` (no predicates of its own).
    pub fn rung(
        self,
        name: impl Into<Arc<str>>,
        work: impl Future<Output = Result<T, E>> + Send + 'a,
    ) -> Self {
        self.push(Rung::new(name, work))
    }

    /// Add a configured [`Rung`].
    pub fn push(mut self, rung: Rung<'a, T, E>) -> Self {
        self.rungs.push(rung);
        self
    }

    /// Add several rungs.
    pub fn rungs(mut self, rungs: impl IntoIterator<Item = Rung<'a, T, E>>) -> Self {
        self.rungs.extend(rungs);
        self
    }

    /// The ladder's fall-through rule, for rungs without their own.
    /// Default: every error falls through.
    pub fn falls_through(mut self, pred: impl Fn(&E) -> bool + Send + Sync + 'a) -> Self {
        self.falls_through = Some(Box::new(pred));
        self
    }

    /// Observe each failure that falls through, with the name of the rung
    /// it falls to (`None`: no rung left applies).
    pub fn on_fall(mut self, hook: impl FnMut(&RungFailure<E>, Option<&str>) + Send + 'a) -> Self {
        self.on_fall = Some(Box::new(hook));
        self
    }

    /// Stop between rungs once `token` fires ([`LadderError::Cancelled`]).
    /// A running rung is the rung's own business (pass it the token).
    pub fn cancel_on(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Number of rungs.
    pub fn len(&self) -> usize {
        self.rungs.len()
    }

    /// True with no rungs.
    pub fn is_empty(&self) -> bool {
        self.rungs.is_empty()
    }

    /// Run the ladder; the shipping rung's value.
    pub async fn run(self) -> Result<T, LadderError<E>> {
        self.run_shipped().await.map(Shipped::into_value)
    }

    /// Run the ladder; the value with the rung that shipped it and every
    /// failure before it.
    pub async fn run_shipped(self) -> Result<Shipped<T, E>, LadderError<E>> {
        let Ladder {
            rungs,
            falls_through,
            mut on_fall,
            cancel,
        } = self;
        let mut failures: Vec<RungFailure<E>> = Vec::new();
        let mut rungs: Vec<Option<Rung<'a, T, E>>> = rungs.into_iter().map(Some).collect();
        let mut next = next_applicable(&rungs, 0, &failures);
        while let Some(index) = next {
            if cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(LadderError::Cancelled { failures });
            }
            let rung = rungs[index].take().expect("each rung runs at most once");
            match rung.work.await {
                Ok(value) => {
                    return Ok(Shipped {
                        value,
                        rung: rung.name,
                        index,
                        failures,
                    })
                }
                Err(error) => {
                    let falls = match (&rung.falls_through, &falls_through) {
                        (Some(own), _) => own(&error),
                        (None, Some(ladder)) => ladder(&error),
                        (None, None) => true,
                    };
                    let failure = RungFailure {
                        rung: rung.name,
                        index,
                        error,
                    };
                    if !falls {
                        return Err(LadderError::Stopped {
                            failure,
                            earlier: failures,
                        });
                    }
                    failures.push(failure);
                    next = next_applicable(&rungs, index + 1, &failures);
                    if let Some(hook) = on_fall.as_mut() {
                        let to = next.and_then(|i| rungs[i].as_ref()).map(|r| &*r.name);
                        hook(failures.last().expect("just pushed"), to);
                    }
                }
            }
        }
        Err(LadderError::Exhausted { failures })
    }
}

fn next_applicable<T, E>(
    rungs: &[Option<Rung<'_, T, E>>],
    from: usize,
    failures: &[RungFailure<E>],
) -> Option<usize> {
    (from..rungs.len()).find(|&i| {
        rungs[i]
            .as_ref()
            .is_some_and(|r| r.when.as_ref().is_none_or(|w| w(failures)))
    })
}

// ── Composition ladders ───────────────────────────────────────────────

/// Fall-through rule of a composition ladder.
pub type FallsThroughFn = Arc<dyn Fn(&BorgError) -> bool + Send + Sync>;
/// `when` rule of a composition rung: the handoff as the ladder reached
/// the rung, and the failures so far.
pub type RungWhenFn = Arc<dyn Fn(&HandoffState, &[RungFailure<BorgError>]) -> bool + Send + Sync>;
/// `on_fall` hook of a composition ladder: the failure that fell through,
/// the next rung's label (`None`: none applies), and the handoff to record
/// a per-rung outcome on.
pub type OnFallFn =
    Arc<dyn Fn(&RungFailure<BorgError>, Option<&str>, &mut HandoffState) + Send + Sync>;

/// One rung of a composition [`FallbackLadder`]: a body and optional
/// `when` / `falls_through` predicates.
#[must_use]
pub struct LadderRung {
    pub(crate) body: Composition,
    pub(crate) when: Option<RungWhenFn>,
    pub(crate) falls_through: Option<FallsThroughFn>,
}

impl LadderRung {
    /// A rung running `body`.
    pub fn new(body: impl Into<Composition>) -> Self {
        Self {
            body: body.into(),
            when: None,
            falls_through: None,
        }
    }

    /// Run this rung only when `pred(handoff, failures so far)` is true.
    pub fn when(
        mut self,
        pred: impl Fn(&HandoffState, &[RungFailure<BorgError>]) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.when = Some(Arc::new(pred));
        self
    }

    /// This rung's own fall-through rule, in place of the ladder's.
    pub fn falls_through(
        mut self,
        pred: impl Fn(&BorgError) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.falls_through = Some(Arc::new(pred));
        self
    }
}

impl From<Composition> for LadderRung {
    fn from(body: Composition) -> Self {
        LadderRung::new(body)
    }
}

impl From<crate::composition::SingleBuilder> for LadderRung {
    fn from(body: crate::composition::SingleBuilder) -> Self {
        LadderRung::new(body)
    }
}

/// Builder for [`Composition::FallbackLadder`]; converts into a
/// [`Composition`] (so it nests in `seq([...])` without `.build()`).
///
/// ```
/// use borg::dsl::*;
/// use borg::ladder::LadderRung;
/// # use borg::testing::ScriptedAgent;
///
/// let comp = fallback_ladder([
///     LadderRung::new(step(ScriptedAgent::new("primary"))),
///     LadderRung::new(step(ScriptedAgent::new("backup")))
///         .when(|h, _failures| h.get("needs_tools").is_none()),
/// ])
/// .falls_through(|e| e.is_retryable())
/// .build();
/// ```
#[must_use]
pub struct FallbackLadder {
    rungs: Vec<LadderRung>,
    falls_through: Option<FallsThroughFn>,
    on_fall: Option<OnFallFn>,
}

impl FallbackLadder {
    /// A ladder over `rungs`, falling through on
    /// [`BorgError::is_retryable`] until told otherwise.
    pub fn new<I, R>(rungs: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<LadderRung>,
    {
        Self {
            rungs: rungs.into_iter().map(Into::into).collect(),
            falls_through: None,
            on_fall: None,
        }
    }

    /// Add a rung.
    pub fn rung(mut self, rung: impl Into<LadderRung>) -> Self {
        self.rungs.push(rung.into());
        self
    }

    /// The ladder's fall-through rule, for rungs without their own.
    pub fn falls_through(
        mut self,
        pred: impl Fn(&BorgError) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.falls_through = Some(Arc::new(pred));
        self
    }

    /// Observe (and record on the handoff) each failure that falls through.
    pub fn on_fall(
        mut self,
        hook: impl Fn(&RungFailure<BorgError>, Option<&str>, &mut HandoffState) + Send + Sync + 'static,
    ) -> Self {
        self.on_fall = Some(Arc::new(hook));
        self
    }

    /// The composition.
    pub fn build(self) -> Composition {
        self.into()
    }
}

impl From<FallbackLadder> for Composition {
    fn from(b: FallbackLadder) -> Self {
        Composition::FallbackLadder {
            rungs: b.rungs,
            falls_through: b
                .falls_through
                .unwrap_or_else(|| Arc::new(BorgError::is_retryable)),
            on_fall: b.on_fall,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    enum Fail {
        Down { cost: u32, id: Option<&'static str> },
        Refused,
    }

    impl fmt::Display for Fail {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Fail::Down { cost, .. } => write!(f, "down({cost})"),
                Fail::Refused => write!(f, "refused"),
            }
        }
    }

    fn down(cost: u32, id: Option<&'static str>) -> Fail {
        Fail::Down { cost, id }
    }

    fn is_down(e: &Fail) -> bool {
        matches!(e, Fail::Down { .. })
    }

    #[tokio::test]
    async fn the_first_rung_that_ships_wins_and_keeps_the_falls_before_it() {
        let shipped = Ladder::new()
            .rung("a", async { Err::<u8, _>(down(1, None)) })
            .rung("b", async { Ok(2) })
            .rung("c", async { panic!("never polled") })
            .falls_through(is_down)
            .run_shipped()
            .await
            .unwrap();
        assert_eq!((shipped.value, &*shipped.rung, shipped.index), (2, "b", 1));
        assert_eq!(shipped.failures.len(), 1);
        assert_eq!(&*shipped.failures[0].rung, "a");
        assert!(!shipped.first_try());
    }

    #[tokio::test]
    async fn a_terminal_error_stops_the_ladder_as_is() {
        let err = Ladder::new()
            .rung("a", async { Err::<(), _>(down(1, None)) })
            .rung("b", async { Err(Fail::Refused) })
            .rung("c", async { Ok(()) })
            .falls_through(is_down)
            .run()
            .await
            .unwrap_err();
        match err {
            LadderError::Stopped { failure, earlier } => {
                assert_eq!(failure.error, Fail::Refused);
                assert_eq!(&*failure.rung, "b");
                assert_eq!(earlier.len(), 1);
            }
            other => panic!("expected Stopped, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rungs_choose_themselves_from_the_failures_so_far() {
        let shipped =
            Ladder::new()
                .rung("a", async { Err::<&str, _>(down(0, Some("x"))) })
                // Skipped: the failure says its provider is down too.
                .push(Rung::new("same_provider", async { Ok("wrong") }).when(
                    |f: &[RungFailure<Fail>]| f.iter().all(|r| r.error != down(0, Some("x"))),
                ))
                .rung("other_provider", async { Ok("right") })
                .run_shipped()
                .await
                .unwrap();
        assert_eq!((shipped.value, shipped.index), ("right", 2));
    }

    #[tokio::test]
    async fn a_rungs_own_rule_beats_the_ladders() {
        let err = Ladder::new()
            .push(Rung::new("a", async { Err::<(), _>(down(1, None)) }).falls_through(|_| false))
            .rung("b", async { Ok(()) })
            .falls_through(is_down)
            .run()
            .await
            .unwrap_err();
        assert!(err.is_stopped());
    }

    #[tokio::test]
    async fn exhausted_keeps_every_failure_and_merges_them() {
        let err = Ladder::new()
            .rung("a", async { Err::<(), _>(down(3, Some("first"))) })
            .rung("b", async { Err(down(5, None)) })
            .run()
            .await
            .unwrap_err();
        assert!(matches!(err, LadderError::Exhausted { .. }));
        assert_eq!(err.failures().count(), 2);
        assert_eq!(&*err.last().unwrap().rung, "b");
        assert_eq!(
            err.to_string(),
            "every fallback rung failed: `a`: down(3); `b`: down(5)"
        );
        let merged = err
            .merge(|later, earlier| match (later, earlier) {
                (Fail::Down { cost: a, id: ia }, Fail::Down { cost: b, id: ib }) => Fail::Down {
                    cost: a + b,
                    id: ia.or(ib),
                },
                (later, _) => later,
            })
            .unwrap();
        assert_eq!(merged, down(8, Some("first")));
    }

    #[tokio::test]
    async fn on_fall_sees_each_fall_and_where_it_goes() {
        let mut seen = Vec::new();
        let _ = Ladder::new()
            .rung("a", async { Err::<(), _>(down(0, None)) })
            .push(Rung::new("skipped", async { Ok(()) }).when(|_| false))
            .rung("b", async { Err(down(0, None)) })
            .on_fall(|f, to| seen.push((f.rung.to_string(), to.map(str::to_string))))
            .run()
            .await;
        assert_eq!(
            seen,
            vec![
                ("a".to_string(), Some("b".to_string())),
                ("b".to_string(), None)
            ]
        );
    }

    #[tokio::test]
    async fn no_applicable_rung_is_an_empty_exhaustion() {
        let err = Ladder::<(), Fail>::new()
            .push(Rung::new("a", async { Ok(()) }).when(|_| false))
            .run()
            .await
            .unwrap_err();
        assert!(matches!(&err, LadderError::Exhausted { failures } if failures.is_empty()));
        assert!(err.merge(|a, _| a).is_none());
    }

    #[tokio::test]
    async fn a_cancelled_ladder_stops_between_rungs() {
        let token = CancellationToken::new();
        let t = token.clone();
        let err = Ladder::new()
            .rung("a", async move {
                t.cancel();
                Err::<(), _>(down(0, None))
            })
            .rung("b", async { Ok(()) })
            .cancel_on(token)
            .run()
            .await
            .unwrap_err();
        assert!(matches!(err, LadderError::Cancelled { failures } if failures.len() == 1));
    }
}
