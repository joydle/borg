//! Context propagation — carry the caller's task-scoped context into every
//! task borg spawns.
//!
//! A `tokio::task_local!` covers the future it wraps and nothing that
//! future `tokio::spawn`s. borg spawns: every [`Composition::Parallel`]
//! branch runs on a supervised task, a parallel tool batch runs one task
//! per call, and [`crate::SubAgentHandle::spawn`] /
//! [`crate::ExecCtx::spawn_sub_agent`] run their work in the background.
//! Without help, an app's per-request scope (a spend ledger, a rate-limit
//! window, an eval recorder, a request id) is simply *gone* inside those
//! branches, and the app has to carry it by hand into every closure.
//!
//! A [`ContextCarrier`] is that hand-carry, written once. borg calls
//! [`ContextCarrier::capture`] on the spawning task, right before the
//! spawn, and re-enters the captured value inside the new task around the
//! branch's whole future. The current [`tracing::Span`] always travels the
//! same way, so a branch's logs nest under the span that spawned it.
//!
//! Register a carrier either
//! - **process-wide**, once at startup: [`register_carrier`] — it then
//!   applies to every spawn borg makes, including
//!   [`crate::SubAgentHandle::spawn`] which has no `ExecCtx`; or
//! - **per execution**: [`crate::ExecCtx::with_carrier`] — it applies to
//!   that composition tree (and every spawn below it) only.
//!
//! A carrier whose value is not set on the spawning task captures nothing
//! and costs nothing, so a process-wide carrier is harmless to code that
//! never sets its value.
//!
//! The common case is a `tokio` task-local; [`TaskLocal`] is the carrier
//! for one:
//!
//! ```
//! use borg::propagation::{self, TaskLocal};
//!
//! tokio::task_local! {
//!     static REQUEST_ID: String;
//! }
//!
//! // Once, at startup.
//! propagation::register_carrier(TaskLocal::new(&REQUEST_ID));
//! ```
//!
//! For anything that is not a task-local, implement [`ContextCarrier`]
//! and [`Captured`] directly; [`Captured::enter`] wraps the branch's
//! future in whatever scope the value needs.
//!
//! [`Composition::Parallel`]: crate::Composition::Parallel

use parking_lot::RwLock;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tracing::Instrument;

/// A boxed, `Send` future borrowed for `'a` — the shape [`Captured::enter`]
/// wraps. `()` output: borg threads the branch's real output around it.
pub type ScopedFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

/// Something an app wants to survive borg's spawns.
///
/// Called on the spawning task. Return `None` when there is nothing to
/// carry (the value is not set here); the spawn then runs without it.
///
/// Implement it once per kind of context and register it with
/// [`register_carrier`] or [`crate::ExecCtx::with_carrier`].
pub trait ContextCarrier: Send + Sync + 'static {
    /// Take what the new task needs, on the task about to spawn it.
    fn capture(&self) -> Option<Box<dyn Captured>>;
}

/// A captured value, ready to be re-entered inside a spawned task.
pub trait Captured: Send + 'static {
    /// Run `fut` inside the captured context (e.g.
    /// `LocalKey::scope(value, fut)`). Called once, inside the new task,
    /// before the branch's first poll.
    fn enter<'a>(self: Box<Self>, fut: ScopedFuture<'a>) -> ScopedFuture<'a>;
}

/// [`ContextCarrier`] for a `tokio` task-local: captures a clone of the
/// value set on the spawning task, and re-enters it with
/// [`tokio::task::LocalKey::scope`] inside the branch.
///
/// The value is cloned once per spawn, so wrap anything non-trivial in an
/// `Arc` (`task_local! { static SPEND: Arc<Ledger>; }`) and every branch
/// shares the one ledger.
pub struct TaskLocal<T: 'static> {
    key: &'static tokio::task::LocalKey<T>,
}

impl<T: 'static> TaskLocal<T> {
    /// A carrier for `key`.
    pub const fn new(key: &'static tokio::task::LocalKey<T>) -> Self {
        Self { key }
    }
}

impl<T: Clone + Send + Sync + 'static> ContextCarrier for TaskLocal<T> {
    fn capture(&self) -> Option<Box<dyn Captured>> {
        let key = self.key;
        key.try_with(T::clone)
            .ok()
            .map(|value| Box::new(TaskLocalValue { key, value }) as Box<dyn Captured>)
    }
}

struct TaskLocalValue<T: 'static> {
    key: &'static tokio::task::LocalKey<T>,
    value: T,
}

impl<T: Send + Sync + 'static> Captured for TaskLocalValue<T> {
    fn enter<'a>(self: Box<Self>, fut: ScopedFuture<'a>) -> ScopedFuture<'a> {
        let TaskLocalValue { key, value } = *self;
        Box::pin(key.scope(value, fut))
    }
}

/// An ordered set of carriers. Cheap to clone (one `Arc`).
#[derive(Clone, Default)]
pub struct Carriers {
    list: Arc<[Arc<dyn ContextCarrier>]>,
}

impl Carriers {
    /// No carriers.
    pub fn new() -> Self {
        Self::default()
    }

    /// These carriers plus `carrier`, entered innermost.
    #[must_use]
    pub fn with(&self, carrier: Arc<dyn ContextCarrier>) -> Self {
        let mut v: Vec<_> = self.list.iter().cloned().collect();
        v.push(carrier);
        Self { list: v.into() }
    }

    /// Number of carriers.
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// True when there are none.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

impl std::fmt::Debug for Carriers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Carriers")
            .field("len", &self.list.len())
            .finish()
    }
}

static GLOBAL: RwLock<Vec<Arc<dyn ContextCarrier>>> = RwLock::new(Vec::new());

/// Register `carrier` for every spawn borg makes in this process, from now
/// on. Call it once per kind of context, at startup; registering the same
/// key twice enters it twice (harmless, but wasted work).
pub fn register_carrier(carrier: impl ContextCarrier) {
    GLOBAL.write().push(Arc::new(carrier));
}

/// How many process-wide carriers are registered.
pub fn registered_carriers() -> usize {
    GLOBAL.read().len()
}

/// Everything a spawned task re-enters: the spawner's [`tracing::Span`]
/// and each carrier's captured value.
///
/// borg builds one per spawn with [`capture`]; apps that spawn their own
/// tasks beside borg's can use it too, so their tasks see the same
/// context:
///
/// ```ignore
/// let ctx = borg::propagation::capture(&borg::propagation::Carriers::new());
/// tokio::spawn(ctx.scope(async move { /* the task-locals are set here */ }));
/// ```
#[must_use = "a captured context does nothing until `scope` runs a future in it"]
pub struct CapturedContext {
    span: tracing::Span,
    captured: Vec<Box<dyn Captured>>,
}

/// Capture the current span, every process-wide carrier and every carrier
/// in `local`, on the calling task.
pub fn capture(local: &Carriers) -> CapturedContext {
    capture_with_span(local, tracing::Span::current())
}

/// [`capture`] with an explicit span in place of the current one.
pub fn capture_with_span(local: &Carriers, span: tracing::Span) -> CapturedContext {
    let mut captured = Vec::new();
    {
        let global = GLOBAL.read();
        captured.extend(global.iter().filter_map(|c| c.capture()));
    }
    captured.extend(local.list.iter().filter_map(|c| c.capture()));
    CapturedContext { span, captured }
}

impl CapturedContext {
    /// How many carrier values were captured (the span is not counted).
    pub fn captured_values(&self) -> usize {
        self.captured.len()
    }

    /// Run `fut` inside this context: under the captured span, with every
    /// captured value re-entered (the first carrier outermost).
    pub fn scope<F>(self, fut: F) -> impl Future<Output = F::Output> + Send
    where
        F: Future + Send,
        F::Output: Send,
    {
        let CapturedContext { span, captured } = self;
        async move {
            if captured.is_empty() {
                return fut.await;
            }
            let mut out = None;
            {
                let slot = &mut out;
                let mut inner: ScopedFuture<'_> = Box::pin(async move {
                    *slot = Some(fut.await);
                });
                for c in captured.into_iter().rev() {
                    inner = c.enter(inner);
                }
                inner.await;
            }
            out.expect("borg: a carrier's `enter` dropped the branch's future without running it")
        }
        .instrument(span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    tokio::task_local! {
        static A: u32;
        static B: Arc<str>;
    }

    #[tokio::test]
    async fn task_local_value_crosses_a_spawn() {
        let local = Carriers::new().with(Arc::new(TaskLocal::new(&A)));
        let seen = A
            .scope(7, async {
                let ctx = capture(&local);
                assert_eq!(ctx.captured_values(), 1);
                tokio::spawn(ctx.scope(async { A.try_with(|v| *v).ok() }))
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(seen, Some(7));
    }

    #[tokio::test]
    async fn an_unset_key_captures_nothing() {
        let local = Carriers::new().with(Arc::new(TaskLocal::new(&A)));
        let ctx = capture(&local);
        assert_eq!(ctx.captured_values(), 0);
        let seen = tokio::spawn(ctx.scope(async { A.try_with(|v| *v).ok() }))
            .await
            .unwrap();
        assert_eq!(seen, None);
    }

    #[tokio::test]
    async fn several_carriers_all_enter() {
        let local = Carriers::new()
            .with(Arc::new(TaskLocal::new(&A)))
            .with(Arc::new(TaskLocal::new(&B)));
        let (a, b) = A
            .scope(
                3,
                B.scope(Arc::from("req-1"), async {
                    let ctx = capture(&local);
                    tokio::spawn(
                        ctx.scope(async {
                            (A.try_with(|v| *v).ok(), B.try_with(|v| v.clone()).ok())
                        }),
                    )
                    .await
                    .unwrap()
                }),
            )
            .await;
        assert_eq!(a, Some(3));
        assert_eq!(b.as_deref(), Some("req-1"));
    }
}
