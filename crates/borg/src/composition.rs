//! `Composition` — orchestration shapes for combining `BorgAgent` impls.
//!
//! Variants nest. Step names are unique within the immediate composition
//! (enforced at builder time, panics on duplicate). Outputs land in
//! `HandoffState.state` keyed by `output_key` when one is declared.
//!
//! Handoff is named, not positional: named keys document intent inline
//! and survive reordering, where positional `Vec<Verdict>` chaining
//! forces consumers to track step indices.
//!
//! Two patterns are intentionally absent: control-transfer handoff
//! (where one agent takes over the conversation — borg only models
//! agents-as-tools, parent stays in control) and OneForAll/RestForOne
//! supervisor strategies (only Parallel has siblings, and "restart all
//! on one failure" isn't useful for LLM agents — borg is OneForOne).

use crate::agent::BorgAgent;
use crate::handoff::HandoffState;
use crate::verdict::Verdict;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Predicate fed to `Composition::LoopUntil`. Returns true to stop the
/// loop. Receives the most-recently-recorded verdict.
///
/// A `fn` pointer (not `Box<dyn Fn>`) keeps `Composition: Send +
/// 'static` trivially. For captured state, use a top-level function.
pub type UntilFn = fn(&Verdict) -> bool;

/// Per-iteration view passed to a [`LoopBody`] factory: the iteration
/// index, read-only handoff state, and last verdict. Lets the body adapt
/// per turn (different tools, throttled `max_tokens`, compacted prompts).
#[non_exhaustive]
pub struct LoopCtx<'a> {
    /// 0-indexed iteration counter. Always < `max_iterations`.
    pub iteration: u32,
    /// Read-only handoff snapshot at the start of this iteration. Writes
    /// happen inside the inner composition as it runs, not here.
    pub handoff: &'a HandoffState,
    /// Last verdict from the prior iteration's tail, or `None` on
    /// iteration 0.
    pub last_verdict: Option<&'a Verdict>,
}

/// Factory closure for `Composition::LoopUntil`, called once per
/// iteration to produce the next body. Boxed `Fn` (not `FnMut`) keeps
/// the composition `Send + Sync`.
pub type LoopBody = Box<dyn Fn(LoopCtx<'_>) -> Composition + Send + Sync>;

/// Selector fed to `Composition::Branch`. Reads the handoff state and
/// returns the key of the branch to run; an unmatched key surfaces
/// `BorgError::UnmatchedBranch` unless a `default` was declared.
///
/// `Arc<dyn Fn>` (not a bare `fn` pointer) so selectors can capture
/// configuration. Bare fn pointers still satisfy the bound and pass
/// through the constructors unchanged.
pub type SelectorFn = Arc<dyn Fn(&HandoffState) -> BranchKey + Send + Sync>;

/// String-newtype for branch keys, routing `Branch` execution.
///
/// Construct via the `From` impls or `.into()`. The inner string is
/// private so the representation can change without a semver bump.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BranchKey(String);

impl BranchKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the key, returning the inner string.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl From<&str> for BranchKey {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for BranchKey {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Error policy for compositions with siblings (`Parallel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorPolicy {
    /// Stop on the first failed branch, cancel all in-flight siblings,
    /// propagate the error.
    FailFast,
    /// Run every branch to completion regardless of failures. Failed
    /// branches surface as `NamedVerdict { is_error: true, … }` in the
    /// audit history.
    CollectAll,
    /// Like `CollectAll` but require at least `n_ok` successful branches;
    /// otherwise surface the composition itself as failed.
    AtLeast(usize),
}

/// Backoff strategy between retry attempts. Selected via
/// [`RestartPolicy::with_backoff`] / [`RestartPolicy::without_backoff`];
/// [`RestartPolicy::on_failure`] picks [`Backoff::default_for_llm`]
/// (exponential with jitter), so `on_failure(N)` does the right thing
/// against rate-limited providers by default.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Backoff {
    /// Retry immediately. Use only against a non-rate-limited resource
    /// (in-memory work, panicked child re-spawn).
    None,
    /// Constant delay between attempts.
    Fixed(Duration),
    /// `delay(n) = min(base * 2^(n-1), max)`, then scaled by a random
    /// value in `[1.0 - jitter_ratio, 1.0]` to break herd effects.
    /// Attempt 1 has no delay.
    ///
    /// `jitter_ratio` clamps to `[0.0, 1.0]` (NaN counts as `0.0`);
    /// `0.0` is deterministic.
    /// [`Backoff::default_for_llm`] uses `0.3`.
    Exponential {
        base: Duration,
        max: Duration,
        jitter_ratio: f32,
    },
}

impl Backoff {
    /// Exponential 200 ms → 4 s with 30% jitter, sized for LLM provider
    /// retry-after windows.
    pub const fn default_for_llm() -> Self {
        Backoff::Exponential {
            base: Duration::from_millis(200),
            max: Duration::from_secs(4),
            jitter_ratio: 0.3,
        }
    }

    /// Delay before `attempt` (1-indexed). Attempt 1 is always
    /// `Duration::ZERO`; only retries are delayed.
    pub fn delay_for_attempt(&self, attempt: u8) -> Duration {
        if attempt <= 1 {
            return Duration::ZERO;
        }
        match self {
            Backoff::None => Duration::ZERO,
            Backoff::Fixed(d) => *d,
            Backoff::Exponential {
                base,
                max,
                jitter_ratio,
            } => {
                let exp = (attempt - 2).min(20) as u32;
                let raw = base.checked_mul(1u32 << exp).unwrap_or(*max);
                let capped = raw.min(*max);
                apply_jitter(capped, *jitter_ratio)
            }
        }
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Backoff::default_for_llm()
    }
}

/// Scale `d` by `[1.0 - ratio, 1.0]` using `SystemTime` subsec-nanos as
/// entropy. Not cryptographic — the only goal is to break herd effects
/// when N siblings retry at once. `ratio = 0.0` is deterministic.
///
/// Never panics on a bad config: a NaN ratio means no jitter, any other
/// ratio clamps to `[0.0, 1.0]`, and the result never exceeds `d` (float
/// rounding near `Duration::MAX` falls back to `d` itself).
fn apply_jitter(d: Duration, ratio: f32) -> Duration {
    let ratio = if ratio.is_nan() {
        0.0
    } else {
        ratio.clamp(0.0, 1.0)
    };
    if ratio == 0.0 {
        return d;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let frac = (f64::from(nanos) / 1_000_000_000.0) * f64::from(ratio);
    let mul = (1.0 - frac).clamp(0.0, 1.0);
    Duration::try_from_secs_f64(d.as_secs_f64() * mul).map_or(d, |j| j.min(d))
}

/// Per-agent restart policy. OneForOne: this agent's failure does not
/// cascade to siblings.
///
/// `OnFailure` retries up to `max_attempts` times, sleeping per
/// `backoff` between attempts. A non-zero `window` also bails once total
/// wall-clock time exceeds it, even with attempts remaining — so a
/// flapping upstream can't mask a caller's `timeout(30s)` behind an
/// inner retry loop.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub enum RestartPolicy {
    /// Single attempt; surface failure. Default.
    #[default]
    Never,
    /// Up to `max_attempts` (including the first). If `window > 0`, also
    /// bail when elapsed time exceeds `window`.
    OnFailure {
        max_attempts: u8,
        window: Duration,
        backoff: Backoff,
    },
}

impl RestartPolicy {
    pub fn never() -> Self {
        RestartPolicy::Never
    }

    /// Retry up to `max_attempts` times (including the first) with
    /// [`Backoff::default_for_llm`] between attempts. Chain
    /// [`RestartPolicy::without_backoff`] / [`RestartPolicy::with_backoff`]
    /// to adjust.
    pub fn on_failure(max_attempts: u8) -> Self {
        RestartPolicy::OnFailure {
            max_attempts: max_attempts.max(1),
            window: Duration::ZERO,
            backoff: Backoff::default_for_llm(),
        }
    }

    /// `on_failure` with a wall-clock cap on the entire retry loop.
    pub fn on_failure_within(max_attempts: u8, window: Duration) -> Self {
        RestartPolicy::OnFailure {
            max_attempts: max_attempts.max(1),
            window,
            backoff: Backoff::default_for_llm(),
        }
    }

    /// Override the backoff strategy:
    /// `RestartPolicy::on_failure(3).with_backoff(Backoff::None)`.
    #[must_use]
    pub fn with_backoff(self, backoff: Backoff) -> Self {
        match self {
            RestartPolicy::Never => RestartPolicy::Never,
            RestartPolicy::OnFailure {
                max_attempts,
                window,
                ..
            } => RestartPolicy::OnFailure {
                max_attempts,
                window,
                backoff,
            },
        }
    }

    /// Opt out of all delay between attempts (`.with_backoff(Backoff::None)`).
    /// Use only for non-network retries.
    #[must_use]
    pub fn without_backoff(self) -> Self {
        self.with_backoff(Backoff::None)
    }

    pub fn max_attempts(&self) -> u8 {
        match self {
            RestartPolicy::Never => 1,
            RestartPolicy::OnFailure { max_attempts, .. } => *max_attempts,
        }
    }

    /// Wall-clock cap on the entire retry loop. `Duration::ZERO` means
    /// no cap (only `max_attempts` bounds the loop).
    pub fn window(&self) -> Duration {
        match self {
            RestartPolicy::Never => Duration::ZERO,
            RestartPolicy::OnFailure { window, .. } => *window,
        }
    }

    /// Backoff between attempts. `Backoff::None` for [`RestartPolicy::Never`].
    pub fn backoff(&self) -> Backoff {
        match self {
            RestartPolicy::Never => Backoff::None,
            RestartPolicy::OnFailure { backoff, .. } => *backoff,
        }
    }
}

/// The composition tree. Every variant carries enough metadata for the
/// executor to render correct events without inferring shape.
///
/// `#[non_exhaustive]` so a minor release can add primitives — match with a
/// `_` arm.
#[non_exhaustive]
pub enum Composition {
    /// Run one agent. The most common leaf.
    Single {
        /// Composition-unique name. Surfaces in `HandoffState.history`
        /// and lifecycle events.
        name: String,
        agent: Box<dyn BorgAgent>,
        /// Optional key under which to write `verdict.body` into
        /// `HandoffState.state`. `None` records the verdict in `history`
        /// only.
        output_key: Option<String>,
        /// Restart-on-transient-failure policy.
        restart: RestartPolicy,
    },
    /// Run named steps in order. Step N reads `HandoffState` mutated by
    /// steps 0..N-1.
    Sequential {
        steps: Vec<Composition>,
        on_error: ErrorPolicy,
    },
    /// Run named branches concurrently. Each sees the same handoff
    /// snapshot taken at fan-out; writes merge at fan-in. The builder
    /// enforces unique branch names and (when declared) output_keys.
    Parallel {
        branches: Vec<(String, Composition)>,
        on_error: ErrorPolicy,
        max_concurrency: usize,
    },
    /// Repeat the body until `until(last_verdict)` returns true or
    /// `max_iterations` is reached.
    ///
    /// The body is a *factory*, called once per iteration with a
    /// [`LoopCtx`] so it can adapt per turn (different tools, throttled
    /// `max_tokens`, compacted prompts). This also sidesteps the need
    /// for `Composition: Clone`.
    LoopUntil {
        body: LoopBody,
        until: UntilFn,
        max_iterations: u32,
    },
    /// Selector picks the branch key; the matching branch runs, or
    /// `default` if none matches.
    Branch {
        selector: SelectorFn,
        branches: Vec<(BranchKey, Composition)>,
        default: Option<Box<Composition>>,
    },
    /// Try `primary`; on retryable error, try each fallback in order.
    Fallback {
        primary: Box<Composition>,
        fallbacks: Vec<Composition>,
    },
    /// Fallback ladder: run rungs in order until one ships. A rung's
    /// error falls through only when its predicate (the rung's own, else
    /// `falls_through`) says so; any other error is returned as-is. A
    /// rung with a `when` rule runs only if it returns true. When every
    /// applicable rung fell through, the error is
    /// [`crate::BorgError::FallbackExhausted`] with every rung's error.
    /// Build with [`crate::dsl::fallback_ladder`] /
    /// [`crate::ladder::FallbackLadder`]; see [`crate::ladder`].
    FallbackLadder {
        rungs: Vec<crate::ladder::LadderRung>,
        falls_through: crate::ladder::FallsThroughFn,
        on_fall: Option<crate::ladder::OnFallFn>,
    },
    /// Tool-loop: call the agent → dispatch any tool calls it emits →
    /// thread results back via handoff → repeat until the agent stops
    /// calling tools or `max_steps` is reached.
    ///
    /// Each iteration is one model call; N tool calls in one response
    /// dispatch together and count as one step. The agent reads
    /// accumulated results from the handoff slot `"tool_results"`
    /// (a `Vec<DispatchedCall>`), which the executor seeds and appends
    /// automatically; `extract_tool_calls(&Verdict)` returning empty
    /// stops the loop.
    Agent {
        /// Step name for events and handoff history.
        name: String,
        /// The agent driving the loop: `invoke()` produces a Verdict,
        /// `extract_tool_calls()` parses tool-use intent from it.
        agent: Box<dyn crate::agent::BorgAgent>,
        /// Context-erased tool dispatch surface. Typically
        /// `Arc<MutexDispatcher<C>>`.
        tools: std::sync::Arc<dyn crate::tools::ToolDispatch>,
        /// Maximum model calls; each counts as one step. If the last
        /// allowed step still asks for tools (its calls are run), the
        /// loop fails with [`BorgError::MaxStepsReached`](crate::BorgError::MaxStepsReached);
        /// a final answer on the last step is a normal `Ok`.
        max_steps: u32,
    },
    /// Run `inner` while beating a [`crate::BorgEvent::Heartbeat`] every
    /// `every` (and once at its start when `at_start`), for as long as it
    /// runs. Transparent: `inner` runs in the same context (composition
    /// id, handoff, cancel, budget), so wrapping a composition changes
    /// none of its own events or step names. Build with
    /// [`crate::dsl::heartbeat`].
    Heartbeat {
        /// The beat's name ([`crate::BorgEvent::Heartbeat::name`]).
        name: String,
        /// Beat period; a missed tick is skipped, not caught up.
        every: Duration,
        /// Beat once (beat 0, `elapsed_ms` 0) before `inner` starts.
        at_start: bool,
        inner: Box<Composition>,
    },
    /// A composition built when it is reached, from the handoff as the
    /// steps before it left it: for a shape that depends on what an
    /// earlier step decided (one branch per item a planner listed). Build
    /// with [`crate::dsl::deferred`].
    ///
    /// For a resumable execution the builder is also called once with an
    /// empty handoff to fingerprint the shape (as a `LoopUntil` body is),
    /// so it must not panic on one.
    Deferred { build: DeferredFn },
}

/// The builder of a [`Composition::Deferred`]: called with the handoff
/// when the composition is reached. `Fn` (not `FnOnce`) so a checkpoint
/// can fingerprint its shape.
pub type DeferredFn = Box<dyn Fn(&HandoffState) -> Composition + Send + Sync>;

impl Composition {
    // ── Idiomatic constructors ──────────────────────────────────────
    //
    // These accept agents directly (no `Box::new`) and simple iterables.
    // Builders cover advanced cases (custom `output_key`, `restart`,
    // `on_error`, `max_concurrency`).

    /// Build a single-agent composition from any `BorgAgent` impl,
    /// auto-boxing the agent. For `output_key` / `restart`, use
    /// [`Composition::single_with`].
    ///
    /// ```
    /// # use borg::*;
    /// # use async_trait::async_trait;
    /// # struct MyAgent;
    /// # #[async_trait]
    /// # impl BorgAgent for MyAgent {
    /// #     fn agent_id(&self) -> &'static str { "my" }
    /// #     fn model(&self) -> &str { "test" }
    /// #     fn max_tokens(&self) -> u32 { 100 }
    /// #     fn system_prompt(&self) -> &'static str { "" }
    /// #     fn user_prompt(&self, _: &AgentContext) -> Option<String> { None }
    /// # }
    /// let comp = Composition::single("reviewer", MyAgent);
    /// ```
    pub fn single(name: impl Into<String>, agent: impl BorgAgent + 'static) -> Composition {
        Composition::Single {
            name: name.into(),
            agent: Box::new(agent),
            output_key: None,
            restart: RestartPolicy::Never,
        }
    }

    /// Like [`Composition::single`] but returns a [`SingleBuilder`] to
    /// configure `output_key`, `restart`, etc. before `.build()`.
    pub fn single_with(name: impl Into<String>, agent: impl BorgAgent + 'static) -> SingleBuilder {
        SingleBuilder {
            name: name.into(),
            agent: Box::new(agent),
            output_key: None,
            restart: RestartPolicy::Never,
        }
    }

    /// Run steps sequentially with `FailFast`. Step N reads handoff
    /// state mutated by 0..N-1. For other policies, use
    /// [`Composition::sequential_with`].
    ///
    /// ```
    /// # use borg::*;
    /// # use async_trait::async_trait;
    /// # struct A; struct B;
    /// # #[async_trait] impl BorgAgent for A {
    /// #     fn agent_id(&self) -> &'static str { "a" } fn model(&self) -> &str { "" }
    /// #     fn max_tokens(&self) -> u32 { 0 } fn system_prompt(&self) -> &'static str { "" }
    /// #     fn user_prompt(&self, _: &AgentContext) -> Option<String> { None }
    /// # }
    /// # #[async_trait] impl BorgAgent for B {
    /// #     fn agent_id(&self) -> &'static str { "b" } fn model(&self) -> &str { "" }
    /// #     fn max_tokens(&self) -> u32 { 0 } fn system_prompt(&self) -> &'static str { "" }
    /// #     fn user_prompt(&self, _: &AgentContext) -> Option<String> { None }
    /// # }
    /// let comp = Composition::sequential([
    ///     Composition::single("a", A),
    ///     Composition::single("b", B),
    /// ]);
    /// ```
    pub fn sequential(steps: impl IntoIterator<Item = Composition>) -> Composition {
        let collected: Vec<_> = steps.into_iter().collect();
        let mut seen = HashSet::new();
        for step in &collected {
            if let Composition::Single { name, .. } = step {
                require_unique_name(&mut seen, name, "sequential composition");
            }
        }
        Composition::Sequential {
            steps: collected,
            on_error: ErrorPolicy::FailFast,
        }
    }

    /// Sequential with an explicit error policy.
    pub fn sequential_with(
        steps: impl IntoIterator<Item = Composition>,
        on_error: ErrorPolicy,
    ) -> Composition {
        let mut comp = Self::sequential(steps);
        if let Composition::Sequential {
            on_error: ref mut e,
            ..
        } = comp
        {
            *e = on_error;
        }
        comp
    }

    /// Run named branches concurrently with `FailFast` and unbounded
    /// concurrency. Each branch sees the same handoff snapshot taken at
    /// fan-out. For configuration, use [`Composition::parallel_with`].
    ///
    /// Pass `(name, composition)` pairs:
    /// ```
    /// # use borg::*;
    /// # use async_trait::async_trait;
    /// # struct A; struct B;
    /// # #[async_trait] impl BorgAgent for A {
    /// #     fn agent_id(&self) -> &'static str { "a" } fn model(&self) -> &str { "" }
    /// #     fn max_tokens(&self) -> u32 { 0 } fn system_prompt(&self) -> &'static str { "" }
    /// #     fn user_prompt(&self, _: &AgentContext) -> Option<String> { None }
    /// # }
    /// # #[async_trait] impl BorgAgent for B {
    /// #     fn agent_id(&self) -> &'static str { "b" } fn model(&self) -> &str { "" }
    /// #     fn max_tokens(&self) -> u32 { 0 } fn system_prompt(&self) -> &'static str { "" }
    /// #     fn user_prompt(&self, _: &AgentContext) -> Option<String> { None }
    /// # }
    /// let comp = Composition::parallel([
    ///     ("a", Composition::single("a", A)),
    ///     ("b", Composition::single("b", B)),
    /// ]);
    /// ```
    pub fn parallel<S, I>(branches: I) -> Composition
    where
        S: Into<String>,
        I: IntoIterator<Item = (S, Composition)>,
    {
        let collected: Vec<(String, Composition)> =
            branches.into_iter().map(|(s, c)| (s.into(), c)).collect();

        // Unique branch names + unique output_keys.
        let mut seen_names = HashSet::new();
        let mut seen_keys = HashSet::new();
        for (name, comp) in &collected {
            require_unique_name(&mut seen_names, name, "parallel branches");
            if let Composition::Single {
                output_key: Some(k),
                ..
            } = comp
            {
                if !seen_keys.insert(k.clone()) {
                    panic!(
                        "borg: duplicate output_key `{k}` across parallel branches; \
                         parallel writers must use distinct keys to avoid races"
                    );
                }
            }
        }

        let max_concurrency = collected.len().max(1);
        Composition::Parallel {
            branches: collected,
            on_error: ErrorPolicy::FailFast,
            max_concurrency,
        }
    }

    /// Parallel with explicit error policy and concurrency cap.
    pub fn parallel_with<S, I>(
        branches: I,
        on_error: ErrorPolicy,
        max_concurrency: usize,
    ) -> Composition
    where
        S: Into<String>,
        I: IntoIterator<Item = (S, Composition)>,
    {
        let mut comp = Self::parallel(branches);
        if let Composition::Parallel {
            on_error: ref mut e,
            max_concurrency: ref mut c,
            ..
        } = comp
        {
            *e = on_error;
            *c = max_concurrency.max(1);
        }
        comp
    }

    /// Repeat a body produced by `body_factory` until the predicate
    /// fires or `max_iterations` is reached. The factory runs once per
    /// iteration (see [`LoopCtx`]). For a fixed body, use
    /// [`Composition::loop_fixed`].
    pub fn loop_until<F>(body_factory: F, until: UntilFn, max_iterations: u32) -> Composition
    where
        F: Fn(LoopCtx<'_>) -> Composition + Send + Sync + 'static,
    {
        Composition::LoopUntil {
            body: Box::new(body_factory),
            until,
            max_iterations: max_iterations.max(1),
        }
    }

    /// Loop a *fixed* body composition, for when the shape doesn't vary
    /// between iterations (per-turn adaptation lives in agents'
    /// `user_prompt`). Takes a `Fn() -> Composition` factory because
    /// `Composition` is not `Clone`.
    pub fn loop_fixed<F>(body_factory: F, until: UntilFn, max_iterations: u32) -> Composition
    where
        F: Fn() -> Composition + Send + Sync + 'static,
    {
        Composition::LoopUntil {
            body: Box::new(move |_ctx| body_factory()),
            until,
            max_iterations: max_iterations.max(1),
        }
    }

    /// Selector picks a branch by key. Pass `(key, composition)` pairs;
    /// for a fallback arm use [`Composition::branch_default`].
    ///
    /// Accepts any `Fn(&HandoffState) -> BranchKey + Send + Sync +
    /// 'static` — bare fn pointers and capturing closures alike. The
    /// selector is `Arc`-wrapped internally so `Composition` stays
    /// `Send + Sync`.
    pub fn branch<K, I, F>(selector: F, branches: I) -> Composition
    where
        F: Fn(&HandoffState) -> BranchKey + Send + Sync + 'static,
        K: Into<BranchKey>,
        I: IntoIterator<Item = (K, Composition)>,
    {
        let collected: Vec<(BranchKey, Composition)> =
            branches.into_iter().map(|(k, c)| (k.into(), c)).collect();
        let mut seen = HashSet::new();
        for (k, _) in &collected {
            if !seen.insert(k.0.clone()) {
                panic!("borg: duplicate branch key `{}` in branch composition", k.0);
            }
        }
        Composition::Branch {
            selector: Arc::new(selector),
            branches: collected,
            default: None,
        }
    }

    /// Branch with a default arm that runs when no key matches. See
    /// [`Composition::branch`] for the selector contract.
    pub fn branch_default<K, I, F>(selector: F, branches: I, default: Composition) -> Composition
    where
        F: Fn(&HandoffState) -> BranchKey + Send + Sync + 'static,
        K: Into<BranchKey>,
        I: IntoIterator<Item = (K, Composition)>,
    {
        let mut comp = Self::branch(selector, branches);
        if let Composition::Branch {
            default: ref mut d, ..
        } = comp
        {
            *d = Some(Box::new(default));
        }
        comp
    }

    /// Try `primary`, and on retryable error try each fallback in order.
    pub fn fallback(
        primary: Composition,
        fallbacks: impl IntoIterator<Item = Composition>,
    ) -> Composition {
        Composition::Fallback {
            primary: Box::new(primary),
            fallbacks: fallbacks.into_iter().collect(),
        }
    }

    // ── Inspection ──────────────────────────────────────────────────

    /// Human-readable label for events. `Single` uses the step name;
    /// grouping shapes use the shape name plus a count.
    pub fn label(&self) -> String {
        match self {
            Composition::Single { name, .. } => name.clone(),
            Composition::Sequential { steps, .. } => {
                format!("sequential({})", steps.len())
            }
            Composition::Parallel { branches, .. } => {
                format!("parallel({})", branches.len())
            }
            Composition::LoopUntil { max_iterations, .. } => {
                format!("loop_until(max={max_iterations})")
            }
            Composition::Branch { branches, .. } => {
                format!("branch({})", branches.len())
            }
            Composition::Fallback { fallbacks, .. } => {
                format!("fallback({})", fallbacks.len() + 1)
            }
            Composition::Agent {
                name, max_steps, ..
            } => {
                format!("agent({name},max_steps={max_steps})")
            }
            Composition::FallbackLadder { rungs, .. } => {
                format!("fallback_ladder({})", rungs.len())
            }
            Composition::Heartbeat { name, .. } => format!("heartbeat({name})"),
            Composition::Deferred { .. } => "deferred".to_string(),
        }
    }

    /// Number of immediate children, for `BorgEvent::CompositionStarted`.
    pub fn child_count(&self) -> u8 {
        match self {
            Composition::Single { .. } => 1,
            Composition::Sequential { steps, .. } => steps.len().min(u8::MAX as usize) as u8,
            Composition::Parallel { branches, .. } => branches.len().min(u8::MAX as usize) as u8,
            Composition::LoopUntil { .. } => 1,
            Composition::Branch { branches, .. } => branches.len().min(u8::MAX as usize) as u8,
            Composition::Fallback { fallbacks, .. } => {
                (fallbacks.len() + 1).min(u8::MAX as usize) as u8
            }
            // One logical child: tool-loop dispatches happen inside the
            // step, not as separate composition children.
            Composition::Agent { .. } => 1,
            Composition::FallbackLadder { rungs, .. } => rungs.len().min(u8::MAX as usize) as u8,
            Composition::Heartbeat { .. } | Composition::Deferred { .. } => 1,
        }
    }

    /// Construct a tool-loop agent composition.
    ///
    /// ```ignore
    /// let registry = Registry::<MyCtx>::new()
    ///     .with(ReadFile)
    ///     .with(ApplyEdit);
    /// let tools = Arc::new(MutexDispatcher::new(registry, my_context));
    ///
    /// let comp = Composition::agent("main", MyAgent, tools, /* max_steps */ 10);
    /// ```
    pub fn agent(
        name: impl Into<String>,
        agent: impl crate::agent::BorgAgent + 'static,
        tools: std::sync::Arc<dyn crate::tools::ToolDispatch>,
        max_steps: u32,
    ) -> Composition {
        Composition::Agent {
            name: name.into(),
            agent: Box::new(agent),
            tools,
            max_steps: max_steps.max(1),
        }
    }

    // ── Builder factory methods ─────────────────────────────────────

    /// Return a [`SingleBuilder`] from a pre-boxed agent. For the common
    /// case prefer [`Composition::single`] / [`Composition::single_with`].
    pub fn single_builder(name: impl Into<String>, agent: Box<dyn BorgAgent>) -> SingleBuilder {
        SingleBuilder {
            name: name.into(),
            agent,
            output_key: None,
            restart: RestartPolicy::Never,
        }
    }

    /// Return a [`SequentialBuilder`] for configuring error policy and
    /// adding steps one by one via `.step()`.
    pub fn sequential_builder() -> SequentialBuilder {
        SequentialBuilder {
            steps: Vec::new(),
            on_error: ErrorPolicy::FailFast,
            seen_names: HashSet::new(),
        }
    }

    /// Return a [`ParallelBuilder`] for configuring error policy,
    /// concurrency cap, and adding branches via `.branch()`.
    pub fn parallel_builder() -> ParallelBuilder {
        ParallelBuilder {
            branches: Vec::new(),
            on_error: ErrorPolicy::FailFast,
            max_concurrency: 4,
            seen_names: HashSet::new(),
            seen_output_keys: HashSet::new(),
        }
    }

    /// Return a [`LoopUntilBuilder`] for configuring `max_iterations`.
    /// Pass a body factory — see [`Composition::loop_until`].
    pub fn loop_until_builder<F>(body_factory: F, until: UntilFn) -> LoopUntilBuilder
    where
        F: Fn(LoopCtx<'_>) -> Composition + Send + Sync + 'static,
    {
        LoopUntilBuilder {
            body: Box::new(body_factory),
            until,
            max_iterations: 3,
        }
    }

    /// Return a [`BranchBuilder`] for adding arms via `.arm()` and an
    /// optional default via `.default()`.
    pub fn branch_builder<F>(selector: F) -> BranchBuilder
    where
        F: Fn(&HandoffState) -> BranchKey + Send + Sync + 'static,
    {
        BranchBuilder {
            selector: Arc::new(selector),
            branches: Vec::new(),
            default: None,
            seen_keys: HashSet::new(),
        }
    }

    /// Return a [`FallbackBuilder`] for adding fallback compositions via
    /// `.fallback()`.
    pub fn fallback_builder(primary: Composition) -> FallbackBuilder {
        FallbackBuilder {
            primary: Box::new(primary),
            fallbacks: Vec::new(),
        }
    }
}

// ── Builders ───────────────────────────────────────────────────────────

#[must_use]
pub struct SingleBuilder {
    name: String,
    agent: Box<dyn BorgAgent>,
    output_key: Option<String>,
    restart: RestartPolicy,
}

impl SingleBuilder {
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }

    pub fn restart(mut self, policy: RestartPolicy) -> Self {
        self.restart = policy;
        self
    }

    pub fn build(self) -> Composition {
        self.into()
    }
}

/// `SingleBuilder` -> `Composition` so leaves don't need `.build()`:
///
/// ```ignore
/// Composition::sequential([
///     Composition::single_with("a", A).output_key("a_out").into(),
///     Composition::single("b", B),
/// ])
/// ```
impl From<SingleBuilder> for Composition {
    fn from(b: SingleBuilder) -> Self {
        Composition::Single {
            name: b.name,
            agent: b.agent,
            output_key: b.output_key,
            restart: b.restart,
        }
    }
}

/// Builder for [`Composition::Heartbeat`] ([`crate::dsl::heartbeat`]);
/// converts into a `Composition`.
#[must_use]
pub struct HeartbeatBuilder {
    name: String,
    every: Duration,
    at_start: bool,
    inner: Composition,
}

impl HeartbeatBuilder {
    pub(crate) fn new(name: impl Into<String>, every: Duration, inner: Composition) -> Self {
        Self {
            name: name.into(),
            every,
            at_start: false,
            inner,
        }
    }

    /// Beat once before the inner composition starts (beat 0 at
    /// `elapsed_ms` 0), so a consumer knows at once that it began; the
    /// periodic beats follow from 1.
    pub fn at_start(mut self) -> Self {
        self.at_start = true;
        self
    }

    pub fn build(self) -> Composition {
        self.into()
    }
}

impl From<HeartbeatBuilder> for Composition {
    fn from(b: HeartbeatBuilder) -> Self {
        Composition::Heartbeat {
            name: b.name,
            every: b.every,
            at_start: b.at_start,
            inner: Box::new(b.inner),
        }
    }
}

#[must_use]
pub struct SequentialBuilder {
    steps: Vec<Composition>,
    on_error: ErrorPolicy,
    seen_names: HashSet<String>,
}

impl SequentialBuilder {
    pub fn step(mut self, comp: Composition) -> Self {
        // Unique step names keep handoff.find(name) unambiguous. Checked
        // against immediate children only; nested compositions scope
        // their own names.
        if let Composition::Single { ref name, .. } = comp {
            require_unique_name(&mut self.seen_names, name, "sequential composition");
        }
        self.steps.push(comp);
        self
    }

    pub fn on_error(mut self, policy: ErrorPolicy) -> Self {
        self.on_error = policy;
        self
    }

    pub fn build(self) -> Composition {
        Composition::Sequential {
            steps: self.steps,
            on_error: self.on_error,
        }
    }
}

#[must_use]
pub struct ParallelBuilder {
    branches: Vec<(String, Composition)>,
    on_error: ErrorPolicy,
    max_concurrency: usize,
    seen_names: HashSet<String>,
    seen_output_keys: HashSet<String>,
}

impl ParallelBuilder {
    pub fn branch(mut self, name: impl Into<String>, comp: Composition) -> Self {
        let name = name.into();
        require_unique_name(&mut self.seen_names, &name, "parallel branches");
        // Unique output_keys across branches — parallel writers to the
        // same key would race.
        if let Composition::Single {
            output_key: Some(ref k),
            ..
        } = comp
        {
            if !self.seen_output_keys.insert(k.clone()) {
                panic!(
                    "borg: duplicate output_key `{k}` across parallel branches; \
                     parallel writers must use distinct keys to avoid races"
                );
            }
        }
        self.branches.push((name, comp));
        self
    }

    pub fn on_error(mut self, policy: ErrorPolicy) -> Self {
        self.on_error = policy;
        self
    }

    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    pub fn build(self) -> Composition {
        Composition::Parallel {
            branches: self.branches,
            on_error: self.on_error,
            max_concurrency: self.max_concurrency,
        }
    }
}

#[must_use]
pub struct LoopUntilBuilder {
    body: LoopBody,
    until: UntilFn,
    max_iterations: u32,
}

impl LoopUntilBuilder {
    pub fn max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    pub fn build(self) -> Composition {
        Composition::LoopUntil {
            body: self.body,
            until: self.until,
            max_iterations: self.max_iterations,
        }
    }
}

#[must_use]
pub struct BranchBuilder {
    selector: SelectorFn,
    branches: Vec<(BranchKey, Composition)>,
    default: Option<Box<Composition>>,
    seen_keys: HashSet<String>,
}

impl BranchBuilder {
    pub fn arm(mut self, key: impl Into<BranchKey>, comp: Composition) -> Self {
        let key = key.into();
        if !self.seen_keys.insert(key.0.clone()) {
            panic!(
                "borg: duplicate branch key `{}` in branch composition",
                key.0
            );
        }
        self.branches.push((key, comp));
        self
    }

    pub fn default(mut self, comp: Composition) -> Self {
        self.default = Some(Box::new(comp));
        self
    }

    pub fn build(self) -> Composition {
        Composition::Branch {
            selector: self.selector,
            branches: self.branches,
            default: self.default,
        }
    }
}

#[must_use]
pub struct FallbackBuilder {
    primary: Box<Composition>,
    fallbacks: Vec<Composition>,
}

impl FallbackBuilder {
    pub fn or_else(mut self, comp: Composition) -> Self {
        self.fallbacks.push(comp);
        self
    }

    pub fn build(self) -> Composition {
        Composition::Fallback {
            primary: self.primary,
            fallbacks: self.fallbacks,
        }
    }
}

// ── Validation helpers ────────────────────────────────────────────────

// Panic when a name is reused. `context` labels the panic message.
fn require_unique_name(seen: &mut HashSet<String>, name: &str, context: &str) {
    if !seen.insert(name.to_string()) {
        panic!("borg: duplicate name `{name}` in {context}; names must be unique");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::BorgAgent;
    use crate::context::AgentContext;
    use async_trait::async_trait;

    struct DummyAgent(&'static str);

    #[async_trait]
    impl BorgAgent for DummyAgent {
        fn agent_id(&self) -> &'static str {
            self.0
        }
        fn model(&self) -> &str {
            "test/model"
        }
        fn max_tokens(&self) -> u32 {
            100
        }
        fn system_prompt(&self) -> &'static str {
            "sys"
        }
        fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
            Some("p".into())
        }
    }

    fn dummy(name: &'static str) -> Composition {
        Composition::single_builder(name, Box::new(DummyAgent(name))).build()
    }

    #[test]
    fn single_builder_produces_named_composition() {
        let c = Composition::single_builder("review", Box::new(DummyAgent("review")))
            .output_key("review_out")
            .build();
        match c {
            Composition::Single {
                name,
                output_key: Some(k),
                ..
            } => {
                assert_eq!(name, "review");
                assert_eq!(k, "review_out");
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn sequential_builder_accepts_distinct_names() {
        let c = Composition::sequential_builder()
            .step(dummy("a"))
            .step(dummy("b"))
            .step(dummy("c"))
            .build();
        match c {
            Composition::Sequential { steps, .. } => assert_eq!(steps.len(), 3),
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    #[should_panic(expected = "duplicate name `a` in sequential composition")]
    fn sequential_builder_panics_on_duplicate_step_name() {
        let _ = Composition::sequential_builder()
            .step(dummy("a"))
            .step(dummy("a"))
            .build();
    }

    #[test]
    #[should_panic(expected = "duplicate name `k` in parallel branches")]
    fn parallel_builder_panics_on_duplicate_branch_name() {
        let _ = Composition::parallel_builder()
            .branch("k", dummy("a"))
            .branch("k", dummy("b"))
            .build();
    }

    #[test]
    #[should_panic(expected = "duplicate output_key")]
    fn parallel_builder_panics_on_duplicate_output_key() {
        let a = Composition::single_builder("a", Box::new(DummyAgent("a")))
            .output_key("shared")
            .build();
        let b = Composition::single_builder("b", Box::new(DummyAgent("b")))
            .output_key("shared")
            .build();
        let _ = Composition::parallel_builder()
            .branch("ka", a)
            .branch("kb", b)
            .build();
    }

    #[test]
    fn parallel_builder_distinct_output_keys_ok() {
        let a = Composition::single_builder("a", Box::new(DummyAgent("a")))
            .output_key("ka")
            .build();
        let b = Composition::single_builder("b", Box::new(DummyAgent("b")))
            .output_key("kb")
            .build();
        let c = Composition::parallel_builder()
            .branch("ba", a)
            .branch("bb", b)
            .build();
        assert_eq!(c.child_count(), 2);
    }

    #[test]
    fn restart_policy_clamps_max_to_at_least_one() {
        match RestartPolicy::on_failure(0) {
            RestartPolicy::OnFailure { max_attempts, .. } => assert_eq!(max_attempts, 1),
            _ => panic!(),
        }
    }

    // ── Backoff math (jitter-free for determinism) ─────────────────────

    #[test]
    fn backoff_none_is_always_zero() {
        let b = Backoff::None;
        for n in 1..=5 {
            assert_eq!(b.delay_for_attempt(n), Duration::ZERO);
        }
    }

    #[test]
    fn backoff_fixed_returns_constant_after_first_attempt() {
        let b = Backoff::Fixed(Duration::from_millis(100));
        assert_eq!(b.delay_for_attempt(1), Duration::ZERO);
        assert_eq!(b.delay_for_attempt(2), Duration::from_millis(100));
        assert_eq!(b.delay_for_attempt(3), Duration::from_millis(100));
    }

    #[test]
    fn backoff_exponential_doubles_until_max() {
        // jitter_ratio = 0.0 makes this fully deterministic.
        let b = Backoff::Exponential {
            base: Duration::from_millis(100),
            max: Duration::from_millis(800),
            jitter_ratio: 0.0,
        };
        assert_eq!(b.delay_for_attempt(1), Duration::ZERO);
        assert_eq!(b.delay_for_attempt(2), Duration::from_millis(100));
        assert_eq!(b.delay_for_attempt(3), Duration::from_millis(200));
        assert_eq!(b.delay_for_attempt(4), Duration::from_millis(400));
        assert_eq!(b.delay_for_attempt(5), Duration::from_millis(800));
        // Caps at max.
        assert_eq!(b.delay_for_attempt(6), Duration::from_millis(800));
        assert_eq!(b.delay_for_attempt(20), Duration::from_millis(800));
    }

    #[test]
    fn backoff_exponential_huge_attempt_does_not_overflow() {
        let b = Backoff::Exponential {
            base: Duration::from_millis(100),
            max: Duration::from_secs(10),
            jitter_ratio: 0.0,
        };
        // u8::MAX wouldn't fit in `1 << 254`; we cap the exponent inside.
        assert_eq!(b.delay_for_attempt(255), Duration::from_secs(10));
    }

    #[test]
    fn backoff_jitter_stays_below_or_at_unjittered_value() {
        // With ratio in (0, 1], the multiplier is in [1-ratio, 1.0],
        // so the jittered delay is never LONGER than the unjittered one
        // and never below `unjittered * (1 - ratio)`.
        let unjittered = Duration::from_millis(1000);
        for _ in 0..50 {
            let jittered = apply_jitter(unjittered, 0.3);
            assert!(jittered <= unjittered);
            assert!(jittered >= Duration::from_millis(700));
        }
    }

    #[test]
    fn restart_policy_on_failure_defaults_to_llm_backoff() {
        match RestartPolicy::on_failure(2) {
            RestartPolicy::OnFailure { backoff, .. } => match backoff {
                Backoff::Exponential {
                    base, jitter_ratio, ..
                } => {
                    assert_eq!(base, Duration::from_millis(200));
                    assert!((jitter_ratio - 0.3).abs() < 0.001);
                }
                other => panic!("expected default-for-llm exponential, got {other:?}"),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn without_backoff_disables_delay() {
        let p = RestartPolicy::on_failure(3).without_backoff();
        assert!(matches!(p.backoff(), Backoff::None));
        assert_eq!(p.max_attempts(), 3);
    }

    #[test]
    fn with_backoff_overrides_default() {
        let custom = Backoff::Fixed(Duration::from_millis(50));
        let p = RestartPolicy::on_failure(3).with_backoff(custom);
        assert!(matches!(p.backoff(), Backoff::Fixed(d) if d == Duration::from_millis(50)));
    }

    #[test]
    fn with_backoff_on_never_is_a_no_op() {
        // `Never` has no attempts to space out, so the variant is unchanged.
        let p = RestartPolicy::Never.with_backoff(Backoff::Fixed(Duration::from_secs(99)));
        assert!(matches!(p, RestartPolicy::Never));
    }

    #[test]
    fn loop_until_max_iterations_at_least_one() {
        let c = Composition::loop_until_builder(|_| dummy("a"), |_| true)
            .max_iterations(0)
            .build();
        match c {
            Composition::LoopUntil { max_iterations, .. } => assert_eq!(max_iterations, 1),
            _ => panic!(),
        }
    }

    #[test]
    fn label_reflects_shape_and_size() {
        assert_eq!(dummy("a").label(), "a");
        let seq = Composition::sequential_builder()
            .step(dummy("a"))
            .step(dummy("b"))
            .build();
        assert_eq!(seq.label(), "sequential(2)");
        let par = Composition::parallel_builder()
            .branch("x", dummy("a"))
            .branch("y", dummy("b"))
            .build();
        assert_eq!(par.label(), "parallel(2)");
    }
}
