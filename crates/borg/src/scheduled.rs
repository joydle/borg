//! `SubAgentHandle` and `Scheduled` — background-spawn sub-agent
//! dispatches that complete out-of-band with the caller.
//!
//! `Composition::Parallel` is fire-all-then-collect-all. These types
//! serve callers that can't block: spawn one or more dispatches at a
//! turn boundary, continue working, and drain results at the next
//! boundary (or later, or never if the user cancels).
//!
//! ## Two layers
//!
//! - [`SubAgentHandle`] — primitive: one spawn, one handle.
//!   `try_await()` is non-blocking; drop cancels.
//! - [`Scheduled`] — convenience over N handles keyed by name.
//!   `collect_ready()` non-blocking drain, `take(name)` consumes the
//!   buffered result, `wait_for(name, timeout)` blocks up to a budget.
//!   Drop cancels all in-flight branches.
//!
//! [`ExecCtx::spawn_sub_agent`] routes an agent through
//! `run_one_with_restart`, so budget gating, lifecycle events, retry
//! policy, circuit breaker, and meltdown ledger all behave identically
//! to a synchronous dispatch, and inner `StepRetrying` / `StepCompleted`
//! events surface the same way.
//!
//! ## Any output, not only a `Verdict`
//!
//! Both types are generic over what a spawn produces, defaulting to
//! [`Verdict`]. A consumer whose background work is "run a sub-agent,
//! then turn its verdict into a domain value" spawns that
//! whole body with [`SubAgentHandle::spawn`] / [`Scheduled::spawn`] and
//! gets its own type back — same cancel, drain and drop semantics.
//!
//! ## Work that outlives a turn
//!
//! A consumer that rebuilds its per-turn state every turn (dropping the
//! old one) would cancel whatever was still running with it. [`Scheduled::detach`]
//! moves the live group out — in-flight handles, buffered results and
//! failures, the completion hook — and [`Scheduled::absorb`] adopts it
//! into the next turn's group. Nothing is cancelled on the way.
//!
//! ## Accounting for every completion
//!
//! [`Scheduled::on_complete`] installs a hook that sees each completion
//! exactly once, at the moment the group observes it — whichever drain
//! that is ([`Scheduled::collect_ready`], [`Scheduled::wait_for`],
//! [`Scheduled::wait_until_settled`], [`Scheduled::cancel_all_and_reap`]),
//! and even when a later run of the same name overwrites the buffered
//! result. It is the place to queue a charge for work that cost money.
//!
//! ## Cancel-on-drop
//!
//! Both `Drop` impls cancel the per-spawn token. Forgetting to call
//! `cancel_all` does not leak a running LLM call past the
//! orchestrator's lifetime.

use crate::composition::RestartPolicy;
use crate::error::BorgError;
use crate::executor::{run_one_with_restart, ExecCtx};
use crate::handoff::HandoffState;
use crate::verdict::Verdict;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;

/// Handle to one spawned dispatch.
///
/// Wraps the Tokio `JoinHandle` plus a cancel token that is a *child*
/// of the parent token (for [`ExecCtx::spawn_sub_agent`], the parent
/// `ExecCtx.cancel`): the parent's cancel propagates here, and the
/// handle's own drop fires the child token, isolating this spawn's
/// cancellation from its siblings.
///
/// The handle owns the right to abort — dropping it aborts the spawn.
/// To let the work outlive the caller's scope, keep the handle alive
/// or call [`Self::detach`].
///
/// A spawn that panics surfaces as [`BorgError::ChildPanicked`]; one
/// aborted before it finished (only a drop or [`Scheduled::cancel_all`]
/// does that) as [`BorgError::Cancelled`].
#[must_use = "SubAgentHandle aborts the spawn on drop; bind it or call .detach()"]
pub struct SubAgentHandle<T = Verdict> {
    name: String,
    inner: JoinHandle<Result<T, BorgError>>,
    cancel: CancellationToken,
    detached: bool,
    /// True once `try_await` has returned `Some(_)`. Polling `inner`
    /// again would panic (Tokio rejects poll-after-Ready), so we track
    /// consumption locally and let `try_await` be called repeatedly.
    consumed: bool,
}

/// What a join failure means, named after the spawn it belonged to.
fn join_failure(name: &str, join_err: JoinError) -> BorgError {
    if join_err.is_panic() {
        let payload = join_err.into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_string());
        BorgError::ChildPanicked {
            agent_id: name.to_string(),
            message,
        }
    } else {
        BorgError::Cancelled
    }
}

impl<T: Send + 'static> SubAgentHandle<T> {
    /// Spawn arbitrary background work under a child of `parent`.
    ///
    /// `work` receives the child token; it should `select!` its long
    /// awaits against it (the handle's drop and the parent's cancel
    /// both fire it). For an agent dispatch with retry, budget and
    /// events, use [`ExecCtx::spawn_sub_agent`] instead.
    pub fn spawn<F, Fut>(name: impl Into<String>, parent: &CancellationToken, work: F) -> Self
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = Result<T, BorgError>> + Send + 'static,
    {
        // Runs on a task of its own: carry the caller's span and every
        // process-wide carrier (task-locals) into it.
        let context = crate::propagation::capture(&crate::propagation::Carriers::new());
        Self::spawn_in(name, parent, context, work)
    }

    /// [`Self::spawn`] inside an explicitly captured context — e.g.
    /// [`ExecCtx::capture_context`], so the work also sees that
    /// execution's own [`ExecCtx::with_carrier`] carriers.
    pub fn spawn_in<F, Fut>(
        name: impl Into<String>,
        parent: &CancellationToken,
        context: crate::propagation::CapturedContext,
        work: F,
    ) -> Self
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = Result<T, BorgError>> + Send + 'static,
    {
        let cancel = parent.child_token();
        let inner = tokio::spawn(context.scope(work(cancel.clone())));
        SubAgentHandle {
            name: name.into(),
            inner,
            cancel,
            detached: false,
            consumed: false,
        }
    }

    /// The name this spawn was started under.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Non-blocking poll. `Some(result)` if the spawn has completed
    /// (success or failure), `None` if still running. After `Some` is
    /// returned the handle is exhausted; store the result if needed.
    pub fn try_await(&mut self) -> Option<Result<T, BorgError>> {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        match self.poll_result(&mut cx) {
            Poll::Ready(r) => Some(r),
            Poll::Pending => None,
        }
    }

    /// Poll with a real waker: `Ready` once, then `Pending` forever
    /// (the result is not kept).
    fn poll_result(&mut self, cx: &mut Context<'_>) -> Poll<Result<T, BorgError>> {
        if self.consumed {
            return Poll::Pending;
        }
        match Pin::new(&mut self.inner).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(joined) => {
                self.consumed = true;
                Poll::Ready(match joined {
                    Ok(result) => result,
                    Err(join_err) => Err(join_failure(&self.name, join_err)),
                })
            }
        }
    }

    /// Whether the spawn has finished, without consuming the result.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Block until the spawn completes (or the cancel token fires).
    /// The handle is consumed.
    pub async fn await_completion(self) -> Result<T, BorgError> {
        // Suppress drop-cancel: we're awaiting to completion here.
        let mut this = self;
        this.detached = true;
        let name = std::mem::take(&mut this.name);
        match (&mut this.inner).await {
            Ok(result) => result,
            Err(join_err) => Err(join_failure(&name, join_err)),
        }
    }

    /// Detach the spawn: the work runs to completion regardless of the
    /// handle's fate. Use when the caller won't keep the handle but
    /// still wants the work to finish (e.g. background telemetry that
    /// publishes its own result via the `EventEmitter`).
    pub fn detach(mut self) {
        self.detached = true;
    }

    /// Cancel this spawn. The task observes it on its next `select!`
    /// against its token.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl<T> Drop for SubAgentHandle<T> {
    fn drop(&mut self) {
        if !self.detached {
            self.cancel.cancel();
            self.inner.abort();
        }
    }
}

/// Called once per completion a [`Scheduled`] observes. `Sync` so the group
/// stays `Sync`: consumers hold it in state that is borrowed across an
/// `.await` inside `Send` futures.
type CompletionHook<T> = Box<dyn FnMut(&str, &Result<T, BorgError>) + Send + Sync>;

/// Group handle: spawn N background dispatches, drain results by name
/// as they finish, cancel everything on drop. The usual pattern: spawn
/// at turn boundary T, drain at T+1 (or later), cancel-all at session
/// end.
///
/// Several spawns may share a name (two runs of one reviewer in a
/// turn): all of them run, the buffer keeps the most recent success and
/// the most recent failure per name, and [`Self::on_complete`] sees
/// every one.
#[must_use = "Scheduled cancels all in-flight spawns on drop"]
pub struct Scheduled<T = Verdict> {
    handles: Vec<SubAgentHandle<T>>,
    ready: HashMap<String, T>,
    failures: HashMap<String, BorgError>,
    on_complete: Option<CompletionHook<T>>,
}

impl<T> Default for Scheduled<T> {
    fn default() -> Self {
        Self {
            handles: Vec::new(),
            ready: HashMap::new(),
            failures: HashMap::new(),
            on_complete: None,
        }
    }
}

impl Scheduled<Verdict> {
    /// Spawn an agent dispatch into the group — [`ExecCtx::spawn_sub_agent`]
    /// + [`Self::push`] in one call.
    pub fn spawn_agent<A>(
        &mut self,
        name: impl Into<String>,
        ctx: ExecCtx,
        agent: A,
        restart: RestartPolicy,
        handoff: HandoffState,
    ) where
        A: crate::agent::BorgAgent + 'static,
    {
        let name = name.into();
        let handle = ctx.spawn_sub_agent(name.clone(), agent, restart, handoff);
        self.push(name, handle);
    }
}

impl<T: Send + 'static> Scheduled<T> {
    /// Construct from an iterator of `(name, SubAgentHandle)` pairs,
    /// typically keyed off the caller's domain enum.
    pub fn new<I, S>(items: I) -> Self
    where
        I: IntoIterator<Item = (S, SubAgentHandle<T>)>,
        S: Into<String>,
    {
        let mut group = Self::default();
        for (name, handle) in items {
            group.push(name, handle);
        }
        group
    }

    /// Builder: run `hook` on every completion this group observes,
    /// exactly once each, on the thread that drains (never inside the
    /// spawned task). See the module docs.
    pub fn on_complete(
        mut self,
        hook: impl FnMut(&str, &Result<T, BorgError>) + Send + Sync + 'static,
    ) -> Self {
        self.on_complete = Some(Box::new(hook));
        self
    }

    /// Install or replace the completion hook on an existing group.
    pub fn set_on_complete(
        &mut self,
        hook: impl FnMut(&str, &Result<T, BorgError>) + Send + Sync + 'static,
    ) {
        self.on_complete = Some(Box::new(hook));
    }

    /// Add another spawn to the group after construction. A spawn already
    /// running under the same name keeps running beside it.
    pub fn push(&mut self, name: impl Into<String>, mut handle: SubAgentHandle<T>) {
        handle.name = name.into();
        self.handles.push(handle);
    }

    /// Spawn arbitrary work into the group under a child of `parent`
    /// ([`SubAgentHandle::spawn`] + [`Self::push`]).
    pub fn spawn<F, Fut>(&mut self, name: impl Into<String>, parent: &CancellationToken, work: F)
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = Result<T, BorgError>> + Send + 'static,
    {
        let handle = SubAgentHandle::spawn(name, parent, work);
        self.handles.push(handle);
    }

    /// Buffer one completion and tell the hook.
    fn accept(&mut self, name: String, result: Result<T, BorgError>) {
        if let Some(hook) = self.on_complete.as_mut() {
            hook(&name, &result);
        }
        match result {
            Ok(v) => {
                self.ready.insert(name, v);
            }
            Err(e) => {
                self.failures.insert(name, e);
            }
        }
    }

    /// Non-blocking drain: poll every in-flight handle once, move any
    /// finished results into the ready buffer. Returns the names that
    /// moved this call (a name twice if two of its runs finished).
    pub fn collect_ready(&mut self) -> Vec<String> {
        let mut newly_ready = Vec::new();
        let mut i = 0;
        while i < self.handles.len() {
            if let Some(result) = self.handles[i].try_await() {
                let handle = self.handles.remove(i);
                let name = handle.name.clone();
                drop(handle); // finished: its drop aborts nothing
                newly_ready.push(name.clone());
                self.accept(name, result);
            } else {
                i += 1;
            }
        }
        newly_ready
    }

    /// Take a buffered result by name: the success if one is buffered,
    /// else the failure. Returns `None` if either:
    /// - the name was never spawned, or
    /// - the spawn is still in flight (call `collect_ready` first), or
    /// - the result was already taken.
    pub fn take(&mut self, name: &str) -> Option<Result<T, BorgError>> {
        if let Some(v) = self.ready.remove(name) {
            return Some(Ok(v));
        }
        self.failures.remove(name).map(Err)
    }

    /// Look at a buffered success without taking it.
    pub fn peek(&self, name: &str) -> Option<&T> {
        self.ready.get(name)
    }

    /// Take the buffered failure for `name` — the last run of it that
    /// died (an error, a panic, a cancel) instead of producing a value.
    /// A buffered success for the same name stays.
    pub fn take_failure(&mut self, name: &str) -> Option<BorgError> {
        self.failures.remove(name)
    }

    /// Look at the buffered failure for `name` without taking it.
    pub fn peek_failure(&self, name: &str) -> Option<&BorgError> {
        self.failures.get(name)
    }

    /// Block up to `timeout` for the named spawn. Returns `None` on
    /// timeout (the spawn stays in flight for a later `collect_ready`)
    /// or if the name wasn't found. Siblings that finish during the
    /// wait are NOT drained here — call `collect_ready()` after (or use
    /// [`Self::wait_until_settled`], which does).
    ///
    /// The result is returned, not buffered; the completion hook still
    /// sees it.
    ///
    /// Cancel-safe: the handle stays in the group while it is polled, so
    /// dropping this future (an outer timeout, a `select!` branch) leaves
    /// the spawn running for a later drain.
    pub async fn wait_for(
        &mut self,
        name: &str,
        timeout: std::time::Duration,
    ) -> Option<Result<T, BorgError>> {
        if let Some(r) = self.take(name) {
            return Some(r);
        }
        if !self.is_inflight(name) {
            return None;
        }
        let handles = &mut self.handles;
        let waited = tokio::time::timeout(
            timeout,
            std::future::poll_fn(|cx| {
                // Poll the handle where it sits; it never leaves the
                // group while pending.
                let Some(idx) = handles.iter().position(|h| h.name == name) else {
                    return Poll::Ready(None);
                };
                handles[idx].poll_result(cx).map(|r| Some((idx, r)))
            }),
        )
        .await;
        // Timed out: the handle never left the group, so a later
        // collect_ready can still pick it up.
        let (idx, result) = waited.ok()??;
        // Finished: removing (and dropping) it aborts nothing.
        drop(self.handles.remove(idx));
        if let Some(hook) = self.on_complete.as_mut() {
            hook(name, &result);
        }
        Some(result)
    }

    /// Block up to `timeout` until a run of `name` completes, buffering
    /// EVERY completion that lands meanwhile — siblings included, in the
    /// order they finish — exactly as [`Self::collect_ready`] would.
    ///
    /// Returns `true` when `name` settled: a success for it is already
    /// buffered, a run of it completed during the wait (success or
    /// failure), or nothing named `name` is in flight. `false` on
    /// timeout. The result stays buffered for [`Self::take`] /
    /// [`Self::take_failure`].
    pub async fn wait_until_settled(&mut self, name: &str, timeout: std::time::Duration) -> bool {
        if self.ready.contains_key(name) || !self.is_inflight(name) {
            return true;
        }
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let next = tokio::time::timeout_at(
                deadline,
                std::future::poll_fn(|cx| {
                    for (i, h) in self.handles.iter_mut().enumerate() {
                        if let Poll::Ready(r) = h.poll_result(cx) {
                            return Poll::Ready((i, r));
                        }
                    }
                    Poll::Pending
                }),
            )
            .await;
            let Ok((idx, result)) = next else {
                return false;
            };
            let handle = self.handles.remove(idx);
            let done = handle.name.clone();
            drop(handle);
            self.accept(done.clone(), result);
            if done == name || !self.is_inflight(name) {
                return true;
            }
        }
    }

    /// Cancel every in-flight spawn. Idempotent. Called automatically
    /// by `Drop`. Nothing is drained: a spawn that had already finished
    /// is dropped with its result — see [`Self::cancel_all_and_reap`].
    pub fn cancel_all(&mut self) {
        for h in self.handles.drain(..) {
            h.cancel();
            // Drop completes the abort.
            drop(h);
        }
    }

    /// Cancel every in-flight spawn, then collect the ones that had
    /// ALREADY FINISHED before the abort reached them: each goes through
    /// the buffer and the completion hook like any drain. Spawns the
    /// abort actually stopped are discarded silently (they produced
    /// nothing). Returns how many finished results were rescued.
    ///
    /// The session-end step for work that costs money: a spawn whose
    /// result landed after the last drain still spent it, and this is
    /// the only place it would ever be seen. Bound it with a timeout on
    /// a latency-sensitive path — dropping the future mid-way loses only
    /// results not yet reached.
    pub async fn cancel_all_and_reap(&mut self) -> usize {
        for h in &self.handles {
            h.cancel();
            h.inner.abort();
        }
        let mut rescued = 0;
        while !self.handles.is_empty() {
            let mut handle = self.handles.remove(0);
            handle.detached = true; // already aborted; drop has nothing to do
            let name = std::mem::take(&mut handle.name);
            match (&mut handle.inner).await {
                Ok(result) => {
                    rescued += 1;
                    self.accept(name, result);
                }
                Err(join_err) if join_err.is_panic() => {
                    rescued += 1;
                    let e = join_failure(&name, join_err);
                    self.accept(name, Err(e));
                }
                Err(_aborted) => {}
            }
        }
        rescued
    }

    /// Move the whole live group out — in-flight spawns, buffered
    /// results and failures, the completion hook — leaving `self` empty
    /// with no hook. Nothing is cancelled. Pair with [`Self::absorb`] to
    /// carry work across a state rebuild.
    pub fn detach(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Adopt a detached group: its in-flight spawns join this one's and
    /// keep running; its buffered results and failures fill names this
    /// group has none for. Nothing is cancelled.
    ///
    /// Hooks: `self`'s hook, when it has one, observes everything
    /// afterwards (the carried group's hook is dropped); otherwise the
    /// carried group's hook is adopted.
    pub fn absorb(&mut self, mut carried: Self) {
        self.handles.append(&mut carried.handles);
        for (name, v) in carried.ready.drain() {
            self.ready.entry(name).or_insert(v);
        }
        for (name, e) in carried.failures.drain() {
            self.failures.entry(name).or_insert(e);
        }
        if self.on_complete.is_none() {
            self.on_complete = carried.on_complete.take();
        }
    }

    /// Number of in-flight spawns (not yet collected).
    pub fn inflight_count(&self) -> usize {
        self.handles.len()
    }

    /// Number of buffered completions (successes and failures) waiting
    /// to be `take`n.
    pub fn ready_count(&self) -> usize {
        self.ready.len() + self.failures.len()
    }

    /// Whether any spawn named `name` is in flight (not yet collected).
    pub fn is_inflight(&self, name: &str) -> bool {
        self.handles.iter().any(|h| h.name == name)
    }

    /// Whether a completion for `name` (success or failure) is buffered.
    pub fn is_ready(&self, name: &str) -> bool {
        self.ready.contains_key(name) || self.failures.contains_key(name)
    }
}

impl<T> Drop for Scheduled<T> {
    fn drop(&mut self) {
        for h in self.handles.drain(..) {
            h.cancel.cancel();
            drop(h);
        }
    }
}

// ── ExecCtx extension ─────────────────────────────────────────────────

impl ExecCtx {
    /// Spawn a sub-agent dispatch on a Tokio task; returns immediately
    /// with a [`SubAgentHandle`].
    ///
    /// Runs through `run_one_with_restart`, so budget gating, lifecycle
    /// events, retry policy, and circuit breaker behave identically to
    /// a synchronous dispatch. The spawn's cancel token is a child of
    /// `self.cancel`: the parent's cancel propagates, and `handle.drop`
    /// cancels independently.
    ///
    /// Poll the handle (`try_await`, `await_completion`, or via
    /// `Scheduled`) to observe the result; dropping it unawaited
    /// cancels the spawn cleanly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let h = ctx.clone().spawn_sub_agent(
    ///     "error_diagnoser",
    ///     ErrorDiagnoserAgent,
    ///     RestartPolicy::on_failure(2),
    ///     handoff,
    /// );
    /// // ... do other work ...
    /// if let Some(result) = h.try_await() {
    ///     // handle the diagnosis
    /// }
    /// ```
    pub fn spawn_sub_agent<A>(
        self,
        name: impl Into<String>,
        agent: A,
        restart: RestartPolicy,
        handoff: HandoffState,
    ) -> SubAgentHandle
    where
        A: crate::agent::BorgAgent + 'static,
    {
        let cancel = self.cancel.child_token();
        let mut child_ctx = self;
        child_ctx.cancel = cancel.clone();
        let name = name.into();
        let task_name = name.clone();
        // The caller's span, process-wide carriers and this context's own.
        let context = child_ctx.capture_context();
        let inner = tokio::spawn(context.scope(async move {
            run_one_with_restart(task_name, agent, restart, child_ctx, handoff).await
        }));
        SubAgentHandle {
            name,
            inner,
            cancel,
            detached: false,
            consumed: false,
        }
    }
}

// ── Noop waker helper ─────────────────────────────────────────────────
//
// `JoinHandle::poll` needs a `Context`/`Waker`, but `try_await` is a
// one-shot poll — the caller re-polls later — so no real wake is needed.

fn noop_waker() -> std::task::Waker {
    std::task::Waker::noop().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::BorgAgent;
    use crate::budget::in_memory_budget;
    use crate::events::noop_emitter;
    use crate::handoff::HandoffState;
    use crate::llm::deterministic_llm_client;
    use std::time::Duration;

    /// Deterministic agent that emits a fixed verdict.
    struct EchoAgent {
        token: &'static str,
        body: &'static str,
    }

    #[async_trait::async_trait]
    impl BorgAgent for EchoAgent {
        fn agent_id(&self) -> &'static str {
            "echo"
        }

        async fn invoke(&self, _ctx: &crate::context::AgentContext) -> Result<Verdict, BorgError> {
            Ok(Verdict {
                token: self.token.to_string(),
                body: self.body.into(),
                token_is_recognized: true,
                ..Verdict::default()
            })
        }
    }

    /// Agent that sleeps, so tests can observe in-flight and cancel.
    struct SlowAgent {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl BorgAgent for SlowAgent {
        fn agent_id(&self) -> &'static str {
            "slow"
        }

        async fn invoke(&self, ctx: &crate::context::AgentContext) -> Result<Verdict, BorgError> {
            tokio::select! {
                _ = tokio::time::sleep(self.delay) => Ok(Verdict::default()),
                _ = ctx.cancel.cancelled() => Err(BorgError::Cancelled),
            }
        }
    }

    fn test_ctx() -> ExecCtx {
        ExecCtx::new(
            deterministic_llm_client(),
            in_memory_budget(10_000),
            noop_emitter(),
            CancellationToken::new(),
            "test",
        )
    }

    #[tokio::test]
    async fn spawn_sub_agent_returns_completed_verdict_via_await() {
        let ctx = test_ctx();
        let handle = ctx.spawn_sub_agent(
            "echo",
            EchoAgent {
                token: "PASS",
                body: "hi",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let verdict = handle.await_completion().await.expect("verdict");
        assert_eq!(verdict.token, "PASS");
        assert_eq!(&*verdict.body, "hi");
    }

    #[tokio::test]
    async fn try_await_is_none_while_inflight_then_some_when_done() {
        let ctx = test_ctx();
        let mut handle = ctx.spawn_sub_agent(
            "slow",
            SlowAgent {
                delay: Duration::from_millis(50),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        assert!(handle.try_await().is_none());
        tokio::time::sleep(Duration::from_millis(120)).await;
        let r = handle.try_await().expect("complete");
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn drop_cancels_inflight_spawn() {
        let cancel = CancellationToken::new();
        let mut ctx = test_ctx();
        ctx.cancel = cancel.clone();
        let handle = ctx.spawn_sub_agent(
            "slow",
            SlowAgent {
                delay: Duration::from_secs(30),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        drop(handle);
        // Give the task a moment to observe the cancel and exit. The
        // test passes if it doesn't hang for 30 seconds.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn detach_lets_spawn_complete_after_handle_dropped() {
        let ctx = test_ctx();
        let handle = ctx.spawn_sub_agent(
            "echo",
            EchoAgent {
                token: "PASS",
                body: "x",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        handle.detach();
        // Verifies detach() doesn't cancel; the task runs to completion.
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn scheduled_collects_ready_results() {
        let ctx = test_ctx();
        let h1 = ctx.clone().spawn_sub_agent(
            "a",
            EchoAgent {
                token: "PASS",
                body: "a",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let h2 = ctx.spawn_sub_agent(
            "b",
            EchoAgent {
                token: "PASS",
                body: "b",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let mut scheduled = Scheduled::new([("a".to_string(), h1), ("b".to_string(), h2)]);
        tokio::time::sleep(Duration::from_millis(30)).await;
        let ready = scheduled.collect_ready();
        assert_eq!(ready.len(), 2);
        assert_eq!(scheduled.ready_count(), 2);
        assert_eq!(scheduled.inflight_count(), 0);
        let a = scheduled.take("a").expect("a ready");
        assert_eq!(&*a.expect("ok").body, "a");
    }

    #[tokio::test]
    async fn scheduled_wait_for_returns_timeout_then_collects_later() {
        let ctx = test_ctx();
        let h = ctx.spawn_sub_agent(
            "slow",
            SlowAgent {
                delay: Duration::from_millis(100),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let mut scheduled = Scheduled::new([("slow", h)]);
        let r = scheduled.wait_for("slow", Duration::from_millis(10)).await;
        assert!(r.is_none(), "expected timeout");
        assert_eq!(scheduled.inflight_count(), 1);
        let r = scheduled.wait_for("slow", Duration::from_millis(200)).await;
        assert!(r.is_some(), "expected completion");
    }

    #[tokio::test]
    async fn dropped_wait_for_future_keeps_the_spawn_running() {
        // Regression: `wait_for` removed the handle before awaiting, so a
        // caller-side timeout (or `select!`) that dropped the future also
        // dropped the handle, aborting the spawn and losing its result.
        let ctx = test_ctx();
        let h = ctx.spawn_sub_agent(
            "slow",
            SlowAgent {
                delay: Duration::from_millis(80),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let mut scheduled = Scheduled::new([("slow", h)]);
        let outer = tokio::time::timeout(
            Duration::from_millis(10),
            scheduled.wait_for("slow", Duration::from_secs(10)),
        )
        .await;
        assert!(outer.is_err(), "the caller's timeout fires first");
        assert!(
            scheduled.is_inflight("slow"),
            "the spawn stays in the group"
        );
        assert!(
            scheduled
                .wait_until_settled("slow", Duration::from_secs(2))
                .await,
            "the spawn still completes"
        );
        let r = scheduled.take("slow").expect("result retrievable");
        assert!(r.is_ok(), "completed, not aborted: {r:?}");
    }

    #[tokio::test]
    async fn scheduled_drop_cancels_inflight() {
        let ctx = test_ctx();
        let h = ctx.spawn_sub_agent(
            "slow",
            SlowAgent {
                delay: Duration::from_secs(30),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let scheduled = Scheduled::new([("slow", h)]);
        drop(scheduled);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    #[tokio::test]
    async fn scheduled_take_returns_none_for_unknown_name() {
        let scheduled = Scheduled::new(std::iter::empty::<(String, SubAgentHandle)>());
        let mut scheduled = scheduled;
        assert!(scheduled.take("does_not_exist").is_none());
    }

    #[tokio::test]
    async fn scheduled_push_adds_after_construction() {
        let ctx = test_ctx();
        let mut scheduled = Scheduled::new(std::iter::empty::<(String, SubAgentHandle)>());
        let h = ctx.spawn_sub_agent(
            "added_later",
            EchoAgent {
                token: "PASS",
                body: "x",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        scheduled.push("added_later", h);
        assert!(scheduled.is_inflight("added_later"));
        tokio::time::sleep(Duration::from_millis(30)).await;
        let _ = scheduled.collect_ready();
        assert!(scheduled.is_ready("added_later"));
    }

    #[tokio::test]
    async fn try_await_after_completion_returns_none_on_second_call() {
        let ctx = test_ctx();
        let mut handle = ctx.spawn_sub_agent(
            "echo",
            EchoAgent {
                token: "PASS",
                body: "once",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
        let first = handle.try_await();
        assert!(first.is_some());
        assert!(handle.try_await().is_none());
        handle.detach();
    }

    #[tokio::test]
    async fn parent_cancel_propagates_to_spawn() {
        let parent_cancel = CancellationToken::new();
        let mut ctx = test_ctx();
        ctx.cancel = parent_cancel.clone();
        let handle = ctx.spawn_sub_agent(
            "slow",
            SlowAgent {
                delay: Duration::from_secs(30),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        parent_cancel.cancel();
        let r = handle.await_completion().await;
        assert!(matches!(r, Err(BorgError::Cancelled)));
    }

    #[tokio::test]
    async fn cancel_all_drains_in_flight() {
        let ctx = test_ctx();
        let h1 = ctx.clone().spawn_sub_agent(
            "a",
            SlowAgent {
                delay: Duration::from_secs(30),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let h2 = ctx.spawn_sub_agent(
            "b",
            SlowAgent {
                delay: Duration::from_secs(30),
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let mut scheduled = Scheduled::new([("a".to_string(), h1), ("b".to_string(), h2)]);
        assert_eq!(scheduled.inflight_count(), 2);
        scheduled.cancel_all();
        assert_eq!(scheduled.inflight_count(), 0);
    }

    #[tokio::test]
    async fn is_finished_reflects_state() {
        let ctx = test_ctx();
        let handle = ctx.spawn_sub_agent(
            "echo",
            EchoAgent {
                token: "PASS",
                body: "x",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(handle.is_finished());
        handle.detach();
    }

    #[tokio::test]
    async fn wait_for_fast_path_returns_buffered_result() {
        let ctx = test_ctx();
        let h = ctx.spawn_sub_agent(
            "echo",
            EchoAgent {
                token: "PASS",
                body: "buffered",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        let mut scheduled = Scheduled::new([("echo", h)]);
        tokio::time::sleep(Duration::from_millis(30)).await;
        let _ = scheduled.collect_ready();
        // wait_for should fast-path return from the ready buffer.
        let r = scheduled
            .wait_for("echo", Duration::from_millis(1))
            .await
            .expect("buffered");
        assert_eq!(&*r.expect("ok").body, "buffered");
    }

    // ── generic spawns, hooks, detach/absorb, reap ────────────────────

    /// A value a consumer's background body produces (not a `Verdict`).
    #[derive(Debug, PartialEq)]
    struct Priced {
        body: &'static str,
        cost_micros: u64,
    }

    fn priced_after(
        group: &mut Scheduled<Priced>,
        parent: &CancellationToken,
        name: &str,
        body: &'static str,
        cost_micros: u64,
        delay: Duration,
    ) {
        group.spawn(name, parent, move |cancel| async move {
            tokio::select! {
                _ = tokio::time::sleep(delay) => Ok(Priced { body, cost_micros }),
                _ = cancel.cancelled() => Err(BorgError::Cancelled),
            }
        });
    }

    type Seen = std::sync::Arc<std::sync::Mutex<Vec<(String, Result<u64, String>)>>>;

    fn recording_hook(seen: &Seen) -> impl FnMut(&str, &Result<Priced, BorgError>) + Send + Sync {
        let seen = seen.clone();
        move |name, r| {
            seen.lock().unwrap().push((
                name.to_string(),
                r.as_ref().map(|p| p.cost_micros).map_err(|e| e.to_string()),
            ))
        }
    }

    /// A group with a hook is still `Send + Sync`: consumers keep it in state
    /// that a `Send` future borrows across an `.await`.
    #[test]
    fn a_group_with_a_hook_is_send_and_sync() {
        fn assert_send_sync<X: Send + Sync>(_: &X) {}
        let group: Scheduled<Priced> = Scheduled::default().on_complete(|_, _| {});
        assert_send_sync(&group);
        assert_send_sync(&Scheduled::<Verdict>::default());
    }

    #[tokio::test]
    async fn a_generic_spawn_hands_back_the_consumers_own_type() {
        let parent = CancellationToken::new();
        let h = SubAgentHandle::spawn("review", &parent, |_| async {
            Ok(Priced {
                body: "ok",
                cost_micros: 40,
            })
        });
        assert_eq!(h.name(), "review");
        let v = h.await_completion().await.expect("value");
        assert_eq!(
            v,
            Priced {
                body: "ok",
                cost_micros: 40
            }
        );
    }

    /// Two runs of one name in a turn: both run to completion, the buffer
    /// keeps the later one, and the hook sees BOTH — the overwritten run
    /// still cost money and must still be charged.
    #[tokio::test]
    async fn two_runs_of_one_name_both_finish_and_both_reach_the_hook() {
        let parent = CancellationToken::new();
        let seen: Seen = Default::default();
        let mut group = Scheduled::default().on_complete(recording_hook(&seen));
        priced_after(
            &mut group,
            &parent,
            "qa",
            "first",
            10,
            Duration::from_millis(5),
        );
        priced_after(
            &mut group,
            &parent,
            "qa",
            "second",
            20,
            Duration::from_millis(40),
        );
        assert_eq!(
            group.inflight_count(),
            2,
            "a same-named push aborts nothing"
        );

        assert!(group.wait_until_settled("qa", Duration::from_secs(5)).await);
        assert!(group.is_inflight("qa"), "the second run is still going");
        assert!(
            group
                .wait_until_settled("qa", Duration::from_millis(1))
                .await,
            "a buffered success settles the name at once"
        );
        tokio::time::sleep(Duration::from_millis(80)).await;
        group.collect_ready();
        assert!(!group.is_inflight("qa"));

        assert_eq!(group.peek("qa").map(|p| p.body), Some("second"));
        let charged: Vec<u64> = seen
            .lock()
            .unwrap()
            .iter()
            .map(|(_, r)| *r.as_ref().unwrap())
            .collect();
        assert_eq!(charged, vec![10, 20]);
    }

    /// The per-turn rebuild: the group is detached from the state that is
    /// about to drop, survives that drop, and is absorbed by the next
    /// turn's group, where an ordinary drain finds the result.
    #[tokio::test]
    async fn a_detached_group_survives_the_drop_and_lands_in_the_absorber() {
        let parent = CancellationToken::new();
        let seen: Seen = Default::default();
        let mut first = Scheduled::default().on_complete(recording_hook(&seen));
        priced_after(
            &mut first,
            &parent,
            "critic",
            "survived",
            7,
            Duration::from_millis(60),
        );

        let carried = first.detach();
        assert_eq!(first.inflight_count(), 0, "the hollow keeps nothing");
        assert!(!first.is_inflight("critic"));
        drop(first);

        let mut next: Scheduled<Priced> = Scheduled::default();
        next.absorb(carried);
        assert!(next.is_inflight("critic"), "the work moved");
        assert!(
            next.wait_until_settled("critic", Duration::from_secs(5))
                .await
        );
        assert_eq!(next.take("critic").unwrap().unwrap().body, "survived");
        assert_eq!(seen.lock().unwrap().len(), 1, "the carried hook came along");
    }

    #[tokio::test]
    async fn absorb_keeps_the_absorbers_hook_and_fills_only_empty_names() {
        let parent = CancellationToken::new();
        let mine: Seen = Default::default();
        let theirs: Seen = Default::default();
        let mut carried = Scheduled::default().on_complete(recording_hook(&theirs));
        priced_after(&mut carried, &parent, "a", "carried-a", 1, Duration::ZERO);
        priced_after(&mut carried, &parent, "b", "carried-b", 2, Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(20)).await;
        carried.collect_ready();
        let mut next = Scheduled::default().on_complete(recording_hook(&mine));
        priced_after(&mut next, &parent, "a", "own-a", 3, Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(20)).await;
        next.collect_ready();

        next.absorb(carried);
        assert_eq!(
            next.peek("a").unwrap().body,
            "own-a",
            "a buffered name is not overwritten"
        );
        assert_eq!(next.peek("b").unwrap().body, "carried-b");
        priced_after(&mut next, &parent, "c", "late", 4, Duration::ZERO);
        assert!(next.wait_until_settled("c", Duration::from_secs(5)).await);
        assert_eq!(mine.lock().unwrap().last().unwrap().0, "c");
    }

    /// A panicking run is a failure of ITS name, readable after the drain;
    /// a sibling that finished fine is untouched.
    #[tokio::test]
    async fn a_panic_is_a_failure_of_its_own_name_only() {
        let parent = CancellationToken::new();
        let mut group: Scheduled<Priced> = Scheduled::default();
        group.spawn("boom", &parent, |_| async {
            if true {
                panic!("reviewer exploded");
            }
            Ok(Priced {
                body: "never",
                cost_micros: 0,
            })
        });
        priced_after(&mut group, &parent, "fine", "ok", 1, Duration::ZERO);
        tokio::time::sleep(Duration::from_millis(30)).await;
        group.collect_ready();

        match group.peek_failure("boom") {
            Some(BorgError::ChildPanicked { agent_id, message }) => {
                assert_eq!(agent_id, "boom");
                assert!(message.contains("reviewer exploded"));
            }
            other => panic!("expected ChildPanicked, got {other:?}"),
        }
        assert!(group.take_failure("fine").is_none());
        assert_eq!(group.peek("fine").unwrap().body, "ok");
        assert!(group.take_failure("boom").is_some());
        assert!(group.take_failure("boom").is_none(), "taken once");
    }

    #[tokio::test]
    async fn take_failure_leaves_a_buffered_success_alone() {
        let parent = CancellationToken::new();
        let mut group: Scheduled<Priced> = Scheduled::default();
        priced_after(&mut group, &parent, "k", "good", 1, Duration::ZERO);
        group.spawn("k", &parent, |_| async {
            Err(BorgError::Internal {
                message: "second run failed".into(),
            })
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        group.collect_ready();
        assert!(matches!(
            group.take_failure("k"),
            Some(BorgError::Internal { .. })
        ));
        assert_eq!(group.peek("k").unwrap().body, "good");
    }

    /// The session-end reap: a run that finished after the last drain is
    /// rescued through the hook (it cost money); a run the abort stopped
    /// produced nothing and is not reported.
    #[tokio::test]
    async fn cancel_all_and_reap_rescues_finished_runs_and_drops_aborted_ones() {
        let parent = CancellationToken::new();
        let seen: Seen = Default::default();
        let mut group = Scheduled::default().on_complete(recording_hook(&seen));
        priced_after(&mut group, &parent, "landed", "late", 9, Duration::ZERO);
        priced_after(
            &mut group,
            &parent,
            "running",
            "never",
            99,
            Duration::from_secs(30),
        );
        tokio::time::sleep(Duration::from_millis(30)).await; // "landed" finishes, undrained

        let rescued = group.cancel_all_and_reap().await;
        assert_eq!(rescued, 1);
        assert_eq!(group.inflight_count(), 0);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], ("landed".to_string(), Ok(9)));
    }

    #[tokio::test]
    async fn wait_until_settled_buffers_siblings_that_finish_first() {
        let parent = CancellationToken::new();
        let mut group: Scheduled<Priced> = Scheduled::default();
        priced_after(
            &mut group,
            &parent,
            "fast",
            "f",
            1,
            Duration::from_millis(5),
        );
        priced_after(
            &mut group,
            &parent,
            "slow",
            "s",
            2,
            Duration::from_millis(50),
        );
        assert!(
            group
                .wait_until_settled("slow", Duration::from_secs(5))
                .await
        );
        assert!(
            group.peek("fast").is_some(),
            "the sibling was buffered on the way"
        );
        assert!(group.peek("slow").is_some());
        assert!(
            group
                .wait_until_settled("absent", Duration::from_millis(1))
                .await,
            "nothing in flight under a name settles at once"
        );
    }

    #[tokio::test]
    async fn wait_until_settled_times_out_and_keeps_the_spawn() {
        let parent = CancellationToken::new();
        let mut group: Scheduled<Priced> = Scheduled::default();
        priced_after(
            &mut group,
            &parent,
            "slow",
            "s",
            2,
            Duration::from_millis(80),
        );
        assert!(
            !group
                .wait_until_settled("slow", Duration::from_millis(5))
                .await
        );
        assert!(group.is_inflight("slow"));
        assert!(
            group
                .wait_until_settled("slow", Duration::from_secs(5))
                .await
        );
    }

    #[tokio::test]
    async fn spawn_agent_runs_an_agent_into_the_group() {
        let mut group = Scheduled::default();
        group.spawn_agent(
            "echo",
            test_ctx(),
            EchoAgent {
                token: "PASS",
                body: "via spawn_agent",
            },
            RestartPolicy::Never,
            HandoffState::new(),
        );
        assert!(
            group
                .wait_until_settled("echo", Duration::from_secs(5))
                .await
        );
        assert_eq!(
            &*group.take("echo").unwrap().unwrap().body,
            "via spawn_agent"
        );
    }
}
