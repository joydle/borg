//! Supervised task group for concurrent LLM agents.
//!
//! Wraps `tokio::task::JoinSet` to give a group of children Erlang-style
//! isolation:
//!
//! - **Panic isolation.** A panicking child surfaces as
//!   `BorgError::ChildPanicked` rather than unwinding into the supervisor.
//! - **Parent-cancel propagation.** Each child holds a `child_token()`
//!   of the supervisor's token, so a parent cancel reaches every
//!   descendant.
//! - **Bounded concurrency.** A `Semaphore` caps simultaneous LLM
//!   round-trips so a wide `Parallel` can't burst-rate-limit the provider.
//! - **Drop-aborts.** Dropping the supervisor aborts all in-flight
//!   children (`JoinSet` semantics), preventing leaks.
//!
//! Not a full actor system: agents are single-shot async functions, no
//! message passing or linking. Restart is OneForOne via `RestartPolicy`;
//! the session-wide token budget is the only circuit breaker (no
//! separate supervisor-level meltdown threshold).
//!
//! Crate-private (users go through `execute` / `run_one`). The
//! executor uses only `spawn_in` / `join_next` / `cancel_all`; the rest of
//! the handle API stays for the unit tests and future executor paths.
#![allow(dead_code)]

use crate::error::BorgError;
use crate::propagation::CapturedContext;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::{AbortHandle, JoinSet};
use tokio_util::sync::CancellationToken;

/// Live handle for a spawned agent. Lets a caller cancel one branch
/// without touching its siblings.
pub struct RunningAgent {
    pub id: String,
    /// The name the child was spawned under. `Arc<str>` (was
    /// `&'static str`, which made the parallel executor leak every
    /// branch name once per run).
    pub agent_id: Arc<str>,
    pub cancel: CancellationToken,
    pub abort: AbortHandle,
}

impl RunningAgent {
    /// Cooperative cancel — fires the child's token. A well-behaved
    /// child exits via its `tokio::select!` on `ctx.cancel`; `abort()`
    /// is the backstop for one that doesn't.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Forceful abort — drops the future with no cleanup. Prefer
    /// `cancel()` so the agent's `Drop` and `select!` arms fire.
    pub fn abort(&self) {
        self.abort.abort();
    }
}

/// Outcome of one child's completed (or aborted) execution.
pub struct ChildOutcome<T> {
    pub id: String,
    /// The name the child was spawned under — kept on a panic or an
    /// abort too, so a failed branch is reported under its own name.
    pub agent_id: Arc<str>,
    pub result: Result<T, BorgError>,
    pub duration_ms: u64,
}

/// Supervised group of agent tasks. `T` is each child's return type,
/// typically `Verdict`.
///
/// Lazy: `new` spawns nothing. Add children via `spawn`, drain via
/// `join_next`.
pub struct Supervisor<T: Send + 'static> {
    cancel: CancellationToken,
    permits: Arc<Semaphore>,
    children: JoinSet<ChildOutcome<T>>,
    handles: Vec<RunningAgent>,
    /// `(id, agent_id)` per task, so a panicked or aborted child (whose
    /// `ChildOutcome` was never built) is still reported by name.
    names: HashMap<tokio::task::Id, (String, Arc<str>)>,
    next_id: u64,
}

impl<T: Send + 'static> Supervisor<T> {
    /// New supervisor with a parent cancel token and a concurrency cap.
    ///
    /// Holds a `child_token()` of `parent_cancel`: a parent cancel
    /// propagates down, but firing the supervisor's own token (Drop,
    /// `cancel_all`) cancels only its children and never bubbles up.
    /// That one-way isolation is what lets a `Sequential` recover from a
    /// `Parallel` step's cleanup without its own `ctx.cancel` flipping.
    pub fn new(parent_cancel: CancellationToken, max_concurrency: usize) -> Self {
        Self {
            cancel: parent_cancel.child_token(),
            permits: Arc::new(Semaphore::new(max_concurrency.max(1))),
            children: JoinSet::new(),
            handles: Vec::new(),
            names: HashMap::new(),
            next_id: 0,
        }
    }

    /// Spawn a child future under supervision.
    ///
    /// `f` receives the child's `CancellationToken` and returns a future,
    /// which the supervisor wraps in permit-acquire and panic-catch so
    /// the module-level contract holds uniformly. The child runs inside
    /// the caller's context: its current span and every process-wide
    /// [`crate::propagation::ContextCarrier`]. Returns the `RunningAgent`
    /// handle.
    pub fn spawn<F, Fut>(&mut self, agent_id: impl Into<Arc<str>>, f: F) -> &RunningAgent
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T, BorgError>> + Send + 'static,
    {
        let context = crate::propagation::capture(&crate::propagation::Carriers::new());
        self.spawn_in(agent_id, context, f)
    }

    /// [`Self::spawn`] inside an explicitly captured context (the span and
    /// carrier values the executor took on the spawning task, including an
    /// `ExecCtx`'s own carriers).
    pub fn spawn_in<F, Fut>(
        &mut self,
        agent_id: impl Into<Arc<str>>,
        context: CapturedContext,
        f: F,
    ) -> &RunningAgent
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T, BorgError>> + Send + 'static,
    {
        let agent_id: Arc<str> = agent_id.into();
        let child_cancel = self.cancel.child_token();
        let permits = self.permits.clone();
        self.next_id += 1;
        let child_id = format!("{agent_id}-{}", self.next_id);
        let id_for_outcome = child_id.clone();
        let agent_for_outcome = agent_id.clone();
        let cancel_for_outcome = child_cancel.clone();

        let abort = self.children.spawn(context.scope(async move {
            let agent_id = agent_for_outcome;
            let started = std::time::Instant::now();
            // Acquire the permit inside the task so spawn rate stays
            // unbounded — the cap is on simultaneous round-trips, not
            // on enqueueing.
            let _permit = match permits.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    return ChildOutcome {
                        id: id_for_outcome,
                        agent_id,
                        result: Err(BorgError::Internal {
                            message: "supervisor semaphore closed".into(),
                        }),
                        duration_ms: started.elapsed().as_millis() as u64,
                    };
                }
            };

            // If cancel fired while we were queued behind the permit,
            // exit without invoking the closure.
            if cancel_for_outcome.is_cancelled() {
                return ChildOutcome {
                    id: id_for_outcome,
                    agent_id,
                    result: Err(BorgError::Cancelled),
                    duration_ms: started.elapsed().as_millis() as u64,
                };
            }

            // The closure honors cancel via its own `select!`; we
            // deliberately do NOT wrap it in an outer select, which
            // would short-circuit its body before its cancel arm can
            // record side effects. A misbehaving agent that ignores
            // cancel is caught by `JoinSet::abort_all` (Drop).
            let fut = f(cancel_for_outcome.clone());
            let result = fut.await;

            ChildOutcome {
                id: id_for_outcome,
                agent_id,
                result,
                duration_ms: started.elapsed().as_millis() as u64,
            }
        }));
        self.names
            .insert(abort.id(), (child_id.clone(), agent_id.clone()));

        self.handles.push(RunningAgent {
            id: child_id,
            agent_id,
            cancel: child_cancel,
            abort,
        });
        self.handles.last().unwrap()
    }

    /// Wait for the next child to finish. `None` when no children are
    /// in flight.
    ///
    /// Panic catch lives here: `JoinError::is_panic()` becomes
    /// `BorgError::ChildPanicked`, so a panic never unwinds across a
    /// child boundary. Under `ErrorPolicy::CollectAll` a panic in A
    /// yields a failed `NamedVerdict` while siblings continue.
    pub async fn join_next(&mut self) -> Option<ChildOutcome<T>> {
        match self.children.join_next_with_id().await {
            None => None,
            Some(Ok((task, outcome))) => {
                self.names.remove(&task);
                Some(outcome)
            }
            Some(Err(join_err)) => {
                let (id, agent_id) = self
                    .names
                    .remove(&join_err.id())
                    .unwrap_or_else(|| ("unknown".into(), Arc::from("unknown")));
                if join_err.is_panic() {
                    let payload = join_err.into_panic();
                    let message = panic_payload_to_string(payload);
                    Some(ChildOutcome {
                        id,
                        agent_id: agent_id.clone(),
                        result: Err(BorgError::ChildPanicked {
                            agent_id: agent_id.to_string(),
                            message,
                        }),
                        duration_ms: 0,
                    })
                } else {
                    // Non-panic JoinError: the task was aborted.
                    Some(ChildOutcome {
                        id,
                        agent_id,
                        result: Err(BorgError::Cancelled),
                        duration_ms: 0,
                    })
                }
            }
        }
    }

    /// Number of in-flight children.
    pub fn in_flight(&self) -> usize {
        self.children.len()
    }

    /// Cooperative shutdown: fire the token so in-flight children exit
    /// through their `select!` arms.
    pub fn cancel_all(&self) {
        self.cancel.cancel();
    }

    /// Forceful shutdown: abort all child tasks (`JoinSet::abort_all`),
    /// dropping futures with no cleanup.
    pub fn abort_all(&mut self) {
        self.children.abort_all();
    }
}

impl<T: Send + 'static> Drop for Supervisor<T> {
    fn drop(&mut self) {
        // Fire the token before JoinSet's own Drop aborts the children,
        // so their `select!` arms see the cooperative signal first.
        if !self.cancel.is_cancelled() {
            self.cancel.cancel();
        }
    }
}

pub(crate) fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "<non-string panic payload>".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn supervisor_runs_a_child_to_completion() {
        let cancel = CancellationToken::new();
        let mut sup: Supervisor<u32> = Supervisor::new(cancel, 4);
        sup.spawn("hello", |_c| async move { Ok(42u32) });
        let out = sup.join_next().await.unwrap();
        assert_eq!(out.result.unwrap(), 42);
        assert_eq!(&*out.agent_id, "hello");
    }

    #[tokio::test]
    async fn parent_cancel_propagates_to_child_token() {
        let cancel = CancellationToken::new();
        let mut sup: Supervisor<u32> = Supervisor::new(cancel.clone(), 4);
        sup.spawn("slow", |c| async move {
            c.cancelled().await;
            Err(BorgError::Cancelled)
        });
        cancel.cancel();
        let out = sup.join_next().await.unwrap();
        assert!(matches!(out.result, Err(BorgError::Cancelled)));
    }

    #[tokio::test]
    async fn child_panic_surfaces_as_child_panicked_error() {
        let cancel = CancellationToken::new();
        let mut sup: Supervisor<u32> = Supervisor::new(cancel, 4);
        sup.spawn("panicker", |_c| async move {
            panic!("boom");
            #[allow(unreachable_code)]
            Ok(0u32)
        });
        let out = sup.join_next().await.unwrap();
        assert_eq!(&*out.agent_id, "panicker", "a panic keeps the child's name");
        match out.result {
            Err(BorgError::ChildPanicked { message, agent_id }) => {
                assert!(message.contains("boom"));
                assert_eq!(agent_id, "panicker");
            }
            other => panic!("expected ChildPanicked, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn semaphore_caps_concurrent_children() {
        // semaphore = 1, so the second child only starts after the first
        // releases its permit — asserted via timing below.
        let cancel = CancellationToken::new();
        let mut sup: Supervisor<u32> = Supervisor::new(cancel, 1);
        let started = std::time::Instant::now();
        sup.spawn("first", |_c| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(1u32)
        });
        sup.spawn("second", |_c| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(2u32)
        });
        let _a = sup.join_next().await.unwrap();
        let _b = sup.join_next().await.unwrap();
        let elapsed = started.elapsed();
        // Two serialized 50ms sleeps; bounds are loose for scheduler jitter
        // yet tight enough to prove they didn't run in parallel.
        assert!(elapsed >= Duration::from_millis(95));
        assert!(elapsed < Duration::from_millis(250));
    }

    #[tokio::test]
    async fn drop_does_not_cancel_parent_token() {
        // Regression: storing the caller's token directly meant Drop
        // cancelled the parent, silently aborting the enclosing
        // Sequential. Holding a `child_token()` confines Drop to children.
        let parent = CancellationToken::new();
        let parent_clone = parent.clone();
        {
            let mut sup: Supervisor<u32> = Supervisor::new(parent, 4);
            sup.spawn("watcher", move |c| async move {
                c.cancelled().await;
                Err(BorgError::Cancelled)
            });
            tokio::task::yield_now().await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(
            !parent_clone.is_cancelled(),
            "supervisor Drop must NOT cancel the parent token"
        );
    }

    // Span-inheritance guard (moved from borg-loop's tracing tests when
    // `Supervisor` stopped being public). `spawn` wraps the child future in
    // `.in_current_span()`; a regression to a bare `JoinSet::spawn` would
    // orphan spans opened in the closure body.
    #[tokio::test(flavor = "current_thread")]
    async fn supervisor_child_inherits_caller_span() {
        use std::sync::Mutex;
        use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
        use tracing_subscriber::registry::LookupSpan;

        /// `(name, parent_id)` of one new span.
        type Row = (&'static str, Option<u64>);

        #[derive(Default, Clone)]
        struct Capture(Arc<Mutex<Vec<Row>>>);

        impl<S> Layer<S> for Capture
        where
            S: tracing::Subscriber + for<'a> LookupSpan<'a>,
        {
            fn on_new_span(
                &self,
                attrs: &tracing::span::Attributes<'_>,
                id: &tracing::Id,
                ctx: Context<'_, S>,
            ) {
                let parent = ctx
                    .span(id)
                    .and_then(|s| s.parent().map(|p| p.id().into_u64()));
                self.0
                    .lock()
                    .unwrap()
                    .push((attrs.metadata().name(), parent));
            }
        }

        let capture = Capture::default();
        let subscriber = tracing_subscriber::Registry::default().with(capture.clone());
        // Thread-local is enough here: current-thread runtime, and both
        // callsites below are unique to this test.
        let _guard = tracing::subscriber::set_default(subscriber);

        let outer = tracing::info_span!("supervisor_test_outer");
        let outer_id = outer.id().expect("outer span enabled").into_u64();

        let mut sup: Supervisor<()> = Supervisor::new(CancellationToken::new(), 4);
        {
            let _g = outer.enter();
            sup.spawn("test_child", |_cancel| async move {
                // Opened inside the spawned task; must parent under `outer`.
                let _g = tracing::info_span!("supervisor_test_inner").entered();
                Ok(())
            });
        }
        let outcome = sup
            .join_next()
            .await
            .expect("supervisor must surface the child outcome");
        assert!(outcome.result.is_ok(), "child returned Ok(())");

        let rows = capture.0.lock().unwrap().clone();
        let inner = rows
            .iter()
            .find(|(name, _)| *name == "supervisor_test_inner")
            .expect("inner span opened inside supervisor child must be captured");
        assert_eq!(
            inner.1,
            Some(outer_id),
            "Supervisor::spawn must propagate the caller's current span into the \
             spawned task (`.in_current_span()` dropped?). Captured rows: {rows:#?}",
        );
    }
}
