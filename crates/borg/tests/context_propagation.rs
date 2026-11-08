//! Context propagation: a task-local the caller set around `execute` is
//! visible inside every task borg spawns — each parallel branch (also
//! nested, also under `CollectAll`), each parallel tool call, and each
//! background spawn — and the spawner's tracing span travels with it.
//!
//! Run with:
//!   cargo test -p borg --test context_propagation --features testing

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::dsl::{par, par_all, seq};
use borg::propagation::{self, Captured, ContextCarrier, ScopedFuture, TaskLocal};
use borg::testing::{FakeLlmClient, FakeOutcome};
use borg::{
    execute_with_handoff, in_memory_budget, noop_emitter, AgentContext, BorgAgent, BorgError,
    CancellationToken, Composition, ConcurrentDispatcher, ConcurrentRegistry, ConcurrentTool,
    ExecCtx, HandoffState, RestartPolicy, SubAgentHandle, ToolCallRequest, ToolCallResult,
    ToolSchema, Verdict,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

tokio::task_local! {
    /// What an app would scope per request: a ledger every model call
    /// inside the request writes to.
    static LEDGER: Arc<Mutex<Vec<String>>>;
    /// A second, process-wide one (registered once, like an app would at
    /// startup) — distinct from LEDGER so tests do not interfere.
    static REQUEST_ID: Arc<str>;
}

/// A step that reads the task-locals where it runs and writes what it
/// saw into its verdict and the ledger.
struct Probe(&'static str);

#[async_trait]
impl BorgAgent for Probe {
    fn agent_id(&self) -> &'static str {
        self.0
    }
    fn model(&self) -> &str {
        "deterministic"
    }
    fn max_tokens(&self) -> u32 {
        0
    }
    fn system_prompt(&self) -> &'static str {
        ""
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        None
    }
    async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
        // Hop through a yield so the probe really runs on its branch task.
        tokio::task::yield_now().await;
        let ledger = LEDGER
            .try_with(|l| {
                l.lock().unwrap().push(self.0.to_string());
                "ledger"
            })
            .unwrap_or("none");
        let rid = REQUEST_ID
            .try_with(|r| r.to_string())
            .unwrap_or_else(|_| "none".into());
        let mut v = Verdict::empty();
        v.body = format!("{ledger}|{rid}").into();
        Ok(v)
    }
}

fn probe(name: &'static str) -> Composition {
    Composition::single(name, Probe(name))
}

fn ctx() -> ExecCtx {
    ExecCtx::deterministic(
        in_memory_budget(1_000_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    )
}

fn bodies(outcome: &borg::ExecutionOutcome) -> Vec<(String, String)> {
    let mut v: Vec<_> = outcome
        .verdicts
        .iter()
        .map(|v| (v.step_name.clone(), v.verdict.body.to_string()))
        .collect();
    v.sort();
    v
}

/// The bug this fixes: without a carrier, a task-local is gone inside
/// every `par_all` branch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_carrier_a_task_local_is_lost_in_branches() {
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let comp = par_all([probe("a"), probe("b")]);
    let outcome = LEDGER
        .scope(
            ledger.clone(),
            execute_with_handoff(comp, ctx(), HandoffState::new()),
        )
        .await
        .unwrap();
    assert!(bodies(&outcome).iter().all(|(_, b)| b.starts_with("none|")));
    assert!(ledger.lock().unwrap().is_empty());
}

/// The pinned contract: a task-local set before `par_all` is visible
/// inside every branch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_task_local_set_before_par_all_is_visible_in_every_branch() {
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let comp = par_all([probe("a"), probe("b"), probe("c"), probe("d")]);
    let ctx = ctx().with_carrier(TaskLocal::new(&LEDGER));
    let outcome = LEDGER
        .scope(
            ledger.clone(),
            execute_with_handoff(comp, ctx, HandoffState::new()),
        )
        .await
        .unwrap();
    for (name, body) in bodies(&outcome) {
        assert!(body.starts_with("ledger|"), "branch {name} saw {body}");
    }
    let mut seen = ledger.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, ["a", "b", "c", "d"]);
}

/// Nested fan-out: a `par` inside a `seq` inside a `par` — every level
/// re-captures on its own branch task.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_parallel_branches_all_see_it() {
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let comp = par([
        seq([probe("s1"), par([probe("n1"), probe("n2")])]),
        par([probe("m1"), probe("m2")]),
    ]);
    let ctx = ctx().with_carrier(TaskLocal::new(&LEDGER));
    let _ = LEDGER
        .scope(
            ledger.clone(),
            execute_with_handoff(comp, ctx, HandoffState::new()),
        )
        .await
        .unwrap();
    let mut seen = ledger.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, ["m1", "m2", "n1", "n2", "s1"]);
}

/// A process-wide carrier (registered once) reaches branches with no
/// `with_carrier` on the context, and background spawns with no
/// `ExecCtx` at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_registered_carrier_reaches_branches_and_background_spawns() {
    propagation::register_carrier(TaskLocal::new(&REQUEST_ID));
    assert!(propagation::registered_carriers() >= 1);

    let comp = par_all([probe("a"), probe("b")]);
    let outcome = REQUEST_ID
        .scope(
            Arc::from("req-42"),
            execute_with_handoff(comp, ctx(), HandoffState::new()),
        )
        .await
        .unwrap();
    for (name, body) in bodies(&outcome) {
        assert!(body.ends_with("|req-42"), "branch {name} saw {body}");
    }

    let seen = REQUEST_ID
        .scope(Arc::from("req-43"), async {
            let parent = CancellationToken::new();
            SubAgentHandle::spawn("bg", &parent, |_| async {
                Ok(REQUEST_ID.try_with(|r| r.to_string()).ok())
            })
            .await_completion()
            .await
        })
        .await
        .unwrap();
    assert_eq!(seen.as_deref(), Some("req-43"));
}

/// `ExecCtx::spawn_sub_agent` carries the context's own carriers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_sub_agent_carries_the_contexts_carriers() {
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let ctx = ctx().with_carrier(TaskLocal::new(&LEDGER));
    let verdict = LEDGER
        .scope(ledger.clone(), async move {
            ctx.spawn_sub_agent(
                "bg",
                Probe("bg"),
                RestartPolicy::never(),
                HandoffState::new(),
            )
            .await_completion()
            .await
        })
        .await
        .unwrap();
    assert!(verdict.body.starts_with("ledger|"));
    assert_eq!(*ledger.lock().unwrap(), ["bg"]);
}

/// The span that spawned a branch is the branch's parent span.
/// (Current-thread runtime: `set_default` installs the subscriber for
/// this thread only.)
#[tokio::test(flavor = "current_thread")]
async fn the_spawners_span_is_the_branchs_span() {
    struct SpanProbe;
    #[async_trait]
    impl BorgAgent for SpanProbe {
        fn agent_id(&self) -> &'static str {
            "span_probe"
        }
        fn model(&self) -> &str {
            "deterministic"
        }
        fn max_tokens(&self) -> u32 {
            0
        }
        fn system_prompt(&self) -> &'static str {
            ""
        }
        fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
            None
        }
        async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
            let mut v = Verdict::empty();
            let span = tracing::Span::current();
            v.body = span.metadata().map_or("none", |m| m.name()).into();
            Ok(v)
        }
    }
    let subscriber = tracing_subscriber::registry();
    let _guard = tracing::subscriber::set_default(subscriber);
    let span = tracing::info_span!("round");
    let comp = par_all([Composition::single("x", SpanProbe)]);
    let outcome = execute_with_handoff(comp, ctx(), HandoffState::new())
        .instrument_in(span.clone())
        .await
        .unwrap();
    // The branch runs under the span that spawned it, not the empty span
    // a bare `tokio::spawn` would give it.
    assert_eq!(&*outcome.verdicts[0].verdict.body, "round");
}

trait InstrumentIn: std::future::Future + Sized {
    fn instrument_in(self, span: tracing::Span) -> tracing::instrument::Instrumented<Self> {
        tracing::Instrument::instrument(self, span)
    }
}
impl<F: std::future::Future> InstrumentIn for F {}

/// A hand-written carrier for something that is not a task-local: the
/// trait pair is the whole contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_custom_carrier_wraps_every_branch() {
    struct CountingCarrier(Arc<AtomicUsize>);
    struct Entered(Arc<AtomicUsize>);
    impl ContextCarrier for CountingCarrier {
        fn capture(&self) -> Option<Box<dyn Captured>> {
            Some(Box::new(Entered(self.0.clone())))
        }
    }
    impl Captured for Entered {
        fn enter<'a>(self: Box<Self>, fut: ScopedFuture<'a>) -> ScopedFuture<'a> {
            let n = self.0;
            Box::pin(async move {
                n.fetch_add(1, Ordering::SeqCst);
                fut.await
            })
        }
    }
    let entered = Arc::new(AtomicUsize::new(0));
    let comp = par_all([probe("a"), probe("b"), probe("c")]);
    let ctx = ctx().with_carrier(CountingCarrier(entered.clone()));
    let _ = execute_with_handoff(comp, ctx, HandoffState::new())
        .await
        .unwrap();
    assert_eq!(entered.load(Ordering::SeqCst), 3, "once per branch");
}

// ── Parallel tool calls ─────────────────────────────────────────────

struct LedgerTool(&'static str);

#[async_trait]
impl ConcurrentTool for LedgerTool {
    type Context = ();
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.0,
            "writes to the ledger",
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }
    fn parallelize(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        _input: serde_json::Value,
        _: &(),
    ) -> Result<ToolCallResult, BorgError> {
        tokio::task::yield_now().await;
        let saw = LEDGER
            .try_with(|l| {
                l.lock().unwrap().push(self.0.to_string());
                true
            })
            .unwrap_or(false);
        ToolCallResult::ok(serde_json::json!({ "saw": saw }))
    }
}

#[derive(borg::BorgAgentDerive)]
#[borg(
    id = "caller",
    model = "fake/caller",
    max_tokens = 100,
    system = "call tools"
)]
struct Caller;

impl Caller {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("go".into())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_parallel_tool_call_sees_it() {
    let mut registry = ConcurrentRegistry::new();
    registry.register(LedgerTool("t1"));
    registry.register(LedgerTool("t2"));
    registry.register(LedgerTool("t3"));
    let tools = Arc::new(ConcurrentDispatcher::new(registry, ()));
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok_with_tool_calls(
            "batch",
            ["t1", "t2", "t3"]
                .iter()
                .map(|n| ToolCallRequest::new(*n, serde_json::json!({}), format!("id-{n}")))
                .collect(),
        ),
        FakeOutcome::ok("done"),
    ]);
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    )
    .with_parallel_tools(borg::BatchFailureMode::CollectAll)
    .with_carrier(TaskLocal::new(&LEDGER));
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let _ = LEDGER
        .scope(
            ledger.clone(),
            execute_with_handoff(
                borg::dsl::agent_loop(Caller, tools, 4),
                ctx,
                HandoffState::new(),
            ),
        )
        .await
        .unwrap();
    let mut seen = ledger.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, ["t1", "t2", "t3"]);
}
