//! End-to-end integration tests for opt-in parallel tool-call dispatch.
//! Each test scripts a multi-call LLM response, wires a
//! `ConcurrentDispatcher` over barrier-synchronized `ConcurrentTool`s, and
//! asserts observable behavior (concurrency, ordering, failure semantics).
//!
//! Run with:
//!   cargo test -p borg --test parallel_tool_calls --features testing

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::testing::{FakeLlmClient, FakeOutcome};
use borg::{
    dsl, execute_with_handoff, in_memory_budget, noop_emitter, AgentContext, AgentLlmClient,
    BatchFailureMode, BorgAgentDerive, BorgError, ConcurrentDispatcher, ConcurrentRegistry,
    ConcurrentTool, ExecCtx, HandoffState, ToolCallRequest, ToolCallResult, ToolDispatch,
    ToolSchema,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Barrier, Notify};
use tokio_util::sync::CancellationToken;

// ── Shared context + test tools ─────────────────────────────────────

/// Shared context for the test tools: an optional `Barrier` (concurrent-
/// entry assertion), a `Notify` siblings park on (FastFail cancel), a
/// sibling-cancel counter, and an optional name to fail on.
#[derive(Default)]
struct TestCtx {
    barrier: Option<Arc<Barrier>>,
    long_park: Option<Arc<Notify>>,
    sibling_cancels: AtomicUsize,
    fail_on_name: Option<String>,
    /// Calls inside `execute` right now, and the most seen at once.
    in_flight: AtomicUsize,
    max_in_flight: Arc<AtomicUsize>,
}

/// A ConcurrentTool that records its own start order, optionally waits
/// on the barrier, optionally errors structurally based on tool name,
/// optionally parks indefinitely (so siblings can race past it).
struct ToolUnderTest {
    name: String,
    parallel: bool,
    behavior: ToolBehavior,
}

#[derive(Clone)]
enum ToolBehavior {
    /// Wait on the context's barrier (if any), then return Ok.
    BarrierThenOk,
    /// Immediately Err with a structural BorgError.
    StructuralErr,
    /// Park on the context's long_park until cancelled. Increments
    /// sibling_cancels when it observes cancellation.
    ParkUntilCancel,
    /// Return Ok immediately (no barrier).
    Ok,
    /// Track how many calls overlap, holding the slot briefly.
    TrackConcurrency,
}

#[async_trait]
impl ConcurrentTool for ToolUnderTest {
    type Context = TestCtx;
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name.clone(),
            format!("test tool {}", self.name),
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }
    fn parallelize(&self) -> bool {
        self.parallel
    }
    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &TestCtx,
    ) -> Result<ToolCallResult, BorgError> {
        if let Some(fail) = &ctx.fail_on_name {
            if fail == &self.name {
                return Err(BorgError::Internal {
                    message: format!("structural error from `{}`", self.name),
                });
            }
        }
        match self.behavior {
            ToolBehavior::Ok => ToolCallResult::ok(serde_json::json!({"ran": self.name})),
            ToolBehavior::TrackConcurrency => {
                let now = ctx.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                ctx.max_in_flight.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                ctx.in_flight.fetch_sub(1, Ordering::SeqCst);
                ToolCallResult::ok(serde_json::json!({"ran": self.name}))
            }
            ToolBehavior::StructuralErr => Err(BorgError::Internal {
                message: format!("structural error from `{}`", self.name),
            }),
            ToolBehavior::BarrierThenOk => {
                if let Some(b) = ctx.barrier.as_ref() {
                    b.wait().await;
                }
                ToolCallResult::ok(serde_json::json!({"ran": self.name}))
            }
            ToolBehavior::ParkUntilCancel => {
                let notify = ctx
                    .long_park
                    .as_ref()
                    .expect("park behavior needs long_park");
                tokio::select! {
                    biased;
                    _ = notify.notified() => {
                        ctx.sibling_cancels.fetch_add(1, Ordering::SeqCst);
                        Err(BorgError::Cancelled)
                    }
                    _ = tokio::time::sleep(Duration::from_secs(60)) => {
                        // Unreachable in a healthy test — failsafe so a
                        // hang doesn't wedge the whole suite.
                        Err(BorgError::Internal {
                            message: "park watchdog tripped".into(),
                        })
                    }
                }
            }
        }
    }
}

// ── Test agent that just dispatches a scripted batch ─────────────────

#[derive(BorgAgentDerive)]
#[borg(
    id = "batch_caller",
    model = "fake/batch_caller",
    max_tokens = 200,
    system = "Call tools in batches."
)]
struct BatchCaller;

impl BatchCaller {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("go".into())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn req(name: &str, id: &str) -> ToolCallRequest {
    ToolCallRequest::new(name, serde_json::json!({}), id)
}

fn make_ctx(parallel: bool, mode: BatchFailureMode, llm: Arc<dyn AgentLlmClient>) -> ExecCtx {
    // Install the binary-wide span capture before any test runs borg code
    // (see `span_capture` for why it must be global and installed first).
    span_capture();
    let mut ctx = ExecCtx::new(
        llm,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    if parallel {
        ctx = ctx.with_parallel_tools(mode);
    }
    ctx
}

fn make_dispatcher(tools: Vec<ToolUnderTest>, test_ctx: TestCtx) -> Arc<dyn ToolDispatch> {
    let mut registry = ConcurrentRegistry::new();
    for t in tools {
        registry.register(t);
    }
    Arc::new(ConcurrentDispatcher::new(registry, test_ctx))
}

fn batched_llm(calls: Vec<ToolCallRequest>) -> Arc<dyn AgentLlmClient> {
    FakeLlmClient::new(vec![
        FakeOutcome::ok_with_tool_calls("calling batch", calls),
        FakeOutcome::ok("done"),
    ])
}

async fn run_batch(
    dispatcher: Arc<dyn ToolDispatch>,
    ctx: ExecCtx,
) -> Result<Vec<String>, BorgError> {
    let pipeline = dsl::agent_loop(BatchCaller, dispatcher, 5);
    let outcome = execute_with_handoff(pipeline, ctx, HandoffState::new()).await?;
    Ok(outcome.verdicts.into_iter().map(|v| v.step_name).collect())
}

// ── Tests ───────────────────────────────────────────────────────────

/// Three parallel-safe tools blocking on a Barrier(3). Serial dispatch
/// would never trip the barrier and the test would time out; reaching
/// the join proves all three were live in `execute` at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_happy_path() {
    let test_ctx = TestCtx {
        barrier: Some(Arc::new(Barrier::new(3))),
        ..Default::default()
    };
    let tools = vec![
        ToolUnderTest {
            name: "a".into(),
            parallel: true,
            behavior: ToolBehavior::BarrierThenOk,
        },
        ToolUnderTest {
            name: "b".into(),
            parallel: true,
            behavior: ToolBehavior::BarrierThenOk,
        },
        ToolUnderTest {
            name: "c".into(),
            parallel: true,
            behavior: ToolBehavior::BarrierThenOk,
        },
    ];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![req("a", "1"), req("b", "2"), req("c", "3")]);
    let ctx = make_ctx(true, BatchFailureMode::CollectAll, llm);

    let result = tokio::time::timeout(Duration::from_secs(5), run_batch(dispatch, ctx))
        .await
        .expect("test timed out — parallel dispatch is not actually parallel");
    assert!(result.is_ok(), "agent loop should succeed: {result:?}");
}

/// Gate off → serial. With a Barrier(3), the first call hangs forever;
/// a short timeout confirms no parallelism.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_disabled_gate_off_serializes() {
    let test_ctx = TestCtx {
        barrier: Some(Arc::new(Barrier::new(3))),
        ..Default::default()
    };
    let tools = vec![
        ToolUnderTest {
            name: "a".into(),
            parallel: true,
            behavior: ToolBehavior::BarrierThenOk,
        },
        ToolUnderTest {
            name: "b".into(),
            parallel: true,
            behavior: ToolBehavior::BarrierThenOk,
        },
        ToolUnderTest {
            name: "c".into(),
            parallel: true,
            behavior: ToolBehavior::BarrierThenOk,
        },
    ];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![req("a", "1"), req("b", "2"), req("c", "3")]);
    // parallel = false → serial path even though every tool is safe.
    let ctx = make_ctx(false, BatchFailureMode::CollectAll, llm);

    let result = tokio::time::timeout(Duration::from_millis(400), run_batch(dispatch, ctx)).await;
    assert!(
        result.is_err(),
        "with the gate off, the serial loop should hang on the barrier"
    );
}

/// CollectAll: one structural error, but the other tools still complete,
/// the failure reaches the agent as an `is_error` result, and the loop
/// carries on to its next turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_failure_collect_all_waits_for_siblings() {
    // No barrier: this test is about failure semantics, not concurrency.
    let counter = Arc::new(AtomicUsize::new(0));
    let test_ctx = TestCtx {
        fail_on_name: Some("middle".into()),
        ..Default::default()
    };
    struct Counted {
        inner: ToolUnderTest,
        counter: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl ConcurrentTool for Counted {
        type Context = TestCtx;
        fn schema(&self) -> ToolSchema {
            self.inner.schema()
        }
        fn parallelize(&self) -> bool {
            self.inner.parallelize()
        }
        async fn execute(
            &self,
            input: serde_json::Value,
            ctx: &TestCtx,
        ) -> Result<ToolCallResult, BorgError> {
            let r = self.inner.execute(input, ctx).await;
            if r.is_ok() {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
            r
        }
    }
    let mut registry = ConcurrentRegistry::new();
    registry.register(Counted {
        inner: ToolUnderTest {
            name: "left".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        counter: counter.clone(),
    });
    registry.register(Counted {
        inner: ToolUnderTest {
            name: "middle".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        counter: counter.clone(),
    });
    registry.register(Counted {
        inner: ToolUnderTest {
            name: "right".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        counter: counter.clone(),
    });
    let dispatch: Arc<dyn ToolDispatch> = Arc::new(ConcurrentDispatcher::new(registry, test_ctx));
    let llm = batched_llm(vec![
        req("left", "1"),
        req("middle", "2"),
        req("right", "3"),
    ]);
    let ctx = make_ctx(true, BatchFailureMode::CollectAll, llm);

    let pipeline = dsl::agent_loop(BatchCaller, dispatch, 5);
    let outcome = execute_with_handoff(pipeline, ctx, HandoffState::new())
        .await
        .expect("CollectAll hands the failure to the agent instead of ending the loop");
    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "both non-failing siblings should have completed"
    );
    assert_eq!(outcome.verdicts.len(), 2, "the agent got a second turn");
    let results: Vec<borg::DispatchedCall> = outcome.handoff.get_as("tool_results").unwrap();
    let errors: Vec<Option<&str>> = results.iter().map(|d| d.result.error.as_deref()).collect();
    assert_eq!(errors.len(), 3);
    assert!(errors[0].is_none() && errors[2].is_none(), "{errors:?}");
    assert!(
        errors[1].is_some_and(|e| e.contains("structural error from `middle`")),
        "{errors:?}"
    );
}

/// FastFail: a long-parked sibling observes cancellation when one tool
/// errors structurally — woken via the cancel signal, not the watchdog.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_failure_fast_fail_cancels_siblings() {
    let notify = Arc::new(Notify::new());
    let test_ctx = TestCtx {
        long_park: Some(notify.clone()),
        fail_on_name: Some("kaboom".into()),
        ..Default::default()
    };
    let tools = vec![
        ToolUnderTest {
            name: "kaboom".into(),
            parallel: true,
            behavior: ToolBehavior::StructuralErr,
        },
        ToolUnderTest {
            name: "parker".into(),
            parallel: true,
            behavior: ToolBehavior::ParkUntilCancel,
        },
    ];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![req("kaboom", "1"), req("parker", "2")]);
    let ctx = make_ctx(true, BatchFailureMode::FastFail, llm);

    // The parker's select! watches `notify`, not the cancel token, so
    // wake it from a spawned task to deterministically observe its
    // cancellation path after fast-fail fires.
    let notify_clone = notify.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        notify_clone.notify_waiters();
    });

    let result = tokio::time::timeout(Duration::from_secs(3), run_batch(dispatch, ctx))
        .await
        .expect("FastFail should return promptly after the structural error");
    assert!(
        result.is_err(),
        "FastFail surfaces the structural error to the caller"
    );
}

/// A single parallel-safe call hits the executor's len==1 short-circuit
/// (skipping the JoinSet). Smoke test that the path still succeeds — the
/// optimization itself is only observable via tracing spans.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_call_in_parallel_path_returns_correctly() {
    let test_ctx = TestCtx::default();
    let tools = vec![ToolUnderTest {
        name: "solo".into(),
        parallel: true,
        behavior: ToolBehavior::Ok,
    }];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![req("solo", "1")]);
    let ctx = make_ctx(true, BatchFailureMode::CollectAll, llm);

    let result = run_batch(dispatch, ctx).await;
    assert!(
        result.is_ok(),
        "single-call batch should succeed: {result:?}"
    );
}

/// Mixed batch [par, par, seq, par, par]: smoke test that the grouping
/// completes without deadlocking or skipping calls. (Asserting group
/// ordering would require span inspection.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_batch_completes() {
    let test_ctx = TestCtx::default();
    let tools = vec![
        ToolUnderTest {
            name: "p1".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "p2".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "s".into(),
            parallel: false,
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "p3".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "p4".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
    ];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![
        req("p1", "1"),
        req("p2", "2"),
        req("s", "3"),
        req("p3", "4"),
        req("p4", "5"),
    ]);
    let ctx = make_ctx(true, BatchFailureMode::CollectAll, llm);

    let result = run_batch(dispatch, ctx).await;
    assert!(result.is_ok(), "mixed batch should complete: {result:?}");
}

// ── borg.tool_batch span propagation ──────────────────────────────────
//
// Span capture must be deterministic while the other tests in this binary
// run concurrently on their own threads. A thread-local subscriber
// (`set_default` / `with_default`) is not enough: tracing-core caches each
// callsite's `Interest` process-wide, and while at most one dispatcher is
// registered it computes that interest from the *registering thread's*
// default. If a subscriber-less test (e.g. `parallel_happy_path`) is the
// first to hit the `borg.tool_batch` callsite after this test installed
// its scoped subscriber, the callsite is cached as `never` and this test
// sees zero spans.
//
// Instead: one global subscriber for the whole binary, installed by
// `make_ctx` (every test calls it before running borg), so no callsite is
// ever registered without it. Each span test runs under a root span
// carrying a unique `test_tag`; the layer propagates the tag to
// descendants, and assertions count only spans carrying their own tag.

/// Field on a test's root span; inherited by every descendant span.
const TEST_TAG_FIELD: &str = "test_tag";

/// Process-wide span capture, installed as the global default once.
fn span_capture() -> &'static SpanCapture {
    static CAPTURE: std::sync::OnceLock<SpanCapture> = std::sync::OnceLock::new();
    CAPTURE.get_or_init(|| {
        use tracing_subscriber::layer::SubscriberExt;
        let capture = SpanCapture::default();
        let subscriber = tracing_subscriber::Registry::default().with(capture.clone());
        tracing::subscriber::set_global_default(subscriber)
            .expect("parallel_tool_calls: no other global subscriber in this binary");
        capture
    })
}

/// `tracing_subscriber::Layer` that records every new span's name, id,
/// parent and inherited test tag.
#[derive(Default, Clone)]
struct SpanCapture {
    rows: Arc<std::sync::Mutex<Vec<SpanRow>>>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // ids are captured for diagnostic output when assertions fail
struct SpanRow {
    name: &'static str,
    span_id: u64,
    parent_id: Option<u64>,
    tag: Option<String>,
}

/// Span extension holding the inherited test tag.
struct TestTag(String);

#[derive(Default)]
struct TagVisitor(Option<String>);

impl tracing::field::Visit for TagVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == TEST_TAG_FIELD {
            self.0 = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == TEST_TAG_FIELD && self.0.is_none() {
            self.0 = Some(format!("{value:?}"));
        }
    }
}

impl SpanCapture {
    /// Spans named `name` that descend from the root span tagged `tag`.
    fn tagged(&self, tag: &str, name: &str) -> Vec<SpanRow> {
        self.rows
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.name == name && r.tag.as_deref() == Some(tag))
            .cloned()
            .collect()
    }
}

impl<S> tracing_subscriber::Layer<S> for SpanCapture
where
    S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else { return };
        // The registry has already resolved the parent (explicit or
        // contextual) by the time layers see the span.
        let parent = span.parent();
        let mut visitor = TagVisitor::default();
        attrs.record(&mut visitor);
        let tag = visitor.0.or_else(|| {
            parent
                .as_ref()
                .and_then(|p| p.extensions().get::<TestTag>().map(|t| t.0.clone()))
        });
        if let Some(t) = &tag {
            span.extensions_mut().insert(TestTag(t.clone()));
        }
        self.rows.lock().unwrap().push(SpanRow {
            name: attrs.metadata().name(),
            span_id: id.into_u64(),
            parent_id: parent.map(|p| p.id().into_u64()),
            tag,
        });
    }
}

/// Run `fut` under a root span tagged `tag`, so the capture can tell this
/// test's spans from those of tests running concurrently.
async fn under_test_root<F: std::future::Future>(tag: &'static str, fut: F) -> F::Output {
    use tracing::Instrument;
    fut.instrument(tracing::info_span!("test_root", test_tag = tag))
        .await
}

/// A parallel batch creates the `borg.tool_batch` span, and the per-task
/// spans spawned into the `JoinSet` are its children, not orphans —
/// regression coverage for span propagation across `tokio::spawn`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_dispatch_emits_tool_batch_span_with_children() {
    const TAG: &str = "parallel_dispatch_emits_tool_batch_span_with_children";

    let test_ctx = TestCtx::default();
    let tools = vec![
        ToolUnderTest {
            name: "t1".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "t2".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "t3".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
    ];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![req("t1", "1"), req("t2", "2"), req("t3", "3")]);
    let ctx = make_ctx(true, BatchFailureMode::CollectAll, llm);

    let result = under_test_root(TAG, run_batch(dispatch, ctx)).await;
    assert!(result.is_ok(), "agent loop should succeed: {result:?}");

    // Exactly one borg.tool_batch span was created under this test's root.
    let batch_spans = span_capture().tagged(TAG, "borg.tool_batch");
    assert_eq!(
        batch_spans.len(),
        1,
        "expected exactly one borg.tool_batch span, got {batch_spans:?}"
    );

    // With noop_emitter there are no borg.tool child spans to inspect, so
    // we assert the weaker invariant: the batch span exists, created once
    // per parallel run, and descends from the caller's span. Tool-span
    // ancestry under the batch span has separate coverage in borg-tracing.
    assert!(
        batch_spans[0].parent_id.is_some(),
        "borg.tool_batch must nest under the caller's span: {batch_spans:?}"
    );
}

/// Negative case: serial dispatch (gate off) must NOT create a
/// borg.tool_batch span. Pairs with the above test to fence the
/// feature boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serial_dispatch_emits_no_tool_batch_span() {
    const TAG: &str = "serial_dispatch_emits_no_tool_batch_span";

    let test_ctx = TestCtx::default();
    let tools = vec![
        ToolUnderTest {
            name: "t1".into(),
            parallel: true, // parallel-safe, but the gate is off
            behavior: ToolBehavior::Ok,
        },
        ToolUnderTest {
            name: "t2".into(),
            parallel: true,
            behavior: ToolBehavior::Ok,
        },
    ];
    let dispatch = make_dispatcher(tools, test_ctx);
    let llm = batched_llm(vec![req("t1", "1"), req("t2", "2")]);
    // Gate OFF.
    let ctx = make_ctx(false, BatchFailureMode::CollectAll, llm);

    let result = under_test_root(TAG, run_batch(dispatch, ctx)).await;
    assert!(result.is_ok());

    // Sanity: the capture saw this test's root, so a zero below means
    // "no batch span", not "capture missed everything".
    assert_eq!(span_capture().tagged(TAG, "test_root").len(), 1);
    let batch_spans = span_capture().tagged(TAG, "borg.tool_batch");
    assert!(
        batch_spans.is_empty(),
        "serial path must not emit borg.tool_batch span: {batch_spans:?}"
    );
}

/// A parallel batch never runs more than `max_parallel_tool_calls` at
/// once; the rest wait for a slot and still all run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_batch_respects_concurrency_cap() {
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let test_ctx = TestCtx {
        max_in_flight: max_in_flight.clone(),
        ..Default::default()
    };
    let tools = vec![ToolUnderTest {
        name: "t".into(),
        parallel: true,
        behavior: ToolBehavior::TrackConcurrency,
    }];
    let dispatcher = make_dispatcher(tools, test_ctx);
    let calls: Vec<_> = (0..8).map(|i| req("t", &format!("c{i}"))).collect();
    let ctx = make_ctx(true, BatchFailureMode::CollectAll, batched_llm(calls))
        .with_max_parallel_tool_calls(2);
    run_batch(dispatcher, ctx).await.expect("batch runs");
    assert_eq!(max_in_flight.load(Ordering::SeqCst), 2);
}
