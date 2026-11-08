//! Parallel tool-call example — opt-in concurrent dispatch.
//!
//! Three independent parallel-safe tools batched in one LLM turn. Two
//! ergonomic shortcuts:
//!
//! 1. `#[tool(..., parallel)]` emits both `impl Tool` and
//!    `impl ConcurrentTool` from one inherent `execute(&self, input,
//!    ctx: &Context)` body (shared `&Context`, not `&mut`).
//! 2. `dsl::agent_loop_parallel(agent, registry, ctx, max_steps)` wires
//!    `ConcurrentDispatcher` + `agent_loop`. The opt-in gate lives on
//!    `ExecCtx::with_parallel_tools(...)`.
//!
//! Run with: `cargo run --example parallel_tools -p borg --features testing`
//!
//! Versus `tool_loop.rs` (serial): same work per turn, but latency drops
//! from `sum(tool_latencies)` to `max(tool_latencies)`.

use borg::testing::{FakeLlmClient, FakeOutcome};
use borg::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shared read-only context. `ConcurrentDispatcher` requires `Sync`; the
/// atomics stand in for real interior-mutable state (RwLock, DashMap, ...).
#[derive(Default)]
struct ReadCtx {
    calls_in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

/// A sleepy tool. `#[tool(..., parallel)]` emits both `impl Tool` and
/// `impl ConcurrentTool`; `execute` takes shared `&ReadCtx`.
#[derive(BorgToolDerive)]
#[tool(
    name = "read_a",
    description = "simulated read of resource A",
    context = ReadCtx,
    input_schema = ::serde_json::json!({"type": "object", "properties": {}}),
    parallel,
)]
struct ReadA;

#[derive(BorgToolDerive)]
#[tool(
    name = "read_b",
    description = "simulated read of resource B",
    context = ReadCtx,
    input_schema = ::serde_json::json!({"type": "object", "properties": {}}),
    parallel,
)]
struct ReadB;

#[derive(BorgToolDerive)]
#[tool(
    name = "read_c",
    description = "simulated read of resource C",
    context = ReadCtx,
    input_schema = ::serde_json::json!({"type": "object", "properties": {}}),
    parallel,
)]
struct ReadC;

/// Shared per-tool body: bump the in-flight gauge, sleep, return.
async fn simulated_read(name: &str, delay_ms: u64, ctx: &ReadCtx) -> ToolCallResult {
    let in_flight = ctx.calls_in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    loop {
        let peak = ctx.peak_in_flight.load(Ordering::SeqCst);
        if in_flight <= peak {
            break;
        }
        if ctx
            .peak_in_flight
            .compare_exchange(peak, in_flight, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            break;
        }
    }
    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    ctx.calls_in_flight.fetch_sub(1, Ordering::SeqCst);
    ToolCallResult::ok(serde_json::json!({"ran": name, "delay_ms": delay_ms})).unwrap()
}

impl ReadA {
    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &ReadCtx,
    ) -> Result<ToolCallResult, BorgError> {
        Ok(simulated_read("read_a", 80, ctx).await)
    }
}

impl ReadB {
    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &ReadCtx,
    ) -> Result<ToolCallResult, BorgError> {
        Ok(simulated_read("read_b", 80, ctx).await)
    }
}

impl ReadC {
    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &ReadCtx,
    ) -> Result<ToolCallResult, BorgError> {
        Ok(simulated_read("read_c", 80, ctx).await)
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "fanout_reader",
    model = "fake/fanout_reader",
    max_tokens = 200,
    system = "Read three things in parallel, then summarize."
)]
struct FanoutReader;

impl FanoutReader {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("Read all three files in parallel.".into())
    }
}

#[tokio::main]
async fn main() -> Result<(), BorgError> {
    let registry = ConcurrentRegistry::<ReadCtx>::new()
        .with(ReadA)
        .with(ReadB)
        .with(ReadC);

    // The registry's context lives inside the dispatcher.
    let pipeline = dsl::agent_loop_parallel(FanoutReader, registry, ReadCtx::default(), 5);

    // One turn batching three tool calls, then a text-only turn to terminate.
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::new(vec![
        FakeOutcome::ok_with_tool_calls(
            "Fanning out three reads.",
            vec![
                ToolCallRequest::new("read_a", serde_json::json!({}), "a"),
                ToolCallRequest::new("read_b", serde_json::json!({}), "b"),
                ToolCallRequest::new("read_c", serde_json::json!({}), "c"),
            ],
        ),
        FakeOutcome::ok("All three reads complete."),
    ]);

    // The gate lives on ExecCtx, not the composition. CollectAll is the
    // default; spelled out here for clarity.
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "parallel_tools_example",
    )
    .with_parallel_tools(BatchFailureMode::CollectAll);

    let started = Instant::now();
    let outcome = execute_with_handoff(pipeline, ctx, HandoffState::new()).await?;
    let elapsed = started.elapsed();

    println!(
        "Pipeline finished in {:?} ({} steps).",
        elapsed,
        outcome.verdicts.len(),
    );
    println!("Each tool sleeps 80ms. Serial would take ~240ms; parallel ~80ms.");

    // < 200ms proves parallel dispatch; serial would need ~240ms (3 × 80ms).
    assert!(
        elapsed < Duration::from_millis(200),
        "expected parallel dispatch (< 200ms), took {elapsed:?}"
    );

    Ok(())
}
