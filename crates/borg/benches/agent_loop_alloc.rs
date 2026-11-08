//! Per-turn allocation hotspots in the agent loop.
//!
//! * **`quadratic_to_value`** — `serde_json::to_value(&accumulated_tool_results)`
//!   is called once per step on a vector that grows by one per step, so total
//!   work over a `max_steps = N` loop is O(N²). Reproduced so a future fix
//!   (append-only `Value::Array`) shows up as a slope change.
//!
//! * **`schemas_call`** — `Registry::schemas()` runs every step and rebuilds
//!   `Vec<ToolSchema>` from scratch. Measures per-call cost as tool count grows.
//!
//! * **`chat_request_alloc`** — `ChatRequest` construction allocates two owned
//!   `String`s (`model`, `system`) plus a deep clone of `tool_schemas`,
//!   isolated from any LLM call.
//!
//! Run with: `cargo bench -p borg --bench agent_loop_alloc`

use borg::{ChatRequest, DispatchedCall, Registry, Tool, ToolCallResult, ToolSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ── fixture: a no-op Tool that just returns its declared schema ─────
//
// We don't dispatch here — just populate a `Registry` so `schemas()`
// has work. Each tool carries a distinct schema so the bench reflects
// the cost of cloning many distinct `ToolSchema` owned-string fields.

struct DummyTool {
    schema: ToolSchema,
}

impl DummyTool {
    fn new(idx: usize) -> Self {
        Self {
            schema: ToolSchema::new(
                format!("dummy_tool_{idx}"),
                "A no-op tool used only to populate the bench Registry.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    },
                    "required": ["value"],
                }),
            ),
        }
    }
}

#[async_trait::async_trait]
impl Tool for DummyTool {
    type Context = ();

    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &mut (),
    ) -> Result<ToolCallResult, borg::BorgError> {
        ToolCallResult::ok("noop")
    }
}

fn build_registry(n: usize) -> Registry<()> {
    let mut r = Registry::<()>::new();
    for i in 0..n {
        r.register(DummyTool::new(i));
    }
    r
}

/// Realistic `DispatchedCall` (~200 bytes of JSON output) — what an
/// agent step appends to `accumulated_tool_results`.
fn dispatched_call_fixture(idx: usize) -> DispatchedCall {
    let result = ToolCallResult::ok(serde_json::json!({
        "status": "ok",
        "duration_ms": 42,
        "summary": format!("step-{idx} completed: read 3 files, modified 1, ran 2 tests"),
        "artifacts": ["a.rs", "b.rs", "c.rs"],
    }))
    .expect("fixture serialize");
    DispatchedCall::new(format!("call-{idx}"), "fixture_tool", result)
}

// ── Bench A: quadratic to_value over a growing Vec<DispatchedCall> ──
//
// Re-serializes the full vec each step. Regression guard: if a refactor
// reintroduces this shape, the slope spikes vs `linear_append` below.

fn quadratic_to_value(c: &mut Criterion) {
    let mut g = c.benchmark_group("agent_loop/quadratic_to_value");
    for &max_steps in &[1usize, 5, 10, 25] {
        // Σ(k+1) for k in 0..N → O(N²) total work over N steps.
        g.throughput(Throughput::Elements(max_steps as u64));
        g.bench_with_input(
            BenchmarkId::new("steps", max_steps),
            &max_steps,
            |b, &steps| {
                b.iter(|| {
                    let mut acc: Vec<DispatchedCall> = Vec::new();
                    for step in 0..steps {
                        acc.push(dispatched_call_fixture(step));
                        let v = serde_json::to_value(&acc).expect("to_value");
                        black_box(v);
                    }
                });
            },
        );
    }
    g.finish();
}

// ── Bench A': linear append-only Value::Array (current pattern) ─────
//
// What `execute_agent_loop` does now: keep a `Value::Array`, `to_value`
// only the new `DispatchedCall` per step, push it. O(1) per step.

fn linear_append(c: &mut Criterion) {
    let mut g = c.benchmark_group("agent_loop/linear_append");
    for &max_steps in &[1usize, 5, 10, 25] {
        g.throughput(Throughput::Elements(max_steps as u64));
        g.bench_with_input(
            BenchmarkId::new("steps", max_steps),
            &max_steps,
            |b, &steps| {
                b.iter(|| {
                    let mut acc: Vec<DispatchedCall> = Vec::with_capacity(steps);
                    let mut acc_json: serde_json::Value =
                        serde_json::Value::Array(Vec::with_capacity(steps));
                    for step in 0..steps {
                        let d = dispatched_call_fixture(step);
                        let dj = serde_json::to_value(&d).expect("to_value");
                        if let serde_json::Value::Array(arr) = &mut acc_json {
                            arr.push(dj);
                        }
                        acc.push(d);
                        black_box(&acc_json);
                    }
                });
            },
        );
    }
    g.finish();
}

// ── Bench B: Registry::schemas() per-call cost ───────────────────────

fn schemas_call(c: &mut Criterion) {
    let mut g = c.benchmark_group("agent_loop/schemas_call");
    for &n_tools in &[5usize, 20, 100] {
        let registry = build_registry(n_tools);
        g.throughput(Throughput::Elements(n_tools as u64));
        g.bench_with_input(BenchmarkId::new("n_tools", n_tools), &n_tools, |b, _| {
            b.iter(|| {
                let schemas = registry.schemas();
                black_box(schemas);
            });
        });
    }
    g.finish();
}

// ── Bench C: ChatRequest construction (model + system + tools clone) ─

fn chat_request_alloc(c: &mut Criterion) {
    let mut g = c.benchmark_group("agent_loop/chat_request_alloc");
    // Realistic ~2 KB system prompt, as a `&'static str` so the bench
    // captures the per-call `.to_string()` cost.
    static SYSTEM: &str = concat!(
        "You are a helpful agent that orchestrates a sequence of tool calls. ",
        "Always prefer tool calls over speculation. Be concise. ",
        "Surface failures as structured errors. Never claim success without ",
        "evidence. Respect the budget; if you run out of token headroom, ",
        "return a partial result with the artifacts you do have. ",
        "When tools are present in the toolchain you may call them by name; ",
        "when none are present, fall back to plain text answers. ",
        "Avoid duplicating prior turns. Do not invent tool names. ",
    );

    for &n_tools in &[0usize, 5, 20, 100] {
        let registry = build_registry(n_tools);
        let schemas = registry.schemas();
        g.throughput(Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::new("n_tools", n_tools), &schemas, |b, sc| {
            b.iter(|| {
                // `ChatRequest` is `#[non_exhaustive]`, so build via the
                // constructor + field mutate, not a struct expression.
                let mut req =
                    ChatRequest::new("anthropic/claude-haiku-5-5", SYSTEM, "do the thing", 1024);
                req.tools = sc.clone();
                black_box(req);
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    quadratic_to_value,
    linear_append,
    schemas_call,
    chat_request_alloc
);
criterion_main!(benches);
