//! Tool-loop example — `Composition::Agent` end-to-end.
//!
//! 1. Two tools (`reverse`, `uppercase`) wired into a [`Registry`] +
//!    [`MutexDispatcher`].
//! 2. One agent (`StringWorker`) that emits tool calls until it has both
//!    transformed strings, then returns its final answer.
//! 3. A [`FakeLlmClient`] script driving the loop deterministically:
//!    turn 1 calls `reverse`, turn 2 `uppercase`, turn 3 the summary.
//! 4. [`agent_loop`] wraps it as a `Composition` run via
//!    `execute_with_handoff`.
//!
//! Run with: `cargo run --example tool_loop -p borg --features testing`

use borg::testing::{FakeLlmClient, FakeOutcome};
use borg::tools::{MutexDispatcher, Registry, ToolCallRequest, ToolDispatch};
use borg::*;
use std::sync::Arc;

// ── Two tools via #[derive(BorgToolDerive)] ─────────────────────────
//
// Both use `()` as context. For tools needing shared state (DB handle,
// cache, ...), pass it via `MutexDispatcher::new(registry, ctx)` and use
// `Registry<MyCtx>`.

#[derive(BorgToolDerive)]
#[tool(
    name = "reverse",
    description = "Reverse the characters in a string.",
    context = (),
    input_schema = ::serde_json::json!({
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
    }),
)]
struct ReverseTool;

impl ReverseTool {
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &mut (),
    ) -> Result<ToolCallResult, BorgError> {
        let text = input["text"].as_str().ok_or_else(|| BorgError::Internal {
            message: "missing `text` arg".into(),
        })?;
        ToolCallResult::ok(text.chars().rev().collect::<String>())
    }
}

#[derive(BorgToolDerive)]
#[tool(
    name = "uppercase",
    description = "Uppercase every letter in a string.",
    context = (),
    input_schema = ::serde_json::json!({
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
    }),
)]
struct UppercaseTool;

impl UppercaseTool {
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &mut (),
    ) -> Result<ToolCallResult, BorgError> {
        let text = input["text"].as_str().ok_or_else(|| BorgError::Internal {
            message: "missing `text` arg".into(),
        })?;
        ToolCallResult::ok(text.to_uppercase())
    }
}

// ── One tool-loop agent via #[derive(BorgAgentDerive)] ──────────────
//
// The default `extract_tool_calls` reads `verdict.tool_calls`, which
// `FakeLlmClient::ok_with_tool_calls` populates — no override needed.

#[derive(BorgAgentDerive)]
#[borg(
    id = "string_worker",
    model = "fake/string_worker",
    max_tokens = 500,
    system = "You transform strings. Call `reverse` then `uppercase` then \
              return a one-line summary of the result."
)]
struct StringWorker;

impl StringWorker {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("Transform 'hello world': reverse it, then uppercase it.".into())
    }
}

#[tokio::main]
async fn main() -> Result<(), BorgError> {
    // MutexDispatcher erases the registry's context to the ToolDispatch
    // trait that Composition::Agent needs.
    let registry = Registry::<()>::new().with(ReverseTool).with(UppercaseTool);
    let dispatcher: Arc<dyn ToolDispatch> = Arc::new(MutexDispatcher::new(registry, ()));

    // Scripted turns: 1 = `reverse`, 2 = `uppercase` (on the prior
    // result), 3 = text only. The loop terminates when the agent stops
    // emitting tool calls.
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::new(vec![
        FakeOutcome::ok_with_tool_calls(
            "Calling reverse...",
            vec![ToolCallRequest::new(
                "reverse",
                serde_json::json!({ "text": "hello world" }),
                "rev-1",
            )],
        ),
        FakeOutcome::ok_with_tool_calls(
            "Calling uppercase...",
            vec![ToolCallRequest::new(
                "uppercase",
                serde_json::json!({ "text": "dlrow olleh" }),
                "up-1",
            )],
        ),
        // Text only, no tool calls → loop terminates.
        FakeOutcome::ok("Reversed and uppercased: DLROW OLLEH"),
    ]);

    // `agent_loop` is dsl shorthand for `Composition::agent(name, agent,
    // dispatcher, max_steps)`; here a per-step cap of 10 turns.
    let pipeline = dsl::agent_loop(StringWorker, dispatcher, 10);

    // `noop_emitter` is fine for examples; production wires a
    // TracingEmitter or domain emitter.
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "tool_loop_example",
    );

    let outcome = execute_with_handoff(pipeline, ctx, HandoffState::new()).await?;

    // The final verdict body is the agent's parting text. Intermediate
    // tool calls are observable via the EventEmitter (wire TracingEmitter
    // to see the borg.composition → step → turn → tool span tree).
    println!(
        "Steps: {}  tokens(in={}, out={})",
        outcome.verdicts.len(),
        outcome.total_input_tokens(),
        outcome.total_output_tokens(),
    );
    for v in &outcome.verdicts {
        println!(
            "  {:<14} body={:?}",
            v.step_name,
            v.verdict.body.lines().next().unwrap_or(""),
        );
    }

    Ok(())
}
