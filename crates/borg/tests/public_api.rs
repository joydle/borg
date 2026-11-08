//! Integration test exercising only the public API — the published-API
//! smoke test. Lives under `tests/` so it sees `borg` as an external
//! crate: if it ever needs a `pub(crate)` path, the public surface is
//! missing something. Breadth-of-features tests live in `src/tests.rs`.
//!
//! Run via `cargo test -p borg --test public_api --features testing`.
//! Gated on `feature = "testing"` for the fakes (`FakeLlmClient`,
//! `FakeOutcome`).

#![cfg(feature = "testing")]

use borg::prelude::*;
use borg::testing::{FakeLlmClient, FakeOutcome};
use std::sync::Arc;

// ── Agent definitions via the prelude alone ────────────────────────────

#[derive(BorgAgentDerive)]
#[borg(
    id = "echo",
    model = "fake/echo",
    max_tokens = 100,
    system = "Echo the input as JSON."
)]
struct Echo;

impl Echo {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("ping".into())
    }
}

#[derive(serde::Deserialize, Default, PartialEq, Debug)]
struct EchoOut {
    msg: String,
}

// ── Tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn run_one_via_prelude() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("body text");
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let v = run_one("echo", Echo, ctx, HandoffState::new())
        .await
        .expect("run_one should succeed");
    assert!(v.body.contains("body text"));
}

#[tokio::test]
async fn run_one_typed_via_prelude() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok(r#"{"msg":"hello"}"#);
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let out: EchoOut = run_one_typed("echo", Echo, ctx, HandoffState::new())
        .await
        .expect("typed parse should succeed");
    assert_eq!(
        out,
        EchoOut {
            msg: "hello".into()
        }
    );
}

#[tokio::test]
async fn sequential_composition_via_prelude() {
    let llm: Arc<dyn AgentLlmClient> =
        FakeLlmClient::new(vec![FakeOutcome::ok("first"), FakeOutcome::ok("second")]);
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let pipeline = Composition::sequential([
        Composition::single("a", Echo),
        Composition::single("b", Echo),
    ]);
    let outcome = execute(pipeline, ctx).await.expect("pipeline runs");
    assert_eq!(outcome.verdicts.len(), 2);
    assert!(outcome.verdicts[0].verdict.body.contains("first"));
    assert!(outcome.verdicts[1].verdict.body.contains("second"));
}

#[tokio::test]
async fn cancellation_propagates_through_prelude() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("never");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let ctx = ExecCtx::new(llm, in_memory_budget(1_000), noop_emitter(), cancel, "test");
    let result = run_one("echo", Echo, ctx, HandoffState::new()).await;
    assert!(matches!(result, Err(BorgError::Cancelled)));
}

#[tokio::test]
async fn restart_policy_via_prelude() {
    // Two retryable failures, then success on attempt 3 (under
    // on_failure(3)).
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("transient 1"),
        FakeOutcome::retryable_err("transient 2"),
        FakeOutcome::ok("third time's the charm"),
    ]);
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let v = run_one_with_restart(
        "echo",
        Echo,
        RestartPolicy::on_failure(3),
        ctx,
        HandoffState::new(),
    )
    .await
    .expect("restart policy should recover");
    assert!(v.body.contains("third time"));
}

// ── #[derive(BorgTool)] proves the macro generates a working Tool ──────

#[derive(borg::BorgToolDerive)]
#[tool(
    name = "uppercase",
    description = "Uppercases the given text.",
    context = (),
    input_schema = ::serde_json::json!({
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
    }),
)]
struct Uppercase;

impl Uppercase {
    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &mut (),
    ) -> Result<borg::ToolCallResult, BorgError> {
        let text = input["text"].as_str().ok_or_else(|| BorgError::Internal {
            message: "missing `text` arg".into(),
        })?;
        borg::ToolCallResult::ok(text.to_uppercase())
    }
}

#[tokio::test]
async fn borg_tool_derive_generates_working_impl() {
    use borg::tools::{Registry, ToolCallRequest};

    let registry = Registry::<()>::new().with(Uppercase);
    let schemas = registry.schemas();
    assert_eq!(schemas.len(), 1);
    assert_eq!(schemas[0].name, "uppercase");
    assert!(schemas[0].description.contains("Uppercases"));

    let mut ctx = ();
    let dispatched = registry
        .dispatch(
            ToolCallRequest::new(
                "uppercase",
                serde_json::json!({ "text": "hello" }),
                "test-1",
            ),
            &mut ctx,
        )
        .await
        .expect("dispatch succeeds");
    assert_eq!(dispatched.result.output.unwrap(), "HELLO");
}

#[tokio::test]
async fn parse_failure_carries_excerpt_diagnostic() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("definitely not json {{{");
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let result: Result<EchoOut, _> = run_one_typed("echo", Echo, ctx, HandoffState::new()).await;
    match result {
        Err(BorgError::Internal { message }) => {
            assert!(
                message.contains("excerpt"),
                "expected excerpt diagnostic: {message}"
            );
            assert!(message.contains("EchoOut"), "expected type name: {message}");
        }
        other => panic!("expected Internal parse error, got {other:?}"),
    }
}

// ── public surface: root re-exports, constructors for #[non_exhaustive] ──

#[test]
fn agent_context_from_testing_helper() {
    let mut handoff = HandoffState::new();
    handoff.set("k", "v");
    let mut ctx = borg::testing::agent_context(handoff, FakeLlmClient::always_ok("x"));
    ctx.turn_number = 3;
    assert_eq!(ctx.turn_number, 3);
    assert_eq!(ctx.handoff.get("k"), Some(&serde_json::json!("v")));
    assert_eq!(Echo.user_prompt(&ctx).as_deref(), Some("ping"));
}

#[test]
fn non_exhaustive_types_build_through_constructors() {
    use borg::guardrails::rules::*;
    use std::time::Duration;

    let engine = GuardrailEngine::new()
        .with_rule(MaxContinuationsRule::new(30))
        .with_rule(TextOnlyEndTurnRule::new(3))
        .with_rule(
            RepeatedToolCallRule::new("read_file", 3, 5, 8)
                .with_window(Duration::from_secs(60))
                .with_messages("warn", "throttle")
                .with_throttle_max_tokens(500),
        )
        .with_rule(IdenticalOutputRule::new("model:text", 3))
        .with_rule(ConsecutiveFailureRule::new("build", 3, 6))
        .with_rule(RunawayCostRule::new(0.95).with_warn_at(0.8));
    let _ = engine;

    let pf = ProviderFailure::new(FailureClass::Throttled, "slow down")
        .with_status(429)
        .with_retry_after(Duration::from_secs(2));
    assert_eq!(pf.http_status, Some(429));
    assert_eq!(
        ProviderFailure::classify_message("ECONNRESET").class,
        FailureClass::Retryable
    );

    let snap = borg::Snapshot::empty("fp".into(), 0);
    assert_eq!(snap.schema_version, borg::SNAPSHOT_SCHEMA_VERSION);
    assert_eq!(
        borg::CheckpointPolicy::default(),
        borg::CheckpointPolicy::OnLoopIteration
    );
}

#[test]
fn builders_and_stream_sink_are_nameable_at_root() {
    let single: borg::SingleBuilder = Composition::single_builder("echo", Box::new(Echo));
    let _: Composition = single.build();
    let _: borg::SequentialBuilder = Composition::sequential_builder();
    // No sink scoped outside `LLM_STREAM.scope(..)`.
    let sink: Option<Arc<dyn borg::LlmStreamSink>> = borg::active_stream_sink();
    assert!(sink.is_none());
    let _ = borg::RETRYABLE_HTTP_STATUS;
}
