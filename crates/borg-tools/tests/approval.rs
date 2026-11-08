//! Integration test for `borg_tools::approval::request_approval`.
//!
//! Verifies the helper builds the right SuspendRequest shape and the executor
//! unwinds via BorgError::Suspended.

use async_trait::async_trait;
use borg::prelude::*;
use borg::suspend::SuspendKind;
use borg::testing::FakeLlmClient;
use borg_tools::approval::request_approval;
use std::sync::Arc;

/// Agent that calls `request_approval` and never produces a verdict.
struct GatedAgent;

#[async_trait]
impl BorgAgent for GatedAgent {
    fn agent_id(&self) -> &'static str {
        "gated"
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn system_prompt(&self) -> &'static str {
        ""
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        None
    }

    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        let _decision = request_approval(ctx, "approve migration?", Some("run_migration")).await?;
        unreachable!("request_approval should unwind");
    }
}

#[tokio::test]
async fn request_approval_unwinds_with_correctly_shaped_request() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("unused");
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );

    let result = run_one("step", GatedAgent, ctx, HandoffState::new()).await;

    match result {
        Err(BorgError::Suspended { signal }) => {
            assert_eq!(signal.request.prompt, "approve migration?");
            assert!(matches!(signal.request.kind, SuspendKind::Approval));
            assert_eq!(
                signal.request.blocked_tool.as_deref(),
                Some("run_migration")
            );
        }
        other => panic!("expected Err(Suspended), got {other:?}"),
    }
}

#[tokio::test]
async fn request_approval_without_blocked_tool_leaves_field_none() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("unused");
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );

    struct UngatedAgent;

    #[async_trait]
    impl BorgAgent for UngatedAgent {
        fn agent_id(&self) -> &'static str {
            "ungated"
        }
        fn model(&self) -> &str {
            "test/model"
        }
        fn max_tokens(&self) -> u32 {
            100
        }
        fn system_prompt(&self) -> &'static str {
            ""
        }
        fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
            None
        }
        async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
            let _ = request_approval(ctx, "approve?", None).await?;
            unreachable!();
        }
    }

    let result = run_one("step", UngatedAgent, ctx, HandoffState::new()).await;
    match result {
        Err(BorgError::Suspended { signal }) => {
            assert!(signal.request.blocked_tool.is_none());
        }
        other => panic!("expected Err(Suspended), got {other:?}"),
    }
}
