//! Integration test for the human-in-the-loop suspend primitive.
//!
//! An agent calls `ctx.request_suspend(req)` mid-invoke and the executor
//! unwinds via `BorgError::Suspended { signal }`, carrying the original
//! request back to the caller.
//!
//! Run via `cargo test -p borg --test suspend_unwind --features testing`.

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::prelude::*;
use borg::suspend::{SuspendKind, SuspendRequest};
use borg::testing::FakeLlmClient;
use std::sync::Arc;

/// Agent whose `invoke()` immediately requests a suspend instead of
/// producing a verdict. Used to exercise the executor's unwind path.
struct SuspendingAgent;

#[async_trait]
impl BorgAgent for SuspendingAgent {
    fn agent_id(&self) -> &'static str {
        "suspender"
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
        let _decision = ctx
            .request_suspend(
                SuspendRequest::new("approve migration?", SuspendKind::Approval)
                    .with_blocked_tool("run_migration"),
            )
            .await?;
        unreachable!("request_suspend unwinds; this line never runs");
    }
}

#[tokio::test]
async fn request_suspend_unwinds_with_suspended_error() {
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("unused");
    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );

    let result = run_one("step", SuspendingAgent, ctx, HandoffState::new()).await;

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
async fn suspended_error_is_not_retryable() {
    // Suspend is a control unwind, not a transient failure. Restart
    // policies must NOT loop on it.
    let req = SuspendRequest::new("approve?", SuspendKind::Approval);
    let signal = borg::suspend::SuspendSignal::from_request(req);
    let err = BorgError::Suspended { signal };
    assert!(!err.is_retryable());
    assert_eq!(err.failure_cause(), "suspended");
}
