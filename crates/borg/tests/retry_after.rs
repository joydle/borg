//! A provider's `Retry-After` is a floor under the restart backoff.
//!
//! An in-flight-budget refusal settles on the PROVIDER's clock: retrying on a
//! 200 ms exponential ladder against a 120 s `Retry-After` is the same failure
//! five times, sooner — and then a dead run that had already paid for its work.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use borg::{
    in_memory_budget, noop_emitter, run_one_with_restart, AgentContext, AgentLlmClient,
    BorgAgentDerive, BorgError, ChatRequest, ChatResponse, ExecCtx, HandoffState, RestartPolicy,
    RetryClass,
};
use tokio_util::sync::CancellationToken;

/// Refuses the first call with a retry hint, answers the second.
struct RefusesOnce {
    calls: AtomicUsize,
    hint: Option<Duration>,
}

#[async_trait]
impl AgentLlmClient for RefusesOnce {
    async fn chat_completion(&self, _request: ChatRequest) -> Result<ChatResponse, BorgError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let e = BorgError::llm("429 in-flight budget", RetryClass::Retryable);
            return Err(match self.hint {
                Some(h) => e.with_retry_after(h),
                None => e,
            });
        }
        Ok(ChatResponse::new("OK", 1, 1))
    }
}

#[derive(BorgAgentDerive)]
#[borg(id = "caller", model = "fake/model", max_tokens = 10, system = "s")]
struct Caller;

impl Caller {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("go".into())
    }
}

async fn elapsed_until_success(hint: Option<Duration>) -> Duration {
    let client = Arc::new(RefusesOnce {
        calls: AtomicUsize::new(0),
        hint,
    });
    let ctx = ExecCtx::new(
        client.clone(),
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let started = tokio::time::Instant::now();
    run_one_with_restart(
        "caller",
        Caller,
        RestartPolicy::on_failure(2).without_backoff(),
        ctx,
        HandoffState::new(),
    )
    .await
    .expect("the second attempt answers");
    assert_eq!(client.calls.load(Ordering::SeqCst), 2);
    started.elapsed()
}

#[tokio::test(start_paused = true)]
async fn a_restart_waits_out_the_providers_retry_after() {
    let waited = elapsed_until_success(Some(Duration::from_secs(120))).await;
    assert!(
        waited >= Duration::from_secs(120),
        "retried after {waited:?}, before the provider's 120 s"
    );
}

#[tokio::test(start_paused = true)]
async fn without_a_hint_the_backoff_alone_decides() {
    let waited = elapsed_until_success(None).await;
    assert!(
        waited < Duration::from_secs(1),
        "no backoff, no hint: {waited:?}"
    );
}
