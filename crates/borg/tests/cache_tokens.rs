//! Prompt-cache tokens reach the places that report and charge them.
//!
//! Before `ChatResponse` carried a cache split, `TurnCompleted` and the turn
//! span reported `cache_read_tokens: 0` for every call, and the budget charged
//! a cached prefix as if it were fresh — a consumer that wanted the discount
//! had to lie about `input_tokens` to get it, which then lied to every reader
//! of the verdict too.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use borg::{
    dsl, execute_with_handoff, AgentContext, AgentLlmClient, BorgAgentDerive, BorgError, BorgEvent,
    Budget, ChatRequest, ChatResponse, EventEmitter, ExecCtx, HandoffState, InMemoryBudget,
};
use tokio_util::sync::CancellationToken;

/// 20K prompt tokens, 18K of them read from the cache and 1K written to it.
struct CachedClient;

#[async_trait]
impl AgentLlmClient for CachedClient {
    async fn chat_completion(&self, _request: ChatRequest) -> Result<ChatResponse, BorgError> {
        Ok(ChatResponse::new("OK", 20_000, 500).with_cache_tokens(18_000, 1_000))
    }
}

#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<Arc<BorgEvent>>>,
}

#[async_trait]
impl EventEmitter for Recorder {
    async fn emit(&self, event: Arc<BorgEvent>) {
        self.events.lock().unwrap().push(event);
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "reviewer",
    model = "fake/model",
    max_tokens = 100,
    system = "review"
)]
struct Reviewer;

impl Reviewer {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("review it".to_string())
    }
}

async fn run(budget: Arc<dyn Budget>) -> (Vec<Arc<BorgEvent>>, borg::Verdict) {
    let recorder = Arc::new(Recorder::default());
    let ctx = ExecCtx::new(
        Arc::new(CachedClient),
        budget,
        recorder.clone(),
        CancellationToken::new(),
        "test",
    );
    let pipeline = dsl::agent_loop(
        Reviewer,
        Arc::new(borg::tools::MutexDispatcher::new(
            borg::tools::Registry::<()>::new(),
            (),
        )),
        1,
    );
    let outcome = execute_with_handoff(pipeline, ctx, HandoffState::new())
        .await
        .expect("one turn");
    let verdict = outcome.verdicts.last().expect("a verdict").verdict.clone();
    let events = recorder.events.lock().unwrap().clone();
    (events, verdict)
}

#[tokio::test]
async fn turn_completed_reports_the_cache_split_the_client_returned() {
    let (events, verdict) = run(borg::in_memory_budget(100_000)).await;
    let turn = events
        .iter()
        .find_map(|e| match e.as_ref() {
            BorgEvent::TurnCompleted {
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                ..
            } => Some((*output_tokens, *cache_read_tokens, *cache_creation_tokens)),
            _ => None,
        })
        .expect("the loop emits TurnCompleted");
    assert_eq!(turn, (500, 18_000, 1_000));
    assert_eq!(
        verdict.input_tokens, 20_000,
        "input stays the real prompt size"
    );
    assert_eq!(
        (verdict.cache_read_tokens, verdict.cache_creation_tokens),
        (18_000, 1_000)
    );
}

#[tokio::test]
async fn the_default_budget_charges_every_token() {
    let budget = borg::in_memory_budget(100_000);
    run(budget.clone()).await;
    assert_eq!(
        budget.consumed(),
        20_500,
        "nothing changes for a budget that did not opt in"
    );
}

#[tokio::test]
async fn a_budget_that_opts_in_charges_cache_reads_at_its_rate() {
    let budget: Arc<dyn Budget> =
        Arc::new(InMemoryBudget::new(100_000).with_cache_read_percent(10));
    run(budget.clone()).await;
    // 2_000 uncached (the 1_000 cache write among them) + 18_000 at 10% + 500.
    assert_eq!(budget.consumed(), 2_000 + 1_800 + 500);
}
