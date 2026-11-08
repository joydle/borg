//! An agent that holds TOOLS can now also SEE.
//!
//! `BorgAgent::images` lands on `ChatRequest::images` next to the prompt and
//! the tool schemas, so one agent loop can both look at images and call tools.
//! These tests pin what the loop actually sends to the client.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use borg::{
    dsl, execute_with_handoff, in_memory_budget, noop_emitter, AgentContext, AgentLlmClient,
    BorgAgentDerive, BorgError, ChatRequest, ChatResponse, ExecCtx, HandoffState,
};
use tokio_util::sync::CancellationToken;

/// Records what the loop actually SENT. That is the only thing worth
/// asserting: a field that exists but never reaches the client is a field the
/// agent cannot use, and the failure would be silent.
#[derive(Default)]
struct RecordingClient {
    seen: Mutex<Vec<ChatRequest>>,
}

#[async_trait]
impl AgentLlmClient for RecordingClient {
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, BorgError> {
        self.seen.lock().unwrap().push(request);
        Ok(ChatResponse::new("OK", 1, 1))
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "text_only",
    model = "fake/text",
    max_tokens = 100,
    system = "you are a test"
)]
struct TextAgent;

impl TextAgent {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("look at nothing".to_string())
    }
}

/// The derive does not generate `images`, so this override lands on the trait
/// itself — which is exactly why every existing derived agent is unaffected.
#[async_trait]
impl borg::BorgAgent for VisionAgentWithEyes {
    fn agent_id(&self) -> &'static str {
        "vision_eyes"
    }
    fn model(&self) -> &str {
        "fake/vision"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn system_prompt(&self) -> &'static str {
        "you are a reviewer"
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("is this readable?".to_string())
    }
    fn images(&self, _ctx: &AgentContext) -> Vec<String> {
        vec![
            "data:image/jpeg;base64,AAAA".to_string(),
            "https://example.test/frame.png".to_string(),
        ]
    }
}

struct VisionAgentWithEyes;

async fn run(agent: impl borg::BorgAgent + 'static, client: Arc<RecordingClient>) {
    let ctx = ExecCtx::new(
        client,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    );
    let pipeline = dsl::agent_loop(
        agent,
        Arc::new(borg::tools::MutexDispatcher::new(
            borg::tools::Registry::<()>::new(),
            (),
        )),
        1,
    );
    let _ = execute_with_handoff(pipeline, ctx, HandoffState::new()).await;
}

/// The default is empty, so an agent written before this field behaves exactly
/// as it did. That is the entire reason the change is additive.
#[tokio::test]
async fn an_agent_that_never_mentions_images_sends_none() {
    let client = Arc::new(RecordingClient::default());
    run(TextAgent, client.clone()).await;
    let seen = client.seen.lock().unwrap();
    assert!(!seen.is_empty(), "the loop must have called the client");
    assert!(
        seen[0].images.is_empty(),
        "a text agent must send no images"
    );
}

/// And an agent that overrides `images()` gets them onto the wire, in order,
/// ALONGSIDE its prompt rather than instead of it.
#[tokio::test]
async fn an_agent_with_eyes_puts_its_frames_on_the_request() {
    let client = Arc::new(RecordingClient::default());
    run(VisionAgentWithEyes, client.clone()).await;
    let seen = client.seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert_eq!(
        seen[0].images.len(),
        2,
        "both frames must reach the client — a dropped one is a review of a \
         screen the agent never saw, and nothing would report it"
    );
    assert_eq!(seen[0].images[0], "data:image/jpeg;base64,AAAA");
    assert_eq!(
        seen[0].images[1], "https://example.test/frame.png",
        "order is preserved: 'image 1 is before, image 2 is after' is a real \
         contract in review prompts"
    );
    assert_eq!(
        seen[0].user, "is this readable?",
        "images ACCOMPANY the prompt; they do not replace it"
    );
}
