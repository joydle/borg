//! `AgentLlmClient` trait — the only LLM-shaped dependency borg has.
//!
//! Production code wires this to a concrete provider client; tests wire
//! a `FakeLlmClient` with controllable latency, scripted responses, and
//! cancellation observation. Borg never names a specific provider.
//!
//! It is a trait (not a struct) so concurrency tests can run against a
//! deterministic fixture without spinning up a real HTTP client.

use crate::error::BorgError;
use async_trait::async_trait;

/// A single chat-completion request.
///
/// When `tools` is non-empty the model gets the schemas and may respond
/// with `tool_calls` in [`ChatResponse`]; implementations route them to
/// the provider's tool-use field. Empty `tools` means text-only.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChatRequest {
    /// Provider-specific model identifier (e.g. `anthropic/claude-haiku-5-5`),
    /// passed through verbatim — model selection is a consumer concern.
    /// `Cow` so a `model()` literal flows through without allocating.
    pub model: std::borrow::Cow<'static, str>,
    /// System prompt — typically a multi-KB `include_str!` literal.
    /// `Cow::Borrowed(self.system_prompt())` avoids a per-call clone.
    pub system: std::borrow::Cow<'static, str>,
    /// Per-call user prompt. Genuinely dynamic — stays `String`.
    pub user: String,
    pub max_tokens: u32,
    /// Tool schemas the model may invoke; empty disables tool use. The
    /// LLM client serializes each [`crate::tools::ToolSchema`] to its
    /// provider's shape. `Arc<[_]>` so the executor hop is one
    /// `Arc::clone`, not a deep copy.
    pub tools: std::sync::Arc<[crate::tools::ToolSchema]>,
    /// Images the model should SEE alongside `user`, as data URLs or
    /// `http(s)` URLs. Empty is the ordinary text-only request.
    ///
    /// Borg does not know what an image is beyond a string; the client impl
    /// decides how its provider carries them (an OpenAI-style content array,
    /// Anthropic image blocks, whatever). Images travel alongside `tools`,
    /// so a single agent loop can both see and call tools.
    ///
    /// `Arc<[_]>` to match `tools`: the executor hop is one `Arc::clone`, and
    /// data URLs are large enough that a deep copy per turn would show up.
    pub images: std::sync::Arc<[String]>,
}

impl ChatRequest {
    /// Construct a tools-free chat request.
    pub fn new(
        model: impl Into<std::borrow::Cow<'static, str>>,
        system: impl Into<std::borrow::Cow<'static, str>>,
        user: impl Into<String>,
        max_tokens: u32,
    ) -> Self {
        Self {
            model: model.into(),
            system: system.into(),
            user: user.into(),
            max_tokens,
            images: std::sync::Arc::from([] as [String; 0]),
            tools: std::sync::Arc::from([] as [crate::tools::ToolSchema; 0]),
        }
    }
}

/// Chat-completion result.
///
/// When the model invokes tools, the client translates the provider's
/// response into a uniform `Vec<ToolCallRequest>`; empty means text-only.
/// Token counts are `u64` because some providers' per-run counters
/// exceed `u32::MAX`.
///
/// ## Prompt-cache tokens
///
/// `cache_read_tokens` and `cache_creation_tokens` are SUBSETS of
/// `input_tokens` — the OpenAI / OpenRouter convention, where
/// `prompt_tokens` already counts the cached prefix. A client whose
/// provider reports them beside the input instead (Anthropic's native
/// API) adds them into `input_tokens` before building the response, so
/// `input_tokens` always means "every prompt token the call carried".
/// [`ChatResponse::with_cache_tokens`] clamps both to that total.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChatResponse {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Whether this response came from a provider-side cache hit.
    /// Optional — providers that don't expose this report `false`.
    pub cache_hit: bool,
    /// Prompt tokens served from the provider's cache (part of
    /// `input_tokens`). 0 when the provider reports none.
    pub cache_read_tokens: u64,
    /// Prompt tokens written to the provider's cache by this call (part
    /// of `input_tokens`). 0 when the provider reports none.
    pub cache_creation_tokens: u64,
    /// Tool calls the model produced, populated from the provider's
    /// tool-use shape. Empty for text-only responses.
    pub tool_calls: Vec<crate::tools::ToolCallRequest>,
}

impl ChatResponse {
    /// Construct a text-only response: `cache_hit` false, `tool_calls`
    /// empty. Use the builder methods to fill in the extras.
    pub fn new(text: impl Into<String>, input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            text: text.into(),
            input_tokens,
            output_tokens,
            cache_hit: false,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            tool_calls: Vec::new(),
        }
    }

    /// Builder: the prompt-cache split of `input_tokens`. Both counts are
    /// clamped so `read + creation <= input_tokens` (a provider that
    /// over-reports cannot make the uncached remainder negative). A
    /// non-zero `read` also sets `cache_hit`.
    #[must_use]
    pub fn with_cache_tokens(mut self, read: u64, creation: u64) -> Self {
        let read = read.min(self.input_tokens);
        let creation = creation.min(self.input_tokens - read);
        self.cache_read_tokens = read;
        self.cache_creation_tokens = creation;
        if read > 0 {
            self.cache_hit = true;
        }
        self
    }

    /// The call's usage as one value, for [`crate::Budget::record_usage`]
    /// and [`crate::Verdict`].
    pub fn usage(&self) -> crate::budget::TokenUsage {
        crate::budget::TokenUsage::new(self.input_tokens, self.output_tokens)
            .with_cache(self.cache_read_tokens, self.cache_creation_tokens)
    }

    /// Builder: mark the response as a provider cache hit.
    #[must_use]
    pub fn with_cache_hit(mut self, cache_hit: bool) -> Self {
        self.cache_hit = cache_hit;
        self
    }

    /// Builder: attach the model's tool-use response.
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<crate::tools::ToolCallRequest>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Total tokens consumed by this call. Used by the budget tracker.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

/// LLM transport. The single trait borg needs to do its job.
///
/// Implementations MUST be `Send + Sync` so one client can be shared
/// across parallel branches. Object-safe, so callers hold
/// `Arc<dyn AgentLlmClient>`.
#[async_trait]
pub trait AgentLlmClient: Send + Sync {
    /// Run a chat completion. Returns `BorgError::Llm { class }` on
    /// failure; the implementation classifies retryability from whatever
    /// signal its provider exposes.
    ///
    /// Implementations SHOULD honor cancellation. Borg passes a
    /// `CancellationToken` through `AgentContext`; a client holding its
    /// own HTTP client should chain it in via `tokio::select!`.
    async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse, BorgError>;
}

/// LLM client stub for deterministic (non-LLM-calling) agents.
///
/// A deterministic `BorgAgent` still needs an `Arc<dyn AgentLlmClient>`
/// in its `ExecCtx`. This stub fails loudly with `BorgError::Internal`
/// if such an agent accidentally reaches for the LLM, surfacing the
/// programming error rather than masking it.
pub fn deterministic_llm_client() -> std::sync::Arc<dyn AgentLlmClient> {
    std::sync::Arc::new(DeterministicLlm)
}

struct DeterministicLlm;

#[async_trait]
impl AgentLlmClient for DeterministicLlm {
    async fn chat_completion(&self, _request: ChatRequest) -> Result<ChatResponse, BorgError> {
        Err(BorgError::Internal {
            message:
                "deterministic agent reached for ctx.llm — override invoke() to do pure logic, \
                      or supply a real AgentLlmClient if the agent needs the model"
                    .into(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
//   Live token streaming sink — opt-in side channel for chat UIs.
// ─────────────────────────────────────────────────────────────────────
//
// `chat_completion` returns the full response, but a chat UI also wants
// to grow the bubble token-by-token. A `LlmStreamSink` trait + the
// `LLM_STREAM` task-local let a client stream under the hood:
//
//   if let Some(sink) = borg::active_stream_sink() {
//       sink.delta(&chunk).await;
//   }
//
// The consumer scopes the sink around its `execute_*` call:
//
//   borg::LLM_STREAM
//       .scope(my_sink, async { execute_with_handoff(...).await })
//       .await
//
// Streaming is a property of the client, not the agent: borg never
// requires a sink, and any client without one just returns the whole
// response.

use std::sync::Arc;

/// Per-call sink for streaming LLM tokens. Consumers (chat UIs) push
/// each chunk onto a transport (SSE, websocket, broadcast). Optional
/// `custom(kind, payload)` is a structured side channel for events
/// like "progress" or "artifact" without coupling borg to any
/// specific wire format.
#[async_trait]
pub trait LlmStreamSink: Send + Sync {
    /// One text chunk from the LLM. The implementation is responsible
    /// for ordering: deltas arrive in production order from the
    /// originating client.
    async fn delta(&self, text: &str);

    /// Optional structured side channel. Default is a no-op; consumers
    /// override to carry typed events (e.g. `("progress", json)`).
    async fn custom(&self, _kind: &str, _payload: serde_json::Value) {}
}

// Per-task slot. Set by the consumer with `LLM_STREAM.scope(...)` so
// `AgentLlmClient` impls find the active sink without threading it
// through every call signature. (Macro-produced items don't get
// rustdoc — see the matching `TOOL_CALL` note in tools.rs.)
tokio::task_local! {
    pub static LLM_STREAM: StreamSlot;
}

/// Slot wrapper so the task-local can default to "no sink" without
/// forcing every consumer to construct one.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct StreamSlot {
    pub sink: Option<Arc<dyn LlmStreamSink>>,
}

impl StreamSlot {
    /// Construct a slot wrapping a concrete sink.
    pub fn new(sink: Arc<dyn LlmStreamSink>) -> Self {
        Self { sink: Some(sink) }
    }
}

/// Convenience — returns the active sink if one is scoped, else `None`.
/// `AgentLlmClient` impls call this in their stream loop:
///
/// ```ignore
/// if let Some(sink) = borg::active_stream_sink() {
///     sink.delta(&chunk).await;
/// }
/// ```
pub fn active_stream_sink() -> Option<Arc<dyn LlmStreamSink>> {
    LLM_STREAM.try_with(|s| s.sink.clone()).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_tokens_sums_input_and_output() {
        let r = ChatResponse {
            text: "x".into(),
            input_tokens: 100,
            output_tokens: 200,
            cache_hit: false,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            tool_calls: Vec::new(),
        };
        assert_eq!(r.total_tokens(), 300);
    }

    #[test]
    fn total_tokens_saturates_on_overflow() {
        let r = ChatResponse {
            text: "x".into(),
            input_tokens: u64::MAX - 5,
            output_tokens: 100,
            cache_hit: false,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            tool_calls: Vec::new(),
        };
        assert_eq!(r.total_tokens(), u64::MAX);
    }

    #[test]
    fn cache_tokens_are_a_clamped_split_of_the_input() {
        let r = ChatResponse::new("x", 1_000, 10).with_cache_tokens(800, 150);
        assert_eq!((r.cache_read_tokens, r.cache_creation_tokens), (800, 150));
        assert!(r.cache_hit, "a cache read is a cache hit");
        assert_eq!(r.total_tokens(), 1_010, "the split does not add tokens");

        let over = ChatResponse::new("x", 100, 0).with_cache_tokens(500, 500);
        assert_eq!(
            (over.cache_read_tokens, over.cache_creation_tokens),
            (100, 0)
        );

        let write_only = ChatResponse::new("x", 100, 0).with_cache_tokens(0, 60);
        assert!(!write_only.cache_hit, "a cache write alone is not a hit");
        assert_eq!(write_only.usage().cache_creation_tokens, 60);
    }
}
