//! `BorgAgent` — the trait every concrete agent implements.
//!
//! A borg agent is a single-shot LLM invocation: build prompt, call LLM,
//! parse response into a `Verdict`. Richer behavior (multi-turn loops,
//! tool use, side effects) is composed from multiple `BorgAgent` impls
//! via `Composition`.
//!
//! The default `invoke()` handles the common flow (`system_prompt` +
//! `user_prompt(ctx)` → `ctx.llm.chat_completion` → token recording →
//! `Verdict::from_response`); override it for custom prompt assembly,
//! side effects, or multi-step parsing.
//!
//! Cancellation contract: implementations MUST honor `ctx.cancel`. The
//! default `invoke()` does this via `tokio::select!`; custom impls must
//! too, or work leaks after a `Stop`.

use crate::context::AgentContext;
use crate::error::BorgError;
use crate::llm::ChatRequest;
use crate::verdict::Verdict;
use async_trait::async_trait;

/// Provider the agent's `model()` string is routed to.
///
/// Informational today — `AgentLlmClient` impls decide their own routing
/// (typically OpenRouter, which multiplexes the named providers).
/// Surfacing it as a typed signal lets observers and future router layers
/// avoid String-matching `model()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Provider {
    /// Multiplexing default. The upstream provider is encoded in the
    /// `model()` string (e.g. `anthropic/claude-sonnet-5.5`).
    OpenRouter,
    /// Direct Anthropic API.
    Anthropic,
    /// Direct OpenAI API.
    OpenAI,
    /// AWS Bedrock-hosted models.
    Bedrock,
    /// Google Gemini API (direct, not via OpenRouter).
    Google,
    /// Together.ai.
    Together,
    /// Escape hatch for unenumerated providers; observers can't render
    /// it specially.
    Custom(&'static str),
}

impl Provider {
    /// Short kebab-case label for events and structured logs. Stable
    /// vocabulary — new providers add a new tag, never widen an existing
    /// one.
    pub fn as_str(&self) -> &str {
        match self {
            Provider::OpenRouter => "openrouter",
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
            Provider::Bedrock => "bedrock",
            Provider::Google => "google",
            Provider::Together => "together",
            Provider::Custom(s) => s,
        }
    }
}

/// Boxed-trait-object form every composition holds.
pub type BoxedAgent = Box<dyn BorgAgent>;

/// Contract every agent implements. Object-safe — composition primitives
/// hold `Box<dyn BorgAgent>`.
///
/// # Typed outputs
///
/// For type-safe parsing of a verdict body, see [`crate::run_one_typed`].
/// The output type is chosen at the call site via turbofish, not an
/// associated type: the same agent may be parsed as different shapes by
/// different consumers, and `type Output` would force every impl to
/// restate `type Output = Verdict;`.
///
/// ```ignore
/// #[derive(serde::Deserialize, Default)]
/// struct ReviewSuggestions { suggestions: Vec<String> }
///
/// let typed: ReviewSuggestions =
///     borg::run_one_typed("review", Review, ctx, h).await?;
/// ```
#[async_trait]
pub trait BorgAgent: Send + Sync {
    /// Stable identifier for this agent class. Appears in events, debug
    /// dumps, and `BorgError::ChildPanicked`. `&'static str` makes
    /// classes 1:1 with names.
    fn agent_id(&self) -> &'static str;

    /// Provider-specific model id, passed verbatim to
    /// `ChatRequest::model`. Default `"deterministic"` covers agents
    /// that don't call an LLM (lint, type check, JSON patching).
    fn model(&self) -> &str {
        "deterministic"
    }

    /// Provider the `model()` string is routed to. Override when the
    /// agent pins to a specific direct-API provider. Informational
    /// today — no router layer reads it.
    fn provider(&self) -> Provider {
        Provider::OpenRouter
    }

    /// Max output tokens for this agent's call. Default `0` for
    /// deterministic agents.
    fn max_tokens(&self) -> u32 {
        0
    }

    /// System prompt — the role/instructions block. Default empty.
    fn system_prompt(&self) -> &'static str {
        ""
    }

    /// Build the user prompt from ambient context. `None` means
    /// "decline to run" — the executor records an empty verdict and
    /// moves on. The default `None` short-circuits the default
    /// `invoke()`, so a deterministic agent that overrides `invoke()`
    /// directly never reaches this path.
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        None
    }

    /// Images this agent needs the model to SEE, as data URLs or `http(s)`
    /// URLs. Empty (the default) is an ordinary text-only turn, so every
    /// existing agent is unaffected.
    ///
    /// The sibling of `user_prompt`: images ride on the same request as the
    /// prompt and the tool registry, so an agent can both look at something
    /// and call tools in the same loop. The derive macro does not generate
    /// this method; derived agents inherit the empty default.
    fn images(&self, _ctx: &AgentContext) -> Vec<String> {
        Vec::new()
    }

    /// Tokens this agent recognizes on line 1 of its response. Empty
    /// slice means no verdict-token contract — `Branch` predicates
    /// won't route on it, but `parse_json` on the body still works.
    /// Used by both LLM and deterministic agents.
    fn known_verdicts(&self) -> &'static [&'static str] {
        &[]
    }

    /// Estimated minimum tokens this agent will consume, used by the
    /// budget admission gate before `Composition::Parallel` fans out.
    /// Override with a conservative undercount: a near-budget agent
    /// still gets admitted, an over-budget one just degrades the
    /// parallel.
    fn min_tokens_estimate(&self) -> u64 {
        0
    }

    /// Extract tool-call requests from a completed verdict.
    ///
    /// Default reads [`Verdict::tool_calls`], populated by the
    /// [`crate::AgentLlmClient`] from the provider's native tool_use
    /// channel — enough for most tool-loop agents. Override when the
    /// LLM client doesn't translate native tool_use (text-only models,
    /// where you parse intent from the body) or to filter/transform
    /// the calls first.
    ///
    /// A non-empty vec tells the tool-loop executor to dispatch and
    /// continue; empty stops the loop.
    ///
    /// See `crates/borg/examples/tool_loop.rs` for an end-to-end demo.
    fn extract_tool_calls(&self, verdict: &Verdict) -> Vec<crate::tools::ToolCallRequest> {
        verdict.tool_calls.clone()
    }

    /// Execute the agent: build prompts, call LLM, record budget, parse
    /// the verdict, honor cancellation. Override only for behavior the
    /// default flow can't express (deterministic work, side-channel
    /// results).
    ///
    /// When `user_prompt(ctx)` is `None` the call short-circuits to an
    /// empty verdict, so a deterministic agent that forgets to override
    /// `invoke()` can't fire a degenerate empty-prompt LLM call.
    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        let Some(user) = self.user_prompt(ctx) else {
            return Ok(Verdict::empty());
        };

        if !ctx.budget.can_invoke() {
            return Err(BorgError::BudgetExhausted {
                needed: self.min_tokens_estimate(),
                remaining: ctx.budget.remaining(),
            });
        }

        let request = ChatRequest {
            // model() is &self-bound so must own; system_prompt() is
            // &'static, borrowed for zero allocation on multi-KB literals.
            model: std::borrow::Cow::Owned(self.model().to_owned()),
            system: std::borrow::Cow::Borrowed(self.system_prompt()),
            user,
            // Applies any pending `Throttle` clamp; no-op without guardrails.
            max_tokens: ctx.effective_max_tokens(self.max_tokens()),
            tools: std::sync::Arc::clone(&ctx.tool_schemas),
            images: std::sync::Arc::from(self.images(ctx)),
        };

        // The select is the backstop guaranteeing we never await past
        // cancel even if the LLM impl ignores the token internally.
        let response = tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => return Err(BorgError::Cancelled),
            r = ctx.llm.chat_completion(request) => r?,
        };

        ctx.budget.record_usage(&response.usage());

        let verdict = Verdict::from_response(
            response.text,
            self.known_verdicts(),
            response.input_tokens,
            response.output_tokens,
        )
        .with_cache_tokens(response.cache_read_tokens, response.cache_creation_tokens)
        .with_tool_calls(response.tool_calls);
        Ok(verdict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::InMemoryBudget;
    use crate::events::NoopEmitter;
    use crate::handoff::HandoffState;
    use crate::llm::{AgentLlmClient, ChatResponse};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    struct FixedClient {
        text: String,
    }

    #[async_trait]
    impl AgentLlmClient for FixedClient {
        async fn chat_completion(&self, _req: ChatRequest) -> Result<ChatResponse, BorgError> {
            Ok(ChatResponse {
                text: self.text.clone(),
                input_tokens: 100,
                output_tokens: 50,
                cache_hit: false,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                tool_calls: Vec::new(),
            })
        }
    }

    struct StaticAgent {
        prompt: Option<String>,
    }

    #[async_trait]
    impl BorgAgent for StaticAgent {
        fn agent_id(&self) -> &'static str {
            "static"
        }
        fn model(&self) -> &str {
            "test/model"
        }
        fn max_tokens(&self) -> u32 {
            1000
        }
        fn system_prompt(&self) -> &'static str {
            "you are helpful"
        }
        fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
            self.prompt.clone()
        }
        fn known_verdicts(&self) -> &'static [&'static str] {
            &["PASS", "FAIL"]
        }
    }

    fn make_ctx(
        handoff: HandoffState,
        llm: Arc<dyn AgentLlmClient>,
        budget: Arc<dyn crate::Budget>,
    ) -> (AgentContext, CancellationToken) {
        let cancel = CancellationToken::new();
        let ctx = AgentContext {
            handoff: Arc::new(handoff),
            cancel: cancel.clone(),
            llm,
            budget,
            emitter: Arc::new(NoopEmitter),
            parent_composition_id: "test".into(),
            step_name: "step".into(),
            turn_number: 0,
            parent_call_id: None,
            tool_schemas: Arc::from([] as [crate::tools::ToolSchema; 0]),
            guardrails: None,
        };
        (ctx, cancel)
    }

    #[tokio::test]
    async fn default_invoke_returns_parsed_verdict() {
        let h = HandoffState::new();
        let llm: Arc<dyn AgentLlmClient> = Arc::new(FixedClient {
            text: "PASS\nlooks good".into(),
        });
        let budget: Arc<dyn crate::Budget> = Arc::new(InMemoryBudget::new(10_000));
        let (ctx, _c) = make_ctx(h, llm, budget.clone());

        let agent = StaticAgent {
            prompt: Some("hi".into()),
        };
        let v = agent.invoke(&ctx).await.unwrap();
        assert_eq!(v.token, "PASS");
        assert_eq!(&*v.body, "looks good");
        assert_eq!(budget.consumed(), 150);
    }

    #[tokio::test]
    async fn user_prompt_none_yields_empty_verdict_without_llm_call() {
        let h = HandoffState::new();
        let llm: Arc<dyn AgentLlmClient> = Arc::new(FixedClient {
            text: "should not be called".into(),
        });
        let budget: Arc<dyn crate::Budget> = Arc::new(InMemoryBudget::new(10_000));
        let (ctx, _c) = make_ctx(h, llm, budget.clone());

        let agent = StaticAgent { prompt: None };
        let v = agent.invoke(&ctx).await.unwrap();
        assert_eq!(v, Verdict::empty());
        assert_eq!(budget.consumed(), 0);
    }

    #[tokio::test]
    async fn budget_exhausted_short_circuits_before_llm() {
        let h = HandoffState::new();
        let llm: Arc<dyn AgentLlmClient> = Arc::new(FixedClient { text: "n/a".into() });
        let budget: Arc<dyn crate::Budget> = Arc::new(InMemoryBudget::new(0));
        let (ctx, _c) = make_ctx(h, llm, budget.clone());

        let agent = StaticAgent {
            prompt: Some("hi".into()),
        };
        match agent.invoke(&ctx).await {
            Err(BorgError::BudgetExhausted { remaining: 0, .. }) => {}
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
        assert_eq!(budget.consumed(), 0);
    }

    #[test]
    fn provider_default_is_openrouter() {
        let agent = StaticAgent { prompt: None };
        assert_eq!(agent.provider(), Provider::OpenRouter);
        assert_eq!(agent.provider().as_str(), "openrouter");
    }

    #[test]
    fn provider_can_be_overridden() {
        struct DirectAnthropicAgent;
        #[async_trait]
        impl BorgAgent for DirectAnthropicAgent {
            fn agent_id(&self) -> &'static str {
                "direct"
            }
            fn model(&self) -> &str {
                "claude-sonnet-5.5"
            }
            fn max_tokens(&self) -> u32 {
                1000
            }
            fn system_prompt(&self) -> &'static str {
                ""
            }
            fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
                None
            }
            fn provider(&self) -> Provider {
                Provider::Anthropic
            }
        }
        let agent = DirectAnthropicAgent;
        assert_eq!(agent.provider(), Provider::Anthropic);
        assert_eq!(agent.provider().as_str(), "anthropic");
    }

    #[test]
    fn provider_custom_carries_label() {
        let p = Provider::Custom("groq");
        assert_eq!(p.as_str(), "groq");
    }

    #[tokio::test]
    async fn cancelled_before_llm_returns_cancelled() {
        let h = HandoffState::new();
        let llm: Arc<dyn AgentLlmClient> = Arc::new(FixedClient { text: "n/a".into() });
        let budget: Arc<dyn crate::Budget> = Arc::new(InMemoryBudget::new(10_000));
        let (ctx, cancel) = make_ctx(h, llm, budget);
        cancel.cancel();

        let agent = StaticAgent {
            prompt: Some("hi".into()),
        };
        match agent.invoke(&ctx).await {
            Err(BorgError::Cancelled) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }
}
