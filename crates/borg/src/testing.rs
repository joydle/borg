//! Test fixtures for borg pipelines.
//!
//! Available via the `testing` feature flag for downstream consumers'
//! integration tests, plus always available within borg's own test
//! suite. Lets you write borg pipeline tests without re-implementing
//! `FakeLlmClient` and friends.
//!
//! ## Available fakes
//!
//! - [`FakeLlmClient`](crate::testing::FakeLlmClient) — scripted-response LLM
//!   with controllable latency, error injection, and infinite-block
//!   (for cancellation tests).
//! - [`ScriptedAgent`](crate::testing::ScriptedAgent) — minimal
//!   [`BorgAgent`] with a configurable id and
//!   prompt; uses the default `invoke()` flow.
//! - [`HandoffEchoAgent`](crate::testing::HandoffEchoAgent) — reads a prior
//!   step's verdict from handoff to verify state flowed correctly.
//! - [`PanickingAgent`](crate::testing::PanickingAgent) — deliberately panics
//!   to exercise actor isolation.
//! - [`CancellationObservingAgent`](crate::testing::CancellationObservingAgent)
//!   — observes whether it was cancelled mid-flight.
//! - [`agent_context`](crate::testing::agent_context) — an [`AgentContext`] with test defaults, for
//!   calling an agent's `user_prompt` / `invoke` without an executor.
//!
//! ## Eval harness (unstable)
//!
//! Scripted scenarios that replay against a composition and assert on
//! the observable event stream — regression locks, not quality
//! benchmarks. The easy shape is the [`eval`](crate::testing::eval) builder:
//!
//! ```ignore
//! use borg::testing::{eval, eval_events, EvalAssert, ScriptedAgent};
//!
//! #[tokio::test]
//! async fn smoke() {
//!     eval("smoke")
//!         .agent(ScriptedAgent::new("a"))
//!         .responses(["done"])
//!         .must_emit([eval_events::STEP_COMPLETED])
//!         .run().await
//!         .assert_ok();
//! }
//! ```
//!
//! For the full surface (multiple budgets, mixed ok/error outcomes, a
//! raw [`FakeOutcome`](crate::testing::FakeOutcome) script), build an [`EvalCase`](crate::testing::EvalCase) directly and pass
//! it to [`run_eval`](crate::testing::run_eval). Both produce the same [`EvalResult`](crate::testing::EvalResult).

pub use crate::evals::{
    eval, events as eval_events, run as run_eval, EvalAssert, EvalBuilder, EvalCase, EvalResult,
    ExpectedOutcome,
};

use crate::agent::BorgAgent;
use crate::context::AgentContext;
use crate::error::{BorgError, RetryClass};
use crate::llm::{AgentLlmClient, ChatRequest, ChatResponse};
use crate::tools::ToolCallRequest;
use crate::verdict::Verdict;
use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

/// Build an [`AgentContext`] for exercising an agent directly — calling
/// its `user_prompt` / `invoke` in a unit test without an executor.
///
/// `AgentContext` is `#[non_exhaustive]` (the executor builds it), so
/// this is the supported way to get one outside borg. Defaults: a fresh
/// cancellation token, a 1M-token in-memory budget, a no-op emitter,
/// `"test"` as both parent composition id and step name, turn 0, no
/// parent call id, no tool schemas, no guardrails. Every field is
/// public, so override what the test needs:
///
/// ```ignore
/// let mut ctx = borg::testing::agent_context(handoff, llm);
/// ctx.turn_number = 3;
/// let prompt = my_agent.user_prompt(&ctx);
/// ```
pub fn agent_context(
    handoff: crate::handoff::HandoffState,
    llm: Arc<dyn AgentLlmClient>,
) -> AgentContext {
    AgentContext {
        handoff: Arc::new(handoff),
        cancel: tokio_util::sync::CancellationToken::new(),
        llm,
        budget: crate::budget::in_memory_budget(1_000_000),
        emitter: Arc::new(crate::events::NoopEmitter),
        parent_composition_id: Arc::from("test"),
        step_name: Arc::from("test"),
        turn_number: 0,
        parent_call_id: None,
        tool_schemas: Arc::from([] as [crate::tools::ToolSchema; 0]),
        guardrails: None,
    }
}

/// Scripted outcome the FakeLlmClient should return on a single call.
#[derive(Clone, Debug)]
pub enum FakeOutcome {
    Ok {
        text: String,
        latency: Duration,
        /// Tool calls to surface as `ChatResponse::tool_calls`. When
        /// non-empty, drives the
        /// [`Composition::Agent`](crate::Composition::Agent) tool-loop
        /// into another iteration.
        tool_calls: Vec<ToolCallRequest>,
    },
    Err {
        message: String,
        class: RetryClass,
        latency: Duration,
    },
    /// Block forever — until cancelled. Cancelled responses surface as
    /// `BorgError::Cancelled` from the agent's `tokio::select!` arm.
    Forever,
}

impl FakeOutcome {
    pub fn ok(text: impl Into<String>) -> Self {
        FakeOutcome::Ok {
            text: text.into(),
            latency: Duration::ZERO,
            tool_calls: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn ok_after(text: impl Into<String>, latency: Duration) -> Self {
        FakeOutcome::Ok {
            text: text.into(),
            latency,
            tool_calls: Vec::new(),
        }
    }

    /// Ok response that also surfaces tool calls — drives the agent
    /// tool-loop into another iteration before producing the final
    /// answer.
    pub fn ok_with_tool_calls(text: impl Into<String>, tool_calls: Vec<ToolCallRequest>) -> Self {
        FakeOutcome::Ok {
            text: text.into(),
            latency: Duration::ZERO,
            tool_calls,
        }
    }

    pub fn retryable_err(message: impl Into<String>) -> Self {
        FakeOutcome::Err {
            message: message.into(),
            class: RetryClass::Retryable,
            latency: Duration::ZERO,
        }
    }

    /// Retryable error that also pauses for `latency` — used to test
    /// `RestartPolicy::on_failure_within` wall-clock cutoffs.
    #[allow(dead_code)]
    pub fn retryable_err_after(message: impl Into<String>, latency: Duration) -> Self {
        FakeOutcome::Err {
            message: message.into(),
            class: RetryClass::Retryable,
            latency,
        }
    }

    pub fn terminal_err(message: impl Into<String>) -> Self {
        FakeOutcome::Err {
            message: message.into(),
            class: RetryClass::Terminal,
            latency: Duration::ZERO,
        }
    }
}

/// Controllable LLM client. Each call consumes one outcome from a
/// scripted FIFO; if the script runs out, calls return `Internal`
/// errors so the test fails clearly instead of hanging.
pub struct FakeLlmClient {
    script: Mutex<std::collections::VecDeque<FakeOutcome>>,
    pub call_count: AtomicUsize,
    /// Reserved: count of calls that observed cancellation at the LLM
    /// layer. Not populated — the agent-layer select fires first.
    #[allow(dead_code)]
    pub cancelled_count: AtomicUsize,
}

impl FakeLlmClient {
    pub fn new(script: Vec<FakeOutcome>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().collect()),
            call_count: AtomicUsize::new(0),
            cancelled_count: AtomicUsize::new(0),
        })
    }

    /// Convenience: a client that returns the same OK response for up
    /// to 1024 calls. Calls beyond that surface as terminal errors.
    pub fn always_ok(text: impl Into<String>) -> Arc<Self> {
        let text = text.into();
        Arc::new(Self {
            script: Mutex::new(std::iter::repeat_n(FakeOutcome::ok(text), 1024).collect()),
            call_count: AtomicUsize::new(0),
            cancelled_count: AtomicUsize::new(0),
        })
    }

    pub fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl AgentLlmClient for FakeLlmClient {
    async fn chat_completion(&self, _req: ChatRequest) -> Result<ChatResponse, BorgError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let outcome = {
            let mut script = self.script.lock().await;
            script.pop_front().unwrap_or_else(|| FakeOutcome::Err {
                message: "fake script exhausted".into(),
                class: RetryClass::Terminal,
                latency: Duration::ZERO,
            })
        };
        match outcome {
            FakeOutcome::Ok {
                text,
                latency,
                tool_calls,
            } => {
                if !latency.is_zero() {
                    tokio::time::sleep(latency).await;
                }
                Ok(ChatResponse {
                    text,
                    input_tokens: 100,
                    output_tokens: 50,
                    cache_hit: false,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    tool_calls,
                })
            }
            FakeOutcome::Err {
                message,
                class,
                latency,
            } => {
                if !latency.is_zero() {
                    tokio::time::sleep(latency).await;
                }
                Err(BorgError::llm(message, class))
            }
            FakeOutcome::Forever => {
                std::future::pending::<()>().await;
                unreachable!()
            }
        }
    }
}

/// Minimal `BorgAgent` for tests. Configurable agent_id, model, and
/// user_prompt resolution. Uses the default `invoke()` flow.
pub struct ScriptedAgent {
    pub id: &'static str,
    pub model_id: String,
    pub max_tokens: u32,
    pub system: &'static str,
    pub prompt: Option<String>,
    pub known_verdicts: &'static [&'static str],
    pub min_estimate: u64,
}

impl ScriptedAgent {
    pub fn new(id: &'static str) -> Self {
        Self {
            id,
            model_id: "test/model".into(),
            max_tokens: 1000,
            system: "test system",
            prompt: Some("test prompt".into()),
            known_verdicts: &[],
            min_estimate: 1_000,
        }
    }

    #[must_use]
    pub fn with_verdicts(mut self, v: &'static [&'static str]) -> Self {
        self.known_verdicts = v;
        self
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn with_prompt(mut self, p: Option<&str>) -> Self {
        self.prompt = p.map(|s| s.to_string());
        self
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn with_min_estimate(mut self, n: u64) -> Self {
        self.min_estimate = n;
        self
    }
}

#[async_trait]
impl BorgAgent for ScriptedAgent {
    fn agent_id(&self) -> &'static str {
        self.id
    }
    fn model(&self) -> &str {
        &self.model_id
    }
    fn max_tokens(&self) -> u32 {
        self.max_tokens
    }
    fn system_prompt(&self) -> &'static str {
        self.system
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        self.prompt.clone()
    }
    fn known_verdicts(&self) -> &'static [&'static str] {
        self.known_verdicts
    }
    fn min_tokens_estimate(&self) -> u64 {
        self.min_estimate
    }
}

/// Reads previous step's verdict token via `ctx.handoff.find(...)` and
/// embeds it in its prompt. Lets tests prove handoff flowed without
/// inspecting handoff state directly.
pub struct HandoffEchoAgent {
    pub id: &'static str,
    pub source_step: &'static str,
}

#[async_trait]
impl BorgAgent for HandoffEchoAgent {
    fn agent_id(&self) -> &'static str {
        self.id
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        1000
    }
    fn system_prompt(&self) -> &'static str {
        "echo previous verdict"
    }
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        let prev = ctx
            .handoff
            .find(self.source_step)
            .map(|nv| nv.verdict.token.clone())
            .unwrap_or_else(|| "MISSING".into());
        Some(format!("previous: {prev}"))
    }
}

/// Declares a `handoff_to` target on its verdict. Used to test that a
/// downstream `Composition::Branch` can route on
/// `HandoffState::pending_handoff()`.
pub struct HandoffAgent {
    pub id: &'static str,
    pub target: &'static str,
}

#[async_trait]
impl BorgAgent for HandoffAgent {
    fn agent_id(&self) -> &'static str {
        self.id
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        1000
    }
    fn system_prompt(&self) -> &'static str {
        "handoff"
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("handoff".into())
    }

    async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
        Ok(Verdict {
            body: format!("routing to {}", self.target).into(),
            token_is_recognized: true,
            input_tokens: 10,
            output_tokens: 5,
            handoff_to: Some(self.target.to_string()),
            ..Verdict::default()
        })
    }
}

/// Always panics on invoke. Exercises supervisor panic isolation.
pub struct PanickingAgent {
    pub id: &'static str,
}

#[async_trait]
impl BorgAgent for PanickingAgent {
    fn agent_id(&self) -> &'static str {
        self.id
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        1000
    }
    fn system_prompt(&self) -> &'static str {
        "panic"
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("p".into())
    }

    async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
        panic!("intentional test panic from {}", self.id);
    }
}

/// Observes whether its invocation was cancelled mid-flight.
/// Sleeps for `block_for` unless the cancel token fires first.
pub struct CancellationObservingAgent {
    pub id: &'static str,
    pub observed_cancel: Arc<std::sync::atomic::AtomicBool>,
    pub block_for: Duration,
}

#[async_trait]
impl BorgAgent for CancellationObservingAgent {
    fn agent_id(&self) -> &'static str {
        self.id
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        1000
    }
    fn system_prompt(&self) -> &'static str {
        "observe"
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("p".into())
    }

    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        let observed = self.observed_cancel.clone();
        let block = self.block_for;
        tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => {
                observed.store(true, std::sync::atomic::Ordering::SeqCst);
                Err(BorgError::Cancelled)
            }
            _ = tokio::time::sleep(block) => {
                Ok(Verdict::with_token("DONE", "completed", 50, 25))
            }
        }
    }
}
