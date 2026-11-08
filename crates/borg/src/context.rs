//! `AgentContext` — what every agent invocation sees.
//!
//! Threaded by reference into `BorgAgent::invoke`. Holds the immutable
//! handoff snapshot, cancellation token, shared LLM client and budget,
//! event emitter, parent attribution, and turn counters. Cheap to clone —
//! every field is an `Arc` or a scalar.

use crate::budget::Budget;
use crate::events::EventEmitter;
use crate::handoff::HandoffState;
use crate::llm::AgentLlmClient;
use crate::tools::ToolSchema;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Read-only context every agent receives during execution.
///
/// All fields are owned (`Arc` or scalar), so `AgentContext` is
/// `'static`-capable — streaming agents can capture it into a
/// `BoxStream<'static, StreamEvent>` without `Box::leak`. Constructed by
/// the executor; agents never build their own. `#[non_exhaustive]` so
/// fields can be added without a breaking change; unit tests that call an
/// agent directly get one from `borg::testing::agent_context` (`testing`
/// feature).
#[non_exhaustive]
pub struct AgentContext {
    /// Immutable handoff snapshot for this step. In `Sequential` it is
    /// the parent's state after step N-1's writes merged; in `Parallel`
    /// every branch sees the same fan-out-time snapshot. Read prior
    /// outputs via `ctx.handoff.get(key)` / `.find(step_name)`.
    pub handoff: Arc<HandoffState>,

    /// Cooperative cancellation token, a `child_token()` of the parent
    /// so a parent cancel fires every descendant. Long-running
    /// `invoke()` impls SHOULD `tokio::select!` on `cancel.cancelled()`.
    pub cancel: CancellationToken,

    /// Shared LLM client used by the default `invoke()` for
    /// `chat_completion`.
    pub llm: Arc<dyn AgentLlmClient>,

    /// Token budget for the running composition tree. Agents call
    /// `budget.can_invoke()` before the LLM and `budget.record(tokens)`
    /// after a successful response.
    pub budget: Arc<dyn Budget>,

    /// Event emitter for *intermediate* events (e.g. partial response).
    /// Lifecycle events are emitted by the composition layer; most
    /// agents leave this alone.
    pub emitter: Arc<dyn EventEmitter>,

    /// Parent composition id, used to attribute emitted events to the
    /// right node. `Arc<str>` so emit-site clones are pointer bumps.
    pub parent_composition_id: Arc<str>,

    /// Step name within the parent composition, as the builder declared
    /// it (e.g. `"code_review"`).
    pub step_name: Arc<str>,

    /// Outer-loop turn number, threaded through for telemetry. `0` when
    /// the consumer isn't running borg inside an outer loop.
    pub turn_number: u32,

    /// Free-form parent-call-id for cross-event attribution. `None` for
    /// pre-session callers and tests.
    pub parent_call_id: Option<Arc<str>>,

    /// Tool schemas attached to this invocation's `ChatRequest`. The
    /// tool-loop executor fills this with registered tools; non-loop
    /// dispatches leave it empty. `Arc<[ToolSchema]>` so fanning into
    /// `ChatRequest::tools` is `Arc::clone`s, not a Vec copy.
    pub tool_schemas: Arc<[ToolSchema]>,

    /// Guardrail harness from [`crate::ExecCtx::with_guardrails`],
    /// `None` without guardrails. Agents MAY drain pending
    /// [`crate::BoundaryMessage`]s in [`crate::BorgAgent::user_prompt`]
    /// and clamp `max_tokens` on `Throttle` — the default
    /// [`crate::BorgAgent::invoke`] does the latter automatically.
    pub guardrails: Option<Arc<crate::guardrails::GuardrailHarness>>,
}

impl AgentContext {
    /// True if cancellation has fired. Cheap poll for short-circuiting
    /// between long operations without a `tokio::select!`.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Drain pending [`crate::BoundaryMessage`]s from the guardrail
    /// queue; empty vec when no harness is wired or the queue is empty.
    /// Draining flushes — messages do not stick. Agents call this in
    /// `user_prompt(ctx)` to weave corrective text into the next turn.
    pub fn boundary_messages(&self) -> Vec<crate::guardrails::BoundaryMessage> {
        match &self.guardrails {
            Some(h) => h.queue.drain(),
            None => Vec::new(),
        }
    }

    /// Apply any pending `Throttle` clamp to `requested`. Consume-once:
    /// a second call returns `requested` unchanged unless a new
    /// `Throttle` fired since. The default [`crate::BorgAgent::invoke`]
    /// calls this; custom `invoke` impls must, for `Throttle` to apply.
    pub fn effective_max_tokens(&self, requested: u32) -> u32 {
        match &self.guardrails {
            Some(h) => h.effective_max_tokens(requested),
            None => requested,
        }
    }

    /// Pause this invocation to ask the human a question.
    ///
    /// Returns [`BorgError::Suspended`](crate::BorgError::Suspended); the
    /// `?` propagates the signal up through `invoke()` into the executor,
    /// where the actor layer (`borg-loop`) persists the request and
    /// re-enters with a [`Decision`](crate::suspend::Decision).
    ///
    /// ```ignore
    /// let decision = ctx.request_suspend(req).await?;
    /// ```
    pub async fn request_suspend(
        &self,
        req: crate::suspend::SuspendRequest,
    ) -> Result<crate::suspend::Decision, crate::error::BorgError> {
        Err(crate::error::BorgError::Suspended {
            signal: crate::suspend::SuspendSignal::from_request(req),
        })
    }
}
