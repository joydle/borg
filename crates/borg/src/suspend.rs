//! Human-in-the-loop suspend primitive.
//!
//! Agents pause mid-invoke via [`AgentContext::request_suspend`]; the
//! executor unwinds with [`BorgError::Suspended`] carrying a
//! [`SuspendSignal`]. An outer actor layer surfaces the
//! [`SuspendRequest`] to the user, persists session state, and re-enters
//! the executor with a [`Decision`] when the human responds.
//!
//! This module provides the types and unwind path; resume mechanics
//! live in the actor layer.
//!
//! [`AgentContext::request_suspend`]: crate::AgentContext::request_suspend
//! [`BorgError::Suspended`]: crate::BorgError::Suspended

use serde::{Deserialize, Serialize};

/// What an agent is pausing to ask the human.
///
/// `prompt` is the human-facing question, rendered verbatim. `kind`
/// picks the UI affordance. `blocked_tool`, if set, names the tool the
/// agent was about to call. `payload` is consumer-defined structured
/// context for the decision UX (proposed diff, planner output, …).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SuspendRequest {
    pub prompt: String,
    pub kind: SuspendKind,
    pub blocked_tool: Option<String>,
    pub payload: serde_json::Value,
}

impl SuspendRequest {
    /// Build a request with the two required fields; optional fields
    /// default to `None` / `Null` and use the builder setters below.
    pub fn new(prompt: impl Into<String>, kind: SuspendKind) -> Self {
        Self {
            prompt: prompt.into(),
            kind,
            blocked_tool: None,
            payload: serde_json::Value::Null,
        }
    }

    /// Name the tool the agent was about to call when it paused.
    #[must_use]
    pub fn with_blocked_tool(mut self, tool: impl Into<String>) -> Self {
        self.blocked_tool = Some(tool.into());
        self
    }

    /// Attach consumer-defined structured context for the decision UX.
    #[must_use]
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }
}

/// Shape of the decision the consumer is being asked to make. Drives
/// UI affordances. `#[non_exhaustive]`: match with a `_` arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SuspendKind {
    /// Binary approve/reject (with optional Edit). The common case.
    Approval,
    /// Multi-choice picker. `options` carries human-readable labels;
    /// the resulting `Decision::Edit { patch }` should carry the
    /// chosen index or label.
    Choice { options: Vec<String> },
}

/// The human's response to a [`SuspendRequest`], delivered through the
/// resume path.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Decision {
    /// Proceed. The agent's `request_suspend.await` returns this and
    /// `invoke` continues with the suspended work.
    Approve,
    /// Stop, with a reason. The agent propagates this through whatever
    /// error path fits its workflow.
    Reject { reason: String },
    /// Proceed with edits. `patch` is consumer-defined and merged into
    /// handoff state by the agent.
    Edit { patch: serde_json::Value },
}

/// Control-flow signal carried by [`BorgError::Suspended`]. Currently
/// holds only the request; `#[non_exhaustive]` reserves room for
/// executor-level snapshot state.
///
/// [`BorgError::Suspended`]: crate::BorgError::Suspended
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SuspendSignal {
    pub request: SuspendRequest,
}

impl SuspendSignal {
    /// Build a signal carrying only the suspend request.
    pub fn from_request(request: SuspendRequest) -> Self {
        Self { request }
    }
}
