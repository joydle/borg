//! Human-in-the-loop approval helper.
//!
//! Thin convenience layer over [`borg::AgentContext::request_suspend`] that
//! constructs a binary-approval [`borg::suspend::SuspendRequest`] and forwards
//! the call. Use it inside [`borg::BorgAgent::invoke`] when the agent is about
//! to take an irreversible action and needs a human to sign off first.
//!
//! ```ignore
//! use async_trait::async_trait;
//! use borg::prelude::*;
//! use borg::suspend::Decision;
//! use borg_tools::approval::request_approval;
//!
//! struct Migrator;
//!
//! #[async_trait]
//! impl BorgAgent for Migrator {
//!     fn agent_id(&self) -> &'static str { "migrator" }
//!     fn model(&self) -> &str { "claude-sonnet-5-5" }
//!     fn max_tokens(&self) -> u32 { 1024 }
//!     fn system_prompt(&self) -> &'static str { "" }
//!     fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> { None }
//!
//!     async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
//!         let decision = request_approval(
//!             ctx,
//!             "I'm about to run the database migration. Proceed?",
//!             Some("run_migration"),
//!         ).await?;
//!         match decision {
//!             Decision::Approve => { /* run migration */ Ok(Default::default()) }
//!             Decision::Reject { reason } => Err(BorgError::Internal { message: reason }),
//!             Decision::Edit { .. } => { /* incorporate edits, retry */ Ok(Default::default()) }
//!             _ => unreachable!("Decision is #[non_exhaustive]"),
//!         }
//!     }
//! }
//! ```
//!
//! This is not a `borg::Tool` impl: the agent's Rust code invokes it directly,
//! the LLM never sees it as a callable function.

use borg::suspend::{Decision, SuspendKind, SuspendRequest};
use borg::{AgentContext, BorgError};

/// Pause the agent for binary human approval.
///
/// The first call from any invocation always unwinds via
/// [`BorgError::Suspended`]; the `Ok(Decision)` arm fires only on the re-entry
/// that follows a resume message.
///
/// `prompt` is the human-facing question rendered verbatim in the UI.
/// `blocked_tool`, if set, names the tool the agent was about to call.
///
/// The payload is left empty; callers that need structured context (proposed
/// diff, planner output, …) should build the [`SuspendRequest`] and call
/// [`AgentContext::request_suspend`] directly.
pub async fn request_approval(
    ctx: &AgentContext,
    prompt: impl Into<String>,
    blocked_tool: Option<&str>,
) -> Result<Decision, BorgError> {
    let mut req = SuspendRequest::new(prompt, SuspendKind::Approval);
    if let Some(tool) = blocked_tool {
        req = req.with_blocked_tool(tool);
    }
    ctx.request_suspend(req).await
}
