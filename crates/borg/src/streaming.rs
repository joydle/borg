//! Streaming agent primitives.
//!
//! `BorgAgent` is single-shot (invoke → `Verdict`). A conversation loop
//! is shaped differently — a long tail of `TextDelta` chunks intermixed
//! with tool calls and intermediate state.
//!
//! `BorgStreamingAgent` covers that shape: `invoke_streaming` returns a
//! `BoxStream<StreamEvent>` that the executor forwards to the consumer's
//! emitter and surfaces as the top-level stream. The final event is
//! always `StreamEvent::Done(Verdict)` carrying cumulative token counts.
//!
//! ## Composition
//!
//! Streaming agents compose with single-shot agents via the
//! [`crate::StreamingComposition`] tree (separate from
//! [`crate::Composition`] — see [Streaming vs one-shot](#streaming-vs-one-shot) below). A pipeline like
//!
//! ```ignore
//! StreamingComposition::single("main_loop", main_streaming_agent)
//!     .then(Composition::single("post_review", post_step), "main_output")
//! ```
//!
//! runs the streaming agent first (events live), then the single-shot
//! review step once the stream completes with `Done`. Borg supervises the
//! whole tree under one cancel token and one budget.
//!
//! ## What this primitive doesn't do
//!
//! - **Reactive backpressure**: slow consumers suspend the producer at
//!   `yield` boundaries; no channel capacity is enforced. Wrap with
//!   `tokio_stream` adapters for bounded buffering.
//! - **Mid-stream cancellation guarantees**: the agent MUST honor
//!   `ctx.cancel`. The executor races each wait for the next event
//!   against the token and drops the agent's stream on cancel, but work
//!   the agent spawned outside that stream runs on, and a synchronous
//!   loop inside one `poll` can't be interrupted.
//!
//! ## Streaming vs one-shot
//!
//! Borg has two parallel composition trees, kept intentionally separate:
//!
//! - [`crate::Composition`] drives [`crate::BorgAgent`] (one-shot, returns
//!   `Verdict`): Sequential / Parallel / Branch / LoopUntil / Fallback /
//!   Agent.
//! - [`StreamingComposition`](crate::StreamingComposition) drives
//!   [`BorgStreamingAgent`] (returns a `BoxStream<StreamEvent>`):
//!   Single / Then / Pre.
//!
//! The return types differ fundamentally — a value vs a tail of events —
//! so each tree carries only the primitives that fit its shape (retry,
//! parallel fan-out, tool-loop `max_steps`, `Branch`, `Fallback` are
//! one-shot only; the streaming tree is forward-only).
//!
//! ### Which one to use
//!
//! Use [`StreamingComposition`](crate::StreamingComposition) when the
//! output is user-visible and incremental (text rendered as it arrives
//! over a streaming wire — SSE, WebSocket, a CLI printing bytes). Use [`crate::Composition`] when the
//! output is consumed programmatically (parsed as JSON, routed by
//! `Branch`, pulled from `HandoffState`) or is intermediate input to a
//! later step, or when you need a one-shot-only primitive.
//!
//! ### Bridge: streaming-then-one-shot
//!
//! [`StreamingComposition::then`](crate::StreamingComposition::then)
//! forwards a streaming agent's events to the consumer, then on `Done(verdict)` runs a one-shot `Composition`
//! with the verdict pre-seeded into handoff. The post-step's events route
//! to `ctx.emitter`, NOT the top-level stream. The post-step runs BEFORE
//! the stream yields its final event: the stream ends with the streaming
//! agent's `Done` if the post-step succeeded, or `Failed(post-step
//! error)` if it didn't.
//!
//! ### Supervision
//!
//! See [`execute_streaming`](crate::execute_streaming) for the exact
//! guarantees: admission (cancel and budget) before dispatch, a panic boundary that turns a panicking
//! agent into `Failed(ChildPanicked)`, cancel raced against every wait on
//! the agent's stream, and lifecycle events emitted before the terminal
//! event. Budget is checked at admission only, not mid-stream.
//!
//! ### Not in scope
//!
//! - Streaming Sequential/Parallel of multiple streaming agents — add a
//!   `StreamingComposition::Parallel` variant + event multiplexer when a
//!   real consumer needs it.
//! - Mid-stream tool dispatch by borg: the streaming agent owns its tool
//!   loop; borg supervises it (see above) and forwards its events.
//! - Retry of a streaming agent: re-delivering already-rendered events is
//!   wrong. The agent surfaces transport failures via
//!   `StreamEvent::Failed`; the caller decides whether to retry the whole
//!   composition.

use crate::context::AgentContext;
use crate::error::BorgError;
use crate::verdict::Verdict;
use async_trait::async_trait;
use futures::stream::BoxStream;

/// Events surfaced by a streaming agent. The executor forwards these to
/// the consumer's emitter and as the top-level stream `execute_streaming`
/// returns.
///
/// Not `Clone` (`BorgError` isn't trivially clonable). Events are
/// single-consumer; wrap with `BroadcastStream` if you need duplication.
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamEvent {
    /// A text chunk. Consumers concatenate these into the full response.
    TextDelta(String),
    /// The agent began a tool call. Borg doesn't dispatch tools (that's
    /// the agent's body), so this is purely observability.
    ToolCallStarted {
        call_id: String,
        name: String,
        args: serde_json::Value,
    },
    /// A tool call completed. `result` is the agent's dispatcher output as JSON.
    ToolCallCompleted {
        call_id: String,
        name: String,
        result: serde_json::Value,
        success: bool,
    },
    /// Extended-thinking chunk. Consumers may render it separately or hide it.
    Thinking(String),
    /// Free-form structured event. Use sparingly; prefer dedicated
    /// variants when a pattern recurs.
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
    /// Terminal success. The stream MUST end with this (anything after is
    /// ignored); carries the final `Verdict` for token accounting and handoff.
    Done(Verdict),
    /// Terminal failure. Agents emit it to signal a controlled mid-stream
    /// failure; the executor also yields it for refused admission
    /// (cancel, budget), a caught panic (`ChildPanicked`), cancel, and a
    /// failed one-shot pre/post step.
    Failed(BorgError),
}

impl StreamEvent {
    /// True for terminal events (`Done` / `Failed`).
    pub fn is_terminal(&self) -> bool {
        matches!(self, StreamEvent::Done(_) | StreamEvent::Failed(_))
    }
}

/// A streaming agent producing `StreamEvent`s terminated by `Done` or
/// `Failed`. Object-safe so the executor can hold `Box<dyn ...>`.
///
/// ## Cancellation contract
///
/// The agent MUST honor `ctx.cancel`; the executor's `tokio::select!`
/// against the token only catches inter-event waits, so an agent looping
/// without polling cancel runs to completion. Idiomatic shape:
///
/// ```ignore
/// async fn invoke_streaming(...) -> Result<BoxStream<StreamEvent>, BorgError> {
///     let cancel = ctx.cancel.clone();
///     Ok(Box::pin(async_stream::stream! {
///         loop {
///             tokio::select! {
///                 _ = cancel.cancelled() => {
///                     yield StreamEvent::Failed(BorgError::Cancelled);
///                     return;
///                 }
///                 event = next_chunk() => yield event,
///             }
///         }
///     }))
/// }
/// ```
#[async_trait]
pub trait BorgStreamingAgent: Send + Sync {
    /// Stable identifier for this agent class — see
    /// [`crate::BorgAgent::agent_id`].
    fn agent_id(&self) -> &'static str;

    /// Provider-specific model id; surfaces in lifecycle events.
    fn model(&self) -> &str;

    /// Drive the agent. Consumes `ctx`; the returned `'static` stream
    /// borrows nothing from the caller.
    ///
    /// `Err(BorgError)` is for setup-time failures (bad config, transport
    /// unavailable). Emit mid-stream failures as `StreamEvent::Failed` so
    /// the consumer keeps the already-streamed events.
    async fn invoke_streaming(
        &self,
        ctx: AgentContext,
    ) -> Result<BoxStream<'static, StreamEvent>, BorgError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn done_is_terminal() {
        let e = StreamEvent::Done(Verdict::empty());
        assert!(e.is_terminal());
    }

    #[test]
    fn failed_is_terminal() {
        let e = StreamEvent::Failed(BorgError::Cancelled);
        assert!(e.is_terminal());
    }

    #[test]
    fn text_delta_is_not_terminal() {
        let e = StreamEvent::TextDelta("hi".into());
        assert!(!e.is_terminal());
    }

    #[test]
    fn thinking_is_not_terminal() {
        let e = StreamEvent::Thinking("...".into());
        assert!(!e.is_terminal());
    }

    #[test]
    fn tool_events_are_not_terminal() {
        let started = StreamEvent::ToolCallStarted {
            call_id: "c".into(),
            name: "n".into(),
            args: serde_json::Value::Null,
        };
        let completed = StreamEvent::ToolCallCompleted {
            call_id: "c".into(),
            name: "n".into(),
            result: serde_json::Value::Null,
            success: true,
        };
        assert!(!started.is_terminal());
        assert!(!completed.is_terminal());
    }
}
