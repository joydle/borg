//! `StreamingActorAgent` — actor variant whose turns produce live `StreamEvent`s.
//!
//! See [`StreamingActorAgent`] for the trait contract and
//! [`crate::streaming_runtime`] for the runtime that drives it.

use crate::ActorCtx;
use async_trait::async_trait;
use borg::{BorgError, StreamEvent};
use futures::stream::BoxStream;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// What the actor reports after one streaming turn completes.
///
/// **`Continue` and `Park` are runtime-identical:** both keep the actor
/// alive at the next `inbox.recv()` and neither auto-loops. The meaning
/// lives in [`TurnHandle::outcome`] consumers (the session driver):
///
/// - `Continue` means "work is unfinished; send another turn now" — a
///   tool-loop continuation needing another LLM call.
/// - `Park` means "work is complete; wait for an external message" — a
///   finished user turn in a chat session.
/// - `Terminal(reason)` exits the loop, checkpoints where appropriate,
///   and deletes the snapshot via `delete_on_terminal`.
///
/// The runtime doesn't enforce the `Continue`/`Park` distinction: an
/// actor that returns `Continue` to a driver that ignores outcomes parks
/// forever. Drive turns through
/// [`StreamingActorHandle::run_session_loop`], which honors it natively.
///
/// [`StreamingActorHandle::run_session_loop`]: crate::StreamingActorHandle::run_session_loop
///
/// [`Continue`]: StreamingHandleOutcome::Continue
/// [`Park`]: StreamingHandleOutcome::Park
/// [`Terminal`]: StreamingHandleOutcome::Terminal
/// [`TurnHandle::outcome`]: TurnHandle::outcome
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StreamingHandleOutcome {
    /// "Work unfinished; send another turn now." Keeps the actor alive
    /// at the next `inbox.recv()`; the driver sends the follow-up.
    Continue,
    /// "Work complete; wait for an external message." Runtime-identical
    /// to `Continue`.
    Park,
    /// Actor is done; the runtime exits the loop. The snapshot is
    /// deleted via `delete_on_terminal`, so no final save is written
    /// (it would be invalidated by the delete).
    Terminal(String),
    /// Human-in-the-loop pause. The actor can't continue without a human
    /// decision. The runtime persists a final snapshot and exits with
    /// [`crate::ActorExit::Suspended`] carrying the request; callers
    /// surface it, then resume with a decision-bearing message.
    ///
    /// Distinct from [`StreamingHandleOutcome::Park`]: Park keeps the actor
    /// alive in the inbox, Suspended exits the loop and writes a checkpoint.
    Suspended {
        request: borg::suspend::SuspendRequest,
    },
}

/// Caller-side view of one completed turn: live events plus the outcome
/// that tells the session driver what to do next.
///
/// Returned by [`StreamingActorHandle::send`]. Call [`next_event`] in a loop
/// until it returns `None`, then call [`outcome`] to learn whether to send
/// `Continue`, wait for a user message (`Park`), or shut down (`Terminal`).
///
/// ## Drop semantics
///
/// Dropping a `TurnHandle` mid-stream tells the runtime the caller is
/// gone: on its next `event_tx.send` (now `Err`) it drops the stream
/// (cancelling in-flight futures) and returns to `inbox.recv()`. The
/// actor stays alive with its checkpoint state intact.
///
/// An agent-declared `Continue` on a caller-gone turn is softened to
/// `Park` — there is no caller to send the next turn to. `Terminal`
/// still terminates.
///
/// ## Two channels — and you MUST dedup
///
/// A single cancel, panic or `handle_streaming` `Err` produces two
/// notifications: a granular [`borg::StreamEvent::Failed`] on the
/// events channel (for in-stream UI rendering inline with partial
/// text), and a coarse [`StreamingHandleOutcome::Terminal`] on the
/// outcome channel (for the parent task's lifecycle dispatch). Both are
/// deliberate, so **callers rendering UI from either MUST dedup** — track
/// whether a `Failed` event was seen and skip the matching `Terminal`:
///
/// ```ignore
/// let mut failure_already_emitted = false;
/// while let Some(ev) = turn.next_event().await {
///     if let StreamEvent::Failed(err) = &ev {
///         emit_failure_ui(err);
///         failure_already_emitted = true;
///     }
///     forward_event(ev);
/// }
/// match turn.outcome().await {
///     StreamingHandleOutcome::Terminal(reason) if !failure_already_emitted => {
///         emit_failure_ui_from_reason(&reason);
///     }
///     StreamingHandleOutcome::Terminal(_) => {
///         // Already emitted from the Failed event above — skip.
///     }
///     StreamingHandleOutcome::Continue | StreamingHandleOutcome::Park => {
///         // Normal exit.
///     }
/// }
/// ```
///
/// Draining only `outcome()` (for "did the turn end normally") or only
/// [`next_event`] (for in-flight observability) is fine; the trap is
/// emitting UI from both for the same event. The runtime can't dedup
/// because it doesn't know which channel the caller renders from.
///
/// [`next_event`]: TurnHandle::next_event
/// [`outcome`]: TurnHandle::outcome
pub struct TurnHandle {
    /// Live event stream for this turn. Drop or drain to completion.
    pub(crate) events: mpsc::Receiver<StreamEvent>,
    /// The turn outcome. The runtime sends it after the last event and
    /// before `events` closes, so awaiting it post-drain never blocks.
    pub(crate) outcome: oneshot::Receiver<StreamingHandleOutcome>,
}

impl TurnHandle {
    /// Receive the next event from this turn's stream. Returns `None` when
    /// the stream is exhausted (all events delivered).
    pub async fn next_event(&mut self) -> Option<StreamEvent> {
        self.events.recv().await
    }

    /// Resolve the turn outcome. Call after `next_event` returns `None`.
    /// The runtime sends the outcome after the last event, so this is
    /// immediately ready once the stream is drained.
    pub async fn outcome(self) -> StreamingHandleOutcome {
        self.outcome
            .await
            .unwrap_or(StreamingHandleOutcome::Terminal(
                "runtime dropped outcome sender".into(),
            ))
    }
}

/// An actor whose turns are streaming BORG calls.
///
/// Each [`handle_streaming`] invocation produces a `BoxStream<StreamEvent>`.
/// The runtime drains it (forwarding events live), then checkpoints and
/// loops. Checkpoint/restore is identical to [`crate::ActorAgent`]; this
/// is a parallel trait only because `ActorAgent::handle` has no channel
/// to thread `StreamEvent`s through.
///
/// ## Cancellation
///
/// While draining a turn, the runtime races cancel against both waits:
/// the stream's next event, and the send into the turn's event channel
/// (so a caller that holds a [`TurnHandle`] without draining it can't
/// pin the actor past a cancel). On cancel it drops the stream, sends
/// `Failed(Cancelled)` to the caller if the channel has room, and exits
/// with [`crate::ActorExit::Cancelled`] after a final checkpoint. It can't
/// interrupt the body of [`handle_streaming`] (see its contract) or a
/// single synchronous `poll` of the stream.
///
/// ## Panics
///
/// Each turn runs behind a panic boundary, as in the plain actor
/// runtime. A panic in [`handle_streaming`] or while polling its stream
/// sends `Failed(BorgError::ChildPanicked)` to the caller if the channel
/// has room, resolves the turn's outcome to `Terminal("panicked: …")`,
/// and exits with [`crate::ActorExit::Panicked`]. The snapshot is not
/// deleted:
///
/// - A panic in `handle_streaming` (which holds `&mut self`) writes no
///   checkpoint, since state may be half-mutated. The store keeps the
///   last snapshot saved before that turn.
/// - A panic while polling the stream (which can't touch the agent)
///   persists the agent's current state, the one `handle_streaming` left
///   behind, as the final checkpoint.
///
/// [`handle_streaming`]: Self::handle_streaming
#[async_trait]
pub trait StreamingActorAgent: Send + 'static {
    /// Inbound message type — same shape as `ActorAgent::Message`.
    type Message: Send + 'static;

    /// Serializable state snapshot — same contract as `ActorAgent::Snapshot`.
    type Snapshot: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Drive one turn. Return its event stream plus whether to continue,
    /// park, or terminate after it drains.
    ///
    /// Return `Err(BorgError)` for setup failures (bad config, transport
    /// down). Surface mid-stream failures as `StreamEvent::Failed` inside
    /// the returned stream so callers still see already-streamed events.
    ///
    /// # Cancellation contract
    ///
    /// 1. **Returned stream:** the runtime races the stream's `next()`
    ///    and each event-channel send against [`ActorCtx::cancel`], so
    ///    cancellation between events is automatic.
    /// 2. **This method body:** NOT raced against `ctx.cancel` — it runs
    ///    to completion before the stream begins. Implementations MUST
    ///    `tokio::select!` against `ctx.cancel.cancelled()` for every
    ///    long-running `.await` (LLM calls, subprocess waits, network
    ///    I/O), or cancellation latency equals that `.await`'s worst
    ///    case.
    ///
    /// ```ignore
    /// async fn handle_streaming(&mut self, msg: Msg, ctx: &ActorCtx)
    ///     -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError>
    /// {
    ///     // Race long setup work against cancel.
    ///     let stream = tokio::select! {
    ///         biased;
    ///         _ = ctx.cancel.cancelled() => return Err(BorgError::Cancelled),
    ///         r = self.llm.chat_completion_streaming(req) => r?,
    ///     };
    ///     Ok((stream, StreamingHandleOutcome::Continue))
    /// }
    /// ```
    async fn handle_streaming(
        &mut self,
        msg: Self::Message,
        ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError>;

    /// Serialize current state for storage — same contract as `ActorAgent::checkpoint`.
    fn checkpoint(&self) -> Self::Snapshot;

    /// Reconstruct state from a snapshot — same contract as `ActorAgent::restore`.
    fn restore(snap: Self::Snapshot) -> Self
    where
        Self: Sized;
}

/// Caller-side handle to a streaming actor.
///
/// `send` delivers a message and returns a [`TurnHandle`] for consuming
/// the turn's events and outcome; the channels are per-message so each
/// turn is independently consumable.
///
/// Dropping the handle fires its cancel token; the actor exits with
/// [`ActorExit::Cancelled`] after a final checkpoint. To let the actor
/// outlive the handle, call [`detach`].
///
/// When multiple owners share the handle via `Arc` (e.g.
/// [`crate::ActorRegistry`]), cancel fires only when the last `Arc`
/// drops — i.e. when no potential senders remain.
///
/// [`ActorExit::Cancelled`]: crate::handle::ActorExit::Cancelled
/// [`detach`]: Self::detach
#[must_use = "StreamingActorHandle cancels the actor on drop; bind it or call .detach()"]
pub struct StreamingActorHandle<A: StreamingActorAgent> {
    pub(crate) addr: Arc<str>,
    pub(crate) inbox: mpsc::Sender<StreamingEnvelope<A::Message>>,
    pub(crate) cancel: CancellationToken,
    pub(crate) join: Option<tokio::task::JoinHandle<crate::handle::ActorExit>>,
    pub(crate) event_channel_capacity: usize,
    pub(crate) detached: bool,
}

/// Internal: one message + its reply channels.
pub(crate) struct StreamingEnvelope<M> {
    pub msg: M,
    /// Runtime forwards streaming events here.
    pub event_tx: mpsc::Sender<StreamEvent>,
    /// Runtime sends the turn outcome here after the stream ends.
    pub outcome_tx: oneshot::Sender<StreamingHandleOutcome>,
}

impl<A: StreamingActorAgent> StreamingActorHandle<A> {
    /// The actor's stable address (store key + log identity).
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Whether the actor's inbox is closed (actor has exited or is exiting).
    pub fn is_closed(&self) -> bool {
        self.inbox.is_closed()
    }

    /// Send a message and get back a [`TurnHandle`] for this turn.
    ///
    /// Drain `turn.events` live, then `await turn.outcome` to learn
    /// whether to `Continue`, `Park`, or shut down.
    ///
    /// Returns `Err(SendError(msg))` carrying the original message when
    /// the inbox is closed (actor exited), so callers can recover it and
    /// retry against a respawned actor. Pre-probe with [`is_closed`] to
    /// drain stale handles before building heavy messages.
    ///
    /// Event channel capacity comes from
    /// [`ActorConfig::event_channel_capacity`] — drain promptly to avoid
    /// backpressuring the runtime.
    ///
    /// [`ActorConfig::event_channel_capacity`]: crate::ActorConfig::event_channel_capacity
    /// [`is_closed`]: Self::is_closed
    pub async fn send(
        &self,
        msg: A::Message,
    ) -> Result<TurnHandle, mpsc::error::SendError<A::Message>> {
        let (event_tx, events) = mpsc::channel::<StreamEvent>(self.event_channel_capacity);
        let (outcome_tx, outcome) = oneshot::channel::<StreamingHandleOutcome>();
        let envelope = StreamingEnvelope {
            msg,
            event_tx,
            outcome_tx,
        };
        self.inbox
            .send(envelope)
            .await
            .map_err(|e| mpsc::error::SendError(e.0.msg))?;
        Ok(TurnHandle { events, outcome })
    }

    /// Cancel the actor. The runtime will checkpoint and exit.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Detach the handle: the actor runs until its own terminal state
    /// (or ancestor cancellation) regardless of this handle. For
    /// fire-and-forget supervision. Mirrors [`crate::ActorHandle::detach`].
    pub fn detach(mut self) {
        self.detached = true;
    }

    /// Wait for the actor to exit and return its exit reason.
    pub async fn join(mut self) -> crate::handle::ActorExit {
        // Caller is awaiting, not abandoning: suppress drop-cancel.
        self.detached = true;
        if let Some(jh) = self.join.take() {
            jh.await
                .unwrap_or(crate::handle::ActorExit::Panicked("join panicked".into()))
        } else {
            crate::handle::ActorExit::Cancelled
        }
    }
}

impl<A: StreamingActorAgent> Drop for StreamingActorHandle<A> {
    fn drop(&mut self) {
        if !self.detached {
            self.cancel.cancel();
        }
        // Cancel-on-drop is cooperative: we don't abort the JoinHandle.
        // The loop observes the cancel and exits cleanly with a final
        // checkpoint; aborting would skip it and lose state.
    }
}
