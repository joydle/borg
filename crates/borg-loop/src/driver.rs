//! Generic session driver for streaming actors.
//!
//! [`StreamingActorHandle::run_session_loop`] drives a [`StreamingActorAgent`]
//! to completion by sending continuation messages until the actor parks
//! (graceful user-turn end) or terminates (error, cancel, or explicit
//! [`Terminal`] outcome). Only the concrete `Continue` message variant
//! differs across apps; the `continuation` closure captures that.
//!
//! ## Example
//!
//! ```ignore
//! enum Msg { Start(String), Continue, Stop }
//!
//! let parked = handle
//!     .run_session_loop(
//!         Msg::Start("do something".into()),
//!         || Msg::Continue,
//!         |ev| println!("{ev:?}"),
//!     )
//!     .await;
//! ```
//!
//! [`Terminal`]: crate::StreamingHandleOutcome::Terminal

use crate::streaming_actor::{StreamingActorAgent, StreamingActorHandle, StreamingHandleOutcome};
use borg::StreamEvent;
use tracing::info;

/// Outcome of a completed [`StreamingActorHandle::run_session_loop`] call.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SessionOutcome {
    /// The actor parked after a graceful user-turn completion. It is still
    /// alive and waiting for the next user message.
    Parked,
    /// The actor exited (Terminal, Cancelled, or inbox closed). It must be
    /// re-spawned for the next user message.
    Terminal,
    /// HITL pause. The actor returned
    /// [`StreamingHandleOutcome::Suspended`] and exited via
    /// [`crate::ActorExit::Suspended`]; the snapshot is persisted.
    /// Callers surface the request to the user, then resume with a
    /// decision-bearing message.
    Suspended {
        request: borg::suspend::SuspendRequest,
    },
}

impl<A: StreamingActorAgent> StreamingActorHandle<A> {
    /// Drive this actor from its initial message to park or terminal.
    ///
    /// Sends `initial` to the actor, drains its event stream (passing each
    /// event to `on_event`), then reads the turn outcome:
    ///
    /// - [`Continue`] → send `continuation()`, repeat.
    /// - [`Park`] → stop; return [`SessionOutcome::Parked`].
    /// - [`Terminal`] or send failure → return [`SessionOutcome::Terminal`].
    ///
    /// `on_event` is called with each [`StreamEvent`] in order. It runs
    /// synchronously in the driver loop — keep it cheap (fan-out to a channel,
    /// not blocking I/O).
    ///
    /// [`Continue`]: StreamingHandleOutcome::Continue
    /// [`Park`]: StreamingHandleOutcome::Park
    /// [`Terminal`]: StreamingHandleOutcome::Terminal
    pub async fn run_session_loop<F, H>(
        &self,
        initial: A::Message,
        mut continuation: F,
        mut on_event: H,
    ) -> SessionOutcome
    where
        F: FnMut() -> A::Message,
        H: FnMut(StreamEvent),
    {
        let mut next_msg = Some(initial);
        let mut turn: u32 = 0;

        loop {
            let msg = match next_msg.take() {
                Some(m) => m,
                None => continuation(),
            };

            let mut turn_handle = match self.send(msg).await {
                Ok(th) => th,
                Err(_actor_gone) => {
                    info!(turns = turn, "session_driver: actor inbox closed");
                    return SessionOutcome::Terminal;
                }
            };

            // Drain events for this turn.
            while let Some(ev) = turn_handle.next_event().await {
                on_event(ev);
            }

            // Read outcome (always resolves after events drain).
            let outcome = turn_handle.outcome().await;

            turn += 1;

            match outcome {
                StreamingHandleOutcome::Continue => {
                    // Another turn needed; loop and call continuation().
                }
                StreamingHandleOutcome::Park => {
                    info!(turns = turn, "session_driver: actor parked");
                    return SessionOutcome::Parked;
                }
                StreamingHandleOutcome::Terminal(reason) => {
                    info!(turns = turn, %reason, "session_driver: actor terminal");
                    return SessionOutcome::Terminal;
                }
                StreamingHandleOutcome::Suspended { request } => {
                    info!(
                        turns = turn,
                        blocked_tool = ?request.blocked_tool,
                        "session_driver: actor suspended for human decision"
                    );
                    return SessionOutcome::Suspended { request };
                }
            }
        }
    }
}
