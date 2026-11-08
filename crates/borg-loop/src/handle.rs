//! `ActorHandle` — the caller-side address of a running actor.
//!
//! Returned by [`crate::spawn_actor`] / [`crate::resume_actor`]. Holds
//! the typed mailbox sender (`send`), the cancel token (`cancel`), the
//! actor's id (`addr`), and the loop's join handle (`join`).
//!
//! Dropping the handle cancels the actor (cancel-on-drop), so a
//! forgotten `cancel` can't leak a running LLM call past the
//! orchestrator's scope. For fire-and-forget, call [`ActorHandle::detach`]
//! first.

use crate::actor::ActorAgent;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// What a finished actor returns to anyone awaiting [`ActorHandle::join`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ActorExit {
    /// The handler returned `HandleOutcome::Terminal(reason)`.
    Terminal(String),
    /// The cancel token was tripped (`cancel`, parent ctx, or `Drop`)
    /// and the loop exited cooperatively.
    Cancelled,
    /// No message for longer than `ActorConfig::idle_timeout`.
    IdleTimeout,
    /// The handler panicked; the runtime caught it at the task
    /// boundary and carries a short formatted string here.
    Panicked(String),
    /// Human-in-the-loop suspend. The actor returned
    /// [`crate::HandleOutcome::Suspended`] and the runtime persisted a
    /// final snapshot before exiting. Callers surface the request, then
    /// call [`crate::resume_actor`] and send a decision-bearing message.
    Suspended {
        request: borg::suspend::SuspendRequest,
    },
}

/// Caller-side handle to a running actor.
///
/// Dropping it cancels the actor; to let the actor outlive the handle,
/// call [`Self::detach`].
#[must_use = "ActorHandle cancels the actor on drop; bind it or call .detach()"]
pub struct ActorHandle<A: ActorAgent> {
    /// Stable identity. The [`crate::ActorStore`] checkpoint key and a
    /// tracing-span / event field.
    pub(crate) addr: Arc<str>,
    /// Bounded mailbox sender; buffer size set at spawn time.
    pub(crate) inbox: mpsc::Sender<A::Message>,
    /// Cancel token, child of the parent context's cancel (if any).
    /// Fires on `cancel()`, `Drop`, or any ancestor cancellation.
    pub(crate) cancel: CancellationToken,
    /// Loop's join handle. `Option` so `join` can take it.
    pub(crate) join: Option<JoinHandle<ActorExit>>,
    /// Whether `Drop` should fire `cancel`. Flipped by `detach()`.
    pub(crate) detached: bool,
}

impl<A: ActorAgent> ActorHandle<A> {
    /// The actor's stable id. Used as the checkpoint store key.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Send a message, failing fast. Returns the message back in the
    /// error if the mailbox is full or the actor has exited. Use
    /// [`Self::send`] to await on backpressure instead.
    pub fn try_send(&self, msg: A::Message) -> Result<(), mpsc::error::TrySendError<A::Message>> {
        self.inbox.try_send(msg)
    }

    /// Send a message, awaiting if the mailbox is full. `Err` only if
    /// the actor has exited (channel closed).
    pub async fn send(&self, msg: A::Message) -> Result<(), mpsc::error::SendError<A::Message>> {
        self.inbox.send(msg).await
    }

    /// Whether the mailbox is closed (actor has exited or is exiting).
    pub fn is_closed(&self) -> bool {
        self.inbox.is_closed()
    }

    /// Cancel the actor. Idempotent. The loop observes it on its next
    /// between-messages `select!` (or immediately if waiting) and exits
    /// with [`ActorExit::Cancelled`] after a final checkpoint.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Detach the handle: the actor runs to its own terminal state (or
    /// ancestor cancellation) regardless of what happens to this
    /// handle. For fire-and-forget supervision.
    pub fn detach(mut self) {
        self.detached = true;
    }

    /// Await the actor's exit. Consumes the handle.
    ///
    /// Returns `ActorExit::Panicked` if the task aborted abnormally or
    /// `join` was called twice; rare in practice since the runtime
    /// catches handler panics internally.
    ///
    /// Dropping the returned future before it completes (an outer
    /// timeout, a losing `select!` branch) drops the handle, which
    /// cancels the actor like any other drop. Call [`Self::detach`]
    /// instead to let the actor outlive its handle.
    pub async fn join(mut self) -> ActorExit {
        let exit = match self.join.take() {
            Some(h) => match h.await {
                Ok(exit) => exit,
                Err(e) => ActorExit::Panicked(format!("join error: {e}")),
            },
            None => ActorExit::Panicked("join called twice".into()),
        };
        // Suppress drop-cancel only once the join has completed: a join
        // abandoned mid-await must still cancel, or the actor leaks.
        self.detached = true;
        exit
    }
}

impl<A: ActorAgent> Drop for ActorHandle<A> {
    fn drop(&mut self) {
        if !self.detached {
            self.cancel.cancel();
        }
        // Cancel-on-drop is cooperative: we don't abort the JoinHandle.
        // The loop observes the cancel and exits cleanly with a final
        // checkpoint; aborting would skip it and lose state.
    }
}
