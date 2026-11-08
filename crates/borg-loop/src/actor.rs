//! `ActorAgent` trait — the contract long-lived agents implement.
//!
//! An actor agent is a stateful, single-threaded message processor.
//! It owns its state, processes one message at a time, occasionally
//! emits a serializable snapshot for checkpointing, and signals when
//! it's done.
//!
//! - **State is owned by the actor.** The runtime holds it by `&mut`
//!   inside the loop and never inspects internals, so each app keeps
//!   its own typed state with no trait-level erasure.
//! - **Messages and snapshots are typed per agent.** The runtime is
//!   monomorphized per actor type; no untyped `dyn Any` mailbox.
//!   Pick a snapshot shape that stays stable across redeploys.
//! - **Single-threaded by construction:** `&mut self` in `handle`
//!   means no concurrent processing within one actor. For
//!   parallelism, spawn N actors.

use crate::ActorCtx;
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};

/// What an actor returns from each [`ActorAgent::handle`] call to
/// signal the runtime what to do next.
#[derive(Debug)]
#[non_exhaustive]
pub enum HandleOutcome {
    /// Keep waiting for the next message. The default outcome.
    Continue,

    /// Terminal state: the runtime shuts down the loop. The reason is
    /// recorded in tracing and surfaced to callers awaiting exit.
    Terminal(String),

    /// Human-in-the-loop pause. The actor can't continue without a
    /// human decision. The runtime persists a final snapshot via
    /// [`ActorAgent::checkpoint`], then exits with
    /// [`crate::ActorExit::Suspended`] carrying the request.
    ///
    /// The actor must encode the pending decision in its `Snapshot`
    /// (e.g. a `pending_suspend: Option<SuspendRequest>` field). On
    /// resume, the consumer sends a decision-bearing message that the
    /// actor processes in its normal `handle` flow.
    Suspended {
        request: borg::suspend::SuspendRequest,
    },
}

/// The contract long-lived agents implement to participate in an
/// actor loop.
///
/// One instance lives for the lifetime of the actor task. Messages
/// arrive on the typed mailbox; the runtime calls [`Self::handle`]
/// once per message. Snapshots are captured periodically via
/// [`Self::checkpoint`] and restored on resume via [`Self::restore`].
///
/// ## Implementing
///
/// ```ignore
/// use async_trait::async_trait;
/// use borg_loop::{ActorAgent, ActorCtx, HandleOutcome};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct CounterSnap { count: u32 }
///
/// struct Counter { count: u32 }
///
/// enum CounterMsg { Bump, Get(tokio::sync::oneshot::Sender<u32>), Stop }
///
/// #[async_trait]
/// impl ActorAgent for Counter {
///     type Message = CounterMsg;
///     type Snapshot = CounterSnap;
///
///     async fn handle(&mut self, msg: CounterMsg, _ctx: &ActorCtx) -> HandleOutcome {
///         match msg {
///             CounterMsg::Bump => { self.count += 1; HandleOutcome::Continue }
///             CounterMsg::Get(reply) => { let _ = reply.send(self.count); HandleOutcome::Continue }
///             CounterMsg::Stop => HandleOutcome::Terminal("stopped".into()),
///         }
///     }
///
///     fn checkpoint(&self) -> Self::Snapshot { CounterSnap { count: self.count } }
///     fn restore(snap: Self::Snapshot) -> Self { Self { count: snap.count } }
/// }
/// ```
#[async_trait]
pub trait ActorAgent: Send + 'static {
    /// Inbound message type. The runtime's mailbox is
    /// `tokio::mpsc::Sender<Self::Message>` — typed end-to-end.
    type Message: Send + 'static;

    /// Serializable state snapshot. Must round-trip cleanly via
    /// `serde_json`. Store only domain state that needs to survive
    /// restart — not cancel tokens, JoinHandles, or file descriptors.
    type Snapshot: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Process one message. Called once per arrival.
    ///
    /// Return [`HandleOutcome::Continue`] to keep looping, or
    /// [`HandleOutcome::Terminal`] to shut down (a final checkpoint is
    /// written first). `&mut self` means no concurrent processing;
    /// ordering is preserved by the mpsc. For fan-out, spawn child
    /// tasks and own their join handles in `self`.
    ///
    /// # Cancellation contract
    ///
    /// The runtime races cancel against the mailbox receive, so a
    /// cancel between messages exits immediately. But once `handle`
    /// runs, the runtime does NOT race its `.await` against
    /// [`ActorCtx::cancel`]. Implementations MUST `tokio::select!`
    /// against `ctx.cancel.cancelled()` for every long-running
    /// `.await` (LLM calls, subprocess waits, network I/O), or
    /// cancellation latency equals that `.await`'s worst case.
    ///
    /// ```ignore
    /// async fn handle(&mut self, msg: Msg, ctx: &ActorCtx) -> HandleOutcome {
    ///     let response = tokio::select! {
    ///         biased;
    ///         _ = ctx.cancel.cancelled() => return HandleOutcome::Continue,
    ///         r = self.llm.chat_completion(req) => r,
    ///     };
    ///     // ... use response ...
    ///     HandleOutcome::Continue
    /// }
    /// ```
    async fn handle(&mut self, msg: Self::Message, ctx: &ActorCtx) -> HandleOutcome;

    /// Serialize the actor's state for the store. Called at
    /// configurable intervals and on graceful shutdown. Keep it cheap:
    /// every call blocks the loop.
    fn checkpoint(&self) -> Self::Snapshot;

    /// Reconstruct the actor from a saved snapshot, before any messages
    /// are processed. Rebuild derived runtime state (caches, in-flight
    /// task handles) here from the snapshot's domain data.
    fn restore(snap: Self::Snapshot) -> Self
    where
        Self: Sized;
}
