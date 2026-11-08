//! `actor_loop` — the runtime that drives an [`ActorAgent`].
//!
//! ## Loop shape
//!
//! Pseudo-code:
//!
//! ```text
//! loop {
//!     tokio::select! {
//!         msg = inbox.recv() => agent.handle(msg, &ctx).await,
//!         _ = cancel.cancelled() => break Cancelled,
//!         _ = idle_timer(cfg.idle_timeout) => break IdleTimeout,
//!     }
//!     if messages_since_save >= cfg.checkpoint_every { save; reset counter }
//! }
//! checkpoint one more time;
//! emit ActorExit;
//! ```
//!
//! ## Concurrency model
//!
//! One actor = one tokio task = one `&mut` state. No lock contention,
//! no inter-message races. For parallelism, spawn N actors.
//!
//! ## Resume
//!
//! [`resume_actor`] loads the snapshot, calls `ActorAgent::restore` to
//! reconstruct state, then spawns the same loop under the same `addr`
//! and store.

use crate::actor::{ActorAgent, HandleOutcome};
use crate::handle::{ActorExit, ActorHandle};
use crate::store::ActorStore;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
#[cfg(feature = "tracing")]
use tracing::Instrument;
use tracing::{debug, info, warn};

/// Read-only context passed to [`ActorAgent::handle`].
#[non_exhaustive]
pub struct ActorCtx {
    /// Cooperative cancellation. The handler SHOULD `tokio::select!`
    /// against `cancel.cancelled()` for long-running `.await`s; the
    /// loop only races cancel against the mailbox, not inside `handle`.
    pub cancel: CancellationToken,

    /// The actor's stable id. Same value as `ActorHandle::addr()`.
    pub addr: Arc<str>,
}

/// Tunable parameters for both plain and streaming actor loops.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ActorConfig {
    /// Inbox capacity. Backpressure: senders block / fail when full.
    /// Default 64 for plain actors; streaming actors typically use 16
    /// (turns are long-running, so less concurrency is needed). `0` is
    /// treated as `1` (a bounded channel needs room for one message).
    pub mailbox_capacity: usize,

    /// Save a checkpoint after this many activations. For plain actors an
    /// activation is one message processed; for streaming actors it is one
    /// LLM turn. Set to `0` to disable periodic checkpointing (the runtime
    /// still writes a final checkpoint on graceful exit).
    pub checkpoint_interval: u32,

    /// Exit with [`ActorExit::IdleTimeout`] if no message arrives within
    /// this window. `None` disables. The timer restarts on each message
    /// receipt — idle is measured from the most recent message, not from
    /// actor start.
    pub idle_timeout: Option<Duration>,

    /// Per-turn event-channel capacity for streaming actors. Each
    /// [`crate::streaming_actor::StreamingActorHandle::send`] creates a
    /// fresh `mpsc::Sender<StreamEvent>` with this capacity. Drain the
    /// [`crate::streaming_actor::TurnHandle`] promptly to avoid
    /// backpressuring the runtime. Ignored by plain `ActorAgent`.
    pub event_channel_capacity: usize,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            mailbox_capacity: 64,
            checkpoint_interval: 5,
            idle_timeout: None,
            event_channel_capacity: 64,
        }
    }
}

impl ActorConfig {
    /// The mailbox capacity to build the channel with: `mailbox_capacity`,
    /// with `0` raised to `1` (tokio's bounded channel panics on zero).
    pub(crate) fn effective_mailbox_capacity(&self) -> usize {
        self.mailbox_capacity.max(1)
    }

    /// Streaming-actor defaults: smaller mailbox (16) and checkpoint every
    /// turn (1) because streaming turns are expensive and shouldn't be skipped.
    pub fn for_streaming() -> Self {
        Self {
            mailbox_capacity: 16,
            checkpoint_interval: 1,
            idle_timeout: None,
            event_channel_capacity: 64,
        }
    }
}

/// Spawn a fresh actor (no prior snapshot). Returns the handle.
///
/// `addr` is the stable identity used as the store key. Does NOT load
/// an existing snapshot under `addr` — use [`resume_actor`] for that.
pub fn spawn_actor<A: ActorAgent>(
    addr: impl Into<String>,
    agent: A,
    config: ActorConfig,
    store: Arc<dyn ActorStore>,
    parent_cancel: Option<CancellationToken>,
) -> ActorHandle<A> {
    let addr: Arc<str> = Arc::from(addr.into());
    let cancel = match parent_cancel {
        Some(p) => p.child_token(),
        None => CancellationToken::new(),
    };
    let (tx, rx) = mpsc::channel::<A::Message>(config.effective_mailbox_capacity());
    let ctx = ActorCtx {
        cancel: cancel.clone(),
        addr: Arc::clone(&addr),
    };
    let loop_addr = Arc::clone(&addr);
    let loop_cancel = cancel.clone();
    // Inherit the caller's current span as the spawned task's ambient
    // parent so the per-turn `borg.session` spans parent under it.
    #[cfg(feature = "tracing")]
    let join = tokio::spawn(
        async move { run_loop(loop_addr, agent, ctx, rx, loop_cancel, config, store).await }
            .in_current_span(),
    );
    #[cfg(not(feature = "tracing"))]
    let join = tokio::spawn(async move {
        run_loop(loop_addr, agent, ctx, rx, loop_cancel, config, store).await
    });
    ActorHandle {
        addr,
        inbox: tx,
        cancel,
        join: Some(join),
        detached: false,
    }
}

/// Resume an actor from a saved snapshot, or spawn fresh if none exists.
///
/// `fresh` and `restore` are closures so actors with runtime
/// dependencies (LLM client, store handles, event bus) can plumb them
/// in at the call site rather than through the trait:
///
/// - **Snapshot loads:** `restore(snap)` produces the actor.
/// - **No snapshot:** `fresh()` produces the actor.
/// - **Corrupt snapshot:** `fresh()` is used and the snapshot is
///   overwritten at the next checkpoint.
///
/// Actors that don't need DI can pass `A::restore` directly.
///
/// Returns `Err(ActorStoreError)` only when the store backend failed.
pub async fn resume_actor<A, F, G>(
    addr: impl Into<String>,
    fresh: F,
    restore: G,
    config: ActorConfig,
    store: Arc<dyn ActorStore>,
    parent_cancel: Option<CancellationToken>,
) -> Result<ActorHandle<A>, crate::store::ActorStoreError>
where
    A: ActorAgent,
    F: FnOnce() -> A,
    G: FnOnce(A::Snapshot) -> A,
{
    let addr_str = addr.into();
    let bytes = store.load(&addr_str).await?;
    let agent = match bytes {
        Some(b) => match serde_json::from_slice::<A::Snapshot>(&b) {
            Ok(snap) => {
                info!(addr = %addr_str, "borg-loop: resuming actor from snapshot");
                restore(snap)
            }
            Err(e) => {
                warn!(
                    addr = %addr_str,
                    error = %e,
                    "borg-loop: snapshot deserialize failed, spawning fresh"
                );
                fresh()
            }
        },
        None => {
            debug!(addr = %addr_str, "borg-loop: no snapshot, spawning fresh");
            fresh()
        }
    };
    Ok(spawn_actor(addr_str, agent, config, store, parent_cancel))
}

async fn run_loop<A: ActorAgent>(
    addr: Arc<str>,
    mut agent: A,
    ctx: ActorCtx,
    mut inbox: mpsc::Receiver<A::Message>,
    cancel: CancellationToken,
    config: ActorConfig,
    store: Arc<dyn ActorStore>,
) -> ActorExit {
    let mut messages_since_save: u32 = 0;
    // `Vec::with_capacity` hint for the next save: steady-state
    // checkpoints are similarly-sized, so pre-allocating skips
    // serde_json's geometric-growth realloc chain.
    let mut last_serialized_size: usize = 0;
    // Per-turn `borg.session` span counter. OTel exports a span only
    // when it closes, so we close one per turn rather than holding a
    // single span for the actor's lifetime.
    #[cfg(feature = "tracing")]
    let mut turn_index: u32 = 0;
    let exit = loop {
        let next: NextStep<A::Message> = if let Some(timeout) = config.idle_timeout {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => NextStep::Cancelled,
                msg = inbox.recv() => match msg {
                    Some(m) => NextStep::Message(m),
                    None => NextStep::InboxClosed,
                },
                _ = tokio::time::sleep(timeout) => NextStep::IdleTimeout,
            }
        } else {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => NextStep::Cancelled,
                msg = inbox.recv() => match msg {
                    Some(m) => NextStep::Message(m),
                    None => NextStep::InboxClosed,
                },
            }
        };

        match next {
            NextStep::Cancelled => break ActorExit::Cancelled,
            NextStep::IdleTimeout => break ActorExit::IdleTimeout,
            NextStep::InboxClosed => {
                // All senders dropped — treat as graceful shutdown.
                debug!(addr = %addr, "borg-loop: inbox closed, exiting");
                break ActorExit::Cancelled;
            }
            NextStep::Message(m) => {
                // Catch panics so one bad handler invocation doesn't take
                // down the loop without leaving a checkpoint behind.
                // AssertUnwindSafe is sound: we never touch `agent` after
                // a panic (we exit the loop).
                use futures::FutureExt;
                // Per-turn span; drops at the end of this match arm,
                // which is what flushes it to OTel exporters.
                #[cfg(feature = "tracing")]
                let turn_session = tracing::info_span!(
                    "borg.session",
                    session.id = %addr,
                    actor.type = std::any::type_name::<A>(),
                    turn.index = turn_index,
                );
                #[cfg(feature = "tracing")]
                {
                    turn_index = turn_index.saturating_add(1);
                }
                #[cfg(feature = "tracing")]
                let result = std::panic::AssertUnwindSafe(agent.handle(m, &ctx))
                    .catch_unwind()
                    .instrument(turn_session)
                    .await
                    .map_panic();
                #[cfg(not(feature = "tracing"))]
                let result = std::panic::AssertUnwindSafe(agent.handle(m, &ctx))
                    .catch_unwind()
                    .await
                    .map_panic();
                match result {
                    Ok(HandleOutcome::Continue) => {
                        messages_since_save = messages_since_save.saturating_add(1);
                        if config.checkpoint_interval > 0
                            && messages_since_save >= config.checkpoint_interval
                        {
                            // Two-stage save: serialize synchronously
                            // (borrowing &agent), then await store.save with
                            // owned bytes. Keeps `A: Sync` off the trait.
                            if let Some(bytes) =
                                serialize_checkpoint(&addr, &agent, last_serialized_size)
                            {
                                last_serialized_size = bytes.len();
                                crate::store::persist_checkpoint(
                                    &addr,
                                    bytes,
                                    store.as_ref(),
                                    "borg-loop",
                                )
                                .await;
                            }
                            messages_since_save = 0;
                        }
                    }
                    Ok(HandleOutcome::Terminal(reason)) => {
                        info!(addr = %addr, %reason, "borg-loop: actor terminal");
                        break ActorExit::Terminal(reason);
                    }
                    Ok(HandleOutcome::Suspended { request }) => {
                        info!(
                            addr = %addr,
                            blocked_tool = ?request.blocked_tool,
                            "borg-loop: actor suspended for human decision"
                        );
                        // Persist before breaking so the awaiting-decision
                        // state survives a restart; the post-loop checkpoint
                        // skips Suspended (see `matches!` below).
                        if let Some(bytes) =
                            serialize_checkpoint(&addr, &agent, last_serialized_size)
                        {
                            last_serialized_size = bytes.len();
                            crate::store::persist_checkpoint(
                                &addr,
                                bytes,
                                store.as_ref(),
                                "borg-loop",
                            )
                            .await;
                        }
                        break ActorExit::Suspended { request };
                    }
                    Err(panic_msg) => {
                        warn!(
                            addr = %addr,
                            panic = %panic_msg,
                            "borg-loop: handler panicked"
                        );
                        break ActorExit::Panicked(panic_msg);
                    }
                }
            }
        }
    };

    // Final checkpoint on graceful exit (cancel, idle). Skip on:
    //  - Panic: agent state may be inconsistent.
    //  - Terminal: the `delete_on_terminal` call below would invalidate
    //    the save (wasted round-trip + cache pollution).
    //  - Suspended: already persisted in the match arm above.
    if !matches!(
        exit,
        ActorExit::Panicked(_) | ActorExit::Terminal(_) | ActorExit::Suspended { .. }
    ) {
        if let Some(bytes) = serialize_checkpoint(&addr, &agent, last_serialized_size) {
            crate::store::persist_checkpoint(&addr, bytes, store.as_ref(), "borg-loop").await;
        }
    }

    crate::store::delete_on_terminal(&exit, &addr, store.as_ref(), "borg-loop").await;

    exit
}

/// Serialize the agent's checkpoint into bytes. Pure (no IO, no await)
/// so the `&agent` borrow doesn't cross an await point — keeps `A: Sync`
/// off the public trait.
///
/// `hint` is the previous serialized size (0 on first call); pre-sizing
/// the Vec skips serde_json's geometric-growth realloc chain. Returns
/// `Arc<[u8]>` so the payload fans out through the
/// [`crate::store::CachingActorStore`] / [`crate::store::ResilientActorStore`]
/// stack via refcount bumps rather than per-layer copies.
fn serialize_checkpoint<A: ActorAgent>(addr: &str, agent: &A, hint: usize) -> Option<Arc<[u8]>> {
    let mut buf = Vec::with_capacity(hint.max(256));
    match serde_json::to_writer(&mut buf, &agent.checkpoint()) {
        Ok(()) => Some(Arc::from(buf)),
        Err(e) => {
            warn!(addr = %addr, error = %e, "borg-loop: snapshot serialize failed (non-fatal)");
            None
        }
    }
}

/// What the `select!` resolved to. Gives the match arms a single shape
/// whether or not the `idle_timeout` branch is present.
enum NextStep<M> {
    Message(M),
    Cancelled,
    IdleTimeout,
    InboxClosed,
}

// ── catch_unwind for async ───────────────────────────────────────────
//
// std::panic::catch_unwind is sync-only, so for an async block we wrap
// the future with AssertUnwindSafe + futures::FutureExt::catch_unwind at
// the call site, and MapPanic flattens the boxed Any to a String.

// Map the boxed Any panic payload to a short string for logs / events.
fn panic_msg(boxed: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = boxed.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = boxed.downcast_ref::<String>() {
        s.clone()
    } else {
        "<unknown panic payload>".to_string()
    }
}

// Flattens catch_unwind's Result<T, Box<dyn Any>> to Result<T, String>.
trait MapPanic<T> {
    fn map_panic(self) -> Result<T, String>;
}
impl<T> MapPanic<T> for Result<T, Box<dyn std::any::Any + Send>> {
    fn map_panic(self) -> Result<T, String> {
        self.map_err(panic_msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_mailbox_capacity_is_raised_to_one() {
        let mut c = ActorConfig {
            mailbox_capacity: 0,
            ..Default::default()
        };
        assert_eq!(c.effective_mailbox_capacity(), 1);
        c.mailbox_capacity = 7;
        assert_eq!(c.effective_mailbox_capacity(), 7);
    }

    #[test]
    fn actor_config_for_streaming_has_streaming_defaults() {
        let c = ActorConfig::for_streaming();
        assert_eq!(c.mailbox_capacity, 16);
        assert_eq!(c.checkpoint_interval, 1);
    }
}
