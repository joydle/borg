//! Runtime for [`StreamingActorAgent`] — `spawn_streaming_actor` and
//! `resume_streaming_actor`.
//!
//! The loop shape is identical to `runtime.rs` but each turn drives a
//! `BoxStream<StreamEvent>` rather than returning a plain `HandleOutcome`.
//! Events are forwarded live to the caller via a per-message mpsc channel.

use crate::handle::ActorExit;
use crate::runtime::ActorConfig;
use crate::store::ActorStore;
use crate::streaming_actor::{
    StreamingActorAgent, StreamingActorHandle, StreamingEnvelope, StreamingHandleOutcome,
};
use borg::StreamEvent;
use futures::{FutureExt, StreamExt};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
#[cfg(feature = "tracing")]
use tracing::Instrument;
use tracing::{debug, info, warn};

/// Spawn a fresh streaming actor. Returns the caller-side handle.
///
/// `addr` is the stable identity used as the store key. Does NOT load
/// an existing snapshot — use [`resume_streaming_actor`] for that.
pub fn spawn_streaming_actor<A: StreamingActorAgent>(
    addr: impl Into<String>,
    agent: A,
    config: ActorConfig,
    store: Arc<dyn ActorStore>,
    parent_cancel: Option<CancellationToken>,
) -> StreamingActorHandle<A> {
    let addr: Arc<str> = Arc::from(addr.into());
    let cancel = match parent_cancel {
        Some(p) => p.child_token(),
        None => CancellationToken::new(),
    };
    let (tx, rx) =
        mpsc::channel::<StreamingEnvelope<A::Message>>(config.effective_mailbox_capacity());
    // tokio's mpsc panics on a zero capacity; treat 0 as 1, like the mailbox.
    let event_capacity = config.event_channel_capacity.max(1);
    let ctx = crate::runtime::ActorCtx {
        cancel: cancel.clone(),
        addr: Arc::clone(&addr),
    };
    let loop_addr = Arc::clone(&addr);
    let loop_cancel = cancel.clone();
    // Inherit the caller's current span as the spawned task's ambient
    // parent so the per-turn `borg.session` spans parent under it.
    #[cfg(feature = "tracing")]
    let join =
        tokio::spawn(
            async move {
                run_streaming_loop(loop_addr, agent, ctx, rx, loop_cancel, config, store).await
            }
            .in_current_span(),
        );
    #[cfg(not(feature = "tracing"))]
    let join = tokio::spawn(async move {
        run_streaming_loop(loop_addr, agent, ctx, rx, loop_cancel, config, store).await
    });
    StreamingActorHandle {
        addr,
        inbox: tx,
        cancel,
        join: Some(join),
        event_channel_capacity: event_capacity,
        detached: false,
    }
}

/// Resume a streaming actor from snapshot, or spawn fresh if none exists.
///
/// `fresh` and `restore` are closures so actors with runtime
/// dependencies (LLM client, store handles, event bus) can plumb them in
/// at the call site rather than through the trait:
///
/// - **Snapshot loads:** `restore(snap)` produces the actor.
/// - **No snapshot:** `fresh()` produces the actor.
/// - **Corrupt snapshot:** `fresh()` is used and the snapshot is
///   overwritten at the next checkpoint.
///
/// Actors that don't need DI can pass `A::restore` directly.
///
/// Returns `Err(ActorStoreError)` only when the store backend failed.
pub async fn resume_streaming_actor<A, F, G>(
    addr: impl Into<String>,
    fresh: F,
    restore: G,
    config: ActorConfig,
    store: Arc<dyn ActorStore>,
    parent_cancel: Option<CancellationToken>,
) -> Result<StreamingActorHandle<A>, crate::store::ActorStoreError>
where
    A: StreamingActorAgent,
    F: FnOnce() -> A,
    G: FnOnce(A::Snapshot) -> A,
{
    let addr_str = addr.into();
    let bytes = store.load(&addr_str).await?;
    let agent = match bytes {
        Some(b) => match serde_json::from_slice::<A::Snapshot>(&b) {
            Ok(snap) => {
                info!(addr = %addr_str, "borg-loop: resuming streaming actor from snapshot");
                restore(snap)
            }
            Err(e) => {
                warn!(
                    addr = %addr_str,
                    error = %e,
                    "borg-loop: streaming actor snapshot deserialize failed, spawning fresh"
                );
                fresh()
            }
        },
        None => {
            debug!(addr = %addr_str, "borg-loop: no snapshot for streaming actor, spawning fresh");
            fresh()
        }
    };
    Ok(spawn_streaming_actor(
        addr_str,
        agent,
        config,
        store,
        parent_cancel,
    ))
}

/// The loop body.
async fn run_streaming_loop<A: StreamingActorAgent>(
    addr: Arc<str>,
    mut agent: A,
    ctx: crate::runtime::ActorCtx,
    mut inbox: mpsc::Receiver<StreamingEnvelope<A::Message>>,
    cancel: CancellationToken,
    config: ActorConfig,
    store: Arc<dyn ActorStore>,
) -> ActorExit {
    let mut turns_since_save: u32 = 0;
    // `Vec::with_capacity` hint for the next save; see
    // `runtime::serialize_checkpoint` for the rationale.
    let mut last_serialized_size: usize = 0;
    // Per-turn `borg.session` span counter. OTel exports a span only
    // when it closes, so we close one per turn rather than holding one
    // for the actor's lifetime.
    #[cfg(feature = "tracing")]
    let mut turn_index: u32 = 0;
    let exit = loop {
        // Wait for the next message, cancel, or idle timeout.
        let envelope: StreamingEnvelope<A::Message> = if let Some(timeout) = config.idle_timeout {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break ActorExit::Cancelled,
                env = inbox.recv() => match env {
                    Some(e) => e,
                    None => {
                        debug!(addr = %addr, "streaming actor inbox closed");
                        break ActorExit::Cancelled;
                    }
                },
                _ = tokio::time::sleep(timeout) => break ActorExit::IdleTimeout,
            }
        } else {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break ActorExit::Cancelled,
                env = inbox.recv() => match env {
                    Some(e) => e,
                    None => {
                        debug!(addr = %addr, "streaming actor inbox closed");
                        break ActorExit::Cancelled;
                    }
                },
            }
        };

        let StreamingEnvelope {
            msg,
            event_tx,
            outcome_tx,
        } = envelope;

        // Per-turn span; drops when the turn finishes, which is what
        // flushes it to OTel exporters. A single lifetime-long span on a
        // parked actor would never close and never export.
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
        // Drive the turn under the session span so spans from
        // `handle_streaming` and the drain loop parent under it.
        //
        // Panic boundary, as in `runtime.rs`: a panicking turn ends the
        // actor with `ActorExit::Panicked` instead of tearing the task
        // down. AssertUnwindSafe is sound: after a panic in
        // `handle_streaming` we never touch `agent` again except to drop it.
        #[cfg(feature = "tracing")]
        let turn_result = AssertUnwindSafe(agent.handle_streaming(msg, &ctx))
            .catch_unwind()
            .instrument(turn_session.clone())
            .await;
        #[cfg(not(feature = "tracing"))]
        let turn_result = AssertUnwindSafe(agent.handle_streaming(msg, &ctx))
            .catch_unwind()
            .await;
        let flow = match turn_result {
            Err(payload) => {
                // `handle_streaming` held `&mut agent`, so its state may be
                // half-mutated: like `runtime.rs`, save nothing. The store
                // keeps the last good snapshot.
                let msg = panic_msg(payload);
                warn!(addr = %addr, panic = %msg, "streaming actor handle_streaming panicked");
                report_panic::<A>(&event_tx, outcome_tx, &msg);
                drop(event_tx);
                TurnFlow::Panicked(msg)
            }
            Ok(Err(e)) => {
                warn!(addr = %addr, error = %e, "streaming actor handle_streaming error");
                // Dual-channel emit: granular `BorgError` on event_tx,
                // coarse Terminal on outcome_tx. Callers rendering UI from
                // both MUST dedup — see the `TurnHandle` doc.
                let _ = event_tx.send(StreamEvent::Failed(e)).await;
                let _ = outcome_tx.send(StreamingHandleOutcome::Terminal(
                    "handle_streaming error".into(),
                ));
                TurnFlow::Terminal("handle_streaming error".into())
            }
            Ok(Ok((mut stream, outcome))) => {
                // Drain the stream, forwarding events to the caller. Every
                // wait (the stream's next event, and a send into a full
                // event channel) races cancel, and each poll of the stream
                // sits behind a panic boundary. Bail out fast if the caller
                // dropped the `TurnHandle` (event_tx.send → Err).
                #[cfg(feature = "tracing")]
                let drain_session = turn_session.clone();
                let drain = async {
                    loop {
                        let next = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => {
                                // Dual-channel emit (as in the
                                // handle_streaming-Err branch): granular
                                // `BorgError::Cancelled` here, coarse
                                // `Terminal("cancelled")` on outcome_tx
                                // below. Callers MUST dedup. `try_send`:
                                // a full channel must not block the exit.
                                let _ = event_tx
                                    .try_send(StreamEvent::Failed(borg::BorgError::Cancelled));
                                break DrainResult::Cancelled;
                            }
                            item = AssertUnwindSafe(stream.next()).catch_unwind() => item,
                        };
                        match next {
                            Err(payload) => break DrainResult::Panicked(panic_msg(payload)),
                            // stream ended without Done — treat as complete
                            Ok(None) => break DrainResult::StreamEnded,
                            Ok(Some(ev)) => {
                                let terminal = ev.is_terminal();
                                // A caller holding the TurnHandle without
                                // draining it must not pin the actor past
                                // a cancel.
                                let sent = tokio::select! {
                                    biased;
                                    _ = cancel.cancelled() => {
                                        let _ = event_tx.try_send(StreamEvent::Failed(
                                            borg::BorgError::Cancelled,
                                        ));
                                        break DrainResult::Cancelled;
                                    }
                                    r = event_tx.send(ev) => r,
                                };
                                if sent.is_err() {
                                    debug!(
                                        addr = %addr,
                                        "turn handle dropped mid-stream; parking actor"
                                    );
                                    // Drop the rest of the stream (cancels its
                                    // in-flight futures) and return to the inbox.
                                    drop(stream);
                                    break DrainResult::CallerGone;
                                }
                                if terminal {
                                    break DrainResult::StreamEnded;
                                }
                            }
                        }
                    }
                };
                #[cfg(feature = "tracing")]
                let drain_result = drain.instrument(drain_session).await;
                #[cfg(not(feature = "tracing"))]
                let drain_result = drain.await;

                if let DrainResult::Panicked(msg) = drain_result {
                    // The stream panicked, but `handle_streaming` had
                    // returned normally and the stream can't touch the
                    // agent: its state is the consistent post-turn state.
                    // Persist it as the last good snapshot, then exit.
                    warn!(addr = %addr, panic = %msg, "streaming actor stream panicked");
                    report_panic::<A>(&event_tx, outcome_tx, &msg);
                    drop(event_tx);
                    if let Some(bytes) = serialize_checkpoint(&addr, &agent, last_serialized_size) {
                        last_serialized_size = bytes.len();
                        crate::store::persist_checkpoint(
                            &addr,
                            bytes,
                            store.as_ref(),
                            "streaming-actor",
                        )
                        .await;
                    }
                    TurnFlow::Panicked(msg)
                } else if drain_result == DrainResult::Cancelled {
                    // Send the outcome before closing event_tx so
                    // `await turn.outcome` is non-blocking after draining.
                    let _ = outcome_tx.send(StreamingHandleOutcome::Terminal("cancelled".into()));
                    // Drop closes the events receiver, signalling end of turn.
                    drop(event_tx);
                    TurnFlow::Cancelled
                } else {
                    // On CallerGone the receiver is gone, so this is
                    // best-effort.
                    let _ = outcome_tx.send(outcome.clone());
                    drop(event_tx);

                    // Checkpoint if due. Serialize synchronously (borrow
                    // ends before the async save) so A: Sync isn't required.
                    turns_since_save = turns_since_save.saturating_add(1);
                    if config.checkpoint_interval > 0
                        && turns_since_save >= config.checkpoint_interval
                    {
                        if let Some(bytes) =
                            serialize_checkpoint(&addr, &agent, last_serialized_size)
                        {
                            last_serialized_size = bytes.len();
                            crate::store::persist_checkpoint(
                                &addr,
                                bytes,
                                store.as_ref(),
                                "streaming-actor",
                            )
                            .await;
                        }
                        turns_since_save = 0;
                    }

                    // CallerGone softens a Continue to Park: no caller to
                    // chain the next turn to, so the actor stays alive
                    // waiting. Terminal still terminates.
                    let effective = match (drain_result, outcome) {
                        (DrainResult::CallerGone, StreamingHandleOutcome::Continue) => {
                            StreamingHandleOutcome::Park
                        }
                        (_, o) => o,
                    };
                    match effective {
                        StreamingHandleOutcome::Continue | StreamingHandleOutcome::Park => {
                            TurnFlow::Continue
                        }
                        StreamingHandleOutcome::Terminal(reason) => {
                            info!(addr = %addr, %reason, "streaming actor terminal");
                            TurnFlow::Terminal(reason)
                        }
                        StreamingHandleOutcome::Suspended { request } => {
                            info!(
                                addr = %addr,
                                blocked_tool = ?request.blocked_tool,
                                "streaming actor suspended for human decision"
                            );
                            // Force-persist regardless of checkpoint_interval:
                            // the awaiting-decision state must survive restart.
                            if let Some(bytes) =
                                serialize_checkpoint(&addr, &agent, last_serialized_size)
                            {
                                last_serialized_size = bytes.len();
                                crate::store::persist_checkpoint(
                                    &addr,
                                    bytes,
                                    store.as_ref(),
                                    "streaming-actor",
                                )
                                .await;
                            }
                            TurnFlow::Suspended { request }
                        }
                    }
                }
            }
        };

        // Span drops here: under tracing-opentelemetry this queues it
        // for export.
        #[cfg(feature = "tracing")]
        drop(turn_session);

        match flow {
            TurnFlow::Continue => {}
            TurnFlow::Cancelled => break ActorExit::Cancelled,
            TurnFlow::Panicked(msg) => break ActorExit::Panicked(msg),
            TurnFlow::Terminal(reason) => break ActorExit::Terminal(reason),
            TurnFlow::Suspended { request } => break ActorExit::Suspended { request },
        }
    };

    // Final checkpoint on graceful exit (cancel, idle). Skip on:
    //  - Panic: a `handle_streaming` panic may leave state inconsistent
    //    (the store keeps the last good snapshot); a stream panic already
    //    persisted the consistent post-turn state in the loop.
    //  - Terminal: the `delete_on_terminal` call below would invalidate
    //    the save (wasted round-trip + cache pollution).
    //  - Suspended: already persisted in the effective-match arm above.
    if !matches!(
        exit,
        ActorExit::Panicked(_) | ActorExit::Terminal(_) | ActorExit::Suspended { .. }
    ) {
        if let Some(bytes) = serialize_checkpoint(&addr, &agent, last_serialized_size) {
            crate::store::persist_checkpoint(&addr, bytes, store.as_ref(), "streaming-actor").await;
        }
    }

    crate::store::delete_on_terminal(&exit, &addr, store.as_ref(), "streaming-actor").await;

    exit
}

/// Control flow out of one turn's processing block.
#[derive(Debug)]
enum TurnFlow {
    /// Loop back to wait for the next message.
    Continue,
    /// HITL pause — exit with [`ActorExit::Suspended`] carrying the
    /// request. The snapshot has already been persisted by the time
    /// this variant is produced.
    Suspended {
        request: borg::suspend::SuspendRequest,
    },
    /// Cancel observed mid-turn — exit with `ActorExit::Cancelled`.
    Cancelled,
    /// The turn panicked — exit with `ActorExit::Panicked(msg)`.
    Panicked(String),
    /// Agent returned `Terminal` (or runtime synthesized one) — exit
    /// with `ActorExit::Terminal(reason)`.
    Terminal(String),
}

/// Why the per-turn drain loop ended. Internal to `run_streaming_loop`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DrainResult {
    /// Stream produced its terminal event or simply ran out — normal end.
    StreamEnded,
    /// Actor cancel token fired while waiting on the stream or on a full
    /// event channel.
    Cancelled,
    /// Polling the stream panicked; carries the panic message.
    Panicked(String),
    /// `event_tx.send` returned `Err` — the caller dropped the
    /// `TurnHandle` mid-turn. The actor stays alive for the next
    /// message; the current turn is abandoned without draining the rest.
    CallerGone,
}

/// Tell the caller its turn panicked, on both channels (callers dedup
/// as for cancel). Non-blocking: a full or abandoned event channel must
/// not stall the actor's exit.
fn report_panic<A: StreamingActorAgent>(
    event_tx: &mpsc::Sender<StreamEvent>,
    outcome_tx: oneshot::Sender<StreamingHandleOutcome>,
    msg: &str,
) {
    let _ = event_tx.try_send(StreamEvent::Failed(borg::BorgError::ChildPanicked {
        agent_id: std::any::type_name::<A>().to_string(),
        message: msg.to_string(),
    }));
    let _ = outcome_tx.send(StreamingHandleOutcome::Terminal(format!("panicked: {msg}")));
}

/// Flatten a caught panic payload to a string. Mirror of the private
/// helper in `runtime.rs`.
fn panic_msg(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<unknown panic payload>".to_string()
    }
}

/// Serialize the agent's checkpoint into bytes. Pure (no IO, no await)
/// so the `&agent` borrow doesn't cross an await point. Mirror of
/// [`crate::runtime::serialize_checkpoint`]; see it for the `hint` and
/// `Arc<[u8]>` rationale.
fn serialize_checkpoint<A: StreamingActorAgent>(
    addr: &str,
    agent: &A,
    hint: usize,
) -> Option<Arc<[u8]>> {
    let mut buf = Vec::with_capacity(hint.max(256));
    match serde_json::to_writer(&mut buf, &agent.checkpoint()) {
        Ok(()) => Some(Arc::from(buf)),
        Err(e) => {
            warn!(addr = %addr, error = %e, "streaming-actor: checkpoint serialize failed");
            None
        }
    }
}
