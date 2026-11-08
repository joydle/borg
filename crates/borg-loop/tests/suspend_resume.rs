//! Integration test for the durable HITL suspend lifecycle.
//!
//! An actor returning [`HandleOutcome::Suspended`] must:
//! - exit with [`ActorExit::Suspended`] carrying the request, and
//! - persist its snapshot before exit, so [`resume_actor`] can rehydrate
//!   the state and deliver a decision.

use async_trait::async_trait;
use borg::suspend::{SuspendKind, SuspendRequest};
use borg::{BorgError, StreamEvent};
use borg_loop::{
    resume_actor, spawn_actor, spawn_streaming_actor, ActorAgent, ActorConfig, ActorCtx, ActorExit,
    ActorStore, HandleOutcome, InMemoryActorStore, StreamingActorAgent, StreamingHandleOutcome,
};
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Test actor: returns Suspended on receiving Trigger; processes
/// `Resume(approved)` after rehydration.
#[derive(Debug)]
enum Msg {
    Trigger,
    Resume { approved: bool },
}

#[derive(Serialize, Deserialize, Default, Debug, PartialEq, Eq)]
struct Snap {
    awaiting_decision: bool,
    final_approved: Option<bool>,
}

struct ApprovalActor {
    awaiting_decision: bool,
    final_approved: Option<bool>,
}

#[async_trait]
impl ActorAgent for ApprovalActor {
    type Message = Msg;
    type Snapshot = Snap;

    async fn handle(&mut self, msg: Msg, _ctx: &ActorCtx) -> HandleOutcome {
        match msg {
            Msg::Trigger => {
                self.awaiting_decision = true;
                HandleOutcome::Suspended {
                    request: SuspendRequest::new("approve migration?", SuspendKind::Approval)
                        .with_blocked_tool("run_migration"),
                }
            }
            Msg::Resume { approved } => {
                self.awaiting_decision = false;
                self.final_approved = Some(approved);
                HandleOutcome::Terminal("decision recorded".into())
            }
        }
    }

    fn checkpoint(&self) -> Snap {
        Snap {
            awaiting_decision: self.awaiting_decision,
            final_approved: self.final_approved,
        }
    }

    fn restore(snap: Snap) -> Self {
        Self {
            awaiting_decision: snap.awaiting_decision,
            final_approved: snap.final_approved,
        }
    }
}

#[tokio::test]
async fn handle_outcome_suspended_drives_actor_exit_suspended() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_actor(
        "session-1",
        ApprovalActor {
            awaiting_decision: false,
            final_approved: None,
        },
        ActorConfig::default(),
        store.clone(),
        None,
    );

    h.send(Msg::Trigger).await.unwrap();
    let exit = h.join().await;

    match exit {
        ActorExit::Suspended { request } => {
            assert_eq!(request.prompt, "approve migration?");
            assert!(matches!(request.kind, SuspendKind::Approval));
            assert_eq!(request.blocked_tool.as_deref(), Some("run_migration"));
        }
        other => panic!("expected ActorExit::Suspended, got {other:?}"),
    }
}

#[tokio::test]
async fn suspended_exit_persists_snapshot_for_resume() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_actor(
        "session-2",
        ApprovalActor {
            awaiting_decision: false,
            final_approved: None,
        },
        ActorConfig::default(),
        store.clone(),
        None,
    );

    h.send(Msg::Trigger).await.unwrap();
    let _ = h.join().await;

    // Suspend exit must persist the snapshot (awaiting_decision = true).
    let bytes = store.load("session-2").await.expect("store load");
    let bytes = bytes.expect("snapshot present after suspend");
    let snap: Snap = serde_json::from_slice(&bytes).expect("snapshot parses");
    assert_eq!(
        snap,
        Snap {
            awaiting_decision: true,
            final_approved: None,
        }
    );
}

#[tokio::test]
async fn resume_after_suspend_delivers_decision() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());

    // 1. Spawn, trigger suspend, await exit.
    let h = spawn_actor(
        "session-3",
        ApprovalActor {
            awaiting_decision: false,
            final_approved: None,
        },
        ActorConfig::default(),
        store.clone(),
        None,
    );
    h.send(Msg::Trigger).await.unwrap();
    let _ = h.join().await;

    // 2. Resume from snapshot; verify awaiting_decision survived.
    let h2 = resume_actor(
        "session-3",
        || panic!("snapshot should exist; fresh constructor should not fire"),
        ApprovalActor::restore,
        ActorConfig::default(),
        store.clone(),
        None,
    )
    .await
    .expect("resume_actor");

    // 3. Deliver the decision; actor terminates cleanly.
    h2.send(Msg::Resume { approved: true }).await.unwrap();
    let exit = h2.join().await;
    assert!(matches!(exit, ActorExit::Terminal(_)));

    // 4. Store should no longer have a snapshot (delete_on_terminal).
    let bytes = store.load("session-3").await.unwrap();
    assert!(bytes.is_none(), "Terminal exit deletes snapshot");
}

#[derive(Debug)]
enum StreamMsg {
    Trigger,
}

#[derive(Serialize, Deserialize, Default, Debug, PartialEq, Eq)]
struct StreamSnap {
    awaiting_decision: bool,
}

struct StreamingApprovalActor {
    awaiting_decision: bool,
}

#[async_trait]
impl StreamingActorAgent for StreamingApprovalActor {
    type Message = StreamMsg;
    type Snapshot = StreamSnap;

    async fn handle_streaming(
        &mut self,
        msg: StreamMsg,
        _ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        match msg {
            StreamMsg::Trigger => {
                self.awaiting_decision = true;
                let ev = StreamEvent::TextDelta("about to migrate…".into());
                Ok((
                    Box::pin(stream::once(async move { ev })),
                    StreamingHandleOutcome::Suspended {
                        request: SuspendRequest::new("approve migration?", SuspendKind::Approval)
                            .with_blocked_tool("run_migration"),
                    },
                ))
            }
        }
    }

    fn checkpoint(&self) -> StreamSnap {
        StreamSnap {
            awaiting_decision: self.awaiting_decision,
        }
    }

    fn restore(snap: StreamSnap) -> Self {
        Self {
            awaiting_decision: snap.awaiting_decision,
        }
    }
}

#[tokio::test]
async fn streaming_suspended_outcome_drives_actor_exit_suspended() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_streaming_actor(
        "stream-session-1",
        StreamingApprovalActor {
            awaiting_decision: false,
        },
        ActorConfig::for_streaming(),
        store.clone(),
        None,
    );

    let mut turn = h.send(StreamMsg::Trigger).await.unwrap();
    while turn.next_event().await.is_some() {}

    let outcome = turn.outcome().await;
    match outcome {
        StreamingHandleOutcome::Suspended { request } => {
            assert_eq!(request.prompt, "approve migration?");
            assert_eq!(request.blocked_tool.as_deref(), Some("run_migration"));
        }
        other => panic!("expected Suspended outcome, got {other:?}"),
    }

    // Actor exits with ActorExit::Suspended after persisting.
    let exit = h.join().await;
    match exit {
        ActorExit::Suspended { request } => {
            assert_eq!(request.prompt, "approve migration?");
        }
        other => panic!("expected ActorExit::Suspended, got {other:?}"),
    }

    // Snapshot must persist for resume.
    let snap_bytes = store
        .load("stream-session-1")
        .await
        .unwrap()
        .expect("snapshot persisted after streaming suspend");
    let snap: StreamSnap = serde_json::from_slice(&snap_bytes).unwrap();
    assert_eq!(
        snap,
        StreamSnap {
            awaiting_decision: true
        }
    );
}
