//! Human-in-the-loop: an actor pauses for an approval decision, persists
//! its state, then resumes once a human answers.
//!
//! The flow:
//!   1. The actor receives work that needs sign-off and returns
//!      `HandleOutcome::Suspended`, carrying a `SuspendRequest`.
//!   2. The runtime persists a snapshot and exits with
//!      `ActorExit::Suspended { request }`.
//!   3. The "human" inspects the request and makes a `Decision`.
//!   4. `resume_actor` rehydrates the actor from its snapshot, and the
//!      decision is delivered as the next message.
//!
//! Run with: `cargo run --example human_in_the_loop -p borg-loop`

use async_trait::async_trait;
use borg::suspend::{Decision, SuspendKind, SuspendRequest};
use borg_loop::{
    resume_actor, spawn_actor, ActorAgent, ActorConfig, ActorCtx, ActorExit, ActorStore,
    HandleOutcome, InMemoryActorStore,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

enum Msg {
    /// Ask the actor to do something that needs approval first.
    Deploy { release: String },
    /// The human's answer, delivered after resume.
    Decision(Decision),
}

/// Persisted state: what we were asked to deploy, and the final outcome.
#[derive(Serialize, Deserialize, Default)]
struct Snap {
    pending_release: Option<String>,
    deployed: Option<String>,
}

struct Deployer {
    pending_release: Option<String>,
    deployed: Option<String>,
}

#[async_trait]
impl ActorAgent for Deployer {
    type Message = Msg;
    type Snapshot = Snap;

    async fn handle(&mut self, msg: Msg, _ctx: &ActorCtx) -> HandleOutcome {
        match msg {
            Msg::Deploy { release } => {
                self.pending_release = Some(release.clone());
                // Pause for a human to approve the deploy.
                HandleOutcome::Suspended {
                    request: SuspendRequest::new(
                        format!("Deploy release {release} to production?"),
                        SuspendKind::Approval,
                    )
                    .with_blocked_tool("deploy"),
                }
            }
            Msg::Decision(Decision::Approve) => {
                let release = self.pending_release.take().unwrap_or_default();
                self.deployed = Some(release.clone());
                HandleOutcome::Terminal(format!("deployed {release}"))
            }
            Msg::Decision(Decision::Reject { reason }) => {
                self.pending_release = None;
                HandleOutcome::Terminal(format!("deploy rejected: {reason}"))
            }
            Msg::Decision(_) => HandleOutcome::Terminal("unsupported decision".into()),
        }
    }

    fn checkpoint(&self) -> Snap {
        Snap {
            pending_release: self.pending_release.clone(),
            deployed: self.deployed.clone(),
        }
    }

    fn restore(snap: Snap) -> Self {
        Self {
            pending_release: snap.pending_release,
            deployed: snap.deployed,
        }
    }
}

#[tokio::main]
async fn main() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let session = "deploy-session";

    // 1. Spawn the actor and ask it to deploy.
    let handle = spawn_actor(
        session,
        Deployer {
            pending_release: None,
            deployed: None,
        },
        ActorConfig::default(),
        store.clone(),
        None,
    );
    handle
        .send(Msg::Deploy {
            release: "v1.0.0".into(),
        })
        .await
        .unwrap();

    // 2. The actor suspends; its snapshot is now persisted.
    let request = match handle.join().await {
        ActorExit::Suspended { request } => request,
        other => panic!("expected Suspended, got {other:?}"),
    };
    println!("⏸  paused for approval: {}", request.prompt);
    println!("   blocked tool: {:?}", request.blocked_tool);

    // 3. A human decides. (In a real app this comes from a UI / API call.)
    let decision = Decision::Approve;
    println!("✅ human approved");

    // 4. Resume from the snapshot and deliver the decision.
    let resumed = resume_actor(
        session,
        || panic!("snapshot should exist after suspend"),
        Deployer::restore,
        ActorConfig::default(),
        store.clone(),
        None,
    )
    .await
    .expect("resume from snapshot");

    resumed.send(Msg::Decision(decision)).await.unwrap();
    match resumed.join().await {
        ActorExit::Terminal(outcome) => println!("🏁 {outcome}"),
        other => panic!("expected Terminal, got {other:?}"),
    }
}
