//! Regression tests for `ActorHandle` lifecycle edges and `ActorConfig`
//! validation, through the public API only.

use async_trait::async_trait;
use borg_loop::{
    spawn_actor, ActorAgent, ActorConfig, ActorCtx, ActorExit, ActorStore, HandleOutcome,
    InMemoryActorStore,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

#[derive(Debug)]
enum Msg {
    Ping(oneshot::Sender<()>),
    /// A long call (an LLM turn) that stops early only on cancel.
    Work,
}

#[derive(Serialize, Deserialize, Default)]
struct Snap;

/// Test actor. `_alive` drops with the actor, so a test holding the
/// receiver learns when the loop has exited.
struct Probe {
    _alive: Option<oneshot::Sender<()>>,
}

#[async_trait]
impl ActorAgent for Probe {
    type Message = Msg;
    type Snapshot = Snap;

    async fn handle(&mut self, msg: Msg, ctx: &ActorCtx) -> HandleOutcome {
        match msg {
            Msg::Ping(reply) => {
                let _ = reply.send(());
                HandleOutcome::Continue
            }
            Msg::Work => {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {}
                    _ = ctx.cancel.cancelled() => {}
                }
                HandleOutcome::Continue
            }
        }
    }

    fn checkpoint(&self) -> Snap {
        Snap
    }

    fn restore(_snap: Snap) -> Self {
        Self { _alive: None }
    }
}

fn store() -> Arc<dyn ActorStore> {
    Arc::new(InMemoryActorStore::new())
}

#[tokio::test]
async fn join_abandoned_by_a_timeout_still_cancels_the_actor() {
    // Regression: `join` marked the handle detached before awaiting, so
    // when an outer timeout dropped the join future the handle's Drop
    // skipped the cancel and the actor ran on, leaked: its in-flight
    // call kept going until it finished on its own.
    let (alive_tx, alive_rx) = oneshot::channel();
    let h = spawn_actor(
        "join-timeout",
        Probe {
            _alive: Some(alive_tx),
        },
        ActorConfig::default(),
        store(),
        None,
    );
    h.send(Msg::Work).await.unwrap();
    let joined = tokio::time::timeout(Duration::from_millis(20), h.join()).await;
    assert!(joined.is_err(), "the busy actor does not exit on its own");
    // Dropping the abandoned join must have cancelled the actor.
    let exited = tokio::time::timeout(Duration::from_secs(2), alive_rx).await;
    assert!(
        exited.is_ok(),
        "actor leaked: still running after its handle was dropped"
    );
}

#[tokio::test]
async fn join_to_completion_does_not_cancel() {
    let h = spawn_actor(
        "join-complete",
        Probe { _alive: None },
        ActorConfig::default(),
        store(),
        None,
    );
    let (tx, rx) = oneshot::channel();
    h.send(Msg::Ping(tx)).await.unwrap();
    rx.await.unwrap();
    h.cancel();
    assert!(matches!(h.join().await, ActorExit::Cancelled));
}

#[tokio::test]
async fn zero_mailbox_capacity_is_treated_as_one() {
    // Regression: tokio's mpsc panics on a zero buffer.
    let mut config = ActorConfig::default();
    config.mailbox_capacity = 0;
    let h = spawn_actor(
        "zero-mailbox",
        Probe { _alive: None },
        config,
        store(),
        None,
    );
    let (tx, rx) = oneshot::channel();
    h.send(Msg::Ping(tx)).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), rx)
        .await
        .expect("message delivered")
        .unwrap();
    h.cancel();
    assert!(matches!(h.join().await, ActorExit::Cancelled));
}
