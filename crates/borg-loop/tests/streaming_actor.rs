//! Integration tests for `StreamingActorAgent` — exercised against the full
//! runtime loop (`spawn_streaming_actor` / `resume_streaming_actor`).
//!
//! These tests run as a separate compilation unit so they can import
//! `borg_loop` and `borg` as external crates exactly like a consumer would.

use async_trait::async_trait;
use borg::{BorgError, StreamEvent, Verdict};
use borg_loop::{
    resume_streaming_actor, spawn_streaming_actor, ActorConfig, ActorCtx, InMemoryActorStore,
    NoopActorStore, StreamingActorAgent, StreamingHandleOutcome,
};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Default, Debug)]
struct CounterSnap {
    count: u32,
}

/// Stateful actor: increments a counter on each turn, emits it as a TextDelta.
struct CounterActor {
    count: u32,
}

enum CounterMsg {
    Increment,
    Stop,
}

#[async_trait]
impl StreamingActorAgent for CounterActor {
    type Message = CounterMsg;
    type Snapshot = CounterSnap;

    async fn handle_streaming(
        &mut self,
        msg: CounterMsg,
        _ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        match msg {
            CounterMsg::Stop => {
                let stream = Box::pin(async_stream::stream! {
                    yield StreamEvent::Done(Verdict::empty());
                });
                Ok((stream, StreamingHandleOutcome::Terminal("stopped".into())))
            }
            CounterMsg::Increment => {
                self.count += 1;
                let count = self.count;
                let stream = Box::pin(async_stream::stream! {
                    yield StreamEvent::TextDelta(count.to_string());
                    yield StreamEvent::Done(Verdict::empty());
                });
                Ok((stream, StreamingHandleOutcome::Continue))
            }
        }
    }

    fn checkpoint(&self) -> CounterSnap {
        CounterSnap { count: self.count }
    }
    fn restore(snap: CounterSnap) -> Self {
        Self { count: snap.count }
    }
}

async fn drain(mut turn: borg_loop::TurnHandle) -> Vec<StreamEvent> {
    let mut events = vec![];
    while let Some(ev) = turn.next_event().await {
        events.push(ev);
    }
    events
}

#[tokio::test]
async fn single_turn_emits_text_delta_and_done() {
    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(NoopActorStore);
    let handle = spawn_streaming_actor(
        "counter-1",
        CounterActor { count: 0 },
        ActorConfig::for_streaming(),
        store,
        None,
    );

    let rx = handle.send(CounterMsg::Increment).await.unwrap();
    let events = drain(rx).await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "1")),
        "expected TextDelta(\"1\")"
    );
    assert!(
        events.iter().any(|e| matches!(e, StreamEvent::Done(_))),
        "expected Done"
    );

    handle.cancel();
    handle.join().await;
}

#[tokio::test]
async fn multiple_turns_accumulate_state() {
    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(NoopActorStore);
    let handle = spawn_streaming_actor(
        "counter-2",
        CounterActor { count: 0 },
        ActorConfig::for_streaming(),
        store,
        None,
    );

    for expected in 1u32..=5 {
        let rx = handle.send(CounterMsg::Increment).await.unwrap();
        let events = drain(rx).await;
        let text: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::TextDelta(t) = e {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, [expected.to_string()], "turn {expected}");
    }

    handle.cancel();
    handle.join().await;
}

#[tokio::test]
async fn terminal_outcome_exits_loop() {
    use borg_loop::ActorExit;

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(NoopActorStore);
    let handle = spawn_streaming_actor(
        "counter-stop",
        CounterActor { count: 0 },
        ActorConfig::for_streaming(),
        store,
        None,
    );

    let rx = handle.send(CounterMsg::Stop).await.unwrap();
    drain(rx).await;

    let exit = handle.join().await;
    assert!(
        matches!(exit, ActorExit::Terminal(ref r) if r == "stopped"),
        "expected Terminal(stopped), got {exit:?}"
    );
}

#[tokio::test]
async fn checkpoint_and_resume_restores_count() {
    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(InMemoryActorStore::new());
    let store2 = Arc::clone(&store);

    // First actor: 3 turns, then cancel (checkpoint on exit).
    let mut config = ActorConfig::for_streaming();
    config.checkpoint_interval = 1;
    let h1 = spawn_streaming_actor(
        "counter-resume",
        CounterActor { count: 0 },
        config.clone(),
        store,
        None,
    );
    for _ in 0..3 {
        let rx = h1.send(CounterMsg::Increment).await.unwrap();
        drain(rx).await;
    }
    h1.cancel();
    h1.join().await;

    // Second actor: resume from snapshot, count should be 3.
    let h2 = resume_streaming_actor(
        "counter-resume",
        || CounterActor { count: 0 },
        <CounterActor as borg_loop::StreamingActorAgent>::restore,
        config,
        store2,
        None,
    )
    .await
    .unwrap();

    let rx = h2.send(CounterMsg::Increment).await.unwrap();
    let events = drain(rx).await;
    // Resumed at count=3, one increment → 4.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "4")),
        "expected TextDelta(\"4\") after resuming at count=3"
    );

    h2.cancel();
    h2.join().await;
}

#[tokio::test]
async fn cancel_token_stops_actor_between_messages() {
    use borg_loop::ActorExit;

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(NoopActorStore);
    let handle = spawn_streaming_actor(
        "counter-cancel",
        CounterActor { count: 0 },
        ActorConfig::for_streaming(),
        store,
        None,
    );

    let rx = handle.send(CounterMsg::Increment).await.unwrap();
    drain(rx).await;

    // Cancel between messages.
    handle.cancel();
    let exit = handle.join().await;
    assert!(
        matches!(exit, ActorExit::Cancelled),
        "expected Cancelled, got {exit:?}"
    );
}
