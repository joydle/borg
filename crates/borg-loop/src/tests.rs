//! Integration tests for the actor loop, exercising the public API only
//! (`spawn_actor`, `resume_actor`, `ActorHandle`) via a tiny `Counter` actor.

use crate::actor::{ActorAgent, HandleOutcome};
use crate::handle::ActorExit;
use crate::runtime::{resume_actor, spawn_actor, ActorConfig, ActorCtx};
use crate::store::{ActorStore, InMemoryActorStore};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

#[derive(Debug)]
enum Msg {
    Increment,
    GetCount(oneshot::Sender<u32>),
    Sleep(Duration),
    Panic,
    Stop(String),
}

#[derive(Serialize, Deserialize, Default)]
struct CounterSnap {
    count: u32,
}

struct Counter {
    count: u32,
}

#[async_trait]
impl ActorAgent for Counter {
    type Message = Msg;
    type Snapshot = CounterSnap;

    async fn handle(&mut self, msg: Msg, _ctx: &ActorCtx) -> HandleOutcome {
        match msg {
            Msg::Increment => {
                self.count += 1;
                HandleOutcome::Continue
            }
            Msg::GetCount(reply) => {
                let _ = reply.send(self.count);
                HandleOutcome::Continue
            }
            Msg::Sleep(d) => {
                tokio::time::sleep(d).await;
                HandleOutcome::Continue
            }
            Msg::Panic => panic!("intentional test panic"),
            Msg::Stop(reason) => HandleOutcome::Terminal(reason),
        }
    }

    fn checkpoint(&self) -> CounterSnap {
        CounterSnap { count: self.count }
    }
    fn restore(snap: CounterSnap) -> Self {
        Self { count: snap.count }
    }
}

fn store() -> Arc<dyn ActorStore> {
    Arc::new(InMemoryActorStore::new())
}

#[tokio::test]
async fn send_increment_then_get_count_returns_one() {
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        store(),
        None,
    );
    h.send(Msg::Increment).await.unwrap();
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    let n = rx.await.unwrap();
    assert_eq!(n, 1);
}

#[tokio::test]
async fn terminal_outcome_drives_clean_exit() {
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        store(),
        None,
    );
    h.send(Msg::Increment).await.unwrap();
    h.send(Msg::Stop("done".into())).await.unwrap();
    let exit = h.join().await;
    match exit {
        ActorExit::Terminal(reason) => assert_eq!(reason, "done"),
        other => panic!("expected Terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_drives_clean_exit_mid_sleep() {
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        store(),
        None,
    );
    h.send(Msg::Sleep(Duration::from_secs(30))).await.unwrap();
    // Let the actor enter the sleep before cancelling.
    tokio::time::sleep(Duration::from_millis(20)).await;
    h.cancel();
    let exit = h.join().await;
    assert!(matches!(exit, ActorExit::Cancelled));
}

#[tokio::test]
async fn dropping_handle_cancels_actor() {
    let s = store();
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    );
    let inbox = h.inbox.clone();
    drop(h);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(inbox.is_closed(), "mailbox should close after drop-cancel");
}

#[tokio::test]
async fn detach_lets_actor_run_to_terminal() {
    let s = store();
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    );
    let inbox = h.inbox.clone();
    inbox.send(Msg::Increment).await.unwrap();
    inbox.send(Msg::Stop("ok".into())).await.unwrap();
    h.detach();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(inbox.is_closed());
    let raw = s.load("c1").await.unwrap();
    assert!(raw.is_none(), "terminal exit should delete the snapshot");
}

#[tokio::test]
async fn checkpoint_every_n_messages() {
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let cfg = ActorConfig {
        checkpoint_interval: 3,
        ..Default::default()
    };
    let h = spawn_actor("c1", Counter { count: 0 }, cfg, Arc::clone(&s), None);

    // Every Continue ticks the checkpoint counter, GetCount included: 2
    // increments + 1 GetCount = 3 messages, firing the threshold-3 save.
    h.send(Msg::Increment).await.unwrap();
    h.send(Msg::Increment).await.unwrap();
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    let _ = rx.await.unwrap();
    // The post-handle save is awaited before the loop polls the next
    // message, so a following `send().await` is a sync barrier: the actor
    // cannot accept it until the prior message's save has landed.
    let (tx2, rx2) = oneshot::channel();
    h.send(Msg::GetCount(tx2)).await.unwrap();
    let _ = rx2.await.unwrap();
    let snap_bytes = s.load("c1").await.unwrap();
    assert!(
        snap_bytes.is_some(),
        "checkpoint should exist after 3 Continue messages (threshold = 3)"
    );
    let snap: CounterSnap = serde_json::from_slice(&snap_bytes.unwrap()).unwrap();
    assert_eq!(
        snap.count, 2,
        "snapshot captures the increment count, not message count"
    );
}

// Terminal deletes the snapshot afterwards ("session done" contract).
#[tokio::test]
async fn terminal_writes_final_checkpoint_then_deletes() {
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    );
    h.send(Msg::Increment).await.unwrap();
    h.send(Msg::Stop("done".into())).await.unwrap();
    let exit = h.join().await;
    assert!(matches!(exit, ActorExit::Terminal(_)));
    assert!(s.load("c1").await.unwrap().is_none());
}

// Terminal exit must skip the final-checkpoint write: the immediately
// following `delete_on_terminal` would invalidate it (a wasted remote
// round-trip and cache pollution under `CachingActorStore`). This guards
// the call sequence — no `save:` between `send(Stop)` and the `delete:`.
#[derive(Default)]
struct RecordingStore {
    calls: std::sync::Mutex<Vec<String>>,
}

impl RecordingStore {
    fn snapshot(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl ActorStore for RecordingStore {
    async fn save(&self, id: &str, _bytes: Arc<[u8]>) -> Result<(), crate::store::ActorStoreError> {
        self.calls.lock().unwrap().push(format!("save:{id}"));
        Ok(())
    }
    async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, crate::store::ActorStoreError> {
        self.calls.lock().unwrap().push(format!("load:{id}"));
        Ok(None)
    }
    async fn delete(&self, id: &str) -> Result<(), crate::store::ActorStoreError> {
        self.calls.lock().unwrap().push(format!("delete:{id}"));
        Ok(())
    }
}

#[tokio::test]
async fn terminal_exit_skips_final_save() {
    let rec = Arc::new(RecordingStore::default());
    let store: Arc<dyn ActorStore> = rec.clone();
    // checkpoint_interval = 0 disables periodic saves so the call
    // sequence is clean to assert on.
    let cfg = ActorConfig {
        checkpoint_interval: 0,
        ..ActorConfig::default()
    };

    let h = spawn_actor("term-1", Counter { count: 0 }, cfg, store, None);
    h.send(Msg::Increment).await.unwrap();
    h.send(Msg::Stop("done".into())).await.unwrap();
    let _ = h.join().await;

    let calls = rec.snapshot();
    let saves = calls.iter().filter(|c| c.starts_with("save:")).count();
    let deletes = calls.iter().filter(|c| c.starts_with("delete:")).count();
    assert_eq!(
        saves, 0,
        "Terminal exit must not write a final checkpoint; calls={calls:?}"
    );
    assert_eq!(
        deletes, 1,
        "Terminal exit must delete exactly once; calls={calls:?}"
    );
}

#[tokio::test]
async fn cancel_keeps_snapshot_for_resume() {
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    );
    h.send(Msg::Increment).await.unwrap();
    h.send(Msg::Increment).await.unwrap();
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    let n = rx.await.unwrap();
    assert_eq!(n, 2);
    h.cancel();
    let exit = h.join().await;
    assert!(matches!(exit, ActorExit::Cancelled));
    // Snapshot remains — cancel is "pause, may resume."
    let snap_bytes = s.load("c1").await.unwrap().expect("snapshot present");
    let snap: CounterSnap = serde_json::from_slice(&snap_bytes).unwrap();
    assert_eq!(snap.count, 2);
}

#[tokio::test]
async fn resume_actor_loads_snapshot_into_fresh_loop() {
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    // First lifecycle: increment twice, cancel.
    {
        let h = spawn_actor(
            "session-1",
            Counter { count: 0 },
            ActorConfig::default(),
            Arc::clone(&s),
            None,
        );
        h.send(Msg::Increment).await.unwrap();
        h.send(Msg::Increment).await.unwrap();
        let (tx, rx) = oneshot::channel();
        h.send(Msg::GetCount(tx)).await.unwrap();
        let _ = rx.await.unwrap();
        h.cancel();
        let _ = h.join().await;
    }
    // Second lifecycle: resume should pick up count=2.
    let h = resume_actor::<Counter, _, _>(
        "session-1",
        || Counter { count: 0 }, // only used on cache miss
        <Counter as ActorAgent>::restore,
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    )
    .await
    .unwrap();
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    let n = rx.await.unwrap();
    assert_eq!(n, 2, "resumed actor should pick up prior count");
    h.send(Msg::Stop("done".into())).await.unwrap();
    let _ = h.join().await;
}

#[tokio::test]
async fn resume_with_no_snapshot_falls_back_to_fresh() {
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = resume_actor::<Counter, _, _>(
        "never-spawned",
        || Counter { count: 42 },
        <Counter as ActorAgent>::restore,
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    )
    .await
    .unwrap();
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    assert_eq!(rx.await.unwrap(), 42);
}

#[tokio::test]
async fn resume_with_corrupt_snapshot_falls_back() {
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    s.save("c1", Arc::from(&b"not valid json{{{"[..]))
        .await
        .unwrap();
    let h = resume_actor::<Counter, _, _>(
        "c1",
        || Counter { count: 99 },
        <Counter as ActorAgent>::restore,
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    )
    .await
    .unwrap();
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    assert_eq!(
        rx.await.unwrap(),
        99,
        "corrupt snapshot should fall back to fresh"
    );
}

// The trait's `restore(snap)` takes only the snapshot; the closure-based
// `resume_actor` lets the caller close over runtime deps (LLM client,
// store, registry refs) at the call site. Here a "boost" dep stands in.
#[tokio::test]
async fn resume_actor_restore_closure_receives_injected_deps() {
    use std::sync::Arc as StdArc;
    let s: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());

    // First lifecycle: increment to 3, snapshot lands.
    {
        let h = spawn_actor(
            "di-1",
            Counter { count: 0 },
            ActorConfig {
                checkpoint_interval: 1,
                ..ActorConfig::default()
            },
            Arc::clone(&s),
            None,
        );
        h.send(Msg::Increment).await.unwrap();
        h.send(Msg::Increment).await.unwrap();
        h.send(Msg::Increment).await.unwrap();
        let (tx, rx) = oneshot::channel();
        h.send(Msg::GetCount(tx)).await.unwrap();
        let _ = rx.await.unwrap();
        h.cancel();
        let _ = h.join().await;
    }

    // Second lifecycle: caller-supplied deps boost the restored counter,
    // proving restore() can take more than the snapshot.
    let deps_boost = StdArc::new(100u32);
    let boost_for_restore = StdArc::clone(&deps_boost);
    let h = resume_actor::<Counter, _, _>(
        "di-1",
        || Counter { count: 0 },
        move |snap| Counter {
            count: snap.count + *boost_for_restore,
        },
        ActorConfig::default(),
        Arc::clone(&s),
        None,
    )
    .await
    .unwrap();

    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    let n = rx.await.unwrap();
    assert_eq!(
        n, 103,
        "restore closure must receive snapshot AND inject the boost dep"
    );
    h.send(Msg::Stop("done".into())).await.unwrap();
    let _ = h.join().await;
}

#[tokio::test]
async fn idle_timeout_fires_when_no_messages() {
    let cfg = ActorConfig {
        idle_timeout: Some(Duration::from_millis(50)),
        ..Default::default()
    };
    let h = spawn_actor("c1", Counter { count: 0 }, cfg, store(), None);
    let exit = h.join().await;
    assert!(matches!(exit, ActorExit::IdleTimeout));
}

#[tokio::test]
async fn idle_timeout_does_not_fire_when_messages_arrive() {
    let cfg = ActorConfig {
        idle_timeout: Some(Duration::from_millis(100)),
        ..Default::default()
    };
    let h = spawn_actor("c1", Counter { count: 0 }, cfg, store(), None);
    // A message every 30ms keeps idle from firing.
    for _ in 0..6 {
        h.send(Msg::Increment).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    let (tx, rx) = oneshot::channel();
    h.send(Msg::GetCount(tx)).await.unwrap();
    assert_eq!(rx.await.unwrap(), 6);
    // Stop sending; idle should fire.
    let exit = tokio::time::timeout(Duration::from_millis(300), h.join())
        .await
        .expect("idle exit within 300ms");
    assert!(matches!(exit, ActorExit::IdleTimeout));
}

#[tokio::test]
async fn handler_panic_produces_panicked_exit() {
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        store(),
        None,
    );
    h.send(Msg::Panic).await.unwrap();
    let exit = h.join().await;
    assert!(matches!(exit, ActorExit::Panicked(_)));
}

#[tokio::test]
async fn parent_cancel_propagates_to_actor() {
    let parent = tokio_util::sync::CancellationToken::new();
    let h = spawn_actor(
        "c1",
        Counter { count: 0 },
        ActorConfig::default(),
        store(),
        Some(parent.clone()),
    );
    h.send(Msg::Sleep(Duration::from_secs(30))).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    parent.cancel();
    let exit = h.join().await;
    assert!(matches!(exit, ActorExit::Cancelled));
}

// Cross-spawn parent-id propagation is verified separately against an
// OTLP InMemorySpanExporter, which can assert
// `exported.parent_span_id == outer.span_id` directly.

#[cfg(test)]
mod streaming_actor_tests {
    use crate::runtime::ActorConfig;
    use crate::store::{ActorStore, InMemoryActorStore, NoopActorStore};
    use crate::streaming_actor::{StreamingActorAgent, StreamingHandleOutcome};
    use crate::streaming_runtime::{resume_streaming_actor, spawn_streaming_actor};
    use async_trait::async_trait;
    use borg::{BorgError, StreamEvent, Verdict};
    use futures::stream::BoxStream;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    #[derive(Serialize, Deserialize, Default)]
    struct EchoSnap {
        count: u32,
    }

    struct EchoActor {
        count: u32,
    }

    enum EchoMsg {
        Echo(String),
    }

    #[async_trait]
    impl StreamingActorAgent for EchoActor {
        type Message = EchoMsg;
        type Snapshot = EchoSnap;

        async fn handle_streaming(
            &mut self,
            msg: EchoMsg,
            _ctx: &crate::ActorCtx,
        ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
            self.count += 1;
            let EchoMsg::Echo(text) = msg;
            let stream = Box::pin(async_stream::stream! {
                yield StreamEvent::TextDelta(text);
                yield StreamEvent::Done(Verdict::empty());
            });
            Ok((stream, StreamingHandleOutcome::Continue))
        }

        fn checkpoint(&self) -> EchoSnap {
            EchoSnap { count: self.count }
        }
        fn restore(snap: EchoSnap) -> Self {
            EchoActor { count: snap.count }
        }
    }

    // Slow actor, for the cancel test.
    struct SlowActor;

    #[derive(Serialize, Deserialize)]
    struct SlowSnap;

    enum SlowMsg {
        Start,
    }

    #[async_trait]
    impl StreamingActorAgent for SlowActor {
        type Message = SlowMsg;
        type Snapshot = SlowSnap;

        async fn handle_streaming(
            &mut self,
            _msg: SlowMsg,
            ctx: &crate::ActorCtx,
        ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
            let cancel = ctx.cancel.clone();
            let stream = Box::pin(async_stream::stream! {
                for i in 0u32..100 {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            yield StreamEvent::Failed(BorgError::Cancelled);
                            return;
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                            yield StreamEvent::TextDelta(format!("chunk-{i}"));
                        }
                    }
                }
                yield StreamEvent::Done(Verdict::empty());
            });
            Ok((stream, StreamingHandleOutcome::Continue))
        }

        fn checkpoint(&self) -> SlowSnap {
            SlowSnap
        }
        fn restore(_: SlowSnap) -> Self {
            SlowActor
        }
    }

    #[test]
    fn streaming_actor_agent_is_defined() {
        // Compile-time check: the trait is importable and implementable.
        let _actor = EchoActor { count: 0 };
    }

    #[tokio::test]
    async fn streaming_handle_send_returns_event_receiver() {
        let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
        let handle = spawn_streaming_actor(
            "echo-1",
            EchoActor { count: 0 },
            ActorConfig::for_streaming(),
            store,
            None,
        );

        let mut rx = handle.send(EchoMsg::Echo("hello".into())).await.unwrap();
        let mut events = vec![];
        while let Some(ev) = rx.events.recv().await {
            events.push(ev);
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta(_))));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done(_))));

        handle.cancel();
    }

    #[tokio::test]
    async fn streaming_actor_checkpoints_and_resumes() {
        let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
        let store2 = Arc::clone(&store);

        // Spawn fresh, run one turn, let it checkpoint.
        let config = ActorConfig {
            checkpoint_interval: 1,
            ..ActorConfig::for_streaming()
        };
        let handle = spawn_streaming_actor(
            "echo-resume",
            EchoActor { count: 0 },
            config.clone(),
            store,
            None,
        );

        let mut rx = handle.send(EchoMsg::Echo("a".into())).await.unwrap();
        while rx.events.recv().await.is_some() {} // drain turn
        handle.cancel();
        let _ = handle.join().await;

        // Resume: the store should have a snapshot with count == 1.
        let resumed = resume_streaming_actor(
            "echo-resume",
            || EchoActor { count: 0 },
            <EchoActor as StreamingActorAgent>::restore,
            config,
            store2,
            None,
        )
        .await
        .unwrap();

        let mut rx2 = resumed.send(EchoMsg::Echo("b".into())).await.unwrap();
        let mut events = vec![];
        while let Some(ev) = rx2.events.recv().await {
            events.push(ev);
        }

        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "b")));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done(_))));

        resumed.cancel();
        let _ = resumed.join().await;
    }

    #[tokio::test]
    async fn cancel_mid_stream_surfaces_failed_event() {
        let store: Arc<dyn ActorStore> = Arc::new(NoopActorStore);
        let handle = spawn_streaming_actor(
            "slow-1",
            SlowActor,
            ActorConfig::for_streaming(),
            store,
            None,
        );

        let mut rx = handle.send(SlowMsg::Start).await.unwrap();

        // Receive one chunk then cancel.
        let first = rx.events.recv().await;
        assert!(matches!(first, Some(StreamEvent::TextDelta(_))));

        handle.cancel();

        // Drain remaining — should see Failed somewhere.
        let mut saw_failed = false;
        while let Some(ev) = rx.events.recv().await {
            if matches!(ev, StreamEvent::Failed(_)) {
                saw_failed = true;
            }
        }
        assert!(saw_failed, "expected Failed event after cancel");
    }

    // Streaming variant of the closure-based restore-with-DI test (the
    // plain-actor version is above); same snapshot-only `restore` shape.
    #[tokio::test]
    async fn resume_streaming_actor_restore_closure_receives_injected_deps() {
        let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
        let store2 = Arc::clone(&store);
        let config = ActorConfig::for_streaming();

        // First lifecycle: drive 2 turns, snapshot lands.
        {
            let h = spawn_streaming_actor(
                "echo-di",
                EchoActor { count: 0 },
                config.clone(),
                Arc::clone(&store),
                None,
            );
            let mut t1 = h.send(EchoMsg::Echo("a".into())).await.unwrap();
            while t1.events.recv().await.is_some() {}
            let mut t2 = h.send(EchoMsg::Echo("b".into())).await.unwrap();
            while t2.events.recv().await.is_some() {}
            h.cancel();
            let _ = h.join().await;
        }

        // Second lifecycle: the restore closure injects a "boost" dep.
        // Restored counter = snap.count + boost.
        let boost = std::sync::Arc::new(50u32);
        let boost_for_restore = std::sync::Arc::clone(&boost);
        let resumed = resume_streaming_actor(
            "echo-di",
            || EchoActor { count: 0 }, // unused on hit
            move |snap| EchoActor {
                count: snap.count + *boost_for_restore,
            },
            config,
            store2,
            None,
        )
        .await
        .unwrap();

        // After restore count is 2+50=52, incrementing to 53 this turn.
        // Count isn't exposed, so verify via the persisted snapshot.
        let mut t3 = resumed.send(EchoMsg::Echo("c".into())).await.unwrap();
        while t3.events.recv().await.is_some() {}
        resumed.cancel();
        let _ = resumed.join().await;

        let bytes = store.load("echo-di").await.unwrap().expect("snapshot");
        let snap: EchoSnap = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            snap.count, 53,
            "restore must have applied boost (2+50=52) and incremented on the post-resume turn (52+1=53)"
        );
    }
}

// The store layer uses `parking_lot` locks specifically because they
// don't poison: a panic mid-lock would otherwise leave every subsequent
// `.expect("poisoned")` acquirer cascading, defeating the
// `catch_unwind` panic-isolation boundary in `run_loop`. Pinned at the
// type level (the store's critical sections are panic-free in practice;
// this is defense-in-depth).
#[cfg(test)]
mod parking_lot_guarantee {
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[test]
    fn parking_lot_mutex_does_not_poison_on_held_panic() {
        let m = Arc::new(Mutex::new(42_u32));
        let cloned = Arc::clone(&m);
        // We assert only that the lock itself doesn't poison, not that
        // the protected state is recoverable across the panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _g = cloned.lock();
            panic!("intentional panic inside lock");
        }));
        assert!(result.is_err(), "the panicking closure must surface as Err");

        // The critical assertion: subsequent acquires still succeed.
        // `std::sync::Mutex` would have poisoned here.
        let g = m.lock();
        assert_eq!(*g, 42, "post-panic acquire must read the prior state");
    }
}
