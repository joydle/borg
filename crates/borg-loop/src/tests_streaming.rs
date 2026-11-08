//! Tests for new borg-loop primitives:
//! `Park` / `TurnHandle`, `ResilientActorStore`, `ActorRegistry`, `run_session_loop`.

use crate::driver::SessionOutcome;
use crate::registry::ActorRegistry;
use crate::runtime::ActorConfig;
use crate::store::{ActorStore, ActorStoreError, InMemoryActorStore, ResilientActorStore};
use crate::streaming_actor::{StreamingActorAgent, StreamingHandleOutcome};
use crate::streaming_runtime::spawn_streaming_actor;
use async_trait::async_trait;
use borg::{BorgError, StreamEvent};
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug)]
enum Msg {
    /// Emit one TextDelta and return Continue.
    Ping(String),
    /// Emit one TextDelta and return Park.
    Done(String),
    /// Emit one TextDelta and return Terminal.
    Stop(String),
    /// Return an error from handle_streaming (surfaces `Failed`, then Terminal).
    Fail,
    /// Emit many TextDeltas with a sleep between each, then Continue.
    SlowPings(u32),
}

#[derive(Serialize, Deserialize, Default)]
struct EchoSnap {
    turns: u32,
}

#[derive(Default)]
struct EchoActor {
    turns: u32,
}

#[async_trait]
impl StreamingActorAgent for EchoActor {
    type Message = Msg;
    type Snapshot = EchoSnap;

    async fn handle_streaming(
        &mut self,
        msg: Msg,
        _ctx: &crate::runtime::ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        self.turns += 1;
        match msg {
            Msg::Ping(text) => {
                let ev = StreamEvent::TextDelta(text);
                Ok((
                    Box::pin(stream::once(async move { ev })),
                    StreamingHandleOutcome::Continue,
                ))
            }
            Msg::Done(text) => {
                let ev = StreamEvent::TextDelta(text);
                Ok((
                    Box::pin(stream::once(async move { ev })),
                    StreamingHandleOutcome::Park,
                ))
            }
            Msg::Stop(text) => {
                let ev = StreamEvent::TextDelta(text);
                Ok((
                    Box::pin(stream::once(async move { ev })),
                    StreamingHandleOutcome::Terminal("stopped".into()),
                ))
            }
            Msg::Fail => Err(BorgError::Cancelled),
            Msg::SlowPings(n) => {
                let stream = async_stream::stream! {
                    for i in 0..n {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        yield StreamEvent::TextDelta(format!("tick-{i}"));
                    }
                };
                Ok((Box::pin(stream), StreamingHandleOutcome::Continue))
            }
        }
    }

    fn checkpoint(&self) -> EchoSnap {
        EchoSnap { turns: self.turns }
    }
    fn restore(snap: EchoSnap) -> Self {
        Self { turns: snap.turns }
    }
}

fn make_handle() -> crate::streaming_actor::StreamingActorHandle<EchoActor> {
    spawn_streaming_actor(
        "echo-1",
        EchoActor::default(),
        ActorConfig::for_streaming(),
        Arc::new(InMemoryActorStore::new()),
        None,
    )
}

#[tokio::test]
async fn turn_handle_events_and_outcome_continue() {
    let h = make_handle();
    let mut turn = h.send(Msg::Ping("hello".into())).await.unwrap();

    let mut events = vec![];
    while let Some(ev) = turn.next_event().await {
        events.push(ev);
    }
    assert!(matches!(events[0], StreamEvent::TextDelta(ref t) if t == "hello"));

    let outcome = turn.outcome().await;
    assert_eq!(outcome, StreamingHandleOutcome::Continue);
}

#[tokio::test]
async fn turn_handle_outcome_park() {
    let h = make_handle();
    let mut turn = h.send(Msg::Done("bye".into())).await.unwrap();

    while turn.next_event().await.is_some() {}
    let outcome = turn.outcome().await;
    assert_eq!(outcome, StreamingHandleOutcome::Park);
}

#[tokio::test]
async fn turn_handle_outcome_terminal() {
    let h = make_handle();
    let mut turn = h.send(Msg::Stop("done".into())).await.unwrap();

    while turn.next_event().await.is_some() {}
    let outcome = turn.outcome().await;
    assert_eq!(outcome, StreamingHandleOutcome::Terminal("stopped".into()));
}

#[tokio::test]
async fn park_keeps_actor_alive_for_next_message() {
    let h = make_handle();

    let mut t1 = h.send(Msg::Done("turn1".into())).await.unwrap();
    while t1.next_event().await.is_some() {}
    assert_eq!(t1.outcome().await, StreamingHandleOutcome::Park);

    // Park keeps the actor alive for the next message.
    assert!(!h.is_closed());
    let mut t2 = h.send(Msg::Ping("turn2".into())).await.unwrap();
    while t2.next_event().await.is_some() {}
    assert_eq!(t2.outcome().await, StreamingHandleOutcome::Continue);
}

#[tokio::test]
async fn handle_streaming_error_surfaces_failed_event_and_terminates() {
    // When handle_streaming returns Err, the runtime surfaces a Failed
    // event then exits with Terminal("handle_streaming error").
    use borg::BorgError;
    let h = make_handle();

    let mut turn = h.send(Msg::Fail).await.unwrap();
    let mut saw_failed = false;
    while let Some(ev) = turn.next_event().await {
        if matches!(ev, StreamEvent::Failed(BorgError::Cancelled)) {
            saw_failed = true;
        }
    }
    assert!(saw_failed, "expected a Failed event surfacing the error");

    let outcome = turn.outcome().await;
    assert_eq!(
        outcome,
        StreamingHandleOutcome::Terminal("handle_streaming error".into())
    );

    let exit = h.join().await;
    assert!(
        matches!(exit, crate::handle::ActorExit::Terminal(ref s) if s == "handle_streaming error"),
        "expected Terminal exit with handle_streaming error reason, got {exit:?}"
    );
}

/// A store that always fails after a configurable number of successes.
struct FailingStore {
    fails_after: std::sync::Mutex<usize>,
}

impl FailingStore {
    fn new(fails_after: usize) -> Self {
        Self {
            fails_after: std::sync::Mutex::new(fails_after),
        }
    }
}

#[async_trait]
impl ActorStore for FailingStore {
    async fn save(&self, _id: &str, _bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        let mut n = self.fails_after.lock().unwrap();
        if *n == 0 {
            return Err(ActorStoreError::Backend("injected failure".into()));
        }
        *n -= 1;
        Ok(())
    }
    async fn load(&self, _id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        Err(ActorStoreError::Backend("injected failure".into()))
    }
    async fn delete(&self, _id: &str) -> Result<(), ActorStoreError> {
        Ok(())
    }
}

/// A store whose ops always hang (simulates TCP stall).
struct HangingStore;

#[async_trait]
impl ActorStore for HangingStore {
    async fn save(&self, _id: &str, _bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        tokio::time::sleep(Duration::from_secs(9999)).await;
        unreachable!()
    }
    async fn load(&self, _id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        tokio::time::sleep(Duration::from_secs(9999)).await;
        unreachable!()
    }
    async fn delete(&self, _id: &str) -> Result<(), ActorStoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn fallback_store_save_load_roundtrip_on_primary() {
    let primary = InMemoryActorStore::new();
    let fallback = InMemoryActorStore::new();
    let store = ResilientActorStore::new(primary, fallback);

    store.save("k", Arc::from(&b"hello"[..])).await.unwrap();
    let loaded = store.load("k").await.unwrap();
    assert_eq!(loaded.as_deref(), Some(&b"hello"[..]));
    assert!(!store.is_degraded());
}

#[tokio::test]
async fn fallback_store_degrades_on_primary_error() {
    let primary = FailingStore::new(0); // fails immediately
    let fallback = InMemoryActorStore::new();
    let store = ResilientActorStore::new(primary, fallback);

    // Primary fails → write goes to fallback, store degrades.
    store.save("k", Arc::from(&b"v1"[..])).await.unwrap();
    assert!(store.is_degraded());

    let loaded = store.load("k").await.unwrap();
    assert_eq!(loaded.as_deref(), Some(&b"v1"[..]));
}

#[tokio::test(start_paused = true)]
async fn fallback_store_degrades_on_primary_timeout() {
    let primary = HangingStore;
    let fallback = InMemoryActorStore::new();
    let store = ResilientActorStore::with_timeout(primary, fallback, Duration::from_millis(50));

    // Primary hangs → times out → falls back.
    store.save("k", Arc::from(&b"v2"[..])).await.unwrap();
    assert!(store.is_degraded());

    let loaded = store.load("k").await.unwrap();
    assert_eq!(loaded.as_deref(), Some(&b"v2"[..]));
}

#[tokio::test]
async fn fallback_store_delete_is_best_effort() {
    let primary = InMemoryActorStore::new();
    let fallback = InMemoryActorStore::new();
    let store = ResilientActorStore::new(primary, fallback);

    store.save("k", Arc::from(&b"x"[..])).await.unwrap();
    store.delete("k").await.unwrap(); // best-effort: must not panic
}

#[tokio::test]
async fn registry_insert_and_get_parked() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    let h = make_handle();

    reg.insert("s1", h);
    assert!(reg.get_parked("s1").is_none(), "not parked yet");

    reg.mark_parked("s1");
    assert!(reg.get_parked("s1").is_some(), "should be parked now");
}

#[tokio::test]
async fn registry_mark_running_clears_parked() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    reg.insert("s1", make_handle());
    reg.mark_parked("s1");
    assert!(reg.get_parked("s1").is_some());

    reg.mark_running("s1");
    assert!(reg.get_parked("s1").is_none());
}

#[tokio::test]
async fn registry_remove_evicts_entry() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    reg.insert("s1", make_handle());
    reg.mark_parked("s1");
    reg.remove("s1");

    assert!(reg.get_parked("s1").is_none());
    assert_eq!(reg.len(), 0);
}

#[tokio::test]
async fn registry_get_parked_returns_none_for_closed_actor() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    let h = make_handle();
    // Cancel and let the task exit so the handle is closed when inserted.
    h.cancel();
    tokio::time::sleep(Duration::from_millis(10)).await;
    reg.insert("s1", h);
    reg.mark_parked("s1");

    // get_parked filters out closed handles.
    assert!(reg.get_parked("s1").is_none());
}

#[tokio::test]
async fn registry_claim_parked_flips_to_running() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    reg.insert("s1", make_handle());
    reg.mark_parked("s1");

    // First claim succeeds and flips parked → running.
    assert!(reg.claim_parked("s1").is_some());
    // Second claim returns None because the entry is no longer parked.
    assert!(reg.claim_parked("s1").is_none());
    assert!(
        reg.get_parked("s1").is_none(),
        "claim_parked must have cleared the parked flag"
    );
}

#[tokio::test]
async fn registry_claim_parked_returns_none_when_not_parked() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    reg.insert("s1", make_handle());
    // Never marked parked.
    assert!(reg.claim_parked("s1").is_none());
}

#[tokio::test]
async fn registry_claim_parked_returns_none_when_absent() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    assert!(reg.claim_parked("nope").is_none());
}

#[tokio::test]
async fn registry_claim_parked_returns_none_for_closed_actor() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    let h = make_handle();
    h.cancel();
    tokio::time::sleep(Duration::from_millis(10)).await;
    reg.insert("s1", h);
    reg.mark_parked("s1");

    assert!(reg.claim_parked("s1").is_none());
}

#[tokio::test]
async fn registry_claim_parked_is_atomic_under_concurrent_callers() {
    let reg: Arc<ActorRegistry<EchoActor>> = Arc::new(ActorRegistry::new());
    reg.insert("s1", make_handle());
    reg.mark_parked("s1");

    // 32 concurrent claim_parked tasks race for the same id; the
    // registry's internal lock guarantees exactly one wins the flip.
    let mut joins = Vec::new();
    for _ in 0..32 {
        let reg = Arc::clone(&reg);
        joins.push(tokio::spawn(
            async move { reg.claim_parked("s1").is_some() },
        ));
    }
    let mut winners = 0;
    for j in joins {
        if j.await.unwrap() {
            winners += 1;
        }
    }
    assert_eq!(winners, 1, "exactly one claim_parked must win");
}

/// The idle reaper's sweep: an actor that exited on its own (here a
/// cancel; in production an idle timeout) leaves, a live parked one —
/// whose inbox is open — stays.
#[tokio::test]
async fn registry_reap_closed_drops_exited_actors_and_keeps_parked_ones() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    let gone = make_handle();
    gone.cancel();
    tokio::time::sleep(Duration::from_millis(10)).await;
    reg.insert("exited", gone);
    reg.insert_parked("resident", make_handle());
    reg.insert("mid-turn", make_handle());

    assert_eq!(reg.reap_closed(), 1);
    assert!(!reg.contains("exited"));
    assert!(
        reg.claim_parked("resident").is_some(),
        "a parked actor survives the sweep"
    );
    assert!(
        reg.contains("mid-turn"),
        "a running actor survives the sweep"
    );
}

#[tokio::test]
async fn registry_retain_sees_every_id_and_counts_removals() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    for id in ["a", "b", "c"] {
        reg.insert(id, make_handle());
    }
    let mut seen = Vec::new();
    let removed = reg.retain(|id, _| {
        seen.push(id.to_string());
        id != "b"
    });
    seen.sort();
    assert_eq!(seen, ["a", "b", "c"]);
    assert_eq!(removed, 1);
    assert_eq!(reg.len(), 2);
    assert!(!reg.contains("b"));
}

/// "An old worker never removes a newer entry": the run that lost its
/// session asks about ITS handle, and a newer actor registered under the
/// same id in the meantime stays.
#[tokio::test]
async fn registry_remove_if_spares_a_newer_entry_under_the_same_id() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    let old = Arc::new(make_handle());
    reg.insert("chat", Arc::clone(&old));
    let newer = Arc::new(make_handle());
    reg.insert("chat", Arc::clone(&newer));

    assert!(!reg.remove_if("chat", |h| Arc::ptr_eq(h, &old)));
    let kept = reg.get("chat").expect("the newer entry stays");
    assert!(Arc::ptr_eq(&kept, &newer));

    assert!(reg.remove_if("chat", |h| Arc::ptr_eq(h, &newer)));
    assert!(!reg.contains("chat"));
    assert!(
        !reg.remove_if("chat", |_| true),
        "absent ids remove nothing"
    );
}

/// The same `Arc` the driver held while the turn ran is the one the
/// registry hands back — what makes `remove_if(.., ptr_eq)` work.
#[tokio::test]
async fn registry_insert_takes_a_shared_handle_without_rewrapping_it() {
    let reg: ActorRegistry<EchoActor> = ActorRegistry::new();
    let mine = Arc::new(make_handle());
    reg.insert_parked("s1", Arc::clone(&mine));
    let claimed = reg.claim_parked("s1").expect("parked at insert");
    assert!(Arc::ptr_eq(&claimed, &mine));
}

/// A parked actor's snapshot is on the store after the turn that parked
/// it (checkpoint interval 1), and a send to an actor that has exited
/// hands the message back — together, what a rehydrate-on-failed-send
/// path needs: the message to resend and a snapshot to resend it to.
#[tokio::test]
async fn park_checkpoints_and_a_failed_send_returns_the_message() {
    let store = Arc::new(InMemoryActorStore::new());
    let mut cfg = ActorConfig::for_streaming();
    cfg.checkpoint_interval = 1;
    let h = spawn_streaming_actor(
        "parks-1",
        EchoActor::default(),
        cfg,
        Arc::clone(&store) as Arc<dyn ActorStore>,
        None,
    );
    let outcome = h
        .run_session_loop(Msg::Ping("a".into()), || Msg::Done("b".into()), |_| {})
        .await;
    assert_eq!(outcome, SessionOutcome::Parked);
    let saved = store.load("parks-1").await.unwrap().expect("park saved");
    let snap: EchoSnap = serde_json::from_slice(&saved).unwrap();
    assert_eq!(snap.turns, 2, "both turns are in the parked snapshot");

    h.cancel();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(h.is_closed());
    match h.send(Msg::Done("again".into())).await {
        Err(tokio::sync::mpsc::error::SendError(Msg::Done(t))) => assert_eq!(t, "again"),
        Err(other) => panic!("wrong message handed back: {:?}", other.0),
        Ok(_) => panic!("an exited actor accepted a message"),
    }
}

#[tokio::test]
async fn session_loop_continue_then_park_returns_parked() {
    let h = make_handle();
    let mut collected_events: Vec<String> = vec![];

    let outcome = h
        .run_session_loop(
            Msg::Ping("t1".into()),    // initial — returns Continue
            || Msg::Done("t2".into()), // continuation — returns Park
            |ev| {
                if let StreamEvent::TextDelta(t) = ev {
                    collected_events.push(t);
                }
            },
        )
        .await;

    assert_eq!(outcome, SessionOutcome::Parked);
    assert_eq!(collected_events, vec!["t1", "t2"]);
}

#[tokio::test]
async fn session_loop_terminal_returns_terminal() {
    let h = make_handle();

    let outcome = h
        .run_session_loop(
            Msg::Stop("bye".into()),
            || unreachable!("should not call continuation after Terminal"),
            |_| {},
        )
        .await;

    assert_eq!(outcome, SessionOutcome::Terminal);
}

#[tokio::test]
async fn session_loop_multiple_continue_then_park() {
    let h = make_handle();
    let mut turn = 0u32;

    let outcome = h
        .run_session_loop(
            Msg::Ping("0".into()),
            || {
                turn += 1;
                if turn < 3 {
                    Msg::Ping(turn.to_string())
                } else {
                    Msg::Done("done".into())
                }
            },
            |_| {},
        )
        .await;

    assert_eq!(outcome, SessionOutcome::Parked);
}

#[tokio::test]
async fn session_loop_actor_gone_returns_terminal() {
    let h = make_handle();
    // Force the actor to exit first, then let the runtime clean up.
    let mut t = h.send(Msg::Stop("pre-exit".into())).await.unwrap();
    while t.next_event().await.is_some() {}
    drop(t);
    tokio::time::sleep(Duration::from_millis(10)).await;

    let outcome = h
        .run_session_loop(
            Msg::Ping("ignored".into()),
            || Msg::Ping("x".into()),
            |_| {},
        )
        .await;
    assert_eq!(outcome, SessionOutcome::Terminal);
}

#[tokio::test]
async fn turn_handle_accessor_methods() {
    let h = make_handle();
    let mut turn = h.send(Msg::Ping("hi".into())).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _outcome = turn.outcome().await;
    h.cancel();
}

#[tokio::test]
async fn turn_handle_drop_mid_stream_parks_actor_alive() {
    // Dropping a TurnHandle mid-stream must not keep the runtime
    // streaming into a dead receiver; the actor parks and returns to
    // inbox.recv(), ready for the next message.
    let h = make_handle();

    // Start a long turn, then drop the handle before draining it.
    let turn = h.send(Msg::SlowPings(50)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    drop(turn);

    // The actor detects the dropped handle on its next event send and
    // returns to inbox.recv(). A follow-up message must be accepted.
    let timeout = tokio::time::Duration::from_millis(500);
    let send_fut = h.send(Msg::Done("after-drop".into()));
    let mut turn2 = tokio::time::timeout(timeout, send_fut)
        .await
        .expect("actor should accept the next message after handle drop")
        .expect("send must succeed");

    let mut events = vec![];
    while let Some(ev) = turn2.next_event().await {
        events.push(ev);
    }
    assert!(
        matches!(events.last(), Some(StreamEvent::TextDelta(t)) if t == "after-drop"),
        "actor must have produced the after-drop turn's event"
    );
    let outcome = turn2.outcome().await;
    assert_eq!(outcome, StreamingHandleOutcome::Park);

    h.cancel();
}

/// Records every save/delete call in order so tests can assert the
/// runtime made the right number of round-trips at exit.
#[derive(Default)]
struct RecordingStore {
    calls: Mutex<Vec<String>>,
}

impl RecordingStore {
    fn snapshot(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl ActorStore for RecordingStore {
    async fn save(&self, id: &str, _bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        self.calls.lock().unwrap().push(format!("save:{id}"));
        Ok(())
    }
    async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        self.calls.lock().unwrap().push(format!("load:{id}"));
        Ok(None)
    }
    async fn delete(&self, id: &str) -> Result<(), ActorStoreError> {
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
    let mut cfg = ActorConfig::for_streaming();
    cfg.checkpoint_interval = 0;

    let h = spawn_streaming_actor("term-1", EchoActor::default(), cfg, store, None);
    let mut turn = h.send(Msg::Stop("done".into())).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;
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
async fn streaming_handle_drop_cancels_actor() {
    // Dropping a (non-Arc'd) StreamingActorHandle fires the cancel token,
    // mirroring ActorHandle's drop semantics.
    let h = make_handle();
    let cancel = h.cancel.clone();
    drop(h);
    assert!(cancel.is_cancelled(), "Drop must cancel the actor's token");
}

#[tokio::test]
async fn send_returns_unsent_message_on_closed_inbox() {
    // When the inbox is closed (actor exited), `send` surfaces the
    // original message via `SendError<A::Message>` so the caller can
    // recover it and retry against a respawned actor without re-acquiring
    // per-message resources (e.g. a pre-opened LLM stream).
    let h = make_handle();

    // Drive a Terminal exit, then wait for the inbox to close.
    let mut turn = h.send(Msg::Stop("done".into())).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !h.is_closed() && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(h.is_closed(), "actor inbox should be closed after Terminal");

    // The message must bounce back inside the error.
    match h.send(Msg::Ping("recover-me".into())).await {
        Ok(_) => panic!("send to a closed inbox must fail"),
        Err(tokio::sync::mpsc::error::SendError(Msg::Ping(text))) => {
            assert_eq!(text, "recover-me");
        }
        Err(tokio::sync::mpsc::error::SendError(other)) => {
            panic!("SendError must carry the original message; got {other:?}");
        }
    }
}

#[tokio::test]
async fn streaming_handle_arc_keeps_actor_alive_until_last_drop() {
    // Under Arc-wrapping (e.g. ActorRegistry) the actor stays alive while
    // any clone is held; only the last drop fires cancel-on-drop.
    let h = Arc::new(make_handle());
    let cancel = h.cancel.clone();
    let h2 = Arc::clone(&h);

    drop(h);
    assert!(
        !cancel.is_cancelled(),
        "cancel must NOT fire while another Arc owner exists"
    );

    drop(h2);
    assert!(
        cancel.is_cancelled(),
        "last Arc drop must fire cancel-on-drop"
    );
}

#[tokio::test]
async fn event_channel_capacity_honored() {
    // event_channel_capacity must be threaded into the per-turn mpsc
    // channel rather than hardcoded.
    let mut cfg = ActorConfig::for_streaming();
    cfg.event_channel_capacity = 4;
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_streaming_actor("ec-1", EchoActor::default(), cfg, store, None);
    // Capacity is private; probe indirectly — a SlowPings(8) burst with
    // capacity=4 must still drain end-to-end (runtime backpressures send).
    let mut turn = h.send(Msg::SlowPings(8)).await.unwrap();
    let mut n = 0;
    while let Some(_ev) = turn.next_event().await {
        n += 1;
    }
    assert_eq!(n, 8);
    assert_eq!(turn.outcome().await, StreamingHandleOutcome::Continue);
    h.cancel();
}

// Cancellation latency contract (negative test): the streaming runtime
// does NOT race `handle_streaming`'s `.await` against `ctx.cancel`. A long
// uninterruptible `.await` in the agent body therefore completes naturally
// even if cancel fires mid-turn. If the runtime ever wraps
// `handle_streaming` in a `select!` against cancel, replace this with the
// positive latency test.
#[derive(Default)]
struct UncooperativeAgent;

#[async_trait]
impl StreamingActorAgent for UncooperativeAgent {
    type Message = std::time::Duration;
    type Snapshot = EchoSnap;

    async fn handle_streaming(
        &mut self,
        sleep_for: std::time::Duration,
        _ctx: &crate::runtime::ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        // Deliberately no `select!` against `_ctx.cancel`.
        tokio::time::sleep(sleep_for).await;
        Ok((
            Box::pin(stream::empty::<StreamEvent>()),
            StreamingHandleOutcome::Terminal("done".into()),
        ))
    }

    fn checkpoint(&self) -> EchoSnap {
        EchoSnap::default()
    }
    fn restore(_: EchoSnap) -> Self {
        Self
    }
}

#[tokio::test]
async fn cancel_does_not_shorten_uncooperative_handle_streaming() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_streaming_actor(
        "uncoop-1",
        UncooperativeAgent,
        ActorConfig::for_streaming(),
        store,
        None,
    );

    let sleep_for = Duration::from_millis(200);
    let mut turn = h.send(sleep_for).await.unwrap();

    // Cancel right after the turn starts; the turn must still take
    // ~sleep_for because nothing races its sleep.
    tokio::time::sleep(Duration::from_millis(20)).await;
    h.cancel();

    let start = std::time::Instant::now();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;
    let _ = h.join().await;
    let elapsed = start.elapsed();

    // Generous lower bound to tolerate slow CI.
    assert!(
        elapsed >= Duration::from_millis(120),
        "uncooperative handle_streaming must NOT be cut short by cancel; \
         elapsed={elapsed:?}, sleep={sleep_for:?}. If this fires, the runtime \
         probably grew a `select!` against cancel around handle_streaming — \
         update the doc on `StreamingActorAgent::handle_streaming` and \
         replace this with the positive latency test."
    );
}

#[tokio::test]
async fn cancel_exit_still_writes_final_save() {
    let rec = Arc::new(RecordingStore::default());
    let store: Arc<dyn ActorStore> = rec.clone();
    let mut cfg = ActorConfig::for_streaming();
    cfg.checkpoint_interval = 0;

    let h = spawn_streaming_actor("cancel-1", EchoActor::default(), cfg, store, None);
    // One ping then cancel — exit is Cancelled, not Terminal.
    let mut turn = h.send(Msg::Ping("hi".into())).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;
    h.cancel();
    let _ = h.join().await;

    let calls = rec.snapshot();
    let saves = calls.iter().filter(|c| c.starts_with("save:")).count();
    let deletes = calls.iter().filter(|c| c.starts_with("delete:")).count();
    assert!(
        saves >= 1,
        "Cancelled exit must write a final checkpoint; calls={calls:?}"
    );
    assert_eq!(
        deletes, 0,
        "Cancelled exit must not delete; calls={calls:?}"
    );
}

// Dual-channel cancel contract: a single cancel fires BOTH
// `StreamEvent::Failed(BorgError::Cancelled)` on the event channel AND
// `StreamingHandleOutcome::Terminal("cancelled")` on the outcome channel.
// The event channel keeps the granular error for in-stream observability;
// the outcome channel gives the parent task a coarse dispatch signal.
// Callers consuming both must dedup (see `TurnHandle`). This test counts
// exactly one Failed event and asserts the Terminal outcome.

/// Streaming actor that holds the stream open until cancel fires so the
/// cancel races the drain loop deterministically.
#[derive(Default)]
struct ParkUntilCancelledAgent;

#[async_trait]
impl StreamingActorAgent for ParkUntilCancelledAgent {
    type Message = ();
    type Snapshot = EchoSnap;

    async fn handle_streaming(
        &mut self,
        _msg: (),
        ctx: &crate::runtime::ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        // Yields nothing until the ambient cancel fires, so the drain
        // loop's cancel arm always wins (no events compete).
        let cancel = ctx.cancel.clone();
        let stream = async_stream::stream! {
            cancel.cancelled().await;
            // Unreachable: the runtime's cancel arm produces the Failed
            // event; this yield only types the stream's item.
            #[allow(unreachable_code)]
            { yield StreamEvent::TextDelta("unreachable".into()); }
        };
        Ok((Box::pin(stream), StreamingHandleOutcome::Continue))
    }

    fn checkpoint(&self) -> EchoSnap {
        EchoSnap::default()
    }
    fn restore(_: EchoSnap) -> Self {
        Self
    }
}

#[tokio::test]
async fn cancel_emits_failed_event_and_terminal_outcome() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_streaming_actor(
        "dual-channel-1",
        ParkUntilCancelledAgent,
        ActorConfig::for_streaming(),
        store,
        None,
    );

    let mut turn = h.send(()).await.unwrap();
    // Let the turn enter its parked-stream state, then cancel.
    tokio::time::sleep(Duration::from_millis(20)).await;
    h.cancel();

    // Expect exactly one Failed(Cancelled); the agent yields nothing else.
    let mut failed_events: Vec<BorgError> = Vec::new();
    let mut other_events = 0;
    while let Some(ev) = turn.next_event().await {
        match ev {
            StreamEvent::Failed(e) => failed_events.push(e),
            _ => other_events += 1,
        }
    }
    assert_eq!(
        failed_events.len(),
        1,
        "expected exactly one Failed event on the event channel; got {} (plus {other_events} non-Failed)",
        failed_events.len()
    );
    assert!(
        matches!(failed_events[0], BorgError::Cancelled),
        "Failed event must carry BorgError::Cancelled; got {:?}",
        failed_events[0]
    );

    // The outcome channel independently surfaces Terminal("cancelled").
    let outcome = turn.outcome().await;
    assert_eq!(
        outcome,
        StreamingHandleOutcome::Terminal("cancelled".into()),
        "outcome channel must surface Terminal(\"cancelled\")"
    );

    let _ = h.join().await;
}
