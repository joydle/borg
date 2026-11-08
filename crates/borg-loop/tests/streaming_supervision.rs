//! Supervision regressions for the streaming actor runtime: panics,
//! cancel under backpressure, and `StreamingComposition::then` follow-ups
//! driven by a consumer that stops at the first terminal event.

use async_trait::async_trait;
use borg::{
    execute_streaming, AgentContext, AgentLlmClient, BorgAgent, BorgError, BorgStreamingAgent,
    ChatRequest, ChatResponse, Composition, ExecCtx, NoopEmitter, StreamEvent,
    StreamingComposition, Verdict,
};
use borg_loop::{
    spawn_streaming_actor, ActorConfig, ActorCtx, ActorExit, ActorStore, InMemoryActorStore,
    NoopActorStore, StreamingActorAgent, StreamingHandleOutcome, TurnHandle,
};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

async fn drain(mut turn: TurnHandle) -> (Vec<StreamEvent>, StreamingHandleOutcome) {
    let mut events = vec![];
    while let Some(ev) = turn.next_event().await {
        events.push(ev);
    }
    (events, turn.outcome().await)
}

// ── Then follow-up under the streaming runtime ─────────────────────────

/// Counts calls; answers `PASS` or fails non-retryably.
struct CountingLlm {
    calls: AtomicUsize,
    fail: bool,
}

#[async_trait]
impl AgentLlmClient for CountingLlm {
    async fn chat_completion(&self, _req: ChatRequest) -> Result<ChatResponse, BorgError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(BorgError::Internal {
                message: "follow-up llm down".into(),
            });
        }
        Ok(ChatResponse::new("PASS\npolished", 10, 5))
    }
}

/// Streaming primary: one delta then `Done`.
struct Primary;

#[async_trait]
impl BorgStreamingAgent for Primary {
    fn agent_id(&self) -> &'static str {
        "primary"
    }
    fn model(&self) -> &str {
        "fake/primary"
    }
    async fn invoke_streaming(
        &self,
        _ctx: AgentContext,
    ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
        Ok(Box::pin(async_stream::stream! {
            yield StreamEvent::TextDelta("draft".into());
            yield StreamEvent::Done(Verdict::with_token("", "draft", 1, 1));
        }))
    }
}

/// One-shot follow-up using the default LLM-backed `invoke`.
struct Polish;

impl BorgAgent for Polish {
    fn agent_id(&self) -> &'static str {
        "polish"
    }
    fn model(&self) -> &str {
        "fake/polish"
    }
    fn system_prompt(&self) -> &'static str {
        "polish"
    }
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        ctx.handoff.get_as::<String>("draft")
    }
    fn known_verdicts(&self) -> &'static [&'static str] {
        &["PASS"]
    }
}

struct ThenActor {
    llm: Arc<CountingLlm>,
}

#[derive(Serialize, Deserialize)]
struct Empty {}

#[async_trait]
impl StreamingActorAgent for ThenActor {
    type Message = ();
    type Snapshot = Empty;

    async fn handle_streaming(
        &mut self,
        _msg: (),
        ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        let exec = ExecCtx::new(
            self.llm.clone(),
            borg::in_memory_budget(10_000),
            Arc::new(NoopEmitter),
            ctx.cancel.child_token(),
            "then-actor",
        );
        let comp = StreamingComposition::single("primary", Primary)
            .then(Composition::single("polish", Polish), "draft");
        Ok((execute_streaming(comp, exec), StreamingHandleOutcome::Park))
    }
    fn checkpoint(&self) -> Empty {
        Empty {}
    }
    fn restore(_: Empty) -> Self {
        unreachable!()
    }
}

async fn run_then_turn(fail: bool) -> (Vec<StreamEvent>, usize) {
    let llm = Arc::new(CountingLlm {
        calls: AtomicUsize::new(0),
        fail,
    });
    let handle = spawn_streaming_actor(
        "then-actor",
        ThenActor { llm: llm.clone() },
        ActorConfig::for_streaming(),
        Arc::new(NoopActorStore),
        None,
    );
    let (events, _) = drain(handle.send(()).await.unwrap()).await;
    handle.cancel();
    handle.join().await;
    (events, llm.calls.load(Ordering::SeqCst))
}

#[tokio::test]
async fn then_follow_up_runs_under_streaming_runtime() {
    let (events, calls) = run_then_turn(false).await;
    assert_eq!(calls, 1, "follow-up LLM must be called exactly once");
    assert!(
        matches!(events.last(), Some(StreamEvent::Done(_))),
        "{events:?}"
    );
}

#[tokio::test]
async fn then_follow_up_failure_surfaces_its_error() {
    let (events, calls) = run_then_turn(true).await;
    assert_eq!(calls, 1);
    match events.last() {
        Some(StreamEvent::Failed(e)) => {
            assert!(e.to_string().contains("follow-up llm down"), "error: {e}");
        }
        other => panic!("expected Failed(follow-up error), got {other:?}"),
    }
    assert!(!events.iter().any(|e| matches!(e, StreamEvent::Done(_))));
}

// ── Panic boundary ─────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
struct CountSnap {
    count: u32,
}

enum PanicMsg {
    Increment,
    PanicInHandle,
    PanicInStream,
}

struct PanicActor {
    count: u32,
}

#[async_trait]
impl StreamingActorAgent for PanicActor {
    type Message = PanicMsg;
    type Snapshot = CountSnap;

    async fn handle_streaming(
        &mut self,
        msg: PanicMsg,
        _ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        match msg {
            PanicMsg::Increment => {
                self.count += 1;
                Ok((
                    Box::pin(async_stream::stream! {
                        yield StreamEvent::Done(Verdict::empty());
                    }),
                    StreamingHandleOutcome::Park,
                ))
            }
            PanicMsg::PanicInHandle => {
                // Half-applied mutation: must never be persisted.
                self.count = 999;
                panic!("handle boom");
            }
            PanicMsg::PanicInStream => Ok((
                Box::pin(async_stream::stream! {
                    yield StreamEvent::TextDelta("partial".into());
                    panic!("stream boom");
                }),
                StreamingHandleOutcome::Park,
            )),
        }
    }
    fn checkpoint(&self) -> CountSnap {
        CountSnap { count: self.count }
    }
    fn restore(snap: CountSnap) -> Self {
        Self { count: snap.count }
    }
}

async fn stored_count(store: &InMemoryActorStore, addr: &str) -> Option<u32> {
    store
        .load(addr)
        .await
        .unwrap()
        .map(|b| serde_json::from_slice::<CountSnap>(&b).unwrap().count)
}

fn assert_panicked_failure(events: &[StreamEvent], needle: &str) {
    match events.last() {
        Some(StreamEvent::Failed(BorgError::ChildPanicked { message, .. })) => {
            assert!(message.contains(needle), "message: {message}");
        }
        other => panic!("expected Failed(ChildPanicked), got {other:?}"),
    }
}

#[tokio::test]
async fn panic_in_handle_streaming_exits_panicked_and_keeps_last_good_snapshot() {
    let store = Arc::new(InMemoryActorStore::new());
    // Checkpoint every turn, so the Increment turn leaves a good snapshot.
    let handle = spawn_streaming_actor(
        "panic-handle",
        PanicActor { count: 0 },
        ActorConfig::for_streaming(),
        store.clone(),
        None,
    );
    drain(handle.send(PanicMsg::Increment).await.unwrap()).await;
    assert_eq!(stored_count(&store, "panic-handle").await, Some(1));

    let (events, outcome) = drain(handle.send(PanicMsg::PanicInHandle).await.unwrap()).await;
    assert_panicked_failure(&events, "handle boom");
    assert!(
        matches!(outcome, StreamingHandleOutcome::Terminal(_)),
        "{outcome:?}"
    );

    let exit = tokio::time::timeout(Duration::from_secs(2), handle.join())
        .await
        .expect("actor must exit");
    match exit {
        ActorExit::Panicked(msg) => assert!(msg.contains("handle boom"), "msg: {msg}"),
        other => panic!("expected Panicked, got {other:?}"),
    }
    // The half-applied 999 is never persisted; the last good snapshot stays.
    assert_eq!(stored_count(&store, "panic-handle").await, Some(1));
}

#[tokio::test]
async fn panic_mid_stream_exits_panicked_and_persists_last_good_snapshot() {
    let store = Arc::new(InMemoryActorStore::new());
    // No periodic checkpoints: the only save is the one the panic boundary writes.
    let mut config = ActorConfig::for_streaming();
    config.checkpoint_interval = 0;
    let handle = spawn_streaming_actor(
        "panic-stream",
        PanicActor { count: 0 },
        config,
        store.clone(),
        None,
    );
    drain(handle.send(PanicMsg::Increment).await.unwrap()).await;
    drain(handle.send(PanicMsg::Increment).await.unwrap()).await;
    assert_eq!(stored_count(&store, "panic-stream").await, None);

    let (events, outcome) = drain(handle.send(PanicMsg::PanicInStream).await.unwrap()).await;
    assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "partial"));
    assert_panicked_failure(&events, "stream boom");
    assert!(
        matches!(outcome, StreamingHandleOutcome::Terminal(_)),
        "{outcome:?}"
    );

    let exit = tokio::time::timeout(Duration::from_secs(2), handle.join())
        .await
        .expect("actor must exit");
    match exit {
        ActorExit::Panicked(msg) => assert!(msg.contains("stream boom"), "msg: {msg}"),
        other => panic!("expected Panicked, got {other:?}"),
    }
    // `handle_streaming` returned normally before the stream panicked, so
    // the agent's state (count = 2) is consistent and is persisted.
    assert_eq!(stored_count(&store, "panic-stream").await, Some(2));
}

// ── Cancel under backpressure ──────────────────────────────────────────

/// Emits deltas forever without awaiting anything but the consumer.
struct Firehose;

#[async_trait]
impl StreamingActorAgent for Firehose {
    type Message = ();
    type Snapshot = Empty;

    async fn handle_streaming(
        &mut self,
        _msg: (),
        _ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        Ok((
            Box::pin(async_stream::stream! {
                loop {
                    yield StreamEvent::TextDelta("x".into());
                }
            }),
            StreamingHandleOutcome::Park,
        ))
    }
    fn checkpoint(&self) -> Empty {
        Empty {}
    }
    fn restore(_: Empty) -> Self {
        Firehose
    }
}

#[tokio::test]
async fn cancel_stops_actor_blocked_on_full_event_channel() {
    let mut config = ActorConfig::for_streaming();
    config.event_channel_capacity = 1;
    let handle =
        spawn_streaming_actor("firehose", Firehose, config, Arc::new(NoopActorStore), None);
    // Hold the turn but never drain it: the runtime fills the channel and
    // blocks on its next send.
    let turn = handle.send(()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    handle.cancel();
    let exit = tokio::time::timeout(Duration::from_secs(2), handle.join())
        .await
        .expect("cancel must stop an actor blocked on a full event channel");
    assert!(matches!(exit, ActorExit::Cancelled), "{exit:?}");
    drop(turn);
}
