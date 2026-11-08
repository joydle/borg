//! Cross-spawn parent-id regression tests.
//!
//! `.entered()` before `tokio::spawn` is thread-local: the guard drops on
//! the caller thread before the spawned task polls, so spans inside the
//! task orphan as trace roots. The spawn paths instead use
//! `.instrument(span)`, which carries the span across `.await` and nested
//! spawns. These tests use a hand-rolled `tracing_subscriber::Layer` that
//! records `(name, span_id, parent_id)` tuples — no OTLP exporter needed.

#![cfg(feature = "tracing")]

use async_trait::async_trait;
use borg::{
    deterministic_llm_client, in_memory_budget, noop_emitter, AgentContext, BorgAgent, BorgError,
    CancellationToken, ExecCtx, HandoffState, RestartPolicy, StreamEvent, Verdict,
};
use borg_loop::{
    spawn_actor, spawn_streaming_actor, ActorAgent, ActorConfig, ActorCtx, HandleOutcome,
    InMemoryActorStore, StreamingActorAgent, StreamingHandleOutcome,
};
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::{span, Id, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::Registry;

/// A `tracing_subscriber::Layer` that records every new span's
/// `(name, span_id, parent_id)` and every span close, so tests can assert
/// both ancestry and finalization (close is what drives OTel export).
#[derive(Default, Clone)]
struct SpanCapture {
    rows: Arc<Mutex<Vec<SpanRow>>>,
    closed: Arc<Mutex<Vec<u64>>>,
}

#[derive(Debug, Clone)]
struct SpanRow {
    name: &'static str,
    span_id: u64,
    parent_id: Option<u64>,
}

impl SpanCapture {
    fn rows(&self) -> Vec<SpanRow> {
        self.rows.lock().unwrap().clone()
    }

    fn find(&self, name: &str) -> Option<SpanRow> {
        self.rows().into_iter().find(|r| r.name == name)
    }

    fn closed_ids(&self) -> Vec<u64> {
        self.closed.lock().unwrap().clone()
    }
}

impl<S> Layer<S> for SpanCapture
where
    S: Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        // Prefer the explicit parent, else the current span at creation
        // (the implicit-parent path that `.instrument(...)` flows through).
        let parent_id = attrs
            .parent()
            .map(|p| p.into_u64())
            .or_else(|| ctx.lookup_current().map(|s| s.id().into_u64()));

        let name = attrs.metadata().name();

        self.rows.lock().unwrap().push(SpanRow {
            name,
            span_id: id.into_u64(),
            parent_id,
        });
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        self.closed.lock().unwrap().push(id.into_u64());
    }
}

fn install_capture() -> (SpanCapture, tracing::subscriber::DefaultGuard) {
    let capture = SpanCapture::default();
    let subscriber = Registry::default().with(capture.clone());
    let guard = tracing::subscriber::set_default(subscriber);
    (capture, guard)
}

#[derive(Serialize, Deserialize, Default)]
struct CounterSnap {
    count: u32,
}

#[derive(Default)]
struct Counter {
    count: u32,
}

#[allow(dead_code)]
enum CounterMsg {
    Inc,
    /// Open an `inner` span inside the handler. The test asserts
    /// `inner.parent_id == borg.session.span_id`, which holds only if
    /// the session span propagated across the `tokio::spawn` boundary.
    EmitInnerSpan,
    Stop,
}

#[async_trait]
impl ActorAgent for Counter {
    type Message = CounterMsg;
    type Snapshot = CounterSnap;

    async fn handle(&mut self, msg: CounterMsg, _ctx: &ActorCtx) -> HandleOutcome {
        match msg {
            CounterMsg::Inc => {
                self.count += 1;
                HandleOutcome::Continue
            }
            CounterMsg::EmitInnerSpan => {
                // Created inside the spawned task; orphans as a root if
                // propagation is broken.
                let _g = tracing::info_span!("inner").entered();
                tracing::info!("inner-span event");
                HandleOutcome::Continue
            }
            CounterMsg::Stop => HandleOutcome::Terminal("done".into()),
        }
    }

    fn checkpoint(&self) -> CounterSnap {
        CounterSnap { count: self.count }
    }

    fn restore(s: CounterSnap) -> Self {
        Self { count: s.count }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn spawn_actor_propagates_session_span_into_spawned_task() {
    let (capture, _g) = install_capture();

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_actor(
        "tracing-prop-test-1",
        Counter::default(),
        ActorConfig::default(),
        store,
        None,
    );

    h.send(CounterMsg::EmitInnerSpan).await.unwrap();
    h.send(CounterMsg::Stop).await.unwrap();
    let _ = h.join().await;

    let session = capture
        .find("borg.session")
        .expect("borg.session span must be captured");
    let inner = capture
        .find("inner")
        .expect("inner span (opened inside the spawned actor task) must be captured");

    assert_eq!(
        inner.parent_id,
        Some(session.span_id),
        "inner.parent_id must equal borg.session.span_id; got parent={:?} session={}.\n\
         If this fires, span propagation across `tokio::spawn` has regressed.\n\
         Captured rows: {:#?}",
        inner.parent_id,
        session.span_id,
        capture.rows(),
    );
}

#[derive(Serialize, Deserialize, Default)]
struct EchoSnap;

#[derive(Default)]
struct StreamingEmitter;

#[async_trait]
impl StreamingActorAgent for StreamingEmitter {
    type Message = ();
    type Snapshot = EchoSnap;

    async fn handle_streaming(
        &mut self,
        _msg: (),
        _ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        // Opened inside the spawned streaming-loop task.
        let _g = tracing::info_span!("inner").entered();
        tracing::info!("streaming inner-span event");
        Ok((
            Box::pin(stream::empty::<StreamEvent>()),
            StreamingHandleOutcome::Terminal("done".into()),
        ))
    }

    fn checkpoint(&self) -> EchoSnap {
        EchoSnap
    }

    fn restore(_: EchoSnap) -> Self {
        Self
    }
}

#[tokio::test(flavor = "current_thread")]
async fn spawn_streaming_actor_propagates_session_span_into_spawned_task() {
    let (capture, _g) = install_capture();

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_streaming_actor(
        "tracing-prop-test-2",
        StreamingEmitter,
        ActorConfig::for_streaming(),
        store,
        None,
    );

    let mut turn = h.send(()).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;
    let _ = h.join().await;

    let session = capture
        .find("borg.session")
        .expect("borg.session span must be captured");
    let inner = capture
        .find("inner")
        .expect("inner span (opened inside the streaming task) must be captured");

    assert_eq!(
        inner.parent_id,
        Some(session.span_id),
        "streaming-actor: inner.parent_id must equal borg.session.span_id.\n\
         Captured rows: {:#?}",
        capture.rows(),
    );
}

// Outer-span propagation: an `outer` grpc-request-like span → borg.session.
#[tokio::test(flavor = "current_thread")]
async fn outer_caller_span_remains_ancestor_through_spawn_streaming_actor() {
    let (capture, _g) = install_capture();

    // Open `outer` on the test task, spawn the actor inside it; `outer`
    // must end up an ancestor of the `inner` span in the spawned task.
    let outer = tracing::info_span!("outer");
    let outer_id = outer.id().unwrap().into_u64();

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = {
        let _g = outer.enter();
        spawn_streaming_actor(
            "tracing-prop-test-3",
            StreamingEmitter,
            ActorConfig::for_streaming(),
            store,
            None,
        )
    };

    let mut turn = h.send(()).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;
    let _ = h.join().await;

    let rows = capture.rows();
    let session = rows
        .iter()
        .find(|r| r.name == "borg.session")
        .expect("borg.session span must be captured");

    // session.parent_id is outer: we entered `outer` before spawning, so
    // the session span was created with current = outer.
    assert_eq!(
        session.parent_id,
        Some(outer_id),
        "borg.session.parent_id must be the caller's outer span; got {:?}.\n\
         If this fires, the caller-side span context isn't being captured at\n\
         span construction. Rows: {:#?}",
        session.parent_id,
        rows,
    );

    // inner must chain up: inner → session → outer.
    let inner = rows
        .iter()
        .find(|r| r.name == "inner")
        .expect("inner span must be captured");
    assert_eq!(
        inner.parent_id,
        Some(session.span_id),
        "inner must parent to session (which itself parents to outer)"
    );
}

// ExecCtx::spawn_sub_agent span-inheritance guard. `spawn_sub_agent`
// (crates/borg/src/scheduled.rs) also uses `.in_current_span()` before
// `tokio::spawn`; same hazard, same test shape.
struct InnerSpanAgent;

#[async_trait]
impl BorgAgent for InnerSpanAgent {
    fn agent_id(&self) -> &'static str {
        "inner_span_agent"
    }

    async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
        let _g = tracing::info_span!("inner").entered();
        tracing::info!("sub-agent invoke event");
        Ok(Verdict::default())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn scheduled_sub_agent_inherits_caller_span() {
    let (capture, _g) = install_capture();

    let outer = tracing::info_span!("outer");
    let outer_id = outer.id().unwrap().into_u64();

    let ctx = ExecCtx::new(
        deterministic_llm_client(),
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "tracing-prop-test",
    );

    let handle = {
        let _g = outer.enter();
        ctx.spawn_sub_agent(
            "child",
            InnerSpanAgent,
            RestartPolicy::Never,
            HandoffState::new(),
        )
    };
    let _verdict = handle.await_completion().await.expect("sub-agent verdict");

    let inner = capture
        .find("inner")
        .expect("inner span opened inside sub-agent invoke must be captured");
    assert_eq!(
        inner.parent_id,
        Some(outer_id),
        "ExecCtx::spawn_sub_agent must propagate the caller's current span into the spawned task.\n\
         If this fires, `.in_current_span()` at scheduled.rs:~356 was likely dropped. \n\
         Captured rows: {:#?}",
        capture.rows(),
    );
}

// `tracing-opentelemetry` only exports an OTLP span when the
// `tracing::Span` *closes*. A lifetime-of-actor `borg.session` span
// therefore never exports for long-lived parked actors, so one
// `borg.session` is opened and closed per turn. These tests drive a
// single turn WITHOUT cancelling or joining (production never cancels
// parked actors) and assert the span closed.
#[tokio::test(flavor = "current_thread")]
async fn streaming_session_span_closes_per_turn_without_cancel() {
    let (capture, _g) = install_capture();

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_streaming_actor(
        "session-export-1",
        StreamingEmitter,
        ActorConfig::for_streaming(),
        store,
        None,
    );

    // One turn, left parked (no cancel/join).
    let mut turn = h.send(()).await.unwrap();
    while turn.next_event().await.is_some() {}
    let _ = turn.outcome().await;
    // Yield so the runtime's post-turn drop logic runs.
    tokio::task::yield_now().await;

    let session = capture
        .find("borg.session")
        .expect("borg.session span must have been created");
    let closed = capture.closed_ids();
    assert!(
        closed.contains(&session.span_id),
        "borg.session must close at end of turn so OTel exports it.\n\
         Created sessions: {:?}\n\
         Closed span ids: {:?}\n\
         If this fires, the per-turn `drop(turn_session)` at the end of the\n\
         message-handling arm in streaming_runtime.rs has regressed back to a\n\
         lifetime-of-actor span.",
        capture
            .rows()
            .iter()
            .filter(|r| r.name == "borg.session")
            .collect::<Vec<_>>(),
        closed,
    );

    // Tidy up so the test runtime exits cleanly.
    h.cancel();
    let _ = h.join().await;
}

#[tokio::test(flavor = "current_thread")]
async fn plain_session_span_closes_per_turn_without_cancel() {
    let (capture, _g) = install_capture();

    let store: Arc<dyn borg_loop::ActorStore> = Arc::new(InMemoryActorStore::new());
    let h = spawn_actor(
        "session-export-2",
        Counter::default(),
        ActorConfig::default(),
        store,
        None,
    );

    h.send(CounterMsg::Inc).await.unwrap();
    // Yield so the runtime processes the message and drops the span.
    tokio::task::yield_now().await;

    let session = capture
        .find("borg.session")
        .expect("borg.session span must have been created");
    let closed = capture.closed_ids();
    assert!(
        closed.contains(&session.span_id),
        "borg.session must close at end of turn so OTel exports it.\n\
         Created sessions: {:?}\n\
         Closed span ids: {:?}",
        capture
            .rows()
            .iter()
            .filter(|r| r.name == "borg.session")
            .collect::<Vec<_>>(),
        closed,
    );

    h.cancel();
    let _ = h.join().await;
}
