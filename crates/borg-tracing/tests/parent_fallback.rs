//! Regression tests for the parent-resolution fallback in
//! `TracingEmitter` (see `crates/borg-tracing/src/emitter.rs`).
//!
//! `TracingEmitter::with_parent(session_span)` stores a session span
//! that downstream events parent under when their registry lookup
//! misses. The fallback must anchor to that session span, using
//! `Span::current()` only when no `with_parent` was supplied — inside a
//! freshly-spawned task `Span::current()` is `Span::none()`, producing
//! orphan trace roots.
//!
//! These tests fire the emitter outside any caller-side span context, so
//! the only thing that can parent the resulting `borg.*` spans is the
//! stored session span; a regression to `Span::current()`-first leaves
//! them parentless and fails the assertions.

use borg::events::{BorgEvent, EventEmitter};
use borg_tracing::TracingEmitter;
use std::sync::{Arc, Mutex};
use tracing::{span, Id, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::Registry;

#[derive(Default, Clone)]
struct SpanCapture {
    rows: Arc<Mutex<Vec<SpanRow>>>,
}

#[derive(Debug, Clone)]
struct SpanRow {
    name: &'static str,
    parent_id: Option<u64>,
}

impl SpanCapture {
    fn rows(&self) -> Vec<SpanRow> {
        self.rows.lock().unwrap().clone()
    }

    fn find(&self, name: &str) -> Option<SpanRow> {
        self.rows().into_iter().find(|r| r.name == name)
    }
}

impl<S> Layer<S> for SpanCapture
where
    S: Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, _id: &Id, ctx: Context<'_, S>) {
        let parent_id = attrs
            .parent()
            .map(|p| p.into_u64())
            .or_else(|| ctx.lookup_current().map(|s| s.id().into_u64()));
        self.rows.lock().unwrap().push(SpanRow {
            name: attrs.metadata().name(),
            parent_id,
        });
    }
}

fn install_capture() -> (SpanCapture, tracing::subscriber::DefaultGuard) {
    let capture = SpanCapture::default();
    let subscriber = Registry::default().with(capture.clone());
    let guard = tracing::subscriber::set_default(subscriber);
    (capture, guard)
}

/// Cache-miss on `SpanKey::Composition` must fall back to the stored
/// session span, not `Span::current()`. Fired outside any caller-side
/// span, a regression surfaces as `borg.step.parent_id == None`.
#[tokio::test(flavor = "current_thread")]
async fn tracing_emitter_cache_miss_falls_back_to_session_span() {
    let (capture, _g) = install_capture();

    // Construct the session span out-of-band so `with_parent` doesn't
    // depend on it being the ambient current span.
    let session = tracing::info_span!("borg.session");
    let session_id = session.id().unwrap().into_u64();
    let emitter = TracingEmitter::with_parent(session);

    // Skip CompositionStarted so StepStarted misses the cache and takes
    // the fallback branch.
    emitter
        .emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "missing-comp".into(),
            step_name: "step1".into(),
            position: 0,
            turn_number: 0,
            model: "test".into(),
        }))
        .await;

    let step = capture
        .find("borg.step")
        .expect("borg.step span must be captured");
    assert_eq!(
        step.parent_id,
        Some(session_id),
        "borg.step must parent under the stored session span on cache miss.\n\
         If parent_id is None, `fallback_parent` regressed to bare `Span::current()` \
         (which is `Span::none()` here because no span is entered on the test task).\n\
         Captured rows: {:#?}",
        capture.rows(),
    );
}

/// Every cache-missing event, not just the first, must anchor to the
/// session span. Drives StepStarted -> TurnStarted -> ToolCalled with
/// distinct keys so each independently takes the fallback path; all
/// three must chain to the session. Fires if `session_span` is ever
/// dropped or reassigned mid-life.
#[tokio::test(flavor = "current_thread")]
async fn tracing_emitter_with_parent_anchors_all_subsequent_emits() {
    let (capture, _g) = install_capture();

    let session = tracing::info_span!("borg.session");
    let session_id = session.id().unwrap().into_u64();
    let emitter = TracingEmitter::with_parent(session);

    // Fresh keys per event so none hit the registry.
    emitter
        .emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "comp-a".into(),
            step_name: "step-a".into(),
            position: 0,
            turn_number: 0,
            model: "test".into(),
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::TurnStarted {
            composition_id: "comp-b".into(),
            step_name: "step-b".into(),
            turn_index: 0,
            model: "test".into(),
            input_tokens: 10,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::ToolCalled {
            composition_id: "comp-c".into(),
            step_name: "step-c".into(),
            turn_index: 0,
            call_id: "call-c".into(),
            tool_name: "tool-c".into(),
            batch_id: None,
        }))
        .await;

    let rows = capture.rows();
    for name in ["borg.step", "borg.turn", "borg.tool"] {
        let span = rows
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("{name} must be captured. Rows: {rows:#?}"));
        assert_eq!(
            span.parent_id,
            Some(session_id),
            "{name} (cache-miss) must parent under the stored session span.\n\
             Stickiness regression: the emitter dropped or replaced its session_span \
             between emits. Rows: {rows:#?}",
        );
    }
}
