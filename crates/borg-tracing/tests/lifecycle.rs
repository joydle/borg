use borg::events::{BorgEvent, CompositionShape, EventEmitter};
use borg_tracing::SpanKey;
use borg_tracing::{SpanRegistry, TracingEmitter};
use std::sync::Arc;
use tracing_test::traced_test;

#[tokio::test]
#[traced_test]
async fn composition_lifecycle_produces_spans() {
    let emitter = TracingEmitter::new();

    emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: "root::step1".into(),
            parent_id: Some("root".into()),
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "root::step1".into(),
            step_name: "step1".into(),
            position: 0,
            turn_number: 0,
            model: "claude".into(),
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::TurnStarted {
            composition_id: "root::step1".into(),
            step_name: "step1".into(),
            turn_index: 0,
            model: "claude".into(),
            input_tokens: 100,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::ToolCalled {
            composition_id: "root::step1".into(),
            step_name: "step1".into(),
            turn_index: 0,
            call_id: "c1".into(),
            tool_name: "read_file".into(),
            batch_id: None,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::ToolReturned {
            composition_id: "root::step1".into(),
            step_name: "step1".into(),
            turn_index: 0,
            call_id: "c1".into(),
            tool_name: "read_file".into(),
            success: true,
            error: None,
            batch_id: None,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::TurnCompleted {
            composition_id: "root::step1".into(),
            step_name: "step1".into(),
            turn_index: 0,
            output_tokens: 50,
            cache_read_tokens: 10,
            cache_creation_tokens: 0,
            first_token_ms: None,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::StepCompleted {
            composition_id: "root::step1".into(),
            step_name: "step1".into(),
            position: 0,
            duration_ms: 100,
            input_tokens: 100,
            output_tokens: 50,
            success: true,
            failure_cause: None,
            turn_number: 0,
            model: "claude".into(),
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id: "root::step1".into(),
            total_duration_ms: 100,
            total_input_tokens: 100,
            total_output_tokens: 50,
            succeeded: 1,
            failed: 0,
        }))
        .await;

    assert!(logs_contain("borg.composition"));
    assert!(logs_contain("borg.step"));
    assert!(logs_contain("borg.turn"));
    assert!(logs_contain("borg.tool"));
}

// Span fields must be pre-declared with `tracing::field::Empty`, or
// `.record()` is a silent no-op. Verifies llm.model / llm.input_tokens
// actually reach span output.
#[tokio::test]
#[traced_test]
async fn turn_span_fields_are_recorded_not_dropped() {
    let emitter = TracingEmitter::new();

    emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: "c1".into(),
            parent_id: None,
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "c1".into(),
            step_name: "s1".into(),
            position: 0,
            turn_number: 0,
            model: "gpt-4o".into(),
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::TurnStarted {
            composition_id: "c1".into(),
            step_name: "s1".into(),
            turn_index: 0,
            model: "gpt-4o".into(),
            input_tokens: 42,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::TurnCompleted {
            composition_id: "c1".into(),
            step_name: "s1".into(),
            turn_index: 0,
            output_tokens: 10,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            first_token_ms: Some(150),
        }))
        .await;

    // Pre-declared fields appear in the structured log event emitted
    // inside the span, which tracing_test captures.
    assert!(
        logs_contain("gpt-4o"),
        "llm.model should be visible in span output"
    );
    assert!(
        logs_contain("input_tokens=42") || logs_contain("input_tokens: 42"),
        "llm.input_tokens should be visible in span output"
    );
}

// SpanRegistry::get_turn_span lets executors instrument agent
// invocations with the correct parent.
#[tokio::test]
#[traced_test]
async fn span_store_get_turn_span_returns_stored_span() {
    let store = SpanRegistry::default();

    let span = tracing::info_span!(
        "borg.turn",
        turn.index = tracing::field::Empty,
        llm.model = tracing::field::Empty,
    );
    store.insert(SpanKey::Turn("c1".into(), "s1".into(), 0), span);

    let retrieved = store.get_turn_span("c1", "s1", 0);
    assert!(
        retrieved.is_some(),
        "get_turn_span should return the stored span"
    );
    assert!(
        !retrieved.unwrap().is_disabled(),
        "retrieved span should be enabled"
    );

    let missing = store.get_turn_span("c1", "s1", 99);
    assert!(
        missing.is_none(),
        "get_turn_span should return None for missing key"
    );
}

// TracingEmitter exposes turn_span() via the EventEmitter trait.
#[tokio::test]
#[traced_test]
async fn tracing_emitter_turn_span_returns_span_after_turn_started() {
    let emitter = TracingEmitter::new();

    emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: "c2".into(),
            parent_id: None,
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "c2".into(),
            step_name: "step-a".into(),
            position: 0,
            turn_number: 0,
            model: "claude-3".into(),
        }))
        .await;

    emitter
        .emit(Arc::new(BorgEvent::TurnStarted {
            composition_id: "c2".into(),
            step_name: "step-a".into(),
            turn_index: 1,
            model: "claude-3".into(),
            input_tokens: 99,
        }))
        .await;

    let span = emitter.turn_span("c2", "step-a", 1);
    assert!(
        !span.is_disabled(),
        "turn_span should return the live borg.turn span"
    );
}
