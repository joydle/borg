use async_trait::async_trait;
use borg::events::{BorgEvent, EventEmitter};
use std::sync::Arc;

/// Fan-out emitter that drives two `EventEmitter`s concurrently.
///
/// Both emitters receive every event. `primary` is consulted first for
/// `turn_span`/`tool_span` lookups, so the emitter holding the span
/// registry returns a live span rather than `Span::none()`.
///
/// ```rust,ignore
/// MultiEmitter::new(TracingEmitter::new(), existing_emitter)
/// ```
#[non_exhaustive]
pub struct MultiEmitter<A, B> {
    primary: A,
    secondary: B,
}

impl<A, B> MultiEmitter<A, B> {
    /// Wire two emitters into a fan-out pair. `primary` is the preferred
    /// source for span lookups; both receive every event.
    pub fn new(primary: A, secondary: B) -> Self {
        Self { primary, secondary }
    }
}

#[async_trait]
impl<A, B> EventEmitter for MultiEmitter<A, B>
where
    A: EventEmitter,
    B: EventEmitter,
{
    async fn emit(&self, event: Arc<BorgEvent>) {
        // Share the Arc across sinks; secondary takes the original handle.
        futures::join!(
            self.primary.emit(Arc::clone(&event)),
            self.secondary.emit(event),
        );
    }

    fn turn_span(&self, composition_id: &str, step_name: &str, turn_index: u32) -> tracing::Span {
        let span = self
            .primary
            .turn_span(composition_id, step_name, turn_index);
        if !span.is_disabled() {
            span
        } else {
            self.secondary
                .turn_span(composition_id, step_name, turn_index)
        }
    }

    fn tool_span(
        &self,
        composition_id: &str,
        step_name: &str,
        turn_index: u32,
        call_id: &str,
    ) -> tracing::Span {
        let span = self
            .primary
            .tool_span(composition_id, step_name, turn_index, call_id);
        if !span.is_disabled() {
            span
        } else {
            self.secondary
                .tool_span(composition_id, step_name, turn_index, call_id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TracingEmitter;
    use borg::events::{BorgEvent, CompositionShape, RecordingEmitter};

    /// MultiEmitter must delegate `turn_span` to the inner emitter that
    /// holds the span, not return `Span::none()`.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn multi_emitter_turn_span_delegates_to_tracing_emitter() {
        let tracing_emitter = TracingEmitter::new();
        let m = MultiEmitter::new(tracing_emitter, RecordingEmitter::new());

        // Drive the lifecycle so TracingEmitter stores a turn span.
        m.emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: "c1".into(),
            parent_id: None,
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;
        m.emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "c1".into(),
            step_name: "plan".into(),
            position: 0,
            turn_number: 0,
            model: "test".into(),
        }))
        .await;
        m.emit(Arc::new(BorgEvent::TurnStarted {
            composition_id: "c1".into(),
            step_name: "plan".into(),
            turn_index: 0,
            model: "test".into(),
            input_tokens: 100,
        }))
        .await;

        let span = m.turn_span("c1", "plan", 0);
        assert!(
            !span.is_disabled(),
            "MultiEmitter::turn_span returned Span::none() — the turn span does not reach MultiEmitter consumers"
        );
    }

    /// MultiEmitter must delegate `tool_span` to the inner emitter that
    /// holds the span, not return `Span::none()`.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn multi_emitter_tool_span_delegates_to_tracing_emitter() {
        let tracing_emitter = TracingEmitter::new();
        let m = MultiEmitter::new(tracing_emitter, RecordingEmitter::new());

        // Drive the lifecycle so TracingEmitter stores a tool span.
        m.emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: "c1".into(),
            parent_id: None,
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;
        m.emit(Arc::new(BorgEvent::StepStarted {
            composition_id: "c1".into(),
            step_name: "plan".into(),
            position: 0,
            turn_number: 0,
            model: "test".into(),
        }))
        .await;
        m.emit(Arc::new(BorgEvent::TurnStarted {
            composition_id: "c1".into(),
            step_name: "plan".into(),
            turn_index: 0,
            model: "test".into(),
            input_tokens: 100,
        }))
        .await;
        m.emit(Arc::new(BorgEvent::ToolCalled {
            composition_id: "c1".into(),
            step_name: "plan".into(),
            turn_index: 0,
            call_id: "call-1".into(),
            tool_name: "read_file".into(),
            batch_id: None,
        }))
        .await;

        let span = m.tool_span("c1", "plan", 0, "call-1");
        assert!(
            !span.is_disabled(),
            "MultiEmitter::tool_span returned Span::none() — the tool span does not reach MultiEmitter consumers"
        );
    }

    #[tokio::test]
    async fn multi_emitter_fans_out_to_both() {
        let m = MultiEmitter::new(RecordingEmitter::new(), RecordingEmitter::new());
        m.emit(Arc::new(BorgEvent::Cancelled {
            composition_id: "c1".into(),
            reason: "test".into(),
        }))
        .await;
        assert_eq!(m.primary.kinds(), vec!["cancelled"]);
        assert_eq!(m.secondary.kinds(), vec!["cancelled"]);
    }
}
