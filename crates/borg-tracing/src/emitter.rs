use crate::fields;
use crate::span_store::{SpanKey, SpanRegistry};
use async_trait::async_trait;
use borg::events::{BorgEvent, EventEmitter};
use std::sync::Arc;
use tracing::Span;

pub struct TracingEmitter {
    session_span: Option<Span>,
    store: Arc<SpanRegistry>,
}

impl TracingEmitter {
    pub fn new() -> Self {
        Self {
            session_span: None,
            store: Arc::new(SpanRegistry::default()),
        }
    }

    pub fn with_parent(parent: Span) -> Self {
        Self {
            session_span: Some(parent),
            store: Arc::new(SpanRegistry::default()),
        }
    }
}

impl Default for TracingEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TracingEmitter {
    /// Resolve the best parent span for a child emit and invoke `f` with
    /// a borrow of it. Resolution order, first non-`None` wins:
    /// `cached` (registry lookup), then `session_span` (set via
    /// `with_parent`), then `Span::current()`.
    fn with_resolved_parent<R>(&self, cached: Option<&Span>, f: impl FnOnce(&Span) -> R) -> R {
        if let Some(s) = cached {
            return f(s);
        }
        if let Some(s) = self.session_span.as_ref() {
            return f(s);
        }
        let current = Span::current();
        f(&current)
    }
}

/// Create a child `borg.composition` span with all fields pre-declared.
/// Fields must be declared as `tracing::field::Empty` up front, or later
/// `span.record(...)` calls are silently dropped by OTLP exporters.
macro_rules! composition_span {
    ($parent:expr) => {{
        let _guard = $parent.enter();
        tracing::info_span!(
            "borg.composition",
            "session.id" = tracing::field::Empty,
            "composition.name" = tracing::field::Empty,
            "composition.kind" = tracing::field::Empty,
            "parallel.degraded" = tracing::field::Empty,
            "composition.total_input_tokens" = tracing::field::Empty,
            "composition.total_output_tokens" = tracing::field::Empty,
            "composition.succeeded" = tracing::field::Empty,
            "composition.failed" = tracing::field::Empty,
            "composition.outcome" = tracing::field::Empty,
        )
    }};
}

/// Create a child `borg.step` span with all fields pre-declared.
macro_rules! step_span {
    ($parent:expr) => {{
        let _guard = $parent.enter();
        tracing::info_span!(
            "borg.step",
            "step.name" = tracing::field::Empty,
            "step.index" = tracing::field::Empty,
            "step.retry" = tracing::field::Empty,
            "step.outcome" = tracing::field::Empty,
            "step.failure_cause" = tracing::field::Empty,
        )
    }};
}

/// Create a child `borg.turn` span with all fields pre-declared.
macro_rules! turn_span {
    ($parent:expr) => {{
        let _guard = $parent.enter();
        tracing::info_span!(
            "borg.turn",
            "turn.index" = tracing::field::Empty,
            "llm.model" = tracing::field::Empty,
            "llm.input_tokens" = tracing::field::Empty,
            "llm.output_tokens" = tracing::field::Empty,
            "llm.cache_read_tokens" = tracing::field::Empty,
            "llm.cache_creation_tokens" = tracing::field::Empty,
            "llm.ttft_ms" = tracing::field::Empty,
            "llm.finish_reason" = tracing::field::Empty,
            "llm.prompt_chars" = tracing::field::Empty,
        )
    }};
}

/// Create a child `borg.tool` span with all fields pre-declared.
macro_rules! tool_span {
    ($parent:expr) => {{
        let _guard = $parent.enter();
        tracing::info_span!(
            "borg.tool",
            "tool.name" = tracing::field::Empty,
            "tool.call_id" = tracing::field::Empty,
            "tool.success" = tracing::field::Empty,
            "tool.error" = tracing::field::Empty,
        )
    }};
}

#[async_trait]
impl EventEmitter for TracingEmitter {
    async fn emit(&self, event: Arc<BorgEvent>) {
        // Fields are `Arc<str>`, so the clones below are refcount bumps.
        match &*event {
            BorgEvent::CompositionStarted {
                composition_id,
                shape,
                ..
            } => {
                // Trace-tree root for a composition; nothing cached yet,
                // so this resolves to the session span or `Span::current()`.
                let span = self.with_resolved_parent(None, |parent| composition_span!(parent));
                span.record(fields::COMPOSITION_NAME, &**composition_id);
                // `_` arm: CompositionShape is `#[non_exhaustive]`.
                let kind: &'static str = match *shape {
                    borg::CompositionShape::Single => "single",
                    borg::CompositionShape::Sequential => "sequential",
                    borg::CompositionShape::Parallel => "parallel",
                    borg::CompositionShape::LoopUntil => "loop_until",
                    borg::CompositionShape::Branch => "branch",
                    borg::CompositionShape::Fallback => "fallback",
                    borg::CompositionShape::AgentLoop => "agent_loop",
                    _ => "other",
                };
                span.record(fields::COMPOSITION_KIND, kind);
                {
                    let _guard = span.enter();
                    tracing::info!(target: "borg.composition", composition_id = %composition_id, "composition started");
                }
                self.store
                    .insert(SpanKey::Composition(composition_id.clone()), span);
            }

            BorgEvent::StepStarted {
                composition_id,
                step_name,
                position,
                ..
            } => {
                let cached = self
                    .store
                    .get(&SpanKey::Composition(composition_id.clone()));
                let span = self.with_resolved_parent(cached.as_ref(), |parent| step_span!(parent));
                span.record(fields::STEP_NAME, &**step_name);
                span.record(fields::STEP_INDEX, *position as i64);
                {
                    let _guard = span.enter();
                    tracing::info!(target: "borg.step", step = %step_name, position, "step started");
                }
                self.store.insert(
                    SpanKey::Step(composition_id.clone(), step_name.clone()),
                    span,
                );
            }

            BorgEvent::TurnStarted {
                composition_id,
                step_name,
                turn_index,
                model,
                input_tokens,
            } => {
                let cached = self
                    .store
                    .get(&SpanKey::Step(composition_id.clone(), step_name.clone()));
                let span = self.with_resolved_parent(cached.as_ref(), |parent| turn_span!(parent));
                span.record(fields::TURN_INDEX, *turn_index as i64);
                span.record(fields::LLM_MODEL, &**model);
                span.record(fields::LLM_INPUT_TOKENS, *input_tokens as i64);
                {
                    let _guard = span.enter();
                    tracing::info!(target: "borg.turn", turn_index, %model, input_tokens, "turn started");
                }
                self.store.insert(
                    SpanKey::Turn(composition_id.clone(), step_name.clone(), *turn_index),
                    span,
                );
            }

            BorgEvent::TurnCompleted {
                composition_id,
                step_name,
                turn_index,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                first_token_ms,
            } => {
                if let Some(span) = self.store.get(&SpanKey::Turn(
                    composition_id.clone(),
                    step_name.clone(),
                    *turn_index,
                )) {
                    span.record(fields::LLM_OUTPUT_TOKENS, *output_tokens as i64);
                    span.record(fields::LLM_CACHE_READ, *cache_read_tokens as i64);
                    span.record(fields::LLM_CACHE_CREATION, *cache_creation_tokens as i64);
                    if let Some(ttft) = first_token_ms {
                        span.record(crate::fields::LLM_TTFT_MS, *ttft as i64);
                    }
                }
                self.store.remove(&SpanKey::Turn(
                    composition_id.clone(),
                    step_name.clone(),
                    *turn_index,
                ));
            }

            BorgEvent::ToolCalled {
                composition_id,
                step_name,
                turn_index,
                call_id,
                tool_name,
                ..
            } => {
                let cached = self.store.get(&SpanKey::Turn(
                    composition_id.clone(),
                    step_name.clone(),
                    *turn_index,
                ));
                let span = self.with_resolved_parent(cached.as_ref(), |parent| tool_span!(parent));
                span.record(fields::TOOL_NAME, &**tool_name);
                span.record(fields::TOOL_CALL_ID, &**call_id);
                {
                    let _guard = span.enter();
                    tracing::info!(target: "borg.tool", %tool_name, %call_id, "tool called");
                }
                self.store.insert(
                    SpanKey::ToolCall(
                        composition_id.clone(),
                        step_name.clone(),
                        *turn_index,
                        call_id.clone(),
                    ),
                    span,
                );
            }

            BorgEvent::ToolReturned {
                composition_id,
                step_name,
                turn_index,
                call_id,
                tool_name: _,
                success,
                error,
                ..
            } => {
                if let Some(span) = self.store.get(&SpanKey::ToolCall(
                    composition_id.clone(),
                    step_name.clone(),
                    *turn_index,
                    call_id.clone(),
                )) {
                    span.record(fields::TOOL_SUCCESS, *success);
                    if let Some(err) = error.as_deref() {
                        span.record(fields::TOOL_ERROR, err);
                    }
                }
                self.store.remove(&SpanKey::ToolCall(
                    composition_id.clone(),
                    step_name.clone(),
                    *turn_index,
                    call_id.clone(),
                ));
            }

            BorgEvent::GuardrailTriggered {
                composition_id,
                step_name,
                rule_name,
                verdict,
                signal,
            } => {
                let cached = self
                    .store
                    .get(&SpanKey::Step(composition_id.clone(), step_name.clone()));
                self.with_resolved_parent(cached.as_ref(), |parent| {
                    let _guard = parent.enter();
                    tracing::warn!(
                        target: "borg.guardrail",
                        %rule_name,
                        %verdict,
                        %signal,
                        "guardrail triggered"
                    );
                });
            }

            BorgEvent::StepRetrying {
                composition_id,
                step_name,
                attempt,
                ..
            } => {
                if let Some(span) = self
                    .store
                    .get(&SpanKey::Step(composition_id.clone(), step_name.clone()))
                {
                    span.record(fields::STEP_RETRY, true);
                }
                self.store
                    .remove(&SpanKey::Step(composition_id.clone(), step_name.clone()));
                if let Some(comp_span) = self
                    .store
                    .get(&SpanKey::Composition(composition_id.clone()))
                {
                    let span = step_span!(comp_span);
                    span.record(fields::STEP_NAME, &**step_name);
                    span.record(fields::STEP_INDEX, *attempt as i64);
                    self.store.insert(
                        SpanKey::Step(composition_id.clone(), step_name.clone()),
                        span,
                    );
                }
            }

            BorgEvent::StepCompleted {
                composition_id,
                step_name,
                success,
                failure_cause,
                ..
            } => {
                if let Some(span) = self
                    .store
                    .get(&SpanKey::Step(composition_id.clone(), step_name.clone()))
                {
                    span.record(
                        fields::STEP_OUTCOME,
                        if *success { "ok" } else { "failure" },
                    );
                    if let Some(cause) = failure_cause.as_deref() {
                        span.record(fields::STEP_FAILURE_CAUSE, cause);
                    }
                }
                self.store
                    .remove(&SpanKey::Step(composition_id.clone(), step_name.clone()));
            }

            BorgEvent::ParallelDegraded { composition_id, .. } => {
                if let Some(span) = self
                    .store
                    .get(&SpanKey::Composition(composition_id.clone()))
                {
                    span.record(crate::fields::PARALLEL_DEGRADED, true);
                }
            }

            BorgEvent::CompositionCompleted {
                composition_id,
                total_input_tokens,
                total_output_tokens,
                succeeded,
                failed,
                ..
            } => {
                if let Some(span) = self
                    .store
                    .get(&SpanKey::Composition(composition_id.clone()))
                {
                    span.record(
                        fields::COMPOSITION_TOTAL_INPUT_TOKENS,
                        *total_input_tokens as i64,
                    );
                    span.record(
                        fields::COMPOSITION_TOTAL_OUTPUT_TOKENS,
                        *total_output_tokens as i64,
                    );
                    span.record(fields::COMPOSITION_SUCCEEDED, *succeeded as i64);
                    span.record(fields::COMPOSITION_FAILED, *failed as i64);
                    let outcome = if *failed == 0 { "ok" } else { "failure" };
                    span.record(fields::COMPOSITION_OUTCOME, outcome);
                }
                self.store
                    .remove(&SpanKey::Composition(composition_id.clone()));
            }

            BorgEvent::Cancelled { composition_id, .. } => {
                if let Some(span) = self
                    .store
                    .get(&SpanKey::Composition(composition_id.clone()))
                {
                    span.record(fields::COMPOSITION_OUTCOME, "cancelled");
                }
                self.store
                    .remove(&SpanKey::Composition(composition_id.clone()));
            }

            BorgEvent::BranchDecided { .. } => {}

            // BorgEvent is `#[non_exhaustive]`.
            _ => {}
        }
    }

    /// Return the live `borg.turn` span so the executor can instrument
    /// agent invocations via `.instrument(turn_span)`.
    fn turn_span(&self, composition_id: &str, step_name: &str, turn_index: u32) -> tracing::Span {
        self.store
            .get_turn_span(composition_id, step_name, turn_index)
            .unwrap_or_else(tracing::Span::none)
    }

    /// Return the live `borg.tool` span so the executor can instrument
    /// `tool.execute(...)` via `.instrument(tool_span)`, nesting spans
    /// created inside tool bodies (outbound RPCs, sub-traces) under it.
    fn tool_span(
        &self,
        composition_id: &str,
        step_name: &str,
        turn_index: u32,
        call_id: &str,
    ) -> tracing::Span {
        self.store
            .get(&SpanKey::ToolCall(
                std::sync::Arc::from(composition_id),
                std::sync::Arc::from(step_name),
                turn_index,
                std::sync::Arc::from(call_id),
            ))
            .unwrap_or_else(tracing::Span::none)
    }
}
