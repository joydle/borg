//! Executor for streaming compositions.
//!
//! The decision record for the two composition trees (one-shot vs
//! streaming) lives in the public [`crate::streaming`] module docs.

use crate::composition::Composition;
use crate::error::BorgError;
use crate::events::{BorgEvent, CompositionShape};
use crate::executor::{execute_with_handoff, ExecCtx};
use crate::handoff::HandoffState;
use crate::streaming::{BorgStreamingAgent, StreamEvent};
use async_stream::stream;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use tracing::Instrument;

/// A composition that produces a stream. Distinct from
/// [`crate::Composition`] (which produces a single `ExecutionOutcome`).
#[non_exhaustive]
pub enum StreamingComposition {
    /// One streaming agent.
    Single {
        name: String,
        agent: Box<dyn BorgStreamingAgent>,
    },
    /// Run a streaming composition; on `Done`, run a single-shot
    /// composition with the streamed verdict pre-seeded into handoff under
    /// `output_key`. The post-step's events fire through `ctx.emitter`,
    /// NOT the top-level stream. The post-step completes before the
    /// top-level stream ends: with the streaming agent's `Done` on
    /// success, or `Failed` carrying the post-step's error. Models
    /// "main agent → post-processing".
    Then {
        primary: Box<StreamingComposition>,
        post: Composition,
        output_key: String,
    },
    /// Inverse of `Then`: run a single-shot composition first, seed its
    /// final verdict body into a handoff slot, then run the streaming
    /// composition. The pre-step's events fire through `ctx.emitter`; the
    /// visible top-level stream is everything from the streaming agent.
    /// Models "preflight (e.g. planner) → live streaming agent".
    ///
    /// `pre_handoff` is the pre-step's initial handoff state (empty via
    /// `with_pre`; supply your own via `with_pre_seeded`).
    Pre {
        pre: Composition,
        output_key: String,
        pre_handoff: HandoffState,
        primary: Box<StreamingComposition>,
    },
}

impl StreamingComposition {
    /// Build a streaming composition from one streaming agent.
    pub fn single(
        name: impl Into<String>,
        agent: impl BorgStreamingAgent + 'static,
    ) -> StreamingComposition {
        StreamingComposition::Single {
            name: name.into(),
            agent: Box::new(agent),
        }
    }

    /// Wrap a streaming composition with a follow-up single-shot
    /// composition. The follow-up runs after the primary's `Done`, before
    /// the stream yields it, and reads the streamed verdict from handoff
    /// under `output_key`. A failed follow-up ends the stream with
    /// `Failed` instead of `Done`.
    #[must_use]
    pub fn then(self, post: Composition, output_key: impl Into<String>) -> StreamingComposition {
        StreamingComposition::Then {
            primary: Box::new(self),
            post,
            output_key: output_key.into(),
        }
    }

    /// Run a single-shot pre-step before the streaming composition,
    /// writing its final verdict body into the streaming agent's handoff
    /// under `output_key`. The pre-step's events fire through
    /// `ctx.emitter`; only the streaming agent's events appear in the
    /// top-level stream.
    #[must_use]
    pub fn with_pre(self, pre: Composition, output_key: impl Into<String>) -> StreamingComposition {
        self.with_pre_seeded(pre, output_key, HandoffState::new())
    }

    /// Variant of [`Self::with_pre`] that supplies an initial handoff
    /// to the pre-step. Use when the pre-step's agents read input
    /// slots (e.g. a Planner that needs `user_message`).
    #[must_use]
    pub fn with_pre_seeded(
        self,
        pre: Composition,
        output_key: impl Into<String>,
        pre_handoff: HandoffState,
    ) -> StreamingComposition {
        StreamingComposition::Pre {
            pre,
            output_key: output_key.into(),
            pre_handoff,
            primary: Box::new(self),
        }
    }
}

/// Drive a streaming composition, returning the `StreamEvent` stream the
/// consumer iterates. Lifecycle events (start/complete) fire through
/// `ctx.emitter` independently.
///
/// Every lifecycle event and every post-step runs BEFORE the stream
/// yields its terminal `Done` / `Failed`, so a consumer that stops at the
/// first terminal event (as the borg-loop streaming runtime does) misses
/// nothing. The stream ends after its terminal event.
///
/// Supervision guarantees, per streaming agent:
///
/// - **Admission:** refused with `Failed(Cancelled)` if `ctx.cancel` has
///   already fired, or `Failed(BudgetExhausted)` if `ctx.budget` can't
///   admit a call (`Budget::can_invoke`, the same gate as
///   [`crate::BorgAgent::invoke`]). The agent is not dispatched.
/// - **Panics:** a panic in `invoke_streaming` or while polling its
///   stream surfaces as `Failed(BorgError::ChildPanicked)`; it never
///   unwinds into the consumer. The panicked stream is dropped, never
///   polled again.
/// - **Cancel:** every wait on the agent's stream races `ctx.cancel`; on
///   cancel the agent's stream is dropped and the consumer sees
///   `Failed(Cancelled)`. A synchronous loop inside one `poll` still
///   can't be interrupted.
/// - **Budget:** checked once, at admission. Tokens a streaming agent
///   spends mid-stream are its own to record; borg does not cut a live
///   stream off when the budget runs out.
///
/// One-shot pre/post steps (`Pre`, `Then`) run under the same panic
/// boundary: a panic there surfaces as `Failed(ChildPanicked)`.
pub fn execute_streaming(
    composition: StreamingComposition,
    ctx: ExecCtx,
) -> BoxStream<'static, StreamEvent> {
    execute_streaming_with_handoff(composition, ctx, HandoffState::new())
}

/// Map a caught panic payload to a `ChildPanicked` error.
fn panicked(agent_id: &str, payload: Box<dyn std::any::Any + Send>) -> BorgError {
    let message = if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<unknown panic payload>".to_string()
    };
    BorgError::ChildPanicked {
        agent_id: agent_id.to_string(),
        message,
    }
}

/// Run a one-shot pre/post step behind a panic boundary.
async fn execute_one_shot_supervised(
    step: Composition,
    ctx: ExecCtx,
    handoff: HandoffState,
    label: &str,
) -> Result<crate::executor::ExecutionOutcome, BorgError> {
    match AssertUnwindSafe(execute_with_handoff(step, ctx, handoff))
        .catch_unwind()
        .await
    {
        Ok(r) => r,
        Err(payload) => Err(panicked(label, payload)),
    }
}

/// Close a streaming step that failed before its stream started:
/// `StepCompleted { success: false }` then `CompositionCompleted`.
async fn emit_setup_failure(
    ctx: &ExecCtx,
    composition_id: Arc<str>,
    name: Arc<str>,
    model: Arc<str>,
    started: std::time::Instant,
    error: &BorgError,
) {
    ctx.emitter
        .emit(Arc::new(BorgEvent::StepCompleted {
            composition_id: composition_id.clone(),
            step_name: name,
            position: 0,
            duration_ms: started.elapsed().as_millis() as u64,
            input_tokens: 0,
            output_tokens: 0,
            success: false,
            failure_cause: Some(error.failure_cause().to_string()),
            turn_number: ctx.turn_number,
            model,
        }))
        .await;
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: 0,
            total_output_tokens: 0,
            succeeded: 0,
            failed: 1,
        }))
        .await;
}

/// Internal entry letting the `Pre` arm seed the streaming agent's
/// handoff. Public callers go through `execute_streaming` (empty handoff).
fn execute_streaming_with_handoff(
    composition: StreamingComposition,
    ctx: ExecCtx,
    seeded_handoff: HandoffState,
) -> BoxStream<'static, StreamEvent> {
    Box::pin(stream! {
        match composition {
            StreamingComposition::Single { name, agent } => {
                let name: Arc<str> = Arc::from(name);
                let composition_id: Arc<str> =
                    Arc::from(format!("{}::{}", ctx.parent_composition_id, name));
                let agent_model: Arc<str> = Arc::from(agent.model());
                let agent_id = agent.agent_id();
                ctx.emitter
                    .emit(Arc::new(BorgEvent::CompositionStarted {
                        composition_id: composition_id.clone(),
                        parent_id: Some(ctx.parent_composition_id.clone()),
                        shape: CompositionShape::Single,
                        branch_count: 1,
                    }))
                    .await;
                ctx.emitter
                    .emit(Arc::new(BorgEvent::StepStarted {
                        composition_id: composition_id.clone(),
                        step_name: name.clone(),
                        position: 0,
                        turn_number: ctx.turn_number,
                        model: agent_model.clone(),
                    }))
                    .await;
                let started = std::time::Instant::now();

                // Admission, before dispatch: the same budget gate the
                // one-shot path applies in `BorgAgent::invoke`.
                let refused = if ctx.cancel.is_cancelled() {
                    Some(BorgError::Cancelled)
                } else if !ctx.budget.can_invoke() {
                    Some(BorgError::BudgetExhausted {
                        needed: 0,
                        remaining: ctx.budget.remaining(),
                    })
                } else {
                    None
                };
                if let Some(e) = refused {
                    emit_setup_failure(
                        &ctx,
                        composition_id,
                        name,
                        agent_model,
                        started,
                        &e,
                    )
                    .await;
                    yield StreamEvent::Failed(e);
                    return;
                }

                // Owned, 'static context (streaming agents hold it across
                // the stream's lifetime). Handoff is seeded by an enclosing
                // `Pre`, or empty via `execute_streaming`.
                let agent_ctx = crate::context::AgentContext {
                    handoff: std::sync::Arc::new(seeded_handoff),
                    cancel: ctx.cancel.child_token(),
                    llm: ctx.llm.clone(),
                    budget: ctx.budget.clone(),
                    emitter: ctx.agent_emitter(),
                    parent_composition_id: composition_id.clone(),
                    step_name: name.clone(),
                    turn_number: ctx.turn_number,
                    parent_call_id: ctx.parent_call_id.clone(),
                    // Single-shot path: no tool-loop, no schemas to attach.
                    tool_schemas: std::sync::Arc::from(
                        [] as [crate::tools::ToolSchema; 0],
                    ),
                    guardrails: ctx.guardrails.clone(),
                };
                ctx.emitter
                    .emit(Arc::new(BorgEvent::TurnStarted {
                        composition_id: composition_id.clone(),
                        step_name: name.clone(),
                        turn_index: 0,
                        model: agent_model.clone(),
                        input_tokens: 0,
                    }))
                    .await;
                // Instrument with the live borg.turn span so spans created
                // inside the agent nest under borg.turn in OTLP exporters.
                // Panic boundary: a panicking `invoke_streaming` becomes
                // `ChildPanicked`. AssertUnwindSafe is sound: on a panic we
                // never touch the agent again.
                let turn_span = ctx.emitter.turn_span(&composition_id, &name, 0);
                let setup = match AssertUnwindSafe(
                    agent.invoke_streaming(agent_ctx).instrument(turn_span),
                )
                .catch_unwind()
                .await
                {
                    Ok(r) => r,
                    Err(payload) => Err(panicked(agent_id, payload)),
                };
                let mut inner = match setup {
                    Ok(s) => s,
                    Err(e) => {
                        emit_setup_failure(
                            &ctx,
                            composition_id,
                            name,
                            agent_model,
                            started,
                            &e,
                        )
                        .await;
                        yield StreamEvent::Failed(e);
                        return;
                    }
                };

                let mut input_tokens: u64 = 0;
                let mut output_tokens: u64 = 0;
                let mut cache_read_tokens: u64 = 0;
                let mut cache_creation_tokens: u64 = 0;
                // The terminal event is held back until the closing
                // lifecycle events have fired (see `execute_streaming`).
                let mut terminal: Option<StreamEvent> = None;
                loop {
                    // Race each poll against cancel, behind a panic
                    // boundary. Once either fires, `inner` is never
                    // polled again.
                    let polled = tokio::select! {
                        biased;
                        _ = ctx.cancel.cancelled() => {
                            terminal = Some(StreamEvent::Failed(BorgError::Cancelled));
                            break;
                        }
                        r = AssertUnwindSafe(inner.next()).catch_unwind() => r,
                    };
                    let ev = match polled {
                        Ok(Some(ev)) => ev,
                        // Stream ended without a terminal event: treat as complete.
                        Ok(None) => break,
                        Err(payload) => {
                            terminal = Some(StreamEvent::Failed(panicked(agent_id, payload)));
                            break;
                        }
                    };
                    match &ev {
                        // Hot path first: TextDelta dominates event volume
                        // (one per token); the loop yields it below.
                        StreamEvent::TextDelta(_) => {}
                        StreamEvent::Done(ref v) => {
                            input_tokens = v.input_tokens;
                            output_tokens = v.output_tokens;
                            cache_read_tokens = v.cache_read_tokens;
                            cache_creation_tokens = v.cache_creation_tokens;
                        }
                        StreamEvent::ToolCallStarted { call_id, name, .. } => {
                            let tool_name_arc: Arc<str> = Arc::from(name.as_str());
                            let call_id_arc: Arc<str> = Arc::from(call_id.as_str());
                            ctx.emitter
                                .emit(Arc::new(BorgEvent::ToolCalled {
                                    composition_id: composition_id.clone(),
                                    step_name: tool_name_arc.clone(),
                                    turn_index: 0,
                                    call_id: call_id_arc,
                                    tool_name: tool_name_arc,
                                    // No executor-level batch for streaming agents.
                                    batch_id: None,
                                }))
                                .await;
                            // The streaming agent dispatches tools internally,
                            // so we can't instrument that body — enter the
                            // borg.tool span briefly for this event only.
                            let tool_span = ctx.emitter.tool_span(&composition_id, name, 0, call_id);
                            tool_span.in_scope(|| {
                                tracing::debug!(target: "borg.tool", %call_id, %name, "streaming: tool dispatch started");
                            });
                        }
                        StreamEvent::ToolCallCompleted {
                            call_id,
                            name,
                            success: tool_success,
                            ..
                        } => {
                            let tool_name_arc: Arc<str> = Arc::from(name.as_str());
                            let call_id_arc: Arc<str> = Arc::from(call_id.as_str());
                            ctx.emitter
                                .emit(Arc::new(BorgEvent::ToolReturned {
                                    composition_id: composition_id.clone(),
                                    step_name: tool_name_arc.clone(),
                                    turn_index: 0,
                                    call_id: call_id_arc,
                                    tool_name: tool_name_arc,
                                    success: *tool_success,
                                    error: None,
                                    batch_id: None,
                                }))
                                .await;
                        }
                        _ => {}
                    }
                    if ev.is_terminal() {
                        terminal = Some(ev);
                        break;
                    }
                    yield ev;
                }
                // Drop the agent's stream before the closing events: on
                // cancel or panic this releases its in-flight futures.
                drop(inner);

                // Closing lifecycle events, emitted BEFORE the terminal
                // event is yielded so a consumer that stops there still
                // observes them.
                let failure_cause = match &terminal {
                    Some(StreamEvent::Failed(e)) => Some(e.failure_cause().to_string()),
                    _ => None,
                };
                let success = failure_cause.is_none();
                ctx.emitter
                    .emit(Arc::new(BorgEvent::TurnCompleted {
                        composition_id: composition_id.clone(),
                        step_name: name.clone(),
                        turn_index: 0,
                        output_tokens,
                        cache_read_tokens,
                        cache_creation_tokens,
                        first_token_ms: None,
                    }))
                    .await;
                ctx.emitter
                    .emit(Arc::new(BorgEvent::StepCompleted {
                        composition_id: composition_id.clone(),
                        step_name: name,
                        position: 0,
                        duration_ms: started.elapsed().as_millis() as u64,
                        input_tokens,
                        output_tokens,
                        success,
                        failure_cause,
                        turn_number: ctx.turn_number,
                        model: agent_model,
                    }))
                    .await;
                ctx.emitter
                    .emit(Arc::new(BorgEvent::CompositionCompleted {
                        composition_id,
                        total_duration_ms: started.elapsed().as_millis() as u64,
                        total_input_tokens: input_tokens,
                        total_output_tokens: output_tokens,
                        succeeded: if success { 1 } else { 0 },
                        failed: if success { 0 } else { 1 },
                    }))
                    .await;
                if let Some(ev) = terminal {
                    yield ev;
                }
            }

            StreamingComposition::Then { primary, post, output_key } => {
                // Forward the primary's non-terminal events; hold back its
                // terminal event so the post-step runs BEFORE the consumer
                // sees the end of the stream.
                let mut inner = execute_streaming(*primary, ctx.clone());
                let mut final_verdict: Option<crate::Verdict> = None;
                while let Some(ev) = inner.next().await {
                    match ev {
                        StreamEvent::Done(v) => {
                            final_verdict = Some(v);
                            break;
                        }
                        StreamEvent::Failed(e) => {
                            // Primary failed: the post-step never runs.
                            yield StreamEvent::Failed(e);
                            return;
                        }
                        other => yield other,
                    }
                }
                drop(inner);
                let Some(verdict) = final_verdict else {
                    // Primary ended without a terminal event; nothing to
                    // hand to the post-step.
                    return;
                };
                if ctx.cancel.is_cancelled() {
                    // Cancel between primary and post-step: the composition
                    // didn't complete, so don't report `Done`.
                    yield StreamEvent::Failed(BorgError::Cancelled);
                    return;
                }

                // Run the post-step; its events go through ctx.emitter,
                // not the top-level stream. Its failure replaces `Done`.
                let mut handoff = HandoffState::new();
                handoff.set(output_key, verdict.body.clone());
                match execute_one_shot_supervised(post, ctx, handoff, "then_post_step").await {
                    Ok(_) => yield StreamEvent::Done(verdict),
                    Err(e) => yield StreamEvent::Failed(e),
                }
            }

            StreamingComposition::Pre { pre, output_key, pre_handoff, primary } => {
                // 1. Run the pre-step to completion; its events fire
                //    through ctx.emitter, not the top-level stream.
                if ctx.cancel.is_cancelled() {
                    yield StreamEvent::Failed(BorgError::Cancelled);
                    return;
                }
                let pre_outcome = match execute_one_shot_supervised(
                    pre,
                    ctx.clone(),
                    pre_handoff,
                    "pre_step",
                )
                .await
                {
                    Ok(o) => o,
                    Err(e) => {
                        yield StreamEvent::Failed(e);
                        return;
                    }
                };
                if ctx.cancel.is_cancelled() {
                    yield StreamEvent::Failed(BorgError::Cancelled);
                    return;
                }
                // 2. Capture the final verdict body for the streaming
                //    agent's `output_key` slot. Reuse the pre-step's merged
                //    handoff so its internal output_keys are preserved.
                let mut seeded = pre_outcome.handoff;
                let pre_body = pre_outcome
                    .verdicts
                    .last()
                    .map(|nv| nv.verdict.body.clone())
                    .unwrap_or_default();
                seeded.set(output_key, pre_body);
                // 3. Drive the streaming primary with the seeded handoff.
                let mut inner = execute_streaming_with_handoff(*primary, ctx, seeded);
                while let Some(ev) = inner.next().await {
                    let is_terminal = ev.is_terminal();
                    yield ev;
                    if is_terminal {
                        break;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::BorgError;
    use crate::events::{NoopEmitter, RecordingEmitter};
    use crate::Verdict;
    use async_trait::async_trait;
    use futures::stream::StreamExt;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    struct ScriptedStreaming {
        text_chunks: Vec<&'static str>,
    }

    #[async_trait]
    impl BorgStreamingAgent for ScriptedStreaming {
        fn agent_id(&self) -> &'static str {
            "scripted"
        }
        fn model(&self) -> &str {
            "fake/test"
        }
        async fn invoke_streaming(
            &self,
            _ctx: crate::context::AgentContext,
        ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
            let chunks: Vec<String> = self.text_chunks.iter().map(|s| s.to_string()).collect();
            Ok(Box::pin(stream! {
                let mut concat = String::new();
                for c in chunks {
                    concat.push_str(&c);
                    yield StreamEvent::TextDelta(c);
                }
                yield StreamEvent::Done(Verdict::with_token("", concat, 100, 50));
            }))
        }
    }

    fn ctx_for() -> ExecCtx {
        ExecCtx::new(
            // Streaming agent doesn't use the LLM here; ExecCtx needs one.
            Arc::new(NoLlm),
            crate::in_memory_budget(10_000),
            Arc::new(NoopEmitter),
            CancellationToken::new(),
            "test",
        )
    }

    struct NoLlm;
    #[async_trait]
    impl crate::AgentLlmClient for NoLlm {
        async fn chat_completion(
            &self,
            _r: crate::ChatRequest,
        ) -> Result<crate::ChatResponse, BorgError> {
            unimplemented!("NoLlm: streaming agent shouldn't reach this in this test")
        }
    }

    #[tokio::test]
    async fn streaming_single_yields_text_then_done() {
        let comp = StreamingComposition::single(
            "scripted",
            ScriptedStreaming {
                text_chunks: vec!["hello, ", "world"],
            },
        );
        let mut events: Vec<StreamEvent> = execute_streaming(comp, ctx_for()).collect().await;
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], StreamEvent::TextDelta(_)));
        assert!(matches!(events[1], StreamEvent::TextDelta(_)));
        let last = events.pop().unwrap();
        match last {
            StreamEvent::Done(v) => {
                assert_eq!(&*v.body, "hello, world");
                assert_eq!(v.input_tokens, 100);
                assert_eq!(v.output_tokens, 50);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_pre_runs_single_shot_then_seeds_streaming_handoff() {
        // Echoes its "seeded_input" handoff slot as a TextDelta, proving
        // the Pre arm wired the pre-step's verdict into the agent's context.
        struct EchoHandoff;
        #[async_trait]
        impl BorgStreamingAgent for EchoHandoff {
            fn agent_id(&self) -> &'static str {
                "echo_handoff"
            }
            fn model(&self) -> &str {
                "fake/echo"
            }
            async fn invoke_streaming(
                &self,
                ctx: crate::context::AgentContext,
            ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
                let seeded: String = ctx
                    .handoff
                    .get_as("seeded_input")
                    .unwrap_or_else(|| "<empty>".into());
                Ok(Box::pin(stream! {
                    yield StreamEvent::TextDelta(format!("echo:{seeded}"));
                    yield StreamEvent::Done(Verdict::with_token("", "echo done", 0, 0));
                }))
            }
        }

        // Pre-step whose final verdict body becomes the streaming agent's
        // `seeded_input` slot.
        struct Stamp;
        #[async_trait]
        impl crate::BorgAgent for Stamp {
            fn agent_id(&self) -> &'static str {
                "stamp"
            }
            fn model(&self) -> &str {
                "deterministic"
            }
            fn known_verdicts(&self) -> &'static [&'static str] {
                &["DONE"]
            }
            async fn invoke(
                &self,
                _ctx: &crate::context::AgentContext,
            ) -> Result<Verdict, BorgError> {
                Ok(Verdict::with_token("DONE", "from-pre", 0, 0))
            }
        }

        let primary = StreamingComposition::single("echo", EchoHandoff);
        let pre = Composition::single("stamp", Stamp);
        let comp = primary.with_pre(pre, "seeded_input");

        let events: Vec<StreamEvent> = execute_streaming(comp, ctx_for()).collect().await;
        // First event is the echoed seeded input: Pre ran first and its
        // verdict landed in the agent's handoff under the configured key.
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "echo:from-pre"));
        assert!(matches!(events.last(), Some(StreamEvent::Done(_))));
    }

    #[tokio::test]
    async fn streaming_then_runs_post_step_after_done() {
        // Checks the visible stream ends with the primary's `Done`; the
        // post-step's effect is covered by integration tests.
        let primary = StreamingComposition::single(
            "scripted",
            ScriptedStreaming {
                text_chunks: vec!["before-post"],
            },
        );
        // Degenerate empty post-step (sequential of zero steps) — avoids
        // needing a non-LLM single-shot agent here.
        let post = Composition::sequential(Vec::<Composition>::new());

        let comp = primary.then(post, "primary_out");
        let events: Vec<StreamEvent> = execute_streaming(comp, ctx_for()).collect().await;

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::TextDelta(_)));
        assert!(matches!(events.last(), Some(StreamEvent::Done(_))));
    }

    #[tokio::test]
    async fn streaming_failed_event_terminates_stream() {
        struct Failer;
        #[async_trait]
        impl BorgStreamingAgent for Failer {
            fn agent_id(&self) -> &'static str {
                "failer"
            }
            fn model(&self) -> &str {
                "fake/failer"
            }
            async fn invoke_streaming(
                &self,
                _ctx: crate::context::AgentContext,
            ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
                Ok(Box::pin(stream! {
                    yield StreamEvent::TextDelta("partial".into());
                    yield StreamEvent::Failed(BorgError::Internal {
                        message: "deliberate".into(),
                    });
                    // Anything after Failed is ignored.
                    yield StreamEvent::TextDelta("never seen".into());
                }))
            }
        }

        let comp = StreamingComposition::single("failer", Failer);
        let events: Vec<StreamEvent> = execute_streaming(comp, ctx_for()).collect().await;

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::TextDelta(_)));
        assert!(matches!(events[1], StreamEvent::Failed(_)));
    }

    /// Regression: the streaming executor must translate
    /// `ToolCallStarted`/`ToolCallCompleted` into
    /// `BorgEvent::ToolCalled`/`ToolReturned` so `borg.tool` spans appear
    /// in OTLP traces for streaming agents.
    #[tokio::test]
    async fn streaming_executor_emits_tool_called_and_returned() {
        struct ToolAgent;
        #[async_trait]
        impl BorgStreamingAgent for ToolAgent {
            fn agent_id(&self) -> &'static str {
                "tool_agent"
            }
            fn model(&self) -> &str {
                "fake/tool"
            }
            async fn invoke_streaming(
                &self,
                _ctx: crate::context::AgentContext,
            ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
                Ok(Box::pin(stream! {
                    yield StreamEvent::ToolCallStarted {
                        call_id: "c1".into(),
                        name: "read_file".into(),
                        args: serde_json::json!({"path": "/foo"}),
                    };
                    yield StreamEvent::ToolCallCompleted {
                        call_id: "c1".into(),
                        name: "read_file".into(),
                        result: serde_json::json!({"content": "hello"}),
                        success: true,
                    };
                    yield StreamEvent::Done(Verdict::with_token("", "done", 10, 5));
                }))
            }
        }

        let recorder = Arc::new(RecordingEmitter::new());
        let ctx = ExecCtx::new(
            Arc::new(NoLlm),
            crate::in_memory_budget(10_000),
            recorder.clone(),
            CancellationToken::new(),
            "test",
        );

        let _events: Vec<StreamEvent> =
            execute_streaming(StreamingComposition::single("tool_agent", ToolAgent), ctx)
                .collect()
                .await;

        let kinds = recorder.kinds();
        assert!(
            kinds.contains(&"tool_called"),
            "expected tool_called in emitter, got: {kinds:?}"
        );
        assert!(
            kinds.contains(&"tool_returned"),
            "expected tool_returned in emitter, got: {kinds:?}"
        );
    }

    // ── Supervision regressions ─────────────────────────────────────────

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Consume like the borg-loop streaming runtime: stop at the first
    /// terminal event and drop the stream without polling again.
    async fn collect_until_terminal(mut s: BoxStream<'static, StreamEvent>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        while let Some(ev) = s.next().await {
            let terminal = ev.is_terminal();
            out.push(ev);
            if terminal {
                break;
            }
        }
        out
    }

    struct PanicsOnInvoke;
    #[async_trait]
    impl BorgStreamingAgent for PanicsOnInvoke {
        fn agent_id(&self) -> &'static str {
            "panics_on_invoke"
        }
        fn model(&self) -> &str {
            "fake/panic"
        }
        async fn invoke_streaming(
            &self,
            _ctx: crate::context::AgentContext,
        ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
            panic!("invoke boom");
        }
    }

    struct PanicsMidStream;
    #[async_trait]
    impl BorgStreamingAgent for PanicsMidStream {
        fn agent_id(&self) -> &'static str {
            "panics_mid_stream"
        }
        fn model(&self) -> &str {
            "fake/panic"
        }
        async fn invoke_streaming(
            &self,
            _ctx: crate::context::AgentContext,
        ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
            Ok(Box::pin(stream! {
                yield StreamEvent::TextDelta("partial".into());
                panic!("stream boom");
            }))
        }
    }

    #[tokio::test]
    async fn streaming_single_panic_in_invoke_yields_failed() {
        let recorder = Arc::new(RecordingEmitter::new());
        let ctx = ExecCtx::new(
            Arc::new(NoLlm),
            crate::in_memory_budget(10_000),
            recorder.clone(),
            CancellationToken::new(),
            "test",
        );
        let comp = StreamingComposition::single("p", PanicsOnInvoke);
        let events: Vec<StreamEvent> = execute_streaming(comp, ctx).collect().await;
        assert_eq!(events.len(), 1, "got {events:?}");
        match &events[0] {
            StreamEvent::Failed(BorgError::ChildPanicked { agent_id, message }) => {
                assert_eq!(agent_id, "panics_on_invoke");
                assert!(message.contains("invoke boom"), "message: {message}");
            }
            other => panic!("expected Failed(ChildPanicked), got {other:?}"),
        }
        let kinds = recorder.kinds();
        assert!(kinds.contains(&"step_completed"), "{kinds:?}");
        assert!(kinds.contains(&"composition_completed"), "{kinds:?}");
    }

    #[tokio::test]
    async fn streaming_single_panic_mid_stream_yields_failed() {
        let comp = StreamingComposition::single("p", PanicsMidStream);
        let events: Vec<StreamEvent> = execute_streaming(comp, ctx_for()).collect().await;
        assert_eq!(events.len(), 2, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "partial"));
        match &events[1] {
            StreamEvent::Failed(BorgError::ChildPanicked { agent_id, message }) => {
                assert_eq!(agent_id, "panics_mid_stream");
                assert!(message.contains("stream boom"), "message: {message}");
            }
            other => panic!("expected Failed(ChildPanicked), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_single_refuses_to_start_on_exhausted_budget() {
        struct Tracked(Arc<AtomicBool>);
        #[async_trait]
        impl BorgStreamingAgent for Tracked {
            fn agent_id(&self) -> &'static str {
                "tracked"
            }
            fn model(&self) -> &str {
                "fake/tracked"
            }
            async fn invoke_streaming(
                &self,
                _ctx: crate::context::AgentContext,
            ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(Box::pin(stream! {
                    yield StreamEvent::Done(Verdict::empty());
                }))
            }
        }

        let invoked = Arc::new(AtomicBool::new(false));
        let ctx = ExecCtx::new(
            Arc::new(NoLlm),
            crate::in_memory_budget(0),
            Arc::new(NoopEmitter),
            CancellationToken::new(),
            "test",
        );
        let comp = StreamingComposition::single("t", Tracked(invoked.clone()));
        let events: Vec<StreamEvent> = execute_streaming(comp, ctx).collect().await;
        assert_eq!(events.len(), 1, "got {events:?}");
        assert!(
            matches!(
                events[0],
                StreamEvent::Failed(BorgError::BudgetExhausted { remaining: 0, .. })
            ),
            "got {events:?}"
        );
        assert!(
            !invoked.load(Ordering::SeqCst),
            "agent must not be dispatched"
        );
    }

    #[tokio::test]
    async fn streaming_single_cancel_interrupts_a_stalled_stream() {
        struct Stalls;
        #[async_trait]
        impl BorgStreamingAgent for Stalls {
            fn agent_id(&self) -> &'static str {
                "stalls"
            }
            fn model(&self) -> &str {
                "fake/stall"
            }
            async fn invoke_streaming(
                &self,
                _ctx: crate::context::AgentContext,
            ) -> Result<BoxStream<'static, StreamEvent>, BorgError> {
                // Ignores ctx.cancel entirely.
                Ok(Box::pin(stream! {
                    yield StreamEvent::TextDelta("one".into());
                    futures::future::pending::<()>().await;
                    yield StreamEvent::Done(Verdict::empty());
                }))
            }
        }

        let cancel = CancellationToken::new();
        let ctx = ExecCtx::new(
            Arc::new(NoLlm),
            crate::in_memory_budget(10_000),
            Arc::new(NoopEmitter),
            cancel.clone(),
            "test",
        );
        let mut s = execute_streaming(StreamingComposition::single("s", Stalls), ctx);
        assert!(matches!(s.next().await, Some(StreamEvent::TextDelta(_))));
        cancel.cancel();
        let next = tokio::time::timeout(std::time::Duration::from_secs(2), s.next())
            .await
            .expect("cancel must interrupt a stalled stream");
        assert!(
            matches!(next, Some(StreamEvent::Failed(BorgError::Cancelled))),
            "got {next:?}"
        );
    }

    /// One-shot post-step that counts its invocations and optionally fails.
    struct PostStep {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }
    #[async_trait]
    impl crate::BorgAgent for PostStep {
        fn agent_id(&self) -> &'static str {
            "post_step"
        }
        fn model(&self) -> &str {
            "deterministic"
        }
        fn known_verdicts(&self) -> &'static [&'static str] {
            &["DONE"]
        }
        async fn invoke(&self, ctx: &crate::context::AgentContext) -> Result<Verdict, BorgError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let seen: Option<String> = ctx.handoff.get_as("primary_out");
            assert_eq!(seen.as_deref(), Some("before-post"));
            if self.fail {
                return Err(BorgError::Internal {
                    message: "post boom".into(),
                });
            }
            Ok(Verdict::with_token("DONE", "post", 0, 0))
        }
    }

    #[tokio::test]
    async fn streaming_then_runs_post_before_terminal_done() {
        let calls = Arc::new(AtomicUsize::new(0));
        let recorder = Arc::new(RecordingEmitter::new());
        let ctx = ExecCtx::new(
            Arc::new(NoLlm),
            crate::in_memory_budget(10_000),
            recorder.clone(),
            CancellationToken::new(),
            "test",
        );
        let comp = StreamingComposition::single(
            "scripted",
            ScriptedStreaming {
                text_chunks: vec!["before-post"],
            },
        )
        .then(
            Composition::single(
                "post",
                PostStep {
                    calls: calls.clone(),
                    fail: false,
                },
            ),
            "primary_out",
        );
        let events = collect_until_terminal(execute_streaming(comp, ctx)).await;
        assert!(
            matches!(events.last(), Some(StreamEvent::Done(_))),
            "{events:?}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "post-step must run before Done"
        );
        // Every closing event fired before the consumer saw `Done`: the
        // streaming step's turn/step/composition completions, then the
        // post-step's step/composition completions.
        let kinds = recorder.kinds();
        assert!(kinds.contains(&"turn_completed"), "{kinds:?}");
        for k in ["step_completed", "composition_completed"] {
            assert_eq!(
                kinds.iter().filter(|x| **x == k).count(),
                2,
                "expected {k} for both the streaming step and the post-step: {kinds:?}"
            );
        }
    }

    #[tokio::test]
    async fn streaming_then_failed_post_surfaces_its_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let comp = StreamingComposition::single(
            "scripted",
            ScriptedStreaming {
                text_chunks: vec!["before-post"],
            },
        )
        .then(
            Composition::single(
                "post",
                PostStep {
                    calls: calls.clone(),
                    fail: true,
                },
            ),
            "primary_out",
        );
        let events = collect_until_terminal(execute_streaming(comp, ctx_for())).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        match events.last() {
            Some(StreamEvent::Failed(e)) => {
                assert!(e.to_string().contains("post boom"), "error: {e}");
            }
            other => panic!("expected Failed(post error), got {other:?}"),
        }
        assert!(
            !events.iter().any(|e| matches!(e, StreamEvent::Done(_))),
            "a failed post-step must not also report Done: {events:?}"
        );
    }
}
