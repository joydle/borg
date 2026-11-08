//! End-to-end evals proving every built-in guardrail rule fires from
//! signals the executor auto-emits — no hand-rolled instrumentation,
//! no manual signal recording.
//!
//! Run via:
//! ```text
//! cargo test -p borg --test guardrails_evals --features testing
//! ```
//!
//! Each scenario:
//! 1. Builds a [`GuardrailHarness`] with one rule.
//! 2. Runs an `agent_loop` composition through the borg executor.
//! 3. Asserts on `BorgError::Guardrail` (halt) or on the harness's
//!    [`BoundaryMessageQueue`] (warn / throttle).
//!
//! Gated on `feature = "testing"` — the fixtures (`FakeOutcome`, `eval`
//! builder) live behind that flag.

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::guardrails::rules::*;
use borg::prelude::*;
use borg::testing::FakeOutcome;
use borg::testing::{eval, EvalAssert};
use borg::tools::{
    DispatchedCall, MutexDispatcher, Registry, Tool, ToolCallRequest, ToolCallResult, ToolDispatch,
    ToolSchema,
};
use std::sync::Arc;

// ── Fixtures ────────────────────────────────────────────────────────────

/// Test agent that emits `calls_per_step` tool calls every turn, driving
/// an agent loop that keeps calling the tool until the LLM script runs out,
/// or until a turn's text is `"done"` (its final answer).
struct RepeaterAgent {
    tool_name: &'static str,
    calls_per_step: usize,
}

#[async_trait]
impl BorgAgent for RepeaterAgent {
    fn agent_id(&self) -> &'static str {
        "repeater"
    }
    fn model(&self) -> &str {
        "fake/repeater"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("call the tool".into())
    }
    fn extract_tool_calls(&self, verdict: &borg::Verdict) -> Vec<ToolCallRequest> {
        if &*verdict.body == "done" {
            return Vec::new();
        }
        (0..self.calls_per_step)
            .map(|i| {
                ToolCallRequest::new(
                    self.tool_name,
                    serde_json::json!({ "i": i }),
                    format!("rc-{i}"),
                )
            })
            .collect()
    }
}

/// Test agent: always emits zero tool calls — every turn is "text-only."
struct TextOnlyAgent;

#[async_trait]
impl BorgAgent for TextOnlyAgent {
    fn agent_id(&self) -> &'static str {
        "text_only"
    }
    fn model(&self) -> &str {
        "fake/text_only"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("hi".into())
    }
}

/// Test agent that records the agent context's `effective_max_tokens`
/// at each invocation into a shared cell, so tests can prove the
/// `Throttle` clamp actually reached the agent.
struct ThrottleProbeAgent {
    observed: Arc<std::sync::Mutex<Vec<u32>>>,
    tool_name: &'static str,
    fire_tool: std::sync::atomic::AtomicBool,
}

#[async_trait]
impl BorgAgent for ThrottleProbeAgent {
    fn agent_id(&self) -> &'static str {
        "throttle_probe"
    }
    fn model(&self) -> &str {
        "fake/throttle_probe"
    }
    fn max_tokens(&self) -> u32 {
        8_000
    }
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        // Probe at prompt-build time — the path the executor uses for
        // max_tokens.
        let eff = ctx.effective_max_tokens(self.max_tokens());
        self.observed.lock().unwrap().push(eff);
        Some(format!("turn (effective={eff})"))
    }
    fn extract_tool_calls(&self, _verdict: &borg::Verdict) -> Vec<ToolCallRequest> {
        // One tool call on turn 0 so the loop runs a second turn, letting
        // turn 0's throttle reach turn 1.
        if self
            .fire_tool
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            vec![ToolCallRequest::new(
                self.tool_name,
                serde_json::json!({}),
                "probe-0",
            )]
        } else {
            Vec::new()
        }
    }
}

/// Tool that always succeeds. Counts invocations via mutable ctx.
struct OkTool {
    name: &'static str,
}

#[async_trait]
impl Tool for OkTool {
    type Context = i32;
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name,
            "Test ok tool",
            serde_json::json!({ "type": "object" }),
        )
    }
    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &mut Self::Context,
    ) -> Result<ToolCallResult, BorgError> {
        *ctx += 1;
        ToolCallResult::ok(serde_json::json!({ "ok": true }))
    }
}

/// Tool that always returns a logical failure (`ToolCallResult::fail`),
/// driving `ConsecutiveFailureRule`.
struct FailingTool {
    name: &'static str,
}

#[async_trait]
impl Tool for FailingTool {
    type Context = i32;
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name,
            "Test failing tool",
            serde_json::json!({ "type": "object" }),
        )
    }
    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &mut Self::Context,
    ) -> Result<ToolCallResult, BorgError> {
        *ctx += 1;
        Ok(ToolCallResult::fail("simulated tool failure").with_summary("fail"))
    }
}

fn dispatcher_for<T>(tool: T) -> Arc<dyn ToolDispatch>
where
    T: Tool<Context = i32> + 'static,
{
    let registry = Registry::<i32>::new().with(tool);
    Arc::new(MutexDispatcher::new(registry, 0))
}

/// Build an LLM script of `n` `ok` outcomes — fuel for a long agent loop.
fn ok_script(n: usize) -> Vec<FakeOutcome> {
    (0..n)
        .map(|i| FakeOutcome::ok(format!("turn {i}")))
        .collect()
}

/// LLM script that returns the SAME text on every call — drives
/// IdenticalOutputRule.
fn repeating_script(n: usize, text: &str) -> Vec<FakeOutcome> {
    (0..n).map(|_| FakeOutcome::ok(text)).collect()
}

// ── Evals ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn max_continuations_rule_halts_after_limit() {
    // 3-turn cap; agent keeps emitting tool calls indefinitely.
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(MaxContinuationsRule::new(3)),
    ));

    let result = eval("max_continuations_halts")
        .composition(borg::dsl::agent_loop(
            RepeaterAgent {
                tool_name: "echo",
                calls_per_step: 1,
            },
            dispatcher_for(OkTool { name: "echo" }),
            20, // executor cap higher than the rule's
        ))
        .outcomes(ok_script(20))
        .guardrails(harness.clone())
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();

    assert!(result.guardrails.is_some());
    let h = result.guardrails.as_ref().unwrap();
    let turns = h
        .detector
        .count("agent:turn_completed", std::time::Duration::from_secs(60));
    assert_eq!(
        turns, 3,
        "expected exactly 3 turns recorded before halt, saw {turns}"
    );
}

#[tokio::test]
async fn text_only_end_turn_rule_halts_after_n_turns() {
    // Agent returns 0 tool calls, so the executor exits the loop on the
    // first text-only turn. halt_after=1 fires on that turn.
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(TextOnlyEndTurnRule::new(1)),
    ));

    // Must be `agent_loop`, not `step`: guardrails evaluate at the
    // agent-loop boundary, which a single agent doesn't have.
    eval("text_only_halts")
        .composition(borg::dsl::agent_loop(
            TextOnlyAgent,
            dispatcher_for(OkTool { name: "noop" }),
            5,
        ))
        .responses(["text"])
        .guardrails(harness)
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();
}

#[tokio::test]
async fn repeated_tool_call_rule_halts_on_spam() {
    // Agent calls `read_file` every turn; rule halts at 4 calls.
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(RepeatedToolCallRule::new("read_file", 2, 3, 4)),
    ));

    let result = eval("repeated_tool_halts")
        .composition(borg::dsl::agent_loop(
            RepeaterAgent {
                tool_name: "read_file",
                calls_per_step: 1,
            },
            dispatcher_for(OkTool { name: "read_file" }),
            10,
        ))
        .outcomes(ok_script(10))
        .guardrails(harness.clone())
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();

    let h = result.guardrails.as_ref().unwrap();
    let n = h
        .detector
        .count("tool:read_file:invoked", std::time::Duration::from_secs(60));
    assert!(n >= 4, "expected >= 4 invoked signals, saw {n}");
}

#[tokio::test]
async fn consecutive_failure_rule_halts_on_repeated_tool_failure() {
    // Tool always returns ToolCallResult::error. Rule halts after 3
    // consecutive failures.
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(ConsecutiveFailureRule::new("flaky", 2, 3)),
    ));

    let result = eval("consecutive_failure_halts")
        .composition(borg::dsl::agent_loop(
            RepeaterAgent {
                tool_name: "flaky",
                calls_per_step: 1,
            },
            dispatcher_for(FailingTool { name: "flaky" }),
            10,
        ))
        .outcomes(ok_script(10))
        .guardrails(harness.clone())
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();

    let h = result.guardrails.as_ref().unwrap();
    // Assert via the same `consecutive_within` semantic the rule uses;
    // a plain consecutive count would be broken by interleaved
    // `tool:flaky:invoked` signals.
    assert!(
        h.detector
            .consecutive_within("tool:flaky:failed", &["tool:flaky:succeeded"])
            >= 3
    );
}

#[tokio::test]
async fn identical_output_rule_halts_on_model_repeating_itself() {
    // Same text every turn. The executor auto-emits `model:text` signals
    // hashing verdict.body, so IdenticalOutputRule sees the repetition.
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(IdenticalOutputRule::new("model:text", 3)),
    ));

    eval("identical_output_halts")
        .composition(borg::dsl::agent_loop(
            RepeaterAgent {
                tool_name: "echo",
                calls_per_step: 1,
            },
            dispatcher_for(OkTool { name: "echo" }),
            10,
        ))
        .outcomes(repeating_script(10, "stuck"))
        .guardrails(harness)
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();
}

#[tokio::test]
async fn runaway_cost_rule_halts_near_budget_exhaustion() {
    // 150 tokens/call, 600-token budget, halt at 50% → halt after 2 turns.
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(RunawayCostRule::new(0.5)),
    ));

    eval("runaway_cost_halts")
        .composition(borg::dsl::agent_loop(
            RepeaterAgent {
                tool_name: "echo",
                calls_per_step: 1,
            },
            dispatcher_for(OkTool { name: "echo" }),
            20,
        ))
        .outcomes(ok_script(20))
        .budget(600)
        .guardrails(harness)
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();
}

#[tokio::test]
async fn warn_messages_reach_boundary_queue_without_halting() {
    // Warn fires on turn 3 (halt disabled). The loop completes normally
    // and the queue holds the corrective message for the next prompt.
    let harness = Arc::new(GuardrailHarness::new(GuardrailEngine::new().with_rule(
        RepeatedToolCallRule::new(
            "echo",
            3,        // warn_at
            u32::MAX, // throttle disabled
            u32::MAX, // halt disabled
        ),
    )));

    let result = eval("warn_reaches_queue")
        .composition(borg::dsl::agent_loop(
            // 4 tool-call turns, then text-only (stops loop). Plenty
            // of warn-fires.
            RepeaterAgent {
                tool_name: "echo",
                calls_per_step: 1,
            },
            dispatcher_for(OkTool { name: "echo" }),
            5,
        ))
        .outcomes({
            let mut script = ok_script(4);
            script.push(FakeOutcome::ok("done"));
            script
        })
        .guardrails(harness.clone())
        .run()
        .await
        .assert_ok();

    assert!(
        result.outcome.is_ok(),
        "warn must not terminate the composition"
    );
    let pending = harness.queue.drain();
    assert!(
        !pending.is_empty(),
        "queue should have boundary messages from the warning rule"
    );
    assert!(pending
        .iter()
        .any(|m| matches!(m.kind, BoundaryMessageKind::Warn)));
}

#[tokio::test]
async fn throttle_clamps_effective_max_tokens_on_next_turn() {
    // Throttle at 1 invocation; probe agent records effective_max_tokens
    // at every prompt-build.
    let observed = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
    let harness = Arc::new(GuardrailHarness::new(GuardrailEngine::new().with_rule(
        RepeatedToolCallRule::new(
            "probe",
            u32::MAX, // warn disabled
            1,        // throttle on first invoke
            u32::MAX, // halt disabled
        ),
    )));

    let agent = ThrottleProbeAgent {
        observed: observed.clone(),
        tool_name: "probe",
        fire_tool: std::sync::atomic::AtomicBool::new(true),
    };

    eval("throttle_clamps")
        .composition(borg::dsl::agent_loop(
            agent,
            dispatcher_for(OkTool { name: "probe" }),
            3,
        ))
        .outcomes(ok_script(3))
        .guardrails(harness)
        .run()
        .await
        .assert_ok();

    // Turn 0: requested 8000, no override → effective 8000.
    // Turn 1: throttle fired at end of turn 0 → effective 2000
    //         (RepeatedToolCallRule's default `throttle_max_tokens`).
    let obs = observed.lock().unwrap();
    assert!(
        obs.len() >= 2,
        "expected at least 2 turns, saw {}",
        obs.len()
    );
    assert_eq!(obs[0], 8_000, "turn 0 should not be clamped");
    assert_eq!(obs[1], 2_000, "turn 1 should be clamped to 2000");
}

#[tokio::test]
async fn tool_dispatch_auto_signals_distinguish_succeeded_and_failed() {
    // No rule registered — just proving the executor wires up the
    // succeeded/failed signals correctly.
    let harness = Arc::new(GuardrailHarness::new(GuardrailEngine::new()));

    // step 0 calls an ok tool, step 1 calls a failing one. RepeaterAgent
    // always emits the same tool name, so use a custom agent to mix.
    struct MixedAgent {
        emitted: std::sync::atomic::AtomicUsize,
    }
    #[async_trait]
    impl BorgAgent for MixedAgent {
        fn agent_id(&self) -> &'static str {
            "mixed"
        }
        fn model(&self) -> &str {
            "fake/mixed"
        }
        fn max_tokens(&self) -> u32 {
            100
        }
        fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
            Some("go".into())
        }
        fn extract_tool_calls(&self, _v: &borg::Verdict) -> Vec<ToolCallRequest> {
            let n = self
                .emitted
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                vec![ToolCallRequest::new("ok", serde_json::json!({}), "c0")]
            } else if n == 1 {
                vec![ToolCallRequest::new("fail", serde_json::json!({}), "c1")]
            } else {
                Vec::new()
            }
        }
    }

    let registry = Registry::<i32>::new()
        .with(OkTool { name: "ok" })
        .with(FailingTool { name: "fail" });
    let tools: Arc<dyn ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    eval("succeeded_vs_failed_signals")
        .composition(borg::dsl::agent_loop(
            MixedAgent {
                emitted: std::sync::atomic::AtomicUsize::new(0),
            },
            tools,
            5,
        ))
        .outcomes(ok_script(5))
        .guardrails(harness.clone())
        .run()
        .await
        .assert_ok();

    let win = std::time::Duration::from_secs(60);
    assert_eq!(harness.detector.count("tool:ok:succeeded", win), 1);
    assert_eq!(harness.detector.count("tool:fail:failed", win), 1);
    assert_eq!(harness.detector.count("tool:ok:failed", win), 0);
    assert_eq!(harness.detector.count("tool:fail:succeeded", win), 0);
    assert_eq!(harness.detector.count("tool:fail:invoked", win), 1);
}

#[tokio::test]
async fn halt_carries_rule_name_and_reason() {
    let harness = Arc::new(GuardrailHarness::new(
        GuardrailEngine::new().with_rule(MaxContinuationsRule::new(1)),
    ));

    let case = eval("halt_carries_metadata")
        .composition(borg::dsl::agent_loop(
            RepeaterAgent {
                tool_name: "echo",
                calls_per_step: 1,
            },
            dispatcher_for(OkTool { name: "echo" }),
            10,
        ))
        .outcomes(ok_script(10))
        .guardrails(harness)
        .expect_error("guardrail-halt")
        .run()
        .await
        .assert_ok();

    match case.outcome {
        Err(BorgError::Guardrail { rule, reason }) => {
            assert_eq!(rule, "max_continuations");
            assert!(
                reason.contains("max-continuations"),
                "reason should mention the cap: got `{reason}`"
            );
        }
        other => panic!("expected BorgError::Guardrail, got {other:?}"),
    }
}

// Quiets the unused-import warning for fixtures not used by every test.
#[allow(dead_code)]
fn _dead_code_helpers() {
    let _ = std::mem::size_of::<DispatchedCall>();
}
