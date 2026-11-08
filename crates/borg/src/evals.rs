//! Eval harness — scripted scenarios that replay against a borg
//! composition and assert on the observable outcome.
//!
//! Pin behavior with deterministic scenarios so refactors can't silently
//! drift. Where unit tests cover individual primitives in isolation, an
//! eval exercises a full composition end-to-end against a scripted
//! `FakeLlmClient` and compares the resulting event stream + verdict
//! shape against an expected outcome.
//!
//! Private module; the public items are re-exported from
//! [`crate::testing`] (`run` as `run_eval`, `events` as `eval_events`).
//!
//! ## Easy shape — [`eval`](crate::evals::eval) builder
//!
//! ```ignore
//! use borg::testing::{eval, eval_events as events, EvalAssert, ScriptedAgent};
//!
//! #[tokio::test]
//! async fn smoke() {
//!     eval("smoke")
//!         .agent(ScriptedAgent::new("a"))
//!         .responses(["done"])
//!         .must_emit([events::STEP_COMPLETED])
//!         .run().await
//!         .assert_ok();
//! }
//! ```
//!
//! That covers most cases.
//!
//! ## Power shape — [`EvalCase`](crate::evals::EvalCase) + [`run`](crate::evals::run)
//!
//! For evals that need the full surface (multiple budgets, mixed
//! ok/error outcomes, raw `FakeOutcome` script), use the low-level
//! [`EvalCase`](crate::evals::EvalCase) struct directly. Both layers
//! produce identical [`EvalResult`](crate::evals::EvalResult)s.
//!
//! ## Limitations
//!
//! - Evals run on `FakeLlmClient`, not a real provider — they're
//!   regression locks, not quality benchmarks.
//! - The harness asserts on observable events, not internal verdict
//!   bodies, staying stable across body-formatting refactors while still
//!   catching behavior drift.

use crate::agent::BorgAgent;
use crate::composition::Composition;
use crate::dsl::step;
use crate::events::{EventEmitter, RecordingEmitter};
use crate::executor;
use crate::handoff::HandoffState;
use crate::testing::{FakeLlmClient, FakeOutcome};
use std::sync::Arc;

/// String constants for every `BorgEvent::kind()`. Use these in
/// `.must_emit(...)` / `.must_not_emit(...)` so typos surface at compile
/// time and renames flow through.
pub mod events {
    pub const COMPOSITION_STARTED: &str = "composition_started";
    pub const STEP_STARTED: &str = "step_started";
    pub const STEP_COMPLETED: &str = "step_completed";
    pub const STEP_RETRYING: &str = "step_retrying";
    pub const COMPOSITION_COMPLETED: &str = "composition_completed";
    pub const PARALLEL_DEGRADED: &str = "parallel_degraded";
    pub const BRANCH_DECIDED: &str = "branch_decided";
    pub const CANCELLED: &str = "cancelled";
}

/// A single scripted scenario.
pub struct EvalCase {
    /// For human-readable assertion failures.
    pub name: &'static str,
    pub composition: Composition,
    /// Scripted client whose responses the agents see.
    pub llm: Arc<FakeLlmClient>,
    pub budget_limit: u64,
    pub expect: ExpectedOutcome,
    /// When present, signal recording + boundary evaluation are active.
    pub guardrails: Option<Arc<crate::guardrails::GuardrailHarness>>,
}

/// What an eval expects to be true after the composition runs.
///
/// Fields are "at least"-style: new events in the stream don't break old
/// evals unless they assert "must NOT emit." Tighten via
/// `must_not_emit_kinds`.
#[derive(Default, Clone)]
pub struct ExpectedOutcome {
    /// Minimum number of `NamedVerdict`s in the outcome. Use `0` for
    /// scenarios that expect a `BorgError` (see `expect_error`).
    pub min_verdicts: usize,
    /// Minimum cumulative input tokens across all verdicts. Catches the
    /// "ran but emitted nothing" failure mode.
    pub min_input_tokens: u64,
    /// `BorgEvent::kind()` strings that must appear at least once.
    pub must_emit_kinds: Vec<&'static str>,
    /// `BorgEvent::kind()` strings that must not appear.
    pub must_not_emit_kinds: Vec<&'static str>,
    /// When `Some`, the composition must return `Err(BorgError)` whose
    /// `failure_cause()` matches this tag; `None` expects success.
    pub expect_error: Option<&'static str>,
}

/// Result of running an eval — the recorded events and final outcome.
///
/// Returned by [`run`] for additional assertions beyond the declarative
/// [`ExpectedOutcome`].
#[derive(Debug)]
pub struct EvalResult {
    pub events: Vec<crate::events::BorgEvent>,
    pub outcome: Result<crate::executor::ExecutionOutcome, crate::error::BorgError>,
    /// Number of LLM calls the scripted client served. Pins "step N
    /// was/wasn't reached" without inspecting the lossy `min_verdicts`.
    pub llm_call_count: usize,
    /// The guardrail harness wired into the eval, if any. Lets authors
    /// drain the [`crate::BoundaryMessageQueue`] and assert on corrective
    /// text.
    pub guardrails: Option<Arc<crate::guardrails::GuardrailHarness>>,
}

/// Run a scenario and assert on its outcome.
///
/// `Ok(EvalResult)` on pass (events + outcome returned for ad-hoc
/// assertions); `Err(String)` on fail, with a diagnostic naming the
/// scenario and which expectation failed.
pub async fn run(case: EvalCase) -> Result<EvalResult, String> {
    let cancel = tokio_util::sync::CancellationToken::new();
    let budget: Arc<dyn crate::budget::Budget> =
        Arc::new(crate::budget::InMemoryBudget::new(case.budget_limit));
    let emitter_inner = Arc::new(RecordingEmitter::new());
    let emitter: Arc<dyn EventEmitter> = emitter_inner.clone();

    let mut ctx = executor::ExecCtx::new(
        case.llm.clone(),
        budget,
        emitter,
        cancel,
        format!("eval::{}", case.name),
    );
    if let Some(harness) = &case.guardrails {
        ctx = ctx.with_guardrails(harness.clone());
    }

    let outcome = executor::execute_with_handoff(case.composition, ctx, HandoffState::new()).await;

    let events = emitter_inner.snapshot();
    let expect = &case.expect;

    if let Some(expected_cause) = expect.expect_error {
        match &outcome {
            Err(e) => {
                let actual = e.failure_cause();
                if actual != expected_cause {
                    return Err(format!(
                        "eval `{}`: expected error cause `{}`, got `{}`",
                        case.name, expected_cause, actual
                    ));
                }
            }
            Ok(_) => {
                return Err(format!(
                    "eval `{}`: expected error cause `{}`, but composition succeeded",
                    case.name, expected_cause
                ));
            }
        }
    } else {
        // Success path: must succeed AND meet thresholds.
        let exec = outcome
            .as_ref()
            .map_err(|e| format!("eval `{}`: expected success, got error: {}", case.name, e))?;

        if exec.verdicts.len() < expect.min_verdicts {
            return Err(format!(
                "eval `{}`: expected >= {} verdicts, got {}",
                case.name,
                expect.min_verdicts,
                exec.verdicts.len()
            ));
        }
        if exec.total_input_tokens() < expect.min_input_tokens {
            return Err(format!(
                "eval `{}`: expected >= {} input tokens, got {}",
                case.name,
                expect.min_input_tokens,
                exec.total_input_tokens()
            ));
        }
    }

    // Runs on both error + success paths: failed compositions may still
    // have emitted partial event streams worth asserting on.
    let kinds: Vec<&'static str> = events.iter().map(|e| e.kind()).collect();
    for required in &expect.must_emit_kinds {
        if !kinds.contains(required) {
            return Err(format!(
                "eval `{}`: required event `{}` not emitted (saw: {:?})",
                case.name, required, kinds
            ));
        }
    }
    for forbidden in &expect.must_not_emit_kinds {
        if kinds.contains(forbidden) {
            return Err(format!(
                "eval `{}`: forbidden event `{}` was emitted (saw: {:?})",
                case.name, forbidden, kinds
            ));
        }
    }

    let llm_call_count = case.llm.calls();
    Ok(EvalResult {
        events,
        outcome,
        llm_call_count,
        guardrails: case.guardrails,
    })
}

// ─────────────────────────────────────────────────────────────────────
//   Builder surface — `eval(name).agent(_).responses([..]).run()`
// ─────────────────────────────────────────────────────────────────────

/// Start an eval. The returned [`EvalBuilder`] is the ergonomic surface
/// over [`EvalCase`] + [`run`].
///
/// Defaults: budget 100_000 tokens, no floors, no event must/must-not
/// lists, no error expected. Call the corresponding method to tighten.
///
/// Minimum viable eval:
///
/// ```ignore
/// eval("smoke")
///     .agent(ScriptedAgent::new("a"))
///     .responses(["done"])
///     .run().await
///     .assert_ok();
/// ```
pub fn eval(name: &'static str) -> EvalBuilder {
    EvalBuilder {
        name,
        composition: None,
        outcomes: Vec::new(),
        budget_limit: 100_000,
        expect: ExpectedOutcome::default(),
        guardrails: None,
    }
}

/// Builder for an eval scenario. Constructed via [`eval`].
///
/// Fields default to "accept anything"; only the `composition` (or
/// `agent`) and response script need to be set. Every other method is
/// opt-in tightening.
pub struct EvalBuilder {
    name: &'static str,
    composition: Option<Composition>,
    outcomes: Vec<FakeOutcome>,
    budget_limit: u64,
    expect: ExpectedOutcome,
    guardrails: Option<Arc<crate::guardrails::GuardrailHarness>>,
}

impl EvalBuilder {
    /// Set the composition under test. Replaces any prior
    /// `agent`/`composition` call.
    #[must_use]
    pub fn composition(mut self, c: Composition) -> Self {
        self.composition = Some(c);
        self
    }

    /// Single-agent shortcut — wraps `a` in `step(a)`. Replaces any
    /// prior `agent`/`composition` call.
    #[must_use]
    pub fn agent<A>(mut self, a: A) -> Self
    where
        A: BorgAgent + 'static,
    {
        self.composition = Some(step(a));
        self
    }

    /// Script the FakeLlmClient with OK responses (one per LLM call, in
    /// order); each text becomes one agent invocation's verdict body.
    /// Replaces any prior `responses`/`response_errors`/`outcomes` call.
    #[must_use]
    pub fn responses<I, S>(mut self, texts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.outcomes = texts.into_iter().map(FakeOutcome::ok).collect();
        self
    }

    /// Script the FakeLlmClient with terminal errors (one per call).
    /// Pair with `expect_error` to test error-path compositions.
    #[must_use]
    pub fn response_errors<I, S>(mut self, msgs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.outcomes = msgs.into_iter().map(FakeOutcome::terminal_err).collect();
        self
    }

    /// Raw escape hatch — script with `FakeOutcome` directly, for mixed
    /// ok/err/forever streams, latency injection, or tool-call outcomes.
    #[must_use]
    pub fn outcomes(mut self, v: Vec<FakeOutcome>) -> Self {
        self.outcomes = v;
        self
    }

    /// Per-eval budget cap. Default 100_000 tokens; lower to test
    /// budget-exhaustion paths.
    #[must_use]
    pub fn budget(mut self, n: u64) -> Self {
        self.budget_limit = n;
        self
    }

    /// Minimum `NamedVerdict`s in the outcome. Default 0.
    #[must_use]
    pub fn min_verdicts(mut self, n: usize) -> Self {
        self.expect.min_verdicts = n;
        self
    }

    /// Minimum cumulative input tokens across all verdicts. Default 0;
    /// catches "ran but emitted nothing."
    #[must_use]
    pub fn min_input_tokens(mut self, n: u64) -> Self {
        self.expect.min_input_tokens = n;
        self
    }

    /// Every kind in `kinds` must appear at least once. Use the
    /// [`events`] constants for compile-time-checked names.
    #[must_use]
    pub fn must_emit<I, S>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for k in kinds {
            self.expect.must_emit_kinds.push(intern(k.as_ref()));
        }
        self
    }

    /// No kind in `kinds` may appear. Use the [`events`] constants.
    #[must_use]
    pub fn must_not_emit<I, S>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for k in kinds {
            self.expect.must_not_emit_kinds.push(intern(k.as_ref()));
        }
        self
    }

    /// Sugar for `must_emit([COMPOSITION_STARTED, COMPOSITION_COMPLETED])`
    /// — "pipeline ran end-to-end."
    #[must_use]
    pub fn must_complete(self) -> Self {
        self.must_emit([events::COMPOSITION_STARTED, events::COMPOSITION_COMPLETED])
    }

    /// Composition is expected to return `Err(BorgError)` whose
    /// `failure_cause()` matches `cause`. Pair with `response_errors`
    /// or hand-rolled `outcomes` to script the failure.
    #[must_use]
    pub fn expect_error(mut self, cause: &'static str) -> Self {
        self.expect.expect_error = Some(cause);
        self
    }

    /// Wire a [`crate::GuardrailHarness`] into the eval's executor.
    /// Auto-emitted signals at every tool dispatch + loop boundary
    /// accumulate against its detector; use `expect_error("guardrail-halt")`
    /// to assert a rule fired.
    #[must_use]
    pub fn guardrails(mut self, harness: Arc<crate::guardrails::GuardrailHarness>) -> Self {
        self.guardrails = Some(harness);
        self
    }

    /// Execute the eval. Call `.assert_ok()` (via [`EvalAssert`]) for a
    /// panic-on-failure terminator, or handle the `Result` manually.
    ///
    /// Panics if no composition was set — a programmer error, not a test
    /// failure.
    pub async fn run(self) -> Result<EvalResult, String> {
        let composition = self.composition.unwrap_or_else(|| {
            panic!(
                "eval `{}`: no composition set — call `.agent(...)` or `.composition(...)`",
                self.name
            )
        });
        let case = EvalCase {
            name: self.name,
            composition,
            llm: FakeLlmClient::new(self.outcomes),
            budget_limit: self.budget_limit,
            expect: self.expect,
            guardrails: self.guardrails,
        };
        run(case).await
    }
}

/// Intern a `&str` as `&'static str` so the builder's `AsRef<str>` args
/// fit the `Vec<&'static str>` surface. The table leaks intentionally:
/// it's bounded by the small set of event-kind names, all of which live
/// for the process lifetime anyway.
fn intern(s: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static POOL: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = POOL.lock().unwrap();
    let pool = guard.get_or_insert_with(HashSet::new);
    if let Some(existing) = pool.get(s) {
        return existing;
    }
    let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
    pool.insert(leaked);
    leaked
}

/// Terminator for `eval(...).run().await` — panics on failure with
/// the harness's formatted error, otherwise returns the [`EvalResult`]
/// for further assertions.
///
/// ```ignore
/// let res = eval("x").agent(a).responses(["ok"]).run().await.assert_ok();
/// assert_eq!(res.llm_call_count, 1);
/// ```
pub trait EvalAssert {
    /// Unwrap to [`EvalResult`] or panic with the eval's diagnostic.
    fn assert_ok(self) -> EvalResult;

    /// Inverse: assert the eval *failed*; panics if it unexpectedly
    /// passed. Returns the failure string for `contains()`-style checks.
    fn assert_err(self) -> String;
}

impl EvalAssert for Result<EvalResult, String> {
    fn assert_ok(self) -> EvalResult {
        match self {
            Ok(r) => r,
            Err(e) => panic!("eval failed:\n  {e}"),
        }
    }

    fn assert_err(self) -> String {
        match self {
            Ok(_) => panic!("eval unexpectedly passed; expected failure"),
            Err(e) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::step;
    use crate::testing::{FakeOutcome, ScriptedAgent};

    #[tokio::test]
    async fn single_step_completion_passes() {
        let case = EvalCase {
            name: "single_step_completion",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::ok("done")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 1,
                min_input_tokens: 1,
                must_emit_kinds: vec!["step_started", "step_completed"],
                must_not_emit_kinds: vec!["step_retrying"],
                expect_error: None,
            },
            guardrails: None,
        };
        let result = run(case).await.expect("scenario should pass");
        assert!(result.outcome.is_ok());
        assert_eq!(result.outcome.as_ref().unwrap().verdicts.len(), 1);
    }

    #[tokio::test]
    async fn must_emit_assertion_fires_on_missing_event() {
        let case = EvalCase {
            name: "missing_event_detection",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::ok("done")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 1,
                min_input_tokens: 0,
                // step_retrying never happens on a clean single-step run.
                must_emit_kinds: vec!["step_retrying"],
                must_not_emit_kinds: vec![],
                expect_error: None,
            },
            guardrails: None,
        };
        let err = run(case).await.expect_err("should fail on missing event");
        assert!(err.contains("step_retrying"));
        assert!(err.contains("not emitted"));
    }

    #[tokio::test]
    async fn must_not_emit_assertion_fires_on_unexpected_event() {
        let case = EvalCase {
            name: "unexpected_event_detection",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::ok("done")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 1,
                min_input_tokens: 0,
                must_emit_kinds: vec![],
                // step_completed always fires on a clean single-step run.
                must_not_emit_kinds: vec!["step_completed"],
                expect_error: None,
            },
            guardrails: None,
        };
        let err = run(case)
            .await
            .expect_err("should fail on unexpected event");
        assert!(err.contains("step_completed"));
        assert!(err.contains("forbidden"));
    }

    #[tokio::test]
    async fn min_verdicts_threshold_enforced() {
        let case = EvalCase {
            name: "min_verdicts_threshold",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::ok("done")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 5, // single-step → 1 verdict; expecting 5 fails
                min_input_tokens: 0,
                must_emit_kinds: vec![],
                must_not_emit_kinds: vec![],
                expect_error: None,
            },
            guardrails: None,
        };
        let err = run(case).await.expect_err("should fail on min_verdicts");
        assert!(err.contains("expected >= 5 verdicts"));
        assert!(err.contains("got 1"));
    }

    #[tokio::test]
    async fn min_input_tokens_threshold_enforced() {
        let case = EvalCase {
            name: "min_input_tokens_threshold",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::ok("done")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 1,
                // FakeLlmClient reports 100 input_tokens per call;
                // assert we'd notice if it ran but produced 0.
                min_input_tokens: 10_000,
                must_emit_kinds: vec![],
                must_not_emit_kinds: vec![],
                expect_error: None,
            },
            guardrails: None,
        };
        let err = run(case)
            .await
            .expect_err("should fail on min_input_tokens");
        assert!(err.contains("expected >= 10000 input tokens"));
    }

    #[tokio::test]
    async fn expect_error_passes_when_composition_fails_as_expected() {
        // ScriptedAgent uses default invoke which calls the LLM. A
        // terminal-error LLM outcome surfaces as BorgError::Llm.
        let case = EvalCase {
            name: "terminal_llm_error",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::terminal_err("upstream 500")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 0,
                min_input_tokens: 0,
                must_emit_kinds: vec!["step_started", "step_completed"],
                must_not_emit_kinds: vec![],
                expect_error: Some("llm-error"),
            },
            guardrails: None,
        };
        let result = run(case).await.expect("scenario should pass");
        assert!(result.outcome.is_err());
    }

    #[tokio::test]
    async fn expect_error_fails_when_composition_unexpectedly_succeeds() {
        let case = EvalCase {
            name: "unexpected_success",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::ok("done")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 0,
                min_input_tokens: 0,
                must_emit_kinds: vec![],
                must_not_emit_kinds: vec![],
                expect_error: Some("llm-error"),
            },
            guardrails: None,
        };
        let err = run(case)
            .await
            .expect_err("should fail when expecting error");
        assert!(err.contains("expected error cause"));
        assert!(err.contains("succeeded"));
    }

    #[tokio::test]
    async fn expect_error_fails_when_wrong_error_cause() {
        let case = EvalCase {
            name: "wrong_error_cause",
            composition: step(ScriptedAgent::new("agent")),
            llm: FakeLlmClient::new(vec![FakeOutcome::terminal_err("upstream 500")]),
            budget_limit: 10_000,
            expect: ExpectedOutcome {
                min_verdicts: 0,
                min_input_tokens: 0,
                must_emit_kinds: vec![],
                must_not_emit_kinds: vec![],
                // The actual cause is "llm-error", not "cancelled".
                expect_error: Some("cancelled"),
            },
            guardrails: None,
        };
        let err = run(case)
            .await
            .expect_err("should fail on wrong error cause");
        assert!(err.contains("expected error cause `cancelled`"));
        assert!(err.contains("got `llm-error`"));
    }

    // Builder surface — exercises the ergonomic chain over EvalCase.

    #[tokio::test]
    async fn builder_one_liner_smoke() {
        let res = eval("smoke")
            .agent(ScriptedAgent::new("a"))
            .responses(["done"])
            .run()
            .await
            .assert_ok();
        assert_eq!(res.llm_call_count, 1);
        assert_eq!(res.outcome.as_ref().unwrap().verdicts.len(), 1);
    }

    #[tokio::test]
    async fn builder_must_complete_sugar() {
        eval("end_to_end")
            .agent(ScriptedAgent::new("a"))
            .responses(["done"])
            .must_complete()
            .run()
            .await
            .assert_ok();
    }

    #[tokio::test]
    async fn builder_multi_step_via_responses() {
        use crate::dsl::seq;
        let pipeline = seq([
            step(ScriptedAgent::new("a")),
            step(ScriptedAgent::new("b")),
            step(ScriptedAgent::new("c")),
        ]);
        let res = eval("three_steps")
            .composition(pipeline)
            .responses(["one", "two", "three"])
            .min_verdicts(3)
            .must_emit([events::STEP_COMPLETED])
            .must_not_emit([events::CANCELLED, events::STEP_RETRYING])
            .run()
            .await
            .assert_ok();
        assert_eq!(res.llm_call_count, 3);
    }

    #[tokio::test]
    async fn builder_error_path() {
        eval("llm_500")
            .agent(ScriptedAgent::new("a"))
            .response_errors(["upstream 500"])
            .expect_error("llm-error")
            .run()
            .await
            .assert_ok();
    }

    #[tokio::test]
    async fn builder_default_budget_is_generous() {
        use crate::dsl::seq;
        let pipeline = seq([
            step(ScriptedAgent::new("a")),
            step(ScriptedAgent::new("b")),
            step(ScriptedAgent::new("c")),
        ]);
        eval("budget_default")
            .composition(pipeline)
            .responses(["x", "y", "z"])
            .run()
            .await
            .assert_ok();
    }

    #[tokio::test]
    async fn builder_assert_err_returns_diagnostic() {
        let msg = eval("forbidden_event_fires")
            .agent(ScriptedAgent::new("a"))
            .responses(["done"])
            .must_not_emit([events::STEP_COMPLETED]) // always fires → must fail
            .run()
            .await
            .assert_err();
        assert!(msg.contains("step_completed"));
        assert!(msg.contains("forbidden"));
    }

    #[tokio::test]
    async fn builder_outcomes_escape_hatch() {
        let res = eval("mixed_outcomes")
            .agent(ScriptedAgent::new("a"))
            .outcomes(vec![FakeOutcome::ok("done")])
            .run()
            .await
            .assert_ok();
        assert_eq!(res.llm_call_count, 1);
    }

    #[tokio::test]
    #[should_panic(expected = "no composition set")]
    async fn builder_missing_composition_panics_clearly() {
        let _ = eval("forgot_composition").responses(["x"]).run().await;
    }

    #[tokio::test]
    async fn builder_event_constants_match_real_kinds() {
        // Sanity: the constants in `events` mod match what BorgEvent
        // actually emits. If a kind gets renamed in events.rs, this
        // test surfaces it before any downstream eval breaks.
        use crate::events::{BorgEvent, CompositionShape};
        assert_eq!(
            BorgEvent::CompositionStarted {
                composition_id: "c".into(),
                shape: CompositionShape::Sequential,
                branch_count: 0,
                parent_id: None,
            }
            .kind(),
            events::COMPOSITION_STARTED
        );
    }
}
