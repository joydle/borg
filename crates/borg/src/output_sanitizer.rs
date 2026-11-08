//! Output transformation pipeline — strip noise and redact sensitive
//! information from tool/subprocess output before it reaches the LLM.
//!
//! ## Two entry points
//!
//! - **Hot path** ([`OutputSanitizerPipeline::run_block`] /
//!   [`OutputSanitizerPipeline::run_line`]) — direct synchronous call, no
//!   borg overhead. Use under high call rates (per-line streaming) where
//!   an event per line would flood the trace.
//! - **Observable path** ([`run_sanitizer`]) — wraps the pipeline in a
//!   `Composition::Sequential` under [`ExecCtx::deterministic`], emitting
//!   one `StepStarted`/`StepCompleted` pair per stage.
//!
//! Both share the same stage impls — no behavior drift between them.
//!
//! ## Implementing a stage
//!
//! ```ignore
//! use borg::output_sanitizer::{OutputTransformStage, SanitizerCtx};
//!
//! struct PathRedactor;
//!
//! impl OutputTransformStage for PathRedactor {
//!     fn stage_id(&self) -> &'static str { "path_redactor" }
//!
//!     fn transform(&self, line: &str, ctx: &SanitizerCtx) -> String {
//!         let id = ctx.get("run_id").unwrap_or("");
//!         line.replace(&format!("/tmp/build-{id}"), "<project>")
//!     }
//! }
//!
//! // Noise filtering — override is_noise_line and return identity from transform:
//! struct NoiseFilter;
//! impl OutputTransformStage for NoiseFilter {
//!     fn stage_id(&self) -> &'static str { "noise_filter" }
//!     fn transform(&self, line: &str, _ctx: &SanitizerCtx) -> String { line.to_string() }
//!     fn is_noise_line(&self, line: &str, _ctx: &SanitizerCtx) -> bool {
//!         line.starts_with("Done in")
//!     }
//! }
//! ```
//!
//! ## Building and running a pipeline
//!
//! ```ignore
//! use borg::output_sanitizer::{OutputSanitizerPipeline, SanitizerCtx, run_sanitizer};
//!
//! let pipeline = OutputSanitizerPipeline::new()
//!     .stage(PathRedactor)
//!     .stage(NoiseFilter);
//!
//! // Hot path:
//! let ctx = SanitizerCtx::new().var("run_id", "run-abc");
//! let clean = pipeline.run_block(raw_output, &ctx);
//!
//! // Observable path (observable via exec_ctx's emitter):
//! let clean = run_sanitizer(&pipeline, raw_output, ctx, exec_ctx).await;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::{BorgAgent, Provider};
use crate::composition::Composition;
use crate::context::AgentContext;
use crate::dsl::{seq, step};
use crate::error::BorgError;
use crate::executor::{execute_with_handoff, ExecCtx};
use crate::handoff::HandoffState;
use crate::verdict::Verdict;

// ── Private handoff keys ─────────────────────────────────────────────────

/// Handoff key for the raw input text. Seeded by [`run_sanitizer`]; the
/// first stage reads it when `handoff.previous()` is `None`.
const INPUT_KEY: &str = "output_sanitizer::input";

/// Handoff key for the serialized [`SanitizerCtx::vars`] map.
const VARS_KEY: &str = "output_sanitizer::vars";

// ── SanitizerCtx ─────────────────────────────────────────────────────────

/// Per-invocation context variables for transform stages.
///
/// Carries caller-supplied metadata so stages produce context-specific
/// output without baking conventions into their signatures.
///
/// ```ignore
/// let ctx = SanitizerCtx::new()
///     .var("run_id", "run-abc123")
///     .var("env", "production");
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SanitizerCtx {
    pub vars: HashMap<String, String>,
}

impl SanitizerCtx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a context variable. Chainable.
    #[must_use]
    pub fn var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.vars.insert(key.into(), value.into());
        self
    }

    /// Look up a variable by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(|s| s.as_str())
    }
}

// ── OutputTransformStage trait ───────────────────────────────────────────

/// One deterministic transformation stage. Pure and synchronous.
///
/// The pipeline calls stages in registration order. For each line:
/// 1. `is_noise_line` runs first — return `true` to drop the line.
/// 2. `transform` runs on surviving lines — return the mutated text.
///
/// Implementations should be cheap: compile regexes at startup via
/// `once_cell::Lazy` or equivalent, avoid I/O, avoid allocation beyond
/// regex capture groups.
pub trait OutputTransformStage: Send + Sync + 'static {
    /// Stable identifier for this stage. Used as the `BorgAgent::agent_id`
    /// in observable runs and in traces. Must be a `'static` string
    /// (string literal or `Box::leak`'d).
    fn stage_id(&self) -> &'static str;

    /// Transform one line of text. Called only for lines that pass
    /// `is_noise_line`. Return the mutated text (may be equal to input
    /// if this stage only filters, not transforms).
    fn transform(&self, line: &str, ctx: &SanitizerCtx) -> String;

    /// Return `true` to drop this line entirely from the output.
    /// Default is always `false` (keep all lines, transform only).
    fn is_noise_line(&self, _line: &str, _ctx: &SanitizerCtx) -> bool {
        false
    }
}

// ── OutputSanitizerPipeline ───────────────────────────────────────────────

/// A sequential multi-stage text transformation pipeline.
///
/// Stages live behind `Arc` so one pipeline serves both the hot path
/// and the observable composition without requiring `Clone` on stages.
///
/// ```ignore
/// let pipeline = OutputSanitizerPipeline::new()
///     .stage(PathRedactor)
///     .stage(VersionStripper)
///     .stage(NoiseFilter);
/// ```
pub struct OutputSanitizerPipeline {
    stages: Vec<Arc<dyn OutputTransformStage>>,
}

impl OutputSanitizerPipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the end of the pipeline.
    #[must_use]
    pub fn stage(mut self, s: impl OutputTransformStage) -> Self {
        self.stages.push(Arc::new(s));
        self
    }

    /// **Hot path** — run all stages over a multi-line block synchronously.
    ///
    /// Trailing-newline discipline is preserved: each line keeps its `\n`
    /// iff it had one in the input.
    pub fn run_block(&self, input: &str, ctx: &SanitizerCtx) -> String {
        if input.is_empty() || self.stages.is_empty() {
            return input.to_string();
        }
        let mut current = input.to_string();
        for stage in &self.stages {
            current = apply_stage(stage.as_ref(), &current, ctx);
        }
        current
    }

    /// **Hot path** — run all stages over a single line synchronously.
    ///
    /// Returns `None` when any stage's `is_noise_line` drops the line.
    pub fn run_line(&self, line: &str, ctx: &SanitizerCtx) -> Option<String> {
        for stage in &self.stages {
            if stage.is_noise_line(line, ctx) {
                return None;
            }
        }
        let mut current = line.to_string();
        for stage in &self.stages {
            current = stage.transform(&current, ctx);
        }
        Some(current)
    }

    /// Build a `Composition::Sequential` with one step per stage.
    ///
    /// Each step is a deterministic `BorgAgent` wrapper; running this
    /// composition under [`ExecCtx::deterministic`] produces one
    /// `StepStarted`/`StepCompleted` event per stage. The composition
    /// reads its input from the [`HandoffState`] seeded by
    /// [`run_sanitizer`] — do not call `execute` on this composition
    /// directly unless you seed `output_sanitizer::input` and
    /// `output_sanitizer::vars` yourself.
    pub fn to_composition(&self) -> Composition {
        let steps: Vec<Composition> = self
            .stages
            .iter()
            .map(|stage| step(SanitizerStageAgent(Arc::clone(stage))))
            .collect();
        seq(steps)
    }
}

impl Default for OutputSanitizerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── run_sanitizer ────────────────────────────────────────────────────────

/// **Observable path** — run the pipeline under a borg composition.
///
/// Seeds the [`HandoffState`] with `input` and `ctx`, builds a
/// `Composition::Sequential` (one step per stage), and executes it under
/// [`ExecCtx::deterministic`]. The `exec_ctx`'s emitter receives one
/// `StepStarted`/`StepCompleted` pair per stage.
///
/// **Fallback:** if the composition errors for any reason (cancellation,
/// internal panic, unexpected failure), falls back to
/// [`OutputSanitizerPipeline::run_block`] — sanitization must never
/// propagate an error to its caller.
pub async fn run_sanitizer(
    pipeline: &OutputSanitizerPipeline,
    input: &str,
    ctx: SanitizerCtx,
    exec_ctx: ExecCtx,
) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut handoff = HandoffState::new();
    handoff.set(INPUT_KEY, input.to_string());
    handoff.set(VARS_KEY, &ctx.vars);

    let comp = pipeline.to_composition();

    match execute_with_handoff(comp, exec_ctx, handoff).await {
        Ok(outcome) => outcome
            .verdicts
            .last()
            .map(|nv| nv.verdict.body.to_string())
            .unwrap_or_else(|| pipeline.run_block(input, &ctx)),
        Err(e) => {
            tracing::warn!(
                error = ?e,
                "output_sanitizer: pipeline failed, falling back to direct run_block"
            );
            pipeline.run_block(input, &ctx)
        }
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Apply a single stage over a multi-line block, returning the filtered
/// and transformed result. Preserves trailing-newline discipline.
fn apply_stage(stage: &dyn OutputTransformStage, input: &str, ctx: &SanitizerCtx) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        let (body, nl) = match line.strip_suffix('\n') {
            Some(s) => (s, "\n"),
            None => (line, ""),
        };
        if stage.is_noise_line(body, ctx) {
            continue;
        }
        let transformed = stage.transform(body, ctx);
        out.push_str(&transformed);
        out.push_str(nl);
    }
    out
}

// ── SanitizerStageAgent — internal BorgAgent wrapper ─────────────────────

/// Wraps an [`OutputTransformStage`] as a deterministic [`BorgAgent`].
///
/// Used by [`OutputSanitizerPipeline::to_composition`] to produce one
/// composition step per stage. Not part of the public API — consumers
/// interact through [`OutputSanitizerPipeline`] and [`run_sanitizer`].
struct SanitizerStageAgent(Arc<dyn OutputTransformStage>);

#[async_trait]
impl BorgAgent for SanitizerStageAgent {
    fn agent_id(&self) -> &'static str {
        self.0.stage_id()
    }

    fn model(&self) -> &str {
        "deterministic"
    }

    fn provider(&self) -> Provider {
        Provider::Custom("deterministic")
    }

    fn max_tokens(&self) -> u32 {
        0
    }

    fn system_prompt(&self) -> &'static str {
        ""
    }

    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        None
    }

    fn known_verdicts(&self) -> &'static [&'static str] {
        &["TRANSFORMED"]
    }

    fn min_tokens_estimate(&self) -> u64 {
        0
    }

    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        if ctx.cancel.is_cancelled() {
            return Err(BorgError::Cancelled);
        }

        // In-flight text: previous stage's verdict body, else seeded input.
        let input = ctx
            .handoff
            .previous()
            .map(|nv| nv.verdict.body.to_string())
            .or_else(|| ctx.handoff.get_as::<String>(INPUT_KEY))
            .unwrap_or_default();

        let vars: HashMap<String, String> = ctx.handoff.get_as(VARS_KEY).unwrap_or_default();
        let san_ctx = SanitizerCtx { vars };

        let output = apply_stage(self.0.as_ref(), &input, &san_ctx);

        Ok(Verdict::with_token("TRANSFORMED", output, 0, 0))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Minimal stage impls ───────────────────────────────────────────────

    struct UpperCase;
    impl OutputTransformStage for UpperCase {
        fn stage_id(&self) -> &'static str {
            "uppercase"
        }
        fn transform(&self, line: &str, _ctx: &SanitizerCtx) -> String {
            line.to_uppercase()
        }
    }

    struct AppendBang;
    impl OutputTransformStage for AppendBang {
        fn stage_id(&self) -> &'static str {
            "append_bang"
        }
        fn transform(&self, line: &str, _ctx: &SanitizerCtx) -> String {
            format!("{line}!")
        }
    }

    struct DropNoise;
    impl OutputTransformStage for DropNoise {
        fn stage_id(&self) -> &'static str {
            "drop_noise"
        }
        fn transform(&self, line: &str, _ctx: &SanitizerCtx) -> String {
            line.to_string()
        }
        fn is_noise_line(&self, line: &str, _ctx: &SanitizerCtx) -> bool {
            line.starts_with("NOISE")
        }
    }

    struct VarSubstitute;
    impl OutputTransformStage for VarSubstitute {
        fn stage_id(&self) -> &'static str {
            "var_substitute"
        }
        fn transform(&self, line: &str, ctx: &SanitizerCtx) -> String {
            if let Some(id) = ctx.get("id") {
                line.replace("<ID>", id)
            } else {
                line.to_string()
            }
        }
    }

    // ── run_block ─────────────────────────────────────────────────────────

    #[test]
    fn run_block_applies_stages_in_order() {
        let p = OutputSanitizerPipeline::new()
            .stage(UpperCase)
            .stage(AppendBang);
        let ctx = SanitizerCtx::new();
        let out = p.run_block("hello\nworld\n", &ctx);
        assert_eq!(out, "HELLO!\nWORLD!\n");
    }

    #[test]
    fn run_block_drops_noise_lines() {
        let p = OutputSanitizerPipeline::new().stage(DropNoise);
        let ctx = SanitizerCtx::new();
        let out = p.run_block("keep\nNOISE: skip this\nalso keep\n", &ctx);
        assert_eq!(out, "keep\nalso keep\n");
    }

    #[test]
    fn run_block_preserves_trailing_newline() {
        let p = OutputSanitizerPipeline::new().stage(AppendBang);
        let ctx = SanitizerCtx::new();
        assert!(p.run_block("line\n", &ctx).ends_with('\n'));
        assert!(!p.run_block("line", &ctx).ends_with('\n'));
    }

    #[test]
    fn run_block_empty_input_returns_empty() {
        let p = OutputSanitizerPipeline::new().stage(UpperCase);
        assert_eq!(p.run_block("", &SanitizerCtx::new()), "");
    }

    #[test]
    fn run_block_no_stages_returns_input() {
        let p = OutputSanitizerPipeline::new();
        let input = "unchanged\n";
        assert_eq!(p.run_block(input, &SanitizerCtx::new()), input);
    }

    #[test]
    fn run_block_idempotent_when_stages_are_idempotent() {
        let p = OutputSanitizerPipeline::new().stage(UpperCase);
        let ctx = SanitizerCtx::new();
        let once = p.run_block("hello\n", &ctx);
        let twice = p.run_block(&once, &ctx);
        assert_eq!(once, twice);
    }

    // ── run_line ─────────────────────────────────────────────────────────

    #[test]
    fn run_line_transforms_surviving_lines() {
        let p = OutputSanitizerPipeline::new().stage(UpperCase);
        let ctx = SanitizerCtx::new();
        assert_eq!(p.run_line("hello", &ctx), Some("HELLO".to_string()));
    }

    #[test]
    fn run_line_returns_none_for_noise() {
        let p = OutputSanitizerPipeline::new().stage(DropNoise);
        let ctx = SanitizerCtx::new();
        assert_eq!(p.run_line("NOISE: drop me", &ctx), None);
        assert_eq!(p.run_line("keep me", &ctx), Some("keep me".to_string()));
    }

    // ── SanitizerCtx ─────────────────────────────────────────────────────

    #[test]
    fn ctx_var_substitution() {
        let p = OutputSanitizerPipeline::new().stage(VarSubstitute);
        let ctx = SanitizerCtx::new().var("id", "run-abc");
        let out = p.run_block("path/<ID>/src\n", &ctx);
        assert_eq!(out, "path/run-abc/src\n");
    }

    #[test]
    fn ctx_missing_var_is_identity() {
        let p = OutputSanitizerPipeline::new().stage(VarSubstitute);
        let ctx = SanitizerCtx::new();
        let out = p.run_block("path/<ID>/src\n", &ctx);
        assert_eq!(out, "path/<ID>/src\n");
    }

    // ── observable path (integration, requires tokio) ─────────────────────

    #[cfg(test)]
    mod observable {
        use super::*;
        use crate::budget::in_memory_budget;
        use crate::events::noop_emitter;
        use crate::executor::ExecCtx;
        use tokio_util::sync::CancellationToken;

        fn make_ctx() -> ExecCtx {
            ExecCtx::deterministic(
                in_memory_budget(0),
                noop_emitter(),
                CancellationToken::new(),
                "test::output_sanitizer",
            )
        }

        #[tokio::test]
        async fn run_sanitizer_matches_run_block() {
            let pipeline = || {
                OutputSanitizerPipeline::new()
                    .stage(UpperCase)
                    .stage(DropNoise)
            };
            let input = "hello\nNOISE: skip\nworld\n";
            let ctx = SanitizerCtx::new();

            let sync_out = pipeline().run_block(input, &ctx);
            let async_out = run_sanitizer(&pipeline(), input, ctx.clone(), make_ctx()).await;

            assert_eq!(sync_out, async_out);
            assert_eq!(async_out, "HELLO\nWORLD\n");
        }

        #[tokio::test]
        async fn run_sanitizer_empty_input_returns_empty() {
            let pipeline = OutputSanitizerPipeline::new().stage(UpperCase);
            let out = run_sanitizer(&pipeline, "", SanitizerCtx::new(), make_ctx()).await;
            assert_eq!(out, "");
        }

        #[tokio::test]
        async fn run_sanitizer_vars_reach_stages() {
            let pipeline = OutputSanitizerPipeline::new().stage(VarSubstitute);
            let ctx = SanitizerCtx::new().var("id", "xyz");
            let out = run_sanitizer(&pipeline, "build-<ID>\n", ctx, make_ctx()).await;
            assert_eq!(out, "build-xyz\n");
        }

        #[tokio::test]
        async fn run_sanitizer_cancelled_falls_back_to_run_block() {
            let pipeline = OutputSanitizerPipeline::new().stage(UpperCase);
            let cancel = CancellationToken::new();
            cancel.cancel();
            let exec_ctx = ExecCtx::deterministic(
                in_memory_budget(0),
                noop_emitter(),
                cancel,
                "test::cancelled",
            );
            let input = "hello\n";
            let ctx = SanitizerCtx::new();
            // Must fall back to run_block on cancellation, not panic or hang.
            let out = run_sanitizer(&pipeline, input, ctx.clone(), exec_ctx).await;
            assert_eq!(out, "HELLO\n");
        }
    }
}
