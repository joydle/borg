//! Showcase — five patterns in one runnable file.
//!
//! Run with: `cargo run --example showcase -p borg --features testing`
//!
//! 1. `#[derive(BorgAgentDerive)]` — write only `user_prompt`
//! 2. `seq` / `par` / `repeat` composition
//! 3. `branch` — one agent decides, three sub-pipelines wait
//! 4. Typed handoff slots — compile-time-checked state between steps
//! 5. The `pipeline!` macro
//!
//! Runs against `FakeLlmClient`, so it's hermetic.

use borg::dsl::*;
use borg::testing::{FakeLlmClient, FakeOutcome};
use borg::*;
use std::collections::HashMap;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────
// 1. Typed handoff slots
// ─────────────────────────────────────────────────────────────────────

struct UserBrief;
impl HandoffSlot for UserBrief {
    type Value = String;
    const KEY: &'static str = "user_brief";
}

struct WorkingFiles;
impl HandoffSlot for WorkingFiles {
    type Value = HashMap<String, String>;
    const KEY: &'static str = "working_files";
}

// ─────────────────────────────────────────────────────────────────────
// 2. Agents
// ─────────────────────────────────────────────────────────────────────

#[derive(BorgAgentDerive)]
#[borg(
    id = "planner",
    model = "fake/planner",
    max_tokens = 800,
    system = "Plan the work."
)]
struct Planner;

impl Planner {
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        ctx.handoff
            .get_typed::<UserBrief>()
            .map(|b| format!("Plan for: {b}"))
    }
}

/// Declares the verdict-token contract `route_by_kind` reads: the token
/// is line 1 of the response, the body is the rest.
#[derive(BorgAgentDerive)]
#[borg(
    id = "triager",
    model = "fake/triager",
    max_tokens = 400,
    system = "Triage. Line 1: BUG | FEATURE | DOCS. Then a one-line rationale.",
    known_verdicts = &["BUG", "FEATURE", "DOCS"],
)]
struct Triager;

impl Triager {
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        ctx.handoff
            .get_typed::<UserBrief>()
            .map(|b| format!("Triage: {b}"))
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "ui_builder",
    model = "fake/builder",
    max_tokens = 1500,
    system = "Emit UI code."
)]
struct Coder;

impl Coder {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("write the UI layer".into())
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "logic_builder",
    model = "fake/builder",
    max_tokens = 1500,
    system = "Emit business logic."
)]
struct TestWriter;

impl TestWriter {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("write the logic layer".into())
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "builder",
    model = "fake/builder",
    max_tokens = 1500,
    system = "Run the build. Line 1: PASS | FAIL.",
    known_verdicts = &["PASS", "FAIL"],
)]
struct Builder;

impl Builder {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("cargo build".into())
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "build_fixer",
    model = "fake/builder",
    max_tokens = 1500,
    system = "Fix build errors."
)]
struct BuildFixer;

impl BuildFixer {
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("fix the last build failure".into())
    }
}

// Three branch-arm agents — one method each.
#[derive(BorgAgentDerive)]
#[borg(
    id = "bug_fixer",
    model = "fake/branch",
    max_tokens = 500,
    system = "Fix bugs."
)]
struct BugFixer;
impl BugFixer {
    fn user_prompt(&self, _: &AgentContext) -> Option<String> {
        Some("fix".into())
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "feature_builder",
    model = "fake/branch",
    max_tokens = 500,
    system = "Build features."
)]
struct FeatureBuilder;
impl FeatureBuilder {
    fn user_prompt(&self, _: &AgentContext) -> Option<String> {
        Some("build".into())
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "doc_writer",
    model = "fake/branch",
    max_tokens = 500,
    system = "Write docs."
)]
struct DocWriter;
impl DocWriter {
    fn user_prompt(&self, _: &AgentContext) -> Option<String> {
        Some("doc".into())
    }
}

// ─────────────────────────────────────────────────────────────────────
// 3. The branch selector — a plain `fn`
// ─────────────────────────────────────────────────────────────────────
//
// `branch` / `branch_default` accept any `impl Fn(&HandoffState) ->
// BranchKey + Send + Sync + 'static`, so bare fn pointers pass through
// without wrapping (capturing closures work too).

fn route_by_kind(h: &HandoffState) -> BranchKey {
    h.find("triager")
        .map(|nv| BranchKey::from(nv.verdict.token.as_str()))
        .unwrap_or_else(|| BranchKey::from("FEATURE"))
}

// ─────────────────────────────────────────────────────────────────────
// 4. Drive it
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), BorgError> {
    // ── A. Build pipeline ──────────────────────────────────────
    //
    // Plan, build UI + logic in parallel, then build-fix-loop up to 3x
    // until PASS.

    let app = seq([
        step(Planner),
        par([step(Coder), step(TestWriter)]),
        // Body ends with `Builder` so the until-predicate sees a verdict
        // carrying the PASS|FAIL token.
        repeat(
            3,
            |v: &Verdict| v.token == "PASS",
            || seq([step(BuildFixer), step(Builder)]),
        ),
    ]);

    // 12 scripted responses:
    //   A: planner(1) + ui(1) + logic(1) + 3×(fixer+builder)(6) = 9
    //   B: triager(1) + chosen-arm(1) = 2
    //   C: builder(1) = 1
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::new(vec![
        // ── Section A ──
        FakeOutcome::ok("plan: build UI then logic then ship"),
        FakeOutcome::ok("<button>OK</button>"),
        FakeOutcome::ok("fn handle_click() {}"),
        FakeOutcome::ok("noop fix"),
        FakeOutcome::ok("FAIL\nmissing semicolon"),
        FakeOutcome::ok("added semicolon"),
        FakeOutcome::ok("FAIL\ntype mismatch"),
        FakeOutcome::ok("coerced type"),
        FakeOutcome::ok("PASS\nbuild green"),
        // ── Section B ──
        FakeOutcome::ok("BUG\nfound a panic in handler"),
        FakeOutcome::ok("fixed the panic"),
        // ── Section C ──
        FakeOutcome::ok("PASS\nall green on first try"),
    ]);

    let mut handoff = HandoffState::new();
    handoff.set_typed::<UserBrief>("a counter app with dark mode".into());
    handoff.set_typed::<WorkingFiles>(HashMap::new());

    let ctx = ExecCtx::new(
        Arc::clone(&llm),
        in_memory_budget(200_000),
        noop_emitter(),
        CancellationToken::new(),
        "showcase-app",
    );

    println!("== A. Build pipeline: seq + par + repeat-until ==");
    let outcome = execute_with_handoff(app, ctx, handoff).await?;
    println!(
        "  steps={}  tokens(in={}, out={})",
        outcome.verdicts.len(),
        outcome.total_input_tokens(),
        outcome.total_output_tokens(),
    );
    for v in &outcome.verdicts {
        println!(
            "    {:<14} token={:<8} body={:?}",
            v.step_name,
            format!("{:?}", v.verdict.token),
            v.verdict.body.lines().next().unwrap_or(""),
        );
    }

    // ── B. Branching — one agent picks the next pipeline ─────────────
    //
    // Triager emits a verdict token (BUG | FEATURE | DOCS); the selector
    // routes into one of three arms. Each arm is independently supervised
    // — a panic in one doesn't touch the others.

    let triage = seq([
        step(Triager),
        branch(
            route_by_kind,
            [
                (BranchKey::from("BUG"), step(BugFixer)),
                (BranchKey::from("FEATURE"), step(FeatureBuilder)),
                (BranchKey::from("DOCS"), step(DocWriter)),
            ],
        ),
    ]);

    let mut handoff_b = HandoffState::new();
    handoff_b.set_typed::<UserBrief>("the date picker crashes on Safari".into());

    let ctx_b = ExecCtx::new(
        Arc::clone(&llm),
        in_memory_budget(50_000),
        noop_emitter(),
        CancellationToken::new(),
        "showcase-branch",
    );

    println!("\n== B. Branch: one agent decides, three pipelines wait ==");
    let outcome_b = execute_with_handoff(triage, ctx_b, handoff_b).await?;
    for v in &outcome_b.verdicts {
        println!(
            "    {:<16} token={:<8} body={:?}",
            v.step_name,
            format!("{:?}", v.verdict.token),
            v.verdict.body.lines().next().unwrap_or(""),
        );
    }

    // ── C. The `pipeline!` macro ─────────────────────────────────────
    //
    // Top-level entries compose sequentially; `parallel { ... }` and
    // `repeat(N, until) { ... }` are recognized as block forms.

    let app2 = borg::pipeline! {
        Builder;
    };

    let ctx_c = ExecCtx::new(
        Arc::clone(&llm),
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "showcase-macro",
    );

    println!("\n== C. The pipeline! macro ==");
    let outcome_c = execute(app2, ctx_c).await?;
    for v in &outcome_c.verdicts {
        println!(
            "    {:<14} token={:<8} body={:?}",
            v.step_name,
            format!("{:?}", v.verdict.token),
            v.verdict.body.lines().next().unwrap_or(""),
        );
    }

    println!("\n✓ All three pipelines ran clean.");
    Ok(())
}
