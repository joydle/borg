//! Dogfood example — critic agent reviews input, summarizer compresses
//! its verdict into one line. Demonstrates:
//!
//! - `#[derive(BorgAgentDerive)]` proc macro (zero trait boilerplate)
//! - Typed handoff slots (refactor-safe key access)
//! - `borg::testing::FakeLlmClient`
//! - `Composition::sequential(...)` with handoff between steps
//! - Free helpers `in_memory_budget()` / `noop_emitter()`
//!
//! Run with: `cargo run --example dogfood -p borg --features testing`

use borg::testing::{FakeLlmClient, FakeOutcome};
use borg::*;
use std::sync::Arc;

// ── Typed handoff slot — schema checked at compile time ──────────────

struct CodeUnderReview;
impl HandoffSlot for CodeUnderReview {
    type Value = String;
    const KEY: &'static str = "code";
}

// ── Two agents via #[derive(BorgAgentDerive)] ───────────────────────
// The derive generates everything from `#[borg(...)]`; you write only
// `user_prompt`.

#[derive(BorgAgentDerive)]
#[borg(
    id = "critic",
    model = "fake/critic",
    max_tokens = 500,
    system = "You are a code reviewer. Output PASS or FAIL on line 1, then an explanation.",
    known_verdicts = &["PASS", "FAIL"],
)]
struct Critic;

impl Critic {
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        let code = ctx.handoff.get_typed::<CodeUnderReview>()?;
        Some(format!("review this code:\n{code}"))
    }
}

#[derive(BorgAgentDerive)]
#[borg(
    id = "summarizer",
    model = "fake/summarizer",
    max_tokens = 200,
    system = "Summarize the previous critic verdict in one line."
)]
struct Summarizer;

impl Summarizer {
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        // Prior step's named verdict from handoff history.
        let critic_verdict = ctx.handoff.find("critic")?;
        Some(format!("Summarize: {}", critic_verdict.verdict.body))
    }
}

// ── Wire it up ───────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), BorgError> {
    // Scripted outputs, consumed in order.
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::new(vec![
        FakeOutcome::ok("PASS\nThe code is fine."),
        FakeOutcome::ok("Critic approved the code."),
    ]);

    // `Composition::single` takes any `BorgAgent + 'static` and boxes internally.
    let pipeline = Composition::sequential([
        Composition::single_with("critic", Critic)
            .output_key("review")
            .into(),
        Composition::single("summarizer", Summarizer),
    ]);

    // Seed input via the typed slot; the first step reads it back.
    let mut handoff = HandoffState::new();
    handoff.set_typed::<CodeUnderReview>("fn main() { println!(\"hi\"); }".into());

    let ctx = ExecCtx::new(
        llm,
        in_memory_budget(10_000),
        noop_emitter(),
        CancellationToken::new(),
        "dogfood",
    );

    let outcome = execute_with_handoff(pipeline, ctx, handoff).await?;

    println!(
        "Steps: {}  tokens(in={}, out={})",
        outcome.verdicts.len(),
        outcome.total_input_tokens(),
        outcome.total_output_tokens(),
    );
    for v in &outcome.verdicts {
        println!(
            "  {:<12} token={:<6} body={:?}",
            v.step_name,
            format!("{:?}", v.verdict.token),
            v.verdict.body.lines().next().unwrap_or(""),
        );
    }

    // ── run_one_typed: one agent, parse its JSON output in one call ─────
    println!();
    println!("== run_one_typed demo ==");

    #[derive(serde::Deserialize, Default)]
    struct Score {
        rating: u8,
        note: String,
    }

    #[derive(BorgAgentDerive)]
    #[borg(
        id = "scorer",
        model = "fake/scorer",
        max_tokens = 100,
        system = "Score 0-10; return JSON {rating, note}."
    )]
    struct Scorer;

    impl Scorer {
        fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
            Some("Rate this commit message: 'fix bug'".into())
        }
    }

    // Empty `known_verdicts` → the full response text becomes `verdict.body`.
    let llm2: Arc<dyn AgentLlmClient> = FakeLlmClient::new(vec![FakeOutcome::ok(
        "{\"rating\": 4, \"note\": \"too terse\"}",
    )]);
    let ctx2 = ExecCtx::new(
        llm2,
        in_memory_budget(1_000),
        noop_emitter(),
        CancellationToken::new(),
        "dogfood-typed",
    );
    let score: Score = run_one_typed("scorer", Scorer, ctx2, HandoffState::new()).await?;
    println!("  rating={} note={:?}", score.rating, score.note);

    // ── #[derive(BorgToolDerive)]: impl Tool from a struct + #[tool(..)] ─
    // You write the inherent `execute`; the derive forwards the trait
    // method to it. Same pattern as BorgAgentDerive.
    println!();
    println!("== BorgTool derive demo ==");

    use borg::tools::{Registry, ToolCallRequest};

    #[derive(borg::BorgToolDerive)]
    #[tool(
        name = "reverse",
        description = "Reverses the given text.",
        context = (),
        input_schema = ::serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        }),
    )]
    struct Reverse;

    impl Reverse {
        async fn execute(
            &self,
            input: serde_json::Value,
            _ctx: &mut (),
        ) -> Result<borg::ToolCallResult, BorgError> {
            let text = input["text"].as_str().ok_or_else(|| BorgError::Internal {
                message: "missing `text` arg".into(),
            })?;
            borg::ToolCallResult::ok(text.chars().rev().collect::<String>())
        }
    }

    let registry = Registry::<()>::new().with(Reverse);
    println!(
        "  registered: {:?}",
        registry
            .schemas()
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );

    let mut tool_ctx = ();
    let dispatched = registry
        .dispatch(
            ToolCallRequest::new(
                "reverse",
                serde_json::json!({ "text": "hello world" }),
                "call-1",
            ),
            &mut tool_ctx,
        )
        .await?;
    println!(
        "  reverse(\"hello world\") = {:?}",
        dispatched.result.output.unwrap()
    );

    Ok(())
}
