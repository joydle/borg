//! borg's app-integration pieces in one runnable file:
//!
//! 1. a request-scoped task-local (a spend ledger) carried into every
//!    `par_all` branch by a `ContextCarrier`;
//! 2. a typed provider ladder (`ladder::Ladder`) with a fall-through rule,
//!    a per-rung filter, a terminal refusal passed through, and the
//!    failures merged into one error;
//! 3. the same rules over compositions (`fallback_ladder`);
//! 4. a composition on an app's own sink with its lifecycle rows off,
//!    inside a pool carved out of the session budget (`sub_budget`).
//!
//! Run with:
//!   cargo run -p borg --example ladders_and_context

use async_trait::async_trait;
use borg::dsl::{fallback_ladder, par_all, seq};
use borg::events::EventKinds;
use borg::ladder::{Ladder, LadderError, LadderRung, Rung};
use borg::propagation::TaskLocal;
use borg::{
    execute_with_handoff, in_memory_budget, noop_emitter, sub_budget, AgentContext, BorgAgent,
    BorgError, CancellationToken, Composition, ExecCtx, HandoffState, RetryClass, Verdict,
};
use std::sync::{Arc, Mutex};

tokio::task_local! {
    /// The request-scoped ledger each step records its work into.
    static SPEND: Arc<Mutex<Vec<(&'static str, u32)>>>;
}

/// A deterministic step that records into the request's ledger.
struct Task(&'static str);

#[async_trait]
impl BorgAgent for Task {
    fn agent_id(&self) -> &'static str {
        self.0
    }
    fn model(&self) -> &str {
        "deterministic"
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
    async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
        // Runs on a branch task of its own; the ledger is still here.
        let _ = SPEND.try_with(|l| l.lock().unwrap().push((self.0, 1)));
        let mut v = Verdict::empty();
        v.body = self.0.into();
        Ok(v)
    }
}

/// A provider failure as an app would type it.
#[derive(Debug)]
struct ProviderError {
    message: &'static str,
    content_policy: bool,
    cost_units: u32,
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

async fn call(model: &'static str, fail: Option<ProviderError>) -> Result<String, ProviderError> {
    match fail {
        Some(e) => Err(e),
        None => Ok(format!("answered by {model}")),
    }
}

#[tokio::main]
async fn main() -> Result<(), BorgError> {
    // ── 1. Context into every branch ─────────────────────────────────
    let ledger = Arc::new(Mutex::new(Vec::new()));
    let ctx = ExecCtx::deterministic(
        in_memory_budget(1_000_000),
        noop_emitter(),
        CancellationToken::new(),
        "request-1",
    )
    .with_carrier(TaskLocal::new(&SPEND));
    let stages = seq([
        par_all([Composition::single("outline", Task("outline"))]),
        par_all([
            Composition::single("intro", Task("intro")),
            Composition::single("body", Task("body")),
            Composition::single("summary", Task("summary")),
        ]),
    ]);
    let _ = SPEND
        .scope(
            ledger.clone(),
            execute_with_handoff(stages, ctx, HandoffState::new()),
        )
        .await?;
    println!(
        "1. ledger entries recorded across branches: {:?}",
        ledger.lock().unwrap().len()
    );

    // ── 2. A typed provider ladder ───────────────────────────────────
    let has_attachment = true;
    let result = Ladder::new()
        .rung(
            "model-a",
            call(
                "model-a",
                Some(ProviderError {
                    message: "520 upstream",
                    content_policy: false,
                    cost_units: 0,
                }),
            ),
        )
        // A model that can't read attachments: skipped for this request.
        .push(Rung::new("model-b", call("model-b", None)).when(move |_| !has_attachment))
        .rung("model-a@mirror", call("model-a@mirror", None))
        .falls_through(|e: &ProviderError| !e.content_policy)
        .on_fall(|f, to| println!("   fell from {} to {to:?}: {}", f.rung, f.error))
        .run_shipped()
        .await;
    let shipped = result.map_err(|_| BorgError::Internal {
        message: "ladder".into(),
    })?;
    println!("2. {} (rung {})", shipped.value, shipped.rung);

    // A refusal stops the ladder and comes back as-is; exhaustion keeps
    // every rung's error, merged into one here.
    let err = Ladder::new()
        .rung(
            "provider-a",
            call(
                "provider-a",
                Some(ProviderError {
                    message: "timeout",
                    content_policy: false,
                    cost_units: 4,
                }),
            ),
        )
        .rung(
            "provider-b",
            call(
                "provider-b",
                Some(ProviderError {
                    message: "run failed",
                    content_policy: false,
                    cost_units: 2,
                }),
            ),
        )
        .run()
        .await
        .unwrap_err();
    assert!(matches!(err, LadderError::Exhausted { .. }));
    let merged = err
        .merge(|mut later, earlier| {
            later.cost_units += earlier.cost_units;
            later
        })
        .unwrap();
    println!(
        "   both providers failed: `{merged}`, {} cost units in total",
        merged.cost_units
    );

    // ── 3. The same rules over compositions ──────────────────────────
    struct Down;
    #[async_trait]
    impl BorgAgent for Down {
        fn agent_id(&self) -> &'static str {
            "primary"
        }
        fn model(&self) -> &str {
            "deterministic"
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
        async fn invoke(&self, _ctx: &AgentContext) -> Result<Verdict, BorgError> {
            Err(BorgError::llm("model_not_available", RetryClass::Terminal))
        }
    }
    let comp = fallback_ladder([
        LadderRung::new(Composition::single("primary", Down)),
        LadderRung::new(Composition::single("backup", Task("backup"))),
    ])
    .falls_through(|e| e.is_retryable() || e.to_string().contains("model_not_available"));

    // ── 4. On the app's sink, no lifecycle rows, in a carved-out pool ──
    let session = in_memory_budget(1_000_000);
    let ctx = ExecCtx::deterministic(
        sub_budget(session, 300_000),
        noop_emitter(),
        CancellationToken::new(),
        "review-step",
    )
    .with_lifecycle_events(EventKinds::NONE);
    let out = execute_with_handoff(comp.build(), ctx, HandoffState::new()).await?;
    println!(
        "3. shipped by `{}`, 4. with no lifecycle rows",
        out.verdicts[0].verdict.body
    );
    Ok(())
}
