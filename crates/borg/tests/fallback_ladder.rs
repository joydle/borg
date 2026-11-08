//! `fallback_ladder`: a caller's fall-through rule, per-rung `when`
//! filters, a terminal error handed back as-is, every rung's error kept
//! on exhaustion, and an `on_fall` hook that records per-rung outcomes on
//! the handoff.
//!
//! Run with:
//!   cargo test -p borg --test fallback_ladder --features testing

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::dsl::{fallback_ladder, seq};
use borg::ladder::LadderRung;
use borg::{
    execute_with_handoff, in_memory_budget, noop_emitter, AgentContext, BorgAgent, BorgError,
    CancellationToken, Composition, ExecCtx, HandoffState, RetryClass, Verdict,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A rung: ships its name, or fails with `fail`'s error. Counts runs.
struct Rung {
    name: &'static str,
    fail: Option<fn() -> BorgError>,
    runs: Arc<AtomicUsize>,
}

#[async_trait]
impl BorgAgent for Rung {
    fn agent_id(&self) -> &'static str {
        self.name
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
        self.runs.fetch_add(1, Ordering::SeqCst);
        match self.fail {
            Some(f) => Err(f()),
            None => {
                let mut v = Verdict::empty();
                v.body = self.name.into();
                Ok(v)
            }
        }
    }
}

fn outage() -> BorgError {
    BorgError::llm("503 upstream", RetryClass::Retryable)
}
fn gone() -> BorgError {
    BorgError::llm("model_not_available", RetryClass::Terminal)
}
fn refused() -> BorgError {
    BorgError::llm("content_policy: refused", RetryClass::Terminal)
}

struct Rungs {
    runs: Vec<(&'static str, Arc<AtomicUsize>)>,
}

impl Rungs {
    fn new() -> Self {
        Self { runs: Vec::new() }
    }
    fn rung(&mut self, name: &'static str, fail: Option<fn() -> BorgError>) -> Composition {
        let runs = Arc::new(AtomicUsize::new(0));
        self.runs.push((name, runs.clone()));
        Composition::single(name, Rung { name, fail, runs })
    }
    fn ran(&self, name: &str) -> usize {
        self.runs
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, r)| r.load(Ordering::SeqCst))
            .unwrap()
    }
}

fn ctx() -> ExecCtx {
    ExecCtx::deterministic(
        in_memory_budget(1_000_000),
        noop_emitter(),
        CancellationToken::new(),
        "test",
    )
}

async fn run(comp: impl Into<Composition>) -> Result<borg::ExecutionOutcome, BorgError> {
    execute_with_handoff(comp.into(), ctx(), HandoffState::new()).await
}

#[tokio::test]
async fn the_callers_rule_decides_what_falls_through() {
    let mut r = Rungs::new();
    // A terminal "model gone" would stop a plain `fallback`; this ladder
    // says it falls through.
    let ladder = fallback_ladder([r.rung("primary", Some(gone)), r.rung("backup", None)])
        .falls_through(
            |e| matches!(e, BorgError::Llm { message, .. } if !message.contains("content_policy")),
        );
    let out = run(ladder).await.unwrap();
    assert_eq!(&*out.verdicts[0].verdict.body, "backup");
    assert_eq!((r.ran("primary"), r.ran("backup")), (1, 1));
}

#[tokio::test]
async fn the_default_rule_is_is_retryable() {
    let mut r = Rungs::new();
    let out = run(fallback_ladder([
        r.rung("a", Some(outage)),
        r.rung("b", None),
    ]))
    .await
    .unwrap();
    assert_eq!(&*out.verdicts[0].verdict.body, "b");

    let mut r = Rungs::new();
    let err = run(fallback_ladder([
        r.rung("a", Some(gone)),
        r.rung("b", None),
    ]))
    .await
    .unwrap_err();
    assert!(err.to_string().contains("model_not_available"));
    assert_eq!(r.ran("b"), 0);
}

#[tokio::test]
async fn a_terminal_refusal_is_passed_through_as_is() {
    let mut r = Rungs::new();
    let err = run(fallback_ladder([
        r.rung("primary", Some(outage)),
        r.rung("mirror", Some(refused)),
        r.rung("backup", None),
    ])
    .falls_through(|e| e.is_retryable()))
    .await
    .unwrap_err();
    match err {
        BorgError::Llm { message, class, .. } => {
            assert_eq!(message, "content_policy: refused");
            assert_eq!(class, RetryClass::Terminal);
        }
        other => panic!("expected the refusal itself, got {other:?}"),
    }
    assert_eq!(r.ran("backup"), 0, "a refusal never reaches the next rung");
}

#[tokio::test]
async fn rungs_filter_themselves_per_request_and_per_failure() {
    let mut r = Rungs::new();
    let ladder = fallback_ladder([
        LadderRung::new(r.rung("primary", Some(outage))),
        // This request needs a reference; the rung cannot take one.
        LadderRung::new(r.rung("no_refs", None)).when(|h, _| h.get("references").is_none()),
        // Only after an outage-class failure of an earlier rung.
        LadderRung::new(r.rung("mirror", None))
            .when(|_, failures| failures.iter().any(|f| f.error.to_string().contains("503"))),
    ]);
    let mut handoff = HandoffState::new();
    handoff.set("references", ["source.png"]);
    let out = execute_with_handoff(ladder.build(), ctx(), handoff)
        .await
        .unwrap();
    assert_eq!(&*out.verdicts[0].verdict.body, "mirror");
    assert_eq!(r.ran("no_refs"), 0);
}

#[tokio::test]
async fn exhaustion_keeps_every_rungs_error_in_order() {
    let mut r = Rungs::new();
    let err = run(fallback_ladder([
        r.rung("a", Some(outage)),
        r.rung("b", Some(outage)),
    ]))
    .await
    .unwrap_err();
    let BorgError::FallbackExhausted { failures } = &err else {
        panic!("expected FallbackExhausted, got {err:?}");
    };
    let names: Vec<_> = failures.iter().map(|f| (&*f.rung, f.index)).collect();
    assert_eq!(names, [("a", 0), ("b", 1)]);
    assert!(failures.iter().all(|f| f.error.is_retryable()));
    assert_eq!(err.failure_cause(), "fallback-exhausted");
    assert!(!err.is_retryable());
    assert!(err
        .to_string()
        .starts_with("every fallback rung failed: `a`:"));
}

#[tokio::test]
async fn no_applicable_rung_is_an_empty_exhaustion() {
    let mut r = Rungs::new();
    let err = run(fallback_ladder([
        LadderRung::new(r.rung("a", None)).when(|_, _| false)
    ]))
    .await
    .unwrap_err();
    assert!(matches!(err, BorgError::FallbackExhausted { ref failures } if failures.is_empty()));
}

#[tokio::test]
async fn on_fall_records_each_rungs_outcome_on_the_handoff() {
    let mut r = Rungs::new();
    let ladder = fallback_ladder([
        r.rung("provider-a", Some(outage)),
        r.rung("provider-b", None),
    ])
    .on_fall(|failure, to, handoff| {
        // e.g. the failed rung's cost or request id, keyed by rung.
        handoff.set(format!("fell:{}", failure.rung), to.map(str::to_string));
    });
    let out = run(ladder).await.unwrap();
    assert_eq!(
        out.handoff.get_as::<Option<String>>("fell:provider-a"),
        Some(Some("provider-b".to_string()))
    );
}

#[tokio::test]
async fn a_rungs_own_rule_beats_the_ladders_and_it_nests_in_seq() {
    let mut r = Rungs::new();
    let ladder = fallback_ladder([
        LadderRung::new(r.rung("a", Some(gone))).falls_through(|_| true),
        LadderRung::new(r.rung("b", None)),
    ]);
    let out = run(seq([Composition::from(ladder), r.rung("after", None)]))
        .await
        .unwrap();
    let bodies: Vec<_> = out
        .verdicts
        .iter()
        .map(|v| v.verdict.body.to_string())
        .collect();
    assert_eq!(bodies, ["b", "after"]);
}

#[tokio::test]
async fn a_cancel_never_falls_through() {
    let mut r = Rungs::new();
    let err = run(fallback_ladder([
        r.rung("a", Some(|| BorgError::Cancelled)),
        r.rung("b", None),
    ])
    .falls_through(|_| true))
    .await
    .unwrap_err();
    assert!(err.is_cancelled());
    assert_eq!(r.ran("b"), 0);
}
