//! `dsl::heartbeat` (liveness beats around a long, silent step),
//! `dsl::deferred` (a composition built when it is reached, from the
//! handoff so far) and `HandoffState::output_as` (a step's typed output).
//!
//! Run with:
//!   cargo test -p borg --test heartbeat_and_deferred --features testing

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::dsl::{deferred, heartbeat, par_all, seq, step_if, step_with};
use borg::events::{EventKind, EventKinds, RecordingEmitter};
use borg::{
    execute_with_handoff, in_memory_budget, noop_emitter, AgentContext, BorgAgent, BorgError,
    BorgEvent, CancellationToken, Composition, ExecCtx, HandoffState, Verdict,
};
use std::sync::Arc;
use std::time::Duration;

/// A step that works for `for_ms` (tokio's clock) and answers `body`.
struct Work {
    id: &'static str,
    for_ms: u64,
    body: String,
}

#[async_trait]
impl BorgAgent for Work {
    fn agent_id(&self) -> &'static str {
        self.id
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
    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        ctx.cancel
            .run_until_cancelled(tokio::time::sleep(Duration::from_millis(self.for_ms)))
            .await
            .ok_or(BorgError::Cancelled)?;
        let mut v = Verdict::empty();
        v.body = self.body.clone().into();
        Ok(v)
    }
}

fn work(id: &'static str, for_ms: u64, body: &str) -> Composition {
    Composition::single(
        id,
        Work {
            id,
            for_ms,
            body: body.to_string(),
        },
    )
}

fn ctx(emitter: Arc<RecordingEmitter>, cancel: CancellationToken) -> ExecCtx {
    ExecCtx::deterministic(in_memory_budget(1_000_000), emitter, cancel, "round")
}

/// `(name, beat, elapsed_ms)` of every heartbeat recorded.
fn beats(rec: &RecordingEmitter) -> Vec<(String, u32, u64)> {
    rec.snapshot()
        .into_iter()
        .filter_map(|e| match e {
            BorgEvent::Heartbeat {
                name,
                beat,
                elapsed_ms,
                ..
            } => Some((name.to_string(), beat, elapsed_ms)),
            _ => None,
        })
        .collect()
}

#[tokio::test(start_paused = true)]
async fn a_heartbeat_beats_on_its_period_while_its_step_works_and_stops_with_it() {
    let rec = Arc::new(RecordingEmitter::new());
    let comp = seq([
        heartbeat(
            "drafting",
            Duration::from_millis(2500),
            work("drafter", 6_000, "plan"),
        )
        .at_start(),
        heartbeat(
            "reviewing",
            Duration::from_millis(2500),
            work("reviewer", 2_000, "ok"),
        ),
    ]);
    let out = execute_with_handoff(
        comp,
        ctx(rec.clone(), CancellationToken::new()),
        HandoffState::new(),
    )
    .await
    .unwrap();
    assert_eq!(
        beats(&rec),
        vec![
            ("drafting".into(), 0, 0),
            ("drafting".into(), 1, 2_500),
            ("drafting".into(), 2, 5_000),
        ],
        "one at the start, then one per period; a step shorter than a period beats never"
    );
    // Transparent: the steps are what they would be unwrapped, ids included.
    let ids: Vec<String> = rec
        .snapshot()
        .into_iter()
        .filter_map(|e| match e {
            BorgEvent::StepStarted { composition_id, .. } => Some(composition_id.to_string()),
            BorgEvent::Heartbeat { composition_id, .. } => Some(composition_id.to_string()),
            _ => None,
        })
        .collect();
    assert!(ids.iter().all(|id| id.starts_with("round::seq")), "{ids:?}");
    assert!(ids.iter().all(|id| !id.contains("drafting")), "{ids:?}");
    let names: Vec<&str> = out.verdicts.iter().map(|v| v.step_name.as_ref()).collect();
    assert_eq!(names, ["drafter", "reviewer"]);
    assert_eq!(
        out.handoff.find("drafter").unwrap().verdict.body.as_ref(),
        "plan"
    );
}

#[tokio::test(start_paused = true)]
async fn a_heartbeat_ends_when_its_step_is_cancelled() {
    let rec = Arc::new(RecordingEmitter::new());
    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(3_000)).await;
        stop.cancel();
    });
    let comp = heartbeat(
        "drafting",
        Duration::from_millis(1_000),
        work("drafter", 60_000, "plan"),
    );
    let err = execute_with_handoff(comp.into(), ctx(rec.clone(), cancel), HandoffState::new())
        .await
        .unwrap_err();
    assert!(err.is_cancelled(), "{err}");
    let n = beats(&rec).len();
    assert!(
        (2..=3).contains(&n),
        "beats until the cancel, none after: {n}"
    );
}

#[tokio::test(start_paused = true)]
async fn heartbeats_pass_a_lifecycle_filter_on_their_own() {
    let rec = Arc::new(RecordingEmitter::new());
    let ctx = ctx(rec.clone(), CancellationToken::new())
        .with_lifecycle_events(EventKinds::only(&[EventKind::Heartbeat]));
    let comp = heartbeat(
        "drafting",
        Duration::from_millis(1_000),
        work("drafter", 2_500, "plan"),
    );
    let _ = execute_with_handoff(comp.into(), ctx, HandoffState::new())
        .await
        .unwrap();
    assert_eq!(
        rec.kinds(),
        vec!["heartbeat", "heartbeat"],
        "only the beats"
    );
}

#[tokio::test]
async fn a_deferred_composition_is_built_from_what_the_steps_before_it_wrote() {
    let comp = seq([
        step_with(Work {
            id: "planner",
            for_ms: 0,
            body: r#"["intro", "summary"]"#.into(),
        })
        .output_key("sections")
        .into(),
        deferred(|h| {
            let names: Vec<String> = h.output_as("sections").unwrap_or_default();
            par_all(names.into_iter().map(|n| {
                let id: &'static str = Box::leak(n.into_boxed_str());
                work(id, 0, id)
            }))
        }),
    ]);
    let out = execute_with_handoff(
        comp,
        ctx(Arc::new(RecordingEmitter::new()), CancellationToken::new()),
        HandoffState::new(),
    )
    .await
    .unwrap();
    let mut written: Vec<&str> = out
        .verdicts
        .iter()
        .skip(1)
        .map(|v| v.verdict.body.as_ref())
        .collect();
    written.sort();
    assert_eq!(written, ["intro", "summary"]);
}

#[tokio::test]
async fn a_deferred_builder_sees_an_empty_handoff_when_fingerprinted() {
    let comp = deferred(|h| {
        let n: Vec<String> = h.output_as("sections").unwrap_or_default();
        par_all(n.into_iter().map(|_| work("x", 0, "x")))
    });
    assert!(borg::composition_fingerprint(&comp).starts_with("Deferred<"));
}

#[derive(serde::Deserialize)]
struct Review {
    approved: bool,
}

#[tokio::test]
async fn step_if_reads_a_steps_typed_output() {
    for (review, revised) in [
        (r#"{"approved": false}"#, true),
        (r#"{"approved": true}"#, false),
        ("not json", false),
    ] {
        let comp = seq([
            step_with(Work {
                id: "reviewer",
                for_ms: 0,
                body: review.into(),
            })
            .output_key("review")
            .into(),
            step_if(
                |h| h.output_as::<Review>("review").is_some_and(|r| !r.approved),
                work("revise", 0, "v2"),
            ),
        ]);
        let out = execute_with_handoff(
            comp,
            ExecCtx::deterministic(
                in_memory_budget(1_000),
                noop_emitter(),
                CancellationToken::new(),
                "t",
            ),
            HandoffState::new(),
        )
        .await
        .unwrap();
        assert_eq!(out.handoff.find("revise").is_some(), revised, "{review}");
    }
}

#[test]
fn output_as_reads_a_body_a_structured_value_and_a_plain_string() {
    let mut h = HandoffState::new();
    h.set("body", r#"{"approved": true}"#);
    h.set("value", serde_json::json!({"approved": false}));
    h.set("word", "hello");
    assert!(h.output_as::<Review>("body").unwrap().approved);
    assert!(!h.output_as::<Review>("value").unwrap().approved);
    assert_eq!(h.output_as::<String>("word").as_deref(), Some("hello"));
    assert!(h.output_as::<Review>("word").is_none());
    assert!(h.output_as::<Review>("missing").is_none());
}
