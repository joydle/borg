//! `ExecCtx::with_lifecycle_events`: a composition on an app's own sink
//! emits only the lifecycle kinds asked for, while its agents keep the
//! full emitter. Plus the parallel executor's naming: a panicked branch
//! is reported under its own name and slot.
//!
//! Run with:
//!   cargo test -p borg --test lifecycle_events --features testing

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::dsl::{par_all, seq};
use borg::events::{EventKind, EventKinds, RecordingEmitter};
use borg::testing::PanickingAgent;
use borg::{
    execute_with_handoff, in_memory_budget, AgentContext, BorgAgent, BorgError, BorgEvent,
    CancellationToken, Composition, EventEmitter, ExecCtx, HandoffState, Verdict,
};
use std::sync::Arc;

/// Emits one app event of its own (a `Cancelled` row stands in for any
/// app-defined event) and ships.
struct Chatty(&'static str);

#[async_trait]
impl BorgAgent for Chatty {
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
    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        ctx.emitter
            .emit(Arc::new(BorgEvent::Cancelled {
                composition_id: Arc::from(self.0),
                reason: "app-event".into(),
            }))
            .await;
        Ok(Verdict::empty())
    }
}

fn ctx(sink: Arc<RecordingEmitter>) -> ExecCtx {
    ExecCtx::deterministic(
        in_memory_budget(1_000_000),
        sink as Arc<dyn EventEmitter>,
        CancellationToken::new(),
        "app",
    )
}

fn comp() -> Composition {
    seq([
        Composition::single("a", Chatty("a")),
        par_all([
            Composition::single("b", Chatty("b")),
            Composition::single("c", Chatty("c")),
        ]),
    ])
}

#[tokio::test]
async fn by_default_every_lifecycle_event_reaches_the_sink() {
    let sink = Arc::new(RecordingEmitter::new());
    let _ = execute_with_handoff(comp(), ctx(sink.clone()), HandoffState::new())
        .await
        .unwrap();
    let kinds = sink.kinds();
    assert!(kinds.contains(&"composition_started"));
    assert!(kinds.contains(&"step_started"));
    assert_eq!(kinds.iter().filter(|k| **k == "cancelled").count(), 3);
}

#[tokio::test]
async fn with_none_only_the_agents_own_events_reach_the_sink() {
    let sink = Arc::new(RecordingEmitter::new());
    let ctx = ctx(sink.clone()).with_lifecycle_events(EventKinds::NONE);
    let _ = execute_with_handoff(comp(), ctx, HandoffState::new())
        .await
        .unwrap();
    let events = sink.snapshot();
    assert_eq!(events.len(), 3, "three app events, no lifecycle rows");
    assert!(events
        .iter()
        .all(|e| matches!(e, BorgEvent::Cancelled { reason, .. } if reason == "app-event")));
}

#[tokio::test]
async fn a_chosen_subset_is_emitted() {
    let sink = Arc::new(RecordingEmitter::new());
    let kinds = EventKinds::only(&[EventKind::StepCompleted]);
    let ctx = ctx(sink.clone()).with_lifecycle_events(kinds);
    let _ = execute_with_handoff(comp(), ctx, HandoffState::new())
        .await
        .unwrap();
    let got = sink.kinds();
    assert!(got
        .iter()
        .all(|k| *k == "step_completed" || *k == "cancelled"));
    assert!(got.contains(&"step_completed"));
    assert!(!got.contains(&"composition_started"));
}

#[tokio::test]
async fn a_panicked_branch_keeps_its_name() {
    let sink = Arc::new(RecordingEmitter::new());
    let comp = par_all([
        Composition::single("fine", Chatty("fine")),
        Composition::single("boom", PanickingAgent { id: "boom" }),
    ]);
    let out = execute_with_handoff(comp, ctx(sink.clone()), HandoffState::new())
        .await
        .unwrap();
    let failed: Vec<_> = out
        .verdicts
        .iter()
        .filter(|v| v.is_error)
        .map(|v| v.step_name.as_str())
        .collect();
    assert_eq!(failed, ["boom"], "named after its branch, not `panic`");
    let positions: Vec<(String, u8)> = sink
        .snapshot()
        .iter()
        .filter_map(|e| match e {
            BorgEvent::StepCompleted {
                success: false,
                position,
                composition_id,
                ..
            } => Some((composition_id.to_string(), *position)),
            _ => None,
        })
        .collect();
    // The single closes its own step (its panic is caught at `invoke`),
    // and the parallel reports the branch at its own position.
    assert_eq!(
        positions,
        [
            ("app::par#1::boom".to_string(), 0),
            ("app::par".to_string(), 1)
        ],
        "the panicked branch's own position"
    );
}
