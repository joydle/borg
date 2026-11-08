//! Production-shape tests covering composition execution, parallelism,
//! handoff, error policies, cancellation, panic isolation, and event
//! ordering. Exercises borg as a whole through the public `testing`
//! module so downstream consumers can reuse the same fakes.

use crate::budget::{Budget, InMemoryBudget};
use crate::composition::{BranchKey, Composition, ErrorPolicy, RestartPolicy};
use crate::error::BorgError;
use crate::events::{EventEmitter, NoopEmitter, RecordingEmitter};
use crate::executor::{execute, ExecCtx};
use crate::llm::AgentLlmClient;
use crate::testing::{
    CancellationObservingAgent, FakeLlmClient, FakeOutcome, HandoffAgent, HandoffEchoAgent,
    PanickingAgent, ScriptedAgent,
};
use crate::verdict::Verdict;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

// ── Helpers ────────────────────────────────────────────────────────────

fn ctx_for(
    llm: Arc<dyn AgentLlmClient>,
    budget: Arc<dyn Budget>,
    emitter: Arc<dyn EventEmitter>,
    cancel: CancellationToken,
) -> ExecCtx {
    ExecCtx::new(llm, budget, emitter, cancel, "test")
}

// ── Sequential pipes via handoff ──────────────────────────────────────

#[tokio::test]
async fn sequential_pipes_verdict_into_next_step_via_handoff() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("PASS\nfirst body"),
        FakeOutcome::ok("DONE\nsecond body"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let first = Composition::single_builder(
        "first",
        Box::new(ScriptedAgent::new("first").with_verdicts(&["PASS", "FAIL"])),
    )
    .output_key("first_out")
    .build();
    let echoer = Composition::single_builder(
        "echoer",
        Box::new(HandoffEchoAgent {
            id: "echoer",
            source_step: "first",
        }),
    )
    .build();

    let comp = Composition::sequential_builder()
        .step(first)
        .step(echoer)
        .build();

    let outcome = execute(
        comp,
        ctx_for(
            llm.clone(),
            budget.clone(),
            emitter,
            CancellationToken::new(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdicts.len(), 2);
    assert_eq!(outcome.verdicts[0].step_name, "first");
    assert_eq!(outcome.verdicts[0].verdict.token, "PASS");
    assert_eq!(outcome.verdicts[1].step_name, "echoer");
    assert_eq!(llm.calls(), 2);
    assert_eq!(
        outcome.handoff.get("first_out").unwrap(),
        &serde_json::Value::String("first body".into())
    );
}

// ── Parallel CollectAll keeps every branch running, errors included ───

#[tokio::test]
async fn parallel_collect_all_returns_every_outcome_including_errors() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("OK_A"),
        FakeOutcome::terminal_err("intentional B failure"),
        FakeOutcome::ok("OK_C"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::parallel_builder()
        .branch(
            "a",
            Composition::single_builder("a_agent", Box::new(ScriptedAgent::new("a"))).build(),
        )
        .branch(
            "b",
            Composition::single_builder("b_agent", Box::new(ScriptedAgent::new("b"))).build(),
        )
        .branch(
            "c",
            Composition::single_builder("c_agent", Box::new(ScriptedAgent::new("c"))).build(),
        )
        .on_error(ErrorPolicy::CollectAll)
        .max_concurrency(3)
        .build();

    let outcome = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdicts.len(), 3);
    let succeeded = outcome.verdicts.iter().filter(|v| !v.is_error).count();
    let failed = outcome.verdicts.iter().filter(|v| v.is_error).count();
    assert_eq!(succeeded, 2, "expected 2 successes");
    assert_eq!(failed, 1, "expected 1 error");
    assert_eq!(llm.calls(), 3);
}

// ── Parallel FailFast cancels in-flight siblings ──────────────────────

#[tokio::test]
async fn parallel_failfast_cancels_in_flight_siblings() {
    // Drive cancel directly rather than racing B's failure against A/C's
    // spawn-and-poll: this tests the cancel-propagation mechanism FailFast
    // relies on without the timing race.
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("unused");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let parent_cancel = CancellationToken::new();

    let a_observed = Arc::new(AtomicBool::new(false));
    let c_observed = Arc::new(AtomicBool::new(false));

    let comp = Composition::parallel_builder()
        .branch(
            "a",
            Composition::single_builder(
                "a",
                Box::new(CancellationObservingAgent {
                    id: "a",
                    observed_cancel: a_observed.clone(),
                    block_for: Duration::from_secs(60),
                }),
            )
            .build(),
        )
        .branch(
            "c",
            Composition::single_builder(
                "c",
                Box::new(CancellationObservingAgent {
                    id: "c",
                    observed_cancel: c_observed.clone(),
                    block_for: Duration::from_secs(60),
                }),
            )
            .build(),
        )
        .on_error(ErrorPolicy::FailFast)
        .max_concurrency(3)
        .build();

    let cancel_for_spawn = parent_cancel.clone();
    let canceller = tokio::spawn(async move {
        // Let A and C reach their select! cancel arm before firing.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_for_spawn.cancel();
    });

    let result = execute(comp, ctx_for(llm, budget, emitter, parent_cancel)).await;
    canceller.await.unwrap();

    assert!(
        matches!(result, Err(BorgError::Cancelled)),
        "expected Cancelled at the executor boundary, got {result:?}"
    );
    assert!(
        a_observed.load(Ordering::SeqCst),
        "a should observe cancel via select arm"
    );
    assert!(
        c_observed.load(Ordering::SeqCst),
        "c should observe cancel via select arm"
    );
}

// ── Cancellation propagates from parent into a never-returning LLM call ─

#[tokio::test]
async fn cancellation_propagates_into_llm_call() {
    let llm = FakeLlmClient::new(vec![FakeOutcome::Forever]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let cancel = CancellationToken::new();

    let comp =
        Composition::single_builder("blocker", Box::new(ScriptedAgent::new("blocker"))).build();

    let cancel_clone = cancel.clone();
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    let result = execute(comp, ctx_for(llm, budget, emitter, cancel)).await;
    canceller.await.unwrap();

    assert!(
        matches!(result, Err(BorgError::Cancelled)),
        "expected Cancelled, got {result:?}"
    );
}

// ── Child panic surfaces as an error without taking down siblings ─────

#[tokio::test]
async fn child_panic_does_not_take_down_siblings_under_collect_all() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("OK_A"),
        // PanickingAgent panics before reaching the LLM.
        FakeOutcome::ok("OK_C"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::parallel_builder()
        .branch(
            "a",
            Composition::single_builder("a", Box::new(ScriptedAgent::new("a"))).build(),
        )
        .branch(
            "panicker",
            Composition::single_builder("p", Box::new(PanickingAgent { id: "p" })).build(),
        )
        .branch(
            "c",
            Composition::single_builder("c", Box::new(ScriptedAgent::new("c"))).build(),
        )
        .on_error(ErrorPolicy::CollectAll)
        .max_concurrency(3)
        .build();

    let outcome = execute(
        comp,
        ctx_for(llm, budget, emitter, CancellationToken::new()),
    )
    .await
    .unwrap();

    let succeeded = outcome.verdicts.iter().filter(|v| !v.is_error).count();
    let failed = outcome.verdicts.iter().filter(|v| v.is_error).count();
    assert_eq!(succeeded, 2, "two siblings should still succeed");
    assert_eq!(failed, 1, "panic surfaces as one error");
}

// ── RestartPolicy::on_failure recovers from a transient error ─────────

#[tokio::test]
async fn restart_policy_on_failure_recovers_from_transient_error() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("503 first"),
        FakeOutcome::retryable_err("503 second"),
        FakeOutcome::ok("PASS\nfinally"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    // `.without_backoff()` keeps the test hermetic — recovery, not timing.
    let comp = Composition::single_builder(
        "flaky",
        Box::new(ScriptedAgent::new("flaky").with_verdicts(&["PASS", "FAIL"])),
    )
    .restart(RestartPolicy::on_failure(3).without_backoff())
    .build();

    let outcome = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdicts.len(), 1);
    assert_eq!(outcome.verdicts[0].verdict.token, "PASS");
    assert_eq!(llm.calls(), 3, "expected 3 attempts before success");
}

// ── Exhausted restart surfaces RetryExhausted ─────────────────────────

#[tokio::test]
async fn restart_policy_exhausted_surfaces_retry_exhausted() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("503 a"),
        FakeOutcome::retryable_err("503 b"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::single_builder("doomed", Box::new(ScriptedAgent::new("doomed")))
        .restart(RestartPolicy::on_failure(2).without_backoff())
        .build();

    let result = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await;

    match result {
        Err(BorgError::RetryExhausted {
            agent_id, attempts, ..
        }) => {
            assert_eq!(agent_id, "doomed");
            assert_eq!(attempts, 2);
        }
        other => panic!("expected RetryExhausted, got {other:?}"),
    }
    assert_eq!(llm.calls(), 2);
}

// ── on_failure_within bails when the wall-clock window expires ────────

#[tokio::test]
async fn restart_policy_window_bails_before_max_attempts() {
    // The window check uses real `Instant::now()`, ignoring tokio's paused
    // clock. Each failure waits 40ms, so with window=50ms the loop bails
    // after the 1st failure (elapsed ≥ window before attempt 2).
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err_after("503 a", Duration::from_millis(40)),
        FakeOutcome::retryable_err_after("503 b", Duration::from_millis(40)),
        FakeOutcome::retryable_err_after("503 c", Duration::from_millis(40)),
        FakeOutcome::retryable_err_after("503 d", Duration::from_millis(40)),
        FakeOutcome::retryable_err_after("503 e", Duration::from_millis(40)),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::single_builder("flap", Box::new(ScriptedAgent::new("flap")))
        .restart(RestartPolicy::on_failure_within(
            5,
            Duration::from_millis(50),
        ))
        .build();

    let result = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await;

    assert!(result.is_err(), "expected loop to bail under window");
    let calls = llm.calls();
    assert!(
        (1..=2).contains(&calls),
        "expected 1-2 attempts before window cap, got {calls}"
    );
}

// ── window=ZERO bounds retries by max_attempts alone ──────────────────

#[tokio::test]
async fn restart_policy_window_zero_uses_max_attempts_only() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("503 a"),
        FakeOutcome::retryable_err("503 b"),
        FakeOutcome::retryable_err("503 c"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::single_builder("doomed", Box::new(ScriptedAgent::new("doomed")))
        .restart(RestartPolicy::on_failure(3).without_backoff())
        .build();

    let result = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await;

    assert!(matches!(result, Err(BorgError::RetryExhausted { .. })));
    assert_eq!(llm.calls(), 3);
}

// ── Pre-seeded snapshot fast-forwards past completed iterations ───────

#[tokio::test]
async fn checkpoint_pre_seeded_snapshot_fast_forwards_completed_iterations() {
    use crate::checkpoint::{
        composition_fingerprint, CheckpointStore, InMemoryCheckpointStore, Snapshot,
        SNAPSHOT_SCHEMA_VERSION,
    };
    use crate::executor::execute_resumable;
    use std::sync::Arc;

    // 4-iteration loop with only 2 scripted responses: running from
    // scratch would error on iter2. Pre-seeding iter0+iter1 as done means
    // only iter2 and iter3 run, consuming exactly the 2 responses.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("PASS\nthird body"),
        FakeOutcome::ok("PASS\nfourth body"),
    ]);

    let build_comp = || {
        Composition::loop_until_builder(
            |_| {
                crate::composition::Composition::single_builder(
                    "iter",
                    Box::new(ScriptedAgent::new("iter")),
                )
                .build()
            },
            |_| false,
        )
        .max_iterations(4)
        .build()
    };

    // Completed-iter ids follow `{parent}::loop::iterN`. parent is "test"
    // (from ctx_for), so the ids are "test::loop::iter0/iter1".
    let fingerprint = composition_fingerprint(&build_comp());
    let seed = Snapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        composition_fingerprint: fingerprint,
        handoff: crate::handoff::HandoffState::new(),
        completed_steps: vec!["test::loop::iter0".into(), "test::loop::iter1".into()],
        turn_number: 0,
        created_at: std::time::SystemTime::now(),
    };
    store.save("session-α", &seed).await.unwrap();

    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let outcome = execute_resumable(build_comp(), ctx, store.clone(), "session-α")
        .await
        .unwrap();

    // Only iter2 + iter3 ran; snapshot deleted on success.
    assert_eq!(llm.calls(), 2);
    assert_eq!(outcome.verdicts.len(), 2);
    assert!(store.load("session-α").await.unwrap().is_none());
}

// ── Fingerprint mismatch refuses to resume ────────────────────────────

#[tokio::test]
async fn checkpoint_fingerprint_mismatch_surfaces_internal_error() {
    use crate::checkpoint::{
        CheckpointStore, InMemoryCheckpointStore, Snapshot, SNAPSHOT_SCHEMA_VERSION,
    };
    use crate::executor::execute_resumable;
    use std::sync::Arc;

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    // Snapshot under a different fingerprint: resume must refuse with
    // Internal carrying a "fingerprint mismatch" message.
    let wrong_fp = "Single(legacy_step)".to_string();
    let snap = Snapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        composition_fingerprint: wrong_fp,
        handoff: crate::handoff::HandoffState::new(),
        completed_steps: Vec::new(),
        turn_number: 0,
        created_at: std::time::SystemTime::now(),
    };
    store.save("session-γ", &snap).await.unwrap();

    let llm = FakeLlmClient::always_ok("PASS\nbody");
    let comp = Composition::loop_until_builder(
        |_| {
            crate::composition::Composition::single_builder(
                "build",
                Box::new(ScriptedAgent::new("build")),
            )
            .build()
        },
        |_| false,
    )
    .max_iterations(2)
    .build();

    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let result = execute_resumable(comp, ctx, store, "session-γ").await;
    match result {
        Err(BorgError::Internal { message }) => {
            assert!(
                message.contains("fingerprint mismatch"),
                "expected fingerprint-mismatch diagnostic, got: {message}"
            );
        }
        other => panic!("expected Internal with fingerprint mismatch, got {other:?}"),
    }
}

// ── Sequential step-level fast-forward under OnEveryStep ──────────────

#[tokio::test]
async fn checkpoint_sequential_fast_forwards_completed_single_steps() {
    use crate::checkpoint::{
        composition_fingerprint, CheckpointPolicy, CheckpointStore, InMemoryCheckpointStore,
        Snapshot, SNAPSHOT_SCHEMA_VERSION,
    };
    use crate::executor::execute_resumable_with;
    use std::sync::Arc;

    // planner → builder → deployer, with planner+builder pre-seeded as
    // complete. Only deployer runs on resume.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let llm = FakeLlmClient::new(vec![FakeOutcome::ok("PASS\ndeployed")]);

    let build_comp = || {
        Composition::sequential_builder()
            .step(
                Composition::single_builder("planner", Box::new(ScriptedAgent::new("planner")))
                    .build(),
            )
            .step(
                Composition::single_builder("builder", Box::new(ScriptedAgent::new("builder")))
                    .build(),
            )
            .step(
                Composition::single_builder("deployer", Box::new(ScriptedAgent::new("deployer")))
                    .build(),
            )
            .build()
    };
    let fingerprint = composition_fingerprint(&build_comp());

    // Sequential id is `{parent}::seq` (parent "test") and each child
    // nests under its position, so step ids are `test::seq#{idx}::{name}`.
    let seed = Snapshot {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        composition_fingerprint: fingerprint,
        handoff: crate::handoff::HandoffState::new(),
        completed_steps: vec!["test::seq#0::planner".into(), "test::seq#1::builder".into()],
        turn_number: 0,
        created_at: std::time::SystemTime::now(),
    };
    store.save("pipeline-α", &seed).await.unwrap();

    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let outcome = execute_resumable_with(
        build_comp(),
        ctx,
        store.clone(),
        "pipeline-α",
        CheckpointPolicy::OnEveryStep,
    )
    .await
    .unwrap();

    // Only deployer ran; snapshot deleted on success.
    assert_eq!(llm.calls(), 1, "planner + builder should be skipped");
    assert_eq!(outcome.verdicts.len(), 1);
    assert_eq!(outcome.verdicts[0].step_name, "deployer");
    assert!(store.load("pipeline-α").await.unwrap().is_none());
}

// ── Default policy does not checkpoint at Sequential step boundaries ──

#[tokio::test]
async fn checkpoint_sequential_default_policy_does_not_save_mid_pipeline() {
    use crate::checkpoint::{CheckpointStore, InMemoryCheckpointStore};
    use crate::executor::execute_resumable;
    use std::sync::Arc;

    // Default policy is OnLoopIteration: a Sequential without an enclosing
    // LoopUntil must not save at step boundaries (opt-in discipline).
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("PASS\nfirst"),
        FakeOutcome::ok("PASS\nsecond"),
    ]);

    let comp = Composition::sequential_builder()
        .step(Composition::single_builder("a", Box::new(ScriptedAgent::new("a"))).build())
        .step(Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build())
        .build();

    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let _ = execute_resumable(comp, ctx, store.clone(), "pipeline-β").await;
    // Weak check: a successful run deletes its (never-saved) snapshot.
    // Proving no mid-pipeline save would need a save-recording store.
    assert!(store.load("pipeline-β").await.unwrap().is_none());
    assert_eq!(llm.calls(), 2);
}

// ── Successful execute_resumable deletes the snapshot ─────────────────

#[tokio::test]
async fn checkpoint_successful_completion_deletes_snapshot() {
    use crate::checkpoint::{CheckpointStore, InMemoryCheckpointStore};
    use crate::executor::execute_resumable;
    use std::sync::Arc;

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("PASS\nbody1"),
        FakeOutcome::ok("PASS\nbody2"),
    ]);

    let comp = Composition::loop_until_builder(
        |_| {
            crate::composition::Composition::single_builder(
                "iter",
                Box::new(ScriptedAgent::new("iter")),
            )
            .build()
        },
        |_| false,
    )
    .max_iterations(2)
    .build();

    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let outcome = execute_resumable(comp, ctx, store.clone(), "task-success")
        .await
        .unwrap();
    assert_eq!(outcome.verdicts.len(), 2);
    assert!(
        store.load("task-success").await.unwrap().is_none(),
        "successful completion must delete the snapshot"
    );
}

// ── Restart storm trips the meltdown ledger ───────────────────────────

#[tokio::test]
async fn restart_storm_surfaces_meltdown_not_retry_exhausted() {
    // One agent retry-fails repeatedly under a shared ExecCtx (shared
    // meltdown ledger). Policy tightened to 4/60s so the storm trips fast.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("a"),
        FakeOutcome::retryable_err("b"),
        FakeOutcome::retryable_err("c"),
        FakeOutcome::retryable_err("d"),
        FakeOutcome::retryable_err("e"),
        FakeOutcome::retryable_err("f"),
        FakeOutcome::retryable_err("g"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let ledger = crate::meltdown::MeltdownLedger::new(crate::meltdown::MeltdownPolicy {
        max_restarts: 4,
        window: Duration::from_secs(60),
    });
    let ctx = ExecCtx::new(
        llm.clone(),
        budget,
        emitter,
        CancellationToken::new(),
        "session",
    )
    .with_meltdown(ledger);

    // max_attempts=10, but meltdown fires on the 4th retry decision and
    // breaks the loop first.
    let comp = Composition::single_builder("flapper", Box::new(ScriptedAgent::new("flapper")))
        .restart(RestartPolicy::on_failure(10).without_backoff())
        .build();

    let result = execute(comp, ctx).await;

    match result {
        Err(BorgError::Meltdown {
            agent_id, restarts, ..
        }) => {
            assert_eq!(agent_id, "flapper");
            assert_eq!(restarts, 4);
        }
        other => panic!("expected Meltdown, got {other:?}"),
    }
    assert_eq!(llm.calls(), 4);
}

// ── Circuit breaker fail-fasts after consecutive failures ─────────────

#[tokio::test]
async fn circuit_breaker_trips_after_threshold_and_fail_fasts_next_call() {
    // Threshold tightened to 3 (default 5) to exercise the open-state path.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("503 a"),
        FakeOutcome::retryable_err("503 b"),
        FakeOutcome::retryable_err("503 c"),
        // Not consumed — the open breaker fail-fasts before the LLM.
        FakeOutcome::ok("never reached"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let breaker =
        crate::circuit_breaker::CircuitBreaker::new(crate::circuit_breaker::BreakerPolicy {
            failure_threshold: 3,
            open_duration: Duration::from_secs(60),
        });
    let ctx = ExecCtx::new(
        llm.clone(),
        budget,
        emitter,
        CancellationToken::new(),
        "test",
    )
    .with_breaker(breaker.clone())
    .with_meltdown(crate::meltdown::MeltdownLedger::disabled());

    // 3 attempts, all fail. Breaker trips Open on the 3rd failure;
    // surfaces RetryExhausted from the composition wrapper.
    let comp = Composition::single_builder(
        "flaky",
        Box::new(crate::testing::ScriptedAgent::new("flaky")),
    )
    .restart(RestartPolicy::on_failure(3).without_backoff())
    .build();
    let result = execute(comp, ctx.clone()).await;
    assert!(matches!(result, Err(BorgError::RetryExhausted { .. })));
    assert_eq!(llm.calls(), 3, "first composition should consume 3 calls");
    assert!(
        breaker.is_open(crate::agent::Provider::OpenRouter, "flaky"),
        "breaker should be Open after the threshold trip"
    );

    // Breaker is Open: the next call fail-fasts before the LLM and
    // surfaces ProviderUnavailable.
    let comp2 = Composition::single_builder(
        "flaky",
        Box::new(crate::testing::ScriptedAgent::new("flaky")),
    )
    .build();
    let result2 = execute(comp2, ctx).await;
    match result2 {
        Err(BorgError::ProviderUnavailable {
            provider, agent_id, ..
        }) => {
            assert_eq!(provider, "openrouter");
            assert_eq!(agent_id, "flaky");
        }
        other => panic!("expected ProviderUnavailable, got {other:?}"),
    }
    assert_eq!(
        llm.calls(),
        3,
        "open breaker must not call the LLM (still 3 from first comp)"
    );
}

// ── Windowed breaker: one window per provider across agents ───────────

#[tokio::test]
async fn windowed_breaker_counts_every_agent_on_a_provider_and_fail_fasts_the_next() {
    use crate::circuit_breaker::{CircuitBreaker, WindowBreaker, WindowPolicy};
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("503 a"),
        FakeOutcome::retryable_err("503 b"),
        FakeOutcome::ok("never reached"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    // Shared with a caller outside the executor (a main loop, say).
    let window: WindowBreaker<crate::agent::Provider> = WindowBreaker::new(WindowPolicy::new(
        3,
        Duration::from_secs(300),
        Duration::from_secs(60),
    ));
    window.record_failure(&crate::agent::Provider::OpenRouter);
    let ctx = ExecCtx::new(
        llm.clone(),
        budget,
        emitter,
        CancellationToken::new(),
        "test",
    )
    .with_breaker(CircuitBreaker::windowed(window.clone()))
    .with_meltdown(crate::meltdown::MeltdownLedger::disabled());

    let comp = Composition::single_builder(
        "flaky",
        Box::new(crate::testing::ScriptedAgent::new("flaky")),
    )
    .restart(RestartPolicy::on_failure(2).without_backoff())
    .build();
    let result = execute(comp, ctx.clone()).await;
    assert!(result.is_err());
    assert_eq!(llm.calls(), 2);
    assert!(
        !window.allow(&crate::agent::Provider::OpenRouter),
        "1 outside + 2 in the executor = the threshold"
    );

    // A DIFFERENT agent on the same provider is fail-fasted too.
    let comp2 = Composition::single_builder(
        "other",
        Box::new(crate::testing::ScriptedAgent::new("other")),
    )
    .build();
    match execute(comp2, ctx).await {
        Err(BorgError::ProviderUnavailable {
            provider,
            agent_id,
            reopens_in,
        }) => {
            assert_eq!(provider, "openrouter");
            assert_eq!(agent_id, "other");
            assert!(reopens_in <= Duration::from_secs(60));
        }
        other => panic!("expected ProviderUnavailable, got {other:?}"),
    }
    assert_eq!(
        llm.calls(),
        2,
        "the open window stops the call before the LLM"
    );
}

// ── Disabled meltdown ledger never trips ──────────────────────────────

#[tokio::test]
async fn disabled_meltdown_lets_retries_run_to_max_attempts() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("a"),
        FakeOutcome::retryable_err("b"),
        FakeOutcome::retryable_err("c"),
        FakeOutcome::retryable_err("d"),
        FakeOutcome::retryable_err("e"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let ctx = ExecCtx::new(
        llm.clone(),
        budget,
        emitter,
        CancellationToken::new(),
        "test",
    )
    .with_meltdown(crate::meltdown::MeltdownLedger::disabled());

    let comp = Composition::single_builder("storm", Box::new(ScriptedAgent::new("storm")))
        .restart(RestartPolicy::on_failure(5).without_backoff())
        .build();

    let result = execute(comp, ctx).await;
    assert!(matches!(result, Err(BorgError::RetryExhausted { .. })));
    assert_eq!(llm.calls(), 5);
}

// ── Parallel admission denied surfaces BudgetExhausted, not Ok([]) ────

#[tokio::test]
async fn parallel_admission_denied_surfaces_budget_exhausted() {
    // Regression: a fan-out that can't admit any branch must surface
    // BudgetExhausted rather than silently returning Ok(Vec::new()).
    let llm = FakeLlmClient::always_ok("never");
    // 100 tokens total; 3 branches × 1000-token min estimate → denial.
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::parallel_builder()
        .branch(
            "a",
            Composition::single_builder("a", Box::new(ScriptedAgent::new("a"))).build(),
        )
        .branch(
            "b",
            Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build(),
        )
        .branch(
            "c",
            Composition::single_builder("c", Box::new(ScriptedAgent::new("c"))).build(),
        )
        .on_error(ErrorPolicy::CollectAll)
        .build();

    let result = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await;

    match result {
        Err(BorgError::BudgetExhausted { .. }) => { /* expected */ }
        Ok(outcome) if outcome.verdicts.is_empty() => {
            panic!("Denied admission must surface BudgetExhausted, not Ok([])");
        }
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    // No LLM calls — denial happens pre-flight.
    assert_eq!(llm.calls(), 0);
}

// ── Cancellation under a restart policy is NOT retry-exhausted ────────

#[tokio::test]
async fn restart_policy_with_cancel_surfaces_cancelled_not_retry_exhausted() {
    // Regression: a Cancelled inner error under a restart policy must not
    // be wrapped as RetryExhausted. The wrap only fires for retryable
    // inner errors.
    let llm = FakeLlmClient::new(vec![FakeOutcome::ok("never")]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let comp = Composition::single_builder("buildfix", Box::new(ScriptedAgent::new("buildfix")))
        .restart(RestartPolicy::on_failure(2).without_backoff())
        .build();

    let result = execute(comp, ctx_for(llm.clone(), budget, emitter, cancel)).await;

    match result {
        Err(BorgError::Cancelled) => { /* expected */ }
        Err(BorgError::RetryExhausted { .. }) => {
            panic!("Cancelled should NOT be wrapped as RetryExhausted");
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }
    // Cancel fired before the first attempt.
    assert_eq!(llm.calls(), 0);
}

// ── Partial budget admission downgrades a parallel run ────────────────

#[tokio::test]
async fn budget_admission_partial_downgrades_parallel() {
    // Budget for 2 of 3 branches at 1k min each.
    let llm = FakeLlmClient::new(vec![FakeOutcome::ok("OK_A"), FakeOutcome::ok("OK_B")]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(2_500));
    let emitter = Arc::new(RecordingEmitter::new());

    let comp = Composition::parallel_builder()
        .branch(
            "a",
            Composition::single_builder("a", Box::new(ScriptedAgent::new("a"))).build(),
        )
        .branch(
            "b",
            Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build(),
        )
        .branch(
            "c",
            Composition::single_builder("c", Box::new(ScriptedAgent::new("c"))).build(),
        )
        .on_error(ErrorPolicy::CollectAll)
        .max_concurrency(3)
        .build();

    let outcome = execute(
        comp,
        ctx_for(
            llm.clone(),
            budget,
            emitter.clone() as Arc<dyn EventEmitter>,
            CancellationToken::new(),
        ),
    )
    .await
    .unwrap();

    // Only 2 of 3 branches invoked the LLM.
    assert_eq!(llm.calls(), 2);
    assert_eq!(outcome.verdicts.len(), 2);
    let kinds = emitter.kinds();
    assert!(
        kinds.contains(&"parallel_degraded"),
        "expected parallel_degraded event, got {kinds:?}"
    );
}

// ── Per-step events emit in order under Sequential ────────────────────

#[tokio::test]
async fn per_step_events_emit_in_correct_order_under_sequential() {
    let llm = FakeLlmClient::always_ok("OK\nbody");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter = Arc::new(RecordingEmitter::new());

    let comp = Composition::sequential_builder()
        .step(Composition::single_builder("a", Box::new(ScriptedAgent::new("a"))).build())
        .step(Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build())
        .build();

    let _ = execute(
        comp,
        ctx_for(
            llm,
            budget,
            emitter.clone() as Arc<dyn EventEmitter>,
            CancellationToken::new(),
        ),
    )
    .await
    .unwrap();

    let kinds = emitter.kinds();
    // Invariant: outer Sequential start → a start/complete → b
    // start/complete → outer complete. (Single steps also emit their own
    // composition_started/completed pair; we assert the outer ordering.)
    let seq_started = kinds
        .iter()
        .position(|k| *k == "composition_started")
        .unwrap();
    let seq_completed = kinds
        .iter()
        .rposition(|k| *k == "composition_completed")
        .unwrap();
    assert!(seq_started < seq_completed, "started before completed");

    let starts: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == "step_started")
        .map(|(i, _)| i)
        .collect();
    let completes: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == "step_completed")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(completes.len(), 2);
    assert!(starts[0] < completes[0]);
    assert!(
        completes[0] < starts[1],
        "step a completes before step b starts"
    );
    assert!(starts[1] < completes[1]);
}

// ── Branch routes to the selected arm ─────────────────────────────────

#[tokio::test]
async fn branch_selector_routes_to_named_arm() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("FAIL\nfirst step"),
        FakeOutcome::ok("FIXED\nfix branch"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let critic = Composition::single_builder(
        "critic",
        Box::new(ScriptedAgent::new("critic").with_verdicts(&["PASS", "FAIL"])),
    )
    .output_key("critic_out")
    .build();

    let fixer = Composition::single_builder("fixer", Box::new(ScriptedAgent::new("fixer"))).build();
    let skip = Composition::single_builder("skip", Box::new(ScriptedAgent::new("skip"))).build();

    fn select_branch(h: &crate::handoff::HandoffState) -> BranchKey {
        h.find("critic")
            .map(|nv| BranchKey::from(nv.verdict.token.as_str()))
            .unwrap_or_else(|| BranchKey::from("UNKNOWN"))
    }

    let comp = Composition::sequential_builder()
        .step(critic)
        .step(
            Composition::branch_builder(select_branch)
                .arm("FAIL", fixer)
                .arm("PASS", skip)
                .build(),
        )
        .build();

    let outcome = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await
    .unwrap();

    // critic + fixer ran; skip did not.
    assert_eq!(outcome.verdicts.len(), 2);
    assert_eq!(outcome.verdicts[0].step_name, "critic");
    assert_eq!(outcome.verdicts[1].step_name, "fixer");
    assert_eq!(llm.calls(), 2);
}

// ── Fallback: primary on success, backup on retryable error ───────────

#[tokio::test]
async fn fallback_runs_primary_on_success() {
    let llm = FakeLlmClient::always_ok("PRIMARY OK");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let primary = Composition::single_builder("p", Box::new(ScriptedAgent::new("p"))).build();
    let backup = Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build();

    let comp = Composition::fallback_builder(primary)
        .or_else(backup)
        .build();
    let outcome = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdicts.len(), 1);
    assert_eq!(outcome.verdicts[0].step_name, "p");
    assert_eq!(llm.calls(), 1, "fallback should not invoke");
}

#[tokio::test]
async fn fallback_uses_backup_when_primary_retryable_fails() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::retryable_err("primary 503"),
        FakeOutcome::ok("BACKUP OK"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let primary = Composition::single_builder("p", Box::new(ScriptedAgent::new("p"))).build();
    let backup = Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build();

    let comp = Composition::fallback_builder(primary)
        .or_else(backup)
        .build();
    let outcome = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await
    .unwrap();

    assert_eq!(outcome.verdicts.len(), 1);
    assert_eq!(outcome.verdicts[0].step_name, "b");
    assert_eq!(llm.calls(), 2);
}

// ── Budget exhausted mid-pipeline surfaces BudgetExhausted ────────────

#[tokio::test]
async fn budget_exhausted_mid_sequential_surfaces_error() {
    let llm = FakeLlmClient::always_ok("OK\nbody");
    // Just enough for one call (150 tokens), nothing for the second.
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(150));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::sequential_builder()
        .step(Composition::single_builder("a", Box::new(ScriptedAgent::new("a"))).build())
        .step(Composition::single_builder("b", Box::new(ScriptedAgent::new("b"))).build())
        .build();

    let result = execute(
        comp,
        ctx_for(llm.clone(), budget, emitter, CancellationToken::new()),
    )
    .await;

    match result {
        Err(BorgError::BudgetExhausted { remaining: 0, .. }) => {}
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    assert_eq!(llm.calls(), 1, "second call must not have happened");
}

// ── DevEx helpers ─────────────────────────────────────────────────────

/// `deterministic_llm_client()` errors loudly if a non-LLM agent ever
/// reaches `ctx.llm`.
#[tokio::test]
async fn deterministic_llm_client_errors_when_reached() {
    use crate::llm::{deterministic_llm_client, ChatRequest};
    let llm = deterministic_llm_client();
    let result = llm
        .chat_completion(ChatRequest::new("x", "x", "x", 10))
        .await;
    match result {
        Err(BorgError::Internal { message }) => {
            assert!(
                message.contains("deterministic agent"),
                "error must explain misuse: {message}"
            );
        }
        other => panic!("expected Internal, got {other:?}"),
    }
}

/// `ExecCtx::deterministic()` wires the stub LLM and preserves the
/// passed-in budget/emitter/cancel/composition-id.
#[tokio::test]
async fn exec_ctx_deterministic_wires_stub_llm() {
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(1000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let cancel = CancellationToken::new();
    let ctx = ExecCtx::deterministic(budget.clone(), emitter, cancel.clone(), "test::det");
    assert_eq!(&*ctx.parent_composition_id, "test::det");
    let result = ctx
        .llm
        .chat_completion(crate::llm::ChatRequest::new("x", "", "x", 10))
        .await;
    assert!(
        matches!(result, Err(BorgError::Internal { .. })),
        "deterministic LLM stub must error if reached"
    );
}

/// `run_one()` is the single-agent dispatch shortcut over
/// `Composition::single → execute_with_handoff → first verdict`.
#[tokio::test]
async fn run_one_returns_single_agent_verdict() {
    use crate::executor::run_one;
    let llm = FakeLlmClient::always_ok("PASS\nbody text");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let agent = ScriptedAgent::new("solo").with_verdicts(&["PASS", "FAIL"]);
    let verdict = run_one("solo", agent, ctx, crate::handoff::HandoffState::new())
        .await
        .expect("run_one should succeed");
    assert_eq!(verdict.token, "PASS");
    assert!(verdict.body.contains("body text"));
}

/// `run_one` surfaces errors verbatim — no log-and-drop fallback.
#[tokio::test]
async fn run_one_propagates_agent_errors() {
    use crate::executor::run_one;
    // An erroring FakeLlmClient is a more reliable failure mode here than
    // a panic: single-step compositions run inline (no JoinSet), so a
    // panic would propagate through the future rather than surface as
    // ChildPanicked.
    let llm = FakeLlmClient::new(vec![FakeOutcome::terminal_err("simulated")]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let agent = ScriptedAgent::new("doomed");
    let result = run_one("doomed", agent, ctx, crate::handoff::HandoffState::new()).await;
    assert!(
        matches!(result, Err(BorgError::Llm { .. })),
        "errors must propagate, got {result:?}"
    );
    let _ = std::marker::PhantomData::<PanickingAgent>; // keep import alive
}

// ── run_one_typed tests ────────────────────────────────────────────────
//
// `run_one_typed` takes an agent that produces JSON in its verdict body
// and returns the parsed struct directly.

#[derive(serde::Deserialize, PartialEq, Debug, Default)]
struct Suggestion {
    file: String,
    fix: String,
}

#[tokio::test]
async fn run_one_typed_parses_verdict_body_into_struct() {
    use crate::executor::run_one_typed;
    // Line 1 (PASS) is the verdict token; the parser sees only the body
    // (the JSON after the newline).
    let body = r#"[{"file":"a.ts","fix":"rename foo to bar"},{"file":"b.ts","fix":"remove unused import"}]"#;
    let llm = FakeLlmClient::always_ok(format!("PASS\n{body}"));
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let agent = ScriptedAgent::new("typed").with_verdicts(&["PASS"]);

    let result: Vec<Suggestion> =
        run_one_typed("typed", agent, ctx, crate::handoff::HandoffState::new())
            .await
            .expect("typed parse should succeed");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].file, "a.ts");
    assert_eq!(result[1].fix, "remove unused import");
}

#[tokio::test]
async fn run_one_typed_returns_internal_error_on_parse_failure() {
    use crate::executor::run_one_typed;
    // Invalid JSON: caller gets a structured error with a body preview,
    // not a silent `unwrap_or_default()` empty Vec.
    let llm = FakeLlmClient::always_ok("PASS\nthis is definitely not json {{{");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let agent = ScriptedAgent::new("typed_fail").with_verdicts(&["PASS"]);

    let result: Result<Vec<Suggestion>, _> = run_one_typed(
        "typed_fail",
        agent,
        ctx,
        crate::handoff::HandoffState::new(),
    )
    .await;
    match result {
        Err(BorgError::Internal { message }) => {
            assert!(
                message.contains("failed to parse"),
                "expected parse-failure message, got: {message}"
            );
            assert!(
                message.contains("Suggestion") || message.contains("Vec"),
                "expected type name in error, got: {message}"
            );
            assert!(
                message.contains("excerpt"),
                "expected body excerpt in error, got: {message}"
            );
        }
        other => panic!("expected BorgError::Internal with parse message, got {other:?}"),
    }
}

#[tokio::test]
async fn run_one_typed_strips_fenced_code_block_on_fallback() {
    use crate::executor::run_one_typed;
    // Models sometimes wrap JSON in ```json fences despite a raw-JSON
    // prompt. `run_one_typed` strips the fence after the direct parse fails.
    let body = "```json\n[{\"file\":\"a.ts\",\"fix\":\"do the thing\"}]\n```";
    let llm = FakeLlmClient::always_ok(format!("PASS\n{body}"));
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let agent = ScriptedAgent::new("fenced").with_verdicts(&["PASS"]);
    let result: Vec<Suggestion> =
        run_one_typed("fenced", agent, ctx, crate::handoff::HandoffState::new())
            .await
            .expect("fenced JSON should parse via fallback");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].file, "a.ts");
}

#[tokio::test]
async fn run_one_typed_propagates_underlying_llm_errors() {
    use crate::executor::run_one_typed;
    // The type-parse wrapper must not swallow non-parse error variants.
    let llm = FakeLlmClient::new(vec![FakeOutcome::terminal_err(
        "simulated upstream failure",
    )]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let agent = ScriptedAgent::new("doomed_typed");

    let result: Result<Vec<Suggestion>, _> = run_one_typed(
        "doomed_typed",
        agent,
        ctx,
        crate::handoff::HandoffState::new(),
    )
    .await;
    assert!(
        matches!(result, Err(BorgError::Llm { .. })),
        "Llm errors must propagate past the type-parse wrapper, got {result:?}"
    );
}

// ── LoopUntil tests ────────────────────────────────────────────────────
//
// LoopUntil takes a factory closure producing a fresh body composition
// per iteration. Contract: stops on predicate, stops at max_iterations,
// factory sees iteration index and prior verdict, cancel breaks cleanly.

#[tokio::test]
async fn loop_until_stops_when_predicate_returns_true() {
    // Predicate fires on iteration 0's "PASS" verdict, so the loop stops
    // after one iteration despite more being available.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("PASS\nfirst iteration body"),
        FakeOutcome::ok("PASS\nsecond — should never run"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = crate::composition::Composition::loop_until(
        |_loop_ctx| {
            crate::composition::Composition::single(
                "step",
                ScriptedAgent::new("step").with_verdicts(&["PASS", "FAIL"]),
            )
        },
        |v| v.token == "PASS",
        5,
    );

    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let outcome = execute(comp, ctx).await.expect("loop should complete");

    assert_eq!(
        outcome.verdicts.len(),
        1,
        "predicate firing on iteration 0 must stop the loop"
    );
    assert_eq!(outcome.verdicts[0].verdict.token, "PASS");
    // Second slot must not be burned.
    assert_eq!(llm.calls(), 1);
}

#[tokio::test]
async fn loop_until_runs_up_to_max_iterations_when_predicate_never_fires() {
    // Predicate always false → loop runs exactly max_iterations times.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("KEEP\niter 0"),
        FakeOutcome::ok("KEEP\niter 1"),
        FakeOutcome::ok("KEEP\niter 2"),
        FakeOutcome::ok("EXTRA\nshould not run"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = crate::composition::Composition::loop_until(
        |_| {
            crate::composition::Composition::single(
                "step",
                ScriptedAgent::new("step").with_verdicts(&["KEEP"]),
            )
        },
        |_| false, // never stop
        3,
    );

    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let outcome = execute(comp, ctx).await.expect("loop should complete");

    assert_eq!(
        outcome.verdicts.len(),
        3,
        "predicate never firing must hit max_iterations exactly"
    );
    assert_eq!(llm.calls(), 3);
}

#[tokio::test]
async fn loop_until_factory_observes_iteration_index() {
    // Factory writes `loop_ctx.iteration` to a shared counter.
    use std::sync::atomic::{AtomicU32, Ordering};

    let observed = Arc::new(AtomicU32::new(0));
    let observed_max = Arc::new(AtomicU32::new(0));
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("GO\n"),
        FakeOutcome::ok("GO\n"),
        FakeOutcome::ok("GO\n"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let observed_for_factory = observed.clone();
    let observed_max_for_factory = observed_max.clone();
    let comp = crate::composition::Composition::loop_until(
        move |loop_ctx| {
            observed_for_factory.store(loop_ctx.iteration, Ordering::SeqCst);
            // Track the highest iteration we ever saw so the assertion
            // doesn't race with the final iteration's overwrite.
            let cur_max = observed_max_for_factory.load(Ordering::SeqCst);
            if loop_ctx.iteration > cur_max {
                observed_max_for_factory.store(loop_ctx.iteration, Ordering::SeqCst);
            }
            crate::composition::Composition::single(
                "step",
                ScriptedAgent::new("step").with_verdicts(&["GO"]),
            )
        },
        |_| false,
        3,
    );

    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let outcome = execute(comp, ctx).await.expect("loop completes");
    assert_eq!(outcome.verdicts.len(), 3);
    // Highest index seen is `max_iterations - 1 = 2`.
    assert_eq!(observed_max.load(Ordering::SeqCst), 2);
}

/// Step limits are `u32`: a loop past 255 iterations runs them all
/// instead of stopping (or failing to compile) at a `u8` ceiling.
#[tokio::test]
async fn loop_until_runs_past_a_u8_iteration_count() {
    let n: u32 = 300;
    let llm = FakeLlmClient::new((0..n).map(|_| FakeOutcome::ok("GO\n")).collect());
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(u64::MAX));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let comp = crate::composition::Composition::loop_until(
        |_| {
            crate::composition::Composition::single(
                "step",
                ScriptedAgent::new("step").with_verdicts(&["GO"]),
            )
        },
        |_| false,
        n,
    );
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let outcome = execute(comp, ctx).await.expect("loop completes");
    assert_eq!(outcome.verdicts.len(), n as usize);
}

#[tokio::test]
async fn loop_until_factory_receives_last_verdict_from_prior_iteration() {
    // Iteration 0 sees `last_verdict = None`; iteration 1 sees iteration
    // 0's verdict token.
    use std::sync::Mutex;

    let captured: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let llm = FakeLlmClient::new(vec![FakeOutcome::ok("ITER0\n"), FakeOutcome::ok("ITER1\n")]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let cap = captured.clone();
    let comp = crate::composition::Composition::loop_until(
        move |loop_ctx| {
            cap.lock()
                .unwrap()
                .push(loop_ctx.last_verdict.map(|v| v.token.clone()));
            crate::composition::Composition::single(
                "step",
                ScriptedAgent::new("step").with_verdicts(&["ITER0", "ITER1"]),
            )
        },
        |v| v.token == "ITER1",
        5,
    );

    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());
    let _ = execute(comp, ctx).await.expect("loop completes");

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "factory called twice (iter 0 + iter 1)");
    assert!(seen[0].is_none(), "iter 0 has no prior verdict");
    assert_eq!(
        seen[1].as_deref(),
        Some("ITER0"),
        "iter 1 sees iter 0's token as last_verdict"
    );
}

#[tokio::test]
async fn loop_until_cancellation_breaks_loop_cleanly() {
    // Pre-cancel → Cancelled without ever invoking the body factory.
    let llm = FakeLlmClient::new(vec![FakeOutcome::ok("X\n")]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let cancel = CancellationToken::new();
    cancel.cancel();

    let factory_invoked = Arc::new(AtomicBool::new(false));
    let f = factory_invoked.clone();
    let comp = crate::composition::Composition::loop_until(
        move |_| {
            f.store(true, Ordering::SeqCst);
            crate::composition::Composition::single("never", ScriptedAgent::new("never"))
        },
        |_| false,
        3,
    );

    let ctx = ctx_for(llm, budget, emitter, cancel);
    let result = execute(comp, ctx).await;
    assert!(matches!(result, Err(BorgError::Cancelled)));
    assert!(
        !factory_invoked.load(Ordering::SeqCst),
        "pre-cancelled loop must not invoke factory"
    );
}

// ── Composition::Agent + tool-loop tests ───────────────────────────────
//
// The agent emits tool calls, borg dispatches them, and results thread
// back to the next turn. Contract: stops when the agent is done, stops at
// max_steps, threads results, propagates LLM errors, honors cancel.

use crate::tools::{
    DispatchedCall, MutexDispatcher, Registry, Tool, ToolCallRequest, ToolCallResult, ToolSchema,
};

/// Test tool that always succeeds and echoes its input.
struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    type Context = i32;

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "echo".into(),
            description: "Echoes input back.".into(),
            input_schema: serde_json::json!({ "type": "object" }),
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &mut Self::Context,
    ) -> Result<ToolCallResult, BorgError> {
        *ctx += 1;
        ToolCallResult::ok(input)
    }
}

/// Test agent that emits a scripted number of tool calls per step, then
/// stops (empty `extract_tool_calls`) once the plan is exhausted.
struct CountingAgent {
    /// Tool calls to emit on each step, consumed front-to-back.
    calls_per_step: std::sync::Mutex<std::collections::VecDeque<usize>>,
    /// Current step, for prompt sequencing.
    step: std::sync::atomic::AtomicUsize,
}

impl CountingAgent {
    fn new(plan: &[usize]) -> Self {
        Self {
            calls_per_step: std::sync::Mutex::new(plan.iter().copied().collect()),
            step: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl crate::agent::BorgAgent for CountingAgent {
    fn agent_id(&self) -> &'static str {
        "counting"
    }
    fn model(&self) -> &str {
        "fake/counting"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn system_prompt(&self) -> &'static str {
        ""
    }
    fn user_prompt(&self, _ctx: &crate::AgentContext) -> Option<String> {
        Some(format!(
            "step {}",
            self.step.load(std::sync::atomic::Ordering::SeqCst)
        ))
    }

    fn extract_tool_calls(&self, _verdict: &Verdict) -> Vec<ToolCallRequest> {
        let n = {
            let mut plan = self.calls_per_step.lock().unwrap();
            plan.pop_front().unwrap_or(0)
        };
        self.step.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (0..n)
            .map(|i| ToolCallRequest {
                name: "echo".into(),
                args: serde_json::json!({ "step": self.step.load(std::sync::atomic::Ordering::SeqCst), "idx": i }),
                call_id: format!("call-{i}"),
            })
            .collect()
    }
}

fn tool_loop_ctx(llm: Arc<dyn AgentLlmClient>, cancel: CancellationToken) -> ExecCtx {
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    ctx_for(llm, budget, emitter, cancel)
}

#[tokio::test]
async fn tool_loop_emits_agent_loop_shape() {
    // Regression: Composition::Agent now advertises its own AgentLoop
    // shape, not LoopUntil, so consumers can render it distinctly.
    let llm = FakeLlmClient::always_ok("done");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let recorder = Arc::new(RecordingEmitter::new());
    let emitter: Arc<dyn EventEmitter> = recorder.clone();
    let ctx = ctx_for(llm, budget, emitter, CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp = crate::composition::Composition::agent("agent", CountingAgent::new(&[0]), tools, 5);
    let _ = execute(comp, ctx).await.expect("loop completes");

    let events = recorder.snapshot();
    let agent_loop_starts = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                crate::events::BorgEvent::CompositionStarted {
                    shape: crate::events::CompositionShape::AgentLoop,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        agent_loop_starts,
        1,
        "expected exactly one CompositionStarted with AgentLoop shape, saw events: {:?}",
        events.iter().map(|e| e.kind()).collect::<Vec<_>>()
    );
    // And no LoopUntil shape from the agent path.
    let loop_until_starts = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                crate::events::BorgEvent::CompositionStarted {
                    shape: crate::events::CompositionShape::LoopUntil,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        loop_until_starts, 0,
        "Composition::Agent must NOT advertise as LoopUntil anymore"
    );
}

#[tokio::test]
async fn tool_loop_stops_when_agent_emits_no_calls() {
    // Plan: step 0 emits 0 tool calls → loop stops immediately.
    let llm = FakeLlmClient::always_ok("done");
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp = crate::composition::Composition::agent("agent", CountingAgent::new(&[0]), tools, 5);
    let outcome = execute(comp, ctx).await.expect("loop completes");

    // Single invocation, no tool calls, one verdict.
    assert_eq!(outcome.verdicts.len(), 1);
    assert_eq!(llm.calls(), 1);
}

#[tokio::test]
async fn tool_loop_dispatches_tools_and_continues() {
    // step 0 emits 2 calls, step 1 emits 0 → 2 steps, 2 tool dispatches.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("step0 verdict"),
        FakeOutcome::ok("step1 verdict"),
    ]);
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tool_ctx_initial: i32 = 0;
    let dispatcher = Arc::new(MutexDispatcher::new(registry, tool_ctx_initial));
    let tools: Arc<dyn crate::tools::ToolDispatch> = dispatcher.clone();

    let comp =
        crate::composition::Composition::agent("agent", CountingAgent::new(&[2, 0]), tools, 5);
    let outcome = execute(comp, ctx).await.expect("loop completes");

    assert_eq!(outcome.verdicts.len(), 2);
    assert_eq!(llm.calls(), 2);
}

#[tokio::test]
async fn tool_loop_caps_at_max_steps() {
    // Agent never stops emitting calls; max_steps=3 caps at 3 model calls.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("v1"),
        FakeOutcome::ok("v2"),
        FakeOutcome::ok("v3"),
        FakeOutcome::ok("would not run"),
    ]);
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp = crate::composition::Composition::agent(
        "agent",
        // Plan emits 1 call/step well past max_steps.
        CountingAgent::new(&[1, 1, 1, 1, 1, 1, 1, 1]),
        tools,
        3,
    );
    // The last allowed step still asked for tools: no final answer.
    match execute(comp, ctx).await {
        Err(BorgError::MaxStepsReached {
            agent_id,
            max_steps,
        }) => {
            assert_eq!(agent_id, "agent");
            assert_eq!(max_steps, 3);
        }
        other => panic!("expected MaxStepsReached, got {other:?}"),
    }
    assert_eq!(llm.calls(), 3, "max_steps cap honored");
}

#[tokio::test]
async fn tool_loop_final_answer_on_last_step_is_ok() {
    // Two tool turns, then a text-only answer on step 3 of 3: Ok.
    let llm = FakeLlmClient::always_ok("v");
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());
    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));
    let comp =
        crate::composition::Composition::agent("agent", CountingAgent::new(&[1, 1, 0]), tools, 3);
    let outcome = execute(comp, ctx)
        .await
        .expect("an answer on the last step is a normal completion");
    assert_eq!(outcome.verdicts.len(), 3);
    assert_eq!(llm.calls(), 3);
}

#[tokio::test]
async fn single_panicking_agent_is_child_panicked_not_an_unwind() {
    // Regression: a one-shot `Single` ran `invoke` inline with no panic
    // boundary, so the panic unwound into the caller's task.
    let llm = FakeLlmClient::always_ok("x");
    let recorder = Arc::new(RecordingEmitter::new());
    let emitter: Arc<dyn EventEmitter> = recorder.clone();
    let ctx = ctx_for(
        llm,
        Arc::new(InMemoryBudget::new(1_000)),
        emitter,
        CancellationToken::new(),
    );
    let comp = Composition::single("p", PanickingAgent { id: "panicker" });
    let result = tokio::spawn(execute(comp, ctx))
        .await
        .expect("the panic must not escape execute()");
    match result {
        Err(BorgError::ChildPanicked { agent_id, message }) => {
            assert_eq!(agent_id, "panicker");
            assert!(!message.is_empty());
        }
        other => panic!("expected ChildPanicked, got {other:?}"),
    }
    // The usual terminal events still close the step.
    let kinds: Vec<&str> = recorder.snapshot().iter().map(|e| e.kind()).collect();
    assert!(kinds.contains(&"step_completed"), "{kinds:?}");
    assert!(kinds.contains(&"composition_completed"), "{kinds:?}");
}

#[tokio::test]
async fn agent_loop_panicking_agent_is_child_panicked() {
    let llm = FakeLlmClient::always_ok("x");
    let ctx = tool_loop_ctx(llm, CancellationToken::new());
    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));
    let comp = crate::composition::Composition::agent("loop", PanickingAgent { id: "p" }, tools, 3);
    let result = tokio::spawn(execute(comp, ctx))
        .await
        .expect("the panic must not escape execute()");
    assert!(
        matches!(result, Err(BorgError::ChildPanicked { .. })),
        "got {result:?}"
    );
}

/// A tool whose `execute` panics.
struct PanicTool;

#[async_trait::async_trait]
impl Tool for PanicTool {
    type Context = i32;

    fn schema(&self) -> ToolSchema {
        ToolSchema::new("echo", "Panics.", serde_json::json!({ "type": "object" }))
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &mut Self::Context,
    ) -> Result<ToolCallResult, BorgError> {
        panic!("tool blew up");
    }
}

#[tokio::test]
async fn serial_panicking_tool_is_a_structural_error_not_an_unwind() {
    // Regression: the serial path awaited `dispatch` with no panic
    // boundary. It now fails like a structural tool error, the same
    // error the parallel path reports for a call dispatched on its own.
    let llm = FakeLlmClient::always_ok("x");
    let ctx = tool_loop_ctx(llm, CancellationToken::new());
    let registry = Registry::<i32>::new().with(PanicTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));
    let comp =
        crate::composition::Composition::agent("loop", CountingAgent::new(&[1, 0]), tools, 3);
    let result = tokio::spawn(execute(comp, ctx))
        .await
        .expect("the panic must not escape execute()");
    match result {
        Err(BorgError::Internal { message }) => {
            assert!(message.contains("tool call `echo` panicked"), "{message}");
            assert!(message.contains("tool blew up"), "{message}");
        }
        other => panic!("expected Internal(tool panicked), got {other:?}"),
    }
}

#[tokio::test]
async fn tool_calls_past_the_per_turn_cap_fail_without_running() {
    // One turn asks for 5 calls with a cap of 2: two run, three come
    // back failed so the model can retry with fewer.
    let llm = FakeLlmClient::always_ok("v");
    let ctx = tool_loop_ctx(llm, CancellationToken::new()).with_max_tool_calls_per_turn(2);
    assert_eq!(ctx.max_tool_calls_per_turn, 2);
    let registry = Registry::<i32>::new().with(EchoTool);
    let dispatcher = Arc::new(MutexDispatcher::new(registry, 0));
    let tools: Arc<dyn crate::tools::ToolDispatch> = dispatcher.clone();
    let comp =
        crate::composition::Composition::agent("agent", CountingAgent::new(&[5, 0]), tools, 3);
    let outcome = execute(comp, ctx).await.expect("loop completes");
    let results = outcome
        .handoff
        .get("tool_results")
        .and_then(|v| v.as_array())
        .expect("tool_results")
        .clone();
    assert_eq!(results.len(), 5, "every requested call gets a result");
    let ran = results
        .iter()
        .filter(|r| r["result"]["error"].is_null())
        .count();
    assert_eq!(ran, 2);
    for rejected in &results[2..] {
        let err = rejected["result"]["error"].as_str().unwrap_or_default();
        assert_eq!(err, "too many tool calls in one turn; limit 2");
    }
}

#[test]
fn tool_call_caps_default_and_clamp() {
    let ctx = tool_loop_ctx(FakeLlmClient::always_ok("v"), CancellationToken::new());
    assert_eq!(
        ctx.max_tool_calls_per_turn,
        crate::executor::DEFAULT_MAX_TOOL_CALLS_PER_TURN
    );
    assert_eq!(crate::executor::DEFAULT_MAX_TOOL_CALLS_PER_TURN, 64);
    assert_eq!(
        ctx.max_parallel_tool_calls,
        crate::executor::DEFAULT_MAX_PARALLEL_TOOL_CALLS
    );
    assert_eq!(crate::executor::DEFAULT_MAX_PARALLEL_TOOL_CALLS, 16);
    let ctx = ctx
        .with_max_tool_calls_per_turn(0)
        .with_max_parallel_tool_calls(0);
    assert_eq!(ctx.max_tool_calls_per_turn, 1);
    assert_eq!(ctx.max_parallel_tool_calls, 1);
}

#[tokio::test]
async fn tool_loop_cancellation_breaks_loop_cleanly() {
    let llm = FakeLlmClient::always_ok("x");
    let cancel = CancellationToken::new();
    cancel.cancel();
    let ctx = tool_loop_ctx(llm.clone(), cancel);

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp =
        crate::composition::Composition::agent("agent", CountingAgent::new(&[1, 1]), tools, 5);
    let result = execute(comp, ctx).await;
    assert!(matches!(result, Err(BorgError::Cancelled)));
    // Pre-cancelled — model never called.
    assert_eq!(llm.calls(), 0);
}

#[tokio::test]
async fn tool_loop_unknown_tool_surfaces_as_logical_failure_continues_loop() {
    // An unknown tool yields `ToolCallResult::fail(...)`, not a BorgError,
    // so the loop continues and the agent sees the failure next turn.

    struct UnknownToolAgent;
    #[async_trait::async_trait]
    impl crate::agent::BorgAgent for UnknownToolAgent {
        fn agent_id(&self) -> &'static str {
            "unk"
        }
        fn model(&self) -> &str {
            "fake/unk"
        }
        fn max_tokens(&self) -> u32 {
            100
        }
        fn system_prompt(&self) -> &'static str {
            ""
        }
        fn user_prompt(&self, _ctx: &crate::AgentContext) -> Option<String> {
            Some("go".into())
        }
        fn extract_tool_calls(&self, verdict: &Verdict) -> Vec<ToolCallRequest> {
            // Turn 1 calls a nonexistent tool; later turns stop.
            if verdict.body.contains("turn1") {
                vec![ToolCallRequest {
                    name: "nonexistent".into(),
                    args: serde_json::json!({}),
                    call_id: "x".into(),
                }]
            } else {
                vec![]
            }
        }
    }

    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("turn1"),
        FakeOutcome::ok("turn2-done"),
    ]);
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());

    // Empty registry — every tool is unknown.
    let registry = Registry::<i32>::new();
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp = crate::composition::Composition::agent("agent", UnknownToolAgent, tools, 5);
    let outcome = execute(comp, ctx)
        .await
        .expect("loop completes despite unknown tool");
    assert_eq!(outcome.verdicts.len(), 2);
    assert_eq!(llm.calls(), 2);
}

#[tokio::test]
async fn tool_loop_propagates_underlying_llm_errors() {
    // First call errors terminally → loop aborts before the second.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::terminal_err("upstream"),
        FakeOutcome::ok("never"),
    ]);
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp = crate::composition::Composition::agent("agent", CountingAgent::new(&[1]), tools, 5);
    let result = execute(comp, ctx).await;
    assert!(matches!(result, Err(BorgError::Llm { .. })));
    assert_eq!(llm.calls(), 1);
    // Keep the DispatchedCall import alive.
    let _ = DispatchedCall {
        call_id: "ignored".into(),
        name: "x".into(),
        result: ToolCallResult::ok(()).unwrap(),
    };
}

#[tokio::test]
async fn tool_loop_default_extract_reads_verdict_tool_calls() {
    // An agent that doesn't override `extract_tool_calls` falls back to
    // the default, which reads `verdict.tool_calls`. The script emits a
    // tool call on turn 1 (loop continues) and none on turn 2 (loop exits).
    use crate::testing::ScriptedAgent;
    use crate::tools::ToolCallRequest;

    let llm = FakeLlmClient::new(vec![
        // Turn 1: intermediate text alongside a tool call.
        FakeOutcome::ok_with_tool_calls(
            "intermediate body — wants docs",
            vec![ToolCallRequest {
                name: "echo".into(),
                args: serde_json::json!({"k": "v"}),
                call_id: "tc-1".into(),
            }],
        ),
        // Turn 2: final answer, no further tool calls.
        FakeOutcome::ok("final-answer-json"),
    ]);
    let ctx = tool_loop_ctx(llm.clone(), CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    // ScriptedAgent uses the default extract_tool_calls.
    let comp = crate::composition::Composition::agent(
        "scripted_agent",
        ScriptedAgent::new("scripted_agent"),
        tools,
        5,
    );
    let outcome = execute(comp, ctx).await.expect("loop completes");

    assert_eq!(outcome.verdicts.len(), 2, "ran turn 1 + turn 2");
    assert_eq!(llm.calls(), 2);

    // Last verdict is the final answer; first carries the tool calls.
    let final_body = &outcome.verdicts.last().unwrap().verdict.body;
    assert_eq!(&**final_body, "final-answer-json");

    let first = &outcome.verdicts[0].verdict;
    assert_eq!(first.tool_calls.len(), 1);
    assert_eq!(first.tool_calls[0].name, "echo");
    assert_eq!(first.tool_calls[0].call_id, "tc-1");
}

// ── Verdict::handoff_to is mirrored into HandoffState ─────────────────

#[tokio::test]
async fn verdict_handoff_to_populates_pending_handoff() {
    // A verdict with handoff_to=Some("reviewer") becomes readable via
    // HandoffState::pending_handoff(), letting a downstream Branch route
    // on agent-declared handoff without out-of-band signaling.
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("unused");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::single_builder(
        "planner",
        Box::new(HandoffAgent {
            id: "planner",
            target: "reviewer",
        }),
    )
    .build();

    let outcome = execute(
        comp,
        ctx_for(llm, budget, emitter, CancellationToken::new()),
    )
    .await
    .expect("composition succeeds");

    assert_eq!(
        outcome.handoff.pending_handoff().as_deref(),
        Some("reviewer"),
        "executor must mirror Verdict.handoff_to into HandoffState::HANDOFF_KEY"
    );
}

#[tokio::test]
async fn verdict_without_handoff_to_leaves_pending_handoff_unset() {
    // Negative case: no handoff_to → the executor must not write the key.
    let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("just text");
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = Composition::single_builder("plain", Box::new(ScriptedAgent::new("plain"))).build();

    let outcome = execute(
        comp,
        ctx_for(llm, budget, emitter, CancellationToken::new()),
    )
    .await
    .expect("composition succeeds");

    assert_eq!(outcome.handoff.pending_handoff(), None);
}

#[tokio::test]
async fn loop_until_inner_error_propagates_and_aborts_loop() {
    // First iteration errors → loop returns it without running more.
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::terminal_err("simulated"),
        FakeOutcome::ok("WOULD_RUN_BUT_WONT\n"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);

    let comp = crate::composition::Composition::loop_until(
        |_| crate::composition::Composition::single("step", ScriptedAgent::new("step")),
        |_| false,
        3,
    );

    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());
    let result = execute(comp, ctx).await;
    assert!(matches!(result, Err(BorgError::Llm { .. })));
    // Only the first iteration ran.
    assert_eq!(llm.calls(), 1);
}

// ── Agent loop emits Turn/Tool events at the right call sites ─────────
//
// Two-step loop: step 0 emits one tool call (iterates), step 1 emits none
// (terminates). All four event kinds must appear, in pair order.

#[tokio::test]
async fn agent_loop_emits_turn_and_tool_events() {
    let llm = FakeLlmClient::new(vec![
        FakeOutcome::ok("step0 verdict"),
        FakeOutcome::ok("step1 verdict"),
    ]);
    let budget: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(100_000));
    let recorder = Arc::new(RecordingEmitter::new());
    let emitter: Arc<dyn EventEmitter> = recorder.clone();
    let ctx = ctx_for(llm.clone(), budget, emitter, CancellationToken::new());

    let registry = Registry::<i32>::new().with(EchoTool);
    let tools: Arc<dyn crate::tools::ToolDispatch> = Arc::new(MutexDispatcher::new(registry, 0));

    let comp =
        crate::composition::Composition::agent("agent", CountingAgent::new(&[1, 0]), tools, 5);
    let _ = execute(comp, ctx).await.expect("agent loop completes");

    let kinds = recorder.kinds();

    assert!(
        kinds.contains(&"turn_started"),
        "expected turn_started in events, got: {kinds:?}"
    );
    assert!(
        kinds.contains(&"turn_completed"),
        "expected turn_completed in events, got: {kinds:?}"
    );
    assert!(
        kinds.contains(&"tool_called"),
        "expected tool_called in events, got: {kinds:?}"
    );
    assert!(
        kinds.contains(&"tool_returned"),
        "expected tool_returned in events, got: {kinds:?}"
    );

    let turn_started_count = kinds.iter().filter(|&&k| k == "turn_started").count();
    let turn_completed_count = kinds.iter().filter(|&&k| k == "turn_completed").count();
    assert_eq!(
        turn_started_count, 2,
        "expected 2 turn_started events, got {turn_started_count}"
    );
    assert_eq!(
        turn_completed_count, 2,
        "expected 2 turn_completed events, got {turn_completed_count}"
    );

    let tool_called_count = kinds.iter().filter(|&&k| k == "tool_called").count();
    let tool_returned_count = kinds.iter().filter(|&&k| k == "tool_returned").count();
    assert_eq!(
        tool_called_count, 1,
        "expected 1 tool_called event, got {tool_called_count}"
    );
    assert_eq!(
        tool_returned_count, 1,
        "expected 1 tool_returned event, got {tool_returned_count}"
    );

    let first_turn_started = kinds.iter().position(|&k| k == "turn_started").unwrap();
    let first_turn_completed = kinds.iter().position(|&k| k == "turn_completed").unwrap();
    assert!(
        first_turn_started < first_turn_completed,
        "turn_started must precede turn_completed"
    );

    let tool_called_pos = kinds.iter().position(|&k| k == "tool_called").unwrap();
    let tool_returned_pos = kinds.iter().position(|&k| k == "tool_returned").unwrap();
    assert!(
        tool_called_pos < tool_returned_pos,
        "tool_called must precede tool_returned"
    );
}

// ── executor regressions ──────────────────────────────────────────────

mod v3_regressions {
    use super::*;
    use crate::checkpoint::{CheckpointPolicy, CheckpointStore, InMemoryCheckpointStore};
    use crate::events::BorgEvent;
    use crate::executor::{execute_resumable, execute_resumable_with, BatchFailureMode};
    use crate::tools::{DispatchedCall, ToolCallRequest, ToolCallResult, ToolDispatch, ToolSchema};
    use async_trait::async_trait;

    fn single(name: &'static str) -> Composition {
        Composition::single_builder(name, Box::new(ScriptedAgent::new(name))).build()
    }

    fn loop_of(name: &'static str, n: u32) -> Composition {
        Composition::loop_until_builder(move |_| single(name), |_| false)
            .max_iterations(n)
            .build()
    }

    fn big_budget() -> Arc<dyn Budget> {
        Arc::new(InMemoryBudget::new(100_000_000))
    }

    fn quiet_ctx(llm: Arc<dyn AgentLlmClient>) -> ExecCtx {
        ctx_for(
            llm,
            big_budget(),
            Arc::new(NoopEmitter),
            CancellationToken::new(),
        )
    }

    fn step_started_ids(rec: &RecordingEmitter) -> Vec<String> {
        rec.snapshot()
            .into_iter()
            .filter_map(|e| match e {
                BorgEvent::StepStarted { composition_id, .. } => Some(composition_id.to_string()),
                _ => None,
            })
            .collect()
    }

    // ── 1. Parallel branches keep their output_key writes ─────────────

    #[tokio::test]
    async fn parallel_branches_merge_output_key_writes_into_parent() {
        let llm = FakeLlmClient::always_ok("PASS\nbody");
        let comp = Composition::parallel_builder()
            .branch(
                "a",
                Composition::single_builder("a", Box::new(ScriptedAgent::new("a")))
                    .output_key("a_out")
                    .build(),
            )
            .branch(
                "b",
                Composition::single_builder("b", Box::new(ScriptedAgent::new("b")))
                    .output_key("b_out")
                    .build(),
            )
            .build();
        let outcome = execute(comp, quiet_ctx(llm)).await.unwrap();
        assert!(outcome.handoff.get("a_out").is_some(), "a_out lost");
        assert!(outcome.handoff.get("b_out").is_some(), "b_out lost");
        assert_eq!(outcome.handoff.history.len(), 2);
    }

    // ── 2. Checkpoint step ids are unique per nested position ─────────

    #[tokio::test]
    async fn resumable_seq_of_two_loops_runs_every_iteration() {
        let llm = FakeLlmClient::always_ok("PASS\nbody");
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let comp = Composition::sequential([loop_of("first", 3), loop_of("second", 3)]);
        let outcome = execute_resumable(comp, quiet_ctx(llm.clone()), store, "two-loops")
            .await
            .unwrap();
        assert_eq!(llm.calls(), 6, "the second loop must not be skipped");
        assert_eq!(outcome.verdicts.len(), 6);
    }

    #[tokio::test]
    async fn resumable_nested_seqs_with_same_step_name_run_both() {
        let llm = FakeLlmClient::always_ok("PASS\nbody");
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let comp = Composition::sequential([
            Composition::sequential([single("review")]),
            Composition::sequential([single("review")]),
        ]);
        let _ = execute_resumable_with(
            comp,
            quiet_ctx(llm.clone()),
            store,
            "nested",
            CheckpointPolicy::OnEveryStep,
        )
        .await
        .unwrap();
        assert_eq!(llm.calls(), 2);
    }

    #[tokio::test]
    async fn parallel_branches_do_not_share_composition_ids() {
        let llm = FakeLlmClient::always_ok("PASS\nbody");
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let rec = Arc::new(RecordingEmitter::new());
        // max_concurrency(1): branches run one after the other, so a
        // shared id would make the second branch skip deterministically.
        let comp = Composition::parallel_builder()
            .branch("left", loop_of("worker", 2))
            .branch("right", loop_of("worker", 2))
            .max_concurrency(1)
            .build();
        let ctx = ctx_for(
            llm.clone(),
            big_budget(),
            rec.clone(),
            CancellationToken::new(),
        );
        let _ = execute_resumable(comp, ctx, store, "par").await.unwrap();
        assert_eq!(llm.calls(), 4, "both branches run every iteration");

        let ids: Vec<String> = step_started_ids(&rec)
            .into_iter()
            .filter(|id| id.ends_with("::worker"))
            .collect();
        assert_eq!(ids.len(), 4, "{ids:?}");
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), 4, "branch step ids collide: {ids:?}");
    }

    #[tokio::test]
    async fn resume_after_crash_skips_exactly_the_finished_iterations() {
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let build = || Composition::sequential([loop_of("first", 3), loop_of("second", 3)]);

        // Crash on the 5th call: the first loop is done (3 calls), the
        // second loop's iter0 is done (4), its iter1 fails (5).
        let mut script: Vec<FakeOutcome> = (0..4).map(|_| FakeOutcome::ok("PASS\nok")).collect();
        script.push(FakeOutcome::terminal_err("crash"));
        let llm = FakeLlmClient::new(script);
        let crashed =
            execute_resumable(build(), quiet_ctx(llm.clone()), store.clone(), "crash").await;
        assert!(crashed.is_err());
        assert_eq!(llm.calls(), 5);
        let snap = store.load("crash").await.unwrap().expect("snapshot kept");
        assert_eq!(snap.completed_steps.len(), 4, "{:?}", snap.completed_steps);

        // Resume: only the second loop's iter1 and iter2 run.
        let llm = FakeLlmClient::always_ok("PASS\nresumed");
        let outcome = execute_resumable(build(), quiet_ctx(llm.clone()), store.clone(), "crash")
            .await
            .unwrap();
        assert_eq!(llm.calls(), 2, "exactly the unfinished work re-runs");
        assert_eq!(outcome.verdicts.len(), 2);
        assert_eq!(outcome.handoff.history.len(), 6);
        assert!(store.load("crash").await.unwrap().is_none());
    }

    // ── 3. Serial tool dispatch honours cancellation ──────────────────

    struct SleepyDispatch;

    #[async_trait]
    impl ToolDispatch for SleepyDispatch {
        async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError> {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(DispatchedCall {
                call_id: request.call_id,
                name: request.name,
                result: ToolCallResult::ok("slept")?,
            })
        }
        fn schemas(&self) -> Arc<[ToolSchema]> {
            Arc::from(Vec::<ToolSchema>::new())
        }
    }

    #[tokio::test]
    async fn serial_tool_dispatch_returns_cancelled_promptly() {
        let llm = FakeLlmClient::new(vec![
            FakeOutcome::ok_with_tool_calls(
                "calling",
                vec![ToolCallRequest::new("sleep", serde_json::json!({}), "1")],
            ),
            FakeOutcome::ok("done"),
        ]);
        let cancel = CancellationToken::new();
        let comp =
            crate::dsl::agent_loop(ScriptedAgent::new("sleeper"), Arc::new(SleepyDispatch), 5);
        let ctx = ctx_for(llm, big_budget(), Arc::new(NoopEmitter), cancel.clone());
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(3), execute(comp, ctx))
            .await
            .expect("serial dispatch ignored cancellation");
        assert!(matches!(result, Err(BorgError::Cancelled)), "{result:?}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    // 4. (max_steps = u32::MAX) lives in tests/agent_loop_capacity.rs: it
    // needs a global allocator of its own to see the oversized request.

    // ── 5. Parallel tool CollectAll delivers every result ─────────────

    struct FlakyParallel;

    #[async_trait]
    impl ToolDispatch for FlakyParallel {
        async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError> {
            if request.name == "boom" {
                return Err(BorgError::Internal {
                    message: "structural boom".into(),
                });
            }
            Ok(DispatchedCall {
                call_id: request.call_id,
                name: request.name,
                result: ToolCallResult::ok("fine")?,
            })
        }
        fn schemas(&self) -> Arc<[ToolSchema]> {
            Arc::from(Vec::<ToolSchema>::new())
        }
        fn parallelize_safe(&self, _tool_name: &str) -> bool {
            true
        }
    }

    fn flaky_llm() -> Arc<FakeLlmClient> {
        FakeLlmClient::new(vec![
            FakeOutcome::ok_with_tool_calls(
                "calling",
                vec![
                    ToolCallRequest::new("ok", serde_json::json!({}), "1"),
                    ToolCallRequest::new("boom", serde_json::json!({}), "2"),
                    ToolCallRequest::new("ok", serde_json::json!({}), "3"),
                ],
            ),
            FakeOutcome::ok("done"),
        ])
    }

    #[tokio::test]
    async fn parallel_tools_collect_all_delivers_failures_as_error_results() {
        let llm = flaky_llm();
        let comp = crate::dsl::agent_loop(ScriptedAgent::new("caller"), Arc::new(FlakyParallel), 5);
        let ctx = quiet_ctx(llm.clone()).with_parallel_tools(BatchFailureMode::CollectAll);
        let outcome = execute(comp, ctx).await.expect("the loop continues");
        assert_eq!(llm.calls(), 2, "the model saw the batch and answered");
        let results: Vec<DispatchedCall> = outcome.handoff.get_as("tool_results").unwrap();
        let ids: Vec<&str> = results.iter().map(|d| d.call_id.as_str()).collect();
        assert_eq!(ids, ["1", "2", "3"]);
        assert!(results[0].result.error.is_none());
        let err = results[1]
            .result
            .error
            .as_deref()
            .expect("an is_error result");
        assert!(err.contains("structural boom"), "{err}");
        assert_eq!(results[1].name, "boom");
        assert!(results[2].result.error.is_none());
    }

    #[tokio::test]
    async fn parallel_tools_fast_fail_surfaces_the_structural_error() {
        let llm = flaky_llm();
        let comp = crate::dsl::agent_loop(ScriptedAgent::new("caller"), Arc::new(FlakyParallel), 5);
        let ctx = quiet_ctx(llm.clone()).with_parallel_tools(BatchFailureMode::FastFail);
        let result = execute(comp, ctx).await;
        assert!(
            matches!(&result, Err(BorgError::Internal { message }) if message.contains("structural boom")),
            "{result:?}"
        );
        assert_eq!(llm.calls(), 1);
    }

    // ── 6. Event counts saturate instead of wrapping ──────────────────

    #[tokio::test]
    async fn completion_counts_saturate_past_255() {
        let llm = FakeLlmClient::always_ok("PASS\nbody");
        let rec = Arc::new(RecordingEmitter::new());
        let comp = Composition::branch(|_| BranchKey::from("go"), [("go", loop_of("tick", 300))]);
        let outcome = execute(
            comp,
            ctx_for(llm, big_budget(), rec.clone(), CancellationToken::new()),
        )
        .await
        .unwrap();
        assert_eq!(outcome.verdicts.len(), 300);
        // The loop's and the branch's completions both cover 300 verdicts.
        let counts: Vec<(String, u8, u8)> = rec
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                BorgEvent::CompositionCompleted {
                    composition_id,
                    succeeded,
                    failed,
                    ..
                } if composition_id.ends_with("::loop") || composition_id.ends_with("::branch") => {
                    Some((composition_id.to_string(), succeeded, failed))
                }
                _ => None,
            })
            .collect();
        assert_eq!(counts.len(), 2, "{counts:?}");
        for (id, succeeded, failed) in &counts {
            assert_eq!((*succeeded, *failed), (u8::MAX, 0), "{id}");
        }
    }

    // ── 7. Bad backoff configs don't panic ────────────────────────────

    #[test]
    fn backoff_with_nan_or_infinite_jitter_does_not_panic() {
        for ratio in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 7.0] {
            let b = crate::composition::Backoff::Exponential {
                base: Duration::from_millis(100),
                max: Duration::from_secs(1),
                jitter_ratio: ratio,
            };
            let d = b.delay_for_attempt(2);
            assert!(d <= Duration::from_millis(100), "{ratio}: {d:?}");
        }
    }

    #[test]
    fn backoff_with_near_max_duration_cap_does_not_panic() {
        for ratio in [f32::MIN_POSITIVE, 0.3, 1.0] {
            let b = crate::composition::Backoff::Exponential {
                base: Duration::MAX,
                max: Duration::MAX,
                jitter_ratio: ratio,
            };
            for attempt in 2..=30 {
                let _ = b.delay_for_attempt(attempt);
            }
        }
    }
}
