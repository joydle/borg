//! Criterion benchmark for the HITL suspend-unwind hot path.
//!
//! 1. `request_suspend_construct_only` — building a `SuspendRequest`; the
//!    cheap baseline.
//! 2. `request_suspend_unwind` — full path from `ctx.request_suspend(req)`
//!    in `invoke` to the executor returning `Err(BorgError::Suspended {
//!    .. })`. Every HITL pause goes through this; it must stay well below
//!    LLM-call latency (10s of ms).
//! 3. `verdict_baseline` — same agent returning an Ok verdict instead of
//!    suspending, as a reference point.
//!
//! Run with: `cargo bench -p borg --bench suspend_unwind --features testing`

use async_trait::async_trait;
use borg::prelude::*;
use borg::suspend::{SuspendKind, SuspendRequest};
use borg::testing::FakeLlmClient;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::sync::Arc;
use tokio::runtime::Builder;

struct SuspendingAgent;

#[async_trait]
impl BorgAgent for SuspendingAgent {
    fn agent_id(&self) -> &'static str {
        "bench-suspender"
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn system_prompt(&self) -> &'static str {
        ""
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        None
    }
    async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
        let _ = ctx
            .request_suspend(
                SuspendRequest::new("approve?", SuspendKind::Approval).with_blocked_tool("migrate"),
            )
            .await?;
        unreachable!();
    }
}

struct VerdictAgent;

#[async_trait]
impl BorgAgent for VerdictAgent {
    fn agent_id(&self) -> &'static str {
        "bench-verdict"
    }
    fn model(&self) -> &str {
        "test/model"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn system_prompt(&self) -> &'static str {
        ""
    }
    fn user_prompt(&self, _ctx: &AgentContext) -> Option<String> {
        Some("p".into())
    }
}

fn bench_construct_only(c: &mut Criterion) {
    let mut g = c.benchmark_group("suspend/construct_only");
    g.throughput(Throughput::Elements(1));
    g.bench_function("construct_request", |b| {
        b.iter(|| {
            let req = SuspendRequest::new(
                black_box("approve migration?"),
                black_box(SuspendKind::Approval),
            )
            .with_blocked_tool(black_box("run_migration"));
            black_box(req);
        });
    });
    g.finish();
}

fn bench_unwind(c: &mut Criterion) {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    let mut g = c.benchmark_group("suspend/unwind");
    g.throughput(Throughput::Elements(1));

    g.bench_function("request_suspend_full_unwind", |b| {
        b.to_async(&rt).iter(|| async {
            let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("unused");
            let ctx = ExecCtx::new(
                llm,
                in_memory_budget(1_000),
                noop_emitter(),
                CancellationToken::new(),
                "bench",
            );
            let result = run_one("step", SuspendingAgent, ctx, HandoffState::new()).await;
            // Confirm the path actually unwinds via Suspended.
            debug_assert!(matches!(result, Err(BorgError::Suspended { .. })));
            black_box(result.is_err())
        });
    });

    g.bench_function("verdict_baseline", |b| {
        b.to_async(&rt).iter(|| async {
            let llm: Arc<dyn AgentLlmClient> = FakeLlmClient::always_ok("body");
            let ctx = ExecCtx::new(
                llm,
                in_memory_budget(1_000),
                noop_emitter(),
                CancellationToken::new(),
                "bench",
            );
            let v = run_one("step", VerdictAgent, ctx, HandoffState::new()).await;
            black_box(v.is_ok())
        });
    });

    g.finish();
}

criterion_group!(benches, bench_construct_only, bench_unwind);
criterion_main!(benches);
