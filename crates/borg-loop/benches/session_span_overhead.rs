//! Criterion benchmark for the per-turn `borg.session` span open/close
//! cost (allocated once per turn so OTel can export it). Confirms the
//! overhead stays well below per-turn LLM latency.
//!
//! Run with: `cargo bench -p borg-loop --bench session_span_overhead`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tokio::runtime::Builder;
use tracing::Instrument;

fn session_span_open_close(c: &mut Criterion) {
    let mut g = c.benchmark_group("session_span/open_close");
    g.throughput(Throughput::Elements(1));
    g.bench_function("open_close", |b| {
        b.iter(|| {
            let span = tracing::info_span!(
                "borg.session",
                session.id = "bench",
                actor.type = "BenchActor",
                turn.index = 0u32,
            );
            black_box(&span);
            drop(span); // close
        });
    });
    g.finish();
}

fn session_span_instrument_noop_async(c: &mut Criterion) {
    let mut g = c.benchmark_group("session_span/instrument_noop_async");
    g.throughput(Throughput::Elements(1));
    g.bench_function("instrument_noop", |b| {
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        b.iter(|| {
            rt.block_on(async {
                let span = tracing::info_span!(
                    "borg.session",
                    session.id = "bench",
                    actor.type = "BenchActor",
                    turn.index = 0u32,
                );
                async { black_box(()) }.instrument(span).await;
            });
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    session_span_open_close,
    session_span_instrument_noop_async
);
criterion_main!(benches);
