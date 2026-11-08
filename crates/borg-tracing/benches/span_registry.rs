//! Criterion benchmarks for `SpanRegistry` insert/get performance.
//!
//! The `Mutex<LruCache<SpanKey, tracing::Span>>` serializes all access;
//! these benches measure the contention cost so regressions stay visible.
//!
//! Run with: `cargo bench -p borg-tracing --bench span_registry`

use borg_tracing::{SpanKey, SpanRegistry};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::sync::Arc;
use std::thread;

fn span_registry_insert_serial(c: &mut Criterion) {
    let mut g = c.benchmark_group("span_registry/insert_serial");
    g.throughput(Throughput::Elements(1));
    g.bench_function("insert", |b| {
        let store = SpanRegistry::default();
        let mut i = 0u32;
        b.iter(|| {
            let key = SpanKey::Turn("c".into(), "s".into(), i);
            store.insert(key, tracing::Span::none());
            i = i.wrapping_add(1);
        });
    });
    g.finish();
}

fn span_registry_get_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("span_registry/get_hit");
    g.throughput(Throughput::Elements(1));
    g.bench_function("get_hit", |b| {
        let store = SpanRegistry::default();
        let key = SpanKey::Composition("hot".into());
        store.insert(key.clone(), tracing::Span::none());
        b.iter(|| {
            let v = store.get(black_box(&key));
            black_box(v);
        });
    });
    g.finish();
}

fn span_registry_get_miss(c: &mut Criterion) {
    let mut g = c.benchmark_group("span_registry/get_miss");
    g.throughput(Throughput::Elements(1));
    g.bench_function("get_miss", |b| {
        let store = SpanRegistry::default();
        let key = SpanKey::Composition("cold".into());
        b.iter(|| {
            let v = store.get(black_box(&key));
            black_box(v);
        });
    });
    g.finish();
}

/// 8-thread contention bench — worst-case load from the executor's
/// parallel-composition path against the serializing `Mutex<LruCache>`.
fn span_registry_insert_parallel(c: &mut Criterion) {
    let mut g = c.benchmark_group("span_registry/insert_parallel_8");
    g.throughput(Throughput::Elements(8 * 1_000));
    g.bench_function("8x1000_inserts", |b| {
        b.iter_custom(|iters| {
            let store = Arc::new(SpanRegistry::default());
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let handles: Vec<_> = (0..8)
                    .map(|tid| {
                        let s = Arc::clone(&store);
                        thread::spawn(move || {
                            for i in 0..1_000u32 {
                                s.insert(
                                    SpanKey::Turn(Arc::from(format!("c-{tid}")), "s".into(), i),
                                    tracing::Span::none(),
                                );
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            }
            start.elapsed()
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    span_registry_insert_serial,
    span_registry_get_hit,
    span_registry_get_miss,
    span_registry_insert_parallel,
);
criterion_main!(benches);
