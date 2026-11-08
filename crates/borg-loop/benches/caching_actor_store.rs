//! Criterion benchmarks for `CachingActorStore` load-hit / load-miss /
//! save performance. The cache is a `Mutex<LruCache>`; this measures the
//! serialization cost so added contention shows up early.
//!
//! Run with: `cargo bench -p borg-loop --bench caching_actor_store`

use borg_loop::{ActorStore, CachingActorStore, InMemoryActorStore};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::runtime::Builder;

fn rt() -> tokio::runtime::Runtime {
    Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt")
}

fn caching_store_load_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("caching_actor_store/load_hit");
    g.throughput(Throughput::Elements(1));
    g.bench_function("load_hit", |b| {
        let rt = rt();
        let store = CachingActorStore::new_with_capacity(
            InMemoryActorStore::new(),
            NonZeroUsize::new(1_024).unwrap(),
        );
        rt.block_on(async {
            store.save("hot", Arc::from(&b"payload"[..])).await.unwrap();
            let _ = store.load("hot").await.unwrap(); // prime the cache
        });
        b.iter(|| {
            rt.block_on(async {
                let v = store.load(black_box("hot")).await.unwrap();
                black_box(v);
            });
        });
    });
    g.finish();
}

fn caching_store_load_miss(c: &mut Criterion) {
    let mut g = c.benchmark_group("caching_actor_store/load_miss");
    g.throughput(Throughput::Elements(1));
    g.bench_function("load_miss", |b| {
        let rt = rt();
        let store = CachingActorStore::new_with_capacity(
            InMemoryActorStore::new(),
            NonZeroUsize::new(1_024).unwrap(),
        );
        b.iter(|| {
            rt.block_on(async {
                let v = store.load(black_box("absent")).await.unwrap();
                black_box(v);
            });
        });
    });
    g.finish();
}

fn caching_store_save(c: &mut Criterion) {
    let mut g = c.benchmark_group("caching_actor_store/save");
    g.throughput(Throughput::Bytes(64));
    g.bench_function("save_64b", |b| {
        let rt = rt();
        let store = CachingActorStore::new_with_capacity(
            InMemoryActorStore::new(),
            NonZeroUsize::new(1_024).unwrap(),
        );
        // Built once; the per-iter `Arc::clone` is just a refcount bump,
        // so the bench measures the store, not allocation.
        let payload: Arc<[u8]> = Arc::from(vec![0u8; 64]);
        let mut i = 0u64;
        b.iter(|| {
            rt.block_on(async {
                let id = format!("e-{i}");
                i = i.wrapping_add(1);
                store
                    .save(black_box(&id), Arc::clone(&payload))
                    .await
                    .unwrap();
            });
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    caching_store_load_hit,
    caching_store_load_miss,
    caching_store_save,
);
criterion_main!(benches);
