//! Concurrent `CachingActorStore` contention bench. The cache is a single
//! `Mutex<LruCache>` over all actors, so every load/save serializes on one
//! mutex; the single-threaded `caching_actor_store` bench can't surface
//! that. A load-hit also clones the cached payload, which dominates on
//! large snapshots.
//!
//! Bench groups:
//! * `store/load_hit_concurrent_8t` — 8 threads × 1k loads on a pre-warmed
//!   100-entry cache.
//! * `store/save_concurrent_8t` — 8 threads × 1k saves, 64 B payload (so
//!   payload work doesn't drown out lock cost).
//! * `store/save_large_payload_serial` — single-thread 1 MB save; pure
//!   payload-clone cost.
//!
//! Run with: `cargo bench -p borg-loop --bench store_concurrent`

use borg_loop::{ActorStore, CachingActorStore, InMemoryActorStore};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::runtime::Builder;

const KEYS_RESIDENT: usize = 100;
const THREADS: usize = 8;
const OPS_PER_THREAD: u64 = 1_000;

/// 4-worker runtime: small enough to keep scheduler noise from drowning
/// the lock-contention signal.
fn multi_rt() -> tokio::runtime::Runtime {
    Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("multi-thread runtime")
}

fn make_warmed_store() -> Arc<CachingActorStore<InMemoryActorStore>> {
    let store = Arc::new(CachingActorStore::new_with_capacity(
        InMemoryActorStore::new(),
        NonZeroUsize::new(1024).expect("nonzero cap"),
    ));
    // Pre-warm `KEYS_RESIDENT` entries so loads are hits.
    {
        let rt = multi_rt();
        let payload: Arc<[u8]> = Arc::from(vec![0xAB; 64]);
        rt.block_on(async {
            for i in 0..KEYS_RESIDENT {
                let key = format!("k-{i}");
                store
                    .save(&key, Arc::clone(&payload))
                    .await
                    .expect("prewarm save");
            }
            // Touch each key so it's resident in the cache.
            for i in 0..KEYS_RESIDENT {
                let key = format!("k-{i}");
                let _ = store.load(&key).await.expect("prewarm load");
            }
        });
    }
    store
}

fn load_hit_concurrent(c: &mut Criterion) {
    let mut g = c.benchmark_group("store/load_hit_concurrent_8t");
    g.throughput(Throughput::Elements((THREADS as u64) * OPS_PER_THREAD));
    g.sample_size(20); // 8k ops/iter is heavy

    g.bench_function("8t_x_1k_load_hit", |b| {
        b.iter_custom(|iters| {
            let store = make_warmed_store();
            let rt = multi_rt();
            let start = std::time::Instant::now();
            rt.block_on(async {
                for _ in 0..iters {
                    let mut handles = Vec::with_capacity(THREADS);
                    for t in 0..THREADS {
                        let store = Arc::clone(&store);
                        handles.push(tokio::spawn(async move {
                            for i in 0..OPS_PER_THREAD {
                                // Spread keys to avoid a single-hot-key pattern.
                                let key = format!("k-{}", (t * 13 + i as usize) % KEYS_RESIDENT);
                                let v = store.load(black_box(&key)).await.expect("load");
                                black_box(v);
                            }
                        }));
                    }
                    for h in handles {
                        h.await.expect("join");
                    }
                }
            });
            start.elapsed()
        });
    });

    g.finish();
}

fn save_concurrent_small(c: &mut Criterion) {
    let mut g = c.benchmark_group("store/save_concurrent_8t");
    g.throughput(Throughput::Elements((THREADS as u64) * OPS_PER_THREAD));
    g.sample_size(20);

    g.bench_function("8t_x_1k_save_64b", |b| {
        b.iter_custom(|iters| {
            let store = make_warmed_store();
            let rt = multi_rt();
            let payload: Arc<[u8]> = Arc::from(vec![0xCDu8; 64]);
            let start = std::time::Instant::now();
            rt.block_on(async {
                for _ in 0..iters {
                    let mut handles = Vec::with_capacity(THREADS);
                    for t in 0..THREADS {
                        let store = Arc::clone(&store);
                        let p = Arc::clone(&payload);
                        handles.push(tokio::spawn(async move {
                            for i in 0..OPS_PER_THREAD {
                                let key = format!("k-{t}-{i}");
                                store
                                    .save(black_box(&key), Arc::clone(&p))
                                    .await
                                    .expect("save");
                            }
                        }));
                    }
                    for h in handles {
                        h.await.expect("join");
                    }
                }
            });
            start.elapsed()
        });
    });

    g.finish();
}

fn save_large_payload_serial(c: &mut Criterion) {
    let mut g = c.benchmark_group("store/save_large_payload_serial");
    const SIZE: usize = 1024 * 1024;
    g.throughput(Throughput::Bytes(SIZE as u64));

    g.bench_function("1mb_save", |b| {
        let rt = multi_rt();
        let store = Arc::new(CachingActorStore::new_with_capacity(
            InMemoryActorStore::new(),
            NonZeroUsize::new(64).expect("nonzero"),
        ));
        let payload: Arc<[u8]> = Arc::from(vec![0xEEu8; SIZE]);
        let mut i = 0u64;
        b.iter(|| {
            rt.block_on(async {
                let key = format!("big-{i}");
                i = i.wrapping_add(1);
                store
                    .save(black_box(&key), Arc::clone(&payload))
                    .await
                    .expect("save");
            });
        });
    });

    g.finish();
}

criterion_group!(
    benches,
    load_hit_concurrent,
    save_concurrent_small,
    save_large_payload_serial,
);
criterion_main!(benches);
