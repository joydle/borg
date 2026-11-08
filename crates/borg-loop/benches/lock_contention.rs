//! Criterion benchmark comparing `parking_lot::Mutex` to
//! `std::sync::Mutex` at the store layer's call patterns. `parking_lot`
//! was chosen for its non-poisoning property; this measures the perf
//! delta on uncontended and 8-thread-contended acquires.
//!
//! Run with: `cargo bench -p borg-loop --bench lock_contention`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::sync::Arc;
use std::thread;

const SERIAL_ITERATIONS: u64 = 100_000;

fn uncontended_acquire(c: &mut Criterion) {
    let mut g = c.benchmark_group("lock_contention/uncontended");
    g.throughput(Throughput::Elements(1));

    g.bench_function("parking_lot", |b| {
        let m = parking_lot::Mutex::new(0u64);
        b.iter(|| {
            let mut g = m.lock();
            *g = g.wrapping_add(1);
            black_box(*g);
        });
    });

    g.bench_function("std_sync", |b| {
        let m = std::sync::Mutex::new(0u64);
        b.iter(|| {
            let mut g = m.lock().expect("poisoned");
            *g = g.wrapping_add(1);
            black_box(*g);
        });
    });

    g.finish();
}

fn contended_8_threads(c: &mut Criterion) {
    let mut g = c.benchmark_group("lock_contention/contended_8t");
    g.throughput(Throughput::Elements(8 * SERIAL_ITERATIONS));

    g.bench_function("parking_lot", |b| {
        b.iter_custom(|iters| {
            let m = Arc::new(parking_lot::Mutex::new(0u64));
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for _ in 0..SERIAL_ITERATIONS {
                                let mut g = m.lock();
                                *g = g.wrapping_add(1);
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

    g.bench_function("std_sync", |b| {
        b.iter_custom(|iters| {
            let m = Arc::new(std::sync::Mutex::new(0u64));
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let handles: Vec<_> = (0..8)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for _ in 0..SERIAL_ITERATIONS {
                                let mut g = m.lock().expect("poisoned");
                                *g = g.wrapping_add(1);
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

criterion_group!(benches, uncontended_acquire, contended_8_threads);
criterion_main!(benches);
