//! Criterion benchmarks for the borg-tools file ops on the in-memory workspace.
//!
//! - `read_file_small_50_lines` — full read of a 50-line file.
//! - `replace_lines_middle_200_lines` — line-range edit mid-file.
//! - `delete_present` / `delete_absent_idempotent` — both delete paths.
//! - `diff_after_one_replace_100_lines` — unified diff via `similar`.
//!
//! Run with: `cargo bench -p borg-tools --bench workspace_ops`

use borg_tools::fs::{delete_file, read_file, read_file_diff, replace_lines};
use borg_tools::workspace::{InMemoryWorkspace, Workspace};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn small_file(lines: usize) -> String {
    let mut s = String::with_capacity(lines * 16);
    for i in 1..=lines {
        s.push_str(&format!("line {i} content goes here\n"));
    }
    s
}

fn rt() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn bench_read_file(c: &mut Criterion) {
    let rt = rt();
    let mut g = c.benchmark_group("borg_tools/read_file");
    g.throughput(Throughput::Elements(1));

    g.bench_function("read_file_small_50_lines", |b| {
        b.iter_batched(
            || {
                let ws = Arc::new(InMemoryWorkspace::new());
                let path = PathBuf::from("src/lib.rs");
                rt.block_on(ws.write(&path, small_file(50))).unwrap();
                (ws, path)
            },
            |(ws, path)| {
                let r = rt
                    .block_on(read_file(ws.as_ref(), &path, None, None))
                    .unwrap();
                black_box(r);
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

fn bench_replace_lines(c: &mut Criterion) {
    let rt = rt();
    let mut g = c.benchmark_group("borg_tools/replace_lines");
    g.throughput(Throughput::Elements(1));

    g.bench_function("replace_lines_middle_200_lines", |b| {
        b.iter_batched(
            || {
                let ws = Arc::new(InMemoryWorkspace::new());
                let path = PathBuf::from("src/main.rs");
                rt.block_on(ws.write(&path, small_file(200))).unwrap();
                (ws, path)
            },
            |(ws, path)| {
                let r = rt
                    .block_on(replace_lines(
                        ws.as_ref(),
                        &path,
                        95,
                        105,
                        "REPLACED\n".into(),
                    ))
                    .unwrap();
                black_box(r);
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

fn bench_delete_file(c: &mut Criterion) {
    let rt = rt();
    let mut g = c.benchmark_group("borg_tools/delete_file");
    g.throughput(Throughput::Elements(1));

    g.bench_function("delete_present", |b| {
        b.iter_batched(
            || {
                let ws = Arc::new(InMemoryWorkspace::new());
                let path = PathBuf::from("temp.txt");
                rt.block_on(ws.write(&path, "hi".into())).unwrap();
                (ws, path)
            },
            |(ws, path)| {
                rt.block_on(delete_file(ws.as_ref(), &path)).unwrap();
                black_box(());
            },
            BatchSize::SmallInput,
        );
    });

    g.bench_function("delete_absent_idempotent", |b| {
        b.iter_batched(
            || {
                let ws: Arc<dyn Workspace> = Arc::new(InMemoryWorkspace::new());
                let path = PathBuf::from("never_existed.txt");
                (ws, path)
            },
            |(ws, path)| {
                rt.block_on(delete_file(ws.as_ref(), &path)).unwrap();
                black_box(());
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

fn bench_read_file_diff(c: &mut Criterion) {
    let rt = rt();
    let mut g = c.benchmark_group("borg_tools/read_file_diff");
    g.throughput(Throughput::Elements(1));

    g.bench_function("diff_after_one_replace_100_lines", |b| {
        b.iter_batched(
            || {
                let ws = Arc::new(InMemoryWorkspace::new());
                let path = PathBuf::from("src/diff.rs");
                rt.block_on(async {
                    ws.write(&path, small_file(100)).await.unwrap();
                    // A single-line edit so the diff renders (single-write path
                    // would yield an empty diff).
                    replace_lines(ws.as_ref(), &path, 50, 50, "EDITED\n".into())
                        .await
                        .unwrap();
                });
                (ws, path)
            },
            |(ws, path)| {
                let d = rt.block_on(read_file_diff(ws.as_ref(), &path)).unwrap();
                black_box(d);
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_read_file,
    bench_replace_lines,
    bench_delete_file,
    bench_read_file_diff
);
criterion_main!(benches);
