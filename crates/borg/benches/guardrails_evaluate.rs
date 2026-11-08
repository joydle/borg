//! `GuardrailEngine::evaluate` per-boundary cost as rule count scales.
//!
//! Tracks three allocation hotspots, all wasted work when rules return
//! `&'static str`:
//!
//! * Rules allocate two owned `String`s per non-Ok verdict via
//!   `rule.name().to_owned().into()` — candidate for `Cow<'static, str>`.
//! * Verdict messages clone the corrective body.
//! * The message `Vec` has no capacity hint, so it reallocates as rules
//!   push.
//!
//! Bench shape: N rules, half `Warn`, half `Ok`, all with `&'static`
//! correctives; measure one `evaluate()` call.
//!
//! Run with: `cargo bench -p borg --bench guardrails_evaluate`

use borg::{
    in_memory_budget, GuardrailCtx, GuardrailEngine, GuardrailRule, GuardrailVerdict,
    WindowedCounterDetector,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Always returns `Ok` — the no-allocation happy path.
struct AlwaysOkRule {
    name: &'static str,
}

impl GuardrailRule for AlwaysOkRule {
    fn name(&self) -> &str {
        self.name
    }
    fn name_static(&self) -> Option<&'static str> {
        Some(self.name)
    }
    fn evaluate(&self, _ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
        GuardrailVerdict::Ok
    }
}

/// Always returns `Warn` with a `&'static` corrective — the path that
/// allocates two `String`s (name and corrective) per call.
struct AlwaysWarnRule {
    name: &'static str,
}

impl GuardrailRule for AlwaysWarnRule {
    fn name(&self) -> &str {
        self.name
    }
    fn name_static(&self) -> Option<&'static str> {
        Some(self.name)
    }
    fn evaluate(&self, _ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
        GuardrailVerdict::Warn {
            corrective: "ratchet down output verbosity".to_string(),
        }
    }
}

fn build_engine(n: usize) -> GuardrailEngine {
    // Half Warn (allocation path), half Ok (fast path).
    let mut engine = GuardrailEngine::new();
    for i in 0..n {
        if i % 2 == 0 {
            engine.register(AlwaysOkRule {
                name: stringify_idx(i),
            });
        } else {
            engine.register(AlwaysWarnRule {
                name: stringify_idx(i),
            });
        }
    }
    engine
}

/// Map an index to a `&'static str` via a static table (can't `format!`
/// to `&'static`). Covers N up to 32.
fn stringify_idx(i: usize) -> &'static str {
    const NAMES: [&str; 32] = [
        "r00", "r01", "r02", "r03", "r04", "r05", "r06", "r07", "r08", "r09", "r10", "r11", "r12",
        "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25",
        "r26", "r27", "r28", "r29", "r30", "r31",
    ];
    NAMES[i % NAMES.len()]
}

fn guardrails_evaluate(c: &mut Criterion) {
    let mut g = c.benchmark_group("guardrails/evaluate");
    let detector = WindowedCounterDetector::new();
    let budget = in_memory_budget(1_000_000);

    for &n_rules in &[1usize, 5, 15, 30] {
        let engine = build_engine(n_rules);
        g.throughput(Throughput::Elements(n_rules as u64));
        g.bench_with_input(
            BenchmarkId::new("n_rules", n_rules),
            &engine,
            |b, engine| {
                b.iter(|| {
                    let ctx =
                        GuardrailCtx::new(&detector, budget.as_ref(), 7, Duration::from_secs(12));
                    let outcome = engine.evaluate(ctx);
                    black_box(outcome);
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, guardrails_evaluate);
criterion_main!(benches);
