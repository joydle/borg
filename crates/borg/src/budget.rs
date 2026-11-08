//! Token budget enforcement.
//!
//! Borg caps total token spend across a whole composition tree (not just
//! turns per agent). The `Budget` trait lets the consumer plug in any
//! accounting strategy; `InMemoryBudget` is the default.
//!
//! Before `Composition::Parallel` fans out, the executor calls
//! `admit_parallel(branches, min_per_branch)`. Admission may downgrade —
//! running 3 of 5 branches when the budget can't cover all 5, emitting
//! `BorgEvent::ParallelDegraded` — rather than letting a branch fail
//! mid-flight when the budget runs out.

use std::sync::atomic::{AtomicU64, Ordering};

/// What one model call consumed, as the budget sees it.
///
/// `cache_read_tokens` and `cache_creation_tokens` are subsets of
/// `input_tokens` (see [`crate::ChatResponse`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
}

impl TokenUsage {
    /// Usage with no prompt-cache split.
    pub const fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
        }
    }

    /// Builder: the cache split of `input_tokens`, clamped so
    /// `read + creation <= input_tokens`.
    #[must_use]
    pub fn with_cache(mut self, read: u64, creation: u64) -> Self {
        let read = read.min(self.input_tokens);
        self.cache_read_tokens = read;
        self.cache_creation_tokens = creation.min(self.input_tokens - read);
        self
    }

    /// Every token the call carried, cached or not: `input + output`.
    pub fn total(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }

    /// The tokens a budget charges when cache reads cost
    /// `cache_read_percent`% of an ordinary input token: uncached input
    /// and output at full weight, cache reads scaled and rounded DOWN.
    ///
    /// `100` (or more) charges cache reads in full — the same number as
    /// [`Self::total`]. `10` is Anthropic's read price ratio.
    pub fn chargeable(&self, cache_read_percent: u8) -> u64 {
        let pct = u64::from(cache_read_percent.min(100));
        let read = self.cache_read_tokens.min(self.input_tokens);
        let uncached = self.input_tokens - read;
        let read_charged = read.saturating_mul(pct) / 100;
        uncached
            .saturating_add(read_charged)
            .saturating_add(self.output_tokens)
    }
}

/// Result of recording a completed call against the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetVerdict {
    /// Within budget; further calls allowed.
    Ok,
    /// Budget consumed exactly at this call. Further calls denied.
    Exhausted,
    /// Budget exceeded. Informational — calls were already denied.
    Overdrawn,
}

/// Outcome of a pre-flight admission check on a parallel composition.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParallelAdmission {
    /// Run all requested branches.
    Full,
    /// Budget covers fewer than requested. Run only the first `allowed`
    /// branches; the executor emits `ParallelDegraded`.
    Partial { allowed: usize },
    /// Not enough budget for even one branch. Caller falls back to the
    /// composition's declared default (often: empty verdict).
    Denied,
}

/// Budget contract. Implementations track a running total and answer
/// admission questions.
///
/// All methods take `&self`: budget state is internally synchronized, so
/// a single `Arc<dyn Budget>` can be shared across parallel branches
/// without a `Mutex`.
pub trait Budget: Send + Sync {
    /// Tokens consumed so far.
    fn consumed(&self) -> u64;

    /// Tokens remaining (saturating).
    fn remaining(&self) -> u64;

    /// Total budget. `consumed + remaining == limit` always holds for
    /// well-behaved implementations.
    fn limit(&self) -> u64;

    /// Whether a single new call is allowed. Cheap pre-check; reserves
    /// nothing.
    fn can_invoke(&self) -> bool {
        self.remaining() > 0
    }

    /// Record `tokens` consumed by a completed call. Implementations
    /// MUST be safe against concurrent calls (atomic add, not
    /// load-modify-store).
    fn record(&self, tokens: u64) -> BudgetVerdict;

    /// Record a completed call from its full usage. Borg's own call sites
    /// (the default [`crate::BorgAgent::invoke`]) charge through this.
    ///
    /// The default charges [`TokenUsage::total`] — every token, cache
    /// reads in full — through [`Self::record`], exactly what borg charged
    /// before usage carried a cache split. Override (or use
    /// [`InMemoryBudget::with_cache_read_percent`]) to charge cache reads
    /// at a discount.
    fn record_usage(&self, usage: &TokenUsage) -> BudgetVerdict {
        self.record(usage.total())
    }

    /// How many of `branches` parallel calls we can afford, assuming
    /// each consumes at least `min_per_branch` tokens. Called for
    /// `Composition::Parallel` admission before any branch spawns.
    fn admit_parallel(&self, branches: usize, min_per_branch: u64) -> ParallelAdmission {
        let need = (branches as u64).saturating_mul(min_per_branch);
        let have = self.remaining();
        if have >= need {
            ParallelAdmission::Full
        } else if min_per_branch == 0 || have == 0 {
            ParallelAdmission::Denied
        } else {
            let allowed = (have / min_per_branch) as usize;
            if allowed == 0 {
                ParallelAdmission::Denied
            } else if allowed >= branches {
                ParallelAdmission::Full
            } else {
                ParallelAdmission::Partial { allowed }
            }
        }
    }
}

impl dyn Budget {
    /// Build an `Arc<dyn Budget>` backed by [`InMemoryBudget`], ready
    /// for [`ExecCtx::new`](crate::ExecCtx::new).
    pub fn in_memory(limit: u64) -> std::sync::Arc<dyn Budget> {
        std::sync::Arc::new(InMemoryBudget::new(limit))
    }
}

/// Default `Budget`, backed by atomic counters. Sized for a single
/// composition tree's lifetime. Cheap to construct and share.
#[derive(Debug)]
pub struct InMemoryBudget {
    limit: u64,
    consumed: AtomicU64,
    /// Share of an ordinary input token a cache read costs, in percent.
    /// 100 = no discount (the default).
    cache_read_percent: u8,
}

impl InMemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            consumed: AtomicU64::new(0),
            cache_read_percent: 100,
        }
    }

    /// Charge prompt-cache reads at `percent`% of an ordinary input token
    /// in [`Budget::record_usage`] (rounded down; clamped to 100). Off by
    /// default: a budget built without this charges every token in full.
    ///
    /// The budget is a proxy for money, and a cached prefix is cheap — a
    /// 20K-token cached brief charged as 20K can exhaust a session pool
    /// after a couple of reviews. `10` matches Anthropic's read price.
    #[must_use]
    pub fn with_cache_read_percent(mut self, percent: u8) -> Self {
        self.cache_read_percent = percent.min(100);
        self
    }

    /// Build an `Arc<dyn Budget>` ready for [`crate::ExecCtx`].
    pub fn shared(limit: u64) -> std::sync::Arc<dyn Budget> {
        std::sync::Arc::new(Self::new(limit))
    }
}

/// Free-function shortcut: `borg::in_memory_budget(40_000)`.
pub fn in_memory_budget(limit: u64) -> std::sync::Arc<dyn Budget> {
    std::sync::Arc::new(InMemoryBudget::new(limit))
}

impl Budget for InMemoryBudget {
    fn consumed(&self) -> u64 {
        self.consumed.load(Ordering::Relaxed)
    }

    fn remaining(&self) -> u64 {
        self.limit
            .saturating_sub(self.consumed.load(Ordering::Relaxed))
    }

    fn limit(&self) -> u64 {
        self.limit
    }

    fn record_usage(&self, usage: &TokenUsage) -> BudgetVerdict {
        self.record(usage.chargeable(self.cache_read_percent))
    }

    fn record(&self, tokens: u64) -> BudgetVerdict {
        // Atomic saturating add: concurrent recorders can't lose updates,
        // and a huge or hostile count pins the total at `u64::MAX`
        // instead of wrapping back under the limit. The verdict compares
        // the post-add total against the limit. (`fetch_update` is
        // deprecated in favour of `try_update` on newer toolchains, which
        // the MSRV lacks.)
        #[allow(deprecated)]
        let prev = self
            .consumed
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
                Some(c.saturating_add(tokens))
            })
            .unwrap_or_else(|c| c);
        let total = prev.saturating_add(tokens);
        if total < self.limit {
            BudgetVerdict::Ok
        } else if total == self.limit {
            BudgetVerdict::Exhausted
        } else {
            BudgetVerdict::Overdrawn
        }
    }
}

/// A pool sized for one composition, carved out of a wider budget.
///
/// Give a composition its own ceiling — a draft, review and revise loop
/// sharing ONE pool, say — while every token it spends still
/// counts against the session's budget. `record` charges both; the pool
/// is spent when either is, so `remaining` is the smaller of the two.
///
/// ```
/// use borg::budget::{sub_budget, Budget};
///
/// let session = borg::in_memory_budget(100_000);
/// let round = sub_budget(session.clone(), 30_000); // the whole round's pool
/// round.record(12_000);
/// assert_eq!(round.remaining(), 18_000);
/// assert_eq!(session.consumed(), 12_000);
/// ```
///
/// Pass it as the composition's [`crate::ExecCtx`] budget: every step in
/// the tree shares it.
pub struct SubBudget {
    parent: std::sync::Arc<dyn Budget>,
    own: InMemoryBudget,
}

impl SubBudget {
    /// A pool of `limit` tokens inside `parent`.
    pub fn new(parent: std::sync::Arc<dyn Budget>, limit: u64) -> Self {
        Self {
            parent,
            own: InMemoryBudget::new(limit),
        }
    }

    /// Charge cache reads at `percent`% against this pool's own limit (as
    /// [`InMemoryBudget::with_cache_read_percent`]); the parent applies its
    /// own rule.
    #[must_use]
    pub fn with_cache_read_percent(mut self, percent: u8) -> Self {
        self.own = self.own.with_cache_read_percent(percent);
        self
    }
}

/// `Arc<dyn Budget>` shortcut for [`SubBudget::new`].
pub fn sub_budget(parent: std::sync::Arc<dyn Budget>, limit: u64) -> std::sync::Arc<dyn Budget> {
    std::sync::Arc::new(SubBudget::new(parent, limit))
}

fn worse(a: BudgetVerdict, b: BudgetVerdict) -> BudgetVerdict {
    use BudgetVerdict::*;
    match (a, b) {
        (Overdrawn, _) | (_, Overdrawn) => Overdrawn,
        (Exhausted, _) | (_, Exhausted) => Exhausted,
        _ => Ok,
    }
}

impl Budget for SubBudget {
    fn consumed(&self) -> u64 {
        self.own.consumed()
    }

    fn remaining(&self) -> u64 {
        self.own.remaining().min(self.parent.remaining())
    }

    fn limit(&self) -> u64 {
        self.own.limit()
    }

    fn record(&self, tokens: u64) -> BudgetVerdict {
        worse(self.own.record(tokens), self.parent.record(tokens))
    }

    fn record_usage(&self, usage: &TokenUsage) -> BudgetVerdict {
        worse(
            self.own.record_usage(usage),
            self.parent.record_usage(usage),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn in_memory_starts_with_full_budget() {
        let b = InMemoryBudget::new(1000);
        assert_eq!(b.consumed(), 0);
        assert_eq!(b.remaining(), 1000);
        assert_eq!(b.limit(), 1000);
        assert!(b.can_invoke());
    }

    #[test]
    fn record_advances_consumption() {
        let b = InMemoryBudget::new(1000);
        assert_eq!(b.record(300), BudgetVerdict::Ok);
        assert_eq!(b.consumed(), 300);
        assert_eq!(b.remaining(), 700);
    }

    #[test]
    fn record_exact_limit_returns_exhausted() {
        let b = InMemoryBudget::new(1000);
        assert_eq!(b.record(1000), BudgetVerdict::Exhausted);
        assert!(!b.can_invoke());
    }

    #[test]
    fn record_saturates_instead_of_wrapping() {
        // Regression: `fetch_add` wrapped 500 + u64::MAX to 499, which
        // put the budget back under its limit.
        let b = InMemoryBudget::new(1000);
        b.record(500);
        assert_eq!(b.record(u64::MAX), BudgetVerdict::Overdrawn);
        assert_eq!(b.consumed(), u64::MAX);
        assert_eq!(b.remaining(), 0);
        assert!(!b.can_invoke());
        // Further charges stay pinned.
        b.record(1);
        assert_eq!(b.consumed(), u64::MAX);
    }

    #[test]
    fn sub_budget_and_cache_weighted_charges_saturate() {
        let session = in_memory_budget(1000);
        let round = SubBudget::new(session.clone(), 800).with_cache_read_percent(10);
        round.record(500);
        let huge = TokenUsage::new(u64::MAX, u64::MAX).with_cache(u64::MAX / 2, 0);
        assert_ne!(round.record_usage(&huge), BudgetVerdict::Ok);
        assert_eq!(round.consumed(), u64::MAX);
        assert_eq!(session.consumed(), u64::MAX);
        assert!(!round.can_invoke());
        assert!(!session.can_invoke());
    }

    #[test]
    fn record_past_limit_returns_overdrawn() {
        let b = InMemoryBudget::new(1000);
        b.record(800);
        assert_eq!(b.record(300), BudgetVerdict::Overdrawn);
    }

    #[test]
    fn record_usage_charges_every_token_by_default() {
        let b = InMemoryBudget::new(100_000);
        let u = TokenUsage::new(20_000, 500).with_cache(18_000, 0);
        b.record_usage(&u);
        assert_eq!(b.consumed(), 20_500, "no discount unless asked for");
    }

    #[test]
    fn record_usage_discounts_cache_reads_when_configured() {
        let b = InMemoryBudget::new(100_000).with_cache_read_percent(10);
        let u = TokenUsage::new(20_000, 500).with_cache(18_000, 1_000);
        b.record_usage(&u);
        // 2_000 uncached + 1_800 (18_000 at 10%) + 500 output. A cache
        // WRITE is not discounted — it is part of the uncached remainder.
        assert_eq!(b.consumed(), 2_000 + 1_800 + 500);
    }

    #[test]
    fn chargeable_rounds_reads_down_and_clamps_the_rate() {
        let u = TokenUsage::new(100, 0).with_cache(15, 0);
        assert_eq!(u.chargeable(10), 85 + 1, "1.5 reads round down to 1");
        assert_eq!(u.chargeable(250), u.total(), "over 100% is 100%");
        assert_eq!(u.chargeable(0), 85, "free reads");
        let over = TokenUsage::new(10, 0).with_cache(50, 50);
        assert_eq!(
            (over.cache_read_tokens, over.cache_creation_tokens),
            (10, 0)
        );
    }

    /// A custom `Budget` that never heard of `record_usage` still charges
    /// the full total through its `record`.
    #[test]
    fn a_budget_without_record_usage_charges_the_total() {
        struct Plain(AtomicU64);
        impl Budget for Plain {
            fn consumed(&self) -> u64 {
                self.0.load(Ordering::Relaxed)
            }
            fn remaining(&self) -> u64 {
                u64::MAX
            }
            fn limit(&self) -> u64 {
                u64::MAX
            }
            fn record(&self, tokens: u64) -> BudgetVerdict {
                self.0.fetch_add(tokens, Ordering::Relaxed);
                BudgetVerdict::Ok
            }
        }
        let b = Plain(AtomicU64::new(0));
        b.record_usage(&TokenUsage::new(1_000, 10).with_cache(900, 0));
        assert_eq!(b.consumed(), 1_010);
    }

    #[test]
    fn admit_parallel_full_when_budget_covers_all() {
        let b = InMemoryBudget::new(10_000);
        match b.admit_parallel(5, 1000) {
            ParallelAdmission::Full => {}
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn admit_parallel_partial_when_budget_short() {
        let b = InMemoryBudget::new(3500);
        match b.admit_parallel(5, 1000) {
            ParallelAdmission::Partial { allowed: 3 } => {}
            other => panic!("expected Partial(3), got {other:?}"),
        }
    }

    #[test]
    fn admit_parallel_denied_when_no_budget() {
        let b = InMemoryBudget::new(500);
        match b.admit_parallel(5, 1000) {
            ParallelAdmission::Denied => {}
            other => panic!("expected Denied, got {other:?}"),
        }
    }

    #[test]
    fn admit_parallel_denied_when_remaining_is_zero() {
        let b = InMemoryBudget::new(1000);
        b.record(1000);
        match b.admit_parallel(2, 1) {
            ParallelAdmission::Denied => {}
            other => panic!("expected Denied, got {other:?}"),
        }
    }

    #[test]
    fn a_sub_budget_charges_its_parent_and_is_capped_by_both() {
        let parent: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(10_000));
        let pool = SubBudget::new(parent.clone(), 3_000);
        assert_eq!(pool.record(1_000), BudgetVerdict::Ok);
        assert_eq!((pool.consumed(), parent.consumed()), (1_000, 1_000));
        assert_eq!(pool.remaining(), 2_000);
        assert_eq!(pool.record(2_000), BudgetVerdict::Exhausted);
        assert!(!pool.can_invoke());
        assert!(parent.can_invoke(), "the session keeps going");

        // The parent's ceiling binds too.
        let parent: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(500));
        let pool = SubBudget::new(parent.clone(), 3_000);
        assert_eq!(pool.remaining(), 500);
        assert_eq!(pool.record(600), BudgetVerdict::Overdrawn);
    }

    #[test]
    fn a_sub_budget_applies_its_own_cache_rule() {
        let parent: Arc<dyn Budget> = Arc::new(InMemoryBudget::new(1_000_000));
        let pool = SubBudget::new(parent.clone(), 100_000).with_cache_read_percent(10);
        let usage = TokenUsage::new(20_000, 500).with_cache(18_000, 0);
        pool.record_usage(&usage);
        assert_eq!(pool.consumed(), 2_000 + 1_800 + 500);
        assert_eq!(
            parent.consumed(),
            20_500,
            "the parent charges by its own rule"
        );
    }

    #[tokio::test]
    async fn concurrent_records_are_atomic() {
        // Smoke-test that fetch_add doesn't lose updates under 100
        // concurrent recorders.
        let b = Arc::new(InMemoryBudget::new(u64::MAX));
        let mut handles = Vec::new();
        for _ in 0..100 {
            let bb = b.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..1000 {
                    bb.record(1);
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(b.consumed(), 100_000);
    }

    #[test]
    fn admit_parallel_with_zero_min_per_branch_admits_full() {
        // `min_per_branch == 0` means "no floor": `need = 0`, so
        // `have >= need` always holds and the first arm returns `Full`
        // regardless of remaining budget.
        let b = InMemoryBudget::new(10_000);
        match b.admit_parallel(5, 0) {
            ParallelAdmission::Full => {}
            other => panic!("expected Full when min is 0, got {other:?}"),
        }
        let b2 = InMemoryBudget::new(0);
        match b2.admit_parallel(5, 0) {
            ParallelAdmission::Full => {}
            other => panic!("expected Full when both are 0, got {other:?}"),
        }
    }
}
