//! Per-(provider, agent) circuit breaker — Polly / Hystrix shape, used
//! by the executor before every `agent.invoke`.
//!
//! During a provider outage (sustained 5xx / 429 / network reset), every
//! sub-agent independently retries the dead provider and burns its
//! `RestartPolicy` budget. The breaker keys on `(provider, agent_id)`,
//! trips after `failure_threshold` consecutive failures, and fail-fasts
//! subsequent calls with
//! [`BorgError::ProviderUnavailable`](crate::BorgError::ProviderUnavailable)
//! for `open_duration`, then half-opens and permits a single trial call
//! to discover whether the upstream recovered.
//!
//! ## Wiring
//!
//! Every [`crate::ExecCtx`] carries a cloneable [`CircuitBreaker`]; child
//! contexts share the inner Arc, so fan-out siblings observe and update
//! the same state. The executor consults [`CircuitBreaker::check`] before
//! `agent.invoke()` and calls [`CircuitBreaker::record_success`] /
//! [`CircuitBreaker::record_failure`] after.
//!
//! ## State machine
//!
//! ```text
//!   Closed { consec_fails: 0 }
//!      │   failure → consec_fails+=1
//!      │
//!      │   if consec_fails >= threshold ──▶  Open { reopens_at: now+T }
//!      │                                          │
//!      │                                          │  check after reopens_at
//!      │                                          ▼
//!      │                                     HalfOpen { trial_in_flight }
//!      │   success                                │
//!      │ ◀────────────────────────────────────────┤  trial succeeds → Closed
//!      │                                          │
//!      │   failure ──────▶  Open { ... }  ◀───────┤  trial fails → Open
//!      │                                          │
//!      │                       second concurrent  │
//!      │                       check in HalfOpen ─┘  → ProviderUnavailable
//! ```
//!
//! HalfOpen permits only one trial in flight; concurrent callers
//! fail-fast, so an outage can't recruit N parallel discovery calls to a
//! dead service.
//!
//! The trial holds its slot on a lease of `open_duration`. A trial that
//! never reports (it ended with a non-retryable error such as a 400,
//! `BudgetExhausted` or `Cancelled`, or it panicked) can't wedge the key:
//! once the lease runs out the next caller becomes the trial. A trial
//! that knows its outcome says nothing about provider health can hand the
//! slot back at once with [`CircuitBreaker::release_trial`].
//!
//! ## Disabling
//!
//! [`CircuitBreaker::disabled`] makes every `check` return Ok and every
//! record a no-op. Use in tests that exercise raw retry logic.
//!
//! ## Sliding-window mode
//!
//! [`WindowBreaker`] counts failures inside a rolling window per key
//! (shared by every clone, so one breaker can gate a whole process) and
//! works standalone — `check` / `allow` / `record_*` around any call, no
//! executor needed. [`CircuitBreaker::windowed`] puts one in front of the
//! executor keyed by provider: every agent on a provider counts into the
//! same window, alongside whatever else holds a clone. The consecutive
//! mode above stays the default.

mod window;

pub use window::{
    counts_toward_breaker, Admission, Recovery, Trip, WindowBreaker, WindowPolicy, WindowStatus,
    PROBE_BUSY_RETRY,
};

use crate::agent::Provider;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Per-(provider, agent) breaker policy.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct BreakerPolicy {
    /// Consecutive failures that trip the breaker from Closed → Open.
    pub failure_threshold: u32,
    /// Time in the Open state before the next check transitions us to
    /// HalfOpen and lets one trial call through.
    pub open_duration: Duration,
}

impl BreakerPolicy {
    /// `failure_threshold` consecutive failures open the breaker for
    /// `open_duration`, then one half-open trial decides.
    #[must_use]
    pub const fn new(failure_threshold: u32, open_duration: Duration) -> Self {
        Self {
            failure_threshold,
            open_duration,
        }
    }
}

impl BreakerPolicy {
    /// 5 consecutive failures, 30-second open window: short enough that a
    /// trial discovers recovery quickly, long enough that a real outage
    /// isn't spammed.
    pub const fn default_for_llm() -> Self {
        Self {
            failure_threshold: 5,
            open_duration: Duration::from_secs(30),
        }
    }

    /// Sentinel: never trip. Every `check` returns Ok.
    pub const fn disabled() -> Self {
        Self {
            failure_threshold: u32::MAX,
            open_duration: Duration::ZERO,
        }
    }

    fn is_disabled(&self) -> bool {
        self.failure_threshold == u32::MAX || self.open_duration.is_zero()
    }
}

impl Default for BreakerPolicy {
    fn default() -> Self {
        Self::default_for_llm()
    }
}

/// Cloneable circuit breaker. Inner state is Arc-shared across clones,
/// so `ExecCtx::clone()` propagates one breaker through every
/// composition child and fan-out siblings see each other's failures.
#[derive(Clone)]
pub struct CircuitBreaker {
    inner: Arc<Mutex<BreakerInner>>,
    /// Sliding-window mode ([`Self::windowed`]): decisions go here, keyed
    /// by provider, and `inner` only answers [`Self::policy`].
    window: Option<WindowBreaker<Provider>>,
}

struct BreakerInner {
    policy: BreakerPolicy,
    breakers: HashMap<(Provider, &'static str), BreakerState>,
}

#[derive(Debug)]
enum BreakerState {
    Closed {
        consecutive_failures: u32,
    },
    /// `reopens_at: None` means `now + open_duration` overflowed
    /// `Instant`: the breaker stays open for the life of the process.
    Open {
        reopens_at: Option<Instant>,
    },
    /// `Some(t)`: one trial has held the slot since `t`, on a lease that
    /// runs out after `open_duration`. `None`: the slot is free (the last
    /// trial released it) and the next check claims it.
    HalfOpen {
        trial_since: Option<Instant>,
    },
}

/// How long concurrent callers are told to wait while a trial is in
/// flight: short, so they retry soon after the trial reports.
const TRIAL_BUSY_RETRY: Duration = Duration::from_millis(50);

/// The Open state entered at `now`. `checked_add` so an `open_duration`
/// near `Duration::MAX` means "far future" instead of a panic.
fn open_from(now: Instant, open_duration: Duration) -> BreakerState {
    BreakerState::Open {
        reopens_at: now.checked_add(open_duration),
    }
}

impl CircuitBreaker {
    /// New breaker with the given policy.
    pub fn new(policy: BreakerPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BreakerInner {
                policy,
                breakers: HashMap::new(),
            })),
            window: None,
        }
    }

    /// Sliding-window mode: `breaker` decides every check and record,
    /// keyed by PROVIDER (the agent id is ignored), so all agents on one
    /// provider share a window — and share it with every other clone of
    /// `breaker`, e.g. a main loop calling the same provider outside the
    /// executor. [`Self::policy`] reports its threshold and cooldown.
    pub fn windowed(breaker: WindowBreaker<Provider>) -> Self {
        let p = breaker.policy();
        Self {
            inner: Arc::new(Mutex::new(BreakerInner {
                policy: BreakerPolicy {
                    failure_threshold: p.failure_threshold,
                    open_duration: p.cooldown,
                },
                breakers: HashMap::new(),
            })),
            window: Some(breaker),
        }
    }

    /// The sliding-window breaker behind [`Self::windowed`], if any.
    pub fn window(&self) -> Option<&WindowBreaker<Provider>> {
        self.window.as_ref()
    }

    /// Default policy ([`BreakerPolicy::default_for_llm`]).
    pub fn with_default_policy() -> Self {
        Self::new(BreakerPolicy::default_for_llm())
    }

    /// Breaker disabled.
    pub fn disabled() -> Self {
        Self::new(BreakerPolicy::disabled())
    }

    /// Check whether a call may proceed. `Ok(())` for Closed and the
    /// permitted-trial branch of HalfOpen; `Err(reopens_in)` for Open or
    /// for concurrent HalfOpen requests beyond the first trial.
    ///
    /// Transitions Open → HalfOpen once the open window has elapsed; only
    /// the caller that wins the race claims the trial slot. A trial that
    /// has not reported within `open_duration` loses the slot to the next
    /// caller, so a trial that ended without recording can't wedge the key.
    pub fn check(&self, provider: Provider, agent_id: &'static str) -> Result<(), Duration> {
        if let Some(window) = &self.window {
            return window.check(&provider).map(|_| ());
        }
        let mut inner = self.inner.lock().expect("circuit breaker poisoned");
        let policy = inner.policy;
        if policy.is_disabled() {
            return Ok(());
        }
        let now = Instant::now();
        let key = (provider, agent_id);
        let state = inner.breakers.entry(key).or_insert(BreakerState::Closed {
            consecutive_failures: 0,
        });
        match state {
            BreakerState::Closed { .. } => Ok(()),
            BreakerState::Open { reopens_at } => match *reopens_at {
                Some(at) if now >= at => {
                    *state = BreakerState::HalfOpen {
                        trial_since: Some(now),
                    };
                    Ok(())
                }
                Some(at) => Err(at - now),
                None => Err(Duration::MAX),
            },
            BreakerState::HalfOpen { trial_since } => {
                let lease_up = match *trial_since {
                    None => true,
                    // The trial never reported (non-retryable error,
                    // cancel, panic): once its lease is up this caller
                    // becomes the new trial.
                    Some(since) => now.saturating_duration_since(since) >= policy.open_duration,
                };
                if lease_up {
                    *trial_since = Some(now);
                    Ok(())
                } else {
                    // Another caller holds the trial slot. Fail-fast with
                    // a near-zero reopens_in so the caller can retry once
                    // the trial result lands.
                    Err(TRIAL_BUSY_RETRY)
                }
            }
        }
    }

    /// Hand back a HalfOpen trial slot without an outcome: for a trial
    /// that ended in a way that says nothing about provider health (a
    /// 400, `BudgetExhausted`, `Cancelled`). The next `check` admits a new
    /// trial at once. No-op in any other state, and in windowed mode
    /// (whose probe lease expires on its own after one cooldown).
    ///
    /// Optional: a trial that reports nothing loses its slot anyway once
    /// `open_duration` has passed.
    pub fn release_trial(&self, provider: Provider, agent_id: &'static str) {
        if self.window.is_some() {
            return;
        }
        let mut inner = self.inner.lock().expect("circuit breaker poisoned");
        if inner.policy.is_disabled() {
            return;
        }
        if let Some(BreakerState::HalfOpen { trial_since }) =
            inner.breakers.get_mut(&(provider, agent_id))
        {
            *trial_since = None;
        }
    }

    /// Record a successful call: resets the failure counter to Closed,
    /// including HalfOpen → Closed on a successful trial.
    pub fn record_success(&self, provider: Provider, agent_id: &'static str) {
        if let Some(window) = &self.window {
            window.record_success(&provider);
            return;
        }
        let mut inner = self.inner.lock().expect("circuit breaker poisoned");
        if inner.policy.is_disabled() {
            return;
        }
        let key = (provider, agent_id);
        inner.breakers.insert(
            key,
            BreakerState::Closed {
                consecutive_failures: 0,
            },
        );
    }

    /// Record a failed call. Trips Closed → Open after `failure_threshold`
    /// consecutive failures; trips HalfOpen → Open immediately (the
    /// trial proved the upstream is still down).
    pub fn record_failure(&self, provider: Provider, agent_id: &'static str) {
        if let Some(window) = &self.window {
            window.record_failure(&provider);
            return;
        }
        let mut inner = self.inner.lock().expect("circuit breaker poisoned");
        let policy = inner.policy;
        if policy.is_disabled() {
            return;
        }
        let now = Instant::now();
        let key = (provider, agent_id);
        let state = inner.breakers.entry(key).or_insert(BreakerState::Closed {
            consecutive_failures: 0,
        });
        match state {
            BreakerState::Closed {
                consecutive_failures,
            } => {
                let next = consecutive_failures.saturating_add(1);
                if next >= policy.failure_threshold {
                    *state = open_from(now, policy.open_duration);
                } else {
                    *state = BreakerState::Closed {
                        consecutive_failures: next,
                    };
                }
            }
            BreakerState::HalfOpen { .. } => {
                *state = open_from(now, policy.open_duration);
            }
            BreakerState::Open { .. } => {
                // `check` would have returned Err, so this shouldn't
                // happen; guard defensively against caller bugs.
            }
        }
    }

    /// Active policy snapshot.
    pub fn policy(&self) -> BreakerPolicy {
        self.inner.lock().expect("circuit breaker poisoned").policy
    }

    /// Inspect the current state — primarily for tests.
    #[cfg(any(test, feature = "testing"))]
    pub fn is_open(&self, provider: Provider, agent_id: &'static str) -> bool {
        if let Some(window) = &self.window {
            return matches!(window.status(&provider), WindowStatus::Open { .. });
        }
        let inner = self.inner.lock().expect("circuit breaker poisoned");
        matches!(
            inner.breakers.get(&(provider, agent_id)),
            Some(BreakerState::Open { .. })
        )
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::with_default_policy()
    }
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("policy", &self.policy())
            .field("windowed", &self.window.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_breaker_permits_calls() {
        let b = CircuitBreaker::with_default_policy();
        assert!(b.check(Provider::OpenRouter, "agent_a").is_ok());
    }

    #[test]
    fn threshold_failures_open_the_breaker() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 3,
            open_duration: Duration::from_secs(60),
        });
        // First two failures: still Closed.
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        // Third failure trips.
        b.record_failure(Provider::OpenRouter, "a");
        let result = b.check(Provider::OpenRouter, "a");
        assert!(result.is_err());
        assert!(b.is_open(Provider::OpenRouter, "a"));
    }

    #[test]
    fn success_resets_failure_counter() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 3,
            open_duration: Duration::from_secs(60),
        });
        b.record_failure(Provider::OpenRouter, "a");
        b.record_failure(Provider::OpenRouter, "a");
        b.record_success(Provider::OpenRouter, "a");
        // 2 failures + 1 success → counter reset.
        b.record_failure(Provider::OpenRouter, "a");
        b.record_failure(Provider::OpenRouter, "a");
        // 2 more failures still below threshold.
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
    }

    #[test]
    fn separate_provider_agent_pairs_are_independent() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 2,
            open_duration: Duration::from_secs(60),
        });
        // Trip OpenRouter/a.
        b.record_failure(Provider::OpenRouter, "a");
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_err());
        // Different agent on same provider: not tripped.
        assert!(b.check(Provider::OpenRouter, "b").is_ok());
        // Same agent on different provider: not tripped.
        assert!(b.check(Provider::Anthropic, "a").is_ok());
    }

    #[tokio::test]
    async fn open_transitions_to_half_open_after_window() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 1,
            open_duration: Duration::from_millis(40),
        });
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_err());
        tokio::time::sleep(Duration::from_millis(60)).await;
        // Now the open window has elapsed; next check transitions to
        // HalfOpen and permits one call.
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        // Concurrent caller in HalfOpen with trial-in-flight: blocked.
        assert!(b.check(Provider::OpenRouter, "a").is_err());
    }

    #[tokio::test]
    async fn half_open_success_closes_breaker() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 1,
            open_duration: Duration::from_millis(20),
        });
        b.record_failure(Provider::OpenRouter, "a");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(b.check(Provider::OpenRouter, "a").is_ok()); // trial granted
        b.record_success(Provider::OpenRouter, "a");
        // Back to Closed; multiple calls permitted.
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
    }

    #[tokio::test]
    async fn half_open_failure_reopens_breaker() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 1,
            open_duration: Duration::from_millis(20),
        });
        b.record_failure(Provider::OpenRouter, "a");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(b.check(Provider::OpenRouter, "a").is_ok()); // trial granted
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.is_open(Provider::OpenRouter, "a"));
    }

    #[test]
    fn disabled_breaker_always_permits() {
        let b = CircuitBreaker::disabled();
        for _ in 0..100 {
            b.record_failure(Provider::OpenRouter, "a");
        }
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
    }

    #[test]
    fn windowed_mode_keys_by_provider_and_shares_with_the_standalone_breaker() {
        let window: WindowBreaker<Provider> = WindowBreaker::new(WindowPolicy::new(
            3,
            Duration::from_secs(60),
            Duration::from_secs(30),
        ));
        let b = CircuitBreaker::windowed(window.clone());
        assert!(b.window().is_some());
        assert_eq!(b.policy().failure_threshold, 3);
        assert_eq!(b.policy().open_duration, Duration::from_secs(30));
        // Two agents and one standalone caller fill the same window.
        b.record_failure(Provider::OpenRouter, "a");
        b.record_failure(Provider::OpenRouter, "b");
        assert!(b.check(Provider::OpenRouter, "c").is_ok());
        window.record_failure(&Provider::OpenRouter);
        assert!(b.is_open(Provider::OpenRouter, "anyone"));
        assert!(b.check(Provider::OpenRouter, "c").is_err());
        assert!(!window.allow(&Provider::OpenRouter));
        // Another provider is untouched.
        assert!(b.check(Provider::Anthropic, "a").is_ok());
        // A success from any agent closes it for all.
        b.record_success(Provider::OpenRouter, "b");
        assert!(window.allow(&Provider::OpenRouter));
    }

    #[tokio::test]
    async fn unreported_trial_is_replaced_after_open_duration() {
        // Regression: a trial that ended with a non-retryable error (a
        // 400, BudgetExhausted, Cancelled) recorded nothing, so the
        // HalfOpen slot stayed taken and every later call failed fast
        // forever.
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 1,
            open_duration: Duration::from_millis(40),
        });
        b.record_failure(Provider::OpenRouter, "a");
        tokio::time::sleep(Duration::from_millis(60)).await;
        // The trial is admitted, then returns a terminal error without
        // recording any outcome.
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        // While its lease runs, other callers still fail fast.
        assert!(b.check(Provider::OpenRouter, "a").is_err());
        tokio::time::sleep(Duration::from_millis(60)).await;
        // Lease expired: the next caller becomes the new trial ...
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        // ... and holds the slot exclusively.
        assert!(b.check(Provider::OpenRouter, "a").is_err());
        // Its success closes the breaker.
        b.record_success(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
    }

    #[tokio::test]
    async fn released_trial_frees_the_slot_immediately() {
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 1,
            open_duration: Duration::from_millis(40),
        });
        b.record_failure(Provider::OpenRouter, "a");
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        // Non-retryable outcome: says nothing about provider health.
        b.release_trial(Provider::OpenRouter, "a");
        assert!(!b.is_open(Provider::OpenRouter, "a"));
        // The next caller becomes the trial without waiting.
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
        assert!(b.check(Provider::OpenRouter, "a").is_err());
        // Releasing while Closed is a no-op.
        b.record_success(Provider::OpenRouter, "a");
        b.release_trial(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_ok());
    }

    #[test]
    fn huge_open_duration_does_not_overflow_instant() {
        // Regression: `now + Duration::MAX` panicked.
        let b = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 1,
            open_duration: Duration::MAX,
        });
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.is_open(Provider::OpenRouter, "a"));
        let reopens_in = b.check(Provider::OpenRouter, "a").unwrap_err();
        assert!(reopens_in > Duration::from_secs(60 * 60 * 24 * 365));
        // A second failure while open is still harmless.
        b.record_failure(Provider::OpenRouter, "a");
        assert!(b.check(Provider::OpenRouter, "a").is_err());
    }

    #[test]
    fn clone_shares_state() {
        let a = CircuitBreaker::new(BreakerPolicy {
            failure_threshold: 2,
            open_duration: Duration::from_secs(60),
        });
        let b = a.clone();
        a.record_failure(Provider::OpenRouter, "x");
        b.record_failure(Provider::OpenRouter, "x");
        assert!(a.check(Provider::OpenRouter, "x").is_err());
        assert!(b.check(Provider::OpenRouter, "x").is_err());
    }
}
