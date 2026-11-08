//! Sliding-window circuit breaker: trips after `failure_threshold`
//! failures inside a rolling `window`, per key, shared by every clone.
//!
//! The consecutive breaker ([`CircuitBreaker`](super::CircuitBreaker)'s
//! default) forgets a run of failures the moment one call succeeds, and
//! keys on `(provider, agent)`. A process that makes many kinds of calls
//! to one upstream (a main loop, helpers, sub-agents) usually wants the
//! other shape: count failures from EVERY caller against one key (say
//! the provider) over a time window, and stop calling that upstream
//! while it is down. That is this breaker.
//!
//! - **Shared.** State lives behind an `Arc`; clone the breaker (or keep
//!   one in a `static`) and every clone sees the same windows.
//! - **Standalone.** [`WindowBreaker::check`] / [`WindowBreaker::allow`]
//!   before a call, [`WindowBreaker::record_success`] /
//!   [`WindowBreaker::record_failure`] after; nothing here needs an
//!   executor. [`CircuitBreaker::windowed`](super::CircuitBreaker::windowed)
//!   hands the same breaker to the executor, keyed by provider.
//! - **Caller decides what counts.** Only call `record_failure` for an
//!   outage signal (a timeout, a throttle, a 5xx);
//!   [`counts_toward_breaker`] is the HTTP-status rule the executor's
//!   retry classifier uses.
//!
//! ## State machine (per key)
//!
//! ```text
//!   Closed { failures in the last `window` }
//!      │  failure → prune the ones older than `window`, push now
//!      │  if failures >= threshold ─────────▶ Open { reopens_at: now + cooldown }
//!      │                                          │  failures while Open are ignored
//!      │  success (from ANY state) → Closed, empty│  check after reopens_at:
//!      │ ◀──────── Recovery::Close ───────────────┤   Close    → Closed, empty, every caller passes
//!      │                                          │   HalfOpen → HalfOpen, ONE probe passes
//!      │ ◀──────── probe succeeds ── HalfOpen ◀───┘
//!      │                               │ probe fails → Open again (fresh cooldown)
//!      │                               │ other callers fail fast; a probe that never
//!      │                               │ reports is replaced after one cooldown
//! ```
//!
//! [`Recovery::HalfOpen`] (the default) is the Hystrix shape the
//! consecutive breaker uses; [`Recovery::Close`] reopens the gate to
//! everyone and lets the window decide again, which is what you want
//! when the callers share an upstream but not a fate (one probe from one
//! caller says little about the rest).

use crate::retry_classify::RETRYABLE_HTTP_STATUS;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a caller turned away during a half-open probe is told to
/// wait: the probe's verdict usually lands well inside it.
pub const PROBE_BUSY_RETRY: Duration = Duration::from_millis(50);

/// The HTTP rule for what is an outage: no status at all (a transport
/// failure or a timeout before any reply) or a status in
/// [`RETRYABLE_HTTP_STATUS`] (408, 429, 500, 502, 503, 504). Caller
/// mistakes (other 4xx) and statuses outside that list do not count.
pub fn counts_toward_breaker(status: Option<u16>) -> bool {
    status.is_none_or(|s| RETRYABLE_HTTP_STATUS.contains(&s))
}

/// What happens once an open breaker's cooldown is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Recovery {
    /// Admit ONE probe; its success closes the breaker, its failure
    /// reopens it for another cooldown. Everyone else fails fast with
    /// [`PROBE_BUSY_RETRY`] meanwhile.
    #[default]
    HalfOpen,
    /// Close with an empty window: every caller passes, and it takes
    /// `failure_threshold` fresh failures to open it again.
    Close,
}

/// Sliding-window breaker policy. Build it with [`WindowPolicy::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct WindowPolicy {
    /// Failures inside `window` that open the breaker (at least 1).
    pub failure_threshold: u32,
    /// How far back a failure still counts.
    pub window: Duration,
    /// How long the breaker stays open before [`Self::recovery`] applies.
    pub cooldown: Duration,
    /// What the first check after the cooldown does.
    pub recovery: Recovery,
}

impl WindowPolicy {
    /// `failure_threshold` failures within `window` open it for
    /// `cooldown`, then one half-open probe ([`Recovery::HalfOpen`]).
    pub const fn new(failure_threshold: u32, window: Duration, cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            window,
            cooldown,
            recovery: Recovery::HalfOpen,
        }
    }

    /// Same policy, another [`Recovery`].
    #[must_use]
    pub const fn with_recovery(mut self, recovery: Recovery) -> Self {
        self.recovery = recovery;
        self
    }

    /// 10 failures in 5 minutes open it for 60 seconds: a process-wide
    /// gate on one upstream, slow to trip on a few bad calls.
    pub const fn default_for_llm() -> Self {
        Self::new(10, Duration::from_secs(300), Duration::from_secs(60))
    }

    /// Never trips: every check passes and every record is a no-op.
    pub const fn disabled() -> Self {
        Self::new(u32::MAX, Duration::ZERO, Duration::ZERO)
    }

    fn is_disabled(&self) -> bool {
        self.failure_threshold == u32::MAX || self.cooldown.is_zero()
    }

    fn threshold(&self) -> usize {
        self.failure_threshold.max(1) as usize
    }
}

impl Default for WindowPolicy {
    fn default() -> Self {
        Self::default_for_llm()
    }
}

/// Why a check let the call through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Admission {
    /// The breaker is closed.
    Closed,
    /// This check ended a cooldown under [`Recovery::Close`]: the breaker
    /// is closed again with an empty window. Once per cooldown.
    Recovered,
    /// This call is the half-open probe ([`Recovery::HalfOpen`]); record
    /// its outcome.
    Probe,
}

/// A `record_failure` that opened the breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Trip {
    /// Failures inside the window when it opened (1 for a failed probe).
    pub failures: u32,
    /// The failure was the half-open probe's.
    pub after_probe: bool,
    /// How long it stays open.
    pub reopens_in: Duration,
}

/// A key's state, read without changing it (gauges, admin pages, tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowStatus {
    /// Closed, with this many failures still inside the window.
    Closed {
        /// Failures younger than the window.
        failures: u32,
    },
    /// Open; the cooldown has this long to go (zero = the next check
    /// recovers).
    Open {
        /// Time left in the cooldown.
        reopens_in: Duration,
    },
    /// A half-open probe is in flight.
    HalfOpen,
}

#[derive(Debug)]
enum State {
    Closed { failures: VecDeque<Instant> },
    Open { reopens_at: Instant },
    HalfOpen { probe_since: Instant },
}

impl State {
    fn closed() -> Self {
        State::Closed {
            failures: VecDeque::new(),
        }
    }
}

struct Inner<K> {
    policy: WindowPolicy,
    keys: HashMap<K, State>,
}

/// Sliding-window circuit breaker keyed by `K`; clones share state.
///
/// ```
/// use borg::circuit_breaker::{WindowBreaker, WindowPolicy, Recovery};
/// use std::time::Duration;
///
/// let breaker: WindowBreaker<&'static str> = WindowBreaker::new(
///     WindowPolicy::new(3, Duration::from_secs(60), Duration::from_secs(30))
///         .with_recovery(Recovery::Close),
/// );
/// for _ in 0..3 {
///     breaker.record_failure(&"upstream");
/// }
/// assert!(!breaker.allow(&"upstream"));
/// assert!(breaker.allow(&"another"));
/// ```
pub struct WindowBreaker<K> {
    inner: Arc<Mutex<Inner<K>>>,
}

impl<K> Clone for WindowBreaker<K> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K: Eq + Hash + Clone> WindowBreaker<K> {
    /// A breaker with no keys yet (every key starts closed).
    pub fn new(policy: WindowPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                policy,
                keys: HashMap::new(),
            })),
        }
    }

    /// Disabled: every check passes, every record is a no-op.
    pub fn disabled() -> Self {
        Self::new(WindowPolicy::disabled())
    }

    /// May a call to `key` go out? `Ok` says why it may; `Err` is how
    /// long until it might (the cooldown left, or [`PROBE_BUSY_RETRY`]
    /// while another caller holds the half-open probe).
    pub fn check(&self, key: &K) -> Result<Admission, Duration> {
        self.check_at(key, Instant::now())
    }

    /// [`Self::check`] as a yes/no.
    pub fn allow(&self, key: &K) -> bool {
        self.check(key).is_ok()
    }

    /// A call to `key` succeeded: closed with an empty window, whatever
    /// the state. Returns `true` when that closed an open or half-open
    /// breaker.
    pub fn record_success(&self, key: &K) -> bool {
        let mut inner = self.inner.lock();
        if inner.policy.is_disabled() {
            return false;
        }
        match inner.keys.get_mut(key) {
            None => false,
            Some(state) => {
                let was_open = !matches!(state, State::Closed { .. });
                *state = State::closed();
                was_open
            }
        }
    }

    /// A call to `key` failed in a way that counts. `Some` when this
    /// failure opened the breaker; failures while it is open are ignored.
    pub fn record_failure(&self, key: &K) -> Option<Trip> {
        self.record_failure_at(key, Instant::now())
    }

    /// `key`'s state, without changing it.
    pub fn status(&self, key: &K) -> WindowStatus {
        self.status_at(key, Instant::now())
    }

    /// Forget every key (all closed and empty).
    pub fn reset(&self) {
        self.inner.lock().keys.clear();
    }

    /// Forget one key.
    pub fn reset_key(&self, key: &K) {
        self.inner.lock().keys.remove(key);
    }

    /// The active policy.
    pub fn policy(&self) -> WindowPolicy {
        self.inner.lock().policy
    }

    /// Replace the policy for every later decision. A breaker already
    /// open keeps the reopen time it was given.
    pub fn set_policy(&self, policy: WindowPolicy) {
        self.inner.lock().policy = policy;
    }

    fn check_at(&self, key: &K, now: Instant) -> Result<Admission, Duration> {
        let mut inner = self.inner.lock();
        let policy = inner.policy;
        if policy.is_disabled() {
            return Ok(Admission::Closed);
        }
        let Some(state) = inner.keys.get_mut(key) else {
            return Ok(Admission::Closed);
        };
        match state {
            State::Closed { .. } => Ok(Admission::Closed),
            State::Open { reopens_at } => {
                if now < *reopens_at {
                    return Err(*reopens_at - now);
                }
                match policy.recovery {
                    Recovery::Close => {
                        *state = State::closed();
                        Ok(Admission::Recovered)
                    }
                    Recovery::HalfOpen => {
                        *state = State::HalfOpen { probe_since: now };
                        Ok(Admission::Probe)
                    }
                }
            }
            State::HalfOpen { probe_since } => {
                // A probe that never reported (cancelled, panicked) must
                // not wedge the key: after one cooldown another caller
                // becomes the probe.
                if now.saturating_duration_since(*probe_since) >= policy.cooldown {
                    *probe_since = now;
                    Ok(Admission::Probe)
                } else {
                    Err(PROBE_BUSY_RETRY)
                }
            }
        }
    }

    fn record_failure_at(&self, key: &K, now: Instant) -> Option<Trip> {
        let mut inner = self.inner.lock();
        let policy = inner.policy;
        if policy.is_disabled() {
            return None;
        }
        let state = inner.keys.entry(key.clone()).or_insert_with(State::closed);
        match state {
            State::Closed { failures } => {
                while failures
                    .front()
                    .is_some_and(|&t| now.saturating_duration_since(t) > policy.window)
                {
                    failures.pop_front();
                }
                failures.push_back(now);
                if failures.len() < policy.threshold() {
                    return None;
                }
                let count = failures.len() as u32;
                *state = State::Open {
                    reopens_at: reopen_deadline(now, policy.cooldown),
                };
                Some(Trip {
                    failures: count,
                    after_probe: false,
                    reopens_in: policy.cooldown,
                })
            }
            State::HalfOpen { .. } => {
                *state = State::Open {
                    reopens_at: reopen_deadline(now, policy.cooldown),
                };
                Some(Trip {
                    failures: 1,
                    after_probe: true,
                    reopens_in: policy.cooldown,
                })
            }
            State::Open { .. } => None,
        }
    }

    fn status_at(&self, key: &K, now: Instant) -> WindowStatus {
        let inner = self.inner.lock();
        let window = inner.policy.window;
        match inner.keys.get(key) {
            None => WindowStatus::Closed { failures: 0 },
            Some(State::Closed { failures }) => WindowStatus::Closed {
                failures: failures
                    .iter()
                    .filter(|&&t| now.saturating_duration_since(t) <= window)
                    .count() as u32,
            },
            Some(State::Open { reopens_at }) => WindowStatus::Open {
                reopens_in: reopens_at.saturating_duration_since(now),
            },
            Some(State::HalfOpen { .. }) => WindowStatus::HalfOpen,
        }
    }
}

impl<K> std::fmt::Debug for WindowBreaker<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowBreaker")
            .field("policy", &self.inner.lock().policy)
            .finish_non_exhaustive()
    }
}

/// `now + cooldown`, saturating to a far-future instant instead of panicking
/// when a caller configures a cooldown near `Duration::MAX`.
fn reopen_deadline(now: Instant, cooldown: Duration) -> Instant {
    const FAR: Duration = Duration::from_secs(60 * 60 * 24 * 365 * 100);
    now.checked_add(cooldown)
        .or_else(|| now.checked_add(FAR))
        .unwrap_or(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    fn breaker(
        threshold: u32,
        window_s: u64,
        cooldown_s: u64,
        recovery: Recovery,
    ) -> WindowBreaker<u8> {
        WindowBreaker::new(
            WindowPolicy::new(threshold, S * window_s as u32, S * cooldown_s as u32)
                .with_recovery(recovery),
        )
    }

    #[test]
    fn opens_at_the_threshold_inside_the_window() {
        let b = breaker(3, 60, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        assert_eq!(b.record_failure_at(&1, t0), None);
        assert_eq!(b.record_failure_at(&1, t0 + S), None);
        assert_eq!(b.check_at(&1, t0 + S), Ok(Admission::Closed));
        let trip = b
            .record_failure_at(&1, t0 + 2 * S)
            .expect("third failure trips");
        assert_eq!(trip.failures, 3);
        assert!(!trip.after_probe);
        assert_eq!(trip.reopens_in, 30 * S);
        assert_eq!(b.check_at(&1, t0 + 2 * S), Err(30 * S));
        assert_eq!(b.check_at(&1, t0 + 12 * S), Err(20 * S));
        // Another key is untouched.
        assert_eq!(b.check_at(&2, t0 + 2 * S), Ok(Admission::Closed));
    }

    #[test]
    fn failures_older_than_the_window_roll_out() {
        let b = breaker(3, 10, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        b.record_failure_at(&1, t0);
        b.record_failure_at(&1, t0 + 5 * S);
        // t0 is 11 s old by now: out. Two in the window, still closed.
        assert_eq!(b.record_failure_at(&1, t0 + 11 * S), None);
        assert_eq!(
            b.status_at(&1, t0 + 11 * S),
            WindowStatus::Closed { failures: 2 }
        );
        // Exactly `window` old still counts (inclusive edge).
        assert!(
            b.record_failure_at(&1, t0 + 15 * S).is_some(),
            "t0+5 is exactly 10 s old"
        );
    }

    #[test]
    fn status_counts_only_the_live_window() {
        let b = breaker(5, 10, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        b.record_failure_at(&1, t0);
        b.record_failure_at(&1, t0 + 3 * S);
        assert_eq!(
            b.status_at(&1, t0 + 5 * S),
            WindowStatus::Closed { failures: 2 }
        );
        assert_eq!(
            b.status_at(&1, t0 + 12 * S),
            WindowStatus::Closed { failures: 1 }
        );
        assert_eq!(b.status_at(&9, t0), WindowStatus::Closed { failures: 0 });
    }

    #[test]
    fn a_success_empties_the_window_and_closes_an_open_breaker() {
        let b = breaker(2, 60, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        b.record_failure_at(&1, t0);
        assert!(!b.record_success(&1), "closed → closed is not a recovery");
        assert_eq!(b.record_failure_at(&1, t0), None, "the window was emptied");
        assert!(b.record_failure_at(&1, t0).is_some());
        assert!(b.check_at(&1, t0).is_err());
        assert!(
            b.record_success(&1),
            "a late success closes an open breaker"
        );
        assert_eq!(b.check_at(&1, t0), Ok(Admission::Closed));
    }

    #[test]
    fn failures_while_open_are_ignored() {
        let b = breaker(2, 60, 30, Recovery::Close);
        let t0 = Instant::now();
        b.record_failure_at(&1, t0);
        b.record_failure_at(&1, t0);
        for i in 0..10 {
            assert_eq!(b.record_failure_at(&1, t0 + S * i), None);
        }
        // The cooldown still runs from the trip.
        assert_eq!(b.check_at(&1, t0 + 30 * S), Ok(Admission::Recovered));
    }

    #[test]
    fn half_open_admits_one_probe_and_its_success_closes() {
        let b = breaker(1, 60, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        b.record_failure_at(&1, t0);
        let after = t0 + 30 * S;
        assert_eq!(b.check_at(&1, after), Ok(Admission::Probe));
        assert_eq!(b.check_at(&1, after), Err(PROBE_BUSY_RETRY));
        assert_eq!(b.status_at(&1, after), WindowStatus::HalfOpen);
        assert!(b.record_success(&1));
        assert_eq!(b.check_at(&1, after), Ok(Admission::Closed));
        assert_eq!(b.check_at(&1, after), Ok(Admission::Closed));
    }

    #[test]
    fn a_failed_probe_reopens_for_a_fresh_cooldown() {
        let b = breaker(3, 60, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record_failure_at(&1, t0);
        }
        let probe_at = t0 + 31 * S;
        assert_eq!(b.check_at(&1, probe_at), Ok(Admission::Probe));
        let trip = b
            .record_failure_at(&1, probe_at)
            .expect("failed probe reopens");
        assert!(trip.after_probe);
        assert_eq!(trip.failures, 1);
        assert_eq!(b.check_at(&1, probe_at + S), Err(29 * S));
    }

    #[test]
    fn a_probe_that_never_reports_is_replaced_after_a_cooldown() {
        let b = breaker(1, 60, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        b.record_failure_at(&1, t0);
        assert_eq!(b.check_at(&1, t0 + 30 * S), Ok(Admission::Probe));
        assert_eq!(b.check_at(&1, t0 + 59 * S), Err(PROBE_BUSY_RETRY));
        assert_eq!(b.check_at(&1, t0 + 60 * S), Ok(Admission::Probe));
    }

    #[test]
    fn close_recovery_lets_everyone_through_and_needs_a_fresh_window_to_reopen() {
        let b = breaker(3, 60, 30, Recovery::Close);
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record_failure_at(&1, t0);
        }
        let after = t0 + 30 * S;
        assert_eq!(b.check_at(&1, after), Ok(Admission::Recovered));
        for _ in 0..5 {
            assert_eq!(b.check_at(&1, after), Ok(Admission::Closed));
        }
        assert_eq!(b.record_failure_at(&1, after), None);
        assert_eq!(b.record_failure_at(&1, after), None);
        assert!(b.record_failure_at(&1, after).is_some());
    }

    #[test]
    fn real_clock_round_trip() {
        let b: WindowBreaker<()> = WindowBreaker::new(
            WindowPolicy::new(2, Duration::from_millis(500), Duration::from_millis(40))
                .with_recovery(Recovery::HalfOpen),
        );
        b.record_failure(&());
        assert!(b.record_failure(&()).is_some());
        assert!(!b.allow(&()));
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(b.check(&()), Ok(Admission::Probe));
        assert!(b.record_success(&()));
        assert!(b.allow(&()));
    }

    #[test]
    fn clones_share_state_and_concurrent_failures_trip_once() {
        let b: WindowBreaker<&'static str> = WindowBreaker::new(WindowPolicy::new(
            10,
            Duration::from_secs(300),
            Duration::from_secs(60),
        ));
        let trips = std::sync::atomic::AtomicU32::new(0);
        std::thread::scope(|s| {
            for _ in 0..32 {
                let b = b.clone();
                let trips = &trips;
                s.spawn(move || {
                    if let Some(trip) = b.record_failure(&"openrouter") {
                        assert_eq!(trip.failures, 10);
                        trips.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(trips.into_inner(), 1, "exactly one failure opens it");
        assert!(!b.allow(&"openrouter"));
    }

    #[test]
    fn concurrent_checks_in_half_open_grant_one_probe() {
        let b: WindowBreaker<()> = WindowBreaker::new(WindowPolicy::new(
            1,
            Duration::from_secs(60),
            Duration::from_millis(20),
        ));
        b.record_failure(&());
        std::thread::sleep(Duration::from_millis(40));
        let probes = std::sync::atomic::AtomicU32::new(0);
        std::thread::scope(|s| {
            for _ in 0..16 {
                let b = b.clone();
                let probes = &probes;
                s.spawn(move || {
                    if b.check(&()) == Ok(Admission::Probe) {
                        probes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(probes.into_inner(), 1);
    }

    #[test]
    fn disabled_never_trips() {
        let b: WindowBreaker<()> = WindowBreaker::disabled();
        for _ in 0..1_000 {
            assert_eq!(b.record_failure(&()), None);
        }
        assert_eq!(b.check(&()), Ok(Admission::Closed));
    }

    #[test]
    fn a_zero_threshold_means_one() {
        let b = breaker(0, 60, 30, Recovery::HalfOpen);
        let t0 = Instant::now();
        assert!(b.record_failure_at(&1, t0).is_some());
    }

    #[test]
    fn set_policy_and_reset() {
        let b = breaker(5, 60, 30, Recovery::HalfOpen);
        b.set_policy(WindowPolicy::new(1, S * 60, S * 30));
        assert!(b.record_failure(&1).is_some());
        assert!(!b.allow(&1));
        b.reset_key(&1);
        assert!(b.allow(&1));
        b.record_failure(&1);
        b.reset();
        assert!(b.allow(&1));
        assert_eq!(b.policy().failure_threshold, 1);
    }

    #[test]
    fn the_outage_rule_is_the_retryable_status_list() {
        assert!(counts_toward_breaker(None));
        for s in [408, 429, 500, 502, 503, 504] {
            assert!(counts_toward_breaker(Some(s)), "{s}");
        }
        for s in [200, 400, 401, 402, 403, 404, 409, 422, 501, 505, 520, 529] {
            assert!(!counts_toward_breaker(Some(s)), "{s}");
        }
    }

    #[test]
    fn huge_cooldown_does_not_overflow_instant() {
        let breaker: WindowBreaker<&str> =
            WindowBreaker::new(WindowPolicy::new(1, Duration::from_secs(60), Duration::MAX));
        assert!(
            breaker.record_failure(&"k").is_some(),
            "one failure trips at threshold 1"
        );
        assert!(
            breaker.check(&"k").is_err(),
            "stays open instead of panicking"
        );
    }
}
