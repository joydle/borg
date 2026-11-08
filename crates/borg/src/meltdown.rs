//! Restart-storm detector — the OTP-style meltdown threshold, used by
//! the executor's retry loop.
//!
//! `RestartPolicy::on_failure(N)` caps retries per-call; this ledger
//! caps them *across* calls within a session. If the same agent class
//! restarts past `max_restarts` within `window`, the supervisor surfaces
//! [`BorgError::Meltdown`](crate::BorgError::Meltdown) rather than grind
//! budget on a broken dependency.
//!
//! ## Wiring
//!
//! Every [`crate::ExecCtx`] carries a cloneable [`MeltdownLedger`]; child
//! contexts share the same Arc, so restarts across siblings count
//! against one budget. The executor calls
//! [`MeltdownLedger::record_restart`] on each retry; a non-`None` return
//! surfaces the typed error.
//!
//! ## Defaults
//!
//! [`MeltdownPolicy::default_for_llm`] caps at 20 restarts / 60s: a real
//! outage trips within seconds (fan-out × per-call retries × short
//! window), while an agent restarting once every few minutes never does.
//!
//! ## Disabling
//!
//! [`MeltdownLedger::disabled`] always returns `None` from
//! `record_restart`. Use in tests that exercise
//! `RestartPolicy::on_failure(many)` without the meltdown net firing.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Policy for restart-storm detection. Borg's analog of OTP's
/// `max_restarts/max_seconds` supervisor parameter.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MeltdownPolicy {
    /// Restarts within `window` that trip the meltdown. A restart is one
    /// decision-to-retry in the executor's `RestartPolicy::OnFailure`
    /// branch, not a session-wide retry count.
    pub max_restarts: u32,
    /// Rolling window over which `max_restarts` is counted; older entries
    /// are pruned on every `record_restart`.
    pub window: Duration,
}

impl MeltdownPolicy {
    /// `max_restarts` within a rolling `window` trip the meltdown.
    #[must_use]
    pub const fn new(max_restarts: u32, window: Duration) -> Self {
        Self {
            max_restarts,
            window,
        }
    }
}

impl MeltdownPolicy {
    /// 20 restarts within 60 seconds: catches real provider outages
    /// (fan-out × retries within ~5s) while leaving a flapping agent room
    /// to recover.
    pub const fn default_for_llm() -> Self {
        Self {
            max_restarts: 20,
            window: Duration::from_secs(60),
        }
    }

    /// Sentinel: never trip. `record_restart` always returns `None`.
    /// For tests or stress paths needing unrestricted retries.
    pub const fn disabled() -> Self {
        Self {
            max_restarts: u32::MAX,
            window: Duration::ZERO,
        }
    }

    fn is_disabled(&self) -> bool {
        self.window.is_zero() || self.max_restarts == u32::MAX
    }
}

impl Default for MeltdownPolicy {
    fn default() -> Self {
        Self::default_for_llm()
    }
}

/// Cloneable ledger of per-agent restart timestamps. Inner state is
/// Arc'd so `ExecCtx::clone()` shares one ledger across the whole
/// composition tree.
#[derive(Clone)]
pub struct MeltdownLedger {
    inner: Arc<Mutex<LedgerInner>>,
}

struct LedgerInner {
    policy: MeltdownPolicy,
    per_agent: HashMap<&'static str, VecDeque<Instant>>,
}

impl MeltdownLedger {
    /// New ledger with the given policy.
    pub fn new(policy: MeltdownPolicy) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LedgerInner {
                policy,
                per_agent: HashMap::new(),
            })),
        }
    }

    /// Default policy ([`MeltdownPolicy::default_for_llm`]).
    pub fn with_default_policy() -> Self {
        Self::new(MeltdownPolicy::default_for_llm())
    }

    /// Meltdown disabled. For tests wanting unrestricted retries.
    pub fn disabled() -> Self {
        Self::new(MeltdownPolicy::disabled())
    }

    /// Record one restart for `agent_id`. Returns `Some(count)` when the
    /// in-window count reaches `policy.max_restarts` — the executor
    /// should then surface
    /// [`BorgError::Meltdown`](crate::BorgError::Meltdown) and stop
    /// retrying — else `None`. Prunes entries outside the window.
    pub fn record_restart(&self, agent_id: &'static str) -> Option<u32> {
        let mut inner = self.inner.lock().expect("meltdown ledger poisoned");
        let policy = inner.policy;
        if policy.is_disabled() {
            return None;
        }
        let now = Instant::now();
        let window = policy.window;
        let entries = inner.per_agent.entry(agent_id).or_default();
        while let Some(front) = entries.front() {
            if now.duration_since(*front) > window {
                entries.pop_front();
            } else {
                break;
            }
        }
        entries.push_back(now);
        let count = entries.len() as u32;
        if count >= policy.max_restarts {
            Some(count)
        } else {
            None
        }
    }

    /// Active policy (cloneable copy).
    pub fn policy(&self) -> MeltdownPolicy {
        self.inner.lock().expect("meltdown ledger poisoned").policy
    }

    /// Current in-window restart count for `agent_id`. For tests;
    /// production reads `record_restart`'s return value.
    #[cfg(any(test, feature = "testing"))]
    pub fn count(&self, agent_id: &'static str) -> u32 {
        let inner = self.inner.lock().expect("meltdown ledger poisoned");
        inner
            .per_agent
            .get(agent_id)
            .map(|q| q.len() as u32)
            .unwrap_or(0)
    }
}

impl Default for MeltdownLedger {
    fn default() -> Self {
        Self::with_default_policy()
    }
}

impl std::fmt::Debug for MeltdownLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let policy = self.policy();
        f.debug_struct("MeltdownLedger")
            .field("policy", &policy)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_threshold_returns_none() {
        let ledger = MeltdownLedger::new(MeltdownPolicy {
            max_restarts: 5,
            window: Duration::from_secs(60),
        });
        for _ in 0..4 {
            assert_eq!(ledger.record_restart("agent_a"), None);
        }
    }

    #[test]
    fn at_threshold_returns_count() {
        let ledger = MeltdownLedger::new(MeltdownPolicy {
            max_restarts: 3,
            window: Duration::from_secs(60),
        });
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(ledger.record_restart("agent_a"), Some(3));
    }

    #[test]
    fn separate_agents_separate_buckets() {
        let ledger = MeltdownLedger::new(MeltdownPolicy {
            max_restarts: 2,
            window: Duration::from_secs(60),
        });
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(ledger.record_restart("agent_b"), None);
        // Each agent has 1 entry; neither trips.
        assert_eq!(ledger.count("agent_a"), 1);
        assert_eq!(ledger.count("agent_b"), 1);
    }

    #[test]
    fn disabled_policy_never_trips() {
        let ledger = MeltdownLedger::disabled();
        for _ in 0..1000 {
            assert_eq!(ledger.record_restart("storm"), None);
        }
    }

    #[test]
    fn clone_shares_state() {
        let ledger = MeltdownLedger::new(MeltdownPolicy {
            max_restarts: 3,
            window: Duration::from_secs(60),
        });
        let twin = ledger.clone();
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(twin.record_restart("agent_a"), None);
        // Third restart on the original; the count includes the
        // twin's entry because the inner state is Arc-shared.
        assert_eq!(ledger.record_restart("agent_a"), Some(3));
    }

    #[tokio::test]
    async fn old_entries_outside_window_get_pruned() {
        let ledger = MeltdownLedger::new(MeltdownPolicy {
            max_restarts: 3,
            window: Duration::from_millis(50),
        });
        // Two early hits, then wait past the window, then two more.
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(ledger.record_restart("agent_a"), None);
        tokio::time::sleep(Duration::from_millis(80)).await;
        // First two are pruned on the next record.
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(ledger.count("agent_a"), 1);
        assert_eq!(ledger.record_restart("agent_a"), None);
        assert_eq!(ledger.count("agent_a"), 2);
    }
}
