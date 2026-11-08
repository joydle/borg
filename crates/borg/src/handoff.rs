//! Handoff state — how step N receives step N-1's output.
//!
//! Named keys are primary (`state: HashMap<String, Value>`); the audit
//! history (`Vec<NamedVerdict>`) is secondary. Callers address prior
//! outputs by meaning, not by positional index.

use crate::verdict::Verdict;
use serde_json::Value;
use std::collections::HashMap;

/// A typed slot in [`HandoffState`].
///
/// Declare a unit struct, impl this trait once, then read/write via
/// [`HandoffState::get_typed`] / [`HandoffState::set_typed`] without
/// restating the key or type at each call site:
///
/// ```
/// use borg::*;
/// use std::collections::HashMap;
///
/// struct WorkingFiles;
/// impl HandoffSlot for WorkingFiles {
///     type Value = HashMap<String, String>;
///     const KEY: &'static str = "working_files";
/// }
///
/// let mut h = HandoffState::new();
/// let mut files = HashMap::new();
/// files.insert("main.tsx".into(), "...".into());
/// h.set_typed::<WorkingFiles>(files);
///
/// let recovered = h.get_typed::<WorkingFiles>().unwrap();
/// assert_eq!(recovered.len(), 1);
/// ```
pub trait HandoffSlot {
    /// Type of the value stored under this slot.
    type Value: serde::Serialize + serde::de::DeserializeOwned;
    /// Key this slot lives under in [`HandoffState::state`]. Uniqueness
    /// is not enforced (no central registry), so use a stable,
    /// namespaced string (`"plan.tasks"`, not `"tasks"`).
    const KEY: &'static str;
}

/// Verdict produced by a named step, plus accounting metadata.
///
/// Lives in `HandoffState::history` as the audit trail. The body is
/// *also* exposed in `HandoffState::state` under the step's `output_key`
/// (when declared) so downstream steps read it without scanning history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct NamedVerdict {
    /// Composition-unique name of the step that produced this verdict.
    pub step_name: String,
    /// The verdict itself.
    pub verdict: Verdict,
    /// Wall-clock duration the agent's `invoke()` took.
    pub duration_ms: u64,
    /// Whether this verdict is a failure surfaced through
    /// `ErrorPolicy::CollectAll`. `false` for successes regardless of
    /// token-recognition status.
    pub is_error: bool,
}

impl NamedVerdict {
    /// Construct a successful named verdict.
    pub fn success(step_name: impl Into<String>, verdict: Verdict, duration_ms: u64) -> Self {
        Self {
            step_name: step_name.into(),
            verdict,
            duration_ms,
            is_error: false,
        }
    }

    /// Construct a failure named verdict (used by
    /// `ErrorPolicy::CollectAll`). The body carries the error message
    /// for downstream consumers to render.
    pub fn failure(
        step_name: impl Into<String>,
        error_message: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            step_name: step_name.into(),
            verdict: Verdict {
                token: "ERROR".to_string(),
                body: std::sync::Arc::from(error_message.into()),
                token_is_recognized: true,
                ..Verdict::default()
            },
            duration_ms,
            is_error: true,
        }
    }
}

/// Shared handoff state threaded through composition execution.
///
/// `Sequential` and `LoopUntil` mutate the parent's state in place so
/// each step sees prior outputs. `Parallel` branches see an immutable
/// fan-out-time snapshot and write back only on success; key uniqueness
/// across branches is enforced at build time, so those writes never
/// conflict.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HandoffState {
    /// Free-form key/value map. Step authors own the schema; borg passes
    /// JSON values through verbatim.
    pub state: HashMap<String, Value>,
    /// Audit trail of completed steps in execution order. Used by
    /// composition events, debug dumps, and `Verdict::token` branch
    /// predicates.
    pub history: Vec<NamedVerdict>,
}

/// Controls how much of [`HandoffState`] is captured into a snapshot at
/// persistence time.
///
/// The live `HandoffState` always keeps unbounded history; this bounds
/// only the slice that flows to a
/// [`crate::checkpoint::CheckpointStore`], so per-save wire bytes don't
/// grow with turn count. Opt in via
/// [`crate::ExecCtx::with_handoff_persistence`].
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct HandoffPersistenceConfig {
    /// Max entries from the tail of `HandoffState::history` to capture.
    /// `0` (default) means unlimited. The `state` map is always captured
    /// in full — only the verdict tail is bounded.
    pub max_persisted_history: usize,
}

impl HandoffState {
    /// State key the executor writes after a step whose verdict carries
    /// `handoff_to = Some(...)`; read it via [`Self::pending_handoff`].
    /// The `__borg_` prefix avoids collision with caller output_keys —
    /// don't write it directly.
    pub const HANDOFF_KEY: &'static str = "__borg_handoff__";

    pub fn new() -> Self {
        Self::default()
    }

    /// Read the most recently-set handoff target. `None` if no step has
    /// emitted `handoff_to = Some(...)`, or its value failed to
    /// deserialize. Natural shape for a `Composition::Branch` selector
    /// dispatching on agent-declared handoff:
    ///
    /// ```ignore
    /// fn route(state: &HandoffState) -> BranchKey {
    ///     state.pending_handoff()
    ///         .map(BranchKey::from)
    ///         .unwrap_or_else(|| "default".into())
    /// }
    /// ```
    pub fn pending_handoff(&self) -> Option<String> {
        self.get_as(Self::HANDOFF_KEY)
    }

    /// Read a JSON value previously written under `key` (e.g. by a
    /// step that declared `output_key = Some("foo")`).
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Read a previously-written value, deserialized via serde. `None`
    /// if the key is absent or doesn't deserialize.
    pub fn get_as<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.state
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// A step's typed output: the value under `key` (a step's
    /// `output_key` writes its verdict body there, as a string) parsed as
    /// JSON into `T`. A value that is already structured (written with
    /// [`Self::set`]), or a body that is not JSON, is read as
    /// [`Self::get_as`] reads it. `None` if the
    /// key is absent or does not parse as a `T`.
    ///
    /// ```
    /// # use borg::HandoffState;
    /// #[derive(serde::Deserialize)]
    /// struct Review { approved: bool }
    /// let mut h = HandoffState::new();
    /// h.set("review", r#"{"approved": false}"#); // a verdict body
    /// assert_eq!(h.output_as::<Review>("review").map(|r| r.approved), Some(false));
    /// ```
    pub fn output_as<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        let value = self.state.get(key)?;
        if let Value::String(body) = value {
            if let Ok(t) = serde_json::from_str(body) {
                return Some(t);
            }
        }
        serde_json::from_value(value.clone()).ok()
    }

    /// Write a value under `key` (last writer wins). Serialization
    /// failures are logged via [`tracing::warn!`] and swallowed; use
    /// [`Self::try_set`] to handle them explicitly.
    pub fn set<T: serde::Serialize>(&mut self, key: impl Into<String>, value: T) {
        let key: String = key.into();
        if let Err(e) = self.try_set(&key, value) {
            tracing::warn!(
                key = %key,
                error = %e,
                "HandoffState::set: serialization failed, value not stored"
            );
        }
    }

    /// Like [`Self::set`] but returns serialization failures instead of
    /// dropping them — e.g. for values with non-finite floats or
    /// non-string map keys.
    pub fn try_set<T: serde::Serialize>(
        &mut self,
        key: impl Into<String>,
        value: T,
    ) -> Result<(), serde_json::Error> {
        let v = serde_json::to_value(value)?;
        self.state.insert(key.into(), v);
        Ok(())
    }

    /// Read a typed slot. The slot's `KEY` and `Value` live on its
    /// [`HandoffSlot`] impl, so renaming a key updates one place.
    pub fn get_typed<S: HandoffSlot>(&self) -> Option<S::Value> {
        self.get_as(S::KEY)
    }

    /// Write a typed slot. Mirrors [`Self::get_typed`] for symmetry.
    pub fn set_typed<S: HandoffSlot>(&mut self, value: S::Value) {
        self.set(S::KEY, value);
    }

    /// Like [`Self::set_typed`] but surfaces serialization failures.
    /// Mirrors [`Self::try_set`] for the typed-slot API.
    pub fn try_set_typed<S: HandoffSlot>(
        &mut self,
        value: S::Value,
    ) -> Result<(), serde_json::Error> {
        self.try_set(S::KEY, value)
    }

    /// The most recent named verdict, regardless of name. Used by
    /// `LoopUntil` predicates and as the "previous step" shorthand
    /// inside `Sequential`.
    pub fn previous(&self) -> Option<&NamedVerdict> {
        self.history.last()
    }

    /// Build a [`HandoffState`] for persistence, honoring `cfg`'s
    /// history-retention policy. The `state` map is captured in full;
    /// `history` is truncated to the last `cfg.max_persisted_history`
    /// entries (`0` means no bound). The live state is untouched.
    #[must_use]
    pub fn to_snapshot(&self, cfg: &HandoffPersistenceConfig) -> Self {
        if cfg.max_persisted_history == 0 || self.history.len() <= cfg.max_persisted_history {
            return self.clone();
        }
        let start = self.history.len() - cfg.max_persisted_history;
        Self {
            state: self.state.clone(),
            history: self.history[start..].to_vec(),
        }
    }

    /// Find a verdict by step name, so a later step can reference a much
    /// earlier one without a positional index.
    pub fn find(&self, step_name: &str) -> Option<&NamedVerdict> {
        self.history.iter().find(|v| v.step_name == step_name)
    }

    /// Append a step's outcome; the executor calls this once per step
    /// completion.
    pub fn record(&mut self, named: NamedVerdict) {
        self.history.push(named);
    }

    /// Snapshot the state for handing to a parallel branch. The branch
    /// will see what was visible at fan-out time but cannot mutate the
    /// parent until it returns and the executor merges its outputs.
    #[must_use]
    pub fn snapshot(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_returns_set_value() {
        let mut h = HandoffState::new();
        h.set("foo", "bar");
        assert_eq!(h.get("foo").unwrap(), &Value::String("bar".to_string()));
    }

    #[test]
    fn to_snapshot_default_is_full_history() {
        let mut h = HandoffState::new();
        h.set("k", "v");
        for i in 0..10u32 {
            h.record(NamedVerdict::success(
                format!("step-{i}"),
                Verdict::default(),
                0,
            ));
        }
        let snap = h.to_snapshot(&HandoffPersistenceConfig::default());
        assert_eq!(snap.history.len(), 10, "default cfg = unlimited");
        assert_eq!(snap.state.len(), 1);
    }

    #[test]
    fn to_snapshot_bounds_history_tail() {
        let mut h = HandoffState::new();
        h.set("k", "v");
        for i in 0..10u32 {
            h.record(NamedVerdict::success(
                format!("step-{i}"),
                Verdict::default(),
                0,
            ));
        }
        let cfg = HandoffPersistenceConfig {
            max_persisted_history: 3,
        };
        let snap = h.to_snapshot(&cfg);
        assert_eq!(snap.history.len(), 3);
        assert_eq!(snap.history[0].step_name, "step-7");
        assert_eq!(snap.history[2].step_name, "step-9");
        assert_eq!(snap.state.len(), 1);
    }

    #[test]
    fn to_snapshot_keeps_full_history_when_under_bound() {
        let mut h = HandoffState::new();
        for i in 0..5u32 {
            h.record(NamedVerdict::success(
                format!("step-{i}"),
                Verdict::default(),
                0,
            ));
        }
        let cfg = HandoffPersistenceConfig {
            max_persisted_history: 100,
        };
        let snap = h.to_snapshot(&cfg);
        assert_eq!(snap.history.len(), 5);
    }

    #[test]
    fn get_as_deserializes_typed_value() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct Plan {
            tasks: u32,
            theme: String,
        }
        let mut h = HandoffState::new();
        h.set(
            "plan",
            Plan {
                tasks: 5,
                theme: "noir".into(),
            },
        );
        let p: Plan = h.get_as("plan").unwrap();
        assert_eq!(p.tasks, 5);
        assert_eq!(p.theme, "noir");
    }

    #[test]
    fn pending_handoff_is_none_until_set() {
        let h = HandoffState::new();
        assert_eq!(h.pending_handoff(), None);
    }

    #[test]
    fn pending_handoff_reads_well_known_key() {
        let mut h = HandoffState::new();
        h.set(HandoffState::HANDOFF_KEY, "reviewer".to_string());
        assert_eq!(h.pending_handoff().as_deref(), Some("reviewer"));
    }

    #[test]
    fn previous_returns_last_recorded_verdict() {
        let mut h = HandoffState::new();
        let a = NamedVerdict::success("a", Verdict::with_token("PASS", "", 0, 0), 10);
        let b = NamedVerdict::success("b", Verdict::with_token("FAIL", "", 0, 0), 20);
        h.record(a);
        h.record(b);
        assert_eq!(h.previous().unwrap().step_name, "b");
    }

    #[test]
    fn find_by_name_returns_correct_verdict() {
        let mut h = HandoffState::new();
        h.record(NamedVerdict::success("a", Verdict::empty(), 1));
        h.record(NamedVerdict::success("b", Verdict::empty(), 2));
        h.record(NamedVerdict::success("c", Verdict::empty(), 3));
        assert_eq!(h.find("b").unwrap().duration_ms, 2);
        assert!(h.find("nonexistent").is_none());
    }

    #[test]
    fn snapshot_is_independent_of_parent() {
        let mut parent = HandoffState::new();
        parent.set("k", "before");
        let snap = parent.snapshot();
        parent.set("k", "after");
        assert_eq!(snap.get("k").unwrap(), &Value::String("before".into()));
        assert_eq!(parent.get("k").unwrap(), &Value::String("after".into()));
    }

    #[test]
    fn failure_named_verdict_carries_error_marker() {
        let nv = NamedVerdict::failure("flaky", "503 Service Unavailable", 100);
        assert!(nv.is_error);
        assert_eq!(nv.verdict.token, "ERROR");
        assert_eq!(&*nv.verdict.body, "503 Service Unavailable");
    }

    #[test]
    fn missing_key_returns_none() {
        let h = HandoffState::new();
        assert!(h.get("nope").is_none());
        assert!(h.get_as::<String>("nope").is_none());
    }

    #[test]
    fn typed_slot_round_trip() {
        struct Plan;
        impl HandoffSlot for Plan {
            type Value = Vec<String>;
            const KEY: &'static str = "plan.tasks";
        }

        let mut h = HandoffState::new();
        h.set_typed::<Plan>(vec!["sword".into(), "shield".into()]);

        let got: Vec<String> = h.get_typed::<Plan>().expect("slot present");
        assert_eq!(got, vec!["sword".to_string(), "shield".to_string()]);

        // Typed slot writes the declared key, so free-form `get` sees it.
        assert!(h.get("plan.tasks").is_some());
    }

    #[test]
    fn typed_slot_missing_returns_none() {
        struct Missing;
        impl HandoffSlot for Missing {
            type Value = String;
            const KEY: &'static str = "missing";
        }
        let h = HandoffState::new();
        assert!(h.get_typed::<Missing>().is_none());
    }
}
