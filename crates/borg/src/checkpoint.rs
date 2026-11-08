//! Composition checkpoint & resume — survive crashes mid-pipeline.
//!
//! Pairs with [`crate::execute_resumable`]. Borg saves
//! composition-step boundaries, not per-event state; the caller owns
//! event-level durability.
//!
//! ## What gets saved
//!
//! - The mutable [`crate::HandoffState`] (state + history).
//! - The set of composition-id paths that have already completed —
//!   the executor consults this on resume to fast-forward past them.
//! - A structural fingerprint of the composition tree. On resume the
//!   fingerprint must match the freshly-constructed composition; a
//!   tree shape change invalidates the snapshot. To restructure
//!   mid-flight, call [`CheckpointStore::delete`] first.
//!
//! ## What's intentionally NOT saved
//!
//! - Per-event progress (TextDelta / ToolCallProgress / …) — the
//!   consumer's durability concern.
//! - In-flight LLM round-trips. A composition that crashes
//!   mid-`agent.invoke()` resumes by re-running that step; completed
//!   sub-steps before it are skipped.
//! - Composition trees themselves. `Composition` holds boxed
//!   `dyn BorgAgent` — not serializable. The caller passes a freshly-
//!   constructed identical composition on resume; the fingerprint
//!   guards against mismatch.
//!
//! ## Save granularity ([`CheckpointPolicy`])
//!
//! - [`CheckpointPolicy::OnLoopIteration`] (default): save after every
//!   [`crate::Composition::LoopUntil`] iteration. ~1 write per LLM turn.
//! - [`CheckpointPolicy::OnEveryStep`]: save after every step
//!   completion. Maximum recovery resolution; write-heavy.
//! - [`CheckpointPolicy::Manual`]: don't auto-save; the caller writes
//!   its own snapshot from [`crate::ExecutionOutcome`].
//!
//! ## Example
//!
//! ```ignore
//! use borg::*;
//! use std::sync::Arc;
//!
//! let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
//!
//! // First run: planner completes, then crash mid-builder. Snapshot
//! // carries planner's verdict + handoff slot it wrote.
//! let _ = execute_resumable(pipeline(), ctx.clone(), store.clone(), "task-42")
//!     .await;
//!
//! // Second run (after restart): same composition shape, same id.
//! // Fast-forwards past planner; resumes at builder.
//! let outcome = execute_resumable(pipeline(), ctx, store, "task-42").await?;
//! ```

use crate::composition::Composition;
use crate::error::BorgError;
use crate::handoff::HandoffState;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Schema version. Bumps on any breaking change to the [`Snapshot`]
/// layout; older snapshots fail to load with
/// `BorgError::Internal { message: "snapshot schema version mismatch" }`.
pub const SNAPSHOT_SCHEMA_VERSION: u32 = 2;

/// Persisted composition state. Travels through [`CheckpointStore`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Snapshot {
    /// Layout version. The executor refuses to load a snapshot whose
    /// version doesn't match [`SNAPSHOT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Structural fingerprint of the composition that produced this
    /// snapshot. On resume the freshly-constructed composition's
    /// fingerprint must match, otherwise the executor fails loud.
    pub composition_fingerprint: String,
    /// Full handoff state captured at the save point: the state map
    /// and the audit history.
    pub handoff: HandoffState,
    /// Composition-ids of steps that have already completed; the
    /// executor's fast-forward checks against this on resume. IDs
    /// match `BorgEvent::StepCompleted.composition_id`'s hierarchical
    /// form (e.g. `"session::review/security"`).
    ///
    /// `Vec<Arc<str>>` so the in-memory tracking set and the persisted
    /// snapshot share the type and `iter().cloned().collect()` is a
    /// chain of `Arc::clone`s. Serde wire format is a JSON string.
    pub completed_steps: Vec<std::sync::Arc<str>>,
    /// `ctx.turn_number` at save time. Resumed executions inherit it so
    /// event attribution stays consistent with pre-crash emission.
    pub turn_number: u32,
    /// Wall-clock save time. Stores may use this for TTL eviction.
    pub created_at: SystemTime,
}

impl Snapshot {
    /// Fresh snapshot for a composition that hasn't started yet. The
    /// executor seeds in-memory state with this on the first
    /// `execute_resumable` call; it isn't persisted until the first
    /// save point.
    pub fn empty(composition_fingerprint: String, turn_number: u32) -> Self {
        Self {
            schema_version: SNAPSHOT_SCHEMA_VERSION,
            composition_fingerprint,
            handoff: HandoffState::new(),
            completed_steps: Vec::new(),
            turn_number,
            created_at: SystemTime::now(),
        }
    }
}

/// Save granularity for [`crate::execute_resumable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CheckpointPolicy {
    /// Save after every [`crate::Composition::LoopUntil`] iteration.
    /// ~1 write per LLM turn.
    #[default]
    OnLoopIteration,
    /// Save after every step completion (Single, branch arm,
    /// parallel branch). Maximum recovery resolution.
    OnEveryStep,
    /// Don't auto-save. The caller owns any persistence.
    Manual,
}

/// Backing store for [`Snapshot`]s. Implement against your durable
/// wire of choice — Redis, Postgres, S3, local disk, in-memory.
///
/// ## Contract
///
/// - `save` is an idempotent overwrite; the executor doesn't merge.
/// - `load` returns `None` for unknown ids and does NOT validate
///   contents — schema version and fingerprint match are the
///   executor's job.
/// - `delete` is idempotent: deleting an absent id is `Ok(())`.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Persist `snap` under `id`. Idempotent overwrite.
    async fn save(&self, id: &str, snap: &Snapshot) -> Result<(), BorgError>;

    /// Read the snapshot for `id`; `None` if none exists (first run).
    async fn load(&self, id: &str) -> Result<Option<Snapshot>, BorgError>;

    /// Remove the snapshot for `id`. Idempotent. The executor calls
    /// this on successful completion so the next call starts fresh.
    async fn delete(&self, id: &str) -> Result<(), BorgError>;
}

/// In-process backing store; dies with the process. Suitable for
/// tests and single-replica services that don't need crash-surviving
/// durability. Production callers swap in a Redis or Postgres impl.
#[derive(Default)]
pub struct InMemoryCheckpointStore {
    inner: Arc<Mutex<HashMap<String, Snapshot>>>,
}

impl InMemoryCheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inspect the underlying map — testing only.
    #[cfg(any(test, feature = "testing"))]
    pub fn len(&self) -> usize {
        self.inner.lock().expect("store mutex poisoned").len()
    }

    #[cfg(any(test, feature = "testing"))]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().expect("store mutex poisoned").is_empty()
    }
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn save(&self, id: &str, snap: &Snapshot) -> Result<(), BorgError> {
        let mut g = self.inner.lock().expect("store mutex poisoned");
        g.insert(id.to_string(), snap.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Snapshot>, BorgError> {
        let g = self.inner.lock().expect("store mutex poisoned");
        Ok(g.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<(), BorgError> {
        let mut g = self.inner.lock().expect("store mutex poisoned");
        g.remove(id);
        Ok(())
    }
}

/// Compute a stable structural fingerprint for a composition tree.
///
/// Two compositions constructed identically (same step names, nesting,
/// variants) produce the same string. The fingerprint is structural,
/// not semantic: compositions that differ in *what they do* but share
/// labels can collide.
///
/// Returns a human-readable canonical form (e.g.
/// `"Seq[Single(plan),Loop<max=10,body=Seq[Single(turn)]>]"`) rather
/// than a hash, to aid debugging; snapshots are small.
pub fn composition_fingerprint(comp: &Composition) -> String {
    let mut s = String::new();
    write_fingerprint(comp, &mut s);
    s
}

fn write_fingerprint(comp: &Composition, out: &mut String) {
    match comp {
        Composition::Single { name, .. } => {
            out.push_str("Single(");
            out.push_str(name);
            out.push(')');
        }
        Composition::Sequential { steps, .. } => {
            out.push_str("Seq[");
            for (i, step) in steps.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_fingerprint(step, out);
            }
            out.push(']');
        }
        Composition::Parallel { branches, .. } => {
            // Sort by name: two construction orders must fingerprint
            // the same.
            let mut sorted: Vec<(&String, &Composition)> =
                branches.iter().map(|(k, v)| (k, v)).collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            out.push_str("Par[");
            for (i, (name, child)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(name);
                out.push(':');
                write_fingerprint(child, out);
            }
            out.push(']');
        }
        Composition::LoopUntil {
            body,
            max_iterations,
            ..
        } => {
            // The body is a closure producing a Composition. Invoke it
            // once at iteration=0 with an empty handoff to materialize
            // and fingerprint the body shape; otherwise two LoopUntils
            // with the same `max_iterations` but different bodies would
            // fingerprint identically.
            //
            // Contract: resumable LoopUntil bodies MUST be safe to
            // invoke at iteration=0 with an empty handoff and
            // `last_verdict = None`. (The first real iteration already
            // runs that way.)
            out.push_str("Loop<max=");
            out.push_str(&max_iterations.to_string());
            out.push_str(",body=");
            let empty_handoff = crate::handoff::HandoffState::new();
            let probe_ctx = crate::composition::LoopCtx {
                iteration: 0,
                handoff: &empty_handoff,
                last_verdict: None,
            };
            let body_comp = body(probe_ctx);
            write_fingerprint(&body_comp, out);
            out.push('>');
        }
        Composition::Branch {
            branches, default, ..
        } => {
            out.push_str("Branch[");
            let mut sorted: Vec<(&crate::composition::BranchKey, &Composition)> =
                branches.iter().map(|(k, v)| (k, v)).collect();
            sorted.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
            for (i, (key, child)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(key.as_str());
                out.push(':');
                write_fingerprint(child, out);
            }
            if let Some(d) = default {
                out.push_str(",default:");
                write_fingerprint(d, out);
            }
            out.push(']');
        }
        Composition::Fallback {
            primary, fallbacks, ..
        } => {
            out.push_str("Fallback[");
            write_fingerprint(primary, out);
            for fb in fallbacks {
                out.push(',');
                write_fingerprint(fb, out);
            }
            out.push(']');
        }
        Composition::FallbackLadder { rungs, .. } => {
            // Predicates are opaque; the rung bodies are the shape.
            out.push_str("FallbackLadder[");
            for (i, r) in rungs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_fingerprint(&r.body, out);
            }
            out.push(']');
        }
        Composition::Agent {
            name, max_steps, ..
        } => {
            // Keyed on step name + cap. Tool sets are NOT fingerprinted
            // (the registry is opaque here); agents that change their
            // tools on resume must surface that themselves.
            out.push_str("AgentLoop(");
            out.push_str(name);
            out.push_str(",max=");
            out.push_str(&max_steps.to_string());
            out.push(')');
        }
        Composition::Heartbeat { name, inner, .. } => {
            // The period is timing, not shape; the name and the inner
            // composition are.
            out.push_str("Heartbeat(");
            out.push_str(name);
            out.push(',');
            write_fingerprint(inner, out);
            out.push(')');
        }
        Composition::Deferred { build } => {
            // Same contract as a LoopUntil body: materialized once with an
            // empty handoff to fingerprint the shape it builds.
            out.push_str("Deferred<");
            let probe = build(&crate::handoff::HandoffState::new());
            write_fingerprint(&probe, out);
            out.push('>');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::ScriptedAgent;

    #[tokio::test]
    async fn in_memory_store_round_trip() {
        let store = InMemoryCheckpointStore::new();
        let snap = Snapshot::empty("Seq[Single(a)]".into(), 0);
        store.save("id1", &snap).await.unwrap();
        let loaded = store.load("id1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().composition_fingerprint, "Seq[Single(a)]");
    }

    #[tokio::test]
    async fn in_memory_store_load_missing_is_none() {
        let store = InMemoryCheckpointStore::new();
        assert!(store.load("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn in_memory_store_delete_is_idempotent() {
        let store = InMemoryCheckpointStore::new();
        let snap = Snapshot::empty("x".into(), 0);
        store.save("id", &snap).await.unwrap();
        assert!(!store.is_empty());
        store.delete("id").await.unwrap();
        assert!(store.is_empty());
        store.delete("id").await.unwrap();
    }

    #[test]
    fn fingerprint_single() {
        let c = Composition::single("plan", ScriptedAgent::new("plan"));
        assert_eq!(composition_fingerprint(&c), "Single(plan)");
    }

    #[test]
    fn fingerprint_sequential() {
        let c = Composition::sequential([
            Composition::single("a", ScriptedAgent::new("a")),
            Composition::single("b", ScriptedAgent::new("b")),
        ]);
        assert_eq!(composition_fingerprint(&c), "Seq[Single(a),Single(b)]");
    }

    #[test]
    fn fingerprint_parallel_is_stable_across_insertion_order() {
        let c1 = Composition::parallel([
            ("first", Composition::single("a", ScriptedAgent::new("a"))),
            ("second", Composition::single("b", ScriptedAgent::new("b"))),
        ]);
        let c2 = Composition::parallel([
            ("second", Composition::single("b", ScriptedAgent::new("b"))),
            ("first", Composition::single("a", ScriptedAgent::new("a"))),
        ]);
        assert_eq!(
            composition_fingerprint(&c1),
            composition_fingerprint(&c2),
            "branch order must not affect fingerprint"
        );
    }

    #[test]
    fn fingerprint_detects_renamed_step() {
        let c1 = Composition::single("plan", ScriptedAgent::new("plan"));
        let c2 = Composition::single("planV2", ScriptedAgent::new("planV2"));
        assert_ne!(composition_fingerprint(&c1), composition_fingerprint(&c2));
    }

    #[test]
    fn fingerprint_loop_until_includes_body_shape() {
        // Same max, different body shapes must not fingerprint alike.
        let a = crate::composition::Composition::loop_until_builder(
            |_| {
                crate::composition::Composition::single_builder(
                    "stepA",
                    Box::new(ScriptedAgent::new("stepA")),
                )
                .build()
            },
            |_| false,
        )
        .max_iterations(3)
        .build();
        let b = crate::composition::Composition::loop_until_builder(
            |_| {
                crate::composition::Composition::single_builder(
                    "stepB",
                    Box::new(ScriptedAgent::new("stepB")),
                )
                .build()
            },
            |_| false,
        )
        .max_iterations(3)
        .build();
        assert_ne!(
            composition_fingerprint(&a),
            composition_fingerprint(&b),
            "LoopUntil body shape must be part of the fingerprint"
        );
    }

    #[test]
    fn fingerprint_loop_until_same_body_same_max_matches() {
        let build = || {
            crate::composition::Composition::loop_until_builder(
                |_| {
                    crate::composition::Composition::single_builder(
                        "step",
                        Box::new(ScriptedAgent::new("step")),
                    )
                    .build()
                },
                |_| false,
            )
            .max_iterations(5)
            .build()
        };
        assert_eq!(
            composition_fingerprint(&build()),
            composition_fingerprint(&build())
        );
    }

    #[test]
    fn snapshot_serializes_round_trip() {
        let snap = Snapshot::empty("Seq[Single(x)]".into(), 7);
        let json = serde_json::to_string(&snap).unwrap();
        let back: Snapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.composition_fingerprint, "Seq[Single(x)]");
        assert_eq!(back.turn_number, 7);
        assert_eq!(back.schema_version, SNAPSHOT_SCHEMA_VERSION);
    }
}
