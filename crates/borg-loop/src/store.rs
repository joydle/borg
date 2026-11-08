//! `ActorStore` — persistence backend for actor snapshots.
//!
//! Decoupled from borg's `CheckpointStore` on purpose: that one is
//! shaped around a composition snapshot (handoff state + step list +
//! fingerprint) for resuming a pipeline, while `ActorStore` carries an
//! opaque application-defined byte blob (the actor's serialized
//! [`crate::ActorAgent::Snapshot`]) for resuming a long-lived session.
//! The two can share one PG table via a `kind` column without sharing
//! the trait — their consumers have different load-time invariants.

use async_trait::async_trait;
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// Persistence backend for actor snapshots. Implement against any
/// durable backend — Postgres, Redis, S3, local disk.
///
/// ## Contract
///
/// - `save(id, bytes)` is idempotent: a duplicate id overwrites. No
///   automatic merge or versioning.
/// - `load(id)` returns `None` for unknown ids. Does NOT validate
///   contents — the runtime's deserialize step does.
/// - `delete(id)` is idempotent. The runtime calls it on terminal exit
///   so the next spawn under the same id starts fresh.
///
/// Payloads are opaque `Arc<[u8]>`, not a typed snapshot: one impl can
/// back any number of `ActorAgent` types, and the `Arc` lets the
/// payload pass through the [`CachingActorStore`] / [`ResilientActorStore`]
/// stack via refcount bumps instead of per-layer deep copies. The
/// runtime owns the `serde_json` round-trip.
#[async_trait]
pub trait ActorStore: Send + Sync {
    /// Persist `bytes` under `id`. Overwrites on duplicate id.
    async fn save(&self, id: &str, bytes: Arc<[u8]>) -> Result<(), ActorStoreError>;

    /// Read the snapshot for `id`. `None` means first run.
    async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError>;

    /// Remove the snapshot for `id`. Idempotent.
    async fn delete(&self, id: &str) -> Result<(), ActorStoreError>;
}

/// Store error. A failed save is non-fatal: the actor keeps running
/// against in-memory state and retries at the next checkpoint.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ActorStoreError {
    /// Underlying store failure (PG connection, Redis timeout, etc.).
    #[error("actor store: {0}")]
    Backend(String),
}

/// In-process backing store. For tests and single-replica services
/// that don't need durability across process crashes; production
/// callers swap in a durable impl.
#[derive(Default)]
pub struct InMemoryActorStore {
    inner: Mutex<HashMap<String, Arc<[u8]>>>,
}

impl InMemoryActorStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored snapshots. Useful in tests.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl ActorStore for InMemoryActorStore {
    async fn save(&self, id: &str, bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        self.inner.lock().insert(id.to_string(), bytes);
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        Ok(self.inner.lock().get(id).map(Arc::clone))
    }

    async fn delete(&self, id: &str) -> Result<(), ActorStoreError> {
        self.inner.lock().remove(id);
        Ok(())
    }
}

/// Write-through in-process cache wrapping any [`ActorStore`].
///
/// `load` serves cached bytes on a hit and falls through to the inner
/// store on a miss (populating the cache). `save` / `delete` update the
/// inner store first, then the cache. Use it when the inner store is
/// remote (Redis, Postgres) and turns are frequent.
///
/// # Capacity
///
/// Prefer [`CachingActorStore::new_with_capacity`] in production. The
/// runtime only `delete`s on terminal exit, so sessions that exit via
/// `Cancelled` / `IdleTimeout` leave their snapshot cached; an
/// unbounded cache then grows linearly until OOM. The bounded variant
/// evicts the least-recently-used entry over `cap`, and a `load` on an
/// evicted key falls through to the inner store, re-warming the cache.
pub struct CachingActorStore<S> {
    inner: S,
    cache: Mutex<LruCache<String, Arc<[u8]>>>,
}

impl<S: ActorStore> CachingActorStore<S> {
    /// Unbounded cache. Prefer [`Self::new_with_capacity`] to avoid the
    /// linear memory growth documented on the type.
    #[deprecated(
        note = "prefer `CachingActorStore::new_with_capacity` for production use — the unbounded cache leaks entries for sessions that exit via Cancelled/IdleTimeout"
    )]
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::unbounded()),
        }
    }

    /// Bounded LRU cache. `capacity` caps resident snapshots;
    /// over-capacity inserts evict the least-recently-used entry, and a
    /// `load` for an evicted key falls through to the inner store.
    ///
    /// Size it to the expected number of concurrently-warm sessions
    /// (e.g. 1_000), adjusting for snapshot size and memory budget.
    pub fn new_with_capacity(inner: S, capacity: NonZeroUsize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Current number of cached entries. Useful for tests and gauges.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().len()
    }
}

#[async_trait]
impl<S: ActorStore> ActorStore for CachingActorStore<S> {
    async fn save(&self, id: &str, bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        self.inner.save(id, Arc::clone(&bytes)).await?;
        self.cache.lock().put(id.to_string(), bytes);
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        // `.get` (not `.peek`) bumps LRU recency. Clone before releasing
        // the lock so we don't hold it across the .await on a miss.
        if let Some(cached) = self.cache.lock().get(id).map(Arc::clone) {
            return Ok(Some(cached));
        }
        let result = self.inner.load(id).await?;
        if let Some(ref bytes) = result {
            self.cache.lock().put(id.to_string(), Arc::clone(bytes));
        }
        Ok(result)
    }

    async fn delete(&self, id: &str) -> Result<(), ActorStoreError> {
        // Evict even if the inner delete fails: its state is unknown
        // then, and a cached snapshot would resurrect the actor.
        let result = self.inner.delete(id).await;
        self.cache.lock().pop(id);
        result
    }
}

/// Noop store: `save` discards, `load` always returns `None`. For
/// callers that want the actor lifecycle without restart resilience
/// (test fixtures, ephemeral one-shot runs).
#[derive(Default)]
pub struct NoopActorStore;

#[async_trait]
impl ActorStore for NoopActorStore {
    async fn save(&self, _id: &str, _bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        Ok(())
    }
    async fn load(&self, _id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        Ok(None)
    }
    async fn delete(&self, _id: &str) -> Result<(), ActorStoreError> {
        Ok(())
    }
}

// ── Shared runtime helpers ────────────────────────────────────────────
//
// Both runtimes (plain + streaming) share identical persist and
// terminal-delete logic; only the log prefix differs.

/// Persist pre-serialized checkpoint bytes. Non-fatal: logs a warning and
/// continues if the store fails (loop runs against in-memory state).
pub(crate) async fn persist_checkpoint(
    addr: &str,
    bytes: Arc<[u8]>,
    store: &dyn ActorStore,
    context: &str,
) {
    if let Err(e) = store.save(addr, bytes).await {
        tracing::warn!(addr = %addr, error = %e, "{context}: checkpoint save failed (non-fatal)");
    }
}

/// Delete the snapshot on terminal exit so future spawns start fresh.
/// Cancel / idle don't delete — those are "pause, may resume."
pub(crate) async fn delete_on_terminal(
    exit: &crate::handle::ActorExit,
    addr: &str,
    store: &dyn ActorStore,
    context: &str,
) {
    let crate::handle::ActorExit::Terminal(_) = exit else {
        return;
    };
    if let Err(e) = store.delete(addr).await {
        tracing::warn!(addr = %addr, error = %e, "{context}: terminal snapshot delete failed (non-fatal)");
    }
}

/// Sentinel key the auto-recovery probe `load`s to test primary
/// liveness. `Ok(None)` for this absent key is the probe's success path.
const RECOVERY_PROBE_KEY: &str = "__borg_loop_health_probe__";

/// Default recovery-probe cadence — comfortably past the single-digit
/// seconds a transient network drop typically lasts.
const DEFAULT_RECOVERY_PROBE_INTERVAL: Duration = Duration::from_secs(30);

/// Timeout-aware primary + fallback store with auto-recovery.
///
/// Each op tries the primary first. On error or timeout the store flips
/// to a **degraded** state and routes subsequent ops to the fallback; a
/// background probe then periodically `load`s a sentinel against the
/// primary and flips back to healthy once it responds.
///
/// The `degraded` flag is per-instance: in multi-replica deployments
/// each replica degrades and recovers independently.
///
/// # Recovery semantics
///
/// - **Degrade (healthy → degraded):** any `save`/`load`/`delete` that
///   errors or times out flips the flag, spawns a single probe task, and
///   warns.
/// - **While degraded:** saves and deletes land on the fallback and are
///   remembered per id (a delete as a tombstone) until the primary has
///   them. Loads read the fallback; a tombstoned id loads as `None`.
/// - **Probe:** every `recovery_probe_interval` (default 30s) runs
///   `primary.load(RECOVERY_PROBE_KEY)` under `timeout`. On success it
///   replays every remembered save and delete onto the primary, then
///   flips back to healthy and exits; if any of that fails it stays
///   degraded and tries again next interval.
/// - **After recovery:** an id whose replay is still owed (one written
///   while the probe was flipping back) is served from the fallback, or
///   as `None` if tombstoned, and replayed on that access — never the
///   stale pre-outage primary snapshot.
/// - **Delete while healthy:** a primary delete that fails returns the
///   error, keeps a tombstone (so this store loads `None`) and degrades,
///   so the probe retries it.
/// - **Disabling:** pass `recovery_probe_interval = None` via
///   [`Self::with_recovery_probe_interval`] to stay degraded until
///   process restart.
/// - **Drop:** the probe task is aborted when the store is dropped.
///
/// ## Typical production wiring
///
/// ```ignore
/// // Redis primary, in-memory fallback, write-through LRU cache on top.
/// let primary = MyRedisStore::new(conn);
/// let store = CachingActorStore::new_with_capacity(
///     ResilientActorStore::new(primary, InMemoryActorStore::default()),
///     std::num::NonZeroUsize::new(1_000).unwrap(),
/// );
/// ```
pub struct ResilientActorStore<P, F = InMemoryActorStore> {
    shared: Arc<Resilient<P, F>>,
    recovery_probe_interval: Option<Duration>,
    probe_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// State shared between the store and its recovery probe.
struct Resilient<P, F> {
    primary: P,
    fallback: F,
    timeout: Duration,
    degraded: AtomicBool,
    probe_running: AtomicBool,
    /// Ops that reached the fallback but not yet the primary.
    pending: Mutex<PendingOps>,
    /// Per-id write locks. Every primary write of an id (a save, a
    /// delete, a replay) holds its lock, so a replay of stale fallback
    /// bytes can't land after a newer save of the same id.
    id_locks: IdLocks,
}

/// Async per-id locks; an id's entry lives only while someone holds or
/// waits for it.
#[derive(Default)]
struct IdLocks(Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>);

impl IdLocks {
    async fn lock(&self, id: &str) -> IdGuard<'_> {
        let lock = Arc::clone(self.0.lock().entry(id.to_string()).or_default());
        let guard = lock.lock_owned().await;
        IdGuard {
            locks: self,
            id: id.to_string(),
            guard: Some(guard),
        }
    }
}

/// Holds one id's lock; drops the map entry when nobody else wants it.
struct IdGuard<'a> {
    locks: &'a IdLocks,
    id: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for IdGuard<'_> {
    fn drop(&mut self) {
        drop(self.guard.take());
        let mut map = self.locks.0.lock();
        // Only the map's own reference left: no holder, no waiter.
        if map.get(&self.id).is_some_and(|l| Arc::strong_count(l) == 1) {
            map.remove(&self.id);
        }
    }
}

#[derive(Default)]
struct PendingOps {
    next_version: u64,
    by_id: HashMap<String, Pending>,
}

/// A save or delete the primary still owes. `version` changes on every
/// newer op for the id, so a replay only clears the op it replayed.
#[derive(Debug, Clone, Copy)]
struct Pending {
    kind: PendingKind,
    version: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingKind {
    /// The fallback holds a newer snapshot than the primary.
    Write,
    /// The id was deleted; the primary may still hold a snapshot.
    Delete,
}

impl<P: ActorStore, F: ActorStore + Default> ResilientActorStore<P, F> {
    /// Create with a 5-second primary timeout and the default 30s
    /// recovery probe.
    pub fn new(primary: P, fallback: F) -> Self {
        Self::with_timeout(primary, fallback, Duration::from_secs(5))
    }

    /// Create with a custom primary timeout. Recovery probe stays
    /// at the default 30s interval.
    pub fn with_timeout(primary: P, fallback: F, timeout: Duration) -> Self {
        Self {
            shared: Arc::new(Resilient {
                primary,
                fallback,
                timeout,
                degraded: AtomicBool::new(false),
                probe_running: AtomicBool::new(false),
                pending: Mutex::new(PendingOps::default()),
                id_locks: IdLocks::default(),
            }),
            recovery_probe_interval: Some(DEFAULT_RECOVERY_PROBE_INTERVAL),
            probe_handle: Mutex::new(None),
        }
    }

    /// Override the recovery-probe interval. `None` disables
    /// auto-recovery (the store stays degraded until restart); also
    /// useful in tests that drive degrade transitions without a ticking
    /// background task.
    #[must_use]
    pub fn with_recovery_probe_interval(mut self, interval: Option<Duration>) -> Self {
        self.recovery_probe_interval = interval;
        self
    }
}

impl<P, F> ResilientActorStore<P, F> {
    /// True if the store has degraded to the fallback backend.
    pub fn is_degraded(&self) -> bool {
        self.shared.degraded.load(Ordering::Acquire)
    }

    /// Saves and deletes that reached the fallback but not yet the
    /// primary. Zero once a recovery has replayed them all. For gauges
    /// and tests.
    pub fn pending_reconcile_count(&self) -> usize {
        self.shared.pending.lock().by_id.len()
    }
}

impl<P, F> Resilient<P, F> {
    /// Remember that `id`'s latest op reached only the fallback.
    fn mark(&self, id: &str, kind: PendingKind) -> u64 {
        let mut pending = self.pending.lock();
        pending.next_version += 1;
        let version = pending.next_version;
        pending
            .by_id
            .insert(id.to_string(), Pending { kind, version });
        version
    }

    fn pending(&self, id: &str) -> Option<Pending> {
        self.pending.lock().by_id.get(id).copied()
    }

    /// Forget `id`'s pending op if it is still the one at `version`.
    fn clear_if(&self, id: &str, version: u64) -> bool {
        let mut pending = self.pending.lock();
        if pending.by_id.get(id).is_some_and(|p| p.version == version) {
            pending.by_id.remove(id);
            true
        } else {
            false
        }
    }

    fn snapshot_pending(&self) -> Vec<(String, Pending)> {
        let pending = self.pending.lock();
        pending
            .by_id
            .iter()
            .map(|(id, p)| (id.clone(), *p))
            .collect()
    }
}

impl<P: ActorStore, F: ActorStore> Resilient<P, F> {
    /// Run a primary op under the timeout; a timeout is an error.
    async fn on_primary<T>(
        &self,
        op: impl std::future::Future<Output = Result<T, ActorStoreError>>,
    ) -> Result<T, ActorStoreError> {
        match tokio::time::timeout(self.timeout, op).await {
            Ok(result) => result,
            Err(_elapsed) => Err(ActorStoreError::Backend(format!(
                "primary timed out after {:?}",
                self.timeout
            ))),
        }
    }

    /// Replay one pending op onto the primary. On success the op is
    /// forgotten (unless a newer one replaced it meanwhile) and a
    /// replayed write's fallback copy is dropped, so a later outage
    /// can't serve it as current.
    ///
    /// The caller holds `id`'s lock and read `op` under it.
    async fn replay(&self, id: &str, op: Pending) -> Result<(), ActorStoreError> {
        match op.kind {
            PendingKind::Delete => {
                self.on_primary(self.primary.delete(id)).await?;
                self.clear_if(id, op.version);
            }
            PendingKind::Write => match self.fallback.load(id).await? {
                Some(bytes) => {
                    self.on_primary(self.primary.save(id, bytes)).await?;
                    if self.clear_if(id, op.version) {
                        let _ = self.fallback.delete(id).await;
                    }
                }
                // The fallback lost it (a lossy fallback): nothing to
                // replay, the primary keeps what it has.
                None => {
                    self.clear_if(id, op.version);
                }
            },
        }
        Ok(())
    }

    /// Replay every pending op. `false` at the first failure: the
    /// primary is still unwell, try again later.
    async fn replay_all(&self) -> bool {
        for (id, _) in self.snapshot_pending() {
            let _lock = self.id_locks.lock(&id).await;
            // Re-read under the lock: a save may have superseded it.
            let Some(op) = self.pending(&id) else {
                continue;
            };
            if let Err(e) = self.replay(&id, op).await {
                warn!(actor_id = %id, error = %e, "actor_store: replay to primary failed, staying degraded");
                return false;
            }
        }
        true
    }
}

impl<P: ActorStore + 'static, F: ActorStore + 'static> ResilientActorStore<P, F> {
    /// Mark degraded and, if not already running, spawn the recovery
    /// probe. The probe loops until the primary recovers and every
    /// pending op is replayed onto it, then flips `degraded` back to
    /// false and exits.
    fn enter_degraded(&self) {
        let shared = &self.shared;
        let was_healthy = !shared.degraded.swap(true, Ordering::AcqRel);
        if !was_healthy {
            return; // already degraded
        }
        let Some(interval) = self.recovery_probe_interval else {
            return; // auto-recovery disabled
        };
        // Single-spawn guard: only one probe task runs at a time.
        if shared
            .probe_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let shared = Arc::clone(shared);
        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let alive = shared
                    .on_primary(shared.primary.load(RECOVERY_PROBE_KEY))
                    .await
                    .is_ok();
                if alive && shared.replay_all().await {
                    // `probe_running` first: a failure landing between
                    // the two stores then finds no probe and spawns one.
                    // The other order leaves it degraded with no probe.
                    shared.probe_running.store(false, Ordering::Release);
                    shared.degraded.store(false, Ordering::Release);
                    tracing::info!(
                        "actor_store: primary recovered after degraded state, exiting probe"
                    );
                    return;
                }
                // Otherwise loop and try again on the next interval.
            }
        });
        *self.probe_handle.lock() = Some(handle);
    }
}

impl<P, F> Drop for ResilientActorStore<P, F> {
    fn drop(&mut self) {
        if let Some(handle) = self.probe_handle.lock().take() {
            handle.abort();
        }
    }
}

#[async_trait]
impl<P: ActorStore + 'static, F: ActorStore + 'static> ActorStore for ResilientActorStore<P, F> {
    async fn save(&self, id: &str, bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
        let s = &*self.shared;
        let _lock = s.id_locks.lock(id).await;
        if self.is_degraded() {
            s.fallback.save(id, bytes).await?;
            s.mark(id, PendingKind::Write);
            return Ok(());
        }
        // Share the payload (refcount bump) between the primary attempt
        // and the fallback retry below.
        match s.on_primary(s.primary.save(id, Arc::clone(&bytes))).await {
            Ok(()) => {
                // This save supersedes anything still owed for the id.
                let superseded = s.pending.lock().by_id.remove(id).is_some();
                if superseded {
                    let _ = s.fallback.delete(id).await;
                }
                Ok(())
            }
            Err(e) => {
                warn!(actor_id = id, error = %e, "actor_store: primary save failed, degrading to fallback");
                self.enter_degraded();
                s.fallback.save(id, bytes).await?;
                s.mark(id, PendingKind::Write);
                Ok(())
            }
        }
    }

    async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
        let s = &*self.shared;
        let pending = s.pending(id);
        if self.is_degraded() {
            return match pending {
                Some(Pending {
                    kind: PendingKind::Delete,
                    ..
                }) => Ok(None),
                _ => s.fallback.load(id).await,
            };
        }
        if pending.is_some() {
            // The replay writes the primary, so it takes the id's lock,
            // and re-reads the op under it: a save that finished in the
            // meantime already superseded it.
            let _lock = s.id_locks.lock(id).await;
            if let Some(op) = s.pending(id) {
                // Not yet replayed: the fallback (or the tombstone) is
                // newer than the primary. Serve it and replay now.
                let newer = match op.kind {
                    PendingKind::Delete => None,
                    PendingKind::Write => s.fallback.load(id).await?,
                };
                if let Err(e) = s.replay(id, op).await {
                    warn!(actor_id = id, error = %e, "actor_store: replay to primary failed, degrading to fallback");
                    self.enter_degraded();
                }
                if op.kind == PendingKind::Delete || newer.is_some() {
                    return Ok(newer);
                }
            }
        }
        match s.on_primary(s.primary.load(id)).await {
            Ok(v) => Ok(v),
            Err(e) => {
                warn!(actor_id = id, error = %e, "actor_store: primary load failed, degrading to fallback");
                self.enter_degraded();
                s.fallback.load(id).await
            }
        }
    }

    async fn delete(&self, id: &str) -> Result<(), ActorStoreError> {
        let s = &*self.shared;
        let _lock = s.id_locks.lock(id).await;
        // Tombstone first: until the primary confirms, this id loads as
        // `None` here even though the primary may still hold it.
        let version = s.mark(id, PendingKind::Delete);
        if self.is_degraded() {
            return s.fallback.delete(id).await;
        }
        let primary = s.on_primary(s.primary.delete(id)).await;
        let _ = s.fallback.delete(id).await;
        match primary {
            Ok(()) => {
                s.clear_if(id, version);
                Ok(())
            }
            Err(e) => {
                warn!(actor_id = id, error = %e, "actor_store: primary delete failed, keeping a tombstone and degrading");
                self.enter_degraded();
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Short literal payload as `Arc<[u8]>`, the store's payload type.
    fn arc_bytes(s: &[u8]) -> Arc<[u8]> {
        Arc::from(s)
    }

    #[tokio::test]
    async fn in_memory_save_and_load_round_trip() {
        let store = InMemoryActorStore::new();
        store.save("a", arc_bytes(b"hello")).await.unwrap();
        let got = store.load("a").await.unwrap();
        assert_eq!(got.as_deref(), Some(&b"hello"[..]));
    }

    #[tokio::test]
    async fn in_memory_load_unknown_returns_none() {
        let store = InMemoryActorStore::new();
        assert!(store.load("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn in_memory_save_overwrites() {
        let store = InMemoryActorStore::new();
        store.save("a", arc_bytes(b"v1")).await.unwrap();
        store.save("a", arc_bytes(b"v2")).await.unwrap();
        assert_eq!(store.load("a").await.unwrap().as_deref(), Some(&b"v2"[..]));
    }

    #[tokio::test]
    async fn in_memory_delete_is_idempotent() {
        let store = InMemoryActorStore::new();
        store.delete("never-existed").await.unwrap();
        store.save("a", arc_bytes(b"x")).await.unwrap();
        store.delete("a").await.unwrap();
        store.delete("a").await.unwrap();
        assert!(store.load("a").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn noop_save_discards() {
        let store = NoopActorStore;
        store.save("a", arc_bytes(b"x")).await.unwrap();
        assert!(store.load("a").await.unwrap().is_none());
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn caching_store_load_populates_cache() {
        let inner = InMemoryActorStore::new();
        inner.save("a", arc_bytes(b"hello")).await.unwrap();
        let store = CachingActorStore::new(inner);
        // Cache miss: delegates to inner.
        assert_eq!(
            store.load("a").await.unwrap().as_deref(),
            Some(&b"hello"[..])
        );
        // Cache hit.
        assert_eq!(
            store.load("a").await.unwrap().as_deref(),
            Some(&b"hello"[..])
        );
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn caching_store_save_updates_cache() {
        let store = CachingActorStore::new(InMemoryActorStore::new());
        store.save("a", arc_bytes(b"v1")).await.unwrap();
        store.save("a", arc_bytes(b"v2")).await.unwrap();
        assert_eq!(store.load("a").await.unwrap().as_deref(), Some(&b"v2"[..]));
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn caching_store_delete_evicts() {
        let store = CachingActorStore::new(InMemoryActorStore::new());
        store.save("a", arc_bytes(b"x")).await.unwrap();
        // Populate cache via load.
        store.load("a").await.unwrap();
        store.delete("a").await.unwrap();
        assert!(store.load("a").await.unwrap().is_none());
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn caching_store_load_unknown_returns_none() {
        let store = CachingActorStore::new(InMemoryActorStore::new());
        assert!(store.load("nope").await.unwrap().is_none());
    }

    // ── Bounded LRU semantics ────────────────────────────────────────────

    #[tokio::test]
    async fn bounded_caching_store_caps_resident_entries() {
        // Cap = 5. Insert 105 entries; assert size stays at 5 and the
        // 100 evicted entries fall through to the inner store on load.
        let cap = NonZeroUsize::new(5).unwrap();
        let store = CachingActorStore::new_with_capacity(InMemoryActorStore::new(), cap);
        for i in 0..105 {
            let id = format!("e-{i}");
            store
                .save(&id, Arc::from(format!("v-{i}").into_bytes()))
                .await
                .unwrap();
        }
        assert_eq!(
            store.cache_len(),
            5,
            "cache must be bounded at capacity; got {}",
            store.cache_len()
        );

        // The most-recently-saved 5 are still resident; the rest were
        // evicted from cache but persist in the inner store, so load
        // still returns them.
        for i in 0..105 {
            let id = format!("e-{i}");
            let expected = format!("v-{i}").into_bytes();
            assert_eq!(
                store.load(&id).await.unwrap().as_deref(),
                Some(expected.as_slice()),
                "evicted entry {id} must miss-fall-through to inner store"
            );
        }
    }

    #[tokio::test]
    async fn bounded_caching_store_load_bumps_lru_recency() {
        // Insert 3 entries into a cap-3 cache. Touch "a" via load.
        // Insert a 4th. "a" must survive — least-recently-used is "b".
        let cap = NonZeroUsize::new(3).unwrap();
        let store = CachingActorStore::new_with_capacity(InMemoryActorStore::new(), cap);
        store.save("a", arc_bytes(b"a")).await.unwrap();
        store.save("b", arc_bytes(b"b")).await.unwrap();
        store.save("c", arc_bytes(b"c")).await.unwrap();
        assert_eq!(store.cache_len(), 3);

        // Touch "a" — moves it to most-recently-used.
        store.load("a").await.unwrap();

        // Insert "d" — evicts the LRU, which should now be "b" (since
        // "a" was just bumped).
        store.save("d", arc_bytes(b"d")).await.unwrap();
        assert_eq!(store.cache_len(), 3);

        // Cache now holds a, c, d; b was evicted. We can't inspect
        // presence directly, so confirm a miss-fill keeps cache_len at 3.
        let _ = store.load("b").await.unwrap();
        assert_eq!(
            store.cache_len(),
            3,
            "cache must stay at capacity after a miss-fill"
        );
    }

    #[test]
    fn resilient_actor_store_name_check() {
        let _ = ResilientActorStore::new(InMemoryActorStore::new(), InMemoryActorStore::new());
    }

    // ── ResilientActorStore auto-recovery probe ───────────────────────────

    /// Primary that fails its first N calls, then succeeds.
    #[derive(Default)]
    struct FlakyStore {
        remaining_failures: std::sync::atomic::AtomicU32,
    }

    impl FlakyStore {
        fn with_initial_failures(n: u32) -> Self {
            Self {
                remaining_failures: std::sync::atomic::AtomicU32::new(n),
            }
        }

        fn maybe_fail(&self) -> Result<(), ActorStoreError> {
            // `fetch_update` is deprecated (renamed `try_update`) on newer
            // toolchains, but `try_update` is not on the 1.88 MSRV.
            #[allow(deprecated)]
            let prev =
                self.remaining_failures
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                        if n > 0 {
                            Some(n - 1)
                        } else {
                            None
                        }
                    });
            match prev {
                Ok(_) => Err(ActorStoreError::Backend("simulated flake".into())),
                Err(_) => Ok(()),
            }
        }
    }

    #[async_trait]
    impl ActorStore for FlakyStore {
        async fn save(&self, _id: &str, _bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
            self.maybe_fail()
        }
        async fn load(&self, _id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
            self.maybe_fail().map(|_| None)
        }
        async fn delete(&self, _id: &str) -> Result<(), ActorStoreError> {
            self.maybe_fail()
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn resilient_store_auto_recovers_after_primary_heals() {
        // FlakyStore fails its very first call, then is healthy. The
        // first save degrades the store; the background probe should
        // then succeed at the next interval and flip back to healthy.
        let primary = FlakyStore::with_initial_failures(1);
        let store = ResilientActorStore::new(primary, InMemoryActorStore::new())
            .with_recovery_probe_interval(Some(Duration::from_millis(50)));

        // Pre-condition: healthy.
        assert!(!store.is_degraded());

        // First save: primary fails → degrade transition + spawn probe.
        store.save("a", arc_bytes(b"x")).await.unwrap();
        assert!(
            store.is_degraded(),
            "store must degrade after a primary failure"
        );

        // Advance past the probe interval and yield so the probe runs;
        // the primary now returns Ok(None) and the store flips healthy.
        for _ in 0..10 {
            tokio::time::advance(Duration::from_millis(60)).await;
            tokio::task::yield_now().await;
            if !store.is_degraded() {
                break;
            }
        }
        assert!(
            !store.is_degraded(),
            "store must auto-recover once primary heals"
        );
    }

    // ── ResilientActorStore reconciliation after an outage ───────────────

    /// Primary backed by a real map that a test can take down and bring
    /// back, and whose deletes can fail on their own. Clones share state,
    /// so the test keeps one to inspect what the primary holds.
    #[derive(Clone, Default)]
    struct SwitchStore {
        data: Arc<InMemoryActorStore>,
        down: Arc<AtomicBool>,
        deletes_fail: Arc<AtomicBool>,
    }

    impl SwitchStore {
        fn set_down(&self, down: bool) {
            self.down.store(down, Ordering::SeqCst);
        }
        fn check(&self) -> Result<(), ActorStoreError> {
            if self.down.load(Ordering::SeqCst) {
                Err(ActorStoreError::Backend("primary down".into()))
            } else {
                Ok(())
            }
        }
    }

    #[async_trait]
    impl ActorStore for SwitchStore {
        async fn save(&self, id: &str, bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
            self.check()?;
            self.data.save(id, bytes).await
        }
        async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
            self.check()?;
            self.data.load(id).await
        }
        async fn delete(&self, id: &str) -> Result<(), ActorStoreError> {
            self.check()?;
            if self.deletes_fail.load(Ordering::SeqCst) {
                return Err(ActorStoreError::Backend("delete refused".into()));
            }
            self.data.delete(id).await
        }
    }

    fn switch_store() -> (SwitchStore, ResilientActorStore<SwitchStore>) {
        let primary = SwitchStore::default();
        let store = ResilientActorStore::new(primary.clone(), InMemoryActorStore::new())
            .with_recovery_probe_interval(Some(Duration::from_millis(50)));
        (primary, store)
    }

    /// Advance virtual time until the probe flips the store healthy.
    async fn wait_recovered<P, F>(store: &ResilientActorStore<P, F>) {
        for _ in 0..20 {
            tokio::time::advance(Duration::from_millis(60)).await;
            tokio::task::yield_now().await;
            if !store.is_degraded() {
                return;
            }
        }
        panic!("store never recovered");
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn save_during_outage_survives_recovery() {
        // Regression: the outage write only reached the fallback, so after
        // the flip back `load` returned the pre-outage primary snapshot.
        let (primary, store) = switch_store();
        store.save("a", arc_bytes(b"v1")).await.unwrap();

        primary.set_down(true);
        store.save("a", arc_bytes(b"v2")).await.unwrap();
        assert!(store.is_degraded());
        assert_eq!(store.load("a").await.unwrap().as_deref(), Some(&b"v2"[..]));

        primary.set_down(false);
        wait_recovered(&store).await;
        assert_eq!(
            store.load("a").await.unwrap().as_deref(),
            Some(&b"v2"[..]),
            "recovered store resumed the pre-outage snapshot"
        );
        // Reconciled to the primary, so other replicas see it too.
        assert_eq!(
            primary.data.load("a").await.unwrap().as_deref(),
            Some(&b"v2"[..])
        );
        assert_eq!(store.pending_reconcile_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn delete_during_outage_survives_recovery() {
        // Regression: the outage delete only reached the fallback, so
        // after the flip back the ended actor was resurrected.
        let (primary, store) = switch_store();
        store.save("a", arc_bytes(b"v1")).await.unwrap();

        primary.set_down(true);
        store.save("other", arc_bytes(b"x")).await.unwrap(); // degrade
        assert!(store.is_degraded());
        store.delete("a").await.unwrap();
        assert!(store.load("a").await.unwrap().is_none());

        primary.set_down(false);
        wait_recovered(&store).await;
        assert!(
            store.load("a").await.unwrap().is_none(),
            "recovered store resurrected a deleted actor"
        );
        assert!(primary.data.load("a").await.unwrap().is_none());
        assert_eq!(store.pending_reconcile_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn failed_primary_delete_is_surfaced_and_tombstoned() {
        // Regression: a failed primary delete while healthy was ignored,
        // so the next load resurrected the ended session.
        let (primary, store) = switch_store();
        store.save("a", arc_bytes(b"v1")).await.unwrap();

        primary.deletes_fail.store(true, Ordering::SeqCst);
        assert!(
            store.delete("a").await.is_err(),
            "a primary delete failure must be surfaced"
        );
        assert!(
            store.load("a").await.unwrap().is_none(),
            "the tombstone hides the undeleted snapshot"
        );

        // Once the primary accepts deletes the tombstone is applied.
        primary.deletes_fail.store(false, Ordering::SeqCst);
        wait_recovered(&store).await;
        assert!(primary.data.load("a").await.unwrap().is_none());
        assert!(store.load("a").await.unwrap().is_none());
        assert_eq!(store.pending_reconcile_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn caching_store_drops_its_entry_when_delete_fails() {
        // The usual stack: a cache over the resilient store. A failed
        // delete must not leave the snapshot cached.
        let primary = SwitchStore::default();
        let store = CachingActorStore::new_with_capacity(
            ResilientActorStore::new(primary.clone(), InMemoryActorStore::new())
                .with_recovery_probe_interval(None),
            NonZeroUsize::new(8).unwrap(),
        );
        store.save("a", arc_bytes(b"v1")).await.unwrap();
        primary.deletes_fail.store(true, Ordering::SeqCst);
        assert!(store.delete("a").await.is_err());
        assert!(store.load("a").await.unwrap().is_none());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn store_degrades_and_recovers_again_after_recovery() {
        // The probe clears `probe_running` before `degraded`, so a failure
        // right after one recovery always gets a fresh probe.
        let (primary, store) = switch_store();
        for round in 0..2 {
            primary.set_down(true);
            store.save("a", arc_bytes(b"x")).await.unwrap();
            assert!(store.is_degraded(), "round {round}: degraded");
            primary.set_down(false);
            wait_recovered(&store).await;
        }
    }

    /// Primary whose save of one payload parks until released, to pin a
    /// replay mid-write.
    #[derive(Clone, Default)]
    struct GatedStore {
        data: Arc<InMemoryActorStore>,
        gated: Arc<Mutex<Option<Vec<u8>>>>,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl ActorStore for GatedStore {
        async fn save(&self, id: &str, bytes: Arc<[u8]>) -> Result<(), ActorStoreError> {
            let gated = self.gated.lock().as_deref() == Some(&bytes[..]);
            if gated {
                self.entered.notify_one();
                self.release.notified().await;
            }
            self.data.save(id, bytes).await
        }
        async fn load(&self, id: &str) -> Result<Option<Arc<[u8]>>, ActorStoreError> {
            self.data.load(id).await
        }
        async fn delete(&self, id: &str) -> Result<(), ActorStoreError> {
            self.data.delete(id).await
        }
    }

    #[tokio::test]
    async fn replay_never_overwrites_a_newer_save() {
        // Regression: a load-triggered replay read the stale fallback
        // copy, a healthy save of newer bytes landed on the primary, then
        // the replay's write landed on top of it.
        let primary = GatedStore::default();
        let store = Arc::new(
            ResilientActorStore::new(primary.clone(), InMemoryActorStore::new())
                .with_recovery_probe_interval(None),
        );
        store.save("a", arc_bytes(b"v0")).await.unwrap();
        // An outage write the primary still owes, with the store healthy
        // again (as after a recovery that raced a degraded save).
        store
            .shared
            .fallback
            .save("a", arc_bytes(b"stale"))
            .await
            .unwrap();
        store.shared.mark("a", PendingKind::Write);
        *primary.gated.lock() = Some(b"stale".to_vec());

        let loader = Arc::clone(&store);
        let load = tokio::spawn(async move { loader.load("a").await });
        // The replay is now writing "stale" to the primary.
        primary.entered.notified().await;

        let saver = Arc::clone(&store);
        let mut save = tokio::spawn(async move { saver.save("a", arc_bytes(b"new")).await });
        // Give the save every chance to finish before the replay does
        // (it may not: it waits for the id's lock).
        let early = tokio::time::timeout(Duration::from_millis(100), &mut save).await;
        primary.release.notify_one();

        load.await.unwrap().unwrap();
        match early {
            Ok(saved) => saved.unwrap().unwrap(),
            Err(_) => save.await.unwrap().unwrap(),
        }
        assert_eq!(
            primary.data.load("a").await.unwrap().as_deref(),
            Some(&b"new"[..]),
            "the replay's stale bytes overwrote the newer save"
        );
        assert_eq!(store.load("a").await.unwrap().as_deref(), Some(&b"new"[..]));
        assert_eq!(store.pending_reconcile_count(), 0);
        assert!(
            store.shared.id_locks.0.lock().is_empty(),
            "lock entries leak"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn resilient_store_stays_degraded_when_recovery_disabled() {
        // Same flaky shape, recovery probe disabled: stays degraded.
        let primary = FlakyStore::with_initial_failures(1);
        let store = ResilientActorStore::new(primary, InMemoryActorStore::new())
            .with_recovery_probe_interval(None);

        store.save("a", arc_bytes(b"x")).await.unwrap();
        assert!(store.is_degraded());

        // Even after generous virtual time, no recovery.
        tokio::time::advance(Duration::from_secs(600)).await;
        tokio::task::yield_now().await;
        assert!(
            store.is_degraded(),
            "with recovery disabled, store must stay degraded"
        );
    }
}
