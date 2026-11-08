//! In-process resident actor registry.
//!
//! Actors that complete a user turn gracefully (returning [`StreamingHandleOutcome::Park`])
//! stay alive in their mailbox, waiting for the next user message. The registry
//! maps a stable string id to the live [`StreamingActorHandle`] so the session
//! layer can look up the resident actor and send [`Park`]-resuming messages
//! directly — skipping actor re-spawn, snapshot deserialization, and full
//! context reconstruction.
//!
//! ## Lifecycle
//!
//! ```text
//! spawn_streaming_actor → handle → registry.insert(id, handle)
//!    ↓
//! actor Parks  → session driver calls registry.mark_parked(id)
//!    ↓
//! next user message → registry.get(id) returns Some → send NextUserTurn
//!    ↓
//! actor Terminals / errors → registry.remove(id)
//!    (or remove_if(id, |h| Arc::ptr_eq(h, &mine)) from a run that may
//!     have been superseded)
//!
//! periodic reaper → registry.reap_closed()
//!    (an actor that idles out exits without telling the registry)
//! ```
//!
//! The registry holds `Arc<StreamingActorHandle<A>>` so multiple owners
//! (registry + session driver) can share a non-Clone handle.
//!
//! [`Park`]: crate::StreamingHandleOutcome::Park

use crate::streaming_actor::{StreamingActorAgent, StreamingActorHandle};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct RegistryEntry<A: StreamingActorAgent> {
    handle: Arc<StreamingActorHandle<A>>,
    parked: bool,
}

/// Thread-safe in-process map of resident streaming actors.
///
/// Keyed by any `String` id — typically the session or conversation id
/// the application uses to identify long-lived agent sessions.
pub struct ActorRegistry<A: StreamingActorAgent> {
    inner: Mutex<HashMap<String, RegistryEntry<A>>>,
}

impl<A: StreamingActorAgent> Default for ActorRegistry<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: StreamingActorAgent> ActorRegistry<A> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Insert a newly-spawned actor handle under `id`.
    /// Replaces any existing entry for the same id (e.g. after a forced
    /// re-spawn following a Terminal exit).
    ///
    /// Takes the handle by value or already shared (`Arc`), so a driver
    /// that kept its own `Arc` while the turn ran can register that same
    /// handle — and later ask [`remove_if`] about it by pointer.
    ///
    /// [`remove_if`]: Self::remove_if
    pub fn insert(&self, id: impl Into<String>, handle: impl Into<Arc<StreamingActorHandle<A>>>) {
        self.insert_entry(id.into(), handle.into(), false);
    }

    /// Insert `handle` under `id` already parked: the turn that produced
    /// it ended with [`Park`], so the next [`claim_parked`] may take it.
    /// One lock acquisition — no window in which another caller sees the
    /// entry mid-turn.
    ///
    /// [`Park`]: crate::StreamingHandleOutcome::Park
    /// [`claim_parked`]: Self::claim_parked
    pub fn insert_parked(
        &self,
        id: impl Into<String>,
        handle: impl Into<Arc<StreamingActorHandle<A>>>,
    ) {
        self.insert_entry(id.into(), handle.into(), true);
    }

    fn insert_entry(&self, id: String, handle: Arc<StreamingActorHandle<A>>, parked: bool) {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        map.insert(id, RegistryEntry { handle, parked });
    }

    /// Whether an entry exists for `id`, parked or mid-turn, live or not.
    pub fn contains(&self, id: &str) -> bool {
        self.inner
            .lock()
            .expect("registry lock poisoned")
            .contains_key(id)
    }

    /// The handle under `id` whatever its state (parked, mid-turn, or
    /// with a closed inbox). For routing that decides on its own — e.g.
    /// "send if alive, rehydrate on a failed send" — rather than through
    /// [`claim_parked`].
    ///
    /// [`claim_parked`]: Self::claim_parked
    pub fn get(&self, id: &str) -> Option<Arc<StreamingActorHandle<A>>> {
        self.inner
            .lock()
            .expect("registry lock poisoned")
            .get(id)
            .map(|e| Arc::clone(&e.handle))
    }

    /// Look up a resident actor by id without changing its state.
    ///
    /// Returns `Some(handle)` only when the actor is alive AND parked.
    /// Returns `None` for an unknown id, an exited actor, or one still
    /// mid-turn.
    ///
    /// Prefer [`claim_parked`] for the "find parked + start using" path:
    /// this method's `get_parked` + `mark_running` pair has a TOCTOU
    /// window between lock acquisitions where a second caller can grab
    /// the same handle.
    ///
    /// [`claim_parked`]: Self::claim_parked
    pub fn get_parked(&self, id: &str) -> Option<Arc<StreamingActorHandle<A>>> {
        let map = self.inner.lock().expect("registry lock poisoned");
        map.get(id)
            .filter(|e| e.parked && !e.handle.is_closed())
            .map(|e| Arc::clone(&e.handle))
    }

    /// Atomically: if the actor is parked, mark it running and return its
    /// handle; otherwise (running, absent, or inbox closed) return `None`.
    ///
    /// The primitive for "next user message arrived; deliver to the
    /// parked actor if one exists, else spawn fresh." A single lock
    /// acquisition closes the `get_parked` + `mark_running` TOCTOU window.
    pub fn claim_parked(&self, id: &str) -> Option<Arc<StreamingActorHandle<A>>> {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        let entry = map.get_mut(id)?;
        if !entry.parked || entry.handle.is_closed() {
            return None;
        }
        entry.parked = false;
        Some(Arc::clone(&entry.handle))
    }

    /// Mark an actor as parked (graceful user-turn completion). The next
    /// call to [`get_parked`] for this id will return `Some`.
    ///
    /// No-op if the id is not in the registry.
    ///
    /// [`get_parked`]: Self::get_parked
    pub fn mark_parked(&self, id: &str) {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        if let Some(entry) = map.get_mut(id) {
            entry.parked = true;
        }
    }

    /// Clear the parked flag so the actor is treated as mid-turn again.
    /// Called at the start of a new user message. No-op for unknown ids.
    ///
    /// Prefer [`claim_parked`] over the [`get_parked`] + `mark_running`
    /// pair: it does the same atomically in one lock acquisition.
    ///
    /// [`get_parked`]: Self::get_parked
    /// [`claim_parked`]: Self::claim_parked
    pub fn mark_running(&self, id: &str) {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        if let Some(entry) = map.get_mut(id) {
            entry.parked = false;
        }
    }

    /// Remove the actor from the registry (e.g. on Terminal or error exit).
    /// The handle is dropped unless other owners (`Arc`) still hold it.
    pub fn remove(&self, id: &str) {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        map.remove(id);
    }

    /// Remove the entry under `id` only if `pred` says so, in one lock
    /// acquisition. Returns whether it was removed.
    ///
    /// The primitive for "an old worker never removes a newer entry": a
    /// run that lost its session hands its own handle and removes the
    /// entry only when it is still that handle —
    ///
    /// ```ignore
    /// registry.remove_if(&id, |h| Arc::ptr_eq(h, &mine));
    /// ```
    ///
    /// — so a newer actor registered under the same id in the meantime
    /// stays. A plain [`remove`] would take it out.
    ///
    /// [`remove`]: Self::remove
    pub fn remove_if(
        &self,
        id: &str,
        pred: impl FnOnce(&Arc<StreamingActorHandle<A>>) -> bool,
    ) -> bool {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        match map.get(id) {
            Some(entry) if pred(&entry.handle) => {
                map.remove(id);
                true
            }
            _ => false,
        }
    }

    /// Keep only the entries `keep` answers `true` for; returns how many
    /// were removed. One lock acquisition for the whole sweep.
    ///
    /// Typical use is a periodic reaper. An actor that exits on its own
    /// (idle timeout, panic, ancestor cancel) gives the registry no
    /// callback, so its entry would otherwise stay until the same id is
    /// routed again:
    ///
    /// ```ignore
    /// let reaped = registry.retain(|_id, h| !h.is_closed());
    /// ```
    ///
    /// `is_closed()` is false for a live parked actor, so that sweep never
    /// drops one that could still take a message.
    pub fn retain(
        &self,
        mut keep: impl FnMut(&str, &Arc<StreamingActorHandle<A>>) -> bool,
    ) -> usize {
        let mut map = self.inner.lock().expect("registry lock poisoned");
        let before = map.len();
        map.retain(|id, entry| keep(id, &entry.handle));
        before - map.len()
    }

    /// Drop every entry whose actor has exited (inbox closed); returns how
    /// many. Shorthand for `retain(|_, h| !h.is_closed())`.
    pub fn reap_closed(&self) -> usize {
        self.retain(|_, h| !h.is_closed())
    }

    /// Number of entries currently tracked (parked or mid-turn).
    pub fn len(&self) -> usize {
        self.inner.lock().expect("registry lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
