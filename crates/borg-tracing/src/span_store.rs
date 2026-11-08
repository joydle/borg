use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// Default capacity for the bounded span registry. Large enough that
/// normal operation never evicts an in-flight entry, but still caps
/// memory when completion events go missing (a panicking turn, a tool
/// that never emits `ToolReturned`, etc.).
const DEFAULT_CAPACITY: usize = 10_000;

/// Key into the open-span registry.
///
/// Identity fields are `Arc<str>` to match `BorgEvent`, so emitters reuse
/// the `Arc::clone` they already hold instead of allocating per lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpanKey {
    Composition(Arc<str>),
    Step(Arc<str>, Arc<str>),      // (composition_id, step_name)
    Turn(Arc<str>, Arc<str>, u32), // (composition_id, step_name, turn_index)
    ToolCall(Arc<str>, Arc<str>, u32, Arc<str>), // (composition_id, step_name, turn_index, call_id)
}

/// Thread-safe registry of open `tracing::Span`s keyed by lifecycle
/// position.
///
/// # Bounded LRU
///
/// Entries are bounded by an LRU policy (default capacity 10,000).
/// When a lifecycle's completion event never arrives, its open entry
/// would otherwise leak until OOM; instead the eldest unused entry
/// evicts at capacity, and later spans for the evicted lifecycle
/// re-anchor to the session span via the parent-resolution fallback.
///
/// `tracing::Span` is `Send + Sync` and cheap to clone (internally
/// reference-counted), so `get` returns a clone rather than a borrow.
pub struct SpanRegistry {
    inner: Mutex<LruCache<SpanKey, tracing::Span>>,
}

impl Default for SpanRegistry {
    fn default() -> Self {
        Self::with_capacity(NonZeroUsize::new(DEFAULT_CAPACITY).expect("non-zero default"))
    }
}

impl SpanRegistry {
    /// Construct a registry with a custom capacity, overriding the
    /// default of 10,000.
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Insert a span under `key`, evicting the least-recently-used entry
    /// if the registry is at capacity.
    pub fn insert(&self, key: SpanKey, span: tracing::Span) {
        self.inner.lock().put(key, span);
    }

    /// Look up the span for `key`, returning a clone and bumping LRU
    /// recency on hit.
    pub fn get(&self, key: &SpanKey) -> Option<tracing::Span> {
        self.inner.lock().get(key).cloned()
    }

    /// Remove and return the span for `key`, called when the lifecycle's
    /// completion event arrives.
    pub fn remove(&self, key: &SpanKey) -> Option<tracing::Span> {
        self.inner.lock().pop(key)
    }

    /// Return a clone of the stored `borg.turn` span for
    /// `(composition_id, step_name, turn_index)`, or `None` if absent.
    /// Does not remove the entry; the executor uses it to instrument the
    /// agent invocation so child spans nest under the turn span.
    pub fn get_turn_span(
        &self,
        composition_id: &str,
        step_name: &str,
        turn_index: u32,
    ) -> Option<tracing::Span> {
        self.get(&SpanKey::Turn(
            Arc::from(composition_id),
            Arc::from(step_name),
            turn_index,
        ))
    }

    /// Current number of entries. A value near capacity signals leaked
    /// lifecycles, useful for operator-side gauges.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// `true` when the registry holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_composition_span() {
        let store = SpanRegistry::default();
        let span = tracing::info_span!("test.composition");
        store.insert(SpanKey::Composition("c1".into()), span);
        assert!(store.get(&SpanKey::Composition("c1".into())).is_some());
        assert!(store.get(&SpanKey::Composition("c2".into())).is_none());
    }

    #[test]
    fn remove_returns_span() {
        let store = SpanRegistry::default();
        let span = tracing::info_span!("test.step");
        store.insert(SpanKey::Step("c1".into(), "step1".into()), span);
        let removed = store.remove(&SpanKey::Step("c1".into(), "step1".into()));
        assert!(removed.is_some());
        assert!(store
            .remove(&SpanKey::Step("c1".into(), "step1".into()))
            .is_none());
    }

    #[test]
    fn span_registry_name_check() {
        let _ = SpanRegistry::default();
    }

    #[test]
    fn registry_bound_prevents_unbounded_growth_on_missing_completions() {
        // Without the LRU bound, 10_000 inserts would all stick around.
        let cap = NonZeroUsize::new(100).expect("non-zero");
        let store = SpanRegistry::with_capacity(cap);
        for i in 0..10_000 {
            let span = tracing::info_span!("leaked.turn");
            store.insert(SpanKey::Turn("c".into(), "s".into(), i as u32), span);
        }
        assert_eq!(
            store.len(),
            100,
            "registry must cap at capacity, not grow unbounded"
        );
    }

    #[test]
    fn registry_len_tracks_inserts_and_removes() {
        let store = SpanRegistry::default();
        assert!(store.is_empty());
        store.insert(SpanKey::Composition("a".into()), tracing::info_span!("a"));
        store.insert(SpanKey::Composition("b".into()), tracing::info_span!("b"));
        assert_eq!(store.len(), 2);
        store.remove(&SpanKey::Composition("a".into()));
        assert_eq!(store.len(), 1);
    }
}
