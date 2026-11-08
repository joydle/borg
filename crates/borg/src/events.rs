//! Lifecycle events emitted during composition execution.
//!
//! Borg fires typed `BorgEvent`s at every composition boundary through
//! the `EventEmitter` trait. The consumer adapter projects them onto
//! whatever durable wire it uses; borg itself stays agnostic.

use async_trait::async_trait;
use std::sync::Arc;

/// Composition shape, surfaced on `CompositionStarted` so consumers can
/// render the right progress UI without inferring from sub-events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CompositionShape {
    Single,
    Sequential,
    Parallel,
    LoopUntil,
    Branch,
    Fallback,
    /// `Composition::Agent` — a tool-loop agent. Distinguishes "one
    /// agent, multiple tool-use iterations" from a fixed `LoopUntil`
    /// over a body.
    AgentLoop,
}

/// Lifecycle event borg fires during composition execution.
///
/// Every event carries `composition_id` so a consumer rendering N
/// concurrent compositions can attribute messages correctly. Per-step
/// events also carry `parent_id` (the enclosing composition) to allow
/// hierarchical rendering.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum BorgEvent {
    /// A composition started executing. Emits at the top level and at
    /// every nested boundary.
    CompositionStarted {
        composition_id: Arc<str>,
        parent_id: Option<Arc<str>>,
        shape: CompositionShape,
        /// Number of immediate children. `Single` → 1,
        /// `Sequential`/`Parallel` → `branches.len()`, `LoopUntil` →
        /// inner count for one iteration.
        branch_count: u8,
    },
    /// A single step (named branch / sequential step / loop iteration)
    /// is about to invoke its agent.
    StepStarted {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        /// Position within the parent composition, 0-indexed. Stable
        /// across runs for `Sequential`/`Parallel`; iteration index for
        /// `LoopUntil`.
        position: u8,
        /// Turn number from `ExecCtx.turn_number`.
        turn_number: u32,
        /// Model id from `BorgAgent::model`.
        model: Arc<str>,
    },
    /// A step finished — successfully or not. Carries token usage and
    /// wall-clock duration so consumers can build cost dashboards.
    StepCompleted {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        position: u8,
        duration_ms: u64,
        input_tokens: u64,
        output_tokens: u64,
        /// True iff the step's agent returned a `Verdict` cleanly. False
        /// indicates an error surfaced through `ErrorPolicy::CollectAll`,
        /// a panicked child, or a cancelled mid-flight call.
        success: bool,
        /// Short kebab-case failure tag when `success: false`, else
        /// `None`. Stable values:
        /// - `"cancelled"` — parent or cooperative cancel fired
        /// - `"budget-exhausted"` — `Budget::can_invoke()` returned false
        /// - `"llm-error"` — LLM call errored (transport, schema, …)
        /// - `"retry-exhausted"` — `RestartPolicy` ran out of attempts
        /// - `"panicked"` — agent code panicked (caught at `JoinSet` boundary)
        /// - `"internal"` — anything we couldn't classify
        ///
        /// Consumers MUST treat unknown values as opaque strings.
        failure_cause: Option<String>,
        turn_number: u32,
        model: Arc<str>,
    },
    /// A step under `RestartPolicy::OnFailure` failed and is about to
    /// retry. `attempt` is the upcoming attempt number (1-indexed).
    StepRetrying {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        attempt: u8,
        last_error: String,
    },
    /// A composition completed all of its branches. Emits whether the
    /// outcome was success, partial-success (CollectAll with errors),
    /// or failure.
    CompositionCompleted {
        composition_id: Arc<str>,
        total_duration_ms: u64,
        total_input_tokens: u64,
        total_output_tokens: u64,
        succeeded: u8,
        failed: u8,
    },
    /// `Composition::Parallel` admitted fewer branches than requested
    /// because the budget couldn't cover them all.
    ParallelDegraded {
        composition_id: Arc<str>,
        requested: u8,
        admitted: u8,
        reason: String,
    },
    /// `Composition::Branch` selector returned a key. `chosen` is the
    /// branch that will run; `matched` is false when no arm matched the
    /// key and the `default` arm took over (or the composition is
    /// about to fail with `UnmatchedBranch`).
    BranchDecided {
        composition_id: Arc<str>,
        chosen: String,
        matched: bool,
    },
    /// A composition was cancelled mid-flight. Emitted *before* draining
    /// children so consumers see the cancel intent without waiting for
    /// stragglers to abort.
    ///
    /// `reason` is a short tag. Borg's own tags: `parent_cancel`,
    /// `failfast_sibling_failed`, `budget_exhausted_mid_flight`;
    /// consumers may pass any string.
    Cancelled {
        composition_id: Arc<str>,
        reason: String,
    },
    /// A [`crate::dsl::heartbeat`] composition is still running: one beat
    /// every `every` while its inner composition works, and one at its
    /// start when asked ([`crate::HeartbeatBuilder::at_start`]). A
    /// liveness signal for a long step that is otherwise silent (a model
    /// call that reasons for minutes before its first byte), not an
    /// accounting record: a tick missed while the emitter was busy is
    /// skipped, not caught up.
    ///
    /// `composition_id` is the enclosing composition's: a heartbeat is
    /// transparent, its inner composition runs in the same context.
    Heartbeat {
        composition_id: Arc<str>,
        /// The heartbeat's name (`dsl::heartbeat(name, …)`).
        name: Arc<str>,
        /// 0-indexed beat count.
        beat: u32,
        /// Time since the heartbeat composition started.
        elapsed_ms: u64,
    },
    /// An LLM turn started inside a `Composition::Agent` tool-loop step.
    TurnStarted {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        turn_index: u32,
        model: Arc<str>,
        input_tokens: u64,
    },
    /// An LLM turn completed.
    TurnCompleted {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        turn_index: u32,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_creation_tokens: u64,
        /// Wall-clock milliseconds from LLM request dispatch to receipt of the
        /// first token byte. `None` when not measured (e.g. non-streaming LLM
        /// clients or consumers that don't track it yet).
        first_token_ms: Option<u64>,
    },
    /// A tool call was dispatched inside a `Composition::Agent` loop.
    ToolCalled {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        turn_index: u32,
        call_id: Arc<str>,
        tool_name: Arc<str>,
        /// Identifier shared by every tool in one parallel batch (a
        /// single `Run::Parallel` group). `None` for serial dispatches
        /// and single-call "parallel" runs that short-circuit to inline
        /// execution. Lets emitters group concurrent calls in a trace UI.
        batch_id: Option<Arc<str>>,
    },
    /// A tool call returned.
    ToolReturned {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        turn_index: u32,
        call_id: Arc<str>,
        tool_name: Arc<str>,
        success: bool,
        /// Failure message from `ToolCallResult.error`. `None` on success.
        error: Option<String>,
        /// Echoes [`BorgEvent::ToolCalled::batch_id`]. See that field
        /// for semantics.
        batch_id: Option<Arc<str>>,
    },
    /// A guardrail rule fired.
    GuardrailTriggered {
        composition_id: Arc<str>,
        step_name: Arc<str>,
        rule_name: Arc<str>,
        verdict: String,
        signal: String,
    },
}

impl BorgEvent {
    /// Convenience for tests that want to assert event ordering by
    /// shape rather than by full enum equality.
    pub fn kind(&self) -> &'static str {
        match self {
            BorgEvent::CompositionStarted { .. } => "composition_started",
            BorgEvent::StepStarted { .. } => "step_started",
            BorgEvent::StepCompleted { .. } => "step_completed",
            BorgEvent::StepRetrying { .. } => "step_retrying",
            BorgEvent::CompositionCompleted { .. } => "composition_completed",
            BorgEvent::ParallelDegraded { .. } => "parallel_degraded",
            BorgEvent::BranchDecided { .. } => "branch_decided",
            BorgEvent::Cancelled { .. } => "cancelled",
            BorgEvent::Heartbeat { .. } => "heartbeat",
            BorgEvent::TurnStarted { .. } => "turn_started",
            BorgEvent::TurnCompleted { .. } => "turn_completed",
            BorgEvent::ToolCalled { .. } => "tool_called",
            BorgEvent::ToolReturned { .. } => "tool_returned",
            BorgEvent::GuardrailTriggered { .. } => "guardrail_triggered",
        }
    }
}

/// Sink for `BorgEvent`s. Implementations route to whatever durable
/// transport the consumer chose (PG, OTLP, stdout, in-memory for tests).
///
/// Implementations MUST be cheap to share via `Arc<dyn EventEmitter>` —
/// borg passes `&dyn` references into spawned children. The trait is
/// not `Clone`-bound, so `Arc::clone` is the fan-out primitive; state
/// expensive to clone (e.g. a transport-backed batcher) belongs behind
/// an `Arc<Mutex<_>>` / `Arc<RwLock<_>>` field.
#[async_trait]
pub trait EventEmitter: Send + Sync {
    /// Emit one event. Errors are logged but never propagated —
    /// observability failures must not break execution.
    ///
    /// Passed as `Arc<BorgEvent>` so fan-out emitters share the event
    /// with a refcount bump instead of cloning the enum per sink.
    async fn emit(&self, event: Arc<BorgEvent>);

    /// Live `borg.turn` span for the given key, so executors can
    /// `.instrument(turn_span)` agent invocations and have spans created
    /// inside the agent nest correctly under the turn span in OTLP.
    /// Defaults to a disabled span so non-tracing emitters need no changes.
    fn turn_span(
        &self,
        _composition_id: &str,
        _step_name: &str,
        _turn_index: u32,
    ) -> tracing::Span {
        tracing::Span::none()
    }

    /// Live `borg.tool` span for the given call, so executors can
    /// `.instrument(tool_span)` `tool.execute(...)` and have spans
    /// created inside tool bodies (outbound RPCs, sub-traces) nest under
    /// `borg.tool` in OTLP. Defaults to a disabled span.
    fn tool_span(
        &self,
        _composition_id: &str,
        _step_name: &str,
        _turn_index: u32,
        _call_id: &str,
    ) -> tracing::Span {
        tracing::Span::none()
    }
}

/// One [`BorgEvent`] kind, without its payload — the unit of an
/// [`EventKinds`] set. Same names as [`BorgEvent::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EventKind {
    CompositionStarted,
    StepStarted,
    StepCompleted,
    StepRetrying,
    CompositionCompleted,
    ParallelDegraded,
    BranchDecided,
    Cancelled,
    TurnStarted,
    TurnCompleted,
    ToolCalled,
    ToolReturned,
    GuardrailTriggered,
    /// [`BorgEvent::Heartbeat`]. Last, so every earlier kind keeps its bit.
    Heartbeat,
}

impl EventKind {
    /// Every kind, in declaration order.
    pub const ALL: [EventKind; 14] = [
        EventKind::CompositionStarted,
        EventKind::StepStarted,
        EventKind::StepCompleted,
        EventKind::StepRetrying,
        EventKind::CompositionCompleted,
        EventKind::ParallelDegraded,
        EventKind::BranchDecided,
        EventKind::Cancelled,
        EventKind::TurnStarted,
        EventKind::TurnCompleted,
        EventKind::ToolCalled,
        EventKind::ToolReturned,
        EventKind::GuardrailTriggered,
        EventKind::Heartbeat,
    ];

    const fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

impl BorgEvent {
    /// This event's [`EventKind`].
    pub fn event_kind(&self) -> EventKind {
        match self {
            BorgEvent::CompositionStarted { .. } => EventKind::CompositionStarted,
            BorgEvent::StepStarted { .. } => EventKind::StepStarted,
            BorgEvent::StepCompleted { .. } => EventKind::StepCompleted,
            BorgEvent::StepRetrying { .. } => EventKind::StepRetrying,
            BorgEvent::CompositionCompleted { .. } => EventKind::CompositionCompleted,
            BorgEvent::ParallelDegraded { .. } => EventKind::ParallelDegraded,
            BorgEvent::BranchDecided { .. } => EventKind::BranchDecided,
            BorgEvent::Cancelled { .. } => EventKind::Cancelled,
            BorgEvent::Heartbeat { .. } => EventKind::Heartbeat,
            BorgEvent::TurnStarted { .. } => EventKind::TurnStarted,
            BorgEvent::TurnCompleted { .. } => EventKind::TurnCompleted,
            BorgEvent::ToolCalled { .. } => EventKind::ToolCalled,
            BorgEvent::ToolReturned { .. } => EventKind::ToolReturned,
            BorgEvent::GuardrailTriggered { .. } => EventKind::GuardrailTriggered,
        }
    }
}

/// A set of [`EventKind`]s — which lifecycle events a composition emits
/// ([`crate::ExecCtx::with_lifecycle_events`]) or a [`FilteredEmitter`]
/// passes. `Copy`, a bitset.
///
/// ```
/// use borg::events::{EventKind, EventKinds};
///
/// // Only the composition tree, no per-step or per-tool rows.
/// let tree = EventKinds::only(&[EventKind::CompositionStarted, EventKind::CompositionCompleted]);
/// assert!(tree.contains(EventKind::CompositionStarted));
/// assert!(!tree.contains(EventKind::StepStarted));
///
/// // Everything but tool rows.
/// let no_tools = EventKinds::ALL.without(EventKind::ToolCalled).without(EventKind::ToolReturned);
/// assert!(!no_tools.contains(EventKind::ToolCalled));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventKinds(u32);

impl EventKinds {
    /// Every kind (what a composition emits by default).
    pub const ALL: EventKinds = EventKinds(u32::MAX);
    /// No kind: the executor's own lifecycle events are dropped.
    pub const NONE: EventKinds = EventKinds(0);

    /// Exactly `kinds`.
    pub const fn only(kinds: &[EventKind]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < kinds.len() {
            bits |= kinds[i].bit();
            i += 1;
        }
        EventKinds(bits)
    }

    /// This set plus `kind`.
    #[must_use]
    pub const fn with(self, kind: EventKind) -> Self {
        EventKinds(self.0 | kind.bit())
    }

    /// This set minus `kind`.
    #[must_use]
    pub const fn without(self, kind: EventKind) -> Self {
        EventKinds(self.0 & !kind.bit())
    }

    /// Whether `kind` is in the set.
    pub const fn contains(self, kind: EventKind) -> bool {
        self.0 & kind.bit() != 0
    }

    /// Whether `event`'s kind is in the set.
    pub fn admits(self, event: &BorgEvent) -> bool {
        self.contains(event.event_kind())
    }
}

impl Default for EventKinds {
    fn default() -> Self {
        EventKinds::ALL
    }
}

/// An [`EventEmitter`] adapter that forwards only the events `keep`
/// accepts to `inner`; spans (`turn_span`, `tool_span`) always come from
/// `inner`. Use it to put a composition on an app's own sink without the
/// rows the app's wire does not carry.
///
/// ```
/// use borg::events::{EventKind, EventKinds, FilteredEmitter};
/// # let sink = borg::noop_emitter();
/// let emitter = FilteredEmitter::kinds(sink, EventKinds::only(&[EventKind::ToolCalled]));
/// ```
pub struct FilteredEmitter {
    inner: std::sync::Arc<dyn EventEmitter>,
    keep: std::sync::Arc<dyn Fn(&BorgEvent) -> bool + Send + Sync>,
}

impl FilteredEmitter {
    /// Forward the events `keep` returns true for.
    pub fn new(
        inner: std::sync::Arc<dyn EventEmitter>,
        keep: impl Fn(&BorgEvent) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner,
            keep: std::sync::Arc::new(keep),
        }
    }

    /// Forward the events whose kind is in `kinds`.
    pub fn kinds(inner: std::sync::Arc<dyn EventEmitter>, kinds: EventKinds) -> Self {
        Self::new(inner, move |e| kinds.admits(e))
    }
}

#[async_trait]
impl EventEmitter for FilteredEmitter {
    async fn emit(&self, event: Arc<BorgEvent>) {
        if (self.keep)(&event) {
            self.inner.emit(event).await;
        }
    }

    fn turn_span(&self, composition_id: &str, step_name: &str, turn_index: u32) -> tracing::Span {
        self.inner.turn_span(composition_id, step_name, turn_index)
    }

    fn tool_span(
        &self,
        composition_id: &str,
        step_name: &str,
        turn_index: u32,
        call_id: &str,
    ) -> tracing::Span {
        self.inner
            .tool_span(composition_id, step_name, turn_index, call_id)
    }
}

/// No-op emitter for callers that don't care about lifecycle events.
/// Used by pre-session contexts and unit tests.
pub struct NoopEmitter;

impl NoopEmitter {
    /// Build an `Arc<dyn EventEmitter>` that drops every event.
    /// Idiomatic shortcut: `NoopEmitter::shared()`.
    pub fn shared() -> std::sync::Arc<dyn EventEmitter> {
        std::sync::Arc::new(Self)
    }
}

/// Free helper: `borg::noop_emitter()`. Sister to [`crate::in_memory_budget`].
pub fn noop_emitter() -> std::sync::Arc<dyn EventEmitter> {
    std::sync::Arc::new(NoopEmitter)
}

#[async_trait]
impl EventEmitter for NoopEmitter {
    async fn emit(&self, _event: Arc<BorgEvent>) {}
}

/// In-memory emitter that records every event. Used by tests to assert
/// event ordering and content. Concurrency-safe via `Mutex`.
#[cfg(any(test, feature = "testing"))]
#[derive(Default)]
pub struct RecordingEmitter {
    pub events: std::sync::Mutex<Vec<BorgEvent>>,
}

#[cfg(any(test, feature = "testing"))]
impl RecordingEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Vec<BorgEvent> {
        self.events.lock().unwrap().clone()
    }

    pub fn kinds(&self) -> Vec<&'static str> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.kind())
            .collect()
    }
}

#[cfg(any(test, feature = "testing"))]
#[async_trait]
impl EventEmitter for RecordingEmitter {
    async fn emit(&self, event: Arc<BorgEvent>) {
        // Slow-path test helper: clone out of the Arc so callers see `Vec<BorgEvent>`.
        self.events.lock().unwrap().push((*event).clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_emitter_swallows_events() {
        let e = NoopEmitter;
        e.emit(Arc::new(BorgEvent::Cancelled {
            composition_id: Arc::from("c1"),
            reason: "test".into(),
        }))
        .await;
    }

    #[tokio::test]
    async fn recording_emitter_captures_in_order() {
        let r = RecordingEmitter::new();
        r.emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: Arc::from("c1"),
            parent_id: None,
            shape: CompositionShape::Sequential,
            branch_count: 2,
        }))
        .await;
        r.emit(Arc::new(BorgEvent::Cancelled {
            composition_id: Arc::from("c1"),
            reason: "test".into(),
        }))
        .await;
        assert_eq!(r.kinds(), vec!["composition_started", "cancelled"]);
    }

    #[test]
    fn every_kind_has_its_own_bit_and_matches_its_name() {
        let mut seen = 0u32;
        for k in EventKind::ALL {
            assert_eq!(seen & k.bit(), 0, "{k:?} shares a bit");
            seen |= k.bit();
        }
        let e = BorgEvent::Cancelled {
            composition_id: Arc::from("x"),
            reason: "r".into(),
        };
        assert_eq!(e.event_kind(), EventKind::Cancelled);
        assert!(EventKinds::ALL.admits(&e));
        assert!(!EventKinds::NONE.admits(&e));
        assert!(!EventKinds::ALL.without(EventKind::Cancelled).admits(&e));
    }

    #[tokio::test]
    async fn filtered_emitter_forwards_only_kept_kinds() {
        let rec = Arc::new(RecordingEmitter::new());
        let f = FilteredEmitter::kinds(rec.clone(), EventKinds::only(&[EventKind::Cancelled]));
        f.emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: Arc::from("c"),
            parent_id: None,
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;
        f.emit(Arc::new(BorgEvent::Cancelled {
            composition_id: Arc::from("c"),
            reason: "r".into(),
        }))
        .await;
        assert_eq!(rec.kinds(), vec!["cancelled"]);
    }

    #[test]
    fn kind_strings_are_distinct() {
        let sample = [
            BorgEvent::CompositionStarted {
                composition_id: Arc::from("x"),
                parent_id: None,
                shape: CompositionShape::Single,
                branch_count: 1,
            },
            BorgEvent::Cancelled {
                composition_id: Arc::from("x"),
                reason: "test".into(),
            },
        ];
        let kinds: Vec<&str> = sample.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["composition_started", "cancelled"]);
    }

    #[test]
    fn new_event_kind_strings() {
        let evts = [
            BorgEvent::TurnStarted {
                composition_id: Arc::from("c1"),
                step_name: Arc::from("s"),
                turn_index: 0,
                model: Arc::from("claude"),
                input_tokens: 100,
            },
            BorgEvent::TurnCompleted {
                composition_id: Arc::from("c1"),
                step_name: Arc::from("s"),
                turn_index: 0,
                output_tokens: 50,
                cache_read_tokens: 10,
                cache_creation_tokens: 5,
                first_token_ms: None,
            },
            BorgEvent::ToolCalled {
                composition_id: Arc::from("c1"),
                step_name: Arc::from("s"),
                turn_index: 0,
                call_id: Arc::from("call-1"),
                tool_name: Arc::from("read_file"),
                batch_id: None,
            },
            BorgEvent::ToolReturned {
                composition_id: Arc::from("c1"),
                step_name: Arc::from("s"),
                turn_index: 0,
                call_id: Arc::from("call-1"),
                tool_name: Arc::from("read_file"),
                success: true,
                error: None,
                batch_id: None,
            },
            BorgEvent::GuardrailTriggered {
                composition_id: Arc::from("c1"),
                step_name: Arc::from("s"),
                rule_name: Arc::from("loop_detector"),
                verdict: "Warn".into(),
                signal: "agent:text_only_turn".into(),
            },
        ];
        let kinds: Vec<&str> = evts.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "turn_started",
                "turn_completed",
                "tool_called",
                "tool_returned",
                "guardrail_triggered",
            ]
        );
    }

    #[test]
    fn token_fields_are_u64() {
        let _e = BorgEvent::TurnStarted {
            composition_id: Arc::from("c"),
            step_name: Arc::from("s"),
            turn_index: 0,
            model: Arc::from("m"),
            input_tokens: u64::MAX,
        };
        let _e = BorgEvent::TurnCompleted {
            composition_id: Arc::from("c"),
            step_name: Arc::from("s"),
            turn_index: 0,
            output_tokens: u64::MAX,
            cache_read_tokens: u64::MAX,
            cache_creation_tokens: u64::MAX,
            first_token_ms: None,
        };
    }
}
