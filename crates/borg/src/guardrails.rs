//! Behavioral guardrails — the fourth kernel primitive after [`Budget`],
//! [`CircuitBreaker`], and [`Meltdown`].
//!
//! [`Budget`]: crate::Budget
//! [`CircuitBreaker`]: crate::CircuitBreaker
//! [`Meltdown`]: crate::MeltdownLedger
//!
//! Budget caps total spend, the circuit breaker reacts to call-level
//! failure rate, and the meltdown ledger reacts to restart storms. None
//! observe what the *model* is doing — re-reading the same file, looping
//! on a broken build, returning text without tool calls, repeating its
//! answer. Guardrails fill that slot: rules see windowed behavioral
//! signals and emit a [`GuardrailVerdict`] that the executor enforces at
//! turn boundaries.
//!
//! ## Pieces
//!
//! ```text
//!   tools / agents ── LoopSignal ──▶ LoopDetector
//!                                         │
//!                                         │  GuardrailCtx
//!                                         ▼
//!                                  GuardrailRule × N
//!                                         │
//!                                         │  Verdict (max-severity wins)
//!                                         ▼
//!                              ┌───────────────────────┐
//!                              │   GuardrailEngine     │
//!                              └───────────┬───────────┘
//!                                          │
//!                            ┌─────────────┼─────────────┐
//!                            ▼             ▼             ▼
//!                          Warn       Throttle         Halt
//!                       (corrective) (corrective +    (terminate
//!                                     max_tokens cap)  session)
//! ```
//!
//! - [`LoopSignal`] — a behavioral event (`"tool:read_file:invoked"`,
//!   `"model:text_only"`) with optional `payload_hash`.
//! - [`LoopDetector`] — aggregates signals over sliding windows and
//!   consecutive runs. The default [`WindowedCounterDetector`] is a
//!   bounded in-memory log; consumers can plug in n-gram or
//!   embedding-similarity detectors without changing the rule API.
//! - [`GuardrailVerdict`] — severity-ordered: `Ok < Warn < Throttle < Halt`.
//! - [`GuardrailRule`] — pure decision function: detector state in,
//!   `GuardrailVerdict` out.
//! - [`GuardrailEngine`] — registry of rules. `evaluate()` runs every
//!   enabled rule, returns the max-severity verdict, and collects
//!   [`BoundaryMessage`]s from all non-`Ok` rules (so multiple `Warn`s
//!   reach the next prompt even when one rule's `Halt` wins enforcement).
//!
//! ## What's NOT in here
//!
//! - Prompt strings. Rules emit `corrective: String`; the consumer's
//!   agent decides how to weave that into the next `user_prompt`.
//! - Cancel reasons. `Halt` surfaces a typed reason string; the
//!   product layer maps that to its own terminal taxonomy.
//! - Wire format. Borg does not persist signals, verdicts, or messages.
//! - Tool-name-specific defaults. Built-in rules take the tool name as
//!   config; there is no `BuildToolRule` baked into the kernel.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ── Signal ───────────────────────────────────────────────────────────────

/// A behavioral event recorded against a [`LoopDetector`].
///
/// Convention for `key`:
/// - `"tool:<name>:invoked"` — tool was dispatched
/// - `"tool:<name>:failed"` — tool returned error
/// - `"tool:<name>:succeeded"` — tool returned ok
/// - `"model:text_only"` — turn ended with text content, zero tool calls
/// - `"model:end_turn"` — provider stop reason was `end_turn`
/// - `"agent:no_state_change"` — round produced no observable mutation
///
/// Borg does not interpret keys; rules do.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoopSignal {
    /// Stable string key. Use a `&'static str` literal to avoid allocating.
    pub key: Cow<'static, str>,

    /// Optional content hash, used by `consecutive_identical` to detect
    /// the model repeating itself or a tool returning the same output
    /// twice. Consumers hash whatever's meaningful; borg never inspects
    /// the bytes.
    pub payload_hash: Option<u64>,

    /// Wall-clock at signal recording. See [`LoopSignal::now`].
    pub at: Instant,
}

impl LoopSignal {
    /// Construct a signal at "now" with no payload hash.
    pub fn now(key: impl Into<Cow<'static, str>>) -> Self {
        Self {
            key: key.into(),
            payload_hash: None,
            at: Instant::now(),
        }
    }

    /// Construct a signal at "now" with a payload hash.
    pub fn now_with_hash(key: impl Into<Cow<'static, str>>, payload_hash: u64) -> Self {
        Self {
            key: key.into(),
            payload_hash: Some(payload_hash),
            at: Instant::now(),
        }
    }
}

// ── Detector ─────────────────────────────────────────────────────────────

/// Aggregates [`LoopSignal`]s into queryable counts.
///
/// All methods are `&self` because detector state is internally
/// synchronized. Cheap to share across parallel branches via
/// `Arc<dyn LoopDetector>`.
pub trait LoopDetector: Send + Sync {
    /// Record a new signal.
    fn record(&self, signal: LoopSignal);

    /// Count of signals with `key` whose `at` is within the last
    /// `window` (inclusive of `now - window`, exclusive of older).
    fn count(&self, key: &str, window: Duration) -> u32;

    /// Count of the most recent signals with `key`, going back from the
    /// latest signal *overall* and stopping at the first differing key.
    /// Returns 0 if the most recent signal's key is not `key`.
    fn consecutive(&self, key: &str) -> u32;

    /// Payload hash of the most recent signal under `key`, or `None` if
    /// no such signal exists or the latest one had no hash.
    fn last_payload_hash(&self, key: &str) -> Option<u64>;

    /// Count of the most recent signals under `key` (ignoring
    /// interleaved other-key signals) whose `payload_hash` equals the
    /// latest's. Returns 0 if the latest signal under `key` has no hash.
    fn consecutive_identical(&self, key: &str) -> u32;

    /// Like [`consecutive`](Self::consecutive) but only considers
    /// signals whose key is `target` or in `breakers`; keys in neither
    /// set are skipped without affecting the run.
    ///
    /// Lets a `tool:X:failed` streak count among the `{failed,
    /// succeeded}` subset: `tool:X:invoked` signals don't reset the run,
    /// but a `tool:X:succeeded` does. The default impl is built on the
    /// other trait methods; custom detectors can override for efficiency.
    fn consecutive_within(&self, target: &str, breakers: &[&str]) -> u32;

    /// Total signals recorded across all keys.
    fn total_recorded(&self) -> u64;
}

/// Default [`LoopDetector`] backed by a bounded ring buffer.
///
/// Stores at most `max_history` signals (FIFO eviction on overflow).
/// Queries scan the buffer; cheap given typical session sizes.
#[derive(Debug)]
pub struct WindowedCounterDetector {
    state: Mutex<DetectorState>,
    max_history: usize,
}

#[derive(Debug)]
struct DetectorState {
    log: VecDeque<LoopSignal>,
    total: u64,
}

impl WindowedCounterDetector {
    /// Default cap: 4096 signals — ample for a session, small enough
    /// that linear scans are negligible.
    pub const DEFAULT_MAX_HISTORY: usize = 4096;

    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_MAX_HISTORY)
    }

    pub fn with_capacity(max_history: usize) -> Self {
        Self {
            state: Mutex::new(DetectorState {
                log: VecDeque::with_capacity(max_history.min(256)),
                total: 0,
            }),
            max_history,
        }
    }

    /// Build a sharable detector as `Arc<dyn LoopDetector>`.
    pub fn shared() -> std::sync::Arc<dyn LoopDetector> {
        std::sync::Arc::new(Self::new())
    }
}

impl Default for WindowedCounterDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopDetector for WindowedCounterDetector {
    fn record(&self, signal: LoopSignal) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if g.log.len() >= self.max_history {
            g.log.pop_front();
        }
        g.log.push_back(signal);
        g.total = g.total.saturating_add(1);
    }

    fn count(&self, key: &str, window: Duration) -> u32 {
        let g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let cutoff = Instant::now().checked_sub(window);
        let mut n: u32 = 0;
        for s in g.log.iter().rev() {
            if let Some(c) = cutoff {
                if s.at < c {
                    break;
                }
            }
            if s.key.as_ref() == key {
                n = n.saturating_add(1);
            }
        }
        n
    }

    fn consecutive(&self, key: &str) -> u32 {
        let g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut n: u32 = 0;
        for s in g.log.iter().rev() {
            if s.key.as_ref() == key {
                n = n.saturating_add(1);
            } else {
                break;
            }
        }
        n
    }

    fn last_payload_hash(&self, key: &str) -> Option<u64> {
        let g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for s in g.log.iter().rev() {
            if s.key.as_ref() == key {
                return s.payload_hash;
            }
        }
        None
    }

    fn consecutive_identical(&self, key: &str) -> u32 {
        let g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut iter = g.log.iter().rev().filter(|s| s.key.as_ref() == key);
        let Some(first) = iter.next() else {
            return 0;
        };
        let Some(target) = first.payload_hash else {
            return 0;
        };
        let mut n: u32 = 1;
        for s in iter {
            if s.payload_hash == Some(target) {
                n = n.saturating_add(1);
            } else {
                break;
            }
        }
        n
    }

    fn consecutive_within(&self, target: &str, breakers: &[&str]) -> u32 {
        let g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut n: u32 = 0;
        for s in g.log.iter().rev() {
            let k = s.key.as_ref();
            if k == target {
                n = n.saturating_add(1);
            } else if breakers.contains(&k) {
                break;
            }
            // else: outside the subset, skip without breaking the run.
        }
        n
    }

    fn total_recorded(&self) -> u64 {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).total
    }
}

// ── Verdict ──────────────────────────────────────────────────────────────

/// Severity-ordered guardrail verdict. Higher variants win when
/// multiple rules fire at the same boundary.
///
/// Named `GuardrailVerdict` to disambiguate from the agent-call
/// [`Verdict`](crate::Verdict) returned by [`BorgAgent`](crate::BorgAgent).
///
/// Severity ladder:
/// - `Ok` (0) — no action
/// - `Warn` (1) — surface a corrective message; next turn proceeds normally
/// - `Throttle` (2) — surface a corrective AND clamp next turn's `max_tokens`
/// - `Halt` (3) — force-terminate the session; the supervisor cancels
///   in-flight work and surfaces the reason to the consumer
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GuardrailVerdict {
    Ok,
    Warn { corrective: String },
    Throttle { corrective: String, max_tokens: u32 },
    Halt { reason: String },
}

impl GuardrailVerdict {
    /// 0..=3. Used for merging multiple rules' verdicts.
    pub fn severity(&self) -> u8 {
        match self {
            GuardrailVerdict::Ok => 0,
            GuardrailVerdict::Warn { .. } => 1,
            GuardrailVerdict::Throttle { .. } => 2,
            GuardrailVerdict::Halt { .. } => 3,
        }
    }

    /// True iff this verdict will halt the session.
    pub fn is_halt(&self) -> bool {
        matches!(self, GuardrailVerdict::Halt { .. })
    }

    /// True iff this verdict requires no action.
    pub fn is_ok(&self) -> bool {
        matches!(self, GuardrailVerdict::Ok)
    }
}

// ── Message ──────────────────────────────────────────────────────────────

/// Distinguishes a boundary message's source verdict. Consumers may
/// render `Warn` and `Throttle` messages differently (icon, color,
/// placement in the prompt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundaryMessageKind {
    Warn,
    Throttle,
}

/// A corrective message surfaced at the turn boundary. The agent's
/// `user_prompt(ctx)` impl drains pending messages from a
/// [`BoundaryMessageQueue`] and weaves them into the next prompt — borg
/// never authors prompt text itself.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BoundaryMessage {
    /// Name of the rule that produced the message.
    pub from: Cow<'static, str>,
    pub kind: BoundaryMessageKind,
    pub body: String,
}

/// FIFO queue of pending [`BoundaryMessage`]s, drained at the next turn
/// boundary by the agent. `drain()` empties the queue, so messages do
/// not stick across turns. Cheap to share via `Arc<BoundaryMessageQueue>`.
#[derive(Debug, Default)]
pub struct BoundaryMessageQueue {
    inner: Mutex<Vec<BoundaryMessage>>,
}

impl BoundaryMessageQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, msg: BoundaryMessage) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(msg);
    }

    /// Take all pending messages, leaving the queue empty.
    pub fn drain(&self) -> Vec<BoundaryMessage> {
        std::mem::take(&mut *self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Rule ─────────────────────────────────────────────────────────────────

/// Context passed to [`GuardrailRule::evaluate`] at each boundary.
///
/// Holds only borrowed references — rules must not retain them across
/// invocations. The `&dyn Budget` lets cost-aware rules project spend.
#[non_exhaustive]
pub struct GuardrailCtx<'a> {
    pub detector: &'a dyn LoopDetector,
    pub budget: &'a dyn crate::Budget,
    /// 0-indexed turn within the current session.
    pub round: u32,
    /// Wall-clock elapsed since session start.
    pub elapsed: Duration,
}

impl<'a> GuardrailCtx<'a> {
    /// What a rule sees at one agent-loop boundary. The executor builds
    /// this; call it yourself only to drive a `GuardrailEngine` directly.
    #[must_use]
    pub fn new(
        detector: &'a dyn LoopDetector,
        budget: &'a dyn crate::Budget,
        round: u32,
        elapsed: Duration,
    ) -> Self {
        Self {
            detector,
            budget,
            round,
            elapsed,
        }
    }
}

/// A pure decision function: detector state in, [`GuardrailVerdict`] out.
///
/// Implementations should be cheap — `evaluate` may be called many
/// times per session. They MUST NOT block on I/O. Side effects belong
/// in the consumer's boundary handler, not in the rule.
pub trait GuardrailRule: Send + Sync {
    /// Stable rule name; appears in [`BoundaryMessage::from`] and
    /// telemetry. Convention: snake_case.
    fn name(&self) -> &str;

    /// Static-string variant of [`Self::name`]. When it returns `Some`,
    /// [`GuardrailEngine::evaluate`] stores the name as `Cow::Borrowed`
    /// instead of allocating a `String` per non-Ok verdict. Override
    /// whenever your `name()` returns a `&'static str`.
    fn name_static(&self) -> Option<&'static str> {
        None
    }

    fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict;

    /// Disable without removing from the registry, e.g. behind a config
    /// flag. Default `true`.
    fn enabled(&self) -> bool {
        true
    }
}

/// Resolve a rule's name into a `Cow<'static, str>`, preferring the
/// allocation-free [`GuardrailRule::name_static`] path when available.
fn rule_name_cow(rule: &dyn GuardrailRule) -> Cow<'static, str> {
    match rule.name_static() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(rule.name().to_owned()),
    }
}

// ── Engine ───────────────────────────────────────────────────────────────

/// Outcome of [`GuardrailEngine::evaluate`].
///
/// The `winning_verdict` is the maximum-severity verdict across all
/// enabled rules — what the executor enforces. `messages` includes
/// corrective text from every non-`Ok` rule, so multiple `Warn`s reach
/// the next prompt even when a different rule's `Halt` decides
/// enforcement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvaluationOutcome {
    pub winning_verdict: GuardrailVerdict,
    pub winning_rule: Option<Cow<'static, str>>,
    pub messages: Vec<BoundaryMessage>,
}

impl EvaluationOutcome {
    /// Route every collected message into a queue.
    pub fn flush_into(&self, queue: &BoundaryMessageQueue) {
        for m in &self.messages {
            queue.push(m.clone());
        }
    }
}

/// Registry of [`GuardrailRule`]s. Evaluates them all at boundary
/// points, returns max-severity verdict.
///
/// Rules are evaluated in registration order; ties at the same
/// severity break by registration order (first-registered wins
/// enforcement). All non-`Ok` verdicts contribute their corrective
/// message to the outcome regardless of who wins.
#[derive(Default)]
pub struct GuardrailEngine {
    rules: Vec<Box<dyn GuardrailRule>>,
}

impl std::fmt::Debug for GuardrailEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailEngine")
            .field("rule_count", &self.rules.len())
            .field(
                "rules",
                &self.rules.iter().map(|r| r.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl GuardrailEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style rule registration.
    #[must_use]
    pub fn with_rule(mut self, rule: impl GuardrailRule + 'static) -> Self {
        self.rules.push(Box::new(rule));
        self
    }

    /// In-place registration, for conditionally adding rules.
    pub fn register(&mut self, rule: impl GuardrailRule + 'static) {
        self.rules.push(Box::new(rule));
    }

    /// Run every enabled rule, merge verdicts by max severity, collect
    /// corrective messages.
    pub fn evaluate(&self, ctx: GuardrailCtx<'_>) -> EvaluationOutcome {
        let mut winning: GuardrailVerdict = GuardrailVerdict::Ok;
        let mut winning_name: Option<Cow<'static, str>> = None;
        // At most one message per rule.
        let mut messages: Vec<BoundaryMessage> = Vec::with_capacity(self.rules.len());

        for rule in &self.rules {
            if !rule.enabled() {
                continue;
            }
            let sub = GuardrailCtx {
                detector: ctx.detector,
                budget: ctx.budget,
                round: ctx.round,
                elapsed: ctx.elapsed,
            };
            let v = rule.evaluate(sub);
            match &v {
                GuardrailVerdict::Warn { corrective } => messages.push(BoundaryMessage {
                    from: rule_name_cow(rule.as_ref()),
                    kind: BoundaryMessageKind::Warn,
                    body: corrective.clone(),
                }),
                GuardrailVerdict::Throttle { corrective, .. } => messages.push(BoundaryMessage {
                    from: rule_name_cow(rule.as_ref()),
                    kind: BoundaryMessageKind::Throttle,
                    body: corrective.clone(),
                }),
                GuardrailVerdict::Halt { .. } | GuardrailVerdict::Ok => {}
            }
            if v.severity() > winning.severity() {
                winning_name = Some(rule_name_cow(rule.as_ref()));
                winning = v;
            }
        }

        EvaluationOutcome {
            winning_verdict: winning,
            winning_rule: winning_name,
            messages,
        }
    }

    /// Number of registered rules (regardless of enabled state).
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── Harness ──────────────────────────────────────────────────────────────

/// The runtime-wired bundle: engine + detector + boundary queue + an
/// atomic `max_tokens_override` slot.
///
/// The executor takes one via [`crate::ExecCtx::with_guardrails`] and
/// threads it into every [`crate::AgentContext`]. Agents call
/// `ctx.effective_max_tokens(self.max_tokens())` to honor `Throttle`
/// clamps and `ctx.boundary_messages()` to read corrective text.
///
/// One harness per session — cheap to clone (`Arc<GuardrailHarness>`).
pub struct GuardrailHarness {
    pub engine: std::sync::Arc<GuardrailEngine>,
    pub detector: std::sync::Arc<dyn LoopDetector>,
    pub queue: std::sync::Arc<BoundaryMessageQueue>,
    /// Per-turn `max_tokens` override set by `Throttle`; `0` means no
    /// override. Read-and-clear via
    /// [`GuardrailHarness::take_max_tokens_override`].
    max_tokens_override: std::sync::atomic::AtomicU32,
    /// Wall-clock at construction; fills `GuardrailCtx::elapsed`.
    started_at: Instant,
}

impl std::fmt::Debug for GuardrailHarness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailHarness")
            .field("engine", &self.engine)
            .field("queue_len", &self.queue.len())
            .field(
                "max_tokens_override",
                &self
                    .max_tokens_override
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("elapsed", &self.started_at.elapsed())
            .finish()
    }
}

impl GuardrailHarness {
    /// Build a harness with a fresh default detector and queue.
    pub fn new(engine: GuardrailEngine) -> Self {
        Self {
            engine: std::sync::Arc::new(engine),
            detector: std::sync::Arc::new(WindowedCounterDetector::new()),
            queue: std::sync::Arc::new(BoundaryMessageQueue::new()),
            max_tokens_override: std::sync::atomic::AtomicU32::new(0),
            started_at: Instant::now(),
        }
    }

    /// Replace the default detector, e.g. with an n-gram or
    /// similarity-based implementation.
    #[must_use]
    pub fn with_detector(mut self, detector: std::sync::Arc<dyn LoopDetector>) -> Self {
        self.detector = detector;
        self
    }

    /// Record a behavioral signal.
    pub fn record(&self, signal: LoopSignal) {
        self.detector.record(signal);
    }

    /// Evaluate the engine at a turn boundary: push every boundary
    /// message into the queue and, on `Throttle`, set the
    /// `max_tokens_override` slot. The returned outcome lets the caller
    /// react to `Halt`.
    pub fn evaluate(&self, round: u32, budget: &dyn crate::Budget) -> EvaluationOutcome {
        let outcome = self.engine.evaluate(GuardrailCtx {
            detector: self.detector.as_ref(),
            budget,
            round,
            elapsed: self.started_at.elapsed(),
        });
        outcome.flush_into(&self.queue);
        if let GuardrailVerdict::Throttle { max_tokens, .. } = &outcome.winning_verdict {
            self.max_tokens_override
                .store(*max_tokens, std::sync::atomic::Ordering::Release);
        }
        outcome
    }

    /// Read and clear the per-turn `max_tokens` override, or `None` if
    /// unset. Consume-once: the slot resets to 0 so a `Throttle` only
    /// clamps the immediately-next turn.
    pub fn take_max_tokens_override(&self) -> Option<u32> {
        let prev = self
            .max_tokens_override
            .swap(0, std::sync::atomic::Ordering::Acquire);
        if prev == 0 {
            None
        } else {
            Some(prev)
        }
    }

    /// Apply the current `Throttle` override (if any) on top of an
    /// agent's requested `max_tokens`, returning the lesser of the two
    /// so a throttle can never *raise* the cap.
    pub fn effective_max_tokens(&self, requested: u32) -> u32 {
        match self.take_max_tokens_override() {
            Some(throttle) => requested.min(throttle),
            None => requested,
        }
    }
}

// ── Built-in rules ──────────────────────────────────────────────────────

/// Built-in generic rules, each a [`GuardrailRule`] impl with config
/// fields. Register what you want on [`GuardrailEngine`]:
///
/// ```
/// use borg::guardrails::{rules::*, GuardrailEngine};
/// use std::time::Duration;
///
/// let engine = GuardrailEngine::new()
///     .with_rule(MaxContinuationsRule::new(30))
///     .with_rule(TextOnlyEndTurnRule::new(3))
///     .with_rule(RepeatedToolCallRule::new("read_file", 3, 5, 8).with_window(Duration::from_secs(60)))
///     .with_rule(IdenticalOutputRule::new("model:text", 3))
///     .with_rule(ConsecutiveFailureRule::new("build_project", 3, 6))
///     .with_rule(RunawayCostRule::new(0.95).with_warn_at(0.80));
/// # let _ = engine;
/// ```
///
/// The config structs are `#[non_exhaustive]`: build them with `new(…)`
/// (plus the `with_*` setters), not struct literals. Fields stay public
/// for reading and in-place tweaks.
pub mod rules {
    use super::{GuardrailCtx, GuardrailRule, GuardrailVerdict};
    use std::borrow::Cow;
    use std::time::Duration;

    /// Hard cap on continuation turns within a session; fires `Halt`
    /// when the count of completed turns reaches `limit`.
    #[derive(Debug, Clone)]
    #[non_exhaustive]
    pub struct MaxContinuationsRule {
        pub limit: u32,
    }

    impl MaxContinuationsRule {
        /// Halt once `limit` turns have completed.
        pub fn new(limit: u32) -> Self {
            Self { limit }
        }
    }

    impl GuardrailRule for MaxContinuationsRule {
        fn name(&self) -> &str {
            "max_continuations"
        }
        fn name_static(&self) -> Option<&'static str> {
            Some("max_continuations")
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            // `round` is 0-indexed and eval fires at the END of round N,
            // so `round + 1` is the count of completed turns: `limit`
            // means "at most N completed turns."
            if ctx.round.saturating_add(1) >= self.limit {
                GuardrailVerdict::Halt {
                    reason: format!(
                        "max-continuations cap reached ({} turns) — session terminated",
                        self.limit
                    ),
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }

    /// Fires when the same tool was invoked N times within a window.
    ///
    /// Uses `count(window)`, not `consecutive`, so interleaving a
    /// recovery call between same-tool spam still trips it — the failure
    /// mode is budget burn on the tool, not literal back-to-back calls.
    #[derive(Debug, Clone)]
    #[non_exhaustive]
    pub struct RepeatedToolCallRule {
        pub tool: Cow<'static, str>,
        pub warn_at: u32,
        pub throttle_at: u32,
        pub halt_at: u32,
        pub window: Duration,
        pub warn_msg: Cow<'static, str>,
        pub throttle_msg: Cow<'static, str>,
        pub throttle_max_tokens: u32,
    }

    impl RepeatedToolCallRule {
        /// Conservative defaults: 120-second window, 2 KB throttle cap,
        /// generic messages. Thresholds are usually overridden per tool.
        pub fn new(
            tool: impl Into<Cow<'static, str>>,
            warn_at: u32,
            throttle_at: u32,
            halt_at: u32,
        ) -> Self {
            let tool = tool.into();
            let warn_msg: Cow<'static, str> =
                format!("You've called `{tool}` repeatedly. Use the content you already have.")
                    .into();
            let throttle_msg: Cow<'static, str> = format!(
                "Stop re-invoking `{tool}`. Make a decision with the context you have or stop."
            )
            .into();
            Self {
                tool,
                warn_at,
                throttle_at,
                halt_at,
                window: Duration::from_secs(120),
                warn_msg,
                throttle_msg,
                throttle_max_tokens: 2_000,
            }
        }

        /// Sliding window the invocation count is taken over (default
        /// 120 s).
        #[must_use]
        pub fn with_window(mut self, window: Duration) -> Self {
            self.window = window;
            self
        }

        /// Replace the corrective text surfaced on `Warn` and `Throttle`.
        #[must_use]
        pub fn with_messages(
            mut self,
            warn_msg: impl Into<Cow<'static, str>>,
            throttle_msg: impl Into<Cow<'static, str>>,
        ) -> Self {
            self.warn_msg = warn_msg.into();
            self.throttle_msg = throttle_msg.into();
            self
        }

        /// `max_tokens` clamp applied on `Throttle` (default 2 000).
        #[must_use]
        pub fn with_throttle_max_tokens(mut self, max_tokens: u32) -> Self {
            self.throttle_max_tokens = max_tokens;
            self
        }
    }

    impl GuardrailRule for RepeatedToolCallRule {
        fn name(&self) -> &str {
            "repeated_tool_call"
        }
        fn name_static(&self) -> Option<&'static str> {
            Some("repeated_tool_call")
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            let key = format!("tool:{}:invoked", self.tool);
            let n = ctx.detector.count(&key, self.window);
            if self.halt_at > 0 && n >= self.halt_at {
                GuardrailVerdict::Halt {
                    reason: format!(
                        "`{}` invoked {n} times in {:?} — halting to prevent budget burn",
                        self.tool, self.window
                    ),
                }
            } else if self.throttle_at > 0 && n >= self.throttle_at {
                GuardrailVerdict::Throttle {
                    corrective: self.throttle_msg.to_string(),
                    max_tokens: self.throttle_max_tokens,
                }
            } else if self.warn_at > 0 && n >= self.warn_at {
                GuardrailVerdict::Warn {
                    corrective: self.warn_msg.to_string(),
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }

    /// Fires when the latest `repeats_to_halt` signals under `key`
    /// share the same `payload_hash` — i.e. the model is repeating its
    /// previous response, or a tool keeps returning the same output.
    #[derive(Debug, Clone)]
    #[non_exhaustive]
    pub struct IdenticalOutputRule {
        pub key: Cow<'static, str>,
        pub repeats_to_halt: u32,
    }

    impl IdenticalOutputRule {
        pub fn new(key: impl Into<Cow<'static, str>>, repeats_to_halt: u32) -> Self {
            Self {
                key: key.into(),
                repeats_to_halt,
            }
        }
    }

    impl GuardrailRule for IdenticalOutputRule {
        fn name(&self) -> &str {
            "identical_output"
        }
        fn name_static(&self) -> Option<&'static str> {
            Some("identical_output")
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            let n = ctx.detector.consecutive_identical(&self.key);
            if n >= self.repeats_to_halt {
                GuardrailVerdict::Halt {
                    reason: format!(
                        "`{}` returned identical output {n}× consecutively",
                        self.key
                    ),
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }

    /// Fires when the model ends its turn text-only (no tool calls)
    /// `halt_after` times in a row — the "stopped using tools but didn't
    /// finish" failure mode.
    ///
    /// Requires the executor or consumer to emit `"agent:text_only_turn"`
    /// signals at end-of-turn when the model produced no tool calls.
    #[derive(Debug, Clone)]
    #[non_exhaustive]
    pub struct TextOnlyEndTurnRule {
        pub halt_after: u32,
    }

    impl TextOnlyEndTurnRule {
        /// Halt after `halt_after` consecutive text-only turns.
        pub fn new(halt_after: u32) -> Self {
            Self { halt_after }
        }
    }

    impl GuardrailRule for TextOnlyEndTurnRule {
        fn name_static(&self) -> Option<&'static str> {
            Some("text_only_end_turn")
        }
        fn name(&self) -> &str {
            "text_only_end_turn"
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            let n = ctx.detector.consecutive("agent:text_only_turn");
            if n >= self.halt_after {
                GuardrailVerdict::Halt {
                    reason: format!(
                        "model produced text-only responses {n}× in a row \
                         without finishing — halting"
                    ),
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }

    /// Fires when budget consumption crosses a fraction of the limit.
    ///
    /// Unlike [`crate::BorgError::BudgetExhausted`], which fires *after*
    /// a call overspends, this trips at the boundary *before* the next
    /// turn — giving the agent a chance to wrap up rather than hit a hard
    /// wall mid-call.
    #[derive(Debug, Clone)]
    #[non_exhaustive]
    pub struct RunawayCostRule {
        /// `0.0..=1.0`. Halt when `consumed / limit >= halt_at_consumed_fraction`.
        pub halt_at_consumed_fraction: f32,
        /// Optional softer warning threshold (e.g. 0.80). `None` to disable.
        pub warn_at_consumed_fraction: Option<f32>,
    }

    impl RunawayCostRule {
        /// Halt when `consumed / limit >= halt_at_consumed_fraction`; no
        /// warning threshold (add one with [`with_warn_at`](Self::with_warn_at)).
        pub fn new(halt_at_consumed_fraction: f32) -> Self {
            Self {
                halt_at_consumed_fraction,
                warn_at_consumed_fraction: None,
            }
        }

        /// Also `Warn` once consumption crosses `fraction` (`0.0..=1.0`).
        #[must_use]
        pub fn with_warn_at(mut self, fraction: f32) -> Self {
            self.warn_at_consumed_fraction = Some(fraction);
            self
        }
    }

    impl GuardrailRule for RunawayCostRule {
        fn name(&self) -> &str {
            "runaway_cost"
        }
        fn name_static(&self) -> Option<&'static str> {
            Some("runaway_cost")
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            let limit = ctx.budget.limit();
            if limit == 0 {
                return GuardrailVerdict::Ok;
            }
            let consumed = ctx.budget.consumed();
            let frac = (consumed as f64) / (limit as f64);
            if frac >= self.halt_at_consumed_fraction as f64 {
                GuardrailVerdict::Halt {
                    reason: format!(
                        "budget at {:.0}% (>= {:.0}% halt threshold) — terminating",
                        frac * 100.0,
                        self.halt_at_consumed_fraction * 100.0
                    ),
                }
            } else if let Some(w) = self.warn_at_consumed_fraction {
                if frac >= w as f64 {
                    GuardrailVerdict::Warn {
                        corrective: format!(
                            "budget at {:.0}% — wrap up the remaining work concisely",
                            frac * 100.0
                        ),
                    }
                } else {
                    GuardrailVerdict::Ok
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }

    /// Halt after N consecutive failures of a specific tool. The failure
    /// run resets on any intervening `tool:<name>:succeeded`.
    #[derive(Debug, Clone)]
    #[non_exhaustive]
    pub struct ConsecutiveFailureRule {
        pub tool: Cow<'static, str>,
        pub warn_at: u32,
        pub halt_at: u32,
    }

    impl ConsecutiveFailureRule {
        pub fn new(tool: impl Into<Cow<'static, str>>, warn_at: u32, halt_at: u32) -> Self {
            Self {
                tool: tool.into(),
                warn_at,
                halt_at,
            }
        }
    }

    impl GuardrailRule for ConsecutiveFailureRule {
        fn name(&self) -> &str {
            "consecutive_failure"
        }
        fn name_static(&self) -> Option<&'static str> {
            Some("consecutive_failure")
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            // `consecutive_within` so per-dispatch `tool:<X>:invoked`
            // signals don't break the failure run — only a
            // `tool:<X>:succeeded` does.
            let failed = format!("tool:{}:failed", self.tool);
            let succeeded = format!("tool:{}:succeeded", self.tool);
            let n = ctx
                .detector
                .consecutive_within(&failed, &[succeeded.as_str()]);
            if self.halt_at > 0 && n >= self.halt_at {
                GuardrailVerdict::Halt {
                    reason: format!("`{}` failed {n}× consecutively", self.tool),
                }
            } else if self.warn_at > 0 && n >= self.warn_at {
                GuardrailVerdict::Warn {
                    corrective: format!(
                        "`{}` has failed {n}× in a row — re-read the error and try a different fix",
                        self.tool
                    ),
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Budget;
    use std::sync::Arc;

    fn det() -> WindowedCounterDetector {
        WindowedCounterDetector::new()
    }

    fn budget() -> Arc<dyn Budget> {
        crate::in_memory_budget(10_000)
    }

    fn ctx<'a>(d: &'a dyn LoopDetector, b: &'a dyn Budget) -> GuardrailCtx<'a> {
        GuardrailCtx {
            detector: d,
            budget: b,
            round: 0,
            elapsed: Duration::ZERO,
        }
    }

    // ── LoopSignal / detector basics ─────────────────────────────────────

    #[test]
    fn loop_signal_now_has_no_payload_hash() {
        let s = LoopSignal::now("tool:read_file:invoked");
        assert_eq!(s.key.as_ref(), "tool:read_file:invoked");
        assert_eq!(s.payload_hash, None);
    }

    #[test]
    fn detector_records_and_counts_total() {
        let d = det();
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("b"));
        assert_eq!(d.total_recorded(), 3);
    }

    #[test]
    fn count_returns_only_signals_in_window() {
        let d = det();
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("b"));
        d.record(LoopSignal::now("a"));
        assert_eq!(d.count("a", Duration::from_secs(60)), 3);
        assert_eq!(d.count("b", Duration::from_secs(60)), 1);
        assert_eq!(d.count("c", Duration::from_secs(60)), 0);
    }

    #[test]
    fn count_ignores_signals_outside_window() {
        // We can't backdate Instant cleanly, so test the boundary by
        // using a zero-duration window: nothing should match.
        let d = det();
        d.record(LoopSignal::now("a"));
        assert_eq!(d.count("a", Duration::ZERO), 0);
    }

    #[test]
    fn consecutive_counts_same_key_run_from_latest() {
        let d = det();
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("b")); // breaks the a-run
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("a"));
        assert_eq!(d.consecutive("a"), 3);
        assert_eq!(d.consecutive("b"), 0); // most recent is "a"
    }

    #[test]
    fn consecutive_returns_zero_when_latest_signal_differs() {
        let d = det();
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("b"));
        assert_eq!(d.consecutive("a"), 0);
        assert_eq!(d.consecutive("b"), 1);
    }

    #[test]
    fn consecutive_returns_zero_for_unknown_key() {
        let d = det();
        d.record(LoopSignal::now("a"));
        assert_eq!(d.consecutive("never_seen"), 0);
    }

    #[test]
    fn last_payload_hash_returns_most_recent_under_key() {
        let d = det();
        d.record(LoopSignal::now_with_hash("a", 1));
        d.record(LoopSignal::now_with_hash("a", 2));
        d.record(LoopSignal::now("b"));
        d.record(LoopSignal::now_with_hash("a", 3));
        assert_eq!(d.last_payload_hash("a"), Some(3));
        assert_eq!(d.last_payload_hash("b"), None);
    }

    #[test]
    fn consecutive_identical_counts_repeated_hashes_under_same_key() {
        let d = det();
        d.record(LoopSignal::now_with_hash("err", 1));
        d.record(LoopSignal::now("noise")); // different key — ignored by filter
        d.record(LoopSignal::now_with_hash("err", 2));
        d.record(LoopSignal::now_with_hash("err", 2));
        d.record(LoopSignal::now_with_hash("err", 2));
        assert_eq!(d.consecutive_identical("err"), 3);
    }

    #[test]
    fn consecutive_identical_returns_zero_when_latest_has_no_hash() {
        let d = det();
        d.record(LoopSignal::now_with_hash("err", 1));
        d.record(LoopSignal::now_with_hash("err", 1));
        d.record(LoopSignal::now("err")); // no hash on latest
        assert_eq!(d.consecutive_identical("err"), 0);
    }

    #[test]
    fn consecutive_within_skips_irrelevant_keys_but_resets_on_breakers() {
        let d = det();
        // `invoked` is irrelevant noise; `succeeded` is the breaker.
        d.record(LoopSignal::now("tool:x:invoked"));
        d.record(LoopSignal::now("tool:x:failed"));
        d.record(LoopSignal::now("tool:x:invoked"));
        d.record(LoopSignal::now("tool:x:failed"));
        d.record(LoopSignal::now("tool:x:invoked"));
        d.record(LoopSignal::now("tool:x:failed"));
        // From latest: failed, [skip invoked], failed, [skip invoked],
        // failed, [skip invoked], end → 3.
        assert_eq!(
            d.consecutive_within("tool:x:failed", &["tool:x:succeeded"]),
            3
        );
    }

    #[test]
    fn consecutive_within_stops_at_breaker_key() {
        let d = det();
        d.record(LoopSignal::now("tool:x:failed"));
        d.record(LoopSignal::now("tool:x:failed"));
        d.record(LoopSignal::now("tool:x:succeeded")); // breaker
        d.record(LoopSignal::now("tool:x:invoked"));
        d.record(LoopSignal::now("tool:x:failed"));
        // From latest: failed, [skip invoked], succeeded → stop. n = 1.
        assert_eq!(
            d.consecutive_within("tool:x:failed", &["tool:x:succeeded"]),
            1
        );
    }

    #[test]
    fn consecutive_within_returns_zero_when_no_target_present() {
        let d = det();
        d.record(LoopSignal::now("tool:x:invoked"));
        d.record(LoopSignal::now("tool:x:succeeded"));
        // Breaker comes before any failure → 0 failures in run.
        assert_eq!(
            d.consecutive_within("tool:x:failed", &["tool:x:succeeded"]),
            0
        );
    }

    #[test]
    fn detector_evicts_oldest_when_capacity_exceeded() {
        let d = WindowedCounterDetector::with_capacity(3);
        d.record(LoopSignal::now("a"));
        d.record(LoopSignal::now("b"));
        d.record(LoopSignal::now("c"));
        d.record(LoopSignal::now("d")); // evicts "a"
        assert_eq!(d.count("a", Duration::from_secs(60)), 0);
        assert_eq!(d.count("d", Duration::from_secs(60)), 1);
        // total_recorded never decreases on eviction — it's a counter,
        // not a gauge.
        assert_eq!(d.total_recorded(), 4);
    }

    #[tokio::test]
    async fn detector_is_safe_under_concurrent_record() {
        let d: Arc<dyn LoopDetector> = Arc::new(WindowedCounterDetector::with_capacity(100_000));
        let mut handles = Vec::new();
        for _ in 0..20 {
            let dd = d.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..500 {
                    dd.record(LoopSignal::now("hot"));
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(d.total_recorded(), 10_000);
    }

    // ── Verdict severity ────────────────────────────────────────────────

    #[test]
    fn verdict_severity_is_strictly_ordered() {
        assert!(
            GuardrailVerdict::Ok.severity()
                < GuardrailVerdict::Warn {
                    corrective: "x".into()
                }
                .severity()
        );
        assert!(
            GuardrailVerdict::Warn {
                corrective: "x".into()
            }
            .severity()
                < GuardrailVerdict::Throttle {
                    corrective: "x".into(),
                    max_tokens: 1
                }
                .severity()
        );
        assert!(
            GuardrailVerdict::Throttle {
                corrective: "x".into(),
                max_tokens: 1
            }
            .severity()
                < GuardrailVerdict::Halt { reason: "x".into() }.severity()
        );
    }

    #[test]
    fn verdict_predicates_match_variant() {
        assert!(GuardrailVerdict::Ok.is_ok());
        assert!(!GuardrailVerdict::Ok.is_halt());
        assert!(GuardrailVerdict::Halt { reason: "x".into() }.is_halt());
        assert!(!GuardrailVerdict::Halt { reason: "x".into() }.is_ok());
    }

    // ── BoundaryMessage queue ───────────────────────────────────────────

    #[test]
    fn boundary_queue_drain_returns_all_and_empties() {
        let q = BoundaryMessageQueue::new();
        q.push(BoundaryMessage {
            from: "r1".into(),
            kind: BoundaryMessageKind::Warn,
            body: "a".into(),
        });
        q.push(BoundaryMessage {
            from: "r2".into(),
            kind: BoundaryMessageKind::Throttle,
            body: "b".into(),
        });
        assert_eq!(q.len(), 2);

        let msgs = q.drain();
        assert_eq!(msgs.len(), 2);
        assert!(q.is_empty());
        assert!(q.drain().is_empty()); // second drain yields nothing
    }

    // ── Engine: rule fixtures + merge semantics ─────────────────────────

    struct StaticRule {
        name: &'static str,
        verdict: GuardrailVerdict,
        enabled: bool,
    }
    impl StaticRule {
        fn new(name: &'static str, verdict: GuardrailVerdict) -> Self {
            Self {
                name,
                verdict,
                enabled: true,
            }
        }
        fn disabled(mut self) -> Self {
            self.enabled = false;
            self
        }
    }
    impl GuardrailRule for StaticRule {
        fn name(&self) -> &str {
            self.name
        }
        fn evaluate(&self, _ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            self.verdict.clone()
        }
        fn enabled(&self) -> bool {
            self.enabled
        }
    }

    #[test]
    fn engine_returns_ok_when_no_rules() {
        let e = GuardrailEngine::new();
        let d = det();
        let b = budget();
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        assert!(outcome.winning_verdict.is_ok());
        assert_eq!(outcome.winning_rule, None);
        assert!(outcome.messages.is_empty());
    }

    #[test]
    fn engine_picks_max_severity_verdict() {
        let e = GuardrailEngine::new()
            .with_rule(StaticRule::new(
                "warn_rule",
                GuardrailVerdict::Warn {
                    corrective: "be careful".into(),
                },
            ))
            .with_rule(StaticRule::new(
                "halt_rule",
                GuardrailVerdict::Halt {
                    reason: "looped too hard".into(),
                },
            ))
            .with_rule(StaticRule::new(
                "throttle_rule",
                GuardrailVerdict::Throttle {
                    corrective: "tighten up".into(),
                    max_tokens: 2_000,
                },
            ));
        let d = det();
        let b = budget();
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        assert!(outcome.winning_verdict.is_halt());
        assert_eq!(outcome.winning_rule.as_deref(), Some("halt_rule"));
        // Warn and Throttle messages STILL surface even though Halt won.
        assert_eq!(outcome.messages.len(), 2);
        assert!(outcome
            .messages
            .iter()
            .any(|m| m.from == "warn_rule" && m.kind == BoundaryMessageKind::Warn));
        assert!(outcome
            .messages
            .iter()
            .any(|m| m.from == "throttle_rule" && m.kind == BoundaryMessageKind::Throttle));
    }

    #[test]
    fn engine_ties_break_by_registration_order() {
        let e = GuardrailEngine::new()
            .with_rule(StaticRule::new(
                "first_warn",
                GuardrailVerdict::Warn {
                    corrective: "1".into(),
                },
            ))
            .with_rule(StaticRule::new(
                "second_warn",
                GuardrailVerdict::Warn {
                    corrective: "2".into(),
                },
            ));
        let d = det();
        let b = budget();
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        assert_eq!(outcome.winning_rule.as_deref(), Some("first_warn"));
        assert_eq!(outcome.messages.len(), 2);
    }

    #[test]
    fn engine_skips_disabled_rules() {
        let e = GuardrailEngine::new()
            .with_rule(
                StaticRule::new(
                    "halt_off",
                    GuardrailVerdict::Halt {
                        reason: "off".into(),
                    },
                )
                .disabled(),
            )
            .with_rule(StaticRule::new(
                "warn_on",
                GuardrailVerdict::Warn {
                    corrective: "on".into(),
                },
            ));
        let d = det();
        let b = budget();
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        // Halt was disabled, so Warn wins.
        assert!(matches!(
            outcome.winning_verdict,
            GuardrailVerdict::Warn { .. }
        ));
        assert_eq!(outcome.winning_rule.as_deref(), Some("warn_on"));
        assert_eq!(outcome.messages.len(), 1);
    }

    #[test]
    fn engine_collects_messages_from_warn_and_throttle_only() {
        let e = GuardrailEngine::new()
            .with_rule(StaticRule::new("ok_rule", GuardrailVerdict::Ok))
            .with_rule(StaticRule::new(
                "warn_rule",
                GuardrailVerdict::Warn {
                    corrective: "w".into(),
                },
            ))
            .with_rule(StaticRule::new(
                "halt_rule",
                GuardrailVerdict::Halt { reason: "h".into() },
            ));
        let d = det();
        let b = budget();
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        // Halt and Ok do NOT produce BoundaryMessages; Halt's reason
        // travels via the winning verdict, not the message queue.
        assert_eq!(outcome.messages.len(), 1);
        assert_eq!(outcome.messages[0].from, "warn_rule");
    }

    #[test]
    fn evaluation_outcome_flushes_into_queue() {
        let e = GuardrailEngine::new()
            .with_rule(StaticRule::new(
                "a",
                GuardrailVerdict::Warn {
                    corrective: "x".into(),
                },
            ))
            .with_rule(StaticRule::new(
                "b",
                GuardrailVerdict::Warn {
                    corrective: "y".into(),
                },
            ));
        let d = det();
        let b = budget();
        let q = BoundaryMessageQueue::new();
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        outcome.flush_into(&q);
        assert_eq!(q.len(), 2);
    }

    // ── End-to-end: rule that reads detector state ──────────────────────

    /// A real-shape rule: halt after N consecutive failures of a given
    /// tool. Exercises the detector → ctx → rule → engine chain.
    struct ConsecutiveFailureRuleFixture {
        tool: &'static str,
        halt_at: u32,
    }
    impl GuardrailRule for ConsecutiveFailureRuleFixture {
        fn name(&self) -> &str {
            "consecutive_failure_fixture"
        }
        fn evaluate(&self, ctx: GuardrailCtx<'_>) -> GuardrailVerdict {
            let key = format!("tool:{}:failed", self.tool);
            let n = ctx.detector.consecutive(&key);
            if n >= self.halt_at {
                GuardrailVerdict::Halt {
                    reason: format!("'{}' failed {n}× consecutively", self.tool),
                }
            } else {
                GuardrailVerdict::Ok
            }
        }
    }

    #[test]
    fn end_to_end_rule_halts_after_threshold() {
        let e = GuardrailEngine::new().with_rule(ConsecutiveFailureRuleFixture {
            tool: "build_project",
            halt_at: 3,
        });
        let d = det();
        let b = budget();

        // Two failures: not yet.
        d.record(LoopSignal::now("tool:build_project:failed"));
        d.record(LoopSignal::now("tool:build_project:failed"));
        assert!(e.evaluate(ctx(&d, b.as_ref())).winning_verdict.is_ok());

        // Third failure trips the rule.
        d.record(LoopSignal::now("tool:build_project:failed"));
        let outcome = e.evaluate(ctx(&d, b.as_ref()));
        assert!(outcome.winning_verdict.is_halt());
        match outcome.winning_verdict {
            GuardrailVerdict::Halt { reason } => assert!(reason.contains("build_project")),
            _ => unreachable!(),
        }
    }

    #[test]
    fn end_to_end_rule_resets_on_interleaved_other_key() {
        let e = GuardrailEngine::new().with_rule(ConsecutiveFailureRuleFixture {
            tool: "build_project",
            halt_at: 3,
        });
        let d = det();
        let b = budget();

        d.record(LoopSignal::now("tool:build_project:failed"));
        d.record(LoopSignal::now("tool:build_project:failed"));
        // A successful intervening call resets the consecutive run.
        d.record(LoopSignal::now("tool:build_project:succeeded"));
        d.record(LoopSignal::now("tool:build_project:failed"));
        d.record(LoopSignal::now("tool:build_project:failed"));
        assert!(e.evaluate(ctx(&d, b.as_ref())).winning_verdict.is_ok());
    }

    // ── Harness ─────────────────────────────────────────────────────────

    #[test]
    fn harness_flushes_messages_into_queue() {
        let engine = GuardrailEngine::new().with_rule(StaticRule::new(
            "r",
            GuardrailVerdict::Warn {
                corrective: "slow down".into(),
            },
        ));
        let h = GuardrailHarness::new(engine);
        let b = budget();

        let outcome = h.evaluate(0, b.as_ref());
        assert!(matches!(
            outcome.winning_verdict,
            GuardrailVerdict::Warn { .. }
        ));

        let pending = h.queue.drain();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].body, "slow down");
    }

    #[test]
    fn harness_sets_max_tokens_override_on_throttle() {
        let engine = GuardrailEngine::new().with_rule(StaticRule::new(
            "throttle",
            GuardrailVerdict::Throttle {
                corrective: "tighten up".into(),
                max_tokens: 1_500,
            },
        ));
        let h = GuardrailHarness::new(engine);
        let b = budget();
        let _ = h.evaluate(0, b.as_ref());
        assert_eq!(h.take_max_tokens_override(), Some(1_500));
        // Consume-once: second read returns None.
        assert_eq!(h.take_max_tokens_override(), None);
    }

    #[test]
    fn harness_effective_max_tokens_caps_below_requested() {
        let engine = GuardrailEngine::new().with_rule(StaticRule::new(
            "throttle",
            GuardrailVerdict::Throttle {
                corrective: "x".into(),
                max_tokens: 1_500,
            },
        ));
        let h = GuardrailHarness::new(engine);
        let b = budget();
        let _ = h.evaluate(0, b.as_ref());
        // Throttle caps below requested.
        assert_eq!(h.effective_max_tokens(8_000), 1_500);
    }

    #[test]
    fn harness_effective_max_tokens_no_override_returns_requested() {
        let h = GuardrailHarness::new(GuardrailEngine::new());
        assert_eq!(h.effective_max_tokens(8_000), 8_000);
    }

    #[test]
    fn harness_record_forwards_to_detector() {
        let h = GuardrailHarness::new(GuardrailEngine::new());
        h.record(LoopSignal::now("tool:read_file:invoked"));
        h.record(LoopSignal::now("tool:read_file:invoked"));
        assert_eq!(
            h.detector
                .count("tool:read_file:invoked", Duration::from_secs(60)),
            2
        );
    }

    // ── Built-in rules ──────────────────────────────────────────────────

    use super::rules::*;

    #[test]
    fn max_continuations_rule_halts_at_limit() {
        // `limit: 5` means "at most 5 completed turns." Rule fires at
        // the END of round N where N+1 == limit — i.e., at round 4,
        // which is the 5th completed turn. Rounds 0..=3 must be Ok.
        let r = MaxContinuationsRule { limit: 5 };
        let d = det();
        let b = budget();

        for round in 0..4 {
            let c = GuardrailCtx {
                detector: &d,
                budget: b.as_ref(),
                round,
                elapsed: Duration::ZERO,
            };
            assert!(r.evaluate(c).is_ok(), "round {round} should be Ok");
        }
        let c4 = GuardrailCtx {
            detector: &d,
            budget: b.as_ref(),
            round: 4,
            elapsed: Duration::ZERO,
        };
        assert!(r.evaluate(c4).is_halt(), "round 4 (5th turn) should Halt");
    }

    #[test]
    fn repeated_tool_call_rule_escalates_with_count() {
        let r = RepeatedToolCallRule::new("read_file", 3, 5, 8);
        let d = det();
        let b = budget();
        let evalf = |round: u32| {
            r.evaluate(GuardrailCtx {
                detector: &d,
                budget: b.as_ref(),
                round,
                elapsed: Duration::ZERO,
            })
        };

        for _ in 0..2 {
            d.record(LoopSignal::now("tool:read_file:invoked"));
        }
        assert!(evalf(0).is_ok(), "2 calls: Ok");

        d.record(LoopSignal::now("tool:read_file:invoked"));
        assert!(
            matches!(evalf(0), GuardrailVerdict::Warn { .. }),
            "3 calls: Warn"
        );

        for _ in 0..2 {
            d.record(LoopSignal::now("tool:read_file:invoked"));
        }
        assert!(
            matches!(evalf(0), GuardrailVerdict::Throttle { .. }),
            "5 calls: Throttle"
        );

        for _ in 0..3 {
            d.record(LoopSignal::now("tool:read_file:invoked"));
        }
        assert!(evalf(0).is_halt(), "8 calls: Halt");
    }

    #[test]
    fn identical_output_rule_halts_on_repeat() {
        let r = IdenticalOutputRule::new("model:text", 3);
        let d = det();
        let b = budget();
        let evalf = || {
            r.evaluate(GuardrailCtx {
                detector: &d,
                budget: b.as_ref(),
                round: 0,
                elapsed: Duration::ZERO,
            })
        };

        d.record(LoopSignal::now_with_hash("model:text", 42));
        d.record(LoopSignal::now_with_hash("model:text", 42));
        assert!(evalf().is_ok(), "2 identical: Ok");

        d.record(LoopSignal::now_with_hash("model:text", 42));
        assert!(evalf().is_halt(), "3 identical: Halt");
    }

    #[test]
    fn text_only_end_turn_rule_halts_after_n() {
        let r = TextOnlyEndTurnRule { halt_after: 3 };
        let d = det();
        let b = budget();
        let evalf = || {
            r.evaluate(GuardrailCtx {
                detector: &d,
                budget: b.as_ref(),
                round: 0,
                elapsed: Duration::ZERO,
            })
        };

        for _ in 0..2 {
            d.record(LoopSignal::now("agent:text_only_turn"));
        }
        assert!(evalf().is_ok());

        d.record(LoopSignal::now("agent:text_only_turn"));
        assert!(evalf().is_halt());
    }

    #[test]
    fn consecutive_failure_rule_escalates_and_resets() {
        let r = ConsecutiveFailureRule::new("build_project", 2, 4);
        let d = det();
        let b = budget();
        let evalf = || {
            r.evaluate(GuardrailCtx {
                detector: &d,
                budget: b.as_ref(),
                round: 0,
                elapsed: Duration::ZERO,
            })
        };

        d.record(LoopSignal::now("tool:build_project:failed"));
        assert!(evalf().is_ok());
        d.record(LoopSignal::now("tool:build_project:failed"));
        assert!(matches!(evalf(), GuardrailVerdict::Warn { .. }));

        // Interleave success — resets the run.
        d.record(LoopSignal::now("tool:build_project:succeeded"));
        assert!(evalf().is_ok());

        for _ in 0..4 {
            d.record(LoopSignal::now("tool:build_project:failed"));
        }
        assert!(evalf().is_halt());
    }

    #[test]
    fn runaway_cost_rule_warns_then_halts() {
        let r = RunawayCostRule {
            halt_at_consumed_fraction: 0.9,
            warn_at_consumed_fraction: Some(0.7),
        };
        let b = crate::in_memory_budget(1_000);
        let d = det();
        let evalf = || {
            r.evaluate(GuardrailCtx {
                detector: &d,
                budget: b.as_ref(),
                round: 0,
                elapsed: Duration::ZERO,
            })
        };

        b.record(500);
        assert!(evalf().is_ok(), "50% Ok");

        b.record(250);
        assert!(matches!(evalf(), GuardrailVerdict::Warn { .. }), "75% Warn");

        b.record(200);
        assert!(evalf().is_halt(), "95% Halt");
    }

    #[test]
    fn runaway_cost_rule_handles_zero_limit() {
        let r = RunawayCostRule {
            halt_at_consumed_fraction: 0.9,
            warn_at_consumed_fraction: None,
        };
        let b = crate::in_memory_budget(0);
        let d = det();
        let outcome = r.evaluate(GuardrailCtx {
            detector: &d,
            budget: b.as_ref(),
            round: 0,
            elapsed: Duration::ZERO,
        });
        // Zero-limit budget: avoid divide-by-zero, return Ok.
        assert!(outcome.is_ok());
    }
}
