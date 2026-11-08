//! Composition execution engine.
//!
//! `execute(composition, ctx)` walks the `Composition` tree, applies the
//! shape's semantics (Sequential threads handoff, Parallel snapshots and
//! merges, Branch routes, Fallback retries, LoopUntil iterates), and
//! returns the final `HandoffState` plus a flat list of `NamedVerdict`s
//! in execution order.
//!
//! ## Event ordering invariants
//!
//! - `CompositionStarted` precedes the first child event; `CompositionCompleted`
//!   follows every child event.
//! - In `Sequential`, step N's `StepCompleted` precedes step N+1's `StepStarted`.
//! - In `Parallel`, `StepStarted`/`StepCompleted` interleave freely; only
//!   well-formedness (pairing), not global ordering, is guaranteed.
//! - On cancel, `Cancelled` fires immediately, before children drain.
//!
//! The executor honors `ctx.cancel` at every boundary: once cancelled it
//! starts no new children, and in-flight children observe derived tokens
//! via the supervisor's child-token chain.

use crate::agent::BorgAgent;
use crate::budget::ParallelAdmission;
use crate::composition::{Composition, ErrorPolicy, RestartPolicy};
use crate::context::AgentContext;
use crate::error::BorgError;
use crate::events::{BorgEvent, CompositionShape};
use crate::handoff::{HandoffState, NamedVerdict};
use crate::supervisor::Supervisor;
use crate::verdict::Verdict;
use futures::FutureExt;
use std::sync::Arc;
use tracing::Instrument;

/// A caught panic from an agent's `invoke`, as the error a panicked
/// parallel branch surfaces.
fn agent_panicked(agent: &dyn BorgAgent, payload: Box<dyn std::any::Any + Send>) -> BorgError {
    BorgError::ChildPanicked {
        agent_id: agent.agent_id().to_string(),
        message: crate::supervisor::panic_payload_to_string(payload),
    }
}

/// Deterministic FNV-1a 64-bit hash for identical-output detection.
///
/// `DefaultHasher` (SipHash) is randomly seeded per-process, so equal
/// strings hash differently across calls. FNV-1a is unseeded and stable.
fn fnv1a_64(data: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }
    hash
}

/// A count for a `u8` event field: saturates at `u8::MAX` rather than
/// wrapping (300 verdicts reports 255, not 44).
fn saturating_u8(n: usize) -> u8 {
    u8::try_from(n).unwrap_or(u8::MAX)
}

/// Result of a top-level composition execution.
#[must_use]
#[derive(Debug)]
#[non_exhaustive]
pub struct ExecutionOutcome {
    /// Final handoff state — every step's output_key write merged in.
    pub handoff: HandoffState,
    /// Flat list of every step's named verdict, in execution order.
    pub verdicts: Vec<NamedVerdict>,
}

impl ExecutionOutcome {
    /// Number of branches that produced a successful (non-error) verdict.
    pub fn succeeded(&self) -> usize {
        self.verdicts.iter().filter(|v| !v.is_error).count()
    }

    /// Number of branches that surfaced an error under
    /// `ErrorPolicy::CollectAll` / `AtLeast`.
    pub fn failed(&self) -> usize {
        self.verdicts.iter().filter(|v| v.is_error).count()
    }

    /// Sum of input tokens across every step.
    pub fn total_input_tokens(&self) -> u64 {
        self.verdicts
            .iter()
            .map(|v| v.verdict.input_tokens)
            .fold(0u64, |a, b| a.saturating_add(b))
    }

    /// Sum of output tokens across every step.
    pub fn total_output_tokens(&self) -> u64 {
        self.verdicts
            .iter()
            .map(|v| v.verdict.output_tokens)
            .fold(0u64, |a, b| a.saturating_add(b))
    }
}

/// Execute a composition under the given top-level context.
///
/// The context provides the LLM client, budget, emitter, and parent
/// cancellation token. The composition is consumed (it owns its agents).
/// Returns the merged handoff state and the flat verdict list.
///
/// Box-pinned because the recursion across composition variants needs a
/// stable future type.
#[tracing::instrument(
    name = "borg.execute",
    skip(composition, ctx),
    fields(parent_composition_id = %ctx.parent_composition_id),
)]
pub fn execute(
    composition: Composition,
    ctx: ExecCtx,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExecutionOutcome, BorgError>> + Send>>
{
    Box::pin(execute_inner(composition, ctx))
}

/// Stable context object the executor threads through recursive calls.
///
/// All fields are owned (`Arc` for shared state); cloning is cheap and
/// `'static`-shareable so spawned children take an owned copy. External
/// crates construct via [`ExecCtx::new`] plus builder methods;
/// `#[non_exhaustive]` reserves room for additive fields.
#[derive(Clone)]
#[non_exhaustive]
pub struct ExecCtx {
    pub llm: Arc<dyn crate::llm::AgentLlmClient>,
    pub budget: Arc<dyn crate::budget::Budget>,
    pub emitter: Arc<dyn crate::events::EventEmitter>,
    pub cancel: tokio_util::sync::CancellationToken,
    pub parent_composition_id: Arc<str>,
    pub turn_number: u32,
    pub parent_call_id: Option<Arc<str>>,
    /// Cross-composition restart-storm detector, defaulting to
    /// [`crate::MeltdownPolicy::default_for_llm`]. Shared via Arc across
    /// spawned siblings so restarts count against one budget. Override
    /// with [`ExecCtx::with_meltdown`].
    pub meltdown: crate::meltdown::MeltdownLedger,
    /// Per-(provider, agent) circuit breaker, defaulting to
    /// [`crate::BreakerPolicy::default_for_llm`] (5 consecutive failures
    /// → 30s open window). Shared via Arc across all descendants, so
    /// fan-out siblings fail-fast in concert during outages.
    pub breaker: crate::circuit_breaker::CircuitBreaker,
    /// Checkpoint configuration; `None` disables resumability. Set by
    /// [`crate::execute_resumable`] and friends. `pub(crate)` because
    /// constructing one requires loading state from a `CheckpointStore`.
    pub(crate) checkpoint: Option<CheckpointConfig>,
    /// Behavioral guardrails; `None` disables them. When set via
    /// [`ExecCtx::with_guardrails`], the executor auto-emits signals at
    /// tool-dispatch and agent-loop boundaries, evaluates the engine
    /// each iteration, and returns [`BorgError::Guardrail`] on `Halt`.
    pub guardrails: Option<std::sync::Arc<crate::guardrails::GuardrailHarness>>,
    /// Bounds the snapshot history tail captured into a
    /// [`crate::checkpoint::Snapshot`]. Default unlimited; opt into a
    /// tail bound via [`ExecCtx::with_handoff_persistence`].
    pub handoff_persistence: crate::handoff::HandoffPersistenceConfig,

    /// Per-execution [`crate::propagation::ContextCarrier`]s, captured on
    /// the spawning task before every spawn this tree makes (parallel
    /// branches, parallel tool batches, [`ExecCtx::spawn_sub_agent`]) and
    /// re-entered inside it. Process-wide carriers
    /// ([`crate::propagation::register_carrier`]) apply on top. Add with
    /// [`ExecCtx::with_carrier`].
    pub(crate) carriers: crate::propagation::Carriers,

    /// The unfiltered emitter agents see as [`AgentContext::emitter`],
    /// set by [`ExecCtx::with_lifecycle_events`] (which narrows
    /// [`ExecCtx::emitter`] to the chosen lifecycle kinds). `None` = agents
    /// see `emitter`.
    pub(crate) agent_emitter: Option<Arc<dyn crate::events::EventEmitter>>,

    /// Whether the executor may dispatch parallel-safe tool calls
    /// concurrently within a single LLM turn (default `false`, fully
    /// serial). When `true`, contiguous runs grouped by
    /// `ToolDispatch::parallelize_safe(name)` run via `JoinSet`;
    /// sequential and single-call runs stay inline.
    ///
    /// No-op unless paired with a dispatcher that answers
    /// `parallelize_safe = true` for some tool (e.g.
    /// `ConcurrentDispatcher` over `ConcurrentTool` impls returning
    /// `parallelize() = true`); a mutex dispatcher serializes regardless.
    pub allow_parallel_tools: bool,

    /// How the executor reacts to a structural `BorgError` from one
    /// tool in a parallel batch. See [`BatchFailureMode`].
    pub parallel_failure_mode: BatchFailureMode,

    /// Most tool calls dispatched from one model turn (default 64). Calls past it are not run:
    /// each gets a failed result ("too many tool calls in one turn;
    /// limit N") so the model can retry with fewer. Set with
    /// [`ExecCtx::with_max_tool_calls_per_turn`].
    pub max_tool_calls_per_turn: usize,

    /// Most calls of one parallel batch running at once (default 16);
    /// the rest wait for a slot.
    /// Set with [`ExecCtx::with_max_parallel_tool_calls`].
    pub max_parallel_tool_calls: usize,
}

/// Default for [`ExecCtx::max_tool_calls_per_turn`].
pub(crate) const DEFAULT_MAX_TOOL_CALLS_PER_TURN: usize = 64;

/// Default for [`ExecCtx::max_parallel_tool_calls`].
pub(crate) const DEFAULT_MAX_PARALLEL_TOOL_CALLS: usize = 16;

/// How the executor reacts to a structural `BorgError` from one tool in
/// a parallel batch.
///
/// Only structural errors (`Err(BorgError)` from `Tool::execute`, or a
/// panic inside it) in a batch of two or more parallel-safe calls
/// participate. Logical failures (`Ok(ToolCallResult::fail(...))`) always
/// collect back to the agent regardless of this mode, and a structural
/// error from a call dispatched on its own (a serial call, or a
/// one-call batch) always ends the loop with that error.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchFailureMode {
    /// Wait for every parallel sibling and hand every call's result to
    /// the agent: a structural error becomes that call's `is_error`
    /// result (`ToolCallResult::fail` carrying the error's message, with
    /// a failed `ToolReturned` event), and the loop continues so the
    /// agent can self-correct. Only a cancellation or a suspension ends
    /// the loop, returned once the batch has drained. Default.
    #[default]
    CollectAll,
    /// On the first structural `BorgError`, cancel sibling tasks via the
    /// batch's `CancellationToken` and return without waiting. Use for
    /// cost-sensitive or transactional workloads where partial work is
    /// worse than no work.
    FastFail,
}

/// Internal checkpoint configuration carried on `ExecCtx`. Cloning shares
/// state via the inner `Arc`s.
#[derive(Clone)]
pub(crate) struct CheckpointConfig {
    pub store: Arc<dyn crate::checkpoint::CheckpointStore>,
    pub id: String,
    pub policy: crate::checkpoint::CheckpointPolicy,
    pub composition_fingerprint: String,
    pub completed_steps: Arc<parking_lot::Mutex<std::collections::HashSet<Arc<str>>>>,
}

impl ExecCtx {
    pub fn new(
        llm: Arc<dyn crate::llm::AgentLlmClient>,
        budget: Arc<dyn crate::budget::Budget>,
        emitter: Arc<dyn crate::events::EventEmitter>,
        cancel: tokio_util::sync::CancellationToken,
        parent_composition_id: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            llm,
            budget,
            emitter,
            cancel,
            parent_composition_id: parent_composition_id.into(),
            turn_number: 0,
            parent_call_id: None,
            meltdown: crate::meltdown::MeltdownLedger::with_default_policy(),
            breaker: crate::circuit_breaker::CircuitBreaker::with_default_policy(),
            checkpoint: None,
            guardrails: None,
            handoff_persistence: crate::handoff::HandoffPersistenceConfig::default(),
            carriers: crate::propagation::Carriers::new(),
            agent_emitter: None,
            allow_parallel_tools: false,
            parallel_failure_mode: BatchFailureMode::CollectAll,
            max_tool_calls_per_turn: DEFAULT_MAX_TOOL_CALLS_PER_TURN,
            max_parallel_tool_calls: DEFAULT_MAX_PARALLEL_TOOL_CALLS,
        }
    }

    /// Cap the tool calls dispatched from one model turn (default 64,
    /// minimum 1). Each call past
    /// the cap is answered with a failed result, "too many tool calls in
    /// one turn; limit N", without running, so the model can recover.
    #[must_use]
    pub fn with_max_tool_calls_per_turn(mut self, n: usize) -> Self {
        self.max_tool_calls_per_turn = n.max(1);
        self
    }

    /// Cap how many calls of one parallel batch run at once (default 16,
    /// minimum 1). Only matters with
    /// [`ExecCtx::with_parallel_tools`]; the rest of the batch waits for
    /// a free slot.
    #[must_use]
    pub fn with_max_parallel_tool_calls(mut self, n: usize) -> Self {
        self.max_parallel_tool_calls = n.max(1);
        self
    }

    /// Override the snapshot history retention policy. The default
    /// captures the full history per save; bound the tail to control
    /// wire bytes when writing snapshots to a remote store.
    ///
    /// ```ignore
    /// let ctx = ExecCtx::new(llm, budget, emitter, cancel, "session")
    ///     .with_handoff_persistence(HandoffPersistenceConfig {
    ///         max_persisted_history: 32,
    ///     });
    /// ```
    #[must_use]
    pub fn with_handoff_persistence(
        mut self,
        cfg: crate::handoff::HandoffPersistenceConfig,
    ) -> Self {
        self.handoff_persistence = cfg;
        self
    }

    /// Opt in to parallel tool-call dispatch within a single LLM turn,
    /// choosing the batch failure mode. Effective only with a
    /// `ToolDispatch` that answers `parallelize_safe = true` for at least
    /// one tool (typically `ConcurrentDispatcher`).
    #[must_use]
    pub fn with_parallel_tools(mut self, mode: BatchFailureMode) -> Self {
        self.allow_parallel_tools = true;
        self.parallel_failure_mode = mode;
        self
    }

    /// Wire in a [`crate::guardrails::GuardrailHarness`]. The executor
    /// auto-emits behavioral signals (tool invoke/succeed/fail, turn
    /// completion, text-only turns) and evaluates the harness's engine at
    /// every agent-loop boundary. A `Halt` verdict terminates with
    /// [`BorgError::Guardrail`]; `Warn`/`Throttle` verdicts surface via
    /// the harness's [`crate::BoundaryMessageQueue`] for the agent's next
    /// `user_prompt`.
    #[must_use]
    pub fn with_guardrails(
        mut self,
        harness: std::sync::Arc<crate::guardrails::GuardrailHarness>,
    ) -> Self {
        self.guardrails = Some(harness);
        self
    }

    /// Replace the meltdown ledger (e.g. `MeltdownLedger::disabled()` to
    /// opt out of the storm detector, or a custom policy).
    #[must_use]
    pub fn with_meltdown(mut self, ledger: crate::meltdown::MeltdownLedger) -> Self {
        self.meltdown = ledger;
        self
    }

    /// Replace the circuit breaker (e.g. [`crate::CircuitBreaker::disabled`]
    /// for raw retry scenarios, or a custom policy).
    #[must_use]
    pub fn with_breaker(mut self, breaker: crate::circuit_breaker::CircuitBreaker) -> Self {
        self.breaker = breaker;
        self
    }

    /// Carry `carrier`'s context into every task this execution spawns:
    /// each parallel branch, each parallel tool call, each
    /// [`ExecCtx::spawn_sub_agent`]. Captured on the spawning task right
    /// before the spawn and re-entered inside the new task, so a
    /// task-local set around [`execute`] is visible in every branch.
    ///
    /// ```
    /// use borg::propagation::TaskLocal;
    /// # use std::sync::Arc;
    /// tokio::task_local! { static LEDGER: Arc<Vec<u8>>; }
    /// # let ctx = borg::ExecCtx::deterministic(borg::in_memory_budget(1),
    /// #     borg::noop_emitter(), borg::CancellationToken::new(), "doc");
    /// let ctx = ctx.with_carrier(TaskLocal::new(&LEDGER));
    /// ```
    ///
    /// For a carrier every execution in the process needs, register it
    /// once with [`crate::propagation::register_carrier`] instead.
    #[must_use]
    pub fn with_carrier(mut self, carrier: impl crate::propagation::ContextCarrier) -> Self {
        self.carriers = self.carriers.with(Arc::new(carrier));
        self
    }

    /// Capture this execution's context (span, process-wide and
    /// per-execution carriers) on the calling task, to run a future of
    /// your own in it — the same capture borg makes before each spawn.
    pub fn capture_context(&self) -> crate::propagation::CapturedContext {
        crate::propagation::capture(&self.carriers)
    }

    /// Emit only the chosen lifecycle events. The executor's own events
    /// (`CompositionStarted`, `StepCompleted`, `ToolCalled`, …) go to
    /// [`ExecCtx::emitter`] only when their kind is in `kinds`; agents
    /// still get the full emitter on [`AgentContext::emitter`], so an app
    /// can run a composition on its own event sink and see only the
    /// events its agents emit:
    ///
    /// ```
    /// use borg::events::EventKinds;
    /// # let ctx = borg::ExecCtx::deterministic(borg::in_memory_budget(1),
    /// #     borg::noop_emitter(), borg::CancellationToken::new(), "doc");
    /// // No `CompositionStarted`/`StepStarted`/… rows on the app's wire.
    /// let ctx = ctx.with_lifecycle_events(EventKinds::NONE);
    /// ```
    ///
    /// Spans (`turn_span`, `tool_span`) still come from the emitter.
    /// Call it after the emitter is final; calling it again narrows the
    /// original emitter afresh.
    #[must_use]
    pub fn with_lifecycle_events(mut self, kinds: crate::events::EventKinds) -> Self {
        let full = self
            .agent_emitter
            .take()
            .unwrap_or_else(|| self.emitter.clone());
        self.emitter = Arc::new(crate::events::FilteredEmitter::kinds(full.clone(), kinds));
        self.agent_emitter = Some(full);
        self
    }

    /// The emitter agents see: the unfiltered one when
    /// [`ExecCtx::with_lifecycle_events`] narrowed [`ExecCtx::emitter`].
    pub(crate) fn agent_emitter(&self) -> Arc<dyn crate::events::EventEmitter> {
        self.agent_emitter
            .clone()
            .unwrap_or_else(|| self.emitter.clone())
    }

    /// Whether checkpoint mode is active on this context.
    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.is_some()
    }

    #[must_use]
    pub fn with_turn(mut self, turn: u32) -> Self {
        self.turn_number = turn;
        self
    }

    #[must_use]
    pub fn with_parent_call(mut self, id: Option<Arc<str>>) -> Self {
        self.parent_call_id = id;
        self
    }

    /// Convenience constructor for compositions made entirely of
    /// deterministic (non-LLM-calling) agents. Wires
    /// [`crate::llm::deterministic_llm_client`], which errors if any
    /// agent reaches for `ctx.llm`. If even one agent in the tree calls
    /// the LLM, use [`ExecCtx::new`] with a real client.
    pub fn deterministic(
        budget: Arc<dyn crate::budget::Budget>,
        emitter: Arc<dyn crate::events::EventEmitter>,
        cancel: tokio_util::sync::CancellationToken,
        parent_composition_id: impl Into<Arc<str>>,
    ) -> Self {
        Self::new(
            crate::llm::deterministic_llm_client(),
            budget,
            emitter,
            cancel,
            parent_composition_id,
        )
    }

    /// Build an `AgentContext` for one step's invocation. `handoff` is
    /// cloned into an `Arc` so the context is `'static`-capable.
    fn agent_ctx(
        &self,
        handoff: &HandoffState,
        cancel: tokio_util::sync::CancellationToken,
        step_name: Arc<str>,
    ) -> AgentContext {
        self.agent_ctx_with_tools(
            handoff,
            cancel,
            step_name,
            Arc::from([] as [crate::tools::ToolSchema; 0]),
        )
    }

    /// Like `agent_ctx` but with tool schemas attached, used by the
    /// `Composition::Agent` tool-loop so the model knows what's callable.
    fn agent_ctx_with_tools(
        &self,
        handoff: &HandoffState,
        cancel: tokio_util::sync::CancellationToken,
        step_name: Arc<str>,
        tool_schemas: Arc<[crate::tools::ToolSchema]>,
    ) -> AgentContext {
        AgentContext {
            handoff: std::sync::Arc::new(handoff.clone()),
            cancel,
            llm: self.llm.clone(),
            budget: self.budget.clone(),
            emitter: self.agent_emitter(),
            guardrails: self.guardrails.clone(),
            parent_composition_id: self.parent_composition_id.clone(),
            step_name,
            turn_number: self.turn_number,
            parent_call_id: self.parent_call_id.clone(),
            tool_schemas,
        }
    }
}

async fn execute_inner(
    composition: Composition,
    ctx: ExecCtx,
) -> Result<ExecutionOutcome, BorgError> {
    let mut handoff = HandoffState::new();
    let verdicts = execute_into(composition, &ctx, &mut handoff).await?;
    Ok(ExecutionOutcome { handoff, verdicts })
}

/// Execute a composition with checkpoint durability, saving snapshots at
/// policy-defined boundaries ([`crate::CheckpointPolicy::OnLoopIteration`]
/// by default; override with [`execute_resumable_with`]).
///
/// First call (`store.load(id)` → `None`): runs from the top, saves as it
/// goes, and deletes the snapshot on success so the next call starts fresh.
///
/// Resume (`store.load(id)` → `Some(snap)`): the composition's structural
/// fingerprint must match `snap`'s, else [`BorgError::Internal`]
/// ([`CheckpointStore::delete`](crate::CheckpointStore::delete) the id if the
/// restructure was intentional). On match, the handoff and completed-steps
/// set are hydrated and execution fast-forwards past completed steps.
///
/// Granularity: resumption works at `Sequential` step and `LoopUntil`
/// iteration boundaries, at any depth. Every nested context's id carries
/// its position (`…::seq#{idx}`, `…::par#{idx}`, `…::branch#{idx}`,
/// `…::loop::iter{n}`, `…::fb#fb{n}`), so two steps of the same shape or
/// name never share a checkpoint id. Nothing inside a `Parallel` branch
/// is checkpointed (a branch's handoff is a private copy), so a
/// `Parallel` re-runs wholesale on resume.
pub fn execute_resumable(
    composition: Composition,
    ctx: ExecCtx,
    store: Arc<dyn crate::checkpoint::CheckpointStore>,
    id: impl Into<String> + Send + 'static,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExecutionOutcome, BorgError>> + Send>>
{
    execute_resumable_with(
        composition,
        ctx,
        store,
        id,
        crate::checkpoint::CheckpointPolicy::OnLoopIteration,
    )
}

/// Like [`execute_resumable`] but with an explicit save policy.
pub fn execute_resumable_with(
    composition: Composition,
    ctx: ExecCtx,
    store: Arc<dyn crate::checkpoint::CheckpointStore>,
    id: impl Into<String> + Send + 'static,
    policy: crate::checkpoint::CheckpointPolicy,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExecutionOutcome, BorgError>> + Send>>
{
    Box::pin(async move {
        use crate::checkpoint::{composition_fingerprint, SNAPSHOT_SCHEMA_VERSION};
        use parking_lot::Mutex;
        use std::collections::HashSet;

        let id_str: String = id.into();
        let fingerprint = composition_fingerprint(&composition);

        // Load any prior snapshot, hydrating handoff + completed_steps;
        // the fingerprint check refuses resuming a changed shape.
        let (mut handoff, completed_set) = match store.load(&id_str).await? {
            Some(snap) => {
                if snap.schema_version != SNAPSHOT_SCHEMA_VERSION {
                    return Err(BorgError::Internal {
                        message: format!(
                            "snapshot schema version mismatch: stored {}, running {}",
                            snap.schema_version, SNAPSHOT_SCHEMA_VERSION
                        ),
                    });
                }
                if snap.composition_fingerprint != fingerprint {
                    return Err(BorgError::Internal {
                        message: format!(
                            "composition fingerprint mismatch for checkpoint `{}`: \
                             stored `{}`, running `{}`. Call `store.delete(id)` if \
                             the restructure was intentional.",
                            id_str, snap.composition_fingerprint, fingerprint
                        ),
                    });
                }
                let set: HashSet<Arc<str>> = snap.completed_steps.iter().cloned().collect();
                (snap.handoff, set)
            }
            None => (HandoffState::new(), HashSet::new()),
        };

        let cfg = CheckpointConfig {
            store: store.clone(),
            id: id_str.clone(),
            policy,
            composition_fingerprint: fingerprint,
            completed_steps: Arc::new(Mutex::new(completed_set)),
        };
        let mut ctx_with_cp = ctx;
        ctx_with_cp.checkpoint = Some(cfg);

        let verdicts = execute_into(composition, &ctx_with_cp, &mut handoff).await?;

        // Clear the snapshot so the next run with this id starts fresh.
        // A failed delete isn't fatal (stale snapshot is ignored on the
        // next resume), but the operator should know.
        if let Err(e) = store.delete(&id_str).await {
            tracing::warn!(
                checkpoint_id = %id_str,
                error = %e,
                "checkpoint delete failed after successful completion; \
                 stale snapshot will be ignored on next execute_resumable \
                 unless the same composition shape is re-run"
            );
        }
        Ok(ExecutionOutcome { handoff, verdicts })
    })
}

/// Execute a composition with a pre-populated `HandoffState`.
///
/// Use this to seed state the first agent's `user_prompt` reads (e.g.
/// session-level `working_files`). The handoff is then mutated and merged
/// exactly as in [`execute`].
pub fn execute_with_handoff(
    composition: Composition,
    ctx: ExecCtx,
    handoff: HandoffState,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExecutionOutcome, BorgError>> + Send>>
{
    Box::pin(async move {
        let mut handoff = handoff;
        let verdicts = execute_into(composition, &ctx, &mut handoff).await?;
        Ok(ExecutionOutcome { handoff, verdicts })
    })
}

/// Run a single agent as a `Composition::single` and return its verdict
/// directly.
///
/// `name` is the step name used in the `composition_id` hierarchy and
/// trace events; pick something stable (typically the agent's
/// `agent_id()`) so dashboards group across runs. Any `BorgError` from
/// the invocation is returned verbatim.
pub async fn run_one(
    name: impl Into<String>,
    agent: impl crate::agent::BorgAgent + 'static,
    ctx: ExecCtx,
    handoff: HandoffState,
) -> Result<crate::verdict::Verdict, BorgError> {
    run_one_with_restart(name, agent, RestartPolicy::Never, ctx, handoff).await
}

/// Like [`run_one`] but with an explicit [`RestartPolicy`]. Only
/// `is_retryable()` errors (transient HTTP 429/5xx, network blips) retry;
/// terminal failures (auth, budget, cancelled) propagate immediately.
/// Retries emit `BorgEvent::StepRetrying`; exhaustion surfaces as
/// `BorgError::RetryExhausted` carrying the last error.
pub async fn run_one_with_restart(
    name: impl Into<String>,
    agent: impl crate::agent::BorgAgent + 'static,
    restart: RestartPolicy,
    ctx: ExecCtx,
    handoff: HandoffState,
) -> Result<crate::verdict::Verdict, BorgError> {
    let composition = match Composition::single(name, agent) {
        Composition::Single {
            name,
            agent,
            output_key,
            ..
        } => Composition::Single {
            name,
            agent,
            output_key,
            restart,
        },
        other => other, // unreachable but keep total
    };
    let outcome = execute_with_handoff(composition, ctx, handoff).await?;
    outcome
        .verdicts
        .into_iter()
        .next()
        .map(|nv| nv.verdict)
        .ok_or_else(|| BorgError::Internal {
            message: "run_one_with_restart: single-step composition produced no verdict".into(),
        })
}

/// Like [`run_one`] but parses the verdict body into `T` via `serde_json`.
/// Parse failures surface as `BorgError::Internal` carrying a 200-char
/// body preview rather than failing silently. Use for sub-agents whose
/// body is always JSON; for free-form text use [`run_one`].
///
/// Parse failures do not re-prompt — that would need a re-prompt cycle
/// the single-shot `BorgAgent::invoke` shape doesn't express.
///
/// ```ignore
/// #[derive(serde::Deserialize, Default)]
/// struct PlannedTask { kind: String, prompt: String }
///
/// let tasks: Vec<PlannedTask> =
///     borg::run_one_typed("planner", Planner, ctx, h).await?;
/// ```
pub async fn run_one_typed<A, T>(
    name: impl Into<String>,
    agent: A,
    ctx: ExecCtx,
    handoff: HandoffState,
) -> Result<T, BorgError>
where
    A: crate::agent::BorgAgent + 'static,
    T: serde::de::DeserializeOwned,
{
    let verdict = run_one(name, agent, ctx, handoff).await?;
    parse_verdict_body::<T>(&verdict.body, "run_one_typed")
}

/// Same as [`run_one_typed`] but with an explicit [`RestartPolicy`] for
/// transient LLM failures (retries the underlying invocation, not the
/// JSON parse).
pub async fn run_one_typed_with_restart<A, T>(
    name: impl Into<String>,
    agent: A,
    restart: RestartPolicy,
    ctx: ExecCtx,
    handoff: HandoffState,
) -> Result<T, BorgError>
where
    A: crate::agent::BorgAgent + 'static,
    T: serde::de::DeserializeOwned,
{
    let verdict = run_one_with_restart(name, agent, restart, ctx, handoff).await?;
    parse_verdict_body::<T>(&verdict.body, "run_one_typed_with_restart")
}

/// Parse a verdict body into `T`. If a direct parse fails and the body is
/// wrapped in a fenced code block, strips the fence and retries — models
/// occasionally wrap JSON in fences even when asked for raw JSON.
fn parse_verdict_body<T: serde::de::DeserializeOwned>(
    body: &str,
    caller: &'static str,
) -> Result<T, BorgError> {
    if let Ok(parsed) = serde_json::from_str::<T>(body) {
        return Ok(parsed);
    }

    // Strip a fenced code block: skip the ``` or ```lang line, take
    // content until the next ```. Only reached on a parse failure.
    let trimmed = body.trim();
    if trimmed.starts_with("```") {
        let inner: String = trimmed
            .lines()
            .skip(1)
            .take_while(|l| !l.starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n");
        if let Ok(parsed) = serde_json::from_str::<T>(&inner) {
            return Ok(parsed);
        }
    }

    let excerpt: String = body.chars().take(200).collect();
    Err(BorgError::Internal {
        message: format!(
            "{caller}: failed to parse verdict body as `{ty}`; body excerpt: {excerpt}",
            caller = caller,
            ty = std::any::type_name::<T>(),
            excerpt = excerpt,
        ),
    })
}

/// Recursive driver. Mutates `handoff` so Sequential/LoopUntil chain;
/// Parallel snapshots and merges atomically on success.
fn execute_into<'a>(
    composition: Composition,
    ctx: &'a ExecCtx,
    handoff: &'a mut HandoffState,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Vec<NamedVerdict>, BorgError>> + Send + 'a>,
> {
    Box::pin(async move {
        if ctx.cancel.is_cancelled() {
            ctx.emitter
                .emit(Arc::new(BorgEvent::Cancelled {
                    composition_id: ctx.parent_composition_id.clone(),
                    reason: "parent_cancel".into(),
                }))
                .await;
            tracing::warn!(
                composition_id = %ctx.parent_composition_id,
                "borg: composition cancelled at boundary check"
            );
            return Err(BorgError::Cancelled);
        }

        match composition {
            Composition::Single {
                name,
                agent,
                output_key,
                restart,
            } => execute_single(name, agent, output_key, restart, ctx, handoff).await,
            Composition::Sequential { steps, on_error } => {
                execute_sequential(steps, on_error, ctx, handoff).await
            }
            Composition::Parallel {
                branches,
                on_error,
                max_concurrency,
            } => execute_parallel(branches, on_error, max_concurrency, ctx, handoff).await,
            Composition::LoopUntil {
                body,
                until,
                max_iterations,
            } => execute_loop_until(body, until, max_iterations, ctx, handoff).await,
            Composition::Branch {
                selector,
                branches,
                default,
            } => execute_branch(selector, branches, default, ctx, handoff).await,
            Composition::Fallback { primary, fallbacks } => {
                execute_fallback(*primary, fallbacks, ctx, handoff).await
            }
            Composition::Agent {
                name,
                agent,
                tools,
                max_steps,
            } => execute_agent_loop(name, agent, tools, max_steps, ctx, handoff).await,
            Composition::FallbackLadder {
                rungs,
                falls_through,
                on_fall,
            } => execute_fallback_ladder(rungs, falls_through, on_fall, ctx, handoff).await,
            Composition::Heartbeat {
                name,
                every,
                at_start,
                inner,
            } => execute_heartbeat(name, every, at_start, *inner, ctx, handoff).await,
            Composition::Deferred { build } => {
                let built = build(handoff);
                execute_into(built, ctx, handoff).await
            }
        }
    })
}

/// Run a [`Composition::Heartbeat`]: `inner` in this same context, a
/// [`BorgEvent::Heartbeat`] every `every` while it runs (on tokio's clock,
/// so a paused test clock drives it), and one at the start when asked.
/// A beat is awaited between polls of `inner`, like any other event; a
/// tick missed meanwhile is skipped ([`tokio::time::MissedTickBehavior::Delay`]).
async fn execute_heartbeat(
    name: String,
    every: std::time::Duration,
    at_start: bool,
    inner: Composition,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let name: Arc<str> = Arc::from(name);
    let every = every.max(std::time::Duration::from_millis(1));
    let started = tokio::time::Instant::now();
    let beat = |beat: u32, elapsed_ms: u64| {
        Arc::new(BorgEvent::Heartbeat {
            composition_id: ctx.parent_composition_id.clone(),
            name: name.clone(),
            beat,
            elapsed_ms,
        })
    };
    let mut beats: u32 = 0;
    if at_start {
        ctx.emitter.emit(beat(0, 0)).await;
        beats = 1;
    }
    let mut ticker = tokio::time::interval_at(started + every, every);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut run = execute_into(inner, ctx, handoff);
    loop {
        tokio::select! {
            out = &mut run => return out,
            _ = ticker.tick() => {
                let elapsed = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                ctx.emitter.emit(beat(beats, elapsed)).await;
                beats = beats.saturating_add(1);
            }
        }
    }
}

async fn execute_single(
    name: String,
    agent: Box<dyn BorgAgent>,
    output_key: Option<String>,
    restart: RestartPolicy,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let name: Arc<str> = Arc::from(name);
    let composition_id: Arc<str> = Arc::from(format!("{}::{}", ctx.parent_composition_id, name));
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::Single,
            branch_count: 1,
        }))
        .await;
    let agent_model: Arc<str> = Arc::from(agent.model());
    ctx.emitter
        .emit(Arc::new(BorgEvent::StepStarted {
            composition_id: composition_id.clone(),
            step_name: name.clone(),
            position: 0,
            turn_number: ctx.turn_number,
            model: agent_model.clone(),
        }))
        .await;

    let max_attempts = restart.max_attempts();
    let window = restart.window();
    let backoff = restart.backoff();
    let started = std::time::Instant::now();
    let mut last_err: Option<BorgError> = None;

    for attempt in 1..=max_attempts {
        if ctx.cancel.is_cancelled() {
            return Err(BorgError::Cancelled);
        }
        // Circuit-breaker pre-check: while the breaker is Open (sustained
        // outage), fail-fast as ProviderUnavailable without an LLM call so
        // the caller can route to a fallback instead of retrying blindly.
        let agent_provider = agent.provider();
        let breaker_agent_id = agent.agent_id();
        if let Err(reopens_in) = ctx.breaker.check(agent_provider, breaker_agent_id) {
            last_err = Some(BorgError::ProviderUnavailable {
                provider: provider_label(agent_provider),
                agent_id: breaker_agent_id.to_string(),
                reopens_in,
            });
            break;
        }
        let agent_ctx = ctx.agent_ctx(handoff, ctx.cancel.child_token(), name.clone());
        // Panic boundary: a panicking `invoke` becomes `ChildPanicked`
        // (retryable, like a panicked parallel branch) instead of
        // unwinding into the caller. AssertUnwindSafe is sound: after a
        // panic the agent is only used for another attempt, which is
        // what a restart of a panicked branch does too.
        let invoked = std::panic::AssertUnwindSafe(agent.invoke(&agent_ctx))
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| Err(agent_panicked(agent.as_ref(), payload)));
        match invoked {
            Ok(verdict) => {
                // Success closes the breaker; done first so a panic in
                // event emission can't leave it stale.
                ctx.breaker.record_success(agent_provider, breaker_agent_id);
                if let Some(key) = output_key.as_ref() {
                    handoff.set(key.clone(), verdict.body.clone());
                }
                // Declarative handoff: write the agent's declared target
                // to the well-known key so a downstream Composition::Branch
                // selector can route on it.
                if let Some(target) = verdict.handoff_to.as_ref() {
                    handoff.set(HandoffState::HANDOFF_KEY, target.clone());
                }
                let duration = started.elapsed().as_millis() as u64;
                let nv = NamedVerdict::success(name.as_ref(), verdict.clone(), duration);
                handoff.record(nv.clone());
                ctx.emitter
                    .emit(Arc::new(BorgEvent::StepCompleted {
                        composition_id: composition_id.clone(),
                        step_name: name.clone(),
                        position: 0,
                        duration_ms: duration,
                        input_tokens: verdict.input_tokens,
                        output_tokens: verdict.output_tokens,
                        success: true,
                        failure_cause: None,
                        turn_number: ctx.turn_number,
                        model: agent_model.clone(),
                    }))
                    .await;
                ctx.emitter
                    .emit(Arc::new(BorgEvent::CompositionCompleted {
                        composition_id,
                        total_duration_ms: duration,
                        total_input_tokens: verdict.input_tokens,
                        total_output_tokens: verdict.output_tokens,
                        succeeded: 1,
                        failed: 0,
                    }))
                    .await;
                return Ok(vec![nv]);
            }
            Err(e) if e.is_retryable() && attempt < max_attempts => {
                // Only retryable failures count toward the breaker's
                // consecutive-failure tally; terminal errors (auth, schema)
                // signal local misconfiguration, not upstream outage.
                ctx.breaker.record_failure(agent_provider, breaker_agent_id);
                if !window.is_zero() && started.elapsed() >= window {
                    last_err = Some(e);
                    break;
                }
                // Restart-storm detector: record on the session-wide ledger
                // before sleeping. Past the threshold (default 20 restarts
                // in 60s) surface a typed meltdown error — a cross-call
                // safety net beyond per-call `max_attempts`.
                let agent_id_static = agent.agent_id();
                if let Some(restarts) = ctx.meltdown.record_restart(agent_id_static) {
                    let policy = ctx.meltdown.policy();
                    return Err(BorgError::Meltdown {
                        agent_id: agent_id_static.to_string(),
                        restarts,
                        window_secs: policy.window.as_secs() as u32,
                    });
                }
                ctx.emitter
                    .emit(Arc::new(BorgEvent::StepRetrying {
                        composition_id: composition_id.clone(),
                        step_name: name.clone(),
                        attempt: attempt + 1,
                        last_error: format!("{e}"),
                    }))
                    .await;
                // Never sooner than the provider asked: its retry-after is
                // a floor under the backoff, not a replacement for it.
                let delay = backoff
                    .delay_for_attempt(attempt + 1)
                    .max(e.retry_after().unwrap_or_default());
                last_err = Some(e);
                // Cancel-aware backoff: a parent cancel during the sleep
                // returns Cancelled cleanly. Skip the sleep entirely if it
                // would by itself blow the wall-clock window.
                if !window.is_zero() && started.elapsed() + delay >= window {
                    break;
                }
                if !delay.is_zero() {
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = ctx.cancel.cancelled() => {
                            return Err(BorgError::Cancelled);
                        }
                    }
                }
                continue;
            }
            Err(e) => {
                // Terminal error, or last attempt of a retryable sequence.
                // Same breaker rule: record only if retryable.
                if e.is_retryable() {
                    ctx.breaker.record_failure(agent_provider, breaker_agent_id);
                } else {
                    // A terminal error says nothing about upstream health,
                    // so it must not hold a half-open trial slot until the
                    // lease runs out: hand it to the next caller now.
                    ctx.breaker.release_trial(agent_provider, breaker_agent_id);
                }
                last_err = Some(e);
                break;
            }
        }
    }

    let duration = started.elapsed().as_millis() as u64;
    let err = last_err.unwrap_or(BorgError::Internal {
        message: "single retry loop exited without verdict".into(),
    });
    // Wrap as RetryExhausted only when the error was retryable and we made
    // more than one attempt — otherwise a single terminal error would get a
    // misleading "retry-exhausted" tag.
    let final_err = if max_attempts > 1 && err.is_retryable() {
        BorgError::RetryExhausted {
            agent_id: name.to_string(),
            attempts: max_attempts,
            last_error: Box::new(err),
        }
    } else {
        err
    };
    let cause = final_err.failure_cause().to_string();
    ctx.emitter
        .emit(Arc::new(BorgEvent::StepCompleted {
            composition_id: composition_id.clone(),
            step_name: name,
            position: 0,
            duration_ms: duration,
            input_tokens: 0,
            output_tokens: 0,
            success: false,
            failure_cause: Some(cause),
            turn_number: ctx.turn_number,
            model: agent_model,
        }))
        .await;
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: duration,
            total_input_tokens: 0,
            total_output_tokens: 0,
            succeeded: 0,
            failed: 1,
        }))
        .await;
    Err(final_err)
}

async fn execute_sequential(
    steps: Vec<Composition>,
    on_error: ErrorPolicy,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let composition_id: Arc<str> = Arc::from(format!("{}::seq", ctx.parent_composition_id));
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::Sequential,
            branch_count: steps.len().min(u8::MAX as usize) as u8,
        }))
        .await;

    let started = std::time::Instant::now();
    let mut all = Vec::new();
    let mut succeeded: u8 = 0;
    let mut failed: u8 = 0;
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;

    for (idx, step) in steps.into_iter().enumerate() {
        if ctx.cancel.is_cancelled() {
            ctx.emitter
                .emit(Arc::new(BorgEvent::Cancelled {
                    composition_id: composition_id.clone(),
                    reason: "parent_cancel".into(),
                }))
                .await;
            return Err(BorgError::Cancelled);
        }
        // Each child nests under its own position (`…::seq#{idx}`), so
        // two children of the same shape or name (two loops, two
        // `review` steps) never share a checkpoint id.
        let child_parent: Arc<str> = Arc::from(format!("{composition_id}#{idx}"));
        // Checkpoint fast-forward for Sequential's Single children; a
        // nested LoopUntil or Sequential tracks its own completion under
        // `child_parent`. The id is the Single's own composition id.
        let single_step_id: Option<Arc<str>> = if let Composition::Single { name, .. } = &step {
            Some(Arc::from(format!("{child_parent}::{name}")))
        } else {
            None
        };
        if let (Some(step_id), Some(cp)) = (&single_step_id, &ctx.checkpoint) {
            let set = cp.completed_steps.lock();
            if set.contains(step_id) {
                // Already done: its output_key write and NamedVerdict are
                // already restored from the snapshot, so skip it. The
                // returned Vec is therefore shorter than the step count on
                // resume (same contract as LoopUntil).
                drop(set);
                continue;
            }
        }
        let nested_ctx = ExecCtx {
            parent_composition_id: child_parent,
            ..ctx.clone()
        };
        match execute_into(step, &nested_ctx, handoff).await {
            Ok(vs) => {
                for v in &vs {
                    total_in += v.verdict.input_tokens;
                    total_out += v.verdict.output_tokens;
                    if v.is_error {
                        failed = failed.saturating_add(1);
                    } else {
                        succeeded = succeeded.saturating_add(1);
                    }
                }
                all.extend(vs);
                // Save point for Single children, only under OnEveryStep;
                // the default OnLoopIteration policy doesn't save at
                // Sequential boundaries.
                if let (Some(step_id), Some(cp)) = (single_step_id.as_ref(), &ctx.checkpoint) {
                    if matches!(cp.policy, crate::checkpoint::CheckpointPolicy::OnEveryStep) {
                        cp.completed_steps.lock().insert(step_id.clone());
                        let snap = crate::checkpoint::Snapshot {
                            schema_version: crate::checkpoint::SNAPSHOT_SCHEMA_VERSION,
                            composition_fingerprint: cp.composition_fingerprint.clone(),
                            handoff: handoff.to_snapshot(&ctx.handoff_persistence),
                            completed_steps: cp.completed_steps.lock().iter().cloned().collect(),
                            turn_number: ctx.turn_number,
                            created_at: std::time::SystemTime::now(),
                        };
                        if let Err(e) = cp.store.save(&cp.id, &snap).await {
                            tracing::warn!(
                                composition_id = %composition_id,
                                step = %step_id,
                                error = %e,
                                "checkpoint save failed at Sequential step; \
                                 execution continues but resume may not pick \
                                 up this step"
                            );
                        }
                    }
                }
            }
            Err(e) => match on_error {
                ErrorPolicy::FailFast => {
                    ctx.emitter
                        .emit(Arc::new(BorgEvent::CompositionCompleted {
                            composition_id,
                            total_duration_ms: started.elapsed().as_millis() as u64,
                            total_input_tokens: total_in,
                            total_output_tokens: total_out,
                            succeeded,
                            failed: failed.saturating_add(1),
                        }))
                        .await;
                    return Err(e);
                }
                ErrorPolicy::CollectAll | ErrorPolicy::AtLeast(_) => {
                    let nv =
                        NamedVerdict::failure(format!("seq_step_{}", all.len()), format!("{e}"), 0);
                    handoff.record(nv.clone());
                    all.push(nv);
                    failed = failed.saturating_add(1);
                }
            },
        }
    }

    if let ErrorPolicy::AtLeast(n) = on_error {
        let ok_count = all.iter().filter(|v| !v.is_error).count();
        if ok_count < n {
            ctx.emitter
                .emit(Arc::new(BorgEvent::CompositionCompleted {
                    composition_id,
                    total_duration_ms: started.elapsed().as_millis() as u64,
                    total_input_tokens: total_in,
                    total_output_tokens: total_out,
                    succeeded,
                    failed,
                }))
                .await;
            return Err(BorgError::InsufficientSuccesses {
                needed: n,
                got: ok_count,
            });
        }
    }

    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: total_in,
            total_output_tokens: total_out,
            succeeded,
            failed,
        }))
        .await;
    Ok(all)
}

/// State keys one parallel branch wrote or changed in its private
/// handoff copy, merged into the parent when the branch succeeds.
type BranchState = std::collections::HashMap<String, serde_json::Value>;

/// What a parallel branch's task hands back: its index, name, verdicts
/// and the state keys it wrote.
type BranchOutcome = (usize, String, Vec<NamedVerdict>, BranchState);

async fn execute_parallel(
    branches: Vec<(String, Composition)>,
    on_error: ErrorPolicy,
    max_concurrency: usize,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let composition_id: Arc<str> = Arc::from(format!("{}::par", ctx.parent_composition_id));
    // A parallel branch is a composition, not a single agent, so its
    // per-step `model` slot is intentionally blank.
    let empty_model: Arc<str> = Arc::from("");
    let requested = branches.len();

    // Pre-flight budget admission, estimating a flat 1k min cost per branch.
    let admission = ctx.budget.admit_parallel(requested, 1_000);
    let admitted = match admission {
        ParallelAdmission::Full => requested,
        ParallelAdmission::Partial { allowed } => {
            ctx.emitter
                .emit(Arc::new(BorgEvent::ParallelDegraded {
                    composition_id: composition_id.clone(),
                    requested: requested.min(u8::MAX as usize) as u8,
                    admitted: allowed.min(u8::MAX as usize) as u8,
                    reason: "budget admission downgrade".into(),
                }))
                .await;
            allowed
        }
        ParallelAdmission::Denied => {
            // Loud-by-default: zero admitted branches must not look like
            // success. Returning Ok(Vec::new()) would let the parent treat
            // "no signal" as a pass.
            ctx.emitter
                .emit(Arc::new(BorgEvent::ParallelDegraded {
                    composition_id: composition_id.clone(),
                    requested: requested.min(u8::MAX as usize) as u8,
                    admitted: 0,
                    reason: "budget admission denied".into(),
                }))
                .await;
            return Err(BorgError::BudgetExhausted {
                needed: requested as u64 * 1_000,
                remaining: ctx.budget.remaining(),
            });
        }
    };

    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::Parallel,
            branch_count: admitted.min(u8::MAX as usize) as u8,
        }))
        .await;

    let started = std::time::Instant::now();
    let snapshot = Arc::new(handoff.snapshot());
    // Each branch hands back its verdicts plus the state keys it wrote.
    let mut sup: Supervisor<BranchOutcome> = Supervisor::new(ctx.cancel.clone(), max_concurrency);

    // Branch name → index, for a failed child's slot and position (a
    // panicked child never returns its own index).
    let mut index_of: std::collections::HashMap<Arc<str>, usize> =
        std::collections::HashMap::with_capacity(admitted);
    for (idx, (branch_name, branch_comp)) in branches.into_iter().take(admitted).enumerate() {
        // Each branch nests under its own position (`…::par#{idx}`), so
        // two branches of the same shape never share step ids. Branches
        // run without checkpointing: a branch's handoff is a private
        // copy, so a snapshot saved from inside one would miss its
        // siblings' writes. A Parallel re-runs wholesale on resume.
        let nested_ctx_outer = ExecCtx {
            parent_composition_id: Arc::from(format!("{composition_id}#{idx}")),
            checkpoint: None,
            ..ctx.clone()
        };
        let snapshot_for_branch = snapshot.clone();
        let branch_name_arc: Arc<str> = Arc::from(branch_name.as_str());
        index_of.insert(branch_name_arc.clone(), idx);
        let branch_name_owned = branch_name;
        let comp_id_for_event = composition_id.clone();
        let emitter_for_event = ctx.emitter.clone();
        let turn_for_event = ctx.turn_number;
        let empty_model_for_event = empty_model.clone();
        // The branch runs on a task of its own: carry the caller's span
        // and carriers (task-locals) into it.
        let context = ctx.capture_context();

        sup.spawn_in(
            branch_name_arc.clone(),
            context,
            move |child_cancel| async move {
                emitter_for_event
                    .emit(Arc::new(BorgEvent::StepStarted {
                        composition_id: comp_id_for_event,
                        step_name: branch_name_arc,
                        position: idx.min(u8::MAX as usize) as u8,
                        turn_number: turn_for_event,
                        // Blank model: see empty_model above. Per-agent model
                        // is on the inner `StepStarted` events.
                        model: empty_model_for_event,
                    }))
                    .await;
                let nested_ctx = ExecCtx {
                    cancel: child_cancel,
                    ..nested_ctx_outer
                };
                let mut local_handoff = (*snapshot_for_branch).clone();
                let res = execute_into(branch_comp, &nested_ctx, &mut local_handoff).await;
                res.map(|vs| {
                    // Only the keys this branch wrote or changed: an
                    // untouched key must not overwrite a sibling's write.
                    let written = local_handoff
                        .state
                        .into_iter()
                        .filter(|(k, v)| snapshot_for_branch.state.get(k) != Some(v))
                        .collect();
                    (idx, branch_name_owned, vs, written)
                })
            },
        );
    }

    let mut outcomes: Vec<Option<(Vec<NamedVerdict>, BranchState)>> =
        (0..admitted).map(|_| None).collect();
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;

    while let Some(child) = sup.join_next().await {
        let failed_idx = index_of.get(&child.agent_id).copied();
        let position = failed_idx.unwrap_or(0).min(u8::MAX as usize) as u8;
        match child.result {
            Ok((idx, name, vs, written)) => {
                let mut input = 0u64;
                let mut output = 0u64;
                for v in &vs {
                    input = input.saturating_add(v.verdict.input_tokens);
                    output = output.saturating_add(v.verdict.output_tokens);
                }
                total_in = total_in.saturating_add(input);
                total_out = total_out.saturating_add(output);
                ctx.emitter
                    .emit(Arc::new(BorgEvent::StepCompleted {
                        composition_id: composition_id.clone(),
                        step_name: Arc::from(name.as_str()),
                        position: idx.min(u8::MAX as usize) as u8,
                        duration_ms: child.duration_ms,
                        input_tokens: input,
                        output_tokens: output,
                        success: true,
                        failure_cause: None,
                        turn_number: ctx.turn_number,
                        // Blank model: see empty_model above.
                        model: empty_model.clone(),
                    }))
                    .await;
                outcomes[idx] = Some((vs, written));
            }
            Err(e) => match on_error {
                ErrorPolicy::FailFast => {
                    sup.cancel_all();
                    // Drain remaining children so we don't leak tasks.
                    while sup.join_next().await.is_some() {}
                    ctx.emitter
                        .emit(Arc::new(BorgEvent::CompositionCompleted {
                            composition_id,
                            total_duration_ms: started.elapsed().as_millis() as u64,
                            total_input_tokens: total_in,
                            total_output_tokens: total_out,
                            succeeded: saturating_u8(
                                outcomes.iter().filter(|o| o.is_some()).count(),
                            ),
                            failed: 1,
                        }))
                        .await;
                    return Err(e);
                }
                ErrorPolicy::CollectAll | ErrorPolicy::AtLeast(_) => {
                    let agent_label = child.agent_id.to_string();
                    let cause = e.failure_cause().to_string();
                    let nv = NamedVerdict::failure(
                        agent_label.clone(),
                        format!("{e}"),
                        child.duration_ms,
                    );
                    ctx.emitter
                        .emit(Arc::new(BorgEvent::StepCompleted {
                            composition_id: composition_id.clone(),
                            step_name: Arc::from(agent_label.as_str()),
                            position,
                            duration_ms: child.duration_ms,
                            input_tokens: 0,
                            output_tokens: 0,
                            success: false,
                            failure_cause: Some(cause),
                            turn_number: ctx.turn_number,
                            model: empty_model.clone(),
                        }))
                        .await;
                    // The child's own slot (its name survives a panic);
                    // the first free one only if the name is unknown.
                    let slot = failed_idx
                        .filter(|&i| outcomes[i].is_none())
                        .or_else(|| outcomes.iter().position(|o| o.is_none()));
                    if let Some(p) = slot {
                        outcomes[p] = Some((vec![nv], BranchState::new()));
                    }
                }
            },
        }
    }

    // Merge results into the parent handoff, in branch order: history,
    // then the state keys each branch wrote. Builder-time validation
    // guarantees distinct output_keys; a key several branches write
    // (e.g. the handoff target) goes to the last branch in order.
    let mut all = Vec::new();
    let mut succeeded: u8 = 0;
    let mut failed: u8 = 0;
    for slot in outcomes.into_iter().flatten() {
        let (vs, written) = slot;
        handoff.state.extend(written);
        for v in &vs {
            if v.is_error {
                failed = failed.saturating_add(1);
            } else {
                succeeded = succeeded.saturating_add(1);
            }
            handoff.record(v.clone());
        }
        all.extend(vs);
    }

    if let ErrorPolicy::AtLeast(n) = on_error {
        let ok_count = all.iter().filter(|v| !v.is_error).count();
        if ok_count < n {
            ctx.emitter
                .emit(Arc::new(BorgEvent::CompositionCompleted {
                    composition_id,
                    total_duration_ms: started.elapsed().as_millis() as u64,
                    total_input_tokens: total_in,
                    total_output_tokens: total_out,
                    succeeded,
                    failed,
                }))
                .await;
            return Err(BorgError::InsufficientSuccesses {
                needed: n,
                got: ok_count,
            });
        }
    }

    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: total_in,
            total_output_tokens: total_out,
            succeeded,
            failed,
        }))
        .await;
    Ok(all)
}

async fn execute_loop_until(
    body: crate::composition::LoopBody,
    until: crate::composition::UntilFn,
    max_iterations: u32,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let composition_id: Arc<str> = Arc::from(format!("{}::loop", ctx.parent_composition_id));
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::LoopUntil,
            branch_count: 1,
        }))
        .await;

    let started = std::time::Instant::now();
    let mut all: Vec<NamedVerdict> = Vec::new();
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;
    let mut last_verdict_owned: Option<Verdict> = None;

    // Per-iteration `ExecCtx` so children attach under
    // `parent::loop::iter<N>`, keeping the event tree readable.
    for iteration in 0..max_iterations {
        if ctx.cancel.is_cancelled() {
            ctx.emitter
                .emit(Arc::new(BorgEvent::Cancelled {
                    composition_id: composition_id.clone(),
                    reason: "parent_cancel".into(),
                }))
                .await;
            return Err(BorgError::Cancelled);
        }

        let iter_id: Arc<str> = Arc::from(format!("{composition_id}::iter{iteration}"));

        // Checkpoint fast-forward: skip the body if this iteration's id is
        // already in the completed-steps set — its handoff state is already
        // restored, so re-running would only burn tokens.
        let mut already_completed = false;
        if let Some(cp) = &ctx.checkpoint {
            let set = cp.completed_steps.lock();
            if set.contains(&iter_id) {
                already_completed = true;
            }
        }
        if already_completed {
            // Recover the iteration's tail verdict so the until-predicate
            // sees what the original run would have. History is keyed by
            // inner step name, so take the most recent entry.
            last_verdict_owned = handoff.history.last().map(|nv| nv.verdict.clone());
            if let Some(v) = last_verdict_owned.as_ref() {
                if until(v) {
                    break;
                }
            }
            continue;
        }

        // Build the iteration's body via the factory, which sees a
        // read-only handoff view plus the prior iteration's last verdict.
        let loop_ctx = crate::composition::LoopCtx {
            iteration,
            handoff,
            last_verdict: last_verdict_owned.as_ref(),
        };
        let inner_comp = body(loop_ctx);

        let iter_ctx = ExecCtx {
            parent_composition_id: Arc::clone(&iter_id),
            ..ctx.clone()
        };

        let vs_inner = execute_into(inner_comp, &iter_ctx, handoff).await?;

        // Accumulate tokens; capture the tail verdict for the next
        // iteration's `LoopCtx.last_verdict`.
        for v in &vs_inner {
            total_in = total_in.saturating_add(v.verdict.input_tokens);
            total_out = total_out.saturating_add(v.verdict.output_tokens);
        }
        last_verdict_owned = vs_inner.last().map(|nv| nv.verdict.clone());
        all.extend(vs_inner);

        // Checkpoint save-point: mark this iteration complete and persist.
        // Cost is one store.save per iteration (~one LLM turn).
        if let Some(cp) = &ctx.checkpoint {
            cp.completed_steps.lock().insert(iter_id.clone());
            if matches!(
                cp.policy,
                crate::checkpoint::CheckpointPolicy::OnLoopIteration
                    | crate::checkpoint::CheckpointPolicy::OnEveryStep
            ) {
                let snap = crate::checkpoint::Snapshot {
                    schema_version: crate::checkpoint::SNAPSHOT_SCHEMA_VERSION,
                    composition_fingerprint: cp.composition_fingerprint.clone(),
                    handoff: handoff.to_snapshot(&ctx.handoff_persistence),
                    completed_steps: cp.completed_steps.lock().iter().cloned().collect(),
                    turn_number: ctx.turn_number,
                    created_at: std::time::SystemTime::now(),
                };
                if let Err(e) = cp.store.save(&cp.id, &snap).await {
                    tracing::warn!(
                        composition_id = %composition_id,
                        iteration,
                        error = %e,
                        "checkpoint save failed; execution continues but resume may not work"
                    );
                }
            }
        }

        // Stop if the predicate matches. Empty iterations (no verdict
        // produced) fall through to the next.
        if let Some(v) = last_verdict_owned.as_ref() {
            if until(v) {
                break;
            }
        }
    }

    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: total_in,
            total_output_tokens: total_out,
            succeeded: saturating_u8(all.iter().filter(|v| !v.is_error).count()),
            failed: saturating_u8(all.iter().filter(|v| v.is_error).count()),
        }))
        .await;
    Ok(all)
}

async fn execute_branch(
    selector: crate::composition::SelectorFn,
    branches: Vec<(crate::composition::BranchKey, Composition)>,
    default: Option<Box<Composition>>,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let composition_id: Arc<str> = Arc::from(format!("{}::branch", ctx.parent_composition_id));
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::Branch,
            branch_count: branches.len().min(u8::MAX as usize) as u8,
        }))
        .await;

    let key = selector(handoff);
    let chosen = branches
        .into_iter()
        .enumerate()
        .find(|(_, (k, _))| k == &key);
    // The arm nests under its own position (`…::branch#{idx}`, or
    // `…::branch#default`), so arms of the same shape never share ids.
    let (composition_to_run, matched, arm) = match (chosen, default) {
        (Some((idx, (_, c))), _) => (c, true, idx.to_string()),
        (None, Some(d)) => (*d, false, "default".to_string()),
        (None, None) => {
            ctx.emitter
                .emit(Arc::new(BorgEvent::BranchDecided {
                    composition_id: composition_id.clone(),
                    chosen: key.as_str().to_string(),
                    matched: false,
                }))
                .await;
            return Err(BorgError::UnmatchedBranch {
                key: key.into_inner(),
            });
        }
    };
    ctx.emitter
        .emit(Arc::new(BorgEvent::BranchDecided {
            composition_id: composition_id.clone(),
            chosen: key.into_inner(),
            matched,
        }))
        .await;

    let nested_ctx = ExecCtx {
        parent_composition_id: Arc::from(format!("{composition_id}#{arm}")),
        ..ctx.clone()
    };
    let started = std::time::Instant::now();
    let result = execute_into(composition_to_run, &nested_ctx, handoff).await;
    let (succeeded, failed, vs) = match &result {
        Ok(v) => (
            saturating_u8(v.iter().filter(|x| !x.is_error).count()),
            saturating_u8(v.iter().filter(|x| x.is_error).count()),
            v.clone(),
        ),
        Err(_) => (0, 1, Vec::new()),
    };
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: vs
                .iter()
                .map(|v| v.verdict.input_tokens)
                .fold(0u64, |a, b| a.saturating_add(b)),
            total_output_tokens: vs
                .iter()
                .map(|v| v.verdict.output_tokens)
                .fold(0u64, |a, b| a.saturating_add(b)),
            succeeded,
            failed,
        }))
        .await;
    result
}

async fn execute_fallback(
    primary: Composition,
    fallbacks: Vec<Composition>,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let composition_id: Arc<str> = Arc::from(format!("{}::fb", ctx.parent_composition_id));
    let total = fallbacks.len() + 1;
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::Fallback,
            branch_count: total.min(u8::MAX as usize) as u8,
        }))
        .await;

    let started = std::time::Instant::now();
    let nested_ctx = ExecCtx {
        parent_composition_id: composition_id.clone(),
        ..ctx.clone()
    };
    match execute_into(primary, &nested_ctx, handoff).await {
        Ok(vs) => {
            let in_tot = vs
                .iter()
                .map(|v| v.verdict.input_tokens)
                .fold(0u64, |a, b| a.saturating_add(b));
            let out_tot = vs
                .iter()
                .map(|v| v.verdict.output_tokens)
                .fold(0u64, |a, b| a.saturating_add(b));
            ctx.emitter
                .emit(Arc::new(BorgEvent::CompositionCompleted {
                    composition_id,
                    total_duration_ms: started.elapsed().as_millis() as u64,
                    total_input_tokens: in_tot,
                    total_output_tokens: out_tot,
                    succeeded: 1,
                    failed: 0,
                }))
                .await;
            return Ok(vs);
        }
        Err(e) if !e.is_retryable() => {
            ctx.emitter
                .emit(Arc::new(BorgEvent::CompositionCompleted {
                    composition_id,
                    total_duration_ms: started.elapsed().as_millis() as u64,
                    total_input_tokens: 0,
                    total_output_tokens: 0,
                    succeeded: 0,
                    failed: 1,
                }))
                .await;
            return Err(e);
        }
        Err(_) => {}
    }

    for (idx, fb) in fallbacks.into_iter().enumerate() {
        if ctx.cancel.is_cancelled() {
            return Err(BorgError::Cancelled);
        }
        let nested_ctx = ExecCtx {
            parent_composition_id: Arc::from(format!("{composition_id}#fb{idx}")),
            ..ctx.clone()
        };
        match execute_into(fb, &nested_ctx, handoff).await {
            Ok(vs) => {
                let in_tot = vs
                    .iter()
                    .map(|v| v.verdict.input_tokens)
                    .fold(0u64, |a, b| a.saturating_add(b));
                let out_tot = vs
                    .iter()
                    .map(|v| v.verdict.output_tokens)
                    .fold(0u64, |a, b| a.saturating_add(b));
                ctx.emitter
                    .emit(Arc::new(BorgEvent::CompositionCompleted {
                        composition_id,
                        total_duration_ms: started.elapsed().as_millis() as u64,
                        total_input_tokens: in_tot,
                        total_output_tokens: out_tot,
                        succeeded: 1,
                        failed: 0,
                    }))
                    .await;
                return Ok(vs);
            }
            Err(e) if !e.is_retryable() => {
                ctx.emitter
                    .emit(Arc::new(BorgEvent::CompositionCompleted {
                        composition_id,
                        total_duration_ms: started.elapsed().as_millis() as u64,
                        total_input_tokens: 0,
                        total_output_tokens: 0,
                        succeeded: 0,
                        failed: 1,
                    }))
                    .await;
                return Err(e);
            }
            Err(_) => continue,
        }
    }

    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: 0,
            total_output_tokens: 0,
            succeeded: 0,
            failed: 1,
        }))
        .await;
    Err(BorgError::Internal {
        message: "all fallbacks exhausted".into(),
    })
}

/// Run a [`Composition::FallbackLadder`]: rungs in order until one ships.
/// Ids match [`Composition::Fallback`]'s and follow the rung's position,
/// not how many rungs ran before it: rung 0 nests under `…::fb`, rung
/// `n > 0` under `…::fb#fb{n-1}`, so a rung keeps its id when a `when`
/// guard skips an earlier one.
async fn execute_fallback_ladder(
    rungs: Vec<crate::ladder::LadderRung>,
    falls_through: crate::ladder::FallsThroughFn,
    on_fall: Option<crate::ladder::OnFallFn>,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    use crate::ladder::RungFailure;

    let composition_id: Arc<str> = Arc::from(format!("{}::fb", ctx.parent_composition_id));
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::Fallback,
            branch_count: rungs.len().min(u8::MAX as usize) as u8,
        }))
        .await;
    let started = std::time::Instant::now();
    let completed = |succeeded: u8, failed: u8, vs: Option<&[NamedVerdict]>| {
        let (i, o) = vs.map_or((0, 0), |vs| {
            vs.iter().fold((0u64, 0u64), |(i, o), v| {
                (
                    i.saturating_add(v.verdict.input_tokens),
                    o.saturating_add(v.verdict.output_tokens),
                )
            })
        });
        Arc::new(BorgEvent::CompositionCompleted {
            composition_id: composition_id.clone(),
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: i,
            total_output_tokens: o,
            succeeded,
            failed,
        })
    };

    let applies =
        |r: &crate::ladder::LadderRung, h: &HandoffState, f: &[RungFailure<BorgError>]| {
            r.when.as_ref().is_none_or(|w| w(h, f))
        };
    let mut failures: Vec<RungFailure<BorgError>> = Vec::new();
    let mut rungs: Vec<Option<crate::ladder::LadderRung>> = rungs.into_iter().map(Some).collect();
    let mut next = (0..rungs.len()).find(|&i| applies(rungs[i].as_ref().unwrap(), handoff, &[]));
    while let Some(index) = next {
        if ctx.cancel.is_cancelled() {
            ctx.emitter
                .emit(completed(
                    0,
                    failures.len().min(u8::MAX as usize) as u8,
                    None,
                ))
                .await;
            return Err(BorgError::Cancelled);
        }
        let rung = rungs[index].take().expect("each rung runs at most once");
        let label: Arc<str> = Arc::from(rung.body.label());
        let nested_ctx = ExecCtx {
            parent_composition_id: if index == 0 {
                composition_id.clone()
            } else {
                Arc::from(format!("{composition_id}#fb{}", index - 1))
            },
            ..ctx.clone()
        };
        match execute_into(rung.body, &nested_ctx, handoff).await {
            Ok(vs) => {
                let failed = failures.len().min(u8::MAX as usize) as u8;
                ctx.emitter.emit(completed(1, failed, Some(&vs))).await;
                return Ok(vs);
            }
            Err(error) => {
                // Control signals never fall through: a cancel or a
                // suspension belongs to the caller, not the next rung.
                let control = error.is_cancelled() || matches!(error, BorgError::Suspended { .. });
                let falls = !control
                    && rung
                        .falls_through
                        .as_ref()
                        .map_or_else(|| falls_through(&error), |own| own(&error));
                if !falls {
                    let failed = (failures.len() + 1).min(u8::MAX as usize) as u8;
                    ctx.emitter.emit(completed(0, failed, None)).await;
                    return Err(error);
                }
                failures.push(RungFailure {
                    rung: label,
                    index,
                    error,
                });
                next = (index + 1..rungs.len()).find(|&i| {
                    rungs[i]
                        .as_ref()
                        .is_some_and(|r| applies(r, handoff, &failures))
                });
                if let Some(hook) = &on_fall {
                    let to = next.and_then(|i| rungs[i].as_ref()).map(|r| r.body.label());
                    hook(
                        failures.last().expect("just pushed"),
                        to.as_deref(),
                        handoff,
                    );
                }
            }
        }
    }

    let failed = failures.len().min(u8::MAX as usize) as u8;
    ctx.emitter.emit(completed(0, failed, None)).await;
    Err(BorgError::FallbackExhausted { failures })
}

/// Static label for a [`crate::Provider`] — the `&'static str` that
/// `BorgError::ProviderUnavailable` needs.
fn provider_label(p: crate::agent::Provider) -> &'static str {
    match p {
        crate::agent::Provider::OpenRouter => "openrouter",
        crate::agent::Provider::Anthropic => "anthropic",
        crate::agent::Provider::OpenAI => "openai",
        crate::agent::Provider::Bedrock => "bedrock",
        crate::agent::Provider::Google => "google",
        crate::agent::Provider::Together => "together",
        crate::agent::Provider::Custom(s) => s,
    }
}

/// Drive an agent → tool dispatch → repeat loop.
///
/// Each iteration is one agent invocation. The agent reads accumulated
/// tool results from the handoff slot `"tool_results"` (a
/// `Vec<DispatchedCall>`); the executor appends after each dispatch round.
///
/// Terminates when `extract_tool_calls` returns empty (normal completion),
/// `max_steps` is reached with the last turn still calling tools
/// (`BorgError::MaxStepsReached`), the cancel token fires
/// (`BorgError::Cancelled`), or `invoke` returns an unrecoverable error.
/// At most [`ExecCtx::max_tool_calls_per_turn`] calls of a turn run; the
/// rest get a failed result.
async fn execute_agent_loop(
    name: String,
    agent: Box<dyn crate::agent::BorgAgent>,
    tools: std::sync::Arc<dyn crate::tools::ToolDispatch>,
    max_steps: u32,
    ctx: &ExecCtx,
    handoff: &mut HandoffState,
) -> Result<Vec<NamedVerdict>, BorgError> {
    let name: Arc<str> = Arc::from(name);
    let composition_id: Arc<str> =
        Arc::from(format!("{}::agent::{}", ctx.parent_composition_id, name));
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: composition_id.clone(),
            parent_id: Some(ctx.parent_composition_id.clone()),
            shape: CompositionShape::AgentLoop,
            branch_count: 1,
        }))
        .await;

    let started = std::time::Instant::now();
    // handoff.history is the single source of truth for per-step verdicts;
    // track the starting index and slice the new range at exit rather than
    // accumulating a parallel Vec.
    let history_start = handoff.history.len();
    // Tool-call accumulator with O(1) per-step append: a JSON
    // Value::Array handed to handoff verbatim, so each step pushes one
    // serialized call instead of re-serializing a growing Vec (O(N²)
    // across max_steps). `max_steps` is a ceiling, not a size hint: a
    // huge one (`u32::MAX`) must not become a huge up-front allocation.
    let mut accumulated_tool_results_json: serde_json::Value =
        serde_json::Value::Array(Vec::with_capacity((max_steps as usize).min(64)));
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;
    let mut last_verdict: Option<Verdict> = None;
    // Set when a turn asked for no tools: the agent gave its answer.
    let mut finished = false;

    for step in 0..max_steps {
        if ctx.cancel.is_cancelled() {
            ctx.emitter
                .emit(Arc::new(BorgEvent::Cancelled {
                    composition_id: composition_id.clone(),
                    reason: "parent_cancel".into(),
                }))
                .await;
            return Err(BorgError::Cancelled);
        }

        // Seed the tool_results slot (the JSON mirror) so the agent's
        // user_prompt sees accumulated calls.
        handoff.set("tool_results", &accumulated_tool_results_json);

        // Per-step composition id so trace viewers distinguish iterations.
        let step_id: Arc<str> = Arc::from(format!("{composition_id}::step{step}"));
        let step_ctx = ExecCtx {
            parent_composition_id: step_id.clone(),
            ..ctx.clone()
        };

        // Schemas go into the agent context so the default `invoke()`
        // attaches them to `ChatRequest::tools` as native tool_use schemas.
        let step_started = std::time::Instant::now();
        let agent_ctx = step_ctx.agent_ctx_with_tools(
            handoff,
            ctx.cancel.child_token(),
            name.clone(),
            tools.schemas(),
        );

        // `step_name` carries the iteration index (`<agent_id>-step0`, ...)
        // so observers can correlate iterations without timing inference.
        let step_name: Arc<str> = Arc::from(format!("{name}-step{step}"));
        let step_model: Arc<str> = Arc::from(agent.model());
        ctx.emitter
            .emit(Arc::new(BorgEvent::StepStarted {
                composition_id: composition_id.clone(),
                step_name: step_name.clone(),
                position: u8::try_from(step).unwrap_or(u8::MAX),
                turn_number: ctx.turn_number,
                model: step_model.clone(),
            }))
            .await;

        ctx.emitter
            .emit(Arc::new(BorgEvent::TurnStarted {
                composition_id: composition_id.clone(),
                step_name: step_name.clone(),
                turn_index: step,
                model: step_model.clone(),
                input_tokens: 0, // not known before the LLM call; TurnCompleted carries actual tokens
            }))
            .await;

        // Instrument with the live borg.turn span so spans created inside
        // the agent nest under borg.turn in OTLP exporters.
        let turn_span = ctx.emitter.turn_span(&composition_id, &step_name, step);
        let invoked = std::panic::AssertUnwindSafe(agent.invoke(&agent_ctx).instrument(turn_span))
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| Err(agent_panicked(agent.as_ref(), payload)));
        let verdict = match invoked {
            Ok(v) => v,
            Err(e) => {
                // Mirror Sequential's failure-emit shape for a uniform
                // step lifecycle.
                let cause = e.failure_cause().to_string();
                ctx.emitter
                    .emit(Arc::new(BorgEvent::StepCompleted {
                        composition_id: composition_id.clone(),
                        step_name: step_name.clone(),
                        position: u8::try_from(step).unwrap_or(u8::MAX),
                        duration_ms: step_started.elapsed().as_millis() as u64,
                        input_tokens: 0,
                        output_tokens: 0,
                        success: false,
                        failure_cause: Some(cause),
                        turn_number: ctx.turn_number,
                        model: step_model.clone(),
                    }))
                    .await;
                ctx.emitter
                    .emit(Arc::new(BorgEvent::TurnCompleted {
                        composition_id: composition_id.clone(),
                        step_name: step_name.clone(),
                        turn_index: step,
                        output_tokens: 0,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                        first_token_ms: None,
                    }))
                    .await;
                return Err(e);
            }
        };
        let step_duration = step_started.elapsed().as_millis() as u64;

        total_in = total_in.saturating_add(verdict.input_tokens);
        total_out = total_out.saturating_add(verdict.output_tokens);

        // Record a `model:text` payload-hash signal so `IdenticalOutputRule`
        // can detect the model repeating its previous response verbatim.
        if let Some(harness) = &ctx.guardrails {
            harness.record(crate::guardrails::LoopSignal {
                key: "model:text".into(),
                payload_hash: Some(fnv1a_64(&verdict.body)),
                at: std::time::Instant::now(),
            });
        }

        ctx.emitter
            .emit(Arc::new(BorgEvent::StepCompleted {
                composition_id: composition_id.clone(),
                step_name: step_name.clone(),
                position: u8::try_from(step).unwrap_or(u8::MAX),
                duration_ms: step_duration,
                input_tokens: verdict.input_tokens,
                output_tokens: verdict.output_tokens,
                success: true,
                failure_cause: None,
                turn_number: ctx.turn_number,
                model: step_model.clone(),
            }))
            .await;

        ctx.emitter
            .emit(Arc::new(BorgEvent::TurnCompleted {
                composition_id: composition_id.clone(),
                step_name: step_name.clone(),
                turn_index: step,
                output_tokens: verdict.output_tokens,
                cache_read_tokens: verdict.cache_read_tokens,
                cache_creation_tokens: verdict.cache_creation_tokens,
                first_token_ms: None,
            }))
            .await;

        // Extract tool calls before recording the verdict (extract borrows it).
        let tool_calls = agent.extract_tool_calls(&verdict);
        let is_text_only = tool_calls.is_empty();

        // Record into handoff history; the exit slice rebuilds the returned Vec.
        let nv = NamedVerdict::success(step_name.as_ref(), verdict.clone(), step_duration);
        handoff.record(nv);
        last_verdict = Some(verdict);

        // Auto-emit one of two mutually-exclusive end-of-turn markers per
        // iteration, so guardrail rules can use plain `consecutive()`.
        if let Some(harness) = &ctx.guardrails {
            if is_text_only {
                harness.record(crate::guardrails::LoopSignal::now("agent:text_only_turn"));
            } else {
                harness.record(crate::guardrails::LoopSignal::now("agent:turn_completed"));
            }
        }

        // Calls past the per-turn cap are answered without running.
        let mut tool_calls = tool_calls;
        let over_cap = if tool_calls.len() > ctx.max_tool_calls_per_turn {
            tool_calls.split_off(ctx.max_tool_calls_per_turn)
        } else {
            Vec::new()
        };

        // Dispatch this turn's tool calls. The default serial path
        // preserves historical event/signal ordering; the parallel path
        // groups contiguous parallel-safe calls into JoinSet runs and keeps
        // the rest serial. Each dispatched call is pushed onto the JSON
        // mirror (O(1), no full re-serialize).
        if !ctx.allow_parallel_tools {
            for request in tool_calls {
                let dispatched = dispatch_one_tool(
                    request,
                    &tools,
                    ctx,
                    &composition_id,
                    &step_name,
                    step,
                    None,
                )
                .await?;
                let dispatched_json =
                    serde_json::to_value(&dispatched).map_err(|e| BorgError::Internal {
                        message: format!("agent loop: failed to serialize dispatched call: {e}"),
                    })?;
                if let serde_json::Value::Array(arr) = &mut accumulated_tool_results_json {
                    arr.push(dispatched_json);
                }
            }
        } else {
            let results =
                dispatch_tool_batch(tool_calls, &tools, ctx, &composition_id, &step_name, step)
                    .await?;
            for dispatched in &results {
                let dispatched_json =
                    serde_json::to_value(dispatched).map_err(|e| BorgError::Internal {
                        message: format!("agent loop: failed to serialize dispatched call: {e}"),
                    })?;
                if let serde_json::Value::Array(arr) = &mut accumulated_tool_results_json {
                    arr.push(dispatched_json);
                }
            }
        }
        for request in over_cap {
            let rejected = reject_tool_call(request, ctx, &composition_id, &step_name, step).await;
            let rejected_json =
                serde_json::to_value(&rejected).map_err(|e| BorgError::Internal {
                    message: format!("agent loop: failed to serialize dispatched call: {e}"),
                })?;
            if let serde_json::Value::Array(arr) = &mut accumulated_tool_results_json {
                arr.push(rejected_json);
            }
        }

        // Evaluate the guardrail engine at the iteration boundary, after
        // every signal for this turn is recorded. `Halt` terminates with
        // `BorgError::Guardrail`; `Warn`/`Throttle` are side-effected into
        // the harness for the next turn to read.
        if let Some(harness) = &ctx.guardrails {
            let outcome = harness.evaluate(step, ctx.budget.as_ref());
            let rule: Arc<str> = Arc::from(outcome.winning_rule.as_deref().unwrap_or("unknown"));
            match &outcome.winning_verdict {
                crate::guardrails::GuardrailVerdict::Ok => {}
                crate::guardrails::GuardrailVerdict::Warn { corrective } => {
                    ctx.emitter
                        .emit(Arc::new(BorgEvent::GuardrailTriggered {
                            composition_id: composition_id.clone(),
                            step_name: step_name.clone(),
                            rule_name: rule.clone(),
                            verdict: "warn".to_string(),
                            signal: corrective.clone(),
                        }))
                        .await;
                }
                crate::guardrails::GuardrailVerdict::Throttle { corrective, .. } => {
                    ctx.emitter
                        .emit(Arc::new(BorgEvent::GuardrailTriggered {
                            composition_id: composition_id.clone(),
                            step_name: step_name.clone(),
                            rule_name: rule.clone(),
                            verdict: "throttle".to_string(),
                            signal: corrective.clone(),
                        }))
                        .await;
                }
                crate::guardrails::GuardrailVerdict::Halt { reason } => {
                    ctx.emitter
                        .emit(Arc::new(BorgEvent::GuardrailTriggered {
                            composition_id: composition_id.clone(),
                            step_name: step_name.clone(),
                            rule_name: rule.clone(),
                            verdict: "halt".to_string(),
                            signal: reason.clone(),
                        }))
                        .await;
                }
            }
            if let crate::guardrails::GuardrailVerdict::Halt { reason } = outcome.winning_verdict {
                let rule = outcome
                    .winning_rule
                    .map(|c| c.into_owned())
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(BorgError::Guardrail { rule, reason });
            }
        }

        // No tool calls → agent is done. Checked after guardrail eval so a
        // Halt on a text-only turn wins over the natural exit.
        if is_text_only {
            tracing::debug!(
                composition_id = %composition_id,
                step,
                "agent loop: stopping (no tool calls emitted)"
            );
            finished = true;
            break;
        }
    }

    // Count succeeded/failed over the new-this-call history slice.
    let succeeded = saturating_u8(
        handoff.history[history_start..]
            .iter()
            .filter(|v| !v.is_error)
            .count(),
    );
    let failed = saturating_u8(
        handoff.history[history_start..]
            .iter()
            .filter(|v| v.is_error)
            .count(),
    );
    ctx.emitter
        .emit(Arc::new(BorgEvent::CompositionCompleted {
            composition_id,
            total_duration_ms: started.elapsed().as_millis() as u64,
            total_input_tokens: total_in,
            total_output_tokens: total_out,
            succeeded,
            failed,
        }))
        .await;

    // Every step ran and the last one still asked for tools: the agent
    // never answered. (No step at all, `max_steps == 0`, is not that.)
    if !finished && last_verdict.is_some() {
        tracing::debug!(
            agent = %name,
            max_steps,
            "agent loop: max_steps reached while the agent was still calling tools"
        );
        return Err(BorgError::MaxStepsReached {
            agent_id: name.to_string(),
            max_steps,
        });
    }

    // Materialize the returned Vec from the history slice at exit.
    let all_verdicts: Vec<NamedVerdict> = handoff.history[history_start..].to_vec();
    Ok(all_verdicts)
}

// ─────────────────────────────────────────────────────────────────────
//   Tool-call dispatch helpers — used by execute_agent_loop above
// ─────────────────────────────────────────────────────────────────────

/// Monotonic batch id for parallel tool runs (process-local).
static BATCH_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-call dispatch: emits guardrail signals + `ToolCalled`/`ToolReturned`
/// events around `ToolDispatch::dispatch`. Used by the serial path and
/// each parallel `JoinSet` task. `Err(BorgError)` propagates from the
/// tool's own `execute`; a cancel of `ctx.cancel` mid-call drops the
/// tool's future and returns `BorgError::Cancelled` (no `ToolReturned`).
async fn dispatch_one_tool(
    request: crate::tools::ToolCallRequest,
    tools: &Arc<dyn crate::tools::ToolDispatch>,
    ctx: &ExecCtx,
    composition_id: &Arc<str>,
    step_name: &Arc<str>,
    turn_index: u32,
    batch_id: Option<&Arc<str>>,
) -> Result<crate::tools::DispatchedCall, BorgError> {
    let tool_name: Arc<str> = Arc::from(request.name.as_str());
    let call_id: Arc<str> = Arc::from(request.call_id.as_str());
    if let Some(harness) = &ctx.guardrails {
        harness.record(crate::guardrails::LoopSignal {
            key: format!("tool:{tool_name}:invoked").into(),
            payload_hash: None,
            at: std::time::Instant::now(),
        });
    }
    ctx.emitter
        .emit(Arc::new(BorgEvent::ToolCalled {
            composition_id: Arc::clone(composition_id),
            step_name: Arc::clone(step_name),
            turn_index,
            call_id: Arc::clone(&call_id),
            tool_name: Arc::clone(&tool_name),
            batch_id: batch_id.map(Arc::clone),
        }))
        .await;
    let tool_span = ctx
        .emitter
        .tool_span(composition_id, step_name, turn_index, &call_id);
    // Race the tool against the execution's cancel token (biased toward
    // cancel, like the parallel path), so a slow tool on the serial path
    // can't hold a cancelled loop open until it finishes.
    // Panic boundary: a panicking tool is a structural error, on the
    // serial path as on the parallel one (where `CollectAll` turns it
    // into that call's failed result).
    let call =
        std::panic::AssertUnwindSafe(tools.dispatch(request).instrument(tool_span)).catch_unwind();
    let dispatched = tokio::select! {
        biased;
        _ = ctx.cancel.cancelled() => return Err(BorgError::Cancelled),
        r = call => r.unwrap_or_else(|payload| Err(tool_panicked(&tool_name, payload)))?,
    };
    let tool_success = dispatched.result.error.is_none();
    if let Some(harness) = &ctx.guardrails {
        let key = if dispatched.result.error.is_some() {
            format!("tool:{tool_name}:failed")
        } else {
            format!("tool:{tool_name}:succeeded")
        };
        harness.record(crate::guardrails::LoopSignal {
            key: key.into(),
            payload_hash: None,
            at: std::time::Instant::now(),
        });
    }
    ctx.emitter
        .emit(Arc::new(BorgEvent::ToolReturned {
            composition_id: Arc::clone(composition_id),
            step_name: Arc::clone(step_name),
            turn_index,
            call_id,
            tool_name,
            success: tool_success,
            error: dispatched.result.error.clone(),
            batch_id: batch_id.map(Arc::clone),
        }))
        .await;
    Ok(dispatched)
}

/// Answer a call past [`ExecCtx::max_tool_calls_per_turn`] without
/// running it: a failed result, with the usual `ToolCalled` /
/// `ToolReturned` pair so observers see every requested call.
async fn reject_tool_call(
    request: crate::tools::ToolCallRequest,
    ctx: &ExecCtx,
    composition_id: &Arc<str>,
    step_name: &Arc<str>,
    turn_index: u32,
) -> crate::tools::DispatchedCall {
    let tool_name: Arc<str> = Arc::from(request.name.as_str());
    let call_id: Arc<str> = Arc::from(request.call_id.as_str());
    let result = crate::tools::ToolCallResult::fail(format!(
        "too many tool calls in one turn; limit {}",
        ctx.max_tool_calls_per_turn
    ));
    ctx.emitter
        .emit(Arc::new(BorgEvent::ToolCalled {
            composition_id: Arc::clone(composition_id),
            step_name: Arc::clone(step_name),
            turn_index,
            call_id: Arc::clone(&call_id),
            tool_name: Arc::clone(&tool_name),
            batch_id: None,
        }))
        .await;
    ctx.emitter
        .emit(Arc::new(BorgEvent::ToolReturned {
            composition_id: Arc::clone(composition_id),
            step_name: Arc::clone(step_name),
            turn_index,
            call_id,
            tool_name,
            success: false,
            error: result.error.clone(),
            batch_id: None,
        }))
        .await;
    crate::tools::DispatchedCall {
        call_id: request.call_id,
        name: request.name,
        result,
    }
}

/// A caught panic from a tool call: the structural error both dispatch
/// paths report.
fn tool_panicked(tool_name: &str, payload: Box<dyn std::any::Any + Send>) -> BorgError {
    BorgError::Internal {
        message: format!(
            "tool call `{tool_name}` panicked: {}",
            crate::supervisor::panic_payload_to_string(payload)
        ),
    }
}

/// Grouped dispatch for the parallel-enabled path.
///
/// Groups contiguous parallel-safe calls into `Run::Parallel` and the rest
/// into `Run::Sequential`. Parallel runs of length > 1 dispatch via
/// `JoinSet`; everything else dispatches inline. Results are reconstructed
/// in original request order regardless of completion order.
async fn dispatch_tool_batch(
    tool_calls: Vec<crate::tools::ToolCallRequest>,
    tools: &Arc<dyn crate::tools::ToolDispatch>,
    ctx: &ExecCtx,
    composition_id: &Arc<str>,
    step_name: &Arc<str>,
    turn_index: u32,
) -> Result<Vec<crate::tools::DispatchedCall>, BorgError> {
    enum Run {
        Parallel(Vec<(usize, crate::tools::ToolCallRequest)>),
        Sequential(Vec<(usize, crate::tools::ToolCallRequest)>),
    }

    // Phase 1: group into ordered runs by parallel_safe(name).
    let mut runs: Vec<Run> = Vec::new();
    for (idx, req) in tool_calls.into_iter().enumerate() {
        let safe = tools.parallelize_safe(&req.name);
        match runs.last_mut() {
            Some(Run::Parallel(v)) if safe => v.push((idx, req)),
            Some(Run::Sequential(v)) if !safe => v.push((idx, req)),
            _ => {
                if safe {
                    runs.push(Run::Parallel(vec![(idx, req)]));
                } else {
                    runs.push(Run::Sequential(vec![(idx, req)]));
                }
            }
        }
    }

    // Phase 2: execute runs in order, collecting into a Vec<Option<_>>
    // indexed by original request position; reconstruct in order at the end.
    let mut total_calls = 0usize;
    for r in &runs {
        total_calls += match r {
            Run::Parallel(v) | Run::Sequential(v) => v.len(),
        };
    }
    let mut slots: Vec<Option<crate::tools::DispatchedCall>> =
        (0..total_calls).map(|_| None).collect();

    for run in runs {
        match run {
            Run::Sequential(calls) => {
                for (idx, req) in calls {
                    // batch_id = None marks serial source-order dispatch,
                    // distinguishing it from a parallel batch downstream.
                    let dispatched = dispatch_one_tool(
                        req,
                        tools,
                        ctx,
                        composition_id,
                        step_name,
                        turn_index,
                        None,
                    )
                    .await?;
                    slots[idx] = Some(dispatched);
                }
            }
            Run::Parallel(calls) => {
                if calls.len() == 1 {
                    // Single-call run: skip JoinSet overhead, batch_id None.
                    let (idx, req) = calls.into_iter().next().unwrap();
                    let dispatched = dispatch_one_tool(
                        req,
                        tools,
                        ctx,
                        composition_id,
                        step_name,
                        turn_index,
                        None,
                    )
                    .await?;
                    slots[idx] = Some(dispatched);
                    continue;
                }

                // True parallel: one task per call into a JoinSet, wrapped
                // in a `borg.tool_batch` span so each per-call `borg.tool`
                // span nests under it (the span propagates across
                // `tokio::spawn` via `.instrument`). The batch_id is also
                // surfaced on every `ToolCalled`/`ToolReturned` event so
                // non-tracing consumers can group concurrent tool events.
                // A child-of-batch CancellationToken lets FastFail abort
                // siblings without nuking the wider session.
                let batch_cancel = ctx.cancel.child_token();
                // Process-local id: it's always paired with the
                // (session-unique) composition_id downstream, so no
                // cross-process uniqueness is needed.
                let batch_id: Arc<str> = Arc::from(format!(
                    "batch-{}",
                    BATCH_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                ));
                let batch_span = tracing::info_span!(
                    "borg.tool_batch",
                    batch.id = %batch_id,
                    batch.size = calls.len(),
                    batch.parallel = true,
                );
                // Each task yields its request index with its outcome, so
                // a failed call still knows which slot it fills. A panic
                // inside the tool is caught in the task and becomes a
                // structural error like any other.
                let mut js: tokio::task::JoinSet<(
                    usize,
                    Result<crate::tools::DispatchedCall, BorgError>,
                )> = tokio::task::JoinSet::new();
                let mut requested: std::collections::HashMap<usize, (String, String)> =
                    std::collections::HashMap::with_capacity(calls.len());
                // At most `max_parallel_tool_calls` of the batch run at
                // once; the rest wait (cancellably) for a slot.
                let slots_free = Arc::new(tokio::sync::Semaphore::new(ctx.max_parallel_tool_calls));
                for (idx, req) in calls {
                    requested.insert(idx, (req.name.clone(), req.call_id.clone()));
                    let tools = tools.clone();
                    let ctx_clone = ctx.clone();
                    let composition_id = Arc::clone(composition_id);
                    let step_name = Arc::clone(step_name);
                    let cancel = batch_cancel.clone();
                    let batch_id = batch_id.clone();
                    // Batch span as parent (the per-tool span inside
                    // dispatch_one_tool nests under it), plus the
                    // caller's carriers: each call is a task of its own.
                    let context =
                        crate::propagation::capture_with_span(&ctx.carriers, batch_span.clone());
                    let slots_free = Arc::clone(&slots_free);
                    js.spawn(context.scope(async move {
                        let _slot = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => return (idx, Err(BorgError::Cancelled)),
                            permit = slots_free.acquire_owned() => match permit {
                                Ok(permit) => permit,
                                // Never closed while the batch runs.
                                Err(_) => return (idx, Err(BorgError::Cancelled)),
                            },
                        };
                        let call = std::panic::AssertUnwindSafe(dispatch_one_tool(
                            req,
                            &tools,
                            &ctx_clone,
                            &composition_id,
                            &step_name,
                            turn_index,
                            Some(&batch_id),
                        ))
                        .catch_unwind();
                        let r = tokio::select! {
                            biased;
                            _ = cancel.cancelled() => Err(BorgError::Cancelled),
                            r = call => r.unwrap_or_else(|panic| {
                                let msg = panic
                                    .downcast_ref::<&str>()
                                    .map(|s| (*s).to_string())
                                    .or_else(|| panic.downcast_ref::<String>().cloned())
                                    .unwrap_or_else(|| "non-string panic payload".into());
                                Err(BorgError::Internal {
                                    message: format!("parallel tool call panicked: {msg}"),
                                })
                            }),
                        };
                        (idx, r)
                    }));
                }

                match ctx.parallel_failure_mode {
                    BatchFailureMode::CollectAll => {
                        // Every call reaches the model: a structural error
                        // becomes that call's `is_error` result. Only a
                        // control signal (cancel, suspend) unwinds, once
                        // the batch has drained.
                        let mut control: Option<BorgError> = None;
                        while let Some(joined) = js.join_next().await {
                            match joined {
                                Ok((idx, Ok(d))) => slots[idx] = Some(d),
                                Ok((_, Err(e)))
                                    if e.is_cancelled()
                                        || matches!(e, BorgError::Suspended { .. }) =>
                                {
                                    control.get_or_insert(e);
                                }
                                Ok((idx, Err(e))) => {
                                    let (name, call_id) =
                                        requested.remove(&idx).unwrap_or_default();
                                    let failed = collected_tool_failure(
                                        ctx,
                                        composition_id,
                                        step_name,
                                        turn_index,
                                        &batch_id,
                                        name,
                                        call_id,
                                        &e,
                                    )
                                    .await;
                                    slots[idx] = Some(failed);
                                }
                                // Tasks are never aborted under CollectAll
                                // and panics are caught in the task.
                                Err(join_err) => {
                                    control.get_or_insert(BorgError::Internal {
                                        message: format!(
                                            "parallel tool batch task failed: {join_err}"
                                        ),
                                    });
                                }
                            }
                        }
                        if let Some(e) = control {
                            return Err(e);
                        }
                    }
                    BatchFailureMode::FastFail => {
                        while let Some(joined) = js.join_next().await {
                            match joined {
                                Ok((idx, Ok(d))) => slots[idx] = Some(d),
                                Ok((_, Err(e))) => {
                                    batch_cancel.cancel();
                                    js.abort_all();
                                    return Err(e);
                                }
                                Err(join_err) if join_err.is_cancelled() => {}
                                Err(join_err) => {
                                    return Err(BorgError::Internal {
                                        message: format!(
                                            "parallel tool batch task failed: {join_err}"
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Reconstruct in original order. A None slot here is an internal bug:
    // every call either yielded a DispatchedCall or short-circuited via Err.
    let out: Vec<crate::tools::DispatchedCall> = slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            s.unwrap_or_else(|| {
                panic!("dispatch_tool_batch: slot {i} unfilled — internal invariant violated")
            })
        })
        .collect();
    Ok(out)
}

/// Turn one parallel call's structural `BorgError` into the `is_error`
/// result [`BatchFailureMode::CollectAll`] hands the model, closing the
/// call's `ToolCalled` with a failed `ToolReturned` and a
/// `tool:{name}:failed` guardrail signal, as a logical failure would.
#[allow(clippy::too_many_arguments)]
async fn collected_tool_failure(
    ctx: &ExecCtx,
    composition_id: &Arc<str>,
    step_name: &Arc<str>,
    turn_index: u32,
    batch_id: &Arc<str>,
    name: String,
    call_id: String,
    error: &BorgError,
) -> crate::tools::DispatchedCall {
    let result = crate::tools::ToolCallResult::fail(format!("{error}"));
    if let Some(harness) = &ctx.guardrails {
        harness.record(crate::guardrails::LoopSignal {
            key: format!("tool:{name}:failed").into(),
            payload_hash: None,
            at: std::time::Instant::now(),
        });
    }
    ctx.emitter
        .emit(Arc::new(BorgEvent::ToolReturned {
            composition_id: Arc::clone(composition_id),
            step_name: Arc::clone(step_name),
            turn_index,
            call_id: Arc::from(call_id.as_str()),
            tool_name: Arc::from(name.as_str()),
            success: false,
            error: result.error.clone(),
            batch_id: Some(Arc::clone(batch_id)),
        }))
        .await;
    crate::tools::DispatchedCall {
        call_id,
        name,
        result,
    }
}
