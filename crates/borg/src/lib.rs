//! `borg` — composable, supervised LLM agent runtime.
//!
//! Borg orchestrates agents: single-shot LLM
//! calls composed into pipelines (sequential, parallel, branching, looping)
//! with explicit supervision, cancellation propagation, fault tolerance,
//! and budget enforcement.
//!
//! ## Why borg, not just `tokio::spawn`
//!
//! A bare `tokio::spawn` of N sub-agent futures gives parallelism but
//! not per-step cancellation, panic isolation, restart on transient
//! failure, budget admission, error policies, or handoff between steps.
//! Borg adds those as thin wrappers over `JoinSet`,
//! `CancellationToken::child_token`, and `Semaphore` — supervised tasks
//! with Erlang-style isolation, not a full actor system.
//!
//! Agents are single-shot async functions, not mailbox-driven
//! processes: no message passing, no bidirectional linking, no
//! OneForAll/RestForOne. Restart is OneForOne via `RestartPolicy`; the
//! session-wide token budget is the meta-circuit-breaker.
//!
//! ## Layering
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────┐
//!   │   Composition: Single | Sequential | Parallel |       │
//!   │     LoopUntil | Branch | Fallback | FallbackLadder |   │  high-level
//!   │     Agent | Heartbeat | Deferred                       │
//!   └──────────────────┬────────────────────────────────────┘
//!                      │
//!   ┌──────────────────▼────────────────────────────────────┐
//!   │   Supervisor: JoinSet + per-child CancellationToken   │  supervision
//!   │              + Semaphore + RestartPolicy + panic catch │
//!   └──────────────────┬────────────────────────────────────┘
//!                      │
//!   ┌──────────────────▼────────────────────────────────────┐
//!   │   Agent: trait `BorgAgent` + `Verdict` return shape   │  contract
//!   └──────────────────┬────────────────────────────────────┘
//!                      │
//!   ┌──────────────────▼────────────────────────────────────┐
//!   │   AgentLlmClient trait + Budget + EventEmitter         │  ports
//!   └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Streaming vs one-shot — which composition tree?
//!
//! Borg ships two intentionally separate trees: [`Composition`] for
//! one-shot [`BorgAgent`]s (returns `Verdict`) and
//! [`StreamingComposition`] for streaming [`BorgStreamingAgent`]s
//! (returns `BoxStream<StreamEvent>`).
//!
//! Rule of thumb: the main user-facing agent is streaming; everything
//! else (planners, reviewers, fixers) is one-shot. Full decision record
//! in the [`streaming`] module docs.
//!
//! ## Domain isolation
//!
//! `borg` MUST NOT depend on any consumer-domain concept; concrete
//! agents live outside and depend *on* borg.
//!
//! ## Public API surface (re-exported below)
//!
//! - `Composition`, `ErrorPolicy`, `RestartPolicy` (+ the `*Builder` types
//!   the `Composition::*_builder` constructors return)
//! - `BorgAgent` trait + `BoxedAgent` type alias
//! - `Verdict`, `AgentContext`, `HandoffState`, `NamedVerdict`
//! - `AgentLlmClient` trait, `ChatRequest`, `ChatResponse`
//! - `Budget` trait, `BudgetVerdict`, `InMemoryBudget`
//! - `EventEmitter` trait, `BorgEvent`, `NoopEmitter`
//! - `BorgError`, `RetryClass`
//! - `LoopDetector` trait, `WindowedCounterDetector`, `LoopSignal`
//! - `GuardrailEngine`, `GuardrailRule`, `GuardrailVerdict`, `GuardrailCtx`
//! - `BoundaryMessage`, `BoundaryMessageQueue`, `EvaluationOutcome`
//! - `GuardrailHarness` — runtime bundle wired in via `ExecCtx::with_guardrails`
//! - `RetryableError` trait, `FailureClass`, `ProviderFailure` — canonical retry taxonomy
//! - `TerminalGuard<T>` — RAII safety net for terminal-state writes
//! - `FileEventRecorder` — append-only JSONL + atomic manifest event sink
//!
//! ## Worked example
//!
//! Building a 3-step pipeline: planner → branch on plan → review.
//! The example uses the public API only.
//!
//! ```ignore
//! use borg::*;
//! use std::sync::Arc;
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn run(
//! #     planner: BoxedAgent,
//! #     fixer: BoxedAgent,
//! #     skipper: BoxedAgent,
//! #     reviewer: BoxedAgent,
//! #     llm: Arc<dyn AgentLlmClient>,
//! # ) -> Result<ExecutionOutcome, BorgError> {
//! // 1. PLAN — outputs JSON plan into handoff[plan]
//! let plan = Composition::single_builder("planner", planner)
//!     .output_key("plan")
//!     .restart(RestartPolicy::on_failure(2))
//!     .build();
//!
//! // 2. BRANCH — route on the plan's first task kind
//! fn route(h: &HandoffState) -> BranchKey {
//!     h.get("plan")
//!         .and_then(|v| v.get("kind").and_then(|k| k.as_str()))
//!         .map(BranchKey::from)
//!         .unwrap_or_else(|| BranchKey::from("default"))
//! }
//! let dispatch = Composition::branch_builder(route)
//!     .arm("fix", Composition::single_builder("fixer", fixer).build())
//!     .arm("skip", Composition::single_builder("skipper", skipper).build())
//!     .build();
//!
//! // 3. REVIEW — single reviewer (could be a Parallel of multiple lenses)
//! let review = Composition::single_builder("reviewer", reviewer)
//!     .output_key("review_notes")
//!     .build();
//!
//! let pipeline = Composition::sequential_builder()
//!     .step(plan)
//!     .step(dispatch)
//!     .step(review)
//!     .build();
//!
//! let ctx = ExecCtx::new(
//!     llm,
//!     Arc::new(InMemoryBudget::new(40_000)),
//!     Arc::new(NoopEmitter),
//!     CancellationToken::new(),
//!     "session-123",
//! );
//!
//! execute(pipeline, ctx).await
//! # }
//! ```

// ── Modules ──────────────────────────────────────────────────────────
//
// Public modules are the deliberate namespaces users import from. Each
// one's items are also re-exported at the crate root where they belong
// to the core API; the module path is the long-form, grouped spelling.
//
// Implementation modules are private: everything a user needs from them
// is re-exported at the crate root below, so each item has exactly one
// public path.

pub mod budget;
pub mod checkpoint;
pub mod circuit_breaker;
pub mod dsl;
pub mod event_recorder;
pub mod events;
pub mod guardrails;
pub mod ladder;
pub mod output_sanitizer;
pub mod propagation;
pub mod retry_classify;
pub mod scheduled;
pub mod streaming;
pub mod suspend;
pub mod tools;

mod agent;
mod composition;
mod context;
mod error;
mod executor;
mod handoff;
mod llm;
mod meltdown;
mod streaming_executor;
mod supervisor;
mod terminal_guard;
mod verdict;

/// Test fakes — [`testing::FakeLlmClient`] and friends. Available to
/// downstream consumers' integration tests via the `testing` feature
/// flag, so they can write borg pipeline tests without rewriting fakes
/// from scratch.
///
/// ```toml
/// [dev-dependencies]
/// borg = { git = "https://github.com/joydle/borg.git", tag = "v1.0.0", features = ["testing"] }
/// ```
#[cfg(any(test, feature = "testing"))]
pub mod testing;

// Eval harness — scripted scenarios for regression-locking composition
// behavior. Gated on the `testing` feature since it consumes
// `FakeLlmClient` + `RecordingEmitter`; its public items are re-exported
// from [`testing`] (`borg::testing::{eval, EvalAssert, …}`).
#[cfg(any(test, feature = "testing"))]
mod evals;

#[cfg(test)]
mod tests;

// ── Public re-exports — the borg API ────────────────────────────────────
//
// Three stability tiers:
//
// - **Stable**: contract kept for the 1.x line; breaking changes bump
//   the major version and get a release note. Growable structs and enums
//   are `#[non_exhaustive]`, so adding a field or variant is not breaking.
// - **Unstable**: shipping but may shift in a minor; release note, no
//   version-bump promise.
// - **Internal (do not depend)**: `#[doc(hidden)]` items reachable only
//   because a macro expansion needs a path to them (see
//   `__pipeline_collect` in `dsl`). Not part of the API; may change in
//   any release.
//
// Supervision (`Supervisor`, `RunningAgent`) is private as of 3.0: use
// `execute` / `run_one`, which wrap the supervisor lifecycle.

// ── Stable — the API surface external users should depend on ──────────

pub use agent::{BorgAgent, BoxedAgent, Provider};
pub use budget::{
    in_memory_budget, sub_budget, Budget, BudgetVerdict, InMemoryBudget, SubBudget, TokenUsage,
};
pub use checkpoint::{
    composition_fingerprint, CheckpointPolicy, CheckpointStore, InMemoryCheckpointStore, Snapshot,
    SNAPSHOT_SCHEMA_VERSION,
};
pub use circuit_breaker::{BreakerPolicy, CircuitBreaker, WindowBreaker, WindowPolicy};
pub use composition::{
    Backoff, BranchBuilder, BranchKey, Composition, DeferredFn, ErrorPolicy, FallbackBuilder,
    HeartbeatBuilder, LoopBody, LoopCtx, LoopUntilBuilder, ParallelBuilder, RestartPolicy,
    SelectorFn, SequentialBuilder, SingleBuilder, UntilFn,
};
pub use context::AgentContext;
pub use error::{BorgError, RetryClass};
pub use event_recorder::{
    FileEventRecorder, DEFAULT_LINE_CAP_BYTES, DEFAULT_ROTATION_THRESHOLD_BYTES,
};
pub use events::{
    noop_emitter, BorgEvent, CompositionShape, EventEmitter, EventKind, EventKinds,
    FilteredEmitter, NoopEmitter,
};
pub use executor::{
    execute, execute_resumable, execute_resumable_with, execute_with_handoff, run_one,
    run_one_typed, run_one_typed_with_restart, run_one_with_restart, BatchFailureMode, ExecCtx,
    ExecutionOutcome,
};
pub use guardrails::{
    BoundaryMessage, BoundaryMessageKind, BoundaryMessageQueue, EvaluationOutcome, GuardrailCtx,
    GuardrailEngine, GuardrailHarness, GuardrailRule, GuardrailVerdict, LoopDetector, LoopSignal,
    WindowedCounterDetector,
};
pub use handoff::{HandoffPersistenceConfig, HandoffSlot, HandoffState, NamedVerdict};
pub use ladder::{FallbackLadder, Ladder, LadderError, LadderRung, Rung, RungFailure, Shipped};
pub use llm::{
    active_stream_sink, deterministic_llm_client, AgentLlmClient, ChatRequest, ChatResponse,
    LlmStreamSink, StreamSlot, LLM_STREAM,
};
pub use meltdown::{MeltdownLedger, MeltdownPolicy};
pub use output_sanitizer::{
    run_sanitizer, OutputSanitizerPipeline, OutputTransformStage, SanitizerCtx,
};
pub use propagation::{register_carrier, ContextCarrier, TaskLocal};
pub use retry_classify::{FailureClass, ProviderFailure, RetryableError, RETRYABLE_HTTP_STATUS};
pub use scheduled::{Scheduled, SubAgentHandle};
pub use terminal_guard::TerminalGuard;
pub use verdict::Verdict;

/// Re-exported so downstream consumers don't need a transitive
/// `tokio-util` dependency declaration just to construct cancel tokens.
pub use tokio_util::sync::CancellationToken;

// DSL surface — function aliases (`step`, `seq`, `par`, …), proc-macro
// derives, and the `pipeline!` macro. The underlying `borg-macros` crate
// is an invisible build-time detail.
pub use dsl::{
    agent_loop, borg_instrument, branch, deferred, fallback, fallback_ladder, heartbeat,
    loop_until, par, par_all, par_with, repeat, seq, step, step_if, step_with, BorgAgentDerive,
    BorgToolDerive, BorgToolSetDerive,
};

// ── Unstable — shipping but contract may shift in a minor ────────────
//
// Streaming primitives are usable but `StreamingComposition` will evolve
// as it unifies with `Composition`. Pin a minor version if you depend on
// these.

pub use streaming::{BorgStreamingAgent, StreamEvent};
pub use streaming_executor::{execute_streaming, StreamingComposition};

// Tool protocol. Trait + schema + result types are stable for
// hand-rolled registries; the `Tool` trait may grow methods, so pin a
// minor version.
pub use tools::{
    active_tool_call, ActiveToolCall, ActiveToolCallSlot, BoxedConcurrentTool, BoxedTool,
    ConcurrentDispatcher, ConcurrentRegistry, ConcurrentTool, DispatchOutcome, DispatchedCall,
    MutexDispatcher, Registry, Tool, ToolCallRequest, ToolCallResult, ToolDispatch, ToolSchema,
    ToolSet, ToolSetRegistry, TOOL_CALL,
};

/// The recommended starting import for borg consumers.
///
/// Brings in everything 90% of agents need:
/// - [`BorgAgent`] trait + [`BorgAgentDerive`] proc macro
/// - [`AgentContext`] (the `&ctx` passed to your `user_prompt`)
/// - [`Composition`] + the common shape constructors
/// - [`ExecCtx`], [`execute`], [`run_one`], [`run_one_with_restart`]
/// - [`HandoffState`] + [`HandoffSlot`]
/// - [`Verdict`] + [`BorgError`]
/// - Common ports: [`AgentLlmClient`], [`Budget`], [`EventEmitter`]
/// - [`CancellationToken`] (re-exported from tokio-util)
///
/// Items intentionally NOT in the prelude:
/// - [`InMemoryBudget`] / [`NoopEmitter`] concrete types — the
///   [`in_memory_budget`] / [`noop_emitter`] constructors handle the
///   common case; reach for the concrete types only if you need to
///   downcast.
/// - Streaming types ([`BorgStreamingAgent`], [`StreamingComposition`],
///   [`execute_streaming`]) — separate sub-system, opt-in.
///
/// ```ignore
/// use borg::prelude::*;
/// ```
pub mod prelude {
    pub use crate::agent::{BorgAgent, BoxedAgent, Provider};
    pub use crate::budget::{in_memory_budget, Budget};
    pub use crate::composition::{BranchKey, Composition, ErrorPolicy, LoopCtx, RestartPolicy};
    pub use crate::context::AgentContext;
    pub use crate::dsl::{
        agent_loop, branch, deferred, fallback, fallback_ladder, heartbeat, loop_until, par,
        par_all, par_with, repeat, seq, step, step_if, step_with, BorgAgentDerive, BorgToolDerive,
    };
    pub use crate::error::{BorgError, RetryClass};
    pub use crate::events::{noop_emitter, BorgEvent, EventEmitter};
    pub use crate::executor::{
        execute, execute_resumable, execute_resumable_with, execute_with_handoff, run_one,
        run_one_typed, run_one_typed_with_restart, run_one_with_restart, ExecCtx, ExecutionOutcome,
    };
    pub use crate::guardrails::{
        BoundaryMessage, BoundaryMessageKind, BoundaryMessageQueue, EvaluationOutcome,
        GuardrailCtx, GuardrailEngine, GuardrailHarness, GuardrailRule, GuardrailVerdict,
        LoopDetector, LoopSignal, WindowedCounterDetector,
    };
    pub use crate::handoff::{HandoffSlot, HandoffState, NamedVerdict};
    pub use crate::llm::{AgentLlmClient, ChatRequest, ChatResponse};
    pub use crate::output_sanitizer::{
        run_sanitizer, OutputSanitizerPipeline, OutputTransformStage, SanitizerCtx,
    };
    pub use crate::retry_classify::{FailureClass, ProviderFailure, RetryableError};
    pub use crate::terminal_guard::TerminalGuard;
    pub use crate::verdict::Verdict;
    pub use tokio_util::sync::CancellationToken;
}
