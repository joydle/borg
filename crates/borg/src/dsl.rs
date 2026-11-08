//! Concise composition surface — function aliases for `Composition::*`
//! plus the [`pipeline!`](crate::pipeline) macro for a statement-style
//! shape. Both are leaner surfaces over the same primitives — no new
//! runtime, just shorter call sites.
//!
//! # Function aliases
//!
//! ```ignore
//! use borg::prelude::*;
//!
//! let pipeline = seq([
//!     step(Planner),
//!     par([step(Coder), step(TestWriter), step(DocWriter)]),
//!     repeat(8, |v: &Verdict| v.token == "PASS", || seq([
//!         step(Builder),
//!         step(BuildFixer),
//!     ])),
//!     step(Reviewer),
//! ]);
//! ```
//!
//! Step names come from [`BorgAgent::agent_id`]. To override the name or
//! attach `output_key` / `restart`, use [`step_with`], which returns a
//! [`SingleBuilder`]:
//!
//! ```ignore
//! step_with(Planner).output_key("plan").restart(RestartPolicy::on_failure(2)).build()
//! ```
//!
//! # `pipeline!` macro
//!
//! The macro reads as a statement block:
//!
//! ```ignore
//! use borg::pipeline;
//!
//! let app = pipeline! {
//!     Planner;
//!     parallel { Coder, TestWriter, DocWriter }
//!     repeat(8, |v: &Verdict| v.token == "PASS") {
//!         Builder;
//!         BuildFixer;
//!     }
//!     Reviewer;
//! };
//! ```
//!
//! Grammar:
//! - `EXPR;` — a single step (expands to [`step`])
//! - `parallel { EXPR, EXPR, … }` — parallel composition
//! - `repeat(MAX, UNTIL) { … }` — fixed-body loop until predicate fires
//!   or `MAX` iterations elapse
//! - Top-level entries compose sequentially
//!
//! Limitations: the macro covers the common case. For `Branch`, custom
//! `ErrorPolicy`, per-iteration `LoopCtx`, or `output_key` chains, drop
//! to the function aliases or to `Composition::*` directly.

use crate::agent::BorgAgent;
use crate::composition::{
    BranchKey, Composition, ErrorPolicy, HeartbeatBuilder, LoopCtx, SingleBuilder, UntilFn,
};
use crate::handoff::HandoffState;

// ─────────────────────────────────────────────────────────────────────
// Proc-macro derives
// ─────────────────────────────────────────────────────────────────────
//
// The derives live in the standalone `borg-macros` crate (Rust requires
// proc-macro crates to be separate) and are re-exported here.

/// `#[derive(BorgAgentDerive)]` — generates the [`BorgAgent`] impl from a
/// `#[borg(id, model, max_tokens, system, …)]` attribute. You write only
/// `user_prompt`.
///
/// Re-exported as `*Derive` so the derive name doesn't shadow the
/// [`BorgAgent`] trait under `borg::*` glob imports; always use the
/// `*Derive` form via `borg::*`.
pub use borg_macros::BorgAgent as BorgAgentDerive;

/// `#[derive(BorgToolDerive)]` — generates the
/// [`Tool`](crate::tools::Tool) impl from a
/// `#[tool(name, description, context, input_schema)]` attribute. You
/// write only the inherent async `execute`. See [`BorgAgentDerive`] for
/// the `*Derive` naming.
pub use borg_macros::BorgTool as BorgToolDerive;

/// `#[borg_instrument(name = "...")]` — wraps an agent's `invoke` in a
/// named `tracing` span under the current `borg.step`. The expansion calls
/// `::tracing`, so the crate using it depends on `tracing`.
pub use borg_macros::borg_instrument;

/// `#[derive(BorgToolSetDerive)]` — generates the
/// [`ToolSet`](crate::tools::ToolSet) impl from an enum of tuple
/// variants whose inner types provide inherent `name()`, `schema()`,
/// `execute(input, &mut ctx)`; the macro forwards to them.
///
/// `#[borg(context = MyCtx<'ctx>)]` declares the GAT instantiation; use
/// `#[borg(context = ())]` for context-free tools. See
/// [`BorgAgentDerive`] for the `*Derive` naming.
pub use borg_macros::BorgToolSet as BorgToolSetDerive;

/// Wrap an agent as a single step, named from [`BorgAgent::agent_id`].
/// Returns a [`Composition`] so items compose uniformly inside
/// `seq([...])`. For chained configuration, use [`step_with`].
pub fn step<A>(agent: A) -> Composition
where
    A: BorgAgent + 'static,
{
    let name = agent.agent_id().to_string();
    Composition::single(name, agent)
}

/// Like [`step`] but returns a [`SingleBuilder`] for chaining
/// `.output_key(...)` / `.restart(...)` / `.build()`.
pub fn step_with<A>(agent: A) -> SingleBuilder
where
    A: BorgAgent + 'static,
{
    let name = agent.agent_id().to_string();
    Composition::single_with(name, agent)
}

/// Sequential composition with `FailFast`. Accepts any
/// `Into<Composition>` — bare values, [`step`], or the [`SingleBuilder`]
/// from [`step_with`], so leaves don't need `.build()`.
pub fn seq<I, C>(steps: I) -> Composition
where
    I: IntoIterator<Item = C>,
    C: Into<Composition>,
{
    Composition::sequential(steps.into_iter().map(Into::into))
}

/// Parallel composition. Branch names derive from inner `Single` step
/// names; non-`Single` branches get synthetic `branch-N` names. Panics
/// on duplicate names (see [`Composition::parallel`]) — give fanned-out
/// instances distinct `agent_id`s.
pub fn par<I, C>(steps: I) -> Composition
where
    I: IntoIterator<Item = C>,
    C: Into<Composition>,
{
    Composition::parallel(name_branches(steps))
}

/// Parallel with an explicit error policy. Concurrency cap is set to the
/// number of branches.
pub fn par_with<I, C>(steps: I, on_error: ErrorPolicy) -> Composition
where
    I: IntoIterator<Item = C>,
    C: Into<Composition>,
{
    let pairs: Vec<(String, Composition)> = name_branches(steps).collect();
    let n = pairs.len();
    Composition::parallel_with(pairs, on_error, n)
}

/// Parallel composition with [`ErrorPolicy::CollectAll`] — every branch
/// runs to completion, failures surfacing as
/// `NamedVerdict { is_error: true, … }` rather than cancelling siblings.
///
/// Use when partial results are still useful (a fan-out where N-of-M
/// succeeding beats rolling back the batch). For fail-fast, use [`par`].
///
/// ```ignore
/// use borg::dsl::*;
///
/// par_all([
///     step(Summarize::new("chapter-1")),
///     step(Summarize::new("chapter-2")),
///     step(Summarize::new("chapter-3")),
/// ])
/// ```
pub fn par_all<I, C>(steps: I) -> Composition
where
    I: IntoIterator<Item = C>,
    C: Into<Composition>,
{
    par_with(steps, ErrorPolicy::CollectAll)
}

/// Repeat a fixed-shape body up to `times` iterations or until `until`
/// fires on the most-recent verdict. The body is a closure returning a
/// fresh [`Composition`] each iteration (since `Composition` isn't
/// `Clone`).
///
/// ```ignore
/// repeat(5, |v: &Verdict| v.token == "DONE", || seq([
///     step(Builder),
///     step(BuildFixer),
/// ]))
/// ```
///
/// For per-iteration adaptation, use [`loop_until`].
pub fn repeat<F>(times: u32, until: UntilFn, body: F) -> Composition
where
    F: Fn() -> Composition + Send + Sync + 'static,
{
    Composition::loop_fixed(body, until, times)
}

/// LoopUntil with per-iteration adaptation: the factory receives a
/// [`LoopCtx`] each iteration. Use when the body shape varies between
/// turns; for a fixed body, [`repeat`] is shorter.
pub fn loop_until<F>(times: u32, until: UntilFn, body: F) -> Composition
where
    F: Fn(LoopCtx<'_>) -> Composition + Send + Sync + 'static,
{
    Composition::loop_until(body, until, times)
}

/// Tool-loop agent — the [`Composition::Agent`] variant. The agent emits
/// tool calls, the dispatcher runs them through `tools`, results thread
/// back via handoff, and the loop ends when the agent stops or
/// `max_steps` is reached. Step name from `agent.agent_id()`.
///
/// ```ignore
/// use borg::dsl::*;
/// use std::sync::Arc;
///
/// let dispatcher = Arc::new(MutexDispatcher::new(registry, my_ctx));
/// let pipeline = agent_loop(MainAgent, dispatcher, 40);
/// ```
pub fn agent_loop<A>(
    agent: A,
    tools: ::std::sync::Arc<dyn crate::tools::ToolDispatch>,
    max_steps: u32,
) -> Composition
where
    A: BorgAgent + 'static,
{
    let name = agent.agent_id().to_string();
    Composition::agent(name, agent, tools, max_steps)
}

/// One-liner constructor for a `ConcurrentDispatcher` as
/// `Arc<dyn ToolDispatch>`. Pair with
/// `ExecCtx::with_parallel_tools(...)` (or `agent_loop_parallel`) to
/// actually exercise parallel dispatch.
pub fn concurrent_dispatcher<C: Send + Sync + 'static>(
    registry: crate::tools::ConcurrentRegistry<C>,
    ctx: C,
) -> ::std::sync::Arc<dyn crate::tools::ToolDispatch> {
    ::std::sync::Arc::new(crate::tools::ConcurrentDispatcher::new(registry, ctx))
}

/// Build a tool-loop composition wired for parallel dispatch in one call
/// (`concurrent_dispatcher` + `agent_loop`). The caller still sets
/// `ExecCtx::with_parallel_tools(mode)` — the gate lives on the
/// executor, not the composition.
pub fn agent_loop_parallel<A, C>(
    agent: A,
    registry: crate::tools::ConcurrentRegistry<C>,
    ctx: C,
    max_steps: u32,
) -> Composition
where
    A: BorgAgent + 'static,
    C: Send + Sync + 'static,
{
    let dispatcher = concurrent_dispatcher(registry, ctx);
    agent_loop(agent, dispatcher, max_steps)
}

/// Branch on a selector. Arms are `(key, composition)` pairs. See
/// [`Composition::branch`] for the selector contract.
pub fn branch<K, I, F>(selector: F, arms: I) -> Composition
where
    F: Fn(&HandoffState) -> BranchKey + Send + Sync + 'static,
    K: Into<BranchKey>,
    I: IntoIterator<Item = (K, Composition)>,
{
    Composition::branch(selector, arms)
}

/// Run `body` if `pred(handoff)` is true, else no-op (an empty
/// `seq([])`). Sugar over [`Composition::branch_default`] with a boolean
/// selector: the body runs under the synthetic key `"then"`, the default
/// arm handles the false case.
///
/// The predicate is any `Fn(&HandoffState) -> bool + Send + Sync +
/// 'static`, so capturing closures are first-class:
///
/// ```ignore
/// use borg::dsl::*;
///
/// step_if(
///     |h| h.get("build_status")
///         .and_then(|v| v.as_str())
///         .map(|s| s == "failed")
///         .unwrap_or(false),
///     step(BuildFix),
/// )
/// ```
///
/// For three-or-more-way dispatch or a real "else" composition, use
/// [`branch`] directly.
pub fn step_if<F, C>(pred: F, body: C) -> Composition
where
    F: Fn(&HandoffState) -> bool + Send + Sync + 'static,
    C: Into<Composition>,
{
    let selector = move |h: &HandoffState| {
        if pred(h) {
            BranchKey::from("then")
        } else {
            BranchKey::from("else")
        }
    };
    Composition::branch_default(
        selector,
        std::iter::once((BranchKey::from("then"), body.into())),
        seq(std::iter::empty::<Composition>()),
    )
}

/// Run `inner` while beating a [`crate::BorgEvent::Heartbeat`] named
/// `name` every `every`, for as long as it runs: liveness for a long step
/// that says nothing of its own (a model call reasoning for minutes, a
/// long batch job). `.at_start()` beats once before it starts. The
/// heartbeat is transparent — `inner` runs in the same context, so its own
/// events, step names and handoff are what they would be unwrapped — and
/// it stops with `inner`, cancel included.
///
/// ```ignore
/// use borg::dsl::*;
/// use std::time::Duration;
///
/// seq([
///     heartbeat("drafting", Duration::from_millis(2500), step(Drafter)).at_start(),
///     heartbeat("reviewing", Duration::from_millis(2500), step(Reviewer)),
/// ])
/// ```
///
/// Map the beats onto your wire in your emitter; with
/// [`crate::ExecCtx::with_lifecycle_events`]`(EventKinds::only(&[EventKind::Heartbeat]))`
/// they are the only lifecycle events it sees.
pub fn heartbeat<C>(
    name: impl Into<String>,
    every: std::time::Duration,
    inner: C,
) -> HeartbeatBuilder
where
    C: Into<Composition>,
{
    HeartbeatBuilder::new(name, every, inner.into())
}

/// A composition built when it is reached, from the handoff the steps
/// before it left: a shape that depends on what an earlier step decided.
///
/// ```ignore
/// use borg::dsl::*;
///
/// seq([
///     step_with(Planner).output_key("items"),
///     // One branch per item the planner listed.
///     deferred(|h| {
///         let items: Vec<String> = h.output_as("items").unwrap_or_default();
///         par_all(items.into_iter().map(|i| Composition::single(i.clone(), Worker(i))))
///     }),
/// ])
/// ```
///
/// Called once with an empty handoff when a resumable execution
/// fingerprints its shape, so the builder must not panic on one.
pub fn deferred<F>(build: F) -> Composition
where
    F: Fn(&HandoffState) -> Composition + Send + Sync + 'static,
{
    Composition::Deferred {
        build: Box::new(build),
    }
}

/// Try `primary`; on retryable failure, fall through each fallback in order.
pub fn fallback<I>(primary: Composition, fallbacks: I) -> Composition
where
    I: IntoIterator<Item = Composition>,
{
    Composition::fallback(primary, fallbacks)
}

/// A fallback ladder over composition rungs — [`fallback`] with the
/// rules a real provider ladder needs: a caller's fall-through predicate
/// (default [`BorgError::is_retryable`](crate::BorgError::is_retryable)),
/// per-rung `when` filters, a terminal error handed back as-is, and every
/// rung's error kept on
/// [`BorgError::FallbackExhausted`](crate::BorgError::FallbackExhausted).
/// Rungs are anything `Into<LadderRung>` (a [`Composition`], a
/// [`step_with`] builder, or a configured
/// [`LadderRung`](crate::ladder::LadderRung)). See [`crate::ladder`].
///
/// ```ignore
/// fallback_ladder([
///     LadderRung::new(step(PrimaryModel)),
///     LadderRung::new(step(MirrorModel)).when(|h, _| h.get("mirror").is_some()),
///     LadderRung::new(step(SmallerModel)),
/// ])
/// .falls_through(|e| e.is_retryable())
/// .on_fall(|failure, to, handoff| { /* record the rung's outcome */ })
/// ```
pub fn fallback_ladder<I, R>(rungs: I) -> crate::ladder::FallbackLadder
where
    I: IntoIterator<Item = R>,
    R: Into<crate::ladder::LadderRung>,
{
    crate::ladder::FallbackLadder::new(rungs)
}

fn name_branches<I, C>(steps: I) -> impl Iterator<Item = (String, Composition)>
where
    I: IntoIterator<Item = C>,
    C: Into<Composition>,
{
    steps.into_iter().map(Into::into).enumerate().map(|(i, c)| {
        let name = match &c {
            Composition::Single { name, .. } => name.clone(),
            _ => format!("branch-{i}"),
        };
        (name, c)
    })
}

// ─────────────────────────────────────────────────────────────────────
// pipeline! — statement-style composition syntax
// ─────────────────────────────────────────────────────────────────────

/// Statement-style composition syntax. See the [module-level docs](self)
/// for grammar + examples.
#[macro_export]
macro_rules! pipeline {
    ( $($body:tt)* ) => {{
        $crate::__pipeline_collect!( [] $($body)* )
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __pipeline_collect {
    // parallel { e1, e2, ... }   (no trailing semicolon required)
    ( [$($acc:expr,)*] parallel { $($pa:expr),+ $(,)? } $($rest:tt)* ) => {
        $crate::__pipeline_collect!(
            [$($acc,)* $crate::dsl::par(::std::vec![
                $( $crate::dsl::step($pa) ),+
            ]),]
            $($rest)*
        )
    };

    // repeat(MAX, UNTIL_CLOSURE) { body }   (no trailing semicolon required)
    ( [$($acc:expr,)*] repeat( $max:expr, $until:expr ) { $($body:tt)* } $($rest:tt)* ) => {
        $crate::__pipeline_collect!(
            [$($acc,)* $crate::dsl::repeat(
                $max,
                $until,
                || $crate::pipeline!($($body)*),
            ),]
            $($rest)*
        )
    };

    // EXPR; — single step
    ( [$($acc:expr,)*] $agent:expr; $($rest:tt)* ) => {
        $crate::__pipeline_collect!(
            [$($acc,)* $crate::dsl::step($agent),]
            $($rest)*
        )
    };

    // Terminator
    ( [$($acc:expr,)*] ) => {{
        let __steps: ::std::vec::Vec<$crate::Composition> =
            ::std::vec![$($acc,)*];
        $crate::dsl::seq(__steps)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::ScriptedAgent;

    fn a() -> ScriptedAgent {
        ScriptedAgent::new("a")
    }
    fn b() -> ScriptedAgent {
        ScriptedAgent::new("b")
    }
    fn c() -> ScriptedAgent {
        ScriptedAgent::new("c")
    }

    #[test]
    fn step_uses_agent_id_as_name() {
        let comp = step(a());
        match comp {
            Composition::Single { name, .. } => assert_eq!(name, "a"),
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn seq_collects_into_sequential() {
        let comp = seq([step(a()), step(b())]);
        match comp {
            Composition::Sequential { steps, .. } => {
                assert_eq!(steps.len(), 2);
            }
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    fn par_derives_branch_names_from_step_names() {
        let comp = par([step(a()), step(b()), step(c())]);
        match comp {
            Composition::Parallel { branches, .. } => {
                let names: Vec<&str> = branches.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(names, vec!["a", "b", "c"]);
            }
            _ => panic!("expected Parallel"),
        }
    }

    #[test]
    fn repeat_wraps_into_loop_until() {
        let comp = repeat(5, |v| v.token == "PASS", || seq([step(a())]));
        match comp {
            Composition::LoopUntil { max_iterations, .. } => {
                assert_eq!(max_iterations, 5);
            }
            _ => panic!("expected LoopUntil"),
        }
    }

    #[test]
    fn pipeline_macro_top_level_is_sequential() {
        let comp = pipeline! {
            a();
            b();
            c();
        };
        match comp {
            Composition::Sequential { steps, .. } => assert_eq!(steps.len(), 3),
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    fn pipeline_macro_parallel_block() {
        let comp = pipeline! {
            a();
            parallel { b(), c() }
        };
        match comp {
            Composition::Sequential { steps, .. } => {
                assert_eq!(steps.len(), 2);
                assert!(matches!(steps[1], Composition::Parallel { .. }));
            }
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    fn pipeline_macro_repeat_block() {
        let comp = pipeline! {
            a();
            repeat(3, |v| v.token == "DONE") {
                b();
                c();
            }
        };
        match comp {
            Composition::Sequential { steps, .. } => {
                assert_eq!(steps.len(), 2);
                assert!(matches!(
                    steps[1],
                    Composition::LoopUntil {
                        max_iterations: 3,
                        ..
                    }
                ));
            }
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    fn par_all_sets_collect_all_error_policy() {
        let comp = par_all([step(a()), step(b()), step(c())]);
        match comp {
            Composition::Parallel {
                branches, on_error, ..
            } => {
                assert_eq!(branches.len(), 3);
                assert_eq!(on_error, ErrorPolicy::CollectAll);
            }
            _ => panic!("expected Parallel"),
        }
    }

    #[test]
    fn par_keeps_fail_fast_default() {
        // Regression: par() must stay FailFast (par_all is the explicit
        // CollectAll variant).
        let comp = par([step(a()), step(b())]);
        match comp {
            Composition::Parallel { on_error, .. } => {
                assert_eq!(on_error, ErrorPolicy::FailFast);
            }
            _ => panic!("expected Parallel"),
        }
    }

    #[test]
    fn step_if_wraps_into_branch_with_default() {
        let comp = step_if(|_| true, step(a()));
        match comp {
            Composition::Branch {
                branches, default, ..
            } => {
                assert_eq!(branches.len(), 1);
                assert_eq!(branches[0].0.as_str(), "then");
                assert!(
                    default.is_some(),
                    "step_if must always install a default no-op arm"
                );
                // Default is an empty seq([]) — Sequential with 0 steps.
                match default.unwrap().as_ref() {
                    Composition::Sequential { steps, .. } => {
                        assert!(steps.is_empty(), "default arm must be a no-op");
                    }
                    _ => panic!("expected Sequential default arm"),
                }
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn step_if_predicate_routes_to_then_or_else() {
        use crate::handoff::HandoffState;

        let comp = step_if(|h: &HandoffState| h.get("flag").is_some(), step(a()));
        let selector = match &comp {
            Composition::Branch { selector, .. } => selector.clone(),
            _ => panic!("expected Branch"),
        };

        let mut h_unset = HandoffState::new();
        assert_eq!(selector(&h_unset).as_str(), "else");

        h_unset.set("flag", serde_json::json!(true));
        assert_eq!(selector(&h_unset).as_str(), "then");
    }

    #[test]
    fn step_if_predicate_can_capture() {
        // The closure captures runtime config (an expected slot value).
        use crate::handoff::HandoffState;

        let expected_status = String::from("failed");
        let comp = step_if(
            move |h: &HandoffState| {
                h.get("build_status")
                    .and_then(|v| v.as_str())
                    .map(|s| s == expected_status.as_str())
                    .unwrap_or(false)
            },
            step(a()),
        );
        let selector = match &comp {
            Composition::Branch { selector, .. } => selector.clone(),
            _ => panic!("expected Branch"),
        };

        let mut h = HandoffState::new();
        assert_eq!(selector(&h).as_str(), "else");

        h.set("build_status", serde_json::json!("failed"));
        assert_eq!(selector(&h).as_str(), "then");

        h.set("build_status", serde_json::json!("passed"));
        assert_eq!(selector(&h).as_str(), "else");
    }

    #[test]
    fn branch_accepts_bare_fn_pointer() {
        // Regression: SelectorFn is `Arc<dyn Fn>`, but a bare fn pointer
        // must still pass through the constructor without explicit wrapping.
        use crate::handoff::HandoffState;

        fn route(_h: &HandoffState) -> BranchKey {
            BranchKey::from("only")
        }

        let comp = branch(route, [(BranchKey::from("only"), step(a()))]);
        assert!(matches!(comp, Composition::Branch { .. }));
    }
}
