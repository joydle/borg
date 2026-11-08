//! Error types for the borg runtime.
//!
//! `BorgError` is the single error returned by composition execution.
//! Adapters at the consumer boundary convert provider-specific errors
//! (e.g. `ClaudeError`) into `BorgError::Llm` — borg itself never names
//! a specific provider.

use std::fmt;

/// Classification borg uses to decide retry behavior on `BorgError::Llm`.
///
/// Borg only branches on this enum — adapters compute it from whatever
/// signal their provider exposes (HTTP status, typed enum, parsed body).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryClass {
    /// Transient — retry with backoff. Maps to 408/429/500/502/503/504,
    /// connection resets, timeouts.
    Retryable,
    /// Permanent — do not retry. Auth, schema, content-policy, 4xx that
    /// won't resolve by waiting.
    Terminal,
}

/// Error returned by every borg execution path.
///
/// Provider-specific errors fold into `Llm`. `#[non_exhaustive]` so a
/// minor release can add variants — match with a `_` arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum BorgError {
    /// LLM call failed. Carries a stringified message and a retry class.
    /// The composition layer reads `class` to decide between `Restart`
    /// and `Stop` under `RestartPolicy`.
    ///
    /// `retry_after` is the provider's own "try again in" (a `Retry-After`
    /// header, or the same hint in an error body). A restart never comes
    /// sooner than it: the backoff before the next attempt is
    /// `max(backoff, retry_after)`. `None` leaves the backoff alone.
    /// Build with [`BorgError::llm`] + [`BorgError::with_retry_after`] to
    /// stay source-compatible when this variant grows.
    Llm {
        message: String,
        class: RetryClass,
        retry_after: Option<std::time::Duration>,
    },
    /// A spawned child agent panicked. The supervisor catches the panic
    /// at the `JoinSet` boundary and surfaces it here so siblings under
    /// `ErrorPolicy::CollectAll` aren't taken down with it.
    ChildPanicked { agent_id: String, message: String },
    /// Parent cancellation token fired before the child completed. Not
    /// itself a failure — supervisors map this to a clean shutdown
    /// outcome. Surfaces here only when a caller explicitly awaits a
    /// cancelled execution.
    Cancelled,
    /// Budget admission denied or exhausted mid-composition.
    BudgetExhausted { needed: u64, remaining: u64 },
    /// `Composition::Branch` selector returned a key with no matching
    /// branch and no `default` fallback was provided.
    UnmatchedBranch { key: String },
    /// `RestartPolicy::Restart { max }` exceeded — the child kept
    /// failing past the retry budget.
    RetryExhausted {
        agent_id: String,
        attempts: u8,
        last_error: Box<BorgError>,
    },
    /// `ErrorPolicy::AtLeast(n)` not met — fewer than `n` branches
    /// produced a successful verdict. The composition itself surfaces
    /// as failed even though some branches succeeded.
    InsufficientSuccesses { needed: usize, got: usize },
    /// Restart storm detector tripped: the session's
    /// [`crate::MeltdownLedger`] recorded `restarts` failures of
    /// `agent_id` within the policy window. The supervisor surfaces this
    /// instead of letting per-call `RestartPolicy` keep grinding budget
    /// on a broken dependency. OTP analog: `max_restarts/max_seconds`.
    Meltdown {
        agent_id: String,
        restarts: u32,
        window_secs: u32,
    },
    /// Per-(provider, agent) circuit breaker is open; this call was
    /// fail-fast'd without hitting the LLM. Surfaces during a provider
    /// outage so callers see a fast typed failure instead of every
    /// sub-agent burning its retry budget independently.
    ProviderUnavailable {
        provider: &'static str,
        agent_id: String,
        reopens_in: std::time::Duration,
    },
    /// A registered [`crate::GuardrailRule`] returned
    /// [`crate::GuardrailVerdict::Halt`] at a turn boundary; the
    /// supervised task is force-terminated. `rule` is the rule's
    /// `name()`, `reason` its halt message — both stable strings the
    /// consumer can route to its own terminal taxonomy.
    Guardrail { rule: String, reason: String },
    /// Every applicable rung of a [`crate::Composition::FallbackLadder`]
    /// fell through (or none applied: `failures` empty). Each rung's error
    /// is kept, in the order the rungs ran.
    FallbackExhausted {
        failures: Vec<crate::ladder::RungFailure<BorgError>>,
    },
    /// A `Composition::Agent` tool loop used all `max_steps` turns and
    /// the last one still asked for tools, so the agent never produced
    /// a final answer. The tool calls of that last turn were run and are
    /// in the handoff; the loop stops anyway. Not retryable: running the
    /// same loop again would hit the same ceiling.
    MaxStepsReached { agent_id: String, max_steps: u32 },
    /// Catch-all for failures that don't fit the above. Use sparingly.
    Internal { message: String },
    /// Control unwind from [`crate::AgentContext::request_suspend`] — not
    /// a failure, the agent is pausing for a human decision and the
    /// signal carries the request. Consumers MUST match this separately
    /// from real errors; `is_retryable()` is false so `RestartPolicy`
    /// won't loop on it.
    Suspended {
        signal: crate::suspend::SuspendSignal,
    },
}

impl BorgError {
    /// An LLM failure with no retry hint.
    pub fn llm(message: impl Into<String>, class: RetryClass) -> Self {
        BorgError::Llm {
            message: message.into(),
            class,
            retry_after: None,
        }
    }

    /// Attach the provider's retry-after hint. A no-op on every variant
    /// but [`BorgError::Llm`].
    #[must_use]
    pub fn with_retry_after(mut self, after: std::time::Duration) -> Self {
        if let BorgError::Llm { retry_after, .. } = &mut self {
            *retry_after = Some(after);
        }
        self
    }

    /// The provider's retry-after hint, when this is an LLM failure that
    /// carried one.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            BorgError::Llm { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Short kebab-case tag for lifecycle events
    /// (`BorgEvent::StepCompleted.failure_cause`). Stable wire
    /// vocabulary — a new variant must add a new tag, never widen an
    /// existing one.
    pub fn failure_cause(&self) -> &'static str {
        match self {
            BorgError::Llm { .. } => "llm-error",
            BorgError::ChildPanicked { .. } => "panicked",
            BorgError::Cancelled => "cancelled",
            BorgError::BudgetExhausted { .. } => "budget-exhausted",
            BorgError::UnmatchedBranch { .. } => "unmatched-branch",
            BorgError::RetryExhausted { .. } => "retry-exhausted",
            BorgError::InsufficientSuccesses { .. } => "insufficient-successes",
            BorgError::Meltdown { .. } => "meltdown",
            BorgError::ProviderUnavailable { .. } => "provider-unavailable",
            BorgError::Guardrail { .. } => "guardrail-halt",
            BorgError::FallbackExhausted { .. } => "fallback-exhausted",
            BorgError::MaxStepsReached { .. } => "max-steps-reached",
            BorgError::Internal { .. } => "internal",
            BorgError::Suspended { .. } => "suspended",
        }
    }
}

impl fmt::Display for BorgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BorgError::Llm {
                message,
                class,
                retry_after,
            } => {
                write!(f, "llm error ({class:?}): {message}")?;
                if let Some(after) = retry_after {
                    write!(f, " (retry after {after:?})")?;
                }
                Ok(())
            }
            BorgError::ChildPanicked { agent_id, message } => {
                write!(f, "agent `{agent_id}` panicked: {message}")
            }
            BorgError::Cancelled => write!(f, "cancelled"),
            BorgError::BudgetExhausted { needed, remaining } => {
                write!(
                    f,
                    "budget exhausted: needed {needed} tokens, {remaining} remaining"
                )
            }
            BorgError::UnmatchedBranch { key } => {
                write!(f, "no branch matched key `{key}` and no default provided")
            }
            BorgError::RetryExhausted {
                agent_id,
                attempts,
                last_error,
            } => {
                write!(
                    f,
                    "agent `{agent_id}` exhausted {attempts} retries; last: {last_error}"
                )
            }
            BorgError::InsufficientSuccesses { needed, got } => {
                write!(
                    f,
                    "AtLeast({needed}) not met: only {got} branch(es) succeeded"
                )
            }
            BorgError::Meltdown {
                agent_id,
                restarts,
                window_secs,
            } => {
                write!(
                    f,
                    "meltdown: agent `{agent_id}` restarted {restarts} times in {window_secs}s — \
                     supervisor halted further retries"
                )
            }
            BorgError::ProviderUnavailable {
                provider,
                agent_id,
                reopens_in,
            } => {
                write!(
                    f,
                    "provider `{provider}` circuit-broken for agent `{agent_id}`: \
                     reopens in {reopens_in:?}"
                )
            }
            BorgError::Guardrail { rule, reason } => {
                write!(f, "guardrail `{rule}` halted: {reason}")
            }
            BorgError::FallbackExhausted { failures } if failures.is_empty() => {
                write!(f, "no fallback rung applied")
            }
            BorgError::FallbackExhausted { failures } => {
                write!(f, "every fallback rung failed")?;
                crate::ladder::write_failures(f, failures)
            }
            BorgError::MaxStepsReached {
                agent_id,
                max_steps,
            } => {
                write!(
                    f,
                    "agent `{agent_id}` reached its {max_steps}-step limit while still calling tools"
                )
            }
            BorgError::Internal { message } => write!(f, "internal: {message}"),
            BorgError::Suspended { signal } => {
                write!(f, "suspended: {}", signal.request.prompt)
            }
        }
    }
}

impl std::error::Error for BorgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BorgError::RetryExhausted { last_error, .. } => Some(last_error.as_ref()),
            BorgError::FallbackExhausted { failures } => failures
                .last()
                .map(|f| &f.error as &(dyn std::error::Error + 'static)),
            _ => None,
        }
    }
}

impl BorgError {
    /// Whether borg's restart policy should attempt another invocation.
    /// `Llm { class: Retryable }` and `ChildPanicked` both return true;
    /// everything else is terminal.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            BorgError::Llm {
                class: RetryClass::Retryable,
                ..
            } | BorgError::ChildPanicked { .. }
        )
    }

    /// Cancellation is a clean shutdown signal, not a user-facing error;
    /// the executor branches on this at its boundary.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, BorgError::Cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_llm_is_retryable() {
        let e = BorgError::llm("503", RetryClass::Retryable);
        assert!(e.is_retryable());
    }

    #[test]
    fn terminal_llm_is_not_retryable() {
        let e = BorgError::llm("401", RetryClass::Terminal);
        assert!(!e.is_retryable());
    }

    #[test]
    fn child_panic_is_retryable_under_restart_policy() {
        let e = BorgError::ChildPanicked {
            agent_id: "a".into(),
            message: "boom".into(),
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn max_steps_reached_is_terminal() {
        let e = BorgError::MaxStepsReached {
            agent_id: "a".into(),
            max_steps: 3,
        };
        assert!(!e.is_retryable());
        assert_eq!(e.failure_cause(), "max-steps-reached");
        assert!(e.to_string().contains("3-step limit"), "{e}");
    }

    #[test]
    fn cancelled_is_neither_retryable_nor_a_failure() {
        let e = BorgError::Cancelled;
        assert!(!e.is_retryable());
        assert!(e.is_cancelled());
    }

    #[test]
    fn budget_exhausted_is_terminal() {
        let e = BorgError::BudgetExhausted {
            needed: 100,
            remaining: 0,
        };
        assert!(!e.is_retryable());
    }

    #[test]
    fn guardrail_is_terminal_and_carries_rule_reason() {
        let e = BorgError::Guardrail {
            rule: "build_cascade".into(),
            reason: "failed 8× consecutively".into(),
        };
        assert!(!e.is_retryable());
        assert_eq!(e.failure_cause(), "guardrail-halt");
        let s = format!("{e}");
        assert!(s.contains("build_cascade"));
        assert!(s.contains("8×"));
    }

    #[test]
    fn retry_after_rides_an_llm_error_and_nothing_else() {
        let e = BorgError::llm("429", RetryClass::Retryable)
            .with_retry_after(std::time::Duration::from_secs(120));
        assert_eq!(e.retry_after(), Some(std::time::Duration::from_secs(120)));
        assert!(format!("{e}").contains("retry after 120s"));
        assert_eq!(
            BorgError::llm("x", RetryClass::Retryable).retry_after(),
            None
        );
        let other = BorgError::Cancelled.with_retry_after(std::time::Duration::from_secs(1));
        assert_eq!(other.retry_after(), None);
    }

    #[test]
    fn display_includes_context() {
        let e = BorgError::RetryExhausted {
            agent_id: "review".into(),
            attempts: 3,
            last_error: Box::new(BorgError::llm("timeout", RetryClass::Retryable)),
        };
        let s = format!("{e}");
        assert!(s.contains("review"));
        assert!(s.contains("3"));
        assert!(s.contains("timeout"));
    }
}
