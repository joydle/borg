//! Canonical retry taxonomy for LLM provider failures.
//!
//! [`RetryClass`] (in `error.rs`) is the coarse 2-way classification
//! the executor consults to decide whether to restart under
//! [`RestartPolicy`](crate::RestartPolicy). [`FailureClass`] is the
//! finer-grained taxonomy here: ten typed variants that product
//! layers route to dashboards, UI hints, or corrective prompts without
//! re-deriving the rule set.
//!
//! Adapters holding a typed provider error classify it through
//! `classify_status` / `classify_provider_code` / `classify_message`
//! and get back a [`ProviderFailure`] carrying the typed `class`, the
//! HTTP status, and any `Retry-After` hint.
//! [`BorgError::Llm`](crate::BorgError::Llm) carries the coarse
//! `class: RetryClass` — convert via [`FailureClass::to_retry_class`].
//!
//! Out of scope: adapter-specific classifiers (they depend on the
//! provider SDK and live in the adapter crate); backoff math (see
//! [`Backoff`](crate::Backoff)); and wire format.

use crate::error::RetryClass;
use std::time::Duration;

/// HTTP status codes worth retrying — borg's single source of truth
/// for transient-status classification. A deliberately conservative
/// set: codes like `409 Conflict` need caller judgment and aren't
/// blanket-retried.
pub const RETRYABLE_HTTP_STATUS: &[u16] = &[408, 429, 500, 502, 503, 504];

/// Transient error message substrings, used by
/// [`ProviderFailure::classify_message`] for connection-level failures
/// that arrive without an HTTP status (TLS reset, socket hang up).
///
/// Each entry must be a substring the upstream message reliably
/// contains; add to this list when telemetry surfaces a new flake.
pub(crate) const RETRYABLE_MESSAGE_SUBSTRINGS: &[&str] = &[
    "ECONNRESET",
    "socket hang up",
    "headers timeout",
    "connection reset",
    "connection closed before response received",
    "stream truncated",
    "broken pipe",
    "Connection refused",
    "request timed out",
    "operation timed out",
    "ERR_PNPM_FETCH_",
    "ERR_PNPM_META_FETCH_FAIL",
    "ERR_PNPM_TARBALL_",
];

/// Substrings indicating a content / safety filter rejection.
/// Terminal: retrying the same prompt won't help; it must be rewritten.
pub(crate) const CONTENT_POLICY_SUBSTRINGS: &[&str] = &[
    "content_policy",
    "content policy",
    "safety filter",
    "moderation",
    "policy_violation",
    "violates the usage policies",
    "harmful or inappropriate",
];

/// Substrings indicating the prompt exceeded the model's context window,
/// matched ASCII-case-insensitively by
/// [`ProviderFailure::message_is_context_length`]. Pairs (`a` AND `b`)
/// live there, not here.
pub(crate) const CONTEXT_LENGTH_SUBSTRINGS: &[&str] = &[
    "prompt is too long",
    "maximum context length",
    "context_length_exceeded",
];

/// Substrings indicating the ACCOUNT is out of money (not a rate limit),
/// matched ASCII-case-insensitively. Terminal: retrying cannot make
/// credits appear.
pub(crate) const OUT_OF_CREDITS_SUBSTRINGS: &[&str] = &[
    "insufficient credits",
    "insufficient_credits",
    "insufficient_quota",
    "out of credits",
    "credit balance is too low",
];

/// Refined provider-failure taxonomy. Wider than [`RetryClass`] so
/// dashboards can distinguish quota (Throttled) from prompt rejection
/// (ContentPolicy) even though both are non-`Retryable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FailureClass {
    /// Generic transient transport / 5xx error. Retry with backoff.
    Retryable,
    /// 429 / `rate_limit_exceeded`. Retryable; its own bucket so
    /// dashboards can attribute quota burn correctly.
    Throttled,
    /// Operation explicitly cancelled. Not retryable.
    Cancellation,
    /// 401 / 403 — credentials wrong or expired. Terminal.
    Authentication,
    /// Malformed 4xx other than auth / rate-limit / content-policy
    /// (size, schema, model unavailable). Terminal.
    InvalidRequest,
    /// Content / safety filter rejected the prompt. Terminal: rewrite
    /// the prompt, don't retry it.
    ContentPolicy,
    /// Idle / total deadline elapsed without a complete response.
    /// Terminal for this attempt; the caller decides whether to retry.
    StreamTimeout,
    /// The composed prompt exceeded the model's context window. Terminal
    /// AND not worth resuming as-is: a retry rebuilds the same oversized
    /// prompt and hits the same wall — the caller must shrink it.
    ContextLengthExceeded,
    /// The account is out of money (HTTP 402, `insufficient_quota`, …).
    /// Terminal: unlike [`FailureClass::Throttled`], nothing settles on
    /// its own. A provider that sends 402 for a CONCURRENCY limit should
    /// be reclassified by its adapter as `Throttled`.
    OutOfCredits,
    /// Unrecognized; treated as permanent. New shapes landing here are
    /// a signal to extend the classifier.
    Unknown,
}

impl FailureClass {
    /// Whether this class is retryable by default.
    #[inline]
    pub fn retryable(self) -> bool {
        matches!(self, FailureClass::Retryable | FailureClass::Throttled)
    }

    /// Lower to the coarse [`RetryClass`] for the 2-way classification
    /// (e.g. [`BorgError::Llm`](crate::BorgError::Llm)).
    #[inline]
    pub fn to_retry_class(self) -> RetryClass {
        if self.retryable() {
            RetryClass::Retryable
        } else {
            RetryClass::Terminal
        }
    }

    /// Short kebab-case label for dashboards / logs. Stable
    /// vocabulary: new variants get a new tag, never reuse an existing.
    pub fn as_str(self) -> &'static str {
        match self {
            FailureClass::Retryable => "retryable",
            FailureClass::Throttled => "throttled",
            FailureClass::Cancellation => "cancellation",
            FailureClass::Authentication => "authentication",
            FailureClass::InvalidRequest => "invalid-request",
            FailureClass::ContentPolicy => "content-policy",
            FailureClass::StreamTimeout => "stream-timeout",
            FailureClass::ContextLengthExceeded => "context-length-exceeded",
            FailureClass::OutOfCredits => "out-of-credits",
            FailureClass::Unknown => "unknown",
        }
    }
}

/// Typed envelope for a classified provider failure. Adapters build
/// one via the `classify_*` functions; the executor reads `.class`,
/// dashboards read `.http_status` / `.provider_code`, UI reads
/// `.message`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderFailure {
    /// The refined taxonomy bucket.
    pub class: FailureClass,
    /// Upstream HTTP status code, when known.
    pub http_status: Option<u16>,
    /// Provider's symbolic code (`"overloaded"`,
    /// `"rate_limit_exceeded"`, …), when known. Free-form; not
    /// normalized.
    pub provider_code: Option<String>,
    /// Human / log-grade error message.
    pub message: String,
    /// Server-suggested retry-after delay from a `Retry-After` header
    /// or error body. `None` if the adapter doesn't surface it.
    pub retry_after: Option<Duration>,
}

impl ProviderFailure {
    /// Hand-build a failure from a class + message. Prefer the
    /// `classify_*` functions when you have richer input.
    pub fn new(class: FailureClass, message: impl Into<String>) -> Self {
        Self {
            class,
            http_status: None,
            provider_code: None,
            message: message.into(),
            retry_after: None,
        }
    }

    #[must_use]
    pub fn with_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    #[must_use]
    pub fn with_provider_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into());
        self
    }

    #[must_use]
    pub fn with_retry_after(mut self, after: Duration) -> Self {
        self.retry_after = Some(after);
        self
    }

    /// Convenience forward to [`FailureClass::retryable`].
    #[inline]
    pub fn is_retryable(&self) -> bool {
        self.class.retryable()
    }

    /// Lower to the coarse [`RetryClass`].
    #[inline]
    pub fn to_retry_class(&self) -> RetryClass {
        self.class.to_retry_class()
    }

    /// Classify from HTTP status alone — the most common entry point.
    pub fn classify_status(status: u16) -> Self {
        let class = match status {
            408 | 500 | 502 | 503 | 504 => FailureClass::Retryable,
            429 => FailureClass::Throttled,
            401 | 403 => FailureClass::Authentication,
            402 => FailureClass::OutOfCredits,
            s if (400..500).contains(&s) => FailureClass::InvalidRequest,
            _ => FailureClass::Unknown,
        };
        Self::new(class, format!("HTTP {status}")).with_status(status)
    }

    /// Classify from a provider's symbolic error code. Accepts numeric
    /// strings (`"502"`, when a provider serializes the status as the
    /// code) and symbolic vocabulary (`"overloaded"`,
    /// `"rate_limit_exceeded"`, `"content_policy_violation"`, …).
    pub fn classify_provider_code(code: &str, message: &str) -> Self {
        // Numeric → route through status.
        if let Ok(status) = code.parse::<u16>() {
            return Self::classify_status(status).with_provider_code(code);
        }
        let class = match code {
            "overloaded"
            | "server_error"
            | "service_unavailable"
            | "gateway_timeout"
            | "internal_server_error" => FailureClass::Retryable,
            "rate_limit_exceeded" | "rate_limited" | "quota_exceeded" => FailureClass::Throttled,
            "content_policy_violation" | "content_policy" | "moderation" | "safety_filter" => {
                FailureClass::ContentPolicy
            }
            "invalid_request_error" | "invalid_request" | "request_too_large" => {
                FailureClass::InvalidRequest
            }
            "authentication_error" | "permission_error" | "unauthorized" => {
                FailureClass::Authentication
            }
            "timeout" | "request_timeout" | "stream_timeout" => FailureClass::StreamTimeout,
            "context_length_exceeded" | "context_window_exceeded" | "prompt_too_long" => {
                FailureClass::ContextLengthExceeded
            }
            "insufficient_quota"
            | "insufficient_credits"
            | "payment_required"
            | "credits_exhausted" => FailureClass::OutOfCredits,
            "cancelled" | "canceled" | "operation_cancelled" => FailureClass::Cancellation,
            _ => FailureClass::Unknown,
        };
        Self::new(class, message).with_provider_code(code)
    }

    /// Whether a provider error message describes a context-window
    /// overflow. Anthropic: "prompt is too long: N tokens > M maximum";
    /// OpenAI: "maximum context length is N tokens"; an aggregator's
    /// passthrough of either. ASCII-case-insensitive.
    pub fn message_is_context_length(message: &str) -> bool {
        let lower = message.to_ascii_lowercase();
        CONTEXT_LENGTH_SUBSTRINGS.iter().any(|n| lower.contains(n))
            || (lower.contains("context length") && lower.contains("exceed"))
            || (lower.contains("too many tokens") && lower.contains("maximum"))
    }

    /// Whether a provider error message says the account is out of money
    /// ("insufficient credits", "insufficient_quota", "credit balance is
    /// too low", …). ASCII-case-insensitive.
    pub fn message_is_out_of_credits(message: &str) -> bool {
        let lower = message.to_ascii_lowercase();
        OUT_OF_CREDITS_SUBSTRINGS.iter().any(|n| lower.contains(n))
    }

    /// Reclassify as a context-window overflow, keeping status, code and
    /// message. For an adapter that learns it only after a generic
    /// classification (e.g. a 400 whose body names the window).
    #[must_use]
    pub fn into_context_length(mut self) -> Self {
        self.class = FailureClass::ContextLengthExceeded;
        self
    }

    /// Last-resort classifier from an unparsed error message. Checks for
    /// content-policy wording, then the context-window and
    /// out-of-credits shapes, then connection-level flake wording
    /// (`ECONNRESET`, "socket hang up", …); unmatched messages fall back
    /// to `Unknown` rather than retryable. The substring lists are an
    /// implementation detail and may grow in any release.
    pub fn classify_message(message: &str) -> Self {
        for needle in CONTENT_POLICY_SUBSTRINGS {
            if message.contains(needle) {
                return Self::new(FailureClass::ContentPolicy, message);
            }
        }
        if Self::message_is_context_length(message) {
            return Self::new(FailureClass::ContextLengthExceeded, message);
        }
        if Self::message_is_out_of_credits(message) {
            return Self::new(FailureClass::OutOfCredits, message);
        }
        for needle in RETRYABLE_MESSAGE_SUBSTRINGS {
            if message.contains(needle) {
                return Self::new(FailureClass::Retryable, message);
            }
        }
        Self::new(FailureClass::Unknown, message)
    }
}

/// Self-classification trait: lets retry policies branch on a typed
/// classification without re-parsing strings. [`BorgError`](crate::BorgError)
/// implements it; custom error types can too.
pub trait RetryableError {
    /// Coarse 2-way classification. The executor reads this when
    /// deciding whether to honor `RestartPolicy::Restart`.
    fn retry_class(&self) -> RetryClass;

    /// Refined classification. Default derives from `retry_class`
    /// (losing the throttled / auth / content-policy distinction);
    /// override when carrying a typed [`ProviderFailure`].
    fn failure_class(&self) -> FailureClass {
        match self.retry_class() {
            RetryClass::Retryable => FailureClass::Retryable,
            RetryClass::Terminal => FailureClass::Unknown,
        }
    }
}

impl RetryableError for crate::BorgError {
    fn retry_class(&self) -> RetryClass {
        if self.is_retryable() {
            RetryClass::Retryable
        } else {
            RetryClass::Terminal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FailureClass basics ─────────────────────────────────────────────

    #[test]
    fn retryable_classes_lower_to_retryable_retry_class() {
        assert_eq!(
            FailureClass::Retryable.to_retry_class(),
            RetryClass::Retryable
        );
        assert_eq!(
            FailureClass::Throttled.to_retry_class(),
            RetryClass::Retryable
        );
    }

    #[test]
    fn terminal_classes_lower_to_terminal_retry_class() {
        for c in [
            FailureClass::Authentication,
            FailureClass::InvalidRequest,
            FailureClass::ContentPolicy,
            FailureClass::StreamTimeout,
            FailureClass::Cancellation,
            FailureClass::ContextLengthExceeded,
            FailureClass::OutOfCredits,
            FailureClass::Unknown,
        ] {
            assert_eq!(c.to_retry_class(), RetryClass::Terminal, "{c:?}");
        }
    }

    #[test]
    fn as_str_is_stable_kebab_case() {
        assert_eq!(FailureClass::Retryable.as_str(), "retryable");
        assert_eq!(FailureClass::Throttled.as_str(), "throttled");
        assert_eq!(FailureClass::ContentPolicy.as_str(), "content-policy");
        assert_eq!(FailureClass::InvalidRequest.as_str(), "invalid-request");
    }

    // ── classify_status ─────────────────────────────────────────────────

    #[test]
    fn classify_status_routes_5xx_to_retryable() {
        for s in [500, 502, 503, 504] {
            assert_eq!(
                ProviderFailure::classify_status(s).class,
                FailureClass::Retryable,
                "{s}"
            );
        }
        assert_eq!(
            ProviderFailure::classify_status(408).class,
            FailureClass::Retryable
        );
    }

    #[test]
    fn classify_status_routes_429_to_throttled() {
        let f = ProviderFailure::classify_status(429);
        assert_eq!(f.class, FailureClass::Throttled);
        assert!(f.is_retryable());
    }

    #[test]
    fn classify_status_routes_401_403_to_authentication() {
        for s in [401, 403] {
            let f = ProviderFailure::classify_status(s);
            assert_eq!(f.class, FailureClass::Authentication, "{s}");
            assert!(!f.is_retryable());
        }
    }

    #[test]
    fn classify_status_routes_other_4xx_to_invalid_request() {
        for s in [400, 404, 409, 422] {
            assert_eq!(
                ProviderFailure::classify_status(s).class,
                FailureClass::InvalidRequest,
                "{s}"
            );
        }
    }

    #[test]
    fn classify_status_carries_status_code() {
        assert_eq!(ProviderFailure::classify_status(503).http_status, Some(503));
    }

    // ── classify_provider_code ─────────────────────────────────────────

    #[test]
    fn provider_code_numeric_routes_through_status() {
        let f = ProviderFailure::classify_provider_code("502", "bad gateway");
        assert_eq!(f.class, FailureClass::Retryable);
        assert_eq!(f.http_status, Some(502));
        assert_eq!(f.provider_code.as_deref(), Some("502"));
    }

    #[test]
    fn provider_code_overloaded_is_retryable() {
        for code in [
            "overloaded",
            "server_error",
            "service_unavailable",
            "gateway_timeout",
        ] {
            let f = ProviderFailure::classify_provider_code(code, "x");
            assert_eq!(f.class, FailureClass::Retryable, "{code}");
        }
    }

    #[test]
    fn provider_code_rate_limit_is_throttled() {
        for code in ["rate_limit_exceeded", "rate_limited", "quota_exceeded"] {
            assert_eq!(
                ProviderFailure::classify_provider_code(code, "x").class,
                FailureClass::Throttled,
                "{code}"
            );
        }
    }

    #[test]
    fn provider_code_content_policy_is_terminal() {
        for code in ["content_policy_violation", "moderation", "safety_filter"] {
            let f = ProviderFailure::classify_provider_code(code, "x");
            assert_eq!(f.class, FailureClass::ContentPolicy, "{code}");
            assert!(!f.is_retryable());
        }
    }

    #[test]
    fn provider_code_unknown_lands_in_unknown() {
        let f = ProviderFailure::classify_provider_code("brand_new_shape", "x");
        assert_eq!(f.class, FailureClass::Unknown);
    }

    // ── classify_message ────────────────────────────────────────────────

    #[test]
    fn message_with_econnreset_is_retryable() {
        let f = ProviderFailure::classify_message("upstream error: ECONNRESET");
        assert_eq!(f.class, FailureClass::Retryable);
    }

    #[test]
    fn message_with_socket_hang_up_is_retryable() {
        let f = ProviderFailure::classify_message("socket hang up");
        assert_eq!(f.class, FailureClass::Retryable);
    }

    #[test]
    fn message_with_content_policy_substring_is_terminal_content_policy() {
        let f = ProviderFailure::classify_message("Request blocked by safety filter");
        assert_eq!(f.class, FailureClass::ContentPolicy);
    }

    #[test]
    fn message_with_unknown_pattern_is_unknown() {
        let f = ProviderFailure::classify_message("unexpected condition occurred");
        assert_eq!(f.class, FailureClass::Unknown);
    }

    #[test]
    fn content_policy_takes_precedence_over_transient() {
        // A message matching both a transient and a content-policy
        // substring lands in ContentPolicy — retrying past a moderation
        // block is wrong.
        let f = ProviderFailure::classify_message(
            "ECONNRESET while reading response (content policy violation)",
        );
        assert_eq!(f.class, FailureClass::ContentPolicy);
    }

    // ── context window / out of credits ────────────────────────────────

    #[test]
    fn context_window_messages_from_each_provider_shape() {
        for m in [
            "prompt is too long: 210000 tokens > 200000 maximum",
            "This model's maximum context length is 128000 tokens",
            "error code: context_length_exceeded",
            "Input context length 300000 exceeds the limit",
            "Too many tokens: the maximum is 1048576",
        ] {
            assert!(ProviderFailure::message_is_context_length(m), "{m}");
            let f = ProviderFailure::classify_message(m);
            assert_eq!(f.class, FailureClass::ContextLengthExceeded, "{m}");
            assert!(!f.is_retryable(), "{m}");
        }
        assert!(!ProviderFailure::message_is_context_length(
            "too many tokens per minute"
        ));
        assert_eq!(
            ProviderFailure::classify_provider_code("context_length_exceeded", "x").class,
            FailureClass::ContextLengthExceeded
        );
        let f = ProviderFailure::classify_status(400).into_context_length();
        assert_eq!(f.class, FailureClass::ContextLengthExceeded);
        assert_eq!(f.http_status, Some(400), "the status survives the reclass");
    }

    #[test]
    fn an_empty_account_is_terminal_not_throttled() {
        let f = ProviderFailure::classify_status(402);
        assert_eq!(f.class, FailureClass::OutOfCredits);
        assert!(!f.is_retryable(), "retrying cannot make money appear");
        for code in [
            "insufficient_quota",
            "insufficient_credits",
            "payment_required",
        ] {
            assert_eq!(
                ProviderFailure::classify_provider_code(code, "x").class,
                FailureClass::OutOfCredits,
                "{code}"
            );
        }
        let m = "Your credit balance is too low to access the API";
        assert_eq!(
            ProviderFailure::classify_message(m).class,
            FailureClass::OutOfCredits
        );
        assert_eq!(FailureClass::OutOfCredits.as_str(), "out-of-credits");
        assert_eq!(
            FailureClass::ContextLengthExceeded.as_str(),
            "context-length-exceeded"
        );
        // A rate limit is still a rate limit.
        assert_eq!(
            ProviderFailure::classify_provider_code("quota_exceeded", "x").class,
            FailureClass::Throttled
        );
    }

    // ── ProviderFailure builders ────────────────────────────────────────

    #[test]
    fn builder_methods_compose() {
        let f = ProviderFailure::new(FailureClass::Throttled, "rate limit hit")
            .with_status(429)
            .with_provider_code("rate_limit_exceeded")
            .with_retry_after(Duration::from_secs(30));
        assert_eq!(f.http_status, Some(429));
        assert_eq!(f.provider_code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(f.retry_after, Some(Duration::from_secs(30)));
    }

    // ── RetryableError trait ────────────────────────────────────────────

    #[test]
    fn retryable_error_impl_for_borg_error() {
        use crate::error::BorgError;
        let llm_retryable = BorgError::llm("503", RetryClass::Retryable);
        assert_eq!(llm_retryable.retry_class(), RetryClass::Retryable);

        let llm_terminal = BorgError::llm("401", RetryClass::Terminal);
        assert_eq!(llm_terminal.retry_class(), RetryClass::Terminal);

        let cancelled = BorgError::Cancelled;
        assert_eq!(cancelled.retry_class(), RetryClass::Terminal);
    }

    // ── Constants ───────────────────────────────────────────────────────

    #[test]
    fn retryable_http_status_set_matches_documented_codes() {
        assert_eq!(RETRYABLE_HTTP_STATUS, &[408, 429, 500, 502, 503, 504]);
    }

    #[test]
    fn retryable_message_substrings_are_nonempty() {
        // `is_empty` would be a compile-time `const false`
        // (clippy::const_is_empty), so assert via `contains` instead.
        assert!(RETRYABLE_MESSAGE_SUBSTRINGS.contains(&"ECONNRESET"));
        assert!(RETRYABLE_MESSAGE_SUBSTRINGS.contains(&"socket hang up"));
    }
}
