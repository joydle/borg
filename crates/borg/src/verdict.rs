//! `Verdict` — the uniform return shape of a single agent invocation.
//!
//! Agents either return a verdict-token-prefixed body (line 1 = token,
//! body = the rest) or raw structured output (no token contract, body
//! = full response). Composition primitives route on `token`; structured
//! consumers call `parse_json`.

/// Parsed agent response: optional line-1 verdict token + body + usage.
///
/// For agents with a known-token contract (e.g. `["PASS", "FAIL"]`),
/// `token` is the matched token (ASCII case-insensitive) and `body` is
/// the response minus the token line. Without a contract, `token` is
/// empty and `body` is the full raw response.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Verdict {
    /// Line-1 token matched against the agent's declared set. Empty if
    /// no contract or no match.
    pub token: String,
    /// Response body: everything after the token line when one matched,
    /// else the full raw response.
    ///
    /// `Arc<str>` because bodies are multi-KB outputs cloned through
    /// `handoff.history` and downstream readers; sharing makes each
    /// clone a pointer bump. Derefs to `&str`.
    pub body: std::sync::Arc<str>,
    /// True when the agent has no contract OR the first line matched a
    /// declared token. False signals an unclassifiable response, which
    /// composition layers may treat as a soft failure.
    pub token_is_recognized: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Prompt tokens served from the provider's cache — a subset of
    /// `input_tokens`. Reported on `BorgEvent::TurnCompleted` and the
    /// turn span. 0 when the client reports none.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Prompt tokens written to the provider's cache — a subset of
    /// `input_tokens`. 0 when the client reports none.
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Native tool-use blocks the LLM emitted, mapped to borg's shape by
    /// the [`crate::AgentLlmClient`]. The default
    /// `BorgAgent::extract_tool_calls` reads this directly.
    pub tool_calls: Vec<crate::tools::ToolCallRequest>,
    /// Declarative handoff: the agent/branch to run next. When set, the
    /// executor writes it to `HandoffState::HANDOFF_KEY` for a
    /// downstream `Composition::Branch` to route on via
    /// [`crate::HandoffState::pending_handoff`]. `None` (default) lets
    /// the composition decide.
    pub handoff_to: Option<String>,
}

impl Verdict {
    /// Empty verdict — an agent declined to run. "No signal."
    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct from a raw model response.
    ///
    /// Empty `known_tokens` means no contract: `body` is the raw
    /// response and `token_is_recognized` is true. Otherwise the first
    /// non-empty line is matched case-insensitively; on match `body` is
    /// everything after the token line, on no match it's the full
    /// response with `token_is_recognized` false.
    pub fn from_response(
        response: String,
        known_tokens: &[&str],
        input_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        if known_tokens.is_empty() {
            return Self {
                body: response.into(),
                token_is_recognized: true,
                input_tokens,
                output_tokens,
                ..Self::default()
            };
        }

        let first_line = response
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim();
        let first_upper = first_line.to_ascii_uppercase();
        let matched = known_tokens
            .iter()
            .find(|t| first_upper == t.to_ascii_uppercase());

        if let Some(t) = matched {
            // Strip through the first non-empty line (the token line).
            let mut body_lines = response.lines().peekable();
            while let Some(&l) = body_lines.peek() {
                if l.trim().is_empty() {
                    body_lines.next();
                } else {
                    body_lines.next();
                    break;
                }
            }
            let body: std::sync::Arc<str> = body_lines.collect::<Vec<_>>().join("\n").trim().into();
            Self {
                token: (*t).to_string(),
                body,
                token_is_recognized: true,
                input_tokens,
                output_tokens,
                ..Self::default()
            }
        } else {
            Self {
                body: response.into(),
                input_tokens,
                output_tokens,
                ..Self::default()
            }
        }
    }

    /// Build a verdict from already-parsed parts, for agents whose
    /// `invoke()` does its own parsing.
    pub fn with_token(
        token: impl Into<String>,
        body: impl Into<std::sync::Arc<str>>,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        Self {
            token: token.into(),
            body: body.into(),
            token_is_recognized: true,
            input_tokens,
            output_tokens,
            ..Self::default()
        }
    }

    /// Attach the prompt-cache split of `input_tokens` (clamped to it).
    /// The default `BorgAgent::invoke` sets it from the
    /// [`crate::ChatResponse`]; an agent with its own `invoke()` sets it
    /// the same way so the executor can report it.
    #[must_use]
    pub fn with_cache_tokens(mut self, read: u64, creation: u64) -> Self {
        let read = read.min(self.input_tokens);
        self.cache_read_tokens = read;
        self.cache_creation_tokens = creation.min(self.input_tokens - read);
        self
    }

    /// Attach tool-use blocks the LLM emitted. Used by the
    /// [`crate::AgentLlmClient`] that translates provider-native
    /// tool_use into borg's shape:
    ///
    /// ```ignore
    /// Verdict::from_response(text, &[], in_tok, out_tok)
    ///     .with_tool_calls(response.tool_calls)
    /// ```
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<crate::tools::ToolCallRequest>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Try to parse the body as JSON. Falls back to extracting a fenced
    /// code block when direct parsing fails. Returns `None` on both
    /// paths failing — the agent returned unparseable output.
    pub fn parse_json<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        if let Ok(v) = serde_json::from_str::<T>(&self.body) {
            return Some(v);
        }
        let text = self.body.trim();
        if text.starts_with("```") {
            let fenced: String = text
                .lines()
                .skip(1)
                .take_while(|l| !l.starts_with("```"))
                .collect::<Vec<_>>()
                .join("\n");
            serde_json::from_str::<T>(&fenced).ok()
        } else {
            None
        }
    }

    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_contract_returns_raw_response_marked_recognized() {
        let v = Verdict::from_response("hello world".to_string(), &[], 10, 20);
        assert_eq!(v.token, "");
        assert_eq!(&*v.body, "hello world");
        assert!(v.token_is_recognized);
    }

    #[test]
    fn matched_token_strips_first_line() {
        let v = Verdict::from_response("PASS\nlooks good".to_string(), &["PASS", "FAIL"], 10, 20);
        assert_eq!(v.token, "PASS");
        assert_eq!(&*v.body, "looks good");
        assert!(v.token_is_recognized);
    }

    #[test]
    fn token_match_is_case_insensitive() {
        let v = Verdict::from_response("pass\nbody".to_string(), &["PASS"], 0, 0);
        assert_eq!(v.token, "PASS");
        assert_eq!(&*v.body, "body");
    }

    #[test]
    fn unmatched_first_line_falls_through_to_raw_body() {
        let v = Verdict::from_response("MAYBE\nuncertain".to_string(), &["PASS", "FAIL"], 10, 20);
        assert_eq!(v.token, "");
        assert_eq!(&*v.body, "MAYBE\nuncertain");
        assert!(!v.token_is_recognized);
    }

    #[test]
    fn parse_json_handles_fenced_block() {
        let v = Verdict::with_token("", "```json\n{\"x\": 1}\n```", 0, 0);
        let parsed: Option<serde_json::Value> = v.parse_json();
        assert_eq!(parsed.unwrap()["x"], 1);
    }

    #[test]
    fn parse_json_handles_bare_object() {
        let v = Verdict::with_token("", "{\"x\": 1}", 0, 0);
        let parsed: Option<serde_json::Value> = v.parse_json();
        assert_eq!(parsed.unwrap()["x"], 1);
    }

    #[test]
    fn parse_json_returns_none_on_garbage() {
        let v = Verdict::with_token("", "not json at all", 0, 0);
        let parsed: Option<serde_json::Value> = v.parse_json();
        assert!(parsed.is_none());
    }

    #[test]
    fn total_tokens_saturates() {
        let v = Verdict {
            token_is_recognized: true,
            input_tokens: u64::MAX,
            output_tokens: 100,
            ..Verdict::default()
        };
        assert_eq!(v.total_tokens(), u64::MAX);
    }
}
