pub const SESSION_ID: &str = "session.id";
pub const COMPOSITION_NAME: &str = "composition.name";
pub const COMPOSITION_KIND: &str = "composition.kind";
pub const STEP_NAME: &str = "step.name";
pub const STEP_INDEX: &str = "step.index";
pub const STEP_RETRY: &str = "step.retry";
pub const STEP_OUTCOME: &str = "step.outcome";
pub const STEP_FAILURE_CAUSE: &str = "step.failure_cause";
pub const TURN_INDEX: &str = "turn.index";
pub const LLM_MODEL: &str = "llm.model";
pub const LLM_INPUT_TOKENS: &str = "llm.input_tokens";
pub const LLM_OUTPUT_TOKENS: &str = "llm.output_tokens";
pub const LLM_CACHE_READ: &str = "llm.cache_read_tokens";
pub const LLM_CACHE_CREATION: &str = "llm.cache_creation_tokens";
pub const LLM_TTFT_MS: &str = "llm.ttft_ms";
pub const LLM_FINISH_REASON: &str = "llm.finish_reason";
pub const LLM_PROMPT_CHARS: &str = "llm.prompt_chars";
pub const TOOL_NAME: &str = "tool.name";
pub const TOOL_CALL_ID: &str = "tool.call_id";
pub const TOOL_SUCCESS: &str = "tool.success";
pub const TOOL_ERROR: &str = "tool.error";
pub const GUARDRAIL_RULE: &str = "guardrail.rule";
pub const GUARDRAIL_VERDICT: &str = "guardrail.verdict";
pub const GUARDRAIL_SIGNAL: &str = "guardrail.signal";
pub const PARALLEL_DEGRADED: &str = "parallel.degraded";
pub const COMPOSITION_TOTAL_INPUT_TOKENS: &str = "composition.total_input_tokens";
pub const COMPOSITION_TOTAL_OUTPUT_TOKENS: &str = "composition.total_output_tokens";
pub const COMPOSITION_SUCCEEDED: &str = "composition.succeeded";
pub const COMPOSITION_FAILED: &str = "composition.failed";
pub const COMPOSITION_OUTCOME: &str = "composition.outcome";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fields_are_dotted_non_empty() {
        let all = [
            SESSION_ID,
            COMPOSITION_NAME,
            COMPOSITION_KIND,
            STEP_NAME,
            STEP_INDEX,
            STEP_RETRY,
            STEP_OUTCOME,
            STEP_FAILURE_CAUSE,
            TURN_INDEX,
            LLM_MODEL,
            LLM_INPUT_TOKENS,
            LLM_OUTPUT_TOKENS,
            LLM_CACHE_READ,
            LLM_CACHE_CREATION,
            LLM_TTFT_MS,
            LLM_FINISH_REASON,
            LLM_PROMPT_CHARS,
            TOOL_NAME,
            TOOL_CALL_ID,
            TOOL_SUCCESS,
            TOOL_ERROR,
            GUARDRAIL_RULE,
            GUARDRAIL_VERDICT,
            GUARDRAIL_SIGNAL,
            PARALLEL_DEGRADED,
            COMPOSITION_TOTAL_INPUT_TOKENS,
            COMPOSITION_TOTAL_OUTPUT_TOKENS,
            COMPOSITION_SUCCEEDED,
            COMPOSITION_FAILED,
            COMPOSITION_OUTCOME,
        ];
        for f in all {
            assert!(!f.is_empty());
            assert!(f.contains('.'), "field should be dotted: {f}");
        }
    }
}
