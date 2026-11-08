//! Conversions between rmcp wire types and borg's tool protocol.
//!
//! Borg's `ToolSchema` and `ToolCallResult` mirror MCP shapes, so these
//! are mechanical projections. The decisions taken here:
//!
//! - **Description fallback**: `Tool::description` is optional; borg
//!   requires a non-empty string, so an empty one falls back to the
//!   tool name.
//! - **Content**: MCP's typed content blocks pass through as a JSON
//!   array, each block keeping its `type` discriminator — no
//!   first-text-wins flattening that would drop image/audio.
//! - **Error mapping**: `isError: true` becomes `ToolCallResult::fail`
//!   with the joined text blocks. Transport/protocol crashes (rmcp
//!   `ServiceError`) become `BorgError::Internal`.

use borg::{BorgError, ToolCallResult, ToolSchema};
use rmcp::model::{CallToolResult, RawContent, Tool};

/// Convert an rmcp `Tool` (from `tools/list`) to a borg `ToolSchema`.
///
/// `namespaced_name` is the already-namespaced name (e.g.
/// `mcp__github__create_issue`), produced by the dispatcher.
pub(crate) fn tool_to_schema(tool: &Tool, namespaced_name: String) -> ToolSchema {
    // Fall back to the bare name when no description: visibly wrong
    // beats a silently empty one.
    let description = tool
        .description
        .as_ref()
        .map(|c| c.as_ref().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| tool.name.as_ref().to_string());

    let input_schema = serde_json::Value::Object((*tool.input_schema).clone());

    ToolSchema::new(namespaced_name, description, input_schema)
}

/// Convert an rmcp `CallToolResult` to a borg `ToolCallResult`.
///
/// `is_error: Some(true)` maps to `ToolCallResult::fail` with the
/// joined text content; otherwise success surfaces
/// `structured_content` when present, else the content vec as a JSON
/// array. Both paths return `Ok` — `Err(BorgError)` is reserved for
/// transport/protocol crashes upstream of this function.
pub(crate) fn result_to_borg(result: CallToolResult) -> Result<ToolCallResult, BorgError> {
    let is_error = result.is_error.unwrap_or(false);

    if is_error {
        return Ok(ToolCallResult::fail(joined_text(&result.content)));
    }

    // Zero-content success returns an empty object so the agent sees a
    // non-null payload.
    let output = if let Some(structured) = result.structured_content {
        structured
    } else if result.content.is_empty() {
        serde_json::json!({})
    } else {
        content_vec_to_json(&result.content)
    };

    ToolCallResult::ok(output)
}

/// Enforce the result-size cap on what reaches the model.
///
/// Measures the serialized `output` plus the `error` and `summary`
/// text. Within `max_bytes` the result passes through untouched. Over
/// it, the result becomes a failure whose message starts with a
/// truncation marker followed by the first `max_bytes` bytes (cut on a
/// char boundary) of the error text, or else of the serialized output,
/// so the agent sees both that it was cut and roughly what came back.
pub(crate) fn cap_result(result: ToolCallResult, max_bytes: usize) -> ToolCallResult {
    let output_json = result
        .output
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let size = output_json.as_ref().map_or(0, String::len)
        + result.error.as_ref().map_or(0, String::len)
        + result.summary.as_ref().map_or(0, String::len);
    if size <= max_bytes {
        return result;
    }
    let text = result
        .error
        .or(output_json)
        .or(result.summary)
        .unwrap_or_default();
    let mut cut = max_bytes.min(text.len());
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    ToolCallResult::fail(format!(
        "[borg-mcp: tool result truncated: {size} bytes exceeds the {max_bytes}-byte limit; \
         first {cut} bytes follow]\n{}",
        &text[..cut]
    ))
}

/// Join all `text` blocks in a content vec with `\n` for an
/// `is_error: true` result. Non-text blocks are skipped.
fn joined_text(content: &[rmcp::model::Content]) -> String {
    let mut buf = String::new();
    for block in content {
        if let RawContent::Text(t) = &block.raw {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&t.text);
        }
    }
    if buf.is_empty() {
        // Error flagged but no text — surface that explicitly.
        "<mcp tool returned is_error=true with no text content>".to_string()
    } else {
        buf
    }
}

/// Serialize a content vec to a JSON array, preserving block types via
/// rmcp's own `Serialize` impls (canonical MCP wire shape).
fn content_vec_to_json(content: &[rmcp::model::Content]) -> serde_json::Value {
    let items: Vec<serde_json::Value> = content
        .iter()
        .filter_map(|block| serde_json::to_value(block).ok())
        .collect();
    serde_json::Value::Array(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Content;
    use std::sync::Arc;

    fn make_tool(name: &'static str, desc: Option<&'static str>) -> Tool {
        let empty_schema: Arc<serde_json::Map<String, serde_json::Value>> =
            Arc::new(serde_json::Map::new());
        Tool::new_with_raw(name, desc.map(Into::into), empty_schema)
    }

    #[test]
    fn tool_to_schema_uses_description() {
        let tool = make_tool("get_weather", Some("Look up the weather"));
        let schema = tool_to_schema(&tool, "mcp__weather__get_weather".to_string());
        assert_eq!(schema.name, "mcp__weather__get_weather");
        assert_eq!(schema.description, "Look up the weather");
    }

    #[test]
    fn tool_to_schema_falls_back_to_name_when_desc_missing() {
        let tool = make_tool("noop", None);
        let schema = tool_to_schema(&tool, "mcp__test__noop".to_string());
        assert_eq!(schema.description, "noop");
    }

    #[test]
    fn tool_to_schema_falls_back_to_name_when_desc_empty() {
        let tool = make_tool("noop", Some(""));
        let schema = tool_to_schema(&tool, "mcp__test__noop".to_string());
        assert_eq!(schema.description, "noop");
    }

    #[test]
    fn result_to_borg_success_with_structured() {
        let result = CallToolResult::structured(serde_json::json!({"answer": 42}));
        let borg_result = result_to_borg(result).unwrap();
        assert!(borg_result.error.is_none());
        assert_eq!(
            borg_result.output.unwrap(),
            serde_json::json!({"answer": 42})
        );
    }

    #[test]
    fn result_to_borg_success_with_content_array() {
        let result = CallToolResult::success(vec![
            Content::text("first line"),
            Content::text("second line"),
        ]);
        let borg_result = result_to_borg(result).unwrap();
        let output = borg_result.output.unwrap();
        assert!(output.is_array(), "expected JSON array of content blocks");
        assert_eq!(output.as_array().unwrap().len(), 2);
    }

    #[test]
    fn result_to_borg_empty_success() {
        let result = CallToolResult::success(vec![]);
        let borg_result = result_to_borg(result).unwrap();
        assert_eq!(borg_result.output.unwrap(), serde_json::json!({}));
    }

    #[test]
    fn result_to_borg_error_joins_text() {
        let result = CallToolResult::error(vec![
            Content::text("file not found"),
            Content::text("path: /tmp/missing"),
        ]);
        let borg_result = result_to_borg(result).unwrap();
        assert!(borg_result.output.is_none());
        assert_eq!(
            borg_result.error.unwrap(),
            "file not found\npath: /tmp/missing"
        );
    }

    #[test]
    fn cap_result_passes_small_results_through() {
        let r = ToolCallResult::ok(serde_json::json!({"a": 1})).unwrap();
        let capped = cap_result(r, 1024);
        assert_eq!(capped.output.unwrap(), serde_json::json!({"a": 1}));
        assert!(capped.error.is_none());
    }

    #[test]
    fn cap_result_truncates_oversized_output_as_failure() {
        let r = ToolCallResult::ok(serde_json::json!({"blob": "y".repeat(10_000)})).unwrap();
        let capped = cap_result(r, 256);
        assert!(capped.output.is_none());
        let err = capped.error.unwrap();
        assert!(err.contains("truncated"), "{err}");
        assert!(err.contains("256"), "{err}");
        assert!(err.len() < 256 + 200, "{} bytes", err.len());
    }

    #[test]
    fn cap_result_truncates_oversized_error_text() {
        let r = ToolCallResult::fail("e".repeat(10_000));
        let err = cap_result(r, 100).error.unwrap();
        assert!(err.contains("truncated"));
        assert!(err.len() < 100 + 200);
    }

    #[test]
    fn cap_result_cuts_on_char_boundary() {
        // Multi-byte chars straddling the cut must not panic.
        let r = ToolCallResult::fail("é".repeat(1000));
        let err = cap_result(r, 101).error.unwrap();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn result_to_borg_error_with_no_text_surfaces_placeholder() {
        let result = CallToolResult::error(vec![]);
        let borg_result = result_to_borg(result).unwrap();
        assert!(borg_result
            .error
            .unwrap()
            .contains("is_error=true with no text content"));
    }
}
