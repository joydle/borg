//! Routing tests for the dispatcher.
//!
//! Exercises `route_request` — `dispatch`'s pre-network short-circuit —
//! without spawning a server. End-to-end coverage (env policy, timeout,
//! cancel, result cap) against a fake stdio server lives in
//! `tests/stdio_server.rs`.

use borg::ToolSchema;

use crate::dispatch::route_request;

fn schema(name: &str) -> ToolSchema {
    ToolSchema::new(
        name.to_string(),
        name.to_string(),
        serde_json::json!({"type": "object"}),
    )
}

#[test]
fn route_accepts_known_namespaced_tool() {
    let schemas = vec![schema("mcp__github__create_issue")];
    let result = route_request("mcp__github__create_issue", "github", &schemas);
    assert_eq!(result.unwrap(), "create_issue");
}

#[test]
fn route_rejects_non_mcp_prefix() {
    let schemas = vec![schema("mcp__github__create_issue")];
    let err = route_request("read_file", "github", &schemas).unwrap_err();
    assert!(err.contains("does not match"), "got: {err}");
}

#[test]
fn route_rejects_wrong_server() {
    // Registered under "github", request names "weather": disclaim it.
    let schemas = vec![schema("mcp__github__create_issue")];
    let err = route_request("mcp__weather__forecast", "github", &schemas).unwrap_err();
    assert!(err.contains("not handled by this dispatcher"), "got: {err}");
}

#[test]
fn route_rejects_tool_not_in_schema_cache() {
    // Right server, but the tool isn't in the cached snapshot.
    let schemas = vec![schema("mcp__github__create_issue")];
    let err = route_request("mcp__github__delete_universe", "github", &schemas).unwrap_err();
    assert!(err.contains("not advertised by mcp server"), "got: {err}");
}

#[test]
fn route_preserves_tool_name_with_underscores() {
    let schemas = vec![schema("mcp__fs__list_all_files")];
    let result = route_request("mcp__fs__list_all_files", "fs", &schemas);
    assert_eq!(result.unwrap(), "list_all_files");
}

/// A `__`-containing label must return `Err(InvalidLabel)`, not panic.
/// Validation fires before any subprocess spawn, so a dummy spec works.
#[tokio::test]
async fn connect_stdio_returns_err_on_invalid_label() {
    use crate::client::StdioServerSpec;
    use crate::error::McpError;
    use crate::McpToolDispatch;

    let spec = StdioServerSpec::new("echo").arg("hello");
    let result = McpToolDispatch::connect_stdio("bad__label", &spec).await;
    assert!(
        matches!(result, Err(McpError::InvalidLabel { .. })),
        "expected InvalidLabel error"
    );
}

fn raw_tool(name: &str) -> rmcp::model::Tool {
    rmcp::model::Tool::new_with_raw(
        name.to_string(),
        None,
        std::sync::Arc::new(serde_json::Map::new()),
    )
}

/// Tool names the model APIs reject, overlong namespaced names and
/// duplicates are dropped; the first of a duplicate pair is kept.
#[test]
fn build_schemas_drops_invalid_overlong_and_duplicate_names() {
    use crate::dispatch::{build_schemas, MAX_TOOL_NAME_LEN};

    let label = "srv";
    // `mcp__srv__` is 10 bytes, so a 54-byte raw name is exactly 64.
    let longest_ok = "a".repeat(MAX_TOOL_NAME_LEN - "mcp__srv__".len());
    let too_long = "b".repeat(MAX_TOOL_NAME_LEN - "mcp__srv__".len() + 1);
    let mut first_dup = raw_tool("dup");
    first_dup.description = Some("first".into());
    let mut second_dup = raw_tool("dup");
    second_dup.description = Some("second".into());
    let tools = vec![
        raw_tool("ok_tool-1"),
        raw_tool("has space"),
        raw_tool("dotted.name"),
        raw_tool("slash/name"),
        raw_tool("unicodé"),
        raw_tool(""),
        raw_tool("new\nline"),
        raw_tool(&longest_ok),
        raw_tool(&too_long),
        first_dup,
        second_dup,
    ];
    let schemas = build_schemas(label, &tools);
    let names: Vec<&str> = schemas.iter().map(|s| s.name.as_str()).collect();
    let expected_long = format!("mcp__srv__{longest_ok}");
    assert_eq!(
        names,
        vec![
            "mcp__srv__ok_tool-1",
            expected_long.as_str(),
            "mcp__srv__dup"
        ]
    );
    let dup = schemas.iter().find(|s| s.name == "mcp__srv__dup").unwrap();
    assert_eq!(dup.description, "first", "the first duplicate wins");
}
