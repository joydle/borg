//! Tool-name namespacing — `mcp__<server>__<tool>` shape.
//!
//! Borrowed from the Claude Agent SDK. The server prefix prevents
//! collisions (two servers exposing `search` would otherwise shadow
//! each other in `Registry::register`, which panics on duplicates) and
//! lets operators allowlist a whole server with `mcp__github__*`.
//!
//! The format is fixed. A tool whose underlying name contains `__`
//! still namespaces correctly — `parse` splits only on the first `__`
//! after the prefix.

/// Prefix every namespaced MCP tool name starts with.
pub(crate) const PREFIX: &str = "mcp__";

/// Build a namespaced name from a server label and raw tool name. Both
/// components pass through verbatim; the consumer owns label uniqueness
/// across dispatchers.
pub(crate) fn namespace(server_label: &str, tool_name: &str) -> String {
    format!("{PREFIX}{server_label}__{tool_name}")
}

/// Inverse of [`namespace`]: parse `mcp__<server>__<tool>` into
/// `(server, tool)`, or `None` if the name doesn't match.
///
/// Splits on the first `__` after the prefix, so a tool name that
/// itself contains `__` (e.g. `mcp__github__list__repos`) round-trips
/// as server=`github`, tool=`list__repos`.
pub(crate) fn parse(namespaced: &str) -> Option<(&str, &str)> {
    let after_prefix = namespaced.strip_prefix(PREFIX)?;
    let sep = after_prefix.find("__")?;
    let server = &after_prefix[..sep];
    let tool = &after_prefix[sep + 2..];
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server, tool))
}

/// Validate a server label for use in `mcp__<label>__<tool>` names.
///
/// A label must be non-empty ASCII letters, digits, `_` and `-` (the
/// characters model providers accept in tool names), must not contain
/// `__`, and must not start or end with `_`. With those rules the first
/// `__` after the prefix is always the separator, so [`parse`] can't
/// split at the wrong boundary and route a call to another server
/// (`a_` would build `mcp__a___t`, which reads back as server `a`).
pub(crate) fn validate_server_label(label: &str) -> Result<(), String> {
    let problem = if label.is_empty() {
        Some("is empty")
    } else if let Some(bad) = label
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-'))
    {
        return Err(format!(
            "MCP server label '{label}' contains '{bad}'; use only ASCII letters, \
             digits, '_' and '-'"
        ));
    } else if label.contains("__") {
        Some("contains '__' (double underscore)")
    } else if label.starts_with('_') || label.ends_with('_') {
        Some("starts or ends with '_'")
    } else {
        None
    };
    match problem {
        None => Ok(()),
        Some(problem) => Err(format!(
            "MCP server label '{label}' {problem}, which makes the \
             mcp__<server>__<tool> name ambiguous. Use single inner \
             underscores or hyphens instead."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_basic() {
        assert_eq!(
            namespace("github", "create_issue"),
            "mcp__github__create_issue"
        );
    }

    #[test]
    fn parse_basic() {
        assert_eq!(
            parse("mcp__github__create_issue"),
            Some(("github", "create_issue"))
        );
    }

    #[test]
    fn parse_round_trips_tool_with_underscores() {
        let combined = namespace("github", "list_all_repos");
        assert_eq!(parse(&combined), Some(("github", "list_all_repos")));
    }

    #[test]
    fn parse_round_trips_tool_with_double_underscore() {
        // Tool name itself contains `__`; the split on the first `__`
        // keeps the server label correct and preserves the separator.
        assert_eq!(
            parse("mcp__github__list__repos"),
            Some(("github", "list__repos"))
        );
    }

    #[test]
    fn parse_rejects_non_mcp_names() {
        assert_eq!(parse("read_file"), None);
        assert_eq!(parse("mcp_github__foo"), None); // single underscore after mcp
    }

    #[test]
    fn parse_rejects_missing_separator() {
        // No `__` after the server label.
        assert_eq!(parse("mcp__github_foo"), None);
    }

    #[test]
    fn parse_rejects_empty_components() {
        assert_eq!(parse("mcp____foo"), None); // empty server
        assert_eq!(parse("mcp__server__"), None); // empty tool
    }

    #[test]
    fn validate_label_rejects_double_underscore() {
        assert!(validate_server_label("foo__bar").is_err());
        assert!(validate_server_label("a__b__c").is_err());
        assert!(validate_server_label("foo").is_ok());
        assert!(validate_server_label("my-server").is_ok());
        assert!(validate_server_label("github_api").is_ok()); // single underscore ok
    }

    #[test]
    fn validate_label_rejects_edge_underscores() {
        // `a_` would build `mcp__a___t`, which `parse` reads back as
        // server `a`, tool `_t`: a different server.
        assert_eq!(parse(&namespace("a_", "t")), Some(("a", "_t")));
        assert!(validate_server_label("a_").is_err());
        assert!(validate_server_label("_a").is_err());
        assert!(validate_server_label("_").is_err());
    }

    #[test]
    fn validate_label_rejects_empty_and_non_name_chars() {
        assert!(validate_server_label("").is_err());
        assert!(validate_server_label("my server").is_err());
        assert!(validate_server_label("github.com").is_err());
        assert!(validate_server_label("a/b").is_err());
        assert!(validate_server_label("Server-2").is_ok());
    }

    #[test]
    fn valid_labels_round_trip_through_parse() {
        for label in ["a", "a_b", "a-b", "a_b-c_d", "A1"] {
            for tool in ["t", "_t", "t_", "list__repos", "__x"] {
                let name = namespace(label, tool);
                assert_eq!(parse(&name), Some((label, tool)), "{name}");
            }
        }
    }
}
