//! Claude Code `.mcp.json` (and compatible) config-file loader.
//!
//! The file shape — used by Claude Code, Claude Desktop, Cursor, and
//! most other MCP-aware tools — is:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "github": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-github"],
//!       "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" }
//!     },
//!     "filesystem": {
//!       "command": "uvx",
//!       "args": ["mcp-server-filesystem", "/tmp"]
//!     }
//!   }
//! }
//! ```
//!
//! The map key becomes the dispatcher's `server_label` (the
//! `mcp__<label>__<tool>` namespacing prefix) and is validated here, so
//! an ambiguous label (`a_`, `_a`, `a__b`, spaces, dots) fails the load
//! with [`McpError::InvalidLabel`].
//!
//! `env` entries are the *only* parent secrets a server sees: the child
//! starts from a cleared environment plus
//! [`DEFAULT_ENV_PASSTHROUGH`](crate::DEFAULT_ENV_PASSTHROUGH). Adjust
//! the returned [`StdioServerSpec`] (`pass_env`, `inherit_env`,
//! `call_timeout`, `max_result_bytes`) before connecting if needed.
//!
//! Only stdio transport is supported. SSE / HTTP entries parse fine but
//! load returns `McpError::UnsupportedTransport` rather than silently
//! skipping them.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

use crate::client::StdioServerSpec;
use crate::error::McpError;

/// One entry from `mcpServers`. Non-stdio fields are parsed only so we
/// can detect and reject them in [`ServerEntry::into_stdio_spec`].
#[derive(Debug, Clone, Deserialize)]
struct ServerEntry {
    /// `"stdio"` | `"sse"` | `"http"`; defaults to `"stdio"`.
    #[serde(default)]
    r#type: Option<String>,

    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,

    /// SSE / HTTP field — parsed only to detect and reject remote entries.
    #[serde(default)]
    url: Option<String>,
}

/// Top-level shape of `.mcp.json`.
#[derive(Debug, Clone, Deserialize)]
struct ConfigFile {
    #[serde(rename = "mcpServers", default)]
    mcp_servers: BTreeMap<String, ServerEntry>,
}

/// One parsed server config, ready to feed to
/// [`McpToolDispatch::connect_stdio`](crate::McpToolDispatch::connect_stdio).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct McpServerConfig {
    /// Map key from the JSON — becomes the namespacing prefix.
    pub label: String,
    /// Stdio launch spec.
    pub spec: StdioServerSpec,
}

/// Read and parse a Claude-Code-compatible `.mcp.json` file. Returns
/// the stdio server configs alphabetically by label (deterministic).
///
/// Errors with [`McpError::ConfigIo`] on read failure,
/// [`McpError::ConfigParse`] on invalid JSON or shape,
/// [`McpError::InvalidLabel`] on an ambiguous server name, and
/// [`McpError::UnsupportedTransport`] on a non-stdio entry. One bad
/// entry fails the whole load.
pub fn load_config_file(path: impl AsRef<Path>) -> Result<Vec<McpServerConfig>, McpError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| McpError::ConfigIo {
        path: path.display().to_string(),
        source: std::sync::Arc::new(e),
    })?;
    parse_config_str(&text)
}

/// Parse JSON text into validated configs. Same errors as
/// [`load_config_file`] minus the IO error.
pub fn parse_config_str(text: &str) -> Result<Vec<McpServerConfig>, McpError> {
    let parsed: ConfigFile = serde_json::from_str(text).map_err(|e| McpError::ConfigParse {
        message: e.to_string(),
    })?;

    let mut out = Vec::with_capacity(parsed.mcp_servers.len());
    for (label, entry) in parsed.mcp_servers {
        crate::namespace::validate_server_label(&label)
            .map_err(|message| McpError::InvalidLabel { message })?;
        let spec = entry.into_stdio_spec(&label)?;
        out.push(McpServerConfig { label, spec });
    }
    Ok(out)
}

impl ServerEntry {
    /// Validate and convert to a stdio spec, rejecting non-stdio entries.
    ///
    /// Env-var values support `${NAME}` references resolved from the
    /// parent environment; an unresolved reference fails the load so a
    /// missing credential can't slip through silently.
    fn into_stdio_spec(self, label: &str) -> Result<StdioServerSpec, McpError> {
        let transport_kind = self.r#type.as_deref().unwrap_or("stdio");
        if transport_kind != "stdio" {
            return Err(McpError::UnsupportedTransport {
                label: label.to_string(),
                kind: transport_kind.to_string(),
            });
        }

        // SSE/HTTP shapes carry `url` instead of `command`; reject them
        // as unsupported transport rather than as a missing-command error.
        if self.url.is_some() {
            return Err(McpError::UnsupportedTransport {
                label: label.to_string(),
                kind: "url-based (sse/http) — only stdio supported in this version".to_string(),
            });
        }
        let command = self.command.ok_or_else(|| McpError::ConfigParse {
            message: format!(
                "server `{label}`: missing `command` field (stdio transport requires it)"
            ),
        })?;

        let mut spec = StdioServerSpec::new(command).args(self.args);
        for (key, raw_value) in self.env {
            let resolved =
                resolve_env_refs(&raw_value).map_err(|missing| McpError::ConfigParse {
                    message: format!(
                        "server `{label}`: env var `{key}` references `${{{missing}}}` \
                     but it's not set in the parent environment"
                    ),
                })?;
            spec = spec.env(key, resolved);
        }

        Ok(spec)
    }
}

/// Substitute `${NAME}` references in `value` from the parent
/// environment. Returns `Err(name)` for the first unresolved reference.
fn resolve_env_refs(value: &str) -> Result<String, String> {
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + 2..];
        let end = after_open.find('}').ok_or_else(|| {
            // Unterminated `${...`: report the rest as the missing var.
            after_open.to_string()
        })?;
        let name = &after_open[..end];
        let resolved = std::env::var(name).map_err(|_| name.to_string())?;
        out.push_str(&resolved);
        rest = &after_open[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_stdio_entry() {
        let json = r#"{
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"]
                }
            }
        }"#;
        let configs = parse_config_str(json).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].label, "github");
        assert_eq!(configs[0].spec.command, "npx");
        assert_eq!(
            configs[0].spec.args,
            vec!["-y", "@modelcontextprotocol/server-github"]
        );
    }

    #[test]
    fn parse_preserves_alphabetical_order() {
        let json = r#"{
            "mcpServers": {
                "zebra": { "command": "z" },
                "alpha": { "command": "a" },
                "mike":  { "command": "m" }
            }
        }"#;
        let configs = parse_config_str(json).unwrap();
        let labels: Vec<&str> = configs.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, vec!["alpha", "mike", "zebra"]);
    }

    #[test]
    fn parse_explicit_stdio_type() {
        let json = r#"{
            "mcpServers": {
                "explicit": { "type": "stdio", "command": "thing" }
            }
        }"#;
        assert!(parse_config_str(json).is_ok());
    }

    #[test]
    fn parse_rejects_sse_transport() {
        let json = r#"{
            "mcpServers": {
                "remote": { "type": "sse", "url": "https://example.com/sse" }
            }
        }"#;
        let err = parse_config_str(json).unwrap_err();
        assert!(matches!(err, McpError::UnsupportedTransport { ref kind, .. } if kind == "sse"));
    }

    #[test]
    fn parse_rejects_http_transport() {
        let json = r#"{
            "mcpServers": {
                "remote": { "type": "http", "url": "https://example.com/mcp" }
            }
        }"#;
        let err = parse_config_str(json).unwrap_err();
        assert!(matches!(err, McpError::UnsupportedTransport { ref kind, .. } if kind == "http"));
    }

    #[test]
    fn parse_rejects_missing_command_in_stdio() {
        let json = r#"{
            "mcpServers": {
                "broken": { "args": ["x"] }
            }
        }"#;
        let err = parse_config_str(json).unwrap_err();
        assert!(
            matches!(err, McpError::ConfigParse { ref message } if message.contains("missing `command`"))
        );
    }

    #[test]
    fn parse_rejects_url_with_stdio_default() {
        // `url` without an explicit type still implies a remote entry.
        let json = r#"{
            "mcpServers": {
                "remote": { "url": "https://example.com/sse" }
            }
        }"#;
        assert!(matches!(
            parse_config_str(json),
            Err(McpError::UnsupportedTransport { .. })
        ));
    }

    #[test]
    fn parse_rejects_ambiguous_labels() {
        for label in ["a_", "_a", "a__b", "", "has space"] {
            let json = format!(r#"{{ "mcpServers": {{ "{label}": {{ "command": "x" }} }} }}"#);
            let err = parse_config_str(&json).unwrap_err();
            assert!(
                matches!(err, McpError::InvalidLabel { .. }),
                "label {label:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn parse_defaults_to_cleared_env() {
        let json = r#"{ "mcpServers": { "s": { "command": "x" } } }"#;
        let spec = &parse_config_str(json).unwrap()[0].spec;
        assert!(!spec.inherit_env);
    }

    #[test]
    fn parse_rejects_invalid_json() {
        let err = parse_config_str("{ this is not json").unwrap_err();
        assert!(matches!(err, McpError::ConfigParse { .. }));
    }

    #[test]
    fn parse_empty_mcp_servers() {
        let json = r#"{ "mcpServers": {} }"#;
        assert_eq!(parse_config_str(json).unwrap().len(), 0);
    }

    #[test]
    fn env_var_resolution_substitutes_known() {
        std::env::set_var("BORG_MCP_TEST_TOKEN", "secret-abc");
        let resolved = resolve_env_refs("prefix-${BORG_MCP_TEST_TOKEN}-suffix").unwrap();
        assert_eq!(resolved, "prefix-secret-abc-suffix");
    }

    #[test]
    fn env_var_resolution_fails_on_unset() {
        let err = resolve_env_refs("${DEFINITELY_NOT_SET_KSJDFKLSDJ}").unwrap_err();
        assert_eq!(err, "DEFINITELY_NOT_SET_KSJDFKLSDJ");
    }

    #[test]
    fn env_var_resolution_passes_through_plain_text() {
        assert_eq!(resolve_env_refs("just text").unwrap(), "just text");
    }

    #[test]
    fn parse_env_block_resolves_refs() {
        std::env::set_var("BORG_MCP_TEST_GH", "gh-token-xyz");
        let json = r#"{
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "env": { "GITHUB_TOKEN": "${BORG_MCP_TEST_GH}" }
                }
            }
        }"#;
        let configs = parse_config_str(json).unwrap();
        assert_eq!(
            configs[0].spec.env,
            vec![("GITHUB_TOKEN".to_string(), "gh-token-xyz".to_string())]
        );
    }

    #[test]
    fn parse_env_block_errors_on_missing_ref() {
        let json = r#"{
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "env": { "TOK": "${DEFINITELY_NOT_SET_QQQQQQ}" }
                }
            }
        }"#;
        let err = parse_config_str(json).unwrap_err();
        assert!(
            matches!(err, McpError::ConfigParse { ref message } if message.contains("DEFINITELY_NOT_SET_QQQQQQ"))
        );
    }
}
