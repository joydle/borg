//! Error type for borg-mcp.
//!
//! [`McpError`] covers spawning, handshaking, and calling the server;
//! it surfaces from `connect` so consumers fail fast at startup. At
//! dispatch time, transport/protocol failures convert into
//! `BorgError::Internal`, while tool-reported failures
//! (`is_error: true`) flow through `ToolCallResult::fail` instead.

use borg::BorgError;
use std::sync::Arc;

/// Errors from borg-mcp's lifecycle and transport layer.
///
/// Source fields are `Arc<...>` because rmcp's error types aren't
/// `Clone` but `McpError` must be (consumers may inspect connect
/// errors from multiple places).
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum McpError {
    /// Failed to spawn the subprocess.
    #[error("failed to spawn mcp server `{command}`: {source}")]
    Spawn {
        command: String,
        #[source]
        source: Arc<std::io::Error>,
    },

    /// MCP initialize handshake failed — child started but the
    /// initialize exchange didn't complete (timeout, protocol-version
    /// mismatch, child died pre-handshake).
    #[error("mcp initialize failed: {source}")]
    Initialize {
        #[source]
        source: Arc<rmcp::service::ClientInitializeError>,
    },

    /// rmcp service-layer error during a post-handshake request
    /// (tool call surfaced a transport error, list_tools after the
    /// server died, etc.). `stage` describes which step was in flight.
    #[error("mcp service error at `{stage}`: {source}")]
    Service {
        stage: &'static str,
        #[source]
        source: Arc<rmcp::ServiceError>,
    },

    /// Failed to read a `.mcp.json` config file from disk.
    #[error("failed to read mcp config `{path}`: {source}")]
    ConfigIo {
        path: String,
        #[source]
        source: Arc<std::io::Error>,
    },

    /// Failed to parse a `.mcp.json` payload (JSON syntax error, or
    /// a semantic mismatch like a missing `command` field).
    #[error("mcp config parse error: {message}")]
    ConfigParse { message: String },

    /// Config entry uses a transport we don't support yet (sse, http).
    #[error("mcp server `{label}` uses unsupported transport `{kind}` — only stdio is supported")]
    UnsupportedTransport { label: String, kind: String },

    /// Server label failed namespace validation (e.g. contains `__`).
    #[error("invalid mcp server label: {message}")]
    InvalidLabel { message: String },

    /// A request didn't complete within the spec's `call_timeout`.
    /// `operation` is `initialize`, `tools/list`, or `tools/call` plus
    /// the tool name.
    /// At dispatch time a timed-out `tools/call` becomes a failed
    /// `ToolCallResult`, not this error.
    #[error("mcp {operation} timed out after {timeout:?}")]
    Timeout {
        operation: String,
        timeout: std::time::Duration,
    },

    /// The server sent one message (one stdout line) longer than the
    /// connection's limit (`2 * max_result_bytes + 1 MiB`). It was not
    /// buffered; the server was killed and the connection closed.
    /// At dispatch time this becomes a failed `ToolCallResult` for the
    /// call in flight; later calls fail as on a dead server.
    #[error("mcp server sent a message over the {limit}-byte limit; connection closed")]
    MessageTooLarge { limit: usize },

    /// The spec's cancel token fired while a tool call was in flight.
    /// Converts to `BorgError::Cancelled`.
    #[error("mcp tool call `{tool}` cancelled")]
    Cancelled { tool: String },
}

impl From<McpError> for BorgError {
    fn from(err: McpError) -> Self {
        match err {
            McpError::Cancelled { .. } => BorgError::Cancelled,
            other => BorgError::Internal {
                message: other.to_string(),
            },
        }
    }
}
