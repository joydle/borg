//! `McpToolDispatch` — an `MCP server` exposed as a borg `ToolDispatch`.
//!
//! The shape:
//!
//! ```text
//!   borg agent loop
//!         │
//!         │  schemas()                 ┌──── cached snapshot (ArcSwap)
//!         │  dispatch(req)             │
//!         ▼                            ▼
//!   McpToolDispatch  ──────────────▶  Vec<ToolSchema>   (namespaced names)
//!         │
//!         │  routes mcp__<server>__<tool> → tool
//!         ▼
//!     McpClient  ──── JSON-RPC over stdio ───▶  mcp server
//! ```
//!
//! - One dispatcher per MCP server; composing multiple servers is the
//!   consumer's job.
//! - Schemas are cached at construction. `refresh()` re-fetches
//!   `tools/list` and atomically swaps the snapshot, so `schemas()`
//!   readers never block.
//! - A name that doesn't match this dispatcher's `mcp__<server>__*`
//!   shape returns `ToolCallResult::fail("unknown tool ...")` — the
//!   same shape `Registry::dispatch` uses, keeping the agent's error
//!   handling uniform.

use arc_swap::ArcSwap;
use async_trait::async_trait;
use borg::{BorgError, DispatchedCall, ToolCallRequest, ToolCallResult, ToolDispatch, ToolSchema};
use std::sync::Arc;

use crate::client::{McpClient, StdioServerSpec};
use crate::convert::{cap_result, result_to_borg, tool_to_schema};
use crate::error::McpError;
use crate::namespace;

/// One MCP server, presented to borg as a `ToolDispatch`.
///
/// Construct via [`McpToolDispatch::connect_stdio`]. Hand the returned
/// `Arc<dyn ToolDispatch>` to a borg `Composition::Agent` exactly as
/// you would a `MutexDispatcher` over a native `Registry`.
pub struct McpToolDispatch {
    /// Label that namespaces every tool from this server
    /// (`mcp__<server_label>__<tool>`). The consumer picks it.
    server_label: String,
    client: McpClient,
    /// Result-size cap from the spec (`StdioServerSpec::max_result_bytes`).
    max_result_bytes: usize,
    /// Cached, namespaced tool schemas. `ArcSwap` keeps `schemas()`
    /// lock-free on the agent's hot path. Double-wrapped because
    /// ArcSwap needs a `Sized` payload while the trait returns
    /// `Arc<[ToolSchema]>`: the outer Arc is the swap handle, the inner
    /// is what `schemas()` clones out.
    schemas: ArcSwap<Arc<[ToolSchema]>>,
}

impl McpToolDispatch {
    /// Spawn an MCP server over stdio, run the initialize handshake,
    /// fetch the initial tool list, and return a connected dispatcher.
    ///
    /// `server_label` is the namespacing prefix component
    /// (`mcp__<server_label>__<tool>`). Pick something stable and short
    /// — operators read it in tool allowlists.
    ///
    /// Errors surface here so a broken config fails agent boot rather
    /// than the first turn that touches a tool.
    pub async fn connect_stdio(
        server_label: impl Into<String>,
        spec: &StdioServerSpec,
    ) -> Result<Arc<Self>, McpError> {
        let server_label = server_label.into();
        namespace::validate_server_label(&server_label)
            .map_err(|message| McpError::InvalidLabel { message })?;
        let client = McpClient::connect(spec).await?;

        let raw_tools = client.list_tools().await?;
        let schemas_vec = build_schemas(&server_label, &raw_tools);

        tracing::info!(
            target: "borg_mcp",
            server = %server_label,
            tool_count = schemas_vec.len(),
            "connected to mcp server"
        );

        let schemas: Arc<[ToolSchema]> = Arc::from(schemas_vec);
        Ok(Arc::new(Self {
            server_label,
            client,
            max_result_bytes: spec.max_result_bytes,
            schemas: ArcSwap::from_pointee(schemas),
        }))
    }

    /// Re-fetch `tools/list` and atomically swap the cached schemas.
    /// Call after a `tools/list_changed` notification, or poll.
    ///
    /// The server can change any tool's description and schema here,
    /// and the new text reaches the model on its next turn unreviewed.
    /// See the crate docs on tool poisoning.
    pub async fn refresh(&self) -> Result<(), McpError> {
        let raw_tools = self.client.list_tools().await?;
        let schemas_vec = build_schemas(&self.server_label, &raw_tools);

        tracing::info!(
            target: "borg_mcp",
            server = %self.server_label,
            tool_count = schemas_vec.len(),
            "refreshed mcp tool list"
        );

        let schemas: Arc<[ToolSchema]> = Arc::from(schemas_vec);
        self.schemas.store(Arc::new(schemas));
        Ok(())
    }

    /// The server label this dispatcher was constructed with — useful
    /// for building allowlist entries (`format!("mcp__{}__*", label)`).
    pub fn server_label(&self) -> &str {
        &self.server_label
    }
}

/// Longest namespaced tool name kept: the tool-name limit of the major
/// model APIs.
pub(crate) const MAX_TOOL_NAME_LEN: usize = 64;

/// Namespace and convert a server's `tools/list`, dropping (with a
/// `warn`) every tool the model APIs would reject or that would be
/// ambiguous: a raw name outside `[A-Za-z0-9_-]+`, a namespaced name over
/// [`MAX_TOOL_NAME_LEN`] bytes, or a repeat of a name already kept (the
/// first one wins).
pub(crate) fn build_schemas(server_label: &str, tools: &[rmcp::model::Tool]) -> Vec<ToolSchema> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let raw: &str = tool.name.as_ref();
        let problem = if raw.is_empty()
            || !raw
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            Some("name must match ^[A-Za-z0-9_-]+$")
        } else if namespace::namespace(server_label, raw).len() > MAX_TOOL_NAME_LEN {
            Some("namespaced name is longer than 64 characters")
        } else if !seen.insert(raw) {
            Some("duplicate tool name")
        } else {
            None
        };
        if let Some(problem) = problem {
            tracing::warn!(
                target: "borg_mcp",
                server = %server_label,
                tool = %raw.escape_debug(),
                problem,
                "dropping tool advertised by mcp server"
            );
            continue;
        }
        out.push(tool_to_schema(
            tool,
            namespace::namespace(server_label, raw),
        ));
    }
    out
}

/// Pure routing decision: `Ok(raw_name)` to forward to
/// `call_tool(raw_name, args)`, or `Err(reason)` to short-circuit as a
/// `ToolCallResult::fail`. Pure so tests can drive it without spawning
/// a server.
pub(crate) fn route_request<'a>(
    name: &'a str,
    server_label: &str,
    schemas: &[ToolSchema],
) -> Result<&'a str, String> {
    let Some((server, raw_name)) = namespace::parse(name) else {
        return Err(format!(
            "unknown tool `{name}` — does not match `mcp__<server>__<tool>` shape"
        ));
    };

    if server != server_label {
        return Err(format!(
            "unknown tool `{name}` — server `{server}` is not handled by this dispatcher \
             (label `{server_label}`)"
        ));
    }

    if !schemas.iter().any(|s| s.name == name) {
        return Err(format!(
            "unknown tool `{name}` — not advertised by mcp server `{server_label}`"
        ));
    }

    Ok(raw_name)
}

#[async_trait]
impl ToolDispatch for McpToolDispatch {
    async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError> {
        let ToolCallRequest {
            name,
            args,
            call_id,
            ..
        } = request;

        // Route before touching the network: short-circuit a namespace
        // mismatch or unknown tool as a logical failure.
        let raw_name = {
            let guard = self.schemas.load();
            let schemas: &[ToolSchema] = guard.as_ref();
            match route_request(&name, &self.server_label, schemas) {
                Ok(raw) => raw.to_string(),
                Err(reason) => {
                    return Ok(DispatchedCall::new(
                        call_id,
                        name,
                        ToolCallResult::fail(reason),
                    ));
                }
            }
        };

        // Transport failure -> `BorgError::Internal` (consumer rebuilds
        // the dispatcher); cancel -> `BorgError::Cancelled`. A timeout is
        // a failed tool result so the agent can recover or move on.
        // Tool-level `is_error: true` flows through `result_to_borg` as
        // `ToolCallResult::fail`.
        let raw_result = match self.client.call_tool(raw_name, args).await {
            Ok(r) => r,
            Err(McpError::Timeout { timeout, .. }) => {
                tracing::warn!(
                    target: "borg_mcp",
                    server = %self.server_label,
                    tool = %name,
                    ?timeout,
                    "mcp tool call timed out"
                );
                let result = ToolCallResult::fail(format!(
                    "mcp tool `{name}` timed out after {timeout:?}; the server may be hung \
                     and the request was cancelled"
                ));
                return Ok(DispatchedCall::new(call_id, name, result));
            }
            Err(McpError::MessageTooLarge { limit }) => {
                let result = ToolCallResult::fail(format!(
                    "mcp tool `{name}` returned a message over the {limit}-byte limit; it was \
                     discarded and the connection to the server was closed"
                ));
                return Ok(DispatchedCall::new(call_id, name, result));
            }
            Err(e) => return Err(BorgError::from(e)),
        };

        let result = cap_result(result_to_borg(raw_result)?, self.max_result_bytes);

        Ok(DispatchedCall::new(call_id, name, result))
    }

    fn schemas(&self) -> Arc<[ToolSchema]> {
        // Lock-free: clone the inner Arc out of the swap guard — two
        // pointer bumps, no per-element copy.
        Arc::clone(&self.schemas.load())
    }
}
