//! Minimal wrapper around an rmcp client peer.
//!
//! Owns the lifecycle of one stdio-MCP-server connection. No reconnect:
//! if the subprocess dies, the next `call_tool` errors and the consumer
//! rebuilds the dispatcher.

use rmcp::{
    model::{
        CallToolRequest, CallToolRequestParams, CallToolResult, CancelledNotificationParam,
        ClientRequest, ServerResult, Tool,
    },
    service::{
        PeerRequestOptions, RequestHandle, RunningService, RxJsonRpcMessage, TxJsonRpcMessage,
    },
    transport::{
        async_rw::{JsonRpcMessageCodec, JsonRpcMessageCodecError},
        Transport,
    },
    RoleClient, ServiceError, ServiceExt,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio_util::bytes::BytesMut;
use tokio_util::codec::{Decoder, Encoder};
use tokio_util::sync::CancellationToken;

use crate::error::McpError;

/// Default per-request timeout for `tools/call` (and for the
/// `initialize` handshake and `tools/list`). See
/// [`StdioServerSpec::call_timeout`].
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(60);

/// Default cap on a tool result's serialized size. See
/// [`StdioServerSpec::max_result_bytes`].
pub const DEFAULT_MAX_RESULT_BYTES: usize = 1024 * 1024;

/// Headroom on top of twice `max_result_bytes` for one JSON-RPC message
/// from the server: the envelope, a `tools/list` reply, and a result
/// carrying both `structuredContent` and its text rendering.
const FRAME_HEADROOM_BYTES: usize = 1024 * 1024;

/// Bytes read from the server's stdout per `read` call.
const READ_CHUNK: usize = 8 * 1024;

/// How long `close` waits for the server to exit after its stdin is
/// closed before killing it.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);

/// Largest single JSON-RPC message (one stdout line) accepted from a
/// server whose results are capped at `max_result_bytes`.
pub(crate) fn max_frame_bytes(max_result_bytes: usize) -> usize {
    max_result_bytes
        .saturating_mul(2)
        .saturating_add(FRAME_HEADROOM_BYTES)
}

/// Parent env vars every server gets by default, when set in the
/// parent. Enough for `npx` / `uvx` / interpreters to find themselves,
/// their caches and a temp dir; nothing that is usually a secret.
#[cfg(not(windows))]
pub const DEFAULT_ENV_PASSTHROUGH: &[&str] = &[
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR",
];

/// Parent env vars every server gets by default, when set in the
/// parent. Enough for `npx` / `uvx` / interpreters to find themselves,
/// their caches and a temp dir; nothing that is usually a secret.
#[cfg(windows)]
pub const DEFAULT_ENV_PASSTHROUGH: &[&str] = &[
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "SystemDrive",
    "windir",
    "ComSpec",
    "USERPROFILE",
    "USERNAME",
    "APPDATA",
    "LOCALAPPDATA",
    "TEMP",
    "TMP",
];

/// Spec for spawning an MCP server as a child process over stdio.
///
/// Mirrors the `command + args + env` shape used by Claude Desktop,
/// Cursor, and the reference servers.
///
/// # Environment
///
/// The child does **not** inherit the parent environment. It gets the
/// parent's values for [`DEFAULT_ENV_PASSTHROUGH`] (`PATH`, `HOME`,
/// locale, temp dir, and on Windows the system vars process launch
/// needs), plus any names added with [`pass_env`](Self::pass_env), plus
/// the explicit [`env`](Self::env) entries, which win. So each server
/// sees only the secrets you hand it. [`inherit_env(true)`](Self::inherit_env)
/// restores the old inherit-everything behaviour.
///
/// # Limits
///
/// Each `tools/call` is bounded by [`call_timeout`](Self::call_timeout)
/// (default [`DEFAULT_CALL_TIMEOUT`]) and its result by
/// [`max_result_bytes`](Self::max_result_bytes) (default
/// [`DEFAULT_MAX_RESULT_BYTES`]); [`cancel_token`](Self::cancel_token)
/// aborts in-flight calls.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StdioServerSpec {
    /// Executable to run (e.g. `npx`, `uvx`, an absolute path).
    pub command: String,
    /// Arguments passed to the executable.
    pub args: Vec<String>,
    /// Env vars set on the child, on top of the passthrough set. These
    /// override passthrough values of the same name.
    pub env: Vec<(String, String)>,
    /// Extra parent env var names to forward (by name, value read from
    /// the parent at spawn), on top of [`DEFAULT_ENV_PASSTHROUGH`].
    /// Ignored when `inherit_env` is set.
    pub pass_env: Vec<String>,
    /// Inherit the parent's whole environment instead of the passthrough
    /// set. Default `false`.
    pub inherit_env: bool,
    /// Per-request timeout for `tools/call`, `tools/list` and the
    /// `initialize` handshake. On expiry a `notifications/cancelled` is
    /// sent and the call returns a failed tool result.
    pub call_timeout: Duration,
    /// Cap on a tool result's serialized JSON size. A larger result is
    /// replaced by a failed tool result holding a truncated prefix and a
    /// marker.
    ///
    /// It also bounds memory: one message from the server (one stdout
    /// line) may be at most `2 * max_result_bytes + 1 MiB`. A longer
    /// line is not buffered; the call in flight gets a failed tool
    /// result, the server is killed, and later calls fail as on a dead
    /// server.
    pub max_result_bytes: usize,
    /// When this token is cancelled, in-flight `tools/call` requests are
    /// abandoned (with a `notifications/cancelled`) and dispatch returns
    /// `BorgError::Cancelled`.
    pub cancel: Option<CancellationToken>,
}

impl StdioServerSpec {
    /// Bare spec — no args, no extra env, default limits.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            pass_env: Vec::new(),
            inherit_env: false,
            call_timeout: DEFAULT_CALL_TIMEOUT,
            max_result_bytes: DEFAULT_MAX_RESULT_BYTES,
            cancel: None,
        }
    }

    /// Append a single argument.
    #[must_use]
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Append multiple arguments.
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Set an env var on the child.
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Forward the parent's value of `name` to the child (when set), on
    /// top of [`DEFAULT_ENV_PASSTHROUGH`]. Use for things like
    /// `HTTPS_PROXY` or `NODE_EXTRA_CA_CERTS`.
    #[must_use]
    pub fn pass_env(mut self, name: impl Into<String>) -> Self {
        self.pass_env.push(name.into());
        self
    }

    /// `true` makes the child inherit the parent's entire environment,
    /// every secret included. Default `false`.
    #[must_use]
    pub fn inherit_env(mut self, inherit: bool) -> Self {
        self.inherit_env = inherit;
        self
    }

    /// Per-request timeout. Default [`DEFAULT_CALL_TIMEOUT`].
    #[must_use]
    pub fn call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = timeout;
        self
    }

    /// Cap on a result's serialized size. Default
    /// [`DEFAULT_MAX_RESULT_BYTES`].
    #[must_use]
    pub fn max_result_bytes(mut self, bytes: usize) -> Self {
        self.max_result_bytes = bytes;
        self
    }

    /// Abort in-flight tool calls when `token` is cancelled (e.g. the
    /// session's cancel token).
    #[must_use]
    pub fn cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Apply the env policy to the child's command.
    fn apply_env(&self, cmd: &mut Command) {
        if !self.inherit_env {
            cmd.env_clear();
            let names = DEFAULT_ENV_PASSTHROUGH
                .iter()
                .copied()
                .chain(self.pass_env.iter().map(String::as_str));
            for name in names {
                if let Some(value) = std::env::var_os(name) {
                    cmd.env(name, value);
                }
            }
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
    }
}

/// Connected MCP client. Owns the rmcp `RunningService` peer.
///
/// On drop, rmcp's `RunningService` signals the child to shut down via
/// the JSON-RPC `shutdown` notification before the pipes close.
pub(crate) struct McpClient {
    peer: RunningService<RoleClient, ()>,
    timeout: Duration,
    cancel: Option<CancellationToken>,
    /// Set by the transport when the server sent an oversize message.
    oversize: Arc<AtomicBool>,
    /// The transport's per-message limit, for error messages.
    max_frame: usize,
}

impl McpClient {
    /// Spawn the child, run the MCP initialize handshake, and return a
    /// connected client. Waits up to `spec.call_timeout` for initialize.
    ///
    /// Errors: `Spawn` if the subprocess can't start, `Initialize` if
    /// the handshake fails (version mismatch, early crash), `Timeout` if
    /// it doesn't finish in time.
    pub(crate) async fn connect(spec: &StdioServerSpec) -> Result<Self, McpError> {
        let mut command = Command::new(&spec.command);
        command.args(&spec.args);
        spec.apply_env(&mut command);
        let max_frame = max_frame_bytes(spec.max_result_bytes);
        let transport =
            BoundedChildTransport::spawn(command, max_frame).map_err(|e| McpError::Spawn {
                command: spec.command.clone(),
                source: Arc::new(e),
            })?;
        let oversize = Arc::clone(&transport.oversize);

        // Dropping the pending handshake drops the transport, which
        // kills the child.
        let peer = tokio::time::timeout(spec.call_timeout, ().serve(transport))
            .await
            .map_err(|_| McpError::Timeout {
                operation: "initialize".to_string(),
                timeout: spec.call_timeout,
            })?
            .map_err(|e| McpError::Initialize {
                source: Arc::new(e),
            })?;

        Ok(Self {
            peer,
            timeout: spec.call_timeout,
            cancel: spec.cancel.clone(),
            oversize,
            max_frame,
        })
    }

    /// `true` once the server sent a message over the frame limit (the
    /// connection is then closed).
    fn closed_by_oversize(&self) -> bool {
        self.oversize.load(Ordering::Acquire)
    }

    /// Map a service error, naming the oversize frame when that is what
    /// closed the transport while this request was in flight.
    fn service_error(&self, stage: &'static str, was_closed: bool, e: ServiceError) -> McpError {
        if !was_closed && self.closed_by_oversize() {
            if let ServiceError::TransportClosed | ServiceError::TransportSend(_) = e {
                return McpError::MessageTooLarge {
                    limit: self.max_frame,
                };
            }
        }
        McpError::Service {
            stage,
            source: Arc::new(e),
        }
    }

    /// One-shot `tools/list` — the server's currently-advertised tools.
    pub(crate) async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let was_closed = self.closed_by_oversize();
        tokio::time::timeout(self.timeout, self.peer.list_all_tools())
            .await
            .map_err(|_| McpError::Timeout {
                operation: "tools/list".to_string(),
                timeout: self.timeout,
            })?
            .map_err(|e| self.service_error("tools/list", was_closed, e))
    }

    /// Call one tool. `name` is the *raw* (un-namespaced) name; the
    /// dispatch layer owns namespacing.
    ///
    /// Bounded by the spec's `call_timeout` and raced against its cancel
    /// token; either one sends `notifications/cancelled` to the server
    /// and returns `McpError::Timeout` / `McpError::Cancelled`.
    pub(crate) async fn call_tool(
        &self,
        name: String,
        args: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        // rmcp wants a `serde_json::Map`. Wrap a non-object value as
        // `{"value": ...}` rather than failing the call.
        let arguments_map = match args {
            serde_json::Value::Object(map) => map,
            serde_json::Value::Null => serde_json::Map::new(),
            other => {
                let mut m = serde_json::Map::new();
                m.insert("value".to_string(), other);
                m
            }
        };

        let mut params = CallToolRequestParams::new(name.clone());
        if !arguments_map.is_empty() {
            params = params.with_arguments(arguments_map);
        }

        let was_closed = self.closed_by_oversize();
        let service_err = |e| self.service_error("tools/call", was_closed, e);

        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = self
            .peer
            .send_cancellable_request(request, PeerRequestOptions::no_options())
            .await
            .map_err(service_err)?;
        let RequestHandle { rx, id, .. } = handle;

        let response = async { rx.await.unwrap_or(Err(ServiceError::TransportClosed)) };
        let cancelled = async {
            match &self.cancel {
                Some(token) => token.cancelled().await,
                None => std::future::pending().await,
            }
        };

        let (err, reason) = tokio::select! {
            res = tokio::time::timeout(self.timeout, response) => match res {
                Ok(Ok(ServerResult::CallToolResult(result))) => return Ok(result),
                Ok(Ok(_)) => return Err(service_err(ServiceError::UnexpectedResponse)),
                Ok(Err(e)) => return Err(service_err(e)),
                Err(_) => (
                    McpError::Timeout {
                        operation: format!("tools/call `{name}`"),
                        timeout: self.timeout,
                    },
                    "request timeout",
                ),
            },
            () = cancelled => (McpError::Cancelled { tool: name }, "cancelled by client"),
        };

        // Best effort: tell the server to stop working on it. Bounded so
        // a wedged transport can't turn a timeout into another hang.
        let notify = self.peer.notify_cancelled(CancelledNotificationParam {
            request_id: id,
            reason: Some(reason.to_string()),
        });
        let _ = tokio::time::timeout(Duration::from_secs(1), notify).await;
        Err(err)
    }
}

/// Stdio transport to a child MCP server that bounds every incoming
/// message.
///
/// rmcp's `TokioChildProcess` reads each stdout line with an unbounded
/// `read_until`, so one huge line is buffered whole before
/// `max_result_bytes` can look at it. This reads in fixed chunks through
/// rmcp's own line codec with a `max_length`: a line past it is never
/// buffered past the limit. The transport then kills the server and
/// ends the stream, which fails the requests in flight; the request id
/// is somewhere in the discarded bytes, so there is no telling which.
struct BoundedChildTransport {
    /// `kill_on_drop`, so the server dies with the transport.
    child: Child,
    stdout: ChildStdout,
    /// `None` once closed.
    stdin: Arc<tokio::sync::Mutex<Option<ChildStdin>>>,
    read_buf: BytesMut,
    decoder: JsonRpcMessageCodec<RxJsonRpcMessage<RoleClient>>,
    oversize: Arc<AtomicBool>,
    eof: bool,
}

impl BoundedChildTransport {
    fn spawn(mut command: Command, max_frame: usize) -> std::io::Result<Self> {
        command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("child stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("child stdout was not piped"))?;
        Ok(Self {
            child,
            stdout,
            stdin: Arc::new(tokio::sync::Mutex::new(Some(stdin))),
            read_buf: BytesMut::new(),
            decoder: JsonRpcMessageCodec::new_with_max_length(max_frame),
            oversize: Arc::new(AtomicBool::new(false)),
            eof: false,
        })
    }

    /// One decode step over the buffered bytes. `Err(())` ends the
    /// stream.
    fn decode_buffered(&mut self) -> Result<Option<RxJsonRpcMessage<RoleClient>>, ()> {
        loop {
            let step = if self.eof {
                self.decoder.decode_eof(&mut self.read_buf)
            } else {
                self.decoder.decode(&mut self.read_buf)
            };
            match step {
                Ok(item) => return Ok(item),
                Err(JsonRpcMessageCodecError::MaxLineLengthExceeded) => {
                    let limit = self.decoder.max_length();
                    tracing::warn!(
                        target: "borg_mcp",
                        limit,
                        "mcp server sent a message over the size limit; closing the connection"
                    );
                    self.oversize.store(true, Ordering::Release);
                    self.read_buf = BytesMut::new();
                    let _ = self.child.start_kill();
                    return Err(());
                }
                // The bad line is already consumed: skip it.
                Err(JsonRpcMessageCodecError::Serde(e)) => {
                    tracing::debug!(target: "borg_mcp", error = %e, "unparseable message from mcp server");
                }
                Err(e) => {
                    tracing::warn!(target: "borg_mcp", error = %e, "mcp transport read failed");
                    return Err(());
                }
            }
        }
    }
}

impl Transport<RoleClient> for BoundedChildTransport {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send + 'static {
        let stdin = Arc::clone(&self.stdin);
        async move {
            let mut buf = BytesMut::new();
            JsonRpcMessageCodec::<TxJsonRpcMessage<RoleClient>>::default()
                .encode(item, &mut buf)
                .map_err(std::io::Error::from)?;
            let mut guard = stdin.lock().await;
            let Some(writer) = guard.as_mut() else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "transport is closed",
                ));
            };
            writer.write_all(&buf).await?;
            writer.flush().await
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        loop {
            match self.decode_buffered() {
                Ok(Some(msg)) => return Some(msg),
                Err(()) => return None,
                Ok(None) if self.eof => return None,
                Ok(None) => {}
            }
            self.read_buf.reserve(READ_CHUNK);
            match self.stdout.read_buf(&mut self.read_buf).await {
                Ok(0) => self.eof = true,
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(target: "borg_mcp", error = %e, "reading from mcp server failed");
                    return None;
                }
            }
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        // Closing stdin is the stdio shutdown signal; give the server a
        // moment to exit on its own, then kill it.
        drop(self.stdin.lock().await.take());
        if tokio::time::timeout(SHUTDOWN_GRACE, self.child.wait())
            .await
            .is_err()
        {
            self.child.kill().await?;
        }
        Ok(())
    }
}
