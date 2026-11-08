use tokio_util::sync::CancellationToken;

/// Per-call context for [`crate::SubprocessTool`].
///
/// Collects the subprocess's log lines during `execute`; inspect `lines`
/// afterward, or stream live via a callback (see `with_callback`).
///
/// One context can serve many calls: `lines` is cleared at the start of
/// every `execute`, so it never grows past one call's capped output.
pub struct SandboxContext {
    /// Lines kept from the most recent call, in order, after the tool's
    /// per-line and total output caps.
    pub lines: Vec<crate::backend::LogLine>,

    /// Optional live-streaming callback. Called for each kept line as it
    /// arrives (lines past the output cap are not delivered).
    /// `None` means lines are only collected into `self.lines`.
    pub(crate) on_line: Option<Box<dyn FnMut(crate::backend::LogLine) + Send>>,

    /// Cancel token forwarded to [`crate::SandboxBackend::run`]. Defaults
    /// to a fresh, never-cancelled token; set via [`with_cancel`] to
    /// propagate a parent agent's cancel to the subprocess.
    ///
    /// [`with_cancel`]: Self::with_cancel
    pub(crate) cancel: CancellationToken,
}

impl SandboxContext {
    /// Create a context that only collects lines (no live streaming).
    pub fn new() -> Self {
        Self {
            lines: vec![],
            on_line: None,
            cancel: CancellationToken::new(),
        }
    }

    /// Create a context that streams lines live AND collects them.
    pub fn with_callback(cb: impl FnMut(crate::backend::LogLine) + Send + 'static) -> Self {
        Self {
            lines: vec![],
            on_line: Some(Box::new(cb)),
            cancel: CancellationToken::new(),
        }
    }

    /// Attach a cancel token. When it fires, the subprocess is killed and
    /// `SandboxError::Cancelled` is surfaced. Typically pass
    /// `agent_ctx.cancel.child_token()`.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }
}

impl Default for SandboxContext {
    fn default() -> Self {
        Self::new()
    }
}
