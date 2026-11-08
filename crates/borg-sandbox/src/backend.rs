use async_trait::async_trait;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

/// Appended to a line of output that was cut at a byte cap (by
/// [`crate::TokioBackend`]'s per-line cap or by [`crate::SubprocessTool`]'s
/// per-line / total caps).
pub const TRUNCATION_MARKER: &str = " …[truncated]";

/// A single line of subprocess output.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LogLine {
    Stdout(String),
    Stderr(String),
}

/// Errors from sandbox operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SandboxError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("process exited with non-zero status {0}")]
    NonZeroExit(i32),
    #[error("setup failed: {0}")]
    Setup(String),
    #[error("timeout")]
    Timeout,
    #[error("cancelled")]
    Cancelled,
}

/// Command specification passed to [`SandboxBackend::run`].
#[non_exhaustive]
pub struct SandboxCommand {
    /// Executable to run inside the sandbox. [`crate::SubprocessTool`]
    /// with an allowlist always passes an absolute path here.
    pub program: String,
    /// Arguments (not including program name).
    pub args: Vec<String>,
    /// Working directory inside the sandbox. Backends map this to their
    /// internal mount layout.
    pub workdir: PathBuf,
    /// Environment overrides, applied on top of the backend's baseline
    /// env. For [`crate::TokioBackend`] the baseline is an *empty*
    /// environment plus its passthrough list; the parent's environment is
    /// not inherited.
    pub env: Vec<(String, String)>,
}

impl SandboxCommand {
    /// Construct a command with the required fields (program + workdir);
    /// builder methods fill in the rest. Lets external crates build a
    /// `SandboxCommand` despite the `#[non_exhaustive]` annotation.
    pub fn new(program: impl Into<String>, workdir: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            workdir: workdir.into(),
            env: Vec::new(),
        }
    }

    /// Builder: replace `args` wholesale.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Builder: replace `env` wholesale.
    #[must_use]
    pub fn with_env<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env = env.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        self
    }
}

/// Streaming subprocess sandbox. Object-safe.
///
/// `run` streams every stdout/stderr line through `on_line` as it
/// appears, returning when the process exits, the `cancel` token fires,
/// or the backend's timeout elapses. Backends MUST honor `cancel`: when
/// it fires, the subprocess (and anything it spawned) is killed, log-pump
/// tasks are aborted, and `Err(SandboxError::Cancelled)` is returned.
///
/// Backends SHOULD bound the size of each line they pass to `on_line`
/// (see [`TRUNCATION_MARKER`]); [`crate::SubprocessTool`] caps lines
/// again, but only after the backend has buffered them.
///
/// Implemented by [`crate::TokioBackend`], which is **not** isolation:
/// implement this trait over a real isolation layer to run untrusted
/// code (see the crate docs).
#[async_trait]
pub trait SandboxBackend: Send + Sync + 'static {
    /// Stable name for this backend — used in logs and traces.
    fn name(&self) -> &'static str;

    /// Execute `cmd` in the sandbox, streaming output lines through `on_line`.
    ///
    /// `on_line` is `FnMut + Send` so the backend may call it from a
    /// separate task. `cancel` is the cooperative cancel token; pass
    /// `CancellationToken::new()` for an un-cancellable run.
    ///
    /// Returns:
    /// - `Ok(())` — exited with code 0.
    /// - `Err(NonZeroExit(code))` — the code is carried structurally so
    ///   callers can distinguish exit 1 ("test failed") from exit 137
    ///   ("OOM-killed") without parsing a message.
    /// - `Err(Cancelled)` — cancel token fired.
    /// - `Err(Timeout)` — backend timeout elapsed.
    /// - `Err(Io(_) | Setup(_))` — spawn / I/O / setup failure; no
    ///   subprocess ran, or the error is recoverable on retry.
    async fn run(
        &self,
        cmd: SandboxCommand,
        on_line: &mut (dyn FnMut(LogLine) + Send),
        cancel: CancellationToken,
    ) -> Result<(), SandboxError>;
}
