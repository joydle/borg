use crate::backend::{LogLine, SandboxBackend, SandboxCommand, SandboxError, TRUNCATION_MARKER};
use async_trait::async_trait;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// Parent environment variables [`TokioBackend`] copies into the child by
/// default. Everything else in the parent's environment (API keys, cloud
/// credentials, tokens) is **not** inherited.
#[cfg(unix)]
pub const DEFAULT_ENV_PASSTHROUGH: &[&str] = &["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM"];

/// Parent environment variables [`TokioBackend`] copies into the child by
/// default. Everything else in the parent's environment (API keys, cloud
/// credentials, tokens) is **not** inherited.
#[cfg(not(unix))]
pub const DEFAULT_ENV_PASSTHROUGH: &[&str] = &[
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TERM",
    "SystemRoot",
    "SystemDrive",
    "windir",
    "ComSpec",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
];

/// Default per-line cap for [`TokioBackend`]: 64 KiB.
pub const DEFAULT_MAX_LINE_BYTES: usize = 64 * 1024;

/// Bytes read from a pipe per `read` call.
const READ_CHUNK: usize = 8 * 1024;

/// Subprocess backend over `tokio::process::Command`.
///
/// # This is not a sandbox
///
/// `TokioBackend` runs an ordinary subprocess **with your privileges**: it
/// can read and write every file you can, open network connections, and
/// so on. What it does do:
///
/// - **Clears the environment.** The child starts from an empty env plus
///   the passthrough list ([`DEFAULT_ENV_PASSTHROUGH`] unless replaced via
///   [`with_env_passthrough`](Self::with_env_passthrough)), then the
///   command's own `env`.
/// - **Kills the whole process tree it can see.** On Unix the child runs
///   in its own process group, and the group is sent `SIGKILL` on timeout,
///   on cancel, when the `run` future is dropped, and once the direct
///   child exits (so `sh -c "sleep 1d & echo hi"` returns as soon as `sh`
///   does). A descendant that calls `setsid` escapes the group; output
///   draining is still bounded by the timeout and the cancel token.
///   On Windows only the direct child is killed (no job object yet), so
///   grandchildren can survive a timeout or cancel.
/// - **Bounds memory.** Pipes are read in fixed-size chunks and each line
///   is capped at [`max_line_bytes`](Self::with_max_line_bytes) (excess is
///   dropped and the line ends with [`TRUNCATION_MARKER`]).
/// - **Times out.** Default 30 s.
#[derive(Clone)]
pub struct TokioBackend {
    timeout: Option<Duration>,
    env_passthrough: Vec<String>,
    max_line_bytes: usize,
}

impl TokioBackend {
    /// Default: 30-second timeout, [`DEFAULT_ENV_PASSTHROUGH`],
    /// [`DEFAULT_MAX_LINE_BYTES`].
    pub fn new() -> Self {
        Self {
            timeout: Some(Duration::from_secs(30)),
            env_passthrough: DEFAULT_ENV_PASSTHROUGH
                .iter()
                .map(|s| s.to_string())
                .collect(),
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
        }
    }

    /// Explicit timeout. The process group is killed and
    /// `SandboxError::Timeout` returned on expiry.
    pub fn with_timeout(d: Duration) -> Self {
        Self {
            timeout: Some(d),
            ..Self::new()
        }
    }

    /// No timeout. Only for benchmarks or tests that intentionally run long.
    pub fn without_timeout() -> Self {
        Self {
            timeout: None,
            ..Self::new()
        }
    }

    /// Replace the list of parent env vars copied into the child. Pass an
    /// empty list for a fully empty environment (then most programs need
    /// at least `PATH` set through [`SandboxCommand::env`]).
    #[must_use]
    pub fn with_env_passthrough<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.env_passthrough = keys.into_iter().map(Into::into).collect();
        self
    }

    /// Cap each emitted line at `n` bytes (minimum 1); excess bytes are
    /// discarded and the line ends with [`TRUNCATION_MARKER`].
    #[must_use]
    pub fn with_max_line_bytes(mut self, n: usize) -> Self {
        self.max_line_bytes = n.max(1);
        self
    }
}

impl Default for TokioBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Kills the child's process group when dropped (Unix), so a dropped
/// `run` future takes its grandchildren with it. Disarmed once the group
/// has been killed after the leader exits: after the leader is reaped its
/// pid may be reused, so a late `killpg` could hit an unrelated group.
struct GroupKill {
    #[cfg(unix)]
    pgid: Option<libc::pid_t>,
}

impl GroupKill {
    fn new(_pid: Option<u32>) -> Self {
        Self {
            #[cfg(unix)]
            pgid: _pid.and_then(|p| libc::pid_t::try_from(p).ok()),
        }
    }

    /// SIGKILL the whole group (Unix). No-op on other platforms.
    fn kill(&self) {
        #[cfg(unix)]
        if let Some(pgid) = self.pgid {
            // SAFETY: killpg has no memory-safety preconditions. The group
            // id is the leader's pid, which cannot be reused while the
            // leader is unreaped or any group member is alive.
            unsafe {
                libc::killpg(pgid, libc::SIGKILL);
            }
        }
    }

    fn disarm(&mut self) {
        #[cfg(unix)]
        {
            self.pgid = None;
        }
    }
}

impl Drop for GroupKill {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Read `reader` in fixed chunks, split on `\n`, cap each line at
/// `max_line` bytes, and send lines through `tx`.
async fn pump<R: AsyncRead + Unpin>(
    mut reader: R,
    max_line: usize,
    tx: mpsc::Sender<LogLine>,
    wrap: fn(String) -> LogLine,
) {
    let mut buf = vec![0u8; READ_CHUNK];
    let mut line: Vec<u8> = Vec::new();
    let mut overflow = false;

    fn push(line: &mut Vec<u8>, bytes: &[u8], max: usize, overflow: &mut bool) {
        let room = max.saturating_sub(line.len());
        if bytes.len() > room {
            *overflow = true;
        }
        line.extend_from_slice(&bytes[..bytes.len().min(room)]);
    }

    fn finish(line: &mut Vec<u8>, overflow: &mut bool) -> String {
        if !*overflow && line.last() == Some(&b'\r') {
            line.pop();
        }
        let mut s = String::from_utf8_lossy(line).into_owned();
        if *overflow {
            s.push_str(TRUNCATION_MARKER);
        }
        line.clear();
        *overflow = false;
        s
    }

    loop {
        let n = match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut chunk = &buf[..n];
        while let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
            push(&mut line, &chunk[..pos], max_line, &mut overflow);
            if tx
                .send(wrap(finish(&mut line, &mut overflow)))
                .await
                .is_err()
            {
                return;
            }
            chunk = &chunk[pos + 1..];
        }
        push(&mut line, chunk, max_line, &mut overflow);
    }
    if !line.is_empty() || overflow {
        let _ = tx.send(wrap(finish(&mut line, &mut overflow))).await;
    }
}

#[async_trait]
impl SandboxBackend for TokioBackend {
    fn name(&self) -> &'static str {
        "tokio"
    }

    async fn run(
        &self,
        cmd: SandboxCommand,
        on_line: &mut (dyn FnMut(LogLine) + Send),
        cancel: CancellationToken,
    ) -> Result<(), SandboxError> {
        let mut command = Command::new(&cmd.program);
        command
            .args(&cmd.args)
            .current_dir(&cmd.workdir)
            .env_clear()
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        for key in &self.env_passthrough {
            if let Some(v) = std::env::var_os(key) {
                command.env(key, v);
            }
        }
        command.envs(cmd.env);
        // Own process group, so timeout / cancel / drop can kill the
        // whole tree rather than only the direct child.
        #[cfg(unix)]
        command.process_group(0);

        let mut child = command.spawn().map_err(SandboxError::Io)?;
        let mut group = GroupKill::new(child.id());

        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let (tx, mut rx) = mpsc::channel::<LogLine>(64);
        let tx_err = tx.clone();

        // Pumps live on a `JoinSet` owned by this future, so dropping the
        // future aborts them; `pumps.abort_all()` on cancel/timeout just
        // makes that immediate. `.in_current_span()` carries the caller's
        // tracing context across the spawn — tracing spans don't propagate
        // into spawned tasks otherwise.
        let max_line = self.max_line_bytes;
        let mut pumps = JoinSet::new();
        pumps.spawn(pump(stdout, max_line, tx, LogLine::Stdout).in_current_span());
        pumps.spawn(pump(stderr, max_line, tx_err, LogLine::Stderr).in_current_span());

        // One deadline covers the run AND the output drain: a grandchild
        // holding the pipe open cannot extend it.
        let deadline = async {
            match self.timeout {
                Some(d) => tokio::time::sleep(d).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline);

        let mut status = None;
        let mut output_done = false;
        while status.is_none() || !output_done {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    group.kill();
                    let _ = child.kill().await;
                    pumps.abort_all();
                    return Err(SandboxError::Cancelled);
                }
                _ = &mut deadline => {
                    group.kill();
                    let _ = child.kill().await;
                    pumps.abort_all();
                    return Err(SandboxError::Timeout);
                }
                line = rx.recv(), if !output_done => match line {
                    Some(l) => on_line(l),
                    None => output_done = true,
                },
                s = child.wait(), if status.is_none() => {
                    let s = s.map_err(SandboxError::Io)?;
                    // The leader has exited (zombie reaped just now, but
                    // the group id stays reserved while any member lives):
                    // kill leftover background processes so they neither
                    // outlive the call nor hold the pipes open.
                    group.kill();
                    group.disarm();
                    status = Some(s);
                }
            }
        }

        drop(pumps);

        let code = status.and_then(|s| s.code()).unwrap_or(-1);
        if code == 0 {
            Ok(())
        } else {
            Err(SandboxError::NonZeroExit(code))
        }
    }
}
