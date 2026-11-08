//! `borg-sandbox`: run subprocesses from a [`borg::Tool`], through a
//! pluggable [`SandboxBackend`].
//!
//! # This is not a sandbox
//!
//! [`TokioBackend`] runs **ordinary subprocesses with your privileges**. It
//! clears the environment, kills the process group on timeout and cancel,
//! and caps output, but the child can still read and write every file you
//! can and reach the network. [`SubprocessTool`]'s program allowlist and
//! workspace root limit what the *model* can ask for; they do not contain
//! what an allowed program then does (allowlisting `sh` or `python3` means
//! the model can run anything).
//!
//! Every field of the tool input (`program`, `args`, `workdir`, `env`) is
//! chosen by the model, so a prompt injection controls all of it. **To run
//! untrusted code, implement [`SandboxBackend`] over a real isolation
//! layer**, for example:
//!
//! - [gVisor](https://gvisor.dev/) (`runsc`).
//! - [Firecracker](https://firecracker-microvm.github.io/) microVMs.
//! - [bubblewrap](https://github.com/containers/bubblewrap) or
//!   [nsjail](https://github.com/google/nsjail).
//! - Hosted sandboxes: E2B, Modal, Daytona, Cloudflare Sandbox.
//! - [agentjail](https://github.com/joydle/agentjail) (beta): lightweight
//!   Linux sandboxes for AI agents; check its own security notes before
//!   relying on it.
//!
//! A backend is one async method. This sketch wraps each command in
//! bubblewrap and reuses [`TokioBackend`] for spawning, streaming,
//! timeouts and kills:
//!
//! ```rust,no_run
//! use async_trait::async_trait;
//! use borg_sandbox::{LogLine, SandboxBackend, SandboxCommand, SandboxError, TokioBackend};
//! use tokio_util::sync::CancellationToken;
//!
//! struct Bwrap {
//!     inner: TokioBackend,
//! }
//!
//! #[async_trait]
//! impl SandboxBackend for Bwrap {
//!     fn name(&self) -> &'static str {
//!         "bwrap"
//!     }
//!
//!     async fn run(
//!         &self,
//!         cmd: SandboxCommand,
//!         on_line: &mut (dyn FnMut(LogLine) + Send),
//!         cancel: CancellationToken,
//!     ) -> Result<(), SandboxError> {
//!         let dir = cmd.workdir.to_string_lossy().into_owned();
//!         let mut args: Vec<String> = [
//!             "--unshare-all", "--die-with-parent", "--new-session",
//!             "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
//!             "--symlink", "usr/lib", "/lib", "--proc", "/proc", "--dev", "/dev",
//!             "--tmpfs", "/tmp", "--bind", &dir, &dir, "--chdir", &dir, "--",
//!         ]
//!         .iter()
//!         .map(|s| s.to_string())
//!         .collect();
//!         args.push(cmd.program.clone());
//!         args.extend(cmd.args.iter().cloned());
//!         let wrapped = SandboxCommand::new("/usr/bin/bwrap", &cmd.workdir)
//!             .with_args(args)
//!             .with_env(cmd.env.clone());
//!         self.inner.run(wrapped, on_line, cancel).await
//!     }
//! }
//! ```
//!
//! # What's here
//!
//! - [`SandboxBackend`]: streaming subprocess trait.
//! - [`TokioBackend`]: plain-subprocess backend (not isolation; see above).
//! - [`SubprocessTool`]: wraps any `SandboxBackend` as a `borg::Tool`.
//!   Built with [`SubprocessTool::builder`], which requires a program
//!   allowlist and a workspace root; [`SubprocessTool::unrestricted`] is
//!   the explicitly dangerous escape hatch.
//! - [`SandboxContext`]: per-call context for `SubprocessTool`.
//!
//! # Platform notes
//!
//! Process-group kill (grandchildren die on timeout, cancel and drop) is
//! Unix-only; on Windows only the direct child is killed. The crate's
//! tests drive `sh`, `sleep` and `/dev/zero`, so most of them are
//! Unix-only and Windows behaviour is far less exercised.

pub mod backend;
pub mod context;
pub mod tokio_backend;
pub mod tool;

pub use backend::{LogLine, SandboxBackend, SandboxCommand, SandboxError, TRUNCATION_MARKER};
pub use context::SandboxContext;
pub use tokio_backend::{TokioBackend, DEFAULT_ENV_PASSTHROUGH};
pub use tool::{
    ConfigError, SubprocessTool, SubprocessToolBuilder, BLOCKED_ENV, BLOCKED_ENV_PREFIXES,
};

// The unit tests drive real `sh`/`echo` processes and Unix paths.
#[cfg(all(test, unix))]
mod tests;
