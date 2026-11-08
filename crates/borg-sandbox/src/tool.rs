use crate::backend::{LogLine, SandboxBackend, SandboxCommand, SandboxError, TRUNCATION_MARKER};
use crate::context::SandboxContext;
use async_trait::async_trait;
use borg::{BorgError, Tool, ToolCallResult, ToolSchema};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Env var names a model may never set, even when the caller allowlisted
/// them: they load code (shared libraries, startup scripts, interpreter
/// options) into whatever program runs, redirect program lookup, or move
/// where programs read their config from (`HOME`, and so `~/.gitconfig`,
/// `~/.cargo/config.toml`, shell rc files). Matched case-insensitively;
/// see also [`BLOCKED_ENV_PREFIXES`].
///
/// The backend may still pass the parent's `HOME` through to the child
/// ([`crate::TokioBackend`] does); the model just can't choose it.
pub const BLOCKED_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "RUSTFLAGS",
    "RUSTDOCFLAGS",
    "RUSTDOC",
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "PS4",
    "PROMPT_COMMAND",
    "IFS",
    "NODE_OPTIONS",
    "NODE_PATH",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    "PERL5OPT",
    "PERL5LIB",
    "PERLLIB",
    "RUBYOPT",
    "RUBYLIB",
    "JAVA_TOOL_OPTIONS",
    "_JAVA_OPTIONS",
    "JDK_JAVA_OPTIONS",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_EXEC_PATH",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
    "EDITOR",
    "VISUAL",
    "PAGER",
];

/// Env var name prefixes a model may never set (case-insensitive):
/// dynamic-loader injection (`LD_*`, `DYLD_*`), every git setting
/// (`GIT_*`: config injection, `GIT_SSH_COMMAND`, `GIT_EXTERNAL_DIFF`, ...),
/// cargo and rustc settings that run programs (`CARGO_*`, `RUSTC*`:
/// `CARGO_TARGET_<triple>_RUNNER`, `RUSTC_WRAPPER`, ...), XDG base
/// directories (`XDG_*`, which relocate config files), and exported bash
/// functions (`BASH_FUNC_*`).
pub const BLOCKED_ENV_PREFIXES: &[&str] = &[
    "LD_",
    "DYLD_",
    "GIT_",
    "CARGO_",
    "RUSTC",
    "XDG_",
    "BASH_FUNC_",
];

/// Default cap on combined stdout + stderr bytes kept per call: 1 MiB.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 1 << 20;

/// Default cap on a single line kept per call: 64 KiB.
pub const DEFAULT_MAX_LINE_BYTES: usize = 64 * 1024;

pub(crate) fn is_blocked_env(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    BLOCKED_ENV.iter().any(|b| *b == upper)
        || BLOCKED_ENV_PREFIXES.iter().any(|p| upper.starts_with(p))
        || key.is_empty()
        || key.contains('=')
        || key.contains('\0')
}

/// Error from [`SubprocessToolBuilder::build`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// No program was allowlisted.
    #[error("no programs allowlisted; call allow_program(s)")]
    NoPrograms,
    /// No workspace root was set.
    #[error("no workspace root; call workspace_root(..)")]
    NoWorkspaceRoot,
    /// The workspace root does not exist or is not a directory.
    #[error("workspace root '{0}' does not exist or is not a directory")]
    BadWorkspaceRoot(PathBuf),
    /// An allowlisted program name was not found on the search path.
    #[error("allowlisted program '{0}' was not found on the search path")]
    ProgramNotFound(String),
    /// An allowlisted program contains a path separator but is not absolute.
    #[error("allowlisted program '{0}' must be a bare name or an absolute path")]
    RelativeProgramPath(String),
    /// An env key the model may never set was allowlisted.
    #[error("env key '{0}' cannot be allowlisted for the model (see BLOCKED_ENV)")]
    BlockedEnvKey(String),
}

/// Which programs the model may run.
enum Programs {
    /// [`SubprocessTool::unrestricted`]: anything.
    Any,
    /// Bare names → their resolved absolute path; plus every allowed
    /// absolute path (caller-given and resolved).
    Allow {
        names: HashMap<String, PathBuf>,
        paths: HashSet<PathBuf>,
    },
}

/// The policy a [`SubprocessTool`] enforces on model input.
struct Policy {
    programs: Programs,
    /// Canonical workspace root; `None` only when unrestricted.
    workspace_root: Option<PathBuf>,
    /// Workdir used when the model gives none.
    default_workdir: PathBuf,
    /// Env keys the model may set; `None` = any non-blocked key.
    allowed_env: Option<HashSet<String>>,
    /// Caller-fixed env, applied after (and over) model env.
    fixed_env: Vec<(String, String)>,
    max_output_bytes: usize,
    max_line_bytes: usize,
}

/// A `borg::Tool` that runs a subprocess via a [`SandboxBackend`].
///
/// Every field of the tool input (`program`, `args`, `workdir`, `env`) is
/// chosen by the model, so treat it as attacker-controlled (prompt
/// injection). Build with [`SubprocessTool::builder`], which requires a
/// program allowlist and a workspace root:
///
/// ```rust,no_run
/// # use borg_sandbox::{SubprocessTool, TokioBackend};
/// let tool = SubprocessTool::builder("run", "Run a build command", TokioBackend::new())
///     .allow_programs(["cargo", "git"])
///     .workspace_root("/srv/agent-workspace")
///     .build()?;
/// # Ok::<(), borg_sandbox::ConfigError>(())
/// ```
///
/// Note that allowlisting a shell or an interpreter (`sh`, `bash`,
/// `python3`, `node`) lets the model run anything that program can.
///
/// Input: `program` (required), `args`, `workdir`, `env` (see `schema()`).
/// Success output: `{ "exit_code", "stdout", "stderr", "truncated" }`.
/// Invalid model input, disallowed programs, non-zero exits, timeouts and
/// spawn failures all surface as `ToolCallResult::fail` (an error the model
/// sees), never as `Err(BorgError)`. A non-zero exit's failure text carries
/// the exit code and the (capped) stdout and stderr.
///
/// Once the output passes [`max_output_bytes`](SubprocessToolBuilder::max_output_bytes)
/// the run is stopped (the backend kills the process and its group) and
/// the output kept so far comes back with `"exit_code": null`,
/// `"truncated": true` and a `"note"` saying so.
pub struct SubprocessTool<B: SandboxBackend> {
    name: &'static str,
    description: &'static str,
    backend: Arc<B>,
    policy: Policy,
}

/// Builder for a restricted [`SubprocessTool`]; see
/// [`SubprocessTool::builder`].
pub struct SubprocessToolBuilder<B: SandboxBackend> {
    name: &'static str,
    description: &'static str,
    backend: B,
    programs: Vec<String>,
    workspace_root: Option<PathBuf>,
    search_path: Option<OsString>,
    allowed_env: HashSet<String>,
    fixed_env: Vec<(String, String)>,
    max_output_bytes: usize,
    max_line_bytes: usize,
}

impl<B: SandboxBackend> SubprocessTool<B> {
    /// Start building a restricted tool. [`build`](SubprocessToolBuilder::build)
    /// fails unless at least one program is allowlisted and a workspace
    /// root is set.
    pub fn builder(
        name: &'static str,
        description: &'static str,
        backend: B,
    ) -> SubprocessToolBuilder<B> {
        SubprocessToolBuilder {
            name,
            description,
            backend,
            programs: Vec::new(),
            workspace_root: None,
            search_path: None,
            allowed_env: HashSet::new(),
            fixed_env: Vec::new(),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
        }
    }

    /// **Dangerous.** A tool with no program allowlist and no workspace
    /// root: the model may run any program, in any directory (default
    /// [`std::env::temp_dir`]), with any env var outside
    /// [`BLOCKED_ENV`] / [`BLOCKED_ENV_PREFIXES`].
    ///
    /// With [`crate::TokioBackend`] that is arbitrary code execution with
    /// your privileges. Use only with a backend that provides real
    /// isolation, or for local experiments with a trusted model.
    pub fn unrestricted(name: &'static str, description: &'static str, backend: B) -> Self {
        Self {
            name,
            description,
            backend: Arc::new(backend),
            policy: Policy {
                programs: Programs::Any,
                workspace_root: None,
                default_workdir: std::env::temp_dir(),
                allowed_env: None,
                fixed_env: Vec::new(),
                max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
                max_line_bytes: DEFAULT_MAX_LINE_BYTES,
            },
        }
    }
}

impl<B: SandboxBackend> SubprocessToolBuilder<B> {
    /// Allow one program: either a bare name (resolved to an absolute path
    /// on the [search path](Self::search_path) at build time) or an
    /// absolute path, allowed exactly.
    #[must_use]
    pub fn allow_program(mut self, program: impl Into<String>) -> Self {
        self.programs.push(program.into());
        self
    }

    /// Allow several programs; see [`allow_program`](Self::allow_program).
    #[must_use]
    pub fn allow_programs<I, S>(mut self, programs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.programs.extend(programs.into_iter().map(Into::into));
        self
    }

    /// Required. The model's `workdir` must resolve (after
    /// canonicalization) inside this directory; a relative `workdir` is
    /// taken relative to it, and an omitted one defaults to it.
    #[must_use]
    pub fn workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// The `PATH`-style list used to resolve allowlisted bare names.
    /// Defaults to this process's `PATH` at build time. When set, it is
    /// also passed to the child as `PATH`. Relative entries are ignored.
    /// The model can never influence program lookup.
    #[must_use]
    pub fn search_path(mut self, path: impl Into<OsString>) -> Self {
        self.search_path = Some(path.into());
        self
    }

    /// Env keys the model may set through the `env` input. By default the
    /// model's `env` is ignored entirely. Keys in [`BLOCKED_ENV`] or
    /// matching [`BLOCKED_ENV_PREFIXES`] are rejected at build time.
    #[must_use]
    pub fn allow_env<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_env.extend(keys.into_iter().map(Into::into));
        self
    }

    /// A caller-fixed env var for every run. Applied after (and over) the
    /// model's env; not subject to the blocklist.
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fixed_env.push((key.into(), value.into()));
        self
    }

    /// Cap on combined stdout + stderr bytes kept per call (default 1 MiB).
    /// Output beyond it is dropped, the process is stopped, and
    /// `"truncated": true` is reported.
    #[must_use]
    pub fn max_output_bytes(mut self, n: usize) -> Self {
        self.max_output_bytes = n;
        self
    }

    /// Cap on a single kept line (default 64 KiB). Longer lines are cut
    /// and end with [`TRUNCATION_MARKER`].
    #[must_use]
    pub fn max_line_bytes(mut self, n: usize) -> Self {
        self.max_line_bytes = n.max(1);
        self
    }

    /// Validate and build. Resolves every bare program name against the
    /// search path now, so a program installed later is not picked up.
    pub fn build(self) -> Result<SubprocessTool<B>, ConfigError> {
        if self.programs.is_empty() {
            return Err(ConfigError::NoPrograms);
        }
        let root = self.workspace_root.ok_or(ConfigError::NoWorkspaceRoot)?;
        let root = root
            .canonicalize()
            .ok()
            .filter(|p| p.is_dir())
            .ok_or(ConfigError::BadWorkspaceRoot(root))?;

        if let Some(key) = self.allowed_env.iter().find(|k| is_blocked_env(k)) {
            return Err(ConfigError::BlockedEnvKey(key.clone()));
        }

        let search_path = self
            .search_path
            .clone()
            .or_else(|| std::env::var_os("PATH"))
            .unwrap_or_default();
        let mut names = HashMap::new();
        let mut paths = HashSet::new();
        for program in self.programs {
            if has_separator(&program) {
                let path = PathBuf::from(&program);
                if !path.is_absolute() {
                    return Err(ConfigError::RelativeProgramPath(program));
                }
                paths.insert(path);
            } else {
                let resolved = resolve_program(&program, &search_path)
                    .ok_or_else(|| ConfigError::ProgramNotFound(program.clone()))?;
                paths.insert(resolved.clone());
                names.insert(program, resolved);
            }
        }

        let mut fixed_env = Vec::new();
        if let Some(p) = self.search_path {
            fixed_env.push(("PATH".to_string(), p.to_string_lossy().into_owned()));
        }
        fixed_env.extend(self.fixed_env);

        Ok(SubprocessTool {
            name: self.name,
            description: self.description,
            backend: Arc::new(self.backend),
            policy: Policy {
                programs: Programs::Allow { names, paths },
                default_workdir: root.clone(),
                workspace_root: Some(root),
                allowed_env: Some(self.allowed_env),
                fixed_env,
                max_output_bytes: self.max_output_bytes,
                max_line_bytes: self.max_line_bytes,
            },
        })
    }
}

fn has_separator(s: &str) -> bool {
    s.chars().any(std::path::is_separator)
}

/// Find `name` in the absolute entries of `search_path`.
fn resolve_program(name: &str, search_path: &OsString) -> Option<PathBuf> {
    #[cfg(windows)]
    let exts: Vec<String> = {
        let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
        std::iter::once(String::new())
            .chain(
                pathext
                    .split(';')
                    .filter(|e| !e.is_empty())
                    .map(str::to_string),
            )
            .collect()
    };
    #[cfg(not(windows))]
    let exts: Vec<String> = vec![String::new()];

    std::env::split_paths(search_path)
        .filter(|dir| dir.is_absolute())
        .flat_map(|dir| {
            exts.iter()
                .map(move |ext| dir.join(format!("{name}{ext}")))
                .collect::<Vec<_>>()
        })
        .find(|p| is_executable(p))
}

fn is_executable(p: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(p) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.is_file() && meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        meta.is_file()
    }
}

/// Cut `s` to at most `max` bytes on a char boundary.
fn cut(s: &mut String, max: usize) {
    if s.len() > max {
        let mut i = max;
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        s.truncate(i);
    }
}

impl<B: SandboxBackend> SubprocessTool<B> {
    /// Validate model input and turn it into a command, or a message for
    /// the model explaining what was rejected.
    fn prepare(&self, input: &serde_json::Value) -> Result<SandboxCommand, String> {
        let p = &self.policy;
        let obj = input
            .as_object()
            .ok_or("input must be a JSON object".to_string())?;

        // ── program ───────────────────────────────────────────────────────
        let program = match obj.get("program") {
            Some(serde_json::Value::String(s)) if !s.is_empty() => s.clone(),
            Some(_) => return Err("'program' must be a non-empty string".into()),
            None => return Err("missing required field 'program'".into()),
        };
        let program = match &p.programs {
            Programs::Any => program,
            Programs::Allow { names, paths } => {
                let resolved = if has_separator(&program) {
                    let path = PathBuf::from(&program);
                    paths.contains(&path).then_some(path)
                } else {
                    names.get(&program).cloned()
                };
                match resolved {
                    Some(path) => path.to_string_lossy().into_owned(),
                    None => {
                        let mut allowed: Vec<&str> = names.keys().map(String::as_str).collect();
                        allowed.sort_unstable();
                        return Err(format!(
                            "program '{program}' is not in the allowed list (allowed: {})",
                            allowed.join(", ")
                        ));
                    }
                }
            }
        };

        // ── args ──────────────────────────────────────────────────────────
        let args: Vec<String> = match obj.get("args") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or("'args' must be an array of strings")?,
            Some(_) => return Err("'args' must be an array of strings".into()),
        };
        if args.iter().any(|a| a.contains('\0')) {
            return Err("'args' must not contain NUL bytes".into());
        }

        // ── workdir ───────────────────────────────────────────────────────
        let workdir = match obj.get("workdir") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(PathBuf::from(s)),
            Some(_) => return Err("'workdir' must be a string".into()),
        };
        let workdir = match &p.workspace_root {
            None => workdir.unwrap_or_else(|| p.default_workdir.clone()),
            Some(root) => {
                let requested = match workdir {
                    None => root.clone(),
                    Some(w) if w.is_relative() => root.join(w),
                    Some(w) => w,
                };
                // Canonicalize first: a raw path lets `..` escape, since
                // `/workspace/x/../../etc` starts_with `/workspace`
                // component-wise but resolves to `/etc`.
                let canonical = requested.canonicalize().map_err(|_| {
                    format!(
                        "workdir '{}' does not exist or is not accessible",
                        requested.display()
                    )
                })?;
                if !canonical.starts_with(root) {
                    return Err(format!(
                        "workdir '{}' is outside workspace root '{}'",
                        requested.display(),
                        root.display()
                    ));
                }
                canonical
            }
        };

        // ── env ───────────────────────────────────────────────────────────
        let mut env: Vec<(String, String)> = match obj.get("env") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::Object(map)) => {
                let mut env = Vec::new();
                for (k, v) in map {
                    let v = v
                        .as_str()
                        .ok_or_else(|| format!("env value for '{k}' must be a string"))?;
                    let allowed = p.allowed_env.as_ref().is_none_or(|a| a.contains(k));
                    if allowed && !is_blocked_env(k) && !v.contains('\0') {
                        env.push((k.clone(), v.to_string()));
                    } else {
                        tracing::debug!(key = %k, "ignoring model-supplied env var");
                    }
                }
                env
            }
            Some(_) => return Err("'env' must be an object of strings".into()),
        };
        env.extend(p.fixed_env.iter().cloned());

        Ok(SandboxCommand::new(program, workdir)
            .with_args(args)
            .with_env(env))
    }
}

#[async_trait]
impl<B: SandboxBackend + 'static> Tool for SubprocessTool<B> {
    type Context = SandboxContext;

    fn schema(&self) -> ToolSchema {
        let program_desc = match &self.policy.programs {
            Programs::Any => "Executable to run".to_string(),
            Programs::Allow { names, .. } => {
                let mut allowed: Vec<&str> = names.keys().map(String::as_str).collect();
                allowed.sort_unstable();
                format!("Executable to run; one of: {}", allowed.join(", "))
            }
        };
        let workdir_desc = if self.policy.workspace_root.is_some() {
            "Working directory, inside the workspace (relative paths are relative to it)"
        } else {
            "Working directory"
        };
        let mut properties = serde_json::json!({
            "program": { "type": "string", "description": program_desc },
            "args": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Arguments (not including program name)"
            },
            "workdir": { "type": "string", "description": workdir_desc },
        });
        let env_keys = match &self.policy.allowed_env {
            None => Some("Environment variable overrides".to_string()),
            Some(keys) if keys.is_empty() => None,
            Some(keys) => {
                let mut keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                keys.sort_unstable();
                Some(format!(
                    "Environment variables; only these keys are honoured: {}",
                    keys.join(", ")
                ))
            }
        };
        if let Some(desc) = env_keys {
            properties["env"] = serde_json::json!({
                "type": "object",
                "additionalProperties": { "type": "string" },
                "description": desc
            });
        }
        ToolSchema::new(
            self.name.to_string(),
            self.description.to_string(),
            serde_json::json!({
                "type": "object",
                "required": ["program"],
                "properties": properties
            }),
        )
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &mut SandboxContext,
    ) -> Result<ToolCallResult, BorgError> {
        // `ctx.lines` holds the most recent call's lines only.
        ctx.lines.clear();

        let cmd = match self.prepare(&input) {
            Ok(cmd) => cmd,
            Err(msg) => return Ok(ToolCallResult::fail(msg)),
        };

        let mut stdout_lines: Vec<String> = vec![];
        let mut stderr_lines: Vec<String> = vec![];
        let mut total_bytes: usize = 0;
        let mut truncated = false;
        let max_bytes = self.policy.max_output_bytes;
        let max_line = self.policy.max_line_bytes;

        // Defaults to a never-fired token unless the caller wired one in
        // via SandboxContext::with_cancel. The run gets a child token so
        // hitting the output cap can stop it without touching the caller's.
        let cancel = ctx.cancel.clone();
        let run_cancel = cancel.child_token();
        let mut hit_cap = false;

        let result = self
            .backend
            .run(
                cmd,
                &mut |line| {
                    // ── output caps: per line, then total ─────────────────
                    // Output past the total cap stops the run: nothing
                    // more would be kept, so there is no point letting
                    // the process run (or spin) until the timeout.
                    if total_bytes >= max_bytes {
                        truncated = true;
                        hit_cap = true;
                        run_cancel.cancel();
                        return;
                    }
                    let (mut s, is_stdout) = match line {
                        LogLine::Stdout(s) => (s, true),
                        LogLine::Stderr(s) => (s, false),
                    };
                    let room = max_bytes - total_bytes;
                    if s.len() > room {
                        hit_cap = true;
                        run_cancel.cancel();
                    }
                    let budget = max_line.min(room);
                    if s.len() > budget {
                        cut(&mut s, budget);
                        s.push_str(TRUNCATION_MARKER);
                        truncated = true;
                    } else if s.ends_with(TRUNCATION_MARKER) {
                        // Already cut by the backend.
                        truncated = true;
                    }
                    // +1 for the newline: also bounds a flood of empty lines.
                    total_bytes = total_bytes.saturating_add(s.len() + 1);
                    let line = if is_stdout {
                        LogLine::Stdout(s.clone())
                    } else {
                        LogLine::Stderr(s.clone())
                    };
                    if let Some(cb) = ctx.on_line.as_mut() {
                        cb(line.clone());
                    }
                    ctx.lines.push(line);
                    if is_stdout {
                        stdout_lines.push(s);
                    } else {
                        stderr_lines.push(s);
                    }
                },
                run_cancel.clone(),
            )
            .await;

        let stdout = stdout_lines.join("\n");
        let stderr = stderr_lines.join("\n");
        match result {
            Ok(()) => ToolCallResult::ok(serde_json::json!({
                "exit_code": 0,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            })),
            // Our own stop at the output cap, not the caller's cancel.
            Err(SandboxError::Cancelled) if hit_cap && !cancel.is_cancelled() => {
                ToolCallResult::ok(serde_json::json!({
                    "exit_code": serde_json::Value::Null,
                    "stdout": stdout,
                    "stderr": stderr,
                    "truncated": true,
                    "note": format!(
                        "output passed the {max_bytes}-byte limit; the process was stopped"
                    ),
                }))
            }
            Err(SandboxError::NonZeroExit(code)) => Ok(ToolCallResult::fail(nonzero_exit_message(
                code, &stdout, &stderr, truncated,
            ))),
            Err(e) => Ok(ToolCallResult::fail(format!("sandbox error: {e}"))),
        }
    }
}

/// The failure text for a non-zero exit: the code plus everything kept of
/// stdout and stderr (already within the output caps).
fn nonzero_exit_message(code: i32, stdout: &str, stderr: &str, truncated: bool) -> String {
    let section = |s: &str| {
        if s.is_empty() {
            "(empty)".to_string()
        } else {
            s.to_string()
        }
    };
    let mut msg = format!(
        "exit {code}\nstdout:\n{}\nstderr:\n{}",
        section(stdout),
        section(stderr)
    );
    if truncated {
        msg.push_str("\n[output truncated]");
    }
    msg
}
