//! Integration tests for `borg-sandbox` subprocess execution.
//!
//! Tests without a `cfg` gate drive `sh` / `echo` from `PATH` and also run
//! on the Windows CI runner (Git for Windows provides both). Tests gated on
//! `cfg(unix)` cover process groups, `/dev/zero`, Unix paths and similar;
//! there is no Windows equivalent for those yet.

use borg::Tool;
use borg_sandbox::{
    ConfigError, LogLine, SandboxBackend, SandboxCommand, SandboxContext, SandboxError,
    SubprocessTool, TokioBackend, TRUNCATION_MARKER,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

fn workdir() -> PathBuf {
    std::env::temp_dir()
}

fn stdout_of(lines: &[LogLine]) -> Vec<&str> {
    lines
        .iter()
        .filter_map(|l| match l {
            LogLine::Stdout(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

/// A restricted tool over a fresh temp workspace.
fn tool_in(root: &tempfile::TempDir, programs: &[&str]) -> SubprocessTool<TokioBackend> {
    SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_programs(programs.iter().copied())
        .workspace_root(root.path())
        .build()
        .unwrap()
}

async fn run_tool(
    tool: &SubprocessTool<TokioBackend>,
    input: serde_json::Value,
) -> borg::ToolCallResult {
    let mut ctx = SandboxContext::new();
    tool.execute(input, &mut ctx).await.unwrap()
}

fn out_stdout(r: &borg::ToolCallResult) -> String {
    r.output
        .as_ref()
        .unwrap_or_else(|| panic!("expected output, got {:?}", r.error))["stdout"]
        .as_str()
        .unwrap()
        .to_string()
}

// ── SandboxBackend trait (TokioBackend) ──────────────────────────────────────

#[tokio::test]
async fn tokio_backend_captures_stdout_lines() {
    let backend = TokioBackend::new();
    let cmd = SandboxCommand::new("echo", workdir()).with_args(["hello", "from", "borg-sandbox"]);

    let mut lines: Vec<LogLine> = vec![];
    backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await
        .unwrap();

    let stdout = stdout_of(&lines);
    assert!(
        stdout.iter().any(|s| s.contains("borg-sandbox")),
        "expected stdout containing 'borg-sandbox', got {stdout:?}"
    );
}

#[tokio::test]
async fn tokio_backend_captures_stderr_lines() {
    let backend = TokioBackend::new();
    // `sh -c` writes to stderr via >&2
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "echo stderr-line >&2"]);

    let mut lines: Vec<LogLine> = vec![];
    backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await
        .unwrap();

    let stderr: Vec<_> = lines
        .iter()
        .filter_map(|l| match l {
            LogLine::Stderr(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        stderr.iter().any(|s| s.contains("stderr-line")),
        "expected stderr containing 'stderr-line', got {stderr:?}"
    );
}

#[tokio::test]
async fn tokio_backend_non_zero_exit_is_err() {
    let backend = TokioBackend::new();
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "exit 7"]);

    let mut lines = vec![];
    let result = backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await;
    assert!(
        matches!(result, Err(SandboxError::NonZeroExit(7))),
        "expected NonZeroExit(7), got {result:?}"
    );
}

#[tokio::test]
async fn tokio_backend_command_env_is_applied() {
    let backend = TokioBackend::new();
    let cmd = SandboxCommand::new("sh", workdir())
        .with_args(["-c", "echo $MY_VAR"])
        .with_env([("MY_VAR", "borg_test_value")]);

    let mut lines = vec![];
    backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await
        .unwrap();

    let stdout = stdout_of(&lines);
    assert!(
        stdout.iter().any(|s| s.contains("borg_test_value")),
        "expected env var in stdout, got {stdout:?}"
    );
}

/// The parent's environment (API keys, tokens) must not reach the child.
/// `cargo test` sets `CARGO_MANIFEST_DIR` in this process's environment,
/// and it is not on the default passthrough list.
#[cfg(unix)]
#[tokio::test]
async fn tokio_backend_does_not_inherit_parent_env() {
    let Some(expected) = std::env::var_os("CARGO_MANIFEST_DIR") else {
        eprintln!("CARGO_MANIFEST_DIR not set; not running under cargo, skipping");
        return;
    };
    let expected = expected.to_string_lossy().into_owned();
    let probe = "echo M=${CARGO_MANIFEST_DIR:-UNSET}";

    let mut lines = vec![];
    TokioBackend::new()
        .run(
            SandboxCommand::new("sh", workdir()).with_args(["-c", probe]),
            &mut |l| lines.push(l),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(stdout_of(&lines), ["M=UNSET"], "parent env leaked");

    // Opting a key into the passthrough list copies it.
    let mut lines = vec![];
    TokioBackend::new()
        .with_env_passthrough(["PATH", "CARGO_MANIFEST_DIR"])
        .run(
            SandboxCommand::new("sh", workdir()).with_args(["-c", probe]),
            &mut |l| lines.push(l),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(stdout_of(&lines), [format!("M={expected}")]);
}

#[tokio::test]
async fn tokio_backend_missing_program_returns_err() {
    let backend = TokioBackend::new();
    let cmd = SandboxCommand::new("/nonexistent_program_borg_xyz_12345", workdir());

    let mut lines = vec![];
    let result = backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await;
    assert!(result.is_err(), "expected Err, got {result:?}");
}

/// stdin is `/dev/null`: the child must not read the agent's own stdin
/// (which, for a stdio MCP server, is the protocol stream).
#[cfg(unix)]
#[tokio::test]
async fn tokio_backend_stdin_is_null() {
    let backend = TokioBackend::with_timeout(Duration::from_secs(5));
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "cat; echo done"]);
    let mut lines = vec![];
    backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(stdout_of(&lines), ["done"]);
}

// ── TokioBackend timeout, cancel and process-tree kill ──────────────────────

#[tokio::test]
async fn tokio_backend_timeout_kills_process() {
    let backend = TokioBackend::with_timeout(Duration::from_millis(200));
    let cmd = SandboxCommand::new("sleep", workdir()).with_args(["10"]);
    let mut lines = vec![];
    let result = backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await;
    assert!(
        matches!(result, Err(SandboxError::Timeout)),
        "expected Timeout, got {result:?}"
    );
}

#[tokio::test]
async fn tokio_backend_cancel_kills_subprocess_fast() {
    // run() must honor the caller's cancel token; otherwise a long-running
    // subprocess outlives a cancelled session and its log pumps leak.
    let backend = TokioBackend::without_timeout();
    let cmd = SandboxCommand::new("sleep", workdir()).with_args(["10"]);
    let cancel = CancellationToken::new();
    let cancel_for_trigger = cancel.clone();

    // Fire cancel once the subprocess is running.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_for_trigger.cancel();
    });

    let started = Instant::now();
    let mut lines = vec![];
    let result = backend.run(cmd, &mut |l| lines.push(l), cancel).await;
    let elapsed = started.elapsed();

    assert!(
        matches!(result, Err(SandboxError::Cancelled)),
        "expected Cancelled, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "cancel should kill subprocess fast; took {elapsed:?}"
    );
}

#[tokio::test]
async fn tokio_backend_cancel_reaps_log_pumps_not_just_subprocess() {
    // Cancel must abort the log-pump tasks, not just kill the child:
    // an un-aborted pump would keep draining buffered output and emit
    // lines past the cancel point. We assert the line count doesn't grow
    // after run() returns Cancelled.
    let backend = TokioBackend::without_timeout();
    // Emit 1000 lines fast, then sleep: the burst leaves output buffered
    // in the pipe at cancel time, the condition that exposes a leaked pump.
    let cmd = SandboxCommand::new("sh", workdir()).with_args([
        "-c",
        "i=0; while [ $i -lt 1000 ]; do echo line-$i; i=$((i+1)); done; sleep 10",
    ]);

    let cancel = CancellationToken::new();
    let cancel_for_trigger = cancel.clone();
    let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<LogLine>::new()));
    let lines_for_cb = std::sync::Arc::clone(&lines);

    // Fire cancel while the subprocess is mid-output.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel_for_trigger.cancel();
    });

    let result = backend
        .run(
            cmd,
            &mut |l| {
                lines_for_cb.lock().unwrap().push(l);
            },
            cancel,
        )
        .await;

    let count_at_cancel_return = lines.lock().unwrap().len();
    assert!(
        matches!(result, Err(SandboxError::Cancelled)),
        "expected Cancelled, got {result:?}"
    );

    // Wait, then confirm no new lines arrived after the Cancelled return.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let count_after_settle = lines.lock().unwrap().len();
    assert_eq!(
        count_at_cancel_return, count_after_settle,
        "pump tasks must be aborted on cancel — no lines should arrive after \
         `run` returned. Saw {count_at_cancel_return} lines at return, \
         {count_after_settle} after settle. Difference indicates a leaked pump."
    );
}

/// True once `pid` no longer exists (a zombie still counts as alive, so
/// poll until init has reaped it).
#[cfg(unix)]
async fn wait_dead(pid: i32, within: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < within {
        // SAFETY: signal 0 only checks for existence.
        if unsafe { libc::kill(pid, 0) } != 0 {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    false
}

/// Parse the first stdout line as a pid.
#[cfg(unix)]
fn pid_line(l: &LogLine) -> Option<i32> {
    match l {
        LogLine::Stdout(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// `sh -c "sleep 1d & echo hi"` used to hang forever: `sh` exits but the
/// backgrounded `sleep` holds stdout open, and the output drain sat outside
/// the timeout. Now the leader's exit kills its process group, so the call
/// returns promptly with the output.
#[cfg(unix)]
#[tokio::test]
async fn background_grandchild_does_not_hang_the_call() {
    let timeout = Duration::from_secs(5);
    let backend = TokioBackend::with_timeout(timeout);
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "sleep 86400 & echo hi"]);
    let started = Instant::now();
    let mut lines = vec![];
    let result = tokio::time::timeout(
        timeout * 2,
        backend.run(cmd, &mut |l| lines.push(l), CancellationToken::new()),
    )
    .await
    .expect("run must return within its own timeout");
    assert!(started.elapsed() < timeout, "took {:?}", started.elapsed());
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(stdout_of(&lines), ["hi"]);
}

/// A descendant that leaves the process group (`setsid`) and keeps the
/// pipe open cannot extend the call past the timeout: the drain is
/// bounded by the same deadline.
#[cfg(unix)]
#[tokio::test]
async fn escaped_descendant_holding_pipe_is_bounded_by_timeout() {
    // The parent waits (on a pipe) until the child has left the group,
    // so the group kill at the parent's exit can't catch it first, which
    // happened under load and made the run return early.
    let script = "use POSIX; $|=1; pipe(my $r, my $w); my $p = fork; \
                  if ($p) { close $w; <$r>; print \"$p\\n\"; exit 0 } \
                  close $r; POSIX::setsid(); close $w; sleep 30;";
    let has_perl = std::process::Command::new("perl")
        .arg("-e1")
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !has_perl {
        eprintln!("perl not available; skipping");
        return;
    }
    let timeout = Duration::from_millis(1500);
    let backend = TokioBackend::with_timeout(timeout);
    let cmd = SandboxCommand::new("perl", workdir()).with_args(["-e", script]);
    let started = Instant::now();
    let mut lines = vec![];
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        backend.run(cmd, &mut |l| lines.push(l), CancellationToken::new()),
    )
    .await
    .expect("run must return within its own timeout");
    let elapsed = started.elapsed();
    // Clean up the escapee.
    if let Some(pid) = lines.iter().find_map(pid_line) {
        // SAFETY: plain kill(2) on a pid we just learned about.
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    assert!(matches!(result, Err(SandboxError::Timeout)), "{result:?}");
    assert!(
        elapsed < timeout + Duration::from_secs(2),
        "took {elapsed:?}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn grandchild_dies_on_cancel() {
    let backend = TokioBackend::without_timeout();
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "sleep 86400 & echo $!; wait"]);
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    let mut pid = None;
    let result = backend
        .run(
            cmd,
            &mut |l| {
                if let Some(p) = pid_line(&l) {
                    pid = Some(p);
                    trigger.cancel();
                }
            },
            cancel,
        )
        .await;
    assert!(matches!(result, Err(SandboxError::Cancelled)), "{result:?}");
    let pid = pid.expect("grandchild pid printed");
    assert!(
        wait_dead(pid, Duration::from_secs(3)).await,
        "grandchild {pid} survived cancel"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn grandchild_dies_on_timeout() {
    let backend = TokioBackend::with_timeout(Duration::from_millis(500));
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "sleep 86400 & echo $!; wait"]);
    let mut pid = None;
    let result = backend
        .run(
            cmd,
            &mut |l| pid = pid.or(pid_line(&l)),
            CancellationToken::new(),
        )
        .await;
    assert!(matches!(result, Err(SandboxError::Timeout)), "{result:?}");
    let pid = pid.expect("grandchild pid printed");
    assert!(
        wait_dead(pid, Duration::from_secs(3)).await,
        "grandchild {pid} survived timeout"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn grandchild_dies_when_run_future_is_dropped() {
    let backend = TokioBackend::without_timeout();
    let cmd = SandboxCommand::new("sh", workdir()).with_args(["-c", "sleep 86400 & echo $!; wait"]);
    let pid = std::sync::Arc::new(std::sync::Mutex::new(None));
    let pid_cb = pid.clone();
    let mut on_line = move |l: LogLine| {
        if let Some(p) = pid_line(&l) {
            *pid_cb.lock().unwrap() = Some(p);
        }
    };
    // Drop the future (via an outer timeout) after the pid has arrived.
    let _ = tokio::time::timeout(
        Duration::from_millis(500),
        backend.run(cmd, &mut on_line, CancellationToken::new()),
    )
    .await;
    let pid = pid.lock().unwrap().expect("grandchild pid printed");
    assert!(
        wait_dead(pid, Duration::from_secs(3)).await,
        "grandchild {pid} survived dropping the run future"
    );
}

// ── Bounded output ───────────────────────────────────────────────────────────

/// 50 MB with no newline used to be buffered whole by `lines()`.
#[cfg(unix)]
#[tokio::test]
async fn tokio_backend_caps_a_huge_newline_free_line() {
    let backend = TokioBackend::new().with_max_line_bytes(1000);
    let cmd = SandboxCommand::new("head", workdir()).with_args(["-c", "50000000", "/dev/zero"]);
    let mut lines = vec![];
    backend
        .run(cmd, &mut |l| lines.push(l), CancellationToken::new())
        .await
        .unwrap();
    let out = stdout_of(&lines);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 1000 + TRUNCATION_MARKER.len());
    assert!(out[0].ends_with(TRUNCATION_MARKER));
}

#[cfg(unix)]
#[tokio::test]
async fn tool_truncates_huge_output() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["head"]);
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "head", "args": ["-c", "50000000", "/dev/zero"] }),
    )
    .await;
    let out = r.output.expect("output");
    assert_eq!(out["truncated"], true);
    let stdout = out["stdout"].as_str().unwrap();
    assert!(stdout.len() <= (1 << 20) + TRUNCATION_MARKER.len());
    assert!(stdout.ends_with(TRUNCATION_MARKER));
}

/// Many lines: total cap applies and the result says so.
#[cfg(unix)]
#[tokio::test]
async fn tool_total_output_cap() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("sh")
        .workspace_root(root.path())
        .max_output_bytes(1000)
        .build()
        .unwrap();
    let mut ctx = SandboxContext::new();
    let r = tool
        .execute(
            serde_json::json!({
                "program": "sh",
                "args": ["-c", "i=0; while [ $i -lt 5000 ]; do echo line-$i; i=$((i+1)); done"]
            }),
            &mut ctx,
        )
        .await
        .unwrap();
    let out = r.output.expect("output");
    assert_eq!(out["truncated"], true);
    assert!(out["stdout"].as_str().unwrap().len() < 1100);
    assert!(ctx.lines.len() < 200, "kept {} lines", ctx.lines.len());
}

// ── SubprocessTool (Tool impl) ───────────────────────────────────────────────

#[tokio::test]
async fn subprocess_tool_execute_captures_full_output() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["sh"]);
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "sh", "args": ["-c", "echo line1 && echo line2"] }),
    )
    .await;
    let stdout = out_stdout(&r);
    assert!(stdout.contains("line1"), "stdout: {stdout}");
    assert!(stdout.contains("line2"), "stdout: {stdout}");
    assert_eq!(r.output.unwrap()["exit_code"], 0);
}

#[tokio::test]
async fn subprocess_tool_logical_failure_on_nonzero_exit() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["sh"]);
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "sh", "args": ["-c", "exit 3"] }),
    )
    .await;
    assert!(r.output.is_none(), "expected no output on failure");
    let err = r.error.expect("expected error message");
    assert!(err.contains("exit 3"), "error: {err}");
}

#[tokio::test]
async fn subprocess_tool_lines_collected_in_context_and_cleared_per_call() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["sh"]);
    let input = serde_json::json!({
        "program": "sh",
        "args": ["-c", "echo a && echo b && echo c"],
    });

    let mut ctx = SandboxContext::new();
    tool.execute(input.clone(), &mut ctx).await.unwrap();
    assert_eq!(ctx.lines.len(), 3);
    tool.execute(input, &mut ctx).await.unwrap();
    assert_eq!(
        ctx.lines.len(),
        3,
        "ctx.lines must not accumulate across calls"
    );
}

#[tokio::test]
async fn subprocess_tool_on_line_callback_fires() {
    use std::sync::{Arc, Mutex};

    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let seen2 = Arc::clone(&seen);

    let mut ctx = SandboxContext::with_callback(move |line| {
        if let LogLine::Stdout(s) = line {
            seen2.lock().unwrap().push(s);
        }
    });
    tool.execute(
        serde_json::json!({ "program": "echo", "args": ["callback-test"] }),
        &mut ctx,
    )
    .await
    .unwrap();

    let captured = seen.lock().unwrap();
    assert!(
        captured.iter().any(|s| s.contains("callback-test")),
        "callback not fired, captured: {captured:?}"
    );
}

// ── Invalid model input is a tool failure, not a run error ───────────────────

#[tokio::test]
async fn invalid_input_is_a_tool_failure_not_an_error() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    for (input, want) in [
        (serde_json::json!({}), "missing required field 'program'"),
        (serde_json::json!({ "program": 5 }), "'program' must be"),
        (serde_json::json!({ "program": "" }), "'program' must be"),
        (serde_json::json!("echo"), "input must be a JSON object"),
        (
            serde_json::json!({ "program": "echo", "args": [1] }),
            "'args' must be an array of strings",
        ),
        (
            serde_json::json!({ "program": "echo", "args": "x" }),
            "'args' must be an array of strings",
        ),
        (
            serde_json::json!({ "program": "echo", "workdir": 1 }),
            "'workdir' must be a string",
        ),
        (
            serde_json::json!({ "program": "echo", "env": [] }),
            "'env' must be an object",
        ),
    ] {
        let mut ctx = SandboxContext::new();
        let r = tool
            .execute(input.clone(), &mut ctx)
            .await
            .unwrap_or_else(|e| panic!("{input} must not be a run error: {e:?}"));
        let err = r
            .error
            .unwrap_or_else(|| panic!("{input}: expected failure"));
        assert!(err.contains(want), "{input}: {err}");
    }
}

// ── Builder validation ───────────────────────────────────────────────────────

#[test]
fn builder_requires_programs_and_workspace_root() {
    let root = tempfile::TempDir::new().unwrap();
    let r = SubprocessTool::builder("t", "t", TokioBackend::new())
        .workspace_root(root.path())
        .build();
    assert!(matches!(r, Err(ConfigError::NoPrograms)));

    let r = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("echo")
        .build();
    assert!(matches!(r, Err(ConfigError::NoWorkspaceRoot)));

    let r = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("echo")
        .workspace_root(root.path().join("missing"))
        .build();
    assert!(matches!(r, Err(ConfigError::BadWorkspaceRoot(_))));
}

#[test]
fn builder_rejects_unknown_and_relative_programs_and_blocked_env() {
    let root = tempfile::TempDir::new().unwrap();
    let base =
        || SubprocessTool::builder("t", "t", TokioBackend::new()).workspace_root(root.path());
    assert!(matches!(
        base().allow_program("no_such_program_borg_xyz").build(),
        Err(ConfigError::ProgramNotFound(_))
    ));
    assert!(matches!(
        base().allow_program("bin/echo").build(),
        Err(ConfigError::RelativeProgramPath(_))
    ));
    for key in [
        "PATH",
        "LD_PRELOAD",
        "BASH_ENV",
        "GIT_CONFIG_COUNT",
        "NODE_OPTIONS",
        "HOME",
        "Home",
        "XDG_CONFIG_HOME",
        "GIT_SSH_COMMAND",
        "CARGO_BUILD_RUSTC_WRAPPER",
        "RUSTC_WRAPPER",
    ] {
        assert!(
            matches!(
                base().allow_program("echo").allow_env([key]).build(),
                Err(ConfigError::BlockedEnvKey(_))
            ),
            "{key}"
        );
    }
}

// ── Program allowlist ────────────────────────────────────────────────────────

#[tokio::test]
async fn allowlist_blocks_unlisted_program() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "sh", "args": ["-c", "echo hi"] }),
    )
    .await;
    let err = r.error.expect("expected logical failure");
    assert!(err.contains("not in the allowed list"), "got: {err}");
}

#[tokio::test]
async fn allowlist_permits_listed_program() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "echo", "args": ["allowed"] }),
    )
    .await;
    assert!(out_stdout(&r).contains("allowed"));
}

/// The allowlist used to match on basename, so a model-planted
/// `/tmp/x/echo` passed an allowlist containing `echo`.
#[cfg(unix)]
#[tokio::test]
async fn allowlist_rejects_paths_with_an_allowlisted_basename() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::TempDir::new().unwrap();
    let planted = root.path().join("echo");
    std::fs::write(&planted, "#!/bin/sh\necho PWNED\n").unwrap();
    std::fs::set_permissions(&planted, std::fs::Permissions::from_mode(0o755)).unwrap();

    let tool = tool_in(&root, &["echo"]);
    for program in [planted.to_str().unwrap(), "./echo", "../echo", "x/echo"] {
        let r = run_tool(&tool, serde_json::json!({ "program": program })).await;
        let err = r
            .error
            .unwrap_or_else(|| panic!("{program} must be rejected, got {:?}", r.output));
        assert!(err.contains("not in the allowed list"), "{program}: {err}");
    }
}

/// An allowlisted name runs the binary resolved at build time; its
/// resolved absolute path is allowed too.
#[cfg(unix)]
#[tokio::test]
async fn allowlisted_name_and_its_resolved_path_both_run() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    let resolved = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .filter(|d| d.is_absolute())
        .map(|d| d.join("echo"))
        .find(|p| p.is_file())
        .unwrap();
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": resolved.to_str().unwrap(), "args": ["ok"] }),
    )
    .await;
    assert_eq!(out_stdout(&r), "ok");
}

/// A caller-given absolute path is allowed exactly; its bare basename is
/// not implicitly allowed.
#[cfg(unix)]
#[tokio::test]
async fn allowlisted_absolute_path_is_exact() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::TempDir::new().unwrap();
    let script = root.path().join("hello.sh");
    std::fs::write(&script, "#!/bin/sh\necho hello-script\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let script = script.to_str().unwrap().to_string();

    let tool = tool_in(&root, &[&script]);
    let r = run_tool(&tool, serde_json::json!({ "program": script })).await;
    assert_eq!(out_stdout(&r), "hello-script");
    let r = run_tool(&tool, serde_json::json!({ "program": "hello.sh" })).await;
    assert!(
        r.error.is_some(),
        "bare basename must not match an absolute entry"
    );
}

/// Program lookup uses the caller's search path; a model-supplied `PATH`
/// is never honoured, and `search_path` is what the child sees.
#[cfg(unix)]
#[tokio::test]
async fn model_path_is_never_honoured() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("sh")
        .workspace_root(root.path())
        .search_path("/usr/bin:/bin")
        .build()
        .unwrap();
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "sh",
            "args": ["-c", "echo $PATH"],
            "env": { "PATH": "/tmp/evil" }
        }),
    )
    .await;
    assert_eq!(out_stdout(&r), "/usr/bin:/bin");

    // Same on the unrestricted tool, which otherwise forwards model env.
    let tool = SubprocessTool::unrestricted("t", "t", TokioBackend::new());
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "sh",
            "args": ["-c", "echo $PATH"],
            "env": { "PATH": "/tmp/evil" }
        }),
    )
    .await;
    assert!(!out_stdout(&r).contains("/tmp/evil"));
}

// ── Model env ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn model_env_ignored_unless_allowlisted() {
    let root = tempfile::TempDir::new().unwrap();
    let input = serde_json::json!({
        "program": "sh",
        "args": ["-c", "echo A=${A:-UNSET} B=${B:-UNSET}"],
        "env": { "A": "1", "B": "2" }
    });

    let tool = tool_in(&root, &["sh"]);
    assert_eq!(
        out_stdout(&run_tool(&tool, input.clone()).await),
        "A=UNSET B=UNSET"
    );

    let tool = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("sh")
        .workspace_root(root.path())
        .allow_env(["A"])
        .env("B", "fixed")
        .build()
        .unwrap();
    assert_eq!(out_stdout(&run_tool(&tool, input).await), "A=1 B=fixed");
}

#[cfg(unix)]
#[tokio::test]
async fn startup_and_config_injection_env_is_dropped() {
    let tool = SubprocessTool::unrestricted("t", "t", TokioBackend::new());
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "sh",
            "args": ["-c", "echo ${BASH_ENV:-u}${ENV:-u}${GIT_CONFIG_COUNT:-u}${NODE_OPTIONS:-u}${PYTHONPATH:-u}${DYLD_INSERT_LIBRARIES:-u}${OK:-u}"],
            "env": {
                "BASH_ENV": "/tmp/x", "ENV": "/tmp/x", "GIT_CONFIG_COUNT": "1",
                "NODE_OPTIONS": "--require /tmp/x", "PYTHONPATH": "/tmp/x",
                "DYLD_INSERT_LIBRARIES": "/tmp/x", "OK": "ok"
            }
        }),
    )
    .await;
    assert_eq!(out_stdout(&r), "uuuuuuok");
}

/// `HOME`, `XDG_*`, `GIT_*`, `CARGO_*` and `RUSTC*` from the model never
/// reach the child, even on the unrestricted tool; the backend's own
/// `HOME` passthrough still does.
#[cfg(unix)]
#[tokio::test]
async fn config_relocation_and_tool_runner_env_is_dropped() {
    let tool = SubprocessTool::unrestricted("t", "t", TokioBackend::new());
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "sh",
            "args": ["-c", "echo ${HOME:-u}; echo ${XDG_CONFIG_HOME:-u}${GIT_EXTERNAL_DIFF:-u}${CARGO_TARGET_X_RUNNER:-u}${RUSTC_WRAPPER:-u}${OK:-u}"],
            "env": {
                "HOME": "/tmp/evil-home", "XDG_CONFIG_HOME": "/tmp/x",
                "GIT_EXTERNAL_DIFF": "/tmp/x", "CARGO_TARGET_X_RUNNER": "/tmp/x",
                "RUSTC_WRAPPER": "/tmp/x", "OK": "ok"
            }
        }),
    )
    .await;
    let stdout = out_stdout(&r);
    let mut lines = stdout.lines();
    let home = lines.next().unwrap();
    assert_ne!(home, "/tmp/evil-home", "the model set HOME");
    if let Ok(parent_home) = std::env::var("HOME") {
        assert_eq!(home, parent_home, "the backend's HOME passthrough");
    }
    assert_eq!(lines.next(), Some("uuuuok"));
}

// ── Output caps and exit status ─────────────────────────────────────────────

/// Hitting the total output cap stops the process instead of letting it
/// run to the timeout: `yes` never ends on its own.
#[cfg(unix)]
#[tokio::test]
async fn output_cap_stops_an_endless_producer() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = SubprocessTool::builder(
        "t",
        "t",
        TokioBackend::with_timeout(Duration::from_secs(60)),
    )
    .allow_program("yes")
    .workspace_root(root.path())
    .max_output_bytes(64 * 1024)
    .build()
    .unwrap();
    let started = Instant::now();
    let r = tokio::time::timeout(
        Duration::from_secs(20),
        run_tool(&tool, serde_json::json!({ "program": "yes" })),
    )
    .await
    .expect("the output cap must stop `yes` long before the timeout");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "took {:?}",
        started.elapsed()
    );
    let out = r
        .output
        .unwrap_or_else(|| panic!("expected the truncated output, got {:?}", r.error));
    assert_eq!(out["truncated"], true);
    assert!(out["exit_code"].is_null(), "{out}");
    assert!(
        out["note"].as_str().unwrap_or_default().contains("stopped"),
        "{out}"
    );
    let stdout = out["stdout"].as_str().unwrap();
    assert!(stdout.starts_with("y\ny\n"));
    assert!(stdout.len() <= 64 * 1024 + TRUNCATION_MARKER.len());
}

/// A non-zero exit shows the model its exit code and both streams, not
/// just the first stderr line.
#[tokio::test]
async fn nonzero_exit_reports_code_stdout_and_stderr() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["sh"]);
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "sh",
            "args": ["-c", "echo out-1; echo out-2; echo err-1 >&2; echo err-2 >&2; exit 4"]
        }),
    )
    .await;
    assert!(r.output.is_none());
    let err = r.error.expect("failure");
    for want in ["exit 4", "out-1", "out-2", "err-1", "err-2"] {
        assert!(err.contains(want), "missing {want:?} in {err:?}");
    }
}

/// The failure text stays within the output caps.
#[cfg(unix)]
#[tokio::test]
async fn nonzero_exit_output_respects_caps() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("sh")
        .workspace_root(root.path())
        .max_line_bytes(100)
        .build()
        .unwrap();
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "sh",
            "args": ["-c", "head -c 10000 /dev/zero | tr '\\0' e >&2; echo; exit 1"]
        }),
    )
    .await;
    let err = r.error.expect("failure");
    assert!(err.starts_with("exit 1"), "{err}");
    assert!(err.contains(TRUNCATION_MARKER), "{err}");
    assert!(err.contains("[output truncated]"), "{err}");
    assert!(err.len() < 400, "{} bytes: {err}", err.len());
}

// ── Workdir ──────────────────────────────────────────────────────────────────

// `/etc` is a Unix path; on Windows it does not resolve, so the error differs.
#[cfg(unix)]
#[tokio::test]
async fn workdir_outside_root_rejected() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "echo", "workdir": "/etc" }),
    )
    .await;
    let err = r.error.expect("expected logical failure");
    assert!(err.contains("outside workspace root"), "got: {err}");
}

#[tokio::test]
async fn workdir_inside_root_permitted() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = tool_in(&root, &["echo"]);
    let r = run_tool(
        &tool,
        serde_json::json!({
            "program": "echo",
            "args": ["ok"],
            "workdir": root.path().to_str().unwrap()
        }),
    )
    .await;
    assert!(r.output.is_some(), "{:?}", r.error);
}

#[cfg(unix)]
#[tokio::test]
async fn workdir_defaults_to_root_and_relative_is_under_root() {
    let root = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(root.path().join("sub")).unwrap();
    let canonical = root.path().canonicalize().unwrap();
    let tool = tool_in(&root, &["pwd"]);

    let r = run_tool(&tool, serde_json::json!({ "program": "pwd" })).await;
    assert_eq!(out_stdout(&r), canonical.to_str().unwrap());

    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "pwd", "workdir": "sub" }),
    )
    .await;
    assert_eq!(out_stdout(&r), canonical.join("sub").to_str().unwrap());

    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "pwd", "workdir": ".." }),
    )
    .await;
    assert!(r.error.unwrap().contains("outside workspace root"));
}

#[tokio::test]
async fn output_truncated_at_max_bytes() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("echo")
        .workspace_root(root.path())
        .max_output_bytes(3) // "hello" = 5 bytes → truncated
        .build()
        .unwrap();
    let r = run_tool(
        &tool,
        serde_json::json!({ "program": "echo", "args": ["hello"] }),
    )
    .await;
    let out = r.output.expect("expected output");
    assert_eq!(out["truncated"], true, "expected truncated=true");
    assert_eq!(out["stdout"], format!("hel{TRUNCATION_MARKER}"));
}
