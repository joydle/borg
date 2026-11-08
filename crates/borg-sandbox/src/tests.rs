use crate::backend::SandboxBackend;

// Compile-time check that SandboxBackend is object-safe.
fn _assert_sandbox_backend_object_safe(_b: Box<dyn SandboxBackend>) {}

#[cfg(test)]
mod tokio_backend_tests {
    use crate::{LogLine, SandboxBackend, SandboxCommand, TokioBackend};
    use std::path::PathBuf;

    #[tokio::test]
    async fn echo_hello_world() {
        let backend = TokioBackend::new();
        let cmd = SandboxCommand {
            program: "echo".into(),
            args: vec!["hello".into(), "world".into()],
            workdir: PathBuf::from("/tmp"),
            env: vec![],
        };
        let mut lines: Vec<LogLine> = vec![];
        backend
            .run(
                cmd,
                &mut |line| lines.push(line),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(lines
            .iter()
            .any(|l| matches!(l, LogLine::Stdout(s) if s.contains("hello world"))));
    }

    #[tokio::test]
    async fn non_zero_exit_returns_exit_code() {
        let backend = TokioBackend::new();
        let cmd = SandboxCommand {
            program: "sh".into(),
            args: vec!["-c".into(), "exit 42".into()],
            workdir: PathBuf::from("/tmp"),
            env: vec![],
        };
        let mut lines = vec![];
        let result = backend
            .run(
                cmd,
                &mut |l| lines.push(l),
                tokio_util::sync::CancellationToken::new(),
            )
            .await;
        match result {
            Err(crate::SandboxError::NonZeroExit(code)) => assert_eq!(code, 42),
            other => panic!("expected NonZeroExit(42), got {other:?}"),
        }
    }
}

#[cfg(test)]
mod subprocess_tool_tests {
    use crate::{SandboxContext, SubprocessTool, TokioBackend};
    use borg::Tool;

    fn make_tool() -> SubprocessTool<TokioBackend> {
        SubprocessTool::unrestricted("run_cmd", "Run a sandboxed command", TokioBackend::new())
    }

    #[test]
    fn schema_has_correct_name() {
        let tool = make_tool();
        let schema = tool.schema();
        assert_eq!(schema.name, "run_cmd");
        assert!(schema.input_schema["properties"]["program"].is_object());
        assert!(schema.input_schema["properties"]["args"].is_object());
    }

    #[test]
    fn restricted_schema_lists_programs_and_hides_env() {
        let root = tempfile::TempDir::new().unwrap();
        let tool = SubprocessTool::builder("run", "d", TokioBackend::new())
            .allow_programs(["echo", "sh"])
            .workspace_root(root.path())
            .build()
            .unwrap();
        let schema = tool.schema();
        let desc = schema.input_schema["properties"]["program"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("echo, sh"), "{desc}");
        assert!(schema.input_schema["properties"].get("env").is_none());
    }

    #[tokio::test]
    async fn execute_echo_returns_output() {
        let tool = make_tool();
        let input = serde_json::json!({
            "program": "echo",
            "args": ["borg-sandbox works"],
            "workdir": "/tmp"
        });
        let mut ctx = SandboxContext::new();
        let result = tool.execute(input, &mut ctx).await.unwrap();
        assert!(result.output.is_some());
        let text = result.output.unwrap()["stdout"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("borg-sandbox works"));
    }

    #[tokio::test]
    async fn execute_bad_program_returns_logical_failure() {
        let tool = make_tool();
        let input = serde_json::json!({
            "program": "/nonexistent_program_xyz",
            "args": [],
            "workdir": "/tmp"
        });
        let mut ctx = SandboxContext::new();
        let result = tool.execute(input, &mut ctx).await.unwrap();
        assert!(result.error.is_some(), "expected logical failure");
    }
}

#[cfg(test)]
mod security_tests {
    use crate::{SandboxContext, SubprocessTool, TokioBackend};
    use borg::Tool;
    use std::path::PathBuf;

    fn tool_with_workspace(root: PathBuf) -> SubprocessTool<TokioBackend> {
        SubprocessTool::builder("run", "test", TokioBackend::new())
            .workspace_root(root)
            .allow_program("echo")
            .build()
            .unwrap()
    }

    /// VULN-1: a non-existent `..` traversal path must be rejected by the
    /// security check, not silently accepted via the canonicalize fallback.
    /// The workspace root is the canonical temp dir so that the raw path
    /// `/canonical_tmp/nonexistent/../../etc` passes a component-wise
    /// `starts_with` while escaping at the OS level — rejection must come
    /// from the security layer, not an OS spawn error.
    #[tokio::test]
    async fn rejects_nonexistent_dotdot_escape() {
        let canonical_tmp = std::env::temp_dir()
            .canonicalize()
            .expect("temp_dir should be canonicalize-able");
        let tool = tool_with_workspace(canonical_tmp.clone());

        let traversal = format!(
            "{}/borg_test_nonexistent_xyz_abc/../../etc",
            canonical_tmp.display()
        );
        let input = serde_json::json!({
            "program": "echo",
            "workdir": traversal
        });
        let mut ctx = SandboxContext::new();
        let result = tool.execute(input, &mut ctx).await.unwrap();
        let err = result
            .error
            .expect("expected rejection of non-existent dotdot path");
        assert!(
            err.contains("does not exist or is not accessible"),
            "expected security-check rejection message, got: {err:?}"
        );
    }

    /// VULN-2: the canonical path (not the original) must reach the
    /// subprocess after the workspace check — a symlink that resolves
    /// inside the workspace should succeed.
    #[tokio::test]
    async fn symlink_within_workspace_is_allowed() {
        use std::fs;

        let base = tempfile::TempDir::new().unwrap();
        let real = base.path().join("real");
        fs::create_dir_all(&real).unwrap();
        let link = base.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let tool = SubprocessTool::builder("run", "test", TokioBackend::new())
            .workspace_root(real.canonicalize().unwrap())
            .allow_program("pwd")
            .build()
            .unwrap();

        // workdir is the symlink, which resolves into workspace_root.
        let input = serde_json::json!({
            "program": "pwd",
            "workdir": link.to_str().unwrap()
        });
        let mut ctx = SandboxContext::new();
        let result = tool.execute(input, &mut ctx).await.unwrap();
        assert!(
            result.output.is_some(),
            "symlink within workspace should succeed, got error: {:?}",
            result.error
        );
    }

    /// VULN-3: LD_PRELOAD and DYLD_* must be stripped from LLM-supplied env
    /// to prevent shared-library injection, even on the unrestricted tool
    /// (which otherwise forwards the model's env).
    #[tokio::test]
    async fn strips_dangerous_loader_env_vars() {
        let tool = SubprocessTool::unrestricted("run", "test", TokioBackend::new());

        // Echo full variable names so substring matching can't false-pass
        // (e.g. "DYLD=UNSET" containing "LD=UNSET").
        let input = serde_json::json!({
            "program": "sh",
            "args": ["-c",
                "echo LD_PRELOAD=${LD_PRELOAD:-UNSET}; echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-UNSET}; echo SAFE_VAR=${SAFE_VAR:-UNSET}"
            ],
            "workdir": "/tmp",
            "env": {
                "LD_PRELOAD": "/tmp/evil.so",
                "LD_LIBRARY_PATH": "/tmp/evil",
                "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
                "DYLD_LIBRARY_PATH": "/tmp/evil",
                "SAFE_VAR": "hello"
            }
        });
        let mut ctx = SandboxContext::new();
        let result = tool.execute(input, &mut ctx).await.unwrap();
        let stdout = result
            .output
            .as_ref()
            .and_then(|o| o["stdout"].as_str())
            .unwrap_or("");
        assert!(
            stdout.contains("LD_PRELOAD=UNSET"),
            "LD_PRELOAD must be stripped, got stdout: {stdout:?}"
        );
        assert!(
            stdout.contains("LD_LIBRARY_PATH=UNSET"),
            "LD_LIBRARY_PATH must be stripped (and not inherited), got stdout: {stdout:?}"
        );
        assert!(
            stdout.contains("SAFE_VAR=hello"),
            "unrestricted tool forwards non-blocked env: {stdout:?}"
        );
    }

    #[test]
    fn blocklist_matches_names_and_prefixes_case_insensitively() {
        use crate::tool::BLOCKED_ENV;
        for k in BLOCKED_ENV {
            assert!(crate::tool::is_blocked_env(k), "{k}");
        }
        for k in [
            "path",
            "LD_AUDIT",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_GLOBAL",
            "BASH_FUNC_x%%",
            // Settings that run programs or relocate config files.
            "GIT_SSH_COMMAND",
            "git_external_diff",
            "GIT_AUTHOR_NAME",
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
            "cargo_build_rustc_wrapper",
            "CARGO_HOME",
            "RUSTC",
            "RUSTC_WRAPPER",
            "rustc_workspace_wrapper",
            "RUSTFLAGS",
            "HOME",
            "home",
            "XDG_CONFIG_HOME",
            "xdg_data_home",
            "A=B",
            "",
        ] {
            assert!(crate::tool::is_blocked_env(k), "{k}");
        }
        for k in ["FOO", "RUST_LOG", "CARGOX", "GITHUB_TOKEN", "XDGX"] {
            assert!(!crate::tool::is_blocked_env(k), "{k}");
        }
    }
}
