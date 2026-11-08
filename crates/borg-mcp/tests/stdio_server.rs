//! End-to-end dispatch tests against a fake MCP server over stdio.
//!
//! `harness = false`: this binary is both the test runner and the fake
//! server. Run with `--borg-mcp-fake-server`, it speaks just enough
//! newline-delimited JSON-RPC (`initialize`, `tools/list`,
//! `tools/call`) to exercise the client, so the tests need no `npx`,
//! Python or network. Each test spawns the binary itself as the server.
//!
//! Fake tools:
//! - `env`   — returns the server's environment as `KEY=VALUE` lines.
//! - `echo`  — returns its arguments.
//! - `big`   — returns `bytes` bytes of text (`stream: true` streams
//!   them without buffering, for multi-hundred-MB replies).
//! - `hang`  — never answers.

use std::future::Future;
use std::io::{BufRead, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use borg::{BorgError, DispatchedCall, ToolCallRequest, ToolDispatch};
use borg_mcp::{McpToolDispatch, StdioServerSpec};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

const FAKE_SERVER_FLAG: &str = "--borg-mcp-fake-server";
/// Set in the test process before any server spawns; must not reach a
/// child that doesn't inherit or explicitly pass it.
const PARENT_SECRET: &str = "BORG_MCP_TEST_PARENT_SECRET";

// ───────────────────────────── fake server ─────────────────────────────

fn fake_server() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { return };
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        // Notifications (no id) need no reply.
        let Some(id) = msg.get("id").cloned() else {
            continue;
        };
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let result = match msg["method"].as_str().unwrap_or_default() {
            "initialize" => json!({
                "protocolVersion": params["protocolVersion"],
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "fake", "version": "0.0.0" }
            }),
            "tools/list" => {
                // The last three must be dropped by the client: a name
                // the model APIs reject, a namespaced name over 64 bytes,
                // and a duplicate.
                let long = "l".repeat(60);
                let tools: Vec<Value> = ["env", "echo", "big", "hang", "bad name", &long, "echo"]
                    .iter()
                    .map(|n| {
                        json!({
                            "name": n,
                            "description": format!("fake {n}"),
                            "inputSchema": { "type": "object" }
                        })
                    })
                    .collect();
                json!({ "tools": tools })
            }
            "tools/call" => {
                let args = &params["arguments"];
                let text = match params["name"].as_str().unwrap_or_default() {
                    "env" => std::env::vars()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    "echo" => args.to_string(),
                    // `stream: true` writes the reply straight to stdout
                    // in chunks, so the server never holds it all; the
                    // client must not either.
                    "big" if args["stream"].as_bool() == Some(true) => {
                        let bytes = args["bytes"].as_u64().unwrap_or(0) as usize;
                        let head = format!(
                            r#"{{"jsonrpc":"2.0","id":{id},"result":{{"content":[{{"type":"text","text":""#
                        );
                        let chunk = "x".repeat(64 * 1024);
                        let mut out = stdout.lock();
                        let mut written = 0;
                        let mut ok = out.write_all(head.as_bytes()).is_ok();
                        while ok && written < bytes {
                            ok = out.write_all(chunk.as_bytes()).is_ok();
                            written += chunk.len();
                        }
                        if !ok
                            || writeln!(out, r#""}}]}}}}"#)
                                .and_then(|_| out.flush())
                                .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                    "big" => "x".repeat(args["bytes"].as_u64().unwrap_or(0) as usize),
                    "hang" => continue,
                    other => format!("unknown tool {other}"),
                };
                json!({ "content": [{ "type": "text", "text": text }], "isError": false })
            }
            _ => json!({}),
        };
        let reply = json!({ "jsonrpc": "2.0", "id": id, "result": result });
        if writeln!(stdout, "{reply}")
            .and_then(|_| stdout.flush())
            .is_err()
        {
            return;
        }
    }
}

// ───────────────────────────── helpers ─────────────────────────────

fn fake_spec() -> StdioServerSpec {
    let exe = std::env::current_exe().expect("current_exe");
    StdioServerSpec::new(exe.to_string_lossy()).arg(FAKE_SERVER_FLAG)
}

async fn call(
    mcp: &Arc<McpToolDispatch>,
    tool: &str,
    args: Value,
) -> Result<DispatchedCall, BorgError> {
    mcp.dispatch(ToolCallRequest::new(
        format!("mcp__fake__{tool}"),
        args,
        "c1",
    ))
    .await
}

/// Text of the first content block of a successful call.
fn first_text(call: &DispatchedCall) -> String {
    let output = call
        .result
        .output
        .as_ref()
        .unwrap_or_else(|| panic!("expected output, got {:?}", call.result.error));
    output[0]["text"].as_str().expect("text block").to_string()
}

/// Names only: a failing assertion must never print env values, which
/// in CI can include secrets.
fn env_names(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| l.split_once('=').map(|(k, _)| k))
        .collect()
}

fn has_var(text: &str, key: &str) -> bool {
    env_names(text).iter().any(|k| k.eq_ignore_ascii_case(key))
}

// ───────────────────────────── tests ─────────────────────────────

async fn round_trip_echo() {
    let mcp = McpToolDispatch::connect_stdio("fake", &fake_spec())
        .await
        .expect("connect");
    let names: Vec<String> = mcp.schemas().iter().map(|s| s.name.clone()).collect();
    assert!(names.contains(&"mcp__fake__echo".to_string()), "{names:?}");
    let out = call(&mcp, "echo", json!({"hello": "world"})).await.unwrap();
    assert!(first_text(&out).contains("world"));
}

/// Invalid, overlong and duplicate tool names never reach the schema
/// list, on connect or on refresh.
async fn bad_tool_names_are_dropped() {
    let mcp = McpToolDispatch::connect_stdio("fake", &fake_spec())
        .await
        .expect("connect");
    let expect = ["env", "echo", "big", "hang"].map(|n| format!("mcp__fake__{n}"));
    let names: Vec<String> = mcp.schemas().iter().map(|s| s.name.clone()).collect();
    assert_eq!(names, expect, "after connect");
    mcp.refresh().await.expect("refresh");
    let names: Vec<String> = mcp.schemas().iter().map(|s| s.name.clone()).collect();
    assert_eq!(names, expect, "after refresh");
    let out = call(&mcp, "bad name", json!({})).await.unwrap();
    assert!(out.result.error.is_some(), "a dropped tool must not route");
}

/// Peak resident set of this process in bytes, where the platform
/// reports it without libc.
fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// A result line far past the frame limit is never buffered whole: the
/// call fails cleanly and quickly, and later calls see a dead server.
async fn huge_result_is_not_buffered() {
    const HUGE: u64 = 256 * 1024 * 1024;
    let spec = fake_spec().max_result_bytes(1024);
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let started = Instant::now();
    let out = tokio::time::timeout(
        Duration::from_secs(30),
        call(&mcp, "big", json!({"stream": true, "bytes": HUGE})),
    )
    .await
    .expect("an oversize message must not hang the call")
    .expect("an oversize message is a failed tool result, not a dispatch error");
    let err = out.result.error.expect("oversize result should fail");
    assert!(
        err.contains("limit") && err.contains("connection"),
        "got: {}",
        &err[..err.len().min(300)]
    );
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "took {:?}",
        started.elapsed()
    );
    if let Some(peak) = peak_rss_bytes() {
        assert!(
            peak < HUGE / 2,
            "peak RSS {peak} bytes: the oversize line was buffered"
        );
    }
    // The connection is gone: a later call is a transport error.
    let later = call(&mcp, "echo", json!({})).await;
    assert!(later.is_err(), "expected a dead connection, got {later:?}");
}

async fn child_env_is_cleared_by_default() {
    let spec = fake_spec().env("BORG_MCP_DECLARED", "declared-value");
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let text = first_text(&call(&mcp, "env", json!({})).await.unwrap());
    assert!(
        !has_var(&text, PARENT_SECRET),
        "parent-only var leaked into the child env; child saw {:?}",
        env_names(&text)
    );
    assert!(
        text.contains("BORG_MCP_DECLARED=declared-value"),
        "declared env missing; child saw {:?}",
        env_names(&text)
    );
    assert!(
        has_var(&text, "PATH"),
        "PATH not passed through; child saw {:?}",
        env_names(&text)
    );
}

async fn child_env_pass_env_forwards_named_var() {
    let spec = fake_spec().pass_env(PARENT_SECRET);
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let text = first_text(&call(&mcp, "env", json!({})).await.unwrap());
    assert!(
        text.contains(&format!("{PARENT_SECRET}=parent-only")),
        "child saw {:?}",
        env_names(&text)
    );
}

async fn child_env_inherit_opt_in() {
    let spec = fake_spec().inherit_env(true);
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let text = first_text(&call(&mcp, "env", json!({})).await.unwrap());
    assert!(
        text.contains(&format!("{PARENT_SECRET}=parent-only")),
        "child saw {:?}",
        env_names(&text)
    );
}

async fn hung_tool_times_out() {
    let spec = fake_spec().call_timeout(Duration::from_millis(300));
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let started = Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(10), call(&mcp, "hang", json!({})))
        .await
        .expect("dispatch itself must not hang")
        .expect("a timeout is a tool failure, not a dispatch error");
    assert!(started.elapsed() < Duration::from_secs(5));
    let err = out.result.error.expect("timed-out call should fail");
    assert!(err.contains("timed out"), "got: {err}");
    // The connection is still usable afterwards.
    let ok = call(&mcp, "echo", json!({"after": "timeout"}))
        .await
        .unwrap();
    assert!(first_text(&ok).contains("timeout"));
}

async fn hung_tool_is_cancelled_by_token() {
    let cancel = CancellationToken::new();
    let spec = fake_spec().cancel_token(cancel.clone());
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let res = tokio::time::timeout(Duration::from_secs(10), call(&mcp, "hang", json!({})))
        .await
        .expect("cancel must interrupt the call");
    assert!(
        matches!(res, Err(BorgError::Cancelled)),
        "expected BorgError::Cancelled, got {res:?}"
    );
}

async fn oversized_result_is_capped() {
    let spec = fake_spec().max_result_bytes(1024);
    let mcp = McpToolDispatch::connect_stdio("fake", &spec)
        .await
        .expect("connect");
    let out = call(&mcp, "big", json!({"bytes": 50_000})).await.unwrap();
    assert!(out.result.output.is_none());
    let err = out.result.error.expect("oversized result should fail");
    assert!(
        err.contains("truncated"),
        "got: {}",
        &err[..err.len().min(200)]
    );
    assert!(err.len() < 2048, "capped error is {} bytes", err.len());

    // Under the cap passes through untouched.
    let small = call(&mcp, "big", json!({"bytes": 100})).await.unwrap();
    assert_eq!(first_text(&small).len(), 100);
}

// ───────────────────────────── runner ─────────────────────────────

type TestFn = fn() -> Pin<Box<dyn Future<Output = ()> + Send>>;

macro_rules! tests {
    ($($name:ident),* $(,)?) => {
        &[$((stringify!($name), (|| Box::pin($name()) as Pin<Box<dyn Future<Output = ()> + Send>>) as TestFn)),*]
    };
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == FAKE_SERVER_FLAG) {
        fake_server();
        return;
    }

    let all: &[(&str, TestFn)] = tests![
        round_trip_echo,
        bad_tool_names_are_dropped,
        huge_result_is_not_buffered,
        child_env_is_cleared_by_default,
        child_env_pass_env_forwards_named_var,
        child_env_inherit_opt_in,
        hung_tool_times_out,
        hung_tool_is_cancelled_by_token,
        oversized_result_is_capped,
    ];

    // `cargo test -- <filter>`: run tests whose name contains a filter.
    let filters: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if args.iter().any(|a| a == "--list") {
        for (name, _) in all {
            println!("{name}: test");
        }
        return;
    }

    // Single-threaded at this point: no other thread reads the env.
    std::env::set_var(PARENT_SECRET, "parent-only");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let mut failed = Vec::new();
    let mut ran = 0;
    for (name, test) in all {
        if !filters.is_empty() && !filters.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        ran += 1;
        // Spawned so a panic surfaces as a JoinError instead of aborting
        // the remaining tests.
        let outcome = rt.block_on(async { tokio::spawn(test()).await });
        match outcome {
            Ok(()) => println!("test {name} ... ok"),
            Err(e) => {
                println!("test {name} ... FAILED ({e})");
                failed.push(*name);
            }
        }
    }
    println!(
        "\ntest result: {}. {} passed; {} failed",
        if failed.is_empty() { "ok" } else { "FAILED" },
        ran - failed.len(),
        failed.len()
    );
    if !failed.is_empty() {
        std::process::exit(1);
    }
}
