<p align="center">
  <a href="https://joydle.github.io/borg/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="media/logo-dark.svg">
      <img src="media/logo-light.svg" alt="borg" width="360">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://github.com/joydle/borg/actions/workflows/ci.yml"><img src="https://github.com/joydle/borg/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-black.svg" alt="License: MIT OR Apache-2.0"></a>
  <img src="https://img.shields.io/badge/rust-1.88+-black.svg" alt="MSRV 1.88">
</p>

**borg is a supervised agent runtime for Rust.** Compose LLM calls into
pipelines that cancel cleanly, isolate panics, retry what's retryable,
stay inside a budget and resume after a crash.

```toml
[dependencies]
# Not on crates.io (the `borg` name there is an unrelated crate).
borg = { git = "https://github.com/joydle/borg.git", tag = "v1.0.0" }
async-trait = "0.1"  # the derive macros expand to #[async_trait]
```

Rust 1.88+, Tokio.

## One agent

```rust
use borg::prelude::*;

#[derive(BorgAgentDerive)]
#[borg(id = "reviewer", model = "anthropic/claude-haiku-5-5",
       max_tokens = 800, system = "Review code quality.")]
struct Reviewer;

impl Reviewer {
    fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
        ctx.handoff.get_as("code")
    }
}

let suggestions: Vec<Suggestion> =
    run_one_typed("review", Reviewer, ctx, handoff).await?;
```

borg ships no provider client: implement `AgentLlmClient` once for the
API you use. The examples run on the built-in `FakeLlmClient`, so they
need no API key.

## A pipeline

```rust
use borg::prelude::*;

let pipeline = seq([
    step(Planner),
    par([step(Coder), step(TestWriter), step(DocWriter)]),
    repeat(8, |v: &Verdict| v.token == "PASS", || seq([
        step(Builder),
        step(BuildFixer),
    ])),
    step(Reviewer),
]);
```

Plan, fan out three workers, loop build→fix until `PASS` (at most 8
rounds), review. Cancelling stops every nested agent and in-flight LLM
call; a panicking child doesn't take its siblings down.

## What you get

- **Composition primitives**: `Single`, `Sequential`, `Parallel`,
  `Branch`, `Fallback`, `LoopUntil`, `Agent`, `FallbackLadder`,
  `Heartbeat`, `Deferred`, written with the `dsl` builders or the
  `pipeline!` macro.
- **Supervision**: per-child cancel, panic isolation, restart with
  backoff and jitter, restart-storm meltdown, sliding-window circuit
  breaker, fallback ladders across models.
- **Guardrails**: behavioral rules at every agent-loop boundary (loops,
  repeated tool calls, text-only turns) with `Warn` / `Throttle` /
  `Halt` verdicts.
- **Budgets**: admission before every dispatch, sub-budgets,
  cache-aware pricing.
- **Tools and MCP**: object-safe `Tool` trait, derive macros, opt-in
  parallel tool calls; any stdio MCP server becomes tools.
- **Vision**: an agent can pass images on the same request as its
  tools.
- **Long-lived sessions** (`borg-loop`): a streaming actor is the
  session's main loop. Each turn streams live, `run_session_loop` keeps
  it going until it parks, and the session checkpoints and resumes
  across restarts. Drop a turn mid-stream and the actor parks with its
  snapshot. See the `streaming_session` example.
- **Pause and resume**: an agent asks for a human decision, the run
  suspends and persists, and it resumes later with the `Decision`
  (approve, reject or edit). See the `human_in_the_loop` example.
- **Checkpoints**: `execute_resumable` snapshots a run (each loop
  iteration by default, or every step) to a `CheckpointStore`. After a
  crash the same call skips finished work, and refuses a snapshot taken
  from a differently shaped pipeline. Back the three-method store with
  any database.
- **Lifecycle**: streaming compositions, background work with
  heartbeats.
- **Observability**: a full `tracing` span tree (session → composition
  → step → turn → tool) and JSONL event recording.
- **Testing**: `FakeLlmClient`, `ScriptedAgent` and an eval harness
  (feature `testing`).

## Crates

| Crate | |
|---|---|
| [`borg`](crates/borg) | Core runtime: compositions, supervisor, tools, checkpoints, streaming |
| [`borg-macros`](crates/borg-macros) | Agent and tool derives, `#[borg_instrument]` (re-exported by `borg`) |
| [`borg-loop`](crates/borg-loop) | Long-lived, resumable actor sessions |
| [`borg-tracing`](crates/borg-tracing) | `tracing` span hierarchy for every borg event |
| [`borg-mcp`](crates/borg-mcp) | MCP servers as borg tools |
| [`borg-sandbox`](crates/borg-sandbox) | Subprocess tool behind a pluggable `SandboxBackend` |
| [`borg-tools`](crates/borg-tools) | File-op tools and an approval helper |

## Learn more

- [Website](https://joydle.github.io/borg/)
- Examples, all runnable without an API key:
  [`borg`](crates/borg/examples) (`cargo run -p borg --example dogfood --features testing`)
  and [`borg-loop`](crates/borg-loop/examples) (`cargo run -p borg-loop --example streaming_session`)
- API docs: `cargo doc --open -p borg`

borg is installed from GitHub, not crates.io, where the `borg` name
belongs to an unrelated crate. Pin a release tag rather than tracking
`main`.

## Contributing

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

MSRV is Rust 1.88. Open an issue before changing public API. Use
conventional-commit PR titles (`feat:`, `fix:`, `feat!:`); release notes
are generated from them. A release is a `vX.Y.Z` tag matching the
workspace version in `Cargo.toml`.

## Security

borg makes it easy to give a model subprocess and file access, so
sandboxing is on you; see [SECURITY.md](SECURITY.md). Report
vulnerabilities privately, not as issues.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT) at your option. Contributions are dual-licensed the
same way.
