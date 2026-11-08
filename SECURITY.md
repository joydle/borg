# Security

Report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/joydle/borg/security/advisories/new),
not public issues. Include the affected crate and version, steps to
reproduce, and the impact. Expect an acknowledgement within 72 hours.

Only the latest release line (currently 1.0.x) receives fixes.

## Threat model

borg makes it easy to hand a model local resources. That's the point,
and it's also the risk: treat everything a model sends to a tool as
attacker-controlled (prompt injection).

In scope: a panic escaping the supervisor, cancellation that doesn't
propagate, a budget that can be bypassed, a guarantee below that doesn't
hold, or a release that ships a dependency with a known advisory.

### `borg-sandbox` is not a sandbox

`TokioBackend` runs ordinary subprocesses with your privileges: the child
can read and write every file you can and reach the network. To run
untrusted code, implement `SandboxBackend` over a real isolation layer:

- [gVisor](https://gvisor.dev/) or [Firecracker](https://firecracker-microvm.github.io/) microVMs
- [bubblewrap](https://github.com/containers/bubblewrap) or [nsjail](https://github.com/google/nsjail)
- hosted: E2B, Modal, Daytona, Cloudflare Sandbox
- [agentjail](https://github.com/joydle/agentjail) (beta): lightweight
  Linux sandboxes for AI agents; check its own security notes before
  relying on it

What `SubprocessTool` and `TokioBackend` do guarantee:

- **No inherited environment.** The child gets only a passthrough list
  (default `PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`, `TERM`, plus the
  Windows essentials). Your API keys don't reach it unless you pass them.
- **Model env is opt-in.** The model's `env` is ignored unless you
  allowlist keys, and `PATH`, `LD_*`, `DYLD_*`, `BASH_ENV`, `NODE_OPTIONS`,
  `PYTHONPATH`, `GIT_SSH_COMMAND`, `GIT_CONFIG_*` and similar are never
  accepted from it.
- **Exact allowlist.** `SubprocessTool::builder` requires a program
  allowlist and a workspace root. Names resolve to absolute paths on your
  search path when the tool is built; anything else with a path separator
  is rejected. Allowing a shell or an interpreter still allows arbitrary
  code.
- **Workdir inside the root**, after canonicalization. stdin is
  `/dev/null`.
- **Process-group kill** on Unix on timeout, cancel, drop, and when the
  direct child exits. On Windows only the direct child is killed.
- **Limits:** 30 s default timeout; output read in bounded chunks, 64 KiB
  per line and 1 MiB in total by default.
- **Bad model input is a tool failure**, never an error that aborts the run.

`SubprocessTool::unrestricted` drops the allowlist and workspace root; use
it only for trusted input. Most sandbox tests are Unix-only.

### `borg-mcp`

- An MCP config names an executable that runs as you. Never load configs
  you don't trust.
- Servers start from a cleared environment: a minimal passthrough
  (`PATH`, `HOME`, `USER`, locale and temp vars, plus the Windows
  essentials), names you add with `pass_env`, and the server's declared
  `env` (its `${VAR}` references resolve from your environment at load
  time). `inherit_env(true)` opts back into the full environment.
- Tool names, descriptions, schemas and results come from the server and
  reach the model verbatim, and `refresh()` can change them (tool
  poisoning). Connect only trusted servers, allowlist tools, and gate
  side-effecting tools behind approval.
- Each call has a timeout (default 60 s), honours cancellation, and has a
  result cap (default 1 MiB, applied after decoding).

### `FileEventRecorder`

Logs contain prompts, tool arguments and results. On Unix it creates
directories `0700` and files `0600` and refuses to write through a
symlink. It doesn't validate a directory that already exists, so point it
at a directory only you can write to.

### File tools

The file tools in the docs are illustrative. Allowlist and normalize paths
before writing anything to disk.
