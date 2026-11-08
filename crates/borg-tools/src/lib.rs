//! `borg-tools` — built-in tools for borg agents.
//!
//! - [`approval::request_approval`] — wrap [`borg::AgentContext::request_suspend`]
//!   with the common binary approve/reject/edit HITL shape.
//! - [`workspace`] — a [`workspace::Workspace`] trait plus an in-memory impl.
//!   Tools mutate this abstraction rather than the filesystem; the consumer
//!   controls when (and whether) bytes hit disk.
//! - [`fs`] — `read_file`, `replace_lines`, `read_file_diff`, `delete_file`,
//!   built on `Workspace`.
//!
//! Running programs is `borg-sandbox`'s job (`SubprocessTool`); network
//! tools are yours to write, since they need their own security model.

pub mod approval;
pub mod fs;
pub mod workspace;
