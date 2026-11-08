//! File-op tools that mutate a [`Workspace`].
//!
//! - [`read_file`] — read a 1-indexed line range, with a clamping notice.
//! - [`replace_lines`] — replace a 1-indexed inclusive range, with a clamping
//!   notice.
//! - [`read_file_diff`] — unified diff between the first write of this session
//!   and the current content.
//! - [`delete_file`] — idempotent removal.
//!
//! All tools take an `&dyn Workspace` (see [`crate::workspace`]).
//!
//! These functions don't take an `AgentContext` — they're pure-data transforms
//! on a workspace. Consumers that want HITL coupling wrap the call in
//! `crate::approval::request_approval` first:
//!
//! ```ignore
//! let decision = borg_tools::approval::request_approval(
//!     ctx, "Replace lines 4-12?", Some("replace_lines"),
//! ).await?;
//! if matches!(decision, Decision::Approve) {
//!     borg_tools::fs::replace_lines(ws.as_ref(), &path, 4, 12, new).await?;
//! }
//! ```
//!
//! [`Workspace`]: crate::workspace::Workspace

use crate::workspace::{FileChange, Workspace, WorkspaceError};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors a file-op tool can return.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FileOpError {
    /// Path is not present in the workspace.
    #[error("file not found in workspace: {path}")]
    NotFound { path: PathBuf },
    /// Caller-provided line range is invalid (e.g. `first_line > last_line`,
    /// or `first_line == 0` — lines are 1-indexed).
    #[error("invalid line range: {reason}")]
    InvalidRange { reason: String },
    /// The backing workspace returned an error.
    #[error(transparent)]
    Workspace(#[from] WorkspaceError),
}

/// Result of a [`read_file`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReadFileResult {
    /// Numbered output, `"{line_number}\t{line_content}\n"` per line. The line
    /// numbers are the model's reference for follow-up [`replace_lines`] calls.
    pub content: String,
    /// First line included (1-indexed, inclusive).
    pub start_line: usize,
    /// Last line included (1-indexed, inclusive). Clamped to the actual last
    /// line if the requested `end_line` exceeded the file length, in which case
    /// [`Self::clamped`] is `true`.
    pub end_line: usize,
    /// True iff the requested range was clamped to the file's actual length —
    /// a hint that the agent's line indices were stale.
    pub clamped: bool,
}

/// Read a file from the workspace, returning a numbered-line view.
///
/// `start_line` defaults to `1`. `end_line` defaults to the file's
/// last line. Both are 1-indexed inclusive. Out-of-range bounds are
/// clamped with [`ReadFileResult::clamped`] set to `true`.
pub async fn read_file(
    ws: &dyn Workspace,
    path: &Path,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<ReadFileResult, FileOpError> {
    let content = ws.read(path).await?.ok_or_else(|| FileOpError::NotFound {
        path: path.to_path_buf(),
    })?;

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len().max(1);

    let start = start_line.unwrap_or(1).max(1);
    let end_requested = end_line.unwrap_or(total);
    let end = end_requested.min(total);
    let clamped = end_requested > total;

    if start > end {
        return Err(FileOpError::InvalidRange {
            reason: format!("start_line ({start}) > effective end_line ({end})"),
        });
    }

    let mut out = String::new();
    for (i, line) in lines
        .iter()
        .enumerate()
        .skip(start.saturating_sub(1))
        .take(end - start + 1)
    {
        out.push_str(&format!("{}\t{}\n", i + 1, line));
    }

    Ok(ReadFileResult {
        content: out,
        start_line: start,
        end_line: end,
        clamped,
    })
}

/// Result of a [`replace_lines`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReplaceLinesResult {
    /// True if the replacement was applied.
    pub succeeded: bool,
    /// True if `last_line` exceeded the file length and was clamped — a hint
    /// that the agent's indices were stale.
    pub clamped: bool,
    /// Effective last line replaced after clamping (1-indexed, inclusive).
    pub effective_last_line: usize,
}

/// Replace lines `[first_line..=last_line]` (1-indexed, inclusive) with
/// `replace`. `last_line` beyond EOF is clamped with a notice.
///
/// Returns `InvalidRange` if `first_line < 1` or `first_line >
/// effective_last_line`. Returns `NotFound` if the path is absent.
pub async fn replace_lines(
    ws: &dyn Workspace,
    path: &Path,
    first_line: usize,
    last_line: usize,
    replace: String,
) -> Result<ReplaceLinesResult, FileOpError> {
    if first_line < 1 {
        return Err(FileOpError::InvalidRange {
            reason: format!("first_line must be >= 1 (got {first_line})"),
        });
    }
    if last_line < first_line {
        return Err(FileOpError::InvalidRange {
            reason: format!("last_line ({last_line}) < first_line ({first_line})"),
        });
    }

    let existing = ws.read(path).await?.ok_or_else(|| FileOpError::NotFound {
        path: path.to_path_buf(),
    })?;

    // Split on '\n' (not `.lines()`) so a trailing newline is preserved as a
    // final empty segment; re-joined with '\n' below.
    let segments: Vec<&str> = existing.split('\n').collect();
    let total = segments.len();
    let effective_last = last_line.min(total);
    let clamped = last_line > total;

    if first_line > effective_last {
        return Err(FileOpError::InvalidRange {
            reason: format!(
                "first_line ({first_line}) past EOF (file has {} line(s))",
                total.saturating_sub(if existing.ends_with('\n') { 1 } else { 0 }),
            ),
        });
    }

    let before = segments[..first_line.saturating_sub(1)].join("\n");
    let after = segments[effective_last..].join("\n");

    // The replacement string carries its own trailing newline iff the caller
    // included one; just stitch.
    let mut next = String::new();
    if !before.is_empty() {
        next.push_str(&before);
        next.push('\n');
    }
    next.push_str(&replace);
    if !next.ends_with('\n') && !after.is_empty() {
        next.push('\n');
    }
    next.push_str(&after);

    ws.write(path, next).await?;

    Ok(ReplaceLinesResult {
        succeeded: true,
        clamped,
        effective_last_line: effective_last,
    })
}

/// Result of a [`read_file_diff`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DiffResult {
    /// Unified-diff text (3-line context radius). Empty when the file's session
    /// edits net to no change (e.g. a single write with no follow-up).
    pub text: String,
    /// True iff this file was created during the session (single recorded
    /// write, no prior baseline).
    pub is_creation: bool,
}

/// Render a unified diff between the first recorded write of `path`
/// this session and its current content. Errors with [`FileOpError::NotFound`]
/// if the path has no write history.
pub async fn read_file_diff(ws: &dyn Workspace, path: &Path) -> Result<DiffResult, FileOpError> {
    let changes = ws.changes().await;
    let first_write = changes.iter().find_map(|c| match c {
        FileChange::Written {
            path: p, content, ..
        } if p == path => Some(content.clone()),
        _ => None,
    });

    let Some(first_content) = first_write else {
        return Err(FileOpError::NotFound {
            path: path.to_path_buf(),
        });
    };

    let writes_count = changes
        .iter()
        .filter(|c| matches!(c, FileChange::Written { path: p, .. } if p == path))
        .count();
    let is_creation = writes_count == 1;

    let current = ws.read(path).await?.unwrap_or_default();

    if first_content == current {
        return Ok(DiffResult {
            text: String::new(),
            is_creation,
        });
    }

    let diff = similar::TextDiff::from_lines(&first_content, &current);
    let text = diff
        .unified_diff()
        .context_radius(3)
        .header("first-write", "current")
        .to_string();

    Ok(DiffResult { text, is_creation })
}

/// Remove `path` from the workspace. Idempotent — calling on an already-absent
/// path returns Ok and records no change, so it's safe for the model to call
/// freely.
pub async fn delete_file(ws: &dyn Workspace, path: &Path) -> Result<(), FileOpError> {
    ws.delete(path).await?;
    Ok(())
}
