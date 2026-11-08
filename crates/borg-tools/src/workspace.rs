//! Workspace abstraction — what tools mutate instead of the filesystem.
//!
//! Agent tools mutate an in-memory map (a `HashMap<PathBuf, String>` plus a
//! change log); commit-to-disk happens elsewhere. The tool layer stays a
//! pure-data transformation and the consumer controls when (and whether) bytes
//! hit disk.
//!
//! Exposed as a trait so the file-op tools don't hard-code an implementation.
//! The default impl is [`InMemoryWorkspace`]; future impls can back the
//! workspace with a sandboxed filesystem for local CLI agents.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::sync::Mutex;

/// Errors a [`Workspace`] impl can return.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WorkspaceError {
    /// The backing storage rejected the operation (sandbox denied
    /// access, disk full, network store unavailable, …).
    #[error("workspace backend error: {message}")]
    Backend { message: String },
}

/// A single mutation in the session change log.
///
/// Entries are ordered by application. `read_file_diff` diffs the FIRST
/// recorded write content against the current file content.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FileChange {
    /// A `write` created or replaced the file. `content` is the post-write
    /// state, not a delta; `read_file_diff` uses the first such entry for a
    /// path as the diff baseline.
    Written {
        path: PathBuf,
        content_hash: u64,
        content: String,
    },
    /// A `delete` removed the file from the workspace.
    Deleted { path: PathBuf },
}

/// Trait the file-op tools mutate.
///
/// Implementations own the entire read/write contract — content storage, change
/// tracking, and any sandboxing or path validation. The trait stays narrow so
/// consumers can back it with a HashMap, a sandboxed FS, or a remote store.
///
/// Shareable as `Arc<dyn Workspace>` — every method takes `&self`.
#[async_trait]
pub trait Workspace: Send + Sync {
    /// Read the current contents of `path`, or `None` if absent.
    async fn read(&self, path: &Path) -> Result<Option<String>, WorkspaceError>;

    /// Write `content` to `path`, creating or replacing. Implementations SHOULD
    /// treat an identical-content write as a no-op for the change log (no
    /// `FileChange::Written` recorded).
    async fn write(&self, path: &Path, content: String) -> Result<(), WorkspaceError>;

    /// Remove `path`. Idempotent — deleting a non-existent file is OK and
    /// records no change.
    async fn delete(&self, path: &Path) -> Result<(), WorkspaceError>;

    /// Snapshot of the session change log in order of application. Drives
    /// `read_file_diff`.
    async fn changes(&self) -> Vec<FileChange>;
}

/// In-memory implementation — the default backing for tests and for consumers
/// that defer commit-to-disk to a separate layer.
///
/// A `HashMap<PathBuf, String>` and an append-only change log behind a single
/// `tokio::sync::Mutex`. No `.await` happens while the guard is held.
pub struct InMemoryWorkspace {
    inner: Mutex<Inner>,
}

struct Inner {
    files: HashMap<PathBuf, String>,
    changes: Vec<FileChange>,
}

impl Default for InMemoryWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryWorkspace {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                files: HashMap::new(),
                changes: Vec::new(),
            }),
        }
    }
}

/// djb2 — cheap and stable across process restarts. Collision-tolerant: it's
/// only a fast-path filter for the "did the content actually change?" check in
/// `write`, which still compares contents on a hash match.
fn djb2(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

#[async_trait]
impl Workspace for InMemoryWorkspace {
    async fn read(&self, path: &Path) -> Result<Option<String>, WorkspaceError> {
        let inner = self.inner.lock().await;
        Ok(inner.files.get(path).cloned())
    }

    async fn write(&self, path: &Path, content: String) -> Result<(), WorkspaceError> {
        let mut inner = self.inner.lock().await;
        let hash = djb2(&content);
        let dedup =
            matches!(inner.files.get(path), Some(prev) if djb2(prev) == hash && prev == &content);
        if !dedup {
            inner.changes.push(FileChange::Written {
                path: path.to_path_buf(),
                content_hash: hash,
                content: content.clone(),
            });
        }
        inner.files.insert(path.to_path_buf(), content);
        Ok(())
    }

    async fn delete(&self, path: &Path) -> Result<(), WorkspaceError> {
        let mut inner = self.inner.lock().await;
        if inner.files.remove(path).is_some() {
            inner.changes.push(FileChange::Deleted {
                path: path.to_path_buf(),
            });
        }
        Ok(())
    }

    async fn changes(&self) -> Vec<FileChange> {
        let inner = self.inner.lock().await;
        inner.changes.clone()
    }
}
