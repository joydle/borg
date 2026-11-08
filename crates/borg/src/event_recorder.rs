//! Structured file-based [`EventEmitter`] — `manifest.json` (atomic
//! session summary) + `events.jsonl` (append-only event log).
//!
//! ## Layout
//!
//! Given `dir = /var/borg/sessions/<id>/`:
//!
//! ```text
//! dir/
//!   manifest.json   — session summary (atomic write via .tmp + rename)
//!   events.jsonl    — newline-delimited BorgEvent JSON, append-only
//!   events.1.jsonl  — rotated event log (rotation_threshold_bytes hit)
//!   events.2.jsonl  — older rotation
//! ```
//!
//! The kernel pattern for structured event tracing: open a dir, append
//! JSON per event, atomically rewrite the manifest at session terminal.
//!
//! Scope: this is local-file, lifecycle-event tracing — not a database
//! (ship to S3/Loki/BigQuery for queries/retention), not distributed
//! (each node writes its own dir), and not a replacement for span-level
//! [`tracing`](https://docs.rs/tracing).
//!
//! ## File safety
//!
//! Event logs carry prompts, tool arguments and tool results, so treat
//! them as sensitive. On Unix, directories the recorder creates are
//! `0700` and files are `0600`. Every file is opened without following
//! symlinks: a symlink planted at `events.jsonl`, `events.<N>.jsonl` or
//! `manifest.json.tmp` makes the open (or that rotation / manifest write)
//! fail rather than write through it. A directory that already exists is
//! used as is, so put sessions under a directory only you can write to,
//! not straight under a shared `/tmp`.
//!
//! Reopening an existing directory appends to `events.jsonl` and
//! continues rotation numbering after the highest `events.<N>.jsonl`
//! already there, so earlier rotated logs are never overwritten.
//! Rotation itself never replaces an existing file either: the log is
//! hard-linked to the first free `events.<N>.jsonl` and then unlinked,
//! so two recorders sharing a directory skip each other's rotations
//! instead of clobbering them. (Sharing a directory still interleaves
//! their events; give each session its own.) A FIFO or other non-regular
//! file at `events.jsonl` is refused, never opened for writing (which
//! could block forever).
//!
//! ## Worked example
//!
//! ```ignore
//! use borg::{FileEventRecorder, ExecCtx, in_memory_budget};
//! use std::sync::Arc;
//!
//! // A per-user, per-process location (or a path your app owns).
//! let dir = std::env::temp_dir()
//!     .join(format!("borg-{}", std::process::id()))
//!     .join("session-42");
//! let recorder = Arc::new(
//!     FileEventRecorder::new(&dir, "session-42").expect("recorder open"),
//! );
//! let ctx = ExecCtx::new(llm, in_memory_budget(40_000), recorder.clone(), cancel, "session-42");
//! let outcome = execute(pipeline, ctx).await;
//!
//! // Always finalize so the manifest gets the terminal_status set.
//! recorder.finalize(if outcome.is_ok() { "completed" } else { "failed" })
//!     .expect("finalize");
//! ```

use crate::events::{BorgEvent, EventEmitter};
use async_trait::async_trait;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default per-line byte cap. Lines longer than this are replaced
/// with a truncation marker carrying the original byte length —
/// prevents a runaway tool result from blowing the file up.
pub const DEFAULT_LINE_CAP_BYTES: usize = 16 * 1024;

/// Default rotation threshold: rename `events.jsonl` →
/// `events.<N>.jsonl` and open a fresh log once the active log hits
/// this size.
pub const DEFAULT_ROTATION_THRESHOLD_BYTES: u64 = 10 * 1024 * 1024;

/// Filesystem-backed event recorder. Append-only JSONL events +
/// atomic manifest. Cheap to share via `Arc<dyn EventEmitter>`.
///
/// Synchronized via `std::sync::Mutex`; writes happen on the runtime
/// thread. Fine for a low event rate (10s/sec) — a workload sustaining
/// kilo-events/sec wants a buffered variant.
pub struct FileEventRecorder {
    state: Mutex<RecorderState>,
}

struct RecorderState {
    /// Output directory. Created on construction if missing.
    dir: PathBuf,
    /// Session identifier surfaced in the manifest.
    session_id: String,
    /// Wall-clock at recorder construction.
    started_at: SystemTime,
    /// Active log file, buffered. Opened append-only; replaced on
    /// rotation. The `BufWriter` collapses one `write(2)` per emit into
    /// one per buffer flush (default 8 KiB).
    ///
    /// Callers needing durable visibility must call
    /// [`FileEventRecorder::flush`]; the recorder also flushes on Drop,
    /// rotation, and [`FileEventRecorder::finalize`].
    events_file: std::io::BufWriter<std::fs::File>,
    /// Path of the active log file (`<dir>/events.jsonl`).
    events_path: PathBuf,
    /// Total events successfully written across the session
    /// (including post-rotation events).
    events_written: u64,
    /// Bytes written to the *current* active log file (resets after
    /// rotation).
    current_log_bytes: u64,
    /// Number of completed rotations. The next rotation will produce
    /// `events.{rotation_count + 1}.jsonl`.
    rotation_count: u32,
    /// Line-byte cap. Set via [`FileEventRecorder::with_line_cap`].
    line_cap_bytes: usize,
    /// Rotation threshold. Set via
    /// [`FileEventRecorder::with_rotation_threshold`].
    rotation_threshold_bytes: u64,
    /// Manifest path (`<dir>/manifest.json`).
    manifest_path: PathBuf,
    /// Terminal status — `None` while the session is active, set by
    /// [`FileEventRecorder::finalize`].
    terminal_status: Option<String>,
}

impl FileEventRecorder {
    /// Open (or create) `dir` and write the initial manifest. Errors
    /// surface a `std::io::Error` so the caller can fail fast on
    /// permission / disk-full / path-issue problems, and when
    /// `events.jsonl` is a symlink (see the module's "File safety").
    ///
    /// Reopening a directory appends to the existing `events.jsonl` and
    /// resumes rotation after the highest existing `events.<N>.jsonl`.
    pub fn new(dir: impl AsRef<Path>, session_id: impl Into<String>) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let session_id = session_id.into();
        create_private_dir_all(&dir)?;

        let events_path = dir.join("events.jsonl");
        let file = open_private_append(&events_path)?;
        let current_log_bytes = file.metadata()?.len();
        let events_file = std::io::BufWriter::new(file);
        let manifest_path = dir.join("manifest.json");
        let rotation_count = highest_rotation_index(&dir)?;

        let state = RecorderState {
            dir,
            session_id,
            started_at: SystemTime::now(),
            events_file,
            events_path,
            events_written: 0,
            current_log_bytes,
            rotation_count,
            line_cap_bytes: DEFAULT_LINE_CAP_BYTES,
            rotation_threshold_bytes: DEFAULT_ROTATION_THRESHOLD_BYTES,
            manifest_path,
            terminal_status: None,
        };

        let recorder = Self {
            state: Mutex::new(state),
        };
        // Best-effort initial manifest; the next emit/finalize retries.
        let _ = recorder.write_manifest_locked(&recorder.state.lock().unwrap());
        Ok(recorder)
    }

    /// Build an `Arc<dyn EventEmitter>` ready to plug into
    /// [`crate::ExecCtx::new`].
    pub fn shared(
        dir: impl AsRef<Path>,
        session_id: impl Into<String>,
    ) -> std::io::Result<std::sync::Arc<dyn EventEmitter>> {
        Ok(std::sync::Arc::new(Self::new(dir, session_id)?))
    }

    /// Override the per-line byte cap. Lines longer than this are
    /// replaced with a truncation marker.
    #[must_use]
    pub fn with_line_cap(self, bytes: usize) -> Self {
        self.state.lock().unwrap().line_cap_bytes = bytes;
        self
    }

    /// Override the rotation threshold (bytes per active log file).
    #[must_use]
    pub fn with_rotation_threshold(self, bytes: u64) -> Self {
        self.state.lock().unwrap().rotation_threshold_bytes = bytes;
        self
    }

    /// Record the session's terminal status in the manifest. Idempotent
    /// (a later call overwrites). Call once at session end, typically
    /// from a [`crate::TerminalGuard`] callback. Flushes the events
    /// buffer first so in-buffer events are durable before the terminal
    /// status becomes observable.
    pub fn finalize(&self, terminal_status: impl Into<String>) -> std::io::Result<()> {
        let mut state = self.state.lock().unwrap();
        // Best-effort flush so events.jsonl matches the manifest count on disk.
        if let Err(e) = state.events_file.flush() {
            tracing::warn!("FileEventRecorder: finalize flush failed: {e}");
        }
        state.terminal_status = Some(terminal_status.into());
        self.write_manifest_locked(&state)
    }

    /// Force the in-memory event buffer to disk. Idempotent.
    ///
    /// Callers inspecting `events.jsonl` out-of-band (tests, log
    /// shippers) must call this — the `BufWriter` holds writes until the
    /// buffer fills (default 8 KiB) or the recorder is dropped/rotated/finalized.
    pub fn flush(&self) -> std::io::Result<()> {
        let mut state = self.state.lock().unwrap();
        state.events_file.flush()
    }

    /// Number of events written so far. Cheap.
    pub fn events_written(&self) -> u64 {
        self.state.lock().unwrap().events_written
    }

    /// Path to the active events log.
    pub fn events_path(&self) -> PathBuf {
        self.state.lock().unwrap().events_path.clone()
    }

    /// Path to the manifest file.
    pub fn manifest_path(&self) -> PathBuf {
        self.state.lock().unwrap().manifest_path.clone()
    }

    /// Atomic manifest write: serialize to `<path>.tmp`, then rename.
    /// Mutex must be held by the caller.
    fn write_manifest_locked(&self, state: &RecorderState) -> std::io::Result<()> {
        let started_unix = state
            .started_at
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let manifest = serde_json::json!({
            "session_id": state.session_id,
            "started_at_unix": started_unix,
            "events_written": state.events_written,
            "current_log_bytes": state.current_log_bytes,
            "rotation_count": state.rotation_count,
            "terminal_status": state.terminal_status,
            "line_cap_bytes": state.line_cap_bytes,
            "rotation_threshold_bytes": state.rotation_threshold_bytes,
        });
        let body = serde_json::to_string_pretty(&manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = state.manifest_path.with_extension("json.tmp");
        // Unlink any leftover (or planted symlink) first; `remove_file`
        // removes the link itself, never its target. Then create the tmp
        // exclusively so nothing at that path is followed or truncated.
        match std::fs::remove_file(&tmp) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        let mut file = private_file_options()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        file.write_all(body.as_bytes())?;
        drop(file);
        // `rename` replaces the destination entry; it doesn't follow it.
        std::fs::rename(&tmp, &state.manifest_path)?;
        Ok(())
    }

    /// Rotate the active log to the next free `events.<N>.jsonl` and
    /// open a fresh one. Mutex must be held by the caller.
    ///
    /// Never replaces an existing file: the log is hard-linked to the new
    /// name (which fails if the name is taken, then the next index is
    /// tried) and only then unlinked from `events.jsonl`. If another
    /// recorder on the same directory already rotated our log away, this
    /// one just reopens the current `events.jsonl`.
    fn rotate_locked(&self, state: &mut RecorderState) -> std::io::Result<()> {
        // Flush before the link so the rotated file is byte-complete;
        // an unflushed tail would be discarded by the next buffer swap.
        if let Err(e) = state.events_file.flush() {
            tracing::warn!("FileEventRecorder: pre-rotate flush failed: {e}");
        }
        if still_at_path(state.events_file.get_ref(), &state.events_path)? {
            state.rotation_count =
                link_to_free_rotation(&state.dir, &state.events_path, state.rotation_count)?;
            std::fs::remove_file(&state.events_path)?;
        }
        let file = open_private_append(&state.events_path)?;
        state.current_log_bytes = file.metadata()?.len();
        state.events_file = std::io::BufWriter::new(file);
        Ok(())
    }

    /// Render the truncation marker line. Stable wire shape.
    fn truncation_line(kind: &str, original_bytes: usize) -> String {
        let v = serde_json::json!({
            "kind": kind,
            "truncated": true,
            "original_bytes": original_bytes,
        });
        v.to_string()
    }
}

/// `OpenOptions` that create files `0600` on Unix.
fn private_file_options() -> OpenOptions {
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut opts = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts
}

/// `create_dir_all`, with newly created directories `0700` on Unix.
/// Existing directories keep their permissions.
fn create_private_dir_all(dir: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

fn symlink_refused(path: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "FileEventRecorder: refusing to open `{}`: it is a symlink or not a regular file",
            path.display()
        ),
    )
}

/// Open `path` for appending without following a symlink at `path`.
///
/// A new file is created exclusively (`O_CREAT|O_EXCL` never follows a
/// symlink) with mode `0600`. An existing path must be a regular file
/// (its own `lstat`, so a symlink, FIFO, socket or device is refused
/// before any open); then [`open_existing_checked`] opens it and checks
/// the handle. Opening in append mode never truncates, so a refused
/// target is left untouched.
fn open_private_append(path: &Path) -> std::io::Result<std::fs::File> {
    match private_file_options()
        .append(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => return Ok(file),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let link_meta = std::fs::symlink_metadata(path)?;
    if !link_meta.file_type().is_file() {
        return Err(symlink_refused(path));
    }
    open_existing_checked(path)
}

/// Open an existing log for appending and prove the handle is the
/// regular file at `path`, closing the window between the `lstat` check
/// and the open.
///
/// Opened read+write: on a FIFO swapped in after the check, a write-only
/// open blocks until a reader shows up (forever, on a planted FIFO),
/// while a read-write open returns at once (Linux, macOS, the BSDs) and
/// is then refused here, with no libc flags needed. On Unix the handle's
/// `dev`/`ino` must also match the path's own `lstat`, so a symlink
/// swapped in is caught before anything is written.
fn open_existing_checked(path: &Path) -> std::io::Result<std::fs::File> {
    let file = OpenOptions::new().read(true).append(true).open(path)?;
    let opened = file.metadata()?;
    if !opened.file_type().is_file() {
        return Err(symlink_refused(path));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let at_path = std::fs::symlink_metadata(path)?;
        if !at_path.file_type().is_file()
            || opened.dev() != at_path.dev()
            || opened.ino() != at_path.ino()
        {
            return Err(symlink_refused(path));
        }
    }
    Ok(file)
}

/// Whether `path` still names the open `file` (Unix: same `dev`/`ino`).
/// `false` once another recorder rotated it away. Elsewhere, `true`
/// while the path exists.
fn still_at_path(file: &std::fs::File, path: &Path) -> std::io::Result<bool> {
    let at_path = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let opened = file.metadata()?;
        Ok(at_path.file_type().is_file()
            && opened.dev() == at_path.dev()
            && opened.ino() == at_path.ino())
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        Ok(at_path.file_type().is_file())
    }
}

/// Most indexes [`link_to_free_rotation`] tries before giving up.
const MAX_ROTATION_ATTEMPTS: u32 = 1024;

/// Hard-link `log` to the first free `events.<N>.jsonl` with `N` above
/// `after`, never replacing an existing file. Returns `N`.
fn link_to_free_rotation(dir: &Path, log: &Path, after: u32) -> std::io::Result<u32> {
    let mut index = after;
    for _ in 0..MAX_ROTATION_ATTEMPTS {
        index = index
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("FileEventRecorder: rotation index overflow"))?;
        let rotated = dir.join(format!("events.{index}.jsonl"));
        match std::fs::hard_link(log, &rotated) {
            Ok(()) => return Ok(index),
            // Taken (another recorder rotated meanwhile): skip past the
            // highest index on disk rather than probing one by one.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                index = index.max(highest_rotation_index(dir)?);
            }
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other(
        "FileEventRecorder: no free rotation index",
    ))
}

/// Highest `N` among `events.<N>.jsonl` files in `dir` (0 if none), so
/// a reopened recorder rotates to `N + 1` instead of overwriting.
fn highest_rotation_index(dir: &Path) -> std::io::Result<u32> {
    let mut highest = 0u32;
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        let index = name
            .strip_prefix("events.")
            .and_then(|rest| rest.strip_suffix(".jsonl"))
            .and_then(|n| n.parse::<u32>().ok());
        if let Some(n) = index {
            highest = highest.max(n);
        }
    }
    Ok(highest)
}

#[async_trait]
impl EventEmitter for FileEventRecorder {
    async fn emit(&self, event: Arc<BorgEvent>) {
        let mut state = match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                tracing::warn!("FileEventRecorder mutex poisoned; reusing inner state");
                poisoned.into_inner()
            }
        };

        let json = match serde_json::to_string(&*event) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("FileEventRecorder: serialize {} failed: {e}", event.kind());
                return;
            }
        };

        let line = if json.len() > state.line_cap_bytes {
            Self::truncation_line(event.kind(), json.len())
        } else {
            json
        };

        if let Err(e) = writeln!(state.events_file, "{line}") {
            tracing::warn!("FileEventRecorder: write failed: {e}");
            return;
        }
        state.events_written = state.events_written.saturating_add(1);
        let bytes_written = (line.len() as u64).saturating_add(1); // +1 for newline
        state.current_log_bytes = state.current_log_bytes.saturating_add(bytes_written);

        // Rotate if threshold hit. Best-effort; failures are logged, not surfaced.
        if state.current_log_bytes >= state.rotation_threshold_bytes {
            if let Err(e) = self.rotate_locked(&mut state) {
                tracing::warn!("FileEventRecorder: rotation failed: {e}");
            }
        }
    }
}

impl std::fmt::Debug for FileEventRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("FileEventRecorder")
            .field("dir", &state.dir)
            .field("session_id", &state.session_id)
            .field("events_written", &state.events_written)
            .field("current_log_bytes", &state.current_log_bytes)
            .field("rotation_count", &state.rotation_count)
            .field("terminal_status", &state.terminal_status)
            .finish()
    }
}

impl Drop for FileEventRecorder {
    fn drop(&mut self) {
        // Flush on drop so a recorder dropped without finalize() doesn't
        // lose its event-log tail. Errors ignored — Drop must not panic.
        if let Ok(mut state) = self.state.lock() {
            let _ = state.events_file.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompositionShape;
    use std::io::{BufRead, BufReader};

    /// Lightweight tempdir under `std::env::temp_dir()`, cleaned up on
    /// Drop. Avoids pulling in `tempfile` just for tests.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("borg-event-recorder-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("create tempdir");
            TempDir(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn read_lines(p: &Path) -> Vec<String> {
        let f = std::fs::File::open(p).expect("open events.jsonl");
        BufReader::new(f)
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("read lines")
    }

    fn read_manifest(p: &Path) -> serde_json::Value {
        let body = std::fs::read_to_string(p).expect("read manifest");
        serde_json::from_str(&body).expect("parse manifest")
    }

    fn step_started(name: &str) -> Arc<BorgEvent> {
        Arc::new(BorgEvent::StepStarted {
            composition_id: "test".into(),
            step_name: name.into(),
            position: 0,
            turn_number: 0,
            model: "fake/model".into(),
        })
    }

    #[tokio::test]
    async fn new_creates_dir_and_writes_initial_manifest() {
        let dir = TempDir::new("init");
        let rec = FileEventRecorder::new(dir.path(), "session-1").unwrap();

        assert!(dir.path().join("events.jsonl").exists());
        assert!(dir.path().join("manifest.json").exists());

        let manifest = read_manifest(&rec.manifest_path());
        assert_eq!(manifest["session_id"], "session-1");
        assert_eq!(manifest["events_written"], 0);
        assert!(manifest["terminal_status"].is_null());
    }

    #[tokio::test]
    async fn emit_writes_jsonl_line_per_event() {
        let dir = TempDir::new("jsonl");
        let rec = FileEventRecorder::new(dir.path(), "sess").unwrap();

        rec.emit(step_started("a")).await;
        rec.emit(step_started("b")).await;

        rec.flush().unwrap();
        let lines = read_lines(&rec.events_path());
        assert_eq!(lines.len(), 2);
        let parsed_a: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed_a["kind"], "step_started");
        assert_eq!(parsed_a["step_name"], "a");
        let parsed_b: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(parsed_b["step_name"], "b");
        assert_eq!(rec.events_written(), 2);
    }

    #[tokio::test]
    async fn finalize_sets_terminal_status_in_manifest() {
        let dir = TempDir::new("finalize");
        let rec = FileEventRecorder::new(dir.path(), "sess").unwrap();
        rec.emit(step_started("a")).await;
        rec.finalize("completed").unwrap();

        let manifest = read_manifest(&rec.manifest_path());
        assert_eq!(manifest["terminal_status"], "completed");
        assert_eq!(manifest["events_written"], 1);
    }

    #[tokio::test]
    async fn line_cap_truncates_oversize_events() {
        let dir = TempDir::new("linecap");
        let rec = FileEventRecorder::new(dir.path(), "sess")
            .unwrap()
            .with_line_cap(64); // forces truncation

        let big = step_started(&"x".repeat(200));
        rec.emit(big).await;

        rec.flush().unwrap();
        let lines = read_lines(&rec.events_path());
        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed["kind"], "step_started");
        assert_eq!(parsed["truncated"], true);
        assert!(parsed["original_bytes"].as_u64().unwrap() > 64);
    }

    #[tokio::test]
    async fn rotation_renames_and_opens_fresh_log() {
        let dir = TempDir::new("rotate");
        let rec = FileEventRecorder::new(dir.path(), "sess")
            .unwrap()
            // Force rotation after every event (each line is > 64 bytes).
            .with_rotation_threshold(64);

        rec.emit(step_started("first")).await;
        rec.emit(step_started("second")).await;

        assert!(dir.path().join("events.1.jsonl").exists());

        // The active log reopens fresh per rotation, so the recorder's
        // count is authoritative, not any single file's line count.
        assert_eq!(rec.events_written(), 2);

        // Active + every rotated file should sum to events written.
        // Flush first so the active log is byte-complete.
        rec.flush().unwrap();
        let mut total_lines = read_lines(&dir.path().join("events.jsonl")).len();
        for n in 1..=rec.state.lock().unwrap().rotation_count {
            let p = dir.path().join(format!("events.{n}.jsonl"));
            if p.exists() {
                total_lines += read_lines(&p).len();
            }
        }
        assert_eq!(total_lines, 2);

        rec.finalize("completed").unwrap();
        let m = read_manifest(&rec.manifest_path());
        assert!(m["rotation_count"].as_u64().unwrap() >= 1);
        assert_eq!(m["events_written"], 2);
    }

    #[tokio::test]
    async fn emits_composition_started_with_shape() {
        let dir = TempDir::new("composition");
        let rec = FileEventRecorder::new(dir.path(), "sess").unwrap();

        rec.emit(Arc::new(BorgEvent::CompositionStarted {
            composition_id: "root".into(),
            parent_id: None,
            shape: CompositionShape::AgentLoop,
            branch_count: 1,
        }))
        .await;

        rec.flush().unwrap();
        let lines = read_lines(&rec.events_path());
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed["kind"], "composition_started");
        assert_eq!(parsed["shape"], "agent_loop");
    }

    #[tokio::test]
    async fn reopen_continues_rotation_numbering() {
        // Reopening a dir must not reset the rotation index, or the
        // next rotation overwrites `events.1.jsonl` and loses its logs.
        let dir = TempDir::new("reopen-rotate");
        {
            let rec = FileEventRecorder::new(dir.path(), "sess")
                .unwrap()
                .with_rotation_threshold(64);
            rec.emit(step_started("first")).await;
            rec.emit(step_started("second")).await;
        }
        assert!(dir.path().join("events.1.jsonl").exists());
        assert!(dir.path().join("events.2.jsonl").exists());
        let first_before = std::fs::read_to_string(dir.path().join("events.1.jsonl")).unwrap();

        let rec = FileEventRecorder::new(dir.path(), "sess")
            .unwrap()
            .with_rotation_threshold(64);
        rec.emit(step_started("third")).await;

        let first_after = std::fs::read_to_string(dir.path().join("events.1.jsonl")).unwrap();
        assert_eq!(first_before, first_after, "events.1.jsonl was overwritten");
        assert!(dir.path().join("events.3.jsonl").exists());
        let third = std::fs::read_to_string(dir.path().join("events.3.jsonl")).unwrap();
        assert!(third.contains("\"third\""), "got: {third}");
    }

    #[tokio::test]
    async fn reopen_counts_existing_active_log_toward_rotation() {
        let dir = TempDir::new("reopen-bytes");
        std::fs::write(dir.path().join("events.jsonl"), "x".repeat(100) + "\n").unwrap();
        let rec = FileEventRecorder::new(dir.path(), "sess")
            .unwrap()
            .with_rotation_threshold(128);
        // 101 existing bytes + one ~100-byte line crosses 128.
        rec.emit(step_started("a")).await;
        assert!(dir.path().join("events.1.jsonl").exists());
    }

    /// Make a FIFO at `path` (no libc: the `mkfifo` tool).
    #[cfg(unix)]
    fn mkfifo(path: &Path) {
        let status = std::process::Command::new("mkfifo")
            .arg(path)
            .status()
            .expect("run mkfifo");
        assert!(status.success(), "mkfifo failed");
    }

    /// Run `f` on a thread; `None` if it is still blocked after 5 s.
    #[cfg(unix)]
    fn within_5s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(std::time::Duration::from_secs(5)).ok()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_fifo_events_file() {
        let dir = TempDir::new("fifo-events");
        mkfifo(&dir.path().join("events.jsonl"));
        let path = dir.path().to_path_buf();
        let res = within_5s(move || FileEventRecorder::new(&path, "sess").map(|_| ()))
            .expect("opening a FIFO events.jsonl hung");
        let err = res.unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "got: {err}");
    }

    /// The open after the `lstat` check: a FIFO swapped in between them
    /// must be refused, not block the open (a write-only open of a FIFO
    /// waits for a reader).
    #[cfg(unix)]
    #[tokio::test]
    async fn checked_open_refuses_a_fifo_without_blocking() {
        let dir = TempDir::new("fifo-swap");
        let fifo = dir.path().join("events.jsonl");
        mkfifo(&fifo);
        let res = within_5s(move || open_existing_checked(&fifo).map(|_| ()))
            .expect("the checked open blocked on a FIFO");
        assert!(res.is_err(), "a FIFO must be refused");
    }

    #[tokio::test]
    async fn concurrent_recorders_never_overwrite_rotations() {
        // Regression: rotation renamed onto `events.<N>.jsonl`, and a
        // second recorder on the same dir (its own rotation count) renamed
        // over the first one's rotated log, losing it.
        let dir = TempDir::new("shared-rotate");
        let a = FileEventRecorder::new(dir.path(), "a")
            .unwrap()
            .with_rotation_threshold(64);
        let b = FileEventRecorder::new(dir.path(), "b")
            .unwrap()
            .with_rotation_threshold(64);
        for i in 0..6 {
            a.emit(step_started(&format!("a{i}"))).await;
            b.emit(step_started(&format!("b{i}"))).await;
        }
        a.flush().unwrap();
        b.flush().unwrap();
        let mut seen = Vec::new();
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if name.starts_with("events") && name.ends_with(".jsonl") {
                for line in read_lines(&path) {
                    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
                    seen.push(v["step_name"].as_str().unwrap().to_string());
                }
            }
        }
        seen.sort();
        let mut want: Vec<String> = (0..6)
            .flat_map(|i| [format!("a{i}"), format!("b{i}")])
            .collect();
        want.sort();
        assert_eq!(seen, want, "events lost to a clobbered rotation");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_symlinked_events_file() {
        let dir = TempDir::new("symlink-events");
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "precious\n").unwrap();
        let session = dir.path().join("session");
        std::fs::create_dir(&session).unwrap();
        std::os::unix::fs::symlink(&victim, session.join("events.jsonl")).unwrap();

        let err = FileEventRecorder::new(&session, "sess").unwrap_err();
        assert!(err.to_string().contains("symlink"), "got: {err}");
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious\n");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn manifest_write_does_not_follow_symlinked_tmp() {
        let dir = TempDir::new("symlink-manifest");
        let victim = dir.path().join("victim.txt");
        std::fs::write(&victim, "precious\n").unwrap();
        let session = dir.path().join("session");
        std::fs::create_dir(&session).unwrap();
        std::os::unix::fs::symlink(&victim, session.join("manifest.json.tmp")).unwrap();

        let rec = FileEventRecorder::new(&session, "sess").unwrap();
        rec.finalize("completed").unwrap();
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious\n");
        let m = read_manifest(&rec.manifest_path());
        assert_eq!(m["terminal_status"], "completed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn files_and_dirs_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new("perms");
        let session = dir.path().join("a").join("b");
        let rec = FileEventRecorder::new(&session, "sess")
            .unwrap()
            .with_rotation_threshold(64);
        rec.emit(step_started("first")).await;
        rec.finalize("completed").unwrap();

        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&session), 0o700);
        assert_eq!(mode(&dir.path().join("a")), 0o700);
        assert_eq!(mode(&session.join("events.jsonl")), 0o600);
        assert_eq!(mode(&session.join("events.1.jsonl")), 0o600);
        assert_eq!(mode(&session.join("manifest.json")), 0o600);
    }

    #[tokio::test]
    async fn manifest_is_atomic_on_concurrent_finalize() {
        // Concurrent finalizes must not leave a partial manifest.
        // Exercises the tmp+rename pattern (true atomicity is FS-dependent).
        let dir = TempDir::new("atomic");
        let rec = std::sync::Arc::new(FileEventRecorder::new(dir.path(), "sess").unwrap());

        let mut handles = Vec::new();
        for i in 0..16 {
            let r = rec.clone();
            handles.push(tokio::spawn(async move {
                r.finalize(format!("status-{i}")).unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // Parseable post-storm — proves no partial write landed in place.
        let m = read_manifest(&rec.manifest_path());
        assert!(m["terminal_status"].is_string());
    }
}
