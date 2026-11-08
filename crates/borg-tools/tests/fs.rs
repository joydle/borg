//! Integration tests for the file-op tools:
//! [`read_file`], [`replace_lines`], [`read_file_diff`], [`delete_file`].
//!
//! Each tool operates on a [`Workspace`] (default impl [`InMemoryWorkspace`]):
//! 1-indexed line ranges, clamping notices on overflow, idempotent deletes,
//! hash-deduped writes.
//!
//! [`Workspace`]: borg_tools::workspace::Workspace

use borg_tools::fs::{
    delete_file, read_file, read_file_diff, replace_lines, FileOpError, ReplaceLinesResult,
};
use borg_tools::workspace::{InMemoryWorkspace, Workspace};
use std::path::PathBuf;
use std::sync::Arc;

const SRC: &str = "line 1\nline 2\nline 3\nline 4\nline 5\n";

async fn ws_with_src() -> Arc<dyn Workspace> {
    let ws = Arc::new(InMemoryWorkspace::new());
    ws.write(&PathBuf::from("src/lib.rs"), SRC.into())
        .await
        .unwrap();
    ws
}

// ── read_file ──────────────────────────────────────────────────────────

#[tokio::test]
async fn read_file_returns_full_content_with_line_numbers() {
    let ws = ws_with_src().await;
    let result = read_file(ws.as_ref(), &PathBuf::from("src/lib.rs"), None, None)
        .await
        .unwrap();

    assert_eq!(result.start_line, 1);
    assert_eq!(result.end_line, 5);
    // Output is 1-indexed and line-prefixed.
    assert!(result.content.contains("1\tline 1"));
    assert!(result.content.contains("5\tline 5"));
}

#[tokio::test]
async fn read_file_with_line_range_returns_just_that_range() {
    let ws = ws_with_src().await;
    let result = read_file(ws.as_ref(), &PathBuf::from("src/lib.rs"), Some(2), Some(4))
        .await
        .unwrap();

    assert_eq!(result.start_line, 2);
    assert_eq!(result.end_line, 4);
    assert!(result.content.contains("2\tline 2"));
    assert!(result.content.contains("4\tline 4"));
    assert!(!result.content.contains("line 1"));
    assert!(!result.content.contains("line 5"));
}

#[tokio::test]
async fn read_file_missing_path_returns_not_found_error() {
    let ws: Arc<dyn Workspace> = Arc::new(InMemoryWorkspace::new());
    let err = read_file(ws.as_ref(), &PathBuf::from("absent.rs"), None, None)
        .await
        .unwrap_err();
    assert!(matches!(err, FileOpError::NotFound { .. }));
}

#[tokio::test]
async fn read_file_out_of_range_end_line_clamps() {
    let ws = ws_with_src().await;
    let result = read_file(
        ws.as_ref(),
        &PathBuf::from("src/lib.rs"),
        Some(4),
        Some(999), // file only has 5 lines
    )
    .await
    .unwrap();

    assert_eq!(result.end_line, 5);
    assert!(result.clamped, "end_line=999 on a 5-line file must clamp");
}

// ── replace_lines ──────────────────────────────────────────────────────

#[tokio::test]
async fn replace_lines_in_middle_mutates_file() {
    let ws = ws_with_src().await;
    let path = PathBuf::from("src/lib.rs");
    let result: ReplaceLinesResult = replace_lines(ws.as_ref(), &path, 2, 3, "X\nY\n".into())
        .await
        .unwrap();

    assert!(result.succeeded);
    let after = ws.read(&path).await.unwrap().unwrap();
    assert_eq!(after, "line 1\nX\nY\nline 4\nline 5\n");
}

#[tokio::test]
async fn replace_lines_clamps_last_line_past_eof() {
    let ws = ws_with_src().await;
    let path = PathBuf::from("src/lib.rs");
    let result = replace_lines(ws.as_ref(), &path, 4, 999, "FINAL\n".into())
        .await
        .unwrap();
    assert!(result.clamped);
    let after = ws.read(&path).await.unwrap().unwrap();
    assert_eq!(after, "line 1\nline 2\nline 3\nFINAL\n");
}

#[tokio::test]
async fn replace_lines_rejects_inverted_range() {
    let ws = ws_with_src().await;
    let err = replace_lines(ws.as_ref(), &PathBuf::from("src/lib.rs"), 4, 2, "x".into())
        .await
        .unwrap_err();
    assert!(matches!(err, FileOpError::InvalidRange { .. }));
}

#[tokio::test]
async fn replace_lines_on_missing_file_errors() {
    let ws: Arc<dyn Workspace> = Arc::new(InMemoryWorkspace::new());
    let err = replace_lines(ws.as_ref(), &PathBuf::from("absent.rs"), 1, 1, "x".into())
        .await
        .unwrap_err();
    assert!(matches!(err, FileOpError::NotFound { .. }));
}

// ── read_file_diff ─────────────────────────────────────────────────────

#[tokio::test]
async fn diff_after_replace_returns_unified_diff() {
    let ws = ws_with_src().await;
    let path = PathBuf::from("src/lib.rs");
    replace_lines(ws.as_ref(), &path, 3, 3, "REPLACED\n".into())
        .await
        .unwrap();

    let diff = read_file_diff(ws.as_ref(), &path).await.unwrap();
    assert!(
        diff.text.contains("-line 3") && diff.text.contains("+REPLACED"),
        "unified diff missing -/+ lines: {}",
        diff.text
    );
}

#[tokio::test]
async fn diff_for_untouched_file_returns_empty() {
    let ws = ws_with_src().await;
    let diff = read_file_diff(ws.as_ref(), &PathBuf::from("src/lib.rs"))
        .await
        .unwrap();
    // Single write with no further changes → no in-session delta.
    assert!(diff.text.is_empty() || diff.is_creation);
}

#[tokio::test]
async fn diff_for_never_touched_file_errors() {
    let ws: Arc<dyn Workspace> = Arc::new(InMemoryWorkspace::new());
    let err = read_file_diff(ws.as_ref(), &PathBuf::from("absent.rs"))
        .await
        .unwrap_err();
    assert!(matches!(err, FileOpError::NotFound { .. }));
}

// ── delete_file ────────────────────────────────────────────────────────

#[tokio::test]
async fn delete_existing_file_removes_it() {
    let ws = ws_with_src().await;
    let path = PathBuf::from("src/lib.rs");
    delete_file(ws.as_ref(), &path).await.unwrap();
    assert!(ws.read(&path).await.unwrap().is_none());
}

#[tokio::test]
async fn delete_missing_file_is_idempotent() {
    let ws: Arc<dyn Workspace> = Arc::new(InMemoryWorkspace::new());
    // Two deletes on a never-written file — both must succeed.
    delete_file(ws.as_ref(), &PathBuf::from("absent.rs"))
        .await
        .unwrap();
    delete_file(ws.as_ref(), &PathBuf::from("absent.rs"))
        .await
        .unwrap();
}
