//! Integration test for `borg_tools::workspace`.
//!
//! Verifies the `Workspace` trait contract on the `InMemoryWorkspace`
//! impl: read/write round-trip, deletion idempotency, and the session
//! change log that `read_file_diff` consumes.

use borg_tools::workspace::{FileChange, InMemoryWorkspace, Workspace};
use std::path::PathBuf;

#[tokio::test]
async fn write_then_read_round_trip() {
    let ws = InMemoryWorkspace::new();
    let path = PathBuf::from("src/main.rs");
    ws.write(&path, "fn main() {}\n".into()).await.unwrap();

    let got = ws.read(&path).await.unwrap();
    assert_eq!(got.as_deref(), Some("fn main() {}\n"));
}

#[tokio::test]
async fn read_missing_path_returns_none() {
    let ws = InMemoryWorkspace::new();
    let got = ws.read(&PathBuf::from("nope.txt")).await.unwrap();
    assert!(got.is_none());
}

#[tokio::test]
async fn delete_is_idempotent() {
    let ws = InMemoryWorkspace::new();
    let path = PathBuf::from("temp.txt");

    // Deleting a non-existent file is OK.
    ws.delete(&path).await.unwrap();

    // Write then delete then delete again — all OK.
    ws.write(&path, "hi".into()).await.unwrap();
    ws.delete(&path).await.unwrap();
    ws.delete(&path).await.unwrap();

    // File is gone after delete.
    let got = ws.read(&path).await.unwrap();
    assert!(got.is_none());
}

#[tokio::test]
async fn changes_log_records_writes_and_deletes_in_order() {
    let ws = InMemoryWorkspace::new();
    let a = PathBuf::from("a.txt");
    let b = PathBuf::from("b.txt");

    ws.write(&a, "first".into()).await.unwrap();
    ws.write(&b, "second".into()).await.unwrap();
    ws.write(&a, "first-updated".into()).await.unwrap();
    ws.delete(&b).await.unwrap();

    let changes = ws.changes().await;
    assert_eq!(changes.len(), 4);
    assert!(matches!(&changes[0], FileChange::Written { path, .. } if path == &a));
    assert!(matches!(&changes[1], FileChange::Written { path, .. } if path == &b));
    assert!(matches!(&changes[2], FileChange::Written { path, .. } if path == &a));
    assert!(matches!(&changes[3], FileChange::Deleted { path } if path == &b));
}

#[tokio::test]
async fn changes_log_skips_no_op_writes() {
    // Writing the same content twice records one Written event, not two.
    let ws = InMemoryWorkspace::new();
    let path = PathBuf::from("dup.txt");
    ws.write(&path, "same".into()).await.unwrap();
    ws.write(&path, "same".into()).await.unwrap();
    assert_eq!(ws.changes().await.len(), 1);
}
