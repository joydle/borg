//! Peak-memory check for a huge newline-free output. Its own test binary
//! so the counting allocator sees only this test. Unix-only (`/dev/zero`).
#![cfg(unix)]

use borg::Tool;
use borg_sandbox::{SandboxContext, SubprocessTool, TokioBackend, TRUNCATION_MARKER};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// `head -c 50000000 /dev/zero` emits 50 MB with no newline. Before the
/// fix, `BufReader::lines()` buffered all of it as one line. Now the peak
/// stays near the caps (64 KiB line, 1 MiB total) regardless of input size.
#[tokio::test]
async fn fifty_megabytes_without_newlines_stays_bounded() {
    let root = tempfile::TempDir::new().unwrap();
    let tool = SubprocessTool::builder("t", "t", TokioBackend::new())
        .allow_program("head")
        .workspace_root(root.path())
        .build()
        .unwrap();

    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);

    let mut ctx = SandboxContext::new();
    let r = tool
        .execute(
            serde_json::json!({ "program": "head", "args": ["-c", "50000000", "/dev/zero"] }),
            &mut ctx,
        )
        .await
        .unwrap();

    let grew = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    let out = r.output.expect("output");
    assert_eq!(out["truncated"], true);
    assert!(out["stdout"].as_str().unwrap().ends_with(TRUNCATION_MARKER));
    assert!(
        grew < 8 << 20,
        "peak heap grew by {grew} bytes for a 50 MB stream; expected < 8 MiB"
    );
}
