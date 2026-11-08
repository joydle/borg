//! An agent loop's `max_steps` is a ceiling, not a size hint: a huge one
//! (`u32::MAX`) must not turn into a huge up-front allocation, which
//! aborts the process where the OS refuses it. A global allocator of this
//! test binary's own records the largest single request to prove it.
//!
//! Run with:
//!   cargo test -p borg --test agent_loop_capacity --features testing

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::testing::{FakeLlmClient, ScriptedAgent};
use borg::{
    dsl, execute, in_memory_budget, noop_emitter, BorgError, CancellationToken, DispatchedCall,
    ExecCtx, ToolCallRequest, ToolCallResult, ToolDispatch, ToolSchema,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct LargestRequest;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for LargestRequest {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: LargestRequest = LargestRequest;

struct NoTools;

#[async_trait]
impl ToolDispatch for NoTools {
    async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError> {
        Ok(DispatchedCall::new(
            request.call_id,
            request.name,
            ToolCallResult::fail("no tools here"),
        ))
    }
    fn schemas(&self) -> Arc<[ToolSchema]> {
        Arc::from(Vec::<ToolSchema>::new())
    }
}

#[tokio::test]
async fn agent_loop_with_max_steps_u32_max_runs_one_turn_without_a_huge_allocation() {
    let llm = FakeLlmClient::always_ok("done");
    let ctx = ExecCtx::new(
        llm.clone(),
        in_memory_budget(1_000_000),
        noop_emitter(),
        CancellationToken::new(),
        "capacity",
    );
    let comp = dsl::agent_loop(ScriptedAgent::new("solo"), Arc::new(NoTools), u32::MAX);
    let outcome = execute(comp, ctx).await.unwrap();
    assert_eq!(llm.calls(), 1);
    assert_eq!(outcome.verdicts.len(), 1);
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(
        largest < 64 * 1024 * 1024,
        "largest single allocation was {largest} bytes"
    );
}
