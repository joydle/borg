//! Public-API integration test for the `ConcurrentTool`,
//! `ConcurrentRegistry`, and `ConcurrentDispatcher` type plumbing,
//! end-to-end without the executor. Executor-side dispatch tests live
//! in `parallel_tool_calls.rs`.

#![cfg(feature = "testing")]

use async_trait::async_trait;
use borg::{
    BorgError, ConcurrentDispatcher, ConcurrentRegistry, ConcurrentTool, Tool, ToolCallRequest,
    ToolCallResult, ToolDispatch, ToolSchema,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Barrier;

/// Shared context with interior mutability — the canonical pattern for
/// `ConcurrentTool` impls that mutate state.
#[derive(Default)]
struct CountCtx {
    calls: AtomicUsize,
    barrier: Option<Arc<Barrier>>,
}

/// Increments the counter, then waits on the barrier (if present) so the
/// test can prove tools enter `execute` concurrently.
struct Counter;

#[async_trait]
impl ConcurrentTool for Counter {
    type Context = CountCtx;

    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "counter",
            "increment a shared counter",
            serde_json::json!({"type": "object", "properties": {}}),
        )
    }

    async fn execute(
        &self,
        _input: serde_json::Value,
        ctx: &CountCtx,
    ) -> Result<ToolCallResult, BorgError> {
        ctx.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(b) = ctx.barrier.as_ref() {
            // Will hang forever unless N tasks all reach this point.
            b.wait().await;
        }
        ToolCallResult::ok(serde_json::json!({"ok": true}))
    }
}

#[tokio::test]
async fn dispatches_concurrently_via_tokio_join() {
    let barrier = Arc::new(Barrier::new(3));
    let ctx = CountCtx {
        calls: AtomicUsize::new(0),
        barrier: Some(barrier),
    };
    let registry = ConcurrentRegistry::new().with(Counter);
    let dispatch = Arc::new(ConcurrentDispatcher::new(registry, ctx));

    // Reaching the join proves all three execute bodies were live at
    // once: serial dispatch would never trip the barrier(3) and would hang.
    let (a, b, c) = tokio::join!(
        dispatch.dispatch(ToolCallRequest::new("counter", serde_json::json!({}), "1")),
        dispatch.dispatch(ToolCallRequest::new("counter", serde_json::json!({}), "2")),
        dispatch.dispatch(ToolCallRequest::new("counter", serde_json::json!({}), "3")),
    );
    assert!(a.unwrap().result.output.is_some());
    assert!(b.unwrap().result.output.is_some());
    assert!(c.unwrap().result.output.is_some());
}

#[tokio::test]
async fn parallelize_safe_returns_per_tool_answer() {
    let registry = ConcurrentRegistry::new().with(Counter);
    let dispatch = ConcurrentDispatcher::new(
        registry,
        CountCtx {
            calls: AtomicUsize::new(0),
            barrier: None,
        },
    );

    // ConcurrentTool::parallelize() defaults to true.
    assert!(ToolDispatch::parallelize_safe(&dispatch, "counter"));
    // Unknown tool → false (treated as not-parallel-safe).
    assert!(!ToolDispatch::parallelize_safe(&dispatch, "no_such_tool"));
}

#[tokio::test]
async fn mutex_dispatcher_keeps_default_parallelize_safe_false() {
    use borg::{MutexDispatcher, Registry};

    // Tool::parallelize() returns true, but MutexDispatcher serializes,
    // so the dispatcher answer must stay false regardless.
    struct MutHopeful;
    #[async_trait]
    impl Tool for MutHopeful {
        type Context = ();
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("hope", "x", serde_json::json!({"type": "object"}))
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _ctx: &mut (),
        ) -> Result<ToolCallResult, BorgError> {
            ToolCallResult::ok(serde_json::json!(null))
        }
        fn parallelize(&self) -> bool {
            true
        }
    }

    let dispatch = MutexDispatcher::new(Registry::new().with(MutHopeful), ());
    assert!(
        !ToolDispatch::parallelize_safe(&dispatch, "hope"),
        "MutexDispatcher must answer false regardless of Tool::parallelize() — its mutex serializes."
    );
}
