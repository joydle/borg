//! Compile + runtime test for `#[derive(BorgToolDerive)] #[tool(parallel)]`.
//!
//! With the flag, the derive emits both `impl Tool` and
//! `impl ConcurrentTool` from one shared-context `execute`, and both
//! `parallelize()` return `true`. Without it, only `impl Tool` is
//! emitted and `parallelize()` is `false`.

use async_trait::async_trait;
use borg::tools::{ConcurrentTool, Tool, ToolCallResult};
use borg::{BorgError, BorgToolDerive};

#[derive(BorgToolDerive)]
#[tool(
    name = "parallel_marker",
    description = "shared-context tool",
    context = (),
    input_schema = ::serde_json::json!({"type": "object"}),
    parallel,
)]
struct ParallelMarker;

impl ParallelMarker {
    // `&()` shared, not `&mut ()`: the parallel flag changes the
    // expected inherent signature.
    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &(),
    ) -> Result<ToolCallResult, BorgError> {
        ToolCallResult::ok(serde_json::json!({"ran": true}))
    }
}

#[derive(BorgToolDerive)]
#[tool(
    name = "serial_only",
    description = "exclusive-context tool",
    context = (),
    input_schema = ::serde_json::json!({"type": "object"}),
)]
struct SerialOnly;

impl SerialOnly {
    async fn execute(
        &self,
        _input: serde_json::Value,
        _ctx: &mut (),
    ) -> Result<ToolCallResult, BorgError> {
        ToolCallResult::ok(serde_json::json!(null))
    }
}

#[tokio::test]
async fn parallel_flag_emits_both_traits_with_true() {
    let t: &dyn Tool<Context = ()> = &ParallelMarker;
    assert!(Tool::parallelize(t), "Tool::parallelize() should be true");
    assert_eq!(t.schema().name, "parallel_marker");

    let ct: &dyn ConcurrentTool<Context = ()> = &ParallelMarker;
    assert!(
        ConcurrentTool::parallelize(ct),
        "ConcurrentTool::parallelize() should be true"
    );

    // Both traits dispatch to the same user body.
    let r1 = Tool::execute(&ParallelMarker, serde_json::json!({}), &mut ())
        .await
        .unwrap();
    let r2 = ConcurrentTool::execute(&ParallelMarker, serde_json::json!({}), &())
        .await
        .unwrap();
    assert_eq!(r1.output, r2.output);
}

#[tokio::test]
async fn no_flag_emits_only_tool_with_default_false() {
    let t: &dyn Tool<Context = ()> = &SerialOnly;
    assert!(
        !Tool::parallelize(t),
        "default Tool::parallelize() should be false"
    );

    // No `impl ConcurrentTool` is emitted; absence can't be asserted at
    // runtime, but the line below would fail to compile.
    fn _proof_no_concurrent_impl() {
        // let _: &dyn ConcurrentTool<Context = ()> = &SerialOnly;
    }
    _proof_no_concurrent_impl();
}

// The `parallel` flag requires `Context: Send + Sync` (enforced by the
// ConcurrentTool bound); a `!Sync` context would fail to compile.
#[async_trait]
trait _AssertObjectSafe: Tool {}
impl _AssertObjectSafe for ParallelMarker {}
impl _AssertObjectSafe for SerialOnly {}
