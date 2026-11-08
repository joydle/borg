//! Tool protocol — the trait every dispatchable tool implements.
//!
//! A tool is a named, schema-described, async function the agent can
//! call. The trait is object-safe: `Vec<Box<dyn Tool<Context = C>>>`
//! is the canonical registry shape.
//!
//! ```text
//! Agent  ──"call read_file({path: \"foo.ts\"})"──▶  Dispatcher
//!                                                       │
//!                                                       │  1. lookup name
//!                                                       │  2. parse args
//!                                                       │  3. tool.execute(args, ctx)
//!                                                       │  4. serialize result
//!                                                       ▼
//! Agent  ◀──ToolCallResult { output | error, summary }──┘
//! ```
//!
//! Schemas are raw [`serde_json::Value`] (JSON Schema), not a typed
//! schema crate — hand-shaped schemas (recursive types, oneOf) don't
//! auto-derive cleanly, and raw JSON forces no dependency. The `Context`
//! is an associated type the consumer picks; borg never inspects it.
//! [`ToolSchema`] mirrors the MCP `tools/list` layout, so an MCP adapter
//! is a serializer, not a redesign.

use crate::error::BorgError;
use async_trait::async_trait;

/// Schema describing a tool's name, intent, and expected input.
///
/// Field shape mirrors MCP `tools/list` entries so an adapter can
/// serialize this directly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ToolSchema {
    /// Unique tool name. The dispatcher's lookup is keyed on this.
    pub name: String,
    /// One-line description shown to the model — it drives the model's
    /// decision to call the tool, so keep it specific and actionable.
    pub description: String,
    /// JSON Schema for the `input` object. Must have `"type": "object"`
    /// at the top level.
    pub input_schema: serde_json::Value,
}

impl ToolSchema {
    /// Construct a tool schema. The constructor lets external crates
    /// build one despite `#[non_exhaustive]`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Result of a tool's `execute` call.
///
/// - **Success**: `output = Some(...)`, `error = None`.
/// - **Logical failure**: `output = None`, `error = Some(msg)` — the
///   tool ran and returned a structured failure the agent should react
///   to (file not found, permission denied, …).
/// - Either may carry a one-line `summary` for compact logging when
///   `output` is large.
///
/// A tool that itself crashed returns `Err(BorgError)` from `execute`,
/// not a `ToolCallResult`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ToolCallResult {
    /// Raw output, surfaced verbatim in the agent's next-turn history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
    /// User-facing logical failure message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// One-line summary the agent can read without parsing `output` —
    /// useful when `output` is large.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl ToolCallResult {
    /// Success with a serializable output. `Err` only if the output
    /// isn't serde-serializable.
    pub fn ok<T: serde::Serialize>(output: T) -> Result<Self, BorgError> {
        let value = serde_json::to_value(output).map_err(|e| BorgError::Internal {
            message: format!("ToolCallResult::ok: failed to serialize output: {e}"),
        })?;
        Ok(Self {
            output: Some(value),
            error: None,
            summary: None,
        })
    }

    /// Success with an explicit summary line.
    pub fn ok_with_summary<T: serde::Serialize>(
        output: T,
        summary: impl Into<String>,
    ) -> Result<Self, BorgError> {
        let mut r = Self::ok(output)?;
        r.summary = Some(summary.into());
        Ok(r)
    }

    /// Logical failure. The tool ran; it surfaces a message the agent
    /// should incorporate into its next decision.
    pub fn fail(error: impl Into<String>) -> Self {
        Self {
            output: None,
            error: Some(error.into()),
            summary: None,
        }
    }

    /// Attach a one-line summary to either an `ok` or `fail` result.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }
}

/// Contract every dispatchable tool implements.
///
/// Generic over the consumer's `Context` so each application threads
/// its own per-call state (session refs, DB handles, working-copy
/// snapshots) without coupling to borg.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Consumer-defined per-call context type.
    ///
    /// `Send` but not `'static`: `Registry::dispatch` takes `&mut C`, so
    /// a context of borrows works at the call site. The `'static` bound
    /// only kicks in for `Box<dyn Tool<...>>` storage — wrap borrows in
    /// an owned `Arc`/`Arc<Mutex<...>>` for that case.
    type Context: Send;

    /// Schema sent to the LLM and used by the dispatcher to parse args.
    fn schema(&self) -> ToolSchema;

    /// Execute the tool. Return:
    /// - `Ok(ToolCallResult::ok(...))` on success
    /// - `Ok(ToolCallResult::fail("..."))` on logical failure the agent
    ///   should see (file not found, etc.)
    /// - `Err(BorgError)` only when the tool itself broke
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &mut Self::Context,
    ) -> Result<ToolCallResult, BorgError>;

    /// Whether this tool is safe to invoke concurrently with siblings
    /// sharing the same `Context`. Consulted only by concurrency-aware
    /// dispatchers; `MutexDispatcher` always serializes.
    fn parallelize(&self) -> bool {
        false
    }
}

/// Boxed-trait-object alias — the common type for tool registries.
pub type BoxedTool<C> = Box<dyn Tool<Context = C>>;

/// Context-erased tool dispatch surface.
///
/// `Registry<C>` is generic over the consumer's `Context`, which won't
/// fit a non-generic [`crate::Composition`] variant. `ToolDispatch` is
/// the object-safe bridge: the consumer wraps a `Registry<C>` plus its
/// per-call context (typically in an `Arc<Mutex<C>>`) and hands borg an
/// `Arc<dyn ToolDispatch>` with the context type erased.
///
/// Implementations must be `Send + Sync` so the dispatcher can ride
/// inside a `Composition::Agent` cloned across branches.
#[async_trait]
pub trait ToolDispatch: Send + Sync {
    /// Run a single tool call. Mirrors [`Registry::dispatch`] without
    /// the context generic.
    async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError>;

    /// Schemas of every dispatchable tool, used to populate
    /// `ChatRequest::tools`.
    ///
    /// Returns `Arc<[ToolSchema]>` so the hot path is pointer bumps with
    /// no deep copies. Implementations should cache the slice and clone
    /// the Arc per call (see [`Registry::schemas`]).
    fn schemas(&self) -> std::sync::Arc<[ToolSchema]>;

    /// Whether a request with this tool name can run concurrently with
    /// siblings on this dispatcher. Default `false` (serial semantics).
    ///
    /// `MutexDispatcher` keeps the default since its mutex serializes
    /// anyway; `ConcurrentDispatcher` surfaces per-tool
    /// `ConcurrentTool::parallelize()`.
    fn parallelize_safe(&self, _tool_name: &str) -> bool {
        false
    }
}

/// Wraps a [`Registry<C>`] plus a context behind a `tokio::sync::Mutex`
/// to satisfy the context-erased [`ToolDispatch`] trait.
///
/// Use this when tools need read/write access to shared session state.
/// The mutex is held only for one tool's `execute` call and serializes
/// concurrent dispatch — keep tool bodies fast against the context. For
/// tools that don't need mutable shared state, write a `ToolDispatch`
/// impl that doesn't lock.
pub struct MutexDispatcher<C: Send + 'static> {
    registry: Registry<C>,
    context: tokio::sync::Mutex<C>,
}

impl<C: Send + 'static> MutexDispatcher<C> {
    /// Construct from a registry and an initial context.
    pub fn new(registry: Registry<C>, context: C) -> Self {
        Self {
            registry,
            context: tokio::sync::Mutex::new(context),
        }
    }
}

#[async_trait]
impl<C: Send + 'static> ToolDispatch for MutexDispatcher<C> {
    async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError> {
        let mut ctx = self.context.lock().await;
        self.registry.dispatch(request, &mut *ctx).await
    }

    fn schemas(&self) -> std::sync::Arc<[ToolSchema]> {
        self.registry.schemas()
    }

    // parallelize_safe stays `false`: the mutex serializes regardless.
}

/// Contract for tools that can dispatch concurrently sharing one context.
///
/// Mirrors [`Tool`] but `execute` takes `&Self::Context` (shared) — tools
/// that mutate state use interior mutability (`RwLock`, atomics,
/// `DashMap`) inside it. Used by [`ConcurrentDispatcher`]. A type may
/// implement both `Tool` and `ConcurrentTool`; the `#[tool(parallel)]`
/// derive flag emits both from one `execute(&self, input, ctx: &Context)`.
#[async_trait]
pub trait ConcurrentTool: Send + Sync {
    /// Per-call context. `Sync` so the dispatcher can hand out `&Context`
    /// to many tasks at once.
    type Context: Send + Sync;

    /// Schema sent to the LLM. Mirrors [`Tool::schema`].
    fn schema(&self) -> ToolSchema;

    /// Execute against a shared `&Context`. Same return-shape contract as
    /// [`Tool::execute`].
    async fn execute(
        &self,
        input: serde_json::Value,
        ctx: &Self::Context,
    ) -> Result<ToolCallResult, BorgError>;

    /// Whether this tool is safe to invoke concurrently with siblings.
    /// Defaults to `true` — implementing this trait declares concurrent
    /// safety. Override to `false` to serialize a tool registered here.
    fn parallelize(&self) -> bool {
        true
    }
}

/// Boxed-trait-object alias for concurrent tools.
pub type BoxedConcurrentTool<C> = Box<dyn ConcurrentTool<Context = C>>;

/// Sibling of [`Registry<C>`] for [`ConcurrentTool`] implementations.
///
/// Same name-keyed lookup, schema enumeration, and duplicate-name panic
/// as `Registry`. Dispatches with `&C` (shared), not `&mut C`.
pub struct ConcurrentRegistry<C: Send + Sync + 'static> {
    tools: std::collections::HashMap<String, BoxedConcurrentTool<C>>,
    /// Memoized schema list — see [`Registry::schema_cache`].
    schema_cache: parking_lot::Mutex<Option<std::sync::Arc<[ToolSchema]>>>,
}

impl<C: Send + Sync + 'static> Default for ConcurrentRegistry<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Send + Sync + 'static> ConcurrentRegistry<C> {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            tools: std::collections::HashMap::new(),
            schema_cache: parking_lot::Mutex::new(None),
        }
    }

    /// Register a concurrent tool.
    ///
    /// # Panics
    ///
    /// If a tool with the same name is already registered. See
    /// [`Registry::register`].
    pub fn register<T: ConcurrentTool<Context = C> + 'static>(&mut self, tool: T) {
        let name = tool.schema().name;
        if self.tools.contains_key(&name) {
            panic!(
                "borg::tools::ConcurrentRegistry: duplicate tool name `{name}`; \
                 two tools cannot share a name"
            );
        }
        self.tools.insert(name, Box::new(tool));
        *self.schema_cache.get_mut() = None;
    }

    /// Builder-style register.
    #[must_use]
    pub fn with<T: ConcurrentTool<Context = C> + 'static>(mut self, tool: T) -> Self {
        self.register(tool);
        self
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True iff no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Schemas of every registered tool. Cached — see [`Registry::schemas`].
    pub fn schemas(&self) -> std::sync::Arc<[ToolSchema]> {
        let mut cache = self.schema_cache.lock();
        if let Some(arc) = cache.as_ref() {
            return std::sync::Arc::clone(arc);
        }
        let built: Vec<ToolSchema> = self.tools.values().map(|t| t.schema()).collect();
        let arc: std::sync::Arc<[ToolSchema]> = std::sync::Arc::from(built);
        *cache = Some(std::sync::Arc::clone(&arc));
        arc
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn ConcurrentTool<Context = C>> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Dispatch a single tool call against a shared context. Unknown
    /// tools surface as [`ToolCallResult::fail`], never a `BorgError`.
    pub async fn dispatch(
        &self,
        request: ToolCallRequest,
        ctx: &C,
    ) -> Result<DispatchedCall, BorgError> {
        let ToolCallRequest {
            name,
            args,
            call_id,
        } = request;

        let Some(tool) = self.tools.get(&name) else {
            return Ok(DispatchedCall {
                call_id,
                name: name.clone(),
                result: ToolCallResult::fail(format!("unknown tool `{name}` — not in registry")),
            });
        };

        let slot = ActiveToolCallSlot::new(ActiveToolCall {
            call_id: call_id.clone(),
            name: name.clone(),
        });
        let result = TOOL_CALL
            .scope(slot, async { tool.execute(args, ctx).await })
            .await?;
        Ok(DispatchedCall {
            call_id,
            name,
            result,
        })
    }
}

/// Wraps a [`ConcurrentRegistry<C>`] + shared `Arc<C>` to satisfy the
/// context-erased [`ToolDispatch`] trait without locking.
///
/// Use this when tools share a context that supports concurrent access
/// (interior mutability or pure reads). Tools returning `true` from
/// [`ConcurrentTool::parallelize`] dispatch in parallel when the
/// executor's `allow_parallel_tools` flag is set.
pub struct ConcurrentDispatcher<C: Send + Sync + 'static> {
    registry: ConcurrentRegistry<C>,
    context: std::sync::Arc<C>,
}

impl<C: Send + Sync + 'static> ConcurrentDispatcher<C> {
    /// Construct from a registry and an initial context.
    pub fn new(registry: ConcurrentRegistry<C>, context: C) -> Self {
        Self {
            registry,
            context: std::sync::Arc::new(context),
        }
    }

    /// Construct sharing an existing `Arc<C>`.
    pub fn from_arc(registry: ConcurrentRegistry<C>, context: std::sync::Arc<C>) -> Self {
        Self { registry, context }
    }
}

#[async_trait]
impl<C: Send + Sync + 'static> ToolDispatch for ConcurrentDispatcher<C> {
    async fn dispatch(&self, request: ToolCallRequest) -> Result<DispatchedCall, BorgError> {
        self.registry.dispatch(request, &self.context).await
    }

    fn schemas(&self) -> std::sync::Arc<[ToolSchema]> {
        self.registry.schemas()
    }

    fn parallelize_safe(&self, tool_name: &str) -> bool {
        self.registry
            .get(tool_name)
            .map(|t| t.parallelize())
            .unwrap_or(false)
    }
}

/// A single tool invocation the agent has decided to perform.
///
/// Produced by the agent (from parsing the LLM response or a native
/// `tool_calls` channel), consumed by [`Registry::dispatch`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ToolCallRequest {
    /// Tool name — must match a registered `Tool::schema().name`.
    pub name: String,
    /// JSON args parsed by the tool's `execute`. Borg doesn't validate
    /// against `input_schema`; the consumer enforces it if desired.
    pub args: serde_json::Value,
    /// Caller-provided correlation id, echoed back in the result so
    /// concurrent calls can be matched to responses. Opaque to borg.
    pub call_id: String,
}

impl ToolCallRequest {
    /// Construct a tool-call request. The constructor lets external
    /// crates build one despite `#[non_exhaustive]`.
    pub fn new(
        name: impl Into<String>,
        args: serde_json::Value,
        call_id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            args,
            call_id: call_id.into(),
        }
    }
}

/// Outcome of a single tool dispatch through a [`Registry`].
///
/// Bundles the original `call_id` with the tool's `ToolCallResult`.
/// `Serialize`/`Deserialize` so consumers can stash accumulated
/// dispatches in `HandoffState` for the next turn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct DispatchedCall {
    /// Echoes [`ToolCallRequest::call_id`] verbatim.
    pub call_id: String,
    /// Tool name that ran (echoes the request).
    pub name: String,
    /// What the tool returned, or a synthetic failure if the tool wasn't
    /// found in the registry.
    pub result: ToolCallResult,
}

impl DispatchedCall {
    /// The outcome of one dispatched call. `ToolDispatch` implementations
    /// outside borg build it with this constructor.
    #[must_use]
    pub fn new(
        call_id: impl Into<String>,
        name: impl Into<String>,
        result: ToolCallResult,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            result,
        }
    }
}

/// A registered collection of tools, keyed by name.
///
/// A thin wrapper around `HashMap<String, BoxedTool<C>>` providing
/// name-keyed dispatch, schema enumeration, and duplicate-name
/// protection (`register` panics on a collision rather than silently
/// shadowing). Unknown tools at dispatch time surface as
/// `ToolCallResult::fail`, not a `BorgError`, so the agent can recover.
///
/// `Registry<C>` pins the consumer's context type, so a program can
/// hold several registries with different contexts.
pub struct Registry<C: Send> {
    tools: std::collections::HashMap<String, BoxedTool<C>>,
    /// Memoized schema list, built lazily and invalidated on
    /// `register()`. The agent loop calls `schemas()` once per iteration
    /// and the result is stable between registrations.
    ///
    /// `parking_lot::Mutex<Option<...>>` lets `schemas()` take `&self`;
    /// the inner `Arc<[ToolSchema]>` makes a cache hit a single
    /// `Arc::clone` with no per-element copy.
    schema_cache: parking_lot::Mutex<Option<std::sync::Arc<[ToolSchema]>>>,
}

impl<C: Send> Default for Registry<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Send> Registry<C> {
    /// Empty registry. Add tools via [`Self::register`] or [`Self::with`].
    pub fn new() -> Self {
        Self {
            tools: std::collections::HashMap::new(),
            schema_cache: parking_lot::Mutex::new(None),
        }
    }

    /// Register a tool.
    ///
    /// # Panics
    ///
    /// If a tool with the same name is already registered. Duplicate
    /// names would silently shadow at dispatch time; registries are
    /// built once at boot, so failing loudly there beats failing
    /// mid-request.
    pub fn register<T: Tool<Context = C> + 'static>(&mut self, tool: T) {
        let name = tool.schema().name;
        if self.tools.contains_key(&name) {
            panic!(
                "borg::tools::Registry: duplicate tool name `{name}`; \
                 two tools cannot share a name"
            );
        }
        self.tools.insert(name, Box::new(tool));
        *self.schema_cache.get_mut() = None;
    }

    /// Builder-style register, for chaining at construction time:
    /// ```ignore
    /// let registry = Registry::new()
    ///     .with(ReadFile)
    ///     .with(ApplyEdit);
    /// ```
    #[must_use]
    pub fn with<T: Tool<Context = C> + 'static>(mut self, tool: T) -> Self {
        self.register(tool);
        self
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True iff no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Schemas of every registered tool, in unspecified order — for the
    /// LLM's tool list or an MCP `tools/list` response.
    ///
    /// Cached internally: the first call materializes the slice,
    /// subsequent calls return `Arc::clone(&cached)`. `register()`
    /// invalidates the cache.
    pub fn schemas(&self) -> std::sync::Arc<[ToolSchema]> {
        let mut cache = self.schema_cache.lock();
        if let Some(arc) = cache.as_ref() {
            return std::sync::Arc::clone(arc);
        }
        let built: Vec<ToolSchema> = self.tools.values().map(|t| t.schema()).collect();
        let arc: std::sync::Arc<[ToolSchema]> = std::sync::Arc::from(built);
        *cache = Some(std::sync::Arc::clone(&arc));
        arc
    }

    /// Look up a tool by name. Returns `None` if not registered.
    pub fn get(&self, name: &str) -> Option<&dyn Tool<Context = C>> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Dispatch a single tool call.
    ///
    /// Errors:
    /// - **Unknown tool** → `DispatchedCall { result: fail("...") }`.
    ///   Not a `BorgError`: the agent should see this as a logical
    ///   failure and try a different approach.
    /// - **Tool itself crashed** → propagates the underlying
    ///   `BorgError` from `Tool::execute`.
    pub async fn dispatch(
        &self,
        request: ToolCallRequest,
        ctx: &mut C,
    ) -> Result<DispatchedCall, BorgError> {
        let ToolCallRequest {
            name,
            args,
            call_id,
        } = request;

        let Some(tool) = self.tools.get(&name) else {
            return Ok(DispatchedCall {
                call_id,
                name: name.clone(),
                result: ToolCallResult::fail(format!("unknown tool `{name}` — not in registry")),
            });
        };

        // Scope the active-call task-local so side-channel emissions
        // inside the tool body (BuildLogLine, ToolArgDelta, …) carry the
        // same `call_id` the agent loop saw on `ToolCallStarted`.
        let slot = ActiveToolCallSlot::new(ActiveToolCall {
            call_id: call_id.clone(),
            name: name.clone(),
        });
        let result = TOOL_CALL
            .scope(slot, async { tool.execute(args, ctx).await })
            .await?;
        Ok(DispatchedCall {
            call_id,
            name,
            result,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Enum-dispatched tools — for consumers with borrowed-context state
// ─────────────────────────────────────────────────────────────────────
//
// `Box<dyn Tool<Context = C>>` transitively requires `C: 'static`. That
// blocks contexts that are a struct of `&mut` borrows from a live
// session, since the obvious GAT workaround (`type Context<'ctx>`) makes
// the trait not `dyn`-compatible.
//
// `ToolSet` is the enum-dispatch alternative: the consumer defines an
// enum of unit-struct tool markers and gets static dispatch with a
// `type Context<'ctx>` GAT — no `dyn`, no `'static`. The trade-off is
// compile-time tool registration (adding a variant means recompiling
// the enum).

/// Contract for an enum-dispatched tool collection.
///
/// Implementations are usually enums whose variants name the tools;
/// `execute` pattern-matches on the variant and delegates. The GAT'd
/// `Context<'ctx>` lets consumers thread borrowed session state into
/// tool bodies — no Arc, Mutex, or clone-per-call.
///
/// ```ignore
/// pub enum MyTools { ReadFile(ReadFile), ListFiles(ListFiles) }
///
/// #[async_trait]
/// impl ToolSet for MyTools {
///     type Context<'ctx> = ToolContext<'ctx>;
///
///     fn name(&self) -> &'static str {
///         match self {
///             MyTools::ReadFile(_) => "read_file",
///             MyTools::ListFiles(_) => "list_files",
///         }
///     }
///
///     fn schema(&self) -> ToolSchema { /* ... */ }
///
///     async fn execute<'ctx>(
///         &self,
///         input: serde_json::Value,
///         ctx: &mut Self::Context<'ctx>,
///     ) -> Result<ToolCallResult, BorgError> {
///         match self {
///             MyTools::ReadFile(t) => t.execute(input, ctx).await,
///             MyTools::ListFiles(t) => t.execute(input, ctx).await,
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait ToolSet: Send + Sync {
    /// Per-call context. `'ctx` is the lifetime of the `&mut` passed
    /// into `execute`. Owned contexts ignore it; borrowing contexts
    /// carry it (`type Context<'ctx> = MyBorrowed<'ctx>;`).
    type Context<'ctx>: Send
    where
        Self: 'ctx;

    /// Consumer-defined output type. Use `ToolCallResult` for LLM-facing
    /// dispatch, or any rich type for consumer-internal dispatch
    /// (avoiding a JSON round-trip).
    type Output: Send;

    /// Stable name of the tool variant `self` represents. The registry key.
    fn name(&self) -> &'static str;

    /// Schema for this tool variant. Surfaced to the LLM.
    fn schema(&self) -> ToolSchema;

    /// Execute this tool variant. `&self` is the variant marker;
    /// pattern-match inside the body to dispatch to the actual tool.
    async fn execute<'ctx>(
        &self,
        input: serde_json::Value,
        ctx: &mut Self::Context<'ctx>,
    ) -> Result<Self::Output, BorgError>;
}

/// Name-keyed registry of enum-dispatched tools.
///
/// `T` is the consumer's tool enum. Dispatch looks up the variant by
/// `request.name` and calls the enum's `execute`. Built once at session
/// start; cheap to clone if `T: Clone`.
pub struct ToolSetRegistry<T: ToolSet> {
    tools: std::collections::HashMap<&'static str, T>,
    /// Memoized schema list — see [`Registry::schemas`].
    schema_cache: parking_lot::Mutex<Option<std::sync::Arc<[ToolSchema]>>>,
}

impl<T: ToolSet> Default for ToolSetRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ToolSet> ToolSetRegistry<T> {
    pub fn new() -> Self {
        Self {
            tools: std::collections::HashMap::new(),
            schema_cache: parking_lot::Mutex::new(None),
        }
    }

    /// Register a tool variant.
    ///
    /// # Panics
    ///
    /// If a variant with the same name is already registered. See
    /// [`Registry::register`].
    pub fn register(&mut self, variant: T) {
        let name = variant.name();
        if self.tools.contains_key(name) {
            panic!(
                "borg::tools::ToolSetRegistry: duplicate tool name `{name}`; \
                 two variants cannot share a name"
            );
        }
        self.tools.insert(name, variant);
        *self.schema_cache.get_mut() = None;
    }

    /// Builder-style register.
    #[must_use]
    pub fn with(mut self, variant: T) -> Self {
        self.register(variant);
        self
    }

    /// True iff no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Schemas of every registered tool, in unspecified order. Cached —
    /// see [`Registry::schemas`].
    pub fn schemas(&self) -> std::sync::Arc<[ToolSchema]> {
        let mut cache = self.schema_cache.lock();
        if let Some(arc) = cache.as_ref() {
            return std::sync::Arc::clone(arc);
        }
        let built: Vec<ToolSchema> = self.tools.values().map(|t| t.schema()).collect();
        let arc: std::sync::Arc<[ToolSchema]> = std::sync::Arc::from(built);
        *cache = Some(std::sync::Arc::clone(&arc));
        arc
    }

    /// True iff a tool with `name` is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Dispatch a single tool call.
    ///
    /// Returns the consumer's [`ToolSet::Output`] directly — no JSON
    /// round-trip. The consumer chooses how to surface it to the LLM.
    ///
    /// Errors:
    /// - **Unknown tool** → [`DispatchOutcome::Unknown`].
    /// - **Tool itself crashed** → propagates the underlying `BorgError`.
    pub async fn dispatch<'ctx>(
        &self,
        request: ToolCallRequest,
        ctx: &mut T::Context<'ctx>,
    ) -> Result<DispatchOutcome<T::Output>, BorgError> {
        let ToolCallRequest {
            name,
            args,
            call_id,
        } = request;

        let Some(tool) = self.tools.get(name.as_str()) else {
            return Ok(DispatchOutcome::Unknown { call_id, name });
        };

        // Same TOOL_CALL scoping as `Registry::dispatch`.
        let slot = ActiveToolCallSlot::new(ActiveToolCall {
            call_id: call_id.clone(),
            name: name.clone(),
        });
        let output = TOOL_CALL
            .scope(slot, async { tool.execute(args, ctx).await })
            .await?;
        Ok(DispatchOutcome::Ok {
            call_id,
            name,
            output,
        })
    }
}

/// Outcome of a [`ToolSetRegistry::dispatch`] call.
///
/// Distinguishes the unknown-tool case at the type level so consumers
/// needn't inspect a sentinel error string. `Ok` carries the consumer's
/// `Output` directly.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DispatchOutcome<O> {
    /// Tool ran and returned its typed output.
    Ok {
        call_id: String,
        name: String,
        output: O,
    },
    /// No tool with this name in the registry. The caller decides how to
    /// surface it (e.g. `ToolCallResult::fail` for LLM-facing dispatch).
    Unknown { call_id: String, name: String },
}

// ─────────────────────────────────────────────────────────────────────
// Per-call task-local — TOOL_CALL
// ─────────────────────────────────────────────────────────────────────
//
// Tools that emit side-channel progress events (a build streaming stdout
// lines, a write_file streaming byte counters) need the *same* call_id
// the supervising agent saw on `ToolCallStarted` so the consumer can
// correlate them. The dispatcher populates this slot before invoking
// `Tool::execute` and clears it after; tool bodies read it via
// `active_tool_call()`. The analogue of `crate::llm::LLM_STREAM`.
//
// Both `Registry::dispatch` and `ToolSetRegistry::dispatch` do the
// scoping; consumers scope nothing. Tools called outside a dispatch
// (unit tests) see `active_tool_call() == None`.

// (A `//` comment, not `///`: rustdoc ignores docs on macro-produced
// items.)
tokio::task_local! {
    pub static TOOL_CALL: ActiveToolCallSlot;
}

/// Slot wrapper so the task-local defaults to "no active call" outside a
/// dispatcher scope (e.g. unit tests).
#[derive(Clone, Default)]
pub struct ActiveToolCallSlot {
    pub call: Option<ActiveToolCall>,
}

impl ActiveToolCallSlot {
    pub fn new(call: ActiveToolCall) -> Self {
        Self { call: Some(call) }
    }
}

/// Identity of the currently-executing tool call. Tool bodies read it to
/// tag side-channel events with the same `call_id` the agent loop
/// emitted on `ToolCallStarted`.
#[derive(Clone, Debug)]
pub struct ActiveToolCall {
    /// The LLM-assigned call id. Stable across `started → progress* →
    /// completed`.
    pub call_id: String,
    /// The tool's registered name (matches `ToolSchema::name`).
    pub name: String,
}

/// The active tool call if one is scoped, else `None`. Mirrors
/// [`crate::llm::active_stream_sink`]; tools call it inside `execute` to
/// tag side-channel events:
///
/// ```ignore
/// async fn execute(&self, input, ctx) -> Result<ToolCallResult, BorgError> {
///     let call_id = borg::tools::active_tool_call()
///         .map(|c| c.call_id)
///         .unwrap_or_default();
///     // emit events tagged with call_id ...
/// }
/// ```
///
/// `None` outside a dispatcher scope (unit tests, custom dispatchers).
/// Tools should degrade gracefully — typically skip the emission.
pub fn active_tool_call() -> Option<ActiveToolCall> {
    TOOL_CALL.try_with(|s| s.call.clone()).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        type Context = ();

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "echo".into(),
                description: "Echoes input back unchanged.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                    "required": ["message"],
                }),
            }
        }

        async fn execute(
            &self,
            input: serde_json::Value,
            _ctx: &mut Self::Context,
        ) -> Result<ToolCallResult, BorgError> {
            ToolCallResult::ok(input)
        }
    }

    #[tokio::test]
    async fn schema_serializes_to_mcp_shape() {
        let schema = Echo.schema();
        let json = serde_json::to_value(&schema).unwrap();
        assert_eq!(json["name"], "echo");
        assert!(json["input_schema"]["properties"]["message"].is_object());
        assert_eq!(json["input_schema"]["required"][0], "message");
    }

    #[tokio::test]
    async fn execute_returns_output_on_success() {
        let result = Echo
            .execute(serde_json::json!({ "message": "hi" }), &mut ())
            .await
            .unwrap();
        assert_eq!(result.output.unwrap()["message"], "hi");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn tool_call_result_constructors() {
        let ok = ToolCallResult::ok(serde_json::json!({ "x": 1 })).unwrap();
        assert!(ok.output.is_some() && ok.error.is_none());

        let with_summary = ToolCallResult::ok_with_summary(42, "read 1 file").unwrap();
        assert_eq!(with_summary.summary.as_deref(), Some("read 1 file"));

        let fail = ToolCallResult::fail("not found");
        assert!(fail.output.is_none() && fail.error.as_deref() == Some("not found"));
    }

    #[tokio::test]
    async fn registry_holds_heterogeneous_tools() {
        // Compile-time check: a `Vec<Box<dyn Tool>>` registry must work.
        let registry: Vec<BoxedTool<()>> = vec![Box::new(Echo)];
        let result = registry[0]
            .execute(serde_json::json!({ "message": "ok" }), &mut ())
            .await
            .unwrap();
        assert!(result.output.is_some());
    }

    // ── Registry tests ──────────────────────────────────────────────

    struct Failer;

    #[async_trait]
    impl Tool for Failer {
        type Context = ();
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "failer".into(),
                description: "Always returns a logical failure.".into(),
                input_schema: serde_json::json!({ "type": "object" }),
            }
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _ctx: &mut Self::Context,
        ) -> Result<ToolCallResult, BorgError> {
            Ok(ToolCallResult::fail("intentional"))
        }
    }

    #[tokio::test]
    async fn registry_dispatches_to_named_tool() {
        let registry = Registry::<()>::new().with(Echo).with(Failer);
        assert_eq!(registry.len(), 2);

        let mut ctx = ();
        let dispatched = registry
            .dispatch(
                ToolCallRequest {
                    name: "echo".into(),
                    args: serde_json::json!({ "message": "hi" }),
                    call_id: "abc".into(),
                },
                &mut ctx,
            )
            .await
            .expect("dispatch succeeds");
        assert_eq!(dispatched.call_id, "abc");
        assert_eq!(dispatched.name, "echo");
        assert!(dispatched.result.output.is_some());
    }

    #[tokio::test]
    async fn registry_dispatch_propagates_tool_logical_failure() {
        let registry = Registry::<()>::new().with(Failer);
        let mut ctx = ();
        let dispatched = registry
            .dispatch(
                ToolCallRequest {
                    name: "failer".into(),
                    args: serde_json::json!({}),
                    call_id: "xyz".into(),
                },
                &mut ctx,
            )
            .await
            .expect("dispatch itself doesn't error");
        assert!(dispatched.result.output.is_none());
        assert_eq!(dispatched.result.error.as_deref(), Some("intentional"));
    }

    #[tokio::test]
    async fn registry_dispatch_unknown_tool_returns_logical_failure() {
        let registry = Registry::<()>::new().with(Echo);
        let mut ctx = ();
        let dispatched = registry
            .dispatch(
                ToolCallRequest {
                    name: "nonexistent".into(),
                    args: serde_json::json!({}),
                    call_id: "ghost".into(),
                },
                &mut ctx,
            )
            .await
            .expect("unknown tool is a logical failure, not a BorgError");
        // call_id round-trips even for the not-found case
        assert_eq!(dispatched.call_id, "ghost");
        assert_eq!(dispatched.name, "nonexistent");
        assert!(dispatched.result.output.is_none());
        let err = dispatched.result.error.expect("error message present");
        assert!(err.contains("unknown tool"), "got: {err}");
        assert!(err.contains("nonexistent"), "got: {err}");
    }

    #[tokio::test]
    async fn registry_schemas_enumerate_all_tools() {
        let registry = Registry::<()>::new().with(Echo).with(Failer);
        let schemas = registry.schemas();
        assert_eq!(schemas.len(), 2);
        let names: std::collections::HashSet<&str> =
            schemas.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains("echo"));
        assert!(names.contains("failer"));
    }

    #[test]
    #[should_panic(expected = "duplicate tool name")]
    fn registry_register_duplicate_name_panics() {
        // Two tools sharing a name would silently shadow — fail loudly
        // at registration instead.
        let mut registry = Registry::<()>::new();
        registry.register(Echo);
        registry.register(Echo);
    }

    // ── ToolSet tests (enum-dispatch, GAT-friendly path) ─────────────

    /// Borrowed context — proves a struct of `&mut` borrows works as
    /// `Context<'ctx>` because there's no `dyn ToolSet`.
    struct BorrowedCtx<'ctx> {
        counter: &'ctx mut u32,
    }

    /// Owned context — proves the same trait shape accepts owned
    /// contexts too.
    #[derive(Default)]
    struct OwnedCtx {
        counter: u32,
    }

    // Two tool markers used in the enum below.
    struct Incrementer;
    struct Reader;

    /// Borrowed-context tool enum. The variants are unit-struct
    /// markers; the enum's `execute` pattern-matches and delegates.
    enum BorrowedTools {
        Increment(Incrementer),
        Read(Reader),
    }

    #[async_trait]
    impl ToolSet for BorrowedTools {
        type Context<'ctx> = BorrowedCtx<'ctx>;
        type Output = ToolCallResult;

        fn name(&self) -> &'static str {
            match self {
                BorrowedTools::Increment(_) => "increment",
                BorrowedTools::Read(_) => "read",
            }
        }

        fn schema(&self) -> ToolSchema {
            let (name, description) = match self {
                BorrowedTools::Increment(_) => ("increment", "Bump the counter by 1."),
                BorrowedTools::Read(_) => ("read", "Read the counter value."),
            };
            ToolSchema {
                name: name.to_string(),
                description: description.to_string(),
                input_schema: serde_json::json!({ "type": "object" }),
            }
        }

        async fn execute<'ctx>(
            &self,
            _input: serde_json::Value,
            ctx: &mut BorrowedCtx<'ctx>,
        ) -> Result<ToolCallResult, BorgError> {
            match self {
                BorrowedTools::Increment(_) => {
                    *ctx.counter += 1;
                    ToolCallResult::ok(serde_json::json!({ "new_value": *ctx.counter }))
                }
                BorrowedTools::Read(_) => ToolCallResult::ok(serde_json::json!({
                    "value": *ctx.counter
                })),
            }
        }
    }

    /// Owned-context tool enum — proves the same trait covers the
    /// owned-context case too (`'ctx` is unused).
    enum OwnedTools {
        Increment(Incrementer),
    }

    #[async_trait]
    impl ToolSet for OwnedTools {
        type Context<'ctx> = OwnedCtx;
        type Output = ToolCallResult;

        fn name(&self) -> &'static str {
            "increment"
        }

        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "increment".to_string(),
                description: "Bump.".to_string(),
                input_schema: serde_json::json!({ "type": "object" }),
            }
        }

        async fn execute<'ctx>(
            &self,
            _input: serde_json::Value,
            ctx: &mut OwnedCtx,
        ) -> Result<ToolCallResult, BorgError> {
            ctx.counter += 1;
            ToolCallResult::ok(serde_json::json!({ "new_value": ctx.counter }))
        }
    }

    #[tokio::test]
    async fn tool_set_borrowed_context_dispatches_and_mutates() {
        // The whole point of ToolSet: a context of `&mut` borrows.
        // `BorrowedCtx<'ctx>` carries `&'ctx mut u32`. The registry can
        // be reused across dispatch calls with different `'ctx`.
        let registry = ToolSetRegistry::<BorrowedTools>::new()
            .with(BorrowedTools::Increment(Incrementer))
            .with(BorrowedTools::Read(Reader));

        let mut counter: u32 = 0;
        {
            let mut ctx = BorrowedCtx {
                counter: &mut counter,
            };
            let req = ToolCallRequest {
                name: "increment".to_string(),
                args: serde_json::json!({}),
                call_id: "c1".to_string(),
            };
            let out = registry.dispatch(req, &mut ctx).await.unwrap();
            match out {
                DispatchOutcome::Ok { name, output, .. } => {
                    assert_eq!(name, "increment");
                    assert_eq!(output.output.unwrap()["new_value"], 1);
                }
                DispatchOutcome::Unknown { .. } => panic!("expected Ok, got Unknown"),
            }
        }
        // `ctx` is dropped, the `&mut counter` borrow released. The
        // counter now reflects the mutation.
        assert_eq!(counter, 1);
    }

    #[tokio::test]
    async fn tool_set_owned_context_also_works() {
        let registry =
            ToolSetRegistry::<OwnedTools>::new().with(OwnedTools::Increment(Incrementer));
        let mut ctx = OwnedCtx::default();
        let req = ToolCallRequest {
            name: "increment".to_string(),
            args: serde_json::json!({}),
            call_id: "c1".to_string(),
        };
        let out = registry.dispatch(req, &mut ctx).await.unwrap();
        match out {
            DispatchOutcome::Ok { output, .. } => {
                assert_eq!(output.output.unwrap()["new_value"], 1);
            }
            DispatchOutcome::Unknown { .. } => panic!("expected Ok"),
        }
        assert_eq!(ctx.counter, 1);
    }

    #[tokio::test]
    async fn tool_set_unknown_tool_surfaces_unknown_outcome() {
        let registry = ToolSetRegistry::<BorrowedTools>::new().with(BorrowedTools::Read(Reader));
        let mut counter: u32 = 0;
        let mut ctx = BorrowedCtx {
            counter: &mut counter,
        };
        let req = ToolCallRequest {
            name: "nonexistent".to_string(),
            args: serde_json::json!({}),
            call_id: "c1".to_string(),
        };
        let out = registry.dispatch(req, &mut ctx).await.unwrap();
        match out {
            DispatchOutcome::Unknown { name, call_id } => {
                assert_eq!(name, "nonexistent");
                assert_eq!(call_id, "c1");
            }
            DispatchOutcome::Ok { .. } => panic!("expected Unknown"),
        }
    }

    #[tokio::test]
    async fn tool_set_schemas_enumerates_variants() {
        let registry = ToolSetRegistry::<BorrowedTools>::new()
            .with(BorrowedTools::Increment(Incrementer))
            .with(BorrowedTools::Read(Reader));
        let schemas = registry.schemas();
        assert_eq!(schemas.len(), 2);
        let names: std::collections::HashSet<&str> =
            schemas.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains("increment"));
        assert!(names.contains("read"));
    }

    #[test]
    #[should_panic(expected = "duplicate tool name")]
    fn tool_set_register_duplicate_name_panics() {
        let mut registry = ToolSetRegistry::<BorrowedTools>::new();
        registry.register(BorrowedTools::Read(Reader));
        registry.register(BorrowedTools::Read(Reader));
    }
    // The BorgToolSet derive is exercised from downstream crates: inside
    // borg, `::borg::tools::ToolSet` doesn't resolve to `crate::tools`,
    // so the derive's emitted paths don't compile here. The manual-impl
    // tests above cover the trait + registry shape.

    // ─── active_tool_call task-local ─────────────────────────────────

    /// A tool that records what `active_tool_call()` returns at the
    /// moment its body runs. Lets us assert the dispatcher scoped the
    /// task-local with the expected (call_id, name).
    struct ProbeActiveCall {
        captured: std::sync::Arc<std::sync::Mutex<Option<ActiveToolCall>>>,
    }
    #[async_trait]
    impl Tool for ProbeActiveCall {
        type Context = ();
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "probe".into(),
                description: "Records active_tool_call inside execute.".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }
        async fn execute(
            &self,
            _input: serde_json::Value,
            _ctx: &mut Self::Context,
        ) -> Result<ToolCallResult, BorgError> {
            *self.captured.lock().unwrap() = active_tool_call();
            ToolCallResult::ok(serde_json::json!({}))
        }
    }

    #[tokio::test]
    async fn dispatch_scopes_active_tool_call_with_request_id() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let registry = Registry::<()>::new().with(ProbeActiveCall {
            captured: captured.clone(),
        });
        let req = ToolCallRequest {
            name: "probe".into(),
            args: serde_json::json!({}),
            call_id: "call_01ABC".into(),
        };
        let _ = registry.dispatch(req, &mut ()).await.unwrap();
        let seen = captured.lock().unwrap().clone().expect("scoped");
        assert_eq!(seen.call_id, "call_01ABC");
        assert_eq!(seen.name, "probe");
    }

    #[tokio::test]
    async fn active_tool_call_is_none_outside_dispatch() {
        // No dispatcher scope around this read — the task-local is
        // unset, so the helper returns None. Tools relying on this
        // should degrade gracefully.
        assert!(active_tool_call().is_none());
    }
}
