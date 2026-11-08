//! Proc macros for the `borg` agent runtime.
//!
//! `#[derive(BorgAgentDerive)]` generates the `borg::BorgAgent` impl
//! from a struct + `#[borg(...)]` attribute args.
//!
//! ## Consumer-visible names
//!
//! Consumers should depend on `borg`, not `borg-macros` directly. The
//! `borg` crate re-exports the macros under `*Derive` suffixes so the
//! derive name doesn't collide with the trait name in `borg::*` glob
//! imports:
//!
//! - `borg_macros::BorgAgent` → `borg::BorgAgentDerive`
//! - `borg_macros::BorgTool` → `borg::BorgToolDerive`
//! - `borg_macros::BorgToolSet` → `borg::BorgToolSetDerive`
//!
//! Examples below use the `*Derive` names so they compile under
//! `borg::*`. When depending on `borg-macros` directly, use the bare
//! `BorgAgent` / `BorgTool` / `BorgToolSet` names.
//!
//! # Example
//!
//! ```ignore
//! use borg::*;
//!
//! #[derive(BorgAgentDerive)]
//! #[borg(
//!     id = "code_review",
//!     model = "anthropic/claude-haiku-5-5",
//!     max_tokens = 2000,
//!     system = include_str!("../prompts/code_review.txt"),
//!     min_tokens = 5_000,
//! )]
//! struct CodeReview;
//!
//! impl CodeReview {
//!     /// User-supplied: `derive` doesn't generate `user_prompt` —
//!     /// it's the only method that varies meaningfully per agent.
//!     fn user_prompt(&self, ctx: &AgentContext) -> Option<String> {
//!         ctx.handoff.get_as("working_files")
//!     }
//! }
//! ```
//!
//! `derive` generates `agent_id`, `model`, `max_tokens`,
//! `system_prompt`, `min_tokens_estimate`, and (optionally)
//! `known_verdicts`. `user_prompt` stays user-defined.
//!
//! ## Custom `invoke`
//!
//! Agents whose `invoke` body differs from the default LLM-call flow
//! opt in with `#[borg(custom_invoke)]`; the derive then forwards
//! `invoke` to an inherent method, mirroring `user_prompt`:
//!
//! ```ignore
//! #[derive(BorgAgentDerive)]
//! #[borg(
//!     id = "planner",
//!     model = "anthropic/claude-sonnet-5-5",
//!     max_tokens = 600,
//!     system = "...",
//!     custom_invoke,
//! )]
//! struct Planner;
//!
//! impl Planner {
//!     fn user_prompt(&self, ctx: &AgentContext) -> Option<String> { ... }
//!     async fn invoke(&self, ctx: &AgentContext) -> Result<Verdict, BorgError> {
//!         // custom streaming, a scoped client, side-channel events, etc.
//!     }
//! }
//! ```
//!
//! Without the flag, the trait's default `invoke` is used.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Expr, ExprLit, Lit};

#[proc_macro_derive(BorgAgent, attributes(borg))]
pub fn derive_borg_agent(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let attrs = match parse_borg_attrs(&input) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    let id = attrs.id.expect("`#[borg(id = \"...\")]` is required");
    let model = attrs.model.expect("`#[borg(model = \"...\")]` is required");
    let max_tokens = attrs
        .max_tokens
        .expect("`#[borg(max_tokens = N)]` is required");
    let system = attrs
        .system
        .expect("`#[borg(system = \"...\" or include_str!(...))]` is required");
    let min_tokens = attrs.min_tokens.unwrap_or_else(|| quote! { 2_500 });
    let known_verdicts = attrs.known_verdicts.unwrap_or_else(|| quote! { &[] });

    // `#[borg(custom_invoke)]`: forward `invoke` to an inherent method,
    // so agents needing a non-default invoke body still get the derived
    // metadata methods.
    let invoke_slot = if attrs.custom_invoke {
        quote! {
            async fn invoke(
                &self,
                ctx: &::borg::AgentContext,
            ) -> ::core::result::Result<::borg::Verdict, ::borg::BorgError> {
                Self::invoke(self, ctx).await
            }
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #[::async_trait::async_trait]
        impl ::borg::BorgAgent for #name {
            fn agent_id(&self) -> &'static str { #id }
            fn model(&self) -> &str { #model }
            fn max_tokens(&self) -> u32 { #max_tokens }
            fn system_prompt(&self) -> &'static str { #system }
            fn user_prompt(&self, ctx: &::borg::AgentContext) -> Option<String> {
                Self::user_prompt(self, ctx)
            }
            fn known_verdicts(&self) -> &'static [&'static str] { #known_verdicts }
            fn min_tokens_estimate(&self) -> u64 { #min_tokens }
            #invoke_slot
        }
    };

    expanded.into()
}

#[derive(Default)]
struct BorgAttrs {
    id: Option<TokenStream2>,
    model: Option<TokenStream2>,
    max_tokens: Option<TokenStream2>,
    system: Option<TokenStream2>,
    min_tokens: Option<TokenStream2>,
    known_verdicts: Option<TokenStream2>,
    /// `#[borg(custom_invoke)]`: forward `BorgAgent::invoke` to an
    /// inherent `Self::invoke` method.
    custom_invoke: bool,
}

fn parse_borg_attrs(input: &DeriveInput) -> syn::Result<BorgAttrs> {
    let mut out = BorgAttrs::default();
    for attr in &input.attrs {
        if !attr.path().is_ident("borg") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            let key = meta
                .path
                .get_ident()
                .ok_or_else(|| meta.error("expected ident"))?
                .to_string();

            // Flag-style attrs (no `= value`) must be matched before the
            // `meta.value()` call below, which errors on a missing `=`.
            if key == "custom_invoke" {
                out.custom_invoke = true;
                return Ok(());
            }

            let value: Expr = meta.value()?.parse()?;
            let toks = quote! { #value };

            match key.as_str() {
                "id" => out.id = Some(string_or_expr(&value, toks)),
                "model" => out.model = Some(string_or_expr(&value, toks)),
                "max_tokens" => out.max_tokens = Some(toks),
                "system" => out.system = Some(toks),
                "min_tokens" => out.min_tokens = Some(toks),
                "known_verdicts" => out.known_verdicts = Some(toks),
                other => {
                    return Err(meta.error(format!(
                        "unknown #[borg] key `{other}`; expected one of: \
                         id, model, max_tokens, system, min_tokens, known_verdicts, \
                         custom_invoke (flag)"
                    )))
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// String literals are emitted as their contents (yielding `&'static str`);
/// anything else (`include_str!(...)`, const refs, integer literals) passes
/// through unchanged.
fn string_or_expr(value: &Expr, default: TokenStream2) -> TokenStream2 {
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(s), ..
    }) = value
    {
        let content = s.value();
        return quote! { #content };
    }
    default
}

// #[derive(BorgToolDerive)] — public name; bare `BorgTool` here.
//
// Generates `impl borg::tools::Tool for #name` from a struct +
// `#[tool(...)]` attribute. The user writes (via `borg::*`):
//
// ```ignore
// use borg::*;
//
// #[derive(BorgToolDerive)]
// #[tool(
//     name = "read_file",
//     description = "Read a file from disk.",
//     context = MyCtx,
//     input_schema = ::serde_json::json!({
//         "type": "object",
//         "properties": { "path": { "type": "string" } },
//         "required": ["path"],
//     }),
// )]
// struct ReadFile;
//
// impl ReadFile {
//     async fn execute(
//         &self,
//         input: ::serde_json::Value,
//         ctx: &mut MyCtx,
//     ) -> Result<::borg::ToolCallResult, ::borg::BorgError> { ... }
// }
// ```
//
// The derive emits `schema()` from the attrs and forwards `execute()`
// to the user's inherent method.

#[proc_macro_derive(BorgTool, attributes(tool))]
pub fn derive_borg_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let attrs = match parse_tool_attrs(&input) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    let tool_name = match attrs.name {
        Some(n) => n,
        None => {
            return syn::Error::new_spanned(&input, "`#[tool(name = \"...\")]` is required")
                .to_compile_error()
                .into();
        }
    };
    let description = match attrs.description {
        Some(d) => d,
        None => {
            return syn::Error::new_spanned(&input, "`#[tool(description = \"...\")]` is required")
                .to_compile_error()
                .into();
        }
    };
    let context = match attrs.context {
        Some(c) => c,
        None => {
            return syn::Error::new_spanned(
                &input,
                "`#[tool(context = MyType)]` is required — \
                 the consumer-defined per-call context type",
            )
            .to_compile_error()
            .into();
        }
    };
    let input_schema = match attrs.input_schema {
        Some(s) => s,
        None => {
            return syn::Error::new_spanned(
                &input,
                "`#[tool(input_schema = ...)]` is required — \
                 typically `::serde_json::json!({ \"type\": \"object\", ... })`",
            )
            .to_compile_error()
            .into();
        }
    };

    // `#[tool(parallel)]`: from ONE inherent `execute(&self, input,
    // ctx: &Context)` (shared), emit both `impl Tool` (re-borrowing
    // &mut down to &) and `impl ConcurrentTool`, both `parallelize()`
    // returning `true`. Requires `Context: Send + Sync` (enforced by
    // the ConcurrentTool bound).
    //
    // Without the flag, the inherent `execute` takes `&mut Context`
    // and only `impl Tool` is emitted.
    let expanded = if attrs.parallel {
        quote! {
            #[::async_trait::async_trait]
            impl ::borg::tools::Tool for #name {
                type Context = #context;

                fn schema(&self) -> ::borg::tools::ToolSchema {
                    ::borg::tools::ToolSchema::new(
                        #tool_name.to_string(),
                        #description.to_string(),
                        #input_schema,
                    )
                }

                async fn execute(
                    &self,
                    input: ::serde_json::Value,
                    ctx: &mut Self::Context,
                ) -> ::core::result::Result<
                    ::borg::tools::ToolCallResult,
                    ::borg::BorgError,
                > {
                    // Re-borrow &mut C down to &C for the shared-context body.
                    Self::execute(self, input, &*ctx).await
                }

                fn parallelize(&self) -> bool {
                    true
                }
            }

            #[::async_trait::async_trait]
            impl ::borg::tools::ConcurrentTool for #name {
                type Context = #context;

                fn schema(&self) -> ::borg::tools::ToolSchema {
                    ::borg::tools::ToolSchema::new(
                        #tool_name.to_string(),
                        #description.to_string(),
                        #input_schema,
                    )
                }

                async fn execute(
                    &self,
                    input: ::serde_json::Value,
                    ctx: &Self::Context,
                ) -> ::core::result::Result<
                    ::borg::tools::ToolCallResult,
                    ::borg::BorgError,
                > {
                    Self::execute(self, input, ctx).await
                }

                fn parallelize(&self) -> bool {
                    true
                }
            }
        }
    } else {
        quote! {
            #[::async_trait::async_trait]
            impl ::borg::tools::Tool for #name {
                type Context = #context;

                fn schema(&self) -> ::borg::tools::ToolSchema {
                    ::borg::tools::ToolSchema::new(
                        #tool_name.to_string(),
                        #description.to_string(),
                        #input_schema,
                    )
                }

                async fn execute(
                    &self,
                    input: ::serde_json::Value,
                    ctx: &mut Self::Context,
                ) -> ::core::result::Result<
                    ::borg::tools::ToolCallResult,
                    ::borg::BorgError,
                > {
                    Self::execute(self, input, ctx).await
                }
            }
        }
    };
    expanded.into()
}

#[derive(Default)]
struct ToolAttrs {
    name: Option<TokenStream2>,
    description: Option<TokenStream2>,
    context: Option<TokenStream2>,
    input_schema: Option<TokenStream2>,
    /// `#[tool(parallel)]`: emit both `impl Tool` and
    /// `impl ConcurrentTool` from one shared-context inherent
    /// `execute`. Requires `Context: Send + Sync`.
    parallel: bool,
}

fn parse_tool_attrs(input: &DeriveInput) -> syn::Result<ToolAttrs> {
    let mut out = ToolAttrs::default();
    for attr in &input.attrs {
        if !attr.path().is_ident("tool") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            let key = meta
                .path
                .get_ident()
                .ok_or_else(|| meta.error("expected ident"))?
                .to_string();

            // Flag-style attrs (no `= value`) must be matched before the
            // `meta.value()` call below, which errors on a missing `=`.
            if key == "parallel" {
                out.parallel = true;
                return Ok(());
            }

            let value: Expr = meta.value()?.parse()?;
            let toks = quote! { #value };
            match key.as_str() {
                "name" => out.name = Some(string_or_expr(&value, toks)),
                "description" => out.description = Some(string_or_expr(&value, toks)),
                "context" => out.context = Some(toks),
                "input_schema" => out.input_schema = Some(toks),
                other => {
                    return Err(meta.error(format!(
                        "unknown #[tool] key `{other}`; expected one of: \
                         name, description, context, input_schema, parallel (flag)"
                    )));
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

// #[derive(BorgToolSetDerive)] — public name; bare `BorgToolSet` here.
//
// Generates `impl borg::tools::ToolSet for #enum_name` from an enum of
// tuple variants holding tool markers. Each variant's inner type must
// provide inherent methods:
//
// ```ignore
// impl ReadFileTool {
//     pub fn name() -> &'static str { "read_file" }
//     pub fn schema() -> borg::tools::ToolSchema { /* ... */ }
//     pub async fn execute(
//         &self,
//         input: serde_json::Value,
//         ctx: &mut MyCtx<'_>,
//     ) -> Result<borg::tools::ToolCallResult, borg::BorgError> { /* ... */ }
// }
// ```
//
// The enum carries `#[borg(context = MyCtx<'ctx>)]` to declare the
// `Context<'ctx>` GAT instantiation.
//
// # Example (via `borg::*`)
//
// ```ignore
// use borg::*;
//
// #[derive(BorgToolSetDerive)]
// #[borg(context = ToolContext<'ctx>)]
// pub enum EditorTools {
//     ReadFile(ReadFileTool),
//     Outline(OutlineTool),
// }
// ```
//
// Generates:
//
// ```ignore
// impl borg::tools::ToolSet for EditorTools {
//     type Context<'ctx> = ToolContext<'ctx>;
//     fn name(&self) -> &'static str { match self { ... } }
//     fn schema(&self) -> ToolSchema { match self { ... } }
//     async fn execute<'ctx>(...) -> ... { match self { ... } }
// }
// ```

#[proc_macro_derive(BorgToolSet, attributes(borg))]
pub fn derive_borg_tool_set(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let enum_name = &input.ident;

    // The GAT instantiation, required because `ToolSet::Context<'ctx>`
    // can't be inferred from the variant tool types alone.
    let attrs = match parse_tool_set_attrs(&input) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let context = attrs.context;
    // Defaults to `ToolCallResult` (the LLM-facing dispatch shape);
    // override with `#[borg(returns = MyType)]` for typed results.
    let output = attrs
        .returns
        .unwrap_or_else(|| quote! { ::borg::tools::ToolCallResult });

    let data_enum = match &input.data {
        syn::Data::Enum(e) => e,
        _ => {
            return syn::Error::new_spanned(
                &input,
                "`#[derive(BorgToolSet)]` only applies to enums whose \
                 variants are tuple variants of one tool-marker type",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut name_arms = Vec::new();
    let mut schema_arms = Vec::new();
    let mut execute_arms = Vec::new();

    for variant in &data_enum.variants {
        let var_ident = &variant.ident;
        let field_ty = match &variant.fields {
            syn::Fields::Unnamed(f) if f.unnamed.len() == 1 => &f.unnamed.first().unwrap().ty,
            _ => {
                return syn::Error::new_spanned(
                    variant,
                    "each variant must be a tuple variant with exactly one \
                     tool-marker type (e.g. `ReadFile(ReadFileTool)`)",
                )
                .to_compile_error()
                .into();
            }
        };

        name_arms.push(quote! {
            Self::#var_ident(_) => <#field_ty>::name()
        });
        schema_arms.push(quote! {
            Self::#var_ident(_) => <#field_ty>::schema()
        });
        execute_arms.push(quote! {
            Self::#var_ident(t) => t.execute(input, ctx).await
        });
    }

    let expanded = quote! {
        #[::async_trait::async_trait]
        impl ::borg::tools::ToolSet for #enum_name {
            type Context<'ctx> = #context where Self: 'ctx;
            type Output = #output;

            fn name(&self) -> &'static str {
                match self {
                    #(#name_arms,)*
                }
            }

            fn schema(&self) -> ::borg::tools::ToolSchema {
                match self {
                    #(#schema_arms,)*
                }
            }

            async fn execute<'ctx>(
                &self,
                input: ::serde_json::Value,
                ctx: &mut Self::Context<'ctx>,
            ) -> ::core::result::Result<
                Self::Output,
                ::borg::BorgError,
            > {
                match self {
                    #(#execute_arms,)*
                }
            }
        }
    };
    expanded.into()
}

/// Wraps `invoke` and/or `invoke_streaming` in a named `tracing::info_span`.
///
/// Usage: `#[borg_instrument(name = "my_agent")]`
///
/// The span inherits the current tracing context as parent (e.g. the
/// `borg.step` span when run under borg's engine with `TracingEmitter`).
/// The async body is instrumented with `tracing::Instrument` so the span
/// propagates across `.await` points and stays `Send` for `#[async_trait]`.
///
/// **Ordering**: place `#[borg_instrument]` *before* `#[async_trait]` so this
/// macro sees the original `async fn` syntax before async_trait rewrites it.
///
/// The consumer's crate must have `tracing` in scope; `borg-macros` does not
/// depend on `tracing` itself.
#[proc_macro_attribute]
pub fn borg_instrument(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as BorgInstrumentArgs);
    let mut impl_block = parse_macro_input!(item as syn::ItemImpl);

    let span_name = args.name;
    // A misspelled method name would otherwise make this attribute a
    // silent no-op; track matches so we can compile_error! instead.
    let mut matched = false;

    for impl_item in &mut impl_block.items {
        if let syn::ImplItem::Fn(method) = impl_item {
            let fn_name = method.sig.ident.to_string();
            if fn_name == "invoke" || fn_name == "invoke_streaming" {
                matched = true;
                let original_stmts = method.block.stmts.clone();
                method.block = syn::parse_quote! {
                    {
                        let __borg_span = ::tracing::info_span!(#span_name);
                        ::tracing::Instrument::instrument(
                            async move { #(#original_stmts)* },
                            __borg_span,
                        ).await
                    }
                };
            }
        }
    }

    if !matched {
        return syn::Error::new_spanned(
            &impl_block,
            "#[borg_instrument] matched no methods — expected an impl \
             block containing `invoke` or `invoke_streaming`. Check the \
             method names for typos.",
        )
        .to_compile_error()
        .into();
    }

    quote! { #impl_block }.into()
}

struct BorgInstrumentArgs {
    name: syn::LitStr,
}

impl syn::parse::Parse for BorgInstrumentArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;
        if ident != "name" {
            return Err(syn::Error::new(ident.span(), "expected `name = \"...\"`"));
        }
        let _: syn::Token![=] = input.parse()?;
        let name: syn::LitStr = input.parse()?;
        Ok(Self { name })
    }
}

struct ToolSetAttrs {
    context: TokenStream2,
    returns: Option<TokenStream2>,
}

fn parse_tool_set_attrs(input: &DeriveInput) -> syn::Result<ToolSetAttrs> {
    let mut context: Option<TokenStream2> = None;
    let mut returns: Option<TokenStream2> = None;
    for attr in &input.attrs {
        if !attr.path().is_ident("borg") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            let key = meta
                .path
                .get_ident()
                .ok_or_else(|| meta.error("expected ident"))?
                .to_string();
            // Parse as a Type, not an Expr: both keys take Rust types,
            // often lifetime-parameterized (e.g. `MyCtx<'ctx>`).
            let value: syn::Type = meta.value()?.parse()?;
            let toks = quote! { #value };
            match key.as_str() {
                "context" => context = Some(toks),
                "returns" => returns = Some(toks),
                other => {
                    return Err(meta.error(format!(
                        "unknown #[borg] key `{other}` on `#[derive(BorgToolSet)]`; \
                         expected one of: context, returns"
                    )));
                }
            }
            Ok(())
        })?;
    }
    let context = context.ok_or_else(|| {
        syn::Error::new_spanned(
            input,
            "`#[derive(BorgToolSet)]` requires `#[borg(context = MyCtx<'ctx>)]` \
             to declare the GAT Context instantiation. \
             Use `#[borg(context = ())]` for owned/contextless tools.",
        )
    })?;
    Ok(ToolSetAttrs { context, returns })
}
