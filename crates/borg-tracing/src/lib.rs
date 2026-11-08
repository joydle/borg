//! Tracing integration for borg — wires [`borg::BorgEvent`] into `tracing` spans.
//!
//! - [`TracingEmitter`] — [`borg::EventEmitter`] that opens/closes `tracing`
//!   spans on step and turn lifecycle events.
//! - [`MultiEmitter`] — fan-out emitter that delegates each [`borg::BorgEvent`]
//!   to a pair of [`borg::EventEmitter`]s (e.g. tracing plus metrics).
//! - [`SpanRegistry`] — the bounded in-process map `TracingEmitter` uses
//!   internally to correlate child events to their parent span. Exposed
//!   for benchmarking; you don't need it to wire tracing.
//!
//! ## Wiring example
//!
//! ```ignore
//! use std::sync::Arc;
//! use borg_tracing::{MultiEmitter, TracingEmitter};
//!
//! // Standalone: spans only.
//! let emitter = Arc::new(TracingEmitter::new());
//!
//! // Or under your request span, fanned out alongside an existing sink.
//! let emitter = Arc::new(MultiEmitter::new(
//!     TracingEmitter::with_parent(request_span),
//!     my_existing_emitter,
//! ));
//!
//! let ctx = borg::ExecCtx::new(llm, budget, emitter, cancel, "session-1");
//! ```

mod emitter;
pub mod fields;
mod multi_emitter;
mod span_store;

pub use emitter::TracingEmitter;
pub use multi_emitter::MultiEmitter;
pub use span_store::{SpanKey, SpanRegistry};
