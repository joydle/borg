//! `borg-loop` — long-lived event-loop primitive on top of [`borg`].
//!
//! Sized for **persistent multi-turn agent sessions**: a session that
//! spans 100+ messages over hours or days, survives the user closing
//! the browser, and resumes where it left off.
//!
//! ## Terminology
//!
//! | Term | Crate | Shape |
//! |------|-------|-------|
//! | **Agent** (`BorgAgent`) | `borg` | Stateless, one-shot, returns `Verdict` |
//! | **Streaming agent** (`BorgStreamingAgent`) | `borg` | Stateless, one-shot, yields `StreamEvent` |
//! | **Actor** (`ActorAgent`) | `borg-loop` | Stateful, message-driven, checkpointable |
//! | **Streaming actor** (`StreamingActorAgent`) | `borg-loop` | Stateful, per-turn `BoxStream<StreamEvent>` |
//!
//! "Agent" and "actor" are **not interchangeable**: agents are ephemeral
//! (constructed per invocation), actors are persistent (one instance per
//! session, revived from a snapshot across restarts).
//!
//! ## What this crate provides
//!
//! 1. [`ActorAgent`] — the trait long-lived actors implement: one
//!    typed message handler plus checkpoint/restore.
//! 2. [`spawn_actor`] / [`resume_actor`] — spawn fresh, or resume from
//!    a saved snapshot.
//! 3. [`ActorHandle`] — caller-side address. Send, cancel, await exit.
//!    Cancel-on-drop.
//! 4. [`ActorStore`] — pluggable persistence trait + in-memory and
//!    noop impls.
//! 5. [`StreamingActorAgent`] — variant whose turns produce a live
//!    `BoxStream<StreamEvent>` (LLM streaming + tool-loop turns).
//! 6. [`spawn_streaming_actor`] / [`resume_streaming_actor`] — runtime
//!    for streaming actors, same checkpoint/resume lifecycle.
//! 7. [`StreamingActorHandle`] — per-message event receiver; each
//!    `send(msg)` returns a `Receiver<StreamEvent>` the caller drains live.
//!
//! ## Scope
//!
//! Not a full actor system: no monitor refs, linked processes,
//! distributed addressing, dead-letter queues, selective receive, or
//! multi-level supervision. For those, reach for `Ractor`.
//!
//! The dependency direction is strictly `borg-loop` → `borg`, never the
//! reverse, so clients needing only `borg::Composition` /
//! `borg::Scheduled` keep a lean footprint.
//!
//! ## Minimal example
//!
//! ```ignore
//! use async_trait::async_trait;
//! use borg_loop::*;
//! use serde::{Deserialize, Serialize};
//! use std::sync::Arc;
//!
//! enum Msg { Increment, Snapshot(tokio::sync::oneshot::Sender<u32>), Stop }
//!
//! #[derive(Serialize, Deserialize)]
//! struct CounterSnap { count: u32 }
//!
//! struct Counter { count: u32 }
//!
//! #[async_trait]
//! impl ActorAgent for Counter {
//!     type Message = Msg;
//!     type Snapshot = CounterSnap;
//!
//!     async fn handle(&mut self, msg: Msg, _ctx: &ActorCtx) -> HandleOutcome {
//!         match msg {
//!             Msg::Increment => { self.count += 1; HandleOutcome::Continue }
//!             Msg::Snapshot(reply) => { let _ = reply.send(self.count); HandleOutcome::Continue }
//!             Msg::Stop => HandleOutcome::Terminal("user requested stop".into()),
//!         }
//!     }
//!     fn checkpoint(&self) -> CounterSnap { CounterSnap { count: self.count } }
//!     fn restore(s: CounterSnap) -> Self { Self { count: s.count } }
//! }
//!
//! # async fn run() {
//! let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());
//! let h = spawn_actor("session-42", Counter { count: 0 }, ActorConfig::default(), store, None);
//! h.send(Msg::Increment).await.unwrap();
//! h.send(Msg::Stop).await.unwrap();
//! let exit = h.join().await;
//! assert!(matches!(exit, ActorExit::Terminal(_)));
//! # }
//! ```

mod actor;
mod driver;
mod handle;
mod registry;
mod runtime;
mod store;
mod streaming_actor;
mod streaming_runtime;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_streaming;

pub use actor::{ActorAgent, HandleOutcome};
pub use driver::SessionOutcome;
pub use handle::{ActorExit, ActorHandle};
pub use registry::ActorRegistry;
pub use runtime::{resume_actor, spawn_actor, ActorConfig, ActorCtx};
pub use store::{
    ActorStore, ActorStoreError, CachingActorStore, InMemoryActorStore, NoopActorStore,
    ResilientActorStore,
};
pub use streaming_actor::{
    StreamingActorAgent, StreamingActorHandle, StreamingHandleOutcome, TurnHandle,
};
pub use streaming_runtime::{resume_streaming_actor, spawn_streaming_actor};
