//! A streaming actor as the main loop of a long-lived chat session.
//!
//! 1. `spawn_streaming_actor` starts the session; each turn streams events
//!    (`TextDelta`, `Done`) as they are produced.
//! 2. `run_session_loop` drives the actor turn after turn: the actor says
//!    `Continue` while it still has work, then `Park` when the reply is done.
//! 3. The process "restarts": the handle is cancelled (the runtime writes a
//!    final snapshot), and `resume_streaming_actor` brings the session back
//!    from the store with its history intact.
//!
//! No model is called; the actor streams scripted text, so this runs offline.
//!
//! Run with:
//!   cargo run -p borg-loop --example streaming_session

use async_trait::async_trait;
use borg::{BorgError, StreamEvent, Verdict};
use borg_loop::{
    resume_streaming_actor, spawn_streaming_actor, ActorConfig, ActorCtx, ActorStore,
    InMemoryActorStore, SessionOutcome, StreamingActorAgent, StreamingHandleOutcome,
};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

enum Msg {
    /// A new message from the user.
    User(String),
    /// "Keep going": the driver sends this while the actor has work left.
    Continue,
}

/// What survives a restart.
#[derive(Serialize, Deserialize, Default)]
struct ChatSnap {
    history: Vec<String>,
}

struct Chat {
    history: Vec<String>,
    /// Steps still to do for the current request.
    pending: Vec<String>,
}

impl Chat {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            pending: Vec::new(),
        }
    }
}

/// Stream a line word by word, the way a model streams tokens.
fn stream_line(line: String) -> BoxStream<'static, StreamEvent> {
    Box::pin(async_stream::stream! {
        for word in line.split_inclusive(' ') {
            yield StreamEvent::TextDelta(word.to_string());
        }
        yield StreamEvent::TextDelta("\n".to_string());
        yield StreamEvent::Done(Verdict::empty());
    })
}

#[async_trait]
impl StreamingActorAgent for Chat {
    type Message = Msg;
    type Snapshot = ChatSnap;

    async fn handle_streaming(
        &mut self,
        msg: Msg,
        _ctx: &ActorCtx,
    ) -> Result<(BoxStream<'static, StreamEvent>, StreamingHandleOutcome), BorgError> {
        let line = match msg {
            Msg::User(text) => {
                let remembered = self.history.len();
                self.history.push(format!("user: {text}"));
                self.pending = vec![
                    "Reading the workspace layout.".into(),
                    "Found 7 crates; the runtime lives in `borg`.".into(),
                ];
                format!("On it ({remembered} lines of history): {text}")
            }
            Msg::Continue => match self.pending.first().cloned() {
                Some(step) => {
                    self.pending.remove(0);
                    step
                }
                None => "Done.".into(),
            },
        };
        self.history.push(format!("agent: {line}"));

        // More steps queued: ask the driver for another turn. Otherwise park
        // and wait for the user's next message.
        let outcome = if self.pending.is_empty() {
            StreamingHandleOutcome::Park
        } else {
            StreamingHandleOutcome::Continue
        };
        Ok((stream_line(line), outcome))
    }

    fn checkpoint(&self) -> ChatSnap {
        ChatSnap {
            history: self.history.clone(),
        }
    }

    fn restore(snap: ChatSnap) -> Self {
        Self {
            history: snap.history,
            pending: Vec::new(),
        }
    }
}

fn print_event(ev: StreamEvent) {
    if let StreamEvent::TextDelta(t) = ev {
        print!("{t}");
    }
}

#[tokio::main]
async fn main() {
    let store: Arc<dyn ActorStore> = Arc::new(InMemoryActorStore::new());

    // ── 1–2. Start the session and drive it until it parks ───────────
    let session = spawn_streaming_actor(
        "session-42",
        Chat::new(),
        ActorConfig::for_streaming(), // checkpoint after every turn
        store.clone(),
        None,
    );
    let outcome = session
        .run_session_loop(
            Msg::User("map this repo".into()),
            || Msg::Continue,
            print_event,
        )
        .await;
    assert_eq!(outcome, SessionOutcome::Parked);
    println!("-- parked, waiting for the user");

    // ── 3. Restart: stop the process, bring the session back ─────────
    session.cancel();
    session.join().await;

    let resumed = resume_streaming_actor(
        "session-42",
        Chat::new,
        Chat::restore,
        ActorConfig::for_streaming(),
        store,
        None,
    )
    .await
    .expect("store is readable");
    let outcome = resumed
        .run_session_loop(
            Msg::User("and the tests?".into()),
            || Msg::Continue,
            print_event,
        )
        .await;
    assert_eq!(outcome, SessionOutcome::Parked);
    println!("-- resumed with history intact, parked again");

    resumed.cancel();
    resumed.join().await;
}
