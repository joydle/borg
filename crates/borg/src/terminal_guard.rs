//! RAII safety net for terminal state writes.
//!
//! [`TerminalGuard`] is a Drop-based backstop for "every supervised
//! task MUST produce a terminal record exactly once." Construct one
//! with a fallback terminal value; the happy path calls
//! [`TerminalGuard::fire`] with the real terminal; the `Drop` impl
//! writes the fallback if `fire` was never reached (panic, abandoned
//! future, early return).
//!
//! Without it, a task that exits without writing its terminal leaves
//! a session stuck in `Running` forever; the guard catches that at the
//! executor boundary with a conservative default and a warn line.
//!
//! Limitations:
//! - Not a substitute for explicit `fire` — routinely relying on Drop
//!   to write terminals is a bug.
//! - The Drop impl can't `.await`. For async terminal writes, wrap the
//!   call in `tokio::spawn` inside the callback.
//! - Not cross-process: a SIGKILL bypasses `Drop` entirely; stale
//!   `Running` rows are a reconciler's problem.
//!
//! ## Worked example
//!
//! ```ignore
//! use borg::TerminalGuard;
//!
//! enum Terminal { Completed(String), Failed { reason: String } }
//!
//! async fn run_session(session_id: String) {
//!     let id = session_id.clone();
//!     let guard = TerminalGuard::new(
//!         Terminal::Failed { reason: "internal: guard fallback".into() },
//!         move |terminal| {
//!             // Fire-and-forget the async write.
//!             tokio::spawn(async move { write_terminal(&id, terminal).await });
//!         },
//!     );
//!
//!     match do_work().await {
//!         Ok(out) => guard.fire(Terminal::Completed(out)),
//!         Err(e) => guard.fire(Terminal::Failed { reason: e.to_string() }),
//!     }
//!     // If we panic between guard construction and `fire`, the Drop
//!     // impl fires the Failed{guard fallback} terminal.
//! }
//! ```

/// RAII guard that runs a terminal callback exactly once, either via
/// explicit [`fire`](Self::fire) or via [`Drop`] (the fallback path).
///
/// `T` is the terminal value type. The `Send` bound lets the guard sit
/// across `.await` boundaries inside `tokio::spawn`.
pub struct TerminalGuard<T: Send + 'static> {
    /// `None` after `fire` / `disarm` / `Drop`; otherwise holds the
    /// fallback value and the callback.
    state: Option<State<T>>,
}

struct State<T> {
    fallback: T,
    callback: Box<dyn FnOnce(T) + Send + 'static>,
}

impl<T: Send + 'static> TerminalGuard<T> {
    /// Build a guard. `callback` runs exactly once: with the value
    /// passed to `fire`, or with `fallback` if `Drop` fires first.
    pub fn new<F>(fallback: T, callback: F) -> Self
    where
        F: FnOnce(T) + Send + 'static,
    {
        Self {
            state: Some(State {
                fallback,
                callback: Box::new(callback),
            }),
        }
    }

    /// Happy path: write the terminal explicitly. Consumes the guard
    /// so `Drop` becomes a no-op.
    pub fn fire(mut self, terminal: T) {
        if let Some(state) = self.state.take() {
            (state.callback)(terminal);
        }
    }

    /// Suppress both the explicit write and the Drop-fallback. Use when
    /// the terminal was already written through some other path. Rare;
    /// most code should `fire`.
    pub fn disarm(mut self) {
        self.state.take();
    }

    /// Whether the guard is still armed; false after `fire` / `disarm`
    /// / `Drop`. Useful for `debug_assert!`.
    pub fn is_armed(&self) -> bool {
        self.state.is_some()
    }
}

impl<T: Send + 'static> Drop for TerminalGuard<T> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            // Hitting the fallback is always an upstream bug.
            tracing::warn!(
                "TerminalGuard fired its fallback — explicit `fire()` was never reached. \
                 This usually indicates a panic, an early-return without explicit terminal \
                 write, or a future being dropped due to cancellation."
            );
            (state.callback)(state.fallback);
        }
    }
}

impl<T: Send + 'static> std::fmt::Debug for TerminalGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalGuard")
            .field("armed", &self.is_armed())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Terminal {
        Completed,
        Failed(String),
    }

    fn capture() -> (
        Arc<Mutex<Vec<Terminal>>>,
        impl FnOnce(Terminal) + Send + 'static,
    ) {
        let captured: Arc<Mutex<Vec<Terminal>>> = Arc::new(Mutex::new(Vec::new()));
        let inner = captured.clone();
        let cb = move |t: Terminal| {
            inner.lock().unwrap().push(t);
        };
        (captured, cb)
    }

    #[test]
    fn fire_runs_callback_with_explicit_value() {
        let (captured, cb) = capture();
        let g = TerminalGuard::new(Terminal::Failed("fallback".into()), cb);
        g.fire(Terminal::Completed);
        let log = captured.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0], Terminal::Completed);
    }

    #[test]
    fn drop_runs_callback_with_fallback_value() {
        let (captured, cb) = capture();
        {
            let _g = TerminalGuard::new(Terminal::Failed("fallback".into()), cb);
            // No fire, no disarm: Drop fires the fallback.
        }
        let log = captured.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0], Terminal::Failed("fallback".into()));
    }

    #[test]
    fn disarm_suppresses_both_paths() {
        let (captured, cb) = capture();
        let g = TerminalGuard::new(Terminal::Failed("fallback".into()), cb);
        g.disarm();
        assert!(captured.lock().unwrap().is_empty());
    }

    #[test]
    fn callback_runs_exactly_once_via_fire() {
        let counter = Arc::new(AtomicU32::new(0));
        let inner = counter.clone();
        let g = TerminalGuard::new(Terminal::Failed("x".into()), move |_| {
            inner.fetch_add(1, Ordering::SeqCst);
        });
        g.fire(Terminal::Completed);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn callback_runs_exactly_once_via_drop() {
        let counter = Arc::new(AtomicU32::new(0));
        let inner = counter.clone();
        {
            let _g = TerminalGuard::new(Terminal::Failed("x".into()), move |_| {
                inner.fetch_add(1, Ordering::SeqCst);
            });
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn is_armed_reflects_lifecycle() {
        let (_captured, cb) = capture();
        let g = TerminalGuard::new(Terminal::Failed("fb".into()), cb);
        assert!(g.is_armed());
        g.fire(Terminal::Completed);
        // post-fire is unobservable: `fire` consumes the guard.

        let (_c2, cb2) = capture();
        let g2 = TerminalGuard::new(Terminal::Failed("fb".into()), cb2);
        assert!(g2.is_armed());
        g2.disarm();
    }

    #[tokio::test]
    async fn callback_can_spawn_async_work() {
        // Async terminal writes spawn inside the callback.
        use tokio::sync::oneshot;

        let (tx, rx) = oneshot::channel();
        let tx = std::sync::Mutex::new(Some(tx));

        let g = TerminalGuard::new(Terminal::Failed("fallback".into()), move |t| {
            let sender = tx.lock().unwrap().take().unwrap();
            tokio::spawn(async move {
                sender.send(t).unwrap();
            });
        });

        g.fire(Terminal::Completed);

        let received = rx.await.unwrap();
        assert_eq!(received, Terminal::Completed);
    }

    #[test]
    fn drop_fires_fallback_on_panic_caught_by_catch_unwind() {
        // A panic in the protected scope still writes the terminal via
        // Drop.
        let (captured, cb) = capture();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = TerminalGuard::new(Terminal::Failed("panic_fallback".into()), cb);
            panic!("simulated work failure");
        }));
        assert!(result.is_err());
        let log = captured.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0], Terminal::Failed("panic_fallback".into()));
    }
}
