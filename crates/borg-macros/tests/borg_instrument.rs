//! Compile-time test: `#[borg_instrument]` expands cleanly and the
//! resulting type still satisfies `BorgAgent`. No runtime assertions.

use async_trait::async_trait;
use borg_macros::borg_instrument;

#[allow(dead_code)]
struct MyAgent;

// Must come BEFORE #[async_trait] to see the original `async fn` syntax.
#[borg_instrument(name = "my_agent")]
#[async_trait]
impl borg::BorgAgent for MyAgent {
    fn agent_id(&self) -> &'static str {
        "my_agent"
    }
    fn model(&self) -> &str {
        "test-model"
    }
    fn max_tokens(&self) -> u32 {
        100
    }
    fn system_prompt(&self) -> &'static str {
        "sys"
    }
    fn user_prompt(&self, _ctx: &borg::AgentContext) -> Option<String> {
        None
    }
    async fn invoke(&self, _ctx: &borg::AgentContext) -> Result<borg::Verdict, borg::BorgError> {
        Ok(borg::Verdict::empty())
    }
}

fn _assert_impl(_: &dyn borg::BorgAgent) {}
fn _check() {
    _assert_impl(&MyAgent);
}
