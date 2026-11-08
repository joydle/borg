.PHONY: test fmt clippy

test:
	cargo test --workspace --all-features

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings
