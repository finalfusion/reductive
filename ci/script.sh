#!/bin/bash

set -euxo pipefail

# On Rust 1.32.0, we only care about passing tests.
if rustc --version | grep -v "^rustc 1.37.0"; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi

cargo build --verbose
cargo test --verbose
cargo build --verbose --features "openblas-test"
cargo test --verbose --features "openblas-test"
