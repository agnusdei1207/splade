#!/usr/bin/env bash
set -euo pipefail

action="$1"
shift

case "$action" in
  lock)
    cargo generate-lockfile "$@"
    ;;
  format)
    rustup component add rustfmt
    cargo fmt --all "$@"
    ;;
  fmt)
    rustup component add rustfmt
    cargo fmt --all -- --check "$@"
    ;;
  test)
    cargo test --locked "$@"
    ;;
  clippy)
    rustup component add clippy
    cargo clippy --locked --all-targets -- -D warnings "$@"
    ;;
  bench)
    cargo run --release --locked --bin rust-benchmark -- "$@"
    ;;
  *)
    echo "unsupported Rust action: $action" >&2
    exit 2
    ;;
esac
