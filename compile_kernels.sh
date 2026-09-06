#!/usr/bin/env bash

set -eu

# Resolve paths relative to this script, even when invoked from another directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
EXAMPLE_MANIFEST="$PWD/Cargo.toml"

# Running from krnlc's directory selects its pinned rust-toolchain.toml.
# Use the local compiler, matching the local krnl dependency in Cargo.toml.
(
    cd ../krnl/krnlc
    cargo run --locked --release -- --manifest-path "$EXAMPLE_MANIFEST" "$@"
)
