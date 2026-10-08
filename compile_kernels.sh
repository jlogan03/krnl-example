#!/usr/bin/env bash

set -eu

# Use the project directory regardless of the caller's working directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
EXAMPLE_MANIFEST="$PWD/Cargo.toml"

krnlc --manifest-path "$EXAMPLE_MANIFEST" "$@"
