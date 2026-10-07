#!/usr/bin/env bash

set -eu

# Resolve paths relative to this script, even when invoked from another directory.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
EXAMPLE_MANIFEST="$PWD/Cargo.toml"

krnlc --manifest-path "$EXAMPLE_MANIFEST" "$@"
