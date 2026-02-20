#!/usr/bin/env bash

TOOLCHAIN=nightly-2025-06-30
KRNLC_REPO=https://github.com/jlogan03/krnl.git
KRNLC_BRANCH=jlogan/update-deps

# Keep krnlc aligned with the krnl git dependency to avoid version mismatch errors.
cargo +"$TOOLCHAIN" install --git "$KRNLC_REPO" --branch "$KRNLC_BRANCH" --locked krnlc

krnlc
