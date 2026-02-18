#!/bin/bash

krnlc || true
rm target/krnlc/crates/affine-gpu-example/Cargo.lock
krnlc
