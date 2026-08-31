#!/bin/bash
# Build the full backend-agnostic FloatyLLM runner for Apple Silicon (Metal). Produces a
# `floaty_run` binary + the metallib. Until the top-level CMake is made CUDA-optional, this
# is the Mac build. Run:  bash src/metal/build_runner.sh && \
#   FLOATY_METALLIB=<out>/floaty_kernels.metallib <out>/floaty_run <model.gguf> <n_gen> [ids...]
set -e
cd "$(dirname "$0")/../.."          # repo root
OUT="${1:-/tmp}"
mkdir -p "$OUT"

echo "[1/2] Metal kernels -> metallib"
xcrun metal -std=metal3.0 -c src/metal/floaty_kernels.metal -o "$OUT/fk.air"
xcrun metallib "$OUT/fk.air" -o "$OUT/floaty_kernels.metallib"

echo "[2/2] compile runner (Metal backend + portable loader)"
clang++ -std=c++17 -fobjc-arc -O2 -DFLOATY_METAL -Isrc \
  src/floaty_run.cpp src/floaty_model.cpp src/loader.cpp src/metal/backend_metal.mm \
  -o "$OUT/floaty_run" -framework Metal -framework Foundation

echo "built $OUT/floaty_run"
echo "run: FLOATY_METALLIB=$OUT/floaty_kernels.metallib $OUT/floaty_run <model.gguf> 40"
