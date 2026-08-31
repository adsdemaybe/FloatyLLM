#!/bin/bash
# Build the FloatyLLM Metal kernels + backend + smoke test on Apple Silicon.
# The main CMake project is CUDA-first (GB10); this builds the Metal side standalone until
# the session is refactored onto the Backend interface (CUDA-optional CMake is the next step).
set -e
cd "$(dirname "$0")"
OUT="${1:-/tmp}"

echo "[1/3] compile Metal kernels -> metallib"
xcrun metal -std=metal3.0 -c floaty_kernels.metal -o "$OUT/floaty_kernels.air"
xcrun metallib "$OUT/floaty_kernels.air" -o "$OUT/floaty_kernels.metallib"

echo "[2/3] compile + link smoke test"
clang++ -fobjc-arc -std=c++17 -O2 test_metal.mm backend_metal.mm -o "$OUT/test_metal" \
  -framework Metal -framework Foundation

echo "[3/3] run"
FLOATY_METALLIB="$OUT/floaty_kernels.metallib" "$OUT/test_metal"
