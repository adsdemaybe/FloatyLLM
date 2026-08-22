# SemiLLM

Latency-hidden mixed-precision streaming LLM inference, written in CUDA.

Runs models larger than VRAM by streaming quantized transformer layers CPU to GPU over an
overlapped 3-stage CUDA pipeline (copy / compute / free), dequantizing on the GPU, and setting
per-layer bit-width by pipeline slack + layer sensitivity. Goal: a perplexity vs tokens/s frontier
that beats uniform-quant streaming (and AirLLM).

See [PLAN.md](PLAN.md) for the full design.

## Conventions

- Code is CUDA / C++ (own scheduler + forward loop; cuBLASLt for GEMM). No Python in the hot loop;
  Python only for the offline packer.
- Single-line `//` comments only. Block / multiline `/* */` comments are rejected by the linter.
  Run `python3 tools/lint_comments.py`.
