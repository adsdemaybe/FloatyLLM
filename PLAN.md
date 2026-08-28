# SemiLLM — Latency-Hidden Mixed-Precision Streaming Inference (CUDA)

## 1. Context & thesis
Run models larger than VRAM by streaming transformer layers CPU→GPU. Bottleneck = PCIe
bandwidth; GPU starves waiting on weights. AirLLM does this but (verified from source):
**ships fp16 over PCIe in every mode** (compression only helps disk→RAM; it dequants to fp16 on
CPU *before* the PCIe transfer), and its overlap is a Python `ThreadPoolExecutor` (~10%, off when
compressed).

**Thesis — latency-hidden mixed-precision streaming.** In the streaming regime you are
*load-bound* (t_load ≫ t_compute), so each layer's **bit-width is both a speed knob (PCIe bytes)
and an accuracy knob (quant error)**. Keep weights quantized all the way to VRAM (dequant on GPU),
and set **each layer's bit-width by pipeline slack + layer sensitivity**: layers with compute
slack ride at higher bits *for free* (hidden behind compute); the freed bit-budget goes to the
most quant-sensitive layers. AirLLM can't — its PCIe leg is always fp16, no per-layer knob.

**Headline result:** a **perplexity ↔ tokens/s Pareto frontier** that dominates uniform-quant
streaming (which dominates fp16 streaming). That is the numerical contribution — not the raw
speedup (~3–6×/token on 70B, mostly from quant).

Target: Llama-7B/8B first (correctness), 70B as the real payload. GGUF Q4_0/Q5_0/Q8_0 (+Q6_K
later). 12–16 GB GPU.

## 2. Key decision: own the loop, borrow only heavy kernels
Reject ggml-as-engine (keeps model resident, owns memory/streams → fights streaming; see §11).
We own the scheduler AND the forward loop; borrow only GEMM + attention. Makes weight-swap free
(like AirLLM's eager model) without Python, and gives native control of streams/events/memory.

- **We write:** scheduler, ring buffer, streams/events, KV cache, residual stream; Llama forward
  kernels (RMSNorm, RoPE, residual, SiLU, mul, causal mask, sampling); multi-format dequant; the
  offline packer/allocator.
- **We borrow:** GEMM → cuBLASLt (all projections). Attention → naive (2 GEMMs + softmax) first,
  FlashAttention later. GGUF → **file format only** (weights + vocab), NOT ggml runtime.
- **Crib from:** `llm.c`, `llama2.c` (forward kernels), `llama.cpp` MMQ (dequant + fused GEMM later).

## 3. System overview
```
OFFLINE (packer/, Python ok)
  fp16 model → sensitivity profile → allocator → mixed-quant file (VRAM-native layout) + manifest
RUNTIME (CUDA)
  startup: mmap → load whole quant model into pinned RAM → alloc VRAM regions once →
           load K resident layers → plan fp16 arena layout → load tokenizer
  per token: embed → for L in 0..L-1: layer_forward(L) → final norm → LM head → sample
  layer_forward(L): [copy quant → dequant → fp16 GEMM forward] overlapped across L via 3 stages
```

## 4. VRAM budget (allocate ONCE at startup, never free in loop)
```
free_VRAM = [resident cache: K quant layers]      ← auto-sized, holds the bulk
          + [ring: R=3–4 quant slots]             ← streamed (L−K) overflow
          + [1 fp16 dequant arena: 1 layer]       ← the ONLY dequantized copy
          + [KV cache] + [activations] + [cuBLASLt workspace]

K = (free_VRAM − ring − arena − KV − workspace) / quant_bytes_per_layer
```
Rules:
- **Resident cache holds QUANTIZED layers** (small), not fp16. Auto-size K at startup.
- **Ring R = 3–4.** Depth only absorbs latency variance; it does NOT add bandwidth. Extra VRAM → K.
- **ONE fp16 arena**, reused every layer (dequant(L)→GEMMs(L)→dequant(L+1) serialize on the compute
  stream). Peak fp16 = 1 layer. (2 arenas only if profiling shows dequant-overlap wins.)
- 7B fits 16 GB whole → **impose an artificial VRAM cap** to force streaming for the demo. Real
  payoff is 70B.

## 5. Core data structures
```c
struct LayerMeta {            // from manifest, per layer
  QuantType type;             // Q4_0 | Q5_0 | Q8_0 | Q6_K
  size_t    quant_bytes;      // packed size of this layer (variable under mixed precision)
  MatDesc   mats[6];          // QKV,O,gate,up,down: {offset_in_arena, rows, cols, ld_padded}
  bool      resident;         // in the permanent cache?
  void*     resident_ptr;     // VRAM ptr if resident, else null
};

struct Slot {                 // one ring buffer entry (transient streamed layer)
  void*       d_quant;        // VRAM, sized to max layer quant_bytes
  cudaEvent_t copy_done;      // recorded on copy_stream after H2D
  cudaEvent_t compute_done;   // recorded on compute_stream after forward; gates slot reuse
  int         holding_layer;  // which layer currently occupies it (-1 = free)
};

struct Ctx {
  void*        h_pinned;      // whole quant model in pinned host RAM
  Slot         ring[R];
  void*        d_fp16_arena;  // single shared dequant scratch (1 layer, padded offsets)
  void*        d_hidden;      // residual stream, resident whole pass
  KVCache      kv;
  cudaStream_t copy_stream, compute_stream;
  cublasLtHandle_t lt;
  LayerMeta    layers[N_LAYERS];
};
```

## 6. The hot loop (scheduler.cu)
```c
// prefetch(L): issue H2D for layer L into a free ring slot (skip if resident)
void prefetch(int L) {
  if (layers[L].resident) return;
  Slot* s = acquire_slot(L);                              // pick ring[L % R]
  cudaStreamWaitEvent(copy_stream, s->compute_done, 0);   // slot free? (device wait, non-blocking)
  cudaMemcpyAsync(s->d_quant, h_pinned + off(L), layers[L].quant_bytes,
                  cudaMemcpyHostToDevice, copy_stream);   // quantized bytes only, coalesced
  cudaEventRecord(s->copy_done, copy_stream);
}

void* weights_of(int L) { return layers[L].resident ? layers[L].resident_ptr : ring[L%R].d_quant; }

for (int L = 0; L < N_LAYERS; ++L) {
  prefetch(L + PREFETCH_AHEAD);                           // keep copy engine busy (a few ahead)
  if (!layers[L].resident)
    cudaStreamWaitEvent(compute_stream, ring[L%R].copy_done, 0);
  dequant_layer(weights_of(L), d_fp16_arena, layers[L].type, compute_stream); // 1 launch/layer
  layer_forward(d_fp16_arena, d_hidden, &kv, L, compute_stream);              // cuBLASLt + kernels
  if (!layers[L].resident)
    cudaEventRecord(ring[L%R].compute_done, compute_stream);// slot reusable + hidden updated
}
```
**Sync invariant:** device-side `cudaStreamWaitEvent` for all ordering; **never**
`cudaEventSynchronize` in the loop (host block → serializes issue). Sync only at token end.

## 7. layer_forward (forward.cu) — one Llama layer, reading the fp16 arena
```
RMSNorm(h)
QKV = cuBLASLt(h, Wqkv);  RoPE(Q,K);  kv.append(K,V)
attn = softmax(Q·Kᵀ·scale + causal_mask) · V        // naive v1 → FlashAttention later
h += cuBLASLt(attn, Wo)                              // residual 1
RMSNorm(h)
g = SiLU(cuBLASLt(h, Wgate));  u = cuBLASLt(h, Wup);  m = g ⊙ u
h += cuBLASLt(m, Wdown)                              // residual 2
```
Batched decode / prefill: `h` carries B rows → same weights serve B tokens (amortizes load ÷B,
shifts toward compute-bound). Expose B as a knob.

## 8. Components / files (all new)
| File | Responsibility |
|---|---|
| `packer/` (Python) | sensitivity profile + allocator → mixed-quant file (VRAM-native layout) + manifest |
| `src/loader.{h,cpp}` | mmap file, load quant model into pinned RAM, parse manifest |
| `src/slot_pool.{h,cu}` | alloc ring slots + arena once; acquire/release with events |
| `src/scheduler.{h,cu}` | hot loop, prefetch, 3-stage sync (§6) |
| `src/kernels.cu` | dequant per type, RMSNorm, RoPE, softmax/attn, SiLU, mul, residual, sampling |
| `src/gemm.{h,cu}` | cuBLASLt wrappers (5 projections + LM head), transpose baked |
| `src/forward.{h,cu}` | assemble one layer (§7) + final norm + head |
| `src/main.cpp` | CLI: model, R, VRAM cap, batch B, serial\|pipelined, uniform\|mixed |

## 9. Build phases (each gate = a test that must pass before next)
1. **Single-layer forward.** Load 1 layer (Q8_0) → dequant → forward → **match llama.cpp layer-0
   hidden state**. Gate: max abs err < 1e-2 / KL tiny. De-risks all kernel math.
2. **Serial full model, uniform Q8_0.** Layer loop, no overlap. Gate: **final logits == llama.cpp**
   (greedy, temp=0), token-for-token on a fixed prompt.
3. **Ring + 2 streams.** Overlap; same output as phase 2. Gate: **Nsight shows copy(L+1)‖compute(L)**;
   pipelined wall < serial.
4. **Resident cache + auto-size K.** Gate: fewer streamed layers, correct output, tokens/s ↑.
5. **Multi-format dequant + packer (uniform).** Add Q4_0/Q5_0 kernels; packer emits uniform-bit
   files. Gate: each type's perplexity within expected band vs llama.cpp.
6. **Allocator + mixed precision.** Sensitivity profiling + solver → mixed-quant model. Gate:
   **perplexity ↔ tokens/s frontier beats uniform** at matched cap.
7. **Report.** Plot frontier (mixed vs uniform vs fp16-streaming); PCIe bytes; overlap %.
Optional later: Q6_K; fused MMQ GEMM (§11); FlashAttention; matrix-granularity streaming (§12);
GPUDirect Storage for model > RAM; batched decode sweep.

## 10. Invariants & gotchas (consolidated checklist)
- [ ] FREE = record event, **never `cudaFree`** in loop (device free stalls everything).
- [ ] All loop sync via **`cudaStreamWaitEvent`** (device); `cudaEventSynchronize` only at token end.
- [ ] **Pinned host memory** (`cudaHostAlloc`) — pageable can't DMA-overlap.
- [ ] **ONE fp16 arena**, not N. Resident cache holds **quantized**, not fp16.
- [ ] Ring depth 3–4; extra VRAM → resident cache K, not deeper prefetch (depth ≠ bandwidth).
- [ ] Coalesce a few layers per `cudaMemcpyAsync` (peak BW); don't do tiny per-matrix copies.
- [ ] Packer emits **VRAM-native layout** (dequant-ready, pre-transposed, aligned, consumption
      order) → runtime transfer is a raw DMA, no reformat. Only runtime reformat = dequant kernel.
- [ ] Dequant whole layer into the arena in **one launch**; pad each matrix ld to mult of 8.
- [ ] Compute is **sequential** (residual dep) — caching/prefetch avoids re-streaming, never
      parallelizes layer math.
- [ ] Weight-only quant: activations stay fp16 (if using fused MMQ later, avoid its int8-activation
      path — it changes perplexity).

## 11. GGUF dequant appendix (exact, verified)
Legacy _0 (32-elem block, single fp16 scale — ship in v1):
- **Q4_0** 18 B: fp16 `d` + 32×4-bit. `x = d·(q−8)`. `qs[i]&0x0F`, `qs[i]>>4`.
- **Q5_0** 22 B: fp16 `d` + 32-bit `qh` + 32×4-bit. `x = d·(q−16)`, 5th bit from `qh`.
- **Q8_0** 34 B: fp16 `d` + 32×int8. `x = d·q`. Phase-1 baseline.

Q6_K (K-quant, phase-4+ only): super-block **QK_K=256**, 210 B: `ql[128]` + `qh[64]` +
`scales[16]` (int8) + `d` **fp16 (2B)**. Signed 6-bit: `q6 = (ql_nib | (qh_2b<<4)) − 32`;
`x = d · scales[sub] · q6`. (6.5625 bpw.)

Kernel: one thread block per GGUF block; `half2` vectorized writes (no warp-shuffle). Unit-test
each type against llama.cpp dequant output.

Fused GEMM (phase-4b optimization): lift **llama.cpp MMQ** kernels (already dequant these exact
formats in-matmul) → drops the fp16 arena + dequant seam. Use the **weight-only variant**
(activations fp16). Do NOT write a GEMM from scratch. Skip streaming-K (load-bound → marginal).

## 12. Deferred optimizations (only if profiling justifies)
- **Matrix-granularity streaming:** stream at matrix unit (QKV,O,gate,up,down) in consumption
  order into a circular buffer → finer overlap for variable-size mixed-quant layers. Constraint:
  matrix is atomic (GEMM needs whole matrix); never split a matrix across the wrap (pad). Benchmark
  vs single-arena dequant (it costs 6 launches/layer). v1 stays layer-granularity.
- **Fused MMQ GEMM, FlashAttention, Q6_K, GPUDirect Storage, batched-decode sweep** (§9 optional).

## 13. Risks (all deterministic, testable)
- **Forward correctness** (RoPE/mask/KV layout) → golden per-layer harness vs llama.cpp
  (`ggml_backend_sched_set_eval_callback` to dump reference states). Phase 1 de-risks it.
- **Multi-format dequant** → per-type unit tests (§11).
- **Allocator validity** → t_load/t_compute must match measured hardware or "free accuracy" breaks;
  calibrate on-device, verify predicted vs actual overlap in Nsight before trusting the solver.
- **Attention** → naive first (correct, slow); FlashAttention is later speed, off the correctness path.

## 14. Projected speedup vs AirLLM (70B; 7B fits VRAM → N/A)
Quant transfer 2–4× (dominant, accuracy trade) · overlap ~1.2× · resident cache ~1.3× · no-Python
~1.15× → **~3–6×/token (batch=1)**. Batching B=8–16 → **~10–50× throughput** (not latency). The
research claim is the **perplexity↔tokens/s frontier**, not the multiplier.

## 15. Verification
- **Correctness:** phase-1 per-layer match; phase-2 logits == llama.cpp (temp=0); per-type
  perplexity in band.
- **Overlap:** Nsight copy(L+1)‖compute(L); wall ≈ max(load,compute)·L.
- **Headline:** perplexity↔tokens/s frontier — mixed strictly dominates uniform at matched cap.
- **Transfer:** PCIe bytes = ¼–½ of AirLLM's fp16 leg (instrumented).

## 16. MoE extension (v2) — streaming trillion-param sparse models
North star: run 200B–1T+ models. Those are Mixture-of-Experts, and streaming ❤ MoE.

### Why streaming wins hardest on MoE
Dense: stream all L layers every token. MoE: total params huge (1T) but only
`expert_used_count` of `expert_count` experts fire per token per layer (Laguna: 8/256 ≈ 3%).
**Stream only the activated experts** → per-token weight traffic is a tiny fraction of the model.
A 1T MoE touches ~tens of B of weights/token; stream those, cache the hot ones → dense-quality
on 121 GB. This is the target use case.

### The core new challenge: dynamic expert selection
Dense prefetch works because layer order is static (you know L+1 ahead). MoE expert choice is
**data-dependent** — the router picks experts from the *current* activations, so you can't know
which expert weights to load until the router runs. This breaks static prefetch for experts.
Split each MoE layer:
- **Dense/shared part** (attn, norms, router gate, shared expert) — data-independent → prefetch
  exactly like v1.
- **Routed experts** — dynamic → load on demand after routing.

### Architecture additions
1. **Router kernel** — `logits = x @ W_gate` [tokens, n_experts]; gating (softmax/sigmoid per
   `expert_gating_func`, with `expert_weights_norm` / `expert_weights_scale`); top-k select →
   (expert_id, weight) per token. Cheap (n_experts small).
2. **Expert FFN (grouped)** — for the k selected experts: gather routed tokens → per-expert SwiGLU
   (down·(silu(gate)·up)) → scatter-add scaled by the gate weight. Grouped/batched GEMM
   (cuBLAS batched or CUTLASS grouped) or per-expert GEMM.
3. **Shared expert** — always-on FFN (`expert_shared_feed_forward_length`), added to the routed sum.
4. **Expert weight streaming + cache** (the new memory engine):
   - **Hot-expert VRAM cache** (LRU / frequency): expert usage is skewed — keep the hottest resident,
     stream only misses. Biggest lever at trillion scale.
   - **On-demand load** — after routing, DMA the missing selected experts host→VRAM.
   - **Intra-layer expert pipeline** — load expert e+1 while computing expert e (pipeline the k).
5. **Config extension** — expert_count, expert_used_count, expert_feed_forward_length,
   expert_shared_feed_forward_length, gating func, weights_scale/norm, leading_dense_block_count
   (first N layers are dense → use the v1 path unchanged).

### Overlap model for MoE (honest)
- Attn / router / shared: overlap-prefetch like dense (data-independent).
- Routed experts: intra-layer overlap is limited (router→experts dependency). Recover it via
  (a) hot-expert cache (hits ⇒ no load), (b) intra-layer expert pipelining, (c) **batching** — a
  batch activates the UNION of experts across its tokens, amortizing each expert load over many
  tokens. Small batch ⇒ few experts, tiny load; large batch ⇒ most experts used but each serves
  many tokens. Batch size is the MoE load/compute knob.

### Trillion-param math (decode, per token)
1T MoE, 256 experts/layer, 8 active. Per token/layer: 8 experts + shared + attn loaded ≈ a few %
of total weights. With Q4 + hot-expert cache, streamable on 121 GB unified (GB10). That is the
whole point of SemiLLM.

### Also needed for Laguna-class (separate from the MoE core)
Sliding-window + gated attention, QK-norm, YaRN rope scaling, per-layer variable head counts,
**K-quant dequant (Q4_K / Q6_K)**. Additive to the attention + dequant modules; tracked apart from
the routing core so the MoE work isn't blocked on them.

### Interface impact (minimal churn to v1)
- `moe_layer_forward` variant: attention identical to v1; MLP replaced by router + experts + shared.
- `LlamaConfig` gains `is_moe` + expert fields; `leading_dense_block_count` layers keep the v1 path.
- `stream_forward` gains an expert-streaming path + expert cache alongside the dense slot ring.
- The v1 dense path stays intact and remains the fallback / foundation.

### Staging
- **v1 (now):** dense, Q8_0, streaming — validated on GB10 (9/9). Test small → large dense.
- **v2:** MoE routing + expert streaming/cache (+ K-quant + Laguna attn features). Test a small MoE
  GGUF (e.g. a Qwen-MoE) → Laguna-XS → Laguna-S.
