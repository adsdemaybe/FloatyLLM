#!/usr/bin/env python3
# FloatyLLM mixed-precision packer.
#
# Builds a mixed-quant GGUF by cherry-picking each layer's weights from several
# same-base quantized source models (e.g. Q2_K / Q3_K_M / Q4_K_M of Llama-2-70B).
# No requantization: every tensor is copied verbatim (bytes + ggml type) from the
# source chosen for its layer, so per-layer bit-width becomes a knob with zero extra
# quant error beyond the source files. FloatyLLM already reads per-tensor mixed types,
# so the output runs directly; llama-perplexity reads it too (standard gguf), giving a
# perplexity <-> tokens/s frontier.
#
# Usage:
#   mix_pack.py --out mixed.gguf --schedule SCHED q2k.gguf q3km.gguf q4km.gguf
# SCHED assigns each transformer layer a source *tier* (0 = first source = lowest bits):
#   --schedule uniform:1                 every layer -> source index 1
#   --schedule ends:2,mid:0              first/last 25% -> source 2, middle -> source 0
#   --schedule csv:0,0,2,1,...           explicit per-layer source index (len = n_layers)
# Non-layer tensors (token_embd, output, output_norm) come from the highest-bit source.

import sys, argparse
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def layer_of(name):
    # blk.<L>.* -> L ; non-block tensors -> -1
    if name.startswith("blk."):
        try:
            return int(name.split(".")[1])
        except (IndexError, ValueError):
            return -1
    return -1


def build_schedule(spec, n_layers, n_src):
    hi = n_src - 1
    if spec.startswith("uniform:"):
        s = int(spec.split(":")[1])
        return [s] * n_layers
    if spec.startswith("csv:"):
        vals = [int(x) for x in spec[4:].split(",") if x != ""]
        if len(vals) != n_layers:
            sys.exit(f"csv schedule has {len(vals)} entries, need {n_layers}")
        return vals
    if spec.startswith("ends:"):
        # ends:HI,mid:LO  -> outer 25% each end at tier HI, middle at tier LO
        parts = dict(p.split(":") for p in spec.split(","))
        hi_t = int(parts["ends"]); lo_t = int(parts.get("mid", "0"))
        edge = max(1, n_layers // 4)
        return [hi_t if (L < edge or L >= n_layers - edge) else lo_t for L in range(n_layers)]
    sys.exit(f"unknown schedule '{spec}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="source ggufs, lowest-bit first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schedule", required=True)
    args = ap.parse_args()

    readers = [GGUFReader(p) for p in args.sources]
    hi = len(readers) - 1

    # Tensor index by name per source; and the layer count from the highest-bit source.
    idx = [{t.name: t for t in r.tensors} for r in readers]
    names = list(idx[hi].keys())
    n_layers = 1 + max((layer_of(n) for n in names), default=-1)
    sched = build_schedule(args.schedule, n_layers, len(readers))

    # Architecture for the writer: read general.architecture from the hi source.
    arch = readers[hi].fields["general.architecture"].contents()
    w = GGUFWriter(args.out, arch)

    # Copy all metadata KV from the hi source (config, tokenizer, etc.). general.architecture
    # is set by the writer ctor; skip it and the structural GGUF.* pseudo-fields.
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for f in readers[hi].fields.values():
        if f.name in skip or not f.types:
            continue
        vtype = f.types[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(f.name, f.contents(), vtype, sub_type=f.types[1])
        else:
            w.add_key_value(f.name, f.contents(), vtype)

    counts = {}
    for name in names:
        L = layer_of(name)
        src = sched[L] if L >= 0 else hi          # non-layer tensors -> highest bits
        t = idx[src].get(name) or idx[hi][name]   # fall back if missing in chosen src
        w.add_tensor(name, t.data, raw_dtype=t.tensor_type)
        counts[t.tensor_type.name] = counts.get(t.tensor_type.name, 0) + 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: {len(names)} tensors, {n_layers} layers")
    print("type mix:", counts)
    print("schedule:", sched)


if __name__ == "__main__":
    main()
