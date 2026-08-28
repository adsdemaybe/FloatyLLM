// Model config + tensor binding from GGUF metadata (llama.cpp conventions).
#include "model.h"
#include <cstdio>

namespace {
std::string key(const std::string& arch, const char* suffix) {
    return arch + "." + suffix;
}
}  // namespace

bool read_config(const GgufFile& g, ModelConfig* out, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    if (!gguf_get_str(g, "general.architecture", &out->arch))
        return fail("missing general.architecture");
    const std::string& a = out->arch;

    uint32_t u;
    if (!gguf_get_u32(g, key(a, "block_count"), &u)) return fail("missing block_count");
    out->n_layers = (int)u;
    if (!gguf_get_u32(g, key(a, "embedding_length"), &u)) return fail("missing embedding_length");
    out->dim = (int)u;
    if (!gguf_get_u32(g, key(a, "attention.head_count"), &u)) return fail("missing head_count");
    out->n_heads = (int)u;
    // head_count_kv defaults to head_count (MHA) when absent.
    if (gguf_get_u32(g, key(a, "attention.head_count_kv"), &u)) out->n_kv_heads = (int)u;
    else out->n_kv_heads = out->n_heads;
    if (!gguf_get_u32(g, key(a, "feed_forward_length"), &u)) return fail("missing feed_forward_length");
    out->ffn_dim = (int)u;

    if (out->n_heads <= 0 || out->dim % out->n_heads != 0) return fail("bad dim/head_count");
    out->head_dim = out->dim / out->n_heads;

    float f;
    if (gguf_get_f32(g, key(a, "attention.layer_norm_rms_epsilon"), &f)) out->eps = f;
    if (gguf_get_f32(g, key(a, "rope.freq_base"), &f)) out->rope_base = f;
    return true;
}

bool get_layer_tensors(const GgufFile& g, int layer, LayerTensors* out, std::string* err) {
    char buf[128];
    auto get = [&](const char* suffix, const TensorInfo** dst) -> bool {
        snprintf(buf, sizeof(buf), "blk.%d.%s", layer, suffix);
        const TensorInfo* t = gguf_find_tensor(g, buf);
        if (!t) { if (err) *err = std::string("missing tensor ") + buf; return false; }
        *dst = t;
        return true;
    };
    return get("attn_norm.weight", &out->attn_norm) &&
           get("attn_q.weight",    &out->wq) &&
           get("attn_k.weight",    &out->wk) &&
           get("attn_v.weight",    &out->wv) &&
           get("attn_output.weight", &out->wo) &&
           get("ffn_norm.weight",  &out->ffn_norm) &&
           get("ffn_gate.weight",  &out->wgate) &&
           get("ffn_up.weight",    &out->wup) &&
           get("ffn_down.weight",  &out->wdown);
}
