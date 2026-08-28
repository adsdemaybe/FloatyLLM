// Model config + layer-tensor binding self-test (CPU, no GPU). Synthesizes a GGUF
// with llama config keys + one layer's 9 tensors, then checks read_config and
// get_layer_tensors. Include impls so it compiles standalone. Single-line comments.
#include "../src/loader.cpp"
#include "../src/model.cpp"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

struct Writer {
    std::vector<uint8_t> b;
    template <typename T> void put(T v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void put_str(const std::string& s) { put<uint64_t>(s.size()); b.insert(b.end(), s.begin(), s.end()); }
    void kv_str(const std::string& k, const std::string& v) { put_str(k); put<uint32_t>(GGUF_STRING); put_str(v); }
    void kv_u32(const std::string& k, uint32_t v) { put_str(k); put<uint32_t>(GGUF_U32); put<uint32_t>(v); }
    void kv_f32(const std::string& k, float v) { put_str(k); put<uint32_t>(GGUF_F32); put<float>(v); }
    void tensor(const std::string& name) {
        put_str(name); put<uint32_t>(1); put<uint64_t>(1); put<uint32_t>(GGML_F32); put<uint64_t>(0);
    }
};

static int g_fail = 0;
static void check(bool c, const char* what) {
    printf(c ? "[PASS] %s\n" : "[FAIL] %s\n", what);
    if (!c) ++g_fail;
}

int main() {
    printf("[INFO] Model config + tensor binding test\n");

    const char* suffixes[9] = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };

    Writer w;
    w.put<uint32_t>(0x46554747u);
    w.put<uint32_t>(3);
    w.put<uint64_t>(9);   // tensor count
    w.put<uint64_t>(7);   // metadata count
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 1);
    w.kv_u32("llama.embedding_length", 8);
    w.kv_u32("llama.attention.head_count", 2);
    w.kv_u32("llama.attention.head_count_kv", 1);
    w.kv_u32("llama.feed_forward_length", 16);
    w.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    for (int i = 0; i < 9; ++i) w.tensor(std::string("blk.0.") + suffixes[i]);
    while (w.b.size() % 32 != 0) w.b.push_back(0);
    for (int i = 0; i < 64; ++i) w.b.push_back(0);

    const char* path = "/tmp/semillm_model.gguf";
    FILE* f = fopen(path, "wb"); fwrite(w.b.data(), 1, w.b.size(), f); fclose(f);

    GgufFile g; std::string err;
    if (!gguf_load(path, &g, &err)) { printf("load fail: %s\n", err.c_str()); return 1; }

    ModelConfig cfg;
    check(read_config(g, &cfg, &err), "read_config succeeds");
    check(cfg.arch == "llama", "arch == llama");
    check(cfg.n_layers == 1, "n_layers == 1");
    check(cfg.dim == 8, "dim == 8");
    check(cfg.n_heads == 2, "n_heads == 2");
    check(cfg.n_kv_heads == 1, "n_kv_heads == 1 (GQA)");
    check(cfg.head_dim == 4, "head_dim == dim/n_heads == 4");
    check(cfg.ffn_dim == 16, "ffn_dim == 16");
    check(cfg.eps > 0.0f, "eps read");
    printf("[INFO]   arch=%s dim=%d heads=%d/%d hd=%d ffn=%d eps=%.6f base=%.1f\n",
           cfg.arch.c_str(), cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ffn_dim, cfg.eps, cfg.rope_base);
    check(cfg.rope_base == 10000.0f, "rope_base defaults to 10000 when absent");

    LayerTensors lt;
    check(get_layer_tensors(g, 0, &lt, &err), "get_layer_tensors(0) succeeds");
    check(lt.attn_norm && lt.wq && lt.wk && lt.wv && lt.wo &&
          lt.ffn_norm && lt.wgate && lt.wup && lt.wdown, "all 9 layer tensors bound");
    check(lt.wq->name == "blk.0.attn_q.weight", "wq resolves to blk.0.attn_q.weight");

    LayerTensors lt1;
    check(!get_layer_tensors(g, 1, &lt1, &err), "missing layer 1 returns false");

    if (g_fail == 0) { printf("[INFO] ALL CHECKS PASSED\n"); return 0; }
    printf("[INFO] %d CHECK(S) FAILED\n", g_fail);
    return 1;
}
