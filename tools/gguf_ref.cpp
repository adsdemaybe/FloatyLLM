// GGUF loader round-trip self-test (CPU, no GPU). Synthesizes a minimal GGUF in
// memory, writes it to a temp file, parses it back, and checks metadata + tensor
// directory + data. Single-line comments only (linter rule).
// Include the implementation so this test compiles standalone (CI compiles each
// tools/*_ref.cpp alone). The real build compiles loader.cpp into semillm_core.
#include "../src/loader.cpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// --- tiny GGUF writer ---
struct Writer {
    std::vector<uint8_t> b;
    template <typename T> void put(T v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void put_str(const std::string& s) {
        put<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void kv_str(const std::string& k, const std::string& v) {
        put_str(k); put<uint32_t>(GGUF_STRING); put_str(v);
    }
    void kv_u32(const std::string& k, uint32_t v) {
        put_str(k); put<uint32_t>(GGUF_U32); put<uint32_t>(v);
    }
    void kv_f32(const std::string& k, float v) {
        put_str(k); put<uint32_t>(GGUF_F32); put<float>(v);
    }
};

static int g_fail = 0;
static void check(bool c, const char* what) {
    printf(c ? "[PASS] %s\n" : "[FAIL] %s\n", what);
    if (!c) ++g_fail;
}

int main() {
    printf("[INFO] GGUF loader round-trip test\n");

    const uint32_t dim = 32, blocks = (dim * dim) / 32; // Q8_0 blocks for a 32x32 tensor
    const uint64_t tensor_bytes = (uint64_t)blocks * 34;

    Writer w;
    w.put<uint32_t>(0x46554747u);   // magic "GGUF"
    w.put<uint32_t>(3);             // version
    w.put<uint64_t>(1);             // tensor count
    w.put<uint64_t>(4);             // metadata count
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 2);
    w.kv_u32("llama.embedding_length", dim);
    w.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    // tensor info: name, n_dims, dims, type, offset
    w.put_str("blk.0.attn_q.weight");
    w.put<uint32_t>(2);
    w.put<uint64_t>(dim); w.put<uint64_t>(dim);
    w.put<uint32_t>(GGML_Q8_0);
    w.put<uint64_t>(0);
    // pad to alignment 32, then tensor data (known bytes)
    while (w.b.size() % 32 != 0) w.b.push_back(0);
    for (uint64_t i = 0; i < tensor_bytes; ++i) w.b.push_back((uint8_t)(i & 0xFF));

    const char* path = "/tmp/semillm_test.gguf";
    FILE* f = fopen(path, "wb");
    fwrite(w.b.data(), 1, w.b.size(), f);
    fclose(f);

    GgufFile g;
    std::string err;
    bool ok = gguf_load(path, &g, &err);
    check(ok, "gguf_load succeeds");
    if (!ok) { printf("  err: %s\n", err.c_str()); return 1; }

    check(g.version == 3, "version == 3");

    std::string arch;
    check(gguf_get_str(g, "general.architecture", &arch) && arch == "llama", "arch string == llama");
    uint32_t bc = 0;
    check(gguf_get_u32(g, "llama.block_count", &bc) && bc == 2, "block_count == 2");
    uint32_t ed = 0;
    check(gguf_get_u32(g, "llama.embedding_length", &ed) && ed == dim, "embedding_length == 32");
    float eps = 0;
    check(gguf_get_f32(g, "llama.attention.layer_norm_rms_epsilon", &eps) && eps > 0, "eps f32 read");
    printf("[INFO]   eps=%.6f\n", eps);

    uint32_t missing;
    check(!gguf_get_u32(g, "nope.absent", &missing), "missing key returns false");
    check(!gguf_get_u32(g, "general.architecture", &missing), "wrong-type getter returns false");

    const TensorInfo* t = gguf_find_tensor(g, "blk.0.attn_q.weight");
    check(t != nullptr, "tensor found by name");
    if (t) {
        check(t->ggml_type == GGML_Q8_0, "tensor type == Q8_0");
        check(t->dims.size() == 2 && t->dims[0] == dim && t->dims[1] == dim, "tensor dims 32x32");
        check(gguf_tensor_elements(*t) == dim * dim, "elements == 1024");
        check(gguf_tensor_bytes(*t) == tensor_bytes, "byte size matches Q8_0 blocks");
        const uint8_t* d = gguf_tensor_data(g, *t);
        bool data_ok = d[0] == 0 && d[1] == 1 && d[100] == (uint8_t)100;
        check(data_ok, "tensor data bytes read back correctly");
    }

    if (g_fail == 0) { printf("[INFO] ALL CHECKS PASSED\n"); return 0; }
    printf("[INFO] %d CHECK(S) FAILED\n", g_fail);
    return 1;
}
