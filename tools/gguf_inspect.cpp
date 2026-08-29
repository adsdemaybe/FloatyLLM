// gguf_inspect <file.gguf>: dump version, metadata, config, and tensor directory.
// A real-file utility (uses the loader). Include impls to build standalone.
#include "../src/loader.cpp"
#include "../src/model.cpp"
#include <cstdio>
#include <algorithm>

static const char* ggml_type_name(uint32_t t) {
    switch (t) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0";
        case 3: return "Q4_1"; case 6: return "Q5_0"; case 7: return "Q5_1";
        case 8: return "Q8_0"; case 9: return "Q8_1"; case 10: return "Q2_K";
        case 11: return "Q3_K"; case 12: return "Q4_K"; case 13: return "Q5_K";
        case 14: return "Q6_K"; case 15: return "Q8_K"; default: return "?";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: gguf_inspect <file.gguf>\n"); return 2; }

    GgufFile g;
    std::string err;
    if (!gguf_load(argv[1], &g, &err)) { printf("load failed: %s\n", err.c_str()); return 1; }

    printf("version=%u  tensors=%zu  meta_keys=%zu  data_offset=%zu\n",
           g.version, g.tensors.size(), g.meta.size(), g.data_offset);

    printf("\n== metadata ==\n");
    for (const auto& kv : g.meta) {
        const MetaValue& v = kv.second;
        if (v.type == GGUF_STRING) {
            std::string s = v.str.size() > 60 ? v.str.substr(0, 60) + "..." : v.str;
            printf("  %-45s str  %s\n", kv.first.c_str(), s.c_str());
        } else if (v.type == GGUF_F32 || v.type == GGUF_F64) {
            printf("  %-45s f    %g\n", kv.first.c_str(), v.fnum);
        } else if (v.type == GGUF_ARRAY) {
            printf("  %-45s [array]\n", kv.first.c_str());
        } else {
            printf("  %-45s int  %lld\n", kv.first.c_str(), (long long)v.inum);
        }
    }

    printf("\n== config (read_config) ==\n");
    ModelConfig cfg;
    if (read_config(g, &cfg, &err)) {
        printf("  arch=%s layers=%d dim=%d heads=%d/%d head_dim=%d ffn=%d eps=%g rope_base=%g\n",
               cfg.arch.c_str(), cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads,
               cfg.head_dim, cfg.ffn_dim, cfg.eps, cfg.rope_base);
    } else {
        printf("  read_config failed: %s\n", err.c_str());
    }

    printf("\n== first tensors ==\n");
    int show = (int)std::min<size_t>(g.tensors.size(), 24);
    for (int i = 0; i < show; ++i) {
        const TensorInfo& t = g.tensors[i];
        printf("  %-34s %-5s [", t.name.c_str(), ggml_type_name(t.ggml_type));
        for (size_t d = 0; d < t.dims.size(); ++d) printf("%s%llu", d ? "," : "", (unsigned long long)t.dims[d]);
        printf("] off=%llu\n", (unsigned long long)t.offset);
    }
    return 0;
}
