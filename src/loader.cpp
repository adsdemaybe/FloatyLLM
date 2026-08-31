// GGUF v3 parser implementation. Reads the whole file into memory, then walks
// header -> metadata KVs -> tensor infos -> aligned data section.
#include "loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <atomic>

namespace {

// Fault the whole mmap'd file into the page cache in PARALLEL. On first (cold) run the model
// is otherwise paged in single-threaded, on demand, during the first prefill/decode (slow disk
// crawl). Touching one byte per page across N threads uses the full disk bandwidth, so the
// cost moves to load time and shrinks ~Nx. On a warm run (pages already cached) this is a
// cheap read sweep. Disable with SEMILLM_NO_WARM=1.
void warm_pages(const uint8_t* p, size_t sz) {
    if (const char* e = getenv("SEMILLM_NO_WARM")) { if (atoi(e)) return; }
    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = (int)(hw ? (hw < 16 ? hw : 16) : 8);
    size_t page = 4096, chunk = (sz + nthreads - 1) / nthreads;
    std::atomic<uint64_t> sink{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; ++t) {
        size_t s = (size_t)t * chunk, e = s + chunk > sz ? sz : s + chunk;
        if (s >= sz) break;
        ts.emplace_back([p, s, e, page, &sink] {
            uint64_t acc = 0;
            for (size_t i = s; i < e; i += page) acc += p[i];   // touch each page -> fault in
            sink += acc;
        });
    }
    for (auto& th : ts) th.join();
    (void)sink;
}

struct Cursor {
    const uint8_t* p;
    size_t pos;
    size_t size;
    bool ok = true;

    bool need(size_t n) {
        if (pos + n > size) { ok = false; return false; }
        return true;
    }
    template <typename T> T read_scalar() {
        T v{};
        if (!need(sizeof(T))) return v;
        memcpy(&v, p + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::string read_string() {
        uint64_t len = read_scalar<uint64_t>();
        std::string s;
        if (!ok || !need(len)) { ok = false; return s; }
        s.assign(reinterpret_cast<const char*>(p + pos), len);
        pos += len;
        return s;
    }
};

// Byte width of a fixed-size gguf scalar type (0 for variable/complex).
size_t scalar_width(int type) {
    switch (type) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}

// Read a metadata value of the given type into mv (scalars/strings stored; string
// arrays retained in mv.strs, numeric arrays consumed but not retained). Advances the cursor.
void read_value(Cursor& c, int type, MetaValue& mv) {
    mv.type = type;
    switch (type) {
        case GGUF_U8:  mv.inum = c.read_scalar<uint8_t>();  break;
        case GGUF_I8:  mv.inum = c.read_scalar<int8_t>();   break;
        case GGUF_U16: mv.inum = c.read_scalar<uint16_t>(); break;
        case GGUF_I16: mv.inum = c.read_scalar<int16_t>();  break;
        case GGUF_U32: mv.inum = c.read_scalar<uint32_t>(); break;
        case GGUF_I32: mv.inum = c.read_scalar<int32_t>();  break;
        case GGUF_U64: mv.inum = (int64_t)c.read_scalar<uint64_t>(); break;
        case GGUF_I64: mv.inum = c.read_scalar<int64_t>();  break;
        case GGUF_BOOL: mv.inum = c.read_scalar<uint8_t>() ? 1 : 0; break;
        case GGUF_F32: mv.fnum = c.read_scalar<float>();    break;
        case GGUF_F64: mv.fnum = c.read_scalar<double>();   break;
        case GGUF_STRING: mv.str = c.read_string();         break;
        case GGUF_ARRAY: {
            uint32_t elem_type = c.read_scalar<uint32_t>();
            uint64_t count = c.read_scalar<uint64_t>();
            if (elem_type == GGUF_STRING) mv.strs.reserve(count);
            for (uint64_t i = 0; i < count && c.ok; ++i) {
                if (elem_type == GGUF_STRING) { mv.strs.push_back(c.read_string()); }  // retained (vocab, merges)
                else {
                    size_t w = scalar_width(elem_type);
                    if (w == 0 || !c.need(w)) { c.ok = false; return; }
                    c.pos += w;
                }
            }
            break;
        }
        default: c.ok = false; break;
    }
}

}  // namespace

// mmap + parse one shard; append its tensors (with shard index) to out.
static bool parse_shard(const char* path, GgufFile* out, int shard_idx, bool first, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail("cannot open shard");
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return fail("empty shard"); }
    size_t sz = (size_t)st.st_size;
    void* p = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { close(fd); return fail("mmap failed"); }
    // WILLNEED (not SEQUENTIAL): the whole model is re-read every token, so keep pages
    // resident and let the kernel prefetch; then warm the page cache in parallel.
    madvise(p, sz, MADV_WILLNEED);
    warm_pages((const uint8_t*)p, sz);
    Shard sh; sh.data = (const uint8_t*)p; sh.size = sz; sh.fd = fd;

    Cursor c{sh.data, 0, sh.size};
    auto bail = [&](const char* m) { munmap(p, sz); close(fd); return fail(m); };
    if (c.read_scalar<uint32_t>() != 0x46554747u) return bail("bad magic (not GGUF)");
    uint32_t ver = c.read_scalar<uint32_t>();
    if (first) out->version = ver;
    if (ver != 2 && ver != 3) return bail("unsupported GGUF version");
    uint64_t n_tensors = c.read_scalar<uint64_t>();
    uint64_t n_meta = c.read_scalar<uint64_t>();
    if (!c.ok) return bail("truncated header");

    for (uint64_t i = 0; i < n_meta; ++i) {
        std::string key = c.read_string();
        uint32_t vtype = c.read_scalar<uint32_t>();
        MetaValue mv;
        read_value(c, (int)vtype, mv);
        if (!c.ok) return bail("truncated metadata");
        if (first) out->meta.emplace(std::move(key), std::move(mv));
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = c.read_string();
        uint32_t nd = c.read_scalar<uint32_t>();
        for (uint32_t d = 0; d < nd && c.ok; ++d) t.dims.push_back(c.read_scalar<uint64_t>());
        t.ggml_type = c.read_scalar<uint32_t>();
        t.offset = c.read_scalar<uint64_t>();
        t.shard = shard_idx;
        if (!c.ok) return bail("truncated tensor info");
        out->tensor_index[t.name] = (int)out->tensors.size();
        out->tensors.push_back(std::move(t));
    }
    uint32_t align = 32;
    { uint32_t a; if (first && gguf_get_u32(*out, "general.alignment", &a) && a) align = a; }
    size_t off = (c.pos + align - 1) / align * align;
    if (off > sh.size) return bail("data section past EOF");
    sh.data_offset = off;
    out->shards.push_back(sh);
    return true;
}

bool gguf_load(const char* path, GgufFile* out, std::string* err) {
    if (!parse_shard(path, out, 0, true, err)) return false;
    // Split model? Load the remaining shards (foo-00001-of-000NN.gguf).
    uint32_t cnt = 0;
    if (gguf_get_u32(*out, "split.count", &cnt) && cnt > 1) {
        std::string p(path);
        size_t of = p.rfind("-of-");
        if (of != std::string::npos && of >= 5) {
            for (uint32_t k = 2; k <= cnt; ++k) {
                std::string sp = p;
                char num[8]; snprintf(num, sizeof(num), "%05u", k);
                sp.replace(of - 5, 5, num);
                if (!parse_shard(sp.c_str(), out, (int)k - 1, false, err)) return false;
            }
        }
    }
    return true;
}

void gguf_close(GgufFile* g) {
    for (auto& sh : g->shards) {
        if (sh.data && sh.size) munmap((void*)sh.data, sh.size);
        if (sh.fd >= 0) close(sh.fd);
    }
    g->shards.clear();
}

bool gguf_get_u32(const GgufFile& g, const std::string& key, uint32_t* v) {
    auto it = g.meta.find(key);
    if (it == g.meta.end()) return false;
    int t = it->second.type;
    if (t == GGUF_F32 || t == GGUF_F64 || t == GGUF_STRING || t == GGUF_ARRAY) return false;
    *v = (uint32_t)it->second.inum;
    return true;
}

bool gguf_get_f32(const GgufFile& g, const std::string& key, float* v) {
    auto it = g.meta.find(key);
    if (it == g.meta.end()) return false;
    if (it->second.type == GGUF_F32 || it->second.type == GGUF_F64) { *v = (float)it->second.fnum; return true; }
    return false;
}

bool gguf_get_str(const GgufFile& g, const std::string& key, std::string* v) {
    auto it = g.meta.find(key);
    if (it == g.meta.end() || it->second.type != GGUF_STRING) return false;
    *v = it->second.str;
    return true;
}

const TensorInfo* gguf_find_tensor(const GgufFile& g, const std::string& name) {
    auto it = g.tensor_index.find(name);
    if (it == g.tensor_index.end()) return nullptr;
    return &g.tensors[it->second];
}

const uint8_t* gguf_tensor_data(const GgufFile& g, const TensorInfo& t) {
    const Shard& sh = g.shards[t.shard];
    return sh.data + sh.data_offset + t.offset;
}

uint64_t gguf_tensor_elements(const TensorInfo& t) {
    uint64_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    return n;
}

uint64_t gguf_tensor_bytes(const TensorInfo& t) {
    uint64_t n = gguf_tensor_elements(t);
    switch (t.ggml_type) {
        case GGML_F32:  return n * 4;
        case GGML_F16:  return n * 2;
        case GGML_Q8_0: return (n / 32) * 34;   // 32 int8 + fp16 scale per block
        case GGML_Q4_0: return (n / 32) * 18;
        case GGML_Q5_0: return (n / 32) * 22;
        case GGML_Q4_K: return (n / 256) * 144;
        case GGML_Q5_K: return (n / 256) * 176;
        case GGML_Q6_K: return (n / 256) * 210;
        default: return 0;
    }
}
