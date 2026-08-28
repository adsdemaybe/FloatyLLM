// GGUF v3 parser implementation. Reads the whole file into memory, then walks
// header -> metadata KVs -> tensor infos -> aligned data section.
#include "loader.h"
#include <cstdio>
#include <cstring>

namespace {

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

// Read a metadata value of the given type into mv (scalars/strings stored;
// arrays consumed but not retained). Advances the cursor.
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
            for (uint64_t i = 0; i < count && c.ok; ++i) {
                if (elem_type == GGUF_STRING) { c.read_string(); }
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

bool gguf_load(const char* path, GgufFile* out, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };

    FILE* f = fopen(path, "rb");
    if (!f) return fail("cannot open file");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return fail("empty file"); }
    out->bytes.resize((size_t)sz);
    size_t got = fread(out->bytes.data(), 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) return fail("short read");

    Cursor c{out->bytes.data(), 0, out->bytes.size()};

    uint32_t magic = c.read_scalar<uint32_t>();
    if (magic != 0x46554747u) return fail("bad magic (not GGUF)");
    out->version = c.read_scalar<uint32_t>();
    if (out->version != 2 && out->version != 3) return fail("unsupported GGUF version");
    uint64_t n_tensors = c.read_scalar<uint64_t>();
    uint64_t n_meta = c.read_scalar<uint64_t>();
    if (!c.ok) return fail("truncated header");

    for (uint64_t i = 0; i < n_meta; ++i) {
        std::string key = c.read_string();
        uint32_t vtype = c.read_scalar<uint32_t>();
        MetaValue mv;
        read_value(c, (int)vtype, mv);
        if (!c.ok) return fail("truncated metadata");
        out->meta.emplace(std::move(key), std::move(mv));
    }

    out->tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = c.read_string();
        uint32_t nd = c.read_scalar<uint32_t>();
        for (uint32_t d = 0; d < nd && c.ok; ++d) t.dims.push_back(c.read_scalar<uint64_t>());
        t.ggml_type = c.read_scalar<uint32_t>();
        t.offset = c.read_scalar<uint64_t>();
        if (!c.ok) return fail("truncated tensor info");
        out->tensor_index[t.name] = (int)out->tensors.size();
        out->tensors.push_back(std::move(t));
    }

    // Data section begins after alignment padding (general.alignment or 32).
    uint32_t align = 32;
    { uint32_t a; if (gguf_get_u32(*out, "general.alignment", &a) && a) align = a; }
    size_t off = c.pos;
    off = (off + align - 1) / align * align;
    if (off > out->bytes.size()) return fail("data section past EOF");
    out->data_offset = off;
    return true;
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
    return g.bytes.data() + g.data_offset + t.offset;
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
        case GGML_Q6_K: return (n / 256) * 210;
        default: return 0;
    }
}
