#include "floaty_model.h"
#include <cstring>
#include <cstdio>

// ---- fp16 <-> fp32 (host) ----
uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000; int e = int((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e <= 0) { if (e < -10) return (uint16_t)s; m |= 0x800000; int sh = 14 - e; uint32_t r = m >> sh; return (uint16_t)(s | r); }
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}
float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, out;
    if (e == 0) { if (m == 0) out = s << 31; else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; out = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) out = (s << 31) | 0x7F800000 | (m << 13);
    else out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &out, 4); return f;
}

void ggml_block_info(uint32_t type, size_t* be, size_t* bb) {
    switch (type) {
        case 0:  *be = 1;   *bb = 4;   break;   // F32
        case 1:  *be = 1;   *bb = 2;   break;   // F16
        case 2:  *be = 32;  *bb = 18;  break;   // Q4_0
        case 8:  *be = 32;  *bb = 34;  break;   // Q8_0
        case 10: *be = 256; *bb = 84;  break;   // Q2_K
        case 11: *be = 256; *bb = 110; break;   // Q3_K
        case 12: *be = 256; *bb = 144; break;   // Q4_K
        case 13: *be = 256; *bb = 176; break;   // Q5_K
        case 14: *be = 256; *bb = 210; break;   // Q6_K
        default: *be = 0;   *bb = 0;   break;
    }
}

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static void unpack_q3_scales(const uint8_t* s, int8_t* sc) {
    uint32_t a0 = s[0]|(uint32_t(s[1])<<8)|(uint32_t(s[2])<<16)|(uint32_t(s[3])<<24);
    uint32_t a1 = s[4]|(uint32_t(s[5])<<8)|(uint32_t(s[6])<<16)|(uint32_t(s[7])<<24);
    uint32_t a2 = s[8]|(uint32_t(s[9])<<8)|(uint32_t(s[10])<<16)|(uint32_t(s[11])<<24);
    const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu; uint32_t aux[4];
    aux[2] = ((a0 >> 4) & km2) | (((a2 >> 4) & km1) << 4);
    aux[3] = ((a1 >> 4) & km2) | (((a2 >> 6) & km1) << 4);
    aux[0] = (a0 & km2) | (((a2 >> 0) & km1) << 4);
    aux[1] = (a1 & km2) | (((a2 >> 2) & km1) << 4);
    memcpy(sc, aux, 16);
}
static inline uint16_t H(float v) { return f32_to_f16(v); }

// CPU dequant of n elements -> fp16. Mirrors the GPU kernels' math exactly.
void cpu_dequant(const uint8_t* q, uint16_t* y, uint32_t type, size_t n) {
    if (type == 0) { const float* f = (const float*)q; for (size_t i = 0; i < n; ++i) y[i] = H(f[i]); return; }
    if (type == 1) { memcpy(y, q, n * 2); return; }
    if (type == 2) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const uint8_t* blk = q + b * 18; float d = f16_to_f32(*(const uint16_t*)blk); const uint8_t* qs = blk + 2; for (int j = 0; j < 16; ++j) { y[b*32+j] = H(d*((qs[j]&0xF)-8)); y[b*32+j+16] = H(d*((qs[j]>>4)-8)); } } return; }
    if (type == 8) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const uint8_t* blk = q + b * 34; float d = f16_to_f32(*(const uint16_t*)blk); const int8_t* qs = (const int8_t*)(blk + 2); for (int j = 0; j < 32; ++j) y[b*32+j] = H(d*qs[j]); } return; }
    size_t nb = n / 256;
    for (size_t b = 0; b < nb; ++b) {
        uint16_t* yy = y + b * 256;
        if (type == 12) {               // Q4_K
            const uint8_t* blk = q + b * 144; float d = f16_to_f32(*(const uint16_t*)blk), dmin = f16_to_f32(*(const uint16_t*)(blk+2));
            const uint8_t* sc = blk + 4; const uint8_t* qs = blk + 16; int yi = 0;
            for (int g = 0; g < 256; g += 64) { uint8_t s1,m1,s2,m2; get_scale_min_k4(g/32+0,sc,s1,m1); float d1=d*s1,mn1=dmin*m1; get_scale_min_k4(g/32+1,sc,s2,m2); float d2=d*s2,mn2=dmin*m2; const uint8_t* p = qs + (g/64)*32; for (int l=0;l<32;++l) yy[yi++]=H(d1*(p[l]&0xF)-mn1); for (int l=0;l<32;++l) yy[yi++]=H(d2*(p[l]>>4)-mn2); }
        } else if (type == 13) {        // Q5_K
            const uint8_t* blk = q + b * 176; float d = f16_to_f32(*(const uint16_t*)blk), dmin = f16_to_f32(*(const uint16_t*)(blk+2));
            const uint8_t* sc = blk + 4; const uint8_t* qh = blk + 16; const uint8_t* ql = blk + 48; int yi = 0; uint8_t u1=1,u2=2;
            for (int j = 0; j < 256; j += 64) { uint8_t s1,m1,s2,m2; get_scale_min_k4(j/32+0,sc,s1,m1); float d1=d*s1,mn1=dmin*m1; get_scale_min_k4(j/32+1,sc,s2,m2); float d2=d*s2,mn2=dmin*m2; for (int l=0;l<32;++l) yy[yi++]=H(d1*((ql[l]&0xF)+((qh[l]&u1)?16:0))-mn1); for (int l=0;l<32;++l) yy[yi++]=H(d2*((ql[l]>>4)+((qh[l]&u2)?16:0))-mn2); ql+=32; u1<<=2; u2<<=2; }
        } else if (type == 14) {        // Q6_K
            const uint8_t* blk = q + b * 210; const uint8_t* ql = blk; const uint8_t* qh = blk + 128; const int8_t* sc = (const int8_t*)(blk+192); float d = f16_to_f32(*(const uint16_t*)(blk+208));
            for (int nn = 0; nn < 256; nn += 128) { const uint8_t* Ql = ql+(nn/128)*64; const uint8_t* Qh = qh+(nn/128)*32; const int8_t* Sc = sc+(nn/128)*8; for (int l=0;l<32;++l) { int is=l/16; int q1=int((Ql[l]&0xF)|(((Qh[l]>>0)&3)<<4))-32; int q2=int((Ql[l+32]&0xF)|(((Qh[l]>>2)&3)<<4))-32; int q3=int((Ql[l]>>4)|(((Qh[l]>>4)&3)<<4))-32; int q4=int((Ql[l+32]>>4)|(((Qh[l]>>6)&3)<<4))-32; yy[nn+l]=H(d*Sc[is+0]*q1); yy[nn+l+32]=H(d*Sc[is+2]*q2); yy[nn+l+64]=H(d*Sc[is+4]*q3); yy[nn+l+96]=H(d*Sc[is+6]*q4); } }
        } else if (type == 11) {        // Q3_K
            const uint8_t* blk = q + b * 110; const uint8_t* hmask = blk; const uint8_t* qs = blk + 32; int8_t scl[16]; unpack_q3_scales(blk+96, scl); float d = f16_to_f32(*(const uint16_t*)(blk+108));
            for (int h = 0; h < 2; ++h) for (int j = 0; j < 4; ++j) for (int r = 0; r < 32; ++r) { int qidx=h*32+r, sidx=h*8+j*2+(r>=16?1:0); uint8_t mbit=uint8_t(1u<<(h*4+j)); int ql=(qs[qidx]>>(2*j))&3; int hb=(hmask[r]&mbit)?0:4; yy[h*128+j*32+r]=H(d*(scl[sidx]-32)*(ql-hb)); }
        } else if (type == 10) {        // Q2_K
            const uint8_t* blk = q + b * 84; const uint8_t* sc = blk; const uint8_t* qs = blk + 16; float d = f16_to_f32(*(const uint16_t*)(blk+80)), dmin = f16_to_f32(*(const uint16_t*)(blk+82));
            for (int h = 0; h < 2; ++h) for (int j = 0; j < 4; ++j) for (int r = 0; r < 32; ++r) { int qidx=h*32+r; uint8_t s=sc[h*8+j*2+(r>=16?1:0)]; float dl=d*(s&0xF), ml=dmin*(s>>4); yy[h*128+j*32+r]=H(dl*((qs[qidx]>>(2*j))&3)-ml); }
        }
    }
}

static bool cfg_u32(const GgufFile& g, const std::string& a, const char* k, int* dst) {
    uint32_t u; if (!gguf_get_u32(g, a + "." + k, &u)) return false; *dst = (int)u; return true;
}

bool floaty_load(const char* path, FModel* m, std::string* err) {
    if (!gguf_load(path, &m->g, err)) return false;
    const GgufFile& g = m->g;
    if (!gguf_get_str(g, "general.architecture", &m->arch)) { if (err) *err = "no arch"; return false; }
    const std::string& a = m->arch;
    if (!cfg_u32(g, a, "block_count", &m->n_layers) || !cfg_u32(g, a, "embedding_length", &m->dim) ||
        !cfg_u32(g, a, "attention.head_count", &m->n_heads)) { if (err) *err = "missing config"; return false; }
    if (!cfg_u32(g, a, "attention.head_count_kv", &m->n_kv_heads)) m->n_kv_heads = m->n_heads;
    if (!cfg_u32(g, a, "feed_forward_length", &m->ffn)) { if (err) *err = "no feed_forward_length"; return false; }
    uint32_t nexp = 0; if (gguf_get_u32(g, a + ".expert_count", &nexp) && nexp > 1) { if (err) *err = "MoE not supported by portable runner (dense only)"; return false; }
    m->head_dim = m->dim / m->n_heads;
    float f; if (gguf_get_f32(g, a + ".attention.layer_norm_rms_epsilon", &f)) m->eps = f;
    if (gguf_get_f32(g, a + ".rope.freq_base", &f)) m->rope_base = f;

    const TensorInfo* embd = gguf_find_tensor(g, "token_embd.weight");
    if (!embd) { if (err) *err = "no token_embd"; return false; }
    m->vocab = embd->dims.size() > 1 ? (int)embd->dims[1] : (int)(gguf_tensor_elements(*embd) / m->dim);

    auto qmat = [&](const char* name, FMat* out) -> bool {
        const TensorInfo* t = gguf_find_tensor(g, name); if (!t) return false;
        size_t be, bb; ggml_block_info(t->ggml_type, &be, &bb); if (!be) return false;
        out->type = t->ggml_type; out->in = (int)t->dims[0]; out->out = t->dims.size() > 1 ? (int)t->dims[1] : 1;
        out->src = gguf_tensor_data(g, *t); out->quant_bytes = ((size_t)out->out * out->in / be) * bb; return true;
    };
    auto normvec = [&](const char* name, std::vector<uint16_t>* out) -> bool {
        const TensorInfo* t = gguf_find_tensor(g, name); if (!t) return false;
        size_t n = gguf_tensor_elements(*t); out->resize(n); cpu_dequant(gguf_tensor_data(g, *t), out->data(), t->ggml_type, n); return true;
    };

    m->layers.resize(m->n_layers);
    char nm[96];
    for (int L = 0; L < m->n_layers; ++L) {
        FLayer& ly = m->layers[L];
        auto nm_of = [&](const char* s) { snprintf(nm, sizeof(nm), "blk.%d.%s", L, s); return (const char*)nm; };
        std::string an = nm_of("attn_norm.weight"); if (!normvec(an.c_str(), &ly.attn_norm)) { if (err) *err = "no attn_norm"; return false; }
        std::string fn = nm_of("ffn_norm.weight");  if (!normvec(fn.c_str(), &ly.ffn_norm))  { if (err) *err = "no ffn_norm"; return false; }
        bool ok = qmat(std::string(nm_of("attn_q.weight")).c_str(), &ly.wq)
               && qmat(std::string(nm_of("attn_k.weight")).c_str(), &ly.wk)
               && qmat(std::string(nm_of("attn_v.weight")).c_str(), &ly.wv)
               && qmat(std::string(nm_of("attn_output.weight")).c_str(), &ly.wo)
               && qmat(std::string(nm_of("ffn_gate.weight")).c_str(), &ly.gate)
               && qmat(std::string(nm_of("ffn_up.weight")).c_str(), &ly.up)
               && qmat(std::string(nm_of("ffn_down.weight")).c_str(), &ly.down);
        if (!ok) { if (err) *err = "missing layer weight"; return false; }
    }
    // token embeddings + final norm -> fp16; LM head stays quantized (gemv).
    { size_t n = gguf_tensor_elements(*embd); m->token_embd.resize(n); cpu_dequant(gguf_tensor_data(g, *embd), m->token_embd.data(), embd->ggml_type, n); }
    if (!normvec("output_norm.weight", &m->final_norm)) { if (err) *err = "no output_norm"; return false; }
    const TensorInfo* ow = gguf_find_tensor(g, "output.weight");
    const TensorInfo* head = ow ? ow : embd;   // tied embeddings fall back to token_embd
    size_t be, bb; ggml_block_info(head->ggml_type, &be, &bb);
    m->output.type = head->ggml_type; m->output.in = (int)head->dims[0]; m->output.out = head->dims.size() > 1 ? (int)head->dims[1] : m->vocab;
    m->output.src = gguf_tensor_data(g, *head); m->output.quant_bytes = ((size_t)m->output.out * m->output.in / be) * bb;
    return true;
}

void floaty_free(FModel* m) { gguf_close(&m->g); }
