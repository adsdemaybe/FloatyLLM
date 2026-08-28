// K-quant dequant (Q4_K, Q6_K), ported from ggml dequantize_row_q4_K/q6_K.
#include "kquant.h"
#include <cstring>

namespace {

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

// Unpack 6-bit scale + min for sub-block j from the 12-byte packed scales.
void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

}  // namespace

void dequant_q4_K_f32(const uint8_t* data, size_t n, float* y) {
    const size_t nb = n / QK_K;
    const uint8_t* p = data;
    for (size_t i = 0; i < nb; ++i) {
        uint16_t db, dmb; memcpy(&db, p, 2); memcpy(&dmb, p + 2, 2);
        const float d = half_to_float(db);
        const float mn = half_to_float(dmb);
        const uint8_t* scales = p + 4;      // 12 bytes
        const uint8_t* q = p + 16;          // 128 bytes
        float* yy = y + i * QK_K;
        int is = 0, yi = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) yy[yi++] = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) yy[yi++] = d2 * (float)(q[l] >> 4) - m2;
            q += 32; is += 2;
        }
        p += 144;
    }
}

void dequant_q6_K_f32(const uint8_t* data, size_t n, float* y) {
    const size_t nb = n / QK_K;
    const uint8_t* p = data;
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t* ql = p;                          // 128
        const uint8_t* qh = p + 128;                    // 64
        const int8_t* sc = (const int8_t*)(p + 192);    // 16
        uint16_t dbits; memcpy(&dbits, p + 208, 2);
        const float d = half_to_float(dbits);
        float* yy = y + i * QK_K;
        for (int nn = 0; nn < QK_K; nn += 128) {
            const uint8_t* Ql = ql + (nn / 128) * 64;
            const uint8_t* Qh = qh + (nn / 128) * 32;
            const int8_t* Sc = sc + (nn / 128) * 8;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((Ql[l +  0] & 0xF) | (((Qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((Ql[l + 32] & 0xF) | (((Qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((Ql[l +  0]  >> 4) | (((Qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((Ql[l + 32]  >> 4) | (((Qh[l] >> 6) & 3) << 4)) - 32;
                yy[nn + l +  0] = d * Sc[is + 0] * q1;
                yy[nn + l + 32] = d * Sc[is + 2] * q2;
                yy[nn + l + 64] = d * Sc[is + 4] * q3;
                yy[nn + l + 96] = d * Sc[is + 6] * q4;
            }
        }
        p += 210;
    }
}
