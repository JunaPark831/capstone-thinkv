// M3 융합 어텐션 (decode, 토큰 하나). 압축된 KV를 16비트 사본으로 풀어 두지 않고, 읽으면서 바로 계산한다.
// 입력 모양은 ondevice/ventuno_cache.py의 Layer·Pool과 같다. 부르는 쪽은 ondevice/ventuno_attn.py.
// KV 헤드마다 따로 계산하므로 OpenMP가 있으면 헤드별로 나눠 돈다(없으면 한 코어).
#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

struct Pool {                         // ventuno_cache.Pool 하나. ventuno_attn.py의 CPool과 순서가 같아야 한다
    const uint8_t *k, *v, *vs, *ks;   // k·v [S, Hkv, D*bits/8], vs [S, Hkv, D/16], ks [n_gid, Hkv, D]
    const int8_t* live;               // [S] 1이면 살아 있는 슬롯 (BlockTable.live_mask)
    const int32_t* gid;               // [S] 슬롯이 속한 묶음 번호
    const float *kg, *vg;             // [n_gid] glob
    int64_t S, n_gid, bits;
};

static const int8_t NV4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};  // nvfp4 코드 → 격자 값 × 2
static const int8_t TER[16] = {0, 1, 0, -1};                                             // ternary 코드 → 값

// 코드 묶음 한 줄(토큰 하나, 헤드 하나)을 정수 값 D개로 푼다. nvfp4는 격자 값의 2배다(0.5를 정수로 담으려고).
static void unpack(const uint8_t* p, int64_t bits, int d, int8_t* out) {
    if (bits == 4) {
        const int8x16_t t = vld1q_s8(NV4);
        for (int i = 0; i < d / 2; i += 16) {
            uint8x16_t b = vld1q_u8(p + i);
            int8x16x2_t z = {{vqtbl1q_s8(t, vandq_u8(b, vdupq_n_u8(15))), vqtbl1q_s8(t, vshrq_n_u8(b, 4))}};
            vst2q_s8(out + 2 * i, z);   // 짝수 채널 = 아래 4비트, 홀수 채널 = 위 4비트
        }
    } else {
        const int8x16_t t = vld1q_s8(TER);
        const uint8x16_t m = vdupq_n_u8(3);
        for (int i = 0; i < d / 4; i += 16) {
            uint8x16_t b = vld1q_u8(p + i);
            int8x16x4_t z = {{vqtbl1q_s8(t, vandq_u8(b, m)), vqtbl1q_s8(t, vandq_u8(vshrq_n_u8(b, 2), m)),
                              vqtbl1q_s8(t, vandq_u8(vshrq_n_u8(b, 4), m)), vqtbl1q_s8(t, vshrq_n_u8(b, 6))}};
            vst4q_s8(out + 4 * i, z);   // 채널 4i+j = 비트 2j
        }
    }
}

static inline float dot(const float* a, const float* b, int d) {
    float32x4_t s = vdupq_n_f32(0.f);
    for (int c = 0; c < d; c += 4) s = vfmaq_f32(s, vld1q_f32(a + c), vld1q_f32(b + c));
    return vaddvq_f32(s);
}
static inline void axpy(float w, const float* x, float* y, int d) {
    for (int c = 0; c < d; c++) y[c] += w * x[c];
}
static inline void to_f32(const float* x, float* y, int d) { std::memcpy(y, x, d * sizeof(float)); }
static inline void to_f32(const int8_t* x, float* y, int d) {
    for (int c = 0; c < d; c++) y[c] = x[c];
}
static inline void to_f32(const uint16_t* x, float* y, int d) {   // bf16 → float32: 위 16비트에 그대로 넣는다
    for (int c = 0; c < d; c++) {
        uint32_t u = uint32_t(x[c]) << 16;
        std::memcpy(y + c, &u, 4);
    }
}

// q [Hq, D] float32, 프롬프트 pk·pv [Hkv, P, D], 열린 묶음 bk·bv [Hkv, 16, D] 중 앞 nb개, 풀 npool개,
// lut[256] = E4M3 바이트 → 값. 결과 out [Hq, D] float32.
// 점수는 한 줄에 모아 softmax를 한 번에 한다. 줄 순서: 프롬프트 → 풀 슬롯(죽은 슬롯은 -inf) → 열린 묶음.
template <class T>
static void attend(const float* q, int hq, int hkv, int d, float scaling, const T* pk, const T* pv, int64_t P,
                   const T* bk, const T* bv, int64_t nb, const Pool* pools, int npool, const float* lut, float* out) {
    const int r = hq / hkv;           // GQA: 쿼리 헤드 h는 KV 헤드 h / r을 본다 (kernels/reference.py와 같은 규칙)
    int64_t n = P + nb;
    for (int i = 0; i < npool; i++) n += pools[i].S;

#pragma omp parallel for
    for (int h = 0; h < hkv; h++) {
        std::vector<float> sc(r * n, -INFINITY), qs(r * d), x(d), acc(r * d, 0.f);
        std::vector<int8_t> code(d);
        for (int j = 0; j < r * d; j++) qs[j] = q[(int64_t)h * r * d + j] * scaling;

        // 1) 점수
        int64_t o = 0;
        for (int64_t t = 0; t < P; t++, o++) {
            to_f32(pk + (h * P + t) * d, x.data(), d);
            for (int j = 0; j < r; j++) sc[j * n + o] = dot(&qs[j * d], x.data(), d);
        }
        for (int i = 0; i < npool; i++) {
            const Pool& p = pools[i];
            const int64_t row = d * p.bits / 8;
            const float half = p.bits == 4 ? 0.5f : 1.f;
            // 묶음마다 q' = q × 키 스케일 × glob을 먼저 만든다. 안쪽 루프는 q'와 정수 코드의 내적만 남는다.
            std::vector<float> qp(p.n_gid * r * d);
            for (int64_t g = 0; g < p.n_gid; g++)
                for (int c = 0; c < d; c++) {
                    float s = std::fmax(lut[p.ks[(g * hkv + h) * d + c]] * p.kg[g], 1e-12f) * half;
                    for (int j = 0; j < r; j++) qp[(g * r + j) * d + c] = qs[j * d + c] * s;
                }
            for (int64_t s = 0; s < p.S; s++, o++) {
                if (!p.live[s]) continue;
                unpack(p.k + (s * hkv + h) * row, p.bits, d, code.data());
                to_f32(code.data(), x.data(), d);
                const float* qg = &qp[(int64_t)p.gid[s] * r * d];
                for (int j = 0; j < r; j++) sc[j * n + o] = dot(qg + j * d, x.data(), d);
            }
        }
        for (int64_t t = 0; t < nb; t++, o++) {
            to_f32(bk + (h * 16 + t) * d, x.data(), d);
            for (int j = 0; j < r; j++) sc[j * n + o] = dot(&qs[j * d], x.data(), d);
        }

        // 2) softmax (쿼리 헤드마다)
        for (int j = 0; j < r; j++) {
            float* s = &sc[j * n];
            float m = -INFINITY, z = 0.f;
            for (int64_t t = 0; t < n; t++) m = std::fmax(m, s[t]);
            for (int64_t t = 0; t < n; t++) z += (s[t] = std::exp(s[t] - m));
            for (int64_t t = 0; t < n; t++) s[t] /= z;
        }

        // 3) 값 누적 (점수와 같은 줄 순서)
        o = 0;
        for (int64_t t = 0; t < P; t++, o++) {
            to_f32(pv + (h * P + t) * d, x.data(), d);
            for (int j = 0; j < r; j++) axpy(sc[j * n + o], x.data(), &acc[j * d], d);
        }
        for (int i = 0; i < npool; i++) {
            const Pool& p = pools[i];
            const int64_t row = d * p.bits / 8;
            const float half = p.bits == 4 ? 0.5f : 1.f;
            for (int64_t s = 0; s < p.S; s++, o++) {
                if (!p.live[s]) continue;
                unpack(p.v + (s * hkv + h) * row, p.bits, d, code.data());
                to_f32(code.data(), x.data(), d);
                const uint8_t* vs = p.vs + (s * hkv + h) * (d / 16);
                const float vg = p.vg[p.gid[s]];
                for (int b = 0; b < d / 16; b++) {          // 값 스케일은 채널 16개마다 하나
                    const float sb = std::fmax(lut[vs[b]] * vg, 1e-12f) * half;
                    for (int j = 0; j < r; j++) axpy(sc[j * n + o] * sb, &x[b * 16], &acc[j * d + b * 16], 16);
                }
            }
        }
        for (int64_t t = 0; t < nb; t++, o++) {
            to_f32(bv + (h * 16 + t) * d, x.data(), d);
            for (int j = 0; j < r; j++) axpy(sc[j * n + o], x.data(), &acc[j * d], d);
        }
        std::memcpy(out + (int64_t)h * r * d, acc.data(), r * d * sizeof(float));
    }
}

extern "C" {
void attend_bf16(const float* q, int hq, int hkv, int d, float scaling, const uint16_t* pk, const uint16_t* pv,
                 int64_t P, const uint16_t* bk, const uint16_t* bv, int64_t nb, const Pool* pools, int npool,
                 const float* lut, float* out) {
    attend(q, hq, hkv, d, scaling, pk, pv, P, bk, bv, nb, pools, npool, lut, out);
}
void attend_f32(const float* q, int hq, int hkv, int d, float scaling, const float* pk, const float* pv, int64_t P,
                const float* bk, const float* bv, int64_t nb, const Pool* pools, int npool, const float* lut,
                float* out) {
    attend(q, hq, hkv, d, scaling, pk, pv, P, bk, bv, nb, pools, npool, lut, out);
}
}
