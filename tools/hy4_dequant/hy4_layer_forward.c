/* hy4 lane: full single-layer forward (CPU, fp32) on real rank bytes.
 * Block 1 (full indexer, sparse MoE) of the deployed rank-02 shard, run
 * sequentially over T=4 tokens exactly as the model executes: each token's
 * hc mixing consumes the stream state at its step, attention attends over
 * the cached kv of positions 0..t, and the MoE routes per token. At T=4 the
 * DSA top-k (2048) covers all positions, so the indexer mask is a no-op.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hy4_rank_loader.h"
#include "hy4_iq_dequant_vendor.h"

#define N_EMBD 6144
#define HC 4
#define T 4
#define HEADS 64
#define HK 256
#define NOPE 192
#define ROT 64
#define KV_LORA 512
#define VD 256
#define N_FF 2048
#define N_EXPERT 256
#define N_USED 8
#define OWN_LO 32   /* rank-02 owns experts 32..47 */
#define SCALE 2.827f

static float silu(float v) { return v / (1.0f + expf(-v)); }

static int dequant_view(const hy4_rank *rank, const hy4_tensor_view *tv,
                        float *y) {
    uint8_t *src = malloc((size_t)tv->nbytes);
    if (hy4_tensor_read(rank, tv, src)) return -1;
    if (tv->type == 0) memcpy(y, src, (size_t)tv->nbytes);
    else if (tv->type == 8) {
        long nb = tv->nbytes / 34;
        for (long b = 0; b < nb; ++b) {
            uint16_t bits = (uint16_t)src[b*34] | ((uint16_t)src[b*34+1] << 8);
            float s = hy4_fp16_to_fp32(bits);
            for (int i = 0; i < 32; ++i) y[b*32+i] = s * (float)(int8_t)src[b*34+2+i];
        }
    } else if (tv->type == 12) hy4_dequant_row_q4_K((const block_q4_K *)src, y, tv->nbytes / 144 * 256);
    else if (tv->type == 13) hy4_dequant_row_q5_K((const block_q5_K *)src, y, tv->nbytes / 176 * 256);
    else if (tv->type == 14) hy4_dequant_row_q6_K((const block_q6_K *)src, y, tv->nbytes / 210 * 256);
    else if (tv->type == 16) hy4_dequant_iq2_xxs(src, y, tv->nbytes / 66);
    else if (tv->type == 18) hy4_dequant_iq3_xxs(src, y, tv->nbytes / 98);
    else if (tv->type == 29) hy4_dequant_iq1_m(src, y, tv->nbytes / 56);
    else { fprintf(stderr, "unsupported type %d\n", tv->type); free(src); return -1; }
    free(src);
    return 0;
}

static void matvec(const float *w, const float *x, float *y, long rows, long cols) {
    for (long r = 0; r < rows; ++r) {
        float acc = 0;
        const float *row = w + r * cols;
        for (long c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] = acc;
    }
}

static void rope_neox(float *v, int rot, int pos) {
    for (int d = 0; d < rot / 2; ++d) {
        float ang = pos * powf(1e7f, -2.0f * d / rot);
        float a = v[d], b = v[d + rot / 2];
        v[d] = a * cosf(ang) - b * sinf(ang);
        v[d + rot / 2] = a * sinf(ang) + b * cosf(ang);
    }
}

static void rms_norm(const float *x, const float *w, float *y, long n, float eps) {
    float ss = 0;
    for (long i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (long i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

/* hc mixing: fn per stream, summed; pre/post gates from scale/base halves */
static void hc_mix(const float *fn, const float *streams, const float *scale,
                   const float *base, float *pre, float *post) {
    /* fn consumes the FLATTENED hc-stream vector: [HC*N_EMBD] -> [2*HC] */
    float mixed[HC * 2];
    matvec(fn, streams, mixed, HC * 2, HC * N_EMBD);
    for (int i = 0; i < HC; ++i)
        pre[i] = 1.0f / (1.0f + expf(-(mixed[i] * scale[0] + base[i]))) + 1e-6f;
    for (int i = 0; i < HC; ++i)
        post[i] = 2.0f / (1.0f + expf(-(mixed[HC + i] * scale[1] + base[HC + i]))) + 1e-6f;
}

static void hc_reduce(const float *streams, const float *pre, float *out) {
    for (int i = 0; i < N_EMBD; ++i) {
        float acc = 0;
        for (int s = 0; s < HC; ++s) acc += streams[s * N_EMBD + i] * pre[s];
        out[i] = acc;
    }
}

static void hc_distribute(float *streams, const float *branch, const float *post) {
    for (int s = 0; s < HC; ++s)
        for (int i = 0; i < N_EMBD; ++i)
            streams[s * N_EMBD + i] += branch[i] * post[s];
}

static double fsum(const float *v, long n) {
    double s = 0;
    for (long i = 0; i < n; ++i) s += v[i];
    return s;
}

static int has_nan(const float *v, long n) {
    for (long i = 0; i < n; ++i) if (v[i] != v[i]) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s pack_dir\n", argv[0]); return 2; }
    hy4_rank *rank = NULL;
    if (hy4_rank_open(argv[1], 0, &rank)) { fprintf(stderr, "open failed\n"); return 1; }

    const hy4_tensor_view *tv;
    float *hc_attn_fn = malloc((size_t)HC * 2 * (size_t)HC * N_EMBD * 4);
    float *hc_attn_scale = malloc(16), *hc_attn_base = malloc((size_t)HC * 2 * 4);
    float *ln1 = malloc((size_t)N_EMBD * 4);
    float *wq_a = malloc((size_t)2048 * N_EMBD * 4), *qan = malloc(2048 * 4);
    float *wq_b = malloc((size_t)HEADS * HK * 2048 * 4);
    float *wkv_a = malloc((size_t)(KV_LORA + ROT) * N_EMBD * 4), *kvan = malloc((size_t)KV_LORA * 4);
    float *wkb = malloc((size_t)HEADS * KV_LORA * NOPE * 4);
    float *wvb = malloc((size_t)HEADS * VD * KV_LORA * 4);
    float *wgate = malloc((size_t)HEADS * VD * N_EMBD * 4);
    float *wo = malloc((size_t)N_EMBD * HEADS * VD * 4);
    float *sinks = malloc(HEADS * 4);
    float *hc_ffn_fn = malloc((size_t)HC * 2 * (size_t)HC * N_EMBD * 4);
    float *hc_ffn_scale = malloc(16), *hc_ffn_base = malloc((size_t)HC * 2 * 4);
    float *ln2 = malloc((size_t)N_EMBD * 4);
    float *wgate_inp = malloc((size_t)N_EXPERT * N_EMBD * 4);
    float *ebias = malloc(N_EXPERT * 4);

#define LOAD(name, buf) do { \
    if (!(tv = hy4_tensor_lookup(rank, name)) || dequant_view(rank, tv, buf)) { \
        fprintf(stderr, "LOAD FAIL: %s\n", name); return 1; } \
    fprintf(stderr, "loaded %s type=%d nbytes=%ld\n", name, tv->type, tv->nbytes); } while(0)
    LOAD("blk.1.hc_attn_fn.weight", hc_attn_fn);
    LOAD("blk.1.hc_attn_scale.weight", hc_attn_scale);
    LOAD("blk.1.hc_attn_base.weight", hc_attn_base);
    LOAD("blk.1.attn_norm.weight", ln1);
    LOAD("blk.1.attn_q_a.weight", wq_a);
    LOAD("blk.1.attn_q_a_norm.weight", qan);
    LOAD("blk.1.attn_q_b.weight", wq_b);
    LOAD("blk.1.attn_kv_a_mqa.weight", wkv_a);
    LOAD("blk.1.attn_kv_a_norm.weight", kvan);
    LOAD("blk.1.attn_k_b.weight", wkb);
    LOAD("blk.1.attn_v_b.weight", wvb);
    LOAD("blk.1.attn_gate.weight", wgate);
    LOAD("blk.1.attn_output.weight", wo);
    LOAD("blk.1.attn_sinks.weight", sinks);
    LOAD("blk.1.hc_ffn_fn.weight", hc_ffn_fn);
    LOAD("blk.1.hc_ffn_scale.weight", hc_ffn_scale);
    LOAD("blk.1.hc_ffn_base.weight", hc_ffn_base);
    LOAD("blk.1.ffn_norm.weight", ln2);
    LOAD("blk.1.ffn_gate_inp.weight", wgate_inp);
    LOAD("blk.1.exp_probs_b.bias", ebias);
    fprintf(stderr, "layer weights loaded\n");

    /* kv cache across tokens */
    static float klat[T][KV_LORA], kpe[T][ROT];

    /* hc streams: deterministic init */
    float *streams = malloc((size_t)HC * N_EMBD * 4);
    for (int s = 0; s < HC; ++s)
        for (int i = 0; i < N_EMBD; ++i)
            streams[s * N_EMBD + i] = sinf((float)(i + 17)) * 0.1f;

    /* shared MoE weights (dequant once) */
    float *shg = malloc((size_t)N_FF * N_EMBD * 4);
    float *shu = malloc((size_t)N_FF * N_EMBD * 4);
    float *shd = malloc((size_t)N_EMBD * N_FF * 4);
    if (!(tv = hy4_tensor_lookup(rank, "blk.1.ffn_gate_shexp.weight")) || dequant_view(rank, tv, shg)) return 1;
    if (!(tv = hy4_tensor_lookup(rank, "blk.1.ffn_up_shexp.weight")) || dequant_view(rank, tv, shu)) return 1;
    if (!(tv = hy4_tensor_lookup(rank, "blk.1.ffn_down_shexp.weight")) || dequant_view(rank, tv, shd)) return 1;
    float *gw = malloc((size_t)N_FF * N_EMBD * 4);
    float *uw = malloc((size_t)N_FF * N_EMBD * 4);
    float *dw = malloc((size_t)N_EMBD * N_FF * 4);

    for (int t = 0; t < T; ++t) {
        /* ---- attention branch ---- */
        float pre[HC], post[HC];
        hc_mix(hc_attn_fn, streams, hc_attn_scale, hc_attn_base, pre, post);
        float cur[N_EMBD];
        hc_reduce(streams, pre, cur);
        rms_norm(cur, ln1, cur, N_EMBD, 1e-5f);

        float qr[2048];
        matvec(wq_a, cur, qr, 2048, N_EMBD);
        rms_norm(qr, qan, qr, 2048, 1e-5f);

        float kvc[KV_LORA + ROT];
        matvec(wkv_a, cur, kvc, KV_LORA + ROT, N_EMBD);
        memcpy(kpe[t], kvc + KV_LORA, ROT * 4);
        rope_neox(kpe[t], ROT, t);
        rms_norm(kvc, kvan, klat[t], KV_LORA, 1e-5f);

        float attn_all[HEADS * VD];
        memset(attn_all, 0, sizeof(attn_all));
        for (int h = 0; h < HEADS; ++h) {
            float qh[HK];
            matvec(wq_b + (size_t)h * HK * 2048, qr, qh, HK, 2048);
            rope_neox(qh + NOPE, ROT, t);
            float q_abs[KV_LORA];
            matvec(wkb + (size_t)h * KV_LORA * NOPE, qh, q_abs, KV_LORA, NOPE);
            float qfull[KV_LORA + ROT];
            memcpy(qfull, q_abs, KV_LORA * 4);
            memcpy(qfull + KV_LORA, qh + NOPE, ROT * 4);

            float logits[T + 1], maxl = -INFINITY;
            for (int tk = 0; tk <= t; ++tk) {
                float s = 0;
                for (int i = 0; i < KV_LORA; ++i) s += q_abs[i] * klat[tk][i];
                for (int i = 0; i < ROT; ++i) s += qh[NOPE + i] * kpe[tk][i];
                logits[tk] = s / sqrtf((float)HK) + sinks[h];
                if (logits[tk] > maxl) maxl = logits[tk];
            }
            float sink_p = expf(sinks[h] - maxl);
            float denom = sink_p, num[T];
            for (int tk = 0; tk <= t; ++tk) {
                num[tk] = expf(logits[tk] - maxl);
                denom += num[tk];
            }
            float vlat[KV_LORA];
            memset(vlat, 0, sizeof(vlat));
            for (int tk = 0; tk <= t; ++tk) {
                float p = num[tk] / denom;
                for (int i = 0; i < KV_LORA; ++i) vlat[i] += p * klat[tk][i];
            }
            matvec(wvb + (size_t)h * VD * KV_LORA, vlat, attn_all + (size_t)h * VD, VD, KV_LORA);
        }
        float *gatev = malloc((size_t)HEADS * VD * 4);
        matvec(wgate, cur, gatev, HEADS * VD, N_EMBD);
        for (int i = 0; i < HEADS * VD; ++i) attn_all[i] *= 1.0f / (1.0f + expf(-gatev[i]));
        free(gatev);
        float *abranch = malloc((size_t)N_EMBD * 4);
        matvec(wo, attn_all, abranch, N_EMBD, HEADS * VD);
        hc_distribute(streams, abranch, post);
        free(abranch);

        /* ---- ffn branch ---- */
        float fpre[HC], fpost[HC];
        hc_mix(hc_ffn_fn, streams, hc_ffn_scale, hc_ffn_base, fpre, fpost);
        float fcur[N_EMBD];
        hc_reduce(streams, fpre, fcur);
        rms_norm(fcur, ln2, fcur, N_EMBD, 1e-5f);

        float logits[N_EXPERT], probs[N_EXPERT], sel_key[N_EXPERT];
        for (int e = 0; e < N_EXPERT; ++e) {
            float acc = 0;
            for (int i = 0; i < N_EMBD; ++i) acc += wgate_inp[e * N_EMBD + i] * fcur[i];
            logits[e] = acc;
            probs[e] = 1.0f / (1.0f + expf(-acc));
            sel_key[e] = probs[e] + ebias[e];
        }
        int sel[N_USED];
        float w[N_USED];
        for (int k = 0; k < N_USED; ++k) {
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < N_EXPERT; ++e)
                if (sel_key[e] > bv) { bv = sel_key[e]; best = e; }
            sel_key[best] = -INFINITY;
            sel[k] = best;
            w[k] = probs[best];
        }
        float wsum = 0;
        for (int k = 0; k < N_USED; ++k) wsum += w[k];
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        for (int k = 0; k < N_USED; ++k) w[k] = w[k] / wsum * SCALE;

        float ffn[N_EMBD];
        memset(ffn, 0, sizeof(ffn));
        float g[N_FF], u[N_FF], h[N_EMBD];
        int used_local = 0;
        const hy4_tensor_view *tvg = hy4_tensor_lookup(rank, "blk.1.ffn_gate_exps.weight");
        const hy4_tensor_view *tvu = hy4_tensor_lookup(rank, "blk.1.ffn_up_exps.weight");
        const hy4_tensor_view *tvd = hy4_tensor_lookup(rank, "blk.1.ffn_down_exps.weight");
        long slab_g = tvg->nbytes / 16, slab_u = tvu->nbytes / 16, slab_d = tvd->nbytes / 16;
        for (int k = 0; k < N_USED; ++k) {
            int li = sel[k] - OWN_LO;
            if (li < 0 || li >= 16) continue;
            used_local++;
            hy4_tensor_view slab;
            slab = *tvg; slab.file_offset += li * slab_g; slab.nbytes = slab_g;
            slab.n_dims = 2; slab.dims[0] = N_FF; slab.dims[1] = N_EMBD; slab.dims[2] = 0;
            if (dequant_view(rank, &slab, gw)) return 1;
            slab = *tvu; slab.file_offset += li * slab_u; slab.nbytes = slab_u;
            slab.n_dims = 2; slab.dims[0] = N_FF; slab.dims[1] = N_EMBD; slab.dims[2] = 0;
            if (dequant_view(rank, &slab, uw)) return 1;
            slab = *tvd; slab.file_offset += li * slab_d; slab.nbytes = slab_d;
            slab.n_dims = 2; slab.dims[0] = N_EMBD; slab.dims[1] = N_FF; slab.dims[2] = 0;
            if (dequant_view(rank, &slab, dw)) return 1;
            float g[N_FF], u[N_FF], h[N_EMBD];
            matvec(gw, fcur, g, N_FF, N_EMBD);
            matvec(uw, fcur, u, N_FF, N_EMBD);
            for (int j = 0; j < N_FF; ++j) {
                if (u[j] > 10.0f) u[j] = 10.0f; if (u[j] < -10.0f) u[j] = -10.0f;
                float ga = silu(g[j]);
                if (ga > 10.0f) ga = 10.0f;
                u[j] = ga * u[j];
            }
            matvec(dw, u, h, N_EMBD, N_FF);
            for (int i = 0; i < N_EMBD; ++i) ffn[i] += w[k] * h[i];
        }
        matvec(shg, fcur, g, N_FF, N_EMBD);
        matvec(shu, fcur, u, N_FF, N_EMBD);
        for (int j = 0; j < N_FF; ++j) u[j] = silu(g[j]) * u[j];
        matvec(shd, u, h, N_EMBD, N_FF);
        for (int i = 0; i < N_EMBD; ++i) ffn[i] += h[i];

        hc_distribute(streams, ffn, fpost);
        printf("t=%d used_local=%d stream_sum=%.6f stream_amax=%.6f nan=%d\n",
               t, used_local, fsum(streams, (long)HC * N_EMBD),
               fabsf(streams[0]), has_nan(streams, (long)HC * N_EMBD));
        if (has_nan(streams, (long)HC * N_EMBD)) { printf("RED STOP: NaN\n"); return 1; }
    }
    printf("LAYER FORWARD DONE\n");
    hy4_rank_close(rank);
    return 0;
}
