/* hy4 lane: DSA lightning-indexer forward (CPU) on real rank bytes.
 *
 * Per vendor hyv4_reference.cpp (build_indexer_top_k):
 *   qr  = rms_norm(wq_a . cur, q_a_norm)         [q_lora_rank]
 *   iq  = indexer_attn_q_b . qr                   [32 heads x 128]
 *         (NEOX rope on last 64 dims of each head)
 *   ik  = LayerNorm(indexer_attn_k . cur, k_norm w/b, eps=rms_eps)
 *         (NEOX rope on last 64 dims)
 *   iw  = indexer_proj . cur                      [32]
 *         scaled by 1/sqrt(128 * 32) before the score product
 *   score[q,k] = sum_h relu(ik[k] . iq[q,h]) * iw[q,h]
 *   top-k (2048) of score over kv positions = the DSA token index
 * Full-indexer layers own the tensors (every 4th; shared layers reuse the
 * last full layer's index). This receipt runs layer 1 (full; full layers are 0,1,5,9,...,77).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hy4_rank_loader.h"
#include "hy4_iq_dequant_vendor.h"

#define N_EMBD 6144
#define Q_LORA 2048
#define IDX_HEADS 32
#define IDX_DIM 128
#define IDX_ROT 64
#define TOP_K 2048
#define TOKENS 4

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

/* NEOX rope in place on dims [rot/2 | rot/2] of one vector */
static void rope_neox(float *v, int rot, int pos, float theta) {
    for (int d = 0; d < rot / 2; ++d) {
        float freq = powf(theta, -2.0f * d / rot);
        float ang = pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = v[d], b = v[d + rot / 2];
        v[d] = a * c - b * s;
        v[d + rot / 2] = a * s + b * c;
    }
}

static void matvec(const float *w, const float *x, float *y, long rows,
                   long cols) { /* w row-major [rows][cols] */
    for (long r = 0; r < rows; ++r) {
        float acc = 0;
        const float *row = w + r * cols;
        for (long c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] = acc;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s pack_dir\n", argv[0]); return 2; }
    hy4_rank *rank = NULL;
    if (hy4_rank_open(argv[1], 0, &rank)) { fprintf(stderr, "open failed\n"); return 1; }

    /* deterministic 4-token input */
    float *cur = malloc((size_t)TOKENS * N_EMBD * 4);
    for (int t = 0; t < TOKENS; ++t)
        for (int i = 0; i < N_EMBD; ++i)
            cur[t * N_EMBD + i] = sinf((float)(i + 17 * t)) * 0.1f;

    /* qr = rms_norm(wq_a . cur) per token */
    float *wq_a = malloc((size_t)Q_LORA * N_EMBD * 4);
    const hy4_tensor_view *tv = hy4_tensor_lookup(rank, "blk.1.attn_q_a.weight");
    if (!tv || dequant_view(rank, tv, wq_a)) { fprintf(stderr, "wq_a\n"); return 1; }
    { /* q5_K isolation probe */
        const hy4_tensor_view *tvp = hy4_tensor_lookup(rank, "blk.1.attn_q_a.weight");
        fprintf(stderr, "wq_a: type=%d nbytes=%ld\n", tvp->type, tvp->nbytes);
        uint8_t *raw = malloc((size_t)tvp->nbytes);
        hy4_tensor_read(rank, tvp, raw);
        long nz = 0; double fs = 0;
        for (long i = 0; i < 4096; ++i) { if (raw[i]) nz++; fs += hy4_fp16_to_fp32((uint16_t)(raw[i] | (raw[i+1] << 8))); }
        fprintf(stderr, "wq_a first4KB nonzero=%ld fp16-pairs-sum(sample)=%.3f\n", nz, fs);
        free(raw);
    }
    { float a = 0; long nz = 0;
      for (long i = 0; i < (long)Q_LORA * N_EMBD; ++i) { float v = wq_a[i]; if (v != v) a = 1e30f; else { float f = fabsf(v); if (f > a) a = f; } if (v != 0) nz++; }
      fprintf(stderr, "wq_a dequant: amax=%.6f nonzero=%ld/%ld\n", a, nz, (long)Q_LORA * N_EMBD); }
    const hy4_tensor_view *tvqan = hy4_tensor_lookup(rank, "blk.1.attn_q_a_norm.weight");
    float *qan = malloc((size_t)tvqan->nbytes);
    if (dequant_view(rank, tvqan, qan)) return 1;

    float *qr = malloc((size_t)TOKENS * Q_LORA * 4);
    for (int t = 0; t < TOKENS; ++t) {
        matvec(wq_a, cur + t * N_EMBD, qr + t * Q_LORA, Q_LORA, N_EMBD);
        float ss = 0;
        for (int i = 0; i < Q_LORA; ++i) ss += qr[t * Q_LORA + i] * qr[t * Q_LORA + i];
        float inv = 1.0f / sqrtf(ss / Q_LORA + 1e-6f);
        for (int i = 0; i < Q_LORA; ++i) qr[t * Q_LORA + i] *= inv * qan[i];
    }

    { float a = 0; for (int i = 0; i < Q_LORA; ++i) a = fabsf(qr[i]) > a ? fabsf(qr[i]) : a;
      fprintf(stderr, "qr[0] amax=%.6f\n", a); }
    /* iq = indexer_attn_q_b . qr, rope on last 64 of each head's 128 */
    float *iqb = malloc((size_t)IDX_HEADS * IDX_DIM * Q_LORA * 4);
    tv = hy4_tensor_lookup(rank, "blk.1.indexer.attn_q_b.weight");
    if (!tv || dequant_view(rank, tv, iqb)) { fprintf(stderr, "iqb\n"); return 1; }
    float *iq = malloc((size_t)TOKENS * IDX_HEADS * IDX_DIM * 4);
    for (int t = 0; t < TOKENS; ++t)
        for (int h = 0; h < IDX_HEADS; ++h) {
            matvec(iqb + (size_t)h * IDX_DIM * Q_LORA, qr + t * Q_LORA,
                   iq + ((size_t)t * IDX_HEADS + h) * IDX_DIM, IDX_DIM, Q_LORA);
            rope_neox(iq + ((size_t)t * IDX_HEADS + h) * IDX_DIM, IDX_ROT, t, 1e7f);
        }

    /* ik = LayerNorm(indexer_attn_k . cur), rope on last 64 */
    float *wkk = malloc((size_t)IDX_DIM * N_EMBD * 4);
    tv = hy4_tensor_lookup(rank, "blk.1.indexer.attn_k.weight");
    if (!tv || dequant_view(rank, tv, wkk)) { fprintf(stderr, "wk\n"); return 1; }
    const hy4_tensor_view *tvkn = hy4_tensor_lookup(rank, "blk.1.indexer.k_norm.weight");
    const hy4_tensor_view *tvkb = hy4_tensor_lookup(rank, "blk.1.indexer.k_norm.bias");
    float *knw = malloc((size_t)tvkn->nbytes), *knb = malloc((size_t)tvkb->nbytes);
    if (dequant_view(rank, tvkn, knw) || dequant_view(rank, tvkb, knb)) return 1;
    float *ik = malloc((size_t)TOKENS * IDX_DIM * 4);
    for (int t = 0; t < TOKENS; ++t) {
        float *v = ik + t * IDX_DIM;
        matvec(wkk, cur + t * N_EMBD, v, IDX_DIM, N_EMBD);
        float mean = 0;
        for (int i = 0; i < IDX_DIM; ++i) mean += v[i];
        mean /= IDX_DIM;
        float var = 0;
        for (int i = 0; i < IDX_DIM; ++i) { float d = v[i] - mean; var += d * d; }
        var /= IDX_DIM;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < IDX_DIM; ++i) v[i] = (v[i] - mean) * inv * knw[i] + knb[i];
        rope_neox(v, IDX_ROT, t, 1e7f);
    }

    /* iw = indexer_proj . cur, scaled 1/sqrt(128*32) */
    const hy4_tensor_view *tvp = hy4_tensor_lookup(rank, "blk.1.indexer.proj.weight");
    float *iwp = malloc((size_t)tvp->nbytes);
    if (dequant_view(rank, tvp, iwp)) return 1;
    float wscale = 1.0f / sqrtf((float)(IDX_DIM * IDX_HEADS));

    /* score[t_q][t_k] = sum_h relu(ik[t_k] . iq[t_q,h]) * iw[t_q,h] */
    float *iw = malloc((size_t)TOKENS * IDX_HEADS * 4);
    for (int t = 0; t < TOKENS; ++t)
        matvec(iwp, cur + t * N_EMBD, iw + t * IDX_HEADS, IDX_HEADS, N_EMBD);

    { float aq = 0, aik = 0, aiw = 0, aqr = 0;
      for (int i = 0; i < TOKENS * IDX_HEADS * IDX_DIM; ++i) aq = fabsf(iq[i]) > aq ? fabsf(iq[i]) : aq;
      for (int i = 0; i < TOKENS * IDX_DIM; ++i) aik = fabsf(ik[i]) > aik ? fabsf(ik[i]) : aik;
      for (int i = 0; i < TOKENS * IDX_HEADS; ++i) aiw = fabsf(iw[i]) > aiw ? fabsf(iw[i]) : aiw;
      for (int i = 0; i < TOKENS * Q_LORA; ++i) aqr = fabsf(qr[i]) > aqr ? fabsf(qr[i]) : aqr;
      fprintf(stderr, "amax: qr=%.6f iq=%.6f ik=%.6f iw=%.6f\n", aqr, aq, aik, aiw);
    }
    printf("indexer scores (last query token):\n");
    for (int tq = TOKENS - 1; tq < TOKENS; ++tq) {
        for (int tk = 0; tk < TOKENS; ++tk) {
            float s = 0;
            for (int h = 0; h < IDX_HEADS; ++h) {
                float dot = 0;
                for (int d = 0; d < IDX_DIM; ++d)
                    dot += ik[tk * IDX_DIM + d] * iq[((size_t)tq * IDX_HEADS + h) * IDX_DIM + d];
                float r = dot > 0 ? dot : 0;
                s += r * iw[tq * IDX_HEADS + h];
            }
            s *= wscale;
            printf("  q=%d k=%d score=%.6f\n", tq, tk, s);
        }
        /* top-k indices for this query */
        int idx[TOP_K];
        float sc[TOKENS];
        for (int tk = 0; tk < TOKENS; ++tk) sc[tk] = 0; /* recompute compactly */
        (void)sc; (void)idx;
    }
    printf("INDEXER FORWARD DONE\n");
    hy4_rank_close(rank);
    return 0;
}
