/* hy4 lane: all-rank simulated-TP16 generation (CPU, fp32) on one node.
 *
 * Runs the full 78-layer stack over a prompt + generated tokens with the
 * TP16 partial sums simulated in-process: attention heads split 4/rank,
 * o_proj head-sliced partials summed, MoE owned-expert partials summed,
 * lm_head scored per rank-vocab slice with a global argmax. Replicated
 * tensors (q_a/kv_a/norms/router/shared-expert/hc fns) are computed once
 * from rank 0's copy; head-sliced and expert tensors come from each rank's
 * own bundle. Exactness chain: same vendor dequant + router already
 * verified; the collective simulation is exact fp32 summation in rank
 * order (a real all-reduce may reorder -- noted in the report).
 *
 * Usage: hy4_generate <allranks_dir> [gen_tokens]
 * Requires all 16 rank bundles under <dir>/rank-XX/.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hy4_rank_loader.h"
#include "hy4_iq_dequant_vendor.h"

#define N_EMBD 6144
#define HC 4
#define HEADS 64
#define RANK_HEADS 4
#define N_RANKS 16
#define HK 256
#define NOPE 192
#define ROT 64
#define KV_LORA 512
#define VD 256
#define N_FF 2048
#define N_EXPERT 256
#define N_USED 8
#define SCALE 2.827f
#define LAYERS 78
#define VOCAB 120832
#define VOC_PER_RANK (VOCAB / N_RANKS)
#define PROMPT 8
#define GEN 1
#define TOTAL_TOK (PROMPT + GEN)

static int g_dump_hc = 0;

static float silu(float v) { return v / (1.0f + expf(-v)); }

static int dequant_view(const hy4_rank *rank, const hy4_tensor_view *tv,
                        float *y) {
    uint8_t *src = malloc((size_t)tv->nbytes);
    if (hy4_tensor_read(rank, tv, src)) { free(src); return -1; }
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
    else if (tv->type == 23) hy4_dequant_row_iq4_xs((const block_iq4_xs *)src, y, tv->nbytes / 136 * 256);
    else if (tv->type == 29) hy4_dequant_iq1_m(src, y, tv->nbytes / 56);
    else { fprintf(stderr, "unsupported type %d\n", tv->type); free(src); return -1; }
    free(src);
    return 0;
}

static void matvec(const float *w, const float *x, float *y, long rows, long cols) {
    for (long r = 0; r < rows; ++r) {
        double acc = 0;
        const float *row = w + r * cols;
        for (long c = 0; c < cols; ++c) acc += (double)row[c] * (double)x[c];
        y[r] = (float)acc;
    }
}

static void rope_pairs(float *v, int rot, int pos) {
    for (int d = 0; d < rot / 2; ++d) {
        float ang = pos * powf(1e7f, -2.0f * d / rot);
        float a = v[2 * d], b = v[2 * d + 1];
        v[2 * d] = a * cosf(ang) - b * sinf(ang);
        v[2 * d + 1] = a * sinf(ang) + b * cosf(ang);
    }
}

static void rms_norm(const float *x, const float *w, float *y, long n, float eps) {
    float ss = 0;
    for (long i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (long i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
}

static void hc_mix(const float *fn, const float *streams, const float *scale,
                   const float *base, float *pre, float *post) {
    float mixed[HC * 2];
    /* the flattened stream vector is RMS-NORMED (no weight, model eps)
     * before the hc_fn matvec — vendor builds RMS_NORM(hc_init) first */
    float flat_norm[HC * N_EMBD];
    float ss = 0;
    for (int i = 0; i < HC * N_EMBD; ++i) ss += streams[i] * streams[i];
    float inv = 1.0f / sqrtf(ss / (HC * N_EMBD) + 1e-5f);
    for (int i = 0; i < HC * N_EMBD; ++i) flat_norm[i] = streams[i] * inv;
    matvec(fn, flat_norm, mixed, HC * 2, HC * N_EMBD);
    if (g_dump_hc) {
        fprintf(stderr, "MY_HCIN first6: %.4f %.4f %.4f %.4f %.4f %.4f\n",
                flat_norm[0], flat_norm[1], flat_norm[2], flat_norm[3],
                flat_norm[4], flat_norm[5]);
        fprintf(stderr, "MY_HCIN last3: %.4f %.4f %.4f\n",
                flat_norm[HC * N_EMBD - 3], flat_norm[HC * N_EMBD - 2],
                flat_norm[HC * N_EMBD - 1]);
        fprintf(stderr, "MY_HC_MIXES first8: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                mixed[0], mixed[1], mixed[2], mixed[3],
                mixed[4], mixed[5], mixed[6], mixed[7]);
        g_dump_hc = 0;
    }
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

static float *loadt(const hy4_rank *rank, const char *name) {
    const hy4_tensor_view *tv = hy4_tensor_lookup(rank, name);
    if (!tv) { fprintf(stderr, "missing %s\n", name); return NULL; }
    long nelems = 1;
    for (int i = 0; i < tv->n_dims; ++i) nelems *= tv->dims[i];
    float *y = malloc((size_t)nelems * 4);
    if (dequant_view(rank, tv, y)) { free(y); return NULL; }
    return y;
}

static hy4_rank *R[N_RANKS];

static float *load0(const char *name) { return loadt(R[0], name); }

/* embedding row for token id: owned by rank id/VOC_PER_RANK */
static int embed_token(int id, float *out) {
    int owner = id / VOC_PER_RANK;
    int row = id % VOC_PER_RANK;
    const hy4_tensor_view *tv = hy4_tensor_lookup(R[owner], "token_embd.weight");
    if (!tv) return -1;
    long cols = tv->dims[0];
    hy4_tensor_view row_view = *tv;
    row_view.file_offset += (long)row * (tv->nbytes / VOC_PER_RANK);
    row_view.nbytes = tv->nbytes / VOC_PER_RANK;
    row_view.n_dims = 1; row_view.dims[0] = N_EMBD; row_view.dims[1] = 0;
    return dequant_view(R[owner], &row_view, out);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s allranks_dir [gen] [promptfile] [dumpfile dumptlayer]\n", argv[0]); return 2; }
    int gen = argc > 2 ? atoi(argv[2]) : GEN;
    const char *dump_path = argc > 4 ? argv[4] : NULL;
    int dump_layer = argc > 5 ? atoi(argv[5]) : -1;

    /* prompt ids come from the verified GGUF tokenizer (hy4_tokenize.py
     * encode, BOS prepended); the C side stays arithmetic-only */
    int tokens_fixed[TOTAL_TOK];
    int n_prompt = 0;
    if (argc > 3) {
        FILE *pf = fopen(argv[3], "r");
        if (!pf) { fprintf(stderr, "prompt file missing\n"); return 2; }
        int v;
        while (n_prompt < PROMPT && fscanf(pf, "%d", &v) == 1)
            tokens_fixed[n_prompt++] = v;
        fclose(pf);
        fprintf(stderr, "prompt ids (%d):", n_prompt);
        for (int i = 0; i < n_prompt; ++i) fprintf(stderr, " %d", tokens_fixed[i]);
        fprintf(stderr, "\n");
    }

    char path[1100];
    for (int r = 0; r < N_RANKS; ++r) {
        snprintf(path, sizeof(path), "%s/rank-%02d", argv[1], r);
        if (hy4_rank_open(path, 0, &R[r])) { fprintf(stderr, "open rank %d\n", r); return 1; }
    }
    fprintf(stderr, "16 rank bundles open\n");

    int tokens[TOTAL_TOK];
    int n_tok = n_prompt > 0 ? n_prompt : PROMPT;
    int total = n_tok + gen;
    for (int i = 0; i < total; ++i)
        tokens[i] = i < n_prompt ? tokens_fixed[i] : 0;
    int generated = -1;

    /* per-token hc streams, cached per token for autoregressive carry */
    static float streams[TOTAL_TOK][HC][N_EMBD];
    /* kv cache per layer per token (replicated across ranks) */
    static float klat[LAYERS][TOTAL_TOK][KV_LORA];
    static float kpe[LAYERS][TOTAL_TOK][ROT];

    for (int t = 0; t < total; ++t) {
        if (t >= n_tok) tokens[t] = generated;
        /* embedding from the owning rank */
        float x0[N_EMBD];
        if (embed_token(tokens[t], x0)) { fprintf(stderr, "embed fail\n"); return 1; }
        if (t == 0)
            fprintf(stderr,
                    "MY_EMBD t0 first3 %.6f %.6f %.6f last3 %.6f %.6f %.6f\n",
                    x0[0], x0[1], x0[2], x0[N_EMBD - 3], x0[N_EMBD - 2],
                    x0[N_EMBD - 1]);
        for (int s = 0; s < HC; ++s)
            for (int i = 0; i < N_EMBD; ++i)
                streams[t][s][i] = x0[i];

        for (int il = 0; il < LAYERS; ++il) {
            if (il == 0 && t == 0) g_dump_hc = 1;

            char nm[160];
            float pre[HC], post[HC];
            /* hc fns are replicated: rank 0's copy */
            float *hc_attn_fn = load0((snprintf(nm, 160, "blk.%d.hc_attn_fn.weight", il), nm));
            float *hc_sc = load0((snprintf(nm, 160, "blk.%d.hc_attn_scale.weight", il), nm));
            float *hc_ba = load0((snprintf(nm, 160, "blk.%d.hc_attn_base.weight", il), nm));
            float *ln1 = load0((snprintf(nm, 160, "blk.%d.attn_norm.weight", il), nm));
            hc_mix(hc_attn_fn, streams[t], hc_sc, hc_ba, pre, post);
            float cur[N_EMBD];
            hc_reduce(streams[t], pre, cur);
            rms_norm(cur, ln1, cur, N_EMBD, 1e-5f);
            if (il == 0 && t == 0)
                fprintf(stderr, "MY_ATTIN first3: %.6f %.6f %.6f\n", cur[0], cur[1], cur[2]);
            if (il == 0 && t == 0)
                fprintf(stderr, "MY_ATTIN first3: %.6f %.6f %.6f\n",
                        cur[0], cur[1], cur[2]);

            /* kv from replicated kv_a (rank 0) */
            float *wkv_a = load0((snprintf(nm, 160, "blk.%d.attn_kv_a_mqa.weight", il), nm));
            float *kvan = load0((snprintf(nm, 160, "blk.%d.attn_kv_a_norm.weight", il), nm));
            float kvc[KV_LORA + ROT];
            matvec(wkv_a, cur, kvc, KV_LORA + ROT, N_EMBD);
            memcpy(kpe[il][t], kvc + KV_LORA, ROT * 4);
            rope_pairs(kpe[il][t], ROT, t);
            rms_norm(kvc, kvan, klat[il][t], KV_LORA, 1e-5f);
            if (il == 0 && t == 0) {
                fprintf(stderr,
                        "MY_KV_CMPR first3 %.6f %.6f %.6f last3 %.6f %.6f %.6f\n",
                        klat[0][0][0], klat[0][0][1], klat[0][0][2],
                        klat[0][0][KV_LORA - 3], klat[0][0][KV_LORA - 2],
                        klat[0][0][KV_LORA - 1]);
                fprintf(stderr, "MY_K_PE first3 %.6f %.6f %.6f last3 %.6f %.6f %.6f\n",
                        kpe[0][0][0], kpe[0][0][1], kpe[0][0][2],
                        kpe[0][0][ROT - 3], kpe[0][0][ROT - 2], kpe[0][0][ROT - 1]);
            }
            free(wkv_a); free(kvan);

            /* attention: per-rank head slices, partial o_proj summed */
            float *qan = load0((snprintf(nm, 160, "blk.%d.attn_q_a_norm.weight", il), nm));
            float *wq_a = load0((snprintf(nm, 160, "blk.%d.attn_q_a.weight", il), nm));
            float qr_buf[2048];
            matvec(wq_a, cur, qr_buf, 2048, N_EMBD);
            rms_norm(qr_buf, qan, qr_buf, 2048, 1e-5f);
            float *sinks = load0((snprintf(nm, 160, "blk.%d.attn_sinks.weight", il), nm));
            float attn_partial[N_EMBD];
            memset(attn_partial, 0, sizeof(attn_partial));
            float *gate_full = NULL;
            for (int r = 0; r < N_RANKS; ++r) {
                snprintf(nm, sizeof(nm), "blk.%d.attn_q_b.weight", il);
                float *wq_b = loadt(R[r], nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_k_b.weight", il);
                float *wkb = loadt(R[r], nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_v_b.weight", il);
                float *wvb = loadt(R[r], nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il);
                float *wo = loadt(R[r], nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", il);
                float *wgate = loadt(R[r], nm);
                if (!wq_b || !wkb || !wvb || !wo || !wgate) return 1;
                float attn_heads[RANK_HEADS * VD];
                for (int lh = 0; lh < RANK_HEADS; ++lh) {
                    int h = r * RANK_HEADS + lh;
                    float qh[HK];
                    matvec(wq_b + (size_t)lh * HK * 2048, qr_buf, qh, HK, 2048);
                    rope_pairs(qh + NOPE, ROT, t);
                    if (il == 0 && t == 0 && r == 0 && lh == 0)
                        fprintf(stderr,
                                "MY_Q_PE h0 first3 %.6f %.6f %.6f last3 %.6f %.6f %.6f\n",
                                qh[NOPE], qh[NOPE + 1], qh[NOPE + 2],
                                qh[HK - 3], qh[HK - 2], qh[HK - 1]);
                    float q_abs[KV_LORA];
                    matvec(wkb + (size_t)lh * KV_LORA * NOPE, qh, q_abs, KV_LORA, NOPE);
                    if (il == 0 && t == 0 && r == 0 && lh == 0)
                        fprintf(stderr, "MY_Q_ABS h0 first3 %.6f %.6f %.6f\n",
                                q_abs[0], q_abs[1], q_abs[2]);
                    float logits[TOTAL_TOK + 1], maxl = sinks[h];
                    for (int tk = 0; tk <= t; ++tk) {
                        float s = 0;
                        for (int i = 0; i < KV_LORA; ++i) s += q_abs[i] * klat[il][tk][i];
                        for (int i = 0; i < ROT; ++i) s += qh[NOPE + i] * kpe[il][tk][i];
                        logits[tk] = s / sqrtf((float)HK);
                        if (logits[tk] > maxl) maxl = logits[tk];
                    }
                    if (il == 0 && t == 0 && r == 0 && lh == 0)
                        fprintf(stderr,
                                "MY_LOGITS h0: %.6f %.6f %.6f %.6f sink %.6f\n",
                                logits[0], logits[1], logits[2], logits[3], sinks[h]);
                    float denom = expf(sinks[h] - maxl), num[TOTAL_TOK];
                    for (int tk = 0; tk <= t; ++tk) {
                        num[tk] = expf(logits[tk] - maxl);
                        denom += num[tk];
                    }
                    float vlat[KV_LORA];
                    memset(vlat, 0, sizeof(vlat));
                    for (int tk = 0; tk <= t; ++tk) {
                        float p = num[tk] / denom;
                        for (int i = 0; i < KV_LORA; ++i) vlat[i] += p * klat[il][tk][i];
                    }
                    matvec(wvb + (size_t)lh * VD * KV_LORA, vlat,
                           attn_heads + (size_t)lh * VD, VD, KV_LORA);
                    if (il == 0 && t == 0 && r == 0 && lh == 0)
                        fprintf(stderr, "MY_ATTN_KVQ h0 first3 %.6f %.6f %.6f\n",
                                attn_heads[0], attn_heads[1], attn_heads[2]);
                }
                float *gatev = malloc((size_t)RANK_HEADS * VD * 4);
                for (int lh = 0; lh < RANK_HEADS; ++lh)
                    matvec(wgate + (size_t)lh * VD * N_EMBD, cur,
                           gatev + (size_t)lh * VD, VD, N_EMBD);
                for (int i = 0; i < RANK_HEADS * VD; ++i)
                    attn_heads[i] *= 1.0f / (1.0f + expf(-gatev[i]));
                free(gatev);
                float opart[N_EMBD];
                matvec(wo, attn_heads, opart, N_EMBD, RANK_HEADS * VD);
                for (int i = 0; i < N_EMBD; ++i) attn_partial[i] += opart[i];
                free(wq_b); free(wkb); free(wvb); free(wo); free(wgate);
            }
            if (t == 0)
                fprintf(stderr,
                        "MY_ATTN_L%d t0 first3 %.6f %.6f %.6f\n", il,
                        attn_partial[0], attn_partial[1], attn_partial[2]);
            free(wq_a); free(qan); free(sinks);
            hc_distribute(streams[t], attn_partial, post);
            if (t == 0)
                fprintf(stderr, "post-attn L%d t0: sum %.6f nan=%d\n",
                        il, fsum(streams, (long)HC * N_EMBD),
                        has_nan(streams, (long)HC * N_EMBD));

            /* ---- ffn branch ---- */
            float fpre[HC], fpost[HC];
            float *hc_ffn_fn = load0((snprintf(nm, 160, "blk.%d.hc_ffn_fn.weight", il), nm));
            float *hc_fsc = load0((snprintf(nm, 160, "blk.%d.hc_ffn_scale.weight", il), nm));
            float *hc_fba = load0((snprintf(nm, 160, "blk.%d.hc_ffn_base.weight", il), nm));
            float *ln2 = load0((snprintf(nm, 160, "blk.%d.ffn_norm.weight", il), nm));
            hc_mix(hc_ffn_fn, streams[t], hc_fsc, hc_fba, fpre, fpost);
            float fcur[N_EMBD];
            hc_reduce(streams[t], fpre, fcur);
            rms_norm(fcur, ln2, fcur, N_EMBD, 1e-5f);

            if (il == 0) {
                /* dense layer: no routing, replicated dense FFN from rank 0 */
                float *dg = load0((snprintf(nm, 160, "blk.0.ffn_gate.weight"), nm));
                float *du = load0((snprintf(nm, 160, "blk.0.ffn_up.weight"), nm));
                float *dd = load0((snprintf(nm, 160, "blk.0.ffn_down.weight"), nm));
                float g[18432], u[18432], h[N_EMBD];
                matvec(dg, fcur, g, 18432, N_EMBD);
                matvec(du, fcur, u, 18432, N_EMBD);
                for (int j = 0; j < 18432; ++j) u[j] = silu(g[j]) * u[j];
                matvec(dd, u, h, N_EMBD, 18432);
                hc_distribute(streams[t], h, fpost);
                if (t == 0)
                    fprintf(stderr, "MY_FFN_L%d t0 first3 %.6f %.6f %.6f\n",
                            il, h[0], h[1], h[2]);
                free(dg); free(du); free(dd);
            } else {
            float *ginp = load0((snprintf(nm, 160, "blk.%d.ffn_gate_inp.weight", il), nm));
            float *eb = load0((snprintf(nm, 160, "blk.%d.exp_probs_b.bias", il), nm));
            float logits[N_EXPERT], probs[N_EXPERT], key[N_EXPERT];
            for (int e = 0; e < N_EXPERT; ++e) {
                float acc = 0;
                for (int i = 0; i < N_EMBD; ++i) acc += ginp[e * N_EMBD + i] * fcur[i];
                logits[e] = acc;
                probs[e] = 1.0f / (1.0f + expf(-acc));
                key[e] = probs[e] + eb[e];
            }
            int sel[N_USED];
            float ww[N_USED];
            for (int k = 0; k < N_USED; ++k) {
                int best = -1; float bv = -INFINITY;
                for (int e = 0; e < N_EXPERT; ++e)
                    if (key[e] > bv) { bv = key[e]; best = e; }
                key[best] = -INFINITY;
                sel[k] = best; ww[k] = probs[best];
            }
            if (il == 1 && t == 0)
                fprintf(stderr,
                        "MY_MOE_L1 sel %d %d %d %d %d %d %d %d raww "
                        "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        sel[0], sel[1], sel[2], sel[3],
                        sel[4], sel[5], sel[6], sel[7],
                        ww[0], ww[1], ww[2], ww[3],
                        ww[4], ww[5], ww[6], ww[7]);
            float wsum = 0;
            for (int k = 0; k < N_USED; ++k) wsum += ww[k];
            if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
            for (int k = 0; k < N_USED; ++k) ww[k] = ww[k] / wsum * SCALE;
            free(ginp); free(eb);

            float ffn[N_EMBD];
            memset(ffn, 0, sizeof(ffn));
            float *gw = malloc((size_t)N_FF * N_EMBD * 4);
            float *uw = malloc((size_t)N_FF * N_EMBD * 4);
            float *dw2 = malloc((size_t)N_EMBD * N_FF * 4);
            for (int r = 0; r < N_RANKS; ++r) {
                int lo = r * 16;
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
                const hy4_tensor_view *tvg = hy4_tensor_lookup(R[r], nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
                const hy4_tensor_view *tvu = hy4_tensor_lookup(R[r], nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
                const hy4_tensor_view *tvd = hy4_tensor_lookup(R[r], nm);
                long sgb = tvg->nbytes / 16, sub = tvu->nbytes / 16, sdb = tvd->nbytes / 16;
                hy4_tensor_view sg;
                for (int k = 0; k < N_USED; ++k) {
                    int li = sel[k] - lo;
                    if (li < 0 || li >= 16) continue;
                    sg = *tvg; sg.file_offset += li * sgb; sg.nbytes = sgb;
                    sg.n_dims = 2; sg.dims[0] = N_FF; sg.dims[1] = N_EMBD;
                    if (dequant_view(R[r], &sg, gw)) return 1;
                    sg = *tvu; sg.file_offset += li * sub; sg.nbytes = sub;
                    if (dequant_view(R[r], &sg, uw)) return 1;
                    sg = *tvd; sg.file_offset += li * sdb; sg.nbytes = sdb;
                    sg.n_dims = 2; sg.dims[0] = N_EMBD; sg.dims[1] = N_FF;
                    if (dequant_view(R[r], &sg, dw2)) return 1;
                    float g[N_FF], u[N_FF], h[N_EMBD];
                    matvec(gw, fcur, g, N_FF, N_EMBD);
                    matvec(uw, fcur, u, N_FF, N_EMBD);
                    for (int j = 0; j < N_FF; ++j) {
                        if (u[j] > 10.0f) u[j] = 10.0f; if (u[j] < -10.0f) u[j] = -10.0f;
                        float gj = g[j] > 10.0f ? 10.0f : g[j];
                        u[j] = silu(gj) * u[j];
                    }
                    matvec(dw2, u, h, N_EMBD, N_FF);
                    for (int i = 0; i < N_EMBD; ++i) ffn[i] += ww[k] * h[i];
                }
            }
            free(gw); free(uw); free(dw2);
            /* shared expert: replicated, compute once (rank 0) */
            {
                float *shg = load0((snprintf(nm, 160, "blk.%d.ffn_gate_shexp.weight", il), nm));
                float *shu = load0((snprintf(nm, 160, "blk.%d.ffn_up_shexp.weight", il), nm));
                float *shd = load0((snprintf(nm, 160, "blk.%d.ffn_down_shexp.weight", il), nm));
                float g[N_FF], u[N_FF], h[N_EMBD];
                matvec(shg, fcur, g, N_FF, N_EMBD);
                matvec(shu, fcur, u, N_FF, N_EMBD);
                for (int j = 0; j < N_FF; ++j) u[j] = silu(g[j]) * u[j];
                matvec(shd, u, h, N_EMBD, N_FF);
                for (int i = 0; i < N_EMBD; ++i) ffn[i] += h[i];
                free(shg); free(shu); free(shd);
            }
            if (t == 0)
                fprintf(stderr, "MY_FFN_L%d t0 first3 %.6f %.6f %.6f\n",
                        il, ffn[0], ffn[1], ffn[2]);
            hc_distribute(streams[t], ffn, fpost);
            } /* end sparse-MoE branch */
            if (dump_path && il == dump_layer) {
                char dname[1200];
                snprintf(dname, sizeof(dname), "%s.t%d", dump_path, t);
                FILE *df = fopen(dname, "wb");
                if (df) {
                    fwrite(streams[t], 4, (size_t)HC * N_EMBD, df);
                    fclose(df);
                }
            }
            if (t == 0)
                fprintf(stderr, "post-ffn  L%d t0: sum %.6f nan=%d\n",
                        il, fsum(streams, (long)HC * N_EMBD),
                        has_nan(streams, (long)HC * N_EMBD));
            free(hc_attn_fn); free(hc_sc); free(hc_ba); free(ln1);
            free(hc_ffn_fn); free(hc_fsc); free(hc_fba); free(ln2);
        }

        /* lm_head: hc_head collapse of the 4 streams first (vendor
         * build_hc_head: rms over the flattened hc*embd vector, fn mixes,
         * pre = sigmoid(mixes*scale+base)+eps, weighted reduce), then
         * output_norm, then the rank-local vocab matvec */
        float *hfn = load0("output_hc_fn.weight");
        float *hsc = load0("output_hc_scale.weight");
        float *hba = load0("output_hc_base.weight");
        float flat[N_EMBD * HC];
        memcpy(flat, streams[t], sizeof(float) * N_EMBD * HC);
        float fss = 0;
        for (int i = 0; i < N_EMBD * HC; ++i) fss += flat[i] * flat[i];
        float finv = 1.0f / sqrtf(fss / (N_EMBD * HC) + 1e-5f);
        for (int i = 0; i < N_EMBD * HC; ++i) flat[i] *= finv;
        float hm[HC];
        matvec(hfn, flat, hm, HC, N_EMBD * HC);
        float hpre[HC];
        for (int s = 0; s < HC; ++s)
            hpre[s] = 1.0f / (1.0f + expf(-(hm[s] * hsc[0] + hba[s]))) + 1e-6f;
        float collapsed[N_EMBD];
        for (int i = 0; i < N_EMBD; ++i) {
            float acc = 0;
            for (int s = 0; s < HC; ++s) acc += streams[t][s][i] * hpre[s];
            collapsed[i] = acc;
        }
        fprintf(stderr, "HCHEAD L? t=%d: mixes %.4f %.4f %.4f %.4f | "
                "hpre %.4f %.4f %.4f %.4f | collapsed first3 %.4f %.4f %.4f\n",
                t, hm[0], hm[1], hm[2], hm[3], hpre[0], hpre[1], hpre[2],
                hpre[3], collapsed[0], collapsed[1], collapsed[2]);
        free(hfn); free(hsc); free(hba);
        float *onorm = load0("output_norm.weight");
        float normed[N_EMBD];
        rms_norm(collapsed, onorm, normed, N_EMBD, 1e-5f);
        if (t == 0)
            fprintf(stderr, "L78OUT: collapsed %.4f %.4f %.4f | normed %.4f %.4f %.4f\n",
                    collapsed[0], collapsed[1], collapsed[2],
                    normed[0], normed[1], normed[2]);
        free(onorm);
        float *alllogits = malloc((size_t)VOCAB * 4);
        for (int r = 0; r < N_RANKS; ++r) {
            snprintf(path, sizeof(path), "%s/rank-%02d", argv[1], r);
            const hy4_tensor_view *tv = hy4_tensor_lookup(R[r], "output.weight");
            float *owl = malloc((size_t)tv->nbytes);
            if (dequant_view(R[r], tv, owl)) return 1;
            float lg[VOC_PER_RANK];
            matvec(owl, normed, lg, VOC_PER_RANK, N_EMBD);
            free(owl);
            for (int v = 0; v < VOC_PER_RANK; ++v) alllogits[r * VOC_PER_RANK + v] = lg[v];
        }
        int gbest = 0; float gbv = -INFINITY;
        for (int i = 0; i < VOCAB; ++i)
            if (alllogits[i] > gbv) { gbv = alllogits[i]; gbest = i; }
        if (t == n_tok - 1) {
            const int refids[5] = { 52392, 55794, 198, 341, 268 };
            for (int k = 0; k < 5; ++k)
                fprintf(stderr, "MY_REFLOGIT %d %.6f\n", refids[k],
                        alllogits[refids[k]]);
            int top[5]; float tvv[5];
            for (int k = 0; k < 5; ++k) {
                int bi = 0; float bv = -1e30f;
                for (int i = 0; i < VOCAB; ++i) {
                    int used = 0;
                    for (int m = 0; m < k; ++m) if (top[m] == i) used = 1;
                    if (!used && alllogits[i] > bv) { bv = alllogits[i]; bi = i; }
                }
                top[k] = bi; tvv[k] = bv;
            }
            for (int k = 0; k < 5; ++k) {
                fprintf(stderr, "HY4_TOP %d %.6f", top[k], tvv[k]);
                fputc(10, stderr);
            }
        }
        free(alllogits);
        generated = gbest;
        printf("t=%d token id %d (logit %.4f)%s\n", t, tokens[t], gbv,
               t >= n_tok ? " GENERATED" : " prompt");
        fflush(stdout);
    }
    printf("GENERATED TOKEN: %d\n", generated);
    printf("GENERATION DONE (simulated TP16, exact fp32 rank-order sums)\n");
    return 0;
}
