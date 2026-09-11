/* hy4 lane: full 78-layer stack prompt forward (CPU, fp32), rank-02 partial.
 *
 * Layer-major prompt processing: all T tokens pass through each layer in
 * turn (filling that layer's kv cache), with each token carrying its own
 * hc-stream state [4 x 6144] across layers. Attention is this rank's 4-head
 * slice (kv_a latent is replicated, so the cache is rank-invariant); the MoE
 * contributes only this rank's owned experts; the final rank-local vocab
 * slice is scored. The TP16 all-reduce of attention/MoE partials and the
 * cross-rank argmax complete a true token at serving time.
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
#define RANK_HEADS 4
#define MY_RANK 2
#define HEAD_LO (MY_RANK * RANK_HEADS)
#define HK 256
#define NOPE 192
#define ROT 64
#define KV_LORA 512
#define VD 256
#define N_FF 2048
#define N_EXPERT 256
#define N_USED 8
#define OWN_LO 32
#define SCALE 2.827f
#define LAYERS 78
#define VOCAB 120832
#define VOC_PER_RANK (VOCAB / 16)

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
    else if (tv->type == 29) hy4_dequant_iq1_m(src, y, tv->nbytes / 56);
    else if (tv->type == 23) hy4_dequant_row_iq4_xs((const block_iq4_xs *)src, y, tv->nbytes / 136 * 256);
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

static void hc_mix(const float *fn, const float *streams, const float *scale,
                   const float *base, float *pre, float *post) {
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

static int has_nan(const float *v, long n) {
    for (long i = 0; i < n; ++i) if (v[i] != v[i]) return 1;
    return 0;
}

static float *loadt(const hy4_rank *rank, const char *name) {
    const hy4_tensor_view *tv = hy4_tensor_lookup(rank, name);
    if (!tv) { fprintf(stderr, "missing %s\n", name); return NULL; }
    /* allocate the DEQUANTIZED element count, not the compressed bytes */
    long nelems = 1;
    for (int i = 0; i < tv->n_dims; ++i) nelems *= tv->dims[i];
    float *y = malloc((size_t)nelems * 4);
    if (dequant_view(rank, tv, y)) { free(y); return NULL; }
    return y;
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s pack_dir\n", argv[0]); return 2; }
    hy4_rank *rank = NULL;
    if (hy4_rank_open(argv[1], 0, &rank)) { fprintf(stderr, "open failed\n"); return 1; }

    /* per-layer kv cache */
    static float klat[LAYERS][T][KV_LORA], kpe[LAYERS][T][ROT];

    /* per-token hc streams: initialized as the broadcast of the token's
     * embedding stand-in (true embedding gather needs the owning rank) */
    static float streams[T][HC][N_EMBD];
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < HC; ++s)
            for (int i = 0; i < N_EMBD; ++i)
                streams[t][s][i] = sinf((float)(i + 31 * t)) * 0.1f;

    /* reusable weight buffers: allocate for the largest shape once
     * (o_proj/gate: [16384, 6144] F32 = 402 MB) */
    float *W = malloc((size_t)HEADS * VD * N_EMBD * 4);
    float *W2 = malloc((size_t)HEADS * VD * N_EMBD * 4);

    for (int il = 0; il < LAYERS; ++il) {
        char nm[160];
        float *hc_attn_fn = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_attn_fn.weight", il), nm));
        float *hc_attn_sc = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_attn_scale.weight", il), nm));
        float *hc_attn_ba = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_attn_base.weight", il), nm));
        float *ln1 = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_norm.weight", il), nm));
        float *wq_a = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_q_a.weight", il), nm));
        float *qan = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_q_a_norm.weight", il), nm));
        float *wq_b = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_q_b.weight", il), nm));
        float *wkv_a = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_kv_a_mqa.weight", il), nm));
        float *kvan = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_kv_a_norm.weight", il), nm));
        float *wkb = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_k_b.weight", il), nm));
        float *wvb = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_v_b.weight", il), nm));
        float *wgate = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_gate.weight", il), nm));
        float *wo = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_output.weight", il), nm));
        float *sinks = loadt(rank, (snprintf(nm, 160, "blk.%d.attn_sinks.weight", il), nm));
        float *hc_ffn_fn = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_ffn_fn.weight", il), nm));
        float *hc_ffn_sc = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_ffn_scale.weight", il), nm));
        float *hc_ffn_ba = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_ffn_base.weight", il), nm));
        float *ln2 = loadt(rank, (snprintf(nm, 160, "blk.%d.ffn_norm.weight", il), nm));
        if (!hc_attn_fn || !hc_attn_sc || !hc_attn_ba || !ln1 || !wq_a || !qan ||
            !wq_b || !wkv_a || !kvan || !wkb || !wvb || !wgate || !wo || !sinks ||
            !hc_ffn_fn || !hc_ffn_sc || !hc_ffn_ba || !ln2) return 1;

        float *hc_ffn_fn2 = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_ffn_fn.weight", il), nm));
        float *hc_ffn_sc2 = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_ffn_scale.weight", il), nm));
        float *hc_ffn_ba2 = loadt(rank, (snprintf(nm, 160, "blk.%d.hc_ffn_base.weight", il), nm));
        float *ln2v = loadt(rank, (snprintf(nm, 160, "blk.%d.ffn_norm.weight", il), nm));
        for (int t = 0; t < T; ++t) {
            float pre[HC], post[HC];
            hc_mix(hc_attn_fn, streams[t], hc_attn_sc, hc_attn_ba, pre, post);
            float att_in[N_EMBD];
            hc_reduce(streams[t], pre, att_in);
            rms_norm(att_in, ln1, att_in, N_EMBD, 1e-5f);
            float cur[N_EMBD];
            memcpy(cur, att_in, N_EMBD * 4);
            (void)cur;

            float qr[2048];
            matvec(wq_a, att_in, qr, 2048, N_EMBD);
            rms_norm(qr, qan, qr, 2048, 1e-5f);
            float kvc[KV_LORA + ROT];
            matvec(wkv_a, att_in, kvc, KV_LORA + ROT, N_EMBD);
            memcpy(kpe[il][t], kvc + KV_LORA, ROT * 4);
            rope_neox(kpe[il][t], ROT, t);
            rms_norm(kvc, kvan, klat[il][t], KV_LORA, 1e-5f);

            float attn_all[RANK_HEADS * VD];
            memset(attn_all, 0, sizeof(attn_all));
            for (int lh = 0; lh < RANK_HEADS; ++lh) {
                int h = HEAD_LO + lh;
                float qh[HK];
                matvec(wq_b + (size_t)lh * HK * 2048, qr, qh, HK, 2048);
                rope_neox(qh + NOPE, ROT, t);
                float q_abs[KV_LORA];
                matvec(wkb + (size_t)lh * KV_LORA * NOPE, qh, q_abs, KV_LORA, NOPE);
                float logits[T + 1], maxl = -INFINITY;
                for (int tk = 0; tk <= t; ++tk) {
                    float s = 0;
                    for (int i = 0; i < KV_LORA; ++i) s += q_abs[i] * klat[il][tk][i];
                    for (int i = 0; i < ROT; ++i) s += qh[NOPE + i] * kpe[il][tk][i];
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
                    for (int i = 0; i < KV_LORA; ++i) vlat[i] += p * klat[il][tk][i];
                }
                matvec(wvb + (size_t)lh * VD * KV_LORA, vlat,
                       attn_all + (size_t)lh * VD, VD, KV_LORA);
            }
            /* gate slice: gate rows [h*VD .. h*VD+VD) for this rank's heads */
            float *gatev = malloc((size_t)RANK_HEADS * VD * 4);
            for (int lh = 0; lh < RANK_HEADS; ++lh)
                /* attn_gate is HEAD-SLICED in the shard: local row index */
                matvec(wgate + (size_t)lh * VD * N_EMBD, att_in,
                       gatev + (size_t)lh * VD, VD, N_EMBD);
            for (int i = 0; i < RANK_HEADS * VD; ++i)
                attn_all[i] *= 1.0f / (1.0f + expf(-gatev[i]));
            free(gatev);
            /* o_proj is head-SLICED: [RANK_HEADS*VD, N_EMBD] local rows;
             * the TP16 all-reduce sums the rank partials */
            float abranch[N_EMBD];
            matvec(wo, attn_all, abranch, N_EMBD, RANK_HEADS * VD);
            hc_distribute(streams[t], abranch, post);


            float fpre[HC], fpost[HC];
            hc_mix(hc_ffn_fn2, streams[t], hc_ffn_sc2, hc_ffn_ba2, fpre, fpost);
            float fcur[N_EMBD];
            hc_reduce(streams[t], fpre, fcur);
            rms_norm(fcur, ln2v, fcur, N_EMBD, 1e-5f);

            if (il == 0) {
                /* dense layer: intermediate 18432 (not the MoE 2048) */
                float *dg = loadt(rank, (snprintf(nm, 160, "blk.0.ffn_gate.weight"), nm));
                float *du = loadt(rank, (snprintf(nm, 160, "blk.0.ffn_up.weight"), nm));
                float *dd = loadt(rank, (snprintf(nm, 160, "blk.0.ffn_down.weight"), nm));
                if (!dg || !du || !dd) return 1;
                float g[18432], u[18432], h[N_EMBD];
                matvec(dg, fcur, g, 18432, N_EMBD);
                matvec(du, fcur, u, 18432, N_EMBD);
                for (int j = 0; j < 18432; ++j) u[j] = silu(g[j]) * u[j];
                matvec(dd, u, h, N_EMBD, 18432);
                hc_distribute(streams[t], h, fpost);
                free(dg); free(du); free(dd);
            } else {
                float *ginp = loadt(rank, (snprintf(nm, 160, "blk.%d.ffn_gate_inp.weight", il), nm));
                float *eb = loadt(rank, (snprintf(nm, 160, "blk.%d.exp_probs_b.bias", il), nm));
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
                float wsum = 0;
                for (int k = 0; k < N_USED; ++k) wsum += ww[k];
                if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
                for (int k = 0; k < N_USED; ++k) ww[k] = ww[k] / wsum * SCALE;
                float ffn[N_EMBD];
                memset(ffn, 0, sizeof(ffn));
                float *gw = malloc((size_t)N_FF * N_EMBD * 4);
                float *uw = malloc((size_t)N_FF * N_EMBD * 4);
                float *dw2 = malloc((size_t)N_EMBD * N_FF * 4);
                for (int k = 0; k < N_USED; ++k) {
                    int li = sel[k] - OWN_LO;
                    if (li < 0 || li >= 16) continue;
                    hy4_tensor_view sg;
                    long sgb = 0, sub = 0, sdb = 0;
                    const hy4_tensor_view *tvg = hy4_tensor_lookup(rank, (snprintf(nm, 160, "blk.%d.ffn_gate_exps.weight", il), nm));
                    const hy4_tensor_view *tvu = hy4_tensor_lookup(rank, (snprintf(nm, 160, "blk.%d.ffn_up_exps.weight", il), nm));
                    const hy4_tensor_view *tvd = hy4_tensor_lookup(rank, (snprintf(nm, 160, "blk.%d.ffn_down_exps.weight", il), nm));
                    sgb = tvg->nbytes / 16; sub = tvu->nbytes / 16; sdb = tvd->nbytes / 16;
                    sg = *tvg; sg.file_offset += li * sgb; sg.nbytes = sgb;
                    sg.n_dims = 2; sg.dims[0] = N_FF; sg.dims[1] = N_EMBD;
                    if (dequant_view(rank, &sg, gw)) return 1;
                    sg = *tvu; sg.file_offset += li * sub; sg.nbytes = sub;
                    sg.n_dims = 2; sg.dims[0] = N_FF; sg.dims[1] = N_EMBD;
                    if (dequant_view(rank, &sg, uw)) return 1;
                    sg = *tvd; sg.file_offset += li * sdb; sg.nbytes = sdb;
                    sg.n_dims = 2; sg.dims[0] = N_EMBD; sg.dims[1] = N_FF;
                    if (dequant_view(rank, &sg, dw2)) return 1;
                    float g[N_FF], u[N_FF], h[N_EMBD];
                    matvec(gw, fcur, g, N_FF, N_EMBD);
                    matvec(uw, fcur, u, N_FF, N_EMBD);
                    for (int j = 0; j < N_FF; ++j) {
                        if (u[j] > 10.0f) u[j] = 10.0f; if (u[j] < -10.0f) u[j] = -10.0f;
                        float ga = silu(g[j]);
                        if (ga > 10.0f) ga = 10.0f;
                        u[j] = ga * u[j];
                    }
                    matvec(dw2, u, h, N_EMBD, N_FF);
                    for (int i = 0; i < N_EMBD; ++i) ffn[i] += ww[k] * h[i];
                }
                float *shg = loadt(rank, (snprintf(nm, 160, "blk.%d.ffn_gate_shexp.weight", il), nm));
                float *shu = loadt(rank, (snprintf(nm, 160, "blk.%d.ffn_up_shexp.weight", il), nm));
                float *shd = loadt(rank, (snprintf(nm, 160, "blk.%d.ffn_down_shexp.weight", il), nm));
                float g[N_FF], u[N_FF], h[N_EMBD];
                matvec(shg, fcur, g, N_FF, N_EMBD);
                matvec(shu, fcur, u, N_FF, N_EMBD);
                for (int j = 0; j < N_FF; ++j) u[j] = silu(g[j]) * u[j];
                matvec(shd, u, h, N_EMBD, N_FF);
                for (int i = 0; i < N_EMBD; ++i) ffn[i] += h[i];
                free(ginp); free(eb); free(gw); free(uw); free(dw2);
                free(shg); free(shu); free(shd);
                hc_distribute(streams[t], ffn, fpost);
            }
        }
        /* layer-scoped frees: the weights serve ALL tokens of this layer */
        free(hc_attn_fn); free(hc_attn_sc); free(hc_attn_ba); free(ln1);
        free(hc_ffn_fn2); free(hc_ffn_sc2); free(hc_ffn_ba2); free(ln2v);
        free(wq_a); free(qan); free(wq_b); free(wkv_a); free(kvan);
        free(wkb); free(wvb); free(wgate); free(wo); free(sinks);
        fprintf(stderr, "layer %d done\n", il);
    }

    /* final: hc_head collapse of the last token's streams, norm, vocab slice */
    float *hfn = loadt(rank, "output_hc_fn.weight");
    float *hsc = loadt(rank, "output_hc_scale.weight");
    float *hba = loadt(rank, "output_hc_base.weight");
    float *onorm = loadt(rank, "output_norm.weight");
    float *owl = loadt(rank, "output.weight");
    if (!hfn || !hsc || !hba || !onorm || !owl) return 1;
    float mixed[HC];
    matvec(hfn, streams[T - 1][0], mixed, HC, HC * N_EMBD);
    float pre[HC];
    for (int i = 0; i < HC; ++i)
        pre[i] = 1.0f / (1.0f + expf(-(mixed[i] * hsc[0] + hba[i]))) + 1e-6f;
    float collapsed[N_EMBD];
    hc_reduce(streams[T - 1][0], pre, collapsed);
    float normed[N_EMBD];
    rms_norm(collapsed, onorm, normed, N_EMBD, 1e-5f);
    float *lg = malloc((size_t)VOC_PER_RANK * 4);
    matvec(owl, normed, lg, VOC_PER_RANK, N_EMBD);
    int best = 0; float bv = -INFINITY;
    for (int v = 0; v < VOC_PER_RANK; ++v)
        if (lg[v] > bv) { bv = lg[v]; best = v; }
    printf("RANK-PARTIAL LAST-TOKEN LOGITS: argmax local id %d (global %d) score %.4f\n",
           best, best + MY_RANK * VOC_PER_RANK, bv);
    printf("STACK FORWARD DONE (rank-02 partial; TP16 all-reduce pending)\n");
    hy4_rank_close(rank);
    return 0;
}
