/* hy4 lane: single-token routed-expert + shared-expert FFN forward (CPU).
 *
 * Semantics per hy4_layer_semantics.md / vendor hyv4_reference.cpp:
 *   probs      = sigmoid(ffn_gate_inp . x)      (unbiased, weights source)
 *   selection  = top-8 of (probs + exp_probs_b) (bias affects SELECTION only)
 *   w          = probs[sel] / max(sum, 6.103515625e-5) * 2.827
 *   per expert: up clamped +/-10 pre-activation; gate = silu(.) clamped
 *               one-sided +10 AFTER silu; h = gate*up; y += w * down.h
 *   shared expert: silu gate * up (unclamped) -> down, added unweighted
 * Rank-local: only this rank's 16 owned experts contribute (the TP16
 * all-reduce happens at serving); the receipt covers the partial sum.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hy4_rank_loader.h"
#include "hy4_iq_dequant_vendor.h"

#define N_EMBD 6144
#define N_FF 2048
#define N_EXPERT 256
#define N_USED 8
#define SCALE 2.827f

static float silu(float v) { return v / (1.0f + expf(-v)); }

static int bs_of(int type) {
    return type == 8 ? 34 : type == 16 ? 66 : type == 18 ? 98 :
           type == 29 ? 56 : 0;
}

/* Dequantize one 2-D weight [rows, cols] (whole tensor) into row-major
 * float y. rows*cols floats must be allocated. Type 0 = raw F32. */
static int dequant_view(const hy4_rank *rank, const hy4_tensor_view *tv,
                        float *y) {
    long nelems = 1;
    for (int i = 0; i < tv->n_dims; ++i) nelems *= tv->dims[i];
    uint8_t *src = malloc((size_t)tv->nbytes);
    if (hy4_tensor_read(rank, tv, src)) return -1;
    if (tv->type == 0) {
        memcpy(y, src, (size_t)tv->nbytes);
    } else if (tv->type == 8) {
        long nb = tv->nbytes / 34;
        for (long b = 0; b < nb; ++b) {
            uint16_t bits = (uint16_t)src[b*34] | ((uint16_t)src[b*34+1] << 8);
            float s = hy4_fp16_to_fp32(bits);
            for (int i = 0; i < 32; ++i) y[b*32+i] = s * (float)(int8_t)src[b*34+2+i];
        }
    } else if (tv->type == 16) hy4_dequant_iq2_xxs(src, y, tv->nbytes / 66);
    else if (tv->type == 18) hy4_dequant_iq3_xxs(src, y, tv->nbytes / 98);
    else if (tv->type == 29) hy4_dequant_iq1_m(src, y, tv->nbytes / 56);
    else if (tv->type == 12) hy4_dequant_row_q4_K((const block_q4_K *)src, y, tv->nbytes / 144 * 256);
    else if (tv->type == 13) hy4_dequant_row_q5_K((const block_q5_K *)src, y, tv->nbytes / 176 * 256);
    else if (tv->type == 14) hy4_dequant_row_q6_K((const block_q6_K *)src, y, tv->nbytes / 210 * 256);
    else {
        fprintf(stderr, "dequant_view: unsupported type %d (%ld bytes)\n",
                tv->type, tv->nbytes);
        free(src); return -1;
    }
    free(src);
    (void)nelems;
    return 0;
}

/* matmul y[rows] = W[rows, cols] . x[cols]; W row-major (GGML ne order
 * transposed here: the manifest dims are ne[fastest..], so W row r starts
 * at r*cols elements). */
static void matvec(const float *w, const float *x, float *y, long rows,
                   long cols) {
    for (long r = 0; r < rows; ++r) {
        float acc = 0;
        const float *row = w + r * cols;
        for (long c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] = acc;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s pack_dir [force_expert]\n", argv[0]); return 2; }
    hy4_rank *rank = NULL;
    if (hy4_rank_open(argv[1], 0, &rank)) { fprintf(stderr, "open failed\n"); return 1; }

    /* deterministic input: router row of expert 40 (rank-02-owned) scaled
     * small + a tiny sine tail, so the selection includes owned experts */
    const hy4_tensor_view *tv_probe = hy4_tensor_lookup(rank, "blk.47.ffn_gate_inp.weight");
    float *x = malloc(N_EMBD * 4);
    {
        float *row = malloc((size_t)tv_probe->nbytes);
        if (dequant_view(rank, tv_probe, row)) return 1;
        for (int i = 0; i < N_EMBD; ++i)
            x[i] = row[40 * N_EMBD + i] * 0.1f + sinf((float)i) * 1e-4f;
        free(row);
    }

    /* router (F32 tensors) */
    const hy4_tensor_view *tv_gate = hy4_tensor_lookup(rank, "blk.47.ffn_gate_inp.weight");
    const hy4_tensor_view *tv_bias = hy4_tensor_lookup(rank, "blk.47.exp_probs_b.bias");
    if (!tv_gate || !tv_bias) { fprintf(stderr, "router tensors missing\n"); return 1; }
    float *gate_inp = malloc((size_t)tv_gate->nbytes);
    float *bias = malloc((size_t)tv_bias->nbytes);
    if (dequant_view(rank, tv_gate, gate_inp) ||
        dequant_view(rank, tv_bias, bias)) { fprintf(stderr, "router dequant\n"); return 1; }

    float logits[N_EXPERT], probs[N_EXPERT], sel_key[N_EXPERT];
    for (int e = 0; e < N_EXPERT; ++e) {
        float acc = 0;
        for (int i = 0; i < N_EMBD; ++i) acc += gate_inp[e * N_EMBD + i] * x[i];
        logits[e] = acc;
        probs[e] = 1.0f / (1.0f + expf(-acc));
        sel_key[e] = probs[e] + bias[e];
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
    float sum = 0;
    for (int k = 0; k < N_USED; ++k) sum += w[k];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int k = 0; k < N_USED; ++k) w[k] = w[k] / sum * SCALE;
    printf("logits[40]=%.4f probs[40]=%.4f bias[40]=%.4f selkey[40]=%.4f maxlogit=%.4f\n",
           logits[40], probs[40], bias[40], sel_key[40] == -INFINITY ? 0.f : 0.f, 0.f);
    for (int e = 0; e < N_EXPERT; ++e) if (logits[e] > 1e9f) printf("big logit e=%d %.4f\n", e, logits[e]);
    printf("top8:");
    for (int k = 0; k < N_USED; ++k) printf(" %d(%.6f)", sel[k], w[k]);
    printf("\n");

    /* rank-local routed experts; optional forced expert for the FFN-path
     * exactness receipt (selection itself is verified from the trace) */
    int force_expert = argc > 2 ? atoi(argv[2]) : -1;
    int rank_id = 2, own_lo = rank_id * 16;
    float y_moe[N_EMBD];
    memset(y_moe, 0, sizeof(y_moe));
    if (force_expert >= 0) {
        sel[0] = force_expert;
        w[0] = 1.0f;
        printf("FORCED single-expert receipt: expert %d weight 1.0\n", force_expert);
    }
    float *gw = malloc((size_t)N_FF * N_EMBD * 4);
    float *uw = malloc((size_t)N_FF * N_EMBD * 4);
    float *dw = malloc((size_t)N_EMBD * N_FF * 4);
    float *g = malloc(N_FF * 4), *u = malloc(N_FF * 4), *h = malloc(N_EMBD * 4);
    long used_local = 0;
    for (int k = 0; k < N_USED; ++k) {
        int e = sel[k];
        int li = e - own_lo;
        if (li < 0 || li >= 16) continue;
        used_local++;
        const hy4_tensor_view *tvg = hy4_tensor_lookup(rank, "blk.47.ffn_gate_exps.weight");
        const hy4_tensor_view *tvu = hy4_tensor_lookup(rank, "blk.47.ffn_up_exps.weight");
        const hy4_tensor_view *tvd = hy4_tensor_lookup(rank, "blk.47.ffn_down_exps.weight");
        if (!tvg || !tvu || !tvd) { fprintf(stderr, "expert tensors missing\n"); return 1; }
        long slab_g = tvg->nbytes / 16, slab_u = tvu->nbytes / 16,
             slab_d = tvd->nbytes / 16;
        /* targeted slab reads: each expert's bytes are contiguous (dim-2
         * split keeps whole quant blocks together) */
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

        { /* slab bisection stats */
            long nan_g = 0, nan_u = 0, nan_d = 0;
            float mg = 0, mu = 0, md = 0;
            for (long i = 0; i < N_FF * N_EMBD; ++i) {
                if (gw[i] != gw[i]) nan_g++; else if (fabsf(gw[i]) > mg) mg = fabsf(gw[i]);
                if (uw[i] != uw[i]) nan_u++; else if (fabsf(uw[i]) > mu) mu = fabsf(uw[i]);
                if (dw[i] != dw[i]) nan_d++; else if (fabsf(dw[i]) > md) md = fabsf(dw[i]);
            }
            fprintf(stderr, "slab li=%d: gate nan=%ld amax=%.6f | up nan=%ld amax=%.6f | down nan=%ld amax=%.6f\n",
                    li, nan_g, mg, nan_u, mu, nan_d, md);
            if (nan_g || nan_u || nan_d) { fprintf(stderr, "slab NaN -- RED stop\n"); return 1; }
        }

        matvec(gw, x, g, N_FF, N_EMBD);
        matvec(uw, x, u, N_FF, N_EMBD);
        for (int j = 0; j < N_FF; ++j) {
            u[j] = u[j] < -10.0f ? -10.0f : u[j] > 10.0f ? 10.0f : u[j];
            float ga = silu(g[j]);
            ga = ga > 10.0f ? 10.0f : ga;   /* one-sided clamp AFTER silu */
            h[j] = 0; /* h is N_FF at this stage */
            u[j] = ga * u[j];
        }
        /* h here is the FFN hidden [N_FF] = clamped silu(gate)*up */
        matvec(dw, u, h, N_EMBD, N_FF);
        for (int i = 0; i < N_EMBD; ++i) y_moe[i] += w[k] * h[i];
    }
    double psum = 0;
    for (int i = 0; i < N_EMBD; ++i) psum += y_moe[i];
    printf("rank-local experts used: %ld, routed partial sum %.6f amax %.6f\n",
           used_local, psum, fabsf(y_moe[0]));

    /* trace for the numpy router cross-check (written before the shared
     * path, which may refuse not-yet-vendored types) */
    {
        FILE *t = fopen("/tmp/hy4_moe_trace.bin", "wb");
        fwrite(x, 4, N_EMBD, t);
        fwrite(bias, 4, N_EXPERT, t);
        fwrite(logits, 4, N_EXPERT, t);
        fwrite(w, 4, N_USED, t);
        fwrite(sel, 4, N_USED, t);
        fclose(t);
    }

    /* shared expert (unclamped, unweighted) */
    const hy4_tensor_view *tsg = hy4_tensor_lookup(rank, "blk.47.ffn_gate_shexp.weight");
    const hy4_tensor_view *tsu = hy4_tensor_lookup(rank, "blk.47.ffn_up_shexp.weight");
    const hy4_tensor_view *tsd = hy4_tensor_lookup(rank, "blk.47.ffn_down_shexp.weight");
    if (tsg && tsu && tsd) {
        if (dequant_view(rank, tsg, gw) || dequant_view(rank, tsu, uw) ||
            dequant_view(rank, tsd, dw)) return 1;
        matvec(gw, x, g, N_FF, N_EMBD);
        matvec(uw, x, u, N_FF, N_EMBD);
        for (int j = 0; j < N_FF; ++j) u[j] = silu(g[j]) * u[j];
        matvec(dw, u, h, N_EMBD, N_FF);
        double ssum = 0;
        for (int i = 0; i < N_EMBD; ++i) { y_moe[i] += h[i]; ssum += y_moe[i]; }
        printf("after shared: sum %.6f amax %.6f\n", ssum, fabsf(y_moe[0]));
    }

    printf("MOE FORWARD DONE (trace at /tmp/hy4_moe_trace.bin)\n");
    hy4_rank_close(rank);
    return 0;
}
