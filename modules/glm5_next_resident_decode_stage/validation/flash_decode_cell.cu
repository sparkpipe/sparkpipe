/* Flash decode qualification cells (single-node, sm_121a) — main@e83851f.
 *
 * Cell A - R1 screened head: at B1 the certified-FP8 path must emit the
 *          SAME (token, score) as the full-vocab head (the certified
 *          bound guarantees the argmax is inside the candidate set; the
 *          rescore is exact BF16). Deployment shard (vocab/16) and full
 *          vocab, 3 seeds, plus per-call timing.
 * Cell B - R3 split-K attention at flash geometry (Glm5NextKv paged pool,
 *          LATENT 512 / ROPE 0, 4 heads/rank at TP16, 256 threads): the
 *          below-threshold launch must be byte-identical to the direct
 *          single-pass kernel; the above-threshold split+combine must be
 *          deterministic and match within bf16 rounding tolerance.
 *          8K and 32K positions, plus timing.
 *
 * Exit 0 only if every property holds. Receipts print as CELL lines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "modules/glm5_next_resident_decode_stage/source/cuda/unity.cu"

#define FULL_VOCAB 154880u
#define ATTN_HEADS 4u          /* per-rank at TP16: the 4-CTA grid */
#define QK_SCALE 0.0625f

static cudaError_t s_last;
#define CU(x) do { s_last = (x); if (s_last != cudaSuccess) { \
    printf("CELL FAIL cuda %s at %d\n", cudaGetErrorString(s_last), __LINE__); \
    exit(2); } } while (0)

static void fill_bf16(uint16_t *device, uint64_t n, unsigned seed)
{
    uint16_t *host = n != 0ull ? (uint16_t *)malloc(n * 2u) : 0;
    if (host == 0 && n != 0ull) { printf("CELL FAIL host alloc\n"); exit(2); }
    srand(seed);
    for (uint64_t i = 0; i < n; i++)
    {
        float v = ((float)(rand() % 20001) - 10000.0f) / 3000.0f;
        __nv_bfloat16 h = __float2bfloat16(v);
        host[i] = *(uint16_t *)&h;
    }
    CU(cudaMemcpy(device, host, n * 2u, cudaMemcpyHostToDevice));
    free(host);
}

/* ---------- Cell A: certified-FP8 head vs full-vocab head ---------- */

static int cell_head(uint32_t vocab, unsigned seed)
{
    const uint64_t dim = GLM5_NEXT_HIDDEN;
    uint16_t *head = 0, *norm_w = 0, *hc_mean = 0, *residual = 0, *normed = 0;
    float *cand_score = 0; uint32_t *cand_token = 0;
    uint32_t *out_tok_full = 0, *out_tok_cert = 0;
    float *out_sc_full = 0, *out_sc_cert = 0;
    uint8_t *payload = 0; float *scale = 0, *cnorm = 0;
    void *scratch = 0; uint32_t *cand_ids = 0; uint32_t *screened = 0;
    Glm5NextLayerBuffers b; cudaStream_t stream;
    const uint32_t tiles = (vocab + GLM5_NEXT_HEAD_TILE - 1u) / GLM5_NEXT_HEAD_TILE;
    uint32_t fail = 0;
    cudaEvent_t t0, t1;
    float ms_full = 0.0f, ms_cert = 0.0f;

    memset(&b, 0, sizeof(b));
    CU(cudaStreamCreate(&stream));
    CU(cudaMalloc(&head, (uint64_t)vocab * dim * 2u));
    CU(cudaMalloc(&norm_w, dim * 2u));
    CU(cudaMalloc(&hc_mean, dim * 2u));
    CU(cudaMalloc(&residual, dim * 2u));
    CU(cudaMalloc(&normed, dim * 2u));
    CU(cudaMalloc(&cand_score, (uint64_t)tiles * 4u * sizeof(float)));
    CU(cudaMalloc(&cand_token, (uint64_t)tiles * 4u * sizeof(uint32_t)));
    CU(cudaMalloc(&out_tok_full, 4u * sizeof(uint32_t)));
    CU(cudaMalloc(&out_tok_cert, 4u * sizeof(uint32_t)));
    CU(cudaMalloc(&out_sc_full, 4u * sizeof(float)));
    CU(cudaMalloc(&out_sc_cert, 4u * sizeof(float)));
    CU(cudaMalloc(&payload, (uint64_t)vocab * dim));
    CU(cudaMalloc(&scale, (uint64_t)vocab * (dim / 32u) * sizeof(float)));
    CU(cudaMalloc(&cnorm, (uint64_t)vocab * (dim / 32u) * sizeof(float)));
    CU(cudaMalloc(&scratch, SparkHeadCertifiedFp8ScratchBytes(vocab, dim)));
    CU(cudaMalloc(&cand_ids, SparkHeadCertifiedFp8CandidateBytes(vocab)));
    CU(cudaMalloc(&screened, sizeof(uint32_t)));

    fill_bf16(head, (uint64_t)vocab * dim, seed);
    fill_bf16(norm_w, dim, seed + 1u);
    fill_bf16(hc_mean, dim, seed + 2u);
    CU(cudaMemset(residual, 0, dim * 2u));

    CU(SparkGlm5NextLaunchHeadCertifiedQuantize(0, head, payload, scale,
        cnorm, vocab, (uint32_t)dim));
    CU(cudaDeviceSynchronize());

    b.hidden_bf16 = residual; b.residual_bf16 = residual;
    b.normed_bf16 = normed; b.hc_mean_bf16 = hc_mean;
    b.head_candidate_score = cand_score; b.head_candidate_token = cand_token;
    b.head_vocabulary = vocab;
    CU(cudaEventCreate(&t0)); CU(cudaEventCreate(&t1));

    b.output_token = out_tok_full; b.output_score = out_sc_full;
    if (Glm5NextHeadFullVocab(&b, norm_w, head, 1u, stream) != LM_LAUNCH_OK)
    { printf("CELL FAIL fullvocab launch\n"); return 3; }
    CU(cudaStreamSynchronize(stream));
    CU(cudaEventRecord(t0, stream));
    for (int r = 0; r < 20; r++)
        (void)Glm5NextHeadFullVocab(&b, norm_w, head, 1u, stream);
    CU(cudaEventRecord(t1, stream)); CU(cudaEventSynchronize(t1));
    CU(cudaEventElapsedTime(&ms_full, t0, t1)); ms_full /= 20.0f;
    CU(cudaStreamSynchronize(stream));

    b.output_token = out_tok_cert; b.output_score = out_sc_cert;
    if (Glm5NextHeadCertifiedB1(&b, norm_w, head, payload, scale, cnorm,
        scratch, cand_ids, screened, 0u, vocab, stream) != LM_LAUNCH_OK)
    { printf("CELL FAIL certified launch\n"); return 3; }
    CU(cudaStreamSynchronize(stream));
    CU(cudaEventRecord(t0, stream));
    for (int r = 0; r < 20; r++)
        (void)Glm5NextHeadCertifiedB1(&b, norm_w, head, payload, scale,
            cnorm, scratch, cand_ids, screened, 0u, vocab, stream);
    CU(cudaEventRecord(t1, stream)); CU(cudaEventSynchronize(t1));
    CU(cudaEventElapsedTime(&ms_cert, t0, t1)); ms_cert /= 20.0f;
    CU(cudaStreamSynchronize(stream));

    uint32_t tok_f = 0, tok_c = 0; float sc_f = 0.0f, sc_c = 0.0f, n_screened = 0.0f;
    CU(cudaMemcpy(&tok_f, out_tok_full, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&tok_c, out_tok_cert, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&sc_f, out_sc_full, sizeof(float), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&sc_c, out_sc_cert, sizeof(float), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&n_screened, screened, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (tok_f != tok_c) fail = 1;
    if (fabsf(sc_f - sc_c) > 1e-3f * (fabsf(sc_f) + 1e-6f)) fail |= 2;

    printf("CELL A %s vocab=%u seed=%u token full=%u cert=%u score full=%.4f cert=%.4f screened=%.0f | full %.3f ms cert %.3f ms (%.2fx)\n",
        fail ? "FAIL" : "PASS", vocab, seed, tok_f, tok_c, sc_f, sc_c,
        n_screened, ms_full, ms_cert, ms_cert > 0.0f ? ms_full / ms_cert : 0.0f);

    cudaFree(head); cudaFree(norm_w); cudaFree(hc_mean); cudaFree(residual);
    cudaFree(normed); cudaFree(cand_score); cudaFree(cand_token);
    cudaFree(out_tok_full); cudaFree(out_tok_cert); cudaFree(out_sc_full);
    cudaFree(out_sc_cert); cudaFree(payload); cudaFree(scale); cudaFree(cnorm);
    cudaFree(scratch); cudaFree(cand_ids); cudaFree(screened);
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaStreamDestroy(stream);
    return fail ? 1 : 0;
}

/* ---------- Cell B: split-K attention vs single-pass, flash geometry ---------- */

static int cell_attn(uint32_t positions)
{
    using Kv = Glm5NextKv;
    const uint32_t heads = ATTN_HEADS, threads = GLM5_NEXT_ATTN_THREADS;
    const uint32_t latent = GLM5_NEXT_LATENT, rope = GLM5_NEXT_ROPE_DIM;
    const uint32_t rows = 1u;
    const uint64_t pages = Kv::PagesForTokens(positions);
    uint16_t *q_lat = 0, *q_rope = 0;
    uint16_t *out_base = 0, *out_split = 0, *out_off = 0, *out_again = 0;
    uint8_t *pool = 0; uint32_t *page_table = 0, *seqof = 0, *ctx = 0, *pos = 0;
    LmKvAccessError *access_error = 0;
    float *partials = 0;
    LmKvView cache; cudaStream_t stream; cudaEvent_t t0, t1;
    uint32_t fail = 0;
    float ms_base = 0.0f, ms_split = 0.0f;
    const uint64_t partial_floats =
        (uint64_t)heads * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS *
        LM_LATENT_ATTN_SPLIT_BLOCK_FLOATS(latent);

    memset(&cache, 0, sizeof(cache));
    CU(cudaStreamCreate(&stream));
    CU(cudaMalloc(&q_lat, (uint64_t)heads * latent * 2u));
    CU(cudaMalloc(&q_rope, rope != 0u ? (uint64_t)heads * rope * 2u : 2u));
    CU(cudaMalloc(&pool, Kv::PoolBytes(pages)));
    CU(cudaMalloc(&page_table, pages * sizeof(uint32_t)));
    CU(cudaMalloc(&access_error, sizeof(LmKvAccessError)));
    CU(cudaMalloc(&seqof, sizeof(uint32_t)));
    CU(cudaMalloc(&ctx, sizeof(uint32_t)));
    CU(cudaMalloc(&pos, sizeof(uint32_t)));
    CU(cudaMalloc(&out_base, (uint64_t)heads * latent * 2u));
    CU(cudaMalloc(&out_split, (uint64_t)heads * latent * 2u));
    CU(cudaMalloc(&out_off, (uint64_t)heads * latent * 2u));
    CU(cudaMalloc(&out_again, (uint64_t)heads * latent * 2u));
    CU(cudaMalloc(&partials, partial_floats * sizeof(float)));

    /* identity page table: sequence 0's page i lives at pool page i */
    {
        uint32_t *host_pages = (uint32_t *)malloc(pages * sizeof(uint32_t));
        for (uint64_t i = 0; i < pages; i++) host_pages[i] = (uint32_t)i;
        CU(cudaMemcpy(page_table, host_pages, pages * sizeof(uint32_t), cudaMemcpyHostToDevice));
        free(host_pages);
    }
    fill_bf16(q_lat, (uint64_t)heads * latent, 77u);
    fill_bf16((uint16_t *)pool,
        Kv::PoolBytes(pages) / 2u > 4000000u ? 4000000u : Kv::PoolBytes(pages) / 2u, 79u);
    {
        uint32_t zero = 0u, ctx_v = positions;
        CU(cudaMemcpy(seqof, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(ctx, &ctx_v, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(pos, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        LmKvAccessError reset; LmKvAccessErrorReset(&reset);
        CU(cudaMemcpy(access_error, &reset, sizeof(LmKvAccessError), cudaMemcpyHostToDevice));
    }
    cache.pool = pool; cache.page_table = page_table;
    cache.page_table_stride = (uint32_t)pages; cache.sequence_count = 1u;
    cache.pool_page_count = (uint32_t)pages; cache.access_error = access_error;

    CU(cudaEventCreate(&t0)); CU(cudaEventCreate(&t1));

    /* reference: the direct single-pass kernel (the below-threshold path
     * launches this same kernel) */
    LmLatentAttentionDecodeKernel<Kv, threads, latent, rope>
        <<<dim3(rows, heads), threads, 0, stream>>>
        (q_lat, q_rope, cache, seqof, ctx, 0, 0u, heads, QK_SCALE, out_base, pos);
    CU(cudaGetLastError());
    CU(cudaStreamSynchronize(stream));
    CU(cudaEventRecord(t0, stream));
    for (int r = 0; r < 10; r++)
        LmLatentAttentionDecodeKernel<Kv, threads, latent, rope>
            <<<dim3(rows, heads), threads, 0, stream>>>
            (q_lat, q_rope, cache, seqof, ctx, 0, 0u, heads, QK_SCALE, out_base, pos);
    CU(cudaEventRecord(t1, stream)); CU(cudaEventSynchronize(t1));
    CU(cudaEventElapsedTime(&ms_base, t0, t1)); ms_base /= 10.0f;

    /* below threshold (0): must be BYTE-identical to the direct launch */
    CU((LmLatentAttentionDecodeSplitLaunch<Kv, threads, latent, rope>(
        q_lat, q_rope, cache, seqof, ctx, 0, 0u, heads, QK_SCALE, out_off,
        pos, rows, positions, 0u, partials, (uint32_t)partial_floats, 48u, stream)));
    CU(cudaStreamSynchronize(stream));

    /* above threshold (1): split+combine, twice (determinism), timed */
    CU((LmLatentAttentionDecodeSplitLaunch<Kv, threads, latent, rope>(
        q_lat, q_rope, cache, seqof, ctx, 0, 0u, heads, QK_SCALE, out_split,
        pos, rows, positions, 1u, partials, (uint32_t)partial_floats, 48u, stream)));
    CU(cudaStreamSynchronize(stream));
    CU(cudaEventRecord(t0, stream));
    for (int r = 0; r < 10; r++)
        (void)LmLatentAttentionDecodeSplitLaunch<Kv, threads, latent, rope>(
            q_lat, q_rope, cache, seqof, ctx, 0, 0u, heads, QK_SCALE,
            out_split, pos, rows, positions, 1u, partials,
            (uint32_t)partial_floats, 48u, stream);
    CU(cudaEventRecord(t1, stream)); CU(cudaEventSynchronize(t1));
    CU(cudaEventElapsedTime(&ms_split, t0, t1)); ms_split /= 10.0f;
    CU((LmLatentAttentionDecodeSplitLaunch<Kv, threads, latent, rope>(
        q_lat, q_rope, cache, seqof, ctx, 0, 0u, heads, QK_SCALE, out_again,
        pos, rows, positions, 1u, partials, (uint32_t)partial_floats, 48u, stream)));
    CU(cudaStreamSynchronize(stream));

    const uint64_t out_elems = (uint64_t)heads * latent;
    uint16_t *hb = (uint16_t *)malloc(out_elems * 2u);
    uint16_t *hs = (uint16_t *)malloc(out_elems * 2u);
    uint16_t *ho = (uint16_t *)malloc(out_elems * 2u);
    uint16_t *ha = (uint16_t *)malloc(out_elems * 2u);
    CU(cudaMemcpy(hb, out_base, out_elems * 2u, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(hs, out_split, out_elems * 2u, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(ho, out_off, out_elems * 2u, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(ha, out_again, out_elems * 2u, cudaMemcpyDeviceToHost));

    if (memcmp(ho, hb, out_elems * 2u) != 0) fail |= 1;
    if (memcmp(hs, ha, out_elems * 2u) != 0) fail |= 2;
    double max_rel = 0.0;
    for (uint64_t i = 0; i < out_elems; i++)
    {
        float a = __bfloat162float(*(const __nv_bfloat16 *)&hb[i]);
        float b = __bfloat162float(*(const __nv_bfloat16 *)&hs[i]);
        double rel = fabs((double)a - (double)b) / (fabs((double)a) + 1e-3);
        if (rel > max_rel) max_rel = rel;
    }
    if (max_rel > 3e-2) fail |= 4;

    printf("CELL B %s positions=%u pages=%llu below-threshold-byte-identical=%s deterministic=%s max_rel=%.5f | base %.3f ms split %.3f ms (%.2fx)\n",
        fail ? "FAIL" : "PASS", positions, (unsigned long long)pages,
        (fail & 1) ? "NO" : "yes", (fail & 2) ? "NO" : "yes", max_rel,
        ms_base, ms_split, ms_split > 0.0f ? ms_base / ms_split : 0.0f);

    free(hb); free(hs); free(ho); free(ha);
    cudaFree(q_lat); cudaFree(q_rope); cudaFree(pool); cudaFree(page_table);
    cudaFree(access_error); cudaFree(seqof); cudaFree(ctx); cudaFree(pos);
    cudaFree(out_base); cudaFree(out_split); cudaFree(out_off);
    cudaFree(out_again); cudaFree(partials);
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaStreamDestroy(stream);
    return fail ? 1 : 0;
}

int main(void)
{
    int failures = 0;
    cudaDeviceProp prop;
    CU(cudaGetDeviceProperties(&prop, 0));
    printf("CELL device=%s sms=%d\n", prop.name, prop.multiProcessorCount);

    for (unsigned seed = 1u; seed <= 3u; seed++)
    {
        failures += cell_head(FULL_VOCAB / 16u, seed);
        failures += cell_head(FULL_VOCAB, seed);
    }
    failures += cell_attn(8192u);
    failures += cell_attn(32768u);

    printf("CELL SUMMARY %s (%d failing properties)\n",
        failures == 0 ? "ALL PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
