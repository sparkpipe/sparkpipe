#include "sparkpipe/spark_fixed_ring.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEGREE_MAX 16
#define ITERATIONS 50
#define WEIGHT_MS 25.0
#define COLLECTIVES_PER_TOKEN 90.0

static int degree_g = 16;
static int rank_g = 0;

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + ts.tv_nsec / 1000ull;
}

static uint16_t pattern_val(int r, int i)
{
    return (uint16_t)((uint32_t)(r * 251 + (i % 241) + 1) & 0x3fffu);
}

static uint16_t expected_sum(int i)
{
    int r;
    uint32_t total = 0;
    for (r = 0; r < degree_g; ++r)
        total += pattern_val(r, i);
    return (uint16_t)(total & 0xffffu);
}

int main(int argc, char **argv)
{
    SparkFixedRing *ring = 0;
    uint16_t *acc;
    const uint16_t *landing;
    uint32_t chunk_elems;
    uint32_t chunk_bytes;
    int i;
    int p;
    int iter;
    uint64_t total_us = 0;
    uint64_t min_us = (uint64_t)-1;
    int verify_fail = 0;
    SparkStatus status;
    if (argc < 5)
    {
        fprintf(stderr, "usage: mock_allreduce rank degree start broker_port\n");
        return 2;
    }
    rank_g = atoi(argv[1]);
    degree_g = atoi(argv[2]);
    if (degree_g != 4 && degree_g != 8 && degree_g != 16)
        return 2;
    setvbuf(stdout, 0, _IOLBF, 0);
    status = SparkFixedRingCreate((uint32_t)rank_g, (uint32_t)degree_g,
        (uint32_t)atoi(argv[3]), (uint32_t)atoi(argv[4]), &ring);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "rank %d: ring create -> %u\n", rank_g,
            (unsigned)status);
        return 1;
    }
    acc = SparkFixedRingAccumulator(ring);
    landing = (const uint16_t *)SparkFixedRingLanding(ring);
    chunk_elems = SparkFixedRingChunkElems(ring);
    chunk_bytes = SparkFixedRingChunkBytes(ring);
    SparkFixedRingSetChunkBytes(ring, chunk_bytes);
    for (iter = 0; iter < ITERATIONS; ++iter)
    {
        uint64_t t0;
        uint64_t us;
        for (i = 0; i < 4096; ++i)
            acc[i] = pattern_val(rank_g, i);
        t0 = now_us();
        for (p = 0; p < degree_g - 1; ++p)
        {
            uint32_t imm = (uint32_t)iter * 64u + (uint32_t)p;
            const uint16_t *src = (const uint16_t *)(
                (const uint8_t *)landing +
                (size_t)(imm & 15u) * (size_t)chunk_bytes);
            uint32_t send_index =
                (uint32_t)((rank_g - p + degree_g) % degree_g);
            uint32_t recv_index =
                (uint32_t)((rank_g - p - 1 + degree_g) % degree_g);
            uint16_t *dst = acc + (size_t)recv_index * chunk_elems;
            status = SparkFixedRingSendNext(ring,
                (const uint8_t *)acc + (size_t)send_index * chunk_bytes,
                chunk_bytes, imm);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr, "rank %d send fail p=%d status=%u\n",
                    rank_g, p, (unsigned)status);
                return 1;
            }
            status = SparkFixedRingWaitPrev(ring, imm, 2500000000ull);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr, "rank %d wait fail p=%d status=%u\n",
                    rank_g, p, (unsigned)status);
                return 1;
            }
            for (i = 0; i < (int)chunk_elems; ++i)
                dst[i] = (uint16_t)(dst[i] + src[i]);
        }
        for (p = 0; p < degree_g - 1; ++p)
        {
            uint32_t imm = (uint32_t)iter * 64u +
                (uint32_t)(degree_g - 1) + (uint32_t)p;
            const uint16_t *src = (const uint16_t *)(
                (const uint8_t *)landing +
                (size_t)(imm & 15u) * (size_t)chunk_bytes);
            uint32_t send_index =
                (uint32_t)((rank_g + 1 - p + degree_g) % degree_g);
            uint32_t recv_index =
                (uint32_t)((rank_g - p + degree_g) % degree_g);
            uint16_t *dst = acc + (size_t)recv_index * chunk_elems;
            status = SparkFixedRingSendNext(ring,
                (const uint8_t *)acc + (size_t)send_index * chunk_bytes,
                chunk_bytes, imm);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr, "rank %d send fail p=%d status=%u\n",
                    rank_g, p, (unsigned)status);
                return 1;
            }
            status = SparkFixedRingWaitPrev(ring, imm, 2500000000ull);
            if (status != SPARK_STATUS_OK)
            {
                fprintf(stderr, "rank %d wait fail p=%d status=%u\n",
                    rank_g, p, (unsigned)status);
                return 1;
            }
            memcpy(dst, src, chunk_bytes);
        }
        us = now_us() - t0;
        total_us += us;
        if (us < min_us)
            min_us = us;
        if ((iter % 10) == 9 || iter == ITERATIONS - 1)
        {
            for (i = 0; i < 4096; ++i)
            {
                if (acc[i] != expected_sum(i))
                {
                    fprintf(stderr,
                        "rank %d VERIFY FAIL iter=%d elem=%d got=%u expect=%u\n",
                        rank_g, iter, i, (unsigned)acc[i],
                        (unsigned)expected_sum(i));
                    ++verify_fail;
                    break;
                }
            }
        }
    }
    {
        double mean_us = (double)total_us / (double)ITERATIONS;
        double b1 = 1000000.0 /
            (WEIGHT_MS * 1000.0 + COLLECTIVES_PER_TOKEN * mean_us);
        printf("ALLREDUCE rank=%2d iters=%d mean_us=%.1f min_us=%.1f payload=%dB verify=%s\n",
            rank_g, ITERATIONS, mean_us, (double)min_us, 4096 * 2,
            verify_fail != 0 ? "FAIL" : "OK");
        printf("TOKS rank=%2d per_op_us=%.1f b1_tok_s=%.1f overlap_ceiling_tok_s=%.1f\n",
            rank_g, mean_us, b1, 1000.0 / WEIGHT_MS);
    }
    SparkFixedRingDestroy(ring);
    return verify_fail != 0 ? 1 : 0;
}
