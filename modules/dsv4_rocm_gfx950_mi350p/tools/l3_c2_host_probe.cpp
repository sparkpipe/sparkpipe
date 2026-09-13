// l3_c2_host_probe.cpp - EXECUTES the C2 integer surfaces of the island TU
// on the host. Built with -DSPARK_DSV4_ROCM_HOST_EXECUTOR, the coordinate
// shims expose distinct grid/block/thread globals; because every executed
// kernel assigns each element wholly to one thread via blockDim-strided
// loops, a single-thread walk with the REAL blockDim reproduces the device
// final state bit for bit. Cross-lane-reduction kernels (W13/down/reduce/
// emission post) are executor-excluded by design; their float trees are
// covered by the encoder/equivalence probes and, later, hardware.
//
// Ran-where: authoring workstation, plain C++. Proves algorithm semantics
// (grouping canonicality, ring-slot addressing, page init/commits,
// compressor boundaries/counters/state), NOT device codegen.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "spark_dsv4_rocm_islands.hip"

// Definitions for the executor coordinate globals the shim declares.
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_grid{1u, 1u, 1u};
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_block{1u, 1u, 1u};
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_threadblock{1u, 1u, 1u};
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_thread{0u, 0u, 1u};

// Stray references from unused entry bodies never run here; the stubs keep
// the link honest if one is ever emitted.
hipError_t hipLaunchKernel(const void *, dim3, dim3, void **, size_t,
                           hipStream_t)
{
    std::fprintf(stderr, "executor: hipLaunchKernel must not be called\n");
    std::abort();
}

hipError_t hipEventRecord(hipEvent_t, hipStream_t)
{
    std::abort();
}

hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int)
{
    std::abort();
}

SparkHwStatus spark_hw_read_ahead(SparkHwQueue, void *, uint32_t, void *)
{
    std::abort();
}

namespace {

uint32_t g_failures = 0u;

// bf16 word back to float for expectations (mirror of the TU decoder).
float DecodeProbe(uint16_t raw)
{
    union { uint32_t b; float f; } u;
    u.b = ((uint32_t)raw) << 16u;
    return u.f;
}

void ExpectU32(const char *what, uint64_t index, uint32_t got, uint32_t want)
{
    if (got != want)
    {
        ++g_failures;
        std::printf("FAIL %s[%llu]: got %u want %u\n", what,
                    (unsigned long long)index, got, want);
    }
}

void ExpectF32Bits(const char *what, uint64_t index, float got, float want)
{
    uint32_t gb, wb;
    std::memcpy(&gb, &got, 4u);
    std::memcpy(&wb, &want, 4u);
    if (gb != wb)
    {
        ++g_failures;
        std::printf("FAIL %s[%llu]: got %.9g (0x%08x) want %.9g (0x%08x)\n",
                    what, (unsigned long long)index, got, gb, want, wb);
    }
}

// Runs one kernel launch under the executor walk: every block, single
// in-order thread, blockDim 1 so every strided loop is walked fully.
template <typename Kernel>
void Execute(Kernel kernel, uint32_t grid_x, uint32_t grid_y)
{
    spark_dsv4_rocm_exec_grid.x = grid_x;
    spark_dsv4_rocm_exec_grid.y = grid_y;
    spark_dsv4_rocm_exec_grid.z = 1u;
    for (uint32_t by = 0u; by < grid_y; ++by)
    {
        for (uint32_t bx = 0u; bx < grid_x; ++bx)
        {
            spark_dsv4_rocm_exec_block.x = bx;
            spark_dsv4_rocm_exec_block.y = by;
            spark_dsv4_rocm_exec_block.z = 1u;
            spark_dsv4_rocm_exec_thread.x = 0u;
            spark_dsv4_rocm_exec_thread.y = 0u;
            spark_dsv4_rocm_exec_thread.z = 1u;
            kernel();
        }
    }
}

// --- Test 1: RouteGroupKernel grouping canonicality (THE L4 C2 surface). ---
void TestRouteGroup()
{
    const uint32_t rows = 8u, k = 3u, experts = 6u, pairs = rows * k;
    const uint32_t routes[24] = {
        0u, 2u, 4u,  1u, 1u, 3u,  5u, 5u, 5u,  0u, 2u, 2u,
        4u, 4u, 4u,  1u, 3u, 3u,  2u, 2u, 2u,  0u, 0u, 5u};
    /* Hand-derived canonical result: exclusive prefix of per-expert counts
     * [4,3,6,3,4,4], within-expert packing in ascending flat-element
     * order. */
    const uint32_t want_offsets[7] = {0u, 4u, 7u, 13u, 16u, 20u, 24u};
    const uint32_t want_source[24] = {
        0u, 3u, 7u, 7u,          /* expert 0: flats 0,9,21,22 -> rows */
        1u, 1u, 5u,              /* expert 1: flats 3,4,15 */
        0u, 3u, 3u, 6u, 6u, 6u,  /* expert 2: flats 1,10,11,18,19,20 */
        1u, 5u, 5u,              /* expert 3: flats 5,16,17 */
        0u, 4u, 4u, 4u,          /* expert 4: flats 2,12,13,14 */
        2u, 2u, 2u, 7u};         /* expert 5: flats 6,7,8,23 */
    const uint32_t want_expert[24] = {
        0u, 0u, 0u, 0u,  1u, 1u, 1u,  2u, 2u, 2u, 2u, 2u, 2u,
        3u, 3u, 3u,  4u, 4u, 4u, 4u,  5u, 5u, 5u, 5u};
    const uint32_t want_inverse[24] = {
        0u, 7u, 16u, 4u, 5u, 13u, 20u, 21u, 22u, 1u, 8u, 9u,
        17u, 18u, 19u, 6u, 14u, 15u, 10u, 11u, 12u, 2u, 3u, 23u};

    uint32_t source[24], expert[24], offsets[7], inverse[24];
    /* Poison with garbage twice: results must be identical regardless of
     * prior buffer contents (no hidden order dependence). */
    for (int pass = 0; pass < 2; ++pass)
    {
        for (uint32_t i = 0u; i < 24u; ++i)
        {
            source[i] = 0xDEADBEEFu + pass;
            expert[i] = 0xFEEDFACEu + pass;
            inverse[i] = 0u - pass;
        }
        for (uint32_t e = 0u; e < 7u; ++e)
            offsets[e] = 0u - pass;

        Execute(
            [&]()
            {
                SparkDsv4RocmRouteGroupKernel(routes, source, expert, offsets,
                                              inverse, pairs, k, experts);
            },
            1u, 1u);

        for (uint32_t e = 0u; e < 7u; ++e)
            ExpectU32("route.offsets", e, offsets[e], want_offsets[e]);
        for (uint32_t i = 0u; i < 24u; ++i)
        {
            ExpectU32("route.source", i, source[i], want_source[i]);
            ExpectU32("route.expert", i, expert[i], want_expert[i]);
            ExpectU32("route.inverse", i, inverse[i], want_inverse[i]);
        }
    }
    std::printf("route-group: %s\n", g_failures == 0u ? "ok" : "FAILED");
}

// --- Test 2: CacheScatterKernel ring addressing + verbatim bytes. --------
void TestCacheScatter()
{
    const uint32_t ring_slots = 4u, width = 4u;
    const uint64_t stride = 64u; /* cache words per lane row */
    uint16_t cache[2u * 64u];
    uint16_t payload[5u * 4u];
    uint32_t lanes_idx[5] = {0u, 1u, 0u, 1u, 0u};
    uint64_t positions[5] = {1u, 3u, 5u, 7u, 9u};
    uint32_t ring_out[5];

    for (uint32_t i = 0u; i < 2u * 64u; ++i)
        cache[i] = (uint16_t)(0xBEEFu);
    for (uint32_t r = 0u; r < 5u; ++r)
        for (uint32_t e = 0u; e < width; ++e)
            payload[r * width + e] = (uint16_t)(r * 100u + e);

    /* ratio == 0: slot = position % ring_slots. */
    Execute(
        [&]()
        {
            SparkDsv4RocmCacheScatterKernel(
                payload, nullptr, cache, stride, lanes_idx, positions, 5u,
                width, 0u, 0u, ring_slots, ring_out);
        },
        5u, 1u);


    const uint32_t want_ring[5] = {1u, 3u, 1u, 3u, 1u};
    for (uint32_t r = 0u; r < 5u; ++r)
        ExpectU32("scatter.ring", r, ring_out[r], want_ring[r]);
    /* Rows 0,2,4 collide on lane0/slot1: last writer (row 4) wins. Lane1
     * slot3 holds row 3. Everything else keeps its poison. */
    for (uint32_t e = 0u; e < width; ++e)
    {
        ExpectU32("scatter.final.l0s1", 4u + e,
                  cache[(uint64_t)width + e], payload[4u * width + e]);
        ExpectU32("scatter.final.l1s3", 76u + e,
                  cache[stride + 3u * width + e], payload[3u * width + e]);
    }
    /* Untouched slots keep their poison. */
    ExpectU32("scatter.poison", 0u, cache[0u], 0xBEEFu);
    ExpectU32("scatter.poison", 1u, cache[stride + 2u], 0xBEEFu);

    /* ratio == 2 with base_slot 8 and an emitted predicate: only emitting
     * rows land, slot = base + ((position % ring_slots)/ratio). */
    const uint32_t emitted[4] = {1u, 1u, 0u, 1u};
    const uint64_t cpositions[4] = {0u, 1u, 2u, 3u}; /* slots 8,8,skip,9 */
    const uint32_t clanes[4] = {0u, 0u, 0u, 1u};
    uint16_t compressed[5u * 4u];
    for (uint32_t r = 0u; r < 4u; ++r)
        for (uint32_t e = 0u; e < width; ++e)
            compressed[r * width + e] = (uint16_t)(0x4000u + r * 10u + e);
    for (uint32_t i = 0u; i < 2u * 64u; ++i)
        cache[i] = (uint16_t)(0xC0DEu);

    Execute(
        [&]()
        {
            SparkDsv4RocmCacheScatterKernel(
                compressed, emitted, cache, stride, clanes, cpositions, 4u,
                width, 8u, 2u, ring_slots, ring_out);
        },
        4u, 1u);

    ExpectU32("cscatter.ring0", 0u, ring_out[0], 8u);
    ExpectU32("cscatter.ring1", 1u, ring_out[1], 8u);
    ExpectU32("cscatter.ring3", 3u, ring_out[3], 9u);
    for (uint32_t e = 0u; e < width; ++e)
        ExpectU32("cscatter.payload",
                  (uint64_t)e + 8u * width,
                  cache[8u * width + e], compressed[1u * width + e]);
    for (uint32_t e = 0u; e < width; ++e)
        ExpectU32("cscatter.payload",
                  (uint64_t)stride + 9u * width + e,
                  cache[stride + 9u * width + e],
                  compressed[3u * width + e]);
    /* Skipped non-emitting row left its slot poisoned. */
    ExpectU32("cscatter.skip-poison", (uint64_t)stride + 8u * width,
              cache[stride + 8u * width], 0xC0DEu);
    std::printf("cache-scatter: %s\n", g_failures == 0u ? "ok" : "FAILED");
}

// --- Test 3: InitializePagesKernel parent copy + span poison. ------------
void TestInitializePages()
{
    const uint64_t stride_words = 8u;
    /* Page ids are pool-row numbers here: row 0 is the root, row 1 its
     * child. (Arbitrary ids are legal; keep the array sized max_id+1.) */
    uint32_t pool[2u * 8u];
    uint32_t pages[2] = {0u, 1u};
    uint32_t parents[2] = {0xffffffffu /* root */, 0u};
    uint64_t span_off[1] = {2u};
    uint64_t span_len[1] = {3u};

    for (uint32_t i = 0u; i < 2u * 8u; ++i)
        pool[i] = 0x13579BDFu; /* garbage everywhere first */

    Execute(
        [&]()
        {
            SparkDsv4RocmInitializePagesKernel(pool, stride_words, pages,
                                               parents, 2u, span_off,
                                               span_len, 1u);
        },
        2u, 1u);

    for (uint32_t w = 0u; w < 8u; ++w)
    {
        if (w >= 2u && w <= 4u)
            ExpectF32Bits("initpage.span-is-inf", w,
                          *(const float *)&pool[w], -INFINITY);
        else
            ExpectU32("initpage.root", w, pool[w], 0u);
    }
    for (uint32_t w = 0u; w < 8u; ++w)
        ExpectU32("initpage.child-copies-parent", w, pool[8u + w],
                  pool[w]);
    std::printf("initialize-pages: %s\n", g_failures == 0u ? "ok" : "FAILED");
}

// --- Test 4: UpdatePageTableKernel scattered commits. --------------------
void TestUpdatePageTable()
{
    uint32_t table[16];
    const uint32_t idx[3] = {3u, 7u, 1u};
    const uint32_t val[3] = {9u, 10u, 11u};
    for (uint32_t i = 0u; i < 16u; ++i)
        table[i] = 0xffffffffu;
    /* One update per block under blockDim 1: grid carries the count. */
    Execute(
        [&]()
        {
            SparkDsv4RocmUpdatePageTableKernel(table, idx, val, 3u);
        },
        3u, 1u);
    ExpectU32("ptable[3]", 3u, table[3u], 9u);
    ExpectU32("ptable[7]", 7u, table[7u], 10u);
    ExpectU32("ptable[1]", 1u, table[1u], 11u);
    ExpectU32("ptable[0]", 0u, table[0u], 0xffffffffu);
    std::printf("update-page-table: %s\n", g_failures == 0u ? "ok" : "FAILED");
}

// --- Test 5: CompressStepKernel plain pool (overlap disabled). -----------
/* ratio 2, width 2, two lanes. Packet order: lane0(pos0), lane0(pos1),
 * lane1(pos3). All scores and APE zero -> every group weight is expf(0),
 * so a boundary pools the plain arithmetic mean of its two kv slots, and
 * non-boundary rows emit zeros. Hand-computable bit-exactly. */
void TestCompressStepPlain()
{
    const uint32_t ratio = 2u, width = 2u, rows = 3u;
    const uint64_t stride = ratio * width; /* 4 state words per lane */
    uint16_t kv[3u * 2u], score[3u * 2u];
    float ape[2u * 2u] = {0.0f, 0.0f, 0.0f, 0.0f};
    float kv_state[2u * 4u], score_state[2u * 4u];
    uint32_t lanes[3] = {0u, 0u, 1u};
    uint64_t positions[3] = {0u, 1u, 3u};
    uint16_t emit[3u * 2u];
    uint32_t emitted[3], counters[2] = {77u, 88u};

    /* kv rows: r0=[10,20] r1=[30,40] r2=[50,60] as bf16 via RNE helper on
     * integral values (exact). */
    const float kv_rows[3][2] = {{10.0f, 20.0f}, {30.0f, 40.0f},
                                 {50.0f, 60.0f}};
    for (uint32_t r = 0u; r < 3u; ++r)
        for (uint32_t e = 0u; e < 2u; ++e)
        {
            /* Small integers are bf16-exact; reuse the TU's own RNE store
             * through a scratch float. */
            uint16_t word = 0u;
            float value = kv_rows[r][e];
            uint32_t bits;
            std::memcpy(&bits, &value, 4u);
            word = (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
            kv[r * 2u + e] = word;
            score[r * 2u + e] = 0u; /* +0.0 */
        }
    memset(kv_state, 0, sizeof(kv_state));
    memset(score_state, 0, sizeof(score_state));

    Execute(
        [&]()
        {
            SparkDsv4RocmCompressStepKernel(kv, score, ape, kv_state,
                                            score_state, stride, lanes,
                                            positions, rows, ratio, 0u,
                                            width, emit, emitted, counters);
        },
        rows, 1u);

    /* Emission decisions: pos0 -> (0+1)%2 != 0 -> 0; pos1 -> 1; pos3 -> 1. */
    ExpectU32("compress.emitted0", 0u, emitted[0], 0u);
    ExpectU32("compress.emitted1", 1u, emitted[1], 1u);
    ExpectU32("compress.emitted2", 2u, emitted[2], 1u);
    /* Emit counters increment once per boundary per lane. */
    ExpectU32("compress.counter.lane0", 0u, counters[0], 78u);
    ExpectU32("compress.counter.lane1", 1u, counters[1], 89u);
    /* Non-boundary emit rows are zeros. */
    ExpectU32("compress.emit-zero0", 0u, emit[0], 0u);
    ExpectU32("compress.emit-zero1", 1u, emit[1], 0u);
    /* Boundary emit rows are the slot means (weights all expf(0)). Lane0
     * group {kv[10,20],[30,40]} -> [20,30]; lane1 wrote only slot 1, so its
     * group holds {kv_state[slot1] = [50,60]} plus slot 0 = 0 from init ->
     * mean [25,30]. Mirror the exact fp32 op order: sequential adds, one
     * divide. */
    {
        float m0 = (10.0f + 30.0f) / 2.0f;
        float m1 = (20.0f + 40.0f) / 2.0f;
        ExpectF32Bits("compress.pool.lane0.ch0", 0u,
                      DecodeProbe(emit[2u + 0u]), m0);
        ExpectF32Bits("compress.pool.lane0.ch1", 1u,
                      DecodeProbe(emit[2u + 1u]), m1);
        float n0 = (0.0f + 50.0f) / 2.0f;
        float n1 = (0.0f + 60.0f) / 2.0f;
        ExpectF32Bits("compress.pool.lane1.ch0", 0u,
                      DecodeProbe(emit[4u + 0u]), n0);
        ExpectF32Bits("compress.pool.lane1.ch1", 1u,
                      DecodeProbe(emit[4u + 1u]), n1);
    }
    /* Ring state: lane0 slot0 = row0 kv, slot1 = row1 kv; lane1 slot1 =
     * row2 kv, slot0 untouched zero. Score states mirror + zero APE. */
    ExpectF32Bits("state.kv.l0.s0.ch0", 0u, kv_state[0], 10.0f);
    ExpectF32Bits("state.kv.l0.s0.ch1", 1u, kv_state[1], 20.0f);
    ExpectF32Bits("state.kv.l0.s1.ch0", 0u, kv_state[2], 30.0f);
    ExpectF32Bits("state.kv.l0.s1.ch1", 1u, kv_state[3], 40.0f);
    ExpectF32Bits("state.score.ape", 0u, score_state[0], 0.0f);
    ExpectF32Bits("state.kv.l1.s1.ch0", 0u, kv_state[4u + 2u], 50.0f);
    std::printf("compress-step plain: %s\n", g_failures == 0u ? "ok" : "FAILED");
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    TestRouteGroup();
    TestCacheScatter();
    TestInitializePages();
    TestUpdatePageTable();
    TestCompressStepPlain();
    if (g_failures != 0u)
    {
        std::printf("L3_C2_HOST_EXEC_PROBE FAILED (%u mismatches)\n",
                    g_failures);
        return 1;
    }
    std::printf("L3_C2_HOST_EXEC_PROBE ok: route grouping, ring scatter, "
                "page init/commits, compressor boundaries/counters/state\n");
    return 0;
}
