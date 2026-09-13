// GEMM-007 descriptor cache, checked on a host with a counting mock encode.
//
// The contract the decode hot path relies on: a key hit must return the bytes
// cuTensorMapEncodeTiled would have produced (the request IS every input to
// the encode), and a full model's worth of (layer, projection) keys must
// cycle through the table with zero re-encodes. Both are checkable without a
// driver, which is the only way they get checked for the next three days.
#include "runtime/gemm_descriptor_cache.h"

#include <stdio.h>
#include <string.h>
#include <thread>

static int failures = 0;
static uint32_t mock_calls = 0u;
static int32_t mock_status = LM_TM_ENCODE_OK;

static void check(int condition, const char *label)
{
    printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
    if (!condition)
    {
        failures++;
    }
}

// Deterministic in the (canonical) request, so a cache hit can be compared
// byte-for-byte against the encode it stands in for. CUtensorMap is 128
// bytes; the request is 48, the marker byte proves the encode ran end to end.
static int32_t MockEncode(CUtensorMap *map, const LmTensorMapRequest *request)
{
    __atomic_fetch_add(&mock_calls, 1u, __ATOMIC_RELAXED);
    if (mock_status != LM_TM_ENCODE_OK)
    {
        return mock_status;
    }
    memset(map, 0, sizeof(*map));
    memcpy(map, request, sizeof(*request));
    ((uint8_t *)map)[sizeof(*request)] = 0xA5u;
    return LM_TM_ENCODE_OK;
}

// What the mock produces for the canonical key the cache hands it.
static void ReferenceEncode(CUtensorMap *map, const LmTensorMapRequest *request)
{
    LmTensorMapRequest key = LmGemmDescriptorKey(request);

    memset(map, 0, sizeof(*map));
    memcpy(map, &key, sizeof(key));
    ((uint8_t *)map)[sizeof(key)] = 0xA5u;
}

static LmTensorMapRequest RequestFor(const void *address, uint32_t rows)
{
    LmTensorMapRequest request;

    memset(&request, 0, sizeof(request));
    request.global_address = address;
    request.rows = rows;
    request.columns = 6144u;
    request.groups = 1u;
    request.box_rows = 16u;
    request.box_columns = 128u;
    request.element_bits = LM_TM_BITS_FP8;
    return request;
}

static uint64_t arena[16384] __attribute__((aligned(64)));

static const void *Address(uint32_t index)
{
    return (const uint8_t *)arena + (size_t)index * 64u;
}

// Collect count keys whose cache sets each hold fewer than WAYS occupants, so
// a sweep of them can evict nothing. The hash is header-static, which is what
// makes this deterministic instead of probabilistic.
static uint32_t CollectSpread(LmTensorMapRequest *out, uint32_t count)
{
    uint8_t load[LM_GEMM_DESCRIPTOR_CACHE_SETS];
    uint32_t found = 0u;
    uint32_t index;

    memset(load, 0, sizeof(load));
    for (index = 0u; index < 16384u && found < count; ++index)
    {
        LmTensorMapRequest request = RequestFor(Address(index), 64u);
        LmTensorMapRequest key = LmGemmDescriptorKey(&request);
        uint64_t set =
            LmGemmDescriptorKeyHash(&key) % LM_GEMM_DESCRIPTOR_CACHE_SETS;

        if (load[set] >= LM_GEMM_DESCRIPTOR_CACHE_WAYS)
        {
            continue;
        }
        load[set]++;
        out[found++] = request;
    }
    return found;
}

// The opposite: count keys that ALL land on one set, to force eviction.
static uint32_t CollectColliding(LmTensorMapRequest *out, uint32_t count)
{
    uint64_t target = UINT64_MAX;
    uint32_t found = 0u;
    uint32_t index;

    for (index = 0u; index < 16384u && found < count; ++index)
    {
        LmTensorMapRequest request = RequestFor(Address(index), 64u);
        LmTensorMapRequest key = LmGemmDescriptorKey(&request);
        uint64_t set =
            LmGemmDescriptorKeyHash(&key) % LM_GEMM_DESCRIPTOR_CACHE_SETS;

        if (target == UINT64_MAX)
        {
            target = set;
        }
        if (set != target)
        {
            continue;
        }
        out[found++] = request;
    }
    return found;
}

static int32_t Fetch(
    LmGemmDescriptorCache *cache,
    const LmTensorMapRequest *request,
    CUtensorMap *out)
{
    memset(out, 0, sizeof(*out));
    return LmGemmDescriptorCacheFetch(cache, request, MockEncode, out);
}

static void test_hit_returns_identical_bytes(void)
{
    LmGemmDescriptorCache cache{};
    LmTensorMapRequest request = RequestFor(Address(1u), 64u);
    CUtensorMap first;
    CUtensorMap second;
    CUtensorMap reference;

    mock_calls = 0u;

    check(Fetch(&cache, &request, &first) == LM_TM_ENCODE_OK,
        "first fetch encodes");
    check(mock_calls == 1u, "first fetch encodes exactly once");
    ReferenceEncode(&reference, &request);
    check(memcmp(&first, &reference, sizeof(first)) == 0,
        "encoded bytes match the reference encode");

    check(Fetch(&cache, &request, &second) == LM_TM_ENCODE_OK,
        "second fetch succeeds");
    check(mock_calls == 1u, "second fetch is a hit, no encode");
    check(memcmp(&second, &first, sizeof(second)) == 0,
        "hit returns byte-identical descriptor");
}

static void test_padding_is_not_part_of_the_key(void)
{
    LmGemmDescriptorCache cache{};
    LmTensorMapRequest request = RequestFor(Address(2u), 64u);
    LmTensorMapRequest padded;
    CUtensorMap first;
    CUtensorMap second;

    mock_calls = 0u;
    check(Fetch(&cache, &request, &first) == LM_TM_ENCODE_OK, "seed encodes");

    // Same field values, garbage in the 4 tail-padding bytes (offset 44..47).
    padded = request;
    memset((uint8_t *)&padded + 44u, 0xFF, sizeof(padded) - 44u);
    check(Fetch(&cache, &padded, &second) == LM_TM_ENCODE_OK,
        "padded twin fetch succeeds");
    check(mock_calls == 1u, "padding garbage does not break the key");
    check(memcmp(&second, &first, sizeof(second)) == 0,
        "padded twin returns identical bytes");
}

static void test_every_key_field_is_keyed(void)
{
    LmGemmDescriptorCache cache{};
    LmTensorMapRequest base = RequestFor(Address(3u), 64u);
    LmTensorMapRequest variant;
    CUtensorMap out;
    uint32_t before;

    mock_calls = 0u;
    check(Fetch(&cache, &base, &out) == LM_TM_ENCODE_OK, "base encodes");
    before = mock_calls;

    variant = base;
    variant.global_address = Address(4u);
    check(Fetch(&cache, &variant, &out) == LM_TM_ENCODE_OK &&
        mock_calls == before + 1u, "address is part of the key");

    variant = base;
    variant.rows = base.rows + 16u;
    check(Fetch(&cache, &variant, &out) == LM_TM_ENCODE_OK &&
        mock_calls == before + 2u, "rows are part of the key");

    variant = base;
    variant.groups = 256u;
    check(Fetch(&cache, &variant, &out) == LM_TM_ENCODE_OK &&
        mock_calls == before + 3u, "groups are part of the key");

    variant = base;
    variant.box_rows = 32u;
    check(Fetch(&cache, &variant, &out) == LM_TM_ENCODE_OK &&
        mock_calls == before + 4u, "box rows are part of the key");

    variant = base;
    variant.element_bits = LM_TM_BITS_BF16;
    check(Fetch(&cache, &variant, &out) == LM_TM_ENCODE_OK &&
        mock_calls == before + 5u, "element width is part of the key");
}

static void test_encode_failure_is_not_cached(void)
{
    LmGemmDescriptorCache cache{};
    LmTensorMapRequest request = RequestFor(Address(5u), 64u);
    CUtensorMap out;

    mock_calls = 0u;
    mock_status = LM_TM_ENCODE_ERR_DRIVER;
    check(Fetch(&cache, &request, &out) == LM_TM_ENCODE_ERR_DRIVER,
        "driver error propagates");
    check(mock_calls == 1u, "failing encode was attempted");
    mock_status = LM_TM_ENCODE_OK;
    check(Fetch(&cache, &request, &out) == LM_TM_ENCODE_OK,
        "retry after failure succeeds");
    check(mock_calls == 2u, "a failed encode cached nothing");
    check(Fetch(&cache, &request, &out) == LM_TM_ENCODE_OK &&
        mock_calls == 2u, "the retry's encode is what got cached");
}

static void test_lru_eviction_within_a_set(void)
{
    LmTensorMapRequest colliding[LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u];
    LmGemmDescriptorCache cache{};
    CUtensorMap out;
    uint32_t found;

    found = CollectColliding(colliding, LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u);
    check(found == LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u,
        "found WAYS+1 keys on one set");
    if (found != LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u)
    {
        return;
    }
    mock_calls = 0u;

    // Fill the set, refresh entry 0, then force one eviction. The victim must
    // be entry 1: the least-recently-used of the remaining three.
    for (uint32_t way = 0u; way < LM_GEMM_DESCRIPTOR_CACHE_WAYS; ++way)
    {
        check(Fetch(&cache, &colliding[way], &out) == LM_TM_ENCODE_OK,
            "set fill encodes");
    }
    check(mock_calls == LM_GEMM_DESCRIPTOR_CACHE_WAYS, "set is full");
    check(Fetch(&cache, &colliding[0], &out) == LM_TM_ENCODE_OK &&
        mock_calls == LM_GEMM_DESCRIPTOR_CACHE_WAYS,
        "refreshing entry 0 is a hit");
    check(Fetch(&cache, &colliding[LM_GEMM_DESCRIPTOR_CACHE_WAYS], &out)
        == LM_TM_ENCODE_OK &&
        mock_calls == LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u,
        "fifth key on the set evicts");

    check(Fetch(&cache, &colliding[0], &out) == LM_TM_ENCODE_OK &&
        mock_calls == LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u,
        "refreshed entry survived eviction");
    check(Fetch(&cache, &colliding[2], &out) == LM_TM_ENCODE_OK &&
        mock_calls == LM_GEMM_DESCRIPTOR_CACHE_WAYS + 1u,
        "untouched entry 2 survived eviction");
    check(Fetch(&cache, &colliding[1], &out) == LM_TM_ENCODE_OK &&
        mock_calls == LM_GEMM_DESCRIPTOR_CACHE_WAYS + 2u,
        "LRU entry was the one evicted");
}

// The acceptance condition: a working set the size of a full model - layers
// times projections weight maps plus the activation maps - cycles through the
// table and the SECOND cycle encodes nothing.
static void test_decode_working_set_encodes_zero_in_steady_state(void)
{
    static LmTensorMapRequest keys[768];
    LmGemmDescriptorCache cache{};
    CUtensorMap out;
    CUtensorMap reference;
    uint32_t found;
    uint32_t index;
    uint32_t pass;
    int identical = 1;

    found = CollectSpread(keys, 768u);
    check(found == 768u, "collected a model-sized spread of keys");
    if (found != 768u)
    {
        return;
    }
    mock_calls = 0u;

    for (pass = 0u; pass < 3u; ++pass)
    {
        for (index = 0u; index < found; ++index)
        {
            if (Fetch(&cache, &keys[index], &out) != LM_TM_ENCODE_OK)
            {
                identical = 0;
                continue;
            }
            if (pass > 0u)
            {
                ReferenceEncode(&reference, &keys[index]);
                if (memcmp(&out, &reference, sizeof(out)) != 0)
                {
                    identical = 0;
                }
            }
        }
        if (pass == 0u)
        {
            check(mock_calls == found, "first pass encodes each key once");
        }
    }
    check(mock_calls == found,
        "passes two and three encoded nothing at all");
    check(identical, "every steady-state hit is byte-identical to an encode");
}

static void test_concurrent_fetches_agree(void)
{
    static LmTensorMapRequest keys[64];
    LmGemmDescriptorCache cache{};
    std::thread threads[4];
    int agreed[4] = { 1, 1, 1, 1 };
    uint32_t found;
    uint32_t calls_after_warmup;

    found = CollectSpread(keys, 64u);
    check(found == 64u, "collected keys for the threaded run");
    if (found != 64u)
    {
        return;
    }
    mock_calls = 0u;

    for (uint32_t t = 0u; t < 4u; ++t)
    {
        threads[t] = std::thread([&, t]
        {
            CUtensorMap out;
            CUtensorMap reference;

            for (uint32_t iteration = 0u; iteration < 2000u; ++iteration)
            {
                const LmTensorMapRequest *request =
                    &keys[(iteration + t) % found];

                if (Fetch(&cache, request, &out) != LM_TM_ENCODE_OK)
                {
                    agreed[t] = 0;
                    return;
                }
                ReferenceEncode(&reference, request);
                if (memcmp(&out, &reference, sizeof(out)) != 0)
                {
                    agreed[t] = 0;
                    return;
                }
            }
        });
    }
    for (uint32_t t = 0u; t < 4u; ++t)
    {
        threads[t].join();
        check(agreed[t], "thread saw only byte-identical descriptors");
    }
    // Racing a cold miss can encode the same key once per thread that saw it
    // uncached; every store is the same bytes, so the bound is keys times
    // threads, never more, and warm keys never re-encode.
    check(mock_calls <= found * 4u,
        "racing misses stayed within one encode per thread per key");
    calls_after_warmup = mock_calls;
    for (uint32_t index = 0u; index < found; ++index)
    {
        CUtensorMap out;

        check(Fetch(&cache, &keys[index], &out) == LM_TM_ENCODE_OK,
            "post-thread fetch succeeds");
    }
    check(mock_calls == calls_after_warmup,
        "everything the threads touched stayed cached");
}

// The production entry point, against the driver stub: LmGemmTensorMapCached
// must route through the real plan build and driver marshalling, and a repeat
// call must return the same bytes.
static void test_production_entry_routes_through_prepare(void)
{
    LmTensorMapRequest request;
    alignas(64) CUtensorMap cached;
    alignas(64) CUtensorMap direct;

    memset(&request, 0, sizeof(request));
    request.global_address = arena;
    request.rows = 64u;
    request.columns = 7168u;
    request.groups = 1u;
    request.box_rows = 16u;
    request.box_columns = 64u;
    request.element_bits = LM_TM_BITS_BF16;
    check(LmGemmTensorMapCached(&cached, &request) == LM_TM_ENCODE_OK,
        "production entry encodes through the driver stub");
    check(LmTensorMapPrepare(&direct, &request) == LM_TM_ENCODE_OK,
        "direct prepare of the same request succeeds");
    check(memcmp(&cached, &direct, sizeof(cached)) == 0,
        "cached bytes equal a direct encode");
    memset(&cached, 0, sizeof(cached));
    check(LmGemmTensorMapCached(&cached, &request) == LM_TM_ENCODE_OK &&
        memcmp(&cached, &direct, sizeof(cached)) == 0,
        "repeat call returns the same bytes");
}

int main(void)
{
    printf("GEMM-007 tensor-map descriptor cache\n");

    printf("\nhit path\n");
    test_hit_returns_identical_bytes();
    test_padding_is_not_part_of_the_key();
    test_every_key_field_is_keyed();
    test_encode_failure_is_not_cached();
    test_production_entry_routes_through_prepare();

    printf("\neviction\n");
    test_lru_eviction_within_a_set();
    test_decode_working_set_encodes_zero_in_steady_state();

    printf("\nthreads\n");
    test_concurrent_fetches_agree();

    printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
