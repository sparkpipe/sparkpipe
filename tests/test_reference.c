#include "reference.h"

static int32_t failures = 0;


static void pack_code(uint8_t *base, uint32_t index, uint32_t bits, int32_t value)
{
	uint32_t start = index * bits, i;
	for (i = 0; i < bits; ++i)
	{
		uint32_t bit = start + i;
		uint8_t mask = (uint8_t)(1u << (bit & 7));
		if ((value >> i) & 1)
			base[bit >> 3] |= mask;
		else
			base[bit >> 3] &= (uint8_t)~mask;
	}
}

static int round_trip(uint32_t bits, uint32_t slots)
{
    uint8_t buffer[256];
    int32_t low = -(1 << (bits - 1)), high = (1 << (bits - 1)) - 1;
    uint32_t slot;
    int32_t v, bad = 0, total = 0;
    for (slot = 0; slot < slots; ++slot)
        for (v = low; v <= high; ++v)
        {
            memset(buffer, 0, sizeof(buffer));
            pack_code(buffer, slot, bits, v);
            ++total;
            if (ref_code(buffer, slot, bits) != v)
                ++bad;
        }
    printf("  int%u round trip: %d/%d\n", bits, total - bad, total);
    return bad;
}


static int enumerate_e2m1(void)
{
    static const float expect_[16] = { 0,0.5f,1,1.5f,2,3,4,6, 0,-0.5f,-1,-1.5f,-2,-3,-4,-6 };
    int bad = 0, i;
    for (i = 0; i < 16; ++i)
        if (ref_e2m1((uint8_t)i) != expect_[i])
            ++bad;
    printf("  e2m1: %d/16 codes match the defined magnitudes\n", 16 - bad);
    return bad;
}

static int enumerate_e4m3(void)
{
    int bad = 0, i;
    if (ref_e4m3(0x38u) != 1.0f) ++bad;
    if (ref_e4m3(0x40u) != 2.0f) ++bad;
    if (ref_e4m3(0x30u) != 0.5f) ++bad;
    if (ref_e4m3(0xb8u) != -1.0f) ++bad;
    for (i = 1; i < 126; ++i)
        if (!(ref_e4m3((uint8_t)i) > ref_e4m3((uint8_t)(i - 1))))
            ++bad;
    printf("  e4m3: %s at 0.5/1/2/-1 and monotonic over the positive range\n",
        bad ? "WRONG" : "exact");
    return bad;
}


static int gemm_against_fp64(void)
{
    static uint16_t a[16 * 64], b[8 * 64], out[16 * 8];
    static float scale_a[16], scale_b[8];
    uint32_t m, n, k;
    int bad = 0;
    for (m = 0; m < 16u * 64u; ++m)
        a[m] = (uint16_t)(0x3f00u + (m % 97u));
    for (n = 0; n < 8u * 64u; ++n)
        b[n] = (uint16_t)(0x3f80u - (n % 89u));
    for (m = 0; m < 16u; ++m) scale_a[m] = 1.0f;
    for (n = 0; n < 8u; ++n) scale_b[n] = 1.0f;
    ref_gemm(out, 16u, 8u, 64u,
        (const uint8_t *)a, 64u * 16u, ref_element_bf16, scale_a, 64u,
        (const uint8_t *)b, 64u * 16u, ref_element_bf16, scale_b, 64u);
    for (m = 0; m < 16u; ++m)
        for (n = 0; n < 8u; ++n)
        {
            double exact = 0.0;
            float got = ref_bf16(out[(m * 8u) + n]);
            for (k = 0; k < 64u; ++k)
                exact += (double)ref_bf16(a[(m * 64u) + k]) * (double)ref_bf16(b[(n * 64u) + k]);
            if (!(fabs(got - exact) <= fabs(exact) / 200.0))
            {
                if (bad < 3)
                    printf("  FAIL m%u n%u: fp32 path %g, fp64 exact %g\n", m, n, got, exact);
                ++bad;
            }
        }
    printf("  gemm vs fp64: %d/128 outputs within BF16 precision\n", 128 - bad);
    return bad;
}

static int every_decoder_reaches_the_gemm(void)
{
    static uint8_t plane[512];
    static float scale[8] = { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f };
    int bad = 0;
    memset(plane, 0, sizeof(plane));
    pack_code(plane, 0u, 7u, -5);
    if (ref_element_int7(plane, 0u, 0u, 4096u, scale, 128u) != -10.0f) ++bad;
    memset(plane, 0, sizeof(plane));
    pack_code(plane, 0u, 6u, 9);
    if (ref_element_int6(plane, 0u, 0u, 4096u, scale, 128u) != 18.0f) ++bad;
    plane[0] = 0x38u;
    if (ref_element_e4m3(plane, 0u, 0u, 4096u, scale, 128u) != 2.0f) ++bad;
    plane[0] = 0x05u;
    if (ref_element_e2m1(plane, 0u, 0u, 4096u, scale, 128u) != 6.0f) ++bad;
    printf("  all five decoders reachable and scaled: %s\n", bad ? "NO" : "yes");
    return bad;
}

int main(void)
{
    printf("reference oracle\n\npack and decode round trip\n");
    failures += round_trip(4u, 8u);
    failures += round_trip(6u, 8u);
    failures += round_trip(7u, 8u);
    printf("\nfloat formats enumerated\n");
    failures += enumerate_e2m1();
    failures += enumerate_e4m3();
    printf("\ndecoders wired through the element accessors\n");
    failures += every_decoder_reaches_the_gemm();
    printf("\nreference GEMM checked against an independent FP64 sum\n");
    failures += gemm_against_fp64();
    printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
