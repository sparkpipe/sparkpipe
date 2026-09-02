#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float bf16_to_float(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static float bias_for(uint32_t bits) { return (float)(128 + (1 << (bits - 1))); }

static float arithmetic(uint32_t word, uint32_t align, uint32_t bits)
{
	return (float)(((int32_t)(word << (32u - bits - align))) >> (32 - (int32_t)bits));
}

static float binary(uint32_t word, uint32_t align, uint32_t bits)
{
	uint32_t code = (word >> align) & ((1u << bits) - 1u);
	uint16_t packed = (uint16_t)(0x4300u | (code ^ (1u << (bits - 1u))));
	return bf16_to_float(packed) - bias_for(bits);
}

static int sweep(uint32_t bits, int expect_exact)
{
	int32_t low = -(1 << (bits - 1)), high = (1 << (bits - 1)) - 1;
	int bad = 0, total = 0;
	for (uint32_t align = 0; align < 8u; ++align)
		for (int32_t v = low; v <= high; ++v)
		{
			uint32_t word = ((uint32_t)v & ((1u << bits) - 1u)) << align;
			float a = arithmetic(word, align, bits), b = binary(word, align, bits);
			++total;
			if (a != b || a != (float)v) ++bad;
		}
	if (expect_exact)
		printf("  %u-bit: %d/%d exact across 8 alignments\n", bits, total - bad, total);
	else
		printf("  %u-bit: %d/%d wrong, correctly REJECTED by static_assert in mma.cuh\n", bits, bad, total);
	return expect_exact ? bad : (bad == 0);
}

int main(void)
{
	int bad = 0;
	printf("free dequantisation, codes to BF16 with no conversion instruction\n");
	bad += sweep(4u, 1);
	bad += sweep(6u, 1);
	bad += sweep(7u, 1);
	printf("\nthe limit, and why INT7 is the widest free code:\n");
	bad += sweep(8u, 0);
	printf("  BF16 has 7 mantissa bits. An 8-bit code overflows into the exponent\n");
	printf("  and the value comes out doubled, so INT8 takes the conversion path.\n");
	printf("\n%s\n", bad ? "FAIL" : "PASS");
	return bad ? 1 : 0;
}
