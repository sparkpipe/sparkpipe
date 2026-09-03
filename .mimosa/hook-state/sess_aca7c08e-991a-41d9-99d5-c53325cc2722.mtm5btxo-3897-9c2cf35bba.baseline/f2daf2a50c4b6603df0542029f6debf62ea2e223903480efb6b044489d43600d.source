#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int32_t extract(const uint8_t *base, uint32_t index, uint32_t bits)
{
	uint32_t bit = index * bits;
	uint32_t word;
	memcpy(&word, base + (bit >> 3u), 4);
	return(((int32_t)(word << (32u - bits - (bit & 7u)))) >> (32 - (int32_t)bits));
}

static int32_t sweep(uint32_t bits, uint32_t slots)
{
	uint8_t buffer[64];
	int32_t low = -(1 << (bits - 1)), high = (1 << (bits - 1)) - 1;
	uint32_t slot, i;
	int32_t value, got, bad = 0, total = 0;
	for (slot = 0; slot < slots; ++slot)
		for (value = low; value <= high; ++value)
		{
			uint64_t packed = ((uint64_t)((uint32_t)value & ((1u << bits) - 1u))) << (slot * bits);
			memset(buffer, 0, sizeof(buffer));
			for (i = 0; i < 8u; ++i)
				buffer[i] = (uint8_t)(packed >> (i * 8u));
			got = extract(buffer, slot, bits);
			++total;
			if (got != value)
			{
				if (bad < 3)
					printf("  FAIL int%u slot %u value %d -> %d\n", bits, slot, value, got);
				++bad;
			}
		}
	printf("  int%u: %d/%d codes exact across %u bit alignments\n", bits, total - bad, total, slots);
	return(bad);
}

int main(void)
{
	int32_t bad = 0;
	printf("sub-byte code extraction\n");
	bad += sweep(6u, 5u);
	bad += sweep(7u, 5u);
	bad += sweep(4u, 8u);
	printf("\n%s\n", bad ? "FAIL" : "PASS");
	return bad ? 1 : 0;
}
