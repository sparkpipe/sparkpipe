#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint16_t ToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
int main(int argc, char **argv)
{
	const long rows = atol(argv[3]), split = atol(argv[4]), width = atol(argv[5]);
	FILE *kernel = fopen(argv[1], "rb"), *ref = fopen(argv[2], "rb");
	float triple[3], r;
	for ( int band = 0; band < 2; band++ )
	{
		const long lo = band == 0 ? 0 : split, hi = band == 0 ? split : rows;
		long n = 0, ref_off = 0, kernel_off = 0, differ = 0;
		for ( long i = lo * width; i < hi * width; i++ )
		{
			if ( fread(triple, sizeof(float), sizeof(triple) / sizeof(triple[0]), kernel) != sizeof(triple) / sizeof(triple[0]) || fread(&r, sizeof(float), 1, ref) != 1 )
				return 1;
			const uint16_t exact = ToBf16(triple[1]), mine = ToBf16(triple[0]), theirs = ToBf16(r);
			ref_off += theirs != exact; kernel_off += mine != exact; differ += mine != theirs; n++;
		}
		printf("%s %s: bf16 not the correctly rounded value: kernel %.4f%%, reference %.4f%%; kernel and reference differ at %.4f%%\n", argv[6],
			band == 0 ? "prefill" : "decode ", 100.0 * kernel_off / n, 100.0 * ref_off / n, 100.0 * differ / n);
	}
	return 0;
}
