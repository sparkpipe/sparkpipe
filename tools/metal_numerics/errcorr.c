#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
	const long rows = atol(argv[3]), split = atol(argv[4]), width = argc > 6 ? atol(argv[6]) : 24 * 256;
	FILE *kernel = fopen(argv[1], "rb"), *ref = fopen(argv[2], "rb");
	float triple[3], r;
	for ( int band = 0; band < 2; band++ )
	{
		const long lo = band == 0 ? 0 : split, hi = band == 0 ? split : rows;
		double kk = 0, rr = 0, kr = 0, dd = 0, n = 0;
		for ( long i = lo * width; i < hi * width; i++ )
		{
			if ( fread(triple, sizeof(float), sizeof(triple) / sizeof(triple[0]), kernel) != sizeof(triple) / sizeof(triple[0]) || fread(&r, sizeof(float), 1, ref) != 1 )
				return 1;
			const double scale = triple[2] > 0 ? 16777216.0 / triple[2] : 0, ek = (triple[0] - triple[1]) * scale, er = (r - triple[1]) * scale;
			kk += ek * ek; rr += er * er; kr += ek * er; dd += (ek - er) * (ek - er); n++;
		}
		printf("%s %s: error vs float64 in 2^-24 sum p|v|: kernel rms %.2f, reference rms %.2f, kernel - reference rms %.2f, correlation %.3f\n",
			argv[5], band == 0 ? "prefill" : "decode ", sqrt(kk / n), sqrt(rr / n), sqrt(dd / n), kr / sqrt(kk * rr));
	}
	return 0;
}
