#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

static const uint16_t *Map(const char *path, uint64_t *bytes)
{
	struct stat st;
	const int fd = open(path, O_RDONLY);
	void *data;
	if ( fd < 0 || fstat(fd, &st) != 0 )
	{
		fprintf(stderr, "cannot open %s\n", path);
		exit(1);
	}
	*bytes = (uint64_t)st.st_size;
	data = mmap(0, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if ( data == MAP_FAILED )
		exit(1);
	return (const uint16_t *)data;
}

static float Bf16(uint16_t bits)
{
	uint32_t word = (uint32_t)bits << 16;
	float value;
	memcpy(&value, &word, sizeof(value));
	return value;
}

static int CompareDouble(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y ? 1 : 0;
}

static double Median(const double *values, uint32_t count)
{
	double *copy = (double *)malloc(count * sizeof(double)), result;
	memcpy(copy, values, count * sizeof(double));
	qsort(copy, count, sizeof(double), CompareDouble);
	result = count % 2u ? copy[count / 2u] : 0.5 * (copy[count / 2u - 1u] + copy[count / 2u]);
	free(copy);
	return result;
}

int main(int argc, char **argv)
{
	uint32_t L, H, P, split, draws, candidates, layer, i, j, p, band;
	const uint16_t *reference, **files;
	uint64_t bytes, reference_stride, *stride;
	double **error;
	char **names;
	int first[2][16];
	if ( argc < 8 )
	{
		fprintf(stderr, "usage: %s LAYERS HIDDEN POSITIONS SPLIT REF_A.hidden DRAWS name=path...\n", argv[0]);
		return 2;
	}
	L = (uint32_t)atoi(argv[1]); H = (uint32_t)atoi(argv[2]); P = (uint32_t)atoi(argv[3]); split = (uint32_t)atoi(argv[4]);
	reference = Map(argv[5], &bytes);
	reference_stride = bytes / 2u / L;
	draws = (uint32_t)atoi(argv[6]);
	candidates = (uint32_t)(argc - 7) - draws;
	files = (const uint16_t **)calloc(argc, sizeof(*files));
	stride = (uint64_t *)calloc(argc, sizeof(*stride));
	names = (char **)calloc(argc, sizeof(*names));
	error = (double **)calloc(argc, sizeof(*error));
	for ( i = 0u; i < draws + candidates; i++ )
	{
		char *eq = strchr(argv[7 + i], '=');
		names[i] = argv[7 + i];
		*eq = 0;
		files[i] = Map(eq + 1, &bytes);
		stride[i] = bytes / 2u / L;
		error[i] = (double *)malloc(P * sizeof(double));
	}
	memset(first, 0xff, sizeof(first));
	printf("per layer: median over positions of |x - a| / |a| as a ratio to the largest median among the %u draws; then the share of positions where x is further from a than every draw (a further draw would sit near %.2f)\n", draws, 1.0 / (draws + 1.0));
	for ( layer = 0u; layer < L; layer++ )
	{
		const uint64_t base = (uint64_t)layer * reference_stride;
		for ( p = 0u; p < P; p++ )
		{
			double norm = 0.0;
			for ( j = 0u; j < H; j++ )
			{
				const double value = Bf16(reference[base + (uint64_t)p * H + j]);
				norm += value * value;
			}
			norm = sqrt(norm) + 1e-30;
			for ( i = 0u; i < draws + candidates; i++ )
			{
				double distance = 0.0;
				const uint64_t offset = (uint64_t)layer * stride[i] + (uint64_t)p * H;
				for ( j = 0u; j < H; j++ )
				{
					const double delta = (double)Bf16(files[i][offset + j]) - Bf16(reference[base + (uint64_t)p * H + j]);
					distance += delta * delta;
				}
				error[i][p] = sqrt(distance) / norm;
			}
		}
		printf("%2u", layer);
		for ( band = 0u; band < 2u; band++ )
		{
			const uint32_t lo = band == 0u ? 0u : split, hi = band == 0u ? split : P;
			double top = 0.0, low = 1e30;
			if ( hi <= lo )
				continue;
			for ( i = 0u; i < draws; i++ )
			{
				const double m = Median(error[i] + lo, hi - lo);
				top = m > top ? m : top;
				low = m < low ? m : low;
			}
			printf(" | %s band %.2f..1.00", band == 0u ? "prefill" : "decode", low / top);
			for ( i = draws; i < draws + candidates; i++ )
			{
				const double ratio = Median(error[i] + lo, hi - lo) / top;
				uint32_t above = 0u;
				for ( p = lo; p < hi; p++ )
				{
					uint32_t k, worse = 1u;
					for ( k = 0u; k < draws; k++ )
						worse &= error[i][p] > error[k][p] ? 1u : 0u;
					above += worse;
				}
				printf(" %s %.2f/%.2f", names[i], ratio, (double)above / (hi - lo));
				if ( ratio > 1.0 && first[band][i] < 0 )
					first[band][i] = (int)layer;
			}
		}
		printf("\n");
		fflush(stdout);
	}
	for ( band = 0u; band < 2u; band++ )
		for ( i = draws; i < draws + candidates; i++ )
			printf("first %s layer above the draws' band, %s: %d\n", band == 0u ? "prefill" : "decode", names[i], first[band][i]);
	return 0;
}
