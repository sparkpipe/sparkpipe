#include "inference/kernels/topk_exact_plan.h"

#include <stdio.h>

static int failures;

static void expect(int condition, const char *what)
{
	if ( !condition )
	{
		printf("FAIL %s\n", what);
		failures++;
	}
}

int main(void)
{
	static const uint32_t capacities[] = {1u, 17u, 256u, 1024u, 4096u, 8192u};
	static const uint32_t positions[] = {1u, 4096u, 65536u, 262144u, 262145u, 524288u, 737280u, 1048576u};
	static const uint32_t degrees[] = {1u, 4u, 16u};
	uint32_t c, p, d;
	expect(LmTopkExactSelectionRows(4096u, 262144u) == 256u && LmTopkExactGatherSelectionRows(4096u, 262144u, 16u) == 256u,
		"at 262144 positions the gather sizing keeps the 2^26-float buffer of 256 rows");
	expect(LmTopkExactSelectionRows(4096u, 737280u) == 91u && LmTopkExactGatherSelectionRows(4096u, 737280u, 16u) == 256u,
		"at 737280 positions a 4096-row TP16 gather wave still gets its 256 rows");
	for ( c = 0u; c < sizeof(capacities) / sizeof(capacities[0]); c++ )
		for ( p = 0u; p < sizeof(positions) / sizeof(positions[0]); p++ )
			for ( d = 0u; d < sizeof(degrees) / sizeof(degrees[0]); d++ )
			{
				const uint32_t rows = LmTopkExactGatherSelectionRows(capacities[c], positions[p], degrees[d]);
				const uint32_t slice = (capacities[c] + degrees[d] - 1u) / degrees[d];
				expect(rows >= slice && rows <= capacities[c] && rows >= LmTopkExactSelectionRows(capacities[c], positions[p]),
					"every gather wave's rows fit and the buffer never exceeds the row capacity");
			}
	if ( failures == 0 )
		printf("test_topk_exact_plan: gather waves never score more rows than the selection buffer holds: ok\n");
	return(failures == 0 ? 0 : 1);
}
