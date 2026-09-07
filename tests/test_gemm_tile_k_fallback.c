#include "runtime/launch.h"
#include <stdio.h>

static int failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if (!condition)
		++failures;
}

int main(void)
{
	expect(LM_LAUNCH_ERR_SHAPE == (-41), "LM_LAUNCH_ERR_SHAPE is -41");
	expect(LmGemmSelectTileK(64u, 224u) == 32u,  "BF16 224 (routed_up TP16 7*32) -> 32");
	expect(LmGemmSelectTileK(64u, 2112u) == 64u, "BF16 2112 (dense_down TP16 33*64) -> 64");
	expect(LmGemmSelectTileK(64u, 192u) == 64u,  "BF16 192 (expert-w2 TP16 3*64) -> 64");
	expect(LmGemmSelectTileK(64u, 64u) == 64u,   "BF16 64 -> 64");
	expect(LmGemmSelectTileK(64u, 128u) == 64u,  "BF16 128 -> 64");
	expect(LmGemmSelectTileK(64u, 32u) == 32u,   "BF16 32 -> 32");
	expect(LmGemmSelectTileK(64u, 96u) == 32u,   "BF16 96 (3*32) -> 32");
	expect(LmGemmSelectTileK(128u, 2112u) == 32u, "MXFP4 2112 (66*32) -> 32");
	expect(LmGemmSelectTileK(128u, 192u) == 32u,  "MXFP4 192 (6*32) -> 32");
	expect(LmGemmSelectTileK(128u, 224u) == 32u,  "MXFP4 224 (7*32) -> 32");
	expect(LmGemmSelectTileK(128u, 256u) == 128u, "MXFP4 256 -> 128");
	expect(LmGemmSelectTileK(128u, 384u) == 128u, "MXFP4 384 -> 128");
	expect(LmGemmSelectTileK(64u, 31u) == 0u,    "BF16 31 -> 0 (no tile)");
	expect(LmGemmSelectTileK(128u, 31u) == 0u,   "MXFP4 31 -> 0 (no tile)");

	if (failures != 0)
	{
		printf("gemm_tile_k_fallback: %d failure(s)\n", failures);
		return 1;
	}
	printf("gemm_tile_k_fallback: pass\n");
	return 0;
}
