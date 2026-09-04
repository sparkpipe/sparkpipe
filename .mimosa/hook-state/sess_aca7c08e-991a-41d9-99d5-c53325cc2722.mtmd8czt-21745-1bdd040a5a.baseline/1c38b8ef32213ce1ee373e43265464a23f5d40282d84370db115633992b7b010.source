#include "runtime/launch.h"
#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if (!condition)
		++failures;
}

static void shape_for(LmLaunchShape *s, uint32_t tokens, uint32_t bits, uint32_t tile_k)
{
	memset(s, 0, sizeof(*s));
	s->tokens = tokens;
	s->top_k = 8u;
	s->expert_count = 256u;
	s->input_dimension = 6144u;
	s->output_dimension = 4096u;
	s->stored_bits = bits;
	s->tile_n = 128u;
	s->tile_k = tile_k;
	s->stages = 2u;
}

int main(void)
{
	LmLaunchShape shape;
	LmLaunchPlan plan;
	uint32_t buckets[5] = { 1u, 128u, 512u, 1024u, 2048u };
	uint32_t i;

	printf("launch planning\n\ntile height per token bucket, FP8\n");
	for (i = 0; i < 5u; ++i)
	{
		uint32_t peak, m_tiles;
		shape_for(&shape, buckets[i], 8u, 128u);
		if (LmLaunchPlanBuild(&shape, 48u, &plan) != LM_LAUNCH_OK)
		{
			printf("    B%-5u did not plan\n", buckets[i]);
			++failures;
			continue;
		}
		peak = LmLaunchPeakRowsPerGroup(&shape);
		m_tiles = (peak + plan.tile_m - 1u) / plan.tile_m;
		printf("    B%-5u peak rows %-4u -> TILE_M %-3u  M tiles %u  shared %6u B  span %uB\n",
			buckets[i], peak, plan.tile_m, m_tiles, plan.shared_bytes, plan.swizzle_span);
		if (m_tiles > 1u && buckets[i] <= 1024u)
		{
			printf("      WEIGHT RE-READ at a supported bucket\n");
			++failures;
		}
	}

	printf("\nswizzle span follows the stored width\n");
	shape_for(&shape, 128u, 8u, 128u);
	LmLaunchPlanBuild(&shape, 48u, &plan);
	expect(plan.swizzle_span == 128u, "8-bit, 128-element tile -> 128-byte span");
	shape_for(&shape, 128u, 7u, 256u);
	LmLaunchPlanBuild(&shape, 48u, &plan);
	expect(plan.swizzle_span == 32u, "7-bit, 256-element tile is 224 bytes -> 32-byte span");
	shape_for(&shape, 128u, 4u, 128u);
	LmLaunchPlanBuild(&shape, 48u, &plan);
	expect(plan.swizzle_span == 64u, "4-bit, 128-element tile is 64 bytes -> 64-byte span");
	{
		const uint32_t expected_128[8] = {0u,1u,2u,3u,4u,5u,6u,7u};
		const uint32_t expected_64[8] = {0u,0u,1u,1u,2u,2u,3u,3u};
		const uint32_t expected_32[8] = {0u,0u,0u,0u,1u,1u,1u,1u};
		uint32_t row,matched = 1u;
		for (row = 0u; row < 8u; row++)
			matched &= (LmSwizzleRowSelector(row,128u,128u) == expected_128[row])
				&& (LmSwizzleRowSelector(row,64u,64u) == expected_64[row])
				&& (LmSwizzleRowSelector(row,32u,32u) == expected_32[row]);
		expect(matched != 0u, "32/64/128-byte TMA selectors follow 128-byte sectors");
	}

	printf("\nshared memory is dynamic, so it may exceed the 48 KB static cap\n");
	shape_for(&shape, 1024u, 16u, 64u);
	expect(LmLaunchPlanBuild(&shape, 48u, &plan) == LM_LAUNCH_OK, "BF16 plans at the production tile depth");
	expect(plan.shared_bytes > 49152u, "and needs more than the static limit allows");
	expect(plan.shared_bytes <= LM_SMEM_SM_TOTAL, "while fitting what the SM has");

	printf("\neach format's declared TILE_K must be swizzleable at its stored width\n");
	{
		struct { const char *name; uint32_t bits; uint32_t tile_k; } table[] = {
			{ "bf16", 16u, 64u }, { "fp8", 8u, 128u }, { "int8", 8u, 128u },
			{ "int7", 7u, 256u }, { "int6", 6u, 128u }, { "nvfp4", 4u, 128u },
		};
		uint32_t i;
		for (i = 0; i < 6u; ++i)
		{
			uint32_t pitch = (table[i].tile_k * table[i].bits) / 8u;
			uint32_t span = LmSwizzleSpanFor(pitch);
			printf("    %-6s %2u bits, TILE_K %3u -> pitch %3u, span %uB\n",
				table[i].name, table[i].bits, table[i].tile_k, pitch, span);
			if (span == 0u) { printf("      no span divides it\n"); ++failures; }
		}
	}
	expect(1, "every format declares a TILE_K its stored width can swizzle");

	printf("\nrejections\n");
	shape_for(&shape, 128u, 8u, 128u);
	shape.stages = 1u;
	expect(LmLaunchPlanBuild(&shape, 48u, &plan) == LM_LAUNCH_ERR_SHAPE,
		"a single stage has no lookahead and is rejected");
	shape_for(&shape, 128u, 8u, 128u);
	expect(LmLaunchPlanBuild(&shape, 0u, &plan) == LM_LAUNCH_ERR_SHAPE,
		"a zero SM count is rejected rather than producing an empty grid");
	shape_for(&shape, 128u, 6u, 32u);
	expect(LmLaunchPlanBuild(&shape, 48u, &plan) == LM_LAUNCH_ERR_MAP,
		"6-bit at a 32-element tile is 24 bytes and no span divides it");

	printf("\ngrid is sized to the machine, not the problem\n");
	shape_for(&shape, 1u, 8u, 128u);
	LmLaunchPlanBuild(&shape, 48u, &plan);
	expect(plan.grid_blocks == 48u, "B1 still launches one CTA per SM");
	expect(plan.block_threads == 256u, "8 warps");

	printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
