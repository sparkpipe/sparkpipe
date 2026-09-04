#include "ring/sideband.h"
#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if (!condition)
		++failures;
}

int main(void)
{
	LmSidebandHeader header;
	LmTapPlan plan;
	uint32_t local[8];

	printf("sidebands\n\nstaleness is three questions, not one\n");
	LmSidebandInitialiseHeader(&header, LM_SIDEBAND_INDEX_SHARE, 3u, 8u, 42u, 128u, 2048u, 4u);
	expect(LmSidebandAcceptable(&header, LM_SIDEBAND_INDEX_SHARE, 42u, 9u, 4u) == LM_SIDEBAND_OK,
		"same group, same generation, right kind: accepted");
	expect(LmSidebandAcceptable(&header, LM_SIDEBAND_INDEX_SHARE, 43u, 9u, 4u) == LM_SIDEBAND_ERR_STALE,
		"a selection from the previous step is rejected");
	expect(LmSidebandAcceptable(&header, LM_SIDEBAND_INDEX_SHARE, 42u, 12u, 4u) == LM_SIDEBAND_ERR_STALE,
		"a selection from another layer group is rejected");
	expect(LmSidebandAcceptable(&header, LM_SIDEBAND_HIDDEN_TAP, 42u, 9u, 4u) == LM_SIDEBAND_ERR_SHAPE,
		"a payload of the wrong kind is rejected");

	printf("\nexport only when the neighbour is still inside the group\n");
	expect(LmSidebandShouldExportIndexShare(8u, 4u) == 1, "layer 8 of a 4-group: next rank continues it");
	expect(LmSidebandShouldExportIndexShare(11u, 4u) == 0, "layer 11 ends the group: nothing to send");
	expect(LmSidebandShouldExportIndexShare(8u, 0u) == 0, "no sharing configured: nothing to send");

	printf("\ntaps are sparse across ranks\n");
	memset(&plan, 0, sizeof(plan));
	plan.tap_layer[0] = 2u;
	plan.tap_layer[1] = 40u;
	plan.tap_layer[2] = 70u;
	plan.tap_count = 3u;
	plan.consumer_rank = 0u;
	expect(LmSidebandTapsOnRank(&plan, 0u, 6u, local, 8u) == 1u, "rank 0 owns one tapped layer");
	expect(local[0] == 2u, "and it is local layer 2");
	expect(LmSidebandTapsOnRank(&plan, 12u, 6u, local, 8u) == 0u, "a rank owning none does no work");

	printf("\ndrafting placement is a field, not the last rank\n");
	expect(LmSidebandRankDrafts(&plan, 0u) == 1,
		"GLM 5.2 drafts on rank 0, whose three dense layers leave it slack");
	expect(LmSidebandRankDrafts(&plan, 12u) == 0,
		"and not on the last rank, which the old code assumed");
	plan.consumer_rank = 12u;
	expect(LmSidebandRankDrafts(&plan, 12u) == 1,
		"a model whose early layers are expensive sets it the other way");

	printf("\npayload sizing fails before the copy, not during it\n");
	{
		uint64_t bytes = 0u;
		LmSidebandInitialiseHeader(&header, LM_SIDEBAND_INDEX_SHARE, 0u, 0u, 1u, 128u, 2048u, 4u);
		expect(LmSidebandPayloadBytes(&header, &bytes) == LM_SIDEBAND_OK && bytes == 1048576u,
			"128 rows x 2048 positions x 4 bytes = 1 MB");
		LmSidebandInitialiseHeader(&header, LM_SIDEBAND_INDEX_SHARE, 0u, 0u, 1u, 0u, 0u, 0u);
		expect(LmSidebandPayloadBytes(&header, &bytes) == LM_SIDEBAND_ERR_SHAPE,
			"a zero-element payload is a shape error, not a zero-byte send");
	}

	printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
