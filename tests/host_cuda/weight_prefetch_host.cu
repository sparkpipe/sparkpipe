#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <string.h>

#include "inference/kernels/weight_prefetch.cuh"

static int prefetch_failures;

static void PrefetchCheck(int condition, const char *message)
{
	if ( condition == 0 )
	{
		printf("FAIL %s\n", message);
		prefetch_failures++;
	}
}

int main(void)
{
	static LmWeightPrefetchGroup host[2], device[2];
	static uint8_t weights[3][4096] __attribute__((aligned(16)));
	LmWeightPrefetchPlan plan;
	uint32_t resume;
	const cudaStream_t stream = 1;
	memset(&plan, 0, sizeof(plan));
	plan.host = host;
	plan.device = device;
	plan.capacity = 2u;
	LmWeightPrefetchArm(&plan, stream);
	PrefetchCheck(LmWeightPrefetchSlot(stream, &resume) == &device[0], "the first kernel gets group 0");
	LmWeightPrefetchNote(stream, weights[0], sizeof(weights[0]));
	PrefetchCheck(LmWeightPrefetchSlot(stream, &resume) == &device[1], "the second kernel gets group 1");
	LmWeightPrefetchNote(stream, weights[1], sizeof(weights[1]));
	PrefetchCheck(LmWeightPrefetchSlot(stream, &resume) == 0 && plan.dropped == 1u, "a third kernel is dropped from a full plan");
	LmWeightPrefetchNote(stream, weights[2], sizeof(weights[2]));
	PrefetchCheck(host[1].count == 1u && host[1].address[0] == (uint64_t)(uintptr_t)weights[1] && host[1].total == sizeof(weights[1]),
		"a dropped kernel's weights are not added to the previous kernel's group");
	PrefetchCheck(host[0].count == 1u && host[0].address[0] == (uint64_t)(uintptr_t)weights[0], "group 0 still holds only the first kernel's weights");
	PrefetchCheck(LmWeightPrefetchDisarm(&plan, 1u) == cudaSuccess && device[1].count == 1u && device[1].total == sizeof(weights[1]),
		"the published plan carries the two planned groups unchanged");
	if ( prefetch_failures != 0 )
	{
		printf("FAIL weight_prefetch_host: %d failures\n", prefetch_failures);
		return 1;
	}
	printf("PASS weight_prefetch_host: a kernel dropped from a full prefetch plan adds nothing to the previous group (0 failures)\n");
	return 0;
}
