#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/sparkpipe/spark_state_pool.h"

int main(void)
{
	static uint8_t storage[8u * 48u];
	static uint32_t next_free[8u];
	SparkStatePool pool;
	uint32_t slots[8u];
	uint32_t index, again;
	int32_t failures = 0;
	if (SparkStatePoolInitialize(&pool,storage,next_free,8u,48u) != 0)
	{
		printf("  FAIL initialize refused valid storage\n");
		return 1;
	}
	for (index = 0u; index < 8u; ++index)
	{
		slots[index] = SparkStatePoolAcquire(&pool);
		if (slots[index] == SPARK_STATE_POOL_NO_SLOT)
		{
			printf("  FAIL acquire %u ran dry early\n",index);
			failures += 1;
		}
		memset(SparkStatePoolSlot(&pool,slots[index]),
			(int)(index + 1u),48u);
	}
	if (SparkStatePoolAcquire(&pool) != SPARK_STATE_POOL_NO_SLOT)
	{
		printf("  FAIL exhaustion was quiet\n");
		failures += 1;
	}
	for (index = 0u; index < 8u; ++index)
		if (SparkStatePoolSlot(&pool,slots[index])[0u] !=
			(uint8_t)(index + 1u))
		{
			printf("  FAIL slot %u was aliased\n",index);
			failures += 1;
		}
	if (SparkStatePoolRelease(&pool,slots[3u]) != 0 ||
		SparkStatePoolRelease(&pool,slots[3u]) == 0)
	{
		printf("  FAIL double release accepted\n");
		failures += 1;
	}
	again = SparkStatePoolAcquire(&pool);
	if (again != slots[3u])
	{
		printf("  FAIL freed slot %u not reissued (got %u)\n",
			slots[3u],again);
		failures += 1;
	}
	printf("slots 8, exhaustion loud, double release refused, reuse exact\n");
	if (failures != 0)
	{
		printf("\nFAIL (%d)\n",failures);
		return 1;
	}
	printf("\nthe pool hands out every slot once and knows it\n");
	return 0;
}
