#include "runtime/device/nvrm/nvrm_occupancy.h"

static uint32_t OccupancyRound(uint32_t value, uint32_t unit)
{
	return(unit != 0u ? (value + unit - 1u) / unit * unit : value);
}

static uint32_t OccupancyLeast(uint32_t first, uint32_t second)
{
	return(first < second ? first : second);
}

uint32_t SparkNvrmOccupancy(const SparkNvrmOccupancyLimits *limits, uint32_t threads, uint32_t registers_per_thread, uint32_t shared_bytes)
{
	const uint32_t warps = (threads + SPARK_NVRM_WARP_THREADS - 1u) / SPARK_NVRM_WARP_THREADS;
	uint32_t blocks;
	if ( limits == 0 || threads == 0u || warps > limits->warps_per_sm )
		return(0u);
	blocks = OccupancyLeast(limits->blocks_per_sm,limits->warps_per_sm / warps);
	if ( registers_per_thread != 0u )
	{
		const uint32_t per_warp = OccupancyRound(registers_per_thread * SPARK_NVRM_WARP_THREADS,limits->register_unit);
		blocks = OccupancyLeast(blocks,limits->registers_per_sm / per_warp / warps);
	}
	if ( shared_bytes != 0u )
	{
		const uint32_t per_block = OccupancyRound(shared_bytes,limits->shared_unit);
		blocks = OccupancyLeast(blocks,per_block <= limits->shared_per_sm ? limits->shared_per_sm / per_block : 0u);
	}
	return(blocks);
}
