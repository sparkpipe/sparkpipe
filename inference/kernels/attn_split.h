#pragma once

#include <stdint.h>

static inline uint32_t LmLatentAttentionContextSplits(uint32_t position_bound,uint32_t split_context_threshold)
{
	return(split_context_threshold != 0u && position_bound >= split_context_threshold ? 1u : 0u);
}
