#pragma once

#include <stdint.h>

static inline uint32_t SparkSpeculationDepthCapNext(uint32_t cap,uint32_t proposed,uint32_t accepted,uint32_t cap_max)
{
	uint32_t next;
	if ( cap_max == 0u )
		return(0u);
	if ( cap == 0u || cap > cap_max )
		cap = cap_max;
	if ( proposed != 0u && accepted >= proposed )
		next = cap > cap_max / 2u ? cap_max : 2u * cap;
	else
		next = accepted + 1u;
	return(next > cap_max ? cap_max : next);
}
