#pragma once

#include <stdint.h>
#include <stdio.h>

static inline void LmSlowPath(const char *site, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension)
{
	static struct { const char *site; uint32_t input_dimension, output_dimension; } seen[128];
	static uint32_t count;
	for ( uint32_t i = 0u; i < count; i++ )
		if ( seen[i].site == site && seen[i].input_dimension == input_dimension && seen[i].output_dimension == output_dimension )
			return;
	if ( count < 128u )
	{
		seen[count].site = site;
		seen[count].input_dimension = input_dimension;
		seen[count].output_dimension = output_dimension;
		count++;
	}
	fprintf(stderr, "SLOW-PATH %s rows=%u input=%u output=%u\n", site, rows, input_dimension, output_dimension);
}
