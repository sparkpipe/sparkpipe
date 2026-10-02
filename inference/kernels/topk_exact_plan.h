#pragma once

#include <stdint.h>

#define LM_TOPK_EXACT_CHUNK 16384u
#define LM_TOPK_EXACT_CHUNKED_ROWS 64u
#define LM_TOPK_EXACT_SELECTION_FLOATS (1u << 26)

static inline uint64_t LmTopkExactCandidateEntries(uint32_t n, uint32_t k, uint32_t chunk)
{
	return(chunk > k && n > chunk ? (uint64_t)((n + chunk - 1u) / chunk) * k : 0u);
}

static inline uint64_t LmTopkExactScratchEntries(uint32_t rows, uint32_t n, uint32_t k)
{
	uint32_t chunked = rows < LM_TOPK_EXACT_CHUNKED_ROWS ? rows : LM_TOPK_EXACT_CHUNKED_ROWS;
	return(2u * (uint64_t)chunked * LmTopkExactCandidateEntries(n,k,LM_TOPK_EXACT_CHUNK));
}

static inline uint32_t LmTopkExactSelectionRows(uint32_t row_capacity, uint32_t n)
{
	uint32_t rows = n != 0u ? LM_TOPK_EXACT_SELECTION_FLOATS / n : row_capacity;
	if ( rows == 0u )
		rows = 1u;
	return(rows < row_capacity ? rows : row_capacity);
}
