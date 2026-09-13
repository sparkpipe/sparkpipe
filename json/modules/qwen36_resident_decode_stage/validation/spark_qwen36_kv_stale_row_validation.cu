#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"

/*
 * KV STALE-ROW HAZARD TEST - the empirical half of the speculative-KV audit.
 *
 * The audit question: a verify frame writes speculative draft K/V into the
 * COMMITTED paged cache at rows the following replay may not cover, so does a
 * later decode ever READ stale speculative bytes? Two invariants say no:
 *   (1) within an attention layer the module launches AttnPrepare (which stores
 *       the row's post-norm, post-RoPE K/V at slot_mapping[row]) and only then
 *       AttnDecode (which reads tokens < context_lengths[row]) - same stream,
 *       so the row's own slot is overwritten before it is read;
 *   (2) context_lengths[row] = position + 1, so any row ABOVE the current
 *       position is outside the read window until its own position re-executes.
 * Reading the code proves the ordering; this proves the BEHAVIOUR, and keeps
 * proving it - the design deviates from docs/SPEC_DECODE_REFERENCE_CONTRACTS.md
 * (a) "draft-KV isolation", so the invariants are the whole safety argument and
 * they deserve a regression test rather than a comment.
 *
 * Four runs, identical inputs, only the cache PRE-STATE differs:
 *   A baseline   history rows written, nothing poisoned
 *   B poison the CURRENT row's slot (the hazard) -> must equal A
 *   C poison a slot ABOVE context_length        -> must equal A
 *   D poison a HISTORY row inside the window    -> must DIFFER from A
 * D is the sensitivity control: without it, A==B==C would also be satisfied by a
 * test that reads nothing at all (the mistake that made my earlier preflight
 * report a false FAIL).
 */

#define SPARK_KVTEST_QUERY_HEADS SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT
#define SPARK_KVTEST_KV_HEADS SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT
#define SPARK_KVTEST_HEAD_DIM SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION
#define SPARK_KVTEST_BLOCK_TOKENS SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS
#define SPARK_KVTEST_POSITION 130u
#define SPARK_KVTEST_ROWS 1u

extern "C" cudaError_t SparkQwen36LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen36AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern "C" cudaError_t SparkQwen36LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen36KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern "C" cudaError_t SparkQwen36TpSetGeometry(uint32_t gdn_qk_channels,uint32_t gdn_value_channels,uint32_t gdn_conv_channels,uint32_t gdn_key_heads,uint32_t gdn_value_heads,uint32_t attn_query_heads,uint32_t attn_kv_heads,uint32_t gdn_qk_channel_base,uint32_t gdn_value_channel_base,uint32_t gdn_key_head_base,uint32_t gdn_value_head_base);

static uint16_t SparkKvTestBf16(float value)
{
	uint32_t bits,lsb;
	memcpy(&bits,&value,sizeof(bits));
	lsb = (bits >> 16u) & 1u;
	bits += 0x7fffu + lsb;
	return((uint16_t)(bits >> 16u));
}

/* One deterministic small integer per (kind, position, head, column). */
static float SparkKvTestValue(uint32_t kind, uint32_t position, uint32_t head, uint32_t column)
{
	uint32_t mixed = (kind * 7919u) ^ (position * 104729u) ^ (head * 1299709u) ^ (column * 15485863u);
	return((float)((int32_t)(mixed % 9u) - 4));
}

#define SPARK_KVTEST_CUDA(expression) \
	do { \
		cudaError_t status = (expression); \
		if ( status != cudaSuccess ) \
		{ \
			fprintf(stderr,"%s failed: %s\n",#expression,cudaGetErrorString(status)); \
			return(2); \
		} \
	} while (0)

int main(void)
{
	const uint32_t position = SPARK_KVTEST_POSITION;
	const uint32_t context_length = position + 1u;
	const uint64_t layer_stride = (uint64_t)SPARK_KVTEST_BLOCK_TOKENS * SPARK_KVTEST_KV_HEADS * SPARK_KVTEST_HEAD_DIM * 2u;
	const uint64_t block_stride = layer_stride;
	const uint32_t block_count = ((position + 8u) / SPARK_KVTEST_BLOCK_TOKENS) + 1u;
	const uint64_t cache_elements = (uint64_t)block_count * block_stride;
	const uint64_t query_elements = (uint64_t)SPARK_KVTEST_ROWS * 2u * SPARK_KVTEST_QUERY_HEADS * SPARK_KVTEST_HEAD_DIM;
	const uint64_t kv_elements = (uint64_t)SPARK_KVTEST_ROWS * SPARK_KVTEST_KV_HEADS * SPARK_KVTEST_HEAD_DIM;
	const uint64_t out_elements = (uint64_t)SPARK_KVTEST_ROWS * SPARK_KVTEST_QUERY_HEADS * SPARK_KVTEST_HEAD_DIM;
	SparkQwen36AttnLayerWeights weights;
	SparkQwen36KvBlockTableView table;
	uint16_t *host_cache,*host_query,*host_kv,*host_norm,*host_out[5];
	uint32_t *host_blocks,*host_counts;
	void *cache = 0,*query = 0,*key = 0,*value = 0,*q_norm = 0,*k_norm = 0,*out = 0;
	uint32_t *slot_mapping = 0,*context_lengths = 0,*row_lanes = 0,*device_blocks = 0,*device_counts = 0;
	uint64_t *row_positions = 0;
	uint32_t run,head,column,kv_head,slot_position,failures = 0u;
	uint64_t index;

	SPARK_KVTEST_CUDA(SparkQwen36TpSetGeometry(0u,0u,0u,0u,0u,SPARK_KVTEST_QUERY_HEADS,SPARK_KVTEST_KV_HEADS,0u,0u,0u,0u));
	host_cache = (uint16_t *)calloc((size_t)cache_elements,sizeof(uint16_t));
	host_query = (uint16_t *)calloc((size_t)query_elements,sizeof(uint16_t));
	host_kv = (uint16_t *)calloc((size_t)kv_elements,sizeof(uint16_t));
	host_norm = (uint16_t *)calloc(SPARK_KVTEST_HEAD_DIM,sizeof(uint16_t));
	host_blocks = (uint32_t *)calloc(block_count,sizeof(uint32_t));
	host_counts = (uint32_t *)calloc(1u,sizeof(uint32_t));
	if ( host_cache == 0 || host_query == 0 || host_kv == 0 || host_norm == 0 || host_blocks == 0 || host_counts == 0 )
		return(2);
	for (run = 0u; run < 5u; run++)
	{
		host_out[run] = (uint16_t *)calloc((size_t)out_elements,sizeof(uint16_t));
		if ( host_out[run] == 0 )
			return(2);
	}
	for (column = 0u; column < SPARK_KVTEST_HEAD_DIM; column++)
		host_norm[column] = SparkKvTestBf16(1.0f);
	for (head = 0u; head < SPARK_KVTEST_QUERY_HEADS; head++)
		for (column = 0u; column < SPARK_KVTEST_HEAD_DIM; column++)
			host_query[((uint64_t)head * 2u * SPARK_KVTEST_HEAD_DIM) + column] = SparkKvTestBf16(SparkKvTestValue(0u,position,head,column));
	for (kv_head = 0u; kv_head < SPARK_KVTEST_KV_HEADS; kv_head++)
		for (column = 0u; column < SPARK_KVTEST_HEAD_DIM; column++)
			host_kv[((uint64_t)kv_head * SPARK_KVTEST_HEAD_DIM) + column] = SparkKvTestBf16(SparkKvTestValue(1u,position,kv_head,column));
	for (index = 0u; index < block_count; index++)
		host_blocks[index] = (uint32_t)index;
	host_counts[0] = block_count;

	SPARK_KVTEST_CUDA(cudaMalloc(&cache,(size_t)cache_elements * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc(&query,(size_t)query_elements * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc(&key,(size_t)kv_elements * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc(&value,(size_t)kv_elements * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc(&q_norm,SPARK_KVTEST_HEAD_DIM * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc(&k_norm,SPARK_KVTEST_HEAD_DIM * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc(&out,(size_t)out_elements * sizeof(uint16_t)));
	SPARK_KVTEST_CUDA(cudaMalloc((void **)&slot_mapping,SPARK_KVTEST_ROWS * sizeof(uint32_t)));
	SPARK_KVTEST_CUDA(cudaMalloc((void **)&context_lengths,SPARK_KVTEST_ROWS * sizeof(uint32_t)));
	SPARK_KVTEST_CUDA(cudaMalloc((void **)&row_lanes,SPARK_KVTEST_ROWS * sizeof(uint32_t)));
	SPARK_KVTEST_CUDA(cudaMalloc((void **)&row_positions,SPARK_KVTEST_ROWS * sizeof(uint64_t)));
	SPARK_KVTEST_CUDA(cudaMalloc((void **)&device_blocks,block_count * sizeof(uint32_t)));
	SPARK_KVTEST_CUDA(cudaMalloc((void **)&device_counts,sizeof(uint32_t)));

	memset(&weights,0,sizeof(weights));
	weights.query_norm_weight_bf16 = q_norm;
	weights.key_norm_weight_bf16 = k_norm;
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	table.descriptor_bytes = sizeof(table);
	table.block_token_count = SPARK_KVTEST_BLOCK_TOKENS;
	table.lane_count = 1u;
	table.lane_stride = block_count;
	table.lane_capacity = block_count;
	table.physical_block_indices = device_blocks;
	table.lane_physical_block_counts = device_counts;
	table.host_physical_block_indices = host_blocks;
	table.host_lane_physical_block_counts = host_counts;

	{
		uint32_t host_slot = (position / SPARK_KVTEST_BLOCK_TOKENS) * SPARK_KVTEST_BLOCK_TOKENS + (position % SPARK_KVTEST_BLOCK_TOKENS);
		uint32_t host_context = context_length,host_lane = 0u;
		uint64_t host_position = position;
		SPARK_KVTEST_CUDA(cudaMemcpy(slot_mapping,&host_slot,sizeof(host_slot),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(context_lengths,&host_context,sizeof(host_context),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(row_lanes,&host_lane,sizeof(host_lane),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(row_positions,&host_position,sizeof(host_position),cudaMemcpyHostToDevice));
	}
	SPARK_KVTEST_CUDA(cudaMemcpy(query,host_query,(size_t)query_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
	SPARK_KVTEST_CUDA(cudaMemcpy(key,host_kv,(size_t)kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
	SPARK_KVTEST_CUDA(cudaMemcpy(value,host_kv,(size_t)kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
	SPARK_KVTEST_CUDA(cudaMemcpy(q_norm,host_norm,SPARK_KVTEST_HEAD_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice));
	SPARK_KVTEST_CUDA(cudaMemcpy(k_norm,host_norm,SPARK_KVTEST_HEAD_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice));

	printf("geometry        = q_heads=%u kv_heads=%u head_dim=%u block_tokens=%u position=%u context=%u blocks=%u\n",
		SPARK_KVTEST_QUERY_HEADS,SPARK_KVTEST_KV_HEADS,SPARK_KVTEST_HEAD_DIM,SPARK_KVTEST_BLOCK_TOKENS,
		position,context_length,block_count);

	/* Run 1 repeats run 0 byte for byte: without that control, "DIFFERS" cannot be
	 * distinguished from a nondeterministic kernel or a dirty output buffer. */
	for (run = 0u; run < 5u; run++)
	{
		/* history rows 0..position-1 hold a committed pattern; the current row's
		 * slot starts EMPTY except in the poison run. */
		memset(host_cache,0,(size_t)cache_elements * sizeof(uint16_t));
		for (slot_position = 0u; slot_position < position; slot_position++)
			for (kv_head = 0u; kv_head < SPARK_KVTEST_KV_HEADS; kv_head++)
				for (column = 0u; column < SPARK_KVTEST_HEAD_DIM; column++)
				{
					uint64_t base = ((uint64_t)(slot_position / SPARK_KVTEST_BLOCK_TOKENS) * block_stride) +
						((uint64_t)(slot_position % SPARK_KVTEST_BLOCK_TOKENS) * 2u * SPARK_KVTEST_KV_HEADS * SPARK_KVTEST_HEAD_DIM) +
						((uint64_t)kv_head * SPARK_KVTEST_HEAD_DIM);
					host_cache[base + column] = SparkKvTestBf16(SparkKvTestValue(2u,slot_position,kv_head,column));
					host_cache[base + ((uint64_t)SPARK_KVTEST_KV_HEADS * SPARK_KVTEST_HEAD_DIM) + column] =
						SparkKvTestBf16(SparkKvTestValue(3u,slot_position,kv_head,column));
				}
		if ( run > 1u )
		{
			uint32_t poison_position = run == 2u ? position : (run == 3u ? position + 3u : position - 2u);
			for (kv_head = 0u; kv_head < SPARK_KVTEST_KV_HEADS; kv_head++)
				for (column = 0u; column < SPARK_KVTEST_HEAD_DIM; column++)
				{
					uint64_t base = ((uint64_t)(poison_position / SPARK_KVTEST_BLOCK_TOKENS) * block_stride) +
						((uint64_t)(poison_position % SPARK_KVTEST_BLOCK_TOKENS) * 2u * SPARK_KVTEST_KV_HEADS * SPARK_KVTEST_HEAD_DIM) +
						((uint64_t)kv_head * SPARK_KVTEST_HEAD_DIM);
					host_cache[base + column] = SparkKvTestBf16(64.0f + (float)(column % 7u));
					host_cache[base + ((uint64_t)SPARK_KVTEST_KV_HEADS * SPARK_KVTEST_HEAD_DIM) + column] = SparkKvTestBf16(-64.0f - (float)(column % 5u));
				}
		}
		/* q_fused is an IN/OUT buffer - AttnPrepare writes the normed, RoPE'd query
		 * back into it - so every run must start from the pristine copy or each
		 * successive run re-normalizes an already-transformed query. That is what
		 * made the first version of this test nondeterministic. */
		SPARK_KVTEST_CUDA(cudaMemcpy(query,host_query,(size_t)query_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(key,host_kv,(size_t)kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(value,host_kv,(size_t)kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(cache,host_cache,(size_t)cache_elements * sizeof(uint16_t),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(device_blocks,host_blocks,block_count * sizeof(uint32_t),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(cudaMemcpy(device_counts,host_counts,sizeof(uint32_t),cudaMemcpyHostToDevice));
		SPARK_KVTEST_CUDA(SparkQwen36LaunchAttnPrepare(0,query,key,value,&weights,cache,slot_mapping,row_positions,SPARK_KVTEST_ROWS,0u,layer_stride,block_stride,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON));
		SPARK_KVTEST_CUDA(SparkQwen36LaunchAttnDecode(0,query,cache,&table,row_lanes,context_lengths,out,SPARK_KVTEST_ROWS,0u,layer_stride,block_stride));
		SPARK_KVTEST_CUDA(cudaDeviceSynchronize());
		SPARK_KVTEST_CUDA(cudaMemcpy(host_out[run],out,(size_t)out_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost));
		{
			uint64_t checksum = 0u,nonzero = 0u;
			for (index = 0u; index < out_elements; index++)
			{
				checksum = (checksum * 1000003u) + host_out[run][index];
				nonzero += host_out[run][index] != 0u ? 1u : 0u;
			}
			printf("run %u (%s): checksum=%016llx nonzero=%llu/%llu\n",run,
				run == 0u ? "baseline" : run == 1u ? "baseline REPEAT" : run == 2u ? "poison current row" :
				run == 3u ? "poison above context" : "poison history row",
				(unsigned long long)checksum,(unsigned long long)nonzero,(unsigned long long)out_elements);
		}
	}

	{
		int deterministic = memcmp(host_out[0],host_out[1],(size_t)out_elements * sizeof(uint16_t)) == 0;
		int same_current = memcmp(host_out[0],host_out[2],(size_t)out_elements * sizeof(uint16_t)) == 0;
		int same_above = memcmp(host_out[0],host_out[3],(size_t)out_elements * sizeof(uint16_t)) == 0;
		int same_history = memcmp(host_out[0],host_out[4],(size_t)out_elements * sizeof(uint16_t)) == 0;
		printf("\nA' repeat of the baseline               : %s (expect IDENTICAL - determinism control)\n",
			deterministic ? "IDENTICAL" : "DIFFERS");
		printf("B  poison the CURRENT row's slot        : %s (expect IDENTICAL - Prepare overwrites before Decode reads)\n",
			same_current ? "IDENTICAL" : "DIFFERS");
		printf("C  poison a slot ABOVE context_length   : %s (expect IDENTICAL - outside the read window)\n",
			same_above ? "IDENTICAL" : "DIFFERS");
		printf("D  poison a HISTORY row inside the window: %s (expect DIFFERS - sensitivity control)\n",
			same_history ? "IDENTICAL" : "DIFFERS");
		if ( deterministic == 0 ) { printf("INCONCLUSIVE: the harness is not deterministic; every other verdict is void\n"); failures++; }
		else
		{
			if ( same_current == 0 ) { printf("FAIL: the decode read stale bytes at its own row\n"); failures++; }
			if ( same_above == 0 ) { printf("FAIL: the decode read a row above its context length\n"); failures++; }
			if ( same_history != 0 ) { printf("FAIL: the test is not sensitive to cache content at all\n"); failures++; }
		}
		printf("\nresult          = %s (%u failing checks)\n",failures == 0u ? "NO STALE-ROW HAZARD" : "SEE ABOVE",failures);
	}
	return(failures == 0u ? 0 : 1);
}
