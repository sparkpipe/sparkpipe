#ifndef SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_CUDA_H
#define SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_CUDA_H

#include <cuda_runtime.h>
#include <stdint.h>

#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_k3_pool_sizing.h"
#include "inference/llms/kimi_k3/slice.cuh"


#define SPARK_K3_DISPATCH_OK 0
#define SPARK_K3_DISPATCH_ERR_ARGUMENT -1
#define SPARK_K3_DISPATCH_ERR_CUDA -2
#define SPARK_K3_DISPATCH_ERR_REGISTER -3
#define SPARK_K3_DISPATCH_ERR_BIND -4

typedef struct SparkK3StepInput
{
	const uint16_t *hidden_in;
	const uint32_t *positions;
	const uint32_t *context_length;
	const uint32_t *sequence_of_row;
	const uint32_t *sequence_row_begin;
	const uint32_t *kda_state_index;
	uint32_t *route_expert;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
} SparkK3StepInput;

typedef struct SparkK3Dispatch
{
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t kda_count;
	uint32_t mla_count;
	uint32_t sequences;
	uint32_t max_rows;
	uint32_t routes_capacity;
	uint32_t kv_pages_per_view;
	uint64_t kv_page_bytes;
	K3LayerWeights *weights;
	K3SliceState *slice_state;
	K3LayerBuffers *buffers;
	K3LayerBuffers *buffers_host;
	LmKvView *mla_cache;
	LmKvAccessError *access_error;
	uint8_t *kv_pool;
	uint32_t *page_table;
	uint8_t *kda_state_pool;
	uint16_t *kda_q_window_pool;
	uint16_t *kda_k_window_pool;
	uint16_t *kda_v_window_pool;
	uint8_t *scratch;
	size_t scratch_bytes;
	int device;
} SparkK3Dispatch;

int32_t SparkK3DispatchCreate(SparkK3Dispatch *d, const SparkK3PoolSizing *sizing,
	uint32_t sequences, uint32_t max_rows, uint32_t kv_pages_per_view,
	uint64_t kv_page_bytes, int device);
void SparkK3DispatchDestroy(SparkK3Dispatch *d);

int32_t SparkK3DispatchRegisterPack(SparkK3Pack *pack);
void SparkK3DispatchUnregisterPack(SparkK3Pack *pack);

int32_t SparkK3DispatchBindWeights(SparkK3Dispatch *d, SparkK3Pack *pack,
	SparkK3BoundLayer *bounds, uint32_t layer_count);

int32_t SparkK3DispatchStep(SparkK3Dispatch *d, const SparkK3StepInput *in,
	uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows,
	uint32_t context, uint32_t multiprocessors, cudaStream_t stream);

#endif
