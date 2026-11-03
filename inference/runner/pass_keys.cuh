#pragma once

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "inference/kernels/kv_shard.cuh"

typedef void (*LmPassCollective)(void *context, void *stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *send, void *receive);

struct LmPassKeys
{
	LmKvShardView store;
	const uint32_t *store_sequence_of_row;
	const uint32_t *pass_positions;
	uint32_t pass_tokens;
	uint32_t first_key;
	const uint32_t *older_sequence;
	const uint32_t *older_offset;
	const uint32_t *older_context;
	uint8_t *older_send;
	uint8_t *older_receive;
	uint64_t older_capacity;
	LmPassCollective collective;
	void *context;
};

template<uint32_t SLOT_BYTES>
__global__ static void LmPassOlderKeysKernel(SparkKvShard shard, const uint8_t *gathered, uint32_t first_local, uint32_t rows_per_rank, uint32_t keys, uint8_t *latent_keys)
{
	static_assert(SLOT_BYTES % 16u == 0u, "a key moves as whole 16-byte words");
	for (uint64_t slot = blockIdx.x; slot < (uint64_t)shard.degree * rows_per_rank; slot += gridDim.x)
	{
		SparkKvShard owner = shard;
		uint32_t position;
		owner.rank = (uint32_t)(slot / rows_per_rank);
		position = SparkKvShardLocalPosition(owner,first_local + (uint32_t)(slot % rows_per_rank));
		if ( position >= keys )
			continue;
		for (uint32_t index = threadIdx.x; index < SLOT_BYTES / 16u; index += blockDim.x)
			((uint4 *)(latent_keys + (uint64_t)position * SLOT_BYTES))[index] = ((const uint4 *)(gathered + slot * SLOT_BYTES))[index];
	}
}

template<class Geometry, uint32_t THREADS>
static int32_t LmPassKeysGather(const LmPassKeys *x, uint8_t *latent_keys, cudaStream_t stream)
{
	const uint32_t degree = x->store.shard.degree;
	uint32_t older_rows, chunk_rows, first_local;
	if ( degree == 0u )
		return(LM_LAUNCH_OK);
	if ( x->store_sequence_of_row == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmKvShardStoreKernel<Geometry,THREADS>), x->pass_tokens, THREADS, 0, stream,
		x->store,(const uint16_t *)(latent_keys + (uint64_t)x->first_key * Geometry::kSlotBytes),x->store_sequence_of_row,x->pass_positions,x->pass_tokens,
		Geometry::kSlotBytes / 2u);
	if ( x->first_key == 0u )
		return(LM_LAUNCH_OK);
	older_rows = SparkKvShardGatherKeys(x->store.shard,x->first_key);
	chunk_rows = (uint32_t)(x->older_capacity / ((uint64_t)degree * Geometry::kSlotBytes));
	chunk_rows = chunk_rows < older_rows ? chunk_rows : older_rows;
	if ( x->older_sequence == 0 || x->older_offset == 0 || x->older_context == 0 || x->older_send == 0 || x->older_receive == 0 || chunk_rows == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	for (first_local = 0u; first_local < older_rows; first_local += chunk_rows)
	{
		const uint32_t count = older_rows - first_local < chunk_rows ? older_rows - first_local : chunk_rows;
		if ( LmKvShardGatherPackRangeLaunch<Geometry,THREADS>(x->store,x->older_sequence,x->older_offset,x->older_context,1u,first_local,first_local + count,
				x->older_send - (uint64_t)first_local * Geometry::kSlotBytes,stream) != cudaSuccess )
			return(LM_LAUNCH_ERR_LAUNCH);
		x->collective(x->context,(void *)(uintptr_t)stream,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,count,Geometry::kSlotBytes / 2u,x->older_send,x->older_receive);
		LM_LAUNCH((LmPassOlderKeysKernel<Geometry::kSlotBytes>), LM_KV_SHARD_UNPACK_BLOCKS, THREADS, 0, stream,
			x->store.shard,x->older_receive,first_local,count,x->first_key,latent_keys);
	}
	return(LM_LAUNCH_OK);
}
