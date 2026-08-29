// The family wiring of the JIT-KV pager for this module's resident decode
// stage (see spark_dsv4_jit_kv.h for the contract): the KV_BLOCKS_SAVE_OUT
// / KV_BLOCKS_RESTORE_IN frame ops behind the pager's module seam, and the
// deployment parkability condition over the shared arena predicate.

#include "spark_dsv4_jit_kv.h"

#include <string.h>

static uint32_t SparkDsv4KvFramesViewIsValid(
	const SparkDsv4KvFrames *frames,
	const SparkKvPagerBlockView *view)
{
	if ( view == 0 ||
		view->abi_version != SPARK_KV_PAGER_ABI_VERSION ||
		view->descriptor_bytes != SPARK_KV_PAGER_BLOCK_VIEW_DESCRIPTOR_BYTES ||
		view->block_count != 1u ||
		view->host_staging == 0 ||
		view->key_bytes != frames->key_block_stride_bytes ||
		view->value_bytes != frames->value_block_stride_bytes ||
		view->key_device_address == 0u ||
		(frames->value_block_stride_bytes != 0u &&
			view->value_device_address == 0u) )
	{
		return 0u;
	}
	return 1u;
}

/* Stage the op (the device-plane copy, exactly as the frame context will
 * execute it), then hand it to the submit primitive - the TERM copy on the
 * host, the frame-context submission behind the spark receipt on device.
 * A refusal stages and counts, and never touches the staging bytes. */
static SparkStatus SparkDsv4KvFramesRun(
	SparkDsv4KvFrames *frames,
	const SparkKvPagerBlockView *view,
	uint32_t op_code)
{
	SparkDsv4KvFrameOp *op = &frames->staged;
	SparkStatus status;

	op->op_code = op_code;
	op->block_count = view->block_count;
	op->key_bytes = view->key_bytes;
	op->value_bytes = view->value_bytes;
	op->key_device_address = view->key_device_address;
	op->value_device_address = view->value_device_address;
	op->host_staging = view->host_staging;
	frames->staged_count += 1u;
	if ( frames->backend == SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS &&
		frames->spark_receipt_valid == 0u )
	{
		frames->refused_count += 1u;
		return SPARK_STATUS_UNSUPPORTED;
	}
	status = frames->submit(frames->submit_context,op);
	if ( status != SPARK_STATUS_OK )
	{
		return status;
	}
	if ( frames->backend == SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS )
	{
		frames->device_run_count += 1u;
	}
	else
	{
		frames->host_copy_count += 1u;
	}
	return SPARK_STATUS_OK;
}

static uint32_t SparkDsv4KvFramesIsValid(const SparkDsv4KvFrames *frames)
{
	return frames != 0 &&
		frames->abi_version == SPARK_DSV4_JIT_KV_ABI_VERSION &&
		frames->descriptor_bytes == SPARK_DSV4_KV_FRAMES_DESCRIPTOR_BYTES &&
		frames->submit != 0;
}

SparkStatus SparkDsv4KvFramesInitialize(
	SparkDsv4KvFrames *frames,
	const SparkDsv4KvFramesConfiguration *configuration)
{
	if ( frames == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_DSV4_JIT_KV_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_DSV4_KV_FRAMES_CONFIGURATION_DESCRIPTOR_BYTES ||
		configuration->reserved0 != 0u ||
		configuration->backend >
			SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS ||
		configuration->key_block_stride_bytes == 0u ||
		(configuration->backend ==
				SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS &&
			configuration->submit == 0) )
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	memset(frames,0,sizeof(*frames));
	frames->abi_version = SPARK_DSV4_JIT_KV_ABI_VERSION;
	frames->descriptor_bytes = SPARK_DSV4_KV_FRAMES_DESCRIPTOR_BYTES;
	frames->backend = configuration->backend;
	frames->key_block_stride_bytes = configuration->key_block_stride_bytes;
	frames->value_block_stride_bytes = configuration->value_block_stride_bytes;
	frames->submit_context = configuration->submit_context;
	frames->submit = configuration->submit != 0 ?
		configuration->submit : SparkDsv4KvFramesSubmitHostCopy;
	return SPARK_STATUS_OK;
}

void SparkDsv4KvFramesAcceptSparkReceipt(SparkDsv4KvFrames *frames)
{
	if ( SparkDsv4KvFramesIsValid(frames) != 0u &&
		frames->backend == SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS )
	{
		frames->spark_receipt_valid = 1u;
	}
}

SparkStatus SparkDsv4KvFramesSave(
	void *module_context,
	const SparkKvPagerBlockView *view)
{
	SparkDsv4KvFrames *frames = (SparkDsv4KvFrames *)module_context;

	if ( SparkDsv4KvFramesIsValid(frames) == 0u ||
		SparkDsv4KvFramesViewIsValid(frames,view) == 0u )
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SparkDsv4KvFramesRun(frames,view,SPARK_DSV4_KV_FRAMES_OP_SAVE_OUT);
}

SparkStatus SparkDsv4KvFramesRestore(
	void *module_context,
	const SparkKvPagerBlockView *view)
{
	SparkDsv4KvFrames *frames = (SparkDsv4KvFrames *)module_context;

	if ( SparkDsv4KvFramesIsValid(frames) == 0u ||
		SparkDsv4KvFramesViewIsValid(frames,view) == 0u )
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	return SparkDsv4KvFramesRun(frames,view,
		SPARK_DSV4_KV_FRAMES_OP_RESTORE_IN);
}

/* The TERM copy primitive: the frame op against host mappings. The layout
 * is the pager's staging contract - key plane then value plane. */
SparkStatus SparkDsv4KvFramesSubmitHostCopy(
	void *submit_context,
	const SparkDsv4KvFrameOp *op)
{
	(void)submit_context;
	if ( op == 0 || op->host_staging == 0 )
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( op->op_code == SPARK_DSV4_KV_FRAMES_OP_SAVE_OUT )
	{
		memcpy(op->host_staging,
			(void *)(uintptr_t)op->key_device_address,op->key_bytes);
		if ( op->value_bytes != 0u )
		{
			memcpy((uint8_t *)op->host_staging + op->key_bytes,
				(void *)(uintptr_t)op->value_device_address,
				op->value_bytes);
		}
		return SPARK_STATUS_OK;
	}
	if ( op->op_code == SPARK_DSV4_KV_FRAMES_OP_RESTORE_IN )
	{
		memcpy((void *)(uintptr_t)op->key_device_address,
			op->host_staging,op->key_bytes);
		if ( op->value_bytes != 0u )
		{
			memcpy((void *)(uintptr_t)op->value_device_address,
				(uint8_t *)op->host_staging + op->key_bytes,
				op->value_bytes);
		}
		return SPARK_STATUS_OK;
	}
	return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkDsv4JitKvDecideParkability(
	const SparkKvCacheArena *arena,
	const uint32_t *active_block_indices,
	uint32_t active_block_count,
	SparkDsv4JitKvParkability *parkability_out)
{
	SparkDsv4JitKvParkability parkability;
	uint32_t block_index,index;

	if ( arena == 0 || parkability_out == 0 ||
		(active_block_count != 0u && active_block_indices == 0) )
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	memset(&parkability,0,sizeof(parkability));
	parkability.abi_version = SPARK_DSV4_JIT_KV_ABI_VERSION;
	parkability.descriptor_bytes =
		SPARK_DSV4_JIT_KV_PARKABILITY_DESCRIPTOR_BYTES;
	for ( block_index = 0u; block_index < arena->logical_block_count;
		++block_index )
	{
		if ( SparkKvCacheArenaBlockIsParkable(arena,block_index) != 0u )
		{
			parkability.parkable_block_count += 1u;
		}
		else if ( (arena->blocks[block_index].flags &
				SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
		{
			parkability.pinned_block_count += 1u;
		}
		else if ( (arena->blocks[block_index].flags &
				SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u )
		{
			parkability.non_resident_block_count += 1u;
		}
	}
	for ( index = 0u; index < active_block_count; ++index )
	{
		if ( active_block_indices[index] >= arena->logical_block_count )
		{
			return SPARK_STATUS_INVALID_ARGUMENT;
		}
		if ( SparkKvCacheArenaBlockIsParkable(
				arena,active_block_indices[index]) != 0u )
		{
			parkability.unprotected_active_count += 1u;
		}
	}
	*parkability_out = parkability;
	if ( parkability.unprotected_active_count != 0u )
	{
		/* the deployment's active set must be pinned before work is
		   offered: dispatch against a parkable block is the cliff. */
		return SPARK_STATUS_BUSY;
	}
	return SPARK_STATUS_OK;
}
