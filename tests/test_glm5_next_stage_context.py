#!/usr/bin/env python3
"""Exercise the real module configurator without allocating CUDA state."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"
#include "cache/kv_page_store.c"
static SparkGlm5NextModuleState state;

const char *cudaGetErrorString(cudaError_t error)
{
	(void)error;
	return("host test CUDA status");
}

cudaError_t cudaMemcpy(void *destination,const void *source,size_t bytes,cudaMemcpyKind kind)
{
	(void)destination;
	(void)source;
	(void)bytes;
	(void)kind;
	return(cudaSuccess);
}

SparkStatus SparkKvBackendInitialize(const SparkKvModelTable *table,SparkKvCacheArena *arena,SparkKvPageCache *cache,SparkKvPageStore *store)
{
	(void)arena;
	(void)cache;
	(void)store;
	assert(SparkKvPageStoreConfigurationIsValid(&table->page_store_config) != 0u);
	assert(table->page_store_config.transfer_capacity <= 2u);
	return(SPARK_STATUS_PENDING);
}

static int32_t check_batch_waves(void)
{
	SparkGlm5NextResidentDecodeStageBatchView batch = {0};
	uint32_t slots[303],width,row;
	atomic_uint *claims = state.lane_states;
	state.resident_sequence_capacity = 101u;
	for (row=0u; row<101u; row++)
		atomic_init(&claims[row],SPARK_STAGE_MODULE_SLOT_FREE);
	uint32_t ragged[8] = {5u,2u,9u,5u,2u,9u,5u,9u};
	batch.row_resident_slots = slots;
	for (width=1u; width<=101u; width++)
	{
		batch.active_sequence_count = width;
		batch.row_count = (width * 3u);
		for (row=0u; row<batch.row_count; row++)
			slots[row] = (width - 1u - (row % width));
		if ( SparkGlm5NextValidateRoundMajor(&state,&batch) != SPARK_STATUS_OK || SparkStageModuleIndexSetClaim(claims,101u,slots,width) != SPARK_STATUS_OK )
			return(-4);
		for (row=0u; row<batch.row_count; row+=width)
			if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,row) != width )
				return(-1);
		SparkStageModuleIndexSetRelease(claims,101u,slots,width);
	}
	batch.row_resident_slots = ragged;
	batch.active_sequence_count = 3u;
	batch.row_count = 8u;
	if ( SparkGlm5NextValidateRoundMajor(&state,&batch) != SPARK_STATUS_OK || SparkStageModuleIndexSetClaim(claims,101u,ragged,3u) != SPARK_STATUS_OK )
		return(-5);
	if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,0u) != 3u || SparkGlm5NextRoundMajorWaveRows(&state,&batch,3u) != 3u || SparkGlm5NextRoundMajorWaveRows(&state,&batch,6u) != 2u )
		return(-2);
	if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,8u) != 0u || SparkGlm5NextRoundMajorWaveRows(&state,0,0u) != 0u )
		return(-3);
	SparkStageModuleIndexSetRelease(claims,101u,ragged,3u);
	if ( SparkGlm5NextRoundMajorWaveRows(&state,&batch,0u) != 0u )
		return(-6);
	ragged[6] = 9u;
	ragged[7] = 5u;
	if ( SparkGlm5NextValidateRoundMajor(&state,&batch) == SPARK_STATUS_OK )
		return(-7);
	return(0);
}

static void check_small_kv(void)
{
	uint32_t pages;
	for (pages=1u; pages<=3u; pages++)
	{
		memset(&state,0,sizeof(state));
		state.kv_layer_count = 1u;
		state.page_count = pages;
		state.pages_per_sequence = pages;
		state.resident_sequence_capacity = 1u;
		state.kv_backing_directory = "/unused-host-fixture";
		assert(SparkGlm5NextKvInitialize(&state) == SPARK_STATUS_PENDING);
		free(state.kv_blocks);
		free(state.kv_resident_slot_logical_block_indices);
		free(state.kv_entries);
		free(state.kv_sequences);
		free(state.kv_hash_bucket_heads);
		free(state.kv_entry_indices_by_logical_page);
		free(state.kv_page_staging);
		free(state.kv_lane_logical_pages);
		free(state.kv_lane_page_count);
		free(state.kv_lane_mutable_page);
		free(state.kv_lane_mutation_flags);
		free(state.kv_lane_cache_lanes);
	}
	memset(&state,0,sizeof(state));
}

static void check_pack_identity(void)
{
	SparkGlm5NextStagePackHeader header = {0};
	header.magic = SPARK_GLM5_NEXT_STAGEPACK_MAGIC;
	header.format_version = SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION;
	header.header_bytes = SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES;
	header.directory_entry_bytes = SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES;
	header.codec_abi_version = SPARK_WEIGHT_CODEC_ABI_VERSION;
	header.tensor_count = 1u;
	header.stage_count = state.stage_count;
	header.stage_index = state.stage_index;
	header.first_layer_index = state.first_layer_index;
	header.layer_count = state.layer_count;
	header.total_layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	header.hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	header.vocab_count = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	header.routed_expert_count = SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	header.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	header.expert_weight_codec = state.expert_weight_codec;
	header.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	header.reserved0 = state.tp_degree;
	header.reserved1 = state.tp_rank;
	header.directory_offset = (((header.header_bytes + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES) * SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES);
	header.file_bytes = (header.directory_offset + header.directory_entry_bytes);
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_OK);
	header.stage_count++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header.stage_count--;
	header.stage_index++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header.stage_index--;
	header.first_layer_index++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	header.first_layer_index--;
	header.layer_count++;
	assert(SparkGlm5NextPackValidateHeader(&state,&header,header.file_bytes) == SPARK_STATUS_SCHEMA_ERROR);
}

int32_t main(void)
{
	SparkGlm5NextResidentDecodeStageNodeContext context = {0};
	SparkFirmwareModuleConfiguration configuration = {0};
	SparkFirmwareModuleHostServices services = {0};
	const char *path = 0;
	uint32_t first[4] = {0u,12u,23u,34u},counts[4] = {12u,11u,11u,11u},stage;
	if ( check_batch_waves() != 0 )
		return(1);
	check_small_kv();
	context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context.descriptor_bytes = sizeof(context);
	context.stage_count = 4u;
	context.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC;
	context.resident_sequence_capacity = 3u;
	context.pipeline_slot_count = 1u;
	context.max_sequence_positions = 64u;
	context.execution_row_capacity = 3u;
	context.tp_degree = 4u;
	context.stage_pack_path = "fixture.g5nsp";
	context.model_revision = "fixture";
	configuration.model_revision = context.model_revision;
	services.node_context = &context;
	services.execution_stream = (void *)(uintptr_t)1u;
	for (stage=0u; stage<4u; stage++)
	{
		context.stage_index = stage;
		context.first_layer_index = first[stage];
		context.layer_count = counts[stage];
		assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_OK);
		assert(state.stage_count == 4u && state.stage_index == stage);
		assert(state.first_layer_index == first[stage] && state.layer_count == counts[stage]);
		assert(state.owns_embedding == (stage == 0u) && state.owns_final_head == (stage == 3u));
		assert(path == context.stage_pack_path);
		check_pack_identity();
	}
	context.abi_version--;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_ABI_MISMATCH);
	context.abi_version++;
	context.stage_index = 0u;
	context.first_layer_index = 0u;
	context.layer_count = 12u;
	context.flags = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.flags = 0u;
	context.stage_count = 1u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_INVALID_ARGUMENT);
	context.layer_count = 45u;
	context.tp_degree = 16u;
	assert(SparkGlm5NextModuleConfigure(&state,&configuration,&services,&path) == SPARK_STATUS_OK);
	assert(state.owns_embedding == 1u && state.owns_final_head == 1u);
	check_pack_identity();
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "probe.c", Path(directory) / "probe"
        source.write_text(HARNESS)
        includes = [".", "include", "tests/cuda_stub", "model-families/common/include",
                    "model-families/glm5_next/include", "modules/glm5_next_resident_decode_stage/include",
                    "modules/glm5_next_resident_decode_stage/source"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O2", "-ffunction-sections", "-fdata-sections",
                        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                        *["-I" + p for p in includes], "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
                        '-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"', '-DGLM5_NEXT_CONTRACT_SHA256="fixture"',
                        str(source), "runtime/stage_module_common.c", "-o", str(binary)], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS actual module TP4/PP4 and TP16 context, ownership and ABI gates")


if __name__ == "__main__":
    main()
