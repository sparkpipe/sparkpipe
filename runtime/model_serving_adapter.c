#include "sparkpipe/spark_model_serving_adapter.h"

#include <dlfcn.h>
#include <string.h>

#include "sparkpipe/spark_model_driver_support.h"

_Static_assert(sizeof(SparkModelServingCacheIdentity) ==
	sizeof(SparkModelDriverCacheIdentity),"cache identity ABI mismatch");

static uint32_t SparkModelServingAdapterTextIsPresent(const char *text)
{
	return(text != 0 && text[0] != '\0');
}

static uint32_t SparkModelServingAdapterSha256IsValid(const char *sha256)
{
	uint32_t index;
	if ( sha256 == 0 )
		return(0u);
	for (index=0u; index<SPARK_MODEL_SERVING_ADAPTER_ARTIFACT_SHA256_LENGTH; index++)
		if ( !((sha256[index] >= '0' && sha256[index] <= '9') || (sha256[index] >= 'a' && sha256[index] <= 'f')) )
			return(0u);
	return(sha256[SPARK_MODEL_SERVING_ADAPTER_ARTIFACT_SHA256_LENGTH] == '\0');
}

SparkStatus SparkModelServingAdapterValidateDescriptor(
	const SparkModelServingAdapterDescriptor *descriptor)
{
	uint32_t group,group_count,hybrid,index,parallel,total;
	if ( descriptor == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( descriptor->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || descriptor->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (descriptor->capability_flags & ~SPARK_MODEL_SERVING_ADAPTER_KNOWN_CAPABILITIES) != 0u || descriptor->stage_count == 0u || descriptor->stage_count > SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT || descriptor->layer_count == 0u || descriptor->layer_count > SPARK_MODEL_SERVING_ADAPTER_MAX_LAYER_COUNT )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->capability_flags &
		(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT)) ==
		(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT) &&
		(descriptor->capability_flags &
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP) == 0u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP) != 0u &&
		(descriptor->capability_flags &
		 (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT)) !=
		 (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( descriptor->boundary_format != SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16 || descriptor->boundary_element_count == 0u || descriptor->boundary_element_bytes != sizeof(uint16_t) || SparkWeightCodecIsKnown(descriptor->linear_weight_codec) == 0u || SparkWeightCodecIsKnown(descriptor->expert_weight_codec) == 0u || SparkWeightCodecIsKnown(descriptor->kv_cache_codec) == 0u || descriptor->max_inflight_submission_count == 0u || descriptor->max_inflight_submission_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INFLIGHT_SUBMISSION_COUNT || descriptor->max_active_sequence_count == 0u || descriptor->max_active_sequence_count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT || descriptor->max_input_row_count < descriptor->max_active_sequence_count || descriptor->max_input_row_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT || descriptor->max_resident_sequence_count < descriptor->max_active_sequence_count || descriptor->max_resident_sequence_count > SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT || descriptor->max_output_token_count == 0u || descriptor->max_output_token_count > SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( ((descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION) != 0u) != (descriptor->max_speculative_token_count != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( descriptor->resident_sequence_slot_reuse > SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( ((descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV) != 0u) != (descriptor->resident_sequence_slot_reuse != SPARK_MODEL_SERVING_SLOT_REUSE_NONE) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( descriptor->resident_sequence_slot_reuse == SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE && (descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE) == 0u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) != 0u &&
		(descriptor->capability_flags &
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV) == 0u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) != 0u &&
		descriptor->resident_sequence_slot_reuse ==
		SPARK_MODEL_SERVING_SLOT_REUSE_NONE )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN) != 0u &&
		(descriptor->capability_flags &
		 (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE)) !=
		 (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV |
		  SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN) != 0u &&
		descriptor->max_output_token_count < 2u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( descriptor->minimum_efficient_submission_row_count > descriptor->max_input_row_count )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	hybrid = (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP) != 0u ? 1u : 0u;
	if ( descriptor->cache_block_token_count > SPARK_MODEL_SERVING_ADAPTER_MAX_CACHE_BLOCK_TOKEN_COUNT ||
		(hybrid == 0u && descriptor->parallel_group_size != 0u) ||
		(hybrid != 0u && (descriptor->parallel_group_size < 2u ||
		 descriptor->parallel_group_size > descriptor->stage_count ||
		 descriptor->stage_count % descriptor->parallel_group_size != 0u)) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (descriptor->cache_block_token_count != 0u) != ((descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV) != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( descriptor->cache_block_token_count != 0u && (descriptor->capability_flags & (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV)) != (SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( SparkModelServingAdapterTextIsPresent(descriptor->adapter_id) == 0u || SparkModelServingAdapterTextIsPresent(descriptor->model_id) == 0u || SparkModelServingAdapterTextIsPresent(descriptor->model_revision) == 0u || SparkModelServingAdapterTextIsPresent(descriptor->driver_program_name) == 0u || SparkModelServingAdapterSha256IsValid(descriptor->artifact_sha256) == 0u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	parallel = (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT) != 0u ? 1u : 0u;
	total = 0u;
	for (index=0u; index<descriptor->stage_count; index++)
	{
		if ( descriptor->stage_layer_counts[index] == 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( index + 1u < descriptor->stage_count && ((descriptor->boundary_sideband_kinds[index] != 0u) != (descriptor->boundary_sideband_bytes_per_sequence[index] != 0u)) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( index + 1u == descriptor->stage_count && (descriptor->boundary_sideband_kinds[index] != 0u || descriptor->boundary_sideband_bytes_per_sequence[index] != 0u) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		total += descriptor->stage_layer_counts[index];
	}
	for (; index<SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT; index++)
		if ( descriptor->stage_layer_counts[index] != 0u || descriptor->boundary_sideband_kinds[index] != 0u || descriptor->boundary_sideband_bytes_per_sequence[index] != 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( hybrid != 0u )
	{
		group_count = descriptor->stage_count / descriptor->parallel_group_size;
		total = 0u;
		for (group=0u; group<group_count; group++)
		{
			uint32_t group_layer_count;
			group_layer_count = descriptor->stage_layer_counts[
				group * descriptor->parallel_group_size];
			for (index=1u; index<descriptor->parallel_group_size; index++)
				if ( descriptor->stage_layer_counts[
					group * descriptor->parallel_group_size + index] !=
					group_layer_count )
					do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
			total += group_layer_count;
		}
		return(total == descriptor->layer_count ? SPARK_STATUS_OK :
			SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( total == descriptor->layer_count )
		return(SPARK_STATUS_OK);
	if ( parallel != 0u )
	{
		for (index=0u; index<descriptor->stage_count; index++)
			if ( descriptor->stage_layer_counts[index] != descriptor->layer_count )
				do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		return(SPARK_STATUS_OK);
	}
	do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
}

SparkStatus SparkModelServingAdapterValidateRuntimeLimits(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits)
{
	SparkStatus status;
	uint32_t index,jit_kv;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK || runtime_limits == 0 )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( runtime_limits->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || runtime_limits->descriptor_bytes != SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( runtime_limits->max_inflight_submission_count == 0u || runtime_limits->max_inflight_submission_count > descriptor->max_inflight_submission_count || runtime_limits->max_active_sequence_count == 0u || runtime_limits->max_active_sequence_count > descriptor->max_active_sequence_count || runtime_limits->max_input_row_count < runtime_limits->max_active_sequence_count || runtime_limits->max_input_row_count > descriptor->max_input_row_count || runtime_limits->resident_sequence_capacity < runtime_limits->max_active_sequence_count || runtime_limits->resident_sequence_capacity > descriptor->max_resident_sequence_count )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	jit_kv = (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV) != 0u ? 1u : 0u;
	if ( jit_kv != 0u &&
		(runtime_limits->kv_physical_page_capacity <
		 runtime_limits->max_active_sequence_count ||
		 runtime_limits->kv_logical_page_capacity <
		 runtime_limits->resident_sequence_capacity ||
		 runtime_limits->kv_physical_page_capacity >
		 runtime_limits->kv_logical_page_capacity) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( jit_kv == 0u &&
		(runtime_limits->kv_logical_page_capacity != 0u ||
		 runtime_limits->kv_physical_page_capacity != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	for (index=0u; index<4u; index++)
		if ( runtime_limits->reserved[index] != 0u )
			return(SPARK_STATUS_ABI_MISMATCH);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateInterface(
	const SparkModelServingAdapterInterface *adapter_interface,
	uint32_t required_capability_flags)
{
	SparkStatus status;
	if ( adapter_interface == 0 || (required_capability_flags & ~SPARK_MODEL_SERVING_ADAPTER_KNOWN_CAPABILITIES) != 0u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( adapter_interface->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || adapter_interface->interface_bytes != SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateDescriptor(adapter_interface->descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (adapter_interface->descriptor->capability_flags & required_capability_flags) != required_capability_flags || adapter_interface->initialize == 0 || adapter_interface->destroy == 0 || adapter_interface->validate_submission == 0 || adapter_interface->submit == 0 || adapter_interface->progress == 0 || adapter_interface->quiesce == 0 || adapter_interface->snapshot == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (adapter_interface->descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH) != 0u && adapter_interface->prefetch == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (adapter_interface->descriptor->capability_flags &
		(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH)) ==
		(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH) &&
		adapter_interface->resolve_prefetch == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( (adapter_interface->descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESET) != 0u && adapter_interface->reset == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelServingAdapterValidateRows(
	const SparkModelServingSubmission *submission,
	uint32_t require_live_rows,
	uint32_t resident_sequence_capacity,
	uint32_t cache_block_token_count)
{
	uint8_t seen[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t seen_slots[SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT];
	uint32_t lane,row,slot;
	uint32_t prefix_identity_present,publish_identity_present;
	memset(seen,0,sizeof(seen));
	memset(seen_slots,0,sizeof(seen_slots));
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		if ( (submission->lanes[lane].flags & ~SPARK_MODEL_SERVING_LANE_KNOWN_FLAGS) != 0u ||
			(submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE && submission->lanes[lane].flags != 0u) ||
			(submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE && lane < submission->active_sequence_count && (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) == 0u) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( lane >= submission->active_sequence_count )
		{
			SparkModelServingCacheIdentity zero_identity;
			memset(&zero_identity,0,sizeof(zero_identity));
			if ( submission->lanes[lane].request_id != 0u || submission->lanes[lane].request_generation != 0u || submission->lanes[lane].step_generation != 0u || submission->lanes[lane].sequence_id != 0u || submission->lanes[lane].sequence_position != 0u || submission->lanes[lane].resident_sequence_slot != SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT || submission->lanes[lane].context_token_count != 0u || submission->lanes[lane].input_token_id != 0u || submission->lanes[lane].flags != 0u || submission->lanes[lane].cache_prefix_token_count != 0u || submission->lanes[lane].cache_publish_token_count != 0u || memcmp(&submission->lanes[lane].cache_prefix_identity,&zero_identity,sizeof(zero_identity)) != 0 || memcmp(&submission->lanes[lane].cache_publish_identity,&zero_identity,sizeof(zero_identity)) != 0 )
				do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
			continue;
		}
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( submission->lanes[lane].request_id == 0u || submission->lanes[lane].request_generation == 0u || submission->lanes[lane].step_generation == 0u || submission->lanes[lane].sequence_id == 0u || slot >= resident_sequence_capacity || seen_slots[slot] != 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		prefix_identity_present = 0u;
		publish_identity_present = 0u;
		for (row=0u; row<sizeof(submission->lanes[lane].cache_prefix_identity.sha256); row++)
			prefix_identity_present |= submission->lanes[lane].cache_prefix_identity.sha256[row];
		for (row=0u; row<sizeof(submission->lanes[lane].cache_publish_identity.sha256); row++)
			publish_identity_present |= submission->lanes[lane].cache_publish_identity.sha256[row];
		if ( ((submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u) != (submission->lanes[lane].cache_prefix_token_count != 0u) || ((submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u) != (prefix_identity_present != 0u) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( submission->lanes[lane].cache_prefix_token_count != 0u && (cache_block_token_count == 0u || submission->lanes[lane].cache_prefix_token_count % cache_block_token_count != 0u || submission->lanes[lane].cache_prefix_token_count > submission->lanes[lane].sequence_position) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( ((submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) != 0u) != (submission->lanes[lane].cache_publish_token_count != 0u) || ((submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) != 0u) != (publish_identity_present != 0u) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( submission->lanes[lane].cache_publish_token_count != 0u && (cache_block_token_count == 0u || submission->lanes[lane].cache_publish_token_count % cache_block_token_count != 0u || submission->lanes[lane].cache_publish_token_count <= submission->lanes[lane].cache_prefix_token_count || submission->lanes[lane].cache_publish_token_count > submission->lanes[lane].context_token_count) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		seen_slots[slot] = 1u;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_sequence_ids[row] != submission->lanes[lane].sequence_id )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		seen[lane] = 1u;
	}
	for (lane=0u; require_live_rows != 0u && lane<submission->active_sequence_count; lane++)
		if ( seen[lane] == 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelServingAdapterFindLastRows(
	const SparkModelServingSubmission *submission,
	uint32_t *last_rows)
{
	uint8_t seen[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t last_positions[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane,row;
	if ( submission == 0 || last_rows == 0 || submission->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || submission->descriptor_bytes != SPARK_MODEL_SERVING_SUBMISSION_BYTES || (submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_PREFILL && submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_DECODE) || submission->active_sequence_count == 0u || submission->active_sequence_count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT || submission->lane_count < submission->active_sequence_count || submission->lane_count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT || submission->row_count < submission->active_sequence_count || submission->row_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT || submission->lanes == 0 || submission->row_lane_indices == 0 || submission->row_positions == 0 || submission->row_sequence_ids == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	memset(seen,0,sizeof(seen));
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		if ( (submission->lanes[lane].flags & ~SPARK_MODEL_SERVING_LANE_KNOWN_FLAGS) != 0u ||
			(lane >= submission->active_sequence_count && submission->lanes[lane].flags != 0u) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( lane >= submission->active_sequence_count )
			continue;
		if ( submission->lanes[lane].sequence_id == 0u || submission->lanes[lane].context_token_count == 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		last_rows[lane] = UINT32_MAX;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_sequence_ids[row] != submission->lanes[lane].sequence_id )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( seen[lane] == 0u && submission->row_positions[row] != submission->lanes[lane].sequence_position )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		if ( seen[lane] != 0u && (last_positions[lane] == UINT64_MAX || submission->row_positions[row] != last_positions[lane] + 1u) )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		seen[lane] = 1u;
		last_positions[lane] = submission->row_positions[row];
		last_rows[lane] = row;
	}
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( seen[lane] == 0u || last_positions[lane] == UINT64_MAX || last_positions[lane] + 1u != submission->lanes[lane].context_token_count )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterSelectEmitRows(
	const SparkModelServingSubmission *submission,
	uint32_t *emit_row_indices,
	uint32_t *emit_lane_indices,
	uint32_t emit_capacity,
	uint32_t *emit_count_out)
{
	uint32_t last_rows[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t count,lane,row;
	SparkStatus status;
	if ( emit_count_out == 0 || ((emit_row_indices == 0) != (emit_lane_indices == 0)) || (emit_row_indices == 0 && emit_capacity != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	*emit_count_out = 0u;
	status = SparkModelServingAdapterFindLastRows(submission,last_rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	count = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		count += (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u ? 1u : 0u;
	if ( emit_row_indices != 0 && count > emit_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( emit_row_indices != 0 )
	{
		count = 0u;
		for (row=0u; row<submission->row_count; row++)
		{
			lane = submission->row_lane_indices[row];
			if ( last_rows[lane] == row && (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u )
			{
				emit_row_indices[count] = row;
				emit_lane_indices[count] = lane;
				count++;
			}
		}
	}
	*emit_count_out = count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	uint32_t required_capability,total_output_tokens;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( submission->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || submission->descriptor_bytes != SPARK_MODEL_SERVING_SUBMISSION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( submission->flags != 0u || submission->submission_id == 0u || submission->control_generation == 0u || submission->transaction_id == 0u || submission->dispatch_generation == 0u || submission->request_generation == 0u || submission->step_generation == 0u || submission->work_kind < SPARK_MODEL_SERVING_WORK_KIND_PREFILL || submission->work_kind > SPARK_MODEL_SERVING_WORK_KIND_RELEASE || submission->lane_count == 0u || submission->lane_count > descriptor->max_active_sequence_count || submission->active_sequence_count == 0u || submission->active_sequence_count > submission->lane_count || submission->lanes == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	required_capability = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL : submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE : SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE;
	if ( (descriptor->capability_flags & required_capability) == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( submission->model_extension_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES || (submission->model_extension_bytes != 0u) != (submission->model_extension != 0) || (submission->model_extension_bytes != 0u) != (submission->model_extension_kind != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		if ( submission->tokens_per_sequence != 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	}
	else if ( submission->tokens_per_sequence == 0u ||
		submission->tokens_per_sequence >
		SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	else if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL &&
		submission->tokens_per_sequence != 1u )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	else if ( submission->tokens_per_sequence > 1u &&
		(descriptor->capability_flags &
		 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN) == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	total_output_tokens = 0u;
	if ( submission->tokens_per_sequence != 0u )
		total_output_tokens = submission->active_sequence_count <= UINT32_MAX /
			submission->tokens_per_sequence ? submission->active_sequence_count *
			submission->tokens_per_sequence : UINT32_MAX;
	if ( total_output_tokens > descriptor->max_output_token_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (submission->hidden_input_address != 0) != (submission->hidden_input_bytes != 0u) || (submission->boundary_sideband_input_address != 0) != (submission->boundary_sideband_input_bytes != 0u) || (submission->hidden_output_address != 0) != (submission->hidden_output_bytes != 0u) || (submission->boundary_sideband_output_address != 0) != (submission->boundary_sideband_output_bytes != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		if ( submission->row_count != 0u || submission->token_count != 0u || submission->new_token_count != 0u || submission->hidden_input_address != 0 || submission->boundary_sideband_input_address != 0 || submission->hidden_output_address != 0 || submission->boundary_sideband_output_address != 0 )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
		return(SparkModelServingAdapterValidateRows(submission,0u,descriptor->max_resident_sequence_count,descriptor->cache_block_token_count));
	}
	if ( submission->row_count == 0u || submission->token_count != submission->row_count || submission->new_token_count != submission->row_count || submission->token_count > descriptor->max_input_row_count || submission->token_ids == 0 || submission->row_lane_indices == 0 || submission->row_positions == 0 || submission->row_sequence_ids == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(SparkModelServingAdapterValidateRows(submission,1u,descriptor->max_resident_sequence_count,descriptor->cache_block_token_count));
}

SparkStatus SparkModelServingAdapterValidateRuntimeSubmission(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	uint32_t lane;
	status = SparkModelServingAdapterValidateRuntimeLimits(descriptor,runtime_limits);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterValidateSubmission(descriptor,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->lane_count > runtime_limits->max_active_sequence_count || submission->row_count > runtime_limits->max_input_row_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( submission->lanes[lane].resident_sequence_slot >= runtime_limits->resident_sequence_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SparkModelServingAdapterValidateRows(submission,submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE,runtime_limits->resident_sequence_capacity,descriptor->cache_block_token_count));
}

SparkStatus SparkModelServingAdapterStreamOrderedProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

SparkStatus SparkModelServingAdapterPrepareSubmission(
	const SparkModelServingAdapterInterface *adapter_interface,
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	if ( adapter_interface == 0 || adapter_interface->descriptor == 0 || adapter_interface->validate_submission == 0 || adapter_state == 0 || submission == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	status = adapter_interface->validate_submission(adapter_state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (adapter_interface->descriptor->capability_flags & SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH) == 0u )
		return(SPARK_STATUS_OK);
	if ( adapter_interface->prefetch == 0 )
		return(SPARK_STATUS_ABI_MISMATCH);
	return(adapter_interface->prefetch(adapter_state,submission,1u));
}

SparkStatus SparkModelServingAdapterResolvePrefetch(
	const SparkModelServingAdapterInterface *adapter_interface,
	void *adapter_state,
	const SparkModelServingSubmission *submission,
	uint32_t resolution)
{
	SparkStatus status;
	uint32_t required_capabilities;
	if ( adapter_interface == 0 || adapter_interface->descriptor == 0 ||
		adapter_state == 0 || submission == 0 ||
		(resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		 resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	required_capabilities = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH;
	if ( (adapter_interface->descriptor->capability_flags &
		required_capabilities) != required_capabilities )
		return(SPARK_STATUS_OK);
	if ( adapter_interface->resolve_prefetch == 0 )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = adapter_interface->resolve_prefetch(adapter_state,submission,
		resolution);
	if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(status);
}

SparkStatus SparkModelServingAdapterBuildDriverCacheLanes(
	const SparkModelServingSubmission *submission,
	SparkModelDriverCacheLane *cache_lanes,
	uint32_t cache_lane_capacity,
	uint32_t *cache_lane_count_out)
{
	SparkModelDriverCacheLane *destination;
	const SparkModelServingLane *source;
	uint32_t lane;
	if ( cache_lane_count_out == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	*cache_lane_count_out = 0u;
	if ( submission == 0 || cache_lanes == 0 ||
		submission->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION ||
		submission->descriptor_bytes != SPARK_MODEL_SERVING_SUBMISSION_BYTES ||
		submission->active_sequence_count == 0u ||
		submission->active_sequence_count > cache_lane_capacity ||
		submission->lanes == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		source = &submission->lanes[lane];
		destination = &cache_lanes[lane];
		memset(destination,0,sizeof(*destination));
		destination->sequence_id = source->sequence_id;
		destination->sequence_position = source->sequence_position;
		destination->request_generation = source->request_generation;
		destination->step_generation = source->step_generation;
		destination->resident_sequence_slot = source->resident_sequence_slot;
		destination->context_token_count = source->context_token_count;
		if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			destination->flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
		else
		{
			destination->prefix_token_count = source->cache_prefix_token_count;
			destination->publish_token_count = source->cache_publish_token_count;
			memcpy(&destination->prefix_identity,&source->cache_prefix_identity,sizeof(destination->prefix_identity));
			memcpy(&destination->publish_identity,&source->cache_publish_identity,sizeof(destination->publish_identity));
			if ( (source->flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u )
				destination->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
			if ( (source->flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) != 0u )
				destination->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
		}
		if ( SparkModelDriverCacheLaneIsValid(destination) == 0u )
			do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	}
	*cache_lane_count_out = submission->active_sequence_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateCompletion(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingCompletion *completion)
{
	SparkStatus status;
	uint32_t has_tokens,has_extension;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK || completion == 0 )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	if ( completion->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || completion->descriptor_bytes != SPARK_MODEL_SERVING_COMPLETION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( completion->submission_id == 0u || completion->control_generation == 0u || completion->transaction_id == 0u || completion->dispatch_generation == 0u || completion->request_generation == 0u || completion->step_generation == 0u || completion->status > SPARK_STATUS_UNSUPPORTED || (completion->completion_flags & ~SPARK_MODEL_SERVING_COMPLETION_KNOWN_FLAGS) != 0u || completion->token_count > descriptor->max_output_token_count || completion->tokens_per_sequence > SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE || completion->model_extension_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	has_tokens = (completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS) != 0u;
	has_extension = (completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION) != 0u;
	if ( has_tokens != (completion->token_count != 0u) || has_tokens != (completion->tokens_per_sequence != 0u) || (completion->tokens_per_sequence != 0u && completion->token_count % completion->tokens_per_sequence != 0u) || has_extension != (completion->model_extension_bytes != 0u) || (completion->model_extension_bytes == 0u && completion->model_extension_kind != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( completion->status != SPARK_STATUS_OK && (completion->completion_flags != 0u || completion->accepted_token_count != 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelServingAdapterValidateCompletionResidency(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelDriverResidencyToken *expected_residency,
	const SparkModelServingCompletion *completion)
{
	SparkStatus status;
	status = SparkModelServingAdapterValidateCompletion(descriptor,completion);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( expected_residency == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(memcmp(expected_residency,&completion->residency,sizeof(*expected_residency)) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

SparkStatus SparkModelServingAdapterValidateStageCompletion(
	const SparkModelServingAdapterDescriptor *descriptor,
	uint32_t stage_index,
	uint32_t work_kind,
	uint32_t active_sequence_count,
	uint32_t tokens_per_sequence,
	const SparkModelDriverResidencyToken *expected_residency,
	const SparkModelServingCompletion *completion)
{
	SparkStatus status;
	uint32_t final_stage,has_tokens,parallel,hybrid;
	status = SparkModelServingAdapterValidateCompletionResidency(descriptor,expected_residency,completion);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( stage_index >= descriptor->stage_count || work_kind < SPARK_MODEL_SERVING_WORK_KIND_PREFILL || work_kind > SPARK_MODEL_SERVING_WORK_KIND_RELEASE || active_sequence_count == 0u || active_sequence_count > descriptor->max_active_sequence_count || tokens_per_sequence > SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE || (work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE && tokens_per_sequence == 0u) )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	has_tokens = (completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS) != 0u;
	if ( completion->status != SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(has_tokens == 0u && tokens_per_sequence == 0u ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
	final_stage = descriptor->stage_count - 1u;
	/* Pure parallel fanout (TP without PP) emits tokens from EVERY rank: each
	 * rank runs the whole stack, so the non-final-stage token prohibition
	 * applies only to transported pipelines. */
	parallel = (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT) != 0u ? 1u : 0u;
	hybrid = (descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP) != 0u ? 1u : 0u;
	if ( stage_index != final_stage && (parallel == 0u || hybrid != 0u) )
		return(has_tokens == 0u && completion->tokens_per_sequence == 0u ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
	/* Pure TP fanout: the TP-sharded adapters publish the token payload only
	 * from the final stage (the other ranks carry status alone); a
	 * status-only completion from a non-final rank is therefore legal. */
	if ( stage_index != final_stage && has_tokens == 0u &&
		completion->tokens_per_sequence == 0u )
		return(SPARK_STATUS_OK);
	/* Speculative completions: the submission carries the CHAIN width
	 * (tokens_per_sequence), while the actual yield rides
	 * accepted_token_count. Draft-chain verify emits 1..chain_width
	 * tokens per sequence (partial acceptance), so the lower bound is 1;
	 * the upper bound is the chain width plus the speculative allowance. */
	if ( has_tokens != 0u &&
		completion->tokens_per_sequence >= 1u &&
		completion->tokens_per_sequence <= tokens_per_sequence + descriptor->max_speculative_token_count &&
		active_sequence_count <= UINT32_MAX / completion->tokens_per_sequence &&
		completion->token_count == active_sequence_count * completion->tokens_per_sequence )
		return(SPARK_STATUS_OK);
	return(SPARK_STATUS_SCHEMA_ERROR);
}

SparkStatus SparkModelServingAdapterLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkModelServingAdapterDynamicLibrary *library)
{
	SparkModelServingAdapterGetInterfaceFunction get_interface;
	const SparkModelServingAdapterInterface *adapter_interface;
	void *dynamic_library;
	SparkStatus status;
	if ( shared_object_path == 0 || library == 0 )
		do { fprintf(stderr,"VA_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	memset(library,0,sizeof(*library));
	dynamic_library = dlopen(shared_object_path,RTLD_NOW | RTLD_LOCAL);
	if ( dynamic_library == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	get_interface = (SparkModelServingAdapterGetInterfaceFunction)dlsym(dynamic_library,SPARK_MODEL_SERVING_ADAPTER_INTERFACE_SYMBOL);
	if ( get_interface == 0 )
	{
		dlclose(dynamic_library);
		return(SPARK_STATUS_NOT_FOUND);
	}
	adapter_interface = get_interface();
	status = SparkModelServingAdapterValidateInterface(adapter_interface,required_capability_flags);
	if ( status != SPARK_STATUS_OK )
	{
		dlclose(dynamic_library);
		return(status);
	}
	library->dynamic_library = dynamic_library;
	library->adapter_interface = *adapter_interface;
	return(SPARK_STATUS_OK);
}

void SparkModelServingAdapterUnloadInterface(
	SparkModelServingAdapterDynamicLibrary *library)
{
	if ( library == 0 )
		return;
	if ( library->dynamic_library != 0 )
		dlclose(library->dynamic_library);
	memset(library,0,sizeof(*library));
}
