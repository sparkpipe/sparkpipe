#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_model_driver_support.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static SparkStatus SparkStageKvBindingDeviceCopy(void *context,uint32_t direction,uintptr_t device_address,void *host_address,uint64_t bytes)
{
	const SparkStageKvBinding *binding = (const SparkStageKvBinding *)context;
	cudaError_t error;
	if ( binding == 0 || device_address == 0u || host_address == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host_address,(const void *)device_address,(size_t)bytes,cudaMemcpyDeviceToHost);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpy((void *)device_address,host_address,(size_t)bytes,cudaMemcpyHostToDevice);
	else
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(binding->module_tag,error,"kv_page_copy"));
}

static SparkStatus SparkStageKvBindingPageCopy(void *context,uint32_t direction,uintptr_t device_address,void *host_address,uint64_t bytes)
{
	const SparkStageKvBinding *binding = (const SparkStageKvBinding *)context;
	SparkKvLayeredPageLayout layout;
	uintptr_t base;
	uint64_t span,offset,packed;
	uint32_t region;
	if ( binding == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (region=0u; region<binding->region_count; region++)
	{
		base = (uintptr_t)binding->region_base[region];
		packed = binding->region_packed_page_bytes[region];
		span = (uint64_t)binding->physical_page_count * packed;
		if ( device_address < base || device_address - base >= span )
			continue;
		offset = device_address - base;
		if ( binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR )
		{
			if ( bytes > span - offset )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			return(SparkStageKvBindingDeviceCopy(context,direction,device_address,host_address,bytes));
		}
		if ( offset % packed != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		layout.device_base = base;
		layout.device_bytes = span;
		layout.layer_stride_bytes = binding->region_layer_stride_bytes[region];
		layout.layer_page_bytes = binding->regions[region].layer_page_bytes;
		layout.layer_count = binding->regions[region].layer_count;
		layout.page_count = binding->physical_page_count;
		return(SparkKvPageStoreCopyLayered(&layout,direction,(uint32_t)(offset / packed),host_address,bytes,SparkStageKvBindingDeviceCopy,context));
	}
	SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkStageKvBindingAllocateHost(SparkStageKvBinding *binding,uint64_t lane_entries)
{
	uint32_t index;
	binding->blocks = (SparkKvCacheBlock *)calloc(binding->logical_page_count,sizeof(*binding->blocks));
	binding->resident_slot_logical_block_indices = (uint32_t *)calloc(binding->physical_page_count,sizeof(uint32_t));
	binding->entries = (SparkKvPageCacheEntry *)calloc(binding->logical_page_count,sizeof(*binding->entries));
	binding->sequences = (SparkKvPageCacheSequence *)calloc(binding->resident_sequence_capacity,sizeof(*binding->sequences));
	binding->hash_bucket_heads = (uint32_t *)calloc(binding->logical_page_count,sizeof(uint32_t));
	binding->entry_indices_by_logical_page = (uint32_t *)calloc(binding->logical_page_count,sizeof(uint32_t));
	binding->lanes = (SparkKvLaneTransaction *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lanes));
	binding->logical_pages = (uint32_t *)calloc((size_t)lane_entries,sizeof(uint32_t));
	binding->page_table_shadow = (uint32_t *)malloc((size_t)lane_entries * sizeof(uint32_t));
	binding->lane_bound = (atomic_uchar *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_bound));
	binding->lane_sequence_ids = (atomic_ullong *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_sequence_ids));
	binding->lane_next_positions = (atomic_ullong *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_next_positions));
	binding->owned_lanes = (SparkModelDriverCacheLane *)calloc(binding->resident_sequence_capacity,sizeof(*binding->owned_lanes));
	binding->owned_slots = (uint32_t *)calloc(binding->resident_sequence_capacity,sizeof(*binding->owned_slots));
	if ( binding->blocks == 0 || binding->resident_slot_logical_block_indices == 0 || binding->entries == 0 || binding->sequences == 0 || binding->hash_bucket_heads == 0 || binding->entry_indices_by_logical_page == 0 ||
		binding->lanes == 0 || binding->logical_pages == 0 || binding->page_table_shadow == 0 || binding->lane_bound == 0 || binding->lane_sequence_ids == 0 || binding->lane_next_positions == 0 ||
		binding->owned_lanes == 0 || binding->owned_slots == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaHostAlloc((void **)&binding->physical_pages,(size_t)lane_entries * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess )
	{
		binding->physical_pages = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( cudaHostAlloc((void **)&binding->staging,(size_t)binding->page_bytes,cudaHostAllocPortable) != cudaSuccess )
	{
		binding->staging = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	memset(binding->page_table_shadow,0xff,(size_t)lane_entries * sizeof(uint32_t));
	for (index=0u; index<binding->resident_sequence_capacity; index++)
	{
		atomic_init(&binding->lane_bound[index],0u);
		atomic_init(&binding->lane_sequence_ids[index],0u);
		atomic_init(&binding->lane_next_positions[index],0u);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingGeometry(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	uint64_t packed,missing;
	uint32_t region,owned;
	binding->pages_per_sequence = (configuration->max_sequence_positions + configuration->block_token_count - 1u) / configuration->block_token_count;
	if ( binding->pages_per_sequence == 0u || configuration->resident_sequence_capacity > UINT32_MAX / binding->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	owned = configuration->owner_count <= 1u ? configuration->resident_sequence_capacity :
		(configuration->resident_sequence_capacity + configuration->owner_count - 1u) / configuration->owner_count;
	if ( configuration->physical_page_count < owned * binding->pages_per_sequence || configuration->logical_page_count < configuration->physical_page_count )
	{
		fprintf(stderr,"%s kv binding refused: physical pages %u must cover %u owned lanes x %u pages and logical pages %u must be >= physical\n",
			configuration->module_tag,configuration->physical_page_count,owned,binding->pages_per_sequence,configuration->logical_page_count);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( configuration->max_input_row_count > SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES * configuration->block_token_count )
	{
		fprintf(stderr,"%s kv binding refused: %u input rows exceed one lane transaction (%u pages of %u tokens)\n",
			configuration->module_tag,configuration->max_input_row_count,SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES,configuration->block_token_count);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	binding->page_bytes = 0u;
	for (region=0u; region<configuration->region_count; region++)
	{
		const SparkStageKvRegion *declared = &configuration->regions[region];
		if ( (declared->layout != SPARK_STAGE_KV_REGION_PAGE_MAJOR && declared->layout != SPARK_STAGE_KV_REGION_LAYER_MAJOR) || declared->layer_count == 0u || declared->layer_page_bytes == 0u ||
			declared->layer_page_bytes > UINT64_MAX / declared->layer_count )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		packed = declared->layer_page_bytes * declared->layer_count;
		if ( packed > UINT64_MAX / configuration->physical_page_count || packed > UINT64_MAX - binding->page_bytes )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		binding->regions[region] = *declared;
		binding->region_packed_page_bytes[region] = packed;
		binding->region_layer_stride_bytes[region] = declared->layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? declared->layer_page_bytes : (uint64_t)configuration->physical_page_count * declared->layer_page_bytes;
		binding->page_bytes += packed;
	}
	missing = (uint64_t)(configuration->logical_page_count - configuration->physical_page_count);
	if ( missing != 0u && (configuration->backing_maximum_bytes == 0u || missing > configuration->backing_maximum_bytes / binding->page_bytes) )
	{
		fprintf(stderr,"%s kv binding refused: %llu spill pages of %llu bytes need backing, %llu bytes configured\n",
			configuration->module_tag,(unsigned long long)missing,(unsigned long long)binding->page_bytes,(unsigned long long)configuration->backing_maximum_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static void SparkStageKvBindingFillTable(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration,SparkKvModelTable *table)
{
	memset(table,0,sizeof(*table));
	table->abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table->descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	table->capacity_request = configuration->capacity_request;
	table->arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table->arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table->arena_configuration.logical_block_count = binding->logical_page_count;
	table->arena_configuration.block_token_count = binding->block_token_count;
	table->arena_configuration.resident_block_capacity = binding->physical_page_count;
	table->arena_configuration.layer_count = binding->regions[0].layer_count;
	table->arena_configuration.kv_head_count = configuration->arena_kv_head_count;
	table->arena_configuration.head_dim = configuration->arena_head_dim;
	table->arena_configuration.bytes_per_scalar = configuration->arena_bytes_per_scalar;
	table->arena_configuration.key_device_base = binding->region_base[0];
	table->arena_configuration.key_block_stride_bytes = binding->region_packed_page_bytes[0];
	if ( binding->region_count > 1u )
	{
		table->arena_configuration.value_device_base = binding->region_base[1];
		table->arena_configuration.value_block_stride_bytes = binding->region_packed_page_bytes[1];
	}
	table->arena_configuration.blocks = binding->blocks;
	table->arena_configuration.resident_slot_logical_block_indices = binding->resident_slot_logical_block_indices;
	table->arena_configuration.evict_function = SparkKvPageStoreWriteback;
	table->arena_configuration.evict_context = &binding->page_store;
	table->page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table->page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table->page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table->page_store_config.logical_page_capacity = binding->logical_page_count;
	table->page_store_config.transfer_capacity = binding->logical_page_count < 2u ? binding->logical_page_count : 2u;
	table->page_store_config.page_bytes = binding->page_bytes;
	table->page_store_config.backing_path = configuration->backing_directory;
	table->page_store_config.maximum_backing_bytes = configuration->backing_maximum_bytes > binding->page_bytes ? configuration->backing_maximum_bytes : binding->page_bytes;
	table->page_store_config.staging_address = binding->staging;
	table->page_store_config.staging_bytes = binding->page_bytes;
	table->page_store_config.copy_function = SparkStageKvBindingPageCopy;
	table->page_store_config.copy_context = binding;
	table->sequence_capacity = binding->resident_sequence_capacity;
	table->entry_capacity = binding->logical_page_count;
	table->hash_bucket_count = binding->logical_page_count;
	table->entries = binding->entries;
	table->sequences = binding->sequences;
	table->hash_bucket_heads = binding->hash_bucket_heads;
	table->entry_indices_by_logical_page = binding->entry_indices_by_logical_page;
	table->model_id = configuration->model_id;
	table->model_revision = configuration->model_revision;
	table->cache_layout_fingerprint = configuration->layout_fingerprint;
}

SparkStatus SparkStageKvBindingInitialize(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkKvModelTable table;
	uint64_t lane_entries;
	uint32_t region;
	SparkStatus status;
	cudaError_t error;
	if ( binding == 0 || configuration == 0 || configuration->ledger == 0 || configuration->module_tag == 0 || configuration->block_token_count == 0u || configuration->region_count == 0u ||
		configuration->region_count > SPARK_STAGE_KV_MAX_REGIONS || configuration->resident_sequence_capacity == 0u || configuration->max_sequence_positions == 0u || configuration->pipeline_slot_count == 0u ||
		configuration->model_id == 0 || configuration->model_revision == 0 || configuration->layout_fingerprint == 0 || configuration->physical_page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->backing_directory == 0 || configuration->backing_directory[0] == '\0' )
	{
		fprintf(stderr,"%s kv binding refused: the deployment names no kv_backing_directory\n",configuration->module_tag);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memset(binding,0,sizeof(*binding));
	binding->module_tag = configuration->module_tag;
	binding->block_token_count = configuration->block_token_count;
	binding->region_count = configuration->region_count;
	binding->logical_page_count = configuration->logical_page_count;
	binding->physical_page_count = configuration->physical_page_count;
	binding->resident_sequence_capacity = configuration->resident_sequence_capacity;
	binding->max_sequence_positions = configuration->max_sequence_positions;
	binding->pipeline_slot_count = configuration->pipeline_slot_count;
	binding->owner_rank = configuration->owner_rank;
	binding->owner_count = configuration->owner_count;
	if ( binding->owner_count > 1u && binding->owner_rank >= binding->owner_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvBindingGeometry(binding,configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (region=0u; region<binding->region_count && status==SPARK_STATUS_OK; region++)
		status = SparkStageModuleDeviceAllocate(configuration->ledger,(uint64_t)binding->physical_page_count * binding->region_packed_page_bytes[region],(void **)&binding->region_base[region]);
	lane_entries = (uint64_t)binding->resident_sequence_capacity * binding->pages_per_sequence;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(configuration->ledger,lane_entries * sizeof(uint32_t),(void **)&binding->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemset(binding->page_table,0xff,(size_t)lane_entries * sizeof(uint32_t));
		status = SparkStageModuleCudaStatus(binding->module_tag,error,"kv_page_table");
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAllocateHost(binding,lane_entries);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	binding->transactions.cache = &binding->page_cache;
	binding->transactions.lanes = binding->lanes;
	binding->transactions.logical_pages = binding->logical_pages;
	binding->transactions.physical_pages = binding->physical_pages;
	binding->transactions.page_capacity = binding->pages_per_sequence;
	SparkStageKvBindingFillTable(binding,configuration,&table);
	status = SparkKvBackendInitialize(&table,&binding->arena,&binding->page_cache,&binding->page_store);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pthread_mutex_init(&binding->mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->mutex_initialized = 1u;
	fprintf(stderr,"%s kv binding logical_pages=%u physical_pages=%u pages_per_sequence=%u page_bytes=%llu regions=%u\n",
		binding->module_tag,binding->logical_page_count,binding->physical_page_count,binding->pages_per_sequence,(unsigned long long)binding->page_bytes,binding->region_count);
	return(SPARK_STATUS_OK);
}

void SparkStageKvBindingDestroy(SparkStageKvBinding *binding)
{
	if ( binding == 0 )
		return;
	if ( binding->page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&binding->page_store);
	if ( binding->physical_pages != 0 )
		(void)cudaFreeHost(binding->physical_pages);
	if ( binding->staging != 0 )
		(void)cudaFreeHost(binding->staging);
	free(binding->blocks);
	free(binding->resident_slot_logical_block_indices);
	free(binding->entries);
	free(binding->sequences);
	free(binding->hash_bucket_heads);
	free(binding->entry_indices_by_logical_page);
	free(binding->lanes);
	free(binding->logical_pages);
	free(binding->page_table_shadow);
	free(binding->lane_bound);
	free(binding->lane_sequence_ids);
	free(binding->lane_next_positions);
	free(binding->owned_lanes);
	free(binding->owned_slots);
	if ( binding->mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&binding->mutex);
	memset(binding,0,sizeof(*binding));
}

static uint32_t SparkStageKvBindingFilterLanes(SparkStageKvBinding *binding,const SparkModelDriverCacheLane *lanes,uint32_t count)
{
	uint32_t lane,owned = 0u;
	for (lane=0u; lane<count && owned<binding->resident_sequence_capacity; lane++)
		if ( SparkStageKvBindingOwns(binding,lanes[lane].resident_sequence_slot) != 0u )
			binding->owned_lanes[owned++] = lanes[lane];
	return(owned);
}

static uint32_t SparkStageKvBindingFilterSlots(SparkStageKvBinding *binding,const uint32_t *slots,uint32_t count)
{
	uint32_t lane,owned = 0u;
	for (lane=0u; lane<count && owned<binding->resident_sequence_capacity; lane++)
		if ( SparkStageKvBindingOwns(binding,slots[lane]) != 0u )
			binding->owned_slots[owned++] = slots[lane];
	return(owned);
}

SparkStatus SparkStageKvBindingAdmit(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	SparkModelDriverAdmissionRequest owned;
	SparkStatus status;
	uint32_t lane;
	if ( binding == 0 || request == 0 || decision == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	owned = *request;
	owned.cache_lanes = binding->owned_lanes;
	owned.cache_lane_count = SparkStageKvBindingFilterLanes(binding,request->cache_lanes,request->cache_lane_count);
	owned.active_slot_count = owned.cache_lane_count;
	status = binding->control_generation != 0u && binding->control_generation != request->control_generation ? SPARK_STATUS_VALIDATION_FAILED :
		owned.cache_lane_count == 0u && request->cache_lane_count != 0u ? SPARK_STATUS_OK : SparkKvLaneTransactionsAdmit(&binding->transactions,&owned);
	if ( status == SPARK_STATUS_OK )
		binding->control_generation = request->control_generation;
	if ( status == SPARK_STATUS_OK && (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
			atomic_store_explicit(&binding->lane_bound[request->cache_lanes[lane].resident_sequence_slot],0u,memory_order_release);
	(void)pthread_mutex_unlock(&binding->mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->driver_dispatch_slot = (uint32_t)(request->request_id % binding->pipeline_slot_count);
	decision->driver_dispatch_generation = request->control_generation;
	decision->driver_dispatch_cookie0 = request->transaction_id;
	decision->driver_dispatch_cookie1 = request->submission_id;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageKvBindingReset(SparkStageKvBinding *binding,uint64_t generation)
{
	SparkStatus status;
	uint32_t lane;
	if ( binding == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = generation > binding->reset_generation ? SparkKvLaneTransactionsReset(&binding->transactions) : SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
	{
		memset(binding->page_table_shadow,0xff,(size_t)binding->resident_sequence_capacity * binding->pages_per_sequence * sizeof(uint32_t));
		for (lane=0u; lane<binding->resident_sequence_capacity; lane++)
		{
			atomic_store_explicit(&binding->lane_bound[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_sequence_ids[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_next_positions[lane],0u,memory_order_release);
		}
		binding->reset_generation = generation;
		binding->control_generation = 0u;
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkStageKvBindingClaimAll(atomic_uint *states,uint32_t capacity,uint32_t *indices)
{
	uint32_t index;
	for (index=0u; index<capacity; index++)
		indices[index] = index;
	return(SparkStageModuleIndexSetClaim(states,capacity,indices,capacity));
}

SparkStatus SparkStageKvBindingAdmitReset(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision,atomic_uint *slot_states,atomic_uint *lane_states,void *stream)
{
	uint32_t *slots,*lanes;
	SparkStatus status;
	cudaError_t drain;
	if ( binding == 0 || request == 0 || decision == 0 || slot_states == 0 || lane_states == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkModelDriverInitializeAdmissionDecision(decision);
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (uint32_t *)malloc((size_t)binding->pipeline_slot_count * sizeof(uint32_t));
	lanes = (uint32_t *)malloc((size_t)binding->resident_sequence_capacity * sizeof(uint32_t));
	if ( slots == 0 || lanes == 0 )
	{
		free(slots);
		free(lanes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkStageKvBindingClaimAll(slot_states,binding->pipeline_slot_count,slots);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkStageKvBindingClaimAll(lane_states,binding->resident_sequence_capacity,lanes);
		if ( status == SPARK_STATUS_OK )
		{
			drain = cudaStreamSynchronize((cudaStream_t)stream);
			if ( drain != cudaSuccess )
			{
				(void)SparkStageModuleCudaStatus(binding->module_tag,drain,"kv_reset_stream_drain");
				fprintf(stderr,"%s reset: execution stream failed; lanes and slots stay claimed\n",binding->module_tag);
				free(slots);
				free(lanes);
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			}
			status = SparkStageKvBindingReset(binding,request->control_generation);
			SparkStageModuleIndexSetRelease(lane_states,binding->resident_sequence_capacity,lanes,binding->resident_sequence_capacity);
		}
		SparkStageModuleIndexSetRelease(slot_states,binding->pipeline_slot_count,slots,binding->pipeline_slot_count);
	}
	free(slots);
	free(lanes);
	if ( status == SPARK_STATUS_OK )
	{
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	}
	SPARK_RETURN(status);
}

static uint32_t SparkStageKvPrefixRestorePending(const SparkKvLaneTransaction *owner)
{
	return(owner != 0 && (owner->mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u && (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && owner->lane.sequence_position != 0u);
}

static uint32_t SparkStageKvFrameRestores(const SparkModelDriverFrame *frame,uint32_t slot,uint64_t sequence_id,uint64_t position)
{
	uint32_t lane;
	for (lane=0u; frame != 0 && frame->cache_lanes != 0 && lane<frame->cache_lane_count; lane++)
	{
		const SparkModelDriverCacheLane *cache_lane = &frame->cache_lanes[lane];
		if ( cache_lane->resident_sequence_slot == slot )
			return((cache_lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && cache_lane->prefix_token_count != 0u &&
				cache_lane->sequence_id == sequence_id && cache_lane->sequence_position == position && position != 0u ? 1u : 0u);
	}
	return(0u);
}

static SparkStatus SparkStageKvBindingLoadContinuity(const SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	const SparkKvLaneTransaction *owner;
	uint32_t lane,slot;
	for (lane=0u; lane<active_count; lane++)
	{
		slot = row_resident_slots[lane];
		if ( slot >= binding->resident_sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		bound[lane] = atomic_load_explicit(&binding->lane_bound[slot],memory_order_acquire);
		sequence_ids[lane] = atomic_load_explicit(&binding->lane_sequence_ids[slot],memory_order_acquire);
		next_positions[lane] = atomic_load_explicit(&binding->lane_next_positions[slot],memory_order_acquire);
		owner = &binding->lanes[slot];
		if ( SparkStageKvBindingOwns(binding,slot) == 0u )
		{
			if ( SparkStageKvFrameRestores(frame,slot,row_sequence_ids[lane],row_positions[lane]) != 0u )
			{
				bound[lane] = 1u;
				sequence_ids[lane] = row_sequence_ids[lane];
				next_positions[lane] = row_positions[lane];
			}
			continue;
		}
		if ( SparkStageKvPrefixRestorePending(owner) == 0u )
			continue;
		if ( owner->phase != SPARK_KV_LANE_TRANSACTION_COMMITTED || owner->lane.sequence_id != row_sequence_ids[lane] || owner->lane.sequence_position != row_positions[lane] )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		bound[lane] = 1u;
		sequence_ids[lane] = owner->lane.sequence_id;
		next_positions[lane] = owner->lane.sequence_position;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingRowContinuity(const SparkStageKvBinding *binding,const atomic_uint *lane_states,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	uint8_t *touched;
	uint32_t lane,row,slot;
	SparkStatus status = SPARK_STATUS_OK;
	touched = (uint8_t *)calloc(active_count != 0u ? active_count : 1u,1u);
	if ( touched == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (row=0u; row<row_count && status==SPARK_STATUS_OK; row++)
	{
		slot = row_resident_slots[row];
		status = SparkStageModuleIndexClaimOrdinal(lane_states,binding->resident_sequence_capacity,slot,&lane);
		if ( status != SPARK_STATUS_OK || lane >= active_count || row_resident_slots[lane] != slot || row_positions[row] >= binding->max_sequence_positions )
		{
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
			break;
		}
		if ( row_positions[row] == 0u && touched[lane] == 0u )
		{
			bound[lane] = 1u;
			sequence_ids[lane] = row_sequence_ids[row];
		}
		else if ( row_positions[row] == 0u || bound[lane] == 0u || sequence_ids[lane] != row_sequence_ids[row] || next_positions[lane] != row_positions[row] )
		{
			status = SPARK_STATUS_SCHEMA_ERROR;
			break;
		}
		next_positions[lane] = row_positions[row] + 1u;
		touched[lane] = 1u;
	}
	free(touched);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingContinuity(SparkStageKvBinding *binding,const atomic_uint *lane_states,const SparkModelDriverFrame *frame,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	SparkStatus status;
	if ( binding == 0 || lane_states == 0 || row_resident_slots == 0 || row_sequence_ids == 0 || row_positions == 0 || bound == 0 || sequence_ids == 0 || next_positions == 0 || row_count < active_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkStageKvBindingLoadContinuity(binding,frame,active_count,row_resident_slots,row_sequence_ids,row_positions,bound,sequence_ids,next_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRowContinuity(binding,lane_states,row_count,active_count,row_resident_slots,row_sequence_ids,row_positions,bound,sequence_ids,next_positions);
	(void)pthread_mutex_unlock(&binding->mutex);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingClaim(SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,const uint64_t *next_positions)
{
	uint32_t lane;
	SparkStatus status;
	if ( binding == 0 || frame == 0 || frame->cache_lanes == 0 || frame->cache_lane_count != active_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u || frame->driver_dispatch_slot != (uint32_t)(frame->request_id % binding->pipeline_slot_count) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	for (lane=0u; lane<active_count; lane++)
		if ( frame->cache_lanes[lane].resident_sequence_slot != row_resident_slots[lane] || frame->cache_lanes[lane].sequence_id != row_sequence_ids[lane] ||
			frame->cache_lanes[lane].sequence_position != row_positions[lane] || frame->cache_lanes[lane].context_token_count != next_positions[lane] )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	{
		SparkModelDriverFrame owned = *frame;
		owned.cache_lanes = binding->owned_lanes;
		owned.cache_lane_count = SparkStageKvBindingFilterLanes(binding,frame->cache_lanes,frame->cache_lane_count);
		owned.active_slot_count = owned.cache_lane_count;
		status = owned.cache_lane_count == 0u ? SPARK_STATUS_OK : SparkKvLaneTransactionsClaim(&binding->transactions,&owned);
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingUploadPageTables(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream)
{
	uint64_t offset,bytes;
	uint32_t lane,resident;
	cudaError_t error;
	if ( binding == 0 || resident_slots == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<lane_count; lane++)
	{
		resident = resident_slots[lane];
		if ( resident >= binding->resident_sequence_capacity || binding->lanes[resident].page_count > binding->pages_per_sequence )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( SparkStageKvBindingOwns(binding,resident) == 0u )
			continue;
		offset = (uint64_t)resident * binding->pages_per_sequence;
		bytes = (uint64_t)binding->lanes[resident].page_count * sizeof(uint32_t);
		if ( memcmp(binding->page_table_shadow + offset,binding->physical_pages + offset,(size_t)bytes) == 0 )
			continue;
		error = cudaMemcpyAsync(binding->page_table + offset,binding->physical_pages + offset,(size_t)bytes,cudaMemcpyHostToDevice,(cudaStream_t)stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(binding->module_tag,error,"kv_page_table_upload"));
		memcpy(binding->page_table_shadow + offset,binding->physical_pages + offset,(size_t)bytes);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageKvBindingFinish(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,SparkStatus status,uint32_t extra_tokens,const uint8_t *bound,const uint64_t *sequence_ids,const uint64_t *next_positions)
{
	uint32_t lane,resident;
	SparkStatus result;
	if ( binding == 0 || resident_slots == 0 || bound == 0 || sequence_ids == 0 || next_positions == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	{
		uint32_t owned = SparkStageKvBindingFilterSlots(binding,resident_slots,lane_count);
		result = owned == 0u ? status : SparkKvLaneTransactionsFinish(&binding->transactions,binding->owned_slots,owned,status,extra_tokens);
	}
	for (lane=0u; lane<lane_count; lane++)
	{
		resident = resident_slots[lane];
		if ( resident >= binding->resident_sequence_capacity )
			continue;
		atomic_store_explicit(&binding->lane_bound[resident],result == SPARK_STATUS_OK ? bound[lane] : 0u,memory_order_release);
		if ( result == SPARK_STATUS_OK )
		{
			atomic_store_explicit(&binding->lane_sequence_ids[resident],sequence_ids[lane],memory_order_release);
			atomic_store_explicit(&binding->lane_next_positions[resident],next_positions[lane],memory_order_release);
		}
		else
			memset(binding->page_table_shadow + (uint64_t)resident * binding->pages_per_sequence,0xff,(size_t)binding->pages_per_sequence * sizeof(uint32_t));
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	return(result);
}

static SparkStatus SparkStageKvBindingPublishLanes(SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,const uint32_t *indices)
{
	SparkModelDriverFrame owned;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t lane,resident,owned_slots;
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (lane=0u; lane<frame->cache_lane_count && status==SPARK_STATUS_OK; lane++)
	{
		const SparkModelDriverCacheLane *cache_lane = &frame->cache_lanes[lane];
		resident = indices[lane];
		if ( cache_lane->flags != SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH || cache_lane->publish_token_count == 0u ||
			cache_lane->sequence_position != cache_lane->publish_token_count || cache_lane->context_token_count != cache_lane->publish_token_count ||
			atomic_load_explicit(&binding->lane_bound[resident],memory_order_acquire) == 0u ||
			atomic_load_explicit(&binding->lane_sequence_ids[resident],memory_order_acquire) != cache_lane->sequence_id ||
			atomic_load_explicit(&binding->lane_next_positions[resident],memory_order_acquire) != cache_lane->sequence_position )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status == SPARK_STATUS_OK )
	{
		owned = *frame;
		owned.cache_lanes = binding->owned_lanes;
		owned.cache_lane_count = SparkStageKvBindingFilterLanes(binding,frame->cache_lanes,frame->cache_lane_count);
		owned.active_slot_count = owned.cache_lane_count;
		owned_slots = SparkStageKvBindingFilterSlots(binding,indices,frame->cache_lane_count);
		if ( owned.cache_lane_count != 0u )
		{
			status = SparkKvLaneTransactionsClaim(&binding->transactions,&owned);
			if ( status == SPARK_STATUS_OK )
				status = SparkKvLaneTransactionsFinish(&binding->transactions,binding->owned_slots,owned_slots,SPARK_STATUS_OK,0u);
		}
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingPublishFrame(SparkStageKvBinding *binding,SparkModelDriverFrame *frame,atomic_uint *lane_states)
{
	SparkModelDriverCompletion completion;
	uint32_t *indices,lane;
	SparkStatus status;
	if ( binding == 0 || frame == 0 || lane_states == 0 || binding->mutex_initialized == 0u || frame->completion_function == 0 ||
		frame->flags != (SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH | SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) ||
		frame->new_token_count != 0u || frame->tokens_per_sequence != 0u || frame->cache_lanes == 0 || frame->cache_lane_count == 0u ||
		frame->cache_lane_count != frame->active_slot_count || frame->cache_lane_count > binding->resident_sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->driver_dispatch_slot != (uint32_t)(frame->request_id % binding->pipeline_slot_count) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	indices = (uint32_t *)malloc((size_t)frame->cache_lane_count * sizeof(uint32_t));
	if ( indices == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		indices[lane] = frame->cache_lanes[lane].resident_sequence_slot;
	status = SparkStageModuleIndexSetClaim(lane_states,binding->resident_sequence_capacity,indices,frame->cache_lane_count);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkStageKvBindingPublishLanes(binding,frame,indices);
		SparkStageModuleIndexSetRelease(lane_states,binding->resident_sequence_capacity,indices,frame->cache_lane_count);
	}
	free(indices);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&completion,0,sizeof(completion));
	completion.status = SPARK_STATUS_OK;
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	completion.residency = frame->residency;
	frame->completion_function(frame->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

uint32_t SparkStageKvBindingResidentCount(const SparkStageKvBinding *binding)
{
	uint32_t index,count = 0u;
	if ( binding == 0 || binding->lane_bound == 0 )
		return(0u);
	for (index=0u; index<binding->resident_sequence_capacity; index++)
		count += atomic_load_explicit(&binding->lane_bound[index],memory_order_acquire) != 0u ? 1u : 0u;
	return(count);
}
