#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_weight_codec.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define SPARK_STAGE_KV_COMPLETION_FREE 0u
#define SPARK_STAGE_KV_COMPLETION_QUEUED 1u

static const char *const SparkStageKvLockSiteNames[SPARK_STAGE_KV_LOCK_SITE_COUNT] =
{
	"admit","reset","continuity","claim","finish","publish","save","restore"
};

static uint64_t SparkStageKvNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static SparkStatus SparkStageKvBindingLock(SparkStageKvBinding *binding,uint64_t *start_ns)
{
	if ( pthread_mutex_lock(&binding->mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	*start_ns = SparkStageKvNowNs();
	return(SPARK_STATUS_OK);
}

static void SparkStageKvBindingAccount(SparkStageKvBinding *binding,uint32_t site,uint64_t start_ns)
{
	SparkStageKvLockSite *counter = &binding->counters.lock_sites[site];
	uint64_t held = SparkStageKvNowNs() - start_ns;
	counter->count++;
	counter->total_ns += held;
	if ( held > counter->max_ns )
		counter->max_ns = held;
}

static void SparkStageKvBindingUnlock(SparkStageKvBinding *binding,uint32_t site,uint64_t start_ns)
{
	SparkStageKvBindingAccount(binding,site,start_ns);
	(void)pthread_mutex_unlock(&binding->mutex);
}

static void SparkStageKvBindingWakeSaver(SparkStageKvBinding *binding)
{
	if ( binding->save_started != 0u && SparkKvPageCacheSavePending(&binding->page_cache) != 0u )
		(void)pthread_cond_signal(&binding->save_ready);
}

static SparkStatus SparkStageKvBindingPageCopy(void *context,uint32_t direction,uintptr_t device_address,void *host_address,uint64_t bytes)
{
	const SparkStageKvBinding *binding = (const SparkStageKvBinding *)context;
	cudaStream_t stream;
	enum cudaMemcpyKind kind;
	cudaError_t error;
	uintptr_t base;
	uint64_t span,offset,packed,layer_page,stride;
	uint32_t region,layers;
	if ( binding == 0 || device_address == 0u || host_address == 0 ||
		(direction != SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST && direction != SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	stream = (cudaStream_t)binding->copy_stream;
	kind = direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
	for (region=0u; region<binding->region_count; region++)
	{
		base = (uintptr_t)binding->region_base[region];
		packed = binding->region_packed_page_bytes[region];
		span = ((uint64_t)binding->physical_page_count + binding->shared_page_count) * packed;
		if ( device_address < base || device_address - base >= span )
			continue;
		offset = device_address - base;
		if ( binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR )
		{
			if ( bytes > span - offset )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			error = direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST ?
				cudaMemcpyAsync(host_address,(const void *)device_address,(size_t)bytes,kind,stream) :
				cudaMemcpyAsync((void *)device_address,host_address,(size_t)bytes,kind,stream);
		}
		else
		{
			if ( offset % packed != 0u || bytes != packed )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			layer_page = binding->regions[region].layer_page_bytes;
			layers = binding->regions[region].layer_count;
			stride = binding->region_layer_stride_bytes[region];
			error = direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST ?
				cudaMemcpy2DAsync(host_address,(size_t)layer_page,(const void *)(base + (offset / packed) * layer_page),(size_t)stride,(size_t)layer_page,layers,kind,stream) :
				cudaMemcpy2DAsync((void *)(base + (offset / packed) * layer_page),(size_t)stride,host_address,(size_t)layer_page,(size_t)layer_page,layers,kind,stream);
		}
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
		return(SparkStageModuleCudaStatus(binding->module_tag,error,"kv_page_copy"));
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
	binding->lane_rewind_floors = (atomic_ullong *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_rewind_floors));
	binding->lane_rewind_ceilings = (atomic_ullong *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_rewind_ceilings));
	binding->lane_pending_floors = (atomic_ullong *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_pending_floors));
	if ( binding->blocks == 0 || binding->resident_slot_logical_block_indices == 0 || binding->entries == 0 || binding->sequences == 0 || binding->hash_bucket_heads == 0 || binding->entry_indices_by_logical_page == 0 ||
		binding->lanes == 0 || binding->logical_pages == 0 || binding->page_table_shadow == 0 || binding->lane_bound == 0 || binding->lane_sequence_ids == 0 || binding->lane_next_positions == 0 ||
		binding->lane_rewind_floors == 0 || binding->lane_rewind_ceilings == 0 || binding->lane_pending_floors == 0 )
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
		atomic_init(&binding->lane_rewind_floors[index],0u);
		atomic_init(&binding->lane_rewind_ceilings[index],0u);
		atomic_init(&binding->lane_pending_floors[index],UINT64_MAX);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingStateBudget(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration,uint64_t spill_pages)
{
	uint64_t lane_bytes = configuration->recurrent.lane_bytes,kv_cap,need;
	if ( lane_bytes == 0u )
	{
		binding->page_store_backing_bytes = configuration->backing_maximum_bytes > binding->page_bytes ? configuration->backing_maximum_bytes : binding->page_bytes;
		return(SPARK_STATUS_OK);
	}
	kv_cap = spill_pages * binding->page_bytes;
	if ( kv_cap < binding->page_bytes )
		kv_cap = binding->page_bytes;
	if ( lane_bytes > UINT64_MAX / configuration->logical_page_count )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	need = lane_bytes * configuration->logical_page_count;
	if ( configuration->backing_maximum_bytes < kv_cap || configuration->backing_maximum_bytes - kv_cap < need )
	{
		fprintf(stderr,"%s kv binding refused: %u pages of recurrent state at %llu bytes need %llu backing bytes beyond %llu page-store bytes, %llu configured\n",
			configuration->module_tag,configuration->logical_page_count,(unsigned long long)lane_bytes,(unsigned long long)need,(unsigned long long)kv_cap,
			(unsigned long long)configuration->backing_maximum_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	binding->page_store_backing_bytes = kv_cap;
	binding->state_slot_count = configuration->logical_page_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingGeometry(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	uint64_t packed,missing,layer_page;
	uint32_t region,shards,in_flight;
	binding->pages_per_sequence = (configuration->max_sequence_positions + configuration->block_token_count - 1u) / configuration->block_token_count;
	if ( binding->pages_per_sequence == 0u || configuration->resident_sequence_capacity > UINT32_MAX / binding->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( configuration->physical_page_count < binding->pages_per_sequence || configuration->logical_page_count < configuration->resident_sequence_capacity * binding->pages_per_sequence ||
		configuration->logical_page_count < configuration->physical_page_count )
	{
		fprintf(stderr,"%s kv binding refused: physical pages %u must hold one lane's %u pages and logical pages %u must cover %u lanes x %u pages and be >= physical\n",
			configuration->module_tag,configuration->physical_page_count,binding->pages_per_sequence,configuration->logical_page_count,configuration->resident_sequence_capacity,binding->pages_per_sequence);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( configuration->max_input_row_count > SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES * configuration->block_token_count )
	{
		fprintf(stderr,"%s kv binding refused: %u input rows exceed one lane transaction (%u pages of %u tokens)\n",
			configuration->module_tag,configuration->max_input_row_count,SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES,configuration->block_token_count);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	shards = SparkStageKvBindingContextSharded(configuration) != 0u ? configuration->context_shard.degree : 1u;
	if ( shards > 1u && (SparkKvShardValid(configuration->context_shard,configuration->block_token_count) == 0u ||
		configuration->arena_head_dim % shards != 0u) )
	{
		fprintf(stderr,"%s kv binding refused: context shard degree %u rank %u grain %u cannot split %u-token pages (arena head %u)\n",
			configuration->module_tag,configuration->context_shard.degree,configuration->context_shard.rank,configuration->context_shard.grain,configuration->block_token_count,configuration->arena_head_dim);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	binding->page_bytes = 0u;
	for (region=0u; region<configuration->region_count; region++)
	{
		const SparkStageKvRegion *declared = &configuration->regions[region];
		if ( (declared->layout != SPARK_STAGE_KV_REGION_PAGE_MAJOR && declared->layout != SPARK_STAGE_KV_REGION_LAYER_MAJOR) || declared->layer_count == 0u || declared->layer_page_bytes == 0u ||
			declared->layer_page_bytes % shards != 0u || declared->layer_page_bytes > UINT64_MAX / declared->layer_count )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		layer_page = declared->layer_page_bytes / shards;
		packed = layer_page * declared->layer_count;
		if ( packed > UINT64_MAX / configuration->physical_page_count || packed > UINT64_MAX - binding->page_bytes )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		binding->regions[region] = *declared;
		binding->regions[region].layer_page_bytes = layer_page;
		binding->region_packed_page_bytes[region] = packed;
		binding->region_layer_stride_bytes[region] = declared->layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? layer_page : (uint64_t)configuration->physical_page_count * layer_page;
		binding->page_bytes += packed;
	}
	missing = (uint64_t)(configuration->logical_page_count - configuration->physical_page_count);
	in_flight = configuration->logical_page_count < 2u ? configuration->logical_page_count : 2u;
	if ( missing != 0u && (configuration->backing_maximum_bytes == 0u || missing + in_flight > configuration->backing_maximum_bytes / binding->page_bytes) )
	{
		fprintf(stderr,"%s kv binding refused: %llu spill pages and %u in flight of %llu bytes need backing, %llu bytes configured\n",
			configuration->module_tag,(unsigned long long)missing,in_flight,(unsigned long long)binding->page_bytes,(unsigned long long)configuration->backing_maximum_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SparkStageKvBindingStateBudget(binding,configuration,missing + in_flight));
}

static SparkStatus SparkStageKvBindingPark(void *context,uint32_t logical_page_index,uint32_t physical_page_index,uint64_t generation,uintptr_t key_device_address,uint64_t key_bytes,uintptr_t value_device_address,uint64_t value_bytes)
{
	SparkStageKvBinding *binding = (SparkStageKvBinding *)context;
	SparkStatus status = SparkKvPageStoreWriteback(&binding->page_store,logical_page_index,physical_page_index,generation,key_device_address,key_bytes,value_device_address,value_bytes);
	if ( status == SPARK_STATUS_OK && binding->sync_initialized != 0u && SparkKvPageCacheSaveParked(&binding->page_cache,logical_page_index) != 0u )
		(void)pthread_cond_signal(&binding->save_ready);
	return(status);
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
	table->arena_configuration.head_dim = configuration->arena_head_dim / (SparkStageKvBindingContextSharded(configuration) != 0u ? configuration->context_shard.degree : 1u);
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
	table->arena_configuration.evict_function = SparkStageKvBindingPark;
	table->arena_configuration.evict_context = binding;
	table->page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table->page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table->page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table->page_store_config.logical_page_capacity = binding->logical_page_count;
	table->page_store_config.transfer_capacity = binding->logical_page_count < 2u ? binding->logical_page_count : 2u;
	table->page_store_config.page_bytes = binding->page_bytes;
	table->page_store_config.backing_path = configuration->backing_directory;
	table->page_store_config.maximum_backing_bytes = binding->page_store_backing_bytes;
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
}

static SparkStatus SparkStageKvBindingAttachCopier(SparkStageKvBinding *binding)
{
	SparkKvDeviceCopyConfiguration copy;
	SparkKvPageCacheDeviceCopy hook;
	uint32_t region;
	SparkStatus status;
	memset(&copy,0,sizeof(copy));
	copy.module_tag = binding->module_tag;
	copy.stream = binding->copy_stream;
	copy.arena = &binding->arena;
	copy.physical_page_count = binding->physical_page_count + binding->shared_page_count;
	copy.region_count = binding->region_count;
	for (region=0u; region<binding->region_count; region++)
	{
		copy.regions[region].layout = binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? SPARK_KV_DEVICE_COPY_PAGE_MAJOR : SPARK_KV_DEVICE_COPY_LAYER_MAJOR;
		copy.regions[region].layer_count = binding->regions[region].layer_count;
		copy.regions[region].device_base = (uintptr_t)binding->region_base[region];
		copy.regions[region].layer_page_bytes = binding->regions[region].layer_page_bytes;
		copy.regions[region].layer_stride_bytes = binding->region_layer_stride_bytes[region];
	}
	copy.pending_capacity = 2u * binding->resident_sequence_capacity;
	status = SparkKvDeviceCopierInitialize(&binding->copier,&copy);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	binding->copier_initialized = 1u;
	hook.copy_page = SparkKvDeviceCopierCopyPage;
	hook.retire_copies = SparkKvDeviceCopierRetire;
	hook.destination_pins = SparkKvDeviceCopierDestinationPins;
	hook.defer_free = SparkKvDeviceCopierDeferFree;
	hook.context = &binding->copier;
	return(SparkKvPageCacheAttachDeviceCopy(&binding->page_cache,&hook));
}

static SparkStatus SparkStageKvBindingAllocateCompletions(SparkStageKvBinding *binding)
{
	SparkStageKvBindingCompletionRecord *record;
	uint32_t slot,lanes = binding->resident_sequence_capacity;
	binding->completion_records = (SparkStageKvBindingCompletionRecord *)calloc(binding->pipeline_slot_count,sizeof(*binding->completion_records));
	binding->completion_queue = (uint32_t *)calloc(binding->pipeline_slot_count,sizeof(uint32_t));
	if ( binding->completion_records == 0 || binding->completion_queue == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (slot=0u; slot<binding->pipeline_slot_count; slot++)
	{
		record = &binding->completion_records[slot];
		record->resident_slots = (uint32_t *)calloc(lanes,sizeof(uint32_t));
		record->bound = (uint8_t *)calloc(lanes,sizeof(uint8_t));
		record->sequence_ids = (uint64_t *)calloc(lanes,sizeof(uint64_t));
		record->next_positions = (uint64_t *)calloc(lanes,sizeof(uint64_t));
		if ( record->resident_slots == 0 || record->bound == 0 || record->sequence_ids == 0 || record->next_positions == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static void SparkStageKvBindingFreeCompletions(SparkStageKvBinding *binding)
{
	uint32_t slot;
	if ( binding->completion_records != 0 )
		for (slot=0u; slot<binding->pipeline_slot_count; slot++)
		{
			free(binding->completion_records[slot].resident_slots);
			free(binding->completion_records[slot].bound);
			free(binding->completion_records[slot].sequence_ids);
			free(binding->completion_records[slot].next_positions);
		}
	free(binding->completion_records);
	free(binding->completion_queue);
	binding->completion_records = 0;
	binding->completion_queue = 0;
}

static void *SparkStageKvBindingCompletionMain(void *context);
static SparkStatus SparkStageKvBindingRecurrentAdmit(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,uint64_t *held);
static void *SparkStageKvBindingSaveMain(void *context);

static SparkStageKvRestoreSlot *SparkStageKvBindingRestoreFind(SparkStageKvBinding *binding,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	uint32_t index;
	for (index=0u; index<SPARK_STAGE_KV_RESTORE_SLOTS; index++)
		if ( binding->restores[index].state != SPARK_STAGE_KV_RESTORE_FREE && binding->restores[index].token_count == token_count &&
			memcmp(&binding->restores[index].identity,identity,sizeof(*identity)) == 0 )
			return(&binding->restores[index]);
	return(0);
}

static SparkStageKvRestoreSlot *SparkStageKvBindingRestoreFree(SparkStageKvBinding *binding)
{
	SparkStageKvRestoreSlot *oldest = 0;
	uint32_t index;
	for (index=0u; index<SPARK_STAGE_KV_RESTORE_SLOTS; index++)
	{
		if ( binding->restores[index].state == SPARK_STAGE_KV_RESTORE_FREE )
			return(&binding->restores[index]);
		if ( binding->restores[index].state == SPARK_STAGE_KV_RESTORE_DONE && (oldest == 0 || binding->restores[index].done_ns < oldest->done_ns) )
			oldest = &binding->restores[index];
	}
	return(oldest);
}

static uint32_t SparkStageKvBindingRestoreWanted(SparkStageKvBinding *binding,const SparkModelDriverCacheLane *lane)
{
	SparkStageKvRestoreSlot *slot,*free_slot;
	uint32_t consumed = 0u;
	if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) == 0u || lane->prefix_token_count == 0u ||
		binding->page_cache.sequences[lane->resident_sequence_slot].sequence_id == lane->sequence_id )
		return(0u);
	slot = SparkStageKvBindingRestoreFind(binding,&lane->prefix_identity,lane->prefix_token_count);
	if ( slot != 0 && slot->state == SPARK_STAGE_KV_RESTORE_DONE )
	{
		slot->state = SPARK_STAGE_KV_RESTORE_FREE;
		consumed = 1u;
		slot = 0;
	}
	if ( consumed != 0u || SparkKvPageCachePrefixReady(&binding->page_cache,&lane->prefix_identity,lane->prefix_token_count) != 0u ||
		SparkKvPageCacheImportShared(&binding->page_cache,&lane->prefix_identity,lane->prefix_token_count) == SPARK_STATUS_OK )
		return(0u);
	if ( slot != 0 )
		return(1u);
	free_slot = SparkStageKvBindingRestoreFree(binding);
	if ( free_slot == 0 )
		return(1u);
	memset(free_slot,0,sizeof(*free_slot));
	free_slot->identity = lane->prefix_identity;
	free_slot->token_count = lane->prefix_token_count;
	free_slot->state = SPARK_STAGE_KV_RESTORE_QUEUED;
	binding->restore_jobs++;
	(void)pthread_cond_signal(&binding->restore_ready);
	return(1u);
}

static void SparkStageKvBindingRestoreQueue(SparkStageKvBinding *binding,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	SparkStageKvRestoreSlot *slot;
	if ( SparkKvPageCachePrefixReady(&binding->page_cache,identity,token_count) != 0u || SparkStageKvBindingRestoreFind(binding,identity,token_count) != 0 ||
		SparkKvPageCacheImportShared(&binding->page_cache,identity,token_count) == SPARK_STATUS_OK )
		return;
	slot = SparkStageKvBindingRestoreFree(binding);
	if ( slot == 0 )
		return;
	memset(slot,0,sizeof(*slot));
	slot->identity = *identity;
	slot->token_count = token_count;
	slot->state = SPARK_STAGE_KV_RESTORE_QUEUED;
	binding->restore_jobs++;
	binding->restore_hinted_jobs++;
	(void)pthread_cond_signal(&binding->restore_ready);
}

static SparkStatus SparkStageKvBindingHint(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	uint64_t held;
	uint32_t lane;
	SparkStatus status;
	SparkModelDriverInitializeAdmissionDecision(decision);
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvBindingLock(binding,&held);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	binding->restore_hints++;
	if ( binding->page_cache.snapshot != 0 && binding->restore_started != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
			SparkStageKvBindingRestoreQueue(binding,&request->cache_lanes[lane].prefix_identity,request->cache_lanes[lane].prefix_token_count);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_RESTORE,held);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingRestoreGate(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t lane,waiting = 0u;
	if ( request->admission_flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE || binding->page_cache.snapshot == 0 || binding->restore_started == 0u )
		return(SPARK_STATUS_OK);
	for (lane=0u; lane<request->cache_lane_count; lane++)
		if ( binding->transactions.lanes[request->cache_lanes[lane].resident_sequence_slot].phase == SPARK_KV_LANE_TRANSACTION_EMPTY )
			waiting += SparkStageKvBindingRestoreWanted(binding,&request->cache_lanes[lane]);
	if ( waiting == 0u )
		return(SPARK_STATUS_OK);
	binding->restore_pending_answers++;
	return(SPARK_STATUS_PENDING);
}

static SparkStatus SparkStageKvBindingRestoreRun(SparkStageKvBinding *binding,SparkStageKvRestoreSlot *slot)
{
	SparkKvPageCacheRestoreJob *job = &binding->restore_job;
	SparkStatus status;
	status = SparkKvPageCacheRestoreBegin(&binding->page_cache,job,&slot->identity,slot->token_count);
	if ( status != SPARK_STATUS_PENDING )
		return(status);
	(void)pthread_mutex_unlock(&binding->mutex);
	status = SparkKvPageCacheRestoreReadChain(&binding->page_cache,job);
	(void)pthread_mutex_lock(&binding->mutex);
	while ( status == SPARK_STATUS_OK )
	{
		status = SparkKvPageCacheRestoreAdvance(&binding->page_cache,job);
		if ( status != SPARK_STATUS_OK || job->need == SPARK_KV_PAGE_CACHE_RESTORE_DONE || binding->restore_stop != 0u )
			break;
		(void)pthread_mutex_unlock(&binding->mutex);
		status = SparkKvPageCacheRestoreRead(&binding->page_cache,job);
		(void)pthread_mutex_lock(&binding->mutex);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageCacheRestoreApply(&binding->page_cache,job);
	}
	binding->restore_imported_pages += job->imported_pages;
	return(SparkKvPageCacheRestoreFinish(&binding->page_cache,job,status));
}

static void *SparkStageKvBindingRestoreMain(void *context)
{
	SparkStageKvBinding *binding = (SparkStageKvBinding *)context;
	SparkStageKvRestoreSlot *slot;
	uint32_t index;
	(void)pthread_mutex_lock(&binding->mutex);
	while ( binding->restore_stop == 0u )
	{
		slot = 0;
		for (index=0u; slot == 0 && index<SPARK_STAGE_KV_RESTORE_SLOTS; index++)
			if ( binding->restores[index].state == SPARK_STAGE_KV_RESTORE_QUEUED )
				slot = &binding->restores[index];
		if ( slot == 0 )
		{
			(void)pthread_cond_wait(&binding->restore_ready,&binding->mutex);
			continue;
		}
		slot->state = SPARK_STAGE_KV_RESTORE_RUNNING;
		slot->status = SparkStageKvBindingRestoreRun(binding,slot);
		slot->done_ns = SparkStageKvNowNs();
		slot->state = SPARK_STAGE_KV_RESTORE_DONE;
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	return(0);
}

static SparkStatus SparkStageKvBindingStartRestore(SparkStageKvBinding *binding)
{
	binding->restore_job.links = (SparkKvPageCacheSnapshotLink *)calloc(binding->pages_per_sequence,sizeof(*binding->restore_job.links));
	if ( cudaHostAlloc((void **)&binding->restore_job.page,(size_t)binding->page_bytes,cudaHostAllocPortable) != cudaSuccess )
		binding->restore_job.page = 0;
	if ( binding->page_cache.state_store != 0 )
		binding->restore_job.state = (uint8_t *)malloc((size_t)binding->page_cache.state_store->page_bytes);
	if ( binding->restore_job.links == 0 || binding->restore_job.page == 0 || (binding->page_cache.state_store != 0 && binding->restore_job.state == 0) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( pthread_cond_init(&binding->restore_ready,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->restore_ready_initialized = 1u;
	binding->transactions.restore_async = 1u;
	if ( pthread_create(&binding->restore_thread,0,SparkStageKvBindingRestoreMain,binding) != 0 )
	{
		binding->transactions.restore_async = 0u;
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	binding->restore_started = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingStartAsync(SparkStageKvBinding *binding)
{
	SparkStatus status;
	status = SparkStageKvBindingAttachCopier(binding);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAllocateCompletions(binding);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pthread_mutex_init(&binding->completion_mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( pthread_cond_init(&binding->completion_ready,0) != 0 || pthread_cond_init(&binding->completion_idle,0) != 0 ||
		pthread_cond_init(&binding->save_ready,0) != 0 || pthread_cond_init(&binding->save_idle,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->sync_initialized = 1u;
	if ( pthread_create(&binding->completion_thread,0,SparkStageKvBindingCompletionMain,binding) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->completion_started = 1u;
	if ( pthread_create(&binding->save_thread,0,SparkStageKvBindingSaveMain,binding) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->save_started = 1u;
	status = SparkStageKvBindingStartRestore(binding);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	fprintf(stderr,"%s kv binding async completion_records=%u copy_on_write=device copy_stream=nonblocking restore=worker\n",binding->module_tag,binding->pipeline_slot_count);
	return(SPARK_STATUS_OK);
}

static void SparkStageKvDigestU32(SparkSha256Context *context,uint32_t value)
{
	uint8_t bytes[sizeof(uint32_t)];
	uint32_t index;
	for (index=0u; index<sizeof(bytes); index++)
		bytes[index] = (uint8_t)(value >> (8u * index));
	SparkSha256Update(context,bytes,sizeof(bytes));
}

static void SparkStageKvDigestU64(SparkSha256Context *context,uint64_t value)
{
	uint8_t bytes[sizeof(uint64_t)];
	uint32_t index;
	for (index=0u; index<sizeof(bytes); index++)
		bytes[index] = (uint8_t)(value >> (8u * index));
	SparkSha256Update(context,bytes,sizeof(bytes));
}

static void SparkStageKvDigestText(SparkSha256Context *context,const char *text)
{
	size_t length = strlen(text);
	SparkStageKvDigestU32(context,(uint32_t)length);
	SparkSha256Update(context,text,length);
}

static uint32_t SparkStageKvDigestIsZero(const uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
	uint32_t index;
	uint8_t any = 0u;
	for (index=0u; index<SPARK_SHA256_DIGEST_BYTES; index++)
		any |= digest[index];
	return(any == 0u ? 1u : 0u);
}

static const char *SparkStageKvLayoutInvalid(const SparkStageKvLayoutIdentity *identity)
{
	uint32_t region;
	if ( identity->model_id == 0 || identity->model_id[0] == '\0' )
		return("model_id");
	if ( identity->model_revision == 0 || identity->model_revision[0] == '\0' )
		return("model_revision");
	if ( SparkStageKvDigestIsZero(identity->pack_sha256) != 0u )
		return("pack_sha256");
	if ( SparkStageKvDigestIsZero(identity->contract_sha256) != 0u )
		return("contract_sha256");
	if ( SparkStageKvDigestIsZero(identity->driver_sha256) != 0u )
		return("driver_sha256");
	if ( SparkWeightCodecIsKnown(identity->expert_codec) == 0u )
		return("expert_codec");
	if ( SparkWeightCodecIsKnown(identity->kv_codec) == 0u )
		return("kv_codec");
	if ( identity->context_shard.degree > 1u && identity->context_shard.rank >= identity->context_shard.degree )
		return("context_shard");
	if ( identity->block_token_count == 0u )
		return("block_token_count");
	if ( identity->region_count == 0u || identity->region_count > SPARK_STAGE_KV_MAX_REGIONS )
		return("region_count");
	for (region=0u; region<identity->region_count; region++)
		if ( identity->regions[region].layer_count == 0u || identity->regions[region].layer_page_bytes == 0u )
			return("region");
	return(identity->page_bytes == 0u ? "page_bytes" : 0);
}

SparkStatus SparkStageKvLayoutDigest(const SparkStageKvLayoutIdentity *identity,uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES],const char **invalid_input)
{
	static const char tag[] = "sparkpipe-kv-layout-v1";
	SparkSha256Context context;
	const char *invalid;
	uint32_t region,sharded;
	if ( invalid_input != 0 )
		*invalid_input = 0;
	invalid = identity == 0 || layout_sha256 == 0 ? "identity" : SparkStageKvLayoutInvalid(identity);
	if ( invalid != 0 )
	{
		if ( invalid_input != 0 )
			*invalid_input = invalid;
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	sharded = identity->context_shard.degree > 1u ? 1u : 0u;
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,tag,sizeof(tag) - 1u);
	SparkStageKvDigestText(&context,identity->model_id);
	SparkStageKvDigestText(&context,identity->model_revision);
	SparkSha256Update(&context,identity->pack_sha256,SPARK_SHA256_DIGEST_BYTES);
	SparkSha256Update(&context,identity->contract_sha256,SPARK_SHA256_DIGEST_BYTES);
	SparkSha256Update(&context,identity->driver_sha256,SPARK_SHA256_DIGEST_BYTES);
	SparkStageKvDigestU32(&context,identity->expert_codec);
	SparkStageKvDigestU32(&context,identity->kv_codec);
	SparkStageKvDigestU32(&context,sharded != 0u ? identity->context_shard.degree : 1u);
	SparkStageKvDigestU32(&context,sharded != 0u ? identity->context_shard.rank : 0u);
	SparkStageKvDigestU32(&context,sharded != 0u ? identity->context_shard.grain : 0u);
	SparkStageKvDigestU32(&context,identity->block_token_count);
	SparkStageKvDigestU32(&context,identity->region_count);
	for (region=0u; region<identity->region_count; region++)
	{
		SparkStageKvDigestU32(&context,identity->regions[region].layout);
		SparkStageKvDigestU32(&context,identity->regions[region].layer_count);
		SparkStageKvDigestU64(&context,identity->regions[region].layer_page_bytes);
	}
	SparkStageKvDigestU64(&context,identity->page_bytes);
	SparkStageKvDigestU64(&context,identity->state_page_bytes);
	SparkSha256Finalize(&context,layout_sha256);
	return(SPARK_STATUS_OK);
}

static uint64_t SparkStageKvAlign(uint64_t bytes)
{
	return((bytes + SPARK_KV_SNAPSHOT_ALIGNMENT - 1u) / SPARK_KV_SNAPSHOT_ALIGNMENT * SPARK_KV_SNAPSHOT_ALIGNMENT);
}

static SparkStatus SparkStageKvBindingLayout(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkStageKvLayoutIdentity identity;
	const char *invalid = 0;
	SparkStatus status;
	memset(&identity,0,sizeof(identity));
	status = SparkKvSnapshotBinaryDigest(configuration->driver_symbol,identity.driver_sha256,binding->driver_path,sizeof(binding->driver_path));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv binding refused: cannot hash the driver binary status=%s\n",binding->module_tag,SparkStatusToString(status));
		SPARK_RETURN(status);
	}
	identity.model_id = configuration->model_id;
	identity.model_revision = configuration->model_revision;
	memcpy(identity.pack_sha256,configuration->pack_sha256,SPARK_SHA256_DIGEST_BYTES);
	memcpy(identity.contract_sha256,configuration->contract_sha256,SPARK_SHA256_DIGEST_BYTES);
	identity.expert_codec = configuration->expert_codec;
	identity.kv_codec = configuration->kv_codec;
	identity.context_shard = configuration->context_shard;
	identity.block_token_count = binding->block_token_count;
	identity.region_count = binding->region_count;
	memcpy(identity.regions,binding->regions,sizeof(identity.regions));
	identity.page_bytes = binding->page_bytes;
	identity.state_page_bytes = binding->page_cache.state_store != 0 ? binding->page_cache.state_store->page_bytes : 0u;
	status = SparkStageKvLayoutDigest(&identity,binding->layout_sha256,&invalid);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv binding refused: layout digest input %s is invalid\n",binding->module_tag,invalid != 0 ? invalid : "identity");
		SPARK_RETURN(status);
	}
	SparkSha256DigestToHex(binding->layout_sha256,binding->layout_hex);
	binding->snapshot_page_file_bytes = SparkStageKvAlign(sizeof(SparkKvSnapshotFileHeader)) +
		SparkStageKvAlign((uint64_t)binding->pages_per_sequence * sizeof(SparkKvPageCacheSnapshotLink)) +
		SparkStageKvAlign(binding->page_bytes) + (identity.state_page_bytes != 0u ? SparkStageKvAlign(identity.state_page_bytes) : 0u);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingOpenStore(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkStatus status;
	if ( binding->snapshot_page_file_bytes > configuration->snapshot_maximum_bytes )
	{
		fprintf(stderr,"%s kv binding refused: kv_snapshot_maximum_bytes %llu cannot hold one %llu-byte page file\n",binding->module_tag,
			(unsigned long long)configuration->snapshot_maximum_bytes,(unsigned long long)binding->snapshot_page_file_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkKvSnapshotStoreOpen(&binding->snapshot_store,configuration->snapshot_directory,configuration->snapshot_maximum_bytes,
		binding->snapshot_page_file_bytes * SPARK_STAGE_KV_SNAPSHOT_QUEUE_PAGES);
	if ( status != SPARK_STATUS_OK )
		memset(&binding->snapshot_store,0,sizeof(binding->snapshot_store));
	if ( status == SPARK_STATUS_BUSY )
	{
		fprintf(stderr,"%s kv binding refused: snapshot directory %s is locked by another engine\n",binding->module_tag,configuration->snapshot_directory);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv binding refused: snapshot store %s open status=%s\n",binding->module_tag,configuration->snapshot_directory,SparkStatusToString(status));
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingAttachSnapshot(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration,uint64_t start_ns)
{
	uint64_t matching = 0u,foreign = 0u;
	SparkStatus status;
	binding->snapshot_links = (SparkKvPageCacheSnapshotLink *)calloc(binding->pages_per_sequence,sizeof(*binding->snapshot_links));
	binding->snapshot_pending = (uint32_t *)calloc(binding->logical_page_count,sizeof(*binding->snapshot_pending));
	if ( binding->page_cache.state_store != 0 )
		binding->snapshot_state = (uint8_t *)malloc((size_t)binding->page_cache.state_store->page_bytes);
	binding->save_order = (SparkKvPageCacheSaveOrder *)calloc(binding->logical_page_count,sizeof(*binding->save_order));
	if ( cudaHostAlloc((void **)&binding->snapshot_page,(size_t)binding->page_bytes,cudaHostAllocPortable) != cudaSuccess )
		binding->snapshot_page = 0;
	if ( binding->snapshot_links == 0 || binding->snapshot_pending == 0 || binding->save_order == 0 || binding->snapshot_page == 0 ||
		(binding->page_cache.state_store != 0 && binding->snapshot_state == 0) )
	{
		fprintf(stderr,"%s kv binding refused: cannot allocate snapshot buffers\n",binding->module_tag);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	memset(&binding->snapshot,0,sizeof(binding->snapshot));
	binding->snapshot.store = &binding->snapshot_store;
	memcpy(binding->snapshot.layout_sha256,binding->layout_sha256,SPARK_SHA256_DIGEST_BYTES);
	binding->snapshot.page_capacity = binding->pages_per_sequence;
	binding->snapshot.links = binding->snapshot_links;
	binding->snapshot.page = binding->snapshot_page;
	binding->snapshot.state = binding->snapshot_state;
	binding->snapshot.pending_terminals = binding->snapshot_pending;
	binding->snapshot.pending_capacity = binding->logical_page_count;
	status = SparkKvPageCacheAttachSnapshot(&binding->page_cache,&binding->snapshot);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv binding refused: snapshot attach status=%s\n",binding->module_tag,SparkStatusToString(status));
		SPARK_RETURN(status);
	}
	(void)SparkKvSnapshotCountLayout(&binding->snapshot_store,binding->layout_sha256,&matching,&foreign);
	fprintf(stderr,"%s kv snapshot store directory=%s maximum_bytes=%llu queue_bytes=%llu page_file_bytes=%llu used_bytes=%llu files=%llu matching_files=%llu foreign_layout_files=%llu evicted_at_open=%llu removed_temporaries=%llu driver=%s identity_us=%llu layout=%s\n",
		binding->module_tag,configuration->snapshot_directory,(unsigned long long)binding->snapshot_store.maximum_bytes,(unsigned long long)binding->snapshot_store.queue_maximum_bytes,
		(unsigned long long)binding->snapshot_page_file_bytes,(unsigned long long)binding->snapshot_store.used_bytes,(unsigned long long)binding->snapshot_store.file_count,
		(unsigned long long)matching,(unsigned long long)foreign,(unsigned long long)binding->snapshot_store.eviction_count,
		(unsigned long long)binding->snapshot_store.removed_temporary_count,binding->driver_path,(unsigned long long)((SparkStageKvNowNs() - start_ns) / 1000u),binding->layout_hex);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingOpenSnapshot(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	uint64_t start_ns = SparkStageKvNowNs();
	SparkStatus status;
	status = SparkStageKvBindingLayout(binding,configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingOpenStore(binding,configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAttachSnapshot(binding,configuration,start_ns);
	SPARK_RETURN(status);
}

static const char *SparkStageKvBindingMissingIdentity(const SparkStageKvConfiguration *configuration)
{
	if ( configuration->model_id == 0 || configuration->model_id[0] == '\0' )
		return("model_id");
	if ( configuration->model_revision == 0 || configuration->model_revision[0] == '\0' )
		return("model_revision");
	if ( SparkStageKvDigestIsZero(configuration->pack_sha256) != 0u )
		return("pack_sha256");
	if ( SparkStageKvDigestIsZero(configuration->contract_sha256) != 0u )
		return("contract_sha256");
	if ( configuration->expert_codec == 0u )
		return("expert_codec");
	if ( configuration->kv_codec == 0u )
		return("kv_codec");
	return(configuration->driver_symbol == 0 ? "driver_symbol" : 0);
}

static SparkStatus SparkStageKvBindingAllocateStates(SparkStageKvBinding *binding)
{
	uint64_t bytes = (uint64_t)binding->resident_sequence_capacity * binding->recurrent.lane_bytes;
	if ( cudaHostAlloc((void **)&binding->lane_state,(size_t)bytes,cudaHostAllocPortable) != cudaSuccess )
		binding->lane_state = 0;
	binding->state_staging = (uint8_t *)malloc((size_t)binding->recurrent.lane_bytes);
	binding->lane_state_flags = (uint8_t *)calloc(binding->resident_sequence_capacity,sizeof(*binding->lane_state_flags));
	if ( binding->lane_state == 0 || binding->state_staging == 0 || binding->lane_state_flags == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingAttachStates(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkKvPageStoreConfiguration store;
	SparkStatus status;
	memset(&store,0,sizeof(store));
	store.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	store.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	store.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	store.logical_page_capacity = binding->logical_page_count;
	store.transfer_capacity = 1u;
	store.page_bytes = binding->recurrent.lane_bytes;
	store.maximum_backing_bytes = (uint64_t)binding->state_slot_count * binding->recurrent.lane_bytes;
	store.backing_path = configuration->backing_directory;
	store.staging_address = binding->state_staging;
	store.staging_bytes = binding->recurrent.lane_bytes;
	status = SparkKvPageStoreInitialize(&binding->state_store,&store);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheAttachStateStore(&binding->page_cache,&binding->state_store);
	fprintf(stderr,"%s kv binding recurrent lane_bytes=%llu state_slots=%u state_backing_bytes=%llu page_store_backing_bytes=%llu status=%s\n",binding->module_tag,
		(unsigned long long)binding->recurrent.lane_bytes,binding->state_slot_count,(unsigned long long)store.maximum_backing_bytes,
		(unsigned long long)binding->page_store_backing_bytes,SparkStatusToString(status));
	SPARK_RETURN(status);
}

typedef struct SparkStageKvPoolSlot
{
	uint32_t need;
	uint32_t reserved0;
	uint64_t offset;
} SparkStageKvPoolSlot;

static int SparkStageKvPoolSlotOrder(const void *left,const void *right)
{
	const SparkStageKvPoolSlot *a = (const SparkStageKvPoolSlot *)left,*b = (const SparkStageKvPoolSlot *)right;
	if ( a->need != b->need )
		return(a->need < b->need ? -1 : 1);
	return(a->offset < b->offset ? -1 : a->offset > b->offset ? 1 : 0);
}

static uint32_t SparkStageKvPoolSlotNeed(const SparkStageKvBinding *binding,uint32_t region,uint64_t local,uint64_t chunk)
{
	uint64_t layer_page,layer_span,layer;
	if ( binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR )
		return((uint32_t)(local / binding->region_packed_page_bytes[region]) + 1u);
	layer_page = binding->regions[region].layer_page_bytes;
	layer_span = binding->region_layer_stride_bytes[region];
	layer = local / layer_span;
	if ( local + chunk > (layer + 1u) * layer_span && layer + 1u < binding->regions[region].layer_count )
		return(1u);
	return((uint32_t)((local - layer * layer_span) / layer_page) + 1u);
}

static uint32_t SparkStageKvBindingPoolPageLimit(const SparkStageKvBinding *binding,uint32_t chunks)
{
	uint32_t need;
	if ( chunks >= binding->kv_pool.chunk_capacity )
		return(binding->physical_page_count);
	need = binding->kv_pool_chunk_need[chunks];
	return(need == 0u ? 0u : need - 1u);
}

static uint32_t SparkStageKvBindingPoolChunksFor(const SparkStageKvBinding *binding,uint32_t pages)
{
	uint32_t chunks = 0u;
	while ( chunks < binding->kv_pool.chunk_capacity && binding->kv_pool_chunk_need[chunks] <= pages )
		chunks++;
	return(chunks);
}

static SparkStatus SparkStageKvBindingPoolLayout(SparkStageKvBinding *binding,uint64_t lane_entries,SparkWeightdKvPoolRequest *request,uint64_t region_offsets[SPARK_STAGE_KV_MAX_REGIONS],uint64_t **offsets_out)
{
	SparkStageKvPoolSlot *slots;
	uint64_t granularity = 0u,raw,chunk,table_span,cursor,spans[SPARK_STAGE_KV_MAX_REGIONS],*offsets;
	uint32_t count,index,region,minimum = 0u;
	SparkStatus status;
	status = SparkWeightdKvPoolGranularity(&granularity);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	raw = lane_entries * sizeof(uint32_t);
	for (region=0u; region<binding->region_count; region++)
		raw += (uint64_t)binding->physical_page_count * binding->region_packed_page_bytes[region];
	chunk = granularity * ((raw + granularity * SPARK_STAGE_KV_POOL_TARGET_CHUNKS - 1u) / (granularity * SPARK_STAGE_KV_POOL_TARGET_CHUNKS));
	if ( chunk == 0u )
		chunk = granularity;
	table_span = (lane_entries * sizeof(uint32_t) + chunk - 1u) / chunk * chunk;
	cursor = table_span;
	for (region=0u; region<binding->region_count; region++)
	{
		region_offsets[region] = cursor;
		spans[region] = ((uint64_t)binding->physical_page_count * binding->region_packed_page_bytes[region] + chunk - 1u) / chunk * chunk;
		cursor += spans[region];
	}
	if ( cursor / chunk > SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX )
	{
		fprintf(stderr,"%s kv binding refused: the KV pool needs %llu chunks of %llu bytes, weightd carries %u\n",binding->module_tag,(unsigned long long)(cursor / chunk),
			(unsigned long long)chunk,SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	count = (uint32_t)(cursor / chunk);
	slots = (SparkStageKvPoolSlot *)calloc(count,sizeof(*slots));
	offsets = (uint64_t *)malloc((size_t)count * sizeof(uint64_t));
	binding->kv_pool_chunk_need = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	if ( slots == 0 || offsets == 0 || binding->kv_pool_chunk_need == 0 )
	{
		free(slots);
		free(offsets);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (index=0u; index<count; index++)
	{
		slots[index].offset = (uint64_t)index * chunk;
		for (region=0u; slots[index].offset >= table_span && region<binding->region_count; region++)
			if ( slots[index].offset >= region_offsets[region] && slots[index].offset < region_offsets[region] + spans[region] )
				slots[index].need = SparkStageKvPoolSlotNeed(binding,region,slots[index].offset - region_offsets[region],chunk);
	}
	qsort(slots,count,sizeof(*slots),SparkStageKvPoolSlotOrder);
	for (index=0u; index<count; index++)
	{
		offsets[index] = slots[index].offset;
		binding->kv_pool_chunk_need[index] = slots[index].need;
		if ( slots[index].need <= binding->pages_per_sequence )
			minimum = index + 1u;
	}
	free(slots);
	binding->kv_pool_minimum_chunks = minimum;
	request->device_bytes = (uint64_t)count * chunk;
	request->minimum_bytes = (uint64_t)minimum * chunk;
	request->chunk_bytes = chunk;
	*offsets_out = offsets;
	return(SPARK_STATUS_OK);
}

static uint64_t SparkStageKvGcd(uint64_t left,uint64_t right)
{
	while ( right != 0u )
	{
		uint64_t rest = left % right;
		left = right;
		right = rest;
	}
	return(left);
}

static uint64_t SparkStageKvSharedAlignment(const SparkStageKvBinding *binding,uint64_t granularity)
{
	uint64_t alignment = 1u,bytes,need;
	uint32_t region;
	for (region=0u; region<binding->region_count; region++)
	{
		bytes = binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? binding->region_packed_page_bytes[region] : binding->regions[region].layer_page_bytes;
		need = granularity / SparkStageKvGcd(granularity,bytes);
		alignment = alignment / SparkStageKvGcd(alignment,need) * need;
		if ( alignment > SPARK_KV_SHARED_INDEX_SLOTS_MAX )
			return(0u);
	}
	return(alignment);
}

static SparkStatus SparkStageKvBindingLayout(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration);

static SparkStatus SparkStageKvBindingAttachShared(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkWeightdKvSharedRequest request;
	SparkSha256Context context;
	uint64_t granularity = 0u,alignment;
	SparkStatus status;
	if ( binding->recurrent.lane_bytes != 0u )
	{
		fprintf(stderr,"%s kv shared prefix off: recurrent state lives in this engine's state store\n",binding->module_tag);
		return(SPARK_STATUS_OK);
	}
	status = SparkWeightdKvPoolGranularity(&granularity);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingLayout(binding,configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	alignment = SparkStageKvSharedAlignment(binding,granularity);
	if ( alignment == 0u || alignment > binding->physical_page_count )
	{
		fprintf(stderr,"%s kv shared prefix off: the %llu-byte mapping granularity aligns no window over %u pages\n",binding->module_tag,(unsigned long long)granularity,binding->physical_page_count);
		return(SPARK_STATUS_OK);
	}
	memset(&request,0,sizeof(request));
	SparkSha256Initialize(&context);
	SparkStageKvDigestText(&context,"sparkpipe.kv-shared.v1");
	SparkSha256Update(&context,binding->layout_sha256,SPARK_SHA256_DIGEST_BYTES);
	SparkSha256Finalize(&context,request.key);
	memcpy(request.layout_sha256,binding->layout_sha256,sizeof(request.layout_sha256));
	request.chunk_bytes = granularity;
	request.page_bytes = binding->page_bytes;
	request.alignment_pages = (uint32_t)alignment;
	request.label = binding->module_tag;
	status = SparkWeightdKvSharedAttach(&request,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS,&binding->kv_shared);
	if ( status == SPARK_STATUS_UNSUPPORTED )
	{
		fprintf(stderr,"%s kv shared prefix off: weightd has no shared window for %u-page groups of %llu bytes\n",binding->module_tag,(uint32_t)alignment,(unsigned long long)binding->page_bytes);
		return(SPARK_STATUS_OK);
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv binding refused: weightd shared prefix attach status=%s\n",binding->module_tag,SparkStatusToString(status));
		SPARK_RETURN(status);
	}
	status = SparkKvSharedIndexAttach(&binding->shared_index,binding->kv_shared.metadata,binding->kv_shared.metadata_bytes,binding->kv_shared.holder,binding->page_bytes,binding->layout_sha256);
	if ( status != SPARK_STATUS_OK || binding->kv_shared.slot_count % alignment != 0u )
	{
		fprintf(stderr,"%s kv binding refused: the weightd shared index does not match this layout status=%s\n",binding->module_tag,SparkStatusToString(status));
		SparkWeightdKvSharedUnmap(&binding->kv_shared);
		SPARK_FAIL(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	}
	binding->shared_page_count = binding->kv_shared.slot_count;
	binding->shared_alignment_pages = (uint32_t)alignment;
	return(SPARK_STATUS_OK);
}

static void SparkStageKvBindingWindowOffsets(const SparkStageKvBinding *binding,const uint64_t region_offsets[SPARK_STAGE_KV_MAX_REGIONS],uint64_t granularity,uint64_t *offsets)
{
	uint64_t start,span,step,count = 0u;
	uint32_t region,layer;
	for (region=0u; region<binding->region_count; region++)
	{
		uint32_t layers = binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? 1u : binding->regions[region].layer_count;
		uint64_t bytes = binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? binding->region_packed_page_bytes[region] : binding->regions[region].layer_page_bytes;
		for (layer=0u; layer<layers; layer++)
		{
			start = region_offsets[region] + (uint64_t)layer * ((uint64_t)binding->physical_page_count + binding->shared_page_count) * bytes + (uint64_t)binding->physical_page_count * bytes;
			span = (uint64_t)binding->shared_page_count * bytes;
			for (step=0u; step<span; step+=granularity)
				offsets[count++] = start + step;
		}
	}
}

static SparkStatus SparkStageKvBindingPoolLayoutWindowed(SparkStageKvBinding *binding,uint64_t lane_entries,SparkWeightdKvPoolRequest *request,uint64_t region_offsets[SPARK_STAGE_KV_MAX_REGIONS],uint64_t **offsets_out)
{
	SparkStageKvPoolSlot *slots;
	uint64_t granularity = 0u,chunk,group,pages,table_span,cursor,raw,bytes,piece,start,step,*offsets;
	uint32_t count = 0u,index,region,layer,layers,minimum = 0u,given_up;
	SparkStatus status = SparkWeightdKvPoolGranularity(&granularity);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	raw = lane_entries * sizeof(uint32_t) + (uint64_t)binding->physical_page_count * binding->page_bytes;
	chunk = granularity;
	while ( raw / (chunk * 2u) >= SPARK_STAGE_KV_POOL_TARGET_CHUNKS && (uint64_t)binding->physical_page_count / ((uint64_t)binding->shared_alignment_pages * (chunk * 2u / granularity)) * ((uint64_t)binding->shared_alignment_pages * (chunk * 2u / granularity)) >= binding->pages_per_sequence )
		chunk *= 2u;
	group = (uint64_t)binding->shared_alignment_pages * (chunk / granularity);
	pages = (uint64_t)binding->physical_page_count / group * group;
	if ( pages < binding->pages_per_sequence )
	{
		fprintf(stderr,"%s kv binding refused: aligning the shared window to %llu-page groups leaves %llu private pages, one lane needs %u\n",binding->module_tag,
			(unsigned long long)group,(unsigned long long)pages,binding->pages_per_sequence);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	given_up = binding->physical_page_count - (uint32_t)pages;
	binding->shared_pages_given_up = given_up;
	binding->physical_page_count = (uint32_t)pages;
	for (region=0u; region<binding->region_count; region++)
		if ( binding->regions[region].layout == SPARK_STAGE_KV_REGION_LAYER_MAJOR )
			binding->region_layer_stride_bytes[region] = ((uint64_t)binding->physical_page_count + binding->shared_page_count) * binding->regions[region].layer_page_bytes;
	table_span = (lane_entries * sizeof(uint32_t) + chunk - 1u) / chunk * chunk;
	cursor = table_span;
	for (region=0u; region<binding->region_count; region++)
	{
		region_offsets[region] = cursor;
		cursor += ((uint64_t)binding->physical_page_count + binding->shared_page_count) * binding->region_packed_page_bytes[region];
	}
	count = (uint32_t)(table_span / chunk);
	for (region=0u; region<binding->region_count; region++)
		count += (uint32_t)((uint64_t)binding->physical_page_count * binding->region_packed_page_bytes[region] / chunk);
	if ( count > SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX || (uint64_t)binding->shared_page_count * binding->page_bytes / granularity != binding->kv_shared.chunk_count )
	{
		fprintf(stderr,"%s kv binding refused: the windowed KV pool needs %u chunks of %llu bytes (weightd carries %u) and a %u-chunk window\n",binding->module_tag,count,
			(unsigned long long)chunk,SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX,binding->kv_shared.chunk_count);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	slots = (SparkStageKvPoolSlot *)calloc(count,sizeof(*slots));
	offsets = (uint64_t *)malloc((size_t)count * sizeof(uint64_t));
	binding->kv_pool_chunk_need = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	binding->shared_chunk_offsets = (uint64_t *)malloc((size_t)binding->kv_shared.chunk_count * sizeof(uint64_t));
	if ( slots == 0 || offsets == 0 || binding->kv_pool_chunk_need == 0 || binding->shared_chunk_offsets == 0 )
	{
		free(slots);
		free(offsets);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	index = 0u;
	for (step=0u; step<table_span; step+=chunk)
		slots[index++].offset = step;
	for (region=0u; region<binding->region_count; region++)
	{
		layers = binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? 1u : binding->regions[region].layer_count;
		bytes = binding->regions[region].layout == SPARK_STAGE_KV_REGION_PAGE_MAJOR ? binding->region_packed_page_bytes[region] : binding->regions[region].layer_page_bytes;
		piece = (uint64_t)binding->physical_page_count * bytes;
		for (layer=0u; layer<layers; layer++)
		{
			start = (uint64_t)layer * ((uint64_t)binding->physical_page_count + binding->shared_page_count) * bytes;
			for (step=0u; step<piece; step+=chunk)
			{
				slots[index].offset = region_offsets[region] + start + step;
				slots[index].need = SparkStageKvPoolSlotNeed(binding,region,start + step,chunk);
				index++;
			}
		}
	}
	qsort(slots,count,sizeof(*slots),SparkStageKvPoolSlotOrder);
	for (index=0u; index<count; index++)
	{
		offsets[index] = slots[index].offset;
		binding->kv_pool_chunk_need[index] = slots[index].need;
		if ( slots[index].need <= binding->pages_per_sequence )
			minimum = index + 1u;
	}
	free(slots);
	SparkStageKvBindingWindowOffsets(binding,region_offsets,granularity,binding->shared_chunk_offsets);
	binding->kv_pool_minimum_chunks = minimum;
	binding->kv_reservation_bytes = cursor;
	request->device_bytes = (uint64_t)count * chunk;
	request->minimum_bytes = (uint64_t)minimum * chunk;
	request->chunk_bytes = chunk;
	request->reservation_bytes = cursor;
	*offsets_out = offsets;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingPoolKey(const SparkStageKvConfiguration *configuration,uint8_t key[SPARK_WEIGHTD_KV_POOL_KEY_BYTES])
{
	char directory[PATH_MAX];
	SparkSha256Context context;
	if ( realpath(configuration->snapshot_directory,directory) == 0 )
	{
		fprintf(stderr,"%s kv binding refused: kv_snapshot_directory %s does not resolve\n",configuration->module_tag,configuration->snapshot_directory);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	SparkSha256Initialize(&context);
	SparkStageKvDigestText(&context,"sparkpipe.kv-pool.v1");
	SparkStageKvDigestText(&context,configuration->module_tag);
	SparkStageKvDigestText(&context,directory);
	SparkSha256Finalize(&context,key);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingAttachPool(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration,uint64_t lane_entries)
{
	SparkWeightdKvPoolRequest request;
	uint64_t region_offsets[SPARK_STAGE_KV_MAX_REGIONS],*offsets = 0;
	uint32_t region;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	status = SparkStageKvBindingPoolKey(configuration,request.key);
	if ( status == SPARK_STATUS_OK )
		status = binding->shared_page_count != 0u ? SparkStageKvBindingPoolLayoutWindowed(binding,lane_entries,&request,region_offsets,&offsets) :
			SparkStageKvBindingPoolLayout(binding,lane_entries,&request,region_offsets,&offsets);
	if ( status == SPARK_STATUS_OK && binding->shared_page_count != 0u )
	{
		status = SparkWeightdKvReserveAddress(binding->kv_reservation_bytes,&binding->kv_reservation);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"%s kv binding refused: cannot reserve %llu bytes of address space for the windowed KV layout\n",binding->module_tag,(unsigned long long)binding->kv_reservation_bytes);
		request.device_base = binding->kv_reservation;
	}
	if ( status != SPARK_STATUS_OK )
	{
		free(offsets);
		SPARK_RETURN(status);
	}
	request.metadata_bytes = sizeof(SparkStageKvPoolSeal) + (uint64_t)binding->physical_page_count * sizeof(SparkKvPageCacheResidentRecord);
	request.label = binding->module_tag;
	status = SparkWeightdKvPoolMap(&request,offsets,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS,&binding->kv_pool);
	free(offsets);
	if ( status == SPARK_STATUS_OK && binding->shared_page_count != 0u )
	{
		status = SparkWeightdKvSharedMapChunks(&binding->kv_shared,binding->kv_reservation,binding->shared_chunk_offsets,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"%s kv binding refused: mapping the shared prefix window failed status=%s mapped=%u of %u chunks\n",binding->module_tag,SparkStatusToString(status),
				binding->kv_shared.mapped_count,binding->kv_shared.chunk_count);
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (region=0u; region<binding->region_count; region++)
		binding->region_base[region] = (uint8_t *)binding->kv_pool.device_base + region_offsets[region];
	binding->page_table = (uint32_t *)binding->kv_pool.device_base;
	binding->pool_mapped_bytes = (uint64_t)binding->kv_pool.mapped_count * binding->kv_pool.chunk_bytes;
	fprintf(stderr,"%s kv pool weightd generation=%llu reattached=%u chunks=%u/%u minimum_chunks=%u chunk_bytes=%llu pages=%u/%u kv_committed=%llu kv_reserve=%llu private_kv_bytes=0\n",binding->module_tag,
		(unsigned long long)binding->kv_pool.pool_generation,binding->kv_pool.reattached,binding->kv_pool.mapped_count,binding->kv_pool.chunk_capacity,binding->kv_pool_minimum_chunks,
		(unsigned long long)binding->kv_pool.chunk_bytes,SparkStageKvBindingPoolPageLimit(binding,binding->kv_pool.mapped_count),binding->physical_page_count,
		(unsigned long long)binding->kv_pool.kv_committed_bytes,(unsigned long long)binding->kv_pool.kv_reserve_bytes);
	return(SPARK_STATUS_OK);
}

static const char *SparkStageKvBindingSealRefusal(const SparkStageKvBinding *binding,const SparkStageKvPoolSeal *seal)
{
	if ( binding->kv_pool.reattached == 0u )
		return("new_pool");
	if ( seal->magic != SPARK_STAGE_KV_POOL_SEAL_MAGIC || seal->version != SPARK_STAGE_KV_POOL_SEAL_VERSION || __atomic_load_n(&seal->sealed,__ATOMIC_ACQUIRE) != 1u )
		return("not_sealed");
	if ( seal->pool_generation != binding->kv_pool.pool_generation )
		return("pool_generation");
	if ( memcmp(seal->layout_sha256,binding->layout_sha256,SPARK_SHA256_DIGEST_BYTES) != 0 )
		return("layout");
	if ( seal->physical_page_count != binding->physical_page_count || seal->page_bytes != binding->page_bytes || seal->record_count > binding->physical_page_count )
		return("geometry");
	return(0);
}

static SparkStatus SparkStageKvBindingAdoptPool(SparkStageKvBinding *binding)
{
	SparkStageKvPoolSeal *seal = (SparkStageKvPoolSeal *)binding->kv_pool.metadata;
	const char *refused;
	uint32_t records = 0u,adopted = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if ( seal == 0 || binding->kv_pool.metadata_bytes < sizeof(*seal) + (uint64_t)binding->physical_page_count * sizeof(SparkKvPageCacheResidentRecord) )
	{
		fprintf(stderr,"%s kv binding refused: the weightd pool carries no seal region\n",binding->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	refused = SparkStageKvBindingSealRefusal(binding,seal);
	if ( refused == 0 )
	{
		records = seal->record_count;
		status = SparkKvPageCacheAdoptResident(&binding->page_cache,(const SparkKvPageCacheResidentRecord *)(seal + 1),records,&adopted);
	}
	__atomic_store_n(&seal->sealed,0u,__ATOMIC_RELEASE);
	seal->record_count = 0u;
	binding->kv_pool_seal_cleared = 1u;
	binding->kv_pool_adopted_pages = adopted;
	fprintf(stderr,"%s kv pool adopt reattached=%u records=%u adopted_pages=%u refused=%s status=%s\n",binding->module_tag,binding->kv_pool.reattached,records,adopted,
		refused != 0 ? refused : "none",SparkStatusToString(status));
	SPARK_RETURN(status);
}

static void SparkStageKvBindingSealPool(SparkStageKvBinding *binding)
{
	SparkStageKvPoolSeal *seal = (SparkStageKvPoolSeal *)binding->kv_pool.metadata;
	uint32_t count = 0u,entry,valid = 0u,private_entries = 0u,nonresident = 0u,shared = 0u;
	SparkStatus status;
	if ( seal == 0 || binding->mutex_initialized == 0u || binding->kv_pool_seal_cleared == 0u )
		return;
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr,"%s kv pool not sealed: the device did not drain\n",binding->module_tag);
		return;
	}
	status = SparkKvPageCacheExportResident(&binding->page_cache,(SparkKvPageCacheResidentRecord *)(seal + 1),binding->physical_page_count,&count);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv pool not sealed: export status=%s\n",binding->module_tag,SparkStatusToString(status));
		return;
	}
	seal->magic = SPARK_STAGE_KV_POOL_SEAL_MAGIC;
	seal->version = SPARK_STAGE_KV_POOL_SEAL_VERSION;
	seal->pool_generation = binding->kv_pool.pool_generation;
	memcpy(seal->layout_sha256,binding->layout_sha256,SPARK_SHA256_DIGEST_BYTES);
	seal->physical_page_count = binding->physical_page_count;
	seal->page_bytes = binding->page_bytes;
	seal->record_count = count;
	__atomic_store_n(&seal->sealed,1u,__ATOMIC_RELEASE);
	binding->kv_pool_sealed_pages = count;
	for (entry=0u; entry<binding->page_cache.entry_capacity; entry++)
	{
		const SparkKvPageCacheEntry *candidate = &binding->page_cache.entries[entry];
		if ( (candidate->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) == 0u )
			continue;
		valid++;
		if ( (candidate->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE) != 0u )
			private_entries++;
		else if ( candidate->logical_page_index < binding->arena.logical_block_count &&
			(binding->arena.blocks[candidate->logical_page_index].flags & SPARK_KV_CACHE_BLOCK_FLAG_SHARED) != 0u )
			shared++;
		else if ( candidate->logical_page_index >= binding->arena.logical_block_count ||
			(binding->arena.blocks[candidate->logical_page_index].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
			binding->arena.blocks[candidate->logical_page_index].resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT )
			nonresident++;
	}
	fprintf(stderr,"%s kv pool sealed generation=%llu resident_pages=%u entries=%u window=%u private=%u nonresident=%u orphaned=%u\n",binding->module_tag,(unsigned long long)binding->kv_pool.pool_generation,count,
		valid,shared,private_entries,nonresident,valid - shared - private_entries - nonresident - count);
}

static SparkStatus SparkStageKvBindingAttachWriteBudget(SparkStageKvBinding *binding)
{
	if ( SparkKvWriteBudgetInitialize(&binding->write_budget,binding->kv_pool.write_budget_bytes_per_day,SparkStageKvNowNs()) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv binding refused: the weightd pool grants no NVMe write budget\n",binding->module_tag);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	binding->page_store.write_budget = &binding->write_budget;
	if ( binding->state_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		binding->state_store.write_budget = &binding->write_budget;
	binding->page_cache.write_budget = &binding->write_budget;
	fprintf(stderr,"%s kv write budget bytes_per_day=%llu: spill and snapshot writes past it are discarded or skipped and recomputed\n",binding->module_tag,
		(unsigned long long)binding->write_budget.bytes_per_day);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingRefuse(const SparkStageKvConfiguration *configuration)
{
	const char *missing;
	if ( configuration->backing_directory == 0 || configuration->backing_directory[0] == '\0' )
	{
		fprintf(stderr,"%s kv binding refused: the deployment names no kv_backing_directory\n",configuration->module_tag);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	missing = SparkStageKvBindingMissingIdentity(configuration);
	if ( missing != 0 )
	{
		fprintf(stderr,"%s kv binding refused: layout identity input %s is missing\n",configuration->module_tag,missing);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( configuration->snapshot_directory == 0 || configuration->snapshot_directory[0] == '\0' )
	{
		fprintf(stderr,"%s kv binding refused: the deployment names no kv_snapshot_directory\n",configuration->module_tag);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( configuration->snapshot_maximum_bytes == 0u )
	{
		fprintf(stderr,"%s kv binding refused: kv_snapshot_maximum_bytes is zero\n",configuration->module_tag);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingStartPool(SparkStageKvBinding *binding);

static SparkStatus SparkStageKvBindingAttachSharedCache(SparkStageKvBinding *binding)
{
	SparkStatus status;
	binding->shared_generations = (uint64_t *)calloc(binding->logical_page_count,sizeof(uint64_t));
	binding->shared_chain_slots = (uint32_t *)calloc(binding->pages_per_sequence,sizeof(uint32_t));
	binding->shared_chain_generations = (uint64_t *)calloc(binding->pages_per_sequence,sizeof(uint64_t));
	if ( binding->shared_generations == 0 || binding->shared_chain_slots == 0 || binding->shared_chain_generations == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkKvCacheArenaSetSharedWindow(&binding->arena,binding->physical_page_count,binding->shared_page_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheAttachShared(&binding->page_cache,&binding->shared_index,binding->shared_generations,binding->shared_chain_slots,binding->shared_chain_generations,binding->pages_per_sequence);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	fprintf(stderr,"%s kv shared prefix window pages=%u base_page=%u holder=%u created=%u alignment_pages=%u private_pages_given_up=%u\n",binding->module_tag,binding->shared_page_count,
		binding->physical_page_count,binding->kv_shared.holder,binding->kv_shared.created,binding->shared_alignment_pages,binding->shared_pages_given_up);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkStageKvBindingInitialize(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkKvModelTable table;
	uint64_t lane_entries;
	SparkStatus status;
	cudaError_t error;
	if ( binding == 0 || configuration == 0 || configuration->module_tag == 0 || configuration->block_token_count == 0u || configuration->region_count == 0u ||
		configuration->region_count > SPARK_STAGE_KV_MAX_REGIONS || configuration->resident_sequence_capacity == 0u || configuration->max_sequence_positions == 0u || configuration->pipeline_slot_count == 0u ||
		configuration->physical_page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (configuration->recurrent.lane_bytes == 0u) != (configuration->recurrent.copy == 0) || (configuration->recurrent.lane_bytes == 0u && configuration->recurrent.context != 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvBindingRefuse(configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(binding,0,sizeof(*binding));
	binding->recurrent = configuration->recurrent;
	binding->module_tag = configuration->module_tag;
	binding->block_token_count = configuration->block_token_count;
	binding->region_count = configuration->region_count;
	binding->logical_page_count = configuration->logical_page_count;
	binding->physical_page_count = configuration->physical_page_count;
	binding->resident_sequence_capacity = configuration->resident_sequence_capacity;
	binding->max_sequence_positions = configuration->max_sequence_positions;
	binding->pipeline_slot_count = configuration->pipeline_slot_count;
	binding->context_shard = configuration->context_shard;
	status = SparkStageKvBindingGeometry(binding,configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAttachShared(binding,configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	lane_entries = (uint64_t)binding->resident_sequence_capacity * binding->pages_per_sequence;
	status = SparkStageKvBindingAttachPool(binding,configuration,lane_entries);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemset(binding->page_table,0xff,(size_t)lane_entries * sizeof(uint32_t));
		status = SparkStageModuleCudaStatus(binding->module_tag,error,"kv_page_table");
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAllocateHost(binding,lane_entries);
	if ( status == SPARK_STATUS_OK && binding->recurrent.lane_bytes != 0u )
		status = SparkStageKvBindingAllocateStates(binding);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaStreamCreateWithFlags((cudaStream_t *)&binding->copy_stream,cudaStreamNonBlocking);
		status = SparkStageModuleCudaStatus(binding->module_tag,error,"kv_copy_stream");
		if ( status != SPARK_STATUS_OK )
			binding->copy_stream = 0;
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	binding->transactions.cache = &binding->page_cache;
	binding->transactions.lanes = binding->lanes;
	binding->transactions.logical_pages = binding->logical_pages;
	binding->transactions.physical_pages = binding->physical_pages;
	binding->transactions.page_capacity = binding->pages_per_sequence;
	SparkStageKvBindingFillTable(binding,configuration,&table);
	status = SparkKvBackendInitialize(&table,&binding->arena,&binding->page_cache,&binding->page_store);
	if ( status == SPARK_STATUS_OK )
		binding->arena.resident_block_capacity = SparkStageKvBindingPoolPageLimit(binding,binding->kv_pool.mapped_count);
	if ( status == SPARK_STATUS_OK && binding->shared_page_count != 0u )
		status = SparkStageKvBindingAttachSharedCache(binding);
	if ( status == SPARK_STATUS_OK && binding->recurrent.lane_bytes != 0u )
		status = SparkStageKvBindingAttachStates(binding,configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAttachWriteBudget(binding);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pthread_mutex_init(&binding->mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->mutex_initialized = 1u;
	status = SparkStageKvBindingOpenSnapshot(binding,configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingAdoptPool(binding);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingStartAsync(binding);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingStartPool(binding);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	fprintf(stderr,"%s kv binding logical_pages=%u physical_pages=%u pages_per_sequence=%u page_bytes=%llu regions=%u context_shard=%u/%u layout=%s\n",
		binding->module_tag,binding->logical_page_count,binding->physical_page_count,binding->pages_per_sequence,(unsigned long long)binding->page_bytes,binding->region_count,
		binding->context_shard.degree > 1u ? binding->context_shard.rank : 0u,binding->context_shard.degree > 1u ? binding->context_shard.degree : 1u,binding->layout_hex);
	return(SPARK_STATUS_OK);
}

static void SparkStageKvBindingLogCounters(const SparkStageKvBinding *binding)
{
	const SparkStageKvBindingCounters *counters = &binding->counters;
	uint32_t site;
	for (site=0u; site<SPARK_STAGE_KV_LOCK_SITE_COUNT; site++)
		fprintf(stderr,"%s kv binding lock site=%s count=%llu mean_us=%.1f max_us=%.1f\n",binding->module_tag,SparkStageKvLockSiteNames[site],
			(unsigned long long)counters->lock_sites[site].count,
			counters->lock_sites[site].count != 0u ? (double)counters->lock_sites[site].total_ns / (double)counters->lock_sites[site].count / 1000.0 : 0.0,
			(double)counters->lock_sites[site].max_ns / 1000.0);
	fprintf(stderr,"%s kv binding completions=%llu entry_max_us=%.1f queue_mean_us=%.1f queue_max_us=%.1f copy_on_write=%llu\n",binding->module_tag,
		(unsigned long long)counters->completion_count,(double)counters->entry_max_ns / 1000.0,
		counters->completion_count != 0u ? (double)counters->completion_queue_total_ns / (double)counters->completion_count / 1000.0 : 0.0,
		(double)counters->completion_queue_max_ns / 1000.0,(unsigned long long)binding->copier.copy_count);
	fprintf(stderr,"%s kv binding arena evicted=%llu park_failures=%llu park_backing_full=%llu park_stalls=%llu store_writes=%llu store_reads=%llu backing_pages=%u backing_reclaims=%llu backing_full=%llu backing_full_queued=%llu degraded=%u degraded_entries=%llu degraded_discards=%llu park_stall_queued=%llu\n",binding->module_tag,
		(unsigned long long)binding->arena.resident_evicted_block_count,(unsigned long long)binding->arena.park_failure_count,
		(unsigned long long)binding->arena.park_backing_full_count,(unsigned long long)binding->arena.park_stall_count,
		(unsigned long long)binding->page_store.write_count,(unsigned long long)binding->page_store.read_count,
		binding->page_store.backing_page_count,(unsigned long long)binding->page_cache.backing_reclaim_count,
		(unsigned long long)binding->page_cache.backing_full_count,(unsigned long long)binding->page_cache.backing_full_queued_count,
		binding->arena.park_degraded,(unsigned long long)binding->arena.park_degraded_count,(unsigned long long)binding->page_cache.degraded_discard_count,
		(unsigned long long)binding->page_cache.park_stall_queued_count);
	if ( binding->shared_page_count != 0u )
		fprintf(stderr,"%s kv binding shared window pages=%u allocated=%llu published=%llu imports=%llu imported_pages=%llu import_misses=%llu shared_blocks=%u\n",binding->module_tag,
			binding->shared_page_count,(unsigned long long)binding->page_cache.shared_allocated_count,(unsigned long long)binding->page_cache.shared_published_count,
			(unsigned long long)binding->page_cache.shared_import_count,(unsigned long long)binding->page_cache.shared_imported_page_count,
			(unsigned long long)binding->page_cache.shared_import_miss_count,binding->arena.shared_block_count);
	fprintf(stderr,"%s kv binding restore worker jobs=%llu hints=%llu hinted_jobs=%llu pending_answers=%llu imported_pages=%llu\n",binding->module_tag,
		(unsigned long long)binding->restore_jobs,(unsigned long long)binding->restore_hints,(unsigned long long)binding->restore_hinted_jobs,
		(unsigned long long)binding->restore_pending_answers,(unsigned long long)binding->restore_imported_pages);
}

static void SparkStageKvBindingSaveAtDestroy(SparkStageKvBinding *binding)
{
	uint32_t saved = 0u,unsaved = 0u,ineligible = 0u;
	SparkStatus status;
	if ( binding->mutex_initialized == 0u || binding->page_cache.snapshot == 0 || binding->save_started == 0u )
		return;
	status = SparkStageKvBindingSaveAll(binding,SparkStageKvNowNs() + SPARK_STAGE_KV_DESTROY_SAVE_TIMEOUT_NS,&saved,&unsaved,&ineligible);
	fprintf(stderr,"%s kv snapshot store close saved_entries=%u unsaved_entries=%u ineligible_entries=%u used_bytes=%llu files=%llu save_failures=%llu save_deferred=%llu evicted_unsaved=%llu demotions_queued=%llu park_saves_queued=%llu status=%s\n",
		binding->module_tag,saved,unsaved,ineligible,(unsigned long long)binding->snapshot_store.used_bytes,(unsigned long long)binding->snapshot_store.file_count,
		(unsigned long long)binding->snapshot.save_failure_count,(unsigned long long)binding->snapshot.save_deferred_count,
		(unsigned long long)binding->snapshot.evicted_unsaved_count,(unsigned long long)binding->snapshot.demote_queued_count,(unsigned long long)binding->snapshot.park_save_queued_count,
		SparkStatusToString(status));
}

void SparkStageKvBindingDestroy(SparkStageKvBinding *binding)
{
	if ( binding == 0 )
		return;
	SparkStageKvBindingSaveAtDestroy(binding);
	SparkStageKvBindingStop(binding);
	if ( binding->page_cache.snapshot != 0 )
	{
		uint64_t before = binding->page_cache.snapshot->save_cancelled_count;
		SparkKvPageCacheSaveCancelAll(&binding->page_cache);
		if ( binding->page_cache.snapshot->save_cancelled_count != before )
			fprintf(stderr,"KV-SNAPSHOT pending saves cancelled count=%llu\n",(unsigned long long)(binding->page_cache.snapshot->save_cancelled_count - before));
	}
	if ( binding->state_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&binding->state_store);
	if ( binding->page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&binding->page_store);
	if ( binding->copier_initialized != 0u )
	{
		(void)SparkKvDeviceCopierDrain(&binding->copier);
		SparkKvDeviceCopierDestroy(&binding->copier);
	}
	if ( binding->copy_stream != 0 )
		(void)cudaStreamDestroy((cudaStream_t)binding->copy_stream);
	SparkStageKvBindingSealPool(binding);
	if ( binding->kv_pool.chunk_capacity != 0u )
		fprintf(stderr,"%s kv binding pool chunks=%u/%u pages=%u/%u grows=%llu shrinks=%llu vacated_pages=%llu\n",binding->module_tag,binding->kv_pool.mapped_count,
			binding->kv_pool.chunk_capacity,binding->arena.resident_block_capacity,binding->physical_page_count,(unsigned long long)binding->pool_grow_count,
			(unsigned long long)binding->pool_shrink_count,(unsigned long long)binding->pool_vacated_pages);
	SparkWeightdKvPoolUnmap(&binding->kv_pool);
	SparkWeightdKvSharedUnmap(&binding->kv_shared);
	SparkWeightdKvFreeAddress(binding->kv_reservation,binding->kv_reservation_bytes);
	free(binding->shared_generations);
	free(binding->shared_chain_slots);
	free(binding->shared_chain_generations);
	free(binding->shared_chunk_offsets);
	free(binding->kv_pool_chunk_need);
	if ( binding->snapshot_store.runtime != 0 )
		SparkKvSnapshotStoreClose(&binding->snapshot_store);
	free(binding->snapshot_links);
	free(binding->snapshot_pending);
	free(binding->snapshot_state);
	if ( binding->lane_state != 0 )
		(void)cudaFreeHost(binding->lane_state);
	free(binding->state_staging);
	free(binding->lane_state_flags);
	free(binding->save_order);
	if ( binding->snapshot_page != 0 )
		(void)cudaFreeHost(binding->snapshot_page);
	free(binding->restore_job.links);
	free(binding->restore_job.state);
	if ( binding->restore_job.page != 0 )
		(void)cudaFreeHost(binding->restore_job.page);
	if ( binding->module_tag != 0 && binding->mutex_initialized != 0u )
		SparkStageKvBindingLogCounters(binding);
	SparkStageKvBindingFreeCompletions(binding);
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
	free(binding->lane_rewind_floors);
	free(binding->lane_rewind_ceilings);
	free(binding->lane_pending_floors);
	if ( binding->mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&binding->mutex);
	if ( binding->sync_initialized != 0u )
	{
		(void)pthread_mutex_destroy(&binding->completion_mutex);
		(void)pthread_cond_destroy(&binding->completion_ready);
		(void)pthread_cond_destroy(&binding->completion_idle);
		(void)pthread_cond_destroy(&binding->save_ready);
		(void)pthread_cond_destroy(&binding->save_idle);
		if ( binding->restore_ready_initialized != 0u )
			(void)pthread_cond_destroy(&binding->restore_ready);
		if ( binding->pool_wake_initialized != 0u )
			(void)pthread_cond_destroy(&binding->pool_wake);
	}
	memset(binding,0,sizeof(*binding));
}

SparkStatus SparkStageKvBindingAdmit(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	SparkStatus status;
	uint32_t lane;
	uint64_t held;
	if ( binding == 0 || request == 0 || decision == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_HINT )
		return(SparkStageKvBindingHint(binding,request,decision));
	status = SparkStageKvBindingLock(binding,&held);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = binding->control_generation != 0u && binding->control_generation != request->control_generation ?
		SPARK_STATUS_VALIDATION_FAILED : SparkStageKvBindingRestoreGate(binding,request);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvLaneTransactionsAdmit(&binding->transactions,request);
	if ( status == SPARK_STATUS_OK && binding->recurrent.lane_bytes != 0u )
		status = SparkStageKvBindingRecurrentAdmit(binding,request,&held);
	if ( status == SPARK_STATUS_OK )
		binding->control_generation = request->control_generation;
	if ( status == SPARK_STATUS_OK && (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
			atomic_store_explicit(&binding->lane_bound[request->cache_lanes[lane].resident_sequence_slot],0u,memory_order_release);
	SparkStageKvBindingWakeSaver(binding);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_ADMIT,held);
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
	uint64_t held;
	uint32_t lane;
	if ( binding == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvBindingLock(binding,&held);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = generation > binding->reset_generation ? SparkKvLaneTransactionsReset(&binding->transactions) : SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
	{
		memset(binding->page_table_shadow,0xff,(size_t)binding->resident_sequence_capacity * binding->pages_per_sequence * sizeof(uint32_t));
		for (lane=0u; lane<binding->resident_sequence_capacity; lane++)
		{
			atomic_store_explicit(&binding->lane_bound[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_sequence_ids[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_next_positions[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_rewind_floors[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_rewind_ceilings[lane],0u,memory_order_release);
			atomic_store_explicit(&binding->lane_pending_floors[lane],UINT64_MAX,memory_order_release);
		}
		binding->reset_generation = generation;
		binding->control_generation = 0u;
		if ( binding->recurrent.lane_bytes != 0u )
			memset(binding->lane_state_flags,0,binding->resident_sequence_capacity);
	}
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_RESET,held);
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
				(void)SparkStageModuleCudaStatus(binding->module_tag,drain,"kv_reset_stream_drain");
			else
			{
				drain = cudaStreamSynchronize((cudaStream_t)binding->copy_stream);
				if ( drain != cudaSuccess )
					(void)SparkStageModuleCudaStatus(binding->module_tag,drain,"kv_reset_copy_stream_drain");
			}
			if ( drain != cudaSuccess )
			{
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

static SparkStatus SparkStageKvBindingLoadContinuity(const SparkStageKvBinding *binding,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions,uint64_t *floors)
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
		floors[lane] = atomic_load_explicit(&binding->lane_rewind_ceilings[slot],memory_order_acquire) == next_positions[lane] &&
			atomic_load_explicit(&binding->lane_rewind_floors[slot],memory_order_acquire) <= next_positions[lane] ?
			atomic_load_explicit(&binding->lane_rewind_floors[slot],memory_order_acquire) : next_positions[lane];
		owner = &binding->lanes[slot];
		if ( SparkKvLaneTransactionPrefixRestorePending(owner) == 0u )
			continue;
		if ( owner->phase != SPARK_KV_LANE_TRANSACTION_COMMITTED || owner->lane.sequence_id != row_sequence_ids[lane] || owner->lane.sequence_position != row_positions[lane] )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		bound[lane] = 1u;
		sequence_ids[lane] = owner->lane.sequence_id;
		next_positions[lane] = owner->lane.sequence_position;
		floors[lane] = next_positions[lane];
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvBindingRowContinuity(const SparkStageKvBinding *binding,const atomic_uint *lane_states,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions,const uint64_t *floors)
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
		else if ( row_positions[row] == 0u || bound[lane] == 0u || sequence_ids[lane] != row_sequence_ids[row] ||
			(touched[lane] != 0u && next_positions[lane] != row_positions[row]) ||
			(touched[lane] == 0u && (row_positions[row] > next_positions[lane] || row_positions[row] < floors[lane])) )
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

uint64_t SparkStageKvBindingLaneSequence(const SparkStageKvBinding *binding,uint32_t slot)
{
	if ( binding == 0 || binding->lane_sequence_ids == 0 || slot >= binding->resident_sequence_capacity )
		return(0u);
	return(atomic_load_explicit(&binding->lane_sequence_ids[slot],memory_order_acquire));
}

SparkStatus SparkStageKvBindingContinuity(SparkStageKvBinding *binding,const atomic_uint *lane_states,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	SparkStatus status;
	uint64_t held,*floors;
	if ( binding == 0 || lane_states == 0 || row_resident_slots == 0 || row_sequence_ids == 0 || row_positions == 0 || bound == 0 || sequence_ids == 0 || next_positions == 0 || row_count < active_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	floors = (uint64_t *)calloc(active_count != 0u ? active_count : 1u,sizeof(*floors));
	if ( floors == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkStageKvBindingLock(binding,&held);
	if ( status != SPARK_STATUS_OK )
	{
		free(floors);
		SPARK_RETURN(status);
	}
	status = SparkStageKvBindingLoadContinuity(binding,active_count,row_resident_slots,row_sequence_ids,row_positions,bound,sequence_ids,next_positions,floors);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRowContinuity(binding,lane_states,row_count,active_count,row_resident_slots,row_sequence_ids,row_positions,bound,sequence_ids,next_positions,floors);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_CONTINUITY,held);
	free(floors);
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
			frame->cache_lanes[lane].sequence_position != row_positions[lane] || frame->cache_lanes[lane].context_token_count != next_positions[lane] ||
			((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_VERIFY) != 0u) != ((frame->cache_lanes[lane].flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_VERIFY) != 0u) )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	for (lane=0u; lane<active_count; lane++)
		atomic_store_explicit(&binding->lane_pending_floors[row_resident_slots[lane]],
			(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_VERIFY) != 0u ? row_positions[lane] + 1u : next_positions[lane],memory_order_release);
	{
		uint64_t held;
		status = SparkStageKvBindingLock(binding,&held);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		status = SparkKvLaneTransactionsClaim(&binding->transactions,frame);
		SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_CLAIM,held);
	}
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

typedef struct SparkStageKvStateJob
{
	uint32_t slot;
	uint32_t page;
	uint64_t generation;
	SparkStatus status;
	uint32_t reserved0;
} SparkStageKvStateJob;

static uint8_t *SparkStageKvLaneState(const SparkStageKvBinding *binding,uint32_t slot)
{
	return(binding->lane_state + (uint64_t)slot * binding->recurrent.lane_bytes);
}

static uint32_t SparkStageKvTerminalMutablePage(const SparkKvPageCacheSequence *sequence)
{
	return(sequence->mutable_page_count <= 1u ? sequence->mutable_logical_page_index : sequence->mutable_following_pages[sequence->mutable_page_count - 2u]);
}

static void SparkStageKvClearSlotStates(SparkStageKvBinding *binding,const uint32_t *slots,uint32_t count,uint8_t mask)
{
	uint32_t index;
	if ( binding->recurrent.lane_bytes == 0u )
		return;
	for (index=0u; index<count; index++)
		if ( slots[index] < binding->resident_sequence_capacity )
			binding->lane_state_flags[slots[index]] &= (uint8_t)~mask;
}

static void SparkStageKvClearLaneStates(SparkStageKvBinding *binding,const SparkModelDriverCacheLane *lanes,uint32_t count,uint8_t mask)
{
	uint32_t index,slot;
	if ( binding->recurrent.lane_bytes == 0u )
		return;
	for (index=0u; index<count; index++)
	{
		slot = lanes[index].resident_sequence_slot;
		if ( slot < binding->resident_sequence_capacity )
			binding->lane_state_flags[slot] &= (uint8_t)~mask;
	}
}

static void SparkStageKvAbortPrepared(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request)
{
	SparkModelDriverAdmissionRequest abort = *request;
	abort.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	(void)SparkKvLaneTransactionsAdmit(&binding->transactions,&abort);
	SparkStageKvClearLaneStates(binding,request->cache_lanes,request->cache_lane_count,0xffu);
}

static const char *SparkStageKvRestoreInconsistent(const SparkStageKvBinding *binding,uint32_t slot,uint32_t *page_out)
{
	const SparkKvLaneTransaction *owner = &binding->lanes[slot];
	const SparkKvPageCache *cache = &binding->page_cache;
	uint32_t entry = cache->sequences[slot].terminal_entry_index,page,mutable_page;
	if ( entry >= cache->entry_capacity || cache->entries[entry].token_count != owner->lane.sequence_position || cache->entries[entry].reference_count == 0u ||
		cache->sequences[slot].sequence_id != owner->lane.sequence_id )
		return("entry");
	page = cache->entries[entry].logical_page_index;
	if ( page >= binding->logical_page_count || binding->blocks[page].reference_count == 0u )
		return("page");
	if ( binding->blocks[page].residency_reference_count == 0u )
	{
		mutable_page = cache->sequences[slot].mutable_logical_page_index;
		if ( owner->lane.sequence_position % binding->block_token_count == 0u || mutable_page >= binding->logical_page_count || owner->page_count == 0u ||
			binding->logical_pages[(uint64_t)slot * binding->pages_per_sequence + owner->page_count - 1u] != mutable_page ||
			binding->blocks[mutable_page].residency_reference_count == 0u )
			return("copy-on-write");
	}
	*page_out = page;
	return(0);
}

static SparkStatus SparkStageKvStateRead(SparkStageKvBinding *binding,const SparkStageKvStateJob *job)
{
	SparkStatus status,wait;
	status = SparkKvPageStoreReadback(&binding->state_store,job->page,job->generation,(uintptr_t)SparkStageKvLaneState(binding,job->slot),binding->recurrent.lane_bytes);
	while ( status == SPARK_STATUS_BUSY )
	{
		wait = SparkKvPageStoreWaitForTransfers(&binding->state_store);
		if ( wait != SPARK_STATUS_OK )
			SPARK_RETURN(wait);
		status = SparkKvPageStoreReadback(&binding->state_store,job->page,job->generation,(uintptr_t)SparkStageKvLaneState(binding,job->slot),binding->recurrent.lane_bytes);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkStageKvStateWrite(SparkStageKvBinding *binding,const SparkStageKvStateJob *job)
{
	SparkStatus status,wait;
	status = SparkKvPageStoreWriteback(&binding->state_store,job->page,job->slot,job->generation,(uintptr_t)SparkStageKvLaneState(binding,job->slot),binding->recurrent.lane_bytes,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		wait = SparkKvPageStoreWaitForTransfers(&binding->state_store);
		if ( wait != SPARK_STATUS_OK )
			SPARK_RETURN(wait);
		status = SparkKvPageStoreWriteback(&binding->state_store,job->page,job->slot,job->generation,(uintptr_t)SparkStageKvLaneState(binding,job->slot),binding->recurrent.lane_bytes,0u,0u);
	}
	SPARK_RETURN(status);
}

static uint32_t SparkStageKvCollectRestores(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkStageKvStateJob *jobs,const char **reason_out,uint32_t *slot_out)
{
	uint32_t index,slot,page = 0u,count = 0u;
	const char *reason;
	*reason_out = 0;
	for (index=0u; index<request->cache_lane_count; index++)
	{
		slot = request->cache_lanes[index].resident_sequence_slot;
		if ( SparkKvLaneTransactionPrefixRestorePending(&binding->lanes[slot]) == 0u || (binding->lane_state_flags[slot] & SPARK_STAGE_KV_LANE_STATE_RESTORE_READY) != 0u )
			continue;
		reason = SparkStageKvRestoreInconsistent(binding,slot,&page);
		if ( reason != 0 )
		{
			*reason_out = reason;
			*slot_out = slot;
			return(count);
		}
		jobs[count] = (SparkStageKvStateJob){.slot=slot,.page=page,.generation=binding->blocks[page].generation};
		binding->lane_state_flags[slot] |= SPARK_STAGE_KV_LANE_STATE_LOADING;
		count++;
	}
	return(count);
}

static SparkStatus SparkStageKvApplyRestores(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,const SparkStageKvStateJob *jobs,uint32_t count)
{
	SparkStatus failure = SPARK_STATUS_OK,read;
	uint32_t index,misses = 0u;
	for (index=0u; index<count; index++)
	{
		const SparkStageKvStateJob *job = &jobs[index];
		read = binding->lanes[job->slot].phase != SPARK_KV_LANE_TRANSACTION_PREPARED || binding->blocks[job->page].generation != job->generation ? SPARK_STATUS_VALIDATION_FAILED : job->status;
		binding->lane_state_flags[job->slot] &= (uint8_t)~SPARK_STAGE_KV_LANE_STATE_LOADING;
		if ( read == SPARK_STATUS_OK )
		{
			binding->lane_state_flags[job->slot] |= SPARK_STAGE_KV_LANE_STATE_RESTORE_READY;
			atomic_fetch_add(&binding->recurrent_restores,1u);
			atomic_fetch_add(&binding->recurrent_restore_bytes,binding->recurrent.lane_bytes);
		}
		else if ( read == SPARK_STATUS_NOT_FOUND || read == SPARK_STATUS_IO_ERROR )
		{
			misses++;
			fprintf(stderr,"%s kv binding state restore miss slot=%u sequence=%llu page=%u status=%u: the prefix is recomputed\n",binding->module_tag,job->slot,
				(unsigned long long)binding->lanes[job->slot].lane.sequence_id,job->page,(unsigned)read);
		}
		else if ( failure == SPARK_STATUS_OK )
			failure = read;
	}
	if ( misses == 0u && failure == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	SparkStageKvAbortPrepared(binding,request);
	if ( failure != SPARK_STATUS_OK )
		SPARK_RETURN(failure);
	SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
}

static SparkStatus SparkStageKvBindingRecurrentAdmit(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,uint64_t *held)
{
	SparkStageKvStateJob *jobs;
	const char *reason = 0;
	uint32_t count,index,slot = 0u;
	uint64_t reset,start;
	SparkStatus status;
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT || (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		SparkStageKvClearLaneStates(binding,request->cache_lanes,request->cache_lane_count,0xffu);
		return(SPARK_STATUS_OK);
	}
	if ( request->admission_flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE )
		return(SPARK_STATUS_OK);
	jobs = (SparkStageKvStateJob *)calloc(request->cache_lane_count,sizeof(*jobs));
	if ( jobs == 0 )
	{
		SparkStageKvAbortPrepared(binding,request);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	count = SparkStageKvCollectRestores(binding,request,jobs,&reason,&slot);
	if ( reason != 0 )
	{
		SparkStageKvAbortPrepared(binding,request);
		free(jobs);
		fprintf(stderr,"%s kv binding refused: restore slot=%u sequence=%llu has inconsistent prefix metadata (%s)\n",binding->module_tag,slot,
			(unsigned long long)binding->page_cache.sequences[slot].sequence_id,reason);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( count == 0u )
	{
		free(jobs);
		return(SPARK_STATUS_OK);
	}
	reset = binding->reset_generation;
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_ADMIT,*held);
	start = SparkStageKvNowNs();
	for (index=0u; index<count; index++)
		jobs[index].status = SparkStageKvStateRead(binding,&jobs[index]);
	atomic_fetch_add(&binding->recurrent_restore_ns,SparkStageKvNowNs() - start);
	(void)SparkStageKvBindingLock(binding,held);
	if ( binding->reset_generation != reset )
	{
		for (index=0u; index<count; index++)
			binding->lane_state_flags[jobs[index].slot] = 0u;
		free(jobs);
		fprintf(stderr,"%s kv binding refused: state restore for request %llu raced a reset\n",binding->module_tag,(unsigned long long)request->request_id);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	status = SparkStageKvApplyRestores(binding,request,jobs,count);
	free(jobs);
	SPARK_RETURN(status);
}

static SparkStatus SparkStageKvCollectWrites(SparkStageKvBinding *binding,const uint32_t *slots,uint32_t lane_count,SparkStageKvStateJob *jobs,uint32_t *count_out)
{
	const SparkKvLaneTransaction *owner;
	uint32_t index,slot,page,count = 0u;
	*count_out = 0u;
	for (index=0u; index<lane_count; index++)
	{
		slot = slots[index];
		if ( slot >= binding->resident_sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		owner = &binding->lanes[slot];
		if ( (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) != 0u && (binding->lane_state_flags[slot] & SPARK_STAGE_KV_LANE_STATE_CAPTURED) == 0u )
		{
			fprintf(stderr,"%s kv binding refused: lane %u publishes without captured recurrent state\n",binding->module_tag,slot);
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		}
		if ( (binding->lane_state_flags[slot] & SPARK_STAGE_KV_LANE_STATE_RESTORE_READY) != 0u )
		{
			fprintf(stderr,"%s kv binding refused: lane %u finished without applying its restored recurrent state\n",binding->module_tag,slot);
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		}
		if ( (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) == 0u )
			continue;
		page = SparkStageKvTerminalMutablePage(&binding->page_cache.sequences[slot]);
		if ( page >= binding->logical_page_count )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		jobs[count++] = (SparkStageKvStateJob){.slot=slot,.page=page,.generation=binding->blocks[page].generation};
	}
	*count_out = count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkStageKvWriteJob(SparkStageKvBinding *binding,const SparkStageKvStateJob *job,uint32_t site,uint64_t *held)
{
	SparkStatus status,evicted;
	uint32_t evictions = 0u;
	status = SparkStageKvStateWrite(binding,job);
	while ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
	{
		(void)SparkStageKvBindingLock(binding,held);
		evicted = SparkKvPageCacheEvictUnused(&binding->page_cache);
		SparkStageKvBindingUnlock(binding,site,*held);
		if ( evicted != SPARK_STATUS_OK )
		{
			fprintf(stderr,"%s kv binding state store full: page %u has no record slot after evicting %u unused prefixes\n",binding->module_tag,job->page,evictions);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		evictions++;
		status = SparkStageKvStateWrite(binding,job);
	}
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"%s kv binding state record write failed page=%u status=%u\n",binding->module_tag,job->page,(unsigned)status);
	SPARK_RETURN(status);
}

static SparkStatus SparkStageKvBindingWriteStates(SparkStageKvBinding *binding,const uint32_t *slots,uint32_t lane_count,SparkStatus status,uint32_t site,uint64_t *held,uint32_t *raced)
{
	SparkStageKvStateJob *jobs;
	uint32_t count = 0u,index;
	uint64_t reset,start;
	SparkStatus result;
	*raced = 0u;
	if ( status != SPARK_STATUS_OK || binding->recurrent.lane_bytes == 0u )
		return(status);
	jobs = (SparkStageKvStateJob *)calloc(lane_count != 0u ? lane_count : 1u,sizeof(*jobs));
	if ( jobs == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	result = SparkStageKvCollectWrites(binding,slots,lane_count,jobs,&count);
	if ( result != SPARK_STATUS_OK || count == 0u )
	{
		free(jobs);
		SPARK_RETURN(result);
	}
	reset = binding->reset_generation;
	SparkStageKvBindingUnlock(binding,site,*held);
	start = SparkStageKvNowNs();
	for (index=0u; index<count && result==SPARK_STATUS_OK; index++)
		result = SparkStageKvWriteJob(binding,&jobs[index],site,held);
	atomic_fetch_add(&binding->recurrent_capture_ns,SparkStageKvNowNs() - start);
	(void)SparkStageKvBindingLock(binding,held);
	free(jobs);
	if ( binding->reset_generation != reset )
	{
		*raced = 1u;
		fprintf(stderr,"%s kv binding refused: state restore for request %s raced a reset\n",binding->module_tag,"finish");
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( result == SPARK_STATUS_OK )
	{
		atomic_fetch_add(&binding->recurrent_captures,count);
		atomic_fetch_add(&binding->recurrent_capture_bytes,(uint64_t)count * binding->recurrent.lane_bytes);
	}
	SPARK_RETURN(result);
}

static SparkStatus SparkStageKvBindingCopyLanes(SparkStageKvBinding *binding,uint32_t direction,const uint32_t *slots,uint32_t count,void *stream)
{
	uint32_t index;
	SparkStatus status = SPARK_STATUS_OK;
	for (index=0u; index<count && status==SPARK_STATUS_OK; index++)
		status = binding->recurrent.copy(binding->recurrent.context,direction,slots[index],SparkStageKvLaneState(binding,slots[index]),binding->recurrent.lane_bytes,stream);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingRecurrentRestore(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream)
{
	uint32_t *slots,index,slot,count = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t held;
	if ( binding == 0 || resident_slots == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( binding->recurrent.lane_bytes == 0u || lane_count == 0u )
		return(SPARK_STATUS_OK);
	slots = (uint32_t *)calloc(lane_count,sizeof(*slots));
	if ( slots == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)SparkStageKvBindingLock(binding,&held);
	for (index=0u; index<lane_count && status==SPARK_STATUS_OK; index++)
	{
		slot = resident_slots[index];
		if ( slot >= binding->resident_sequence_capacity )
			status = SPARK_STATUS_INVALID_ARGUMENT;
		else if ( (binding->lane_state_flags[slot] & SPARK_STAGE_KV_LANE_STATE_RESTORE_READY) != 0u )
		{
			binding->lane_state_flags[slot] &= (uint8_t)~SPARK_STAGE_KV_LANE_STATE_RESTORE_READY;
			slots[count++] = slot;
		}
		else if ( binding->lanes[slot].phase == SPARK_KV_LANE_TRANSACTION_EXECUTING && SparkKvLaneTransactionPrefixRestorePending(&binding->lanes[slot]) != 0u )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_CLAIM,held);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingCopyLanes(binding,SPARK_STAGE_KV_RECURRENT_FROM_BUFFER,slots,count,stream);
	free(slots);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingRecurrentCapture(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream)
{
	uint32_t *slots,index,slot,count = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t held;
	if ( binding == 0 || resident_slots == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( binding->recurrent.lane_bytes == 0u || lane_count == 0u )
		return(SPARK_STATUS_OK);
	slots = (uint32_t *)calloc(lane_count,sizeof(*slots));
	if ( slots == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)SparkStageKvBindingLock(binding,&held);
	for (index=0u; index<lane_count && status==SPARK_STATUS_OK; index++)
	{
		slot = resident_slots[index];
		if ( slot >= binding->resident_sequence_capacity )
			status = SPARK_STATUS_INVALID_ARGUMENT;
		else if ( binding->lanes[slot].phase == SPARK_KV_LANE_TRANSACTION_EXECUTING && (binding->lanes[slot].lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) != 0u )
		{
			binding->lane_state_flags[slot] |= SPARK_STAGE_KV_LANE_STATE_CAPTURED;
			slots[count++] = slot;
		}
	}
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_FINISH,held);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingCopyLanes(binding,SPARK_STAGE_KV_RECURRENT_TO_BUFFER,slots,count,stream);
	free(slots);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingCopyPage(const SparkStageKvBinding *binding,uint32_t direction,uintptr_t device_address,void *host,uint64_t bytes)
{
	return(SparkStageKvBindingPageCopy((void *)binding,direction,device_address,host,bytes));
}

SparkStatus SparkStageKvBindingInspect(SparkStageKvBinding *binding,SparkStageKvInspectFunction inspect,void *context)
{
	SparkStatus status;
	uint64_t held;
	if ( binding == 0 || inspect == 0 || binding->mutex_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	(void)SparkStageKvBindingLock(binding,&held);
	status = inspect(context,binding);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_FINISH,held);
	SPARK_RETURN(status);
}

uint32_t SparkStageKvBindingResetIsNew(SparkStageKvBinding *binding,uint64_t generation)
{
	uint32_t fresh;
	uint64_t held;
	if ( binding == 0 || binding->mutex_initialized == 0u )
		return(0u);
	(void)SparkStageKvBindingLock(binding,&held);
	fresh = generation > binding->reset_generation ? 1u : 0u;
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_RESET,held);
	return(fresh);
}

void SparkStageKvBindingTakeRecurrentCounters(SparkStageKvBinding *binding,SparkStageKvRecurrentCounters *counters)
{
	if ( counters == 0 )
		return;
	memset(counters,0,sizeof(*counters));
	if ( binding == 0 )
		return;
	counters->restores = atomic_exchange(&binding->recurrent_restores,0u);
	counters->restore_bytes = atomic_exchange(&binding->recurrent_restore_bytes,0u);
	counters->restore_ns = atomic_exchange(&binding->recurrent_restore_ns,0u);
	counters->captures = atomic_exchange(&binding->recurrent_captures,0u);
	counters->capture_bytes = atomic_exchange(&binding->recurrent_capture_bytes,0u);
	counters->capture_ns = atomic_exchange(&binding->recurrent_capture_ns,0u);
}

typedef struct SparkStageKvFinishWaiter
{
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	uint32_t done;
	SparkStatus status;
} SparkStageKvFinishWaiter;

static void SparkStageKvFinishWaitDone(void *context,SparkStatus status)
{
	SparkStageKvFinishWaiter *waiter = (SparkStageKvFinishWaiter *)context;
	(void)pthread_mutex_lock(&waiter->mutex);
	waiter->status = status;
	waiter->done = 1u;
	(void)pthread_cond_signal(&waiter->condition);
	(void)pthread_mutex_unlock(&waiter->mutex);
}

SparkStatus SparkStageKvBindingFinishWait(SparkStageKvBinding *binding,uint32_t dispatch_slot,const SparkStageKvBindingCompletion *completion)
{
	SparkStageKvBindingCompletion queued;
	SparkStageKvFinishWaiter waiter;
	SparkStatus status;
	if ( completion == 0 || completion->finished_function != 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&waiter,0,sizeof(waiter));
	if ( pthread_mutex_init(&waiter.mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( pthread_cond_init(&waiter.condition,0) != 0 )
	{
		(void)pthread_mutex_destroy(&waiter.mutex);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	queued = *completion;
	queued.finished_function = SparkStageKvFinishWaitDone;
	queued.finished_context = &waiter;
	status = SparkStageKvBindingFinishAsync(binding,dispatch_slot,&queued);
	if ( status == SPARK_STATUS_OK )
	{
		(void)pthread_mutex_lock(&waiter.mutex);
		while ( waiter.done == 0u )
			(void)pthread_cond_wait(&waiter.condition,&waiter.mutex);
		status = waiter.status;
		(void)pthread_mutex_unlock(&waiter.mutex);
	}
	(void)pthread_cond_destroy(&waiter.condition);
	(void)pthread_mutex_destroy(&waiter.mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkStageKvBindingApplyCompletion(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,SparkStatus status,uint32_t extra_tokens,const uint8_t *bound,const uint64_t *sequence_ids,const uint64_t *next_positions)
{
	uint32_t lane,resident;
	SparkStatus result;
	result = SparkKvLaneTransactionsFinish(&binding->transactions,resident_slots,lane_count,status,extra_tokens);
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
			uint64_t floor = atomic_exchange_explicit(&binding->lane_pending_floors[resident],UINT64_MAX,memory_order_acq_rel);
			atomic_store_explicit(&binding->lane_rewind_floors[resident],floor < next_positions[lane] ? floor : next_positions[lane],memory_order_release);
			atomic_store_explicit(&binding->lane_rewind_ceilings[resident],floor < next_positions[lane] ? next_positions[lane] : 0u,memory_order_release);
		}
		else
			memset(binding->page_table_shadow + (uint64_t)resident * binding->pages_per_sequence,0xff,(size_t)binding->pages_per_sequence * sizeof(uint32_t));
	}
	return(result);
}

static SparkStatus SparkStageKvBindingPublishStates(SparkStageKvBinding *binding,const uint32_t *indices,uint32_t lane_count,uint64_t *held)
{
	SparkStatus status,result;
	uint64_t reset = binding->reset_generation;
	uint32_t raced = 0u;
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_PUBLISH,*held);
	status = SparkStageKvBindingCopyLanes(binding,SPARK_STAGE_KV_RECURRENT_TO_BUFFER,indices,lane_count,0);
	(void)SparkStageKvBindingLock(binding,held);
	if ( binding->reset_generation != reset )
	{
		fprintf(stderr,"%s kv binding refused: state restore for request %s raced a reset\n",binding->module_tag,"publish");
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t lane;
		for (lane=0u; lane<lane_count; lane++)
			binding->lane_state_flags[indices[lane]] |= SPARK_STAGE_KV_LANE_STATE_CAPTURED;
		status = SparkStageKvBindingWriteStates(binding,indices,lane_count,SPARK_STATUS_OK,SPARK_STAGE_KV_LOCK_SITE_PUBLISH,held,&raced);
	}
	result = raced == 0u ? SparkKvLaneTransactionsFinish(&binding->transactions,indices,lane_count,status,0u) : status;
	SparkStageKvClearSlotStates(binding,indices,lane_count,SPARK_STAGE_KV_LANE_STATE_CAPTURED | SPARK_STAGE_KV_LANE_STATE_RESTORE_READY);
	SPARK_RETURN(status != SPARK_STATUS_OK ? status : result);
}

static SparkStatus SparkStageKvBindingPublishLanes(SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,const uint32_t *indices)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t lane,resident;
	uint64_t held;
	if ( SparkStageKvBindingLock(binding,&held) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (lane=0u; lane<frame->cache_lane_count && status==SPARK_STATUS_OK; lane++)
	{
		const SparkModelDriverCacheLane *cache_lane = &frame->cache_lanes[lane];
		resident = indices[lane];
		if ( cache_lane->flags != SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH || cache_lane->publish_token_count == 0u ||
			cache_lane->sequence_position != cache_lane->publish_token_count || cache_lane->context_token_count != cache_lane->publish_token_count ||
			atomic_load_explicit(&binding->lane_bound[resident],memory_order_acquire) == 0u ||
			atomic_load_explicit(&binding->lane_sequence_ids[resident],memory_order_acquire) != cache_lane->sequence_id ||
			(atomic_load_explicit(&binding->lane_next_positions[resident],memory_order_acquire) != cache_lane->sequence_position &&
			 (atomic_load_explicit(&binding->lane_rewind_ceilings[resident],memory_order_acquire) != atomic_load_explicit(&binding->lane_next_positions[resident],memory_order_acquire) ||
			  atomic_load_explicit(&binding->lane_next_positions[resident],memory_order_acquire) < cache_lane->sequence_position ||
			  atomic_load_explicit(&binding->lane_rewind_floors[resident],memory_order_acquire) > cache_lane->sequence_position)) )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkKvLaneTransactionsClaim(&binding->transactions,frame);
	if ( status == SPARK_STATUS_OK && binding->recurrent.lane_bytes != 0u )
		status = SparkStageKvBindingPublishStates(binding,indices,frame->cache_lane_count,&held);
	else if ( status == SPARK_STATUS_OK )
		status = SparkKvLaneTransactionsFinish(&binding->transactions,indices,frame->cache_lane_count,SPARK_STATUS_OK,0u);
	SparkStageKvBindingWakeSaver(binding);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_PUBLISH,held);
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

SparkStatus SparkStageKvBindingFinishAsync(SparkStageKvBinding *binding,uint32_t dispatch_slot,const SparkStageKvBindingCompletion *completion)
{
	SparkStageKvBindingCompletionRecord *record;
	uint64_t entry_ns,elapsed;
	uint32_t state;
	entry_ns = SparkStageKvNowNs();
	if ( binding == 0 || completion == 0 || binding->sync_initialized == 0u || dispatch_slot >= binding->pipeline_slot_count ||
		completion->lane_count == 0u || completion->lane_count > binding->resident_sequence_capacity || completion->resident_slots == 0 ||
		completion->bound == 0 || completion->sequence_ids == 0 || completion->next_positions == 0 || completion->finished_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&binding->completion_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( binding->completion_stop != 0u )
	{
		(void)pthread_mutex_unlock(&binding->completion_mutex);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	record = &binding->completion_records[dispatch_slot];
	state = record->state;
	if ( state != SPARK_STAGE_KV_COMPLETION_FREE )
	{
		(void)pthread_mutex_unlock(&binding->completion_mutex);
		fprintf(stderr,"%s kv binding completion refused slot=%u state=%u\n",binding->module_tag,dispatch_slot,state);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	record->lane_count = completion->lane_count;
	record->extra_tokens = completion->extra_tokens;
	record->status = completion->status;
	memcpy(record->resident_slots,completion->resident_slots,(size_t)completion->lane_count * sizeof(uint32_t));
	memcpy(record->bound,completion->bound,(size_t)completion->lane_count * sizeof(uint8_t));
	memcpy(record->sequence_ids,completion->sequence_ids,(size_t)completion->lane_count * sizeof(uint64_t));
	memcpy(record->next_positions,completion->next_positions,(size_t)completion->lane_count * sizeof(uint64_t));
	record->finished_function = completion->finished_function;
	record->finished_context = completion->finished_context;
	record->queued_ns = SparkStageKvNowNs();
	record->state = SPARK_STAGE_KV_COMPLETION_QUEUED;
	binding->completion_queue[(binding->completion_head + binding->completion_count) % binding->pipeline_slot_count] = dispatch_slot;
	binding->completion_count++;
	elapsed = SparkStageKvNowNs() - entry_ns;
	binding->counters.entry_count++;
	binding->counters.entry_total_ns += elapsed;
	if ( elapsed > binding->counters.entry_max_ns )
		binding->counters.entry_max_ns = elapsed;
	(void)pthread_cond_signal(&binding->completion_ready);
	(void)pthread_mutex_unlock(&binding->completion_mutex);
	return(SPARK_STATUS_OK);
}

static void *SparkStageKvBindingCompletionMain(void *context)
{
	SparkStageKvBinding *binding = (SparkStageKvBinding *)context;
	SparkStageKvBindingCompletionRecord *record;
	SparkStageKvBindingFinishedFunction function;
	void *function_context;
	SparkStatus result;
	uint64_t held,waited;
	uint32_t slot,raced;
	for (;;)
	{
		(void)pthread_mutex_lock(&binding->completion_mutex);
		while ( binding->completion_count == 0u && binding->completion_stop == 0u )
			(void)pthread_cond_wait(&binding->completion_ready,&binding->completion_mutex);
		if ( binding->completion_count == 0u )
		{
			(void)pthread_mutex_unlock(&binding->completion_mutex);
			break;
		}
		slot = binding->completion_queue[binding->completion_head];
		binding->completion_head = (binding->completion_head + 1u) % binding->pipeline_slot_count;
		binding->completion_count--;
		binding->completion_running = 1u;
		(void)pthread_mutex_unlock(&binding->completion_mutex);
		record = &binding->completion_records[slot];
		if ( SparkStageKvBindingLock(binding,&held) == SPARK_STATUS_OK )
		{
			result = SparkStageKvBindingWriteStates(binding,record->resident_slots,record->lane_count,record->status,SPARK_STAGE_KV_LOCK_SITE_FINISH,&held,&raced);
			if ( raced == 0u )
				result = SparkStageKvBindingApplyCompletion(binding,record->resident_slots,record->lane_count,result,record->extra_tokens,record->bound,record->sequence_ids,record->next_positions);
			SparkStageKvClearSlotStates(binding,record->resident_slots,record->lane_count,SPARK_STAGE_KV_LANE_STATE_CAPTURED | SPARK_STAGE_KV_LANE_STATE_RESTORE_READY);
			SparkStageKvBindingWakeSaver(binding);
			SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_FINISH,held);
		}
		else
			result = SPARK_STATUS_INTERNAL_ERROR;
		function = record->finished_function;
		function_context = record->finished_context;
		(void)pthread_mutex_lock(&binding->completion_mutex);
		waited = SparkStageKvNowNs() - record->queued_ns;
		binding->counters.completion_count++;
		binding->counters.completion_queue_total_ns += waited;
		if ( waited > binding->counters.completion_queue_max_ns )
			binding->counters.completion_queue_max_ns = waited;
		record->state = SPARK_STAGE_KV_COMPLETION_FREE;
		(void)pthread_mutex_unlock(&binding->completion_mutex);
		function(function_context,result);
		(void)pthread_mutex_lock(&binding->completion_mutex);
		binding->completion_running = 0u;
		if ( binding->completion_count == 0u )
			(void)pthread_cond_broadcast(&binding->completion_idle);
		(void)pthread_mutex_unlock(&binding->completion_mutex);
	}
	return(0);
}

static void SparkStageKvBindingSaveOne(SparkStageKvBinding *binding,uint64_t *held)
{
	SparkKvPageCacheSaveWork work;
	SparkStatus status,copy;
	status = SparkKvPageCacheSaveTake(&binding->page_cache,&work);
	if ( status == SPARK_STATUS_BUSY )
	{
		SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_SAVE,*held);
		(void)SparkKvSnapshotFlush(binding->page_cache.snapshot->store);
		(void)SparkStageKvBindingLock(binding,held);
		status = SparkKvPageCacheSaveTake(&binding->page_cache,&work);
		if ( status == SPARK_STATUS_BUSY )
		{
			SparkStageKvBindingAccount(binding,SPARK_STAGE_KV_LOCK_SITE_SAVE,*held);
			(void)pthread_cond_wait(&binding->save_ready,&binding->mutex);
			*held = SparkStageKvNowNs();
			return;
		}
	}
	if ( status != SPARK_STATUS_OK )
		return;
	binding->save_running = 1u;
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_SAVE,*held);
	copy = SparkKvPageCacheSaveCopy(&binding->page_cache,&work);
	(void)SparkStageKvBindingLock(binding,held);
	if ( SparkKvPageCacheSaveFinish(&binding->page_cache,&work,copy) == SPARK_STATUS_OK )
		binding->counters.save_worker_page_count++;
	else
		binding->counters.save_worker_failure_count++;
	binding->save_running = 0u;
}

static void *SparkStageKvBindingSaveMain(void *context)
{
	SparkStageKvBinding *binding = (SparkStageKvBinding *)context;
	uint64_t held;
	(void)SparkStageKvBindingLock(binding,&held);
	for (;;)
	{
		while ( binding->save_stop == 0u && (binding->page_cache.snapshot == 0 || SparkKvPageCacheSavePending(&binding->page_cache) == 0u) )
		{
			(void)pthread_cond_broadcast(&binding->save_idle);
			SparkStageKvBindingAccount(binding,SPARK_STAGE_KV_LOCK_SITE_SAVE,held);
			(void)pthread_cond_wait(&binding->save_ready,&binding->mutex);
			held = SparkStageKvNowNs();
		}
		if ( binding->save_stop != 0u )
			break;
		SparkStageKvBindingSaveOne(binding,&held);
	}
	(void)pthread_cond_broadcast(&binding->save_idle);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_SAVE,held);
	return(0);
}

SparkStatus SparkStageKvBindingFenceExecution(SparkStageKvBinding *binding,void *stream)
{
	if ( binding == 0 || binding->copier_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvDeviceCopierFence(&binding->copier,stream));
}

static void SparkStageKvBindingDeadline(uint64_t timeout_ns,struct timespec *deadline)
{
	clock_gettime(CLOCK_REALTIME,deadline);
	deadline->tv_sec += (time_t)(timeout_ns / 1000000000ull);
	deadline->tv_nsec += (long)(timeout_ns % 1000000000ull);
	if ( deadline->tv_nsec >= 1000000000L )
	{
		deadline->tv_sec += 1;
		deadline->tv_nsec -= 1000000000L;
	}
}

SparkStatus SparkStageKvBindingQuiesce(SparkStageKvBinding *binding,uint64_t timeout_ns)
{
	struct timespec deadline;
	uint32_t queued,completing,saving;
	int waited = 0;
	if ( binding == 0 || binding->sync_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageKvBindingDeadline(timeout_ns,&deadline);
	(void)pthread_mutex_lock(&binding->completion_mutex);
	while ( waited == 0 && (binding->completion_count != 0u || binding->completion_running != 0u) )
		waited = pthread_cond_timedwait(&binding->completion_idle,&binding->completion_mutex,&deadline);
	queued = binding->completion_count;
	completing = binding->completion_running;
	(void)pthread_mutex_unlock(&binding->completion_mutex);
	(void)pthread_mutex_lock(&binding->mutex);
	while ( waited == 0 && binding->save_running != 0u )
		waited = pthread_cond_timedwait(&binding->save_idle,&binding->mutex,&deadline);
	saving = binding->save_running;
	(void)pthread_mutex_unlock(&binding->mutex);
	if ( queued != 0u || completing != 0u || saving != 0u )
	{
		fprintf(stderr,"%s kv binding quiesce timeout queued=%u completing=%u saving=%u\n",binding->module_tag,queued,completing,saving);
		SPARK_FAIL(SPARK_STATUS_BUSY);
	}
	return(SPARK_STATUS_OK);
}

static void SparkStageKvBindingPoolMark(SparkStageKvBinding *binding)
{
	binding->pool_pressure_mark = binding->arena.resident_capacity_stall_count + binding->arena.resident_evicted_block_count + binding->page_cache.evicted_entry_count;
}

static void SparkStageKvBindingPoolOutcome(SparkStageKvBinding *binding,const char *step,SparkStatus status)
{
	if ( (uint32_t)status != binding->pool_last_status && status != SPARK_STATUS_OK )
		fprintf(stderr,"%s kv pool %s failed status=%s chunks=%u of %u\n",binding->module_tag,step,SparkStatusToString(status),binding->kv_pool.mapped_count,binding->kv_pool.chunk_capacity);
	binding->pool_last_status = (uint32_t)status;
}

static void SparkStageKvBindingPoolGrow(SparkStageKvBinding *binding)
{
	SparkWeightdKvPoolState state;
	uint32_t before = binding->kv_pool.mapped_count,pages = binding->arena.resident_block_capacity,target,limit;
	SparkStatus status;
	target = before > binding->kv_pool.chunk_capacity / 2u ? binding->kv_pool.chunk_capacity : 2u * before;
	(void)pthread_mutex_unlock(&binding->mutex);
	status = SparkWeightdKvPoolGrow(&binding->kv_pool,target,&state,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS);
	(void)pthread_mutex_lock(&binding->mutex);
	SparkStageKvBindingPoolOutcome(binding,"grow",status);
	limit = SparkStageKvBindingPoolPageLimit(binding,binding->kv_pool.mapped_count);
	if ( limit > binding->arena.resident_block_capacity )
		binding->arena.resident_block_capacity = limit;
	binding->pool_mapped_bytes = (uint64_t)binding->kv_pool.mapped_count * binding->kv_pool.chunk_bytes;
	if ( status == SPARK_STATUS_OK )
		binding->kv_pool_wanted_chunks = state.wanted_chunks;
	if ( binding->kv_pool.mapped_count == before )
		return;
	binding->pool_grow_count++;
	fprintf(stderr,"%s kv pool grew chunks=%u->%u of %u pages=%u->%u of %u wanted_chunks=%u kv_committed=%llu kv_reserve=%llu\n",binding->module_tag,before,binding->kv_pool.mapped_count,
		binding->kv_pool.chunk_capacity,pages,binding->arena.resident_block_capacity,binding->physical_page_count,binding->kv_pool_wanted_chunks,
		(unsigned long long)binding->kv_pool.kv_committed_bytes,(unsigned long long)binding->kv_pool.kv_reserve_bytes);
}

static void SparkStageKvBindingPoolShrink(SparkStageKvBinding *binding,uint64_t reclaim_bytes)
{
	SparkWeightdKvPoolState state;
	uint32_t before = binding->kv_pool.mapped_count,pages = binding->arena.resident_block_capacity,give,target,kept_pages = 0u,kept,vacated = 0u;
	SparkStatus status;
	give = (uint32_t)((reclaim_bytes + binding->kv_pool.chunk_bytes - 1u) / binding->kv_pool.chunk_bytes);
	if ( give > before - binding->kv_pool_minimum_chunks )
		give = before - binding->kv_pool_minimum_chunks;
	target = before - give;
	status = SparkKvPageCacheVacateResident(&binding->page_cache,SparkStageKvBindingPoolPageLimit(binding,target),&kept_pages,&vacated);
	SparkStageKvBindingPoolMark(binding);
	binding->pool_vacated_pages += vacated;
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageKvBindingPoolOutcome(binding,"vacate",status);
		return;
	}
	kept = SparkStageKvBindingPoolChunksFor(binding,kept_pages);
	if ( kept < target )
		kept = target;
	if ( kept >= before )
		return;
	binding->arena.resident_block_capacity = SparkStageKvBindingPoolPageLimit(binding,kept);
	binding->arena.next_resident_slot_scan = 0u;
	(void)pthread_mutex_unlock(&binding->mutex);
	status = cudaDeviceSynchronize() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdKvPoolShrink(&binding->kv_pool,kept,&state,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS);
	(void)pthread_mutex_lock(&binding->mutex);
	SparkStageKvBindingPoolOutcome(binding,"shrink",status);
	binding->pool_mapped_bytes = (uint64_t)binding->kv_pool.mapped_count * binding->kv_pool.chunk_bytes;
	binding->pool_shrink_count++;
	fprintf(stderr,"%s kv pool shrank chunks=%u->%u of %u pages=%u->%u vacated_pages=%u reclaim_wanted=%llu kv_committed=%llu status=%s\n",binding->module_tag,before,binding->kv_pool.mapped_count,
		binding->kv_pool.chunk_capacity,pages,binding->arena.resident_block_capacity,vacated,(unsigned long long)reclaim_bytes,(unsigned long long)binding->kv_pool.kv_committed_bytes,
		SparkStatusToString(status));
}

static void SparkStageKvBindingPoolTick(SparkStageKvBinding *binding)
{
	SparkWeightdKvPoolState state;
	uint64_t pressure = binding->arena.resident_capacity_stall_count + binding->arena.resident_evicted_block_count + binding->page_cache.evicted_entry_count;
	SparkStatus status = SPARK_STATUS_OK;
	if ( pressure != binding->pool_pressure_mark )
	{
		binding->pool_pressure_mark = pressure;
		binding->pool_idle_ticks = 0u;
		if ( binding->kv_pool.mapped_count < binding->kv_pool.chunk_capacity )
			SparkStageKvBindingPoolGrow(binding);
		return;
	}
	if ( binding->pool_idle_ticks < SPARK_STAGE_KV_POOL_IDLE_TICKS )
	{
		binding->pool_idle_ticks++;
		return;
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	if ( binding->kv_pool_wanted_chunks != 0u )
		status = SparkWeightdKvPoolGrow(&binding->kv_pool,binding->kv_pool.mapped_count,&state,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdKvPoolStatus(&binding->kv_pool,&state,SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS);
	(void)pthread_mutex_lock(&binding->mutex);
	SparkStageKvBindingPoolOutcome(binding,"status",status);
	if ( status != SPARK_STATUS_OK )
		return;
	binding->kv_pool_wanted_chunks = state.wanted_chunks;
	if ( state.reclaim_wanted_bytes != 0u && binding->kv_pool.mapped_count > binding->kv_pool_minimum_chunks )
		SparkStageKvBindingPoolShrink(binding,state.reclaim_wanted_bytes);
}

static void *SparkStageKvBindingPoolMain(void *context)
{
	SparkStageKvBinding *binding = (SparkStageKvBinding *)context;
	struct timespec wake;
	(void)pthread_mutex_lock(&binding->mutex);
	SparkStageKvBindingPoolMark(binding);
	while ( binding->pool_stop == 0u )
	{
		SparkStageKvBindingDeadline(SPARK_STAGE_KV_POOL_TICK_NS,&wake);
		(void)pthread_cond_timedwait(&binding->pool_wake,&binding->mutex,&wake);
		if ( binding->pool_stop == 0u )
			SparkStageKvBindingPoolTick(binding);
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	return(0);
}

static SparkStatus SparkStageKvBindingStartPool(SparkStageKvBinding *binding)
{
	if ( pthread_cond_init(&binding->pool_wake,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->pool_wake_initialized = 1u;
	if ( pthread_create(&binding->pool_thread,0,SparkStageKvBindingPoolMain,binding) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->pool_started = 1u;
	return(SPARK_STATUS_OK);
}

void SparkStageKvBindingStop(SparkStageKvBinding *binding)
{
	if ( binding == 0 || binding->sync_initialized == 0u )
		return;
	(void)pthread_mutex_lock(&binding->mutex);
	binding->pool_stop = 1u;
	if ( binding->pool_wake_initialized != 0u )
		(void)pthread_cond_broadcast(&binding->pool_wake);
	(void)pthread_mutex_unlock(&binding->mutex);
	if ( binding->pool_started != 0u )
		(void)pthread_join(binding->pool_thread,0);
	binding->pool_started = 0u;
	(void)pthread_mutex_lock(&binding->completion_mutex);
	binding->completion_stop = 1u;
	(void)pthread_cond_broadcast(&binding->completion_ready);
	(void)pthread_mutex_unlock(&binding->completion_mutex);
	if ( binding->completion_started != 0u )
		(void)pthread_join(binding->completion_thread,0);
	binding->completion_started = 0u;
	(void)pthread_mutex_lock(&binding->mutex);
	binding->save_stop = 1u;
	(void)pthread_cond_broadcast(&binding->save_ready);
	(void)pthread_mutex_unlock(&binding->mutex);
	if ( binding->save_started != 0u )
		(void)pthread_join(binding->save_thread,0);
	binding->save_started = 0u;
	(void)pthread_mutex_lock(&binding->mutex);
	binding->restore_stop = 1u;
	if ( binding->restore_ready_initialized != 0u )
		(void)pthread_cond_broadcast(&binding->restore_ready);
	(void)pthread_mutex_unlock(&binding->mutex);
	if ( binding->restore_started != 0u )
		(void)pthread_join(binding->restore_thread,0);
	binding->restore_started = 0u;
	binding->transactions.restore_async = 0u;
}

static SparkStatus SparkStageKvBindingSaveDrain(SparkStageKvBinding *binding,uint64_t deadline_ns)
{
	struct timespec wake;
	SparkStatus status = SPARK_STATUS_OK;
	(void)pthread_mutex_lock(&binding->mutex);
	while ( SparkKvPageCacheSavePending(&binding->page_cache) != 0u || binding->save_running != 0u )
	{
		if ( binding->snapshot_store.failed_status != SPARK_STATUS_OK )
		{
			status = binding->snapshot_store.failed_status;
			break;
		}
		if ( SparkStageKvNowNs() >= deadline_ns )
		{
			status = SPARK_STATUS_BUSY;
			break;
		}
		(void)pthread_cond_signal(&binding->save_ready);
		SparkStageKvBindingDeadline(10000000u,&wake);
		(void)pthread_cond_timedwait(&binding->save_idle,&binding->mutex,&wake);
	}
	(void)pthread_mutex_unlock(&binding->mutex);
	SPARK_RETURN(status);
}

SparkStatus SparkStageKvBindingSaveAll(SparkStageKvBinding *binding,uint64_t deadline_ns,uint32_t *saved_out,uint32_t *unsaved_out,uint32_t *ineligible_out)
{
	uint32_t unsaved_start = 0u,unsaved = 0u,ineligible = 0u,marked,deferred,skipped;
	SparkStatus status = SPARK_STATUS_OK,drain = SPARK_STATUS_OK;
	if ( binding == 0 || binding->mutex_initialized == 0u || binding->page_cache.snapshot == 0 || saved_out == 0 || unsaved_out == 0 || ineligible_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	(void)pthread_mutex_lock(&binding->mutex);
	status = SparkKvPageCacheCountUnsaved(&binding->page_cache,&unsaved_start,&ineligible);
	(void)pthread_mutex_unlock(&binding->mutex);
	unsaved = unsaved_start;
	while ( status == SPARK_STATUS_OK && unsaved != 0u )
	{
		(void)pthread_mutex_lock(&binding->mutex);
		status = SparkKvPageCacheMarkAllUnsaved(&binding->page_cache,binding->save_order,binding->logical_page_count,&marked,&deferred,&skipped);
		SparkStageKvBindingWakeSaver(binding);
		(void)pthread_mutex_unlock(&binding->mutex);
		if ( status != SPARK_STATUS_OK )
			break;
		drain = SparkStageKvBindingSaveDrain(binding,deadline_ns);
		(void)pthread_mutex_lock(&binding->mutex);
		status = SparkKvPageCacheCountUnsaved(&binding->page_cache,&unsaved,&ineligible);
		(void)pthread_mutex_unlock(&binding->mutex);
		if ( drain != SPARK_STATUS_OK || SparkStageKvNowNs() >= deadline_ns || binding->snapshot_store.failed_status != SPARK_STATUS_OK || (marked == 0u && deferred == 0u) )
			break;
	}
	(void)SparkKvSnapshotFlush(&binding->snapshot_store);
	*saved_out = unsaved_start >= unsaved ? unsaved_start - unsaved : 0u;
	*unsaved_out = unsaved;
	*ineligible_out = ineligible;
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( unsaved == 0u )
		return(SPARK_STATUS_OK);
	if ( binding->snapshot_store.failed_status != SPARK_STATUS_OK )
		SPARK_RETURN(binding->snapshot_store.failed_status);
	if ( drain != SPARK_STATUS_OK )
		SPARK_RETURN(drain);
	SPARK_FAIL(SPARK_STATUS_BUSY);
}

void SparkStageKvBindingKvStoreCounters(SparkStageKvBinding *binding,SparkModelDriverKvStoreCounters *counters)
{
	const SparkKvPageCacheSnapshot *snapshot;
	SparkKvSnapshotStore sample;
	uint64_t matching = 0u,foreign = 0u;
	if ( counters == 0 )
		return;
	memset(counters,0,sizeof(*counters));
	if ( binding == 0 || binding->mutex_initialized == 0u )
		return;
	(void)pthread_mutex_lock(&binding->mutex);
	counters->pool_resident_pages = binding->arena.resident_block_count;
	counters->pool_physical_pages = binding->arena.resident_block_capacity;
	counters->pool_logical_pages = binding->logical_page_count;
	counters->pool_retained_pages = binding->arena.retained_block_count;
	counters->pool_evicted_entries = binding->page_cache.evicted_entry_count;
	counters->pool_resident_evictions = binding->arena.resident_evicted_block_count;
	counters->spill_write_bytes = binding->page_store.write_bytes;
	counters->spill_read_bytes = binding->page_store.read_bytes;
	counters->spill_digest_mismatches = binding->page_store.read_digest_mismatch_count;
	counters->spill_read_errors = binding->page_store.read_error_count;
	counters->pool_device_bytes = binding->pool_mapped_bytes;
	counters->pool_generation = binding->kv_pool.pool_generation;
	counters->pool_reattached = binding->kv_pool.reattached;
	counters->pool_adopted_pages = binding->kv_pool_adopted_pages;
	SparkKvWriteBudgetRefill(&binding->write_budget,SparkStageKvNowNs());
	counters->write_budget_bytes_per_day = binding->write_budget.bytes_per_day;
	counters->write_budget_available_bytes = binding->write_budget.available_bytes;
	counters->write_budget_overrun_bytes = binding->write_budget.overrun_bytes;
	counters->write_budget_refused_saves = binding->write_budget.refused_saves;
	counters->write_budget_discarded_pages = binding->write_budget.discarded_pages;
	snapshot = binding->page_cache.snapshot;
	if ( snapshot == 0 )
	{
		(void)pthread_mutex_unlock(&binding->mutex);
		return;
	}
	counters->save_count = snapshot->save_count;
	counters->save_page_count = snapshot->save_page_count;
	counters->save_ns = snapshot->save_ns;
	counters->save_failure_count = snapshot->save_failure_count;
	counters->save_deferred_count = snapshot->save_deferred_count + snapshot->save_skipped_count;
	counters->restore_count = snapshot->restore_count;
	counters->restore_page_count = snapshot->restore_page_count;
	counters->restore_ns = snapshot->restore_ns;
	counters->restore_miss_count = snapshot->restore_miss_count;
	counters->restore_corrupt_count = snapshot->restore_corrupt_count;
	counters->restore_read_error_count = snapshot->restore_read_error_count;
	counters->restore_failure_count = snapshot->restore_failure_count;
	if ( SparkKvSnapshotStoreSample(&binding->snapshot_store,&sample) == SPARK_STATUS_OK )
	{
		counters->store_failed_status = (uint32_t)sample.failed_status;
		counters->store_used_bytes = sample.used_bytes;
		counters->store_maximum_bytes = sample.maximum_bytes;
		counters->store_file_count = sample.file_count;
		counters->store_checksum_failure_count = sample.checksum_failure_count;
		counters->store_removed_temporary_count = sample.removed_temporary_count;
		counters->store_eviction_count = sample.eviction_count;
		counters->store_write_failure_count = sample.write_failure_count;
		counters->store_queue_full_count = sample.queue_full_count;
		counters->store_queued_count = sample.queued_count;
	}
	if ( SparkKvSnapshotCountLayout(&binding->snapshot_store,binding->layout_sha256,&matching,&foreign) == SPARK_STATUS_OK )
		counters->store_foreign_layout_file_count = foreign;
	counters->attached = 1u;
	(void)pthread_mutex_unlock(&binding->mutex);
}

SparkStatus SparkStageKvBindingSampleCounters(SparkStageKvBinding *binding,SparkStageKvBindingCounters *counters)
{
	SparkStageKvBindingCounters sample;
	if ( binding == 0 || counters == 0 || binding->sync_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	(void)pthread_mutex_lock(&binding->completion_mutex);
	sample = binding->counters;
	(void)pthread_mutex_unlock(&binding->completion_mutex);
	(void)pthread_mutex_lock(&binding->mutex);
	memcpy(sample.lock_sites,binding->counters.lock_sites,sizeof(sample.lock_sites));
	sample.save_worker_page_count = binding->counters.save_worker_page_count;
	sample.save_worker_failure_count = binding->counters.save_worker_failure_count;
	sample.copy_on_write_count = binding->copier.copy_count;
	sample.copy_on_write_bytes = binding->copier.copy_bytes;
	(void)pthread_mutex_unlock(&binding->mutex);
	*counters = sample;
	return(SPARK_STATUS_OK);
}
