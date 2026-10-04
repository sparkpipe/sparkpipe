#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_weight_codec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define SPARK_STAGE_KV_COMPLETION_FREE 0u
#define SPARK_STAGE_KV_COMPLETION_QUEUED 1u

static const char *const SparkStageKvLockSiteNames[SPARK_STAGE_KV_LOCK_SITE_COUNT] =
{
	"admit","reset","continuity","claim","finish","publish","save"
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
		span = (uint64_t)binding->physical_page_count * packed;
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
	if ( binding->blocks == 0 || binding->resident_slot_logical_block_indices == 0 || binding->entries == 0 || binding->sequences == 0 || binding->hash_bucket_heads == 0 || binding->entry_indices_by_logical_page == 0 ||
		binding->lanes == 0 || binding->logical_pages == 0 || binding->page_table_shadow == 0 || binding->lane_bound == 0 || binding->lane_sequence_ids == 0 || binding->lane_next_positions == 0 )
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
	copy.physical_page_count = binding->physical_page_count;
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
static void *SparkStageKvBindingSaveMain(void *context);

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
	fprintf(stderr,"%s kv binding async completion_records=%u copy_on_write=device copy_stream=nonblocking\n",binding->module_tag,binding->pipeline_slot_count);
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
	binding->save_order = (SparkKvPageCacheSaveOrder *)calloc(binding->logical_page_count,sizeof(*binding->save_order));
	if ( cudaHostAlloc((void **)&binding->snapshot_page,(size_t)binding->page_bytes,cudaHostAllocPortable) != cudaSuccess )
		binding->snapshot_page = 0;
	if ( binding->snapshot_links == 0 || binding->snapshot_pending == 0 || binding->save_order == 0 || binding->snapshot_page == 0 )
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

SparkStatus SparkStageKvBindingInitialize(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration)
{
	SparkKvModelTable table;
	uint64_t lane_entries;
	uint32_t region;
	SparkStatus status;
	cudaError_t error;
	if ( binding == 0 || configuration == 0 || configuration->ledger == 0 || configuration->module_tag == 0 || configuration->block_token_count == 0u || configuration->region_count == 0u ||
		configuration->region_count > SPARK_STAGE_KV_MAX_REGIONS || configuration->resident_sequence_capacity == 0u || configuration->max_sequence_positions == 0u || configuration->pipeline_slot_count == 0u ||
		configuration->physical_page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvBindingRefuse(configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(binding,0,sizeof(*binding));
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
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pthread_mutex_init(&binding->mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	binding->mutex_initialized = 1u;
	status = SparkStageKvBindingOpenSnapshot(binding,configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingStartAsync(binding);
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
	fprintf(stderr,"%s kv binding arena evicted=%llu park_failures=%llu park_backing_full=%llu park_stalls=%llu store_writes=%llu store_reads=%llu backing_pages=%u backing_reclaims=%llu\n",binding->module_tag,
		(unsigned long long)binding->arena.resident_evicted_block_count,(unsigned long long)binding->arena.park_failure_count,
		(unsigned long long)binding->arena.park_backing_full_count,(unsigned long long)binding->arena.park_stall_count,
		(unsigned long long)binding->page_store.write_count,(unsigned long long)binding->page_store.read_count,
		binding->page_store.backing_page_count,(unsigned long long)binding->page_cache.backing_reclaim_count);
}

static void SparkStageKvBindingSaveAtDestroy(SparkStageKvBinding *binding)
{
	uint32_t saved = 0u,unsaved = 0u,ineligible = 0u;
	SparkStatus status;
	if ( binding->mutex_initialized == 0u || binding->page_cache.snapshot == 0 || binding->save_started == 0u )
		return;
	status = SparkStageKvBindingSaveAll(binding,SparkStageKvNowNs() + SPARK_STAGE_KV_DESTROY_SAVE_TIMEOUT_NS,&saved,&unsaved,&ineligible);
	fprintf(stderr,"%s kv snapshot store close saved_entries=%u unsaved_entries=%u ineligible_entries=%u used_bytes=%llu files=%llu save_failures=%llu save_deferred=%llu status=%s\n",
		binding->module_tag,saved,unsaved,ineligible,(unsigned long long)binding->snapshot_store.used_bytes,(unsigned long long)binding->snapshot_store.file_count,
		(unsigned long long)binding->snapshot.save_failure_count,(unsigned long long)binding->snapshot.save_deferred_count,SparkStatusToString(status));
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
	if ( binding->page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&binding->page_store);
	if ( binding->copier_initialized != 0u )
	{
		(void)SparkKvDeviceCopierDrain(&binding->copier);
		SparkKvDeviceCopierDestroy(&binding->copier);
	}
	if ( binding->copy_stream != 0 )
		(void)cudaStreamDestroy((cudaStream_t)binding->copy_stream);
	if ( binding->snapshot_store.runtime != 0 )
		SparkKvSnapshotStoreClose(&binding->snapshot_store);
	free(binding->snapshot_links);
	free(binding->snapshot_pending);
	free(binding->save_order);
	if ( binding->snapshot_page != 0 )
		(void)cudaFreeHost(binding->snapshot_page);
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
	if ( binding->mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&binding->mutex);
	if ( binding->sync_initialized != 0u )
	{
		(void)pthread_mutex_destroy(&binding->completion_mutex);
		(void)pthread_cond_destroy(&binding->completion_ready);
		(void)pthread_cond_destroy(&binding->completion_idle);
		(void)pthread_cond_destroy(&binding->save_ready);
		(void)pthread_cond_destroy(&binding->save_idle);
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
	status = SparkStageKvBindingLock(binding,&held);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = binding->control_generation != 0u && binding->control_generation != request->control_generation ?
		SPARK_STATUS_VALIDATION_FAILED : SparkKvLaneTransactionsAdmit(&binding->transactions,request);
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
		}
		binding->reset_generation = generation;
		binding->control_generation = 0u;
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

static uint32_t SparkStageKvPrefixRestorePending(const SparkKvLaneTransaction *owner)
{
	return(owner != 0 && (owner->mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u && (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && owner->lane.sequence_position != 0u);
}

static SparkStatus SparkStageKvBindingLoadContinuity(const SparkStageKvBinding *binding,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
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

SparkStatus SparkStageKvBindingContinuity(SparkStageKvBinding *binding,const atomic_uint *lane_states,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	SparkStatus status;
	uint64_t held;
	if ( binding == 0 || lane_states == 0 || row_resident_slots == 0 || row_sequence_ids == 0 || row_positions == 0 || bound == 0 || sequence_ids == 0 || next_positions == 0 || row_count < active_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvBindingLock(binding,&held);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkStageKvBindingLoadContinuity(binding,active_count,row_resident_slots,row_sequence_ids,row_positions,bound,sequence_ids,next_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRowContinuity(binding,lane_states,row_count,active_count,row_resident_slots,row_sequence_ids,row_positions,bound,sequence_ids,next_positions);
	SparkStageKvBindingUnlock(binding,SPARK_STAGE_KV_LOCK_SITE_CONTINUITY,held);
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
		}
		else
			memset(binding->page_table_shadow + (uint64_t)resident * binding->pages_per_sequence,0xff,(size_t)binding->pages_per_sequence * sizeof(uint32_t));
	}
	return(result);
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
			atomic_load_explicit(&binding->lane_next_positions[resident],memory_order_acquire) != cache_lane->sequence_position )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkKvLaneTransactionsClaim(&binding->transactions,frame);
	if ( status == SPARK_STATUS_OK )
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
	uint32_t slot;
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
			result = SparkStageKvBindingApplyCompletion(binding,record->resident_slots,record->lane_count,record->status,record->extra_tokens,record->bound,record->sequence_ids,record->next_positions);
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

void SparkStageKvBindingStop(SparkStageKvBinding *binding)
{
	if ( binding == 0 || binding->sync_initialized == 0u )
		return;
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
	if ( binding == 0 || binding->mutex_initialized == 0u || binding->page_cache.snapshot == 0 )
		return;
	(void)pthread_mutex_lock(&binding->mutex);
	snapshot = binding->page_cache.snapshot;
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
