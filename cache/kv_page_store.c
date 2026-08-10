#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "sparkpipe/spark_kv_page_store.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define SPARK_KV_PAGE_STORE_JOB_FREE 0u
#define SPARK_KV_PAGE_STORE_JOB_QUEUED 1u
#define SPARK_KV_PAGE_STORE_JOB_ACTIVE 2u
#define SPARK_KV_PAGE_STORE_JOB_COMPLETE 3u

typedef struct SparkKvPageStoreJob
{
	uint32_t state;
	uint32_t direction;
	uint32_t logical_page_index;
	uint32_t physical_page_index;
	uint64_t generation;
	uintptr_t key_device_address;
	uintptr_t value_device_address;
	uint64_t key_bytes;
	uint64_t value_bytes;
	SparkStatus terminal_status;
	uint32_t reserved0;
	SparkKvCachePrefetchBlock prefetch_block;
}
SparkKvPageStoreJob;

typedef struct SparkKvPageStoreWorker
{
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	SparkKvPageStore *store;
	SparkKvPageStoreJob *jobs;
	uint32_t mutex_initialized;
	uint32_t condition_initialized;
	uint32_t thread_started;
	uint32_t stop;
	uint32_t next_job_index;
}
SparkKvPageStoreWorker;

static uint64_t SparkKvPageStoreHashText(uint64_t hash,const char *text)
{
	while ( text != 0 && text[0] != '\0' )
	{
		hash ^= (uint8_t)text[0];
		hash *= UINT64_C(1099511628211);
		text++;
	}
	hash ^= UINT8_C(0xff);
	return(hash * UINT64_C(1099511628211));
}

SparkStatus SparkKvPageStoreBuildPath(
	char *path,
	uint32_t path_capacity,
	const char *backing_directory,
	const char *model_id,
	const char *model_revision,
	const char *node_id,
	uint32_t stage_index)
{
	uint64_t hash;
	int32_t written;
	if ( path == 0 || path_capacity == 0u || backing_directory == 0 ||
		backing_directory[0] == '\0' || model_id == 0 || model_id[0] == '\0' ||
		model_revision == 0 || model_revision[0] == '\0' ||
		node_id == 0 || node_id[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	hash = UINT64_C(1469598103934665603);
	hash = SparkKvPageStoreHashText(hash,model_id);
	hash = SparkKvPageStoreHashText(hash,model_revision);
	hash = SparkKvPageStoreHashText(hash,node_id);
	hash ^= stage_index;
	hash *= UINT64_C(1099511628211);
	written = snprintf(path,path_capacity,"%s/sparkpipe-kv-%016llx.bin",
		backing_directory,(unsigned long long)hash);
	return(written < 0 || (uint32_t)written >= path_capacity ?
		SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_OK);
}

static uint32_t SparkKvPageStoreConfigurationIsValid(
	const SparkKvPageStoreConfiguration *configuration)
{
	uint64_t required_bytes;
	if ( configuration == 0 ||
		configuration->abi_version != SPARK_KV_PAGE_STORE_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES ||
		(configuration->flags & ~SPARK_KV_PAGE_STORE_KNOWN_FLAGS) != 0u ||
		(configuration->flags == 0u ||
		 configuration->flags == SPARK_KV_PAGE_STORE_KNOWN_FLAGS) ||
		configuration->logical_page_capacity == 0u ||
		configuration->transfer_capacity == 0u ||
		configuration->transfer_capacity > configuration->logical_page_capacity ||
		configuration->reserved0 != 0u || configuration->page_bytes == 0u ||
		configuration->page_bytes > SIZE_MAX ||
		configuration->backing_path == 0 ||
		configuration->backing_path[0] == '\0' ||
		configuration->staging_address == 0 ||
		configuration->staging_bytes < configuration->page_bytes )
		return(0u);
	required_bytes = configuration->page_bytes *
		configuration->logical_page_capacity;
	if ( required_bytes / configuration->logical_page_capacity !=
		configuration->page_bytes || required_bytes > INT64_MAX )
		return(0u);
	return(configuration->maximum_backing_bytes == 0u ||
		required_bytes <= configuration->maximum_backing_bytes ? 1u : 0u);
}

static uint32_t SparkKvPageStoreIsValid(const SparkKvPageStore *store)
{
	return(store != 0 &&
		store->abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION &&
		store->descriptor_bytes == SPARK_KV_PAGE_STORE_BYTES &&
		store->logical_page_capacity != 0u && store->transfer_capacity != 0u &&
		store->reserved0 == 0u && store->reserved1 == 0u &&
		store->page_bytes != 0u &&
		store->file_descriptor >= 0 && store->staging_address != 0 &&
		store->staging_bytes >= store->page_bytes && store->worker_state != 0 &&
		store->generations != 0 && store->valid_pages != 0 ? 1u : 0u);
}

static SparkStatus SparkKvPageStoreOpen(
	SparkKvPageStore *store,
	const SparkKvPageStoreConfiguration *configuration)
{
	int32_t flags;
	flags = O_RDWR;
#ifdef O_CLOEXEC
	flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
	flags |= O_NOFOLLOW;
#endif
	if ( (configuration->flags &
		SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE) != 0u )
		flags |= O_CREAT | O_EXCL;
	else
	{
#ifdef O_TMPFILE
		flags |= O_TMPFILE;
#else
		return(SPARK_STATUS_UNSUPPORTED);
#endif
	}
	store->file_descriptor = open(configuration->backing_path,flags,0600);
	return(store->file_descriptor < 0 ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageStoreCopy(
	SparkKvPageStore *store,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	if ( bytes == 0u )
		return(SPARK_STATUS_OK);
	if ( device_address == 0u || host_address == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( store->copy_function != 0 )
		return(store->copy_function(store->copy_context,direction,
			device_address,host_address,bytes));
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		memcpy(host_address,(const void *)device_address,(size_t)bytes);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		memcpy((void *)device_address,host_address,(size_t)bytes);
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageStoreWriteExact(
	int32_t file_descriptor,
	uint64_t offset,
	const void *payload,
	uint64_t bytes)
{
	const uint8_t *cursor;
	ssize_t written;
	cursor = (const uint8_t *)payload;
	while ( bytes != 0u )
	{
		written = pwrite(file_descriptor,cursor,(size_t)bytes,(off_t)offset);
		if ( written < 0 && errno == EINTR )
			continue;
		if ( written <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		cursor += (uint64_t)written;
		offset += (uint64_t)written;
		bytes -= (uint64_t)written;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageStoreReadExact(
	int32_t file_descriptor,
	uint64_t offset,
	void *payload,
	uint64_t bytes)
{
	uint8_t *cursor;
	ssize_t received;
	cursor = (uint8_t *)payload;
	while ( bytes != 0u )
	{
		received = pread(file_descriptor,cursor,(size_t)bytes,(off_t)offset);
		if ( received < 0 && errno == EINTR )
			continue;
		if ( received <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		cursor += (uint64_t)received;
		offset += (uint64_t)received;
		bytes -= (uint64_t)received;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageStoreExecuteWrite(
	SparkKvPageStore *store,
	const SparkKvPageStoreJob *job)
{
	uint8_t *staging;
	uint64_t offset;
	SparkStatus status;
	staging = (uint8_t *)store->staging_address;
	status = SparkKvPageStoreCopy(store,job->direction,
		job->key_device_address,staging,job->key_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageStoreCopy(store,job->direction,
			job->value_device_address,staging + job->key_bytes,job->value_bytes);
	offset = (uint64_t)job->logical_page_index * store->page_bytes;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageStoreWriteExact(store->file_descriptor,offset,
			staging,store->page_bytes);
	return(status);
}

static SparkStatus SparkKvPageStoreExecuteRead(
	SparkKvPageStore *store,
	const SparkKvPageStoreJob *job)
{
	uint8_t *staging;
	uint64_t offset;
	SparkStatus status;
	staging = (uint8_t *)store->staging_address;
	offset = (uint64_t)job->logical_page_index * store->page_bytes;
	status = SparkKvPageStoreReadExact(store->file_descriptor,offset,staging,
		store->page_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageStoreCopy(store,job->direction,
			job->key_device_address,staging,job->key_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageStoreCopy(store,job->direction,
			job->value_device_address,staging + job->key_bytes,job->value_bytes);
	return(status);
}

static SparkStatus SparkKvPageStoreExecuteJob(
	SparkKvPageStore *store,
	const SparkKvPageStoreJob *job)
{
	if ( job->direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		return(SparkKvPageStoreExecuteWrite(store,job));
	if ( job->direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		return(SparkKvPageStoreExecuteRead(store,job));
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static uint32_t SparkKvPageStoreFindQueuedJob(
	SparkKvPageStoreWorker *worker)
{
	uint32_t index,job_index;
	for (index=0u; index<worker->store->transfer_capacity; index++)
	{
		job_index = (worker->next_job_index + index) %
			worker->store->transfer_capacity;
		if ( worker->jobs[job_index].state == SPARK_KV_PAGE_STORE_JOB_QUEUED )
			return(job_index);
	}
	return(worker->store->transfer_capacity);
}

static void SparkKvPageStoreRecordJob(
	SparkKvPageStoreWorker *worker,
	SparkKvPageStoreJob *job,
	SparkStatus status)
{
	SparkKvPageStore *store;
	store = worker->store;
	job->terminal_status = status;
	if ( status == SPARK_STATUS_OK &&
		job->direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
	{
		store->generations[job->logical_page_index] = job->generation;
		store->valid_pages[job->logical_page_index] = 1u;
		store->write_count++;
		store->write_bytes += store->page_bytes;
	}
	else if ( status == SPARK_STATUS_OK )
	{
		store->read_count++;
		store->read_bytes += store->page_bytes;
	}
	job->state = SPARK_KV_PAGE_STORE_JOB_COMPLETE;
}

static void *SparkKvPageStoreWorkerMain(void *context)
{
	SparkKvPageStoreWorker *worker;
	SparkKvPageStoreJob *job;
	SparkStatus status;
	uint32_t job_index;
	worker = (SparkKvPageStoreWorker *)context;
	for (;;)
	{
		(void)pthread_mutex_lock(&worker->mutex);
		job_index = SparkKvPageStoreFindQueuedJob(worker);
		while ( worker->stop == 0u &&
			job_index == worker->store->transfer_capacity )
		{
			(void)pthread_cond_wait(&worker->condition,&worker->mutex);
			job_index = SparkKvPageStoreFindQueuedJob(worker);
		}
		if ( worker->stop != 0u )
		{
			(void)pthread_mutex_unlock(&worker->mutex);
			break;
		}
		job = &worker->jobs[job_index];
		job->state = SPARK_KV_PAGE_STORE_JOB_ACTIVE;
		worker->next_job_index = (job_index + 1u) % worker->store->transfer_capacity;
		(void)pthread_mutex_unlock(&worker->mutex);
		status = SparkKvPageStoreExecuteJob(worker->store,job);
		(void)pthread_mutex_lock(&worker->mutex);
		SparkKvPageStoreRecordJob(worker,job,status);
		(void)pthread_cond_broadcast(&worker->condition);
		(void)pthread_mutex_unlock(&worker->mutex);
	}
	return(0);
}

static void SparkKvPageStoreWorkerDestroy(SparkKvPageStore *store)
{
	SparkKvPageStoreWorker *worker;
	worker = store != 0 ? (SparkKvPageStoreWorker *)store->worker_state : 0;
	if ( worker == 0 )
		return;
	if ( worker->thread_started != 0u )
	{
		(void)pthread_mutex_lock(&worker->mutex);
		worker->stop = 1u;
		(void)pthread_cond_broadcast(&worker->condition);
		(void)pthread_mutex_unlock(&worker->mutex);
		(void)pthread_join(worker->thread,0);
	}
	if ( worker->condition_initialized != 0u )
		(void)pthread_cond_destroy(&worker->condition);
	if ( worker->mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&worker->mutex);
	free(worker->jobs);
	free(worker);
	store->worker_state = 0;
}

static SparkStatus SparkKvPageStoreWorkerInitialize(SparkKvPageStore *store)
{
	SparkKvPageStoreWorker *worker;
	worker = (SparkKvPageStoreWorker *)calloc(1u,sizeof(*worker));
	if ( worker == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	store->worker_state = worker;
	worker->store = store;
	worker->jobs = (SparkKvPageStoreJob *)calloc(store->transfer_capacity,
		sizeof(worker->jobs[0]));
	if ( worker->jobs == 0 || pthread_mutex_init(&worker->mutex,0) != 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	worker->mutex_initialized = 1u;
	if ( pthread_cond_init(&worker->condition,0) != 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	worker->condition_initialized = 1u;
	if ( pthread_create(&worker->thread,0,SparkKvPageStoreWorkerMain,worker) != 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	worker->thread_started = 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageStoreInitialize(
	SparkKvPageStore *store,
	const SparkKvPageStoreConfiguration *configuration)
{
	SparkStatus status;
	if ( store == 0 || SparkKvPageStoreConfigurationIsValid(configuration) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(store,0,sizeof(*store));
	store->file_descriptor = -1;
	store->abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	store->descriptor_bytes = SPARK_KV_PAGE_STORE_BYTES;
	store->flags = configuration->flags;
	store->logical_page_capacity = configuration->logical_page_capacity;
	store->transfer_capacity = configuration->transfer_capacity;
	store->page_bytes = configuration->page_bytes;
	store->maximum_backing_bytes = configuration->maximum_backing_bytes;
	store->staging_address = configuration->staging_address;
	store->staging_bytes = configuration->staging_bytes;
	store->copy_function = configuration->copy_function;
	store->copy_context = configuration->copy_context;
	store->generations = (uint64_t *)calloc(store->logical_page_capacity,
		sizeof(store->generations[0]));
	store->valid_pages = (uint8_t *)calloc(store->logical_page_capacity,
		sizeof(store->valid_pages[0]));
	status = store->generations != 0 && store->valid_pages != 0 ?
		SparkKvPageStoreOpen(store,configuration) :
		SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageStoreWorkerInitialize(store);
	if ( status != SPARK_STATUS_OK )
		SparkKvPageStoreDestroy(store);
	return(status);
}

void SparkKvPageStoreDestroy(SparkKvPageStore *store)
{
	if ( store == 0 )
		return;
	SparkKvPageStoreWorkerDestroy(store);
	if ( store->file_descriptor >= 0 )
		(void)close(store->file_descriptor);
	free(store->valid_pages);
	free(store->generations);
	memset(store,0,sizeof(*store));
	store->file_descriptor = -1;
}

static SparkKvPageStoreJob *SparkKvPageStoreFindJob(
	SparkKvPageStoreWorker *worker,
	uint32_t direction,
	uint32_t logical_page_index,
	uint64_t generation)
{
	SparkKvPageStoreJob *job;
	uint32_t index;
	for (index=0u; index<worker->store->transfer_capacity; index++)
	{
		job = &worker->jobs[index];
		if ( job->state != SPARK_KV_PAGE_STORE_JOB_FREE &&
			job->direction == direction &&
			job->logical_page_index == logical_page_index &&
			job->generation == generation )
			return(job);
	}
	return(0);
}

static SparkKvPageStoreJob *SparkKvPageStoreFindFreeJob(
	SparkKvPageStoreWorker *worker)
{
	uint32_t index;
	for (index=0u; index<worker->store->transfer_capacity; index++)
		if ( worker->jobs[index].state == SPARK_KV_PAGE_STORE_JOB_FREE )
			return(&worker->jobs[index]);
	return(0);
}

static void SparkKvPageStoreQueueJob(
	SparkKvPageStoreWorker *worker,
	SparkKvPageStoreJob *job)
{
	job->state = SPARK_KV_PAGE_STORE_JOB_QUEUED;
	job->terminal_status = SPARK_STATUS_BUSY;
	(void)pthread_cond_signal(&worker->condition);
}

SparkStatus SparkKvPageStoreWriteback(
	void *context,
	uint32_t logical_page_index,
	uint32_t physical_page_index,
	uint64_t generation,
	uintptr_t key_device_address,
	uint64_t key_bytes,
	uintptr_t value_device_address,
	uint64_t value_bytes)
{
	SparkKvPageStore *store;
	SparkKvPageStoreWorker *worker;
	SparkKvPageStoreJob *job;
	SparkStatus status;
	store = (SparkKvPageStore *)context;
	if ( SparkKvPageStoreIsValid(store) == 0u ||
		logical_page_index >= store->logical_page_capacity || generation == 0u ||
		key_bytes > store->page_bytes ||
		value_bytes != store->page_bytes - key_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	worker = (SparkKvPageStoreWorker *)store->worker_state;
	if ( pthread_mutex_lock(&worker->mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	job = SparkKvPageStoreFindJob(worker,
		SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,logical_page_index,generation);
	if ( job != 0 )
	{
		status = job->state == SPARK_KV_PAGE_STORE_JOB_COMPLETE ?
			job->terminal_status : SPARK_STATUS_BUSY;
		if ( job->state == SPARK_KV_PAGE_STORE_JOB_COMPLETE )
			memset(job,0,sizeof(*job));
		(void)pthread_mutex_unlock(&worker->mutex);
		return(status);
	}
	job = SparkKvPageStoreFindFreeJob(worker);
	if ( job == 0 )
	{
		(void)pthread_mutex_unlock(&worker->mutex);
		return(SPARK_STATUS_BUSY);
	}
	job->direction = SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST;
	job->logical_page_index = logical_page_index;
	job->physical_page_index = physical_page_index;
	job->generation = generation;
	job->key_device_address = key_device_address;
	job->value_device_address = value_device_address;
	job->key_bytes = key_bytes;
	job->value_bytes = value_bytes;
	SparkKvPageStoreQueueJob(worker,job);
	(void)pthread_mutex_unlock(&worker->mutex);
	return(SPARK_STATUS_BUSY);
}

static void SparkKvPageStoreBuildCompletedPlan(
	const SparkKvPageStoreJob *job,
	SparkKvCachePrefetchPlan *plan)
{
	memset(plan,0,sizeof(*plan));
	plan->abi_version = SPARK_KV_CACHE_ABI_VERSION;
	plan->descriptor_bytes = SPARK_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES;
	plan->lane_count = 1u;
	plan->requested_logical_block_count = 1u;
	plan->prefetch_block_count = 1u;
	plan->reserved_block_count = 1u;
	plan->lane_block_counts[0u] = 1u;
	plan->blocks[0u] = job->prefetch_block;
}

static SparkStatus SparkKvPageStoreFinishPrefetch(
	SparkKvPageStore *store,
	SparkKvCacheArena *arena,
	uint32_t logical_page_index,
	uint64_t generation)
{
	SparkKvPageStoreWorker *worker;
	SparkKvPageStoreJob *job,completed;
	SparkKvCachePrefetchPlan plan;
	SparkStatus status;
	worker = (SparkKvPageStoreWorker *)store->worker_state;
	if ( pthread_mutex_lock(&worker->mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	job = SparkKvPageStoreFindJob(worker,
		SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,logical_page_index,generation);
	if ( job == 0 || job->state != SPARK_KV_PAGE_STORE_JOB_COMPLETE )
	{
		status = job == 0 ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_BUSY;
		(void)pthread_mutex_unlock(&worker->mutex);
		return(status);
	}
	completed = *job;
	memset(job,0,sizeof(*job));
	(void)pthread_mutex_unlock(&worker->mutex);
	SparkKvPageStoreBuildCompletedPlan(&completed,&plan);
	status = completed.terminal_status;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaMarkPrefetchPlanResident(arena,&plan);
	if ( status != SPARK_STATUS_OK )
		(void)SparkKvCacheArenaCancelPrefetchPlan(arena,&plan);
	return(status);
}

static SparkStatus SparkKvPageStoreStartPrefetch(
	SparkKvPageStore *store,
	SparkKvCacheArena *arena,
	uint32_t logical_page_index)
{
	SparkKvPageStoreWorker *worker;
	SparkKvPageStoreJob *job;
	SparkKvCachePrefetchPlan plan;
	SparkStatus status;
	status = SparkKvCacheArenaBuildPrefetchPlan(arena,&logical_page_index,1u,
		1u,&plan);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( plan.prefetch_block_count != 1u )
	{
		(void)SparkKvCacheArenaCancelPrefetchPlan(arena,&plan);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	worker = (SparkKvPageStoreWorker *)store->worker_state;
	if ( pthread_mutex_lock(&worker->mutex) != 0 )
	{
		(void)SparkKvCacheArenaCancelPrefetchPlan(arena,&plan);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	job = SparkKvPageStoreFindFreeJob(worker);
	if ( job != 0 )
	{
		job->direction = SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE;
		job->logical_page_index = logical_page_index;
		job->physical_page_index = plan.blocks[0u].resident_slot_index;
		job->generation = plan.blocks[0u].generation;
		job->key_device_address = plan.blocks[0u].key_device_address;
		job->value_device_address = plan.blocks[0u].value_device_address;
		job->key_bytes = arena->key_block_stride_bytes;
		job->value_bytes = arena->value_block_stride_bytes;
		job->prefetch_block = plan.blocks[0u];
		SparkKvPageStoreQueueJob(worker,job);
	}
	(void)pthread_mutex_unlock(&worker->mutex);
	if ( job == 0 )
		(void)SparkKvCacheArenaCancelPrefetchPlan(arena,&plan);
	return(SPARK_STATUS_BUSY);
}

SparkStatus SparkKvPageStorePrefetch(
	SparkKvPageStore *store,
	SparkKvCacheArena *arena,
	uint32_t logical_page_index)
{
	SparkKvCacheBlockView view;
	SparkKvPageStoreWorker *worker;
	SparkKvPageStoreJob *job;
	SparkStatus status;
	uint32_t backing_valid;
	if ( SparkKvPageStoreIsValid(store) == 0u || arena == 0 ||
		logical_page_index >= store->logical_page_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvCacheArenaResolveBlock(arena,logical_page_index,&view);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
		return(SPARK_STATUS_OK);
	worker = (SparkKvPageStoreWorker *)store->worker_state;
	if ( pthread_mutex_lock(&worker->mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	backing_valid = (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u &&
		store->valid_pages[logical_page_index] != 0u &&
		store->generations[logical_page_index] == view.generation ? 1u : 0u;
	job = SparkKvPageStoreFindJob(worker,
		SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,logical_page_index,view.generation);
	(void)pthread_mutex_unlock(&worker->mutex);
	if ( backing_valid == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	if ( job != 0 )
		return(SparkKvPageStoreFinishPrefetch(store,arena,logical_page_index,
			view.generation));
	return(SparkKvPageStoreStartPrefetch(store,arena,logical_page_index));
}
