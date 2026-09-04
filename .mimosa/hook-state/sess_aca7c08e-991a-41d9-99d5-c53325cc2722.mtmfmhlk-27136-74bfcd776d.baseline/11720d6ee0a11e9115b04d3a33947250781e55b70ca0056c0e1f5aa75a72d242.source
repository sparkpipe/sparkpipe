#include "sparkpipe/spark_kv_store.h"

#include "dummy_client.h"

#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace
{

enum SparkMooncakeRequestState : uint32_t
{
	SPARK_MOONCAKE_REQUEST_FREE = 0u,
	SPARK_MOONCAKE_REQUEST_QUEUED = 1u,
	SPARK_MOONCAKE_REQUEST_WORKING = 2u,
	SPARK_MOONCAKE_REQUEST_COMPLETE = 3u
};

struct SparkMooncakeRequest
{
	SparkKvStoreBatch batch;
	SparkStatus status;
	uint32_t completed_block_count;
	uint32_t state;
};

struct SparkMooncakeBuffer
{
	void *pointer;
	uint64_t bytes;
};

struct SparkMooncakeState
{
	SparkKvStoreConfiguration configuration;
	std::unique_ptr<mooncake::DummyClient> client;
	std::mutex mutex;
	std::condition_variable condition;
	std::vector<std::thread> workers;
	std::vector<SparkMooncakeBuffer> buffers;
	uint64_t allocated_buffer_bytes;
	SparkMooncakeRequest requests[SPARK_KV_STORE_MAX_INFLIGHT_BATCHES];
	bool stopping;
};

static SparkMooncakeRequest *SparkMooncakeFindRequest(
	SparkMooncakeState *state,
	uint64_t batch_id)
{
	uint32_t request_index;
	for (request_index = 0u;
		 request_index < state->configuration.maximum_inflight_batch_count;
		 ++request_index)
	{
		if (state->requests[request_index].state != SPARK_MOONCAKE_REQUEST_FREE &&
			state->requests[request_index].batch.batch_id == batch_id)
			return &state->requests[request_index];
	}
	return nullptr;
}

static SparkMooncakeRequest *SparkMooncakeFindQueuedRequest(
	SparkMooncakeState *state)
{
	SparkMooncakeRequest *selected;
	uint32_t request_index;
	selected = nullptr;
	for (request_index = 0u;
		 request_index < state->configuration.maximum_inflight_batch_count;
		 ++request_index)
	{
		SparkMooncakeRequest *request;
		request = &state->requests[request_index];
		if (request->state != SPARK_MOONCAKE_REQUEST_QUEUED)
			continue;
		if (selected == nullptr || request->batch.priority < selected->batch.priority ||
			(request->batch.priority == selected->batch.priority &&
			 request->batch.batch_id < selected->batch.batch_id))
			selected = request;
	}
	return selected;
}

static SparkStatus SparkMooncakeRunBatch(
	SparkMooncakeState *state,
	SparkMooncakeRequest *request)
{
	std::vector<std::string> keys;
	std::vector<void *> buffers;
	std::vector<size_t> sizes;
	uint32_t block_index;
	uint32_t operation;
	keys.reserve(request->batch.block_count);
	buffers.reserve(request->batch.block_count);
	sizes.reserve(request->batch.block_count);
	operation = request->batch.blocks[0u].operation;
	for (block_index = 0u; block_index < request->batch.block_count; ++block_index)
	{
		const SparkKvStoreBlock *block;
		block = &request->batch.blocks[block_index];
		if (block->operation != operation)
			return SPARK_STATUS_INVALID_ARGUMENT;
		keys.emplace_back(block->key,block->key_bytes);
		buffers.push_back(block->payload);
		sizes.push_back(block->payload_bytes);
	}
	if (operation == SPARK_KV_STORE_OPERATION_GET)
	{
		std::vector<int64_t> results;
		SparkStatus get_status;
		results = state->client->batch_get_into(keys,buffers,sizes);
		if (results.size() != request->batch.block_count)
			return SPARK_STATUS_IO_ERROR;
		get_status = SPARK_STATUS_OK;
		for (block_index = 0u; block_index < results.size(); ++block_index)
		{
			if (results[block_index] == (int64_t)sizes[block_index])
			{
				request->completed_block_count += 1u;
				continue;
			}
			if (results[block_index] < 0)
			{
				if (get_status == SPARK_STATUS_OK)
					get_status = SPARK_STATUS_NOT_FOUND;
			}
			else
				get_status = SPARK_STATUS_VALIDATION_FAILED;
		}
		return get_status;
	}
	mooncake::ReplicateConfig replicate_config;
	std::vector<std::string> pending_keys;
	std::vector<void *> pending_buffers;
	std::vector<size_t> pending_sizes;
	replicate_config.replica_num = 1u;
	replicate_config.preferred_segments.push_back(
		state->client->get_hostname());
	pending_keys = std::move(keys);
	pending_buffers = std::move(buffers);
	pending_sizes = std::move(sizes);
	for (uint32_t retry = 0u; retry <= 1200u; ++retry)
	{
		std::vector<int> results = state->client->batch_put_from(
			pending_keys,pending_buffers,pending_sizes,replicate_config);
		std::vector<std::string> retry_keys;
		std::vector<void *> retry_buffers;
		std::vector<size_t> retry_sizes;
		if (results.size() != pending_keys.size())
			return SPARK_STATUS_IO_ERROR;
		for (block_index = 0u; block_index < results.size(); ++block_index)
		{
			if (results[block_index] == 0)
			{
				request->completed_block_count += 1u;
				continue;
			}
			if (results[block_index] !=
				static_cast<int>(mooncake::ErrorCode::NO_AVAILABLE_HANDLE))
				return SPARK_STATUS_IO_ERROR;
			retry_keys.push_back(std::move(pending_keys[block_index]));
			retry_buffers.push_back(pending_buffers[block_index]);
			retry_sizes.push_back(pending_sizes[block_index]);
		}
		if (retry_keys.empty())
			return SPARK_STATUS_OK;
		if (retry == 1200u)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		pending_keys = std::move(retry_keys);
		pending_buffers = std::move(retry_buffers);
		pending_sizes = std::move(retry_sizes);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static void SparkMooncakeWorker(SparkMooncakeState *state)
{
	for (;;)
	{
		SparkMooncakeRequest *request;
		SparkStatus status;
		{
			std::unique_lock<std::mutex> lock(state->mutex);
			state->condition.wait(lock,[state]() {
				return state->stopping || SparkMooncakeFindQueuedRequest(state) != nullptr;
			});
			if (state->stopping)
				return;
			request = SparkMooncakeFindQueuedRequest(state);
			request->state = SPARK_MOONCAKE_REQUEST_WORKING;
		}
		status = SparkMooncakeRunBatch(state,request);
		{
			std::lock_guard<std::mutex> lock(state->mutex);
			request->status = status;
			request->state = SPARK_MOONCAKE_REQUEST_COMPLETE;
		}
	}
}

static SparkStatus SparkMooncakeInitialize(
	const SparkKvStoreConfiguration *configuration,
	void **store_state_out)
{
	std::unique_ptr<SparkMooncakeState> state;
	uint32_t worker_index;
	int status;
	if (SparkKvStoreValidateConfiguration(configuration) != SPARK_STATUS_OK ||
		configuration->ipc_socket_path == nullptr ||
		configuration->ipc_socket_path[0] == '\0' || store_state_out == nullptr)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state.reset(new (std::nothrow) SparkMooncakeState());
	if (!state)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->configuration = *configuration;
	state->stopping = false;
	state->allocated_buffer_bytes = 0u;
	std::memset(state->requests,0,sizeof(state->requests));
	state->client.reset(new (std::nothrow) mooncake::DummyClient());
	if (!state->client)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = state->client->setup_dummy(
		configuration->client_memory_pool_bytes,
		0u,
		configuration->service_address,
		configuration->ipc_socket_path);
	if (status != 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	try
	{
		state->workers.reserve(configuration->worker_count);
		for (worker_index = 0u; worker_index < configuration->worker_count;
			 ++worker_index)
			state->workers.emplace_back(SparkMooncakeWorker,state.get());
	}
	catch (...)
	{
		state->stopping = true;
		state->condition.notify_all();
		for (std::thread &worker : state->workers)
			if (worker.joinable())
				worker.join();
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	*store_state_out = state.release();
	return SPARK_STATUS_OK;
}

static void SparkMooncakeDestroy(void *store_state)
{
	std::unique_ptr<SparkMooncakeState> state(
		static_cast<SparkMooncakeState *>(store_state));
	if (!state)
		return;
	{
		std::lock_guard<std::mutex> lock(state->mutex);
		state->stopping = true;
	}
	state->condition.notify_all();
	for (std::thread &worker : state->workers)
		if (worker.joinable())
			worker.join();
	for (const SparkMooncakeBuffer &buffer : state->buffers)
	{
		(void)state->client->unregister_buffer(buffer.pointer);
		(void)mooncake::ShmHelper::getInstance()->free(buffer.pointer);
	}
	state->buffers.clear();
	state->client->tearDownAll();
}

static SparkStatus SparkMooncakeSubmit(
	void *store_state,
	const SparkKvStoreBatch *batch)
{
	SparkMooncakeState *state;
	uint32_t request_index;
	SparkStatus status;
	state = static_cast<SparkMooncakeState *>(store_state);
	if (state == nullptr)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkKvStoreValidateBatch(batch);
	if (status != SPARK_STATUS_OK)
		return status;
	if (batch->block_count > state->configuration.maximum_batch_block_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	{
		std::lock_guard<std::mutex> lock(state->mutex);
		if (SparkMooncakeFindRequest(state,batch->batch_id) != nullptr)
			return SPARK_STATUS_DUPLICATE;
		for (request_index = 0u;
			 request_index < state->configuration.maximum_inflight_batch_count;
			 ++request_index)
		{
			SparkMooncakeRequest *request;
			request = &state->requests[request_index];
			if (request->state != SPARK_MOONCAKE_REQUEST_FREE)
				continue;
			request->batch = *batch;
			request->status = SPARK_STATUS_BUSY;
			request->completed_block_count = 0u;
			request->state = SPARK_MOONCAKE_REQUEST_QUEUED;
			state->condition.notify_one();
			return SPARK_STATUS_OK;
		}
	}
	return SPARK_STATUS_BUSY;
}

static SparkStatus SparkMooncakePoll(
	void *store_state,
	uint64_t batch_id,
	SparkKvStoreCompletion *completion)
{
	SparkMooncakeState *state;
	SparkMooncakeRequest *request;
	state = static_cast<SparkMooncakeState *>(store_state);
	if (state == nullptr || batch_id == 0u || completion == nullptr)
		return SPARK_STATUS_INVALID_ARGUMENT;
	std::lock_guard<std::mutex> lock(state->mutex);
	request = SparkMooncakeFindRequest(state,batch_id);
	if (request == nullptr)
		return SPARK_STATUS_NOT_FOUND;
	if (request->state != SPARK_MOONCAKE_REQUEST_COMPLETE)
		return SPARK_STATUS_BUSY;
	std::memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_KV_STORE_ABI_VERSION;
	completion->descriptor_bytes = SPARK_KV_STORE_COMPLETION_BYTES;
	completion->status = request->status;
	completion->completed_block_count = request->completed_block_count;
	completion->batch_id = batch_id;
	std::memset(request,0,sizeof(*request));
	return SPARK_STATUS_OK;
}

static SparkStatus SparkMooncakeAllocateBuffer(
	void *store_state,
	uint64_t buffer_bytes,
	void **buffer_out)
{
	SparkMooncakeState *state;
	void *buffer;
	state = static_cast<SparkMooncakeState *>(store_state);
	if (state == nullptr || buffer_out == nullptr || buffer_bytes == 0u ||
		buffer_bytes > SIZE_MAX ||
		buffer_bytes > state->configuration.client_memory_pool_bytes)
		return SPARK_STATUS_INVALID_ARGUMENT;
	std::lock_guard<std::mutex> lock(state->mutex);
	if (state->allocated_buffer_bytes >
		state->configuration.client_memory_pool_bytes - buffer_bytes)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	buffer = reinterpret_cast<void *>(
		state->client->alloc_from_mem_pool((size_t)buffer_bytes));
	if (buffer == nullptr)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (state->client->register_buffer(buffer,(size_t)buffer_bytes) != 0)
	{
		(void)mooncake::ShmHelper::getInstance()->free(buffer);
		return SPARK_STATUS_IO_ERROR;
	}
	if (state->client->health_check() != 0)
	{
		(void)state->client->unregister_buffer(buffer);
		(void)mooncake::ShmHelper::getInstance()->free(buffer);
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	}
	try
	{
		state->buffers.push_back({buffer,buffer_bytes});
	}
	catch (...)
	{
		(void)state->client->unregister_buffer(buffer);
		(void)mooncake::ShmHelper::getInstance()->free(buffer);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	state->allocated_buffer_bytes += buffer_bytes;
	*buffer_out = buffer;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkMooncakeReleaseBuffer(
	void *store_state,
	void *buffer)
{
	SparkMooncakeState *state;
	size_t buffer_index;
	state = static_cast<SparkMooncakeState *>(store_state);
	if (state == nullptr || buffer == nullptr)
		return SPARK_STATUS_INVALID_ARGUMENT;
	std::lock_guard<std::mutex> lock(state->mutex);
	for (buffer_index = 0u; buffer_index < state->buffers.size(); ++buffer_index)
	{
		int unregister_status;
		int free_status;
		if (state->buffers[buffer_index].pointer != buffer)
			continue;
		unregister_status = state->client->unregister_buffer(buffer);
		free_status = mooncake::ShmHelper::getInstance()->free(buffer);
		state->allocated_buffer_bytes -= state->buffers[buffer_index].bytes;
		state->buffers.erase(state->buffers.begin() + buffer_index);
		return (unregister_status != 0 || free_status != 0)
			? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
	}
	return SPARK_STATUS_NOT_FOUND;
}

static const SparkKvStoreInterface SparkMooncakeInterface =
{
	SPARK_KV_STORE_ABI_VERSION,
	SPARK_KV_STORE_INTERFACE_BYTES,
	SPARK_KV_STORE_REQUIRED_CAPS,
	0u,
	SparkMooncakeInitialize,
	SparkMooncakeDestroy,
	SparkMooncakeSubmit,
	SparkMooncakePoll,
	SparkMooncakeAllocateBuffer,
	SparkMooncakeReleaseBuffer
};

}

extern "C" const SparkKvStoreInterface *SparkKvStoreGetInterface(void)
{
	return &SparkMooncakeInterface;
}
