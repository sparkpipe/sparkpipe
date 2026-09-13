#include "dummy_client.h"

#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace
{
std::mutex SparkTestMooncakeMutex;
std::unordered_map<std::string,std::vector<uint8_t>> SparkTestMooncakeObjects;
uint32_t SparkTestMooncakeCapacityFailures = 1u;
}

namespace mooncake
{

DummyClient::DummyClient() = default;
DummyClient::~DummyClient() = default;

int DummyClient::setup_dummy(
	size_t mem_pool_size,
	size_t local_buffer_size,
	const std::string &server_address,
	const std::string &ipc_socket_path)
{
	(void)local_buffer_size;
	return mem_pool_size != 0u &&
		!server_address.empty() && !ipc_socket_path.empty() ? 0 : -1;
}

int DummyClient::health_check()
{
	return 0;
}

std::vector<int64_t> DummyClient::batch_get_into(
	const std::vector<std::string> &keys,
	const std::vector<void *> &buffers,
	const std::vector<size_t> &sizes)
{
	std::vector<int64_t> results(keys.size(),-1);
	std::lock_guard<std::mutex> lock(SparkTestMooncakeMutex);
	for (size_t index = 0u; index < keys.size(); ++index)
	{
		auto object = SparkTestMooncakeObjects.find(keys[index]);
		if (object == SparkTestMooncakeObjects.end() ||
			object->second.size() != sizes[index])
			continue;
		std::memcpy(buffers[index],object->second.data(),sizes[index]);
		results[index] = (int64_t)sizes[index];
	}
	return results;
}

std::vector<int> DummyClient::batch_put_from(
	const std::vector<std::string> &keys,
	const std::vector<void *> &buffers,
	const std::vector<size_t> &sizes,
	const ReplicateConfig &configuration)
{
	std::vector<int> results(keys.size(),-1);
	std::lock_guard<std::mutex> lock(SparkTestMooncakeMutex);
	if (SparkTestMooncakeCapacityFailures != 0u)
	{
		SparkTestMooncakeCapacityFailures -= 1u;
		return std::vector<int>(
			keys.size(),static_cast<int>(ErrorCode::NO_AVAILABLE_HANDLE));
	}
	if (configuration.replica_num != 1u ||
		configuration.preferred_segments.size() != 1u)
		return results;
	for (size_t index = 0u; index < keys.size(); ++index)
	{
		const uint8_t *source = static_cast<const uint8_t *>(buffers[index]);
		SparkTestMooncakeObjects[keys[index]] =
			std::vector<uint8_t>(source,source + sizes[index]);
		results[index] = 0;
	}
	return results;
}

std::string DummyClient::get_hostname() const
{
	return "spark-test";
}

uint64_t DummyClient::alloc_from_mem_pool(size_t bytes)
{
	return reinterpret_cast<uint64_t>(std::malloc(bytes));
}

int DummyClient::register_buffer(void *buffer,size_t bytes)
{
	return buffer != nullptr && bytes != 0u ? 0 : -1;
}

int DummyClient::unregister_buffer(void *buffer)
{
	return buffer != nullptr ? 0 : -1;
}

int DummyClient::tearDownAll()
{
	return 0;
}

ShmHelper *ShmHelper::getInstance()
{
	static ShmHelper helper;
	return &helper;
}

int ShmHelper::free(void *buffer)
{
	std::free(buffer);
	return 0;
}

}
