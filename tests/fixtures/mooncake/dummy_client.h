#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mooncake
{

struct ReplicateConfig
{
	uint32_t replica_num;
	std::vector<std::string> preferred_segments;
};

enum class ErrorCode : int32_t
{
	NO_AVAILABLE_HANDLE = -200
};

class DummyClient
{
public:
	DummyClient();
	~DummyClient();
	int setup_dummy(size_t,size_t,const std::string &,const std::string &);
	int health_check();
	std::vector<int64_t> batch_get_into(
		const std::vector<std::string> &,const std::vector<void *> &,
		const std::vector<size_t> &);
	std::vector<int> batch_put_from(
		const std::vector<std::string> &,const std::vector<void *> &,
		const std::vector<size_t> &,const ReplicateConfig &);
	std::string get_hostname() const;
	uint64_t alloc_from_mem_pool(size_t);
	int register_buffer(void *,size_t);
	int unregister_buffer(void *);
	int tearDownAll();
};

class ShmHelper
{
public:
	static ShmHelper *getInstance();
	int free(void *);
};

}
