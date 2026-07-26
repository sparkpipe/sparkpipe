#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"

#define SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_WORK 0x00000001u
#define SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_LAYER_MAJOR 0x00000002u

typedef struct SparkGlm52CudaResidentGateConfiguration
{
	const char *socket_path;
	uint32_t rank_index;
	uint32_t rank_is_set;
	uint32_t flags;
	uint32_t model_quantization_mode;
} SparkGlm52CudaResidentGateConfiguration;

static SparkStatus SparkGlm52CudaResidentGateReadFull(
	int32_t fd,
	void *buffer,
	uint32_t bytes)
{
	uint8_t *destination;
	uint32_t offset;

	destination = (uint8_t *)buffer;
	offset = 0u;
	while (offset < bytes)
	{
		ssize_t got;
		got = read(fd,destination + offset,(size_t)(bytes - offset));
		if (got > 0)
		{
			offset += (uint32_t)got;
			continue;
		}
		if (got < 0 && errno == EINTR)
			continue;
		return SPARK_STATUS_IO_ERROR;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentGateWriteFull(
	int32_t fd,
	const void *buffer,
	uint32_t bytes)
{
	const uint8_t *source;
	uint32_t offset;

	source = (const uint8_t *)buffer;
	offset = 0u;
	while (offset < bytes)
	{
		ssize_t written;
		written = write(fd,source + offset,(size_t)(bytes - offset));
		if (written > 0)
		{
			offset += (uint32_t)written;
			continue;
		}
		if (written < 0 && errno == EINTR)
			continue;
		return SPARK_STATUS_IO_ERROR;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CudaResidentGateValidateStats(
	const SparkGlm52CudaResidentIpcStats *stats,
	uint32_t expected_rank_index,
	uint32_t expected_quantization_mode,
	uint32_t flags)
{
	const uint32_t required_capability_flags =
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_DRIVER_RESIDENT |
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_BUILDER_RESIDENT |
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_TRANSPORT_RESIDENT |
		SPARK_GLM52_CUDA_RESIDENT_IPC_FLAG_CUDA_STATE_RESIDENT;
	uint32_t expected_moe_backend_kind;

	if (stats == 0 ||
		SparkGlm52Pp13RuntimeExpectedMoeBackendKind(
			expected_quantization_mode,
			&expected_moe_backend_kind) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (
		stats->descriptor_bytes != SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES ||
		stats->state != SPARK_GLM52_CUDA_RESIDENT_IPC_STATE_READY ||
		(stats->capability_flags & required_capability_flags) !=
			required_capability_flags ||
		stats->rank_index != expected_rank_index ||
		stats->max_active_sequence_count !=
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
		stats->logical_lane_capacity !=
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
		stats->execution_row_capacity !=
			SparkGlm52Pp13RuntimeExecutionRowCapacity(
				SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET) ||
		stats->model_quantization_mode != expected_quantization_mode ||
		stats->moe_backend_kind != expected_moe_backend_kind ||
		stats->moe_bound_layer_count == 0u ||
		stats->moe_bound_layer_count !=
			stats->moe_expected_layer_count ||
		SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
			stats->model_quantization_mode,
			stats->fp8_scaled_gemm_bound_plan_count,
			stats->fp8_scaled_gemm_expected_plan_count) != SPARK_STATUS_OK ||
		stats->kv_nvme_enabled == 0u ||
		(stats->kv_nvme_mode !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT &&
		 stats->kv_nvme_mode !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT) ||
		stats->kv_physical_block_capacity == 0u ||
		stats->kv_logical_block_capacity < stats->kv_physical_block_capacity ||
		stats->kv_resident_bytes_per_token == 0u ||
		stats->kv_resident_pool_bytes == 0u ||
		stats->kv_nvme_record_bytes == 0u ||
		stats->kv_nvme_capacity_bytes == 0u ||
		stats->kv_nvme_batch_block_capacity == 0u ||
		stats->kv_nvme_pending_store_count != 0u ||
		stats->kv_nvme_pending_load_count != 0u ||
		stats->asynchronous_failure_count != 0u ||
		stats->layer_major_failure_count != 0u ||
		stats->work_queue_error_count != 0u ||
		stats->rejected_count != 0u)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	if ((flags & SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_WORK) != 0u &&
		(stats->work_queue_depth != 0u ||
		 stats->builder_pending_work != 0u ||
		 stats->resident_driver_inflight != 0u ||
		 stats->work_queue_accepted_count == 0u ||
		 stats->work_queue_accepted_count != stats->work_queue_submit_count ||
		 stats->asynchronous_submit_count == 0u ||
		 stats->asynchronous_submit_count !=
			stats->asynchronous_completion_count))
		return SPARK_STATUS_VALIDATION_FAILED;
	if ((flags & SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_LAYER_MAJOR) != 0u &&
		(stats->layer_major_submit_count == 0u ||
		 stats->layer_major_submit_count !=
			stats->layer_major_completion_count ||
		 stats->last_layer_major_logical_lane_count == 0u ||
		 stats->last_layer_major_rows_per_lane <= 1u ||
		 stats->last_layer_major_execution_row_count !=
			stats->last_layer_major_logical_lane_count *
			stats->last_layer_major_rows_per_lane))
		return SPARK_STATUS_VALIDATION_FAILED;
	return SPARK_STATUS_OK;
}

static int32_t SparkGlm52CudaResidentGateParseU32(
	const char *text,
	uint32_t *value_out)
{
	char *end;
	unsigned long value;

	if (text == 0 || text[0] == '\0' || value_out == 0)
		return -1;
	end = 0;
	value = strtoul(text,&end,10);
	if (end == 0 || end[0] != '\0' || value > UINT32_MAX)
		return -1;
	*value_out = (uint32_t)value;
	return 0;
}

static int32_t SparkGlm52CudaResidentGateParseArguments(
	SparkGlm52CudaResidentGateConfiguration *configuration,
	int argc,
	char **argv)
{
	int index;

	memset(configuration,0,sizeof(*configuration));
	configuration->model_quantization_mode =
		SPARK_GLM52_PP13_RUNTIME_DEFAULT_QUANTIZATION_MODE;
	for (index = 1; index < argc; ++index)
	{
		if (strcmp(argv[index],"--socket") == 0 && index + 1 < argc)
			configuration->socket_path = argv[++index];
		else if (strcmp(argv[index],"--rank") == 0 && index + 1 < argc &&
			SparkGlm52CudaResidentGateParseU32(
				argv[++index],&configuration->rank_index) == 0)
			configuration->rank_is_set = 1u;
		else if (strcmp(argv[index],"--model-quantization") == 0)
		{
			if (index + 1 >= argc ||
				SparkGlm52Pp13RuntimeParseQuantizationMode(
					argv[index + 1],&configuration->model_quantization_mode) !=
					SPARK_STATUS_OK)
				return -1;
			index += 1;
		}
		else if (strcmp(argv[index],"--require-work") == 0)
			configuration->flags |=
				SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_WORK;
		else if (strcmp(argv[index],"--require-layer-major") == 0)
			configuration->flags |=
				SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_LAYER_MAJOR |
				SPARK_GLM52_CUDA_RESIDENT_GATE_REQUIRE_WORK;
		else
			return -1;
	}
	return configuration->socket_path != 0 && configuration->rank_is_set != 0u
		? 0 : -1;
}

static SparkStatus SparkGlm52CudaResidentGateQuery(
	const SparkGlm52CudaResidentGateConfiguration *configuration,
	SparkGlm52CudaResidentIpcStats *stats)
{
	SparkGlm52CudaResidentIpcHeader request_header;
	SparkGlm52CudaResidentIpcHeader response_header;
	SparkGlm52CudaResidentIpcQuery query;
	struct sockaddr_un address;
	SparkStatus status;
	int32_t fd;

	if (strlen(configuration->socket_path) >= sizeof(address.sun_path))
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	fd = socket(AF_UNIX,SOCK_STREAM,0);
	if (fd < 0)
		return SPARK_STATUS_IO_ERROR;
	memset(&address,0,sizeof(address));
	address.sun_family = AF_UNIX;
	snprintf(address.sun_path,sizeof(address.sun_path),"%s",
		configuration->socket_path);
	if (connect(fd,(const struct sockaddr *)&address,sizeof(address)) != 0)
	{
		close(fd);
		return SPARK_STATUS_ROUTE_NOT_FOUND;
	}
	memset(&query,0,sizeof(query));
	query.descriptor_bytes = SPARK_GLM52_CUDA_RESIDENT_IPC_QUERY_BYTES;
	SparkGlm52CudaResidentIpcInitializeHeader(
		&request_header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_QUERY,
		configuration->rank_index,1u,sizeof(query));
	status = SparkGlm52CudaResidentGateWriteFull(
		fd,&request_header,sizeof(request_header));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52CudaResidentGateWriteFull(fd,&query,sizeof(query));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52CudaResidentGateReadFull(
			fd,&response_header,sizeof(response_header));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52CudaResidentIpcValidateHeader(
			&response_header,SPARK_GLM52_CUDA_RESIDENT_IPC_KIND_STATS,
			SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES);
	if (status == SPARK_STATUS_OK &&
		response_header.payload_bytes != SPARK_GLM52_CUDA_RESIDENT_IPC_STATS_BYTES)
		status = SPARK_STATUS_ABI_MISMATCH;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52CudaResidentGateReadFull(fd,stats,sizeof(*stats));
	close(fd);
	return status;
}

int main(int argc,char **argv)
{
	SparkGlm52CudaResidentGateConfiguration configuration;
	SparkGlm52CudaResidentIpcStats stats;
	SparkStatus status;

	if (SparkGlm52CudaResidentGateParseArguments(
			&configuration,argc,argv) != 0)
	{
		fprintf(stderr,
			"usage: %s --socket path --rank n [--model-quantization fp8|nvfp4] [--require-work] "
			"[--require-layer-major]\n",argv[0]);
		return 2;
	}
	memset(&stats,0,sizeof(stats));
	status = SparkGlm52CudaResidentGateQuery(&configuration,&stats);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52CudaResidentGateValidateStats(
			&stats,configuration.rank_index,
			configuration.model_quantization_mode,configuration.flags);
	printf("rank=%u state=%u logical_lanes=%u execution_rows=%u "
		"pipeline_requests=%u quantization=%s moe=%u/%u fp8_scaled_gemm=%u/%u "
		"nvme_mode=%u nvme_record_bytes=%llu resident_pool_bytes=%llu "
		"kv_blocks=%llu/%u resident_blocks=%llu swapped_blocks=%llu "
		"queue=%u pending=%u inflight=%u control_generation=%llu "
		"work=%llu/%llu async=%llu/%llu/%llu layer_major=%llu/%llu/%llu "
		"gate=%s status=%u\n",
		configuration.rank_index,stats.state,stats.logical_lane_capacity,
		stats.execution_row_capacity,
		SPARK_GLM52_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY,
		SparkGlm52Pp13RuntimeQuantizationModeName(
			configuration.model_quantization_mode),
		stats.moe_bound_layer_count,stats.moe_expected_layer_count,
		stats.fp8_scaled_gemm_bound_plan_count,
		stats.fp8_scaled_gemm_expected_plan_count,stats.kv_nvme_mode,
		(unsigned long long)stats.kv_nvme_record_bytes,
		(unsigned long long)stats.kv_resident_pool_bytes,
		(unsigned long long)stats.kv_logical_block_count,
		stats.kv_logical_block_capacity,
		(unsigned long long)stats.kv_resident_block_count,
		(unsigned long long)stats.kv_swapped_block_count,
		stats.work_queue_depth,
		stats.builder_pending_work,
		stats.resident_driver_inflight,
		(unsigned long long)stats.control_generation,
		(unsigned long long)stats.work_queue_accepted_count,
		(unsigned long long)stats.work_queue_submit_count,
		(unsigned long long)stats.asynchronous_submit_count,
		(unsigned long long)stats.asynchronous_completion_count,
		(unsigned long long)stats.asynchronous_failure_count,
		(unsigned long long)stats.layer_major_submit_count,
		(unsigned long long)stats.layer_major_completion_count,
		(unsigned long long)stats.layer_major_failure_count,
		status == SPARK_STATUS_OK ? "pass" : "fail",(uint32_t)status);
	return status == SPARK_STATUS_OK ? 0 : 1;
}
