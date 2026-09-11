#include <cuda_runtime_api.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"

#define SPARK_HY4_SERVING_RUNG_DEGREE SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE
#define SPARK_HY4_SERVING_RUNG_RANK 0u
#define SPARK_HY4_SERVING_RUNG_IDENTIFIER UINT64_C(1002)
#define SPARK_HY4_SERVING_RUNG_PORT_BASE 60600u
#define SPARK_HY4_SERVING_RUNG_CONNECT_MS 60000u
#define SPARK_HY4_SERVING_RUNG_OPERATION_MS 60000u
#define SPARK_HY4_SERVING_RUNG_MODULE_PATH "tp_hidden_transport"
#define SPARK_HY4_SERVING_RUNG_MAX_ACTIVE 8u
#define SPARK_HY4_SERVING_RUNG_SLOTS 2u
#define SPARK_HY4_SERVING_RUNG_KV_BLOCKS 4096u
#define SPARK_HY4_SERVING_RUNG_PACK_BYTES UINT64_C(262144)
#define SPARK_HY4_SERVING_RUNG_EXPERT_POOL_BYTES UINT64_C(1048576)
#define SPARK_HY4_SERVING_RUNG_SPINE_BUDGET_BYTES (UINT64_C(4) * 1024u * 1024u)
#define SPARK_HY4_SERVING_RUNG_TIMEOUT_NS UINT64_C(60000000000)
#define SPARK_HY4_SERVING_RUNG_BREAK_COUNT 10u

static const char *const SparkHy4ServingRungBreakNames[
	SPARK_HY4_SERVING_RUNG_BREAK_COUNT] =
{
	"degree_over_max",
	"rank_outside_degree",
	"backend_not_hidden_transport",
	"mesh_addr_zero",
	"connect_timeout_zero",
	"operation_timeout_zero",
	"topology_degree_mismatch",
	"module_path_missing",
	"degree_one_dormant",
	"identifier_zero_dormant"
};

static const SparkStatus SparkHy4ServingRungBreakExpected[
	SPARK_HY4_SERVING_RUNG_BREAK_COUNT] =
{
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_INVALID_ARGUMENT,
	SPARK_STATUS_OK,
	SPARK_STATUS_OK
};

static FILE *rung_tsv;
static uint32_t rung_green_cells;
static uint32_t rung_total_cells;static uint64_t SparkHy4ServingRungNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) +
		(uint64_t)now.tv_nsec);
}

static void SparkHy4ServingRungCell(const char *cell,SparkStatus expected,
	SparkStatus got)
{
	uint32_t ok;
	ok = expected == got ? 1u : 0u;
	rung_green_cells += ok;
	rung_total_cells++;
	printf("SERVINGRUNG cell=%s expected=%d got=%d ok=%u\n",cell,
		(int)expected,(int)got,(unsigned)ok);
	if ( rung_tsv != 0 )
		fprintf(rung_tsv,"cell\t%s\texpected\t%d\tgot\t%d\tok\t%u\n",
			cell,(int)expected,(int)got,(unsigned)ok);
}

static void *SparkHy4ServingRungMeshBase(void)
{
	static void *mesh_base;
	if ( mesh_base == 0 )
	{
		mesh_base = malloc(SPARK_WEIGHTD_MESH_REGION_BYTES);
		if ( mesh_base != 0 )
			memset(mesh_base,0,SPARK_WEIGHTD_MESH_REGION_BYTES);
	}
	return(mesh_base);
}

static SparkStatus SparkHy4ServingRungWriteZeros(FILE *pack,
	uint64_t remaining)
{
	uint8_t buffer[4096];
	memset(buffer,0,sizeof(buffer));
	while ( remaining != 0u )
	{
		uint64_t chunk = remaining < sizeof(buffer) ? remaining :
			(uint64_t)sizeof(buffer);
		if ( fwrite(buffer,1u,(size_t)chunk,pack) != (size_t)chunk )
			return(SPARK_STATUS_IO_ERROR);
		remaining -= chunk;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkHy4ServingRungWriteFixture(const char *pack_path,
	char *sha256_hex_out)
{
	char manifest_path[2048];
	FILE *pack,*manifest;
	uint8_t payload[64],digest[16],record[48];
	uint32_t header[4] =
	{
		SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,
		SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,
		1u,
		0u
	};
	uint32_t layer = 0u,expert = 0u,kind = 0u,zero = 0u;
	uint64_t offset = 0u,range_bytes = (uint64_t)sizeof(payload);
	SparkCk128Context ck;
	SparkStatus status;
	int written;
	written = snprintf(manifest_path,sizeof(manifest_path),"%s.experts",
		pack_path);
	if ( written <= 0 || (size_t)written >= sizeof(manifest_path) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pack = fopen(pack_path,"wb");
	manifest = fopen(manifest_path,"wb");
	if ( pack == 0 || manifest == 0 )
	{
		if ( pack != 0 )
			(void)fclose(pack);
		if ( manifest != 0 )
			(void)fclose(manifest);
		return(SPARK_STATUS_IO_ERROR);
	}
	memset(payload,0x5a,sizeof(payload));
	SparkCk128Initialize(&ck);
	SparkCk128Update(&ck,payload,sizeof(payload));
	SparkCk128Finalize(&ck,digest);
	memset(record,0,sizeof(record));
	memcpy(record + 0u,&layer,4u);
	memcpy(record + 4u,&expert,4u);
	memcpy(record + 8u,&kind,4u);
	memcpy(record + 12u,&zero,4u);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&range_bytes,8u);
	memcpy(record + 32u,digest,16u);
	if ( fwrite(header,1u,sizeof(header),manifest) != sizeof(header) ||
		fwrite(record,1u,sizeof(record),manifest) != sizeof(record) ||
		fwrite(payload,1u,sizeof(payload),pack) != sizeof(payload) )
	{
		(void)fclose(pack);
		(void)fclose(manifest);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkHy4ServingRungWriteZeros(pack,
		SPARK_HY4_SERVING_RUNG_PACK_BYTES - sizeof(payload));
	if ( fclose(pack) != 0 || fclose(manifest) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkSha256File(pack_path,sha256_hex_out));
}

static void SparkHy4ServingRungFillNodeContext(
	SparkHy4ResidentDecodeStageNodeContext *context,uint64_t mesh_addr)
{
	memset(context,0,sizeof(*context));
	context->abi_version = 1u;
	context->max_active_sequence_count = SPARK_HY4_SERVING_RUNG_MAX_ACTIVE;
	context->pipeline_slot_count = SPARK_HY4_SERVING_RUNG_SLOTS;
	context->kv_cache_block_count = SPARK_HY4_SERVING_RUNG_KV_BLOCKS;
	context->tp_degree = SPARK_HY4_SERVING_RUNG_DEGREE;
	context->tp_rank = SPARK_HY4_SERVING_RUNG_RANK;
	context->tp_collective_backend_kind =
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	context->tp_collective_identifier = SPARK_HY4_SERVING_RUNG_IDENTIFIER;
	context->tp_connect_timeout_milli = SPARK_HY4_SERVING_RUNG_CONNECT_MS;
	context->tp_operation_timeout_milli =
		SPARK_HY4_SERVING_RUNG_OPERATION_MS;
	context->tp_collective_control_port_base =
		SPARK_HY4_SERVING_RUNG_PORT_BASE;
	context->tp_collective_mesh_addr = mesh_addr;
	context->tp_collective_topology.rank_count =
		SPARK_HY4_SERVING_RUNG_DEGREE;
	context->tp_collective_backend_module_path =
		SPARK_HY4_SERVING_RUNG_MODULE_PATH;
}

static void SparkHy4ServingRungBreakContext(uint32_t kind,
	SparkHy4ResidentDecodeStageNodeContext *context)
{
	switch (kind)
	{
		case 0u: context->tp_degree =
			SPARK_HY4_SERVING_RUNG_DEGREE + 1u; break;
		case 1u: context->tp_rank = SPARK_HY4_SERVING_RUNG_DEGREE; break;
		case 2u: context->tp_collective_backend_kind =
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL; break;
		case 3u: context->tp_collective_mesh_addr = 0u; break;
		case 4u: context->tp_connect_timeout_milli = 0u; break;
		case 5u: context->tp_operation_timeout_milli = 0u; break;
		case 6u: context->tp_collective_topology.rank_count =
			SPARK_HY4_SERVING_RUNG_DEGREE - 1u; break;
		case 7u: context->tp_collective_backend_module_path = 0; break;
		case 8u: context->tp_degree = 1u; break;
		case 9u: context->tp_collective_identifier = 0u; break;
		default: break;
	}
}

static void SparkHy4ServingRungFillHostServices(
	SparkFirmwareModuleHostServices *services,cudaStream_t stream,
	const SparkHy4ResidentDecodeStageNodeContext *context)
{
	memset(services,0,sizeof(*services));
	services->abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	services->descriptor_bytes = sizeof(*services);
	services->node_id = "spark2";
	services->node_target = "cuda.sm121.hy4.resident_decode_stage.fp8";
	services->node_context = (void *)context;
	services->execution_stream = (void *)stream;
}

static SparkStatus SparkHy4ServingRungModuleInitialize(
	const SparkHy4ResidentDecodeStageNodeContext *context,
	cudaStream_t stream,void **module_state)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = SPARK_HY4_MODEL_ID;
	configuration.model_revision = SPARK_HY4_MODEL_SOURCE_REVISION;
	configuration.stage_name = "hy4_resident_decode_stage";
	configuration.program_name = "resident_decode";
	SparkHy4ServingRungFillHostServices(&services,stream,context);
	*module_state = 0;
	return(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&services,module_state));
}

static void SparkHy4ServingRungBrokenMatrix(cudaStream_t stream,
	uint64_t stand_in_mesh_addr)
{
	SparkHy4ResidentDecodeStageNodeContext context;
	void *module_state;
	SparkStatus got;
	uint32_t kind;
	for (kind=0u; kind<SPARK_HY4_SERVING_RUNG_BREAK_COUNT; kind++)
	{
		SparkHy4ServingRungFillNodeContext(&context,stand_in_mesh_addr);
		SparkHy4ServingRungBreakContext(kind,&context);
		got = SparkHy4ServingRungModuleInitialize(&context,stream,
			&module_state);
		SparkHy4ServingRungCell(SparkHy4ServingRungBreakNames[kind],
			SparkHy4ServingRungBreakExpected[kind],got);
		if ( got == SPARK_STATUS_OK )
			SparkHy4ResidentDecodeStageDestroy(module_state);
	}
}

static SparkStatus SparkHy4ServingRungAttachWeightd(
	const char *pack_path,const char *sha256_hex,uint64_t *mesh_addr_out,
	uint32_t *mesh_ready_out)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPack *pack;
	SparkStatus status,status_release;
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&request,0,sizeof(request));
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = SPARK_HY4_SERVING_RUNG_PACK_BYTES;
	request.identity.topology = SPARK_HY4_SERVING_RUNG_DEGREE;
	(void)snprintf(request.identity.model,sizeof(request.identity.model),
		"%s","spark.hy4.tp16-serving-rung");
	(void)snprintf(request.identity.revision,
		sizeof(request.identity.revision),"%s",
		SPARK_HY4_MODEL_SOURCE_REVISION);
	memcpy(request.identity.pack_sha256,sha256_hex,65u);
	memcpy(request.pack_path,pack_path,strlen(pack_path) + 1u);
	request.expert_pool_bytes = SPARK_HY4_SERVING_RUNG_EXPERT_POOL_BYTES;
	status = SparkWeightdLazyPackCreate(
		getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,
		SPARK_HY4_SERVING_RUNG_SPINE_BUDGET_BYTES,
		SPARK_HY4_SERVING_RUNG_TIMEOUT_NS,&pack);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*mesh_ready_out = pack->attached.mesh_ready;
	*mesh_addr_out = pack->attached.mesh_send_buffer_addr;
	status_release = SparkWeightdLazyPackDestroy(pack);
	return(status_release == SPARK_STATUS_OK ? SPARK_STATUS_OK :
		status_release);
}

static void SparkHy4ServingRungAttachCell(cudaStream_t stream,
	uint64_t attach_mesh_addr,uint32_t attach_mesh_ready)
{
	SparkHy4ResidentDecodeStageNodeContext context;
	void *module_state;
	SparkStatus got;
	SparkHy4ServingRungFillNodeContext(&context,attach_mesh_addr);
	got = SparkHy4ServingRungModuleInitialize(&context,stream,&module_state);
	SparkHy4ServingRungCell("attach_mesh_addr_module_init",
		attach_mesh_ready != 0u ? SPARK_STATUS_OK :
		SPARK_STATUS_INVALID_ARGUMENT,got);
	if ( got == SPARK_STATUS_OK )
		SparkHy4ResidentDecodeStageDestroy(module_state);
}

static void SparkHy4ServingRungStandInCells(cudaStream_t stream,
	uint64_t stand_in_mesh_addr,const char *socket_path)
{
	SparkHy4ResidentDecodeStageNodeContext context;
	void *module_state;
	SparkStatus got;
	SparkHy4ServingRungFillNodeContext(&context,stand_in_mesh_addr);
	(void)unsetenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
	got = SparkHy4ServingRungModuleInitialize(&context,stream,&module_state);
	SparkHy4ServingRungCell("engine_unavailable_fail_closed",
		SPARK_STATUS_UNSUPPORTED,got);
	(void)setenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET,socket_path,1);
	got = SparkHy4ServingRungModuleInitialize(&context,stream,&module_state);
	SparkHy4ServingRungCell("engine_backed_module_init",SPARK_STATUS_OK,got);
	if ( got == SPARK_STATUS_OK )
		SparkHy4ResidentDecodeStageDestroy(module_state);
}

static int SparkHy4ServingRungOpenTsv(void)
{
	const char *tsv_path = getenv("HY4_TP16_SERVING_TSV");
	if ( tsv_path == 0 || tsv_path[0] == '\0' )
		return(0);
	rung_tsv = fopen(tsv_path,"w");
	if ( rung_tsv == 0 )
	{
		fprintf(stderr,"SERVINGRUNG tsv_open errno=%d\n",errno);
		return(1);
	}
	return(0);
}

static int SparkHy4ServingRungPaths(char *pack_path,size_t bytes)
{
	const char *workdir = getenv("HY4_TP16_SERVING_WORKDIR");
	int written;
	if ( workdir == 0 || workdir[0] == '\0' )
		workdir = ".";
	written = snprintf(pack_path,bytes,"%s/serving_rung.pack",workdir);
	return(written > 0 && (size_t)written < bytes ? 0 : 1);
}

int main(int argument_count,char **arguments)
{
	char pack_path[2048],sha256_hex[SPARK_SHA256_HEX_BYTES];
	const char *socket_path;
	uint64_t attach_mesh_addr,started_ns;
	uint32_t attach_mesh_ready;
	cudaStream_t stream;
	SparkStatus got;
	(void)arguments;
	if ( argument_count != 1 )
		return(2);
	started_ns = SparkHy4ServingRungNowNs();
	if ( SparkHy4ServingRungOpenTsv() != 0 )
		return(1);
	if ( SparkHy4ServingRungPaths(pack_path,sizeof(pack_path)) != 0 )
	{
		fprintf(stderr,"SERVINGRUNG paths\n");
		return(1);
	}
	if ( SparkHy4ServingRungMeshBase() == 0 )
	{
		fprintf(stderr,"SERVINGRUNG mesh_alloc\n");
		return(1);
	}
	if ( cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) !=
		cudaSuccess )
	{
		fprintf(stderr,"SERVINGRUNG stream\n");
		return(1);
	}
	if ( SparkHy4ServingRungWriteFixture(pack_path,sha256_hex) !=
		SPARK_STATUS_OK )
	{
		fprintf(stderr,"SERVINGRUNG fixture errno=%d\n",errno);
		return(1);
	}
	SparkHy4ServingRungBrokenMatrix(stream,(uint64_t)(uintptr_t)
		SparkHy4ServingRungMeshBase());
	socket_path = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
	if ( socket_path == 0 || socket_path[0] == '\0' )
	{
		fprintf(stderr,"SERVINGRUNG no_socket\n");
		return(1);
	}
	attach_mesh_addr = 0u;
	attach_mesh_ready = 0u;
	got = SparkHy4ServingRungAttachWeightd(pack_path,sha256_hex,
		&attach_mesh_addr,&attach_mesh_ready);
	if ( got != SPARK_STATUS_OK )
	{
		fprintf(stderr,"SERVINGRUNG attach status=%d\n",(int)got);
		return(1);
	}
	printf("SERVINGRUNG attach mesh_ready=%u mesh_addr=%llu\n",
		(unsigned)attach_mesh_ready,
		(unsigned long long)attach_mesh_addr);
	SparkHy4ServingRungAttachCell(stream,attach_mesh_addr,
		attach_mesh_ready);
	if ( attach_mesh_ready == 0u )
		SparkHy4ServingRungStandInCells(stream,(uint64_t)(uintptr_t)
			SparkHy4ServingRungMeshBase(),socket_path);
	(void)cudaStreamDestroy(stream);
	if ( rung_tsv != 0 )
		(void)fclose(rung_tsv);
	printf("SERVINGRUNG green_cells=%u/%u wall_s=%.1f\n",
		(unsigned)rung_green_cells,(unsigned)rung_total_cells,
		(double)(SparkHy4ServingRungNowNs() - started_ns) / 1e9);
	if ( rung_green_cells != rung_total_cells )
	{
		fprintf(stderr,"SERVINGRUNG RED\n");
		return(1);
	}
	printf("SERVINGRUNG GREEN\n");
	printf("TP16_SERVING_RUNG_DONE\n");
	return(0);
}
