#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#include "sparkpipe/spark_dsv41_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_error_site.h"

#include "../source/spark_dsv41_flash_stagepack_format.h"

extern SparkStatus SparkDsv41FlashResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state);
extern void SparkDsv41FlashResidentDecodeStageDestroy(void *module_state);
extern SparkStatus SparkDsv41FlashResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame);

#define DSV41_FLASH_PROBE_MODEL_REVISION "dba1be0a40aa45a94ad051997016db3960a90277"
#define DSV41_FLASH_PROBE_TP_DEGREE 8u
#define DSV41_FLASH_PROBE_EXPERT_WEIGHT_CODEC 7u

typedef struct SparkDsv41FlashProbePackCensus
{
	uint64_t spine_bytes;
	uint64_t expert_bytes;
	uint64_t layer_seen_bits[SPARK_DSV41_FLASH_MODEL_LAYER_COUNT];
	uint64_t global_seen_bits;
	uint32_t entries;
} SparkDsv41FlashProbePackCensus;

static double SparkDsv41FlashProbeNowMilliseconds(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0);
}

static SparkStatus SparkDsv41FlashProbeCudaCheck(cudaError_t error,const char *site)
{
	if ( error != cudaSuccess )
	{
		(void)fprintf(stderr,"probe cuda_error site=%s code=%d text=%s\n",site,(int)error,cudaGetErrorString(error));
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashProbeCensus(const char *path,SparkDsv41FlashProbePackCensus *census)
{
	SparkDsv41FlashStagePackHeader header;
	SparkDsv41FlashStagePackEntry *entries;
	FILE *file;
	uint64_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( fread(&header,sizeof(header),1u,file) != 1u )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	entries = (SparkDsv41FlashStagePackEntry *)malloc((size_t)header.tensor_count * sizeof(*entries));
	if ( entries == 0 )
	{
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( fseeko(file,(off_t)header.directory_offset,SEEK_SET) != 0 ||
		fread(entries,sizeof(*entries),(size_t)header.tensor_count,file) != header.tensor_count )
	{
		free(entries);
		(void)fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	(void)fclose(file);
	census->spine_bytes = 0u;
	census->expert_bytes = 0u;
	census->entries = header.tensor_count;
	census->global_seen_bits = 0u;
	memset(census->layer_seen_bits,0,sizeof(census->layer_seen_bits));
	for (index=0u; index<header.tensor_count; index++)
	{
		uint64_t bytes = entries[index].payload_bytes + entries[index].scale_bytes;
		if ( SparkDsv41FlashStagePackKindIsGlobal(entries[index].tensor_kind) != 0u )
		{
			census->global_seen_bits |= UINT64_C(1) << entries[index].tensor_kind;
			census->spine_bytes += bytes;
		}
		else if ( SparkDsv41FlashStagePackKindIsExpert(entries[index].tensor_kind) != 0u )
			census->expert_bytes += bytes;
		else
			census->spine_bytes += bytes;
		if ( entries[index].layer_index != SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER &&
			entries[index].layer_index < SPARK_DSV41_FLASH_MODEL_LAYER_COUNT )
			census->layer_seen_bits[entries[index].layer_index] |= UINT64_C(1) << entries[index].tensor_kind;
	}
	free(entries);
	return(SPARK_STATUS_OK);
}

static void SparkDsv41FlashProbeKindList(uint64_t bits,char *text,uint32_t bytes)
{
	uint32_t kind,used;
	used = 0u;
	text[0] = '\0';
	for (kind=0u; kind<SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT && used<bytes; kind++)
	{
		if ( (bits & (UINT64_C(1) << kind)) == 0u )
			continue;
		used += (uint32_t)snprintf(text + used,bytes - used,"%s%u",used == 0u ? "" : ",",kind);
	}
}

static void SparkDsv41FlashProbeInventoryDiagnose(
	const SparkDsv41FlashProbePackCensus *census,
	uint32_t tp_rank)
{
	SparkDsv41FlashStagePackTensorShape shape;
	uint64_t expected_bits,seen_bits;
	char seen_text[512],expected_text[512];
	uint32_t kind,layer;
	for (layer=0u; layer<SPARK_DSV41_FLASH_MODEL_LAYER_COUNT; layer++)
	{
		expected_bits = 0u;
		for (kind=0u; kind<SPARK_DSV41_FLASH_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		{
			if ( SparkDsv41FlashStagePackKindIsGlobal(kind) != 0u )
				continue;
			if ( SparkDsv41FlashStagePackExpectedShape(kind,DSV41_FLASH_PROBE_EXPERT_WEIGHT_CODEC,DSV41_FLASH_PROBE_TP_DEGREE,&shape) == 0u )
				continue;
			if ( SparkDsv41FlashStagePackKindInLayer(kind,layer) != 0u )
				expected_bits |= UINT64_C(1) << kind;
		}
		seen_bits = census->layer_seen_bits[layer];
		if ( seen_bits != expected_bits )
		{
			SparkDsv41FlashProbeKindList(expected_bits & ~seen_bits,expected_text,sizeof(expected_text));
			SparkDsv41FlashProbeKindList(seen_bits & ~expected_bits,seen_text,sizeof(seen_text));
			(void)printf("probe diag rank=%u layer=%u expected=0x%llx seen=0x%llx missing_kinds=%s extra_kinds=%s\n",
				tp_rank,layer,(unsigned long long)expected_bits,(unsigned long long)seen_bits,
				expected_text,seen_text);
		}
	}
	if ( census->global_seen_bits != ((UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING) |
		(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM) |
		(UINT64_C(1) << SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD)) )
		(void)printf("probe diag rank=%u global_seen=0x%llx\n",tp_rank,(unsigned long long)census->global_seen_bits);
}

static SparkStatus SparkDsv41FlashProbeNodeContextPrepare(
	const char *pack_path,
	uint32_t tp_rank,
	SparkDsv41FlashResidentDecodeStageNodeContext *context)
{
	context->abi_version = SPARK_DSV41_FLASH_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = (uint32_t)sizeof(*context);
	context->stage_count = 1u;
	context->stage_index = 0u;
	context->first_layer_index = 0u;
	context->layer_count = SPARK_DSV41_FLASH_MODEL_LAYER_COUNT;
	context->expert_weight_codec = DSV41_FLASH_PROBE_EXPERT_WEIGHT_CODEC;
	context->resident_sequence_capacity = 1u;
	context->pipeline_slot_count = 1u;
	context->max_sequence_positions = 1024u;
	context->execution_row_capacity = 1u;
	context->tp_degree = DSV41_FLASH_PROBE_TP_DEGREE;
	context->tp_rank = tp_rank;
	context->flags = 0u;
	context->stage_pack_path = pack_path;
	context->model_revision = DSV41_FLASH_PROBE_MODEL_REVISION;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashProbeConfigurationPrepare(
	const SparkDsv41FlashResidentDecodeStageNodeContext *context,
	SparkFirmwareModuleConfiguration *configuration)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration->descriptor_bytes = (uint32_t)sizeof(*configuration);
	configuration->model_id = "dsv41_flash_attach_probe";
	configuration->model_revision = context->model_revision;
	configuration->stage_name = "dsv41_flash";
	configuration->program_name = "resident_decode";
	configuration->operation_name = "initialize";
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashProbeServicesPrepare(
	const SparkDsv41FlashResidentDecodeStageNodeContext *context,
	cudaStream_t stream,
	SparkFirmwareModuleHostServices *services)
{
	memset(services,0,sizeof(*services));
	services->abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	services->descriptor_bytes = (uint32_t)sizeof(*services);
	services->node_context = (void *)context;
	services->execution_stream = (void *)stream;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashProbeEventsSynchronize(
	cudaEvent_t begin,cudaEvent_t end,cudaStream_t stream,float *device_milliseconds)
{
	if ( cudaEventRecord(begin,stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( cudaEventRecord(end,stream) != cudaSuccess ||
		cudaEventSynchronize(end) != cudaSuccess ||
		cudaEventElapsedTime(device_milliseconds,begin,end) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashProbeFramePrepare(
	cudaStream_t stream,
	SparkModelDriverFrame *frame,
	SparkModelDriverBuffer *buffers,
	uint32_t *input_token,
	uint32_t *output_token)
{
	*input_token = 123u;
	*output_token = 0xdeadbeefu;
	memset(frame,0,sizeof(*frame));
	frame->active_slot_count = 1u;
	frame->new_token_count = 1u;
	frame->sequence_position = 0u;
	frame->execution_stream = (void *)stream;
	memset(buffers,0,sizeof(*buffers) * 2u);
	buffers[0].address = input_token;
	buffers[0].bytes = sizeof(*input_token);
	buffers[1].address = output_token;
	buffers[1].bytes = sizeof(*output_token);
	frame->buffers = buffers;
	frame->buffer_count = 2u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv41FlashProbeArgumentsParse(
	char **argv,
	uint32_t *tp_rank,
	uint32_t *expect_ok,
	uint32_t *do_execute)
{
	if ( sscanf(argv[5],"%u:%u",expect_ok,do_execute) != 2 || *expect_ok > 1u || *do_execute > 1u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*tp_rank = (uint32_t)strtoul(argv[2],0,10);
	if ( setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES",argv[3],1) != 0 ||
		setenv("SPARK_WEIGHTD_SPINE_BUDGET_BYTES",argv[4],1) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

int main(int argc,char **argv)
{
	SparkDsv41FlashResidentDecodeStageNodeContext context;
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	SparkDsv41FlashProbePackCensus census;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	cudaEvent_t begin,end;
	cudaStream_t stream;
	uint32_t input_token,output_token;
	uint32_t expect_ok,do_execute,tp_rank;
	float device_milliseconds;
	void *state;
	double wall_begin,wall_end;
	SparkStatus status;
	if ( argc != 6 )
	{
		(void)fprintf(stderr,"usage: %s PACK TP_RANK EXPERT_POOL_BYTES SPINE_BUDGET_BYTES EXPECT_OK:EXECUTE\n",argv[0]);
		return(2);
	}
	expect_ok = 0u;
	do_execute = 0u;
	tp_rank = 0u;
	status = SparkDsv41FlashProbeArgumentsParse(argv,&tp_rank,&expect_ok,&do_execute);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeCensus(argv[1],&census);
	if ( status != SPARK_STATUS_OK )
	{
		(void)fprintf(stderr,"probe preflight_status=%d\n",(int)status);
		return(1);
	}
	SparkDsv41FlashProbeInventoryDiagnose(&census,tp_rank);
	status = SparkDsv41FlashProbeCudaCheck(cudaFree(0),"context_ensure");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeCudaCheck(cudaStreamCreate(&stream),"stream_create");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeCudaCheck(cudaEventCreate(&begin),"event_create");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeCudaCheck(cudaEventCreate(&end),"event_create");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeNodeContextPrepare(argv[1],tp_rank,&context);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeConfigurationPrepare(&context,&configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv41FlashProbeServicesPrepare(&context,stream,&services);
	if ( status != SPARK_STATUS_OK )
		return(1);
	state = 0;
	wall_begin = SparkDsv41FlashProbeNowMilliseconds();
	status = SparkDsv41FlashProbeEventsSynchronize(begin,end,stream,&device_milliseconds);
	if ( status == SPARK_STATUS_OK )
	{
		SparkStatus init_status = SparkDsv41FlashResidentDecodeStageInitialize(&configuration,&services,&state);
		status = SparkDsv41FlashProbeEventsSynchronize(begin,end,stream,&device_milliseconds);
		wall_end = SparkDsv41FlashProbeNowMilliseconds();
		(void)printf("probe rank=%u pack=%s init_status=%d init_wall_ms=%.3f init_device_ms=%.3f entries=%u spine_bytes=%llu expert_bytes=%llu\n",
			tp_rank,argv[1],(int)init_status,wall_end - wall_begin,(double)device_milliseconds,
			census.entries,(unsigned long long)census.spine_bytes,(unsigned long long)census.expert_bytes);
		status = init_status;
	}
	if ( expect_ok == 0u )
	{
		if ( state != 0 || status != SPARK_STATUS_UNSUPPORTED )
		{
			(void)fprintf(stderr,"probe fail_closed_violation status=%d state=%p\n",(int)status,state);
			SparkDsv41FlashResidentDecodeStageDestroy(state);
			return(1);
		}
		(void)printf("probe fail_closed_ok status=%d\n",(int)status);
		return(0);
	}
	if ( status != SPARK_STATUS_OK )
		return(1);
	status = SparkDsv41FlashProbeFramePrepare(stream,&frame,buffers,&input_token,&output_token);
	if ( status == SPARK_STATUS_OK )
	{
		wall_begin = SparkDsv41FlashProbeNowMilliseconds();
		status = SparkDsv41FlashProbeEventsSynchronize(begin,end,stream,&device_milliseconds);
		if ( status == SPARK_STATUS_OK )
		{
			SparkStatus exec_status = do_execute != 0u ?
				SparkDsv41FlashResidentDecodeStageExecute(state,&frame) : SPARK_STATUS_OK;
			status = SparkDsv41FlashProbeEventsSynchronize(begin,end,stream,&device_milliseconds);
			wall_end = SparkDsv41FlashProbeNowMilliseconds();
			(void)printf("probe rank=%u exec_status=%d exec_wall_ms=%.3f exec_device_ms=%.3f output_token=%u\n",
				tp_rank,(int)exec_status,wall_end - wall_begin,(double)device_milliseconds,output_token);
		}
	}
	SparkDsv41FlashResidentDecodeStageDestroy(state);
	(void)printf("probe destroy_status=0\n");
	if ( cudaStreamDestroy(stream) != cudaSuccess )
		return(1);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}
