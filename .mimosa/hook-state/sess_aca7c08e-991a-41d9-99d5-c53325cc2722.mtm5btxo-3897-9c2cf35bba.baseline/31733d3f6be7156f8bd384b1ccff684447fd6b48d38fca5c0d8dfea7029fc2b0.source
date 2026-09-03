#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
extern SparkStatus SparkQwen38ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *, const SparkFirmwareModuleHostServices *, void **);
extern void SparkQwen38ResidentDecodeStageDestroy(void *);
extern SparkStatus SparkQwen38ResidentDecodeStageExecute(void *, SparkModelDriverFrame *);

static double NowSeconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC,&ts);
	return((double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9);
}

int main(int argc, char **argv)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	uint32_t *input_tokens,*output_tokens;
	void *state;
	SparkStatus status;
	uint32_t batch,steps,step,row;
	double begin,end,elapsed;
	char batch_text[32],steps_text[32];
	if (argc != 4)
	{
		fprintf(stderr,"usage: qwen38_timing_probe PACK BATCH STEPS\n");
		return(2);
	}
	batch = (uint32_t)strtoul(argv[2],0,10);
	steps = (uint32_t)strtoul(argv[3],0,10);
	if (batch == 0u || batch > 512u || steps == 0u)
		return(2);
	snprintf(batch_text,sizeof(batch_text),"%u",batch);
	snprintf(steps_text,sizeof(steps_text),"%u",steps);
	setenv("SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION","1",1);
	setenv("SPARK_QWEN38_STAGE_COUNT","4",1);
	setenv("SPARK_QWEN38_STAGE_INDEX","1",1);
	setenv("SPARK_QWEN38_STAGE_FIRST_LAYER","1",1);
	setenv("SPARK_QWEN38_STAGE_LAYER_COUNT","1",1);
	setenv("SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES",batch_text,1);
	setenv("SPARK_QWEN38_STAGE_PIPELINE_SLOTS","1",1);
	setenv("SPARK_QWEN38_STAGE_KV_BLOCKS","8",1);
	setenv("SPARK_QWEN38_STAGE_PACK_PATH",argv[1],1);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "probe";
	configuration.model_revision = "test";
	configuration.stage_name = "qwen38";
	configuration.program_name = "resident_decode";
	configuration.operation_name = "initialize";
	memset(&services,0,sizeof(services));
	services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	services.descriptor_bytes = sizeof(services);
	status = SparkQwen38ResidentDecodeStageInitialize(&configuration,&services,&state);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"initialize status=%d\n",(int)status);
		return(1);
	}
	input_tokens = (uint32_t *)calloc(batch,sizeof(uint32_t));
	output_tokens = (uint32_t *)calloc(batch,sizeof(uint32_t));
	for (row = 0u; row < batch; row++)
		input_tokens[row] = 123u;
	memset(&frame,0,sizeof(frame));
	frame.active_slot_count = batch;
	frame.new_token_count = batch;
	memset(buffers,0,sizeof(buffers));
	buffers[0].address = input_tokens;
	buffers[0].bytes = batch * 4u;
	buffers[1].address = output_tokens;
	buffers[1].bytes = batch * 4u;
	frame.buffers = buffers;
	frame.buffer_count = 2u;
	frame.sequence_position = 0u;
	status = SparkQwen38ResidentDecodeStageExecute(state,&frame);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"warmup status=%d\n",(int)status);
		return(1);
	}
	begin = NowSeconds();
	for (step = 1u; step <= steps; step++)
	{
		frame.sequence_position = (uint64_t)step;
		status = SparkQwen38ResidentDecodeStageExecute(state,&frame);
		if (status != SPARK_STATUS_OK)
		{
			fprintf(stderr,"step %u status=%d\n",step,(int)status);
			return(1);
		}
	}
	end = NowSeconds();
	elapsed = (end - begin) / (double)steps;
	fprintf(stderr,"batch=%u steps=%u ms_per_step=%.3f tok_per_s=%.1f\n",batch,steps,elapsed * 1000.0,(double)batch / elapsed);
	SparkQwen38ResidentDecodeStageDestroy(state);
	return(0);
}
