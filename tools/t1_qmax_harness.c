#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"

extern SparkStatus SparkQwen38MaxResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state);
extern void SparkQwen38MaxResidentDecodeStageDestroy(void *module_state);
extern SparkStatus SparkQwen38MaxResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);

#define T1_QMAX_PROMPT_CAPACITY 64u

static uint32_t t1_parse_prompt(const char *text, uint32_t *prompt)
{
	const char *scan = text;
	uint32_t count = 0u;
	while ( *scan != '\0' && count < T1_QMAX_PROMPT_CAPACITY )
	{
		char *end;
		unsigned long value;
		errno = 0;
		value = strtoul(scan,&end,10);
		if ( end == scan || errno != 0 || value >= 0x80000000ul )
		{
			fprintf(stderr,"t1_qmax_harness bad prompt token near '%s'\n",scan);
			exit(2);
		}
		prompt[count++] = (uint32_t)value;
		scan = *end == ',' ? end + 1 : end;
	}
	if ( count == 0u )
	{
		fprintf(stderr,"t1_qmax_harness empty prompt\n");
		exit(2);
	}
	return(count);
}

int main(int argc, char **argv)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	uint32_t prompt[T1_QMAX_PROMPT_CAPACITY];
	uint32_t prompt_count,new_tokens,step,status,generated;
	uint32_t input_token,output_token;
	void *state;
	if ( argc != 3 )
	{
		fprintf(stderr,"usage: t1_qmax_harness PROMPT_CSV NEW_TOKENS\n");
		return(2);
	}
	prompt_count = t1_parse_prompt(argv[1],prompt);
	new_tokens = (uint32_t)strtoul(argv[2],0,10);
	if ( new_tokens == 0u )
	{
		fprintf(stderr,"t1_qmax_harness NEW_TOKENS must be positive\n");
		return(2);
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "qwen38max-t1";
	configuration.model_revision = "t1";
	configuration.stage_name = "qwen38";
	configuration.program_name = "resident_decode";
	configuration.operation_name = "initialize";
	memset(&services,0,sizeof(services));
	services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	services.descriptor_bytes = sizeof(services);
	state = 0;
	status = SparkQwen38MaxResidentDecodeStageInitialize(&configuration,&services,&state);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"t1_qmax_harness initialize status=%d\n",(int)status);
		return(1);
	}
	generated = 0u;
	{
		struct timespec total_begin;
		double total_seconds;
		uint32_t timing = getenv("T1_QMAX_TIMING") != 0 ? 1u : 0u;
		clock_gettime(CLOCK_MONOTONIC,&total_begin);
		for ( step = 0u; step < prompt_count + new_tokens; step++ )
		{
			struct timespec step_begin,step_end;
			int have_step = 0;
			input_token = step < prompt_count ? prompt[step] : generated;
			output_token = 0xdeadbeefu;
			memset(&frame,0,sizeof(frame));
			frame.active_slot_count = 1u;
			frame.new_token_count = 1u;
			frame.sequence_position = step;
			buffers[0].address = &input_token;
			buffers[0].bytes = sizeof(input_token);
			buffers[1].address = &output_token;
			buffers[1].bytes = sizeof(output_token);
			frame.buffers = buffers;
			frame.buffer_count = 2u;
			if ( timing != 0u )
			{
				clock_gettime(CLOCK_MONOTONIC,&step_begin);
				have_step = 1;
			}
			status = SparkQwen38MaxResidentDecodeStageExecute(state,&frame);
			if ( status != SPARK_STATUS_OK )
			{
				fprintf(stderr,"t1_qmax_harness execute step=%u status=%d\n",step,(int)status);
				SparkQwen38MaxResidentDecodeStageDestroy(state);
				return(1);
			}
			if ( have_step != 0 )
			{
				clock_gettime(CLOCK_MONOTONIC,&step_end);
				fprintf(stderr,"STEP seconds=%.6f step=%u\n",
					(double)(step_end.tv_sec - step_begin.tv_sec) +
					(double)(step_end.tv_nsec - step_begin.tv_nsec) / 1e9,step);
			}
			generated = output_token;
			printf("TOKEN step=%u input=%u output=%u\n",step,input_token,output_token);
			fflush(stdout);
		}
		if ( timing != 0u )
		{
			struct timespec total_end;
			clock_gettime(CLOCK_MONOTONIC,&total_end);
			total_seconds = (double)(total_end.tv_sec - total_begin.tv_sec) +
				(double)(total_end.tv_nsec - total_begin.tv_nsec) / 1e9;
			fprintf(stderr,"TOTAL steps=%u seconds=%.6f tok_per_s=%.6f\n",
				prompt_count + new_tokens,total_seconds,
				(double)(prompt_count + new_tokens) / total_seconds);
		}
	}
	SparkQwen38MaxResidentDecodeStageDestroy(state);
	fprintf(stderr,"t1_qmax_harness done steps=%u generated=%u\n",prompt_count + new_tokens,generated);
	return(0);
}
