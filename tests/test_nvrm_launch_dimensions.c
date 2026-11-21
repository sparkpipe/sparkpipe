#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif

#define PROBE_TIMEOUT_NS 10000000000ull
#define PROBE_SENTINEL 0x5e000000u
#define PROBE_SENTINEL_MASK 0xffffff00u
#define PROBE_DRIVER_WORDS (SPARK_NVRM_BANK0_DRIVER_BYTES / 4u)
#define PROBE_VALUES 6u

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

typedef struct Probe
{
	SparkNvrmGpu gpu;
	SparkNvrmProgram program;
	const SparkNvrmCubinKernel *kernel;
	SparkNvrmMemory output;
	SparkNvrmMemory slot;
	SparkNvrmMemory semaphore;
	uint64_t sequence;
} Probe;

static const char *const probe_names[PROBE_VALUES] = {"block x", "block y", "block z", "grid x", "grid y", "grid z"};

static uint8_t *ReadFile(const char *path, uint64_t *bytes)
{
	FILE *file = fopen(path,"rb");
	uint8_t *data;
	long length;
	if ( file == 0 )
		return(0);
	fseek(file,0,SEEK_END);
	length = ftell(file);
	fseek(file,0,SEEK_SET);
	data = (uint8_t *)malloc((size_t)length);
	if ( data != 0 && fread(data,1u,(size_t)length,file) != (size_t)length )
	{
		free(data);
		data = 0;
	}
	fclose(file);
	*bytes = (uint64_t)length;
	return(data);
}

static int ProbeFailed(const char *step)
{
	const SparkDeviceError *error = SparkDeviceLastError();
	printf("FAIL %s: %s [%s, %s, code %d]\n",step,error->text,error->operation,error->owner,error->backend_code);
	return(1);
}

static const uint32_t probe_words[PROBE_VALUES] = {SPARK_NVRM_BANK0_WORD_BLOCK_X, SPARK_NVRM_BANK0_WORD_BLOCK_Y, SPARK_NVRM_BANK0_WORD_BLOCK_Z,
	SPARK_NVRM_BANK0_WORD_GRID_X, SPARK_NVRM_BANK0_WORD_GRID_Y, SPARK_NVRM_BANK0_WORD_GRID_Z};

static int ProbeSentinelWord(uint32_t word)
{
	uint32_t index;
	for (index = 0u; index < PROBE_VALUES; index++)
		if ( probe_words[index] == word )
			return(1);
	return(!SparkNvrmBank0WordClaimed(word));
}

static int ProbeRun(Probe *probe, const uint32_t grid[3], const uint32_t block[3], uint32_t sentinels, uint32_t *values)
{
	SparkNvrmChannel *channel = &probe->gpu.compute;
	SparkNvrmLaunchRequest request;
	const uint64_t value = ++probe->sequence,output = probe->output.address;
	uint32_t index;
	memset(probe->output.cpu,0xee,PROBE_VALUES * sizeof(uint32_t));
	memset(&request,0,sizeof(request));
	request.kernel = probe->kernel;
	memcpy(request.grid,grid,sizeof(request.grid));
	memcpy(request.block,block,sizeof(request.block));
	request.arguments = &output;
	request.argument_bytes = (uint32_t)sizeof(output);
	request.slot = probe->slot.cpu;
	request.slot_address = probe->slot.address;
	request.slot_bytes = (uint32_t)probe->slot.bytes;
	request.release_address = probe->semaphore.address;
	request.release_payload = value;
	if ( SparkNvrmLaunchQueue(&probe->gpu,channel,&probe->program,&request,"probe") != SPARK_STATUS_OK )
		return(ProbeFailed("queue the probe"));
	if ( sentinels != 0u )
	{
		uint32_t *words = (uint32_t *)(void *)(probe->slot.cpu + SPARK_NVRM_LAUNCH_QMD_BYTES);
		for (index = 0u; index < PROBE_DRIVER_WORDS; index++)
			if ( ProbeSentinelWord(index) )
				words[index] = PROBE_SENTINEL | index;
	}
	if ( SparkNvrmSubmit(&probe->gpu,channel,"probe") != SPARK_STATUS_OK ||
		SparkNvrmWaitValue((const volatile uint64_t *)(void *)probe->semaphore.cpu,value,PROBE_TIMEOUT_NS,"kernel_launch","probe") != SPARK_STATUS_OK )
		return(ProbeFailed("run the probe"));
	memcpy(values,probe->output.cpu,PROBE_VALUES * sizeof(uint32_t));
	return(0);
}

static void TestSentinels(Probe *probe)
{
	static const uint32_t one[3] = {1u, 1u, 1u};
	uint32_t values[PROBE_VALUES],index;
	if ( ProbeRun(probe,one,one,1u,values) != 0 )
	{
		failures++;
		return;
	}
	for (index = 0u; index < PROBE_VALUES; index++)
	{
		char what[160];
		const uint32_t word = values[index] & ~PROBE_SENTINEL_MASK;
		printf("probe: %s is read from bank 0 word %u (byte %#x), raw %#x\n",probe_names[index],word,word * 4u,values[index]);
		snprintf(what,sizeof(what),"the compiled code reads %s from one driver word, and it is the word the launcher fills (%u)",probe_names[index],
			probe_words[index]);
		CHECK((values[index] & PROBE_SENTINEL_MASK) == PROBE_SENTINEL && word == probe_words[index],what);
	}
}

static void TestShapes(Probe *probe)
{
	static const uint32_t shapes[][6] = {
		{1u, 1u, 1u, 1u, 1u, 1u},
		{7u, 1u, 1u, 96u, 1u, 1u},
		{5u, 3u, 1u, 16u, 8u, 1u},
		{2u, 3u, 4u, 4u, 4u, 4u},
		{3u, 5u, 7u, 7u, 7u, 7u},
		{1u, 65535u, 1u, 33u, 3u, 1u},
		{1u, 1u, 65535u, 1000u, 1u, 1u},
		{1048575u, 1u, 1u, 1u, 1u, 1u},
		{9u, 7u, 5u, 1u, 1u, 64u},
		{2u, 2u, 2u, 1024u, 1u, 1u},
		{1u, 1u, 1u, 32u, 32u, 1u},
	};
	uint32_t index;
	for (index = 0u; index < sizeof(shapes) / sizeof(shapes[0]); index++)
	{
		uint32_t values[PROBE_VALUES];
		char what[160];
		if ( ProbeRun(probe,shapes[index],shapes[index] + 3,0u,values) != 0 )
		{
			failures++;
			return;
		}
		snprintf(what,sizeof(what),"grid %ux%ux%u of %ux%ux%u threads reads its own shape",shapes[index][0],shapes[index][1],shapes[index][2],
			shapes[index][3],shapes[index][4],shapes[index][5]);
		CHECK(values[0] == shapes[index][3] && values[1] == shapes[index][4] && values[2] == shapes[index][5] && values[3] == shapes[index][0] &&
			values[4] == shapes[index][1] && values[5] == shapes[index][2],what);
	}
}

static void Release(Probe *probe)
{
	if ( probe->semaphore.bytes != 0u )
		SparkNvrmMemoryRelease(&probe->gpu,&probe->semaphore);
	if ( probe->slot.bytes != 0u )
		SparkNvrmMemoryRelease(&probe->gpu,&probe->slot);
	if ( probe->output.bytes != 0u )
		SparkNvrmMemoryRelease(&probe->gpu,&probe->output);
	SparkNvrmProgramRelease(&probe->gpu,&probe->program);
	SparkNvrmGpuClose(&probe->gpu);
}

int main(void)
{
	static Probe probe;
	uint64_t bytes = 0u;
	uint8_t *blob;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_launch_dimensions (no open kernel module on this host)\n");
		return(0);
	}
	blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes);
	if ( blob == 0 )
	{
		printf("FAIL cannot read %s\n",SPARK_NVRM_FIXTURE_CUBIN);
		return(1);
	}
	if ( SparkNvrmGpuOpen(&probe.gpu,0u,SPARK_NVRM_MEMORY_HOST,"probe") != SPARK_STATUS_OK )
		return(ProbeFailed("open the GPU"));
	if ( SparkNvrmProgramLoad(&probe.gpu,blob,bytes,&probe.program,"probe") != SPARK_STATUS_OK )
		return(ProbeFailed("load the fixture"));
	free(blob);
	probe.kernel = SparkNvrmCubinKernelNamed(&probe.program.cubin,"SparkNvrmProbeDimensions");
	if ( probe.kernel == 0 ||
		SparkNvrmMemoryAllocate(&probe.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&probe.output,"probe") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&probe.gpu,SparkNvrmLaunchSlotBytes(probe.kernel),SPARK_NVRM_MEMORY_HOST,&probe.slot,"probe") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&probe.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&probe.semaphore,"probe") != SPARK_STATUS_OK )
		return(ProbeFailed("prepare the probe"));
	TestSentinels(&probe);
	TestShapes(&probe);
	Release(&probe);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the compiled code reads the launch shape from the bank 0 words the launcher fills, for every shape tried\n");
	return(0);
}
