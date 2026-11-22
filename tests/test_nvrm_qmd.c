#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_cubin.h"
#include "runtime/device/nvrm/nvrm_qmd.h"
#include "runtime/device/nvrm/nvrm_launch.h"

#define MW(range) range
#include "classes/compute/clcec0qmd.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif

#define HIGH(field) ((uint32_t)(1 ? field))
#define LOW(field) ((uint32_t)(0 ? field))
#define FIELD(qmd, field) Field((qmd),HIGH(field),LOW(field))

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

static uint64_t Field(const SparkNvrmQmd *qmd, uint32_t high, uint32_t low)
{
	uint8_t bytes[SPARK_NVRM_QMD_WORDS * 4u];
	unsigned __int128 window = 0;
	uint32_t index;
	const uint32_t width = high - low + 1u;
	for (index = 0u; index < SPARK_NVRM_QMD_WORDS * 4u; index++)
		bytes[index] = (uint8_t)(qmd->word[index / 4u] >> (8u * (index % 4u)));
	for (index = high / 8u + 1u; index > low / 8u; index--)
		window = (window << 8) | bytes[index - 1u];
	window >>= low % 8u;
	return((uint64_t)(window & (width >= 64u ? (unsigned __int128)UINT64_MAX : (((unsigned __int128)1 << width) - 1u))));
}

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

static SparkNvrmLaunchShape ShapeFor(const SparkNvrmCubinKernel *kernel)
{
	SparkNvrmLaunchShape shape;
	memset(&shape,0,sizeof(shape));
	shape.program_address = 0x7a0000000000ull + kernel->code_offset;
	shape.program_bytes = kernel->code_bytes;
	shape.grid[0] = 4u;
	shape.grid[1] = 2u;
	shape.grid[2] = 1u;
	shape.block[0] = 128u;
	shape.block[1] = 1u;
	shape.block[2] = 1u;
	shape.registers = kernel->registers;
	shape.barriers = 1u;
	shape.shared_bytes = kernel->shared_bytes;
	shape.local_bytes = 0x240u;
	shape.sass_version = 0x81u;
	shape.bank_address[0] = 0x7b0000000100ull;
	shape.bank_bytes[0] = (uint32_t)kernel->bank[0].bytes;
	shape.bank_mask = 1u;
	shape.release_address = 0x7c0000000040ull;
	shape.release_payload = 0x1122334455667788ull;
	return(shape);
}

static void TestLaunch(const SparkNvrmCubinKernel *kernel)
{
	const SparkNvrmLaunchShape shape = ShapeFor(kernel);
	const uint64_t prefetch_size = kernel->code_bytes >> 8 < 0x1ffu ? kernel->code_bytes >> 8 : 0x1ffu;
	SparkNvrmQmd qmd;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"first light") == SPARK_STATUS_OK,"a launch of the first-light kernel builds a descriptor");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_QMD_MAJOR_VERSION) == 5u && FIELD(&qmd,NVCEC0_QMDV05_00_QMD_MINOR_VERSION) == 0u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_QMD_TYPE) == NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA && FIELD(&qmd,NVCEC0_QMDV05_00_QMD_GROUP_ID) == 0x3fu,
		"the descriptor is a version 5.0 CTA grid in every group");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_INVALIDATE_TEXTURE_HEADER_CACHE) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_INVALIDATE_TEXTURE_SAMPLER_CACHE) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_INVALIDATE_TEXTURE_DATA_CACHE) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_INVALIDATE_SHADER_DATA_CACHE) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_API_VISIBLE_CALL_LIMIT) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_SAMPLER_INDEX) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CWD_MEMBAR_TYPE) == NVCEC0_QMDV05_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_INVALIDATE(0)) == 1u,"cache invalidation and the end-of-grid membar are set");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_PROGRAM_ADDRESS_LOWER_SHIFTED4) == ((shape.program_address >> 4) & 0xffffffffu) &&
		FIELD(&qmd,NVCEC0_QMDV05_00_PROGRAM_ADDRESS_UPPER_SHIFTED4) == shape.program_address >> 36 &&
		FIELD(&qmd,NVCEC0_QMDV05_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED) == ((shape.program_address >> 8) & 0xffffffffu) &&
		FIELD(&qmd,NVCEC0_QMDV05_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED) == shape.program_address >> 40 &&
		FIELD(&qmd,NVCEC0_QMDV05_00_PROGRAM_PREFETCH_SIZE) == prefetch_size && FIELD(&qmd,NVCEC0_QMDV05_00_SASS_VERSION) == 0x81u,
		"the program address, its prefetch and the instruction set version are set");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_GRID_WIDTH) == 4u && FIELD(&qmd,NVCEC0_QMDV05_00_GRID_HEIGHT) == 2u && FIELD(&qmd,NVCEC0_QMDV05_00_GRID_DEPTH) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CTA_THREAD_DIMENSION0) == 128u && FIELD(&qmd,NVCEC0_QMDV05_00_CTA_THREAD_DIMENSION1) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CTA_THREAD_DIMENSION2) == 1u,"the grid and group shape are set");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_REGISTER_COUNT) == kernel->registers && FIELD(&qmd,NVCEC0_QMDV05_00_BARRIER_COUNT) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_SHARED_MEMORY_SIZE_SHIFTED7) == (kernel->shared_bytes + 127u) / 128u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_MIN_SM_CONFIG_SHARED_MEM_SIZE) == 9u && FIELD(&qmd,NVCEC0_QMDV05_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE) == 9u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_MAX_SM_CONFIG_SHARED_MEM_SIZE) == 0x1au &&
		FIELD(&qmd,NVCEC0_QMDV05_00_SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4) == 0x24u,"registers, barriers, shared and local memory are set");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(0)) == ((shape.bank_address[0] >> 6) & 0xffffffffu) &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6(0)) == shape.bank_address[0] >> 38 &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0)) == (kernel->bank[0].bytes + 15u) / 16u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_VALID(0)) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_VALID(1)) == 0u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(1)) == 0u,"constant bank 0 alone is bound, with its address and size");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_ADDR_LOWER) == 0x00000040u && FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_ADDR_UPPER) == 0x7c00u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_PAYLOAD_LOWER) == 0x55667788u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_PAYLOAD_UPPER) == 0x11223344u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_ENABLE(0)) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_ENABLE(1)) == 0u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE(0)) == NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_PAYLOAD64B(0)) == NVCEC0_QMDV05_00_RELEASE_PAYLOAD64B_TRUE &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_PAYLOAD64B(1)) == 0u,
		"the grid releases a 64-bit payload to its semaphore");
	CHECK(FIELD(&qmd,NVCEC0_QMDV05_00_DEPENDENT_QMD0_ENABLE) == 0u && FIELD(&qmd,NVCEC0_QMDV05_00_GPC_CGA_WIDTH) == 0u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_TPC_DISABLE_MASK_VALID) == 0u && FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE1_ADDR_LOWER) == 0u,
		"fields the launch does not ask for stay zero");
	CHECK(qmd.word[39] == 4u && (qmd.word[34] & 0xffffu) == 128u && ((qmd.word[14] >> 20) & 0xfu) == 5u && qmd.word[15] == 0x40u &&
		qmd.word[17] == 0x55667788u && qmd.word[18] == 0x11223344u,"the grid width, group width, version and release land in their descriptor words");
}

static void TestReleaseTimestamp(const SparkNvrmCubinKernel *kernel)
{
	SparkNvrmLaunchShape shape = ShapeFor(kernel);
	SparkNvrmQmd qmd;
	shape.release_timestamp = 1u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"timed") == SPARK_STATUS_OK &&
		FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE(0)) == NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS,
		"a timed release writes the four-word semaphore");
	shape.release_address = 0x7c0000000048ull;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"timed") == SPARK_STATUS_INVALID_ARGUMENT,"a four-word release needs 16-byte alignment");
	shape.release_address = 0u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"untimed") == SPARK_STATUS_OK && FIELD(&qmd,NVCEC0_QMDV05_00_RELEASE_ENABLE(0)) == 0u,
		"a launch without a semaphore releases nothing");
}

static void TestCluster(const SparkNvrmCubinKernel *kernel)
{
	SparkNvrmLaunchShape shape = ShapeFor(kernel);
	SparkNvrmQmd qmd;
	shape.grid[0] = 8u;
	shape.cluster[0] = 2u;
	shape.cluster[1] = 2u;
	shape.cluster[2] = 1u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"cluster") == SPARK_STATUS_OK &&
		FIELD(&qmd,NVCEC0_QMDV05_00_QMD_TYPE) == NVCEC0_QMDV05_00_QMD_TYPE_GRID_GPC_CGA && FIELD(&qmd,NVCEC0_QMDV05_00_GPC_CGA_WIDTH) == 2u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_GPC_CGA_HEIGHT) == 2u && FIELD(&qmd,NVCEC0_QMDV05_00_GPC_CGA_DEPTH) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_GRID_WIDTH) == 4u && FIELD(&qmd,NVCEC0_QMDV05_00_GRID_HEIGHT) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_GRID_DEPTH) == 1u,
		"a cluster launch is a GPC cluster grid with the cluster's width, height and depth, and its grid counted in clusters");
	shape.grid[0] = 7u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"cluster") == SPARK_STATUS_INVALID_ARGUMENT &&
		strstr(SparkDeviceLastError()->text,"whole number of clusters") != 0,"a grid that is not a whole number of clusters is refused");
	shape.grid[0] = 64u;
	shape.grid[1] = 1u;
	shape.cluster[0] = 64u;
	shape.cluster[1] = 1u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"cluster") == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"GPC_CGA_WIDTH") != 0,
		"a cluster wider than its descriptor field is refused");
}

static void TestLatch(const SparkNvrmCubinKernel *kernel)
{
	SparkNvrmLaunchShape shape = ShapeFor(kernel);
	SparkNvrmQmd qmd;
	shape.latch_arrive = 41u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"latch") == SPARK_STATUS_OK && FIELD(&qmd,NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_VALID) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_ID) == 41u && FIELD(&qmd,NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 0u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_HOLD_CTA_LAUNCH_UNTIL_PARENT_LATCH_ACQUIRE_AND_CTA_COMPLETE) == 0u,
		"every launch arrives at its own latch and waits on none");
	shape.latch_wait = 40u;
	CHECK(SparkNvrmQmdBuild(&shape,&qmd,"latch") == SPARK_STATUS_OK && FIELD(&qmd,NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_WAIT_ON_LATCH_ID) == 40u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_HOLD_CTA_LAUNCH_UNTIL_PARENT_LATCH_ACQUIRE_AND_CTA_COMPLETE) == 1u &&
		FIELD(&qmd,NVCEC0_QMDV05_00_HOLD_MEMBAR_UNTIL_LATCH_ACQUIRE) == 1u && FIELD(&qmd,NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_ID) == 41u,
		"a dependent launch waits on the launch before it, holds its blocks until that one ends and holds its membar until the latch");
}

static SparkStatus ClusterShape(const SparkNvrmCubinKernel *kernel, uint32_t groups, uint32_t asked, uint32_t cluster[3])
{
	SparkNvrmLaunchRequest request;
	memset(&request,0,sizeof(request));
	request.kernel = kernel;
	request.grid[0] = groups;
	request.grid[1] = request.grid[2] = 1u;
	request.cluster[0] = asked;
	request.cluster[1] = request.cluster[2] = asked != 0u ? 1u : 0u;
	return(SparkNvrmLaunchClusterShape(&request,cluster,"cluster shape"));
}

static void TestClusterShape(const SparkNvrmCubinKernel *fixture)
{
	SparkNvrmCubinKernel kernel = *fixture;
	uint32_t cluster[3] = {9u, 9u, 9u};
	CHECK(ClusterShape(&kernel,8u,0u,cluster) == SPARK_STATUS_OK && cluster[0] == 0u && cluster[1] == 0u && cluster[2] == 0u,
		"a kernel without clusters launches without one");
	CHECK(ClusterShape(&kernel,8u,4u,cluster) == SPARK_STATUS_OK && cluster[0] == 4u && cluster[1] == 1u && cluster[2] == 1u,
		"any kernel can be launched in clusters the launch names");
	kernel.clustered = 1u;
	kernel.cluster[0] = 2u;
	kernel.cluster[1] = kernel.cluster[2] = 1u;
	CHECK(ClusterShape(&kernel,8u,0u,cluster) == SPARK_STATUS_OK && cluster[0] == 2u && cluster[1] == 1u && cluster[2] == 1u,
		"a kernel compiled for clusters runs in them when the launch names none");
	CHECK(ClusterShape(&kernel,8u,2u,cluster) == SPARK_STATUS_OK && cluster[0] == 2u,"a launch may repeat the compiled cluster shape");
	CHECK(ClusterShape(&kernel,8u,4u,cluster) == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"compiled for clusters of 2x1x1") != 0,
		"a launch that names another cluster shape than the compiled one is refused");
	CHECK(ClusterShape(&kernel,7u,0u,cluster) == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"whole number of clusters") != 0,
		"a grid that is not a whole number of compiled clusters is refused");
	kernel.cluster[0] = kernel.cluster[1] = kernel.cluster[2] = 0u;
	CHECK(ClusterShape(&kernel,8u,0u,cluster) == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"launch it with a cluster shape") != 0,
		"a kernel that must run in clusters without a compiled shape needs one at launch");
	CHECK(ClusterShape(&kernel,16u,16u,cluster) == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"over the limit of 8") != 0,
		"a cluster over eight blocks is refused");
	kernel.cluster_limit = 2u;
	CHECK(ClusterShape(&kernel,8u,4u,cluster) == SPARK_STATUS_INVALID_ARGUMENT && strstr(SparkDeviceLastError()->text,"over the limit of 2") != 0 &&
		ClusterShape(&kernel,8u,2u,cluster) == SPARK_STATUS_OK,"a kernel's own cluster rank bound lowers the limit");
}

typedef struct Broken
{
	const char *what;
	void (*apply)(SparkNvrmLaunchShape *shape);
	const char *named;
} Broken;

static void NoGroups(SparkNvrmLaunchShape *shape) { shape->grid[0] = 0u; }
static void TallGrid(SparkNvrmLaunchShape *shape) { shape->grid[1] = 65536u; }
static void WideGroup(SparkNvrmLaunchShape *shape) { shape->block[0] = 1025u; }
static void DeepGroup(SparkNvrmLaunchShape *shape) { shape->block[0] = 1u; shape->block[2] = 65u; }
static void CrookedProgram(SparkNvrmLaunchShape *shape) { shape->program_address += 8u; }
static void HugeShared(SparkNvrmLaunchShape *shape) { shape->shared_bytes = 100u * 1024u + 1u; }
static void CrookedLocal(SparkNvrmLaunchShape *shape) { shape->local_bytes = 8u; }
static void NoBankZero(SparkNvrmLaunchShape *shape) { shape->bank_mask = 2u; shape->bank_address[1] = 0x7b0000000200ull; }
static void CrookedBank(SparkNvrmLaunchShape *shape) { shape->bank_address[0] += 32u; }
static void CrookedRelease(SparkNvrmLaunchShape *shape) { shape->release_address += 4u; }
static void ManyRegisters(SparkNvrmLaunchShape *shape) { shape->registers = 512u; }
static void WideVersion(SparkNvrmLaunchShape *shape) { shape->sass_version = 256u; }
static void ManyBarriers(SparkNvrmLaunchShape *shape) { shape->barriers = 32u; }
static void HugeBank(SparkNvrmLaunchShape *shape) { shape->bank_bytes[0] = 0x20000u; }

static const Broken broken[] = {
	{"a launch without groups",NoGroups,0},
	{"a grid taller than 65535 groups",TallGrid,0},
	{"a group of more than 1024 threads",WideGroup,0},
	{"a group deeper than 64 threads",DeepGroup,0},
	{"a program address off 16 bytes",CrookedProgram,0},
	{"more shared memory than an SM configuration offers",HugeShared,0},
	{"a local memory size off 16 bytes",CrookedLocal,0},
	{"a launch without constant bank 0",NoBankZero,0},
	{"a constant bank off 64 bytes",CrookedBank,0},
	{"a release semaphore off 8 bytes",CrookedRelease,0},
	{"512 registers",ManyRegisters,"REGISTER_COUNT"},
	{"an instruction set version wider than its field",WideVersion,"SASS_VERSION"},
	{"32 barriers",ManyBarriers,"BARRIER_COUNT"},
	{"a 128 KiB constant bank",HugeBank,"CONSTANT_BUFFER_SIZE_SHIFTED4"},
};

static void TestRefusals(const SparkNvrmCubinKernel *kernel)
{
	uint32_t index;
	for (index = 0u; index < sizeof(broken) / sizeof(broken[0]); index++)
	{
		SparkNvrmLaunchShape shape = ShapeFor(kernel);
		SparkNvrmQmd qmd;
		char what[160];
		broken[index].apply(&shape);
		snprintf(what,sizeof(what),"%s is refused%s%s",broken[index].what,broken[index].named != 0 ? ", naming " : "",
			broken[index].named != 0 ? broken[index].named : "");
		CHECK(SparkNvrmQmdBuild(&shape,&qmd,"broken") == SPARK_STATUS_INVALID_ARGUMENT &&
			strcmp(SparkDeviceLastError()->operation,"kernel_launch") == 0 &&
			(broken[index].named == 0 || strstr(SparkDeviceLastError()->text,broken[index].named) != 0),what);
	}
	CHECK(SparkNvrmQmdBuild(0,0,"broken") == SPARK_STATUS_INVALID_ARGUMENT,"a build without a shape is refused");
}

static void TestSharedConfiguration(const SparkNvrmCubinKernel *kernel)
{
	static const uint32_t shared[] = {0u, 32u * 1024u, 32u * 1024u + 1u, 64u * 1024u, 64u * 1024u + 1u, 100u * 1024u};
	static const uint32_t config[] = {9u, 9u, 17u, 17u, 26u, 26u};
	uint32_t index;
	for (index = 0u; index < sizeof(shared) / sizeof(shared[0]); index++)
	{
		SparkNvrmLaunchShape shape = ShapeFor(kernel);
		SparkNvrmQmd qmd;
		char what[96];
		shape.shared_bytes = shared[index];
		snprintf(what,sizeof(what),"%u bytes of shared memory pick SM configuration %u",shared[index],config[index]);
		CHECK(SparkNvrmQmdBuild(&shape,&qmd,"shared") == SPARK_STATUS_OK &&
			FIELD(&qmd,NVCEC0_QMDV05_00_MIN_SM_CONFIG_SHARED_MEM_SIZE) == config[index] &&
			FIELD(&qmd,NVCEC0_QMDV05_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE) == config[index],what);
	}
}

static void TestFieldAccess(void)
{
	SparkNvrmQmd qmd;
	const uint64_t value = 0x5555555555555ull & ((1ull << 51) - 1u);
	memset(&qmd,0xff,sizeof(qmd));
	CHECK(SparkNvrmQmdWrite(&qmd,70u,20u,0u,"field") == SPARK_STATUS_OK && Field(&qmd,70u,20u) == 0u &&
		Field(&qmd,19u,19u) == 1u && Field(&qmd,71u,71u) == 1u,"clearing a field that spans three words leaves its neighbours");
	CHECK(SparkNvrmQmdWrite(&qmd,70u,20u,value,"field") == SPARK_STATUS_OK && Field(&qmd,70u,20u) == value &&
		SparkNvrmQmdRead(&qmd,70u,20u) == value,"a written field reads back through both readers");
	memset(&qmd,0,sizeof(qmd));
	CHECK(SparkNvrmQmdWrite(&qmd,127u,64u,UINT64_MAX,"field") == SPARK_STATUS_OK && qmd.word[2] == UINT32_MAX && qmd.word[3] == UINT32_MAX &&
		qmd.word[1] == 0u && qmd.word[4] == 0u,"a 64-bit field fills exactly two whole words");
	CHECK(SparkNvrmQmdWrite(&qmd,3u,0u,16u,"field") == SPARK_STATUS_INVALID_ARGUMENT,"a value wider than its field is refused");
	CHECK(SparkNvrmQmdWrite(&qmd,0u,3u,1u,"field") == SPARK_STATUS_INVALID_ARGUMENT,"a reversed field is refused");
	CHECK(SparkNvrmQmdWrite(&qmd,3072u,3072u,1u,"field") == SPARK_STATUS_INVALID_ARGUMENT,"a field past the descriptor is refused");
	CHECK(SparkNvrmQmdWrite(&qmd,64u,0u,1u,"field") == SPARK_STATUS_INVALID_ARGUMENT,"a field wider than 64 bits is refused");
}

static void TestBank0(const SparkNvrmCubinKernel *kernel)
{
	struct { uint32_t *output; uint32_t value; } arguments = {(uint32_t *)(uintptr_t)0x7d0000001000ull,0xabcdef01u};
	const uint64_t shared_window = 0x729400000000ull,local_window = 0x729300000000ull,bank_address = 0x7d0000200200ull;
	const uint32_t grid[3] = {7u, 5u, 3u},block[3] = {96u, 2u, 1u};
	const uint32_t bytes = (uint32_t)kernel->bank[0].bytes;
	uint8_t *bank = (uint8_t *)malloc(bytes);
	SparkNvrmCubinKernel overlapping = *kernel;
	uint32_t *words;
	if ( bank == 0 )
		return;
	memset(bank,0xee,bytes);
	CHECK(kernel->parameter_bytes == 12u && SparkNvrmBank0Build(kernel,shared_window,local_window,grid,block,0,&arguments,12u,bank,bank_address,bytes,"bank") == SPARK_STATUS_OK,
		"constant bank 0 builds for the first-light kernel");
	words = (uint32_t *)(void *)bank;
	CHECK(words[188] == 0x00000000u && words[189] == 0x7294u && words[190] == 0x00000000u && words[191] == 0x7293u && words[223] == 0xfffdc0u,
		"the shared and local memory windows and the last driver word are set");
	CHECK(words[SPARK_NVRM_BANK0_WORD_BLOCK_X] == 96u && words[SPARK_NVRM_BANK0_WORD_BLOCK_Y] == 2u && words[SPARK_NVRM_BANK0_WORD_BLOCK_Z] == 1u &&
		words[SPARK_NVRM_BANK0_WORD_GRID_X] == 7u && words[SPARK_NVRM_BANK0_WORD_GRID_Y] == 5u && words[SPARK_NVRM_BANK0_WORD_GRID_Z] == 3u,
		"the launch shape lands in the words the compiled code reads it from");
	CHECK(words[SPARK_NVRM_BANK0_WORD_PARAMETERS] == 0x00200580u && words[SPARK_NVRM_BANK0_WORD_PARAMETERS + 1u] == 0x7d00u,
		"the parameters' GPU address, bank 0's address plus the parameter offset, lands where a parameter's generic address is formed");
	CHECK(words[168] == 1u && words[169] == 1u && words[170] == 1u && words[172] == 0x3f800000u && words[173] == 0x3f800000u &&
		words[174] == 0x3f800000u && words[176] == 7u && words[177] == 5u && words[178] == 3u && words[179] == 1u,
		"a launch without clusters fills the cluster words as clusters of one block: sizes 1, reciprocals 1.0, the grid in clusters and one rank");
	CHECK(words[0] == 0u && words[187] == 0u && words[192] == 0u && words[209] == 0u && words[212] == 0u && words[215] == 0u && words[219] == 0u,
		"the other driver words are zero");
	CHECK(SparkNvrmBank0WordClaimed(216u) && SparkNvrmBank0WordClaimed(222u) && SparkNvrmBank0WordClaimed(188u) && SparkNvrmBank0WordClaimed(223u) &&
		SparkNvrmBank0WordClaimed(210u) && SparkNvrmBank0WordClaimed(211u) && !SparkNvrmBank0WordClaimed(209u) && !SparkNvrmBank0WordClaimed(212u) &&
		SparkNvrmBank0WordClaimed(168u) && SparkNvrmBank0WordClaimed(170u) && SparkNvrmBank0WordClaimed(172u) && SparkNvrmBank0WordClaimed(174u) &&
		SparkNvrmBank0WordClaimed(176u) && SparkNvrmBank0WordClaimed(179u) && !SparkNvrmBank0WordClaimed(171u) && !SparkNvrmBank0WordClaimed(175u) &&
		!SparkNvrmBank0WordClaimed(167u) && !SparkNvrmBank0WordClaimed(180u) && !SparkNvrmBank0WordClaimed(219u) && !SparkNvrmBank0WordClaimed(0u),
		"exactly the cluster, window, parameter address, last and launch-shape words are claimed");
	{
		const uint32_t cluster_grid[3] = {8u, 6u, 3u},cluster[3] = {2u, 3u, 1u},uneven[3] = {3u, 3u, 1u};
		CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,cluster_grid,block,cluster,&arguments,12u,bank,bank_address,bytes,"bank") ==
			SPARK_STATUS_OK && words[168] == 2u && words[169] == 3u && words[170] == 1u && words[172] == 0x3f000000u && words[173] == 0x3eaaaaabu &&
			words[174] == 0x3f800000u && words[176] == 4u && words[177] == 2u && words[178] == 3u && words[179] == 6u,
			"a cluster launch fills the cluster sizes, their float reciprocals, the grid in clusters and the blocks per cluster");
		CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,cluster_grid,block,uneven,&arguments,12u,bank,bank_address,bytes,"bank") ==
			SPARK_STATUS_INVALID_ARGUMENT,"a cluster that does not divide the grid is refused");
		CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,grid,block,0,&arguments,12u,bank,bank_address,bytes,"bank") == SPARK_STATUS_OK,
			"rebuild the plain bank for the checks that follow");
	}
	CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,grid,block,0,&arguments,12u,bank,0u,bytes,"bank") == SPARK_STATUS_INVALID_ARGUMENT &&
		SparkNvrmBank0Build(kernel,shared_window,local_window,grid,block,0,&arguments,12u,bank,bank_address + 8u,bytes,"bank") == SPARK_STATUS_INVALID_ARGUMENT,
		"a bank without its 64-byte aligned GPU address is refused");
	CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,0,block,0,&arguments,12u,bank,bank_address,bytes,"bank") == SPARK_STATUS_INVALID_ARGUMENT,
		"a bank without a launch shape is refused");
	CHECK(memcmp(bank + SPARK_NVRM_BANK0_DRIVER_BYTES,&arguments,12u) == 0,"the arguments follow the driver words at the kernel's parameter offset");
	CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,grid,block,0,&arguments,8u,bank,bank_address,bytes,"bank") == SPARK_STATUS_INVALID_ARGUMENT,
		"an argument block of the wrong size is refused");
	CHECK(SparkNvrmBank0Build(kernel,shared_window,local_window,grid,block,0,&arguments,12u,bank,bank_address,bytes - 1u,"bank") == SPARK_STATUS_INVALID_ARGUMENT,
		"a constant bank smaller than the parameters is refused");
	overlapping.parameter_offset = 0x160u;
	CHECK(SparkNvrmBank0Build(&overlapping,shared_window,local_window,grid,block,0,&arguments,12u,bank,bank_address,bytes,"bank") == SPARK_STATUS_INVALID_ARGUMENT,
		"parameters overlapping the driver words are refused");
	free(bank);
}

int main(void)
{
	uint64_t bytes = 0u;
	uint8_t *blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes);
	SparkNvrmCubin cubin;
	const SparkNvrmCubinKernel *kernel;
	CHECK(blob != 0 && SparkNvrmCubinParse(blob,bytes,"fixture",&cubin) == SPARK_STATUS_OK,"the fixture cubin parses");
	kernel = blob != 0 ? SparkNvrmCubinKernelNamed(&cubin,"SparkNvrmFirstLight") : 0;
	CHECK(kernel != 0,"the fixture holds the first-light kernel");
	if ( kernel != 0 )
	{
		TestLaunch(kernel);
		TestReleaseTimestamp(kernel);
		TestRefusals(kernel);
		TestSharedConfiguration(kernel);
		TestBank0(kernel);
		TestCluster(kernel);
		TestLatch(kernel);
		TestClusterShape(kernel);
		SparkNvrmCubinRelease(&cubin);
	}
	TestFieldAccess();
	free(blob);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS QMD version 5 launch descriptors and constant bank 0 are packed field by field and refuse what does not fit\n");
	return(0);
}
