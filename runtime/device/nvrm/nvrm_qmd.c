#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_qmd.h"

#define MW(range) range
#include "classes/compute/clcec0qmd.h"

#define QMD_HIGH(field) ((uint32_t)(1 ? field))
#define QMD_LOW(field) ((uint32_t)(0 ? field))
#define QMD_PUT(builder, field, value) QmdPut((builder),QMD_HIGH(field),QMD_LOW(field),(uint64_t)(value),#field)
#define QMD_BITS 3072u
#define QMD_PREFETCH_SIZE_MAX 0x1ffu
#define QMD_GROUP_ALL 0x3fu
#define QMD_GRID_WIDTH_MAX 0x7fffffffu
#define QMD_GRID_EXTENT_MAX 0xffffu
#define QMD_BLOCK_THREADS_MAX 1024u
#define QMD_BLOCK_DEPTH_MAX 64u
#define QMD_SHARED_CONFIG_MAX 0x1au
#define BANK0_WINDOW_WORD_SHARED 188u
#define BANK0_WINDOW_WORD_LOCAL 190u
#define BANK0_WORD_LAST 223u
#define BANK0_LAST_VALUE 0xfffdc0u

typedef struct QmdBuilder
{
	SparkNvrmQmd *qmd;
	const char *refused;
} QmdBuilder;

static uint64_t QmdMask(uint32_t width)
{
	return(width >= 64u ? UINT64_MAX : ((uint64_t)1 << width) - 1u);
}

static void QmdStore(SparkNvrmQmd *qmd, uint32_t high, uint32_t low, uint64_t value)
{
	uint32_t bit;
	for (bit = low; bit <= high; bit++)
	{
		const uint32_t word = bit / 32u,shift = bit % 32u;
		qmd->word[word] = (qmd->word[word] & ~(1u << shift)) | ((uint32_t)((value >> (bit - low)) & 1u) << shift);
	}
}

static void QmdPut(QmdBuilder *builder, uint32_t high, uint32_t low, uint64_t value, const char *field)
{
	if ( builder->refused != 0 )
		return;
	if ( high < low || high >= QMD_BITS || high - low >= 64u || (value & ~QmdMask(high - low + 1u)) != 0u )
	{
		builder->refused = field;
		return;
	}
	QmdStore(builder->qmd,high,low,value);
}

SparkStatus SparkNvrmQmdWrite(SparkNvrmQmd *qmd, uint32_t high, uint32_t low, uint64_t value, const char *owner)
{
	QmdBuilder builder = {qmd,0};
	if ( qmd == 0 )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,"needs a launch descriptor");
		SPARK_FAIL(status);
	}
	QmdPut(&builder,high,low,value,"field");
	if ( builder.refused != 0 )
	{
		char text[96];
		snprintf(text,sizeof(text),"%#llx does not fit launch descriptor bits %u:%u",(unsigned long long)value,high,low);
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,text);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

uint64_t SparkNvrmQmdRead(const SparkNvrmQmd *qmd, uint32_t high, uint32_t low)
{
	uint64_t value = 0u;
	uint32_t bit;
	if ( qmd == 0 || high < low || high >= QMD_BITS || high - low >= 64u )
		return(0u);
	for (bit = high + 1u; bit > low; bit--)
		value = (value << 1) | ((qmd->word[(bit - 1u) / 32u] >> ((bit - 1u) % 32u)) & 1u);
	return(value);
}

static uint32_t QmdSharedConfig(uint32_t shared_bytes)
{
	static const uint32_t choices[] = {32u, 64u, SPARK_NVRM_SHARED_CONFIG_KIB_MAX};
	uint32_t index;
	for (index = 0u; index < sizeof(choices) / sizeof(choices[0]); index++)
		if ( choices[index] * 1024u >= shared_bytes )
			return(choices[index] * 1024u / 4096u + 1u);
	return(0u);
}

static const char *QmdShapeProblem(const SparkNvrmLaunchShape *shape)
{
	uint32_t index;
	if ( shape->grid[0] == 0u || shape->grid[1] == 0u || shape->grid[2] == 0u || shape->block[0] == 0u || shape->block[1] == 0u || shape->block[2] == 0u )
		return("a launch needs at least one group of one thread");
	if ( shape->grid[0] > QMD_GRID_WIDTH_MAX || shape->grid[1] > QMD_GRID_EXTENT_MAX || shape->grid[2] > QMD_GRID_EXTENT_MAX )
		return("the grid is larger than a launch allows");
	if ( shape->block[2] > QMD_BLOCK_DEPTH_MAX || (uint64_t)shape->block[0] * shape->block[1] * shape->block[2] > QMD_BLOCK_THREADS_MAX )
		return("the group has more threads than a launch allows");
	if ( (shape->program_address & 0xfu) != 0u || shape->program_address == 0u )
		return("the program address is not 16-byte aligned");
	if ( QmdSharedConfig(shape->shared_bytes) == 0u )
		return("the group needs more shared memory than an SM configuration offers");
	if ( (shape->local_bytes & 0xfu) != 0u )
		return("the per-thread local memory size is not a multiple of 16 bytes");
	if ( (shape->bank_mask & 1u) == 0u || shape->bank_mask >> SPARK_NVRM_CUBIN_BANKS != 0u )
		return("a launch needs constant bank 0 and at most 8 banks");
	for (index = 0u; index < SPARK_NVRM_CUBIN_BANKS; index++)
		if ( (shape->bank_mask >> index & 1u) != 0u && ((shape->bank_address[index] & 0x3fu) != 0u || shape->bank_address[index] == 0u) )
			return("a constant bank address is not 64-byte aligned");
	if ( shape->release_address != 0u && (shape->release_address & (shape->release_timestamp != 0u ? 0xfu : 0x7u)) != 0u )
		return("the release semaphore is not aligned to its structure");
	if ( shape->cluster[0] != 0u && (shape->cluster[1] == 0u || shape->cluster[2] == 0u || shape->grid[0] % shape->cluster[0] != 0u ||
		shape->grid[1] % shape->cluster[1] != 0u || shape->grid[2] % shape->cluster[2] != 0u) )
		return("a cluster launch's grid is a whole number of clusters on every axis");
	if ( shape->next_qmd_address != 0u && ((shape->next_qmd_address & (SPARK_NVRM_QMD_ALIGN - 1u)) != 0u || (shape->next_qmd_address >> 40) != 0u) )
		return("a dependent launch descriptor address is 256-byte aligned and below 2^40");
	return(0);
}

static void QmdBanks(QmdBuilder *builder, const SparkNvrmLaunchShape *shape)
{
	uint32_t index;
	QMD_PUT(builder,NVCEC0_QMDV05_00_CONSTANT_BUFFER_INVALIDATE(0),NVCEC0_QMDV05_00_CONSTANT_BUFFER_INVALIDATE_TRUE);
	for (index = 0u; index < SPARK_NVRM_CUBIN_BANKS; index++)
	{
		const uint64_t shifted = shape->bank_address[index] >> 6;
		if ( (shape->bank_mask >> index & 1u) == 0u )
			continue;
		QMD_PUT(builder,NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(index),shifted & 0xffffffffu);
		QMD_PUT(builder,NVCEC0_QMDV05_00_CONSTANT_BUFFER_ADDR_UPPER_SHIFTED6(index),shifted >> 32);
		QMD_PUT(builder,NVCEC0_QMDV05_00_CONSTANT_BUFFER_SIZE_SHIFTED4(index),((uint64_t)shape->bank_bytes[index] + 15u) >> 4);
		QMD_PUT(builder,NVCEC0_QMDV05_00_CONSTANT_BUFFER_VALID(index),NVCEC0_QMDV05_00_CONSTANT_BUFFER_VALID_TRUE);
	}
}

static void QmdProgram(QmdBuilder *builder, const SparkNvrmLaunchShape *shape)
{
	const uint64_t program = shape->program_address >> 4,prefetch = shape->program_address >> 8;
	const uint64_t prefetch_size = shape->program_bytes >> 8 < QMD_PREFETCH_SIZE_MAX ? shape->program_bytes >> 8 : QMD_PREFETCH_SIZE_MAX;
	QMD_PUT(builder,NVCEC0_QMDV05_00_PROGRAM_ADDRESS_LOWER_SHIFTED4,program & 0xffffffffu);
	QMD_PUT(builder,NVCEC0_QMDV05_00_PROGRAM_ADDRESS_UPPER_SHIFTED4,program >> 32);
	QMD_PUT(builder,NVCEC0_QMDV05_00_PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED,prefetch & 0xffffffffu);
	QMD_PUT(builder,NVCEC0_QMDV05_00_PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED,prefetch >> 32);
	QMD_PUT(builder,NVCEC0_QMDV05_00_PROGRAM_PREFETCH_SIZE,prefetch_size);
	QMD_PUT(builder,NVCEC0_QMDV05_00_SASS_VERSION,shape->sass_version);
	QMD_PUT(builder,NVCEC0_QMDV05_00_REGISTER_COUNT,shape->registers);
	QMD_PUT(builder,NVCEC0_QMDV05_00_BARRIER_COUNT,shape->barriers);
	QMD_PUT(builder,NVCEC0_QMDV05_00_SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4,shape->local_bytes >> 4);
}

static void QmdShape(QmdBuilder *builder, const SparkNvrmLaunchShape *shape)
{
	const uint32_t config = QmdSharedConfig(shape->shared_bytes);
	const uint32_t target = shape->resident_shared_bytes > shape->shared_bytes && QmdSharedConfig(shape->resident_shared_bytes) != 0u ?
		QmdSharedConfig(shape->resident_shared_bytes) : QmdSharedConfig(SPARK_NVRM_SHARED_CONFIG_KIB_MAX * 1024u);
	const uint32_t wanted = shape->resident_shared_bytes > shape->shared_bytes ? target : config;
	const uint32_t clustered = shape->cluster[0] != 0u ? 1u : 0u;
	QMD_PUT(builder,NVCEC0_QMDV05_00_GRID_WIDTH,clustered != 0u ? shape->grid[0] / shape->cluster[0] : shape->grid[0]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_GRID_HEIGHT,clustered != 0u ? shape->grid[1] / shape->cluster[1] : shape->grid[1]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_GRID_DEPTH,clustered != 0u ? shape->grid[2] / shape->cluster[2] : shape->grid[2]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_CTA_THREAD_DIMENSION0,shape->block[0]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_CTA_THREAD_DIMENSION1,shape->block[1]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_CTA_THREAD_DIMENSION2,shape->block[2]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_SHARED_MEMORY_SIZE_SHIFTED7,((uint64_t)shape->shared_bytes + 127u) >> 7);
	QMD_PUT(builder,NVCEC0_QMDV05_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,config);
	QMD_PUT(builder,NVCEC0_QMDV05_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,wanted);
	QMD_PUT(builder,NVCEC0_QMDV05_00_MAX_SM_CONFIG_SHARED_MEM_SIZE,QMD_SHARED_CONFIG_MAX);
}

static void QmdRelease(QmdBuilder *builder, const SparkNvrmLaunchShape *shape)
{
	if ( shape->release_address == 0u )
		return;
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_ADDR_LOWER,shape->release_address & 0xffffffffu);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_ADDR_UPPER,shape->release_address >> 32);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_PAYLOAD_LOWER,shape->release_payload & 0xffffffffu);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_SEMAPHORE0_PAYLOAD_UPPER,shape->release_payload >> 32);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_ENABLE(0),NVCEC0_QMDV05_00_RELEASE_ENABLE_TRUE);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_MEMBAR_TYPE(0),NVCEC0_QMDV05_00_RELEASE_MEMBAR_TYPE_FE_SYSMEMBAR);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_PAYLOAD64B(0),NVCEC0_QMDV05_00_RELEASE_PAYLOAD64B_TRUE);
	QMD_PUT(builder,NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE(0),shape->release_timestamp != 0u ?
		NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE_SEMAPHORE_FOUR_WORDS : NVCEC0_QMDV05_00_RELEASE_STRUCTURE_SIZE_SEMAPHORE_TWO_WORDS);
}

static void QmdCluster(QmdBuilder *builder, const SparkNvrmLaunchShape *shape)
{
	if ( shape->cluster[0] == 0u )
		return;
	QMD_PUT(builder,NVCEC0_QMDV05_00_QMD_TYPE,NVCEC0_QMDV05_00_QMD_TYPE_GRID_GPC_CGA);
	QMD_PUT(builder,NVCEC0_QMDV05_00_GPC_CGA_WIDTH,shape->cluster[0]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_GPC_CGA_HEIGHT,shape->cluster[1]);
	QMD_PUT(builder,NVCEC0_QMDV05_00_GPC_CGA_DEPTH,shape->cluster[2]);
}

static void QmdLatch(QmdBuilder *builder, const SparkNvrmLaunchShape *shape)
{
	if ( shape->latch_arrive != 0u )
	{
		QMD_PUT(builder,NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_VALID,NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_VALID_TRUE);
		QMD_PUT(builder,NVCEC0_QMDV05_00_ARRIVE_AT_LATCH_ID,shape->latch_arrive);
	}
	if ( shape->next_qmd_address != 0u )
	{
		QMD_PUT(builder,NVCEC0_QMDV05_00_DEPENDENT_QMD0_ENABLE,NVCEC0_QMDV05_00_DEPENDENT_QMD0_ENABLE_TRUE);
		QMD_PUT(builder,NVCEC0_QMDV05_00_DEPENDENT_QMD0_ACTION,NVCEC0_QMDV05_00_DEPENDENT_QMD0_ACTION_QMD_SCHEDULE);
		QMD_PUT(builder,NVCEC0_QMDV05_00_DEPENDENT_QMD0_PREFETCH,NVCEC0_QMDV05_00_DEPENDENT_QMD0_PREFETCH_TRUE);
		QMD_PUT(builder,NVCEC0_QMDV05_00_DEPENDENT_QMD0_POINTER,shape->next_qmd_address >> 8);
	}
	if ( shape->latch_wait == 0u )
		return;
	QMD_PUT(builder,NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID,NVCEC0_QMDV05_00_WAIT_ON_LATCH_VALID_TRUE);
	QMD_PUT(builder,NVCEC0_QMDV05_00_WAIT_ON_LATCH_ID,shape->latch_wait);
	QMD_PUT(builder,NVCEC0_QMDV05_00_HOLD_MEMBAR_UNTIL_LATCH_ACQUIRE,NVCEC0_QMDV05_00_HOLD_MEMBAR_UNTIL_LATCH_ACQUIRE_TRUE);
	QMD_PUT(builder,NVCEC0_QMDV05_00_LATCH_ACQUIRE_INVALIDATE_SHADER_DATA_CACHE,1u);
	QMD_PUT(builder,NVCEC0_QMDV05_00_LATCH_ACQUIRE_INVALIDATE_TEXTURE_DATA_CACHE,1u);
}

SparkStatus SparkNvrmQmdBuild(const SparkNvrmLaunchShape *shape, SparkNvrmQmd *qmd, const char *owner)
{
	QmdBuilder builder = {qmd,0};
	const char *problem;
	if ( shape == 0 || qmd == 0 )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,"needs a launch shape and a descriptor");
		SPARK_FAIL(status);
	}
	problem = QmdShapeProblem(shape);
	if ( problem != 0 )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,problem);
		SPARK_FAIL(status);
	}
	memset(qmd,0,sizeof(*qmd));
	QMD_PUT(&builder,NVCEC0_QMDV05_00_QMD_MAJOR_VERSION,5u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_QMD_MINOR_VERSION,0u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_QMD_TYPE,NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_QMD_GROUP_ID,QMD_GROUP_ALL);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_INVALIDATE_TEXTURE_HEADER_CACHE,1u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_INVALIDATE_TEXTURE_SAMPLER_CACHE,1u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_INVALIDATE_TEXTURE_DATA_CACHE,1u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_INVALIDATE_SHADER_DATA_CACHE,1u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_API_VISIBLE_CALL_LIMIT,1u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_SAMPLER_INDEX,1u);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_CWD_MEMBAR_TYPE,NVCEC0_QMDV05_00_CWD_MEMBAR_TYPE_L1_MEMBAR);
	QMD_PUT(&builder,NVCEC0_QMDV05_00_CWD_MEMBAR_TASK_CHASING_ENABLE,NVCEC0_QMDV05_00_CWD_MEMBAR_TASK_CHASING_ENABLE_TRUE);
	QmdProgram(&builder,shape);
	QmdShape(&builder,shape);
	QmdBanks(&builder,shape);
	QmdRelease(&builder,shape);
	QmdCluster(&builder,shape);
	QmdLatch(&builder,shape);
	if ( builder.refused != 0 )
	{
		char text[128];
		snprintf(text,sizeof(text),"the launch value for %s does not fit its field",builder.refused + strlen("NVCEC0_QMDV05_00_"));
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,text);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

static uint32_t Bank0Reciprocal(uint32_t size)
{
	const float reciprocal = 1.0f / (float)size;
	uint32_t bits;
	memcpy(&bits,&reciprocal,sizeof(bits));
	return(bits);
}

static void Bank0Word(uint8_t *bank, uint32_t word, uint32_t value)
{
	uint32_t index;
	for (index = 0u; index < 4u; index++)
		bank[word * 4u + index] = (uint8_t)(value >> (8u * index));
}

_Static_assert(SPARK_NVRM_BANK0_WORD_BLOCK_X > BANK0_WINDOW_WORD_LOCAL + 1u && SPARK_NVRM_BANK0_WORD_GRID_Z < BANK0_WORD_LAST,
	"the launch-shape words lie in the driver area, clear of the window and last words");
_Static_assert(SPARK_NVRM_BANK0_WORD_PARAMETERS > BANK0_WINDOW_WORD_LOCAL + 1u && SPARK_NVRM_BANK0_WORD_PARAMETERS + 1u < SPARK_NVRM_BANK0_WORD_BLOCK_X,
	"the parameters' address lies in the driver area, clear of the window and launch-shape words");

_Static_assert(SPARK_NVRM_BANK0_WORD_CLUSTER_DIM + 2u < SPARK_NVRM_BANK0_WORD_CLUSTER_RECIPROCAL &&
	SPARK_NVRM_BANK0_WORD_CLUSTER_RECIPROCAL + 2u < SPARK_NVRM_BANK0_WORD_CLUSTER_COUNT &&
	SPARK_NVRM_BANK0_WORD_CLUSTER_COUNT + 2u < SPARK_NVRM_BANK0_WORD_CLUSTER_RANKS && SPARK_NVRM_BANK0_WORD_CLUSTER_RANKS < BANK0_WINDOW_WORD_SHARED,
	"the cluster words lie in the driver area below the window words");

static uint32_t Bank0ClusterWord(uint32_t word)
{
	return((word >= SPARK_NVRM_BANK0_WORD_CLUSTER_DIM && word < SPARK_NVRM_BANK0_WORD_CLUSTER_DIM + 3u) ||
		(word >= SPARK_NVRM_BANK0_WORD_CLUSTER_RECIPROCAL && word < SPARK_NVRM_BANK0_WORD_CLUSTER_RECIPROCAL + 3u) ||
		(word >= SPARK_NVRM_BANK0_WORD_CLUSTER_COUNT && word < SPARK_NVRM_BANK0_WORD_CLUSTER_COUNT + 3u) || word == SPARK_NVRM_BANK0_WORD_CLUSTER_RANKS ?
		1u : 0u);
}

static const uint32_t bank0_dimension_words[6] = {SPARK_NVRM_BANK0_WORD_BLOCK_X, SPARK_NVRM_BANK0_WORD_BLOCK_Y, SPARK_NVRM_BANK0_WORD_BLOCK_Z,
	SPARK_NVRM_BANK0_WORD_GRID_X, SPARK_NVRM_BANK0_WORD_GRID_Y, SPARK_NVRM_BANK0_WORD_GRID_Z};

int SparkNvrmBank0WordClaimed(uint32_t word)
{
	uint32_t index;
	if ( word == BANK0_WINDOW_WORD_SHARED || word == BANK0_WINDOW_WORD_SHARED + 1u || word == BANK0_WINDOW_WORD_LOCAL ||
		word == BANK0_WINDOW_WORD_LOCAL + 1u || word == BANK0_WORD_LAST || word == SPARK_NVRM_BANK0_WORD_PARAMETERS ||
		word == SPARK_NVRM_BANK0_WORD_PARAMETERS + 1u || Bank0ClusterWord(word) != 0u )
		return(1);
	for (index = 0u; index < sizeof(bank0_dimension_words) / sizeof(bank0_dimension_words[0]); index++)
		if ( bank0_dimension_words[index] == word )
			return(1);
	return(0);
}

SparkStatus SparkNvrmBank0Build(const SparkNvrmCubinKernel *kernel, uint64_t shared_window, uint64_t local_window,
	const uint32_t grid[3], const uint32_t block[3], const uint32_t cluster[3], const void *arguments, uint32_t argument_bytes, uint8_t *bank,
	uint64_t bank_address, uint32_t bank_bytes, const char *owner)
{
	const uint32_t single[3] = {1u, 1u, 1u};
	const uint32_t *shape = cluster != 0 && cluster[0] != 0u ? cluster : single;
	uint32_t index;
	uint64_t parameters;
	const char *problem = 0;
	if ( kernel == 0 || bank == 0 || grid == 0 || block == 0 || (arguments == 0 && argument_bytes != 0u) )
		problem = "needs a kernel, its launch shape, its arguments and a constant bank";
	else if ( bank_address == 0u || (bank_address & 0x3fu) != 0u )
		problem = "needs the GPU address of constant bank 0, aligned to 64 bytes";
	else if ( shape[0] == 0u || shape[1] == 0u || shape[2] == 0u || grid[0] % shape[0] != 0u || grid[1] % shape[1] != 0u || grid[2] % shape[2] != 0u )
		problem = "a cluster has three nonzero sizes that divide the grid";
	else if ( kernel->parameter_offset < SPARK_NVRM_BANK0_DRIVER_BYTES )
		problem = "the kernel's parameters overlap the driver words of constant bank 0";
	else if ( argument_bytes != kernel->parameter_bytes )
		problem = "the argument block is not the size of the kernel's parameters";
	else if ( (uint64_t)kernel->parameter_offset + kernel->parameter_bytes > bank_bytes )
		problem = "the constant bank is smaller than the kernel's parameters";
	if ( problem != 0 )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"kernel_launch",owner,0,problem);
		SPARK_FAIL(status);
	}
	memset(bank,0,bank_bytes);
	Bank0Word(bank,BANK0_WINDOW_WORD_SHARED,(uint32_t)shared_window);
	Bank0Word(bank,BANK0_WINDOW_WORD_SHARED + 1u,(uint32_t)(shared_window >> 32));
	Bank0Word(bank,BANK0_WINDOW_WORD_LOCAL,(uint32_t)local_window);
	Bank0Word(bank,BANK0_WINDOW_WORD_LOCAL + 1u,(uint32_t)(local_window >> 32));
	Bank0Word(bank,BANK0_WORD_LAST,BANK0_LAST_VALUE);
	parameters = bank_address + kernel->parameter_offset;
	Bank0Word(bank,SPARK_NVRM_BANK0_WORD_PARAMETERS,(uint32_t)parameters);
	Bank0Word(bank,SPARK_NVRM_BANK0_WORD_PARAMETERS + 1u,(uint32_t)(parameters >> 32));
	for (index = 0u; index < 3u; index++)
	{
		Bank0Word(bank,bank0_dimension_words[index],block[index]);
		Bank0Word(bank,bank0_dimension_words[3u + index],grid[index]);
		Bank0Word(bank,SPARK_NVRM_BANK0_WORD_CLUSTER_DIM + index,shape[index]);
		Bank0Word(bank,SPARK_NVRM_BANK0_WORD_CLUSTER_RECIPROCAL + index,Bank0Reciprocal(shape[index]));
		Bank0Word(bank,SPARK_NVRM_BANK0_WORD_CLUSTER_COUNT + index,grid[index] / shape[index]);
	}
	Bank0Word(bank,SPARK_NVRM_BANK0_WORD_CLUSTER_RANKS,shape[0] * shape[1] * shape[2]);
	if ( argument_bytes != 0u )
		memcpy(bank + kernel->parameter_offset,arguments,argument_bytes);
	return(SPARK_STATUS_OK);
}
