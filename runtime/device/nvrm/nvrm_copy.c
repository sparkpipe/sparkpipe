#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "classes/dma-copy/clcab5.h"

#define COPY_SUBCHANNEL 4u
#define COPY_LINE_BYTES (1ull << 31)
#define COPY_WORD_BYTES 4u
#define COPY_BYTE_PATTERN 0x01010101u

_Static_assert(COPY_LINE_BYTES <= 0xffffffffull && COPY_LINE_BYTES % COPY_WORD_BYTES == 0u,
	"a line's bytes stay below 2^32 and whole four-byte elements");

#define COPY_TRY(call) \
	do { \
		const SparkStatus copy_status = (call); \
		if ( copy_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(copy_status); \
	} while (0)

static uint32_t CopyLaunch(uint32_t remap)
{
	return((uint32_t)(SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_DATA_TRANSFER_TYPE,NVCAB5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED) |
		SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_SRC_MEMORY_LAYOUT,NVCAB5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH) |
		SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_DST_MEMORY_LAYOUT,NVCAB5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH) |
		SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_REMAP_ENABLE,remap != 0u ? NVCAB5_LAUNCH_DMA_REMAP_ENABLE_TRUE : NVCAB5_LAUNCH_DMA_REMAP_ENABLE_FALSE)));
}

static SparkStatus CopyRelease(SparkNvrmChannel *channel, const char *owner)
{
	const uint64_t address = channel->fence.address,value = channel->fence_value + 1u;
	uint32_t words[4];
	words[0] = (uint32_t)SPARK_NVRM_VALUE(NVCAB5_SET_SEMAPHORE_A_UPPER,address >> 32);
	words[1] = (uint32_t)address;
	words[2] = (uint32_t)value;
	words[3] = (uint32_t)(value >> 32);
	COPY_TRY(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_SET_SEMAPHORE_A,words,4u,owner));
	words[0] = (uint32_t)(SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_DATA_TRANSFER_TYPE,NVCAB5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE) |
		SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_FLUSH_ENABLE,NVCAB5_LAUNCH_DMA_FLUSH_ENABLE_TRUE) |
		SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_SEMAPHORE_TYPE,NVCAB5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE) |
		SPARK_NVRM_VALUE(NVCAB5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE,NVCAB5_LAUNCH_DMA_SEMAPHORE_PAYLOAD_SIZE_TWO_WORD));
	return(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_LAUNCH_DMA,words,1u,owner));
}

static SparkStatus CopyFinish(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, SparkStatus status, uint64_t *value, const char *owner)
{
	if ( status == SPARK_STATUS_OK )
		status = CopyRelease(channel,owner);
	if ( status == SPARK_STATUS_OK )
		status = SparkNvrmOpEnd(gpu,channel,SPARK_NVRM_ENGINE_COPY,value,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmOpAbandon(channel);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus CopyLines(SparkNvrmChannel *channel, uint64_t destination, uint64_t source, uint64_t elements, uint32_t element_bytes,
	uint32_t remap, const char *owner)
{
	const uint64_t line_elements = COPY_LINE_BYTES / element_bytes;
	uint64_t done;
	for (done = 0u; done < elements; done += line_elements)
	{
		const uint64_t count = elements - done < line_elements ? elements - done : line_elements;
		const uint64_t from = source + done * element_bytes,to = destination + done * element_bytes;
		uint32_t words[4];
		words[0] = (uint32_t)SPARK_NVRM_VALUE(NVCAB5_OFFSET_IN_UPPER_UPPER,from >> 32);
		words[1] = (uint32_t)from;
		words[2] = (uint32_t)SPARK_NVRM_VALUE(NVCAB5_OFFSET_OUT_UPPER_UPPER,to >> 32);
		words[3] = (uint32_t)to;
		if ( remap != 0u )
			COPY_TRY(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_OFFSET_OUT_UPPER,words + 2,2u,owner));
		else
			COPY_TRY(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_OFFSET_IN_UPPER,words,4u,owner));
		words[0] = (uint32_t)count;
		COPY_TRY(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_LINE_LENGTH_IN,words,1u,owner));
		words[0] = CopyLaunch(remap);
		COPY_TRY(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_LAUNCH_DMA,words,1u,owner));
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmCopy(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t destination, uint64_t source, uint64_t bytes, uint64_t *value,
	const char *owner)
{
	COPY_TRY(SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_COPY,owner));
	return(CopyFinish(gpu,channel,CopyLines(channel,destination,source,bytes,1u,0u,owner),value,owner));
}

static SparkStatus CopyFillPart(SparkNvrmChannel *channel, uint64_t destination, uint32_t pattern, uint64_t elements, uint32_t element_bytes,
	const char *owner)
{
	uint32_t words[3];
	words[0] = pattern;
	words[1] = 0u;
	words[2] = (uint32_t)(SPARK_NVRM_VALUE(NVCAB5_SET_REMAP_COMPONENTS_DST_X,NVCAB5_SET_REMAP_COMPONENTS_DST_X_CONST_A) |
		SPARK_NVRM_VALUE(NVCAB5_SET_REMAP_COMPONENTS_COMPONENT_SIZE,element_bytes == COPY_WORD_BYTES ? NVCAB5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_FOUR :
			NVCAB5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_ONE) |
		SPARK_NVRM_VALUE(NVCAB5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS,NVCAB5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS_ONE) |
		SPARK_NVRM_VALUE(NVCAB5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS,NVCAB5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS_ONE));
	COPY_TRY(SparkNvrmPush(channel,COPY_SUBCHANNEL,NVCAB5_SET_REMAP_CONST_A,words,3u,owner));
	return(CopyLines(channel,destination,0u,elements,element_bytes,1u,owner));
}

SparkStatus SparkNvrmFill(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, uint64_t destination, uint8_t byte, uint64_t bytes, uint64_t *value,
	const char *owner)
{
	const uint64_t misalign = destination % COPY_WORD_BYTES;
	const uint64_t head = misalign == 0u ? 0u : (COPY_WORD_BYTES - misalign < bytes ? COPY_WORD_BYTES - misalign : bytes);
	const uint64_t body = (bytes - head) / COPY_WORD_BYTES * COPY_WORD_BYTES,tail = bytes - head - body;
	SparkStatus status;
	COPY_TRY(SparkNvrmOpBegin(channel,SPARK_NVRM_ENGINE_COPY,owner));
	status = SPARK_STATUS_OK;
	if ( head != 0u )
		status = CopyFillPart(channel,destination,byte,head,1u,owner);
	if ( status == SPARK_STATUS_OK && body != 0u )
		status = CopyFillPart(channel,destination + head,byte * COPY_BYTE_PATTERN,body / COPY_WORD_BYTES,COPY_WORD_BYTES,owner);
	if ( status == SPARK_STATUS_OK && tail != 0u )
		status = CopyFillPart(channel,destination + head + body,byte,tail,1u,owner);
	return(CopyFinish(gpu,channel,status,value,owner));
}
