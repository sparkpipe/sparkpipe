#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_image.h"
#include "runtime/device/nvrm/nvrm_launch.h"

static uint32_t ImageStepWords(const SparkNvrmStep *step, uint32_t index)
{
	const uint32_t order = index != 0u ? SPARK_NVRM_SEMAPHORE_METHOD_WORDS : 0u;
	if ( step->kind == SPARK_NVRM_STEP_LAUNCH )
		return(step->chained != 0u ? SPARK_NVRM_LAUNCH_WORDS : order + SPARK_NVRM_LAUNCH_WORDS);
	if ( step->kind == SPARK_NVRM_STEP_AWAIT )
		return(order + 2u * SPARK_NVRM_SEMAPHORE_METHOD_WORDS);
	return(order + 3u * SPARK_NVRM_SEMAPHORE_METHOD_WORDS);
}

void SparkNvrmImageChain(SparkNvrmStep *steps, uint32_t count, uint32_t segment)
{
	uint32_t index,run = 0u;
	for ( index = 1u; index < count; index++ )
		if ( steps[index].kind == SPARK_NVRM_STEP_LAUNCH && steps[index - 1u].kind == SPARK_NVRM_STEP_LAUNCH && run + 1u < segment )
		{
			steps[index].chained = 1u;
			run++;
		}
		else
			run = 0u;
}

uint32_t SparkNvrmImageWords(const SparkNvrmStep *steps, uint32_t count)
{
	uint32_t index,words = SPARK_NVRM_SEMAPHORE_METHOD_WORDS;
	for ( index = 0u; index < count; index++ )
		words += ImageStepWords(&steps[index],index);
	return(words);
}

static uint32_t ImageSemaphore(uint32_t *words, uint32_t at, uint64_t address, uint64_t value, SparkNvrmSemaphoreOperation operation)
{
	SparkNvrmSemaphoreWords(words + at,address,value,operation);
	return(at + SPARK_NVRM_SEMAPHORE_METHOD_WORDS);
}

void SparkNvrmImageWrite(uint32_t *words, uint64_t semaphore, uint64_t slots, SparkNvrmStep *steps, uint32_t count)
{
	uint32_t index,at = ImageSemaphore(words,0u,semaphore,0u,SPARK_NVRM_SEMAPHORE_RELEASE);
	for ( index = 0u; index < count; index++ )
	{
		SparkNvrmStep *step = &steps[index];
		if ( index != 0u && step->chained == 0u )
			at = ImageSemaphore(words,at,semaphore,index,SPARK_NVRM_SEMAPHORE_ACQUIRE);
		if ( step->kind == SPARK_NVRM_STEP_LAUNCH )
		{
			SparkNvrmLaunchWords(words + at,slots + step->slot);
			at += SPARK_NVRM_LAUNCH_WORDS;
			continue;
		}
		step->payload = at;
		if ( step->kind == SPARK_NVRM_STEP_AWAIT )
			at = ImageSemaphore(words,at,step->address,0u,SPARK_NVRM_SEMAPHORE_ACQUIRE);
		else
		{
			at = ImageSemaphore(words,at,step->address,0u,step->kind == SPARK_NVRM_STEP_RECORD ? SPARK_NVRM_SEMAPHORE_RELEASE_TIME :
				SPARK_NVRM_SEMAPHORE_RELEASE);
			step->second = at;
			at = ImageSemaphore(words,at,step->second_address,0u,step->kind == SPARK_NVRM_STEP_RECORD ? SPARK_NVRM_SEMAPHORE_MAX :
				SPARK_NVRM_SEMAPHORE_ACQUIRE);
		}
		at = ImageSemaphore(words,at,semaphore,(uint64_t)index + 1u,SPARK_NVRM_SEMAPHORE_RELEASE);
	}
}

void SparkNvrmImagePatch(uint32_t *words, uint32_t method, uint64_t value)
{
	words[method + SPARK_NVRM_SEMAPHORE_PAYLOAD_WORD] = (uint32_t)value;
	words[method + SPARK_NVRM_SEMAPHORE_PAYLOAD_WORD + 1u] = (uint32_t)(value >> 32);
}

void SparkNvrmImagePatchAddress(uint32_t *words, uint32_t method, uint64_t address)
{
	words[method + 1u] = (uint32_t)address;
	words[method + 2u] = (uint32_t)(address >> 32);
}

uint64_t SparkNvrmImagePayload(const uint32_t *words, uint32_t method)
{
	return((uint64_t)words[method + SPARK_NVRM_SEMAPHORE_PAYLOAD_WORD] | ((uint64_t)words[method + SPARK_NVRM_SEMAPHORE_PAYLOAD_WORD + 1u] << 32));
}
