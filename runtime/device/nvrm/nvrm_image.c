#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_image.h"
#include "runtime/device/nvrm/nvrm_launch.h"

static uint32_t ImageOrderWords(const SparkNvrmStep *steps, uint32_t index)
{
	if ( index == 0u )
		return(0u);
	if ( steps[index].kind == SPARK_NVRM_STEP_LAUNCH && steps[index - 1u].kind == SPARK_NVRM_STEP_LAUNCH )
		return(SPARK_NVRM_WAIT_IDLE_WORDS);
	return(SPARK_NVRM_SEMAPHORE_METHOD_WORDS);
}

static uint32_t ImageStepWords(const SparkNvrmStep *steps, uint32_t index)
{
	const SparkNvrmStep *step = &steps[index];
	const uint32_t order = ImageOrderWords(steps,index);
	if ( step->kind == SPARK_NVRM_STEP_LAUNCH )
	{
		if ( step->chained == SPARK_NVRM_CHAIN_QMD )
			return(0u);
		return(step->chained == SPARK_NVRM_CHAIN_LATCH ? SPARK_NVRM_LAUNCH_WORDS : order + SPARK_NVRM_LAUNCH_WORDS);
	}
	if ( step->kind == SPARK_NVRM_STEP_AWAIT )
		return(order + 2u * SPARK_NVRM_SEMAPHORE_METHOD_WORDS);
	return(order + 3u * SPARK_NVRM_SEMAPHORE_METHOD_WORDS);
}

void SparkNvrmImageChain(SparkNvrmStep *steps, uint32_t count, uint32_t segment)
{
	uint32_t index,run = 0u;
	if ( count != 0u )
		steps[0].chained = SPARK_NVRM_CHAIN_NONE;
	for ( index = 1u; index < count; index++ )
	{
		steps[index].chained = SPARK_NVRM_CHAIN_NONE;
		if ( steps[index].kind == SPARK_NVRM_STEP_LAUNCH && steps[index - 1u].kind == SPARK_NVRM_STEP_LAUNCH && run + 1u < segment )
		{
			steps[index].chained = steps[index].dependent != 0u && steps[index - 1u].chained != SPARK_NVRM_CHAIN_QMD ?
				SPARK_NVRM_CHAIN_LATCH : SPARK_NVRM_CHAIN_QMD;
			run++;
		}
		else
			run = 0u;
	}
}

uint32_t SparkNvrmImageWords(const SparkNvrmStep *steps, uint32_t count)
{
	uint32_t index,words = SPARK_NVRM_SEMAPHORE_METHOD_WORDS;
	for ( index = 0u; index < count; index++ )
		words += ImageStepWords(steps,index);
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
		if ( step->kind == SPARK_NVRM_STEP_LAUNCH && step->chained == SPARK_NVRM_CHAIN_QMD )
			continue;
		if ( index != 0u && step->chained == SPARK_NVRM_CHAIN_NONE )
		{
			if ( step->kind == SPARK_NVRM_STEP_LAUNCH && steps[index - 1u].kind == SPARK_NVRM_STEP_LAUNCH )
			{
				SparkNvrmWaitIdleWords(words + at);
				at += SPARK_NVRM_WAIT_IDLE_WORDS;
			}
			else
				at = ImageSemaphore(words,at,semaphore,index,SPARK_NVRM_SEMAPHORE_ACQUIRE);
		}
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
