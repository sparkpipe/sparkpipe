#ifndef SPARKPIPE_TOOLS_TOKEN_PROMPTS_H
#define SPARKPIPE_TOOLS_TOKEN_PROMPTS_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#define SPARK_TOKEN_PROMPT_NAME_BYTES 64u

typedef struct SparkTokenPrompt
{
	char name[SPARK_TOKEN_PROMPT_NAME_BYTES];
	uint32_t *tokens;
	uint32_t count;
} SparkTokenPrompt;

typedef struct SparkTokenPrompts
{
	SparkTokenPrompt *prompts;
	uint32_t count;
} SparkTokenPrompts;

SparkStatus SparkTokenPromptsLoad(const char *path, SparkTokenPrompts *prompts);
void SparkTokenPromptsFree(SparkTokenPrompts *prompts);

#endif
