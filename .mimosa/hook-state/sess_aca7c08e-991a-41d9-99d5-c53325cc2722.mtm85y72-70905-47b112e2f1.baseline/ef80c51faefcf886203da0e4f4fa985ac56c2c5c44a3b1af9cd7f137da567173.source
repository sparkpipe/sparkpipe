#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT 0x00000001u
#define SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING 0x00000002u
#define SPARK_GLM52_CHAT_TEMPLATE_KNOWN_FLAGS \
	(SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT | \
	 SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING)

typedef enum SparkGlm52ChatTemplateRole
{
	SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM = 0,
	SPARK_GLM52_CHAT_TEMPLATE_ROLE_USER,
	SPARK_GLM52_CHAT_TEMPLATE_ROLE_ASSISTANT,
	SPARK_GLM52_CHAT_TEMPLATE_ROLE_TOOL
} SparkGlm52ChatTemplateRole;

typedef struct SparkGlm52ChatTemplateWriter
{
	char *text;
	uint32_t text_capacity;
	uint32_t text_bytes;
} SparkGlm52ChatTemplateWriter;

SparkStatus SparkGlm52ChatTemplateInitializeWriter(
	SparkGlm52ChatTemplateWriter *writer,
	char *text,
	uint32_t text_capacity,
	uint32_t text_bytes);

SparkStatus SparkGlm52ChatTemplateAppend(
	SparkGlm52ChatTemplateWriter *writer,
	const char *text,
	uint32_t text_bytes);

SparkStatus SparkGlm52ChatTemplateBegin(
	SparkGlm52ChatTemplateWriter *writer,
	const char *reasoning_effort,
	uint32_t flags);

SparkStatus SparkGlm52ChatTemplateBeginMessage(
	SparkGlm52ChatTemplateWriter *writer,
	SparkGlm52ChatTemplateRole role);

SparkStatus SparkGlm52ChatTemplateEndMessage(
	SparkGlm52ChatTemplateWriter *writer,
	SparkGlm52ChatTemplateRole role);

SparkStatus SparkGlm52ChatTemplateFinish(
	SparkGlm52ChatTemplateWriter *writer,
	uint32_t flags);

SparkStatus SparkGlm52ChatTemplateRenderSimple(
	SparkGlm52ChatTemplateWriter *writer,
	const char *prompt,
	uint32_t prompt_bytes,
	const char *system_prompt,
	uint32_t system_prompt_bytes,
	const char *reasoning_effort,
	uint32_t flags);

#ifdef __cplusplus
}
#endif
