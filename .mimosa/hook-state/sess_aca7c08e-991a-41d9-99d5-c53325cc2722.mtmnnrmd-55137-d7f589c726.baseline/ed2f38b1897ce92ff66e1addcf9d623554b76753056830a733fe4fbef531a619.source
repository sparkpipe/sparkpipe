#include "sparkpipe/spark_glm52_chat_template.h"

#include <string.h>

static const char SparkGlm52ChatTemplatePrefix[] = "[gMASK]<sop>";
static const char SparkGlm52ChatTemplateReasoning[] =
	"<|system|>Reasoning Effort: ";
static const char SparkGlm52ChatTemplateSystem[] = "<|system|>";
static const char SparkGlm52ChatTemplateUser[] = "<|user|>";
static const char SparkGlm52ChatTemplateAssistant[] = "<|assistant|>";
static const char SparkGlm52ChatTemplateObservation[] = "<|observation|>";
static const char SparkGlm52ChatTemplateThink[] = "<think>";
static const char SparkGlm52ChatTemplateEmptyThink[] = "<think></think>";
static const char SparkGlm52ChatTemplateToolResponse[] = "<tool_response>";
static const char SparkGlm52ChatTemplateToolResponseEnd[] = "</tool_response>";
static const char SparkGlm52ChatTemplateReasoningHigh[] = "High";
static const char SparkGlm52ChatTemplateReasoningMax[] = "Max";

static SparkStatus SparkGlm52ChatTemplateAppendLiteral(
	SparkGlm52ChatTemplateWriter *writer,
	const char *text)
{
	return SparkGlm52ChatTemplateAppend(
		writer,
		text,
		(uint32_t)strlen(text));
}

SparkStatus SparkGlm52ChatTemplateInitializeWriter(
	SparkGlm52ChatTemplateWriter *writer,
	char *text,
	uint32_t text_capacity,
	uint32_t text_bytes)
{
	if ( writer == 0 || text_bytes > text_capacity )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( text != 0 && text_bytes >= text_capacity )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( text == 0 && text_bytes != 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	writer->text = text;
	writer->text_capacity = text_capacity;
	writer->text_bytes = text_bytes;
	if ( text != 0 )
		text[text_bytes] = '\0';
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ChatTemplateAppend(
	SparkGlm52ChatTemplateWriter *writer,
	const char *text,
	uint32_t text_bytes)
{
	uint32_t next_bytes;

	if ( writer == 0 || (text == 0 && text_bytes != 0u) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( text_bytes > (UINT32_MAX - writer->text_bytes) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	next_bytes = writer->text_bytes + text_bytes;
	if ( next_bytes > writer->text_capacity ||
		(writer->text != 0 && next_bytes >= writer->text_capacity) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( writer->text != 0 && text_bytes != 0u )
		memcpy(writer->text + writer->text_bytes, text, text_bytes);
	writer->text_bytes = next_bytes;
	if ( writer->text != 0 )
		writer->text[next_bytes] = '\0';
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ChatTemplateBegin(
	SparkGlm52ChatTemplateWriter *writer,
	const char *reasoning_effort,
	uint32_t flags)
{
	SparkStatus status;

	if ( writer == 0 || (flags & ~SPARK_GLM52_CHAT_TEMPLATE_KNOWN_FLAGS) != 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( reasoning_effort != 0 &&
		(strcmp(reasoning_effort, "high") == 0 ||
		 strcmp(reasoning_effort, "High") == 0) )
		reasoning_effort = SparkGlm52ChatTemplateReasoningHigh;
	else
		reasoning_effort = SparkGlm52ChatTemplateReasoningMax;
	status = SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplatePrefix);
	if ( status != SPARK_STATUS_OK ||
		(flags & SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING) == 0u )
		return status;
	status = SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplateReasoning);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGlm52ChatTemplateAppendLiteral(writer, reasoning_effort);
}

SparkStatus SparkGlm52ChatTemplateBeginMessage(
	SparkGlm52ChatTemplateWriter *writer,
	SparkGlm52ChatTemplateRole role)
{
	SparkStatus status;

	if ( role == SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM )
		return SparkGlm52ChatTemplateAppendLiteral(
			writer,
			SparkGlm52ChatTemplateSystem);
	if ( role == SPARK_GLM52_CHAT_TEMPLATE_ROLE_USER )
		return SparkGlm52ChatTemplateAppendLiteral(
			writer,
			SparkGlm52ChatTemplateUser);
	if ( role == SPARK_GLM52_CHAT_TEMPLATE_ROLE_ASSISTANT )
	{
		status = SparkGlm52ChatTemplateAppendLiteral(
			writer,
			SparkGlm52ChatTemplateAssistant);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkGlm52ChatTemplateAppendLiteral(
			writer,
			SparkGlm52ChatTemplateEmptyThink);
	}
	if ( role != SPARK_GLM52_CHAT_TEMPLATE_ROLE_TOOL )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplateObservation);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplateToolResponse);
}

SparkStatus SparkGlm52ChatTemplateEndMessage(
	SparkGlm52ChatTemplateWriter *writer,
	SparkGlm52ChatTemplateRole role)
{
	if ( writer == 0 || role < SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM ||
		role > SPARK_GLM52_CHAT_TEMPLATE_ROLE_TOOL )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( role != SPARK_GLM52_CHAT_TEMPLATE_ROLE_TOOL )
		return SPARK_STATUS_OK;
	return SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplateToolResponseEnd);
}

SparkStatus SparkGlm52ChatTemplateFinish(
	SparkGlm52ChatTemplateWriter *writer,
	uint32_t flags)
{
	SparkStatus status;

	if ( writer == 0 || (flags & ~SPARK_GLM52_CHAT_TEMPLATE_KNOWN_FLAGS) != 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( (flags & SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT) == 0u )
		return SPARK_STATUS_OK;
	status = SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplateAssistant);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( (flags & SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING) != 0u )
		return SparkGlm52ChatTemplateAppendLiteral(
			writer,
			SparkGlm52ChatTemplateThink);
	return SparkGlm52ChatTemplateAppendLiteral(
		writer,
		SparkGlm52ChatTemplateEmptyThink);
}

SparkStatus SparkGlm52ChatTemplateRenderSimple(
	SparkGlm52ChatTemplateWriter *writer,
	const char *prompt,
	uint32_t prompt_bytes,
	const char *system_prompt,
	uint32_t system_prompt_bytes,
	const char *reasoning_effort,
	uint32_t flags)
{
	SparkStatus status;

	if ( writer == 0 || prompt == 0 ||
		(system_prompt == 0 && system_prompt_bytes != 0u) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52ChatTemplateBegin(writer, reasoning_effort, flags);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( system_prompt_bytes != 0u )
	{
		status = SparkGlm52ChatTemplateBeginMessage(
			writer,
			SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkGlm52ChatTemplateAppend(
			writer,
			system_prompt,
			system_prompt_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	status = SparkGlm52ChatTemplateBeginMessage(
		writer,
		SPARK_GLM52_CHAT_TEMPLATE_ROLE_USER);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkGlm52ChatTemplateAppend(writer, prompt, prompt_bytes);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGlm52ChatTemplateFinish(writer, flags);
}
