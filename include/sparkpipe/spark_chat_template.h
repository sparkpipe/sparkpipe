#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkChatTemplateOutcome
{
	SPARK_CHAT_TEMPLATE_RENDERED = 0,
	SPARK_CHAT_TEMPLATE_NOT_CHAT = 1,
	SPARK_CHAT_TEMPLATE_MISSING = 2,
	SPARK_CHAT_TEMPLATE_INVALID_MESSAGES = 3,
	SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED = 4,
	SPARK_CHAT_TEMPLATE_THINKING_UNSUPPORTED = 5,
	SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY = 6
} SparkChatTemplateOutcome;

SparkChatTemplateOutcome SparkChatTemplateRender(
	const SparkModelResidentChatTemplate *chat_template,
	const SparkJsonDocument *document,
	int32_t root,
	bool thinking,
	char **text,
	uint32_t *text_bytes);
const char *SparkChatTemplateOutcomeCode(SparkChatTemplateOutcome outcome);
const char *SparkChatTemplateOutcomeMessage(SparkChatTemplateOutcome outcome);
SparkStatus SparkChatTemplateResolveStops(
	const SparkModelResidentChatTemplate *chat_template,
	const SparkTokenizerSpecialToken *special_tokens,
	uint32_t special_token_count,
	uint32_t *stop_ids,
	uint32_t *stop_count,
	const char **unresolved_marker);

#ifdef __cplusplus
}
#endif
