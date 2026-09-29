#include "sparkpipe/spark_chat_template.h"
#include "sparkpipe/spark_error_site.h"

#include <stdlib.h>
#include <string.h>

typedef struct SparkChatTemplateText
{
	char *bytes;
	size_t length;
	size_t capacity;
} SparkChatTemplateText;

static int SparkChatTemplateAppend(SparkChatTemplateText *text, const char *piece)
{
	size_t piece_bytes = strlen(piece);
	if ( text->length + piece_bytes + 1u > text->capacity )
	{
		size_t capacity = text->capacity;
		char *grown;
		while ( text->length + piece_bytes + 1u > capacity )
			capacity *= 2u;
		grown = realloc(text->bytes,capacity);
		if ( grown == 0 )
			return(0);
		text->bytes = grown;
		text->capacity = capacity;
	}
	memcpy(text->bytes + text->length,piece,piece_bytes);
	text->length += piece_bytes;
	text->bytes[text->length] = '\0';
	return(1);
}

static SparkChatTemplateOutcome SparkChatTemplateRoleHeader(
	const SparkModelResidentChatTemplate *chat_template,
	const SparkJsonDocument *document,
	int32_t role,
	bool thinking,
	uint32_t leading,
	const char **header)
{
	*header = 0;
	if ( role < 0 || !SparkJsonTokenIsType(document,role,SPARK_JSON_TOKEN_STRING) )
		return(SPARK_CHAT_TEMPLATE_INVALID_MESSAGES);
	if ( SparkJsonStringEquals(document,role,"user") )
		*header = chat_template->user;
	else if ( SparkJsonStringEquals(document,role,"assistant") )
		*header = thinking ? chat_template->assistant_thinking : chat_template->assistant;
	else if ( SparkJsonStringEquals(document,role,"system") )
		*header = thinking && leading != 0u ? chat_template->system_thinking : chat_template->system;
	else if ( SparkJsonStringEquals(document,role,"observation") )
		*header = chat_template->observation;
	return(*header != 0 ? SPARK_CHAT_TEMPLATE_RENDERED : SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED);
}

static const char *SparkChatTemplateRoleSuffix(
	const SparkModelResidentChatTemplate *chat_template,
	const SparkJsonDocument *document,
	int32_t role)
{
	if ( chat_template->assistant_suffix != 0 && role >= 0 &&
		SparkJsonStringEquals(document,role,"assistant") )
		return(chat_template->assistant_suffix);
	return("");
}

SparkChatTemplateOutcome SparkChatTemplateRender(
	const SparkModelResidentChatTemplate *chat_template,
	const SparkJsonDocument *document,
	int32_t root,
	bool thinking,
	char **text,
	uint32_t *text_bytes)
{
	SparkChatTemplateText rendered;
	SparkChatTemplateOutcome outcome = SPARK_CHAT_TEMPLATE_RENDERED;
	int32_t messages;
	uint32_t message_count,message_index;
	uint32_t leading_system = 0u;
	if ( document == 0 || root < 0 || text == 0 || text_bytes == 0 )
		return(SPARK_CHAT_TEMPLATE_INVALID_MESSAGES);
	*text = 0;
	*text_bytes = 0u;
	messages = SparkJsonFindObjectMember(document,root,"messages");
	if ( messages < 0 )
		return(SPARK_CHAT_TEMPLATE_NOT_CHAT);
	if ( chat_template == 0 || chat_template->declared == 0u )
		return(SPARK_CHAT_TEMPLATE_MISSING);
	if ( !SparkJsonTokenIsType(document,messages,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_CHAT_TEMPLATE_INVALID_MESSAGES);
	message_count = SparkJsonGetArrayElementCount(document,messages);
	if ( message_count == 0u )
		return(SPARK_CHAT_TEMPLATE_INVALID_MESSAGES);
	if ( thinking && chat_template->generation_thinking == 0 )
		return(SPARK_CHAT_TEMPLATE_THINKING_UNSUPPORTED);
	rendered.capacity = 4096u;
	rendered.length = 0u;
	rendered.bytes = malloc(rendered.capacity);
	if ( rendered.bytes == 0 || !SparkChatTemplateAppend(&rendered,chat_template->prefix) )
	{
		free(rendered.bytes);
		return(SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY);
	}
	{
		int32_t first = SparkJsonGetArrayElement(document,messages,0u);
		int32_t first_role = first >= 0 && SparkJsonTokenIsType(document,first,SPARK_JSON_TOKEN_OBJECT) ?
			SparkJsonFindObjectMember(document,first,"role") : -1;
		leading_system = first_role >= 0 && SparkJsonTokenIsType(document,first_role,SPARK_JSON_TOKEN_STRING) &&
			SparkJsonStringEquals(document,first_role,"system") ? 1u : 0u;
	}
	if ( thinking && leading_system == 0u && !SparkChatTemplateAppend(&rendered,chat_template->thinking_prefix) )
		outcome = SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY;
	for ( message_index = 0u; outcome == SPARK_CHAT_TEMPLATE_RENDERED && message_index < message_count; message_index++ )
	{
		int32_t entry = SparkJsonGetArrayElement(document,messages,message_index);
		int32_t content;
		const char *header;
		char *piece = 0;
		if ( entry < 0 || !SparkJsonTokenIsType(document,entry,SPARK_JSON_TOKEN_OBJECT) )
		{
			outcome = SPARK_CHAT_TEMPLATE_INVALID_MESSAGES;
			break;
		}
		outcome = SparkChatTemplateRoleHeader(chat_template,document,
			SparkJsonFindObjectMember(document,entry,"role"),thinking,
			message_index == 0u ? 1u : 0u,&header);
		if ( outcome != SPARK_CHAT_TEMPLATE_RENDERED )
			break;
		content = SparkJsonFindObjectMember(document,entry,"content");
		if ( content < 0 || !SparkJsonTokenIsType(document,content,SPARK_JSON_TOKEN_STRING) ||
			SparkJsonCopyString(document,content,&piece) != SPARK_STATUS_OK )
		{
			outcome = SPARK_CHAT_TEMPLATE_INVALID_MESSAGES;
			break;
		}
		if ( !SparkChatTemplateAppend(&rendered,header) || !SparkChatTemplateAppend(&rendered,piece) ||
			!SparkChatTemplateAppend(&rendered,SparkChatTemplateRoleSuffix(chat_template,document,
				SparkJsonFindObjectMember(document,entry,"role"))) ||
			!SparkChatTemplateAppend(&rendered,chat_template->turn_suffix) )
			outcome = SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY;
		free(piece);
	}
	if ( outcome == SPARK_CHAT_TEMPLATE_RENDERED &&
		!SparkChatTemplateAppend(&rendered,thinking ? chat_template->generation_thinking : chat_template->generation) )
		outcome = SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY;
	if ( outcome != SPARK_CHAT_TEMPLATE_RENDERED || rendered.length > UINT32_MAX )
	{
		free(rendered.bytes);
		return(outcome != SPARK_CHAT_TEMPLATE_RENDERED ? outcome : SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY);
	}
	*text = rendered.bytes;
	*text_bytes = (uint32_t)rendered.length;
	return(SPARK_CHAT_TEMPLATE_RENDERED);
}

const char *SparkChatTemplateOutcomeCode(SparkChatTemplateOutcome outcome)
{
	switch ( outcome )
	{
	case SPARK_CHAT_TEMPLATE_RENDERED: return("rendered");
	case SPARK_CHAT_TEMPLATE_NOT_CHAT: return("not_chat");
	case SPARK_CHAT_TEMPLATE_MISSING: return("chat_template_missing");
	case SPARK_CHAT_TEMPLATE_INVALID_MESSAGES: return("invalid_messages");
	case SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED: return("role_unsupported");
	case SPARK_CHAT_TEMPLATE_THINKING_UNSUPPORTED: return("thinking_unsupported");
	case SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY: return("oom");
	}
	return("invalid_messages");
}

const char *SparkChatTemplateOutcomeMessage(SparkChatTemplateOutcome outcome)
{
	switch ( outcome )
	{
	case SPARK_CHAT_TEMPLATE_MISSING:
		return("this deployment declares no chat_template; send prompt or prompt_token_ids");
	case SPARK_CHAT_TEMPLATE_INVALID_MESSAGES:
		return("messages must be a non-empty array of objects with string role and content");
	case SPARK_CHAT_TEMPLATE_ROLE_UNSUPPORTED:
		return("a message role is not declared by this deployment's chat_template");
	case SPARK_CHAT_TEMPLATE_THINKING_UNSUPPORTED:
		return("this deployment's chat_template declares no thinking generation");
	case SPARK_CHAT_TEMPLATE_OUT_OF_MEMORY:
		return("out of memory rendering the chat prompt");
	case SPARK_CHAT_TEMPLATE_RENDERED:
	case SPARK_CHAT_TEMPLATE_NOT_CHAT:
		break;
	}
	return("chat prompt rendered");
}

SparkStatus SparkChatTemplateResolveStops(
	const SparkModelResidentChatTemplate *chat_template,
	const SparkTokenizerSpecialToken *special_tokens,
	uint32_t special_token_count,
	uint32_t *stop_ids,
	uint32_t *stop_count,
	const char **unresolved_marker)
{
	uint32_t marker_index,token_index,matches;
	if ( chat_template == 0 || stop_ids == 0 || stop_count == 0 || unresolved_marker == 0 ||
		chat_template->declared == 0u ||
		chat_template->stop_marker_count > SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*stop_count = 0u;
	*unresolved_marker = 0;
	for ( marker_index = 0u; marker_index < chat_template->stop_marker_count; marker_index++ )
	{
		const char *marker = chat_template->stop_markers[marker_index];
		size_t marker_bytes = strlen(marker);
		matches = 0u;
		for ( token_index = 0u; special_tokens != 0 && token_index < special_token_count; token_index++ )
			if ( special_tokens[token_index].is_special != 0u && special_tokens[token_index].text != 0 &&
				special_tokens[token_index].text_bytes == marker_bytes &&
				memcmp(special_tokens[token_index].text,marker,marker_bytes) == 0 )
			{
				stop_ids[marker_index] = special_tokens[token_index].token_id;
				matches++;
			}
		if ( matches != 1u )
		{
			*unresolved_marker = marker;
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	*stop_count = chat_template->stop_marker_count;
	return(SPARK_STATUS_OK);
}
