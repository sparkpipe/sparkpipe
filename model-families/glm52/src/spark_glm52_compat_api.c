#include "sparkpipe/spark_glm52_compat_api.h"

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_chat_template.h"
#include "sparkpipe/spark_json.h"

#define SPARK_GLM52_COMPAT_DEFAULT_CHAT_FLAGS \
	(SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT | \
	 SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING)

static SparkStatus SparkGlm52CompatAppendBytes(
    SparkGlm52CompatTextRequest *request,
    const char *text,
    uint32_t text_bytes)
{
    if ( request == 0 || request->text == 0 || text == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( text_bytes > request->text_capacity ||
        request->text_bytes > (request->text_capacity - text_bytes) )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    memcpy(&request->text[request->text_bytes], text, text_bytes);
    request->text_bytes += text_bytes;
    if ( request->text_bytes < request->text_capacity )
        request->text[request->text_bytes] = '\0';
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendLiteral(
    SparkGlm52CompatTextRequest *request,
    const char *text)
{
    return SparkGlm52CompatAppendBytes(
        request,
        text,
        (uint32_t)strlen(text));
}

static SparkStatus SparkGlm52CompatAppendJsonString(
    const SparkJsonDocument *document,
    int32_t token_index,
    SparkGlm52CompatTextRequest *request)
{
    SparkStatus status;
    char *decoded_text;

    decoded_text = 0;
    status = SparkJsonCopyString(document, token_index, &decoded_text);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatAppendLiteral(request, decoded_text);
    free(decoded_text);
    return status;
}

static SparkStatus SparkGlm52CompatReadOptionalUInt32(
    const SparkJsonDocument *document,
    int32_t root_token_index,
    const char *member_name,
    uint32_t *value,
    uint32_t *present_out)
{
    int32_t member_token_index;

    if ( present_out != 0 )
        *present_out = 0u;
    member_token_index =
        SparkJsonFindObjectMember(document, root_token_index, member_name);
    if ( member_token_index < 0 )
        return SPARK_STATUS_OK;
    if ( present_out != 0 )
        *present_out = 1u;
    return SparkJsonGetUInt32(document, member_token_index, value);
}


static SparkStatus SparkGlm52CompatAppendOptionalJsonString(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    SparkGlm52CompatTextRequest *request,
    uint32_t *appended_out)
{
    int32_t member_token_index;
    SparkStatus status;

    if (appended_out != 0)
    {
        *appended_out = 0u;
    }
    member_token_index = SparkJsonFindObjectMember(
        document,
        object_token_index,
        member_name);
    if (member_token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document, member_token_index, SPARK_JSON_TOKEN_STRING))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52CompatAppendJsonString(document, member_token_index, request);
    if (status == SPARK_STATUS_OK && appended_out != 0)
    {
        *appended_out = 1u;
    }
    return status;
}

static SparkStatus SparkGlm52CompatAppendFileName(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    SparkGlm52CompatTextRequest *request)
{
    uint32_t appended;
    SparkStatus status;

    status = SparkGlm52CompatAppendOptionalJsonString(
        document,
        object_token_index,
        "filename",
        request,
        &appended);
    if (status != SPARK_STATUS_OK || appended != 0u)
    {
        return status;
    }
    status = SparkGlm52CompatAppendOptionalJsonString(
        document,
        object_token_index,
        "file_name",
        request,
        &appended);
    if (status != SPARK_STATUS_OK || appended != 0u)
    {
        return status;
    }
    return SparkGlm52CompatAppendOptionalJsonString(
        document,
        object_token_index,
        "name",
        request,
        &appended);
}

static SparkStatus SparkGlm52CompatAppendFileContentField(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    SparkGlm52CompatTextRequest *request,
    uint32_t *appended_out)
{
    static const char *FieldNames[] = {
        "content",
        "file_content",
        "file_text",
        "text",
        "data"
    };
    uint32_t field_index;
    SparkStatus status;

    if (appended_out != 0)
    {
        *appended_out = 0u;
    }
    for (field_index = 0u;
         field_index < (uint32_t)(sizeof(FieldNames) / sizeof(FieldNames[0u]));
         ++field_index)
    {
        uint32_t appended;

        status = SparkGlm52CompatAppendOptionalJsonString(
            document,
            object_token_index,
            FieldNames[field_index],
            request,
            &appended);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (appended != 0u)
        {
            if (appended_out != 0)
            {
                *appended_out = 1u;
            }
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendFileObject(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    SparkGlm52CompatTextRequest *request,
    uint32_t *appended_out)
{
    uint32_t appended;
    SparkStatus status;

    if (appended_out != 0)
    {
        *appended_out = 0u;
    }
    status = SparkGlm52CompatAppendLiteral(request, "\n[uploaded file");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52CompatAppendLiteral(request, ": ");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52CompatAppendFileName(document, object_token_index, request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52CompatAppendLiteral(request, "]\n");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52CompatAppendFileContentField(
        document,
        object_token_index,
        request,
        &appended);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (appended == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52CompatAppendLiteral(request, "\n[/uploaded file]\n");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (appended_out != 0)
    {
        *appended_out = 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendContentObject(
    const SparkJsonDocument *document,
    int32_t element_token_index,
    SparkGlm52CompatTextRequest *request)
{
    int32_t source_token_index;
    int32_t text_token_index;
    uint32_t appended;
    SparkStatus status;

    text_token_index = SparkJsonFindObjectMember(
        document,
        element_token_index,
        "text");
    if (text_token_index >= 0)
    {
        return SparkGlm52CompatAppendJsonString(
            document,
            text_token_index,
            request);
    }
    if (SparkJsonFindObjectMember(document, element_token_index, "content") >= 0 ||
        SparkJsonFindObjectMember(document, element_token_index, "file_content") >= 0 ||
        SparkJsonFindObjectMember(document, element_token_index, "file_text") >= 0 ||
        SparkJsonFindObjectMember(document, element_token_index, "data") >= 0 ||
        SparkJsonFindObjectMember(document, element_token_index, "filename") >= 0 ||
        SparkJsonFindObjectMember(document, element_token_index, "file_name") >= 0 ||
        SparkJsonFindObjectMember(document, element_token_index, "name") >= 0)
    {
        return SparkGlm52CompatAppendFileObject(
            document,
            element_token_index,
            request,
            &appended);
    }
    source_token_index = SparkJsonFindObjectMember(
        document,
        element_token_index,
        "source");
    if (source_token_index >= 0 &&
        SparkJsonTokenIsType(document, source_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        status = SparkGlm52CompatAppendFileContentField(
            document,
            source_token_index,
            request,
            &appended);
        if (status != SPARK_STATUS_OK || appended != 0u)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendContentToken(
    const SparkJsonDocument *document,
    int32_t content_token_index,
    SparkGlm52CompatTextRequest *request)
{
    uint32_t element_count;
    uint32_t element_index;
    SparkStatus status;

    if ( SparkJsonTokenIsType(document, content_token_index, SPARK_JSON_TOKEN_STRING) )
        return SparkGlm52CompatAppendJsonString(
            document,
            content_token_index,
            request);
    if ( !SparkJsonTokenIsType(document, content_token_index, SPARK_JSON_TOKEN_ARRAY) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    element_count = SparkJsonGetArrayElementCount(document, content_token_index);
    for (element_index=0u; element_index<element_count; element_index++)
    {
        int32_t element_token_index;

        element_token_index =
            SparkJsonGetArrayElement(document, content_token_index, element_index);
        if ( SparkJsonTokenIsType(document, element_token_index, SPARK_JSON_TOKEN_STRING) )
        {
            status = SparkGlm52CompatAppendJsonString(
                document,
                element_token_index,
                request);
            if ( status != SPARK_STATUS_OK )
                return status;
            continue;
        }
        if ( !SparkJsonTokenIsType(document, element_token_index, SPARK_JSON_TOKEN_OBJECT) )
            return SPARK_STATUS_INVALID_ARGUMENT;
        status = SparkGlm52CompatAppendContentObject(
            document,
            element_token_index,
            request);
        if ( status != SPARK_STATUS_OK )
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatInitializeTemplateWriter(
	SparkGlm52CompatTextRequest *request,
	SparkGlm52ChatTemplateWriter *writer)
{
	if ( request == 0 || writer == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52ChatTemplateInitializeWriter(
		writer,
		request->text,
		request->text_capacity,
		request->text_bytes);
}

static SparkStatus SparkGlm52CompatBeginTemplate(
	SparkGlm52CompatTextRequest *request)
{
	SparkGlm52ChatTemplateWriter writer;
	SparkStatus status;

	status = SparkGlm52CompatInitializeTemplateWriter(request, &writer);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkGlm52ChatTemplateBegin(
		&writer,
		"Max",
		request->chat_template_flags);
	request->text_bytes = writer.text_bytes;
	return status;
}

static SparkStatus SparkGlm52CompatGetMessageRole(
	const SparkJsonDocument *document,
	int32_t message_token_index,
	SparkGlm52ChatTemplateRole *role_out)
{
	int32_t role_token_index;

	if ( document == 0 || role_out == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	role_token_index =
		SparkJsonFindObjectMember(document, message_token_index, "role");
	if ( role_token_index < 0 ||
		SparkJsonStringEquals(document, role_token_index, "user") )
		*role_out = SPARK_GLM52_CHAT_TEMPLATE_ROLE_USER;
	else if ( SparkJsonStringEquals(document, role_token_index, "system") )
		*role_out = SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM;
	else if ( SparkJsonStringEquals(document, role_token_index, "assistant") )
		*role_out = SPARK_GLM52_CHAT_TEMPLATE_ROLE_ASSISTANT;
	else if ( SparkJsonStringEquals(document, role_token_index, "tool") )
		*role_out = SPARK_GLM52_CHAT_TEMPLATE_ROLE_TOOL;
	else
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatBeginMessage(
	SparkGlm52CompatTextRequest *request,
	SparkGlm52ChatTemplateRole role)
{
	SparkGlm52ChatTemplateWriter writer;
	SparkStatus status;

	status = SparkGlm52CompatInitializeTemplateWriter(request, &writer);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkGlm52ChatTemplateBeginMessage(&writer, role);
	request->text_bytes = writer.text_bytes;
	return status;
}

static SparkStatus SparkGlm52CompatEndMessage(
	SparkGlm52CompatTextRequest *request,
	SparkGlm52ChatTemplateRole role)
{
	SparkGlm52ChatTemplateWriter writer;
	SparkStatus status;

	status = SparkGlm52CompatInitializeTemplateWriter(request, &writer);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkGlm52ChatTemplateEndMessage(&writer, role);
	request->text_bytes = writer.text_bytes;
	return status;
}

static SparkStatus SparkGlm52CompatFinishTemplate(
	SparkGlm52CompatTextRequest *request)
{
	SparkGlm52ChatTemplateWriter writer;
	SparkStatus status;

	status = SparkGlm52CompatInitializeTemplateWriter(request, &writer);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkGlm52ChatTemplateFinish(
		&writer,
		request->chat_template_flags);
	request->text_bytes = writer.text_bytes;
	return status;
}

static SparkStatus SparkGlm52CompatAppendMessages(
    const SparkJsonDocument *document,
    int32_t messages_token_index,
    SparkGlm52CompatTextRequest *request)
{
    uint32_t message_count;
    uint32_t message_index;
    SparkStatus status;

    if ( !SparkJsonTokenIsType(document, messages_token_index, SPARK_JSON_TOKEN_ARRAY) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    message_count = SparkJsonGetArrayElementCount(document, messages_token_index);
    if ( message_count == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (message_index=0u; message_index<message_count; message_index++)
    {
        int32_t message_token_index;
        int32_t content_token_index;
		SparkGlm52ChatTemplateRole role;

        message_token_index =
            SparkJsonGetArrayElement(document, messages_token_index, message_index);
        if ( !SparkJsonTokenIsType(document, message_token_index, SPARK_JSON_TOKEN_OBJECT) )
            return SPARK_STATUS_INVALID_ARGUMENT;
        content_token_index =
            SparkJsonFindObjectMember(document, message_token_index, "content");
        if ( content_token_index < 0 )
            return SPARK_STATUS_INVALID_ARGUMENT;
		status = SparkGlm52CompatGetMessageRole(
			document,
			message_token_index,
			&role);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkGlm52CompatBeginMessage(request, role);
        if ( status != SPARK_STATUS_OK )
            return status;
        status = SparkGlm52CompatAppendContentToken(
            document,
            content_token_index,
            request);
        if ( status != SPARK_STATUS_OK )
            return status;
		status = SparkGlm52CompatEndMessage(request, role);
        if ( status != SPARK_STATUS_OK )
            return status;
    }
    return SPARK_STATUS_OK;
}


static SparkStatus SparkGlm52CompatAppendFilesArray(
    const SparkJsonDocument *document,
    int32_t files_token_index,
    SparkGlm52CompatTextRequest *request)
{
    uint32_t file_count;
    uint32_t file_index;
    SparkStatus status;

    if (files_token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document, files_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file_count = SparkJsonGetArrayElementCount(document, files_token_index);
    for (file_index = 0u; file_index < file_count; ++file_index)
    {
        int32_t file_token_index;
        uint32_t appended;

        file_token_index = SparkJsonGetArrayElement(
            document,
            files_token_index,
            file_index);
        if (!SparkJsonTokenIsType(document, file_token_index, SPARK_JSON_TOKEN_OBJECT))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkGlm52CompatAppendFileObject(
            document,
            file_token_index,
            request,
            &appended);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatAppendRequestFiles(
	const SparkJsonDocument *document,
	int32_t root_token_index,
	SparkGlm52CompatTextRequest *request)
{
	int32_t attachments_token_index;
	int32_t files_token_index;
	SparkStatus status;

	files_token_index =
		SparkJsonFindObjectMember(document, root_token_index, "files");
	status = SparkGlm52CompatAppendFilesArray(
		document,
		files_token_index,
		request);
	if ( status != SPARK_STATUS_OK )
		return status;
	attachments_token_index =
		SparkJsonFindObjectMember(document, root_token_index, "attachments");
	return SparkGlm52CompatAppendFilesArray(
		document,
		attachments_token_index,
		request);
}

static SparkStatus SparkGlm52CompatPrepareCommon(
    SparkJsonDocument *document,
    SparkGlm52CompatTextRequest *request)
{
    int32_t root_token_index;
    uint32_t thinking_budget;
    uint32_t thinking_budget_alias;
    uint32_t thinking_budget_present;
    uint32_t thinking_budget_alias_present;
    SparkStatus status;

    if ( request == 0 || request->abi_version != SPARK_GLM52_COMPAT_API_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_COMPAT_TEXT_REQUEST_BYTES ||
        request->text == 0 || request->text_capacity == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    root_token_index = SparkJsonGetRootToken(document);
    if ( !SparkJsonTokenIsType(document, root_token_index, SPARK_JSON_TOKEN_OBJECT) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    request->text_bytes = 0u;
    request->text[0] = '\0';
    request->thinking_token_budget = 0u;
    request->chat_template_flags = SPARK_GLM52_COMPAT_DEFAULT_CHAT_FLAGS;
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "priority",
        &request->priority,
        0);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "max_tokens",
        &request->output_token_budget,
        0);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "max_completion_tokens",
        &request->output_token_budget,
        0);
    if ( status != SPARK_STATUS_OK )
        return status;
    thinking_budget = 0u;
    thinking_budget_alias = 0u;
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "thinking_budget_tokens",
        &thinking_budget,
        &thinking_budget_present);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatReadOptionalUInt32(
        document,
        root_token_index,
        "thinking_token_budget",
        &thinking_budget_alias,
        &thinking_budget_alias_present);
    if ( status != SPARK_STATUS_OK )
        return status;
    if ( thinking_budget_present != 0u &&
        thinking_budget_alias_present != 0u &&
        thinking_budget != thinking_budget_alias )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( thinking_budget_present == 0u && thinking_budget_alias_present == 0u )
        return SPARK_STATUS_OK;
    if ( thinking_budget_present == 0u )
        thinking_budget = thinking_budget_alias;
    request->thinking_token_budget = thinking_budget;
    request->chat_template_flags =
        SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT;
    if ( thinking_budget != 0u )
        request->chat_template_flags |=
            SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52CompatSubmitPrepared(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52CompatTextRequest *compat_request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkGlm52ServiceSubmitTextRequest request;

    SparkGlm52ServiceInitializeSubmitTextRequest(&request);
    request.flags = compat_request->flags;
    request.priority = compat_request->priority;
    request.thinking_token_budget = compat_request->thinking_token_budget;
    request.output_token_budget = compat_request->output_token_budget;
    request.max_prefill_tokens_per_step =
        compat_request->max_prefill_tokens_per_step;
    request.tokenizer_encode_flags = compat_request->tokenizer_encode_flags;
    request.client_id = compat_request->client_id;
    request.client_request_id = compat_request->client_request_id;
    request.sequence_id = compat_request->sequence_id;
    request.text = compat_request->text;
    request.text_bytes = compat_request->text_bytes;
    return SparkGlm52ServiceSubmitText(service, &request, result);
}

void SparkGlm52CompatInitializeTextRequest(
    SparkGlm52CompatTextRequest *request,
    char *text,
    uint32_t text_capacity)
{
    if ( request == 0 )
        return;
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_COMPAT_API_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_COMPAT_TEXT_REQUEST_BYTES;
    request->chat_template_flags = SPARK_GLM52_COMPAT_DEFAULT_CHAT_FLAGS;
    request->text = text;
    request->text_capacity = text_capacity;
    if ( text != 0 && text_capacity != 0u )
        text[0] = '\0';
}

SparkStatus SparkGlm52CompatPrepareOpenAiJson(
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t messages_token_index;
    int32_t prompt_token_index;
    SparkStatus status;

    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(json_text, json_bytes, &document);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatPrepareCommon(&document, request);
    if ( status == SPARK_STATUS_OK )
    {
        root_token_index = SparkJsonGetRootToken(&document);
        messages_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "messages");
        prompt_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "prompt");
        if ( messages_token_index >= 0 )
		{
			status = SparkGlm52CompatBeginTemplate(request);
			if ( status == SPARK_STATUS_OK )
				status = SparkGlm52CompatAppendMessages(
					&document,
					messages_token_index,
					request);
		}
        else if ( prompt_token_index >= 0 )
            status = SparkGlm52CompatAppendJsonString(
                &document,
                prompt_token_index,
                request);
        else
            status = SPARK_STATUS_INVALID_ARGUMENT;
        if ( status == SPARK_STATUS_OK )
			status = SparkGlm52CompatAppendRequestFiles(
				&document,
				root_token_index,
				request);
		if ( status == SPARK_STATUS_OK && messages_token_index >= 0 )
			status = SparkGlm52CompatFinishTemplate(request);
    }
    SparkJsonDocumentDestroy(&document);
    return status;
}

SparkStatus SparkGlm52CompatPrepareAnthropicJson(
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t system_token_index;
    int32_t messages_token_index;
    SparkStatus status;

    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(json_text, json_bytes, &document);
    if ( status != SPARK_STATUS_OK )
        return status;
    status = SparkGlm52CompatPrepareCommon(&document, request);
    if ( status == SPARK_STATUS_OK )
    {
        root_token_index = SparkJsonGetRootToken(&document);
		status = SparkGlm52CompatBeginTemplate(request);
        system_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "system");
		if ( status == SPARK_STATUS_OK && system_token_index >= 0 )
        {
			status = SparkGlm52CompatBeginMessage(
				request,
				SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM);
            if ( status == SPARK_STATUS_OK )
                status = SparkGlm52CompatAppendJsonString(
                    &document,
                    system_token_index,
                    request);
            if ( status == SPARK_STATUS_OK )
				status = SparkGlm52CompatEndMessage(
					request,
					SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM);
        }
        messages_token_index =
            SparkJsonFindObjectMember(&document, root_token_index, "messages");
        if ( status == SPARK_STATUS_OK && messages_token_index >= 0 )
            status = SparkGlm52CompatAppendMessages(
                &document,
                messages_token_index,
                request);
        else if ( status == SPARK_STATUS_OK )
            status = SPARK_STATUS_INVALID_ARGUMENT;
        if ( status == SPARK_STATUS_OK )
			status = SparkGlm52CompatAppendRequestFiles(
				&document,
				root_token_index,
				request);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52CompatFinishTemplate(request);
    }
    SparkJsonDocumentDestroy(&document);
    return status;
}

SparkStatus SparkGlm52CompatSubmitOpenAiJson(
    SparkGlm52ServiceRuntime *service,
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkStatus status;

    status = SparkGlm52CompatPrepareOpenAiJson(json_text, json_bytes, request);
    if ( status != SPARK_STATUS_OK )
        return status;
    return SparkGlm52CompatSubmitPrepared(service, request, result);
}

SparkStatus SparkGlm52CompatSubmitAnthropicJson(
    SparkGlm52ServiceRuntime *service,
    const char *json_text,
    uint32_t json_bytes,
    SparkGlm52CompatTextRequest *request,
    SparkGlm52ServiceSubmitResult *result)
{
    SparkStatus status;

    status = SparkGlm52CompatPrepareAnthropicJson(json_text, json_bytes, request);
    if ( status != SPARK_STATUS_OK )
        return status;
    return SparkGlm52CompatSubmitPrepared(service, request, result);
}
