#include "sparkpipe/spark_glm52_http_gateway.h"

#include <stdio.h>
#include <string.h>

static uint32_t SparkGlm52HttpStringEquals(const char *left,const char *right)
{
	if (left == 0 || right == 0)
		return 0u;
	return strcmp(left,right) == 0;
}

static uint32_t SparkGlm52HttpBytesMatch(
	const char *left,
	uint32_t left_bytes,
	const char *right)
{
	uint32_t right_bytes;

	if (left == 0 || right == 0)
		return 0u;
	right_bytes = (uint32_t)strlen(right);
	if (left_bytes != right_bytes)
		return 0u;
	return memcmp(left,right,right_bytes) == 0;
}

static SparkStatus SparkGlm52HttpWriteBody(
	SparkGlm52HttpGatewayResponse *response,
	const char *content_type,
	uint32_t status_code,
	uint32_t flags,
	const char *body)
{
	int32_t written;

	if (response == 0 || response->body == 0 || body == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(response->body,response->body_capacity,"%s",body);
	if (written < 0)
		return SPARK_STATUS_INTERNAL_ERROR;
	if ((uint32_t)written >= response->body_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	response->status_code = status_code;
	response->flags = flags;
	response->content_type = content_type;
	response->body_bytes = (uint32_t)written;
	return SPARK_STATUS_OK;
}

void SparkGlm52HttpGatewayInitializeRequest(
	SparkGlm52HttpGatewayRequest *request)
{
	if (request == 0)
		return;
	memset(request,0,sizeof(*request));
	request->abi_version = SPARK_GLM52_HTTP_GATEWAY_ABI_VERSION;
	request->descriptor_bytes = SPARK_GLM52_HTTP_GATEWAY_REQUEST_BYTES;
}

void SparkGlm52HttpGatewayInitializeResponse(
	SparkGlm52HttpGatewayResponse *response,
	char *body,
	uint32_t body_capacity)
{
	if (response == 0)
		return;
	memset(response,0,sizeof(*response));
	response->abi_version = SPARK_GLM52_HTTP_GATEWAY_ABI_VERSION;
	response->descriptor_bytes = SPARK_GLM52_HTTP_GATEWAY_RESPONSE_BYTES;
	response->body = body;
	response->body_capacity = body_capacity;
}

uint32_t SparkGlm52HttpGatewayRoute(
	const SparkGlm52HttpGatewayRequest *request)
{
	if (request == 0 || request->method == 0 || request->path == 0)
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE;
	if (SparkGlm52HttpStringEquals(request->method,"OPTIONS"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_CORS_PREFLIGHT;
	if (SparkGlm52HttpStringEquals(request->method,"GET") &&
		SparkGlm52HttpStringEquals(request->path,"/"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_DEMO_UI;
	if (SparkGlm52HttpStringEquals(request->method,"GET") &&
		SparkGlm52HttpStringEquals(request->path,"/health"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_HEALTH;
	if (SparkGlm52HttpStringEquals(request->method,"POST") &&
		SparkGlm52HttpStringEquals(request->path,"/v1/chat/completions"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT;
	if (SparkGlm52HttpStringEquals(request->method,"POST") &&
		SparkGlm52HttpStringEquals(request->path,"/v1/completions"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS;
	if (SparkGlm52HttpStringEquals(request->method,"POST") &&
		SparkGlm52HttpStringEquals(request->path,"/v1/messages"))
		return SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES;
	return SPARK_GLM52_HTTP_GATEWAY_ROUTE_NONE;
}

SparkStatus SparkGlm52HttpGatewayBuildDemoUi(
	SparkGlm52HttpGatewayResponse *response)
{
	static const char Body[] =
		"<!doctype html><html><head><meta charset=\"utf-8\">"
		"<title>SparkPipe GLM 5.2</title>"
		"<style>body{font-family:system-ui;margin:2rem;max-width:960px}"
		"textarea{width:100%;height:16rem}pre{white-space:pre-wrap;background:#111;color:#eee;padding:1rem;min-height:10rem}"
		"input,button,textarea{font:inherit}button{padding:.55rem 1rem}.row{margin:.75rem 0}.small{color:#555}</style></head>"
		"<body><h1>SparkPipe GLM 5.2</h1>"
		"<p>Public demo path: paste a prompt, attach text files, and stream the response from <code>/v1/chat/completions</code>.</p>"
		"<div class=\"row\"><input id=\"key\" placeholder=\"Bearer token, if required\" type=\"password\" style=\"width:100%\"></div>"
		"<div class=\"row\"><textarea id=\"prompt\">Summarize the attached file and point out anything surprising.</textarea></div>"
		"<div class=\"row\"><input id=\"files\" type=\"file\" multiple></div>"
		"<div class=\"row small\">Text files are folded into the model prompt inside SparkPipe; clients do not manage context windows.</div>"
		"<div class=\"row\"><button id=\"run\">Send</button></div><pre id=\"out\"></pre>"
		"<script>"
		"document.getElementById('run').onclick=async()=>{"
		"const out=document.getElementById('out');out.textContent='';"
		"const key=document.getElementById('key').value;"
		"const prompt=document.getElementById('prompt').value;"
		"const files=await Promise.all([...document.getElementById('files').files].map(async f=>({filename:f.name,content:await f.text()})));"
		"const body={model:'glm-5.2',stream:true,max_tokens:1024,messages:[{role:'user',content:prompt}],files};"
		"const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json','authorization':key?('Bearer '+key):''},body:JSON.stringify(body)});"
		"const reader=r.body.getReader();const dec=new TextDecoder();"
		"for(;;){const x=await reader.read();if(x.done)break;out.textContent+=dec.decode(x.value);}"
		"};"
		"</script></body></html>";
	return SparkGlm52HttpWriteBody(response,"text/html; charset=utf-8",200u,0u,Body);
}

SparkStatus SparkGlm52HttpGatewayBuildHealth(
	SparkGlm52HttpGatewayResponse *response,
	uint32_t backend_ready,
	uint32_t pp13_ready)
{
	int32_t written;

	if (response == 0 || response->body == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(
		response->body,
		response->body_capacity,
		"{\"service\":\"sparkpipe-glm52\",\"backend_ready\":%u,\"pp13_ready\":%u}\n",
		backend_ready != 0u ? 1u : 0u,
		pp13_ready != 0u ? 1u : 0u);
	if (written < 0)
		return SPARK_STATUS_INTERNAL_ERROR;
	if ((uint32_t)written >= response->body_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	response->status_code = 200u;
	response->flags = 0u;
	response->content_type = "application/json";
	response->body_bytes = (uint32_t)written;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52HttpGatewayBuildBackendUnavailable(
	SparkGlm52HttpGatewayResponse *response,
	uint32_t stream)
{
	static const char JsonBody[] =
		"{\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 PP13 backend is not attached\"}}\n";
	static const char StreamBody[] =
		"event: error\n"
		"data: {\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 PP13 backend is not attached\"}}\n\n";

	if (stream != 0u)
		return SparkGlm52HttpWriteBody(
			response,
			"text/event-stream",
			503u,
			SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM,
			StreamBody);
	return SparkGlm52HttpWriteBody(response,"application/json",503u,0u,JsonBody);
}

SparkStatus SparkGlm52HttpGatewayBuildRequestTimeout(
	SparkGlm52HttpGatewayResponse *response,
	uint32_t stream)
{
	static const char JsonBody[] =
		"{\"error\":{\"type\":\"request_timeout\",\"message\":\"GLM52 PP13 request produced no terminal event before the stream poll budget\"}}\n";
	static const char StreamBody[] =
		"event: error\n"
		"data: {\"error\":{\"type\":\"request_timeout\",\"message\":\"GLM52 PP13 request produced no terminal event before the stream poll budget\"}}\n\n";

	if (stream != 0u)
		return SparkGlm52HttpWriteBody(
			response,
			"text/event-stream",
			504u,
			SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM,
			StreamBody);
	return SparkGlm52HttpWriteBody(response,"application/json",504u,0u,JsonBody);
}

SparkStatus SparkGlm52HttpGatewayBuildUnauthorized(
	SparkGlm52HttpGatewayResponse *response)
{
	return SparkGlm52HttpWriteBody(
		response,
		"application/json",
		401u,
		0u,
		"{\"error\":{\"type\":\"unauthorized\",\"message\":\"missing or invalid bearer token\"}}\n");
}

SparkStatus SparkGlm52HttpGatewayBuildNotFound(
	SparkGlm52HttpGatewayResponse *response)
{
	return SparkGlm52HttpWriteBody(
		response,
		"application/json",
		404u,
		0u,
		"{\"error\":{\"type\":\"not_found\",\"message\":\"unknown endpoint\"}}\n");
}


SparkStatus SparkGlm52HttpGatewayBuildCorsPreflight(
	SparkGlm52HttpGatewayResponse *response)
{
	return SparkGlm52HttpWriteBody(
		response,
		"text/plain",
		200u,
		0u,
		"");
}

uint32_t SparkGlm52HttpGatewayBodyRequestsStream(
	const char *body,
	uint32_t body_bytes)
{
	uint32_t index;
	uint32_t cursor;

	if (body == 0 || body_bytes < 12u)
		return 0u;
	for (index = 0u; (index + 8u) <= body_bytes; ++index)
	{
		if (memcmp(&body[index],"\"stream\"",8u) != 0)
			continue;
		cursor = index + 8u;
		while (cursor < body_bytes &&
			(body[cursor] == ' ' || body[cursor] == '\t' ||
			 body[cursor] == '\r' || body[cursor] == '\n'))
			++cursor;
		if (cursor >= body_bytes || body[cursor] != ':')
			continue;
		++cursor;
		while (cursor < body_bytes &&
			(body[cursor] == ' ' || body[cursor] == '\t' ||
			 body[cursor] == '\r' || body[cursor] == '\n'))
			++cursor;
		if ((cursor + 4u) <= body_bytes &&
			memcmp(&body[cursor],"true",4u) == 0)
			return 1u;
	}
	return 0u;
}

uint32_t SparkGlm52HttpGatewayAuthorizationMatches(
	const SparkGlm52HttpGatewayRequest *request,
	const char *api_key)
{
	char expected[512];
	int32_t written;

	if (api_key == 0 || api_key[0] == '\0')
		return 1u;
	if (request == 0)
		return 0u;
	written = snprintf(expected,sizeof(expected),"Bearer %s",api_key);
	if (written < 0 || (uint32_t)written >= sizeof(expected))
		return 0u;
	return SparkGlm52HttpBytesMatch(
		request->authorization,
		request->authorization_bytes,
		expected);
}

static SparkStatus SparkGlm52HttpAppendJsonEscapedBytes(
    char *destination,
    uint32_t destination_capacity,
    uint32_t *destination_bytes_inout,
    const char *text,
    uint32_t text_bytes)
{
    uint32_t index;
    uint32_t used_bytes;

    if (destination == 0 || destination_bytes_inout == 0 ||
        (text == 0 && text_bytes != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    used_bytes = *destination_bytes_inout;
    for (index = 0u; index < text_bytes; ++index)
    {
        unsigned char byte;

        byte = (unsigned char)text[index];
        if (byte == '"' || byte == '\\')
        {
            if (used_bytes + 2u >= destination_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            destination[used_bytes++] = '\\';
            destination[used_bytes++] = (char)byte;
        }
        else if (byte == '\n')
        {
            if (used_bytes + 2u >= destination_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            destination[used_bytes++] = '\\';
            destination[used_bytes++] = 'n';
        }
        else if (byte == '\r')
        {
            if (used_bytes + 2u >= destination_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            destination[used_bytes++] = '\\';
            destination[used_bytes++] = 'r';
        }
        else if (byte == '\t')
        {
            if (used_bytes + 2u >= destination_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            destination[used_bytes++] = '\\';
            destination[used_bytes++] = 't';
        }
        else if (byte < 0x20u)
        {
            int32_t written;

            if (used_bytes + 6u >= destination_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            written = snprintf(
                &destination[used_bytes],
                destination_capacity - used_bytes,
                "\\u%04x",
                (uint32_t)byte);
            if (written != 6)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
            used_bytes += 6u;
        }
        else
        {
            if (used_bytes + 1u >= destination_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            destination[used_bytes++] = (char)byte;
        }
    }
    destination[used_bytes] = '\0';
    *destination_bytes_inout = used_bytes;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52HttpServiceEventIsTerminal(
    const SparkGlm52ServiceEvent *service_event)
{
    if (service_event == 0)
    {
        return 0u;
    }
    return service_event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_COMPLETED ||
        service_event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_REQUEST_CANCELLED ||
        service_event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_ERROR;
}

SparkStatus SparkGlm52HttpGatewayBuildServiceHealth(
    SparkGlm52HttpGatewayResponse *response,
    const SparkGlm52ServiceStats *stats,
    uint32_t backend_ready,
    uint32_t pp13_ready,
    uint32_t max_context_tokens,
    uint32_t production_contract_flags,
    const char *first_blocker)
{
    SparkGlm52ServiceStats empty_stats;
    const SparkGlm52ServiceStats *stats_view;
    char escaped_blocker[512];
    uint32_t escaped_blocker_bytes;
    int32_t written;

    if (response == 0 || response->body == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&empty_stats, 0, sizeof(empty_stats));
    stats_view = stats != 0 ? stats : &empty_stats;
    escaped_blocker_bytes = 0u;
    if (first_blocker == 0)
    {
        first_blocker = "";
    }
    if (SparkGlm52HttpAppendJsonEscapedBytes(
            escaped_blocker,
            sizeof(escaped_blocker),
            &escaped_blocker_bytes,
            first_blocker,
            (uint32_t)strlen(first_blocker)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    written = snprintf(
        response->body,
        response->body_capacity,
        "{"
        "\"service\":\"sparkpipe-glm52\","
        "\"backend_ready\":%u,"
        "\"pp13_ready\":%u,"
        "\"max_context_tokens\":%u,"
        "\"production_contract_flags\":%u,"
        "\"connected_clients\":%u,"
        "\"live_requests\":%u,"
        "\"queued_requests\":%u,"
        "\"event_backlog\":%u,"
        "\"dropped_events\":%u,"
        "\"jit_prefetch_dispatches\":%llu,"
        "\"jit_prefetch_blocks\":%llu,"
        "\"async_jit_prefetch_starts\":%llu,"
        "\"async_jit_prefetch_completions\":%llu,"
        "\"first_blocker\":\"%s\""
        "}\n",
        backend_ready != 0u ? 1u : 0u,
        pp13_ready != 0u ? 1u : 0u,
        max_context_tokens,
        production_contract_flags,
        stats_view->connected_client_count,
        stats_view->live_request_count,
        stats_view->serving_stats.queued_request_count,
        stats_view->event_count,
        stats_view->dropped_event_count,
        (unsigned long long)stats_view->serving_stats.jit_prefetch_dispatch_count,
        (unsigned long long)stats_view->serving_stats.jit_prefetch_block_count,
        (unsigned long long)stats_view->serving_stats.async_jit_prefetch_start_count,
        (unsigned long long)stats_view->serving_stats.async_jit_prefetch_completion_count,
        escaped_blocker);
    if (written < 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((uint32_t)written >= response->body_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    response->status_code = 200u;
    response->flags = 0u;
    response->content_type = "application/json";
    response->body_bytes = (uint32_t)written;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52HttpGatewayBuildSubmitAccepted(
    SparkGlm52HttpGatewayResponse *response,
    const SparkGlm52ServiceSubmitResult *submit_result,
    uint32_t stream)
{
    int32_t written;

    if (response == 0 || response->body == 0 || submit_result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (stream != 0u)
    {
        written = snprintf(
            response->body,
            response->body_capacity,
            "event: accepted\n"
            "data: {\"client_request_id\":%llu,\"serving_request_id\":%llu,"
            "\"sequence_id\":%llu,\"prompt_tokens\":%u,"
            "\"output_token_budget\":%u}\n\n",
            (unsigned long long)submit_result->client_request_id,
            (unsigned long long)submit_result->serving_request_id,
            (unsigned long long)submit_result->sequence_id,
            submit_result->prompt_token_count,
            submit_result->output_token_budget);
        if (written < 0)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        if ((uint32_t)written >= response->body_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        response->status_code = 202u;
        response->flags = SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM;
        response->content_type = "text/event-stream";
        response->body_bytes = (uint32_t)written;
        return SPARK_STATUS_OK;
    }
    written = snprintf(
        response->body,
        response->body_capacity,
        "{\"id\":\"spreq-%llu\",\"object\":\"sparkpipe.request\","
        "\"client_request_id\":%llu,\"serving_request_id\":%llu,"
        "\"sequence_id\":%llu,\"prompt_tokens\":%u,"
        "\"output_token_budget\":%u,\"status\":\"queued\"}\n",
        (unsigned long long)submit_result->serving_request_id,
        (unsigned long long)submit_result->client_request_id,
        (unsigned long long)submit_result->serving_request_id,
        (unsigned long long)submit_result->sequence_id,
        submit_result->prompt_token_count,
        submit_result->output_token_budget);
    if (written < 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((uint32_t)written >= response->body_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    response->status_code = 202u;
    response->flags = 0u;
    response->content_type = "application/json";
    response->body_bytes = (uint32_t)written;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52HttpGatewaySubmitJsonToService(
    SparkGlm52ServiceRuntime *service,
    const SparkGlm52HttpGatewayRequest *request,
    SparkGlm52ServiceClientId client_id,
    SparkGlm52ServiceRequestId client_request_id,
    SparkGlm52CompatTextRequest *compat_request,
    SparkGlm52ServiceSubmitResult *submit_result,
    SparkGlm52HttpGatewayResponse *response)
{
    uint32_t route;
    uint32_t stream;
    SparkStatus status;

    if (service == 0 || request == 0 || compat_request == 0 ||
        submit_result == 0 || response == 0 || request->body == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route = SparkGlm52HttpGatewayRoute(request);
    compat_request->client_id = client_id;
    compat_request->client_request_id = client_request_id;
    stream = SparkGlm52HttpGatewayBodyRequestsStream(
        request->body,
        request->body_bytes);
    if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_CHAT ||
        route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS)
    {
        status = SparkGlm52CompatSubmitOpenAiJson(
            service,
            request->body,
            request->body_bytes,
            compat_request,
            submit_result);
    }
    else if (route == SPARK_GLM52_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES)
    {
        status = SparkGlm52CompatSubmitAnthropicJson(
            service,
            request->body,
            request->body_bytes,
            compat_request,
            submit_result);
    }
    else
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52HttpGatewayBuildSubmitAccepted(
        response,
        submit_result,
        stream);
}

SparkStatus SparkGlm52HttpGatewayBuildServiceEventStream(
    SparkGlm52HttpGatewayResponse *response,
    const SparkGlm52ServiceEvent *service_event,
    const SparkTokenizer *tokenizer)
{
    char token_text[1024u];
    uint32_t token_text_bytes;
    uint32_t body_bytes;
    int32_t written;
    SparkStatus status;

    if (response == 0 || response->body == 0 || service_event == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    token_text[0u] = '\0';
    token_text_bytes = 0u;
    if (service_event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_TOKEN &&
        tokenizer != 0)
    {
        uint32_t token_id;

        token_id = service_event->token_id;
        status = SparkTokenizerDecodeTokenIds(
            tokenizer,
            &token_id,
            1u,
            SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
            token_text,
            sizeof(token_text),
            &token_text_bytes);
        if (status != SPARK_STATUS_OK)
        {
            token_text[0u] = '\0';
            token_text_bytes = 0u;
        }
    }

    written = snprintf(
        response->body,
        response->body_capacity,
        "event: %s\n"
        "data: {\"kind\":%u,\"status\":%u,\"client_id\":%llu,"
        "\"client_request_id\":%llu,\"serving_request_id\":%llu,"
        "\"sequence_id\":%llu,\"token_id\":%u,\"token_index\":%u,"
        "\"text\":\"",
        SparkGlm52HttpServiceEventIsTerminal(service_event) != 0u ? "done" :
            (service_event->kind == SPARK_GLM52_SERVICE_EVENT_KIND_TOKEN ? "token" : "event"),
        service_event->kind,
        service_event->status,
        (unsigned long long)service_event->client_id,
        (unsigned long long)service_event->client_request_id,
        (unsigned long long)service_event->serving_request_id,
        (unsigned long long)service_event->sequence_id,
        service_event->token_id,
        service_event->token_index);
    if (written < 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((uint32_t)written >= response->body_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    body_bytes = (uint32_t)written;
    status = SparkGlm52HttpAppendJsonEscapedBytes(
        response->body,
        response->body_capacity,
        &body_bytes,
        token_text,
        token_text_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    written = snprintf(
        &response->body[body_bytes],
        response->body_capacity - body_bytes,
        "\"}\n\n");
    if (written < 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((uint32_t)written >= response->body_capacity - body_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    body_bytes += (uint32_t)written;
    response->status_code = 200u;
    response->flags = SPARK_GLM52_HTTP_GATEWAY_RESPONSE_FLAG_STREAM;
    response->content_type = "text/event-stream";
    response->body_bytes = body_bytes;
    return SPARK_STATUS_OK;
}
