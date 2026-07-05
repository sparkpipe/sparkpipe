#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_compat_api.h"

static void SparkTestCompatOpenAiChat(void)
{
    static const char RequestJson[] =
        "{"
        "\"model\":\"glm-5.2\","
        "\"max_tokens\":17,"
        "\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are terse.\"},"
        "{\"role\":\"user\",\"content\":\"Read this C code.\"}"
        "]"
        "}";
    SparkGlm52CompatTextRequest request;
    char text[512];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    request.client_id = 10u;
    request.client_request_id = 20u;
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(request.output_token_budget == 17u);
    assert(request.text_bytes == strlen("system: You are terse.\nuser: Read this C code.\n"));
    assert(strcmp(text, "system: You are terse.\nuser: Read this C code.\n") == 0);
}

static void SparkTestCompatOpenAiPrompt(void)
{
    static const char RequestJson[] =
        "{\"model\":\"glm-5.2\",\"max_completion_tokens\":5,\"prompt\":\"plain prompt\"}";
    SparkGlm52CompatTextRequest request;
    char text[64];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(request.output_token_budget == 5u);
    assert(strcmp(text, "plain prompt") == 0);
}


static void SparkTestCompatOpenAiChatWithFiles(void)
{
    static const char RequestJson[] =
        "{"
        "\"model\":\"glm-5.2\","
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Use the attachment.\"}]}],"
        "\"files\":[{\"filename\":\"notes.txt\",\"content\":\"alpha\\nbeta\"}]"
        "}";
    SparkGlm52CompatTextRequest request;
    char text[512];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(strstr(text, "user: Use the attachment.\n") != 0);
    assert(strstr(text, "[uploaded file: notes.txt]") != 0);
    assert(strstr(text, "alpha\nbeta") != 0);
    assert(strstr(text, "[/uploaded file]") != 0);
}

static void SparkTestCompatAnthropicMessages(void)
{
    static const char RequestJson[] =
        "{"
        "\"model\":\"glm-5.2\","
        "\"system\":\"Stay exact.\","
        "\"max_tokens\":9,"
        "\"messages\":["
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"First\"},"
        "{\"type\":\"text\",\"text\":\" second\"}]},"
        "{\"role\":\"assistant\",\"content\":\"Ack\"}"
        "]"
        "}";
    SparkGlm52CompatTextRequest request;
    char text[512];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareAnthropicJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_OK);
    assert(request.output_token_budget == 9u);
    assert(strcmp(text, "system: Stay exact.\nuser: First second\nassistant: Ack\n") == 0);
}

static void SparkTestCompatRejectsSmallBuffer(void)
{
    static const char RequestJson[] =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"too long for buffer\"}]}";
    SparkGlm52CompatTextRequest request;
    char text[8];

    SparkGlm52CompatInitializeTextRequest(&request, text, sizeof(text));
    assert(SparkGlm52CompatPrepareOpenAiJson(
        RequestJson,
        ((uint32_t)strlen(RequestJson)),
        &request) == SPARK_STATUS_CAPACITY_EXCEEDED);
}

int main(void)
{
    SparkTestCompatOpenAiChat();
    SparkTestCompatOpenAiPrompt();
    SparkTestCompatOpenAiChatWithFiles();
    SparkTestCompatAnthropicMessages();
    SparkTestCompatRejectsSmallBuffer();
    return 0;
}
