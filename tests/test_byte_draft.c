#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_byte_draft.h"

#define TEST_BANK "5bc8f158616aef2b58e88cae6af7154c"
#define TEST_DIGEST "0fedab7a6bcdee67c26091670c1aba184dacda6b69a321611f1dd53910e917c9"
#define TEST_TAIL "beta "
#define TEST_TOKENIZER_PATH "build/test_byte_draft_tokenizer.json"
#define TOKEN_A 1u
#define TOKEN_B 2u
#define TOKEN_C 3u
#define TOKEN_SPACE 4u
#define TOKEN_NEWLINE 5u
#define TOKEN_AB 6u
#define TOKEN_ABC 7u
#define TOKEN_SPACE_A 8u
#define TOKEN_SPACE_AB 9u
#define TOKEN_SPACE_ABC 10u
#define TOKEN_UNKNOWN 11u

static uint32_t test_failures;

#define CHECK(condition, name) do { if (!(condition)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, name); test_failures++; } } while (0)

typedef struct TestServer
{
    int listen_fd;
    uint16_t port;
    pthread_t thread;
    const char *status_line;
    const char *body;
    uint32_t silent;
    char request[65536];
    size_t request_bytes;
} TestServer;

static void *TestServerMain(void *argument)
{
    TestServer *server = (TestServer *)argument;
    char header[512];
    size_t used = 0u,length = 0u;
    char *end;
    int fd = accept(server->listen_fd,0,0);
    if ( fd < 0 )
        return 0;
    for (;;)
    {
        ssize_t got = recv(fd,server->request + used,sizeof(server->request) - 1u - used,0);
        if ( got <= 0 )
            break;
        used += (size_t)got;
        server->request[used] = '\0';
        end = strstr(server->request,"\r\n\r\n");
        if ( end != 0 )
        {
            char *field = strstr(server->request,"Content-Length: ");
            length = field != 0 ? (size_t)strtoul(field + 16,0,10) : 0u;
            if ( used >= (size_t)(end + 4 - server->request) + length )
                break;
        }
    }
    server->request_bytes = used;
    if ( server->silent != 0u )
    {
        struct timespec pause = {1, 0};
        (void)nanosleep(&pause,0);
    }
    else
    {
        int header_bytes = snprintf(header,sizeof(header),"%s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",server->status_line,strlen(server->body));
        (void)send(fd,header,(size_t)header_bytes,0);
        (void)send(fd,server->body,strlen(server->body),0);
    }
    (void)close(fd);
    return 0;
}

static void TestServerStart(TestServer *server,const char *status_line,const char *body,uint32_t silent)
{
    struct sockaddr_in address;
    socklen_t length = sizeof(address);
    memset(server,0,sizeof(*server));
    server->status_line = status_line;
    server->body = body;
    server->silent = silent;
    server->listen_fd = socket(AF_INET,SOCK_STREAM,0);
    memset(&address,0,sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    (void)bind(server->listen_fd,(struct sockaddr *)&address,sizeof(address));
    (void)listen(server->listen_fd,4);
    (void)getsockname(server->listen_fd,(struct sockaddr *)&address,&length);
    server->port = ntohs(address.sin_port);
    (void)pthread_create(&server->thread,0,TestServerMain,server);
}

static void TestServerStop(TestServer *server)
{
    (void)pthread_join(server->thread,0);
    (void)close(server->listen_fd);
}

static SparkByteDraftForest test_forest = {"self", TEST_BANK, 1u};

static void TestConfiguration(SparkByteDraftConfiguration *configuration,uint16_t port)
{
    memset(configuration,0,sizeof(*configuration));
    configuration->host = "127.0.0.1";
    configuration->port = port;
    configuration->path = "/v1/memory";
    configuration->bearer_token = "test-token";
    configuration->forests = &test_forest;
    configuration->forest_count = 1u;
    configuration->min_match_bytes = 5u;
    configuration->max_candidates = 4u;
    configuration->max_continuation_bytes = 64u;
    configuration->budget = 8192u;
    configuration->work_budget.index_steps = 1000000u;
    configuration->work_budget.candidates_checked = 10000u;
    configuration->work_budget.physical_bytes_read = 16777216u;
    configuration->work_budget.decoded_bytes = 16777216u;
    configuration->work_budget.peak_scratch_bytes = 8388608u;
    configuration->work_budget.cache_bytes = 0u;
    configuration->work_budget.elapsed_ms = 1000u;
    configuration->io_timeout_ms = 2000u;
}

#define TEST_ENVELOPE(candidates, complete, stop) \
    "{\"ok\":true,\"candidates\":[" candidates "]," \
    "\"forest_snapshots\":{\"" TEST_BANK "\":1},\"index_generations\":{\"" TEST_BANK "\":1}," \
    "\"indexed_through\":{\"" TEST_BANK "\":1},\"complete\":" complete ",\"stop_reason\":\"" stop "\"," \
    "\"usage\":{\"index_steps\":33,\"candidates_checked\":1,\"physical_bytes_read\":124,\"decoded_bytes\":161," \
    "\"peak_scratch_bytes\":8388608,\"cache_bytes\":0,\"elapsed_ms\":2,\"indivisible_overrun\":false}}"

#define TEST_CANDIDATE(bank, start, end, matched, cend, b64, more, next) \
    "{\"source\":{\"bank_id\":\"" bank "\",\"event\":1,\"artifact\":{\"id\":\"fixture.txt\",\"revision\":\"r1\"}," \
    "\"content_sha256\":\"" TEST_DIGEST "\"},\"match_start\":" start ",\"match_end\":" end ",\"matched_bytes\":" matched "," \
    "\"continuation_end\":" cend ",\"continuation_b64\":\"" b64 "\",\"continuation_has_more\":" more ",\"next_source_offset\":" next "}"

#define TEST_GOOD TEST_CANDIDATE(TEST_BANK, "6", "11", "5", "23", "Z2FtbWEgZGVsdGEK", "false", "null")

static SparkStatus TestQuery(const char *status_line,const char *body,uint32_t silent,SparkByteDraftResult *result,TestServer *server)
{
    SparkByteDraftConfiguration configuration;
    SparkStatus status;
    TestServerStart(server,status_line,body,silent);
    TestConfiguration(&configuration,server->port);
    if ( silent != 0u )
        configuration.io_timeout_ms = 300u;
    status = SparkByteDraftQuery(&configuration,(const uint8_t *)TEST_TAIL,5u,result);
    TestServerStop(server);
    return status;
}

static void TestQueries(void)
{
    static SparkByteDraftResult result;
    static TestServer server;
    SparkByteDraftConfiguration configuration;
    SparkByteDraftForest twins[2] = {{"self", TEST_BANK, 1u}, {"other", TEST_BANK, 2u}};
    SparkStatus status;

    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_GOOD,"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_OK && result.complete == 1u && result.stop_reason == SPARK_BYTE_DRAFT_STOP_COMPLETE &&
        result.candidate_count == 1u,"the spec fixture parses as one complete candidate");
    CHECK(result.candidates[0].continuation_bytes == 12u && memcmp(result.candidates[0].continuation,"gamma delta\n",12u) == 0 &&
        result.candidates[0].matched_bytes == 5u && result.candidates[0].match_start == 6u && result.candidates[0].continuation_end == 23u &&
        result.candidates[0].has_more == 0u && strcmp(result.candidates[0].bank_id,TEST_BANK) == 0 &&
        strcmp(result.candidates[0].content_sha256,TEST_DIGEST) == 0,"candidate bytes, offsets and source match the fixture");
    CHECK(strstr(server.request,"POST /v1/memory HTTP/1.1\r\n") != 0 && strstr(server.request,"Authorization: Bearer test-token\r\n") != 0 &&
        strstr(server.request,"\"op\":\"draft_bytes\",\"tail_b64\":\"YmV0YSA=\"") != 0 &&
        strstr(server.request,"\"forests\":[\"self\"],\"forest_snapshots\":{\"" TEST_BANK "\":1}") != 0 &&
        strstr(server.request,"\"min_match_bytes\":5,\"max_candidates\":4,\"max_continuation_bytes\":64,\"budget\":8192") != 0 &&
        strstr(server.request,"\"elapsed_ms\":1000}") != 0 && strstr(server.request,"scope") == 0,"the request is the spec's draft_bytes object");
    SparkByteDraftResultRelease(&result);

    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE("","true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_OK && result.candidate_count == 0u && result.complete == 1u,"an empty complete answer is an exact miss");

    status = TestQuery("HTTP/1.1 200 OK","{\"ok\":false,\"code\":-5066,\"error\":\"tail\"}",0u,&result,&server);
    CHECK(status == SPARK_STATUS_IO_ERROR && result.service_error_code == -5066 && result.candidate_count == 0u,
        "a service error records its exact code and yields no proposal");

    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_GOOD,"false","work_exhausted"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_OK && result.complete == 0u && result.stop_reason == SPARK_BYTE_DRAFT_STOP_WORK_EXHAUSTED &&
        result.candidate_count == 1u,"a partial answer keeps its candidates and says it is partial");
    SparkByteDraftResultRelease(&result);

    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE(TEST_BANK,"6","11","4","23","Z2FtbWEgZGVsdGEK","false","null"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"matched_bytes that disagree with the span are refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE(TEST_BANK,"6","11","5","24","Z2FtbWEgZGVsdGEK","false","null"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"a continuation whose length disagrees with its span is refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE(TEST_BANK,"6","11","5","23","Z2FtbWEgZGVsdGEK","true","null"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"more bytes without a next offset are refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE(TEST_BANK,"6","11","5","23","Z2FtbWEgZGVsdGEK","false","23"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"a next offset without more bytes is refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_GOOD,"true","work_exhausted"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"complete with a partial stop reason is refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE("ffffffffffffffffffffffffffffffff","6","11","5","23","Z2FtbWEgZGVsdGEK","false","null"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"a candidate from a bank that was not selected is refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE(TEST_BANK,"6","11","5","12","Zh==","false","null"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"non-canonical base64 is refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_CANDIDATE(TEST_BANK,"0","5","5","6","Zg==","false","null") "," TEST_GOOD,"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_OK && result.candidate_count == 2u,"complete candidates in order parse");
    SparkByteDraftResultRelease(&result);
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_GOOD "," TEST_CANDIDATE(TEST_BANK,"0","5","5","6","Zg==","false","null"),"true","complete"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"complete candidates out of the documented order are refused");
    status = TestQuery("HTTP/1.1 200 OK",TEST_ENVELOPE(TEST_GOOD "," TEST_GOOD,"false","candidate_limit"),0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"two candidates for one continuation start are refused");
    status = TestQuery("HTTP/1.1 200 OK","{\"ok\":true,\"candidates\":[],\"forest_snapshots\":{\"" TEST_BANK "\":2},\"index_generations\":{},\"indexed_through\":{},\"complete\":true,\"stop_reason\":\"complete\",\"usage\":{}}",0u,&result,&server);
    CHECK(status == SPARK_STATUS_VALIDATION_FAILED,"an answer for another snapshot is refused");
    status = TestQuery("HTTP/1.1 400 Bad Request","{}",0u,&result,&server);
    CHECK(status == SPARK_STATUS_IO_ERROR,"an HTTP framing or authorization error yields no proposal");
    status = TestQuery("HTTP/1.1 200 OK","{}",1u,&result,&server);
    CHECK(status == SPARK_STATUS_BUSY,"a silent service is bounded by the IO timeout");

    TestConfiguration(&configuration,1u);
    configuration.min_match_bytes = 6u;
    CHECK(SparkByteDraftQuery(&configuration,(const uint8_t *)TEST_TAIL,5u,&result) == SPARK_STATUS_INVALID_ARGUMENT,
        "a minimum match longer than the tail is refused before any request");
    TestConfiguration(&configuration,1u);
    configuration.forests = twins;
    configuration.forest_count = 2u;
    CHECK(SparkByteDraftQuery(&configuration,(const uint8_t *)TEST_TAIL,5u,&result) == SPARK_STATUS_INVALID_ARGUMENT,
        "two forests resolving to one bank are refused");
    TestConfiguration(&configuration,1u);
    configuration.scope = "";
    CHECK(SparkByteDraftQuery(&configuration,(const uint8_t *)TEST_TAIL,5u,&result) == SPARK_STATUS_INVALID_ARGUMENT,
        "an empty scope is refused");
}

static void TestWriteTokenizer(void)
{
    FILE *file = fopen(TEST_TOKENIZER_PATH,"wb");
    if ( file == 0 )
    {
        test_failures++;
        return;
    }
    fprintf(file,
        "{\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"byte_fallback\":false,\"vocab\":{"
        "\"a\":%u,\"b\":%u,\"c\":%u,\"\\u0120\":%u,\"\\u010a\":%u,\"ab\":%u,\"abc\":%u,"
        "\"\\u0120a\":%u,\"\\u0120ab\":%u,\"\\u0120abc\":%u,\"<unk>\":%u},"
        "\"merges\":[\"\\u0120 a\",\"\\u0120a b\",\"\\u0120ab c\",\"a b\",\"ab c\"]},"
        "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false},\"added_tokens\":[]}\n",
        TOKEN_A,TOKEN_B,TOKEN_C,TOKEN_SPACE,TOKEN_NEWLINE,TOKEN_AB,TOKEN_ABC,TOKEN_SPACE_A,TOKEN_SPACE_AB,TOKEN_SPACE_ABC,TOKEN_UNKNOWN);
    (void)fclose(file);
}

static void TestAlignment(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    static const uint32_t committed[2] = {TOKEN_AB, TOKEN_SPACE_AB};
    static const uint32_t short_committed[1] = {TOKEN_AB};
    uint32_t draft[8],count = 99u,tokens = 0u,bytes = 0u;
    uint8_t tail[64];
    TestWriteTokenizer();
    SparkTokenizerReset(&tokenizer);
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes = SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = TEST_TOKENIZER_PATH;
    CHECK(SparkTokenizerLoadHuggingFaceJson(&tokenizer,&configuration) == SPARK_STATUS_OK,"test tokenizer loads");

    CHECK(SparkByteDraftTail(&tokenizer,committed,2u,8u,tail,sizeof(tail),&bytes,&tokens) == SPARK_STATUS_OK &&
        bytes == 5u && memcmp(tail,"ab ab",5u) == 0 && tokens == 2u,"the tail is the exact decoded bytes of the committed window");
    CHECK(SparkByteDraftAlign(&tokenizer,committed,2u,8u,(const uint8_t *)" abc\nab",7u,8u,draft,&count) == SPARK_STATUS_OK &&
        count == 2u && draft[0] == TOKEN_SPACE_ABC && draft[1] == TOKEN_NEWLINE,
        "an aligned continuation proposes its tokens after the committed boundary, without the possibly cut final token");
    CHECK(SparkByteDraftAlign(&tokenizer,committed,2u,8u,(const uint8_t *)" abc\nab",7u,1u,draft,&count) == SPARK_STATUS_OK &&
        count == 1u && draft[0] == TOKEN_SPACE_ABC,"truncation keeps the first proposed tokens");
    CHECK(SparkByteDraftAlign(&tokenizer,short_committed,1u,8u,(const uint8_t *)"c ab ab",7u,8u,draft,&count) == SPARK_STATUS_OK &&
        count == 0u,"a continuation that merges across the committed boundary yields no draft");
    CHECK(SparkByteDraftAlign(&tokenizer,committed,2u,8u,(const uint8_t *)" abc\n\xc3",6u,8u,draft,&count) == SPARK_STATUS_OK &&
        count == 1u && draft[0] == TOKEN_SPACE_ABC,"a continuation ending inside a UTF-8 scalar is cut at the last whole scalar");
    CHECK(SparkByteDraftAlign(&tokenizer,committed,2u,8u,(const uint8_t *)"\xc3",1u,8u,draft,&count) == SPARK_STATUS_OK &&
        count == 0u,"a continuation of one partial scalar yields no draft");
    SparkTokenizerDestroy(&tokenizer);
}

int main(void)
{
    TestQueries();
    TestAlignment();
    if ( test_failures != 0u )
    {
        fprintf(stderr,"test_byte_draft: %u failures\n",test_failures);
        return 1;
    }
    printf("test_byte_draft: PASS (spec fixture, invariants, service errors, timeouts, tokenizer alignment)\n");
    return 0;
}
