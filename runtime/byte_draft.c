#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_byte_draft.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SPARK_BYTE_DRAFT_HEADER_BYTES_MAX 16384u

static const char spark_byte_draft_null[] = "null";
static const char spark_byte_draft_http[] = "HTTP/1.";
static const char spark_byte_draft_ok[] = " 200";
static const char spark_byte_draft_length[] = "Content-Length:";

typedef struct SparkByteDraftBuffer
{
    char *data;
    size_t used;
    size_t capacity;
    uint32_t failed;
} SparkByteDraftBuffer;

static const char spark_byte_draft_base64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void SparkByteDraftAppendBytes(SparkByteDraftBuffer *buffer,const char *bytes,size_t count)
{
    char *grown;
    size_t capacity;
    if ( buffer->failed != 0u )
        return;
    if ( buffer->used + count + 1u > buffer->capacity )
    {
        capacity = buffer->capacity != 0u ? buffer->capacity : 1024u;
        while ( capacity < buffer->used + count + 1u )
            capacity *= 2u;
        grown = (char *)realloc(buffer->data,capacity);
        if ( grown == 0 )
        {
            buffer->failed = 1u;
            return;
        }
        buffer->data = grown;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->used,bytes,count);
    buffer->used += count;
    buffer->data[buffer->used] = '\0';
}

static void SparkByteDraftAppendText(SparkByteDraftBuffer *buffer,const char *text)
{
    SparkByteDraftAppendBytes(buffer,text,strlen(text));
}

static void SparkByteDraftAppendUnsigned(SparkByteDraftBuffer *buffer,uint64_t value)
{
    char text[24];
    (void)snprintf(text,sizeof(text),"%llu",(unsigned long long)value);
    SparkByteDraftAppendText(buffer,text);
}

static void SparkByteDraftAppendJsonString(SparkByteDraftBuffer *buffer,const char *text)
{
    char escape[8];
    const unsigned char *cursor;
    SparkByteDraftAppendBytes(buffer,"\"",1u);
    for ( cursor = (const unsigned char *)text; *cursor != 0u; cursor++ )
    {
        if ( *cursor == '"' || *cursor == '\\' )
        {
            escape[0] = '\\';
            escape[1] = (char)*cursor;
            SparkByteDraftAppendBytes(buffer,escape,2u);
        }
        else if ( *cursor < 0x20u )
        {
            (void)snprintf(escape,sizeof(escape),"\\u%04x",(unsigned)*cursor);
            SparkByteDraftAppendText(buffer,escape);
        }
        else
            SparkByteDraftAppendBytes(buffer,(const char *)cursor,1u);
    }
    SparkByteDraftAppendBytes(buffer,"\"",1u);
}

static void SparkByteDraftAppendBase64(SparkByteDraftBuffer *buffer,const uint8_t *bytes,uint32_t count)
{
    char quad[4];
    uint32_t index,value;
    for ( index = 0u; index < count; index += 3u )
    {
        value = (uint32_t)bytes[index] << 16u;
        if ( index + 1u < count )
            value |= (uint32_t)bytes[index + 1u] << 8u;
        if ( index + 2u < count )
            value |= bytes[index + 2u];
        quad[0] = spark_byte_draft_base64[(value >> 18u) & 63u];
        quad[1] = spark_byte_draft_base64[(value >> 12u) & 63u];
        quad[2] = index + 1u < count ? spark_byte_draft_base64[(value >> 6u) & 63u] : '=';
        quad[3] = index + 2u < count ? spark_byte_draft_base64[value & 63u] : '=';
        SparkByteDraftAppendBytes(buffer,quad,4u);
    }
}

static int32_t SparkByteDraftBase64Value(char symbol)
{
    const char *found = symbol != '\0' ? strchr(spark_byte_draft_base64,symbol) : 0;
    return found != 0 ? (int32_t)(found - spark_byte_draft_base64) : -1;
}

static SparkStatus SparkByteDraftDecodeBase64(const char *text,uint8_t **bytes_out,uint32_t *count_out)
{
    size_t length = strlen(text),index;
    uint32_t padding,count = 0u,value;
    int32_t digits[4];
    uint8_t *bytes;
    *bytes_out = 0;
    *count_out = 0u;
    if ( length == 0u || length % 4u != 0u )
        return SPARK_STATUS_VALIDATION_FAILED;
    padding = text[length - 1u] == '=' ? (text[length - 2u] == '=' ? 2u : 1u) : 0u;
    bytes = (uint8_t *)malloc(length / 4u * 3u);
    if ( bytes == 0 )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    for ( index = 0u; index < length; index += 4u )
    {
        uint32_t last = index + 4u == length ? 1u : 0u,slot;
        for ( slot = 0u; slot < 4u; slot++ )
        {
            uint32_t pad_slot = last != 0u && slot >= 4u - padding ? 1u : 0u;
            digits[slot] = pad_slot != 0u ? (text[index + slot] == '=' ? 0 : -1) : SparkByteDraftBase64Value(text[index + slot]);
            if ( digits[slot] < 0 )
            {
                free(bytes);
                return SPARK_STATUS_VALIDATION_FAILED;
            }
        }
        value = (uint32_t)digits[0] << 18u | (uint32_t)digits[1] << 12u | (uint32_t)digits[2] << 6u | (uint32_t)digits[3];
        bytes[count++] = (uint8_t)(value >> 16u);
        if ( last == 0u || padding < 2u )
            bytes[count++] = (uint8_t)(value >> 8u);
        if ( last == 0u || padding < 1u )
            bytes[count++] = (uint8_t)value;
        if ( last != 0u && ((padding == 1u && (value & 0xffu) != 0u) || (padding == 2u && (value & 0xffffu) != 0u)) )
        {
            free(bytes);
            return SPARK_STATUS_VALIDATION_FAILED;
        }
    }
    *bytes_out = bytes;
    *count_out = count;
    return SPARK_STATUS_OK;
}

static uint64_t SparkByteDraftNowMilli(void)
{
    struct timespec now;
    if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
        return 0u;
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static int SparkByteDraftRemainingMilli(uint64_t deadline)
{
    uint64_t now = SparkByteDraftNowMilli();
    return now >= deadline ? 0 : (int)(deadline - now);
}

static SparkStatus SparkByteDraftConnect(const SparkByteDraftConfiguration *configuration,uint64_t deadline,int *fd_out)
{
    struct addrinfo hints,*addresses = 0,*address;
    struct pollfd poll_fd;
    char port[8];
    socklen_t length = sizeof(int);
    int fd = -1,error = ETIMEDOUT,flags;
    memset(&hints,0,sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    (void)snprintf(port,sizeof(port),"%u",(unsigned)configuration->port);
    if ( getaddrinfo(configuration->host,port,&hints,&addresses) != 0 )
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    for ( address = addresses; address != 0; address = address->ai_next )
    {
        fd = socket(address->ai_family,address->ai_socktype,address->ai_protocol);
        if ( fd < 0 )
            continue;
        flags = fcntl(fd,F_GETFL);
        if ( flags < 0 || fcntl(fd,F_SETFL,flags | O_NONBLOCK) != 0 )
            error = errno;
        else if ( connect(fd,address->ai_addr,address->ai_addrlen) == 0 )
            error = 0;
        else if ( errno == EINPROGRESS )
        {
            poll_fd.fd = fd;
            poll_fd.events = POLLOUT;
            poll_fd.revents = 0;
            if ( poll(&poll_fd,1u,SparkByteDraftRemainingMilli(deadline)) <= 0 )
                error = ETIMEDOUT;
            else if ( getsockopt(fd,SOL_SOCKET,SO_ERROR,&error,&length) != 0 )
                error = errno;
        }
        else
            error = errno;
        if ( error == 0 )
            break;
        (void)close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    if ( fd < 0 )
    {
        if ( error == ETIMEDOUT )
            SPARK_FAIL(SPARK_STATUS_BUSY);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    *fd_out = fd;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftExchange(const SparkByteDraftConfiguration *configuration,const SparkByteDraftBuffer *body,SparkByteDraftBuffer *response)
{
    SparkByteDraftBuffer request = {0};
    struct pollfd poll_fd;
    uint64_t deadline = SparkByteDraftNowMilli() + configuration->io_timeout_ms;
    size_t sent = 0u;
    char chunk[16384];
    ssize_t moved;
    int fd = -1;
    SparkStatus status;
    SparkByteDraftAppendText(&request,"POST ");
    SparkByteDraftAppendText(&request,configuration->path);
    SparkByteDraftAppendText(&request," HTTP/1.1\r\nHost: ");
    SparkByteDraftAppendText(&request,configuration->host);
    SparkByteDraftAppendText(&request,"\r\nContent-Type: application/json\r\nAuthorization: Bearer ");
    SparkByteDraftAppendText(&request,configuration->bearer_token);
    SparkByteDraftAppendText(&request,"\r\nConnection: close\r\nContent-Length: ");
    SparkByteDraftAppendUnsigned(&request,body->used);
    SparkByteDraftAppendText(&request,"\r\n\r\n");
    SparkByteDraftAppendBytes(&request,body->data,body->used);
    if ( request.failed != 0u )
    {
        free(request.data);
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    status = SparkByteDraftConnect(configuration,deadline,&fd);
    while ( status == SPARK_STATUS_OK && sent < request.used )
    {
        poll_fd.fd = fd;
        poll_fd.events = POLLOUT;
        poll_fd.revents = 0;
        if ( poll(&poll_fd,1u,SparkByteDraftRemainingMilli(deadline)) <= 0 )
            status = SPARK_STATUS_BUSY;
        else if ( (moved = send(fd,request.data + sent,request.used - sent,0)) > 0 )
            sent += (size_t)moved;
        else if ( moved < 0 && (errno == EINTR || errno == EAGAIN) )
            continue;
        else
            status = SPARK_STATUS_IO_ERROR;
    }
    free(request.data);
    while ( status == SPARK_STATUS_OK )
    {
        poll_fd.fd = fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        if ( poll(&poll_fd,1u,SparkByteDraftRemainingMilli(deadline)) <= 0 )
        {
            status = SPARK_STATUS_BUSY;
            break;
        }
        moved = recv(fd,chunk,sizeof(chunk),0);
        if ( moved == 0 )
            break;
        if ( moved < 0 )
        {
            if ( errno == EINTR || errno == EAGAIN )
                continue;
            status = SPARK_STATUS_IO_ERROR;
            break;
        }
        if ( response->used + (size_t)moved > SPARK_BYTE_DRAFT_MAX_RESPONSE_BYTES + SPARK_BYTE_DRAFT_HEADER_BYTES_MAX )
        {
            status = SPARK_STATUS_CAPACITY_EXCEEDED;
            break;
        }
        SparkByteDraftAppendBytes(response,chunk,(size_t)moved);
        if ( response->failed != 0u )
            status = SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if ( fd >= 0 )
        (void)close(fd);
    if ( status != SPARK_STATUS_OK )
    {
        fprintf(stderr,"BYTE-DRAFT-TRANSPORT-FAIL host=%s port=%u status=%d\n",configuration->host,(unsigned)configuration->port,(int)status);
        SPARK_RETURN(status);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftHttpBody(const SparkByteDraftBuffer *response,const char **body_out,size_t *body_bytes_out)
{
    const char *end,*cursor,*line;
    unsigned long long length = 0u;
    uint32_t have_length = 0u;
    if ( response->used < sizeof(spark_byte_draft_http) + sizeof(spark_byte_draft_ok) ||
         strncmp(response->data,spark_byte_draft_http,sizeof(spark_byte_draft_http) - 1u) != 0 ||
         strncmp(response->data + sizeof(spark_byte_draft_http),spark_byte_draft_ok,sizeof(spark_byte_draft_ok) - 1u) != 0 )
        return SPARK_STATUS_IO_ERROR;
    end = strstr(response->data,"\r\n\r\n");
    if ( end == 0 )
        return SPARK_STATUS_IO_ERROR;
    for ( line = strstr(response->data,"\r\n"); line != 0 && line < end; line = strstr(line + 2u,"\r\n") )
    {
        cursor = line + 2u;
        if ( strncasecmp(cursor,spark_byte_draft_length,sizeof(spark_byte_draft_length) - 1u) == 0 )
        {
            const char *value = cursor + sizeof(spark_byte_draft_length) - 1u;
            char *parsed_end;
            errno = 0;
            length = strtoull(value,&parsed_end,10);
            if ( errno != 0 || parsed_end == value )
                return SPARK_STATUS_IO_ERROR;
            have_length = 1u;
        }
    }
    *body_out = end + 4u;
    *body_bytes_out = response->used - (size_t)(end + 4u - response->data);
    if ( have_length == 0u || length != *body_bytes_out || length > SPARK_BYTE_DRAFT_MAX_RESPONSE_BYTES )
        return SPARK_STATUS_IO_ERROR;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftBuildRequest(const SparkByteDraftConfiguration *configuration,const uint8_t *tail,uint32_t tail_bytes,SparkByteDraftBuffer *body)
{
    const SparkByteDraftWorkBudget *work = &configuration->work_budget;
    uint32_t index;
    SparkByteDraftAppendText(body,"{\"op\":\"draft_bytes\",\"tail_b64\":\"");
    SparkByteDraftAppendBase64(body,tail,tail_bytes);
    SparkByteDraftAppendText(body,"\",\"forests\":[");
    for ( index = 0u; index < configuration->forest_count; index++ )
    {
        if ( index != 0u )
            SparkByteDraftAppendText(body,",");
        SparkByteDraftAppendJsonString(body,configuration->forests[index].alias);
    }
    SparkByteDraftAppendText(body,"],\"forest_snapshots\":{");
    for ( index = 0u; index < configuration->forest_count; index++ )
    {
        if ( index != 0u )
            SparkByteDraftAppendText(body,",");
        SparkByteDraftAppendJsonString(body,configuration->forests[index].bank_id);
        SparkByteDraftAppendText(body,":");
        SparkByteDraftAppendUnsigned(body,configuration->forests[index].snapshot);
    }
    SparkByteDraftAppendText(body,"},\"min_match_bytes\":");
    SparkByteDraftAppendUnsigned(body,configuration->min_match_bytes);
    SparkByteDraftAppendText(body,",\"max_candidates\":");
    SparkByteDraftAppendUnsigned(body,configuration->max_candidates);
    SparkByteDraftAppendText(body,",\"max_continuation_bytes\":");
    SparkByteDraftAppendUnsigned(body,configuration->max_continuation_bytes);
    SparkByteDraftAppendText(body,",\"budget\":");
    SparkByteDraftAppendUnsigned(body,configuration->budget);
    SparkByteDraftAppendText(body,",\"work_budget\":{\"index_steps\":");
    SparkByteDraftAppendUnsigned(body,work->index_steps);
    SparkByteDraftAppendText(body,",\"candidates_checked\":");
    SparkByteDraftAppendUnsigned(body,work->candidates_checked);
    SparkByteDraftAppendText(body,",\"physical_bytes_read\":");
    SparkByteDraftAppendUnsigned(body,work->physical_bytes_read);
    SparkByteDraftAppendText(body,",\"decoded_bytes\":");
    SparkByteDraftAppendUnsigned(body,work->decoded_bytes);
    SparkByteDraftAppendText(body,",\"peak_scratch_bytes\":");
    SparkByteDraftAppendUnsigned(body,work->peak_scratch_bytes);
    SparkByteDraftAppendText(body,",\"cache_bytes\":");
    SparkByteDraftAppendUnsigned(body,work->cache_bytes);
    SparkByteDraftAppendText(body,",\"elapsed_ms\":");
    SparkByteDraftAppendUnsigned(body,work->elapsed_ms);
    SparkByteDraftAppendText(body,"}");
    if ( configuration->scope != 0 )
    {
        SparkByteDraftAppendText(body,",\"scope\":");
        SparkByteDraftAppendJsonString(body,configuration->scope);
    }
    SparkByteDraftAppendText(body,"}");
    return body->failed != 0u ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftValidateConfiguration(const SparkByteDraftConfiguration *configuration,const uint8_t *tail,uint32_t tail_bytes)
{
    const SparkByteDraftWorkBudget *work;
    uint32_t index,other;
    if ( configuration == 0 || tail == 0 || tail_bytes == 0u || tail_bytes > SPARK_BYTE_DRAFT_MAX_TAIL_BYTES ||
         configuration->host == 0 || configuration->port == 0u || configuration->path == 0 || configuration->path[0] != '/' ||
         configuration->bearer_token == 0 || configuration->bearer_token[0] == '\0' ||
         configuration->forests == 0 || configuration->forest_count == 0u || configuration->forest_count > SPARK_BYTE_DRAFT_MAX_FORESTS ||
         (configuration->scope != 0 && configuration->scope[0] == '\0') ||
         configuration->min_match_bytes == 0u || configuration->min_match_bytes > tail_bytes ||
         configuration->max_candidates == 0u || configuration->max_candidates > SPARK_BYTE_DRAFT_MAX_CANDIDATES ||
         configuration->max_continuation_bytes == 0u || configuration->max_continuation_bytes > SPARK_BYTE_DRAFT_MAX_CONTINUATION_BYTES ||
         configuration->budget < 2u || configuration->budget > SPARK_BYTE_DRAFT_MAX_BUDGET || configuration->io_timeout_ms == 0u )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    work = &configuration->work_budget;
    if ( work->index_steps > (uint64_t)INT64_MAX || work->candidates_checked > (uint64_t)INT64_MAX ||
         work->physical_bytes_read > (uint64_t)INT64_MAX || work->decoded_bytes > (uint64_t)INT64_MAX ||
         work->peak_scratch_bytes > (uint64_t)INT64_MAX || work->cache_bytes > (uint64_t)INT64_MAX || work->elapsed_ms > (uint64_t)INT64_MAX )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    for ( index = 0u; index < configuration->forest_count; index++ )
    {
        const SparkByteDraftForest *forest = &configuration->forests[index];
        if ( forest->alias == 0 || forest->alias[0] == '\0' || forest->bank_id == 0 || forest->bank_id[0] == '\0' ||
             strlen(forest->bank_id) >= SPARK_BYTE_DRAFT_BANK_ID_BYTES || forest->snapshot > (uint64_t)INT64_MAX )
            SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
        for ( other = 0u; other < index; other++ )
            if ( strcmp(configuration->forests[other].bank_id,forest->bank_id) == 0 || strcmp(configuration->forests[other].alias,forest->alias) == 0 )
                SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftMemberU64(const SparkJsonDocument *document,int32_t object,const char *name,uint64_t *value)
{
    int32_t token = SparkJsonFindObjectMember(document,object,name);
    if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_PRIMITIVE) )
        return SPARK_STATUS_VALIDATION_FAILED;
    return SparkJsonGetUInt64(document,token,value) == SPARK_STATUS_OK ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}

static SparkStatus SparkByteDraftMemberBool(const SparkJsonDocument *document,int32_t object,const char *name,uint32_t *value)
{
    int32_t token = SparkJsonFindObjectMember(document,object,name);
    bool parsed = false;
    if ( token < 0 || SparkJsonGetBoolean(document,token,&parsed) != SPARK_STATUS_OK )
        return SPARK_STATUS_VALIDATION_FAILED;
    *value = parsed ? 1u : 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftMemberString(const SparkJsonDocument *document,int32_t object,const char *name,char **text)
{
    int32_t token = SparkJsonFindObjectMember(document,object,name);
    *text = 0;
    if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_STRING) )
        return SPARK_STATUS_VALIDATION_FAILED;
    return SparkJsonCopyString(document,token,text) == SPARK_STATUS_OK ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}

static uint32_t SparkByteDraftIsNull(const SparkJsonDocument *document,int32_t token)
{
    char *raw = 0;
    uint32_t bytes = 0u,is_null;
    if ( token < 0 || SparkJsonCopyRawValue(document,token,&raw,&bytes) != SPARK_STATUS_OK )
        return 0u;
    is_null = bytes == sizeof(spark_byte_draft_null) - 1u && memcmp(raw,spark_byte_draft_null,sizeof(spark_byte_draft_null) - 1u) == 0 ? 1u : 0u;
    free(raw);
    return is_null;
}

static uint32_t SparkByteDraftDigestValid(const char *text)
{
    uint32_t index;
    for ( index = 0u; index < 64u; index++ )
        if ( !((text[index] >= '0' && text[index] <= '9') || (text[index] >= 'a' && text[index] <= 'f')) )
            return 0u;
    return text[64] == '\0' ? 1u : 0u;
}

static uint32_t SparkByteDraftStopReason(const char *text)
{
    static const char *const names[] = {"","complete","candidate_limit","budget_exhausted","work_exhausted","pending_index","cancelled"};
    uint32_t index;
    for ( index = 1u; index < sizeof(names) / sizeof(names[0]); index++ )
        if ( strcmp(text,names[index]) == 0 )
            return index;
    return 0u;
}

static SparkStatus SparkByteDraftInvalid(const char *reason)
{
    fprintf(stderr,"BYTE-DRAFT-INVALID %s\n",reason);
    SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
}

static SparkStatus SparkByteDraftParseCandidate(const SparkByteDraftConfiguration *configuration,const SparkJsonDocument *document,int32_t object,const uint8_t *tail,uint32_t tail_bytes,SparkByteDraftCandidate *candidate)
{
    static const char *const members[] = {"source","match_start","match_end","matched_bytes","continuation_end","continuation_b64","continuation_has_more","next_source_offset"};
    static const char *const source_members[] = {"bank_id","event","artifact","content_sha256"};
    uint64_t matched = 0u,next = 0u;
    char *bank = 0,*digest = 0,*encoded = 0;
    int32_t source,next_token;
    uint32_t index,known = 0u;
    SparkStatus status;
    (void)tail;
    if ( !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) ||
         SparkJsonValidateObjectMembersExact(document,object,members,sizeof(members) / sizeof(members[0])) != SPARK_STATUS_OK )
        return SparkByteDraftInvalid("candidate members");
    source = SparkJsonFindObjectMember(document,object,"source");
    if ( !SparkJsonTokenIsType(document,source,SPARK_JSON_TOKEN_OBJECT) ||
         SparkJsonValidateObjectMembersExact(document,source,source_members,sizeof(source_members) / sizeof(source_members[0])) != SPARK_STATUS_OK )
        return SparkByteDraftInvalid("source members");
    status = SparkByteDraftMemberString(document,source,"bank_id",&bank);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberString(document,source,"content_sha256",&digest);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberU64(document,source,"event",&candidate->event);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberU64(document,object,"match_start",&candidate->match_start);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberU64(document,object,"match_end",&candidate->match_end);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberU64(document,object,"matched_bytes",&matched);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberU64(document,object,"continuation_end",&candidate->continuation_end);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberBool(document,object,"continuation_has_more",&candidate->has_more);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberString(document,object,"continuation_b64",&encoded);
    if ( status != SPARK_STATUS_OK )
    {
        free(bank);
        free(digest);
        free(encoded);
        return SparkByteDraftInvalid("candidate field types");
    }
    for ( index = 0u; index < configuration->forest_count; index++ )
        known |= strcmp(configuration->forests[index].bank_id,bank) == 0 ? 1u : 0u;
    if ( known == 0u || strlen(bank) >= SPARK_BYTE_DRAFT_BANK_ID_BYTES || SparkByteDraftDigestValid(digest) == 0u )
        status = SparkByteDraftInvalid("candidate bank or digest");
    else
    {
        memcpy(candidate->bank_id,bank,strlen(bank) + 1u);
        memcpy(candidate->content_sha256,digest,SPARK_BYTE_DRAFT_DIGEST_HEX_BYTES);
        status = SparkByteDraftDecodeBase64(encoded,&candidate->continuation,&candidate->continuation_bytes);
        if ( status != SPARK_STATUS_OK )
            status = SparkByteDraftInvalid("continuation base64");
    }
    free(bank);
    free(digest);
    free(encoded);
    if ( status != SPARK_STATUS_OK )
        return status;
    next_token = SparkJsonFindObjectMember(document,object,"next_source_offset");
    if ( matched == 0u || matched > tail_bytes || matched < configuration->min_match_bytes ||
         candidate->match_end < candidate->match_start || candidate->match_end - candidate->match_start != matched )
        return SparkByteDraftInvalid("matched span");
    if ( candidate->continuation_end <= candidate->match_end ||
         candidate->continuation_end - candidate->match_end != candidate->continuation_bytes ||
         candidate->continuation_bytes > configuration->max_continuation_bytes )
        return SparkByteDraftInvalid("continuation span");
    if ( candidate->has_more != 0u )
    {
        if ( SparkByteDraftMemberU64(document,object,"next_source_offset",&next) != SPARK_STATUS_OK || next != candidate->continuation_end )
            return SparkByteDraftInvalid("next_source_offset with more bytes");
    }
    else if ( SparkByteDraftIsNull(document,next_token) == 0u )
        return SparkByteDraftInvalid("next_source_offset without more bytes");
    candidate->matched_bytes = (uint32_t)matched;
    return SPARK_STATUS_OK;
}

static int SparkByteDraftCompare(const SparkByteDraftCandidate *left,const SparkByteDraftCandidate *right)
{
    int bank;
    if ( left->matched_bytes != right->matched_bytes )
        return left->matched_bytes > right->matched_bytes ? -1 : 1;
    bank = strcmp(left->bank_id,right->bank_id);
    if ( bank != 0 )
        return bank;
    if ( left->event != right->event )
        return left->event < right->event ? -1 : 1;
    if ( left->match_start != right->match_start )
        return left->match_start < right->match_start ? -1 : 1;
    return 0;
}

static SparkStatus SparkByteDraftParseSnapshots(const SparkByteDraftConfiguration *configuration,const SparkJsonDocument *document,int32_t root)
{
    int32_t map = SparkJsonFindObjectMember(document,root,"forest_snapshots"),token;
    uint64_t value;
    uint32_t index;
    if ( !SparkJsonTokenIsType(document,map,SPARK_JSON_TOKEN_OBJECT) )
        return SparkByteDraftInvalid("forest_snapshots type");
    for ( index = 0u; index < configuration->forest_count; index++ )
    {
        token = SparkJsonFindObjectMember(document,map,configuration->forests[index].bank_id);
        if ( token < 0 || SparkJsonGetUInt64(document,token,&value) != SPARK_STATUS_OK || value != configuration->forests[index].snapshot )
            return SparkByteDraftInvalid("forest_snapshots differ from the request");
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftParse(const SparkByteDraftConfiguration *configuration,const char *body,size_t body_bytes,const uint8_t *tail,uint32_t tail_bytes,SparkByteDraftResult *result)
{
    static const char *const members[] = {"ok","candidates","forest_snapshots","index_generations","indexed_through","complete","stop_reason","usage"};
    SparkJsonDocument document;
    char *text = 0,*raw = 0;
    int32_t root,candidates,element;
    uint32_t ok = 0u,count,index,raw_bytes = 0u;
    SparkStatus status;
    SparkJsonDocumentReset(&document);
    if ( SparkJsonParseText(body,body_bytes,&document) != SPARK_STATUS_OK )
        return SparkByteDraftInvalid("response is not JSON");
    root = SparkJsonGetRootToken(&document);
    status = SparkByteDraftMemberBool(&document,root,"ok",&ok);
    if ( status == SPARK_STATUS_OK && ok == 0u )
    {
        int32_t code = SparkJsonFindObjectMember(&document,root,"code");
        char *error_text = 0;
        if ( code < 0 || SparkJsonCopyRawValue(&document,code,&raw,&raw_bytes) != SPARK_STATUS_OK || raw_bytes == 0u || raw[0] != '-' )
            status = SparkByteDraftInvalid("ok:false without a negative code");
        else
        {
            result->service_error_code = strtoll(raw,0,10);
            (void)SparkByteDraftMemberString(&document,root,"error",&error_text);
            fprintf(stderr,"BYTE-DRAFT-SERVICE-ERROR code=%lld error=%s\n",(long long)result->service_error_code,error_text != 0 ? error_text : "");
            free(error_text);
            status = SPARK_STATUS_IO_ERROR;
        }
        free(raw);
        SparkJsonDocumentDestroy(&document);
        SPARK_RETURN(status);
    }
    if ( status != SPARK_STATUS_OK || SparkJsonValidateObjectMembersExact(&document,root,members,sizeof(members) / sizeof(members[0])) != SPARK_STATUS_OK )
    {
        SparkJsonDocumentDestroy(&document);
        return SparkByteDraftInvalid("response members");
    }
    status = SparkByteDraftMemberBool(&document,root,"complete",&result->complete);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftMemberString(&document,root,"stop_reason",&text);
    if ( status == SPARK_STATUS_OK )
    {
        result->stop_reason = SparkByteDraftStopReason(text);
        if ( result->stop_reason == 0u ||
             (result->complete != 0u) != (result->stop_reason == SPARK_BYTE_DRAFT_STOP_COMPLETE) )
            status = SparkByteDraftInvalid("complete and stop_reason disagree");
    }
    else
        status = SparkByteDraftInvalid("complete or stop_reason type");
    free(text);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftParseSnapshots(configuration,&document,root);
    candidates = SparkJsonFindObjectMember(&document,root,"candidates");
    if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,candidates,SPARK_JSON_TOKEN_ARRAY) )
        status = SparkByteDraftInvalid("candidates type");
    count = status == SPARK_STATUS_OK ? SparkJsonGetArrayElementCount(&document,candidates) : 0u;
    if ( count > configuration->max_candidates )
        status = SparkByteDraftInvalid("more candidates than requested");
    for ( index = 0u; status == SPARK_STATUS_OK && index < count; index++ )
    {
        uint32_t other;
        element = SparkJsonGetArrayElement(&document,candidates,index);
        status = SparkByteDraftParseCandidate(configuration,&document,element,tail,tail_bytes,&result->candidates[index]);
        if ( status == SPARK_STATUS_OK )
            result->candidate_count = index + 1u;
        for ( other = 0u; status == SPARK_STATUS_OK && other < index; other++ )
            if ( strcmp(result->candidates[other].bank_id,result->candidates[index].bank_id) == 0 &&
                 result->candidates[other].event == result->candidates[index].event &&
                 result->candidates[other].match_end == result->candidates[index].match_end )
                status = SparkByteDraftInvalid("two candidates for one continuation start");
        if ( status == SPARK_STATUS_OK && result->complete != 0u && index != 0u &&
             SparkByteDraftCompare(&result->candidates[index - 1u],&result->candidates[index]) > 0 )
            status = SparkByteDraftInvalid("complete candidates out of order");
    }
    SparkJsonDocumentDestroy(&document);
    return status;
}

void SparkByteDraftResultRelease(SparkByteDraftResult *result)
{
    uint32_t index;
    if ( result == 0 )
        return;
    for ( index = 0u; index < SPARK_BYTE_DRAFT_MAX_CANDIDATES; index++ )
    {
        free(result->candidates[index].continuation);
        result->candidates[index].continuation = 0;
        result->candidates[index].continuation_bytes = 0u;
    }
    result->candidate_count = 0u;
}

SparkStatus SparkByteDraftQuery(
    const SparkByteDraftConfiguration *configuration,
    const uint8_t *tail,
    uint32_t tail_bytes,
    SparkByteDraftResult *result)
{
    SparkByteDraftBuffer body = {0},response = {0};
    const char *payload;
    size_t payload_bytes;
    SparkStatus status;
    if ( result == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(result,0,sizeof(*result));
    status = SparkByteDraftValidateConfiguration(configuration,tail,tail_bytes);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftBuildRequest(configuration,tail,tail_bytes,&body);
    if ( status == SPARK_STATUS_OK )
        status = SparkByteDraftExchange(configuration,&body,&response);
    if ( status == SPARK_STATUS_OK )
    {
        status = SparkByteDraftHttpBody(&response,&payload,&payload_bytes);
        if ( status != SPARK_STATUS_OK )
            fprintf(stderr,"BYTE-DRAFT-HTTP-INVALID host=%s port=%u\n",configuration->host,(unsigned)configuration->port);
        else
            status = SparkByteDraftParse(configuration,payload,payload_bytes,tail,tail_bytes,result);
    }
    free(body.data);
    free(response.data);
    if ( status != SPARK_STATUS_OK )
    {
        SparkByteDraftResultRelease(result);
        SPARK_RETURN(status);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkByteDraftDecode(const SparkTokenizer *tokenizer,const uint32_t *ids,uint32_t count,char **text_out,uint32_t *bytes_out)
{
    uint32_t capacity = 256u * count + 64u,bytes = 0u;
    char *text = (char *)malloc(capacity);
    SparkStatus status;
    *text_out = 0;
    *bytes_out = 0u;
    if ( text == 0 )
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    status = SparkTokenizerDecodeTokenIds(tokenizer,ids,count,0u,text,capacity,&bytes);
    if ( status != SPARK_STATUS_OK )
    {
        free(text);
        SPARK_RETURN(status);
    }
    *text_out = text;
    *bytes_out = bytes;
    return SPARK_STATUS_OK;
}

static uint32_t SparkByteDraftWindowStart(const SparkTokenizer *tokenizer,const uint32_t *committed_ids,uint32_t committed_count,uint32_t window_tokens)
{
    uint32_t window = committed_count < window_tokens ? committed_count : window_tokens,start = committed_count - window,index;
    for ( index = start; index + 1u < committed_count; index++ )
    {
        char piece[64];
        uint32_t bytes = 0u;
        if ( SparkTokenizerDecodeTokenIds(tokenizer,&committed_ids[index],1u,0u,piece,sizeof(piece),&bytes) == SPARK_STATUS_OK &&
             bytes != 0u && (piece[0] == ' ' || piece[0] == '\n' || piece[0] == '\t') )
            return index;
    }
    return start;
}

SparkStatus SparkByteDraftTail(
    const SparkTokenizer *tokenizer,
    const uint32_t *committed_ids,
    uint32_t committed_count,
    uint32_t window_tokens,
    uint8_t *tail,
    uint32_t tail_capacity,
    uint32_t *tail_bytes,
    uint32_t *tail_tokens)
{
    uint32_t start,bytes = 0u,keep;
    char *text = 0;
    SparkStatus status;
    if ( tokenizer == 0 || committed_ids == 0 || committed_count == 0u || window_tokens == 0u ||
         window_tokens > SPARK_BYTE_DRAFT_MAX_WINDOW_TOKENS || tail == 0 || tail_capacity == 0u ||
         tail_bytes == 0 || tail_tokens == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    start = committed_count - (committed_count < window_tokens ? committed_count : window_tokens);
    status = SparkByteDraftDecode(tokenizer,&committed_ids[start],committed_count - start,&text,&bytes);
    if ( status != SPARK_STATUS_OK )
        SPARK_RETURN(status);
    keep = bytes < tail_capacity ? bytes : tail_capacity;
    keep = keep < SPARK_BYTE_DRAFT_MAX_TAIL_BYTES ? keep : SPARK_BYTE_DRAFT_MAX_TAIL_BYTES;
    memcpy(tail,text + (bytes - keep),keep);
    free(text);
    *tail_bytes = keep;
    *tail_tokens = committed_count - start;
    return keep != 0u ? SPARK_STATUS_OK : SPARK_STATUS_NOT_FOUND;
}

static uint32_t SparkByteDraftCompleteUtf8(const uint8_t *bytes,uint32_t count)
{
    uint32_t start = count,need;
    while ( start > 0u && count - start < 4u && (bytes[start - 1u] & 0xc0u) == 0x80u )
        start--;
    if ( start == 0u )
        return count;
    need = (bytes[start - 1u] & 0x80u) == 0u ? 1u : (bytes[start - 1u] & 0xe0u) == 0xc0u ? 2u :
        (bytes[start - 1u] & 0xf0u) == 0xe0u ? 3u : (bytes[start - 1u] & 0xf8u) == 0xf0u ? 4u : 0u;
    if ( need == 0u )
        return count;
    return count - (start - 1u) == need ? count : start - 1u;
}

SparkStatus SparkByteDraftAlign(
    const SparkTokenizer *tokenizer,
    const uint32_t *committed_ids,
    uint32_t committed_count,
    uint32_t window_tokens,
    const uint8_t *continuation,
    uint32_t continuation_bytes,
    uint32_t max_tokens,
    uint32_t *draft_ids,
    uint32_t *draft_count)
{
    SparkTokenizerEncoding encoding;
    uint32_t start,window,text_bytes = 0u,usable,proposed,index;
    char *window_text = 0,*joined;
    uint32_t *ids;
    SparkStatus status;
    if ( tokenizer == 0 || committed_ids == 0 || committed_count == 0u || window_tokens == 0u ||
         window_tokens > SPARK_BYTE_DRAFT_MAX_WINDOW_TOKENS || continuation == 0 || continuation_bytes == 0u ||
         continuation_bytes > SPARK_BYTE_DRAFT_MAX_CONTINUATION_BYTES || max_tokens == 0u || draft_ids == 0 || draft_count == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *draft_count = 0u;
    usable = SparkByteDraftCompleteUtf8(continuation,continuation_bytes);
    if ( usable == 0u )
        return SPARK_STATUS_OK;
    start = SparkByteDraftWindowStart(tokenizer,committed_ids,committed_count,window_tokens);
    window = committed_count - start;
    status = SparkByteDraftDecode(tokenizer,&committed_ids[start],window,&window_text,&text_bytes);
    if ( status != SPARK_STATUS_OK )
        SPARK_RETURN(status);
    joined = (char *)malloc((size_t)text_bytes + usable);
    ids = (uint32_t *)malloc(((size_t)text_bytes + usable + 16u) * sizeof(uint32_t));
    if ( joined == 0 || ids == 0 )
    {
        free(window_text);
        free(joined);
        free(ids);
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    memcpy(joined,window_text,text_bytes);
    memcpy(joined + text_bytes,continuation,usable);
    free(window_text);
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = text_bytes + usable + 16u;
    encoding.token_ids = ids;
    status = SparkTokenizerEncodeUtf8(tokenizer,joined,text_bytes + usable,SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH,&encoding);
    free(joined);
    if ( status != SPARK_STATUS_OK )
    {
        free(ids);
        SPARK_RETURN(status);
    }
    if ( encoding.overflow_token_count != 0u || encoding.invalid_segment_count != 0u || encoding.token_count <= window + 1u ||
         memcmp(ids,&committed_ids[start],(size_t)window * sizeof(uint32_t)) != 0 )
    {
        free(ids);
        return SPARK_STATUS_OK;
    }
    proposed = encoding.token_count - 1u - window;
    proposed = proposed < max_tokens ? proposed : max_tokens;
    for ( index = 0u; index < proposed; index++ )
        draft_ids[index] = ids[window + index];
    *draft_count = proposed;
    free(ids);
    return SPARK_STATUS_OK;
}
