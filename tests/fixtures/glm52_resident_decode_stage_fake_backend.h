#ifndef SPARKPIPE_TEST_GLM52_RESIDENT_DECODE_STAGE_FAKE_BACKEND_H
#define SPARKPIPE_TEST_GLM52_RESIDENT_DECODE_STAGE_FAKE_BACKEND_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct SparkGlm52ResidentDecodeStageFakeStream
{
    bool defer_completion;
    uint32_t submit_count;
    uint32_t last_pipeline_slot;
    uint32_t last_active_sequence_count;
    void (*pending_completion_function)(void *completion_context);
    void *pending_completion_context;
} SparkGlm52ResidentDecodeStageFakeStream;

static inline void SparkGlm52ResidentDecodeStageFakeStreamInitialize(
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream)
{
    assert(fake_stream != 0);
    memset(fake_stream, 0, sizeof(*fake_stream));
}

static inline void SparkGlm52ResidentDecodeStageFakeStreamSetDeferred(
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream,
    bool defer_completion)
{
    assert(fake_stream != 0);
    fake_stream->defer_completion = defer_completion;
}

static inline bool SparkGlm52ResidentDecodeStageFakeStreamHasPending(
    const SparkGlm52ResidentDecodeStageFakeStream *fake_stream)
{
    assert(fake_stream != 0);
    return fake_stream->pending_completion_function != 0;
}

static inline void SparkGlm52ResidentDecodeStageFakeStreamComplete(
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream)
{
    void (*completion_function)(void *completion_context);
    void *completion_context;

    assert(fake_stream != 0);
    assert(fake_stream->pending_completion_function != 0);
    completion_function = fake_stream->pending_completion_function;
    completion_context = fake_stream->pending_completion_context;
    fake_stream->pending_completion_function = 0;
    fake_stream->pending_completion_context = 0;
    completion_function(completion_context);
}

#endif
