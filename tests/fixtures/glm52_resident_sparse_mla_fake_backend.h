#ifndef SPARKPIPE_TEST_GLM52_RESIDENT_SPARSE_MLA_FAKE_BACKEND_H
#define SPARKPIPE_TEST_GLM52_RESIDENT_SPARSE_MLA_FAKE_BACKEND_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct SparkGlm52ResidentSparseMlaFakeStream
{
    bool defer_completion;
    uint32_t submit_count;
    uint32_t last_pipeline_slot;
    uint32_t last_active_sequence_count;
    void (*pending_completion_function)(void *completion_context);
    void *pending_completion_context;
} SparkGlm52ResidentSparseMlaFakeStream;

static inline void SparkGlm52ResidentSparseMlaFakeStreamInitialize(
    SparkGlm52ResidentSparseMlaFakeStream *fake_stream)
{
    assert(fake_stream != 0);
    memset(fake_stream, 0, sizeof(*fake_stream));
}

static inline void SparkGlm52ResidentSparseMlaFakeStreamSetDeferred(
    SparkGlm52ResidentSparseMlaFakeStream *fake_stream,
    bool defer_completion)
{
    assert(fake_stream != 0);
    fake_stream->defer_completion = defer_completion;
}

static inline bool SparkGlm52ResidentSparseMlaFakeStreamHasPending(
    const SparkGlm52ResidentSparseMlaFakeStream *fake_stream)
{
    assert(fake_stream != 0);
    return fake_stream->pending_completion_function != 0;
}

static inline void SparkGlm52ResidentSparseMlaFakeStreamComplete(
    SparkGlm52ResidentSparseMlaFakeStream *fake_stream)
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
