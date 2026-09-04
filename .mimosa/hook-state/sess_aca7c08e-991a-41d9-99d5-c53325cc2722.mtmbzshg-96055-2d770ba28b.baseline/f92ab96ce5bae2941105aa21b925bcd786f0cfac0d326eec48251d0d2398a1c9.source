#ifndef SPARKPIPE_HARDWARE_PROBE_COMMON_H
#define SPARKPIPE_HARDWARE_PROBE_COMMON_H

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkProbeLatencySummary
{
    uint64_t minimum_ns;
    uint64_t p50_ns;
    uint64_t p95_ns;
    uint64_t p99_ns;
    uint64_t maximum_ns;
} SparkProbeLatencySummary;

static inline uint64_t SparkProbeMix64(uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static inline uint64_t SparkProbePatternWord(uint64_t absolute_word_index)
{
    return SparkProbeMix64(absolute_word_index ^ UINT64_C(0x535041524b4e564d));
}

static inline uint64_t SparkProbeFingerprintWords(
    const uint64_t *words,
    uint64_t word_count,
    uint64_t absolute_word_offset)
{
    uint64_t fingerprint;
    uint64_t word_index;

    if (words == NULL && word_count != 0u)
    {
        return 0u;
    }
    fingerprint = SparkProbeMix64(word_count ^ absolute_word_offset);
    for (word_index = 0u; word_index < word_count; ++word_index)
    {
        fingerprint ^= SparkProbeMix64(
            words[word_index] ^ (absolute_word_offset + word_index));
    }
    return fingerprint;
}

static inline uint64_t SparkProbeMonotonicNanoseconds(void)
{
    struct timespec value;

    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
}

static inline int SparkProbeParseU64(
    const char *text,
    uint64_t minimum,
    uint64_t maximum,
    uint64_t *value_out)
{
    char *end;
    unsigned long long value;

    if (text == NULL || text[0] == '\0' || value_out == NULL || minimum > maximum)
    {
        return 0;
    }
    errno = 0;
    end = NULL;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < minimum || value > maximum)
    {
        return 0;
    }
    *value_out = (uint64_t)value;
    return 1;
}

static inline int SparkProbeParseU32(
    const char *text,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t *value_out)
{
    uint64_t value;

    if (value_out == NULL ||
        !SparkProbeParseU64(text, minimum, maximum, &value))
    {
        return 0;
    }
    *value_out = (uint32_t)value;
    return 1;
}

static inline int SparkProbeHexSha256IsValid(const char *text)
{
    size_t index;

    if (text == NULL || strlen(text) != 64u)
    {
        return 0;
    }
    for (index = 0u; index < 64u; ++index)
    {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f')))
        {
            return 0;
        }
    }
    return 1;
}

static inline int SparkProbeIdentifierIsValid(const char *text)
{
    size_t index;
    size_t length;

    if (text == NULL)
    {
        return 0;
    }
    length = strlen(text);
    if (length == 0u || length > 255u)
    {
        return 0;
    }
    for (index = 0u; index < length; ++index)
    {
        unsigned char character;

        character = (unsigned char)text[index];
        if (!((character >= 'a' && character <= 'z') ||
              (character >= 'A' && character <= 'Z') ||
              (character >= '0' && character <= '9') ||
              character == '.' || character == '_' || character == '-' ||
              character == ':' || character == '/'))
        {
            return 0;
        }
    }
    return 1;
}

static inline int SparkProbeCompareU64(const void *left, const void *right)
{
    uint64_t left_value;
    uint64_t right_value;

    left_value = *(const uint64_t *)left;
    right_value = *(const uint64_t *)right;
    return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
}

static inline uint64_t SparkProbePercentileIndex(size_t count, uint32_t percentile)
{
    uint64_t numerator;

    if (count == 0u)
    {
        return 0u;
    }
    numerator = (uint64_t)(count - 1u) * percentile;
    return (numerator + 99u) / 100u;
}

static inline SparkProbeLatencySummary SparkProbeSummarizeLatency(
    uint64_t *samples,
    size_t sample_count)
{
    SparkProbeLatencySummary summary;

    memset(&summary, 0, sizeof(summary));
    if (samples == NULL || sample_count == 0u)
    {
        return summary;
    }
    qsort(samples, sample_count, sizeof(*samples), SparkProbeCompareU64);
    summary.minimum_ns = samples[0];
    summary.p50_ns = samples[SparkProbePercentileIndex(sample_count, 50u)];
    summary.p95_ns = samples[SparkProbePercentileIndex(sample_count, 95u)];
    summary.p99_ns = samples[SparkProbePercentileIndex(sample_count, 99u)];
    summary.maximum_ns = samples[sample_count - 1u];
    return summary;
}

static inline void SparkProbeWriteJsonString(FILE *output, const char *text)
{
    const unsigned char *cursor;

    if (output == NULL)
    {
        return;
    }
    if (text == NULL)
    {
        fputs("null", output);
        return;
    }
    fputc('"', output);
    cursor = (const unsigned char *)text;
    while (*cursor != '\0')
    {
        unsigned char character;

        character = *cursor++;
        if (character == '"' || character == '\\')
        {
            fputc('\\', output);
            fputc((int)character, output);
        }
        else if (character == '\b')
        {
            fputs("\\b", output);
        }
        else if (character == '\f')
        {
            fputs("\\f", output);
        }
        else if (character == '\n')
        {
            fputs("\\n", output);
        }
        else if (character == '\r')
        {
            fputs("\\r", output);
        }
        else if (character == '\t')
        {
            fputs("\\t", output);
        }
        else if (character < 0x20u)
        {
            fprintf(output, "\\u%04x", (unsigned int)character);
        }
        else
        {
            fputc((int)character, output);
        }
    }
    fputc('"', output);
}

#ifdef __cplusplus
}
#endif

#endif
