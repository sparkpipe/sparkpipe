#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tokenizer.h"

static int SparkSplitReadAll(char **text_out, size_t *length_out)
{
    size_t capacity = 1u << 20;
    size_t length = 0u;
    size_t chunk;
    char *text = (char *)malloc(capacity);
    if (text == 0)
    {
        return 0;
    }
    while ((chunk = fread(text + length, 1u, capacity - length, stdin)) > 0u)
    {
        length += chunk;
        if (length == capacity)
        {
            char *grown = (char *)realloc(text, capacity + capacity);
            if (grown == 0)
            {
                free(text);
                return 0;
            }
            text = grown;
            capacity += capacity;
        }
    }
    *text_out = text;
    *length_out = length;
    return 1;
}

int main(int argc, char **argv)
{
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    char *text = 0;
    size_t length = 0u;
    size_t record = 0u;
    uint32_t *ends;
    int encode = argc == 3 && strcmp(argv[2], "--encode") == 0;
    if (argc != 2 && !encode)
    {
        fprintf(stderr, "usage: %s TOKENIZER_JSON [--encode] < NUL-separated records\n", argv[0]);
        return 2;
    }
    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes = SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = argv[1];
    if (SparkTokenizerLoadHuggingFaceJson(&tokenizer, &configuration) != SPARK_STATUS_OK || !SparkSplitReadAll(&text, &length))
    {
        fprintf(stderr, "sparkpipe_split: cannot load %s or read input\n", argv[1]);
        return 1;
    }
    ends = (uint32_t *)malloc((length + 1u) * sizeof(uint32_t));
    if (ends == 0 || length > UINT32_MAX)
    {
        return 1;
    }
    while (record < length)
    {
        size_t stop = record;
        uint32_t count = 0u;
        SparkStatus status;
        while (stop < length && text[stop] != '\0')
        {
            stop += 1u;
        }
        if (encode)
        {
            SparkTokenizerEncoding encoding;
            SparkTokenizerEncodingReset(&encoding);
            encoding.token_capacity = (uint32_t)length + 1u;
            encoding.token_ids = ends;
            status = SparkTokenizerEncodeUtf8(&tokenizer, text + record, (uint32_t)(stop - record), 0u, &encoding);
            count = encoding.token_count;
        }
        else
        {
            status = SparkTokenizerSplitUtf8(&tokenizer, text + record, (uint32_t)(stop - record), ends, (uint32_t)length + 1u, &count);
        }
        if (status != SPARK_STATUS_OK)
        {
            printf("error %d\n", (int)status);
        }
        else
        {
            for (uint32_t index = 0u; index < count; ++index)
            {
                printf(index == 0u ? "%u" : " %u", ends[index]);
            }
            printf("\n");
        }
        record = stop + 1u;
    }
    free(ends);
    free(text);
    SparkTokenizerDestroy(&tokenizer);
    return 0;
}
