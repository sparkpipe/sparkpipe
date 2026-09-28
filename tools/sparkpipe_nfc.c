#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sparkpipe/spark_tokenizer.h"

int main(void)
{
    size_t capacity = 1u << 20;
    size_t length = 0u;
    char *text = (char *)malloc(capacity);
    char *normalized = 0;
    uint32_t normalized_bytes = 0u;
    size_t chunk;
    if (text == 0)
    {
        return 1;
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
                return 1;
            }
            text = grown;
            capacity += capacity;
        }
    }
    if (length > UINT32_MAX || SparkTokenizerNormalizeNfcUtf8(text, (uint32_t)length, &normalized, &normalized_bytes) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "sparkpipe_nfc: input is not valid UTF-8\n");
        free(text);
        return 2;
    }
    fwrite(normalized, 1u, normalized_bytes, stdout);
    free(normalized);
    free(text);
    return 0;
}
