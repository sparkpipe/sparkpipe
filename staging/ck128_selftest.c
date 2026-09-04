#include "sparkpipe/spark_ck128.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: ck128_selftest <file> <chunk-bytes>\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (f == 0)
    {
        fprintf(stderr, "open failed\n");
        return 2;
    }
    size_t chunk = (size_t)strtoul(argv[2], 0, 10);
    uint8_t *buffer = (uint8_t *)malloc(chunk);
    SparkCk128Context context;
    SparkCk128Initialize(&context);
    for (;;)
    {
        size_t n = fread(buffer, 1u, chunk, f);
        if (n != 0u)
        {
            SparkCk128Update(&context, buffer, n);
        }
        if (n < chunk)
        {
            break;
        }
    }
    uint8_t digest[16];
    SparkCk128Finalize(&context, digest);
    char hex[SPARK_CK128_HEX_BYTES];
    SparkCk128DigestToHex(digest, hex);
    printf("%s\n", hex);
    fclose(f);
    free(buffer);
    return 0;
}
