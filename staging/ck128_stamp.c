#include "sparkpipe/spark_ck128.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: ck128_stamp <file>\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (f == 0)
    {
        fprintf(stderr, "open failed\n");
        return 2;
    }
    SparkCk128Context c;
    SparkCk128Initialize(&c);
    static unsigned char buf[67108864];
    size_t n;
    while ((n = fread(buf, 1u, sizeof(buf), f)) != 0u)
    {
        SparkCk128Update(&c, buf, n);
    }
    unsigned char d[16];
    SparkCk128Finalize(&c, d);
    char hex[SPARK_CK128_HEX_BYTES];
    SparkCk128DigestToHex(d, hex);
    char out[4096];
    snprintf(out, sizeof(out), "%s.ck128", argv[1]);
    FILE *o = fopen(out, "w");
    if (o == 0)
    {
        fprintf(stderr, "sidecar write failed\n");
        return 2;
    }
    fprintf(o, "%s\n", hex);
    fclose(o);
    fclose(f);
    printf("ck128 %s -> %s\n", hex, out);
    return 0;
}
