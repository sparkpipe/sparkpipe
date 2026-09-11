// hy4 lane: verify the placed FP8 rank pack against its .experts sidecar
// (WEPX v1, 40-byte records: reserved u32, chunk ordinal u32, offset u64,
// bytes u64, ck128 16B). Recomputes SparkCk128 over every chunk range and
// prints one line per mismatch plus a final verdict. Exit 0 only when all
// chunks match.
//
// Usage: experts_ck128_check <pack.safetensors> <pack.safetensors.experts>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "sparkpipe/spark_ck128.h"

struct Rec {
    uint32_t ordinal;
    uint64_t offset;
    uint64_t bytes;
    unsigned char ck[16];
};

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: experts_ck128_check <pack> <experts>\n");
        return 2;
    }
    FILE* side = fopen(argv[2], "rb");
    if (!side)
    {
        fprintf(stderr, "NO SIDECAR\n");
        return 2;
    }
    unsigned char head[16];
    if (fread(head, 1, 16, side) != 16 || memcmp(head, "WEPX", 4) != 0)
    {
        fprintf(stderr, "BAD SIDECAR MAGIC\n");
        return 2;
    }
    uint64_t count = 0;
    memcpy(&count, head + 8, 8);
    std::vector<Rec> recs(count);
    for (uint64_t i = 0; i < count; ++i)
    {
        unsigned char raw[40];
        if (fread(raw, 1, 40, side) != 40)
        {
            fprintf(stderr, "SIDECAR TRUNCATED at %llu\n",
                (unsigned long long)i);
            return 2;
        }
        uint32_t reserved = 0;
        memcpy(&reserved, raw, 4);
        memcpy(&recs[i].ordinal, raw + 4, 4);
        memcpy(&recs[i].offset, raw + 8, 8);
        memcpy(&recs[i].bytes, raw + 16, 8);
        memcpy(recs[i].ck, raw + 24, 16);
        if (reserved != 0 || recs[i].ordinal != (uint32_t)i)
        {
            fprintf(stderr, "BAD RECORD %llu\n", (unsigned long long)i);
            return 2;
        }
    }
    fclose(side);
    FILE* f = fopen(argv[1], "rb");
    if (!f)
    {
        fprintf(stderr, "NO PACK\n");
        return 2;
    }
    std::vector<unsigned char> buf;
    size_t worst = 0;
    for (const Rec& r : recs)
        if (r.bytes > worst)
            worst = (size_t)r.bytes;
    buf.resize(worst);
    uint64_t bad = 0;
    char hex[SPARK_CK128_HEX_BYTES];
    for (const Rec& r : recs)
    {
        if (fseeko(f, (off_t)r.offset, SEEK_SET) != 0 ||
            fread(buf.data(), 1, (size_t)r.bytes, f) != (size_t)r.bytes)
        {
            fprintf(stderr, "READ FAIL ordinal %u\n", r.ordinal);
            return 2;
        }
        SparkCk128Context c;
        SparkCk128Initialize(&c);
        SparkCk128Update(&c, buf.data(), (size_t)r.bytes);
        unsigned char d[16];
        SparkCk128Finalize(&c, d);
        SparkCk128DigestToHex(d, hex);
        if (memcmp(d, r.ck, 16) != 0)
        {
            char want[SPARK_CK128_HEX_BYTES];
            for (int i = 0; i < 16; ++i)
                snprintf(want + 2 * i, 3, "%02x", r.ck[i]);
            printf("CK128 MISMATCH ordinal=%u off=%llu bytes=%llu got=%s want=%s\n",
                r.ordinal, (unsigned long long)r.offset,
                (unsigned long long)r.bytes, hex, want);
            ++bad;
        }
    }
    fclose(f);
    printf("CK128_CHECK %s chunks=%llu mismatches=%llu\n",
        bad == 0 ? "GREEN" : "RED", (unsigned long long)count,
        (unsigned long long)bad);
    return bad == 0 ? 0 : 1;
}
