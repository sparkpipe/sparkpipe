#include <stdio.h>
#include <stdlib.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd_manifest.h"

int main(int argc, char **argv)
{
    SparkWeightdManifest m;
    const SparkWeightdRangeGroup *g;
    uint32_t i, layer, expert;
    unsigned long long pack_bytes;
    if (argc != 5)
    {
        fprintf(stderr, "usage: %s <manifest> <pack-bytes> <layer> <expert>\n",
            argv[0]);
        return 2;
    }
    pack_bytes = strtoull(argv[2], 0, 10);
    layer = (uint32_t)strtoul(argv[3], 0, 10);
    expert = (uint32_t)strtoul(argv[4], 0, 10);
    if (SparkWeightdManifestLoad(argv[1], pack_bytes, &m) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "LOAD-FAIL\n");
        return 1;
    }
    printf("ranges=%u groups=%u spine_count=%u spine_bytes=%llu\n",
        m.range_count, m.group_count, m.spine_count,
        (unsigned long long)m.spine_bytes);
    g = SparkWeightdManifestFind(&m, layer, expert);
    if (g == 0)
    {
        fprintf(stderr, "FIND-FAIL l%u e%u\n", layer, expert);
        SparkWeightdManifestDestroy(&m);
        return 1;
    }
    printf("group l%u e%u: first_range=%u count=%u\n", layer, expert,
        g->first_range, g->range_count);
    for (i = 0u; i < g->range_count; i++)
        printf("  range kind=%u off=%llu bytes=%llu\n",
            m.ranges[g->first_range + i].kind,
            (unsigned long long)m.ranges[g->first_range + i].offset,
            (unsigned long long)m.ranges[g->first_range + i].bytes);
    SparkWeightdManifestDestroy(&m);
    printf("MANIFEST-LOAD-PASS\n");
    return 0;
}
