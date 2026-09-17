#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define TEST_FAIL(msg) do { \
    fprintf(stderr, "FAIL: %s:%d %s\n", __FILE__, __LINE__, msg); \
    failures++; \
} while (0)

static int check_marker(const char *path, const char *marker)
{
    char cmd[1024];
    char line[256];
    FILE *pipe;
    int found = 0;

    snprintf(cmd, sizeof(cmd), "strings '%s' 2>/dev/null | grep -c '%s'",
        path, marker);
    pipe = popen(cmd, "r");
    if (pipe == 0)
        return -1;
    if (fgets(line, sizeof(line), pipe) != 0) {
        if (atoi(line) > 0)
            found = 1;
    }
    pclose(pipe);

    if (!found) {
        fprintf(stderr,
            "FAIL: marker '%s' NOT in %s — stale/private kernel copy compiled\n",
            marker, path);
        failures++;
    } else {
        fprintf(stderr, "PASS: marker '%s' present in %s\n", marker, path);
    }
    return found;
}

static int check_kernel_present(const char *path, const char *kernel)
{
    char cmd[1024];
    char line[512];
    FILE *pipe;
    int found = 0;

    snprintf(cmd, sizeof(cmd),
        "cuobjdump -sass '%s' 2>/dev/null | grep -c '%s' || strings '%s' 2>/dev/null | grep -c '%s'",
        path, kernel, path, kernel);
    pipe = popen(cmd, "r");
    if (pipe == 0)
        return -1;
    while (fgets(line, sizeof(line), pipe) != 0) {
        if (atoi(line) > 0)
            found = 1;
    }
    pclose(pipe);
    return found;
}

int main(int argc, char **argv)
{
    const char *path;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_driver.so or weightd binary>\n", argv[0]);
        return 2;
    }
    path = argv[1];

    fprintf(stderr, "=== kernel ABI checks on %s ===\n\n", path);

    fprintf(stderr, "1. Mesh kernel build marker:\n");
    check_marker(path, "SPARK-TP-MESH-KERNELS-V3-PARITY-TAIL-ABORT-DIAG");

    fprintf(stderr, "\n2. Graph capture marker:\n");
    check_marker(path, "GRAPH-CAPTURE-OK");

    fprintf(stderr, "\n3. Key kernel symbols:\n");
    {
        const char *kernels[] = {
            "SparkGlm5NextMeshWaitKernel",
            "SparkGlm5NextMeshPublishKernel",
            "SparkGlm5NextMeshCopyDownKernel",
            "SparkGlm5NextSumRanksF32Kernel",
        };
        uint32_t i;
        for (i = 0; i < 4; i++) {
            int found = check_kernel_present(path, kernels[i]);
            if (found)
                fprintf(stderr, "PASS: %s found\n", kernels[i]);
            else {
                fprintf(stderr, "FAIL: %s NOT found\n", kernels[i]);
                failures++;
            }
        }
    }

    fprintf(stderr, "\n%s: %d failures\n", argv[0], failures);
    return failures ? 1 : 0;
}
