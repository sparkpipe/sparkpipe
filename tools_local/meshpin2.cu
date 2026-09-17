#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <ctime>

#define SLOT_BYTES (16ull * 1024 * 1024)
#define SLOTS 32
#define BANDS 4
#define REGION (SLOT_BYTES * SLOTS * BANDS + 4096)

static double now_s()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    size_t bytes = (size_t)atoll(argv[1]) * 1024;
    (void)argc;
    DIR *dir = opendir("/proc");
    struct dirent *e;
    long wpid = -1;
    while ((e = readdir(dir)) != 0)
    {
        long pid = atol(e->d_name);
        if (pid <= 0) continue;
        char p[256], ex[256];
        snprintf(p, sizeof(p), "/proc/%ld/exe", pid);
        ssize_t n = readlink(p, ex, sizeof(ex) - 1);
        if (n <= 0) continue;
        ex[n] = 0;
        if (strstr(ex, "sparkpipe_weightd")) { wpid = pid; break; }
    }
    closedir(dir);
    if (wpid < 0) { printf("no-weightd\n"); return 2; }
    char path[256];
    int fd = -1;
    snprintf(path, sizeof(path), "/proc/%ld/fd", wpid);
    dir = opendir(path);
    while ((e = readdir(dir)) != 0)
    {
        char lk[256], p2[300];
        snprintf(p2, sizeof(p2), "/proc/%ld/fd/%s", wpid, e->d_name);
        ssize_t n = readlink(p2, lk, sizeof(lk) - 1);
        if (n <= 0) continue;
        lk[n] = 0;
        if (strstr(lk, "spark-mesh")) { fd = open(p2, O_RDWR); break; }
    }
    closedir(dir);
    if (fd < 0) { printf("no-fd\n"); return 3; }
    uint8_t *m = (uint8_t *)mmap(0, REGION, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { printf("mmap-fail\n"); return 4; }
    uint64_t band1 = 32ull * SLOT_BYTES;
    double t0 = now_s();
    cudaError_t r = cudaHostRegister(m + band1, bytes, 0u);
    double dt1 = now_s() - t0;
    printf("band %zu KB rc=%d in %.3fs\n", bytes >> 10, (int)r, dt1);
    fflush(stdout);
    double t2 = now_s();
    cudaError_t r2 = cudaHostRegister(m + SLOTS * SLOT_BYTES * BANDS, 4096, 0u);
    double dt2 = now_s() - t2;
    printf("doorbell rc=%d in %.3fs\n", (int)r2, dt2);
    return 0;
}
