/* mesh-register EINVAL reproduction (lane 0, glm5_next M3 blocker).
 *
 * a7 failed: cudaHostRegister(mesh memfd mmap, 134283264, PORTABLE|MAPPED)
 * -> cudaErrorInvalidValue, inside SparkTpDeviceCollectivePrepareReceiveBf16
 * (ring/transport/tp_device_collective.c:2061) against the shared release
 * weightd's exported mesh memfd. The PR #1082 campaign passed the same call
 * against a private daemon, so the trigger is environmental. This probe
 * isolates the registration itself: mmap a memfd (the same size/alignment
 * discipline the client uses) and try every alignment/flag combination.
 *
 * build: cc -O2 -I include tools/mesh_register_repro.c -o repro \
 *            -L/usr/local/cuda/lib64 -lcudart
 */
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define REGION_BYTES 134283264ull     /* SPARK_WEIGHTD_MESH_REGION_BYTES */

int memfd = -1;

static int attempt(const char *label, size_t alignment, unsigned flags)
{
    size_t reserved = (size_t)REGION_BYTES + alignment;
    uint8_t *raw = mmap(0, reserved, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED)
    {
        printf("%-28s reserve failed errno=%d\n", label, errno);
        return 1;
    }
    uintptr_t aligned = ((uintptr_t)raw + alignment - 1u) & ~(uintptr_t)(alignment - 1u);
    if (munmap(raw, reserved) != 0)
    {
        printf("%-28s unmap failed errno=%d\n", label, errno);
        return 1;
    }
    void *mapped = mmap((void *)aligned, (size_t)REGION_BYTES,
                        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memfd, 0);
    if (mapped == MAP_FAILED)
    {
        printf("%-28s map failed errno=%d\n", label, errno);
        return 1;
    }
    memset(mapped, 0xa5, 4096);
    cudaError_t err = cudaHostRegister(mapped, (size_t)REGION_BYTES, flags);
    printf("%-28s ptr=%p aligned=%zu flags=%u -> %s (%d)\n",
           label, mapped, alignment, flags, cudaGetErrorString(err), (int)err);
    if (err == cudaSuccess)
    {
        cudaHostUnregister(mapped);
        munmap(mapped, (size_t)REGION_BYTES);
        return 0;
    }
    return 1;
}

int main(void)
{
    int failures = 0;
    memfd = memfd_create("spark-mesh-repro", 0u);
    if (memfd < 0 || ftruncate(memfd, REGION_BYTES) != 0)
    {
        printf("memfd setup failed errno=%d\n", errno);
        return 2;
    }
    printf("cuda runtime: %s\n", cudaGetErrorString(cudaFree(0)));
    failures += attempt("64KiB PORTABLE|MAPPED", 64u * 1024u,
                        cudaHostRegisterPortable | cudaHostRegisterMapped);
    failures += attempt("2MiB PORTABLE|MAPPED", 2u * 1024u * 1024u,
                        cudaHostRegisterPortable | cudaHostRegisterMapped);
    failures += attempt("64KiB PORTABLE", 64u * 1024u, cudaHostRegisterPortable);
    failures += attempt("2MiB PORTABLE", 2u * 1024u * 1024u, cudaHostRegisterPortable);
    failures += attempt("64KiB default", 64u * 1024u, 0u);
    failures += attempt("2MiB default", 2u * 1024u * 1024u, 0u);
    printf("failures=%d\n", failures);
    return failures != 0;
}
