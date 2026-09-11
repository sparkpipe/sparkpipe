#include <cuda_runtime.h>
#include <cstdio>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define POC_BAND 1u
#define POC_ROUNDS 20u
#define POC_HIDDEN 4096u
#define SLOT_BYTES (16u * 1024u * 1024u)
#define RANKS_PER_BAND 16u
#define SLOTS_PER_BAND 32u
#define BANDS 4u
#define DOORBELL_BYTES 4096u
#define BUFFER_BYTES (SLOT_BYTES * SLOTS_PER_BAND * BANDS)
#define REGION_BYTES (BUFFER_BYTES + DOORBELL_BYTES)
#define DOORBELL_OFFSET BUFFER_BYTES

static uint32_t rank_of_host()
{
    char host[64];
    gethostname(host, sizeof(host));
    char tail = host[strlen(host) - 1];
    if (tail >= '0' && tail <= '9') return (uint32_t)(tail - '0');
    if (tail >= 'a' && tail <= 'f') return (uint32_t)(tail - 'a' + 10);
    return 999u;
}

static double now_s()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void *attach_mesh(uint32_t *rank_out)
{
    DIR *dir;
    struct dirent *entry;
    char path[256];
    long resident_pid = -1;
    dir = opendir("/proc");
    if (dir == 0) return 0;
    while ((entry = readdir(dir)) != 0)
    {
        long pid = atol(entry->d_name);
        if (pid <= 0) continue;
        char exe[256];
        snprintf(path, sizeof(path), "/proc/%ld/exe", pid);
        ssize_t n = readlink(path, exe, sizeof(exe) - 1);
        if (n <= 0) continue;
        exe[n] = 0;
        if (strstr(exe, "sparkpipe_weightd") != 0)
        {
            resident_pid = pid;
            break;
        }
    }
    closedir(dir);
    if (resident_pid < 0) { fprintf(stderr, "poc: no weightd\n"); return 0; }
    fprintf(stderr, "poc: weightd pid %ld\n", resident_pid);
    snprintf(path, sizeof(path), "/proc/%ld/fd", resident_pid);
    dir = opendir(path);
    if (dir == 0) return 0;
    int fd = -1;
    while ((entry = readdir(dir)) != 0)
    {
        char link[256];
        snprintf(path, sizeof(path), "/proc/%ld/fd/%s", resident_pid, entry->d_name);
        ssize_t n = readlink(path, link, sizeof(link) - 1);
        if (n <= 0) continue;
        link[n] = 0;
        if (strstr(link, "spark-mesh") != 0)
        {
            fd = open(path, O_RDWR);
            break;
        }
    }
    closedir(dir);
    if (fd < 0) { fprintf(stderr, "poc: no mesh fd\n"); return 0; }
    fprintf(stderr, "poc: mesh fd ok\n");
    void *mapping = mmap(0, REGION_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return 0;
    uint64_t band_off = (uint64_t)POC_BAND * SLOTS_PER_BAND * SLOT_BYTES;
    fprintf(stderr, "poc: registering band+doorbell\n");
    cudaError_t r1 = cudaHostRegister((uint8_t *)mapping + band_off,
        SLOTS_PER_BAND * SLOT_BYTES, 0u);
    cudaError_t r2 = cudaHostRegister((uint8_t *)mapping + DOORBELL_OFFSET,
        DOORBELL_BYTES, 0u);
    if (r1 != cudaSuccess || r2 != cudaSuccess)
    {
        fprintf(stderr, "poc rank %u: register band rc=%d doorbell rc=%d\n",
            rank_of_host(), (int)r1, (int)r2);
        munmap(mapping, REGION_BYTES);
        return 0;
    }
    (void)rank_out;
    return mapping;
}

__global__ void poc_compute(float *buffer, uint32_t iterations)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (float)(i + 1);
    for (uint32_t k = 0; k < iterations; k++)
        v = fmaf(v, 1.0000001f, 0.0000001f);
    if (i < POC_HIDDEN) buffer[i % POC_HIDDEN] = v;
}

__global__ void poc_fill_partial(uint16_t *slot_bf16, uint32_t rank)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POC_HIDDEN) return;
    float v = (float)(rank * 1000u + i);
    uint32_t bits = __float_as_uint(v);
    slot_bf16[i] = (uint16_t)(bits >> 16);
}

__device__ __forceinline__ unsigned long long poc_globaltimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void poc_publish(uint64_t *entry, uint64_t *seq_cell,
    uint64_t *round_seq, uint64_t bytes, uint64_t slot_index,
    unsigned long long *stamps, uint32_t round_index)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long seq = 1ull + atomicAdd((unsigned long long *)seq_cell, 1ull);
    round_seq[0] = seq;
    stamps[round_index * 2u] = poc_globaltimer();
    __threadfence_system();
    volatile uint64_t *e = (volatile uint64_t *)entry;
    e[2] = slot_index;
    e[1] = bytes;
    __threadfence_system();
    e[0] = seq;
}

__device__ __forceinline__ uint64_t poc_load_cv(const volatile uint64_t *address)
{
    return __ldcv((const uint64_t *)address);
}

__global__ void poc_wait(volatile uint64_t *band_base, uint64_t *round_seq,
    uint32_t rank, uint64_t parity, unsigned long long *stamps,
    uint32_t round_index)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t seq = round_seq[0];
    for (uint32_t peer = 0; peer < RANKS_PER_BAND; peer++)
    {
        if (peer == rank) continue;
        const uint64_t *end_word =
            (const uint64_t *)((uint8_t *)band_base +
            ((uint64_t)peer * 2u + parity) * SLOT_BYTES + SLOT_BYTES - 8u);
        while (poc_load_cv(end_word) < seq)
        {
            __nanosleep(200);
        }
    }
    stamps[round_index * 2u + 1u] = poc_globaltimer();
}

__global__ void poc_combine(uint16_t *destination, const uint16_t *source)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POC_HIDDEN) return;
    float d = __uint_as_float(((uint32_t)destination[i]) << 16);
    float s = __uint_as_float(((uint32_t)source[i]) << 16);
    float r = d + s;
    uint32_t bits = __uint_as_float(r);
    destination[i] = (uint16_t)(bits >> 16);
}

__global__ void poc_reset(uint16_t *destination, uint32_t rank)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POC_HIDDEN) return;
    float v = (float)(rank * 1000u + i);
    uint32_t bits = __float_as_uint(v);
    destination[i] = (uint16_t)(bits >> 16);
}

static unsigned long long *g_stamps;
static volatile uint32_t g_alive;

static void *poc_watchdog(void *)
{
    for (;;)
    {
        struct timespec pause = {1, 0};
        nanosleep(&pause,0);
        if ( g_alive == 0u ) continue;
        if ( g_alive > 12u ) break;
        uint32_t done = 0;
        for (uint32_t q = 0; q < 64u; q++)
            if (g_stamps[q] != 0ull) done++;
        fprintf(stderr,"poc WATCHDOG tick=%u stamps-written=%u/64 last=%llu\n",
            g_alive, done, g_stamps[done != 0u ? done - 1u : 0u]);
        fflush(stderr);
    }
    return 0;
}

int main(int argc, char **argv)
{
    pthread_t watchdog_thread;
    pthread_create(&watchdog_thread,0,poc_watchdog,0);
    uint32_t mode = argc > 1 ? (uint32_t)atoi(argv[1]) : 2u;
    uint32_t tokens = argc > 2 ? (uint32_t)atoi(argv[2]) : 8u;
    uint32_t compute_iters = argc > 3 ? (uint32_t)atoi(argv[3]) : 60000u;
    uint32_t rank = rank_of_host();
    if (rank >= RANKS_PER_BAND)
    {
        fprintf(stderr, "poc: unknown rank\n");
        return 1;
    }
    uint8_t *mesh = (uint8_t *)attach_mesh(&rank);
    if (mesh == 0)
    {
        fprintf(stderr, "poc rank %u: mesh attach failed\n", rank);
        return 1;
    }
    uint64_t band_base_offset = (uint64_t)POC_BAND * SLOTS_PER_BAND * SLOT_BYTES;
    uint8_t *band = mesh + band_base_offset;
    uint64_t entry_offset = DOORBELL_OFFSET + ((uint64_t)POC_BAND * RANKS_PER_BAND + rank) * 24u;
    uint64_t *entry = (uint64_t *)(mesh + entry_offset);

    uint16_t *partial;
    float *work;
    uint64_t *seq_cell;
    uint64_t *round_seq;
    uint16_t *result_host;
    fprintf(stderr, "poc: init mallocs\n");
    cudaMalloc(&partial, POC_HIDDEN * 2u);
    cudaMalloc(&work, 4u * POC_HIDDEN * sizeof(float));
    cudaMalloc(&seq_cell, sizeof(uint64_t));
    cudaMalloc(&round_seq, sizeof(uint64_t));
    fprintf(stderr, "poc: mallochost\n");
    cudaMallocHost(&result_host, POC_HIDDEN * 2u);
    unsigned long long *stamps;
    unsigned long long *stamps_host;
    cudaMallocHost(&stamps, POC_ROUNDS * 4u * sizeof(unsigned long long));
    stamps_host = stamps;
    uint64_t seq_base = argc > 4 ? (uint64_t)strtoull(argv[4], 0, 10) : 1ull;
    cudaMemcpy(seq_cell, &seq_base, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    fprintf(stderr, "poc: stream ok\n");

    double t0 = now_s();
    fprintf(stderr, "poc: calibration launch\n");
    poc_compute<<<1024, 256, 0, stream>>>(work, compute_iters);
    cudaStreamSynchronize(stream);
    fprintf(stderr, "poc: calibration synced\n");
    double compute_ms = (now_s() - t0) * 1000.0;
    if (rank == 0)
        fprintf(stderr, "poc: compute kernel %.3f ms per call target\n", compute_ms);

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaGraphExec_t exec_alt;
    uint8_t *band_alt = mesh + 2u * 32 * SLOT_BYTES;
    uint64_t entry_alt_offset = DOORBELL_OFFSET + ((uint64_t)2u * RANKS_PER_BAND + rank) * 24u;
    uint64_t *entry_alt = (uint64_t *)(mesh + entry_alt_offset);
    if (mode == 2u || mode == 3u)
    {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }
    for (uint32_t r = 0; r < POC_ROUNDS; r++)
    {
        uint64_t parity = r & 1u;
        uint16_t *slot = (uint16_t *)(band + ((uint64_t)rank * 2u + parity) * SLOT_BYTES);
        poc_reset<<<POC_HIDDEN / 256u, 256u, 0, stream>>>(partial, rank);
        poc_compute<<<1024, 256, 0, stream>>>(work, compute_iters);
        cudaMemcpyAsync(slot, partial, POC_HIDDEN * 2u, cudaMemcpyDeviceToHost, stream);
        poc_publish<<<1, 32, 0, stream>>>(entry, seq_cell, round_seq,
            (uint64_t)POC_HIDDEN * 2u, (uint64_t)rank * 2u + parity,
            stamps, r);
        poc_wait<<<1, 32, 0, stream>>>((volatile uint64_t *)band, round_seq,
            rank, parity, stamps, r);
        for (uint32_t peer = 0; peer < RANKS_PER_BAND; peer++)
        {
            if (peer == rank) continue;
            uint16_t *peer_slot = (uint16_t *)(band + ((uint64_t)peer * 2u + parity) * SLOT_BYTES);
            poc_combine<<<POC_HIDDEN / 256u, 256u, 0, stream>>>(partial, peer_slot);
        }
    }
    cudaMemcpyAsync(result_host, partial, POC_HIDDEN * 2u, cudaMemcpyDeviceToHost, stream);
    if (mode == 2u || mode == 3u)
    {
        fprintf(stderr, "poc: end capture rc=%d\n", (int)cudaStreamEndCapture(stream, &graph));
        cudaError_t ri = cudaGraphInstantiate(&exec, graph, 0);
        fprintf(stderr, "poc: instantiate rc=%d\n", (int)ri);
        cudaError_t ru = cudaGraphUpload(exec, stream);
        fprintf(stderr, "poc: upload rc=%d\n", (int)ru);
        cudaError_t rs = cudaStreamSynchronize(stream);
        fprintf(stderr, "poc: upload sync rc=%d\n", (int)rs);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (uint32_t r = 0; r < POC_ROUNDS; r++)
        {
            uint64_t parity = r & 1u;
            uint16_t *slot = (uint16_t *)(band_alt + ((uint64_t)rank * 2u + parity) * SLOT_BYTES);
            poc_reset<<<POC_HIDDEN / 256u, 256u, 0, stream>>>(partial, rank);
            poc_compute<<<1024, 256, 0, stream>>>(work, compute_iters);
            cudaMemcpyAsync(slot, partial, POC_HIDDEN * 2u, cudaMemcpyDeviceToHost, stream);
            poc_publish<<<1, 32, 0, stream>>>(entry_alt, seq_cell, round_seq,
                (uint64_t)POC_HIDDEN * 2u, (uint64_t)rank * 2u + parity,
                stamps, r + POC_ROUNDS);
            poc_wait<<<1, 32, 0, stream>>>((volatile uint64_t *)band_alt, round_seq,
                rank, parity, stamps, r + POC_ROUNDS);
            for (uint32_t peer = 0; peer < RANKS_PER_BAND; peer++)
            {
                if (peer == rank) continue;
                uint16_t *peer_slot = (uint16_t *)(band_alt + ((uint64_t)peer * 2u + parity) * SLOT_BYTES);
                poc_combine<<<POC_HIDDEN / 256u, 256u, 0, stream>>>(partial, peer_slot);
            }
        }
        cudaMemcpyAsync(result_host, partial, POC_HIDDEN * 2u, cudaMemcpyDeviceToHost, stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec_alt, graph, 0);
        cudaGraphUpload(exec_alt, stream);
        cudaStreamSynchronize(stream);
        fprintf(stderr, "poc: alt graph ready\n");
    }

    double best = 1e9;
    for (uint32_t t = 0; t < tokens; t++)
    {
        g_stamps = stamps;
        g_alive = t + 1u;
        double s0 = now_s();
        if (mode == 2u || mode == 3u)
        {
            cudaError_t rl = cudaGraphLaunch((t & 1u) == 0u ? exec : exec_alt, stream);
            fprintf(stderr, "poc: launch t=%u graph=%s rc=%d\n", t,
                (t & 1u) == 0u ? "A" : "B", (int)rl);
            cudaError_t rys = cudaStreamSynchronize(stream);
            fprintf(stderr, "poc: sync t=%u rc=%d\n", t, (int)rys);
        }
        else
        {
            cudaStreamSynchronize(stream);
            for (uint32_t r = 0; r < POC_ROUNDS; r++)
            {
                uint64_t parity = r & 1u;
                uint16_t *slot = (uint16_t *)(band + ((uint64_t)rank * 2u + parity) * SLOT_BYTES);
                poc_reset<<<POC_HIDDEN / 256u, 256u, 0, stream>>>(partial, rank);
                poc_compute<<<1024, 256, 0, stream>>>(work, compute_iters);
                cudaMemcpyAsync(slot, partial, POC_HIDDEN * 2u, cudaMemcpyDeviceToHost, stream);
                poc_publish<<<1, 32, 0, stream>>>(entry, seq_cell, round_seq,
                    (uint64_t)POC_HIDDEN * 2u, (uint64_t)rank * 2u + parity,
                    stamps, r);
                cudaStreamSynchronize(stream);
                uint64_t seq = 0;
                cudaMemcpy(&seq, round_seq, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                uint64_t want = seq;
                for (uint32_t peer = 0; peer < RANKS_PER_BAND; peer++)
                {
                    if (peer == rank) continue;
                    volatile uint64_t *end_word = (volatile uint64_t *)
                        (band + ((uint64_t)peer * 2u + parity) * SLOT_BYTES + SLOT_BYTES - 8u);
                    while (*end_word < want)
                    {
                    }
                }
                for (uint32_t peer = 0; peer < RANKS_PER_BAND; peer++)
                {
                    if (peer == rank) continue;
                    uint16_t *peer_slot = (uint16_t *)(band + ((uint64_t)peer * 2u + parity) * SLOT_BYTES);
                    poc_combine<<<POC_HIDDEN / 256u, 256u, 0, stream>>>(partial, peer_slot);
                }
                cudaStreamSynchronize(stream);
            }
            cudaMemcpyAsync(result_host, partial, POC_HIDDEN * 2u, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        double dt = now_s() - s0;
        if (dt < best) best = dt;
        float expect_first = 0.0f;
        float got_first = 0.0f;
        {
            uint32_t bits = ((uint32_t)result_host[0]) << 16;
            memcpy(&got_first, &bits, 4);
            for (uint32_t p = 0; p < RANKS_PER_BAND; p++)
                expect_first += (float)(p * 1000u);
        }
        if (rank == 0)
            fprintf(stderr, "poc mode %u token %u: %.1f ms  check[0]=%.2f expect=%.2f\n",
                mode, t, dt * 1000.0, got_first, expect_first);
    }
    if (rank == 0)
        fprintf(stderr, "poc RESULT mode=%u rounds=%u best=%.1f ms/token -> %.2f tok/s-equivalent\n",
            mode, POC_ROUNDS, best * 1000.0, 1.0 / best);
    cudaMemcpy(stamps, stamps, 0, cudaMemcpyDeviceToDevice);
    if (mode == 3u)
    {
        double lat[POC_ROUNDS - 10u];
        uint32_t n = 0;
        for (uint32_t r = 10u; r < POC_ROUNDS; r++)
        {
            unsigned long long a = stamps_host[r * 2u];
            unsigned long long b = stamps_host[r * 2u + 1u];
            if (b > a) lat[n++] = (double)(b - a) / 1000.0;
        }
        for (uint32_t i = 1u; i < n; i++)
        {
            double key = lat[i];
            uint32_t j = i;
            while (j > 0u && lat[j - 1u] > key) { lat[j] = lat[j - 1u]; j--; }
            lat[j] = key;
        }
        if (n > 0)
            fprintf(stderr, "poc LATENCY rank0 rounds=%u min=%.1fus p50=%.1fus p90=%.1fus max=%.1fus\n",
                n, lat[0], lat[n / 2u], lat[(n * 9u) / 10u], lat[n - 1u]);
    }
    return 0;
}
