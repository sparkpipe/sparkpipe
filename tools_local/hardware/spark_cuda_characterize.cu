#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <pthread.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

#define SPARK_CUDA_PROBE_BLOCK_THREADS 256u
#define SPARK_CUDA_PROBE_MAX_BYTES (8ull * 1024ull * 1024ull * 1024ull)
#define SPARK_CUDA_PROBE_MAX_ITERATIONS 1000000u
#define SPARK_CUDA_PROBE_DEFAULT_THERMAL_SECONDS 1800u

struct SparkCudaProbeOptions
{
    const char *question_id;
    const char *candidate;
    const char *mode;
    const char *load_mode;
    const char *sample_phase;
    uint64_t working_set_bytes;
    uint64_t payload_bytes;
    uint32_t dynamic_shared_bytes;
    uint32_t batch_size;
    uint32_t kernel_count;
    uint32_t stream_count;
    uint64_t operations;
    uint32_t iterations;
    uint32_t sustained_seconds;
    const char *source_package_sha256;
    const char *run_id;
    const char *topology;
    const char *node;
    const char *output_path;
};

struct SparkCudaProbeLatency
{
    uint64_t p50_ns;
    uint64_t p95_ns;
    uint64_t p99_ns;
};

struct SparkCudaProbeTelemetry
{
    double temperature_c;
    double sm_clock_mhz;
    double memory_clock_mhz;
    double power_w;
    uint32_t available;
};

struct SparkCudaProbeCpuTraffic
{
    std::atomic<uint32_t> stop;
    uint8_t *buffer;
    uint64_t bytes;
    uint32_t write_mode;
    std::atomic<uint64_t> completed_bytes;
};

static std::atomic<uint64_t> SparkCudaProbeCallbackCounter(0u);

static uint64_t SparkCudaProbeMonotonicNanoseconds()
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0)
    {
        return 0u;
    }
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ull +
        static_cast<uint64_t>(value.tv_nsec);
}

static bool SparkCudaProbeCheck(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static bool SparkCudaProbeHexIsValid(const char *text)
{
    if (text == nullptr || std::strlen(text) != 64u)
    {
        return false;
    }
    for (uint32_t index = 0u; index < 64u; ++index)
    {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f')))
        {
            return false;
        }
    }
    return true;
}

static bool SparkCudaProbeParseU64(const char *text, uint64_t *value_out)
{
    char *end = nullptr;
    unsigned long long value;
    if (text == nullptr || value_out == nullptr || text[0] == '\0')
    {
        return false;
    }
    errno = 0;
    value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
    {
        return false;
    }
    *value_out = static_cast<uint64_t>(value);
    return true;
}

static bool SparkCudaProbeParseU32(
    const char *text,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t *value_out)
{
    uint64_t value;
    if (!SparkCudaProbeParseU64(text, &value) || value < minimum || value > maximum)
    {
        return false;
    }
    *value_out = static_cast<uint32_t>(value);
    return true;
}

static int SparkCudaProbeCompareU64(const void *left, const void *right)
{
    uint64_t a = *static_cast<const uint64_t *>(left);
    uint64_t b = *static_cast<const uint64_t *>(right);
    return a < b ? -1 : a > b ? 1 : 0;
}

static SparkCudaProbeLatency SparkCudaProbeSummarize(std::vector<uint64_t> samples)
{
    SparkCudaProbeLatency result{};
    if (samples.empty())
    {
        return result;
    }
    std::qsort(samples.data(), samples.size(), sizeof(samples[0]), SparkCudaProbeCompareU64);
    auto percentile = [&samples](uint32_t value) -> uint64_t
    {
        uint64_t index = ((samples.size() - 1u) * value + 99u) / 100u;
        return samples[static_cast<size_t>(index)];
    };
    result.p50_ns = percentile(50u);
    result.p95_ns = percentile(95u);
    result.p99_ns = percentile(99u);
    return result;
}

static void SparkCudaProbeWriteJsonString(FILE *output, const char *text)
{
    std::fputc('"', output);
    if (text != nullptr)
    {
        for (const unsigned char *cursor = reinterpret_cast<const unsigned char *>(text);
             *cursor != 0u; ++cursor)
        {
            if (*cursor == '"' || *cursor == '\\')
            {
                std::fputc('\\', output);
                std::fputc(*cursor, output);
            }
            else if (*cursor >= 0x20u && *cursor < 0x7fu)
            {
                std::fputc(*cursor, output);
            }
            else
            {
                std::fprintf(output, "\\u%04x", static_cast<unsigned>(*cursor));
            }
        }
    }
    std::fputc('"', output);
}

__global__ static void SparkCudaProbeEmptyKernel()
{
}

__global__ static void SparkCudaProbeReadKernel(
    const uint4 *source,
    uint64_t element_count,
    unsigned long long *sink,
    uint32_t repetitions)
{
    uint64_t index;
    uint64_t stride;
    unsigned long long accumulator;

    index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    accumulator = 0ull;
    for (uint32_t repetition = 0u; repetition < repetitions; ++repetition)
    {
        for (uint64_t current = index; current < element_count; current += stride)
        {
            uint4 value;

            value = source[current];
            accumulator += static_cast<unsigned long long>(value.x) +
                static_cast<unsigned long long>(value.y) +
                static_cast<unsigned long long>(value.z) +
                static_cast<unsigned long long>(value.w);
        }
    }
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u)
    {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if ((threadIdx.x & 31u) == 0u)
    {
        atomicAdd(sink, accumulator);
    }
}

__global__ static void SparkCudaProbeWriteKernel(
    uint4 *destination,
    uint64_t element_count,
    uint4 value,
    uint32_t repetitions)
{
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint32_t repetition = 0u; repetition < repetitions; ++repetition)
    {
        for (uint64_t current = index; current < element_count; current += stride)
        {
            destination[current] = value;
        }
    }
}

__global__ static void SparkCudaProbeCopyKernel(
    const uint4 *source,
    uint4 *destination,
    uint64_t element_count,
    uint32_t repetitions)
{
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint32_t repetition = 0u; repetition < repetitions; ++repetition)
    {
        for (uint64_t current = index; current < element_count; current += stride)
        {
            destination[current] = source[current];
        }
    }
}

__global__ static void SparkCudaProbePointerChaseKernel(
    const uint32_t *next,
    uint64_t steps,
    uint32_t *result)
{
    uint32_t index = 0u;
    for (uint64_t step = 0u; step < steps; ++step)
    {
        index = next[index];
    }
    *result = index;
}

__global__ static void SparkCudaProbeAtomicKernel(
    unsigned long long *counters,
    uint64_t operations,
    uint32_t distributed)
{
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t operation = index; operation < operations; operation += stride)
    {
        uint64_t counter = distributed != 0u ? operation & 4095u : 0u;
        atomicAdd(&counters[counter], 1ull);
    }
}

__global__ static void SparkCudaProbeDynamicSharedKernel(uint32_t bytes)
{
    extern __shared__ unsigned char storage[];
    for (uint32_t index = threadIdx.x; index < bytes; index += blockDim.x)
    {
        storage[index] = static_cast<unsigned char>(index + blockIdx.x);
    }
}

static uint32_t SparkCudaProbePatternWord(uint8_t byte_value)
{
    return static_cast<uint32_t>(byte_value) * 0x01010101u;
}

static unsigned long long SparkCudaProbeExpectedReadSum(
    uint64_t element_count,
    uint8_t byte_value,
    uint32_t repetitions)
{
    unsigned long long word;

    word = SparkCudaProbePatternWord(byte_value);
    return static_cast<unsigned long long>(element_count) *
        static_cast<unsigned long long>(repetitions) * 4ull * word;
}

static bool SparkCudaProbeReadSinkMatches(
    const unsigned long long *device_sink,
    unsigned long long expected,
    const char *operation)
{
    unsigned long long observed;

    observed = 0ull;
    if (!SparkCudaProbeCheck(cudaMemcpy(&observed, device_sink, sizeof(observed),
            cudaMemcpyDeviceToHost), operation))
    {
        return false;
    }
    return observed == expected;
}

static bool SparkCudaProbeDevicePatternMatches(
    const void *device_buffer,
    uint64_t bytes,
    uint4 expected,
    const char *operation)
{
    uint4 observed[2];
    const uint8_t *source;

    if (bytes < sizeof(uint4))
    {
        return false;
    }
    source = static_cast<const uint8_t *>(device_buffer);
    if (!SparkCudaProbeCheck(cudaMemcpy(&observed[0], source, sizeof(uint4),
            cudaMemcpyDeviceToHost), operation) ||
        !SparkCudaProbeCheck(cudaMemcpy(&observed[1], source + bytes - sizeof(uint4),
            sizeof(uint4), cudaMemcpyDeviceToHost), operation))
    {
        return false;
    }
    for (uint32_t index = 0u; index < 2u; ++index)
    {
        if (observed[index].x != expected.x || observed[index].y != expected.y ||
            observed[index].z != expected.z || observed[index].w != expected.w)
        {
            return false;
        }
    }
    return true;
}

static bool SparkCudaProbeHostPatternMatches(
    const void *host_buffer,
    uint64_t bytes,
    uint4 expected)
{
    uint4 observed[2];
    const uint8_t *source;

    if (host_buffer == nullptr || bytes < sizeof(uint4))
    {
        return false;
    }
    source = static_cast<const uint8_t *>(host_buffer);
    std::memcpy(&observed[0], source, sizeof(uint4));
    std::memcpy(&observed[1], source + bytes - sizeof(uint4), sizeof(uint4));
    for (uint32_t index = 0u; index < 2u; ++index)
    {
        if (observed[index].x != expected.x || observed[index].y != expected.y ||
            observed[index].z != expected.z || observed[index].w != expected.w)
        {
            return false;
        }
    }
    return true;
}

static uint32_t SparkCudaProbeGrid(const cudaDeviceProp &properties)
{
    return std::max(1, properties.multiProcessorCount * 8);
}

static double SparkCudaProbeElapsedGbPerSecond(uint64_t bytes, float milliseconds)
{
    return milliseconds <= 0.0f ? 0.0 : static_cast<double>(bytes) /
        (static_cast<double>(milliseconds) * 1000000.0);
}

static bool SparkCudaProbeMeasureKernelBandwidth(
    const cudaDeviceProp &properties,
    uint64_t bytes,
    uint32_t iterations,
    double *read_gb_s,
    double *write_gb_s,
    double *copy_gb_s)
{
    void *source;
    void *destination;
    unsigned long long *sink;
    cudaEvent_t start;
    cudaEvent_t finish;
    uint64_t aligned_bytes;
    uint64_t elements;
    uint32_t grid;
    bool ok;

    source = nullptr;
    destination = nullptr;
    sink = nullptr;
    start = nullptr;
    finish = nullptr;
    aligned_bytes = bytes / sizeof(uint4) * sizeof(uint4);
    if (aligned_bytes == 0u ||
        !SparkCudaProbeCheck(cudaMalloc(&source, aligned_bytes), "cudaMalloc source") ||
        !SparkCudaProbeCheck(cudaMalloc(&destination, aligned_bytes), "cudaMalloc destination") ||
        !SparkCudaProbeCheck(cudaMalloc(&sink, sizeof(*sink)), "cudaMalloc sink") ||
        !SparkCudaProbeCheck(cudaMemset(source, 0x5a, aligned_bytes), "cudaMemset source") ||
        !SparkCudaProbeCheck(cudaMemset(destination, 0, aligned_bytes), "cudaMemset destination") ||
        !SparkCudaProbeCheck(cudaMemset(sink, 0, sizeof(*sink)), "cudaMemset sink") ||
        !SparkCudaProbeCheck(cudaEventCreate(&start), "cudaEventCreate start") ||
        !SparkCudaProbeCheck(cudaEventCreate(&finish), "cudaEventCreate finish"))
    {
        if (finish != nullptr) cudaEventDestroy(finish);
        if (start != nullptr) cudaEventDestroy(start);
        if (sink != nullptr) cudaFree(sink);
        if (destination != nullptr) cudaFree(destination);
        if (source != nullptr) cudaFree(source);
        return false;
    }
    grid = SparkCudaProbeGrid(properties);
    elements = aligned_bytes / sizeof(uint4);
    auto time_kernel = [&](auto launch, uint64_t traffic_bytes, double *value_out) -> bool
    {
        float milliseconds;

        milliseconds = 0.0f;
        if (!SparkCudaProbeCheck(cudaEventRecord(start), "cudaEventRecord start"))
        {
            return false;
        }
        launch();
        if (!SparkCudaProbeCheck(cudaGetLastError(), "bandwidth kernel launch") ||
            !SparkCudaProbeCheck(cudaEventRecord(finish), "cudaEventRecord finish") ||
            !SparkCudaProbeCheck(cudaEventSynchronize(finish), "cudaEventSynchronize finish") ||
            !SparkCudaProbeCheck(cudaEventElapsedTime(&milliseconds, start, finish),
                "cudaEventElapsedTime"))
        {
            return false;
        }
        *value_out = SparkCudaProbeElapsedGbPerSecond(traffic_bytes, milliseconds);
        return *value_out > 0.0 && std::isfinite(*value_out);
    };
    ok = SparkCudaProbeCheck(cudaMemset(sink, 0, sizeof(*sink)), "reset read sink") &&
        time_kernel([&]() {
            SparkCudaProbeReadKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
                static_cast<const uint4 *>(source), elements, sink, iterations);
        }, aligned_bytes * iterations, read_gb_s) &&
        SparkCudaProbeReadSinkMatches(
            sink,
            SparkCudaProbeExpectedReadSum(elements, 0x5au, iterations),
            "verify read sink");
    if (ok)
    {
        ok = time_kernel([&]() {
                SparkCudaProbeWriteKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
                    static_cast<uint4 *>(destination), elements,
                    make_uint4(1u, 2u, 3u, 4u), iterations);
            }, aligned_bytes * iterations, write_gb_s) &&
            SparkCudaProbeDevicePatternMatches(
                destination,
                aligned_bytes,
                make_uint4(1u, 2u, 3u, 4u),
                "verify write pattern");
    }
    if (ok)
    {
        uint32_t source_word;

        source_word = SparkCudaProbePatternWord(0x5au);
        ok = time_kernel([&]() {
                SparkCudaProbeCopyKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
                    static_cast<const uint4 *>(source), static_cast<uint4 *>(destination),
                    elements, iterations);
            }, aligned_bytes * 2u * iterations, copy_gb_s) &&
            SparkCudaProbeDevicePatternMatches(
                destination,
                aligned_bytes,
                make_uint4(source_word, source_word, source_word, source_word),
                "verify copy pattern");
    }
    cudaEventDestroy(finish);
    cudaEventDestroy(start);
    cudaFree(sink);
    cudaFree(destination);
    cudaFree(source);
    return ok;
}


static bool SparkCudaProbeMeasureReadReuse(
    const cudaDeviceProp &properties,
    uint64_t bytes,
    uint32_t repetitions,
    double *single_pass_gb_s,
    double *repeated_pass_gb_s)
{
    void *source;
    unsigned long long *sink;
    cudaEvent_t start;
    cudaEvent_t finish;
    uint64_t aligned_bytes;
    uint64_t elements;
    uint32_t grid;
    uint32_t repeated;
    bool ok;

    source = nullptr;
    sink = nullptr;
    start = nullptr;
    finish = nullptr;
    aligned_bytes = bytes / sizeof(uint4) * sizeof(uint4);
    grid = SparkCudaProbeGrid(properties);
    repeated = std::max(2u, repetitions);
    if (aligned_bytes == 0u ||
        !SparkCudaProbeCheck(cudaMalloc(&source, aligned_bytes), "cache-reuse source") ||
        !SparkCudaProbeCheck(cudaMalloc(&sink, sizeof(*sink)), "cache-reuse sink") ||
        !SparkCudaProbeCheck(cudaMemset(source, 0x69, aligned_bytes), "cache-reuse initialize") ||
        !SparkCudaProbeCheck(cudaEventCreate(&start), "cache-reuse start") ||
        !SparkCudaProbeCheck(cudaEventCreate(&finish), "cache-reuse finish"))
    {
        if (finish != nullptr) cudaEventDestroy(finish);
        if (start != nullptr) cudaEventDestroy(start);
        if (sink != nullptr) cudaFree(sink);
        if (source != nullptr) cudaFree(source);
        return false;
    }
    elements = aligned_bytes / sizeof(uint4);
    auto measure = [&](uint32_t pass_count, double *throughput_out) -> bool
    {
        float milliseconds;

        milliseconds = 0.0f;
        if (!SparkCudaProbeCheck(cudaMemset(sink, 0, sizeof(*sink)), "cache-reuse reset sink") ||
            !SparkCudaProbeCheck(cudaEventRecord(start), "cache-reuse record start"))
        {
            return false;
        }
        SparkCudaProbeReadKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
            static_cast<const uint4 *>(source), elements, sink, pass_count);
        if (!SparkCudaProbeCheck(cudaGetLastError(), "cache-reuse launch") ||
            !SparkCudaProbeCheck(cudaEventRecord(finish), "cache-reuse record finish") ||
            !SparkCudaProbeCheck(cudaEventSynchronize(finish), "cache-reuse synchronize") ||
            !SparkCudaProbeCheck(cudaEventElapsedTime(&milliseconds, start, finish),
                "cache-reuse elapsed"))
        {
            return false;
        }
        *throughput_out = SparkCudaProbeElapsedGbPerSecond(
            aligned_bytes * pass_count, milliseconds);
        return *throughput_out > 0.0 && std::isfinite(*throughput_out) &&
            SparkCudaProbeReadSinkMatches(
                sink,
                SparkCudaProbeExpectedReadSum(elements, 0x69u, pass_count),
                "cache-reuse verify sink");
    };
    ok = measure(1u, single_pass_gb_s) && measure(repeated, repeated_pass_gb_s);
    cudaEventDestroy(finish);
    cudaEventDestroy(start);
    cudaFree(sink);
    cudaFree(source);
    return ok;
}


static bool SparkCudaProbeMeasureMapped(
    const cudaDeviceProp &properties,
    uint64_t bytes,
    uint32_t iterations,
    double *read_gb_s,
    double *write_gb_s)
{
    void *host;
    void *device_alias;
    unsigned long long *sink;
    cudaEvent_t start;
    cudaEvent_t finish;
    uint64_t aligned_bytes;
    uint64_t elements;
    uint32_t grid;
    bool ok;

    host = nullptr;
    device_alias = nullptr;
    sink = nullptr;
    start = nullptr;
    finish = nullptr;
    aligned_bytes = bytes / sizeof(uint4) * sizeof(uint4);
    if (aligned_bytes == 0u ||
        !SparkCudaProbeCheck(cudaHostAlloc(&host, aligned_bytes,
                cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc mapped") ||
        !SparkCudaProbeCheck(cudaHostGetDevicePointer(&device_alias, host, 0u),
            "cudaHostGetDevicePointer") ||
        !SparkCudaProbeCheck(cudaMalloc(&sink, sizeof(*sink)), "cudaMalloc mapped sink") ||
        !SparkCudaProbeCheck(cudaEventCreate(&start), "cudaEventCreate mapped start") ||
        !SparkCudaProbeCheck(cudaEventCreate(&finish), "cudaEventCreate mapped finish"))
    {
        if (finish != nullptr) cudaEventDestroy(finish);
        if (start != nullptr) cudaEventDestroy(start);
        if (sink != nullptr) cudaFree(sink);
        if (host != nullptr) cudaFreeHost(host);
        return false;
    }
    std::memset(host, 0x37, aligned_bytes);
    elements = aligned_bytes / sizeof(uint4);
    grid = SparkCudaProbeGrid(properties);
    auto time_kernel = [&](auto launch, double *value_out) -> bool
    {
        float milliseconds;

        milliseconds = 0.0f;
        if (!SparkCudaProbeCheck(cudaEventRecord(start), "mapped event start"))
        {
            return false;
        }
        launch();
        if (!SparkCudaProbeCheck(cudaGetLastError(), "mapped kernel launch") ||
            !SparkCudaProbeCheck(cudaEventRecord(finish), "mapped event finish") ||
            !SparkCudaProbeCheck(cudaEventSynchronize(finish), "mapped event synchronize") ||
            !SparkCudaProbeCheck(cudaEventElapsedTime(&milliseconds, start, finish),
                "mapped elapsed"))
        {
            return false;
        }
        *value_out = SparkCudaProbeElapsedGbPerSecond(
            aligned_bytes * iterations, milliseconds);
        return *value_out > 0.0 && std::isfinite(*value_out);
    };
    ok = SparkCudaProbeCheck(cudaMemset(sink, 0, sizeof(*sink)), "mapped reset sink") &&
        time_kernel([&]() {
            SparkCudaProbeReadKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
                static_cast<const uint4 *>(device_alias), elements, sink, iterations);
        }, read_gb_s) &&
        SparkCudaProbeReadSinkMatches(
            sink,
            SparkCudaProbeExpectedReadSum(elements, 0x37u, iterations),
            "mapped verify read sink");
    if (ok)
    {
        ok = time_kernel([&]() {
                SparkCudaProbeWriteKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
                    static_cast<uint4 *>(device_alias), elements,
                    make_uint4(5u, 6u, 7u, 8u), iterations);
            }, write_gb_s) &&
            SparkCudaProbeHostPatternMatches(
                host,
                aligned_bytes,
                make_uint4(5u, 6u, 7u, 8u));
    }
    cudaEventDestroy(finish);
    cudaEventDestroy(start);
    cudaFree(sink);
    cudaFreeHost(host);
    return ok;
}


static bool SparkCudaProbeMeasureCopies(
    uint64_t bytes,
    uint32_t iterations,
    double *d2d_gb_s,
    double *h2d_gb_s,
    double *d2h_gb_s,
    SparkCudaProbeLatency *d2d_latency,
    SparkCudaProbeLatency *h2d_latency,
    SparkCudaProbeLatency *d2h_latency)
{
    void *host_source;
    void *host_destination;
    void *device_a;
    void *device_b;
    cudaStream_t stream;
    bool ok;

    host_source = nullptr;
    host_destination = nullptr;
    device_a = nullptr;
    device_b = nullptr;
    stream = nullptr;
    if (!SparkCudaProbeCheck(cudaHostAlloc(&host_source, bytes, cudaHostAllocPortable),
            "cudaHostAlloc copy source") ||
        !SparkCudaProbeCheck(cudaHostAlloc(&host_destination, bytes, cudaHostAllocPortable),
            "cudaHostAlloc copy destination") ||
        !SparkCudaProbeCheck(cudaMalloc(&device_a, bytes), "cudaMalloc copy A") ||
        !SparkCudaProbeCheck(cudaMalloc(&device_b, bytes), "cudaMalloc copy B") ||
        !SparkCudaProbeCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreate copy"))
    {
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (device_b != nullptr) cudaFree(device_b);
        if (device_a != nullptr) cudaFree(device_a);
        if (host_destination != nullptr) cudaFreeHost(host_destination);
        if (host_source != nullptr) cudaFreeHost(host_source);
        return false;
    }
    std::memset(host_source, 0x4d, bytes);
    std::memset(host_destination, 0, bytes);
    ok = SparkCudaProbeCheck(cudaMemcpy(device_a, host_source, bytes,
            cudaMemcpyHostToDevice), "initialize copy source") &&
        SparkCudaProbeCheck(cudaMemset(device_b, 0, bytes), "initialize copy destination");
    struct CopyCase
    {
        void *destination;
        const void *source;
        cudaMemcpyKind kind;
        double *bandwidth;
        SparkCudaProbeLatency *latency;
    } cases[] = {
        {device_b, device_a, cudaMemcpyDeviceToDevice, d2d_gb_s, d2d_latency},
        {device_a, host_source, cudaMemcpyHostToDevice, h2d_gb_s, h2d_latency},
        {host_destination, device_a, cudaMemcpyDeviceToHost, d2h_gb_s, d2h_latency},
    };
    for (CopyCase &copy_case : cases)
    {
        std::vector<uint64_t> samples;
        uint64_t total_start;
        uint64_t elapsed;

        samples.reserve(iterations);
        total_start = SparkCudaProbeMonotonicNanoseconds();
        for (uint32_t iteration = 0u; ok && iteration < iterations; ++iteration)
        {
            uint64_t start_ns;
            uint64_t finish_ns;

            start_ns = SparkCudaProbeMonotonicNanoseconds();
            ok = SparkCudaProbeCheck(cudaMemcpyAsync(
                    copy_case.destination,
                    copy_case.source,
                    bytes,
                    copy_case.kind,
                    stream),
                "cudaMemcpyAsync") &&
                SparkCudaProbeCheck(cudaStreamSynchronize(stream), "copy synchronize");
            finish_ns = SparkCudaProbeMonotonicNanoseconds();
            if (ok && finish_ns > start_ns)
            {
                samples.push_back(finish_ns - start_ns);
            }
            else if (ok)
            {
                ok = false;
            }
        }
        elapsed = SparkCudaProbeMonotonicNanoseconds() - total_start;
        if (!ok || samples.size() != iterations || elapsed == 0u)
        {
            ok = false;
            break;
        }
        *copy_case.bandwidth = static_cast<double>(bytes) * iterations /
            static_cast<double>(elapsed);
        *copy_case.latency = SparkCudaProbeSummarize(samples);
        if (!std::isfinite(*copy_case.bandwidth) || *copy_case.bandwidth <= 0.0 ||
            copy_case.latency->p50_ns == 0u || copy_case.latency->p99_ns == 0u)
        {
            ok = false;
            break;
        }
        if (copy_case.kind == cudaMemcpyDeviceToDevice)
        {
            uint32_t word;

            word = SparkCudaProbePatternWord(0x4du);
            ok = SparkCudaProbeDevicePatternMatches(
                device_b,
                bytes,
                make_uint4(word, word, word, word),
                "verify D2D copy");
        }
        else if (copy_case.kind == cudaMemcpyHostToDevice)
        {
            uint32_t word;

            word = SparkCudaProbePatternWord(0x4du);
            ok = SparkCudaProbeDevicePatternMatches(
                device_a,
                bytes,
                make_uint4(word, word, word, word),
                "verify H2D copy");
        }
        else
        {
            uint32_t word;

            word = SparkCudaProbePatternWord(0x4du);
            ok = SparkCudaProbeHostPatternMatches(
                host_destination,
                bytes,
                make_uint4(word, word, word, word));
        }
    }
    cudaStreamDestroy(stream);
    cudaFree(device_b);
    cudaFree(device_a);
    cudaFreeHost(host_destination);
    cudaFreeHost(host_source);
    return ok;
}


static uint64_t SparkCudaProbeSplitMix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

static bool SparkCudaProbeMeasurePointerChase(
    uint64_t bytes,
    uint32_t iterations,
    double *latency_ns,
    double *latency_p99_ns,
    double *effective_gb_s)
{
    uint64_t element_count;
    std::vector<uint32_t> order;
    std::vector<uint32_t> next;
    std::vector<uint64_t> samples;
    uint32_t *device_next;
    uint32_t *device_result;
    cudaEvent_t start;
    cudaEvent_t finish;
    uint64_t steps;
    uint32_t expected_result;
    bool ok;

    element_count = bytes / sizeof(uint32_t);
    if (element_count < 1024u || element_count > UINT32_MAX || iterations == 0u)
    {
        return false;
    }
    order.resize(static_cast<size_t>(element_count));
    for (uint64_t index = 0u; index < element_count; ++index)
    {
        order[static_cast<size_t>(index)] = static_cast<uint32_t>(index);
    }
    for (uint64_t index = element_count - 1u; index > 0u; --index)
    {
        uint64_t selected;

        selected = SparkCudaProbeSplitMix64(index) % (index + 1u);
        std::swap(order[static_cast<size_t>(index)], order[static_cast<size_t>(selected)]);
    }
    next.resize(static_cast<size_t>(element_count));
    for (uint64_t index = 0u; index < element_count; ++index)
    {
        next[order[static_cast<size_t>(index)]] =
            order[static_cast<size_t>((index + 1u) % element_count)];
    }
    device_next = nullptr;
    device_result = nullptr;
    start = nullptr;
    finish = nullptr;
    if (!SparkCudaProbeCheck(cudaMalloc(&device_next, bytes), "cudaMalloc pointer next") ||
        !SparkCudaProbeCheck(cudaMalloc(&device_result, sizeof(*device_result)),
            "cudaMalloc pointer result") ||
        !SparkCudaProbeCheck(cudaMemcpy(device_next, next.data(), bytes,
            cudaMemcpyHostToDevice), "cudaMemcpy pointer next") ||
        !SparkCudaProbeCheck(cudaEventCreate(&start), "pointer event start") ||
        !SparkCudaProbeCheck(cudaEventCreate(&finish), "pointer event finish"))
    {
        if (finish != nullptr) cudaEventDestroy(finish);
        if (start != nullptr) cudaEventDestroy(start);
        if (device_result != nullptr) cudaFree(device_result);
        if (device_next != nullptr) cudaFree(device_next);
        return false;
    }
    steps = std::min<uint64_t>(element_count * 4u, 10000000u);
    expected_result = 0u;
    for (uint64_t step = 0u; step < steps; ++step)
    {
        expected_result = next[expected_result];
    }
    samples.reserve(iterations);
    ok = true;
    for (uint32_t iteration = 0u; ok && iteration < iterations; ++iteration)
    {
        uint32_t observed_result;
        float milliseconds;
        uint64_t elapsed_ns;

        observed_result = UINT32_MAX;
        milliseconds = 0.0f;
        ok = SparkCudaProbeCheck(cudaEventRecord(start), "pointer chase record start");
        if (ok)
        {
            SparkCudaProbePointerChaseKernel<<<1, 1>>>(device_next, steps, device_result);
            ok = SparkCudaProbeCheck(cudaGetLastError(), "pointer chase launch") &&
                SparkCudaProbeCheck(cudaEventRecord(finish), "pointer chase record finish") &&
                SparkCudaProbeCheck(cudaEventSynchronize(finish), "pointer chase synchronize") &&
                SparkCudaProbeCheck(cudaEventElapsedTime(&milliseconds, start, finish),
                    "pointer chase elapsed") &&
                SparkCudaProbeCheck(cudaMemcpy(&observed_result, device_result,
                    sizeof(observed_result), cudaMemcpyDeviceToHost),
                    "pointer chase result copy") &&
                observed_result == expected_result;
        }
        elapsed_ns = static_cast<uint64_t>(milliseconds * 1000000.0f);
        if (ok)
        {
            ok = elapsed_ns > 0u;
        }
        if (ok)
        {
            samples.push_back(elapsed_ns);
        }
    }
    if (ok)
    {
        SparkCudaProbeLatency summary;

        summary = SparkCudaProbeSummarize(samples);
        *latency_ns = static_cast<double>(summary.p50_ns) /
            static_cast<double>(steps);
        *latency_p99_ns = static_cast<double>(summary.p99_ns) /
            static_cast<double>(steps);
        *effective_gb_s = static_cast<double>(steps * sizeof(uint32_t)) /
            static_cast<double>(summary.p50_ns);
        ok = std::isfinite(*latency_ns) && *latency_ns > 0.0 &&
            std::isfinite(*latency_p99_ns) && *latency_p99_ns > 0.0 &&
            std::isfinite(*effective_gb_s) && *effective_gb_s > 0.0;
    }
    cudaEventDestroy(finish);
    cudaEventDestroy(start);
    cudaFree(device_result);
    cudaFree(device_next);
    return ok;
}


static bool SparkCudaProbeMeasureLaunch(
    uint32_t iterations,
    uint32_t batch_size,
    uint32_t kernel_count,
    bool synchronize_each,
    const char *load_mode,
    const cudaDeviceProp &properties,
    uint64_t working_set_bytes,
    SparkCudaProbeLatency *latency)
{
    void *load_buffer = nullptr;
    unsigned long long *load_sink = nullptr;
    cudaStream_t load_stream = nullptr;
    std::vector<uint64_t> samples;
    bool loaded;
    bool ok;
    uint64_t load_bytes;
    uint32_t load_repetitions;

    loaded = std::strcmp(load_mode, "memory_loaded") == 0;
    load_bytes = 0u;
    load_repetitions = 0u;
    ok = batch_size != 0u && kernel_count != 0u;
    if (ok && loaded)
    {
        load_bytes = std::max<uint64_t>(working_set_bytes, 64ull * 1024ull * 1024ull);
        load_bytes = load_bytes / sizeof(uint4) * sizeof(uint4);
        load_repetitions = std::max<uint32_t>(64u, iterations / 4u);
        ok = SparkCudaProbeCheck(cudaMalloc(&load_buffer, load_bytes), "launch load buffer") &&
            SparkCudaProbeCheck(cudaMalloc(&load_sink, sizeof(*load_sink)), "launch load sink") &&
            SparkCudaProbeCheck(cudaStreamCreateWithFlags(&load_stream, cudaStreamNonBlocking),
                "launch load stream") &&
            SparkCudaProbeCheck(cudaMemset(load_buffer, 0x39, load_bytes), "launch load initialize") &&
            SparkCudaProbeCheck(cudaMemset(load_sink, 0, sizeof(*load_sink)), "launch sink initialize");
        if (ok)
        {
            SparkCudaProbeReadKernel<<<SparkCudaProbeGrid(properties),
                SPARK_CUDA_PROBE_BLOCK_THREADS, 0, load_stream>>>(
                static_cast<const uint4 *>(load_buffer), load_bytes / sizeof(uint4),
                load_sink, load_repetitions);
            ok = SparkCudaProbeCheck(cudaGetLastError(), "launch background load");
        }
    }
    samples.reserve(iterations);
    for (uint32_t iteration = 0u; ok && iteration < iterations; ++iteration)
    {
        uint64_t start;

        start = SparkCudaProbeMonotonicNanoseconds();
        for (uint32_t kernel_index = 0u; ok && kernel_index < kernel_count; ++kernel_index)
        {
            SparkCudaProbeEmptyKernel<<<batch_size, 1>>>();
            ok = SparkCudaProbeCheck(cudaGetLastError(), "launch probe kernel");
        }
        if (ok && synchronize_each)
        {
            ok = SparkCudaProbeCheck(cudaDeviceSynchronize(), "launch probe synchronize");
        }
        if (ok)
        {
            samples.push_back(SparkCudaProbeMonotonicNanoseconds() - start);
        }
    }
    if (ok && !synchronize_each)
    {
        ok = SparkCudaProbeCheck(cudaDeviceSynchronize(), "launch probe final synchronize");
    }
    if (load_stream != nullptr)
    {
        if (!SparkCudaProbeCheck(cudaStreamSynchronize(load_stream), "launch load synchronize") ||
            !SparkCudaProbeReadSinkMatches(
                load_sink,
                SparkCudaProbeExpectedReadSum(
                    load_bytes / sizeof(uint4), 0x39u, load_repetitions),
                "launch load verify"))
        {
            ok = false;
        }
    }
    if (load_stream != nullptr)
    {
        cudaStreamDestroy(load_stream);
    }
    if (load_sink != nullptr)
    {
        cudaFree(load_sink);
    }
    if (load_buffer != nullptr)
    {
        cudaFree(load_buffer);
    }
    *latency = SparkCudaProbeSummarize(samples);
    return ok && samples.size() == iterations &&
        latency->p50_ns != 0u && latency->p99_ns != 0u;
}


static bool SparkCudaProbeMeasureGraph(
    const char *candidate,
    uint32_t batch_size,
    uint32_t kernel_count,
    uint32_t iterations,
    SparkCudaProbeLatency *latency,
    uint64_t *instantiate_ns)
{
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<uint64_t> samples;
    bool graph_mode;
    bool ok;

    graph_mode = std::strcmp(candidate, "graph") == 0;
    *instantiate_ns = 0u;
    ok = SparkCudaProbeCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "graph stream");
    if (ok && graph_mode)
    {
        ok = SparkCudaProbeCheck(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            "graph begin capture");
        for (uint32_t kernel_index = 0u; ok && kernel_index < kernel_count; ++kernel_index)
        {
            SparkCudaProbeEmptyKernel<<<batch_size, 1, 0, stream>>>();
            ok = SparkCudaProbeCheck(cudaGetLastError(), "graph captured kernel");
        }
        if (ok)
        {
            ok = SparkCudaProbeCheck(cudaStreamEndCapture(stream, &graph), "graph end capture");
        }
        if (ok)
        {
            uint64_t instantiate_start;
            cudaError_t instantiate_status;

            instantiate_start = SparkCudaProbeMonotonicNanoseconds();
            instantiate_status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u);
            *instantiate_ns = SparkCudaProbeMonotonicNanoseconds() - instantiate_start;
            ok = SparkCudaProbeCheck(instantiate_status, "graph instantiate");
        }
    }
    samples.reserve(iterations);
    for (uint32_t iteration = 0u; ok && iteration < iterations; ++iteration)
    {
        uint64_t start;

        start = SparkCudaProbeMonotonicNanoseconds();
        if (graph_mode)
        {
            ok = SparkCudaProbeCheck(cudaGraphLaunch(executable, stream), "graph launch");
        }
        else
        {
            for (uint32_t kernel_index = 0u; ok && kernel_index < kernel_count; ++kernel_index)
            {
                SparkCudaProbeEmptyKernel<<<batch_size, 1, 0, stream>>>();
                ok = SparkCudaProbeCheck(cudaGetLastError(), "direct graph-comparison kernel");
            }
        }
        if (ok)
        {
            ok = SparkCudaProbeCheck(cudaStreamSynchronize(stream),
                graph_mode ? "graph synchronize" : "direct synchronize");
        }
        if (ok)
        {
            samples.push_back(SparkCudaProbeMonotonicNanoseconds() - start);
        }
    }
    *latency = SparkCudaProbeSummarize(samples);
    if (executable != nullptr)
    {
        cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr)
    {
        cudaGraphDestroy(graph);
    }
    if (stream != nullptr)
    {
        cudaStreamDestroy(stream);
    }
    return ok && latency->p50_ns != 0u && latency->p99_ns != 0u;
}

static void CUDART_CB SparkCudaProbeHostCallback(void *)
{
    SparkCudaProbeCallbackCounter.fetch_add(1u, std::memory_order_relaxed);
}

static bool SparkCudaProbeMeasureCompletionMechanism(
    const char *candidate,
    const char *load_mode,
    const cudaDeviceProp &properties,
    uint32_t iterations,
    uint64_t working_set_bytes,
    SparkCudaProbeLatency *latency)
{
    void *load_buffer = nullptr;
    unsigned long long *sink = nullptr;
    cudaStream_t stream = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaEvent_t event = nullptr;
    std::vector<uint64_t> samples;
    uint64_t load_bytes;
    uint32_t load_repetitions;
    bool ok = SparkCudaProbeCheck(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "completion stream") &&
        SparkCudaProbeCheck(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "completion event");
    bool loaded = std::strcmp(load_mode, "memory_loaded") == 0;
    load_bytes = 0u;
    load_repetitions = 0u;
    if (ok && loaded)
    {
        ok = SparkCudaProbeCheck(
            cudaStreamCreateWithFlags(&load_stream, cudaStreamNonBlocking), "completion load stream");
        load_bytes = std::max<uint64_t>(working_set_bytes, 64ull * 1024ull * 1024ull);
        load_bytes = load_bytes / sizeof(uint4) * sizeof(uint4);
        load_repetitions = 64u;
        ok = SparkCudaProbeCheck(cudaMalloc(&load_buffer, load_bytes), "completion load buffer") &&
            SparkCudaProbeCheck(cudaMalloc(&sink, sizeof(*sink)), "completion load sink") &&
            SparkCudaProbeCheck(cudaMemset(load_buffer, 0x28, load_bytes), "completion load initialize") &&
            SparkCudaProbeCheck(cudaMemset(sink, 0, sizeof(*sink)), "completion sink initialize");
        if (ok)
        {
            SparkCudaProbeReadKernel<<<SparkCudaProbeGrid(properties),
                SPARK_CUDA_PROBE_BLOCK_THREADS, 0, load_stream>>>(
                static_cast<const uint4 *>(load_buffer), load_bytes / sizeof(uint4),
                sink, load_repetitions);
            ok = SparkCudaProbeCheck(cudaGetLastError(), "completion load launch");
        }
    }
    SparkCudaProbeCallbackCounter.store(0u, std::memory_order_relaxed);
    samples.reserve(iterations);
    for (uint32_t iteration = 0u; ok && iteration < iterations; ++iteration)
    {
        uint64_t start_ns = SparkCudaProbeMonotonicNanoseconds();
        SparkCudaProbeEmptyKernel<<<1, 1, 0, stream>>>();
        ok = SparkCudaProbeCheck(cudaGetLastError(), "completion empty launch");
        if (ok && std::strcmp(candidate, "host_callback") == 0)
        {
            ok = SparkCudaProbeCheck(
                cudaLaunchHostFunc(stream, SparkCudaProbeHostCallback, nullptr),
                "completion host callback") &&
                SparkCudaProbeCheck(cudaStreamSynchronize(stream), "completion callback synchronize");
        }
        else if (ok && std::strcmp(candidate, "event") == 0)
        {
            ok = SparkCudaProbeCheck(cudaEventRecord(event, stream), "completion event record") &&
                SparkCudaProbeCheck(cudaEventSynchronize(event), "completion event synchronize");
        }
        else if (ok)
        {
            ok = SparkCudaProbeCheck(cudaStreamSynchronize(stream), "completion stream synchronize");
        }
        if (ok)
        {
            samples.push_back(SparkCudaProbeMonotonicNanoseconds() - start_ns);
        }
    }
    if (loaded && load_stream != nullptr)
    {
        if (!SparkCudaProbeCheck(cudaStreamSynchronize(load_stream),
                "completion load synchronize") ||
            !SparkCudaProbeReadSinkMatches(
                sink,
                SparkCudaProbeExpectedReadSum(
                    load_bytes / sizeof(uint4), 0x28u, load_repetitions),
                "completion load verify"))
        {
            ok = false;
        }
    }
    if (sink != nullptr) cudaFree(sink);
    if (load_buffer != nullptr) cudaFree(load_buffer);
    if (event != nullptr) cudaEventDestroy(event);
    if (load_stream != nullptr) cudaStreamDestroy(load_stream);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (ok && std::strcmp(candidate, "host_callback") == 0 &&
        SparkCudaProbeCallbackCounter.load(std::memory_order_relaxed) != iterations)
    {
        ok = false;
    }
    *latency = SparkCudaProbeSummarize(samples);
    return ok && samples.size() == iterations &&
        latency->p50_ns != 0u && latency->p99_ns != 0u;
}

static bool SparkCudaProbeMeasureConcurrency(
    const char *candidate,
    const cudaDeviceProp &properties,
    uint64_t bytes,
    uint32_t stream_count,
    uint32_t iterations,
    double *overlap_ratio,
    double *aggregate_throughput_gb_s,
    uint64_t *serial_ns_out,
    uint64_t *parallel_ns_out)
{
    std::vector<void *> sources(stream_count, nullptr);
    std::vector<void *> destinations(stream_count, nullptr);
    std::vector<unsigned long long *> sinks(stream_count, nullptr);
    std::vector<cudaStream_t> streams(stream_count, nullptr);
    uint64_t traffic_bytes;
    uint64_t serial_start;
    uint64_t serial_ns;
    uint64_t parallel_start;
    uint64_t parallel_ns;
    uint32_t grid;
    bool ok;

    bytes = bytes / sizeof(uint4) * sizeof(uint4);
    ok = bytes != 0u && stream_count != 0u;
    for (uint32_t index = 0u; ok && index < stream_count; ++index)
    {
        ok = SparkCudaProbeCheck(cudaMalloc(&sources[index], bytes), "concurrency source") &&
            SparkCudaProbeCheck(cudaMalloc(&destinations[index], bytes), "concurrency destination") &&
            SparkCudaProbeCheck(cudaMalloc(&sinks[index], sizeof(*sinks[index])), "concurrency sink") &&
            SparkCudaProbeCheck(cudaStreamCreateWithFlags(&streams[index], cudaStreamNonBlocking),
                "concurrency stream") &&
            SparkCudaProbeCheck(cudaMemset(sources[index], (int)(0x11u + index), bytes),
                "concurrency source initialize") &&
            SparkCudaProbeCheck(cudaMemset(destinations[index], 0, bytes),
                "concurrency destination initialize") &&
            SparkCudaProbeCheck(cudaMemset(sinks[index], 0, sizeof(*sinks[index])),
                "concurrency sink initialize");
    }
    grid = SparkCudaProbeGrid(properties);
    auto launch = [&](uint32_t index, cudaStream_t stream) -> bool
    {
        bool copy_operation;

        copy_operation = std::strcmp(candidate, "copy_copy") == 0 ||
            (std::strcmp(candidate, "copy_compute") == 0 && (index & 1u) == 0u);
        if (copy_operation)
        {
            return SparkCudaProbeCheck(
                cudaMemcpyAsync(destinations[index], sources[index], bytes,
                    cudaMemcpyDeviceToDevice, stream),
                "concurrency copy launch");
        }
        SparkCudaProbeReadKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS, 0, stream>>>(
            static_cast<const uint4 *>(sources[index]), bytes / sizeof(uint4),
            sinks[index], iterations);
        return SparkCudaProbeCheck(cudaGetLastError(), "concurrency compute launch");
    };
    serial_start = SparkCudaProbeMonotonicNanoseconds();
    for (uint32_t index = 0u; ok && index < stream_count; ++index)
    {
        ok = launch(index, streams[0]) &&
            SparkCudaProbeCheck(cudaStreamSynchronize(streams[0]),
                "concurrency serial synchronize");
    }
    serial_ns = SparkCudaProbeMonotonicNanoseconds() - serial_start;
    parallel_start = SparkCudaProbeMonotonicNanoseconds();
    for (uint32_t index = 0u; ok && index < stream_count; ++index)
    {
        ok = launch(index, streams[index]);
    }
    for (uint32_t index = 0u; ok && index < stream_count; ++index)
    {
        ok = SparkCudaProbeCheck(cudaStreamSynchronize(streams[index]),
            "concurrency parallel synchronize");
    }
    parallel_ns = SparkCudaProbeMonotonicNanoseconds() - parallel_start;
    traffic_bytes = 0u;
    for (uint32_t index = 0u; ok && index < stream_count; ++index)
    {
        bool copy_operation;
        uint8_t byte_value;

        copy_operation = std::strcmp(candidate, "copy_copy") == 0 ||
            (std::strcmp(candidate, "copy_compute") == 0 && (index & 1u) == 0u);
        byte_value = static_cast<uint8_t>(0x11u + index);
        traffic_bytes += copy_operation ? bytes * 2u : bytes * iterations;
        if (copy_operation)
        {
            uint32_t word;

            word = SparkCudaProbePatternWord(byte_value);
            ok = SparkCudaProbeDevicePatternMatches(
                destinations[index],
                bytes,
                make_uint4(word, word, word, word),
                "verify concurrency copy");
        }
        else
        {
            ok = SparkCudaProbeReadSinkMatches(
                sinks[index],
                SparkCudaProbeExpectedReadSum(
                    bytes / sizeof(uint4), byte_value, iterations) * 2ull,
                "verify concurrency read");
        }
    }
    *serial_ns_out = serial_ns;
    *parallel_ns_out = parallel_ns;
    *overlap_ratio = parallel_ns == 0u ? 0.0 :
        static_cast<double>(serial_ns) / static_cast<double>(parallel_ns);
    *aggregate_throughput_gb_s = parallel_ns == 0u ? 0.0 :
        static_cast<double>(traffic_bytes) / static_cast<double>(parallel_ns);
    ok = ok && serial_ns != 0u && parallel_ns != 0u &&
        std::isfinite(*overlap_ratio) && *overlap_ratio > 0.0 &&
        std::isfinite(*aggregate_throughput_gb_s) && *aggregate_throughput_gb_s > 0.0;
    for (uint32_t index = 0u; index < stream_count; ++index)
    {
        if (streams[index] != nullptr)
        {
            cudaStreamDestroy(streams[index]);
        }
        if (sinks[index] != nullptr)
        {
            cudaFree(sinks[index]);
        }
        if (destinations[index] != nullptr)
        {
            cudaFree(destinations[index]);
        }
        if (sources[index] != nullptr)
        {
            cudaFree(sources[index]);
        }
    }
    return ok;
}


static bool SparkCudaProbeMeasureAtomic(
    const cudaDeviceProp &properties,
    bool distributed,
    uint64_t operations,
    uint32_t iterations,
    double *operations_per_second)
{
    unsigned long long *counters;
    cudaEvent_t start;
    cudaEvent_t finish;
    std::vector<unsigned long long> observed;
    unsigned long long observed_total;
    unsigned long long expected_total;
    bool ok;

    counters = nullptr;
    start = nullptr;
    finish = nullptr;
    observed.resize(4096u);
    ok = SparkCudaProbeCheck(cudaMalloc(&counters, observed.size() * sizeof(*counters)),
            "atomic counters") &&
        SparkCudaProbeCheck(cudaMemset(counters, 0, observed.size() * sizeof(*counters)),
            "atomic memset") &&
        SparkCudaProbeCheck(cudaEventCreate(&start), "atomic start") &&
        SparkCudaProbeCheck(cudaEventCreate(&finish), "atomic finish") &&
        SparkCudaProbeCheck(cudaEventRecord(start), "atomic start record");
    for (uint32_t iteration = 0u; ok && iteration < iterations; ++iteration)
    {
        SparkCudaProbeAtomicKernel<<<SparkCudaProbeGrid(properties),
            SPARK_CUDA_PROBE_BLOCK_THREADS>>>(counters, operations, distributed ? 1u : 0u);
        ok = SparkCudaProbeCheck(cudaGetLastError(), "atomic launch");
    }
    if (ok)
    {
        ok = SparkCudaProbeCheck(cudaEventRecord(finish), "atomic finish record") &&
            SparkCudaProbeCheck(cudaEventSynchronize(finish), "atomic synchronize");
    }
    float milliseconds = 0.0f;
    if (ok)
    {
        ok = SparkCudaProbeCheck(cudaEventElapsedTime(&milliseconds, start, finish),
                "atomic elapsed") &&
            SparkCudaProbeCheck(cudaMemcpy(observed.data(), counters,
                observed.size() * sizeof(observed[0]), cudaMemcpyDeviceToHost),
                "atomic result copy");
    }
    observed_total = 0ull;
    for (unsigned long long value : observed)
    {
        observed_total += value;
    }
    expected_total = static_cast<unsigned long long>(operations) * iterations;
    if (ok)
    {
        ok = observed_total == expected_total;
        if (!distributed)
        {
            ok = ok && observed[0] == expected_total;
            for (size_t index = 1u; ok && index < observed.size(); ++index)
            {
                ok = observed[index] == 0ull;
            }
        }
    }
    if (ok)
    {
        *operations_per_second = milliseconds <= 0.0f ? 0.0 :
            static_cast<double>(operations) * static_cast<double>(iterations) * 1000.0 /
                static_cast<double>(milliseconds);
        ok = *operations_per_second > 0.0 && std::isfinite(*operations_per_second);
    }
    if (finish != nullptr)
    {
        cudaEventDestroy(finish);
    }
    if (start != nullptr)
    {
        cudaEventDestroy(start);
    }
    if (counters != nullptr)
    {
        cudaFree(counters);
    }
    return ok;
}


static void *SparkCudaProbeCpuTrafficMain(void *context)
{
    SparkCudaProbeCpuTraffic *traffic = static_cast<SparkCudaProbeCpuTraffic *>(context);
    uint64_t completed = 0u;
    volatile uint64_t sink = 0u;
    while (traffic->stop.load(std::memory_order_relaxed) == 0u)
    {
        if (traffic->write_mode != 0u)
        {
            for (uint64_t index = 0u; index < traffic->bytes; index += 64u)
            {
                traffic->buffer[index] = static_cast<uint8_t>(index + completed);
            }
        }
        else
        {
            for (uint64_t index = 0u; index < traffic->bytes; index += 64u)
            {
                sink += traffic->buffer[index];
            }
        }
        completed += traffic->bytes;
    }
    traffic->completed_bytes.store(completed + sink * 0u, std::memory_order_relaxed);
    return nullptr;
}

static bool SparkCudaProbeMeasureUnifiedContention(
    const SparkCudaProbeOptions &options,
    const cudaDeviceProp &properties,
    double *gpu_read_gb_s,
    double *gpu_write_gb_s,
    double *cpu_gb_s)
{
    *gpu_read_gb_s = 0.0;
    *gpu_write_gb_s = 0.0;
    *cpu_gb_s = 0.0;
    if (std::strcmp(options.candidate, "mapped_host_gpu_read") == 0 ||
        std::strcmp(options.candidate, "mapped_host_gpu_write") == 0)
    {
        double read_value = 0.0;
        double write_value = 0.0;
        if (!SparkCudaProbeMeasureMapped(properties, options.working_set_bytes,
                options.iterations, &read_value, &write_value))
        {
            return false;
        }
        *gpu_read_gb_s = read_value;
        *gpu_write_gb_s = write_value;
        return true;
    }
    SparkCudaProbeCpuTraffic traffic{};
    pthread_t thread{};
    uint64_t cpu_bytes = std::min<uint64_t>(options.working_set_bytes,
        2ull * 1024ull * 1024ull * 1024ull);
    if (std::strcmp(options.candidate, "cpu_read_contention") == 0 ||
        std::strcmp(options.candidate, "cpu_write_contention") == 0)
    {
        if (posix_memalign(reinterpret_cast<void **>(&traffic.buffer), 4096u,
                static_cast<size_t>(cpu_bytes)) != 0)
        {
            return false;
        }
        std::memset(traffic.buffer, 0x61, static_cast<size_t>(cpu_bytes));
        traffic.bytes = cpu_bytes;
        traffic.write_mode = std::strcmp(options.candidate, "cpu_write_contention") == 0 ? 1u : 0u;
        if (pthread_create(&thread, nullptr, SparkCudaProbeCpuTrafficMain, &traffic) != 0)
        {
            std::free(traffic.buffer);
            return false;
        }
    }
    uint64_t start = SparkCudaProbeMonotonicNanoseconds();
    double copy_value = 0.0;
    bool ok = SparkCudaProbeMeasureKernelBandwidth(properties, options.working_set_bytes,
        options.iterations, gpu_read_gb_s, gpu_write_gb_s, &copy_value);
    uint64_t elapsed = SparkCudaProbeMonotonicNanoseconds() - start;
    if (traffic.buffer != nullptr)
    {
        traffic.stop.store(1u, std::memory_order_relaxed);
        pthread_join(thread, nullptr);
        *cpu_gb_s = elapsed == 0u ? 0.0 :
            static_cast<double>(traffic.completed_bytes.load(std::memory_order_relaxed)) /
                static_cast<double>(elapsed);
        std::free(traffic.buffer);
    }
    return ok;
}

static bool SparkCudaProbeParseTelemetryValue(char *text, double *value_out)
{
    char *cursor;
    char *end;
    double value;

    if (text == nullptr || value_out == nullptr)
    {
        return false;
    }
    cursor = text;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n')
    {
        ++cursor;
    }
    if (std::strcmp(cursor, "N/A") == 0 || std::strcmp(cursor, "[N/A]") == 0)
    {
        *value_out = -1.0;
        return true;
    }
    errno = 0;
    end = nullptr;
    value = std::strtod(cursor, &end);
    if (errno != 0 || end == cursor)
    {
        return false;
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')
    {
        ++end;
    }
    if (*end != '\0')
    {
        return false;
    }
    *value_out = value;
    return true;
}

static SparkCudaProbeTelemetry SparkCudaProbeReadTelemetry()
{
    SparkCudaProbeTelemetry telemetry{};
    telemetry.temperature_c = -1.0;
    telemetry.sm_clock_mhz = -1.0;
    telemetry.memory_clock_mhz = -1.0;
    telemetry.power_w = -1.0;
    FILE *pipe = popen(
        "nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw "
        "--format=csv,noheader,nounits 2>/dev/null | head -n 1",
        "r");
    if (pipe == nullptr)
    {
        return telemetry;
    }
    char line[256u];
    if (std::fgets(line, sizeof(line), pipe) != nullptr)
    {
        char memory_clock_text[32u];
        double temperature;
        double sm_clock;
        double memory_clock;
        double power;
        if (std::sscanf(line, " %lf , %lf , %31[^,] , %lf",
                &temperature, &sm_clock, memory_clock_text, &power) == 4 &&
            SparkCudaProbeParseTelemetryValue(memory_clock_text, &memory_clock))
        {
            telemetry.temperature_c = temperature;
            telemetry.sm_clock_mhz = sm_clock;
            telemetry.memory_clock_mhz = memory_clock;
            telemetry.power_w = power;
            telemetry.available = 1u;
        }
    }
    pclose(pipe);
    return telemetry;
}

static bool SparkCudaProbeMeasureThermalCopySample(
    cudaEvent_t start,
    cudaEvent_t finish,
    const void *source,
    void *destination,
    uint64_t element_count,
    uint64_t aligned_bytes,
    uint32_t grid,
    uint32_t repetitions,
    double *copy_gb_s)
{
    float milliseconds;

    milliseconds = 0.0f;
    if (!SparkCudaProbeCheck(cudaEventRecord(start), "thermal cudaEventRecord start"))
    {
        return false;
    }
    SparkCudaProbeCopyKernel<<<grid, SPARK_CUDA_PROBE_BLOCK_THREADS>>>(
        static_cast<const uint4 *>(source),
        static_cast<uint4 *>(destination),
        element_count,
        repetitions);
    if (!SparkCudaProbeCheck(cudaGetLastError(), "thermal copy kernel launch") ||
        !SparkCudaProbeCheck(cudaEventRecord(finish), "thermal cudaEventRecord finish") ||
        !SparkCudaProbeCheck(cudaEventSynchronize(finish), "thermal cudaEventSynchronize finish") ||
        !SparkCudaProbeCheck(cudaEventElapsedTime(&milliseconds, start, finish),
            "thermal cudaEventElapsedTime"))
    {
        return false;
    }
    *copy_gb_s = SparkCudaProbeElapsedGbPerSecond(
        aligned_bytes * 2ull * repetitions,
        milliseconds);
    return *copy_gb_s > 0.0 && std::isfinite(*copy_gb_s);
}

static bool SparkCudaProbeMeasureThermal(
    const cudaDeviceProp &properties,
    uint64_t bytes,
    uint32_t seconds,
    SparkCudaProbeTelemetry telemetry[3],
    double bandwidth[3],
    uint32_t *cuda_error_count,
    uint64_t *sample_count_out,
    double *actual_seconds_out)
{
    void *source;
    void *destination;
    cudaEvent_t start_event;
    cudaEvent_t finish_event;
    uint64_t aligned_bytes;
    uint64_t element_count;
    uint64_t start_ns;
    uint64_t midpoint_ns;
    uint64_t deadline_ns;
    uint64_t now_ns;
    uint32_t source_word;
    uint32_t grid;
    std::vector<double> samples;
    std::vector<double> steady_samples;
    bool ok;

    source = nullptr;
    destination = nullptr;
    start_event = nullptr;
    finish_event = nullptr;
    *cuda_error_count = 0u;
    *sample_count_out = 0u;
    *actual_seconds_out = 0.0;
    bandwidth[0] = 0.0;
    bandwidth[1] = 0.0;
    bandwidth[2] = 0.0;
    telemetry[0] = {};
    telemetry[1] = {};
    telemetry[2] = {};
    if (seconds < 3u)
    {
        return false;
    }
    aligned_bytes = bytes / sizeof(uint4) * sizeof(uint4);
    if (aligned_bytes == 0u ||
        !SparkCudaProbeCheck(cudaMalloc(&source, aligned_bytes), "thermal cudaMalloc source") ||
        !SparkCudaProbeCheck(cudaMalloc(&destination, aligned_bytes),
            "thermal cudaMalloc destination") ||
        !SparkCudaProbeCheck(cudaMemset(source, 0x5a, aligned_bytes),
            "thermal cudaMemset source") ||
        !SparkCudaProbeCheck(cudaMemset(destination, 0, aligned_bytes),
            "thermal cudaMemset destination") ||
        !SparkCudaProbeCheck(cudaEventCreate(&start_event),
            "thermal cudaEventCreate start") ||
        !SparkCudaProbeCheck(cudaEventCreate(&finish_event),
            "thermal cudaEventCreate finish"))
    {
        if (finish_event != nullptr) cudaEventDestroy(finish_event);
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (destination != nullptr) cudaFree(destination);
        if (source != nullptr) cudaFree(source);
        return false;
    }
    telemetry[0] = SparkCudaProbeReadTelemetry();
    if (telemetry[0].available == 0u)
    {
        cudaEventDestroy(finish_event);
        cudaEventDestroy(start_event);
        cudaFree(destination);
        cudaFree(source);
        return false;
    }
    element_count = aligned_bytes / sizeof(uint4);
    grid = SparkCudaProbeGrid(properties);
    start_ns = SparkCudaProbeMonotonicNanoseconds();
    if (start_ns == 0u)
    {
        cudaEventDestroy(finish_event);
        cudaEventDestroy(start_event);
        cudaFree(destination);
        cudaFree(source);
        return false;
    }
    now_ns = start_ns;
    midpoint_ns = start_ns + (static_cast<uint64_t>(seconds) * 1000000000ull) / 2ull;
    deadline_ns = start_ns + static_cast<uint64_t>(seconds) * 1000000000ull;
    samples.reserve(static_cast<size_t>(seconds) * 256u);
    steady_samples.reserve(static_cast<size_t>(seconds) * 128u);
    ok = true;
    do
    {
        double sample_gb_s;

        sample_gb_s = 0.0;
        if (!SparkCudaProbeMeasureThermalCopySample(
                start_event,
                finish_event,
                source,
                destination,
                element_count,
                aligned_bytes,
                grid,
                4u,
                &sample_gb_s))
        {
            ++*cuda_error_count;
            ok = false;
            break;
        }
        samples.push_back(sample_gb_s);
        now_ns = SparkCudaProbeMonotonicNanoseconds();
        if (now_ns == 0u)
        {
            ok = false;
            break;
        }
        if (now_ns >= midpoint_ns)
        {
            steady_samples.push_back(sample_gb_s);
            if (telemetry[1].available == 0u)
            {
                telemetry[1] = SparkCudaProbeReadTelemetry();
            }
        }
    } while (now_ns < deadline_ns);
    telemetry[2] = SparkCudaProbeReadTelemetry();
    *actual_seconds_out = now_ns > start_ns ?
        static_cast<double>(now_ns - start_ns) / 1000000000.0 : 0.0;
    *sample_count_out = samples.size();
    source_word = SparkCudaProbePatternWord(0x5au);
    if (ok)
    {
        ok = !samples.empty() && !steady_samples.empty() &&
            telemetry[1].available != 0u && telemetry[2].available != 0u &&
            *actual_seconds_out >= static_cast<double>(seconds) &&
            SparkCudaProbeDevicePatternMatches(
                destination,
                aligned_bytes,
                make_uint4(source_word, source_word, source_word, source_word),
                "verify thermal copy pattern");
    }
    if (ok)
    {
        std::sort(steady_samples.begin(), steady_samples.end());
        bandwidth[0] = samples.front();
        bandwidth[1] = steady_samples[steady_samples.size() / 2u];
        bandwidth[2] = samples.back();
        ok = bandwidth[0] > 0.0 && bandwidth[1] > 0.0 && bandwidth[2] > 0.0 &&
            std::isfinite(bandwidth[0]) && std::isfinite(bandwidth[1]) &&
            std::isfinite(bandwidth[2]);
    }
    cudaEventDestroy(finish_event);
    cudaEventDestroy(start_event);
    cudaFree(destination);
    cudaFree(source);
    return ok;
}

static FILE *SparkCudaProbeOpenOutput(const SparkCudaProbeOptions &options)
{
    return options.output_path == nullptr ? stdout : std::fopen(options.output_path, "wb");
}

static void SparkCudaProbeWriteReceiptPrefix(
    FILE *output,
    const SparkCudaProbeOptions &options,
    const char *status,
    const char *summary_json)
{
    std::fprintf(output,
        "{\n  \"schema_version\": 1,\n  \"receipt_kind\": \"spark_hardware_probe\",\n"
        "  \"run_id\": ");
    SparkCudaProbeWriteJsonString(output, options.run_id);
    std::fprintf(output,
        ",\n  \"probe_id\": \"cuda_characterize\",\n"
        "  \"source_identity\": {\"source_package_sha256\": \"%s\"},\n"
        "  \"scope\": {\"topology\": ", options.source_package_sha256);
    SparkCudaProbeWriteJsonString(output, options.topology);
    std::fprintf(output, ", \"node\": ");
    SparkCudaProbeWriteJsonString(output, options.node);
    std::fprintf(output,
        "},\n  \"answers\": [\n    {\"question_id\": ");
    SparkCudaProbeWriteJsonString(output, options.question_id);
    std::fprintf(output, ", \"status\": \"%s\", \"summary\": %s, \"observations\": [",
        status, summary_json == nullptr ? "{}" : summary_json);
}

static void SparkCudaProbeWriteReceiptSuffix(FILE *output, const char *error)
{
    std::fprintf(output, "]");
    if (error != nullptr)
    {
        std::fprintf(output, ", \"error\": ");
        SparkCudaProbeWriteJsonString(output, error);
    }
    std::fprintf(output, "}\n  ]\n}\n");
}

static bool SparkCudaProbeRunQuestion(
    const SparkCudaProbeOptions &options,
    const cudaDeviceProp &properties,
    int driver_version,
    int runtime_version,
    int memory_clock_rate)
{
    FILE *output;
    bool ok;

    output = SparkCudaProbeOpenOutput(options);
    if (output == nullptr)
    {
        return false;
    }
    ok = true;
    if (std::strcmp(options.question_id, "GB10-IDENTITY-001") == 0)
    {
        bool identity_pass;

        identity_pass = properties.major == 12 && properties.minor == 1 &&
            std::strstr(properties.name, "GB10") != nullptr;
        SparkCudaProbeWriteReceiptPrefix(output, options, "measured", "{}");
        std::fprintf(output,
            "{\"parameters\": {\"candidate\": \"identity\", \"iterations\": %u}, "
            "\"metrics\": {\"device_name\": ",
            options.iterations);
        SparkCudaProbeWriteJsonString(output, properties.name);
        std::fprintf(output,
            ", \"device_count\": 1, \"compute_major\": %d, \"compute_minor\": %d, "
            "\"driver_version\": %d, \"runtime_version\": %d, "
            "\"global_memory_bytes\": %zu, \"sm_count\": %d, "
            "\"memory_clock_khz\": %d, \"memory_bus_width_bits\": %d, "
            "\"device_is_gb10_sm121\": %s, \"integrity_pass\": %s, "
            "\"numerical_pass\": %s}}",
            properties.major,
            properties.minor,
            driver_version,
            runtime_version,
            properties.totalGlobalMem,
            properties.multiProcessorCount,
            memory_clock_rate,
            properties.memoryBusWidth,
            identity_pass ? "true" : "false",
            identity_pass ? "true" : "false",
            identity_pass ? "true" : "false");
        SparkCudaProbeWriteReceiptSuffix(output, nullptr);
        ok = identity_pass;
    }
    else if (std::strcmp(options.question_id, "GB10-MEM-001") == 0)
    {
        double read_gb_s;
        double write_gb_s;
        double copy_gb_s;

        read_gb_s = 0.0;
        write_gb_s = 0.0;
        copy_gb_s = 0.0;
        ok = SparkCudaProbeMeasureKernelBandwidth(
            properties,
            options.working_set_bytes,
            options.iterations,
            &read_gb_s,
            &write_gb_s,
            &copy_gb_s);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output,
                "{\"parameters\": {\"candidate\": \"bandwidth\", "
                "\"working_set_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"read_gb_s\": %.17g, \"write_gb_s\": %.17g, "
                "\"copy_gb_s\": %.17g, \"integrity_pass\": true, "
                "\"numerical_pass\": true}}",
                options.working_set_bytes,
                options.iterations,
                read_gb_s,
                write_gb_s,
                copy_gb_s);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "memory bandwidth measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-MEM-002") == 0)
    {
        double single_pass_gb_s;
        double repeated_pass_gb_s;
        double reuse_gain;

        single_pass_gb_s = 0.0;
        repeated_pass_gb_s = 0.0;
        ok = SparkCudaProbeMeasureReadReuse(
            properties,
            options.working_set_bytes,
            options.iterations,
            &single_pass_gb_s,
            &repeated_pass_gb_s);
        reuse_gain = single_pass_gb_s == 0.0 ? 0.0 :
            repeated_pass_gb_s / single_pass_gb_s;
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output,
                "{\"parameters\": {\"candidate\": \"reuse\", "
                "\"working_set_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"single_pass_read_gb_s\": %.17g, "
                "\"repeated_read_gb_s\": %.17g, \"reuse_gain\": %.17g, "
                "\"integrity_pass\": true, \"numerical_pass\": true}}",
                options.working_set_bytes,
                options.iterations,
                single_pass_gb_s,
                repeated_pass_gb_s,
                reuse_gain);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "cache-reuse measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-MEM-003") == 0)
    {
        double latency_ns;
        double latency_p99_ns;
        double effective_gb_s;

        latency_ns = 0.0;
        latency_p99_ns = 0.0;
        effective_gb_s = 0.0;
        ok = SparkCudaProbeMeasurePointerChase(
            options.working_set_bytes,
            options.iterations,
            &latency_ns,
            &latency_p99_ns,
            &effective_gb_s);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output,
                "{\"parameters\": {\"candidate\": \"pointer_chase\", "
                "\"working_set_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"pointer_chase_ns_per_step\": %.17g, "
                "\"pointer_chase_p99_ns_per_step\": %.17g, "
                "\"throughput_gb_s\": %.17g, \"sample_count\": %u, "
                "\"integrity_pass\": true, "
                "\"numerical_pass\": true}}",
                options.working_set_bytes,
                options.iterations,
                latency_ns,
                latency_p99_ns,
                effective_gb_s,
                options.iterations);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "pointer chase failed");
    }
    else if (std::strcmp(options.question_id, "GB10-UMEM-001") == 0)
    {
        double baseline_read_gb_s;
        double baseline_write_gb_s;
        double candidate_read_gb_s;
        double candidate_write_gb_s;
        double candidate_cpu_gb_s;
        double baseline_gpu_gb_s;
        double candidate_gpu_gb_s;
        double gpu_bandwidth_ratio;
        SparkCudaProbeOptions baseline_options;

        baseline_read_gb_s = 0.0;
        baseline_write_gb_s = 0.0;
        candidate_read_gb_s = 0.0;
        candidate_write_gb_s = 0.0;
        candidate_cpu_gb_s = 0.0;
        baseline_options = options;
        baseline_options.candidate = "gpu_only";
        ok = SparkCudaProbeMeasureUnifiedContention(
            baseline_options,
            properties,
            &baseline_read_gb_s,
            &baseline_write_gb_s,
            &candidate_cpu_gb_s);
        if (ok && std::strcmp(options.candidate, "gpu_only") == 0)
        {
            candidate_read_gb_s = baseline_read_gb_s;
            candidate_write_gb_s = baseline_write_gb_s;
            candidate_cpu_gb_s = 0.0;
        }
        else if (ok)
        {
            ok = SparkCudaProbeMeasureUnifiedContention(
                options,
                properties,
                &candidate_read_gb_s,
                &candidate_write_gb_s,
                &candidate_cpu_gb_s);
        }
        baseline_gpu_gb_s = std::max(baseline_read_gb_s, baseline_write_gb_s);
        candidate_gpu_gb_s = std::max(candidate_read_gb_s, candidate_write_gb_s);
        gpu_bandwidth_ratio = baseline_gpu_gb_s == 0.0 ? 0.0 :
            candidate_gpu_gb_s / baseline_gpu_gb_s;
        ok = ok && std::isfinite(gpu_bandwidth_ratio) && gpu_bandwidth_ratio > 0.0;
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output, "{\"parameters\": {\"candidate\": ");
            SparkCudaProbeWriteJsonString(output, options.candidate);
            std::fprintf(output,
                ", \"working_set_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"baseline_gpu_read_gb_s\": %.17g, "
                "\"baseline_gpu_write_gb_s\": %.17g, "
                "\"gpu_read_gb_s\": %.17g, \"gpu_write_gb_s\": %.17g, "
                "\"cpu_traffic_gb_s\": %.17g, \"gpu_bandwidth_ratio\": %.17g, "
                "\"integrity_pass\": true, \"numerical_pass\": true}}",
                options.working_set_bytes,
                options.iterations,
                baseline_read_gb_s,
                baseline_write_gb_s,
                candidate_read_gb_s,
                candidate_write_gb_s,
                candidate_cpu_gb_s,
                gpu_bandwidth_ratio);
        }
        SparkCudaProbeWriteReceiptSuffix(
            output,
            ok ? nullptr : "unified-memory contention measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-MAPPED-001") == 0)
    {
        double read_gb_s;
        double write_gb_s;

        read_gb_s = 0.0;
        write_gb_s = 0.0;
        ok = SparkCudaProbeMeasureMapped(
            properties,
            options.payload_bytes,
            options.iterations,
            &read_gb_s,
            &write_gb_s);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output,
                "{\"parameters\": {\"candidate\": \"mapped_host\", "
                "\"payload_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"gpu_read_gb_s\": %.17g, \"gpu_write_gb_s\": %.17g, "
                "\"throughput_gb_s\": %.17g, \"integrity_pass\": true, "
                "\"numerical_pass\": true}}",
                options.payload_bytes,
                options.iterations,
                read_gb_s,
                write_gb_s,
                std::max(read_gb_s, write_gb_s));
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "mapped-host measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-COPY-001") == 0)
    {
        double d2d;
        double h2d;
        double d2h;
        SparkCudaProbeLatency d2d_latency;
        SparkCudaProbeLatency h2d_latency;
        SparkCudaProbeLatency d2h_latency;

        d2d = 0.0;
        h2d = 0.0;
        d2h = 0.0;
        d2d_latency = {};
        h2d_latency = {};
        d2h_latency = {};
        ok = SparkCudaProbeMeasureCopies(
            options.payload_bytes,
            options.iterations,
            &d2d,
            &h2d,
            &d2h,
            &d2d_latency,
            &h2d_latency,
            &d2h_latency);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output,
                "{\"parameters\": {\"candidate\": \"copy\", "
                "\"payload_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"d2d_gb_s\": %.17g, \"h2d_gb_s\": %.17g, "
                "\"d2h_gb_s\": %.17g, \"throughput_gb_s\": %.17g, "
                "\"d2d_latency_p99_ns\": %" PRIu64 ", "
                "\"h2d_latency_p99_ns\": %" PRIu64 ", "
                "\"d2h_latency_p99_ns\": %" PRIu64 ", "
                "\"integrity_pass\": true, \"numerical_pass\": true}}",
                options.payload_bytes,
                options.iterations,
                d2d,
                h2d,
                d2h,
                std::max(h2d, d2h),
                d2d_latency.p99_ns,
                h2d_latency.p99_ns,
                d2h_latency.p99_ns);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "copy measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-LAUNCH-001") == 0)
    {
        bool synchronize_each;
        SparkCudaProbeLatency latency;

        synchronize_each = std::strcmp(options.mode, "launch_sync") == 0;
        latency = {};
        ok = SparkCudaProbeMeasureLaunch(
            options.iterations,
            options.batch_size,
            options.kernel_count,
            synchronize_each,
            options.load_mode,
            properties,
            options.working_set_bytes,
            &latency);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output, "{\"parameters\": {\"candidate\": \"direct\", \"mode\": ");
            SparkCudaProbeWriteJsonString(output, options.mode);
            std::fputs(", \"load_mode\": ", output);
            SparkCudaProbeWriteJsonString(output, options.load_mode);
            std::fprintf(output,
                ", \"batch_size\": %u, \"kernel_count\": %u, "
                "\"working_set_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"latency_p50_ns\": %" PRIu64 ", "
                "\"latency_p95_ns\": %" PRIu64 ", \"latency_p99_ns\": %" PRIu64 ", "
                "\"integrity_pass\": true, \"numerical_pass\": true}}",
                options.batch_size,
                options.kernel_count,
                options.working_set_bytes,
                options.iterations,
                latency.p50_ns,
                latency.p95_ns,
                latency.p99_ns);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "launch measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-GRAPH-001") == 0)
    {
        SparkCudaProbeLatency latency;
        uint64_t instantiate_ns;
        char summary[160u];

        latency = {};
        instantiate_ns = 0u;
        ok = SparkCudaProbeMeasureGraph(
            options.candidate,
            options.batch_size,
            options.kernel_count,
            options.iterations,
            &latency,
            &instantiate_ns);
        std::snprintf(
            summary,
            sizeof(summary),
            "{\"graph_instantiate_ns\": %" PRIu64 "}",
            instantiate_ns);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", summary);
        if (ok)
        {
            std::fprintf(output, "{\"parameters\": {\"candidate\": ");
            SparkCudaProbeWriteJsonString(output, options.candidate);
            std::fprintf(output,
                ", \"mode\": ");
            SparkCudaProbeWriteJsonString(output, options.mode);
            std::fprintf(output,
                ", \"batch_size\": %u, \"kernel_count\": %u, \"iterations\": %u}, "
                "\"metrics\": {\"latency_p50_ns\": %" PRIu64 ", "
                "\"latency_p95_ns\": %" PRIu64 ", \"latency_p99_ns\": %" PRIu64 ", "
                "\"graph_instantiate_ns\": %" PRIu64 ", "
                "\"integrity_pass\": true, \"numerical_pass\": true}}",
                options.batch_size,
                options.kernel_count,
                options.iterations,
                latency.p50_ns,
                latency.p95_ns,
                latency.p99_ns,
                instantiate_ns);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "graph comparison failed");
    }
    else if (std::strcmp(options.question_id, "GB10-CALLBACK-001") == 0)
    {
        SparkCudaProbeLatency latency;

        latency = {};
        ok = SparkCudaProbeMeasureCompletionMechanism(
            options.candidate,
            options.load_mode,
            properties,
            options.iterations,
            options.working_set_bytes,
            &latency);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output, "{\"parameters\": {\"candidate\": ");
            SparkCudaProbeWriteJsonString(output, options.candidate);
            std::fputs(", \"load_mode\": ", output);
            SparkCudaProbeWriteJsonString(output, options.load_mode);
            std::fprintf(output,
                ", \"working_set_bytes\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"latency_p50_ns\": %" PRIu64 ", "
                "\"latency_p95_ns\": %" PRIu64 ", \"latency_p99_ns\": %" PRIu64 ", "
                "\"integrity_pass\": true, \"numerical_pass\": true}}",
                options.working_set_bytes,
                options.iterations,
                latency.p50_ns,
                latency.p95_ns,
                latency.p99_ns);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "completion measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-CONCURRENCY-001") == 0)
    {
        double ratio;
        double aggregate_throughput_gb_s;
        uint64_t serial_ns;
        uint64_t parallel_ns;

        ratio = 0.0;
        aggregate_throughput_gb_s = 0.0;
        serial_ns = 0u;
        parallel_ns = 0u;
        ok = SparkCudaProbeMeasureConcurrency(
            options.candidate,
            properties,
            options.working_set_bytes,
            options.stream_count,
            options.iterations,
            &ratio,
            &aggregate_throughput_gb_s,
            &serial_ns,
            &parallel_ns);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output, "{\"parameters\": {\"candidate\": ");
            SparkCudaProbeWriteJsonString(output, options.candidate);
            std::fprintf(output,
                ", \"stream_count\": %u, \"working_set_bytes\": %" PRIu64 ", "
                "\"iterations\": %u}, \"metrics\": {\"overlap_ratio\": %.17g, "
                "\"aggregate_throughput_gb_s\": %.17g, \"serial_ns\": %" PRIu64 ", "
                "\"parallel_ns\": %" PRIu64 ", \"integrity_pass\": true, "
                "\"numerical_pass\": true}}",
                options.stream_count,
                options.working_set_bytes,
                options.iterations,
                ratio,
                aggregate_throughput_gb_s,
                serial_ns,
                parallel_ns);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "concurrency measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-SMEM-001") == 0)
    {
        cudaError_t attribute_status;
        int active_blocks;
        cudaError_t occupancy_status;
        cudaError_t launch_status;
        cudaError_t synchronize_status;
        bool launch_succeeded;
        double occupancy;

        attribute_status = cudaSuccess;
        if (options.dynamic_shared_bytes > 0u)
        {
            attribute_status = cudaFuncSetAttribute(
                SparkCudaProbeDynamicSharedKernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(options.dynamic_shared_bytes));
        }
        active_blocks = 0;
        occupancy_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            SparkCudaProbeDynamicSharedKernel,
            SPARK_CUDA_PROBE_BLOCK_THREADS,
            options.dynamic_shared_bytes);
        SparkCudaProbeDynamicSharedKernel<<<1, SPARK_CUDA_PROBE_BLOCK_THREADS,
            options.dynamic_shared_bytes>>>(options.dynamic_shared_bytes);
        launch_status = cudaGetLastError();
        synchronize_status = launch_status == cudaSuccess ? cudaDeviceSynchronize() : launch_status;
        launch_succeeded = attribute_status == cudaSuccess &&
            occupancy_status == cudaSuccess && launch_status == cudaSuccess &&
            synchronize_status == cudaSuccess;
        occupancy = properties.maxThreadsPerMultiProcessor == 0 ? 0.0 :
            static_cast<double>(active_blocks * SPARK_CUDA_PROBE_BLOCK_THREADS) /
                properties.maxThreadsPerMultiProcessor;
        ok = true;
        SparkCudaProbeWriteReceiptPrefix(output, options, "measured", "{}");
        std::fprintf(output,
            "{\"parameters\": {\"candidate\": \"dynamic_shared\", "
            "\"dynamic_shared_bytes\": %u, \"iterations\": %u}, "
            "\"metrics\": {"
            "\"launch_succeeded\": %s, \"active_blocks_per_sm\": %d, "
            "\"theoretical_occupancy\": %.17g, "
            "\"attribute_status\": %d, \"occupancy_status\": %d, "
            "\"launch_status\": %d, \"synchronize_status\": %d, "
            "\"integrity_pass\": true, \"numerical_pass\": true}}",
            options.dynamic_shared_bytes,
            options.iterations,
            launch_succeeded ? "true" : "false",
            launch_succeeded ? active_blocks : 0,
            launch_succeeded ? occupancy : 0.0,
            static_cast<int>(attribute_status),
            static_cast<int>(occupancy_status),
            static_cast<int>(launch_status),
            static_cast<int>(synchronize_status));
        (void)cudaGetLastError();
        SparkCudaProbeWriteReceiptSuffix(output, nullptr);
    }
    else if (std::strcmp(options.question_id, "GB10-ATOMIC-001") == 0)
    {
        bool distributed;
        double operations_per_second;

        distributed = std::strcmp(options.candidate, "distributed") == 0;
        operations_per_second = 0.0;
        ok = SparkCudaProbeMeasureAtomic(
            properties,
            distributed,
            options.operations,
            options.iterations,
            &operations_per_second);
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output, "{\"parameters\": {\"candidate\": ");
            SparkCudaProbeWriteJsonString(output, options.candidate);
            std::fprintf(output,
                ", \"mode\": ");
            SparkCudaProbeWriteJsonString(output, options.mode);
            std::fprintf(output,
                ", \"operations\": %" PRIu64 ", \"iterations\": %u}, "
                "\"metrics\": {\"operations_per_second\": %.17g, "
                "\"giga_ops\": %.17g, \"integrity_pass\": true, "
                "\"numerical_pass\": true}}",
                options.operations,
                options.iterations,
                operations_per_second,
                operations_per_second / 1000000000.0);
        }
        SparkCudaProbeWriteReceiptSuffix(output, ok ? nullptr : "atomic measurement failed");
    }
    else if (std::strcmp(options.question_id, "GB10-THERMAL-001") == 0)
    {
        SparkCudaProbeTelemetry telemetry[3];
        double bandwidth[3];
        uint32_t cuda_error_count;
        uint64_t sample_count;
        double actual_seconds;
        double steady_ratio;
        double end_ratio;

        telemetry[0] = {};
        telemetry[1] = {};
        telemetry[2] = {};
        bandwidth[0] = 0.0;
        bandwidth[1] = 0.0;
        bandwidth[2] = 0.0;
        cuda_error_count = 0u;
        sample_count = 0u;
        actual_seconds = 0.0;
        ok = SparkCudaProbeMeasureThermal(
            properties,
            options.working_set_bytes,
            options.sustained_seconds,
            telemetry,
            bandwidth,
            &cuda_error_count,
            &sample_count,
            &actual_seconds);
        steady_ratio = bandwidth[0] == 0.0 ? 0.0 : bandwidth[1] / bandwidth[0];
        end_ratio = bandwidth[0] == 0.0 ? 0.0 : bandwidth[2] / bandwidth[0];
        SparkCudaProbeWriteReceiptPrefix(output, options, ok ? "measured" : "failed", "{}");
        if (ok)
        {
            std::fprintf(output,
                "{\"parameters\": {\"candidate\": \"sustained_memory_copy\", "
                "\"sample_phase\": \"all\", \"working_set_bytes\": %" PRIu64 ", "
                "\"sustained_seconds\": %u, \"iterations\": %u}, "
                "\"metrics\": {"
                "\"start_copy_gb_s\": %.17g, \"steady_copy_gb_s\": %.17g, "
                "\"end_copy_gb_s\": %.17g, \"steady_state_throughput_ratio\": %.17g, "
                "\"end_throughput_ratio\": %.17g, "
                "\"start_temperature_c\": %.17g, \"steady_temperature_c\": %.17g, "
                "\"end_temperature_c\": %.17g, \"start_sm_clock_mhz\": %.17g, "
                "\"steady_sm_clock_mhz\": %.17g, \"end_sm_clock_mhz\": %.17g, "
                "\"start_memory_clock_mhz\": %.17g, \"steady_memory_clock_mhz\": %.17g, "
                "\"end_memory_clock_mhz\": %.17g, \"start_power_w\": %.17g, "
                "\"steady_power_w\": %.17g, \"end_power_w\": %.17g, "
                "\"sample_count\": %" PRIu64 ", \"actual_seconds\": %.17g, "
                "\"cuda_error_count\": %u, \"integrity_pass\": %s, "
                "\"numerical_pass\": %s}}",
                options.working_set_bytes,
                options.sustained_seconds,
                options.iterations,
                bandwidth[0],
                bandwidth[1],
                bandwidth[2],
                steady_ratio,
                end_ratio,
                telemetry[0].temperature_c,
                telemetry[1].temperature_c,
                telemetry[2].temperature_c,
                telemetry[0].sm_clock_mhz,
                telemetry[1].sm_clock_mhz,
                telemetry[2].sm_clock_mhz,
                telemetry[0].memory_clock_mhz,
                telemetry[1].memory_clock_mhz,
                telemetry[2].memory_clock_mhz,
                telemetry[0].power_w,
                telemetry[1].power_w,
                telemetry[2].power_w,
                sample_count,
                actual_seconds,
                cuda_error_count,
                cuda_error_count == 0u ? "true" : "false",
                cuda_error_count == 0u ? "true" : "false");
        }
        SparkCudaProbeWriteReceiptSuffix(
            output,
            ok ? nullptr : "thermal test requires working nvidia-smi telemetry and sustained CUDA execution");
    }
    else
    {
        SparkCudaProbeWriteReceiptPrefix(output, options, "failed", "{}");
        SparkCudaProbeWriteReceiptSuffix(output, "unknown question ID");
        ok = false;
    }
    if (output != stdout)
    {
        std::fclose(output);
    }
    return ok;
}

static bool SparkCudaProbeOptionsAreValidForQuestion(
    const SparkCudaProbeOptions &options);

static void SparkCudaProbeUsage(const char *program_name)
{
    std::fprintf(stderr,
        "usage: %s --question ID [--candidate ID] [--mode ID] [--load-mode ID] "
        "[--sample-phase start|steady|end|all] [--working-set-bytes N] [--payload-bytes N] "
        "[--dynamic-shared-bytes N] [--batch-size N] [--kernel-count N] [--stream-count N] "
        "[--operations N] [--iterations N] [--sustained-seconds N] "
        "--source-package-sha256 HASH --run-id ID --topology NAME --node NAME "
        "[--output FILE]\n",
        program_name);
}

static bool SparkCudaProbeParseOptions(
    int argument_count,
    char **arguments,
    SparkCudaProbeOptions *options)
{
    std::memset(options, 0, sizeof(*options));
    options->candidate = "none";
    options->mode = "none";
    options->load_mode = "idle";
    options->sample_phase = "all";
    options->working_set_bytes = 256ull * 1024ull * 1024ull;
    options->payload_bytes = 64ull * 1024ull;
    options->batch_size = 1u;
    options->kernel_count = 1u;
    options->stream_count = 2u;
    options->operations = 16ull * 1024ull * 1024ull;
    options->iterations = 20u;
    options->sustained_seconds = SPARK_CUDA_PROBE_DEFAULT_THERMAL_SECONDS;
    for (int index = 1; index < argument_count; ++index)
    {
        const char *argument = arguments[index];
#define SPARK_PARSE_STRING(NAME, FIELD) \
        if (std::strcmp(argument, NAME) == 0 && index + 1 < argument_count) \
        { \
            options->FIELD = arguments[++index]; \
        }
        SPARK_PARSE_STRING("--question", question_id)
        else SPARK_PARSE_STRING("--candidate", candidate)
        else SPARK_PARSE_STRING("--mode", mode)
        else SPARK_PARSE_STRING("--load-mode", load_mode)
        else SPARK_PARSE_STRING("--sample-phase", sample_phase)
        else SPARK_PARSE_STRING("--source-package-sha256", source_package_sha256)
        else SPARK_PARSE_STRING("--run-id", run_id)
        else SPARK_PARSE_STRING("--topology", topology)
        else SPARK_PARSE_STRING("--node", node)
        else SPARK_PARSE_STRING("--output", output_path)
#undef SPARK_PARSE_STRING
        else if (std::strcmp(argument, "--working-set-bytes") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU64(arguments[++index], &options->working_set_bytes) ||
                options->working_set_bytes < 4096u ||
                options->working_set_bytes > SPARK_CUDA_PROBE_MAX_BYTES)
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--payload-bytes") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU64(arguments[++index], &options->payload_bytes) ||
                options->payload_bytes == 0u || options->payload_bytes > SPARK_CUDA_PROBE_MAX_BYTES)
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--dynamic-shared-bytes") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU32(arguments[++index], 0u, 1024u * 1024u,
                    &options->dynamic_shared_bytes))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--batch-size") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU32(arguments[++index], 1u, 1048576u,
                    &options->batch_size))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--kernel-count") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU32(arguments[++index], 1u, 4096u,
                    &options->kernel_count))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--stream-count") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU32(arguments[++index], 1u, 64u,
                    &options->stream_count))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--operations") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU64(arguments[++index], &options->operations) ||
                options->operations == 0u || options->operations > (1ull << 40u))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--iterations") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU32(arguments[++index], 1u,
                    SPARK_CUDA_PROBE_MAX_ITERATIONS, &options->iterations))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--sustained-seconds") == 0 && index + 1 < argument_count)
        {
            if (!SparkCudaProbeParseU32(arguments[++index], 3u, 7200u,
                    &options->sustained_seconds))
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    return options->question_id != nullptr &&
        SparkCudaProbeHexIsValid(options->source_package_sha256) &&
        options->run_id != nullptr && options->topology != nullptr && options->node != nullptr &&
        SparkCudaProbeOptionsAreValidForQuestion(*options);
}


static bool SparkCudaProbeTextEqualsOneOf(
    const char *value,
    std::initializer_list<const char *> allowed)
{
    for (const char *candidate : allowed)
    {
        if (std::strcmp(value, candidate) == 0)
        {
            return true;
        }
    }
    return false;
}

static bool SparkCudaProbeOptionsAreValidForQuestion(const SparkCudaProbeOptions &options)
{
    if (std::strcmp(options.question_id, "GB10-IDENTITY-001") == 0 ||
        std::strcmp(options.question_id, "GB10-MEM-001") == 0 ||
        std::strcmp(options.question_id, "GB10-MEM-002") == 0 ||
        std::strcmp(options.question_id, "GB10-MEM-003") == 0 ||
        std::strcmp(options.question_id, "GB10-MAPPED-001") == 0 ||
        std::strcmp(options.question_id, "GB10-COPY-001") == 0)
    {
        return true;
    }
    if (std::strcmp(options.question_id, "GB10-UMEM-001") == 0)
    {
        return SparkCudaProbeTextEqualsOneOf(options.candidate,
            {"gpu_only", "cpu_read_contention", "cpu_write_contention",
             "mapped_host_gpu_read", "mapped_host_gpu_write"});
    }
    if (std::strcmp(options.question_id, "GB10-LAUNCH-001") == 0)
    {
        return SparkCudaProbeTextEqualsOneOf(options.mode, {"enqueue", "launch_sync"}) &&
            SparkCudaProbeTextEqualsOneOf(options.load_mode, {"idle", "memory_loaded"});
    }
    if (std::strcmp(options.question_id, "GB10-GRAPH-001") == 0)
    {
        return SparkCudaProbeTextEqualsOneOf(options.candidate, {"direct", "graph"});
    }
    if (std::strcmp(options.question_id, "GB10-CALLBACK-001") == 0)
    {
        return SparkCudaProbeTextEqualsOneOf(options.candidate,
                {"event", "host_callback", "stream_sync"}) &&
            SparkCudaProbeTextEqualsOneOf(options.load_mode, {"idle", "memory_loaded"});
    }
    if (std::strcmp(options.question_id, "GB10-CONCURRENCY-001") == 0)
    {
        return SparkCudaProbeTextEqualsOneOf(options.candidate,
            {"copy_copy", "copy_compute", "compute_compute"});
    }
    if (std::strcmp(options.question_id, "GB10-SMEM-001") == 0)
    {
        return true;
    }
    if (std::strcmp(options.question_id, "GB10-ATOMIC-001") == 0)
    {
        return SparkCudaProbeTextEqualsOneOf(options.candidate, {"contended", "distributed"});
    }
    if (std::strcmp(options.question_id, "GB10-THERMAL-001") == 0)
    {
        return std::strcmp(options.candidate, "sustained_memory_copy") == 0 &&
            std::strcmp(options.sample_phase, "all") == 0;
    }
    return false;
}

int main(int argument_count, char **arguments)
{
    SparkCudaProbeOptions options{};
    if (!SparkCudaProbeParseOptions(argument_count, arguments, &options))
    {
        SparkCudaProbeUsage(arguments[0]);
        return 2;
    }
    if (!SparkCudaProbeCheck(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags"))
    {
        return 1;
    }
    int device = 0;
    int driver_version = 0;
    int runtime_version = 0;
    int memory_clock_rate = 0;
    cudaDeviceProp properties{};
    if (!SparkCudaProbeCheck(cudaGetDevice(&device), "cudaGetDevice") ||
        !SparkCudaProbeCheck(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties") ||
        !SparkCudaProbeCheck(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion") ||
        !SparkCudaProbeCheck(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion"))
    {
        return 1;
    }
    if (!SparkCudaProbeCheck(
            cudaDeviceGetAttribute(
                &memory_clock_rate,
                cudaDevAttrMemoryClockRate,
                device),
            "cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)"))
    {
        return 1;
    }
    return SparkCudaProbeRunQuestion(
        options,
        properties,
        driver_version,
        runtime_version,
        memory_clock_rate) ? 0 : 1;
}
