#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#if defined(__linux__)
#include <linux/aio_abi.h>
#else
typedef uint64_t aio_context_t;

struct iocb
{
    uint64_t aio_data;
    uint16_t aio_lio_opcode;
    uint32_t aio_fildes;
    uint64_t aio_buf;
    uint64_t aio_nbytes;
    int64_t aio_offset;
};

struct io_event
{
    uint64_t data;
    uint64_t obj;
    int64_t res;
    int64_t res2;
};

#define IOCB_CMD_PREAD 0u
#endif
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sys/stat.h>
#if !defined(O_DIRECT)
#define O_DIRECT 0
#endif
#include <sys/syscall.h>
#include <sys/types.h>

#include "spark_probe_common.h"

#define SPARK_NVME_PROBE_ALIGNMENT_BYTES 4096u
#define SPARK_NVME_PROBE_MAX_BLOCK_BYTES (64u * 1024u * 1024u)
#define SPARK_NVME_PROBE_MAX_QUEUE_DEPTH 256u
#define SPARK_NVME_PROBE_MAX_WORKERS 64u
#define SPARK_NVME_PROBE_MAX_ITERATIONS 1000000u

typedef struct SparkNvmeProbeOptions
{
    const char *question_id;
    const char *candidate;
    const char *file_path;
    uint64_t file_bytes;
    uint32_t prepare_file;
    uint32_t block_bytes;
    uint32_t queue_depth;
    uint32_t worker_count;
    uint32_t iterations;
    uint32_t cuda_device;
    const char *source_package_sha256;
    const char *run_id;
    const char *topology;
    const char *node;
    const char *output_path;
} SparkNvmeProbeOptions;

typedef struct SparkNvmeProbeSlot
{
    void *host_buffer;
    void *device_buffer;
    unsigned long long *device_fingerprint;
    cudaStream_t stream;
    struct iocb control_block;
    uint64_t byte_offset;
    uint64_t start_ns;
    uint32_t iteration;
    uint32_t host_registered;
    uint32_t in_flight;
} SparkNvmeProbeSlot;

typedef struct SparkNvmeProbeWorkerResult
{
    std::vector<uint64_t> latency_samples;
    uint64_t transferred_bytes;
    uint64_t cpu_fingerprint;
    uint64_t device_fingerprint;
    uint64_t integrity_error_count;
    uint64_t read_error_count;
    uint64_t verified_sample_count;
} SparkNvmeProbeWorkerResult;

typedef struct SparkNvmeProbeSharedState
{
    std::atomic<uint32_t> ready_count;
    std::atomic<uint32_t> start;
    std::atomic<uint32_t> stop;
    std::atomic<uint32_t> initialization_failed;
    std::atomic<uint32_t> execution_failed;
} SparkNvmeProbeSharedState;

typedef struct SparkNvmeProbeWorkerContext
{
    const SparkNvmeProbeOptions *options;
    SparkNvmeProbeSharedState *shared;
    SparkNvmeProbeWorkerResult *result;
    uint32_t worker_index;
} SparkNvmeProbeWorkerContext;

__device__ static unsigned long long SparkNvmeProbeDeviceMix64(unsigned long long value)
{
    value += 0x9e3779b97f4a7c15ull;
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return value;
}

__global__ static void SparkNvmeProbeFingerprintKernel(
    const unsigned long long *words,
    uint64_t word_count,
    uint64_t absolute_word_offset,
    unsigned long long *fingerprint)
{
    uint64_t index;
    uint64_t stride;
    unsigned long long local;

    index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
    stride = (uint64_t)gridDim.x * blockDim.x;
    local = 0ull;
    while (index < word_count)
    {
        local ^= SparkNvmeProbeDeviceMix64(words[index] ^
            (absolute_word_offset + index));
        index += stride;
    }
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u)
    {
        local ^= __shfl_down_sync(0xffffffffu, local, offset);
    }
    if ((threadIdx.x & 31u) == 0u)
    {
        atomicXor(fingerprint, local);
    }
}

static bool SparkNvmeProbeCudaCheck(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static bool SparkNvmeProbeAligned(uint64_t value)
{
    return (value & (SPARK_NVME_PROBE_ALIGNMENT_BYTES - 1u)) == 0u;
}

static bool SparkNvmeProbePrepareFile(const SparkNvmeProbeOptions &options)
{
    int descriptor;
    void *buffer;
    uint64_t offset;

    if (!SparkNvmeProbeAligned(options.file_bytes) ||
        !SparkNvmeProbeAligned(options.block_bytes) ||
        options.file_bytes < options.block_bytes)
    {
        return false;
    }
    descriptor = open(options.file_path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
    if (descriptor < 0)
    {
        std::perror("open prepare file");
        return false;
    }
    buffer = nullptr;
    if (posix_memalign(&buffer, SPARK_NVME_PROBE_ALIGNMENT_BYTES,
            options.block_bytes) != 0)
    {
        close(descriptor);
        return false;
    }
    for (offset = 0u; offset < options.file_bytes; offset += options.block_bytes)
    {
        uint64_t *words;
        uint64_t word_count;
        uint64_t word_index;
        ssize_t written;

        words = static_cast<uint64_t *>(buffer);
        word_count = options.block_bytes / sizeof(uint64_t);
        for (word_index = 0u; word_index < word_count; ++word_index)
        {
            words[word_index] = SparkProbePatternWord((offset / sizeof(uint64_t)) + word_index);
        }
        written = pwrite(descriptor, buffer, options.block_bytes, (off_t)offset);
        if (written != (ssize_t)options.block_bytes)
        {
            std::perror("pwrite prepare file");
            free(buffer);
            close(descriptor);
            return false;
        }
    }
    if (fsync(descriptor) != 0)
    {
        std::perror("fsync prepare file");
        free(buffer);
        close(descriptor);
        return false;
    }
    free(buffer);
    close(descriptor);
    return true;
}

static bool SparkNvmeProbeValidateFile(const SparkNvmeProbeOptions &options)
{
    struct stat status;

    if (stat(options.file_path, &status) != 0)
    {
        std::perror("stat NVMe file");
        return false;
    }
    if (!S_ISREG(status.st_mode) || (uint64_t)status.st_size < options.file_bytes)
    {
        std::fprintf(stderr, "NVMe file is not a sufficiently large regular file\n");
        return false;
    }
    return true;
}

static void SparkNvmeProbeDestroySlot(SparkNvmeProbeSlot *slot)
{
    if (slot->stream != nullptr)
    {
        (void)cudaStreamDestroy(slot->stream);
    }
    if (slot->device_fingerprint != nullptr)
    {
        (void)cudaFree(slot->device_fingerprint);
    }
    if (slot->device_buffer != nullptr)
    {
        (void)cudaFree(slot->device_buffer);
    }
    if (slot->host_buffer != nullptr)
    {
        if (slot->host_registered != 0u)
        {
            (void)cudaHostUnregister(slot->host_buffer);
        }
        free(slot->host_buffer);
    }
    std::memset(slot, 0, sizeof(*slot));
}

static bool SparkNvmeProbeCreateSlot(
    const SparkNvmeProbeOptions &options,
    SparkNvmeProbeSlot *slot)
{
    std::memset(slot, 0, sizeof(*slot));
    if (posix_memalign(&slot->host_buffer, SPARK_NVME_PROBE_ALIGNMENT_BYTES,
            options.block_bytes) != 0)
    {
        return false;
    }
    if (std::strcmp(options.question_id, "NVME-GPU-001") == 0)
    {
        if (!SparkNvmeProbeCudaCheck(cudaHostRegister(slot->host_buffer,
                options.block_bytes, cudaHostRegisterPortable), "cudaHostRegister"))
        {
            SparkNvmeProbeDestroySlot(slot);
            return false;
        }
        slot->host_registered = 1u;
        if (!SparkNvmeProbeCudaCheck(cudaMalloc(&slot->device_buffer,
                options.block_bytes), "cudaMalloc device buffer") ||
            !SparkNvmeProbeCudaCheck(cudaMalloc(&slot->device_fingerprint,
                sizeof(*slot->device_fingerprint)), "cudaMalloc fingerprint") ||
            !SparkNvmeProbeCudaCheck(cudaStreamCreateWithFlags(&slot->stream,
                cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
        {
            SparkNvmeProbeDestroySlot(slot);
            return false;
        }
    }
    return true;
}

static bool SparkNvmeProbeVerifyPattern(
    const void *buffer,
    uint32_t bytes,
    uint64_t absolute_byte_offset,
    uint64_t *fingerprint_out)
{
    const uint64_t *words;
    uint64_t word_count;
    uint64_t word_index;
    uint64_t absolute_word_offset;

    words = static_cast<const uint64_t *>(buffer);
    word_count = bytes / sizeof(uint64_t);
    absolute_word_offset = absolute_byte_offset / sizeof(uint64_t);
    for (word_index = 0u; word_index < word_count; ++word_index)
    {
        if (words[word_index] != SparkProbePatternWord(absolute_word_offset + word_index))
        {
            return false;
        }
    }
    *fingerprint_out = SparkProbeFingerprintWords(words, word_count, absolute_word_offset);
    return true;
}

static bool SparkNvmeProbeCopyAndFingerprint(
    const SparkNvmeProbeOptions &options,
    SparkNvmeProbeSlot *slot,
    uint64_t absolute_byte_offset,
    uint64_t expected_fingerprint,
    uint64_t *device_fingerprint_out)
{
    unsigned long long device_fingerprint;
    uint64_t initial_fingerprint;
    uint32_t blocks;
    uint64_t word_count;

    word_count = options.block_bytes / sizeof(uint64_t);
    blocks = (uint32_t)std::min<uint64_t>(65535u,
        std::max<uint64_t>(1u, (word_count + 255u) / 256u));
    device_fingerprint = 0ull;
    initial_fingerprint = SparkProbeMix64(
        word_count ^ (absolute_byte_offset / sizeof(uint64_t)));
    if (!SparkNvmeProbeCudaCheck(cudaMemcpyAsync(slot->device_fingerprint,
            &initial_fingerprint, sizeof(initial_fingerprint),
            cudaMemcpyHostToDevice, slot->stream), "cudaMemcpyAsync fingerprint") ||
        !SparkNvmeProbeCudaCheck(cudaMemcpyAsync(slot->device_buffer,
            slot->host_buffer, options.block_bytes, cudaMemcpyHostToDevice,
            slot->stream), "cudaMemcpyAsync H2D"))
    {
        return false;
    }
    SparkNvmeProbeFingerprintKernel<<<blocks, 256u, 0u, slot->stream>>>(
        static_cast<const unsigned long long *>(slot->device_buffer),
        word_count,
        absolute_byte_offset / sizeof(uint64_t),
        slot->device_fingerprint);
    if (!SparkNvmeProbeCudaCheck(cudaGetLastError(), "fingerprint kernel launch") ||
        !SparkNvmeProbeCudaCheck(cudaMemcpyAsync(&device_fingerprint,
            slot->device_fingerprint, sizeof(device_fingerprint),
            cudaMemcpyDeviceToHost, slot->stream), "cudaMemcpyAsync fingerprint D2H") ||
        !SparkNvmeProbeCudaCheck(cudaStreamSynchronize(slot->stream),
            "cudaStreamSynchronize NVMe pipeline"))
    {
        return false;
    }
    *device_fingerprint_out = (uint64_t)device_fingerprint;
    return (uint64_t)device_fingerprint == expected_fingerprint;
}


static int SparkNvmeProbeAioSetup(uint32_t queue_depth, aio_context_t *context)
{
#if defined(__linux__)
    *context = 0u;
    return (int)syscall(SYS_io_setup, queue_depth, context);
#else
    (void)queue_depth;
    *context = 0u;
    return -1;
#endif
}

static int SparkNvmeProbeAioSubmit(aio_context_t context, struct iocb *control_block)
{
#if defined(__linux__)
    struct iocb *control_blocks[1];

    control_blocks[0] = control_block;
    return (int)syscall(SYS_io_submit, context, 1L, control_blocks);
#else
    (void)context;
    (void)control_block;
    return -1;
#endif
}

static int SparkNvmeProbeAioGetEvents(
    aio_context_t context,
    long minimum_count,
    long maximum_count,
    struct io_event *events)
{
#if defined(__linux__)
    return (int)syscall(SYS_io_getevents, context, minimum_count, maximum_count,
        events, (struct timespec *)0);
#else
    (void)context;
    (void)minimum_count;
    (void)maximum_count;
    (void)events;
    return -1;
#endif
}

static int SparkNvmeProbeAioDestroy(aio_context_t context)
{
#if defined(__linux__)
    return (int)syscall(SYS_io_destroy, context);
#else
    (void)context;
    return 0;
#endif
}

static bool SparkNvmeProbeSubmitRead(
    const SparkNvmeProbeOptions &options,
    int descriptor,
    aio_context_t aio_context,
    uint32_t worker_index,
    uint32_t iteration,
    SparkNvmeProbeSlot *slot)
{
    uint64_t block_count;
    uint64_t block_index;

    block_count = options.file_bytes / options.block_bytes;
    block_index = ((uint64_t)iteration * options.worker_count + worker_index) % block_count;
    slot->byte_offset = block_index * options.block_bytes;
    slot->iteration = iteration;
    slot->start_ns = SparkProbeMonotonicNanoseconds();
    std::memset(&slot->control_block, 0, sizeof(slot->control_block));
    slot->control_block.aio_data = (uint64_t)(uintptr_t)slot;
    slot->control_block.aio_lio_opcode = IOCB_CMD_PREAD;
    slot->control_block.aio_fildes = (uint32_t)descriptor;
    slot->control_block.aio_buf = (uint64_t)(uintptr_t)slot->host_buffer;
    slot->control_block.aio_nbytes = options.block_bytes;
    slot->control_block.aio_offset = (int64_t)slot->byte_offset;
    if (SparkNvmeProbeAioSubmit(aio_context, &slot->control_block) != 1)
    {
        std::perror("io_submit");
        return false;
    }
    slot->in_flight = 1u;
    return true;
}

static bool SparkNvmeProbeCompleteRead(
    const SparkNvmeProbeOptions &options,
    uint32_t worker_index,
    const struct io_event &event,
    SparkNvmeProbeWorkerResult *result)
{
    SparkNvmeProbeSlot *slot;
    uint64_t finish_ns;
    uint64_t cpu_fingerprint;
    uint64_t device_fingerprint;
    uint64_t sample_identity;

    slot = (SparkNvmeProbeSlot *)(uintptr_t)event.data;
    if (slot == nullptr || slot->in_flight == 0u ||
        event.obj != (uint64_t)(uintptr_t)&slot->control_block)
    {
        result->read_error_count += 1u;
        return false;
    }
    slot->in_flight = 0u;
    if ((int64_t)event.res != (int64_t)options.block_bytes || (int64_t)event.res2 != 0)
    {
        result->read_error_count += 1u;
        return false;
    }
    if (!SparkNvmeProbeVerifyPattern(slot->host_buffer,
            options.block_bytes, slot->byte_offset, &cpu_fingerprint))
    {
        result->integrity_error_count += 1u;
        return false;
    }
    device_fingerprint = cpu_fingerprint;
    if (std::strcmp(options.question_id, "NVME-GPU-001") == 0 &&
        !SparkNvmeProbeCopyAndFingerprint(options, slot,
            slot->byte_offset, cpu_fingerprint, &device_fingerprint))
    {
        result->integrity_error_count += 1u;
        return false;
    }
    finish_ns = SparkProbeMonotonicNanoseconds();
    if (finish_ns <= slot->start_ns)
    {
        result->read_error_count += 1u;
        return false;
    }
    result->latency_samples.push_back(finish_ns - slot->start_ns);
    result->transferred_bytes += options.block_bytes;
    sample_identity = SparkProbeMix64(
        slot->byte_offset ^
        ((uint64_t)slot->iteration << 32u) ^
        ((uint64_t)worker_index << 56u));
    result->cpu_fingerprint ^= SparkProbeMix64(cpu_fingerprint ^ sample_identity);
    result->device_fingerprint ^= SparkProbeMix64(device_fingerprint ^ sample_identity);
    result->verified_sample_count += 1u;
    return true;
}

static void SparkNvmeProbeWorker(SparkNvmeProbeWorkerContext context)
{
    const SparkNvmeProbeOptions &options = *context.options;
    SparkNvmeProbeWorkerResult &result = *context.result;
    std::vector<SparkNvmeProbeSlot> slots;
    std::vector<struct io_event> events;
    aio_context_t aio_context;
    int descriptor;
    uint32_t initialized_slots;
    uint32_t submitted_iterations;
    uint32_t completed_iterations;

    aio_context = 0u;
    descriptor = -1;
    initialized_slots = 0u;
    submitted_iterations = 0u;
    completed_iterations = 0u;
    try
    {
        slots.resize(options.queue_depth);
        events.resize(options.queue_depth);
        result.latency_samples.reserve(options.iterations);
    }
    catch (const std::bad_alloc &)
    {
        context.shared->initialization_failed.store(1u, std::memory_order_release);
        context.shared->ready_count.fetch_add(1u, std::memory_order_acq_rel);
        return;
    }
    if (std::strcmp(options.question_id, "NVME-GPU-001") == 0 &&
        !SparkNvmeProbeCudaCheck(cudaSetDevice((int)options.cuda_device), "cudaSetDevice worker"))
    {
        context.shared->initialization_failed.store(1u, std::memory_order_release);
    }
    descriptor = open(options.file_path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (descriptor < 0)
    {
        std::perror("open O_DIRECT NVMe file");
        context.shared->initialization_failed.store(1u, std::memory_order_release);
    }
    for (initialized_slots = 0u;
         initialized_slots < options.queue_depth &&
             context.shared->initialization_failed.load(std::memory_order_acquire) == 0u;
         ++initialized_slots)
    {
        if (!SparkNvmeProbeCreateSlot(options, &slots[initialized_slots]))
        {
            context.shared->initialization_failed.store(1u, std::memory_order_release);
            break;
        }
    }
    if (context.shared->initialization_failed.load(std::memory_order_acquire) == 0u &&
        SparkNvmeProbeAioSetup(options.queue_depth, &aio_context) != 0)
    {
        std::perror("io_setup");
        context.shared->initialization_failed.store(1u, std::memory_order_release);
    }
    context.shared->ready_count.fetch_add(1u, std::memory_order_acq_rel);
    while (context.shared->start.load(std::memory_order_acquire) == 0u &&
           context.shared->stop.load(std::memory_order_acquire) == 0u)
    {
        std::this_thread::yield();
    }
    if (context.shared->stop.load(std::memory_order_acquire) == 0u &&
        context.shared->initialization_failed.load(std::memory_order_acquire) == 0u)
    {
        while (submitted_iterations < options.iterations &&
               submitted_iterations < options.queue_depth)
        {
            if (!SparkNvmeProbeSubmitRead(options, descriptor, aio_context,
                    context.worker_index, submitted_iterations,
                    &slots[submitted_iterations]))
            {
                context.shared->execution_failed.store(1u, std::memory_order_release);
                break;
            }
            submitted_iterations += 1u;
        }
        while (completed_iterations < options.iterations &&
               context.shared->stop.load(std::memory_order_acquire) == 0u &&
               context.shared->execution_failed.load(std::memory_order_acquire) == 0u)
        {
            int event_count;

            event_count = SparkNvmeProbeAioGetEvents(
                aio_context, 1L, (long)options.queue_depth, events.data());
            if (event_count <= 0)
            {
                if (event_count < 0)
                {
                    std::perror("io_getevents");
                }
                context.shared->execution_failed.store(1u, std::memory_order_release);
                break;
            }
            for (int event_index = 0; event_index < event_count; ++event_index)
            {
                SparkNvmeProbeSlot *slot;

                slot = (SparkNvmeProbeSlot *)(uintptr_t)events[event_index].data;
                if (!SparkNvmeProbeCompleteRead(options, context.worker_index,
                        events[event_index], &result))
                {
                    context.shared->execution_failed.store(1u, std::memory_order_release);
                    break;
                }
                completed_iterations += 1u;
                if (submitted_iterations < options.iterations)
                {
                    if (!SparkNvmeProbeSubmitRead(options, descriptor, aio_context,
                            context.worker_index, submitted_iterations, slot))
                    {
                        context.shared->execution_failed.store(1u, std::memory_order_release);
                        break;
                    }
                    submitted_iterations += 1u;
                }
            }
        }
    }
    if (context.shared->execution_failed.load(std::memory_order_acquire) != 0u)
    {
        context.shared->stop.store(1u, std::memory_order_release);
    }
    if (aio_context != 0u && SparkNvmeProbeAioDestroy(aio_context) != 0)
    {
        std::perror("io_destroy");
        context.shared->execution_failed.store(1u, std::memory_order_release);
    }
    for (uint32_t slot_index = 0u; slot_index < initialized_slots; ++slot_index)
    {
        SparkNvmeProbeDestroySlot(&slots[slot_index]);
    }
    if (descriptor >= 0)
    {
        close(descriptor);
    }
}

static bool SparkNvmeProbeRunWorkers(
    const SparkNvmeProbeOptions &options,
    std::vector<SparkNvmeProbeWorkerResult> *results_out,
    uint64_t *elapsed_ns_out)
{
    SparkNvmeProbeSharedState shared{};
    std::vector<SparkNvmeProbeWorkerContext> contexts;
    std::vector<std::thread> threads;
    uint32_t created;
    uint64_t start_ns;
    uint64_t finish_ns;

    results_out->resize(options.worker_count);
    contexts.resize(options.worker_count);
    threads.reserve(options.worker_count);
    created = 0u;
    try
    {
        for (uint32_t worker = 0u; worker < options.worker_count; ++worker)
        {
            contexts[worker].options = &options;
            contexts[worker].shared = &shared;
            contexts[worker].result = &(*results_out)[worker];
            contexts[worker].worker_index = worker;
            threads.emplace_back(SparkNvmeProbeWorker, contexts[worker]);
            created += 1u;
        }
    }
    catch (const std::system_error &error)
    {
        std::fprintf(stderr, "worker creation failed after %u workers: %s\n",
            created, error.what());
        shared.stop.store(1u, std::memory_order_release);
        shared.start.store(1u, std::memory_order_release);
        for (std::thread &thread : threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        return false;
    }
    while (shared.ready_count.load(std::memory_order_acquire) < options.worker_count &&
           shared.initialization_failed.load(std::memory_order_acquire) == 0u)
    {
        std::this_thread::yield();
    }
    if (shared.initialization_failed.load(std::memory_order_acquire) != 0u)
    {
        shared.stop.store(1u, std::memory_order_release);
    }
    start_ns = SparkProbeMonotonicNanoseconds();
    shared.start.store(1u, std::memory_order_release);
    for (std::thread &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    finish_ns = SparkProbeMonotonicNanoseconds();
    *elapsed_ns_out = finish_ns - start_ns;
    return shared.initialization_failed.load(std::memory_order_acquire) == 0u &&
        shared.execution_failed.load(std::memory_order_acquire) == 0u;
}

static bool SparkNvmeProbeWriteReceipt(
    const SparkNvmeProbeOptions &options,
    const std::vector<SparkNvmeProbeWorkerResult> &results,
    uint64_t elapsed_ns)
{
    std::vector<uint64_t> latencies;
    SparkProbeLatencySummary summary;
    uint64_t transferred_bytes;
    uint64_t cpu_fingerprint;
    uint64_t device_fingerprint;
    uint64_t integrity_errors;
    uint64_t read_errors;
    uint64_t verified_samples;
    double throughput_gb_s;
    FILE *output;

    transferred_bytes = 0u;
    cpu_fingerprint = 0u;
    device_fingerprint = 0u;
    integrity_errors = 0u;
    read_errors = 0u;
    verified_samples = 0u;
    for (const SparkNvmeProbeWorkerResult &result : results)
    {
        latencies.insert(latencies.end(), result.latency_samples.begin(),
            result.latency_samples.end());
        transferred_bytes += result.transferred_bytes;
        cpu_fingerprint ^= result.cpu_fingerprint;
        device_fingerprint ^= result.device_fingerprint;
        integrity_errors += result.integrity_error_count;
        read_errors += result.read_error_count;
        verified_samples += result.verified_sample_count;
    }
    if (latencies.empty() || elapsed_ns == 0u ||
        integrity_errors != 0u || read_errors != 0u ||
        verified_samples != latencies.size() ||
        verified_samples != (uint64_t)options.worker_count * options.iterations ||
        cpu_fingerprint != device_fingerprint)
    {
        return false;
    }
    summary = SparkProbeSummarizeLatency(latencies.data(), latencies.size());
    throughput_gb_s = (double)transferred_bytes / (double)elapsed_ns;
    output = options.output_path == nullptr ? stdout : std::fopen(options.output_path, "wb");
    if (output == nullptr)
    {
        return false;
    }
    std::fputs("{\n  \"schema_version\": 1,\n  \"receipt_kind\": \"spark_hardware_probe\",\n  \"run_id\": ", output);
    SparkProbeWriteJsonString(output, options.run_id);
    std::fputs(",\n  \"probe_id\": \"nvme_characterize\",\n  \"source_identity\": {\"source_package_sha256\": ", output);
    SparkProbeWriteJsonString(output, options.source_package_sha256);
    std::fputs("},\n  \"scope\": {\"topology\": ", output);
    SparkProbeWriteJsonString(output, options.topology);
    std::fputs(", \"node\": ", output);
    SparkProbeWriteJsonString(output, options.node);
    std::fputs("},\n  \"answers\": [\n    {\"question_id\": ", output);
    SparkProbeWriteJsonString(output, options.question_id);
    std::fputs(", \"status\": \"measured\", \"summary\": {\"direct_io\": true, \"device_integrity_verified\": ", output);
    std::fputs(std::strcmp(options.question_id, "NVME-GPU-001") == 0 ? "true" : "false", output);
    std::fputs(", \"partial_worker_creation_unwind_safe\": true}, \"observations\": [{\"parameters\": {\"candidate\": ", output);
    SparkProbeWriteJsonString(output, options.candidate);
    std::fprintf(output,
        ", \"block_bytes\": %u, \"queue_depth\": %u, \"worker_count\": %u, "
        "\"iterations\": %u}, \"metrics\": {\"sample_count\": %zu, "
        "\"transferred_bytes\": %" PRIu64 ", \"throughput_gb_s\": %.17g, "
        "\"latency_min_ns\": %" PRIu64 ", \"latency_p50_ns\": %" PRIu64 ", "
        "\"latency_p95_ns\": %" PRIu64 ", \"latency_p99_ns\": %" PRIu64 ", "
        "\"latency_max_ns\": %" PRIu64 ", \"verified_sample_count\": %" PRIu64 ", "
        "\"cpu_fingerprint\": %" PRIu64 ", \"device_fingerprint\": %" PRIu64 ", "
        "\"integrity_error_count\": 0, "
        "\"read_error_count\": 0, \"integrity_pass\": true}}]}\n  ]\n}\n",
        options.block_bytes,
        options.queue_depth,
        options.worker_count,
        options.iterations,
        latencies.size(),
        transferred_bytes,
        throughput_gb_s,
        summary.minimum_ns,
        summary.p50_ns,
        summary.p95_ns,
        summary.p99_ns,
        summary.maximum_ns,
        verified_samples,
        cpu_fingerprint,
        device_fingerprint);
    if (output != stdout)
    {
        std::fclose(output);
    }
    return true;
}

static void SparkNvmeProbeUsage(const char *program_name)
{
    std::fprintf(stderr,
        "usage: %s --question NVME-RAW-001|NVME-GPU-001 --candidate ID "
        "--file PATH --file-bytes N [--prepare] --block-bytes N --queue-depth N "
        "--worker-count N --iterations N [--cuda-device N] "
        "--source-package-sha256 HASH --run-id ID --topology ID --node ID "
        "[--output FILE]\n",
        program_name);
}

static bool SparkNvmeProbeParseOptions(
    int argument_count,
    char **arguments,
    SparkNvmeProbeOptions *options)
{
    int index;

    std::memset(options, 0, sizeof(*options));
    options->cuda_device = 0u;
    for (index = 1; index < argument_count; ++index)
    {
        const char *argument = arguments[index];
        const char **text_destination = nullptr;

        if (std::strcmp(argument, "--question") == 0)
        {
            text_destination = &options->question_id;
        }
        else if (std::strcmp(argument, "--candidate") == 0)
        {
            text_destination = &options->candidate;
        }
        else if (std::strcmp(argument, "--file") == 0)
        {
            text_destination = &options->file_path;
        }
        else if (std::strcmp(argument, "--source-package-sha256") == 0)
        {
            text_destination = &options->source_package_sha256;
        }
        else if (std::strcmp(argument, "--run-id") == 0)
        {
            text_destination = &options->run_id;
        }
        else if (std::strcmp(argument, "--topology") == 0)
        {
            text_destination = &options->topology;
        }
        else if (std::strcmp(argument, "--node") == 0)
        {
            text_destination = &options->node;
        }
        else if (std::strcmp(argument, "--output") == 0)
        {
            text_destination = &options->output_path;
        }
        if (text_destination != nullptr)
        {
            if (index + 1 >= argument_count)
            {
                return false;
            }
            *text_destination = arguments[++index];
            continue;
        }
        if (std::strcmp(argument, "--prepare") == 0)
        {
            options->prepare_file = 1u;
        }
        else if (std::strcmp(argument, "--file-bytes") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU64(arguments[++index], SPARK_NVME_PROBE_ALIGNMENT_BYTES,
                    UINT64_MAX, &options->file_bytes))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--block-bytes") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], SPARK_NVME_PROBE_ALIGNMENT_BYTES,
                    SPARK_NVME_PROBE_MAX_BLOCK_BYTES, &options->block_bytes))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--queue-depth") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u,
                    SPARK_NVME_PROBE_MAX_QUEUE_DEPTH, &options->queue_depth))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--worker-count") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u,
                    SPARK_NVME_PROBE_MAX_WORKERS, &options->worker_count))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--iterations") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 1u,
                    SPARK_NVME_PROBE_MAX_ITERATIONS, &options->iterations))
            {
                return false;
            }
        }
        else if (std::strcmp(argument, "--cuda-device") == 0)
        {
            if (index + 1 >= argument_count ||
                !SparkProbeParseU32(arguments[++index], 0u, 1024u,
                    &options->cuda_device))
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    const char *question_id;

    question_id = options->question_id == nullptr ? "" : options->question_id;
    return ((std::strcmp(question_id, "NVME-RAW-001") == 0 &&
                std::strcmp(options->candidate == nullptr ? "" : options->candidate,
                    "direct_io") == 0) ||
            (std::strcmp(question_id, "NVME-GPU-001") == 0 &&
                std::strcmp(options->candidate == nullptr ? "" : options->candidate,
                    "nvme_to_gpu") == 0)) &&
        options->file_path != nullptr &&
        options->file_bytes >= options->block_bytes &&
        SparkNvmeProbeAligned(options->file_bytes) &&
        SparkNvmeProbeAligned(options->block_bytes) &&
        options->queue_depth != 0u && options->worker_count != 0u &&
        options->iterations != 0u &&
        SparkProbeHexSha256IsValid(options->source_package_sha256) &&
        SparkProbeIdentifierIsValid(options->run_id) &&
        SparkProbeIdentifierIsValid(options->topology) &&
        SparkProbeIdentifierIsValid(options->node);
}

int main(int argument_count, char **arguments)
{
    SparkNvmeProbeOptions options{};
    std::vector<SparkNvmeProbeWorkerResult> results;
    uint64_t elapsed_ns;

    if (!SparkNvmeProbeParseOptions(argument_count, arguments, &options))
    {
        SparkNvmeProbeUsage(arguments[0]);
        return 2;
    }
    if (options.prepare_file != 0u && !SparkNvmeProbePrepareFile(options))
    {
        return 1;
    }
    if (!SparkNvmeProbeValidateFile(options))
    {
        return 1;
    }
    if (std::strcmp(options.question_id, "NVME-GPU-001") == 0)
    {
        if (!SparkNvmeProbeCudaCheck(cudaSetDeviceFlags(cudaDeviceMapHost),
                "cudaSetDeviceFlags") ||
            !SparkNvmeProbeCudaCheck(cudaSetDevice((int)options.cuda_device),
                "cudaSetDevice"))
        {
            return 1;
        }
    }
    elapsed_ns = 0u;
    if (!SparkNvmeProbeRunWorkers(options, &results, &elapsed_ns) ||
        !SparkNvmeProbeWriteReceipt(options, results, elapsed_ns))
    {
        return 1;
    }
    return 0;
}
