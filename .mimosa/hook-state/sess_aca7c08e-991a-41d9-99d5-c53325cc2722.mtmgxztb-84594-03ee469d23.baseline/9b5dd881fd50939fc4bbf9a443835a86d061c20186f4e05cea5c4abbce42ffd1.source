
#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_weightd_attach.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

static void SparkWeightdAttachSetReason(char *reason, const char *text)
{
    if (reason != 0)
    {
        memset(reason, 0, SPARK_WEIGHTD_ATTACH_REASON_BYTES);
        memcpy(reason, text,
            strlen(text) + 1u <= SPARK_WEIGHTD_ATTACH_REASON_BYTES
                ? strlen(text) + 1u
                : SPARK_WEIGHTD_ATTACH_REASON_BYTES);
    }
}

static int SparkWeightdAttachEnvIsOff(const char *name)
{
    const char *text = getenv(name);
    return text != 0 && text[0] == '0' && text[1] == '\0';
}

static const char *SparkWeightdAttachEnvText(const char *name)
{
    const char *text = getenv(name);
    return text != 0 && text[0] != '\0' ? text : 0;
}

SparkStatus SparkWeightdAttachRequested(void)
{
    const char *socket = SparkWeightdAttachEnvText(
        SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0 || SparkWeightdAttachEnvIsOff(
        SPARK_WEIGHTD_ATTACH_ENV_SWITCH) != 0)
    {
        return SPARK_STATUS_BUSY;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdAttachPack(const SparkWeightdPackSlice *slice,
    const char *pack_path,
    uint64_t timeout_nanoseconds,
    SparkWeightdAttachOutcome *outcome,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES])
{
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    const char *socket;
    const char *digest;
    SparkStatus status;

    if (outcome == 0 || slice == 0 || pack_path == 0 ||
        pack_path[0] == '\0')
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(outcome, 0, sizeof(*outcome));
    if (reason != 0)
    {
        reason[0] = '\0';
    }

    if (SparkWeightdAttachEnvIsOff(SPARK_WEIGHTD_ATTACH_ENV_SWITCH) != 0)
    {
        SparkWeightdAttachSetReason(reason, "env_off");
        return SPARK_STATUS_OK;
    }
    socket = SparkWeightdAttachEnvText(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0)
    {
        SparkWeightdAttachSetReason(reason, "no_socket");
        return SPARK_STATUS_OK;
    }
    digest = SparkWeightdAttachEnvText(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
    if (digest == 0)
    {
        SparkWeightdAttachSetReason(reason, "no_identity");
        return SPARK_STATUS_OK;
    }
    memset(&identity, 0, sizeof(identity));
    {
        const char *model = SparkWeightdAttachEnvText(
            SPARK_WEIGHTD_ATTACH_ENV_MODEL);
        const char *revision = SparkWeightdAttachEnvText(
            SPARK_WEIGHTD_ATTACH_ENV_REVISION);
        model = model != 0 ? model : slice->model;
        revision = revision != 0 ? revision : slice->revision;
        revision = revision != 0 ? revision : "";
        if (model == 0)
        {
            SparkWeightdAttachSetReason(reason, "no_identity");
            return SPARK_STATUS_OK;
        }
        memcpy(identity.model, model,
            strlen(model) + 1u <= SPARK_WEIGHTD_ID_BYTES
                ? strlen(model) + 1u
                : SPARK_WEIGHTD_ID_BYTES);
        memcpy(identity.revision, revision,
            strlen(revision) + 1u <= SPARK_WEIGHTD_REVISION_BYTES
                ? strlen(revision) + 1u
                : SPARK_WEIGHTD_REVISION_BYTES);
    }
    memcpy(identity.pack_sha256, digest,
        strlen(digest) + 1u <= SPARK_WEIGHTD_SHA256_HEX_BYTES
            ? strlen(digest) + 1u
            : SPARK_WEIGHTD_SHA256_HEX_BYTES);
    identity.topology = slice->topology;
    identity.geometry_fingerprint = slice->geometry_fingerprint;
    identity.arena_bytes = slice->pack_bytes;
    identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    if (SparkWeightdIdentityPrepare(&identity) != SPARK_STATUS_OK)
    {
        SparkWeightdAttachSetReason(reason, "identity");
        return SPARK_STATUS_OK;
    }

    status = SparkWeightdClientConnect(socket, &outcome->client, 0);
    if (status != SPARK_STATUS_OK)
    {
        outcome->client = 0;
        SparkWeightdAttachSetReason(reason, "no_daemon");
        return SPARK_STATUS_OK;
    }
    memset(&request, 0, sizeof(request));
    request.identity = identity;
    memcpy(request.pack_path, pack_path,
        strlen(pack_path) + 1u <= SPARK_WEIGHTD_PATH_BYTES
            ? strlen(pack_path) + 1u
            : SPARK_WEIGHTD_PATH_BYTES);
    memset(&result, 0, sizeof(result));
    status = SparkWeightdClientAttach(outcome->client, &request, &result,
        timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, SparkStatusToString(status));
        return SPARK_STATUS_OK;
    }
    if (result.status != SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason,
            SparkStatusToString(result.status));
        return SPARK_STATUS_OK;
    }
    outcome->device_handle = result.device_handle;
    outcome->arena_bytes = result.arena_bytes;
    outcome->arena_generation = result.arena_generation;
    outcome->loaded_from_pack = result.loaded_from_pack;
    outcome->refcount = result.refcount;
    return SPARK_STATUS_OK;
}


static void SparkWeightdAttachCloseBatchFds(SparkWeightdExportBatch *batch)
{
    uint32_t index;
    for (index = 0u; index < batch->batch_count; index++)
    {
        (void)close(batch->fds[index]);
    }
    batch->batch_count = 0u;
}

static int SparkWeightdAttachDeviceId(void)
{
    int device = 0;
    (void)cudaFree(0);
    (void)cudaGetDevice(&device);
    return device;
}

static void SparkWeightdAttachMapUndo(SparkWeightdAttachOutcome *outcome)
{
    if (outcome->map_base != 0)
    {
        CUdeviceptr base = (CUdeviceptr)(uintptr_t)outcome->map_base;
        uint32_t index;
        for (index = 0u; index < outcome->map_mapped_count; index++)
        {
            (void)cuMemUnmap(base + (CUdeviceptr)index *
                    outcome->map_chunk_bytes,
                (size_t)outcome->map_chunk_bytes);
        }
        for (index = 0u; index < outcome->map_handle_count; index++)
        {
            (void)cuMemRelease(
                (CUmemGenericAllocationHandle)outcome->map_handles[index]);
        }
        (void)cuMemAddressFree(base, (size_t)outcome->map_span_bytes);
    }
    free(outcome->map_handles);
    outcome->map_handles = 0;
    outcome->map_base = 0;
    outcome->map_span_bytes = 0ull;
    outcome->map_chunk_bytes = 0ull;
    outcome->map_chunk_count = 0u;
    outcome->map_handle_count = 0u;
    outcome->map_mapped_count = 0u;
}

SparkStatus SparkWeightdAttachImportMap(SparkWeightdAttachOutcome *outcome,
    uint64_t expected_arena_bytes,
    uint64_t timeout_nanoseconds,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES])
{
    SparkWeightdExportBatch batch;
    uint64_t chunk_bytes = 0ull;
    uint64_t covered_bytes = 0ull;
    uint32_t chunk_count = 0u;
    uint32_t batch_offset = 0u;
    CUmemAccessDesc access;
    CUdeviceptr base = 0;

    if (outcome == 0 || reason == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    (void)SparkWeightdAttachDeviceId();
    if (outcome->client == 0)
    {
        SparkWeightdAttachSetReason(reason, "not_attached");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (outcome->map_base != 0)
    {
        SparkWeightdAttachSetReason(reason, "already_mapped");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&batch, 0, sizeof(batch));

    if (SparkWeightdClientExportBatch(outcome->client,
            outcome->arena_generation, 0u, &batch, timeout_nanoseconds) !=
        SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_exchange");
        return SPARK_STATUS_OK;
    }
    if (batch.status != SPARK_STATUS_OK)
    {
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, SparkStatusToString(batch.status));
        return SPARK_STATUS_OK;
    }

    chunk_bytes = batch.chunk_bytes;
    chunk_count = batch.chunk_count;
    if (chunk_bytes == 0ull || chunk_count == 0u ||
        chunk_count > SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX ||
        chunk_bytes > UINT64_MAX / (uint64_t)chunk_count)
    {
        SparkWeightdAttachCloseBatchFds(&batch);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_shape");
        return SPARK_STATUS_OK;
    }
    covered_bytes = chunk_bytes * (uint64_t)chunk_count;
    if (covered_bytes < expected_arena_bytes)
    {
        SparkWeightdAttachCloseBatchFds(&batch);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_short");
        return SPARK_STATUS_OK;
    }

    outcome->map_handles = (void **)calloc(chunk_count, sizeof(void *));
    if (outcome->map_handles == 0)
    {
        SparkWeightdAttachCloseBatchFds(&batch);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_shape");
        return SPARK_STATUS_OK;
    }
    outcome->map_chunk_bytes = chunk_bytes;
    outcome->map_chunk_count = chunk_count;

    while (batch_offset < chunk_count)
    {
        uint32_t index;
        if (batch.batch_count == 0u)
        {
            memset(&batch, 0, sizeof(batch));
            if (SparkWeightdClientExportBatch(outcome->client,
                    outcome->arena_generation, batch_offset, &batch,
                    timeout_nanoseconds) != SPARK_STATUS_OK)
            {
                SparkWeightdAttachMapUndo(outcome);
                SparkWeightdAttachRelease(outcome);
                SparkWeightdAttachSetReason(reason, "import_exchange");
                return SPARK_STATUS_OK;
            }
        }
        if (batch.status != SPARK_STATUS_OK ||
            batch.batch_offset != batch_offset ||
            batch.chunk_bytes != chunk_bytes ||
            batch.chunk_count != chunk_count ||
            batch.batch_count == 0u ||
            batch_offset > chunk_count - batch.batch_count)
        {
            SparkWeightdAttachCloseBatchFds(&batch);
            SparkWeightdAttachMapUndo(outcome);
            SparkWeightdAttachRelease(outcome);
            SparkWeightdAttachSetReason(reason, "import_shape");
            return SPARK_STATUS_OK;
        }
        for (index = 0u; index < batch.batch_count; index++)
        {
            CUmemGenericAllocationHandle handle = 0;
            CUresult import_rc;
            import_rc = cuMemImportFromShareableHandle(&handle,
                    (void *)(uintptr_t)batch.fds[index],
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
            if (import_rc != CUDA_SUCCESS)
            {
                if (getenv("SPARK_WEIGHTD_IMPORT_DIAG") != 0)
                    fprintf(stderr,
                        "weightd import diag: fd=%d curesult=%d\n",
                        batch.fds[index], (int)import_rc);
                (void)close(batch.fds[index]);
                SparkWeightdAttachCloseBatchFds(&batch);
                SparkWeightdAttachMapUndo(outcome);
                SparkWeightdAttachRelease(outcome);
                SparkWeightdAttachSetReason(reason, "import_handle");
                return SPARK_STATUS_OK;
            }
            (void)close(batch.fds[index]);
            outcome->map_handles[outcome->map_handle_count] = (void *)handle;
            outcome->map_handle_count++;
        }
        batch_offset += batch.batch_count;
        memset(&batch, 0, sizeof(batch));
    }

    outcome->map_span_bytes = covered_bytes;
    if (cuMemAddressReserve(&base, (size_t)covered_bytes, 0u, 0ull, 0ull) !=
        CUDA_SUCCESS)
    {
        SparkWeightdAttachMapUndo(outcome);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_map");
        return SPARK_STATUS_OK;
    }
    outcome->map_base = (void *)(uintptr_t)base;
    {
        uint32_t index;
        for (index = 0u; index < chunk_count; index++)
        {
            if (cuMemMap(base + (CUdeviceptr)index * chunk_bytes,
                    (size_t)chunk_bytes, 0u,
                    (CUmemGenericAllocationHandle)outcome->map_handles[index],
                    0ull) != CUDA_SUCCESS)
            {
                SparkWeightdAttachMapUndo(outcome);
                SparkWeightdAttachRelease(outcome);
                SparkWeightdAttachSetReason(reason, "import_map");
                return SPARK_STATUS_OK;
            }
            outcome->map_mapped_count++;
        }
    }
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = SparkWeightdAttachDeviceId();
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(base, (size_t)covered_bytes, &access, 1u) !=
        CUDA_SUCCESS)
    {
        SparkWeightdAttachMapUndo(outcome);
        SparkWeightdAttachRelease(outcome);
        SparkWeightdAttachSetReason(reason, "import_access");
        return SPARK_STATUS_OK;
    }

    outcome->device_handle = (uint64_t)(uintptr_t)base;
    SparkWeightdAttachSetReason(reason, "");
    return SPARK_STATUS_OK;
}

void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome)
{
    if (outcome == 0)
    {
        return;
    }
    if (outcome->map_base != 0 && outcome->map_handles != 0)
    {
        CUdeviceptr base = (CUdeviceptr)(uintptr_t)outcome->map_base;
        uint32_t index;
        for (index = 0u; index < outcome->map_mapped_count; index++)
        {
            (void)cuMemUnmap(base + (CUdeviceptr)index *
                    outcome->map_chunk_bytes,
                (size_t)outcome->map_chunk_bytes);
        }
        for (index = 0u; index < outcome->map_handle_count; index++)
        {
            (void)cuMemRelease(
                (CUmemGenericAllocationHandle)outcome->map_handles[index]);
        }
        (void)cuMemAddressFree(base, (size_t)outcome->map_span_bytes);
    }
    free(outcome->map_handles);
    outcome->map_handles = 0;
    outcome->map_base = 0;
    outcome->map_span_bytes = 0ull;
    outcome->map_chunk_bytes = 0ull;
    outcome->map_chunk_count = 0u;
    outcome->map_handle_count = 0u;
    outcome->map_mapped_count = 0u;
    if (outcome->client != 0)
    {
        SparkWeightdClientClose(outcome->client);
        outcome->client = 0;
    }
    outcome->device_handle = 0ull;
    outcome->arena_bytes = 0ull;
    outcome->arena_generation = 0ull;
    outcome->loaded_from_pack = 0u;
    outcome->refcount = 0u;
}
