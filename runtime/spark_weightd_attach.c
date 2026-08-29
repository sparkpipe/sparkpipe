/* spark_weightd serving-side attach (docs/WEIGHTD_DESIGN.md W2b): the
 * consumer helper over the deadline-client. Every gate below falls back —
 * the direct pack load is the contract, the attach is the optimization —
 * and the daemon's digest verification stays the only bytes authority. */

#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_weightd_attach.h"

#include <stdlib.h>
#include <string.h>

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

    /* gate 1: the kill-switch parity (off wins over anything) */
    if (SparkWeightdAttachEnvIsOff(SPARK_WEIGHTD_ATTACH_ENV_SWITCH) != 0)
    {
        SparkWeightdAttachSetReason(reason, "env_off");
        return SPARK_STATUS_OK;
    }
    /* gate 2: no socket configured — the daemon is absent by deployment */
    socket = SparkWeightdAttachEnvText(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0)
    {
        SparkWeightdAttachSetReason(reason, "no_socket");
        return SPARK_STATUS_OK;
    }
    /* gate 3: the identity is published, never computed per process */
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

    /* the deadline-client: a refused/missing daemon fails the connect
     * closed at the front door — that IS the fallback signal */
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
        /* transport fault mid-exchange: deadline expiry (busy) or the
         * daemon dying under us (io_error) — fail closed to the fallback */
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

void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome)
{
    if (outcome == 0)
    {
        return;
    }
    if (outcome->client != 0)
    {
        /* close WITHOUT detaching: the daemon reaps the dead socket, drops
         * this process's reference, and the arena stays warm — the crash
         * path and the clean teardown are the same code */
        SparkWeightdClientClose(outcome->client);
        outcome->client = 0;
    }
    outcome->device_handle = 0ull;
    outcome->arena_bytes = 0ull;
    outcome->arena_generation = 0ull;
    outcome->loaded_from_pack = 0u;
    outcome->refcount = 0u;
}
