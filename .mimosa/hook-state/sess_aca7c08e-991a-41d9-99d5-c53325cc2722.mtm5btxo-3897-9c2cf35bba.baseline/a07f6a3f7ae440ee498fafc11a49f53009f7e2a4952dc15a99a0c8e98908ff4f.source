#ifndef SPARKPIPE_SPARK_RELEASE_H
#define SPARKPIPE_SPARK_RELEASE_H

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_RELEASE_ABI_VERSION 1u
#define SPARK_RELEASE_DESCRIPTOR_BYTES ((uint32_t)sizeof(SparkReleaseManifest))
#define SPARK_RELEASE_MAX_FILES 256u
#define SPARK_RELEASE_MAX_ROLES 16u
#define SPARK_RELEASE_MAX_ARGUMENTS 96u
#define SPARK_RELEASE_MAX_ENVIRONMENT 64u
#define SPARK_RELEASE_MAX_HOSTS 32u
#define SPARK_RELEASE_MAX_STRING_BYTES 512u
#define SPARK_RELEASE_MAX_PATH_BYTES 4096u
#define SPARK_RELEASE_DEFAULT_POLL_INTERVAL_MS 1000u
#define SPARK_RELEASE_DEFAULT_STOP_GRACE_MS 5000u

#define SPARK_RELEASE_NODE_SELECTOR_ALL 0u
#define SPARK_RELEASE_NODE_SELECTOR_SPARK0 1u
#define SPARK_RELEASE_NODE_SELECTOR_RANK 2u
#define SPARK_RELEASE_NODE_SELECTOR_EXPLICIT_HOST 3u
#define SPARK_RELEASE_NODE_SELECTOR_DISABLED 4u

#define SPARK_RELEASE_FILE_FLAG_EXECUTABLE 0x00000001u
#define SPARK_RELEASE_FILE_FLAG_COMMON 0x00000002u
#define SPARK_RELEASE_FILE_FLAG_RANK_LOCAL 0x00000004u
#define SPARK_RELEASE_FILE_FLAG_CUDA_PACK 0x00000008u
#define SPARK_RELEASE_FILE_FLAG_RESTART_ON_CHANGE 0x00000010u
#define SPARK_RELEASE_FILE_FLAG_RESIDENT_RELOAD_BOUNDARY 0x00000020u

#define SPARK_RELEASE_ROLE_FLAG_RESTART_ON_UPDATE 0x00000001u
#define SPARK_RELEASE_ROLE_FLAG_KEEP_ALIVE 0x00000002u
#define SPARK_RELEASE_ROLE_FLAG_ALLOW_RESIDENT_PACK_CACHE 0x00000004u
#define SPARK_RELEASE_ROLE_FLAG_REQUIRE_EXACT_GENERATION 0x00000008u

#define SPARK_RELEASE_ACTION_FLAG_MANIFEST_CHANGED 0x00000001u
#define SPARK_RELEASE_ACTION_FLAG_FILES_CHANGED 0x00000002u
#define SPARK_RELEASE_ACTION_FLAG_ROLE_CHANGED 0x00000004u
#define SPARK_RELEASE_ACTION_FLAG_RESTART_REQUIRED 0x00000008u
#define SPARK_RELEASE_ACTION_FLAG_RESIDENT_CACHE_REUSE_ALLOWED 0x00000010u


typedef struct SparkReleaseFile
{
    char path[SPARK_RELEASE_MAX_STRING_BYTES];
    char sha256[SPARK_SHA256_HEX_BYTES];
    uint64_t bytes;
    uint32_t flags;
} SparkReleaseFile;

typedef struct SparkReleaseRole
{
    char name[SPARK_RELEASE_MAX_STRING_BYTES];
    char command[SPARK_RELEASE_MAX_STRING_BYTES];
    char pid_file[SPARK_RELEASE_MAX_STRING_BYTES];
    char readiness_file[SPARK_RELEASE_MAX_STRING_BYTES];
    uint32_t selector;
    uint32_t flags;
    uint32_t argument_count;
    char arguments[SPARK_RELEASE_MAX_ARGUMENTS][SPARK_RELEASE_MAX_STRING_BYTES];
    uint32_t environment_count;
    char environment[SPARK_RELEASE_MAX_ENVIRONMENT][SPARK_RELEASE_MAX_STRING_BYTES];
    uint32_t explicit_host_count;
    char explicit_hosts[SPARK_RELEASE_MAX_HOSTS][SPARK_RELEASE_MAX_STRING_BYTES];
} SparkReleaseRole;

typedef struct SparkReleaseManifest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t schema_version;
    uint64_t generation;
    char release_id[SPARK_RELEASE_MAX_STRING_BYTES];
    char git_commit[SPARK_RELEASE_MAX_STRING_BYTES];
    char install_root[SPARK_RELEASE_MAX_STRING_BYTES];
    char state_root[SPARK_RELEASE_MAX_STRING_BYTES];
    uint32_t rank_count;
    uint32_t max_active_sequence_count;
    uint32_t poll_interval_ms;
    uint32_t stop_grace_ms;
    uint32_t file_count;
    SparkReleaseFile files[SPARK_RELEASE_MAX_FILES];
    uint32_t role_count;
    SparkReleaseRole roles[SPARK_RELEASE_MAX_ROLES];
} SparkReleaseManifest;

typedef struct SparkReleaseNodeIdentity
{
    char host[SPARK_RELEASE_MAX_STRING_BYTES];
    uint32_t rank;
    uint32_t rank_is_set;
    uint32_t rank_count;
} SparkReleaseNodeIdentity;

typedef struct SparkReleaseResolvedRole
{
    const SparkReleaseRole *role;
    char command[SPARK_RELEASE_MAX_PATH_BYTES];
    uint32_t argument_count;
    char arguments[SPARK_RELEASE_MAX_ARGUMENTS][SPARK_RELEASE_MAX_PATH_BYTES];
    uint32_t environment_count;
    char environment[SPARK_RELEASE_MAX_ENVIRONMENT][SPARK_RELEASE_MAX_PATH_BYTES];
    char pid_file[SPARK_RELEASE_MAX_PATH_BYTES];
    char readiness_file[SPARK_RELEASE_MAX_PATH_BYTES];
} SparkReleaseResolvedRole;

typedef struct SparkReleaseSyncResult
{
    uint32_t copied_file_count;
    uint32_t verified_file_count;
    uint32_t changed_file_count;
    uint32_t executable_file_count;
    uint32_t cuda_pack_file_count;
    uint32_t action_flags;
    char manifest_sha256[SPARK_SHA256_HEX_BYTES];
} SparkReleaseSyncResult;

void SparkReleaseManifestInitialize(SparkReleaseManifest *manifest);
void SparkReleaseNodeIdentityInitialize(SparkReleaseNodeIdentity *identity);
void SparkReleaseResolvedRoleInitialize(SparkReleaseResolvedRole *resolved_role);
void SparkReleaseSyncResultInitialize(SparkReleaseSyncResult *result);

SparkStatus SparkReleaseManifestLoadFile(
    const char *path,
    SparkReleaseManifest *manifest);
SparkStatus SparkReleaseManifestParseText(
    const char *text,
    uint32_t text_bytes,
    SparkReleaseManifest *manifest);
SparkStatus SparkReleaseManifestValidate(
    const SparkReleaseManifest *manifest);
SparkStatus SparkReleaseManifestFindRoleForNode(
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    const char *preferred_role_name,
    const SparkReleaseRole **role_out);
SparkStatus SparkReleaseResolveRole(
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    const SparkReleaseRole *role,
    SparkReleaseResolvedRole *resolved_role);
SparkStatus SparkReleaseResolveInstallRoot(
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    char *install_root,
    uint32_t install_root_bytes);
SparkStatus SparkReleaseSyncFilesFromDirectory(
    const SparkReleaseManifest *manifest,
    const char *release_directory,
    const char *install_directory_override,
    SparkReleaseSyncResult *result);
SparkStatus SparkReleaseWriteExampleManifest(const char *path);
SparkStatus SparkReleaseFormatResolvedCommandLine(
    const SparkReleaseResolvedRole *resolved_role,
    char *text,
    uint32_t text_bytes);

#ifdef __cplusplus
}
#endif


SparkStatus SparkReleaseEnsureParentDirectory(const char *path);
#endif
