#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_release.h"
#include "sparkpipe/spark_sha256.h"

#define SPARK_TEST_RELEASE_ROOT "build/test_release_runtime"

static const char *SparkTestReleaseArgumentValue(
    const SparkReleaseResolvedRole *role,
    const char *name)
{
    uint32_t argument_index;

    assert(role != 0);
    assert(name != 0);
    for (argument_index = 0u;
         argument_index + 1u < role->argument_count;
         ++argument_index)
    {
        if (strcmp(role->arguments[argument_index],name) == 0)
        {
            return role->arguments[argument_index + 1u];
        }
    }
    return 0;
}

static void SparkTestReleaseWriteFixtureFile(
    const char *path,
    const char *text,
    char sha256[SPARK_SHA256_HEX_BYTES])
{
    assert(&SparkReleaseWriteExampleManifest != 0);
    assert(SparkWriteEntireFile(path,text,strlen(text)) == SPARK_STATUS_OK);
    assert(SparkSha256File(path,sha256) == SPARK_STATUS_OK);
}

static void SparkTestReleaseBuildManifest(
    const char *manifest_path,
    const char *daemon_sha256,
    const char *library_sha256)
{
    char manifest[8192];

    snprintf(
        manifest,
        sizeof(manifest),
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"generation\": 42,\n"
        "  \"release_id\": \"test-release\",\n"
        "  \"git_commit\": \"abcdef\",\n"
        "  \"install_root\": \"%s/install/{host}\",\n"
        "  \"state_root\": \"%s/state/{host}\",\n"
        "  \"rank_count\": 13,\n"
        "  \"max_active_sequence_count\": 512,\n"
        "  \"files\": [\n"
        "    {\"path\": \"bin/rank_daemon\", \"sha256\": \"%s\", \"executable\": true},\n"
        "    {\"path\": \"lib/backend.so\", \"sha256\": \"%s\"}\n"
        "  ],\n"
        "  \"roles\": [\n"
        "    {\n"
        "      \"name\": \"rank\",\n"
        "      \"selector\": \"rank\",\n"
        "      \"command\": \"bin/rank_daemon\",\n"
        "      \"pid_file\": \"{state_root}/run/{role}.pid\",\n"
        "      \"argv\": [\"--rank\", \"{rank}\", \"--rank-hex\", \"{rank_hex}\", \"--root\", \"{install_root}\", \"--max-active\", \"{max_active}\"],\n"
        "      \"env\": [\"LD_LIBRARY_PATH={install_root}/lib\"]\n"
        "    }\n"
        "  ]\n"
        "}\n",
        SPARK_TEST_RELEASE_ROOT,
        SPARK_TEST_RELEASE_ROOT,
        daemon_sha256,
        library_sha256);
    assert(SparkWriteEntireFile(manifest_path,manifest,strlen(manifest)) == SPARK_STATUS_OK);
}

static void SparkTestReleaseParseResolveAndSync(void)
{
    char daemon_sha256[SPARK_SHA256_HEX_BYTES];
    char library_sha256[SPARK_SHA256_HEX_BYTES];
    SparkReleaseManifest manifest;
    SparkReleaseNodeIdentity identity;
    const SparkReleaseRole *role;
    SparkReleaseResolvedRole resolved_role;
    SparkReleaseSyncResult sync_result;
    char command_line[4096];
    char resolved_install_root[SPARK_RELEASE_MAX_PATH_BYTES];
    char install_path[SPARK_RELEASE_MAX_PATH_BYTES];

    assert(SparkRemoveDirectoryTree(SPARK_TEST_RELEASE_ROOT) == SPARK_STATUS_OK);
    assert(SparkCreateDirectories(SPARK_TEST_RELEASE_ROOT "/release/bin") == SPARK_STATUS_OK);
    assert(SparkCreateDirectories(SPARK_TEST_RELEASE_ROOT "/release/lib") == SPARK_STATUS_OK);
    SparkTestReleaseWriteFixtureFile(
        SPARK_TEST_RELEASE_ROOT "/release/bin/rank_daemon",
        "rank daemon\n",
        daemon_sha256);
    SparkTestReleaseWriteFixtureFile(
        SPARK_TEST_RELEASE_ROOT "/release/lib/backend.so",
        "backend\n",
        library_sha256);
    SparkTestReleaseBuildManifest(
        SPARK_TEST_RELEASE_ROOT "/release/sparkpipe.json",
        daemon_sha256,
        library_sha256);

    assert(SparkReleaseManifestLoadFile(
        SPARK_TEST_RELEASE_ROOT "/release/sparkpipe.json",
        &manifest) == SPARK_STATUS_OK);
    assert(manifest.generation == 42u);
    assert(manifest.max_active_sequence_count == 512u);
    assert(manifest.file_count == 2u);
    assert(manifest.role_count == 1u);

    SparkReleaseNodeIdentityInitialize(&identity);
    assert(SparkCopyString(identity.host,sizeof(identity.host),"sparkb") == SPARK_STATUS_OK);
    identity.rank = 11u;
    identity.rank_is_set = 1u;
    identity.rank_count = 13u;
    assert(SparkReleaseManifestFindRoleForNode(&manifest,&identity,"rank",&role) == SPARK_STATUS_OK);
    assert(SparkReleaseResolveRole(&manifest,&identity,role,&resolved_role) == SPARK_STATUS_OK);
    assert(strstr(resolved_role.command,"/install/sparkb/bin/rank_daemon") != 0);
    assert(strcmp(resolved_role.arguments[1],"11") == 0);
    assert(strcmp(resolved_role.arguments[3],"b") == 0);
    assert(strstr(resolved_role.arguments[5],"/install/sparkb") != 0);
    assert(strcmp(resolved_role.arguments[7],"512") == 0);
    assert(strstr(resolved_role.environment[0],"/install/sparkb/lib") != 0);
    assert(SparkReleaseFormatResolvedCommandLine(&resolved_role,command_line,sizeof(command_line)) == SPARK_STATUS_OK);
    assert(strstr(command_line,"--rank 11") != 0);
    assert(SparkReleaseResolveInstallRoot(
        &manifest,
        &identity,
        resolved_install_root,
        sizeof(resolved_install_root)) == SPARK_STATUS_OK);
    assert(strcmp(
        resolved_install_root,
        SPARK_TEST_RELEASE_ROOT "/install/sparkb") == 0);

    assert(SparkReleaseSyncFilesFromDirectory(
        &manifest,
        SPARK_TEST_RELEASE_ROOT "/release",
        resolved_install_root,
        &sync_result) == SPARK_STATUS_OK);
    assert(sync_result.verified_file_count == 2u);
    assert(sync_result.changed_file_count == 2u);
    assert(sync_result.executable_file_count == 1u);
    assert(SparkJoinPath(
        resolved_install_root,
        "bin/rank_daemon",
        install_path,
        sizeof(install_path)) == SPARK_STATUS_OK);
    assert(SparkPathExists(install_path));
    assert(SparkReleaseSyncFilesFromDirectory(
        &manifest,
        SPARK_TEST_RELEASE_ROOT "/release",
        resolved_install_root,
        &sync_result) == SPARK_STATUS_OK);
    assert(sync_result.verified_file_count == 2u);
    assert(sync_result.changed_file_count == 0u);
}

static void SparkTestReleaseExampleResidentDeployment(void)
{
    SparkReleaseManifest manifest;
    SparkReleaseNodeIdentity identity;
    const SparkReleaseRole *role;
    SparkReleaseResolvedRole resolved_role;

    assert(SparkCreateDirectories(SPARK_TEST_RELEASE_ROOT "/example") == SPARK_STATUS_OK);
    assert(SparkReleaseWriteExampleManifest(
        SPARK_TEST_RELEASE_ROOT "/example/sparkpipe.json") == SPARK_STATUS_OK);
    assert(SparkReleaseManifestLoadFile(
        SPARK_TEST_RELEASE_ROOT "/example/sparkpipe.json",
        &manifest) == SPARK_STATUS_OK);
    assert(manifest.max_active_sequence_count == 512u);
    assert(manifest.role_count == 1u);
    SparkReleaseNodeIdentityInitialize(&identity);
    assert(SparkCopyString(identity.host,sizeof(identity.host),"spark8") == SPARK_STATUS_OK);
    identity.rank = 8u;
    identity.rank_is_set = 1u;
    identity.rank_count = 13u;
    assert(SparkReleaseManifestFindRoleForNode(
        &manifest,&identity,0,&role) == SPARK_STATUS_OK);
    assert(strcmp(role->name,"model_resident") == 0);
    assert(SparkReleaseManifestFindRoleForNode(
        &manifest,&identity,"model_resident",&role) == SPARK_STATUS_OK);
    assert(SparkReleaseResolveRole(&manifest,&identity,role,&resolved_role) == SPARK_STATUS_OK);
    assert(strstr(resolved_role.command,
        "/home/spark8/sparkdata/example.pp13/bin/sparkpipe_model_residentd") != 0);
    assert(strstr(SparkTestReleaseArgumentValue(
        &resolved_role,"--deployment"),
        "/home/spark8/sparkdata/example.pp13/config/model_resident.json") != 0);
    assert(strcmp(SparkTestReleaseArgumentValue(
        &resolved_role,"--rank-index"),"8") == 0);
    SparkReleaseNodeIdentityInitialize(&identity);
    assert(SparkCopyString(identity.host,sizeof(identity.host),"spark0") == SPARK_STATUS_OK);
    identity.rank = 0u;
    identity.rank_is_set = 1u;
    identity.rank_count = 13u;
    assert(SparkReleaseManifestFindRoleForNode(
        &manifest,&identity,"model_resident",&role) == SPARK_STATUS_OK);
}

static void SparkTestReleaseRejectsSymlinkDirectories(void)
{
    assert(SparkRemoveDirectoryTree(SPARK_TEST_RELEASE_ROOT) == SPARK_STATUS_OK);
    assert(SparkCreateDirectories(SPARK_TEST_RELEASE_ROOT "/real") ==
        SPARK_STATUS_OK);
    assert(SparkPathIsRealDirectoryTree(SPARK_TEST_RELEASE_ROOT "/real"));
    assert(symlink("real",SPARK_TEST_RELEASE_ROOT "/link") == 0);
    assert(!SparkPathIsRealDirectoryTree(SPARK_TEST_RELEASE_ROOT "/link"));
    assert(SparkCreateDirectories(SPARK_TEST_RELEASE_ROOT "/link/child") ==
        SPARK_STATUS_IO_ERROR);
}

int main(void)
{
    SparkTestReleaseParseResolveAndSync();
    SparkTestReleaseExampleResidentDeployment();
    SparkTestReleaseRejectsSymlinkDirectories();
    return 0;
}
