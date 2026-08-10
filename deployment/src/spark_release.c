#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_release.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"

static const char SparkReleaseTemplateInstallRoot[] = "{install_root}";
static const char SparkReleaseTemplateStateRoot[] = "{state_root}";
static const char SparkReleaseTemplateHost[] = "{host}";
static const char SparkReleaseTemplateRank[] = "{rank}";
static const char SparkReleaseTemplateRankHex[] = "{rank_hex}";
static const char SparkReleaseTemplateRankCount[] = "{rank_count}";
static const char SparkReleaseTemplateMaxActive[] = "{max_active}";
static const char SparkReleaseTemplateReleaseId[] = "{release_id}";
static const char SparkReleaseTemplateGeneration[] = "{generation}";
static const char SparkReleaseTemplateRole[] = "{role}";

static SparkStatus SparkReleaseCopyStringField(
    char *destination,
    uint32_t destination_bytes,
    const char *source)
{
    return SparkCopyString(destination,destination_bytes,source == 0 ? "" : source);
}

static SparkStatus SparkReleaseCopyJsonString(
    const SparkJsonDocument *document,
    int32_t token_index,
    char *destination,
    uint32_t destination_bytes)
{
    char *text;
    SparkStatus status;

    if (destination == 0 || destination_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    destination[0] = '\0';
    if (token_index < 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkJsonCopyString(document,token_index,&text);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkReleaseCopyStringField(destination,destination_bytes,text);
    free(text);
    return status;
}

static SparkStatus SparkReleaseCopyOptionalJsonString(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    char *destination,
    uint32_t destination_bytes,
    const char *default_value)
{
    int32_t value_token_index;

    value_token_index = SparkJsonFindObjectMember(document,object_token_index,member_name);
    if (value_token_index < 0)
    {
        return SparkReleaseCopyStringField(destination,destination_bytes,default_value);
    }
    return SparkReleaseCopyJsonString(document,value_token_index,destination,destination_bytes);
}

static SparkStatus SparkReleaseGetOptionalU32(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    uint32_t default_value,
    uint32_t *value_out)
{
    int32_t value_token_index;

    if (value_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    value_token_index = SparkJsonFindObjectMember(document,object_token_index,member_name);
    if (value_token_index < 0)
    {
        *value_out = default_value;
        return SPARK_STATUS_OK;
    }
    return SparkJsonGetUInt32(document,value_token_index,value_out);
}

static SparkStatus SparkReleaseGetOptionalU64(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    uint64_t default_value,
    uint64_t *value_out)
{
    int32_t value_token_index;

    if (value_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    value_token_index = SparkJsonFindObjectMember(document,object_token_index,member_name);
    if (value_token_index < 0)
    {
        *value_out = default_value;
        return SPARK_STATUS_OK;
    }
    return SparkJsonGetUInt64(document,value_token_index,value_out);
}

static SparkStatus SparkReleaseGetOptionalBoolean(
    const SparkJsonDocument *document,
    int32_t object_token_index,
    const char *member_name,
    bool default_value,
    bool *value_out)
{
    int32_t value_token_index;

    if (value_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    value_token_index = SparkJsonFindObjectMember(document,object_token_index,member_name);
    if (value_token_index < 0)
    {
        *value_out = default_value;
        return SPARK_STATUS_OK;
    }
    return SparkJsonGetBoolean(document,value_token_index,value_out);
}

static uint32_t SparkReleaseSelectorFromText(const char *text)
{
    if (text == 0 || strcmp(text,"all") == 0)
    {
        return SPARK_RELEASE_NODE_SELECTOR_ALL;
    }
    if (strcmp(text,"spark0") == 0 || strcmp(text,"gateway") == 0)
    {
        return SPARK_RELEASE_NODE_SELECTOR_SPARK0;
    }
    if (strcmp(text,"rank") == 0 || strcmp(text,"ranks") == 0 || strcmp(text,"ring_rank") == 0)
    {
        return SPARK_RELEASE_NODE_SELECTOR_RANK;
    }
    if (strcmp(text,"host") == 0 || strcmp(text,"hosts") == 0)
    {
        return SPARK_RELEASE_NODE_SELECTOR_EXPLICIT_HOST;
    }
    if (strcmp(text,"disabled") == 0)
    {
        return SPARK_RELEASE_NODE_SELECTOR_DISABLED;
    }
    return UINT32_MAX;
}

static SparkStatus SparkReleaseParseStringArray(
    const SparkJsonDocument *document,
    int32_t array_token_index,
    uint32_t maximum_count,
    uint32_t string_bytes,
    uint32_t *count_out,
    char strings[][SPARK_RELEASE_MAX_STRING_BYTES])
{
    uint32_t count;
    uint32_t element_index;
    SparkStatus status;

    if (count_out == 0 || strings == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *count_out = 0u;
    if (array_token_index < 0)
    {
        return SPARK_STATUS_OK;
    }
    if (!SparkJsonTokenIsType(document,array_token_index,SPARK_JSON_TOKEN_ARRAY))
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    count = SparkJsonGetArrayElementCount(document,array_token_index);
    if (count > maximum_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (element_index = 0u; element_index < count; ++element_index)
    {
        int32_t element_token_index;

        element_token_index = SparkJsonGetArrayElement(document,array_token_index,element_index);
        status = SparkReleaseCopyJsonString(document,element_token_index,strings[element_index],string_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    *count_out = count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseParseFile(
    const SparkJsonDocument *document,
    int32_t file_token_index,
    SparkReleaseFile *file)
{
    SparkStatus status;
    bool flag_value;

    if (!SparkJsonTokenIsType(document,file_token_index,SPARK_JSON_TOKEN_OBJECT) || file == 0)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    memset(file,0,sizeof(*file));
    status = SparkReleaseCopyOptionalJsonString(document,file_token_index,"path",file->path,sizeof(file->path),0);
    if (status != SPARK_STATUS_OK || file->path[0] == '\0')
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status;
    }
    status = SparkReleaseCopyOptionalJsonString(document,file_token_index,"sha256",file->sha256,sizeof(file->sha256),0);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkReleaseGetOptionalU64(document,file_token_index,"bytes",0u,&file->bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkReleaseGetOptionalBoolean(document,file_token_index,"executable",false,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        file->flags |= SPARK_RELEASE_FILE_FLAG_EXECUTABLE;
    }
    status = SparkReleaseGetOptionalBoolean(document,file_token_index,"common",true,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        file->flags |= SPARK_RELEASE_FILE_FLAG_COMMON;
    }
    status = SparkReleaseGetOptionalBoolean(document,file_token_index,"rank_local",false,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        file->flags |= SPARK_RELEASE_FILE_FLAG_RANK_LOCAL;
    }
    status = SparkReleaseGetOptionalBoolean(document,file_token_index,"cuda_pack",false,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        file->flags |= SPARK_RELEASE_FILE_FLAG_CUDA_PACK;
    }
    status = SparkReleaseGetOptionalBoolean(document,file_token_index,"restart_on_change",true,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        file->flags |= SPARK_RELEASE_FILE_FLAG_RESTART_ON_CHANGE;
    }
    status = SparkReleaseGetOptionalBoolean(document,file_token_index,"resident_reload_boundary",false,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        file->flags |= SPARK_RELEASE_FILE_FLAG_RESIDENT_RELOAD_BOUNDARY;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseParseRole(
    const SparkJsonDocument *document,
    int32_t role_token_index,
    SparkReleaseRole *role)
{
    SparkStatus status;
    int32_t token_index;
    char selector_text[SPARK_RELEASE_MAX_STRING_BYTES];
    bool flag_value;

    if (!SparkJsonTokenIsType(document,role_token_index,SPARK_JSON_TOKEN_OBJECT) || role == 0)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    memset(role,0,sizeof(*role));
    status = SparkReleaseCopyOptionalJsonString(document,role_token_index,"name",role->name,sizeof(role->name),0);
    if (status != SPARK_STATUS_OK || role->name[0] == '\0')
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status;
    }
    status = SparkReleaseCopyOptionalJsonString(document,role_token_index,"command",role->command,sizeof(role->command),0);
    if (status != SPARK_STATUS_OK || role->command[0] == '\0')
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status;
    }
    status = SparkReleaseCopyOptionalJsonString(document,role_token_index,"selector",selector_text,sizeof(selector_text),"all");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    role->selector = SparkReleaseSelectorFromText(selector_text);
    if (role->selector == UINT32_MAX)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkReleaseCopyOptionalJsonString(document,role_token_index,"pid_file",role->pid_file,sizeof(role->pid_file),"{state_root}/run/{role}.pid");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkReleaseCopyOptionalJsonString(document,role_token_index,"readiness_file",role->readiness_file,sizeof(role->readiness_file),"{state_root}/run/{role}.ready");
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkReleaseGetOptionalBoolean(document,role_token_index,"restart_on_update",true,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        role->flags |= SPARK_RELEASE_ROLE_FLAG_RESTART_ON_UPDATE;
    }
    status = SparkReleaseGetOptionalBoolean(document,role_token_index,"keep_alive",true,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        role->flags |= SPARK_RELEASE_ROLE_FLAG_KEEP_ALIVE;
    }
    status = SparkReleaseGetOptionalBoolean(document,role_token_index,"allow_resident_pack_cache",false,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        role->flags |= SPARK_RELEASE_ROLE_FLAG_ALLOW_RESIDENT_PACK_CACHE;
    }
    status = SparkReleaseGetOptionalBoolean(document,role_token_index,"require_exact_generation",true,&flag_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (flag_value)
    {
        role->flags |= SPARK_RELEASE_ROLE_FLAG_REQUIRE_EXACT_GENERATION;
    }
    token_index = SparkJsonFindObjectMember(document,role_token_index,"argv");
    status = SparkReleaseParseStringArray(document,token_index,SPARK_RELEASE_MAX_ARGUMENTS,SPARK_RELEASE_MAX_STRING_BYTES,&role->argument_count,role->arguments);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    token_index = SparkJsonFindObjectMember(document,role_token_index,"env");
    status = SparkReleaseParseStringArray(document,token_index,SPARK_RELEASE_MAX_ENVIRONMENT,SPARK_RELEASE_MAX_STRING_BYTES,&role->environment_count,role->environment);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    token_index = SparkJsonFindObjectMember(document,role_token_index,"hosts");
    return SparkReleaseParseStringArray(document,token_index,SPARK_RELEASE_MAX_HOSTS,SPARK_RELEASE_MAX_STRING_BYTES,&role->explicit_host_count,role->explicit_hosts);
}

void SparkReleaseManifestInitialize(SparkReleaseManifest *manifest)
{
    if (manifest == 0)
    {
        return;
    }
    memset(manifest,0,sizeof(*manifest));
    manifest->abi_version = SPARK_RELEASE_ABI_VERSION;
    manifest->descriptor_bytes = SPARK_RELEASE_DESCRIPTOR_BYTES;
    manifest->schema_version = 1u;
    manifest->rank_count = 13u;
    manifest->poll_interval_ms = SPARK_RELEASE_DEFAULT_POLL_INTERVAL_MS;
    manifest->stop_grace_ms = SPARK_RELEASE_DEFAULT_STOP_GRACE_MS;
}

void SparkReleaseNodeIdentityInitialize(SparkReleaseNodeIdentity *identity)
{
    if (identity == 0)
    {
        return;
    }
    memset(identity,0,sizeof(*identity));
    identity->rank_count = 13u;
}

void SparkReleaseResolvedRoleInitialize(SparkReleaseResolvedRole *resolved_role)
{
    if (resolved_role == 0)
    {
        return;
    }
    memset(resolved_role,0,sizeof(*resolved_role));
}

void SparkReleaseSyncResultInitialize(SparkReleaseSyncResult *result)
{
    if (result == 0)
    {
        return;
    }
    memset(result,0,sizeof(*result));
}

SparkStatus SparkReleaseManifestParseText(
    const char *text,
    uint32_t text_bytes,
    SparkReleaseManifest *manifest)
{
    SparkJsonDocument document;
    SparkStatus status;
    int32_t root_token_index;
    int32_t token_index;
    uint32_t item_count;
    uint32_t item_index;

    if (text == 0 || manifest == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkReleaseManifestInitialize(manifest);
    SparkJsonDocumentReset(&document);
    status = SparkJsonParseText(text,text_bytes,&document);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    if (!SparkJsonTokenIsType(&document,root_token_index,SPARK_JSON_TOKEN_OBJECT))
    {
        SparkJsonDocumentDestroy(&document);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkReleaseGetOptionalU32(&document,root_token_index,"schema_version",1u,&manifest->schema_version);
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseGetOptionalU64(&document,root_token_index,"generation",0u,&manifest->generation);
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseCopyOptionalJsonString(&document,root_token_index,"release_id",manifest->release_id,sizeof(manifest->release_id),"");
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseCopyOptionalJsonString(&document,root_token_index,"git_commit",manifest->git_commit,sizeof(manifest->git_commit),"");
    if (status != SPARK_STATUS_OK) goto cleanup;
    token_index = SparkJsonFindObjectMember(&document,root_token_index,
        "install_root");
    status = SparkReleaseCopyJsonString(&document,token_index,
        manifest->install_root,sizeof(manifest->install_root));
    if (status != SPARK_STATUS_OK) goto cleanup;
    token_index = SparkJsonFindObjectMember(&document,root_token_index,
        "state_root");
    status = SparkReleaseCopyJsonString(&document,token_index,
        manifest->state_root,sizeof(manifest->state_root));
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseGetOptionalU32(&document,root_token_index,"rank_count",13u,&manifest->rank_count);
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseGetOptionalU32(&document,root_token_index,"max_active_sequence_count",1024u,&manifest->max_active_sequence_count);
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseGetOptionalU32(&document,root_token_index,"poll_interval_ms",SPARK_RELEASE_DEFAULT_POLL_INTERVAL_MS,&manifest->poll_interval_ms);
    if (status != SPARK_STATUS_OK) goto cleanup;
    status = SparkReleaseGetOptionalU32(&document,root_token_index,"stop_grace_ms",SPARK_RELEASE_DEFAULT_STOP_GRACE_MS,&manifest->stop_grace_ms);
    if (status != SPARK_STATUS_OK) goto cleanup;

    token_index = SparkJsonFindObjectMember(&document,root_token_index,"files");
    if (!SparkJsonTokenIsType(&document,token_index,SPARK_JSON_TOKEN_ARRAY))
    {
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    item_count = SparkJsonGetArrayElementCount(&document,token_index);
    if (item_count > SPARK_RELEASE_MAX_FILES)
    {
        status = SPARK_STATUS_CAPACITY_EXCEEDED;
        goto cleanup;
    }
    manifest->file_count = item_count;
    for (item_index = 0u; item_index < item_count; ++item_index)
    {
        status = SparkReleaseParseFile(&document,SparkJsonGetArrayElement(&document,token_index,item_index),&manifest->files[item_index]);
        if (status != SPARK_STATUS_OK) goto cleanup;
    }

    token_index = SparkJsonFindObjectMember(&document,root_token_index,"roles");
    if (!SparkJsonTokenIsType(&document,token_index,SPARK_JSON_TOKEN_ARRAY))
    {
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    item_count = SparkJsonGetArrayElementCount(&document,token_index);
    if (item_count > SPARK_RELEASE_MAX_ROLES)
    {
        status = SPARK_STATUS_CAPACITY_EXCEEDED;
        goto cleanup;
    }
    manifest->role_count = item_count;
    for (item_index = 0u; item_index < item_count; ++item_index)
    {
        status = SparkReleaseParseRole(&document,SparkJsonGetArrayElement(&document,token_index,item_index),&manifest->roles[item_index]);
        if (status != SPARK_STATUS_OK) goto cleanup;
    }

    status = SparkReleaseManifestValidate(manifest);

cleanup:
    SparkJsonDocumentDestroy(&document);
    return status;
}

SparkStatus SparkReleaseManifestLoadFile(
    const char *path,
    SparkReleaseManifest *manifest)
{
    char *text;
    size_t text_bytes;
    SparkStatus status;

    if (path == 0 || manifest == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkReadEntireFile(path,&text,&text_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (text_bytes > UINT32_MAX)
    {
        free(text);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkReleaseManifestParseText(text,(uint32_t)text_bytes,manifest);
    free(text);
    return status;
}

static bool SparkReleasePathIsSafe(const char *path)
{
    const char *cursor;

    if (path == 0 || path[0] == '\0' || path[0] == '/')
    {
        return false;
    }
    cursor = path;
    while (*cursor != '\0')
    {
        if ((cursor[0] == '.' && cursor[1] == '.' && (cursor[2] == '/' || cursor[2] == '\0')) ||
            (cursor[0] == '/' && cursor[1] == '.' && cursor[2] == '.' && (cursor[3] == '/' || cursor[3] == '\0')))
        {
            return false;
        }
        ++cursor;
    }
    return true;
}

SparkStatus SparkReleaseManifestValidate(
    const SparkReleaseManifest *manifest)
{
    uint32_t file_index;
    uint32_t role_index;

    if (manifest == 0 || manifest->schema_version != 1u || manifest->file_count == 0u ||
        manifest->role_count == 0u || manifest->rank_count == 0u || manifest->poll_interval_ms == 0u)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    if (manifest->install_root[0] == '\0' || manifest->state_root[0] == '\0')
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    for (file_index = 0u; file_index < manifest->file_count; ++file_index)
    {
        const SparkReleaseFile *file;

        file = &manifest->files[file_index];
        if (!SparkReleasePathIsSafe(file->path) || !SparkSha256HexIsValid(file->sha256))
        {
            return SPARK_STATUS_SCHEMA_ERROR;
        }
    }
    for (role_index = 0u; role_index < manifest->role_count; ++role_index)
    {
        const SparkReleaseRole *role;

        role = &manifest->roles[role_index];
        if (role->name[0] == '\0' || role->command[0] == '\0' || role->selector == SPARK_RELEASE_NODE_SELECTOR_DISABLED)
        {
            return SPARK_STATUS_SCHEMA_ERROR;
        }
        if (role->selector == SPARK_RELEASE_NODE_SELECTOR_EXPLICIT_HOST && role->explicit_host_count == 0u)
        {
            return SPARK_STATUS_SCHEMA_ERROR;
        }
        if (!SparkReleasePathIsSafe(role->command) && role->command[0] != '/' && strstr(role->command,"{install_root}") == 0)
        {
            return SPARK_STATUS_SCHEMA_ERROR;
        }
    }
    return SPARK_STATUS_OK;
}

static bool SparkReleaseRoleMatchesNode(
    const SparkReleaseRole *role,
    const SparkReleaseNodeIdentity *identity)
{
    uint32_t host_index;

    if (role == 0 || identity == 0)
    {
        return false;
    }
    switch (role->selector)
    {
        case SPARK_RELEASE_NODE_SELECTOR_ALL:
        {
            return true;
        }
        case SPARK_RELEASE_NODE_SELECTOR_SPARK0:
        {
            return strcmp(identity->host,"spark0") == 0 || (identity->rank_is_set != 0u && identity->rank == 0u);
        }
        case SPARK_RELEASE_NODE_SELECTOR_RANK:
        {
            return identity->rank_is_set != 0u;
        }
        case SPARK_RELEASE_NODE_SELECTOR_EXPLICIT_HOST:
        {
            for (host_index = 0u; host_index < role->explicit_host_count; ++host_index)
            {
                if (strcmp(role->explicit_hosts[host_index],identity->host) == 0)
                {
                    return true;
                }
            }
            return false;
        }
        default:
        {
            return false;
        }
    }
}

SparkStatus SparkReleaseManifestFindRoleForNode(
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    const char *preferred_role_name,
    const SparkReleaseRole **role_out)
{
    uint32_t role_index;

    if (manifest == 0 || identity == 0 || role_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *role_out = 0;
    for (role_index = 0u; role_index < manifest->role_count; ++role_index)
    {
        const SparkReleaseRole *role;

        role = &manifest->roles[role_index];
        if (preferred_role_name != 0 && preferred_role_name[0] != '\0' && strcmp(role->name,preferred_role_name) != 0)
        {
            continue;
        }
        if (SparkReleaseRoleMatchesNode(role,identity))
        {
            *role_out = role;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}

static SparkStatus SparkReleaseAppendText(char *destination,uint32_t destination_bytes,uint32_t *offset,const char *text)
{
    uint32_t text_index;

    if (destination == 0 || offset == 0 || text == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    text_index = 0u;
    while (text[text_index] != '\0')
    {
        if (*offset + 1u >= destination_bytes)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        destination[*offset] = text[text_index];
        *offset += 1u;
        text_index += 1u;
    }
    destination[*offset] = '\0';
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseAppendU32(char *destination,uint32_t destination_bytes,uint32_t *offset,uint32_t value)
{
    char text[32];

    snprintf(text,sizeof(text),"%u",value);
    return SparkReleaseAppendText(destination,destination_bytes,offset,text);
}

static SparkStatus SparkReleaseAppendRankHex(char *destination,uint32_t destination_bytes,uint32_t *offset,uint32_t rank)
{
    char text[8];

    if (rank < 10u)
    {
        snprintf(text,sizeof(text),"%u",rank);
    }
    else
    {
        snprintf(text,sizeof(text),"%c",(char)('a' + (rank - 10u)));
    }
    return SparkReleaseAppendText(destination,destination_bytes,offset,text);
}

static SparkStatus SparkReleaseExpandTemplate(
    const char *source,
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    const SparkReleaseRole *role,
    char *destination,
    uint32_t destination_bytes)
{
    uint32_t source_index;
    uint32_t offset;
    SparkStatus status;

    if (source == 0 || manifest == 0 || identity == 0 || destination == 0 || destination_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    destination[0] = '\0';
    source_index = 0u;
    offset = 0u;
    while (source[source_index] != '\0')
    {
        if (source[source_index] != '{')
        {
            char character[2];

            character[0] = source[source_index++];
            character[1] = '\0';
            status = SparkReleaseAppendText(destination,destination_bytes,&offset,character);
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateInstallRoot,sizeof(SparkReleaseTemplateInstallRoot) - 1u) == 0)
        {
            status = SparkReleaseExpandTemplate(manifest->install_root,manifest,identity,role,destination + offset,destination_bytes - offset);
            if (status == SPARK_STATUS_OK)
            {
                offset = (uint32_t)strlen(destination);
            }
            source_index += sizeof(SparkReleaseTemplateInstallRoot) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateStateRoot,sizeof(SparkReleaseTemplateStateRoot) - 1u) == 0)
        {
            status = SparkReleaseExpandTemplate(manifest->state_root,manifest,identity,role,destination + offset,destination_bytes - offset);
            if (status == SPARK_STATUS_OK)
            {
                offset = (uint32_t)strlen(destination);
            }
            source_index += sizeof(SparkReleaseTemplateStateRoot) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateHost,sizeof(SparkReleaseTemplateHost) - 1u) == 0)
        {
            status = SparkReleaseAppendText(destination,destination_bytes,&offset,identity->host);
            source_index += sizeof(SparkReleaseTemplateHost) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateRank,sizeof(SparkReleaseTemplateRank) - 1u) == 0)
        {
            status = SparkReleaseAppendU32(destination,destination_bytes,&offset,identity->rank);
            source_index += sizeof(SparkReleaseTemplateRank) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateRankHex,sizeof(SparkReleaseTemplateRankHex) - 1u) == 0)
        {
            status = SparkReleaseAppendRankHex(destination,destination_bytes,&offset,identity->rank);
            source_index += sizeof(SparkReleaseTemplateRankHex) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateRankCount,sizeof(SparkReleaseTemplateRankCount) - 1u) == 0)
        {
            status = SparkReleaseAppendU32(destination,destination_bytes,&offset,identity->rank_count != 0u ? identity->rank_count : manifest->rank_count);
            source_index += sizeof(SparkReleaseTemplateRankCount) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateMaxActive,sizeof(SparkReleaseTemplateMaxActive) - 1u) == 0)
        {
            status = SparkReleaseAppendU32(destination,destination_bytes,&offset,manifest->max_active_sequence_count);
            source_index += sizeof(SparkReleaseTemplateMaxActive) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateReleaseId,sizeof(SparkReleaseTemplateReleaseId) - 1u) == 0)
        {
            status = SparkReleaseAppendText(destination,destination_bytes,&offset,manifest->release_id);
            source_index += sizeof(SparkReleaseTemplateReleaseId) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateGeneration,sizeof(SparkReleaseTemplateGeneration) - 1u) == 0)
        {
            char text[32];

            snprintf(text,sizeof(text),"%llu",(unsigned long long)manifest->generation);
            status = SparkReleaseAppendText(destination,destination_bytes,&offset,text);
            source_index += sizeof(SparkReleaseTemplateGeneration) - 1u;
        }
        else if (strncmp(source + source_index,SparkReleaseTemplateRole,sizeof(SparkReleaseTemplateRole) - 1u) == 0)
        {
            status = SparkReleaseAppendText(destination,destination_bytes,&offset,role == 0 ? "sparkpipe" : role->name);
            source_index += sizeof(SparkReleaseTemplateRole) - 1u;
        }
        else
        {
            status = SparkReleaseAppendText(destination,destination_bytes,&offset,"{");
            source_index += 1u;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkReleaseResolveCommandPath(
    const char *command,
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    const SparkReleaseRole *role,
    char *destination,
    uint32_t destination_bytes)
{
    char expanded[SPARK_RELEASE_MAX_PATH_BYTES];
    SparkStatus status;

    status = SparkReleaseExpandTemplate(command,manifest,identity,role,expanded,sizeof(expanded));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (expanded[0] == '/' || strstr(expanded,"{install_root}") != 0)
    {
        return SparkReleaseCopyStringField(destination,destination_bytes,expanded);
    }
    {
        char combined[SPARK_RELEASE_MAX_PATH_BYTES];

        if (snprintf(combined,sizeof(combined),"%s/%s",manifest->install_root,expanded) >= (int)sizeof(combined))
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        return SparkReleaseExpandTemplate(combined,manifest,identity,role,destination,destination_bytes);
    }
}

SparkStatus SparkReleaseResolveInstallRoot(
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    char *install_root,
    uint32_t install_root_bytes)
{
    if (manifest == 0 || identity == 0 || install_root == 0 ||
        install_root_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkReleaseExpandTemplate(
        manifest->install_root,
        manifest,
        identity,
        0,
        install_root,
        install_root_bytes);
}

SparkStatus SparkReleaseResolveRole(
    const SparkReleaseManifest *manifest,
    const SparkReleaseNodeIdentity *identity,
    const SparkReleaseRole *role,
    SparkReleaseResolvedRole *resolved_role)
{
    uint32_t argument_index;
    uint32_t environment_index;
    SparkStatus status;

    if (manifest == 0 || identity == 0 || role == 0 || resolved_role == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkReleaseResolvedRoleInitialize(resolved_role);
    resolved_role->role = role;
    status = SparkReleaseResolveCommandPath(role->command,manifest,identity,role,resolved_role->command,sizeof(resolved_role->command));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    resolved_role->argument_count = role->argument_count;
    for (argument_index = 0u; argument_index < role->argument_count; ++argument_index)
    {
        status = SparkReleaseExpandTemplate(role->arguments[argument_index],manifest,identity,role,resolved_role->arguments[argument_index],sizeof(resolved_role->arguments[argument_index]));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    resolved_role->environment_count = role->environment_count;
    for (environment_index = 0u; environment_index < role->environment_count; ++environment_index)
    {
        status = SparkReleaseExpandTemplate(role->environment[environment_index],manifest,identity,role,resolved_role->environment[environment_index],sizeof(resolved_role->environment[environment_index]));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    status = SparkReleaseExpandTemplate(role->pid_file,manifest,identity,role,resolved_role->pid_file,sizeof(resolved_role->pid_file));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkReleaseExpandTemplate(role->readiness_file,manifest,identity,role,resolved_role->readiness_file,sizeof(resolved_role->readiness_file));
}

static SparkStatus SparkReleaseBuildInstallPath(
    const char *install_root,
    const char *relative_path,
    char *path,
    uint32_t path_bytes)
{
    if (install_root == 0 || relative_path == 0 || path == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (snprintf(path,path_bytes,"%s/%s",install_root,relative_path) >= (int)path_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkReleaseEnsureParentDirectory(const char *path)
{
    char parent_path[SPARK_RELEASE_MAX_PATH_BYTES];
    char *slash;

    if (path == 0 || strlen(path) >= sizeof(parent_path))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    strcpy(parent_path,path);
    slash = strrchr(parent_path,'/');
    if (slash == 0)
    {
        return SPARK_STATUS_OK;
    }
    *slash = '\0';
    if (parent_path[0] == '\0')
    {
        return SPARK_STATUS_OK;
    }
    return SparkCreateDirectories(parent_path);
}


static SparkStatus SparkReleaseCopyFileAtomically(const char *source_path,const char *destination_path)
{
	char *data;
	size_t data_bytes;
	SparkStatus status;
	status = SparkReadEntireFile(source_path, &data, &data_bytes);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkWriteEntireFileAtomically(destination_path, data, data_bytes);
	free(data);
	return status;
}

SparkStatus SparkReleaseSyncFilesFromDirectory(
    const SparkReleaseManifest *manifest,
    const char *release_directory,
    const char *install_directory_override,
    SparkReleaseSyncResult *result)
{
    char install_root[SPARK_RELEASE_MAX_PATH_BYTES];
    uint32_t file_index;
    SparkStatus status;

    if (manifest == 0 || release_directory == 0 || result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkReleaseSyncResultInitialize(result);
    status = SparkSha256File("sparkpipe.json",result->manifest_sha256);
    (void)status;
    if (install_directory_override != 0 && install_directory_override[0] != '\0')
    {
        status = SparkReleaseCopyStringField(install_root,sizeof(install_root),install_directory_override);
    }
    else
    {
        status = SparkReleaseCopyStringField(install_root,sizeof(install_root),manifest->install_root);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCreateDirectories(install_root);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (file_index = 0u; file_index < manifest->file_count; ++file_index)
    {
        const SparkReleaseFile *file;
        char source_path[SPARK_RELEASE_MAX_PATH_BYTES];
        char destination_path[SPARK_RELEASE_MAX_PATH_BYTES];
        char source_hash[SPARK_SHA256_HEX_BYTES];
        char destination_hash[SPARK_SHA256_HEX_BYTES];
        uint32_t destination_changed;

        file = &manifest->files[file_index];
        if (snprintf(source_path,sizeof(source_path),"%s/%s",release_directory,file->path) >= (int)sizeof(source_path))
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = SparkSha256File(source_path,source_hash);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (strcmp(source_hash,file->sha256) != 0)
        {
            return SPARK_STATUS_HASH_MISMATCH;
        }
        status = SparkReleaseBuildInstallPath(install_root,file->path,destination_path,sizeof(destination_path));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        destination_changed = 1u;
        if (SparkPathExists(destination_path))
        {
            status = SparkSha256File(destination_path,destination_hash);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            if (strcmp(destination_hash,file->sha256) == 0)
            {
                destination_changed = 0u;
            }
        }
        if (destination_changed != 0u)
        {
            status = SparkReleaseEnsureParentDirectory(destination_path);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            status = SparkReleaseCopyFileAtomically(source_path,destination_path);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            result->copied_file_count += 1u;
            result->changed_file_count += 1u;
            result->action_flags |= SPARK_RELEASE_ACTION_FLAG_FILES_CHANGED;
        }
        status = SparkSha256File(destination_path,destination_hash);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (strcmp(destination_hash,file->sha256) != 0)
        {
            return SPARK_STATUS_HASH_MISMATCH;
        }
        if ((file->flags & SPARK_RELEASE_FILE_FLAG_EXECUTABLE) != 0u)
        {
            chmod(destination_path,0755);
            result->executable_file_count += 1u;
        }
        if ((file->flags & SPARK_RELEASE_FILE_FLAG_CUDA_PACK) != 0u)
        {
            result->cuda_pack_file_count += 1u;
        }
        if ((file->flags & SPARK_RELEASE_FILE_FLAG_RESTART_ON_CHANGE) != 0u && destination_changed != 0u)
        {
            result->action_flags |= SPARK_RELEASE_ACTION_FLAG_RESTART_REQUIRED;
        }
        if ((file->flags & SPARK_RELEASE_FILE_FLAG_RESIDENT_RELOAD_BOUNDARY) == 0u && (file->flags & SPARK_RELEASE_FILE_FLAG_CUDA_PACK) != 0u)
        {
            result->action_flags |= SPARK_RELEASE_ACTION_FLAG_RESIDENT_CACHE_REUSE_ALLOWED;
        }
        result->verified_file_count += 1u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkReleaseWriteExampleManifest(const char *path)
{
    static const char ExampleManifest[] =
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"generation\": 1,\n"
        "  \"release_id\": \"model-resident-production-001\",\n"
        "  \"git_commit\": \"replace-with-main-hash\",\n"
        "  \"install_root\": \"/home/{host}/sparkdata/example.pp13\",\n"
        "  \"state_root\": \"/home/{host}/sparkdata/.layout/sparkpipe_state/example.pp13\",\n"
        "  \"rank_count\": 13,\n"
        "  \"max_active_sequence_count\": 512,\n"
        "  \"poll_interval_ms\": 1000,\n"
        "  \"stop_grace_ms\": 5000,\n"
        "  \"files\": [\n"
        "    {\n"
        "      \"path\": \"bin/sparkpipe_model_residentd\",\n"
        "      \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
        "      \"executable\": true,\n"
        "      \"resident_reload_boundary\": true\n"
        "    },\n"
        "    {\n"
        "      \"path\": \"bin/sparkpipe_release_manager\",\n"
        "      \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
        "      \"executable\": true\n"
        "    },\n"
        "    {\n"
        "      \"path\": \"lib/model_serving_adapter.so\",\n"
        "      \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
        "      \"resident_reload_boundary\": true\n"
        "    },\n"
        "    {\n"
        "      \"path\": \"lib/model_driver.so\",\n"
        "      \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
        "      \"resident_reload_boundary\": true\n"
        "    },\n"
        "    {\n"
        "      \"path\": \"lib/hidden_transport.so\",\n"
        "      \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"\n"
        "    },\n"
        "    {\n"
        "      \"path\": \"config/model_resident.json\",\n"
        "      \"sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\",\n"
        "      \"restart_on_change\": true\n"
        "    }\n"
        "  ],\n"
        "  \"roles\": [\n"
        "    {\n"
        "      \"name\": \"model_resident\",\n"
        "      \"selector\": \"rank\",\n"
        "      \"command\": \"bin/sparkpipe_model_residentd\",\n"
        "      \"argv\": [\n"
        "        \"--deployment\",\n"
        "        \"{install_root}/config/model_resident.json\",\n"
        "        \"--rank-index\",\n"
        "        \"{rank}\"\n"
        "      ],\n"
        "      \"env\": [\n"
        "        \"LD_LIBRARY_PATH={install_root}/lib:{install_root}/lib/runtime_libs\"\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n";

    if (path == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkWriteEntireFileAtomically(path,ExampleManifest,sizeof(ExampleManifest) - 1u);
}

SparkStatus SparkReleaseFormatResolvedCommandLine(
    const SparkReleaseResolvedRole *resolved_role,
    char *text,
    uint32_t text_bytes)
{
    uint32_t offset;
    uint32_t argument_index;
    SparkStatus status;

    if (resolved_role == 0 || text == 0 || text_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    text[0] = '\0';
    offset = 0u;
    status = SparkReleaseAppendText(text,text_bytes,&offset,resolved_role->command);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (argument_index = 0u; argument_index < resolved_role->argument_count; ++argument_index)
    {
        status = SparkReleaseAppendText(text,text_bytes,&offset," ");
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkReleaseAppendText(text,text_bytes,&offset,resolved_role->arguments[argument_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}
