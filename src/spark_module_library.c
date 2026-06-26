#include "sparkpipe/spark_module_library.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_module_abi.h"

#define SPARK_STATIC_ARCHIVE_MAGIC "!<arch>\n"
#define SPARK_THIN_ARCHIVE_MAGIC "!<thin>\n"
#define SPARK_ARCHIVE_MAGIC_BYTES 8u

static SparkStatus SparkModuleCopyJsonStringMember(
    const SparkJsonDocument *document,
    int32_t root_token_index,
    const char *member_name,
    char *destination,
    uint32_t destination_bytes,
    bool allow_empty,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    int32_t member_token_index;
    char *member_text;
    SparkStatus status;

    member_token_index = SparkJsonFindObjectMember(document, root_token_index, member_name);
    if (member_token_index < 0 ||
        !SparkJsonTokenIsType(document, member_token_index, SPARK_JSON_TOKEN_STRING))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record field '%s' is missing", member_name);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkJsonCopyString(document, member_token_index, &member_text);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!allow_empty && member_text[0] == '\0')
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record field '%s' is empty", member_name);
        free(member_text);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkCopyString(destination, destination_bytes, member_text);
    free(member_text);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record field '%s' is too long", member_name);
    }
    return status;
}

const char *SparkModuleLinkUnitKindToString(SparkModuleLinkUnitKind link_unit_kind)
{
    switch (link_unit_kind)
    {
        case SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT:
            return "relocatable_object";
        case SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE:
            return "static_archive";
        default:
            return "invalid";
    }
}

static SparkStatus SparkModuleParseLinkUnitKind(
    const char *text,
    SparkModuleLinkUnitKind *link_unit_kind)
{
    if (text == 0 || link_unit_kind == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (strcmp(text, "relocatable_object") == 0)
    {
        *link_unit_kind = SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT;
        return SPARK_STATUS_OK;
    }
    if (strcmp(text, "static_archive") == 0)
    {
        *link_unit_kind = SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_SCHEMA_ERROR;
}

static SparkStatus SparkModuleDetectLinkUnitKind(
    const char *path,
    SparkModuleLinkUnitKind *link_unit_kind,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    FILE *file;
    unsigned char magic[SPARK_ARCHIVE_MAGIC_BYTES];
    size_t magic_bytes;
    int close_result;

    if (path == 0 || link_unit_kind == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file = fopen(path, "rb");
    if (file == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot open module link unit '%s'", path);
        return SPARK_STATUS_IO_ERROR;
    }
    memset(magic, 0, sizeof(magic));
    magic_bytes = fread(magic, 1u, sizeof(magic), file);
    if (ferror(file) != 0)
    {
        fclose(file);
        SparkSetError(error_buffer, error_buffer_bytes, "cannot read module link unit '%s'", path);
        return SPARK_STATUS_IO_ERROR;
    }
    close_result = fclose(file);
    if (close_result != 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot close module link unit '%s'", path);
        return SPARK_STATUS_IO_ERROR;
    }
    if (magic_bytes == sizeof(magic) &&
        memcmp(magic, SPARK_THIN_ARCHIVE_MAGIC, sizeof(magic)) == 0)
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "thin archive '%s' is not self-contained and cannot be published",
            path);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (magic_bytes == sizeof(magic) &&
        memcmp(magic, SPARK_STATIC_ARCHIVE_MAGIC, sizeof(magic)) == 0)
    {
        *link_unit_kind = SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE;
    }
    else
    {
        *link_unit_kind = SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT;
    }
    return SPARK_STATUS_OK;
}

static const char *SparkModuleLinkUnitExtension(SparkModuleLinkUnitKind link_unit_kind)
{
    if (link_unit_kind == SPARK_MODULE_LINK_UNIT_STATIC_ARCHIVE)
    {
        return "a";
    }
    if (link_unit_kind == SPARK_MODULE_LINK_UNIT_RELOCATABLE_OBJECT)
    {
        return "o";
    }
    return 0;
}

void SparkModuleArtifactReset(SparkModuleArtifact *artifact)
{
    if (artifact == 0)
    {
        return;
    }
    memset(artifact, 0, sizeof(*artifact));
}

SparkStatus SparkLoadModuleArtifactRecord(
    const char *record_path,
    SparkModuleArtifact *artifact,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t member_token_index;
    char *link_unit_kind_text;
    char *validation_state;
    SparkStatus status;

    if (record_path == 0 || artifact == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    SparkModuleArtifactReset(artifact);
    SparkJsonDocumentReset(&document);

    status = SparkJsonLoadFile(record_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot parse module record '%s'", record_path);
        return status;
    }
    root_token_index = SparkJsonGetRootToken(&document);
    if (!SparkJsonTokenIsType(&document, root_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record root must be an object");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }

    member_token_index = SparkJsonFindObjectMember(&document, root_token_index, "schema_version");
    if (member_token_index < 0 ||
        SparkJsonGetUInt32(&document, member_token_index, &artifact->schema_version) != SPARK_STATUS_OK ||
        artifact->schema_version != SPARK_MODULE_ARTIFACT_SCHEMA_VERSION)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "unsupported module record schema_version");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    member_token_index = SparkJsonFindObjectMember(&document, root_token_index, "module_abi_version");
    if (member_token_index < 0 ||
        SparkJsonGetUInt32(&document, member_token_index, &artifact->module_abi_version) != SPARK_STATUS_OK ||
        artifact->module_abi_version == 0u)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record has invalid module_abi_version");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    member_token_index = SparkJsonFindObjectMember(&document, root_token_index, "link_unit_kind");
    if (member_token_index < 0 ||
        SparkJsonCopyString(&document, member_token_index, &link_unit_kind_text) != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record link_unit_kind is missing");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    status = SparkModuleParseLinkUnitKind(link_unit_kind_text, &artifact->link_unit_kind);
    free(link_unit_kind_text);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record has invalid link_unit_kind");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }

    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "module_id",
        artifact->module_id,
        sizeof(artifact->module_id),
        false,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "target",
        artifact->target,
        sizeof(artifact->target),
        false,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "artifact_sha256",
        artifact->artifact_sha256,
        sizeof(artifact->artifact_sha256),
        false,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "link_unit",
        artifact->link_unit_path,
        sizeof(artifact->link_unit_path),
        false,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "validation_recipe",
        artifact->validation_recipe,
        sizeof(artifact->validation_recipe),
        false,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "initialize_symbol",
        artifact->initialize_symbol,
        sizeof(artifact->initialize_symbol),
        true,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "execute_symbol",
        artifact->execute_symbol,
        sizeof(artifact->execute_symbol),
        false,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "admit_symbol",
        artifact->admit_symbol,
        sizeof(artifact->admit_symbol),
        true,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "snapshot_symbol",
        artifact->snapshot_symbol,
        sizeof(artifact->snapshot_symbol),
        true,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }
    status = SparkModuleCopyJsonStringMember(
        &document,
        root_token_index,
        "destroy_symbol",
        artifact->destroy_symbol,
        sizeof(artifact->destroy_symbol),
        true,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        goto cleanup;
    }

    member_token_index = SparkJsonFindObjectMember(&document, root_token_index, "validation_state");
    if (member_token_index < 0 ||
        SparkJsonCopyString(&document, member_token_index, &validation_state) != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record validation_state is missing");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    artifact->validated = strcmp(validation_state, "passed") == 0;
    free(validation_state);
    if (!artifact->validated)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record is not validated");
        status = SPARK_STATUS_MODULE_NOT_VALIDATED;
        goto cleanup;
    }
    if (!SparkSha256HexIsValid(artifact->artifact_sha256))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record artifact hash is invalid");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }
    if (!SparkCIdentifierIsValid(artifact->initialize_symbol, true) ||
        !SparkCIdentifierIsValid(artifact->execute_symbol, false) ||
        !SparkCIdentifierIsValid(artifact->admit_symbol, true) ||
        !SparkCIdentifierIsValid(artifact->snapshot_symbol, true) ||
        !SparkCIdentifierIsValid(artifact->destroy_symbol, true))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module record contains an invalid C symbol");
        status = SPARK_STATUS_SCHEMA_ERROR;
        goto cleanup;
    }

    status = SPARK_STATUS_OK;

cleanup:
    SparkJsonDocumentDestroy(&document);
    if (status != SPARK_STATUS_OK)
    {
        SparkModuleArtifactReset(artifact);
    }
    return status;
}

static SparkStatus SparkModuleComputeIdentityKey(
    const char *module_id,
    const char *target,
    char key[SPARK_SHA256_HEX_BYTES])
{
    SparkSha256Context hash_context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];

    if (module_id == 0 || target == 0 || key == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkSha256Initialize(&hash_context);
    SparkSha256Update(&hash_context, module_id, strlen(module_id));
    SparkSha256Update(&hash_context, "\n", 1u);
    SparkSha256Update(&hash_context, target, strlen(target));
    SparkSha256Finalize(&hash_context, digest);
    SparkSha256DigestToHex(digest, key);
    return SPARK_STATUS_OK;
}

static void SparkModuleHashValidationField(
    SparkSha256Context *hash_context,
    const char *field)
{
    const char *field_value;

    field_value = field != 0 ? field : "";
    SparkSha256Update(hash_context, field_value, strlen(field_value));
    SparkSha256Update(hash_context, "\n", 1u);
}

static SparkStatus SparkModuleComputeValidationKey(
    const SparkModulePublishRequest *request,
    SparkModuleLinkUnitKind link_unit_kind,
    const char *artifact_sha256,
    char key[SPARK_SHA256_HEX_BYTES])
{
    SparkSha256Context hash_context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    char module_abi_text[32];
    char validator_argument_count_text[32];
    uint32_t validator_argument_index;

    if (request == 0 || artifact_sha256 == 0 || key == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (snprintf(
            module_abi_text,
            sizeof(module_abi_text),
            "%u",
            request->module_abi_version) < 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (snprintf(
            validator_argument_count_text,
            sizeof(validator_argument_count_text),
            "%u",
            request->validator_argument_count) < 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkSha256Initialize(&hash_context);
    SparkModuleHashValidationField(&hash_context, request->module_id);
    SparkModuleHashValidationField(&hash_context, request->target);
    SparkModuleHashValidationField(&hash_context, artifact_sha256);
    SparkModuleHashValidationField(&hash_context, SparkModuleLinkUnitKindToString(link_unit_kind));
    SparkModuleHashValidationField(&hash_context, request->validation_recipe);
    SparkModuleHashValidationField(&hash_context, module_abi_text);
    SparkModuleHashValidationField(&hash_context, request->initialize_symbol);
    SparkModuleHashValidationField(&hash_context, request->execute_symbol);
    SparkModuleHashValidationField(&hash_context, request->admit_symbol);
    SparkModuleHashValidationField(&hash_context, request->snapshot_symbol);
    SparkModuleHashValidationField(&hash_context, request->destroy_symbol);
    SparkModuleHashValidationField(
        &hash_context,
        validator_argument_count_text);
    for (validator_argument_index = 0u;
         validator_argument_index < request->validator_argument_count;
         ++validator_argument_index)
    {
        SparkModuleHashValidationField(
            &hash_context,
            request->validator_arguments[validator_argument_index]);
    }
    SparkSha256Finalize(&hash_context, digest);
    SparkSha256DigestToHex(digest, key);
    return SPARK_STATUS_OK;
}

static char *SparkModuleEscapeJsonString(const char *text)
{
    size_t source_bytes;
    size_t destination_capacity;
    char *escaped_text;
    size_t source_index;
    size_t destination_index;

    if (text == 0)
    {
        return 0;
    }
    source_bytes = strlen(text);
    if (source_bytes > (SIZE_MAX - 1u) / 6u)
    {
        return 0;
    }
    destination_capacity = (source_bytes * 6u) + 1u;
    escaped_text = (char *)malloc(destination_capacity);
    if (escaped_text == 0)
    {
        return 0;
    }

    destination_index = 0u;
    for (source_index = 0u; source_index < source_bytes; ++source_index)
    {
        unsigned char character;

        character = (unsigned char)text[source_index];
        switch (character)
        {
            case '"':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = '"';
                break;
            case '\\':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = '\\';
                break;
            case '\b':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = 'b';
                break;
            case '\f':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = 'f';
                break;
            case '\n':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = 'n';
                break;
            case '\r':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = 'r';
                break;
            case '\t':
                escaped_text[destination_index++] = '\\';
                escaped_text[destination_index++] = 't';
                break;
            default:
            {
                if (character < 0x20u)
                {
                    static const char HexDigits[] = "0123456789abcdef";

                    escaped_text[destination_index++] = '\\';
                    escaped_text[destination_index++] = 'u';
                    escaped_text[destination_index++] = '0';
                    escaped_text[destination_index++] = '0';
                    escaped_text[destination_index++] = HexDigits[character >> 4u];
                    escaped_text[destination_index++] = HexDigits[character & 0x0fu];
                }
                else
                {
                    escaped_text[destination_index++] = (char)character;
                }
                break;
            }
        }
    }
    escaped_text[destination_index] = '\0';
    return escaped_text;
}

static SparkStatus SparkModuleFormatRecord(
    const SparkModulePublishRequest *request,
    SparkModuleLinkUnitKind link_unit_kind,
    const char *artifact_sha256,
    const char *relative_link_unit_path,
    char **record_text,
    size_t *record_bytes)
{
    char *module_id;
    char *target;
    char *link_unit_path;
    char *validation_recipe;
    char *initialize_symbol;
    char *execute_symbol;
    char *admit_symbol;
    char *snapshot_symbol;
    char *destroy_symbol;
    int formatted_bytes;
    char *buffer;

    if (request == 0 || artifact_sha256 == 0 || relative_link_unit_path == 0 ||
        record_text == 0 || record_bytes == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *record_text = 0;
    *record_bytes = 0u;

    module_id = SparkModuleEscapeJsonString(request->module_id);
    target = SparkModuleEscapeJsonString(request->target);
    link_unit_path = SparkModuleEscapeJsonString(relative_link_unit_path);
    validation_recipe = SparkModuleEscapeJsonString(request->validation_recipe);
    initialize_symbol = SparkModuleEscapeJsonString(
        request->initialize_symbol != 0 ? request->initialize_symbol : "");
    execute_symbol = SparkModuleEscapeJsonString(request->execute_symbol);
    admit_symbol = SparkModuleEscapeJsonString(
        request->admit_symbol != 0 ? request->admit_symbol : "");
    snapshot_symbol = SparkModuleEscapeJsonString(
        request->snapshot_symbol != 0 ? request->snapshot_symbol : "");
    destroy_symbol = SparkModuleEscapeJsonString(
        request->destroy_symbol != 0 ? request->destroy_symbol : "");
    if (module_id == 0 || target == 0 || link_unit_path == 0 || validation_recipe == 0 ||
        initialize_symbol == 0 || execute_symbol == 0 || admit_symbol == 0 ||
        snapshot_symbol == 0 || destroy_symbol == 0)
    {
        free(module_id);
        free(target);
        free(link_unit_path);
        free(validation_recipe);
        free(initialize_symbol);
        free(execute_symbol);
        free(admit_symbol);
        free(snapshot_symbol);
        free(destroy_symbol);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    formatted_bytes = snprintf(
        0,
        0,
        "{\n"
        "  \"schema_version\": %u,\n"
        "  \"module_id\": \"%s\",\n"
        "  \"target\": \"%s\",\n"
        "  \"module_abi_version\": %u,\n"
        "  \"artifact_sha256\": \"%s\",\n"
        "  \"link_unit_kind\": \"%s\",\n"
        "  \"link_unit\": \"%s\",\n"
        "  \"validation_recipe\": \"%s\",\n"
        "  \"validation_state\": \"passed\",\n"
        "  \"initialize_symbol\": \"%s\",\n"
        "  \"execute_symbol\": \"%s\",\n"
        "  \"admit_symbol\": \"%s\",\n"
        "  \"snapshot_symbol\": \"%s\",\n"
        "  \"destroy_symbol\": \"%s\"\n"
        "}\n",
        SPARK_MODULE_ARTIFACT_SCHEMA_VERSION,
        module_id,
        target,
        request->module_abi_version,
        artifact_sha256,
        SparkModuleLinkUnitKindToString(link_unit_kind),
        link_unit_path,
        validation_recipe,
        initialize_symbol,
        execute_symbol,
        admit_symbol,
        snapshot_symbol,
        destroy_symbol);
    if (formatted_bytes < 0)
    {
        free(module_id);
        free(target);
        free(link_unit_path);
        free(validation_recipe);
        free(initialize_symbol);
        free(execute_symbol);
        free(admit_symbol);
        free(snapshot_symbol);
        free(destroy_symbol);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    buffer = (char *)malloc((size_t)formatted_bytes + 1u);
    if (buffer == 0)
    {
        free(module_id);
        free(target);
        free(link_unit_path);
        free(validation_recipe);
        free(initialize_symbol);
        free(execute_symbol);
        free(admit_symbol);
        free(snapshot_symbol);
        free(destroy_symbol);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    snprintf(
        buffer,
        (size_t)formatted_bytes + 1u,
        "{\n"
        "  \"schema_version\": %u,\n"
        "  \"module_id\": \"%s\",\n"
        "  \"target\": \"%s\",\n"
        "  \"module_abi_version\": %u,\n"
        "  \"artifact_sha256\": \"%s\",\n"
        "  \"link_unit_kind\": \"%s\",\n"
        "  \"link_unit\": \"%s\",\n"
        "  \"validation_recipe\": \"%s\",\n"
        "  \"validation_state\": \"passed\",\n"
        "  \"initialize_symbol\": \"%s\",\n"
        "  \"execute_symbol\": \"%s\",\n"
        "  \"admit_symbol\": \"%s\",\n"
        "  \"snapshot_symbol\": \"%s\",\n"
        "  \"destroy_symbol\": \"%s\"\n"
        "}\n",
        SPARK_MODULE_ARTIFACT_SCHEMA_VERSION,
        module_id,
        target,
        request->module_abi_version,
        artifact_sha256,
        SparkModuleLinkUnitKindToString(link_unit_kind),
        link_unit_path,
        validation_recipe,
        initialize_symbol,
        execute_symbol,
        admit_symbol,
        snapshot_symbol,
        destroy_symbol);

    free(module_id);
    free(target);
    free(link_unit_path);
    free(validation_recipe);
    free(initialize_symbol);
    free(execute_symbol);
    free(admit_symbol);
    free(snapshot_symbol);
    free(destroy_symbol);
    *record_text = buffer;
    *record_bytes = (size_t)formatted_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkModuleValidatePublishRequest(
    const SparkModulePublishRequest *request,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    const char *initialize_symbol;
    const char *admit_symbol;
    const char *snapshot_symbol;
    const char *destroy_symbol;
    uint32_t validator_argument_index;

    if (request == 0 || request->library_root == 0 || request->module_id == 0 ||
        request->target == 0 || request->link_unit_path == 0 ||
        request->validation_recipe == 0 || request->execute_symbol == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->library_root[0] == '\0' || request->module_id[0] == '\0' ||
        request->target[0] == '\0' || request->link_unit_path[0] == '\0' ||
        request->validation_recipe[0] == '\0' || request->execute_symbol[0] == '\0' ||
        request->module_abi_version == 0u)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module publish request contains an empty required field");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    initialize_symbol = request->initialize_symbol != 0 ? request->initialize_symbol : "";
    admit_symbol = request->admit_symbol != 0 ? request->admit_symbol : "";
    snapshot_symbol = request->snapshot_symbol != 0 ? request->snapshot_symbol : "";
    destroy_symbol = request->destroy_symbol != 0 ? request->destroy_symbol : "";
    if (strlen(request->module_id) >= SPARK_MODULE_ID_BYTES ||
        strlen(request->target) >= SPARK_MODULE_TARGET_BYTES ||
        strlen(request->validation_recipe) >= SPARK_MODULE_RECIPE_BYTES ||
        strlen(initialize_symbol) >= SPARK_MODULE_SYMBOL_BYTES ||
        strlen(request->execute_symbol) >= SPARK_MODULE_SYMBOL_BYTES ||
        strlen(admit_symbol) >= SPARK_MODULE_SYMBOL_BYTES ||
        strlen(snapshot_symbol) >= SPARK_MODULE_SYMBOL_BYTES ||
        strlen(destroy_symbol) >= SPARK_MODULE_SYMBOL_BYTES)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module publish request exceeds an artifact field capacity");
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (!SparkCIdentifierIsValid(initialize_symbol, true) ||
        !SparkCIdentifierIsValid(request->execute_symbol, false) ||
        !SparkCIdentifierIsValid(admit_symbol, true) ||
        !SparkCIdentifierIsValid(snapshot_symbol, true) ||
        !SparkCIdentifierIsValid(destroy_symbol, true))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module publish request contains an invalid C symbol");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->validator_argument_count != 0u && request->validator_arguments == 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module validator arguments are missing");
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (validator_argument_index = 0u;
         validator_argument_index < request->validator_argument_count;
         ++validator_argument_index)
    {
        if (request->validator_arguments[validator_argument_index] == 0)
        {
            SparkSetError(
                error_buffer,
                error_buffer_bytes,
                "module validator argument %u is null",
                validator_argument_index);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!SparkPathExists(request->link_unit_path))
    {
        SparkSetError(error_buffer, error_buffer_bytes, "module link unit '%s' does not exist", request->link_unit_path);
        return SPARK_STATUS_NOT_FOUND;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkModuleRunValidator(
    const SparkModulePublishRequest *request,
    const char *stored_link_unit_path,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char **arguments;
    uint32_t argument_index;
    int exit_code;
    SparkStatus status;

    if (request->validator_path == 0 || request->validator_path[0] == '\0')
    {
        SparkSetError(error_buffer, error_buffer_bytes, "new module artifacts require a validator executable");
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    arguments = (char **)calloc(
        (size_t)request->validator_argument_count + 3u,
        sizeof(*arguments));
    if (arguments == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    arguments[0] = (char *)request->validator_path;
    for (argument_index = 0u;
         argument_index < request->validator_argument_count;
         ++argument_index)
    {
        arguments[argument_index + 1u] = (char *)request->validator_arguments[argument_index];
    }
    arguments[request->validator_argument_count + 1u] = (char *)stored_link_unit_path;
    arguments[request->validator_argument_count + 2u] = 0;

    status = SparkRunProcess(request->validator_path, arguments, &exit_code);
    free(arguments);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot execute validator '%s'", request->validator_path);
        return status;
    }
    if (exit_code != 0)
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "validator '%s' failed with exit code %d",
            request->validator_path,
            exit_code);
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    return SPARK_STATUS_OK;
}

static bool SparkModuleRecordMatchesRequest(
    const SparkModuleArtifact *artifact,
    const SparkModulePublishRequest *request,
    SparkModuleLinkUnitKind link_unit_kind,
    const char *artifact_sha256)
{
    return artifact != 0 && request != 0 && artifact_sha256 != 0 &&
           artifact->schema_version == SPARK_MODULE_ARTIFACT_SCHEMA_VERSION &&
           artifact->module_abi_version == request->module_abi_version &&
           artifact->link_unit_kind == link_unit_kind &&
           strcmp(artifact->module_id, request->module_id) == 0 &&
           strcmp(artifact->target, request->target) == 0 &&
           strcmp(artifact->artifact_sha256, artifact_sha256) == 0 &&
           strcmp(artifact->validation_recipe, request->validation_recipe) == 0 &&
           strcmp(
               artifact->initialize_symbol,
               request->initialize_symbol != 0 ? request->initialize_symbol : "") == 0 &&
           strcmp(artifact->execute_symbol, request->execute_symbol) == 0 &&
           strcmp(
               artifact->admit_symbol,
               request->admit_symbol != 0 ? request->admit_symbol : "") == 0 &&
           strcmp(
               artifact->snapshot_symbol,
               request->snapshot_symbol != 0 ? request->snapshot_symbol : "") == 0 &&
           strcmp(
               artifact->destroy_symbol,
               request->destroy_symbol != 0 ? request->destroy_symbol : "") == 0 &&
           artifact->validated;
}

SparkStatus SparkPublishValidatedModule(
    const SparkModulePublishRequest *request,
    SparkModulePublishReport *report,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char identity_key[SPARK_SHA256_HEX_BYTES];
    char validation_key[SPARK_SHA256_HEX_BYTES];
    char link_units_directory[SPARK_MODULE_PATH_BYTES];
    char records_directory[SPARK_MODULE_PATH_BYTES];
    char active_directory[SPARK_MODULE_PATH_BYTES];
    char relative_link_unit_path[SPARK_MODULE_PATH_BYTES];
    char immutable_record_name[SPARK_MODULE_PATH_BYTES];
    char active_record_name[SPARK_MODULE_PATH_BYTES];
    char *record_text;
    size_t record_bytes;
    const char *link_unit_extension;
    SparkModuleArtifact existing_artifact;
    SparkStatus status;

    if (report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(report, 0, sizeof(*report));
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    status = SparkModuleValidatePublishRequest(request, error_buffer, error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModuleDetectLinkUnitKind(
        request->link_unit_path,
        &report->link_unit_kind,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    link_unit_extension = SparkModuleLinkUnitExtension(report->link_unit_kind);
    if (link_unit_extension == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkSha256File(request->link_unit_path, report->artifact_sha256);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot hash module link unit '%s'", request->link_unit_path);
        return status;
    }
    status = SparkModuleComputeIdentityKey(request->module_id, request->target, identity_key);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkModuleComputeValidationKey(
        request,
        report->link_unit_kind,
        report->artifact_sha256,
        validation_key);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (SparkJoinPath(
            request->library_root,
            "link_units",
            link_units_directory,
            sizeof(link_units_directory)) != SPARK_STATUS_OK ||
        SparkJoinPath(
            request->library_root,
            "records",
            records_directory,
            sizeof(records_directory)) != SPARK_STATUS_OK ||
        SparkJoinPath(
            request->library_root,
            "active",
            active_directory,
            sizeof(active_directory)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (SparkCreateDirectories(link_units_directory) != SPARK_STATUS_OK ||
        SparkCreateDirectories(records_directory) != SPARK_STATUS_OK ||
        SparkCreateDirectories(active_directory) != SPARK_STATUS_OK)
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "cannot create module library directories under '%s'",
            request->library_root);
        return SPARK_STATUS_IO_ERROR;
    }

    if (snprintf(
            relative_link_unit_path,
            sizeof(relative_link_unit_path),
            "link_units/%s.%s",
            report->artifact_sha256,
            link_unit_extension) >= (int)sizeof(relative_link_unit_path) ||
        SparkJoinPath(
            request->library_root,
            relative_link_unit_path,
            report->stored_link_unit_path,
            sizeof(report->stored_link_unit_path)) != SPARK_STATUS_OK ||
        snprintf(
            immutable_record_name,
            sizeof(immutable_record_name),
            "%s-%s.json",
            identity_key,
            validation_key) >= (int)sizeof(immutable_record_name) ||
        SparkJoinPath(
            records_directory,
            immutable_record_name,
            report->immutable_record_path,
            sizeof(report->immutable_record_path)) != SPARK_STATUS_OK ||
        snprintf(
            active_record_name,
            sizeof(active_record_name),
            "%s.json",
            identity_key) >= (int)sizeof(active_record_name) ||
        SparkJoinPath(
            active_directory,
            active_record_name,
            report->active_record_path,
            sizeof(report->active_record_path)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    if (!SparkPathExists(report->stored_link_unit_path))
    {
        status = SparkCopyFile(request->link_unit_path, report->stored_link_unit_path);
        if (status != SPARK_STATUS_OK)
        {
            SparkSetError(
                error_buffer,
                error_buffer_bytes,
                "cannot store module link unit '%s'",
                report->stored_link_unit_path);
            return status;
        }
    }
    {
        char stored_hash[SPARK_SHA256_HEX_BYTES];
        SparkModuleLinkUnitKind stored_kind;

        status = SparkSha256File(report->stored_link_unit_path, stored_hash);
        if (status != SPARK_STATUS_OK || strcmp(stored_hash, report->artifact_sha256) != 0)
        {
            unlink(report->stored_link_unit_path);
            SparkSetError(
                error_buffer,
                error_buffer_bytes,
                "stored module link-unit hash does not match its content address");
            return SPARK_STATUS_HASH_MISMATCH;
        }
        status = SparkModuleDetectLinkUnitKind(
            report->stored_link_unit_path,
            &stored_kind,
            error_buffer,
            error_buffer_bytes);
        if (status != SPARK_STATUS_OK || stored_kind != report->link_unit_kind)
        {
            unlink(report->stored_link_unit_path);
            SparkSetError(error_buffer, error_buffer_bytes, "stored module link-unit kind mismatch");
            return SPARK_STATUS_HASH_MISMATCH;
        }
    }

    SparkModuleArtifactReset(&existing_artifact);
    if (SparkPathExists(report->immutable_record_path))
    {
        status = SparkLoadModuleArtifactRecord(
            report->immutable_record_path,
            &existing_artifact,
            error_buffer,
            error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkModuleRecordMatchesRequest(
                &existing_artifact,
                request,
                report->link_unit_kind,
                report->artifact_sha256))
        {
            SparkSetError(
                error_buffer,
                error_buffer_bytes,
                "existing immutable module record does not match the requested artifact contract");
            return SPARK_STATUS_HASH_MISMATCH;
        }
        report->validation_reused = true;
    }
    else
    {
        status = SparkModuleRunValidator(
            request,
            report->stored_link_unit_path,
            error_buffer,
            error_buffer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        report->validation_reused = false;
    }
    {
        char validated_hash[SPARK_SHA256_HEX_BYTES];
        SparkModuleLinkUnitKind validated_kind;

        status = SparkSha256File(report->stored_link_unit_path, validated_hash);
        if (status != SPARK_STATUS_OK || strcmp(validated_hash, report->artifact_sha256) != 0)
        {
            unlink(report->stored_link_unit_path);
            SparkSetError(
                error_buffer,
                error_buffer_bytes,
                "validator modified the content-addressed module link unit");
            return SPARK_STATUS_HASH_MISMATCH;
        }
        status = SparkModuleDetectLinkUnitKind(
            report->stored_link_unit_path,
            &validated_kind,
            error_buffer,
            error_buffer_bytes);
        if (status != SPARK_STATUS_OK || validated_kind != report->link_unit_kind)
        {
            unlink(report->stored_link_unit_path);
            SparkSetError(error_buffer, error_buffer_bytes, "validator changed the module link-unit kind");
            return SPARK_STATUS_HASH_MISMATCH;
        }
    }
    if (chmod(report->stored_link_unit_path, S_IRUSR | S_IRGRP | S_IROTH) != 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot make validated module link unit read-only");
        return SPARK_STATUS_IO_ERROR;
    }

    status = SparkModuleFormatRecord(
        request,
        report->link_unit_kind,
        report->artifact_sha256,
        relative_link_unit_path,
        &record_text,
        &record_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkPathExists(report->immutable_record_path))
    {
        status = SparkWriteEntireFileAtomically(
            report->immutable_record_path,
            record_text,
            record_bytes);
        if (status != SPARK_STATUS_OK)
        {
            free(record_text);
            SparkSetError(error_buffer, error_buffer_bytes, "cannot write immutable module record");
            return status;
        }
    }
    status = SparkWriteEntireFileAtomically(
        report->active_record_path,
        record_text,
        record_bytes);
    free(record_text);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "cannot activate validated module record");
    }
    return status;
}

SparkStatus SparkResolveValidatedModule(
    const char *library_root,
    const char *module_id,
    const char *target,
    SparkModuleArtifact *artifact,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    char identity_key[SPARK_SHA256_HEX_BYTES];
    char active_directory[SPARK_MODULE_PATH_BYTES];
    char active_record_name[SPARK_MODULE_PATH_BYTES];
    char active_record_path[SPARK_MODULE_PATH_BYTES];
    char resolved_link_unit_path[SPARK_MODULE_PATH_BYTES];
    char actual_hash[SPARK_SHA256_HEX_BYTES];
    SparkModuleLinkUnitKind actual_kind;
    SparkStatus status;

    if (library_root == 0 || module_id == 0 || target == 0 || artifact == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        error_buffer[0] = '\0';
    }
    status = SparkModuleComputeIdentityKey(module_id, target, identity_key);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkJoinPath(
            library_root,
            "active",
            active_directory,
            sizeof(active_directory)) != SPARK_STATUS_OK ||
        snprintf(
            active_record_name,
            sizeof(active_record_name),
            "%s.json",
            identity_key) >= (int)sizeof(active_record_name) ||
        SparkJoinPath(
            active_directory,
            active_record_name,
            active_record_path,
            sizeof(active_record_path)) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (!SparkPathExists(active_record_path))
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "validated module '%s' for target '%s' is not in the library",
            module_id,
            target);
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    status = SparkLoadModuleArtifactRecord(
        active_record_path,
        artifact,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (strcmp(artifact->module_id, module_id) != 0 ||
        strcmp(artifact->target, target) != 0)
    {
        SparkSetError(error_buffer, error_buffer_bytes, "active module record identity mismatch");
        SparkModuleArtifactReset(artifact);
        return SPARK_STATUS_HASH_MISMATCH;
    }

    if (artifact->link_unit_path[0] == '/')
    {
        status = SparkCopyString(
            resolved_link_unit_path,
            sizeof(resolved_link_unit_path),
            artifact->link_unit_path);
    }
    else
    {
        status = SparkJoinPath(
            library_root,
            artifact->link_unit_path,
            resolved_link_unit_path,
            sizeof(resolved_link_unit_path));
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkModuleArtifactReset(artifact);
        return status;
    }
    status = SparkSha256File(resolved_link_unit_path, actual_hash);
    if (status != SPARK_STATUS_OK)
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "validated module link unit '%s' is missing",
            resolved_link_unit_path);
        SparkModuleArtifactReset(artifact);
        return SPARK_STATUS_IO_ERROR;
    }
    if (strcmp(actual_hash, artifact->artifact_sha256) != 0)
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "validated module link-unit hash mismatch for '%s'",
            module_id);
        SparkModuleArtifactReset(artifact);
        return SPARK_STATUS_HASH_MISMATCH;
    }
    status = SparkModuleDetectLinkUnitKind(
        resolved_link_unit_path,
        &actual_kind,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK || actual_kind != artifact->link_unit_kind)
    {
        SparkSetError(
            error_buffer,
            error_buffer_bytes,
            "validated module link-unit kind mismatch for '%s'",
            module_id);
        SparkModuleArtifactReset(artifact);
        return SPARK_STATUS_HASH_MISMATCH;
    }
    status = SparkCopyString(
        artifact->link_unit_path,
        sizeof(artifact->link_unit_path),
        resolved_link_unit_path);
    if (status != SPARK_STATUS_OK)
    {
        SparkModuleArtifactReset(artifact);
    }
    return status;
}
