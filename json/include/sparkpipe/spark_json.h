#ifndef SPARKPIPE_SPARK_JSON_H
#define SPARKPIPE_SPARK_JSON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkJsonTokenType
{
    SPARK_JSON_TOKEN_UNDEFINED = 0,
    SPARK_JSON_TOKEN_OBJECT,
    SPARK_JSON_TOKEN_ARRAY,
    SPARK_JSON_TOKEN_STRING,
    SPARK_JSON_TOKEN_PRIMITIVE
} SparkJsonTokenType;

typedef struct SparkJsonToken
{
    SparkJsonTokenType type;
    int32_t start;
    int32_t end;
    int32_t parent;
    uint32_t child_count;
} SparkJsonToken;

typedef struct SparkJsonDocument
{
    char *text;
    size_t text_bytes;
    SparkJsonToken *tokens;
    uint32_t token_count;
} SparkJsonDocument;

void SparkJsonDocumentReset(SparkJsonDocument *document);
void SparkJsonDocumentDestroy(SparkJsonDocument *document);
SparkStatus SparkJsonParseText(const char *text, size_t text_bytes, SparkJsonDocument *document);
SparkStatus SparkJsonLoadFile(const char *path, SparkJsonDocument *document);
int32_t SparkJsonGetRootToken(const SparkJsonDocument *document);
int32_t SparkJsonFindObjectMember(const SparkJsonDocument *document, int32_t object_token_index, const char *member_name);
SparkStatus SparkJsonValidateObjectMembersExact(const SparkJsonDocument *document, int32_t object_token_index, const char *const *member_names, uint32_t member_count);
uint32_t SparkJsonGetArrayElementCount(const SparkJsonDocument *document, int32_t array_token_index);
int32_t SparkJsonGetArrayElement(const SparkJsonDocument *document, int32_t array_token_index, uint32_t element_index);
bool SparkJsonTokenIsType(const SparkJsonDocument *document, int32_t token_index, SparkJsonTokenType expected_type);
bool SparkJsonStringEquals(const SparkJsonDocument *document, int32_t token_index, const char *expected_text);
SparkStatus SparkJsonCopyString(const SparkJsonDocument *document, int32_t token_index, char **text);
SparkStatus SparkJsonCopyRawValue(const SparkJsonDocument *document, int32_t token_index, char **text, uint32_t *text_bytes);
SparkStatus SparkJsonGetUInt32(const SparkJsonDocument *document, int32_t token_index, uint32_t *value);
SparkStatus SparkJsonGetUInt64(const SparkJsonDocument *document, int32_t token_index, uint64_t *value);
SparkStatus SparkJsonGetBoolean(const SparkJsonDocument *document, int32_t token_index, bool *value);
/* Convenience used by every model serving adapter: look up a named object
 * member and read it as an unsigned integer. A missing member is a schema
 * error (the adapters' long-standing private-helper semantics). */
SparkStatus SparkJsonGetUInt32Member(const SparkJsonDocument *document, int32_t object_token_index, const char *member_name, uint32_t *value);

#ifdef __cplusplus
}
#endif

#endif
