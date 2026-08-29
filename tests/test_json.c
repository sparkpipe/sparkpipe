#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_sha256.h"

int main(void)
{
    static const char ValidJson[] =
        "{\"name\":\"firmware\\nmodel\",\"unicode\":\"\\u03bb\",\"values\":[1,true,{\"x\":7}]}";
    static const char InvalidJson[] = "{\"value\":1,}";
    static const char DuplicateJson[] = "{\"name\":1,\"name\":2,\"values\":[]}";
    static const char UnknownJson[] = "{\"name\":1,\"values\":[],\"stale\":true}";
    static const char *const ExactMembers[] = {"name", "unicode", "values"};
    static const char *const ShortMembers[] = {"name", "values"};
    static const char ExpectedAbcSha256[] = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    SparkJsonDocument document;
    int32_t root_token;
    int32_t name_token;
    int32_t unicode_token;
    int32_t values_token;
    char *decoded_text;
    char sha256[SPARK_SHA256_HEX_BYTES];

    SparkJsonDocumentReset(&document);
    assert(SparkJsonParseText(ValidJson, strlen(ValidJson), &document) == SPARK_STATUS_OK);
    root_token = SparkJsonGetRootToken(&document);
    assert(SparkJsonTokenIsType(&document, root_token, SPARK_JSON_TOKEN_OBJECT));
    assert(SparkJsonValidateObjectMembersExact(&document, root_token, ExactMembers, 3u) == SPARK_STATUS_OK);
    assert(SparkJsonValidateObjectMembersExact(&document, root_token, ShortMembers, 2u) == SPARK_STATUS_SCHEMA_ERROR);

    name_token = SparkJsonFindObjectMember(&document, root_token, "name");
    assert(SparkJsonCopyString(&document, name_token, &decoded_text) == SPARK_STATUS_OK);
    assert(strcmp(decoded_text, "firmware\nmodel") == 0);
    free(decoded_text);

    unicode_token = SparkJsonFindObjectMember(&document, root_token, "unicode");
    assert(SparkJsonCopyString(&document, unicode_token, &decoded_text) == SPARK_STATUS_OK);
    assert(strcmp(decoded_text, "\xce\xbb") == 0);
    free(decoded_text);

    values_token = SparkJsonFindObjectMember(&document, root_token, "values");
    assert(SparkJsonGetArrayElementCount(&document, values_token) == 3u);
    SparkJsonDocumentDestroy(&document);

    /* Sequential array access walks the same tokens as indexed access
     * (the one-pass contract request-scale loops rely on). */
    {
        static const char ArrayJson[] = "{\"ids\":[7,13,0,999,42,1]}";
        SparkJsonDocumentReset(&document);
        assert(SparkJsonParseText(ArrayJson, strlen(ArrayJson), &document) == SPARK_STATUS_OK);
        root_token = SparkJsonGetRootToken(&document);
        values_token = SparkJsonFindObjectMember(&document, root_token, "ids");
        assert(SparkJsonGetArrayElementCount(&document, values_token) == 6u);
        {
            int32_t walk = SparkJsonGetArrayElementFirst(&document, values_token);
            uint32_t index;
            for (index = 0u; index < 6u; index++)
            {
                assert(walk == SparkJsonGetArrayElement(&document, values_token, index));
                walk = SparkJsonGetArrayElementNext(&document, values_token, walk);
            }
            assert(walk < 0);
        }
        assert(SparkJsonGetArrayElementFirst(&document, root_token) < 0);
        SparkJsonDocumentDestroy(&document);
    }

    SparkJsonDocumentReset(&document);
    assert(SparkJsonParseText(InvalidJson, strlen(InvalidJson), &document) == SPARK_STATUS_PARSE_ERROR);
    SparkJsonDocumentDestroy(&document);

    SparkJsonDocumentReset(&document);
    assert(SparkJsonParseText(DuplicateJson, strlen(DuplicateJson), &document) == SPARK_STATUS_OK);
    root_token = SparkJsonGetRootToken(&document);
    assert(SparkJsonValidateObjectMembersExact(&document, root_token, ShortMembers, 2u) == SPARK_STATUS_SCHEMA_ERROR);
    SparkJsonDocumentDestroy(&document);

    SparkJsonDocumentReset(&document);
    assert(SparkJsonParseText(UnknownJson, strlen(UnknownJson), &document) == SPARK_STATUS_OK);
    root_token = SparkJsonGetRootToken(&document);
    assert(SparkJsonValidateObjectMembersExact(&document, root_token, ShortMembers, 2u) == SPARK_STATUS_SCHEMA_ERROR);
    SparkJsonDocumentDestroy(&document);

    assert(SparkSha256Bytes("abc", 3u, sha256) == SPARK_STATUS_OK);
    assert(strcmp(sha256, ExpectedAbcSha256) == 0);
    assert(SparkSha256HexIsValid(sha256));
    return 0;
}
