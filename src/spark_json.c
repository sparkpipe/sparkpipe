#include "sparkpipe/spark_json.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"

#define SPARK_JSON_INITIAL_TOKEN_CAPACITY 256u
#define SPARK_JSON_MAX_NESTING_DEPTH 256u

typedef struct SparkJsonParser
{
    const char *text;
    size_t text_bytes;
    size_t position;
    SparkJsonToken *tokens;
    uint32_t token_count;
    uint32_t token_capacity;
} SparkJsonParser;

static void SparkJsonSkipWhitespace(SparkJsonParser *parser)
{
    while (parser->position < parser->text_bytes)
    {
        char character;

        character = parser->text[parser->position];
        if (character != ' ' && character != '\t' && character != '\r' && character != '\n')
        {
            break;
        }
        parser->position += 1u;
    }
}

static SparkStatus SparkJsonAllocateToken(
    SparkJsonParser *parser,
    SparkJsonTokenType type,
    int32_t start,
    int32_t parent,
    int32_t *token_index)
{
    SparkJsonToken *resized_tokens;
    uint32_t resized_capacity;
    SparkJsonToken *token;

    if (parser == 0 || token_index == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (parser->token_count == parser->token_capacity)
    {
        if (parser->token_capacity > UINT32_MAX / 2u)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        resized_capacity = parser->token_capacity == 0u ? SPARK_JSON_INITIAL_TOKEN_CAPACITY : parser->token_capacity * 2u;
        resized_tokens = (SparkJsonToken *)realloc(parser->tokens, (size_t)resized_capacity * sizeof(*resized_tokens));
        if (resized_tokens == 0)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        parser->tokens = resized_tokens;
        parser->token_capacity = resized_capacity;
    }

    *token_index = (int32_t)parser->token_count;
    token = &parser->tokens[parser->token_count++];
    token->type = type;
    token->start = start;
    token->end = -1;
    token->parent = parent;
    token->child_count = 0u;
    if (parent >= 0)
    {
        parser->tokens[parent].child_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static bool SparkJsonCharacterIsHex(char character)
{
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
}

static SparkStatus SparkJsonParseStringToken(SparkJsonParser *parser, int32_t parent, int32_t *token_index)
{
    SparkStatus status;
    size_t string_start;

    if (parser == 0 || token_index == 0 || parser->position >= parser->text_bytes || parser->text[parser->position] != '"')
    {
        return SPARK_STATUS_PARSE_ERROR;
    }

    parser->position += 1u;
    string_start = parser->position;
    while (parser->position < parser->text_bytes)
    {
        char character;

        character = parser->text[parser->position];
        if (character == '"')
        {
            status = SparkJsonAllocateToken(parser, SPARK_JSON_TOKEN_STRING, (int32_t)string_start, parent, token_index);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            parser->tokens[*token_index].end = (int32_t)parser->position;
            parser->position += 1u;
            return SPARK_STATUS_OK;
        }
        if ((unsigned char)character < 0x20u)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (character == '\\')
        {
            parser->position += 1u;
            if (parser->position >= parser->text_bytes)
            {
                return SPARK_STATUS_PARSE_ERROR;
            }
            character = parser->text[parser->position];
            if (character == 'u')
            {
                uint32_t hex_index;

                if (parser->position + 4u >= parser->text_bytes)
                {
                    return SPARK_STATUS_PARSE_ERROR;
                }
                for (hex_index = 1u; hex_index <= 4u; ++hex_index)
                {
                    if (!SparkJsonCharacterIsHex(parser->text[parser->position + hex_index]))
                    {
                        return SPARK_STATUS_PARSE_ERROR;
                    }
                }
                parser->position += 4u;
            }
            else if (strchr("\"\\/bfnrt", character) == 0)
            {
                return SPARK_STATUS_PARSE_ERROR;
            }
        }
        parser->position += 1u;
    }
    return SPARK_STATUS_PARSE_ERROR;
}

static bool SparkJsonPrimitiveDelimiter(char character)
{
    return character == ' ' || character == '\t' || character == '\r' || character == '\n' ||
           character == ',' || character == ']' || character == '}';
}

static bool SparkJsonValidateNumber(const char *text, size_t text_bytes)
{
    size_t position;

    if (text == 0 || text_bytes == 0u)
    {
        return false;
    }
    position = 0u;
    if (text[position] == '-')
    {
        position += 1u;
        if (position == text_bytes)
        {
            return false;
        }
    }
    if (text[position] == '0')
    {
        position += 1u;
    }
    else
    {
        if (text[position] < '1' || text[position] > '9')
        {
            return false;
        }
        while (position < text_bytes && text[position] >= '0' && text[position] <= '9')
        {
            position += 1u;
        }
    }
    if (position < text_bytes && text[position] == '.')
    {
        position += 1u;
        if (position == text_bytes || text[position] < '0' || text[position] > '9')
        {
            return false;
        }
        while (position < text_bytes && text[position] >= '0' && text[position] <= '9')
        {
            position += 1u;
        }
    }
    if (position < text_bytes && (text[position] == 'e' || text[position] == 'E'))
    {
        position += 1u;
        if (position < text_bytes && (text[position] == '+' || text[position] == '-'))
        {
            position += 1u;
        }
        if (position == text_bytes || text[position] < '0' || text[position] > '9')
        {
            return false;
        }
        while (position < text_bytes && text[position] >= '0' && text[position] <= '9')
        {
            position += 1u;
        }
    }
    return position == text_bytes;
}

static SparkStatus SparkJsonParsePrimitiveToken(SparkJsonParser *parser, int32_t parent, int32_t *token_index)
{
    size_t primitive_start;
    size_t primitive_bytes;
    SparkStatus status;

    if (parser == 0 || token_index == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    primitive_start = parser->position;
    while (parser->position < parser->text_bytes && !SparkJsonPrimitiveDelimiter(parser->text[parser->position]))
    {
        parser->position += 1u;
    }
    primitive_bytes = parser->position - primitive_start;
    if (primitive_bytes == 0u)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    if (!((primitive_bytes == 4u && memcmp(parser->text + primitive_start, "true", 4u) == 0) ||
          (primitive_bytes == 5u && memcmp(parser->text + primitive_start, "false", 5u) == 0) ||
          (primitive_bytes == 4u && memcmp(parser->text + primitive_start, "null", 4u) == 0) ||
          SparkJsonValidateNumber(parser->text + primitive_start, primitive_bytes)))
    {
        return SPARK_STATUS_PARSE_ERROR;
    }

    status = SparkJsonAllocateToken(parser, SPARK_JSON_TOKEN_PRIMITIVE, (int32_t)primitive_start, parent, token_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    parser->tokens[*token_index].end = (int32_t)parser->position;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkJsonParseValue(SparkJsonParser *parser, int32_t parent, uint32_t depth, int32_t *token_index);

static SparkStatus SparkJsonParseObject(SparkJsonParser *parser, int32_t parent, uint32_t depth, int32_t *token_index)
{
    SparkStatus status;
    int32_t object_token_index;

    if (depth > SPARK_JSON_MAX_NESTING_DEPTH)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkJsonAllocateToken(parser, SPARK_JSON_TOKEN_OBJECT, (int32_t)parser->position, parent, &object_token_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    parser->position += 1u;
    SparkJsonSkipWhitespace(parser);
    if (parser->position < parser->text_bytes && parser->text[parser->position] == '}')
    {
        parser->position += 1u;
        parser->tokens[object_token_index].end = (int32_t)parser->position;
        *token_index = object_token_index;
        return SPARK_STATUS_OK;
    }

    while (parser->position < parser->text_bytes)
    {
        int32_t key_token_index;
        int32_t value_token_index;

        if (parser->text[parser->position] != '"')
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        status = SparkJsonParseStringToken(parser, object_token_index, &key_token_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        (void)key_token_index;
        SparkJsonSkipWhitespace(parser);
        if (parser->position >= parser->text_bytes || parser->text[parser->position] != ':')
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        parser->position += 1u;
        SparkJsonSkipWhitespace(parser);
        status = SparkJsonParseValue(parser, object_token_index, depth + 1u, &value_token_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        (void)value_token_index;
        SparkJsonSkipWhitespace(parser);
        if (parser->position >= parser->text_bytes)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (parser->text[parser->position] == '}')
        {
            parser->position += 1u;
            parser->tokens[object_token_index].end = (int32_t)parser->position;
            *token_index = object_token_index;
            return SPARK_STATUS_OK;
        }
        if (parser->text[parser->position] != ',')
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        parser->position += 1u;
        SparkJsonSkipWhitespace(parser);
    }
    return SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkJsonParseArray(SparkJsonParser *parser, int32_t parent, uint32_t depth, int32_t *token_index)
{
    SparkStatus status;
    int32_t array_token_index;

    if (depth > SPARK_JSON_MAX_NESTING_DEPTH)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkJsonAllocateToken(parser, SPARK_JSON_TOKEN_ARRAY, (int32_t)parser->position, parent, &array_token_index);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    parser->position += 1u;
    SparkJsonSkipWhitespace(parser);
    if (parser->position < parser->text_bytes && parser->text[parser->position] == ']')
    {
        parser->position += 1u;
        parser->tokens[array_token_index].end = (int32_t)parser->position;
        *token_index = array_token_index;
        return SPARK_STATUS_OK;
    }

    while (parser->position < parser->text_bytes)
    {
        int32_t value_token_index;

        status = SparkJsonParseValue(parser, array_token_index, depth + 1u, &value_token_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        (void)value_token_index;
        SparkJsonSkipWhitespace(parser);
        if (parser->position >= parser->text_bytes)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (parser->text[parser->position] == ']')
        {
            parser->position += 1u;
            parser->tokens[array_token_index].end = (int32_t)parser->position;
            *token_index = array_token_index;
            return SPARK_STATUS_OK;
        }
        if (parser->text[parser->position] != ',')
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        parser->position += 1u;
        SparkJsonSkipWhitespace(parser);
    }
    return SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkJsonParseValue(SparkJsonParser *parser, int32_t parent, uint32_t depth, int32_t *token_index)
{
    if (parser == 0 || token_index == 0 || parser->position >= parser->text_bytes)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    switch (parser->text[parser->position])
    {
        case '{':
        {
            return SparkJsonParseObject(parser, parent, depth, token_index);
        }
        case '[':
        {
            return SparkJsonParseArray(parser, parent, depth, token_index);
        }
        case '"':
        {
            return SparkJsonParseStringToken(parser, parent, token_index);
        }
        default:
        {
            return SparkJsonParsePrimitiveToken(parser, parent, token_index);
        }
    }
}

void SparkJsonDocumentReset(SparkJsonDocument *document)
{
    if (document == 0)
    {
        return;
    }
    memset(document, 0, sizeof(*document));
}

void SparkJsonDocumentDestroy(SparkJsonDocument *document)
{
    if (document == 0)
    {
        return;
    }
    free(document->text);
    free(document->tokens);
    SparkJsonDocumentReset(document);
}

SparkStatus SparkJsonParseText(const char *text, size_t text_bytes, SparkJsonDocument *document)
{
    SparkJsonParser parser;
    SparkStatus status;
    int32_t root_token_index;
    char *owned_text;

    if (text == 0 || document == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkJsonDocumentDestroy(document);

    owned_text = (char *)malloc(text_bytes + 1u);
    if (owned_text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(owned_text, text, text_bytes);
    owned_text[text_bytes] = '\0';

    memset(&parser, 0, sizeof(parser));
    parser.text = owned_text;
    parser.text_bytes = text_bytes;
    SparkJsonSkipWhitespace(&parser);
    status = SparkJsonParseValue(&parser, -1, 0u, &root_token_index);
    if (status == SPARK_STATUS_OK)
    {
        SparkJsonSkipWhitespace(&parser);
        if (parser.position != parser.text_bytes || root_token_index != 0)
        {
            status = SPARK_STATUS_PARSE_ERROR;
        }
    }
    if (status != SPARK_STATUS_OK)
    {
        free(owned_text);
        free(parser.tokens);
        return status;
    }

    document->text = owned_text;
    document->text_bytes = text_bytes;
    document->tokens = parser.tokens;
    document->token_count = parser.token_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkJsonLoadFile(const char *path, SparkJsonDocument *document)
{
    char *file_text;
    size_t file_bytes;
    SparkStatus status;

    if (path == 0 || document == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkReadEntireFile(path, &file_text, &file_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkJsonParseText(file_text, file_bytes, document);
    free(file_text);
    return status;
}

int32_t SparkJsonGetRootToken(const SparkJsonDocument *document)
{
    if (document == 0 || document->token_count == 0u)
    {
        return -1;
    }
    return 0;
}

bool SparkJsonTokenIsType(const SparkJsonDocument *document, int32_t token_index, SparkJsonTokenType expected_type)
{
    return document != 0 && token_index >= 0 && (uint32_t)token_index < document->token_count && document->tokens[token_index].type == expected_type;
}

static int32_t SparkJsonFindNextDirectChild(const SparkJsonDocument *document, int32_t parent_token_index, int32_t after_token_index)
{
    uint32_t token_index;

    if (document == 0 || parent_token_index < 0)
    {
        return -1;
    }
    token_index = after_token_index < 0 ? 0u : (uint32_t)after_token_index + 1u;
    while (token_index < document->token_count)
    {
        if (document->tokens[token_index].parent == parent_token_index)
        {
            return (int32_t)token_index;
        }
        token_index += 1u;
    }
    return -1;
}

int32_t SparkJsonFindObjectMember(const SparkJsonDocument *document, int32_t object_token_index, const char *member_name)
{
    int32_t key_token_index;

    if (!SparkJsonTokenIsType(document, object_token_index, SPARK_JSON_TOKEN_OBJECT) || member_name == 0)
    {
        return -1;
    }

    key_token_index = SparkJsonFindNextDirectChild(document, object_token_index, object_token_index);
    while (key_token_index >= 0)
    {
        int32_t value_token_index;

        if (!SparkJsonTokenIsType(document, key_token_index, SPARK_JSON_TOKEN_STRING))
        {
            return -1;
        }
        value_token_index = SparkJsonFindNextDirectChild(document, object_token_index, key_token_index);
        if (value_token_index < 0)
        {
            return -1;
        }
        if (SparkJsonStringEquals(document, key_token_index, member_name))
        {
            return value_token_index;
        }
        key_token_index = SparkJsonFindNextDirectChild(document, object_token_index, value_token_index);
    }
    return -1;
}

uint32_t SparkJsonGetArrayElementCount(const SparkJsonDocument *document, int32_t array_token_index)
{
    if (!SparkJsonTokenIsType(document, array_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return 0u;
    }
    return document->tokens[array_token_index].child_count;
}

int32_t SparkJsonGetArrayElement(const SparkJsonDocument *document, int32_t array_token_index, uint32_t element_index)
{
    int32_t child_token_index;
    uint32_t current_element;

    if (!SparkJsonTokenIsType(document, array_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return -1;
    }
    child_token_index = SparkJsonFindNextDirectChild(document, array_token_index, array_token_index);
    current_element = 0u;
    while (child_token_index >= 0)
    {
        if (current_element == element_index)
        {
            return child_token_index;
        }
        current_element += 1u;
        child_token_index = SparkJsonFindNextDirectChild(document, array_token_index, child_token_index);
    }
    return -1;
}

static uint32_t SparkJsonHexValue(char character)
{
    if (character >= '0' && character <= '9')
    {
        return (uint32_t)(character - '0');
    }
    if (character >= 'a' && character <= 'f')
    {
        return 10u + (uint32_t)(character - 'a');
    }
    return 10u + (uint32_t)(character - 'A');
}

static uint32_t SparkJsonParseUnicodeEscape(const char *text)
{
    return (SparkJsonHexValue(text[0]) << 12u) |
           (SparkJsonHexValue(text[1]) << 8u) |
           (SparkJsonHexValue(text[2]) << 4u) |
           SparkJsonHexValue(text[3]);
}

static uint32_t SparkJsonAppendUtf8(char *destination, uint32_t destination_offset, uint32_t code_point)
{
    if (code_point <= 0x7fu)
    {
        destination[destination_offset++] = (char)code_point;
    }
    else if (code_point <= 0x7ffu)
    {
        destination[destination_offset++] = (char)(0xc0u | (code_point >> 6u));
        destination[destination_offset++] = (char)(0x80u | (code_point & 0x3fu));
    }
    else if (code_point <= 0xffffu)
    {
        destination[destination_offset++] = (char)(0xe0u | (code_point >> 12u));
        destination[destination_offset++] = (char)(0x80u | ((code_point >> 6u) & 0x3fu));
        destination[destination_offset++] = (char)(0x80u | (code_point & 0x3fu));
    }
    else
    {
        destination[destination_offset++] = (char)(0xf0u | (code_point >> 18u));
        destination[destination_offset++] = (char)(0x80u | ((code_point >> 12u) & 0x3fu));
        destination[destination_offset++] = (char)(0x80u | ((code_point >> 6u) & 0x3fu));
        destination[destination_offset++] = (char)(0x80u | (code_point & 0x3fu));
    }
    return destination_offset;
}

SparkStatus SparkJsonCopyString(const SparkJsonDocument *document, int32_t token_index, char **text)
{
    const SparkJsonToken *token;
    char *decoded_text;
    uint32_t source_offset;
    uint32_t destination_offset;
    uint32_t source_bytes;

    if (!SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_STRING) || text == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *text = 0;
    token = &document->tokens[token_index];
    source_bytes = (uint32_t)(token->end - token->start);
    decoded_text = (char *)malloc((size_t)source_bytes + 1u);
    if (decoded_text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    source_offset = 0u;
    destination_offset = 0u;
    while (source_offset < source_bytes)
    {
        char character;

        character = document->text[token->start + (int32_t)source_offset++];
        if (character != '\\')
        {
            decoded_text[destination_offset++] = character;
            continue;
        }

        character = document->text[token->start + (int32_t)source_offset++];
        switch (character)
        {
            case '"': decoded_text[destination_offset++] = '"'; break;
            case '\\': decoded_text[destination_offset++] = '\\'; break;
            case '/': decoded_text[destination_offset++] = '/'; break;
            case 'b': decoded_text[destination_offset++] = '\b'; break;
            case 'f': decoded_text[destination_offset++] = '\f'; break;
            case 'n': decoded_text[destination_offset++] = '\n'; break;
            case 'r': decoded_text[destination_offset++] = '\r'; break;
            case 't': decoded_text[destination_offset++] = '\t'; break;
            case 'u':
            {
                uint32_t code_point;

                code_point = SparkJsonParseUnicodeEscape(document->text + token->start + (int32_t)source_offset);
                source_offset += 4u;
                if (code_point >= 0xd800u && code_point <= 0xdbffu)
                {
                    uint32_t low_surrogate;

                    if (source_offset + 6u > source_bytes ||
                        document->text[token->start + (int32_t)source_offset] != '\\' ||
                        document->text[token->start + (int32_t)source_offset + 1] != 'u')
                    {
                        free(decoded_text);
                        return SPARK_STATUS_PARSE_ERROR;
                    }
                    low_surrogate = SparkJsonParseUnicodeEscape(document->text + token->start + (int32_t)source_offset + 2);
                    if (low_surrogate < 0xdc00u || low_surrogate > 0xdfffu)
                    {
                        free(decoded_text);
                        return SPARK_STATUS_PARSE_ERROR;
                    }
                    source_offset += 6u;
                    code_point = 0x10000u + (((code_point - 0xd800u) << 10u) | (low_surrogate - 0xdc00u));
                }
                else if (code_point >= 0xdc00u && code_point <= 0xdfffu)
                {
                    free(decoded_text);
                    return SPARK_STATUS_PARSE_ERROR;
                }
                destination_offset = SparkJsonAppendUtf8(decoded_text, destination_offset, code_point);
                break;
            }
            default:
            {
                free(decoded_text);
                return SPARK_STATUS_PARSE_ERROR;
            }
        }
    }
    decoded_text[destination_offset] = '\0';
    *text = decoded_text;
    return SPARK_STATUS_OK;
}

bool SparkJsonStringEquals(const SparkJsonDocument *document, int32_t token_index, const char *expected_text)
{
    char *decoded_text;
    bool equal;

    if (expected_text == 0 || SparkJsonCopyString(document, token_index, &decoded_text) != SPARK_STATUS_OK)
    {
        return false;
    }
    equal = strcmp(decoded_text, expected_text) == 0;
    free(decoded_text);
    return equal;
}

SparkStatus SparkJsonCopyRawValue(const SparkJsonDocument *document, int32_t token_index, char **text, uint32_t *text_bytes)
{
    const SparkJsonToken *token;
    uint32_t value_bytes;
    char *copied_text;

    if (document == 0 || token_index < 0 || (uint32_t)token_index >= document->token_count || text == 0 || text_bytes == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *text = 0;
    *text_bytes = 0u;
    token = &document->tokens[token_index];
    if (token->start < 0 || token->end < token->start)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    value_bytes = (uint32_t)(token->end - token->start);
    copied_text = (char *)malloc((size_t)value_bytes + 1u);
    if (copied_text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(copied_text, document->text + token->start, value_bytes);
    copied_text[value_bytes] = '\0';
    *text = copied_text;
    *text_bytes = value_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkJsonCopyPrimitive(const SparkJsonDocument *document, int32_t token_index, char **primitive)
{
    uint32_t primitive_bytes;

    if (!SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_PRIMITIVE))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkJsonCopyRawValue(document, token_index, primitive, &primitive_bytes);
}

SparkStatus SparkJsonGetUInt64(const SparkJsonDocument *document, int32_t token_index, uint64_t *value)
{
    char *primitive;
    char *end;
    unsigned long long parsed_value;
    SparkStatus status;

    if (value == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkJsonCopyPrimitive(document, token_index, &primitive);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (primitive[0] == '-')
    {
        free(primitive);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    errno = 0;
    end = 0;
    parsed_value = strtoull(primitive, &end, 10);
    if (errno != 0 || end == primitive || *end != '\0')
    {
        free(primitive);
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    free(primitive);
    *value = (uint64_t)parsed_value;
    return SPARK_STATUS_OK;
}

SparkStatus SparkJsonGetUInt32(const SparkJsonDocument *document, int32_t token_index, uint32_t *value)
{
    uint64_t parsed_value;
    SparkStatus status;

    if (value == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkJsonGetUInt64(document, token_index, &parsed_value);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (parsed_value > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *value = (uint32_t)parsed_value;
    return SPARK_STATUS_OK;
}

SparkStatus SparkJsonGetBoolean(const SparkJsonDocument *document, int32_t token_index, bool *value)
{
    const SparkJsonToken *token;
    size_t primitive_bytes;

    if (!SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_PRIMITIVE) || value == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    token = &document->tokens[token_index];
    primitive_bytes = (size_t)(token->end - token->start);
    if (primitive_bytes == 4u && memcmp(document->text + token->start, "true", 4u) == 0)
    {
        *value = true;
        return SPARK_STATUS_OK;
    }
    if (primitive_bytes == 5u && memcmp(document->text + token->start, "false", 5u) == 0)
    {
        *value = false;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_SCHEMA_ERROR;
}
