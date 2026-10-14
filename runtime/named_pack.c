#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct SparkNamedPackPrivate
{
	SparkJsonDocument document;
};

static uint64_t SparkNamedPackReadU64Le(const uint8_t *p)
{
	uint64_t value = 0u;
	for ( uint32_t index = 0u; index < 8u; index++ )
		value |= (uint64_t)p[index] << (8u * index);
	return(value);
}

static uint32_t SparkNamedPackReadU32Le(const uint8_t *p)
{
	return((uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u));
}

static SparkStatus SparkNamedPackReadManifest(const char *path, SparkNamedPack *pack, uint32_t magic,
	uint32_t version, uint32_t alignment)
{
	struct stat stat_buffer;
	uint8_t header[16];
	char *manifest_text;
	uint64_t manifest_bytes, base;
	SparkStatus status;
	int32_t fd = open(path, O_RDONLY);
	if ( fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( fstat(fd, &stat_buffer) != 0 || pread(fd, header, sizeof(header), 0) != (ssize_t)sizeof(header) )
	{
		(void)close(fd);
		return(SPARK_STATUS_IO_ERROR);
	}
	pack->file_bytes = (uint64_t)stat_buffer.st_size;
	pack->magic = SparkNamedPackReadU32Le(header);
	pack->version = SparkNamedPackReadU32Le(header + 4u);
	manifest_bytes = SparkNamedPackReadU64Le(header + 8u);
	if ( pack->magic != magic || pack->version != version || manifest_bytes == 0u ||
		sizeof(header) + manifest_bytes > pack->file_bytes )
	{
		(void)close(fd);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	manifest_text = (char *)malloc((size_t)manifest_bytes);
	if ( manifest_text == 0 )
	{
		(void)close(fd);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( pread(fd, manifest_text, (size_t)manifest_bytes, (off_t)sizeof(header)) != (ssize_t)manifest_bytes )
	{
		free(manifest_text);
		(void)close(fd);
		return(SPARK_STATUS_IO_ERROR);
	}
	(void)close(fd);
	base = sizeof(header) + manifest_bytes;
	pack->payload_base = base + (alignment - (base % alignment)) % alignment;
	status = SparkJsonParseText(manifest_text, (size_t)manifest_bytes, &pack->private_state->document);
	free(manifest_text);
	return(status);
}

SparkStatus SparkNamedPackOpen(const char *path, uint32_t magic, uint32_t version, uint32_t alignment, SparkNamedPack *pack)
{
	SparkStatus status;
	if ( pack == 0 || path == 0 || alignment == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pack, 0, sizeof(*pack));
	pack->private_state = (struct SparkNamedPackPrivate *)calloc(1u, sizeof(*pack->private_state));
	if ( pack->private_state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkJsonDocumentReset(&pack->private_state->document);
	status = SparkNamedPackReadManifest(path, pack, magic, version, alignment);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNamedPackClose(pack);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkNamedPackClose(SparkNamedPack *pack)
{
	if ( pack == 0 || pack->private_state == 0 )
		return;
	SparkJsonDocumentDestroy(&pack->private_state->document);
	free(pack->private_state);
	pack->private_state = 0;
}

static int32_t SparkNamedPackTensor(const SparkNamedPack *pack, const char *name)
{
	const SparkJsonDocument *document = &pack->private_state->document;
	int32_t tensors = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "tensors");
	return(tensors < 0 ? -1 : SparkJsonFindObjectMember(document, tensors, name));
}

SparkStatus SparkNamedPackEntryLoad(const SparkNamedPack *pack, const char *name, SparkNamedPackEntry *entry)
{
	const SparkJsonDocument *document;
	int32_t member, field, shape, element;
	char *kind = 0;
	if ( pack == 0 || pack->private_state == 0 || name == 0 || entry == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	document = &pack->private_state->document;
	memset(entry, 0, sizeof(*entry));
	if ( strlen(name) >= sizeof(entry->name) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(entry->name, name, strlen(name) + 1u);
	member = SparkNamedPackTensor(pack, name);
	if ( member < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	field = SparkJsonFindObjectMember(document, member, "offset");
	if ( field < 0 || SparkJsonGetUInt64(document, field, &entry->payload_offset) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	field = SparkJsonFindObjectMember(document, member, "bytes");
	if ( field < 0 || SparkJsonGetUInt64(document, field, &entry->bytes) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	field = SparkJsonFindObjectMember(document, member, "kind");
	if ( field < 0 || SparkJsonCopyString(document, field, &kind) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( strlen(kind) >= sizeof(entry->kind) )
	{
		free(kind);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	memcpy(entry->kind, kind, strlen(kind) + 1u);
	free(kind);
	shape = SparkJsonFindObjectMember(document, member, "shape");
	if ( shape < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	entry->shape_count = SparkJsonGetArrayElementCount(document, shape);
	if ( entry->shape_count > SPARK_NAMED_PACK_MAX_SHAPE )
		return(SPARK_STATUS_VALIDATION_FAILED);
	for ( uint32_t index = 0u; index < entry->shape_count; index++ )
	{
		element = SparkJsonGetArrayElement(document, shape, index);
		if ( element < 0 || SparkJsonGetUInt32(document, element, &entry->shape[index]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( pack->payload_base + entry->payload_offset + entry->bytes > pack->file_bytes )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNamedPackConfigU32(const SparkNamedPack *pack, const char *key, uint32_t *value)
{
	const SparkJsonDocument *document;
	int32_t config, token;
	if ( pack == 0 || pack->private_state == 0 || key == 0 || value == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	document = &pack->private_state->document;
	config = SparkJsonFindObjectMember(document, SparkJsonGetRootToken(document), "config");
	token = config < 0 ? -1 : SparkJsonFindObjectMember(document, config, key);
	if ( token < 0 || SparkJsonGetUInt32(document, token, value) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNamedPackTensorFieldU32(const SparkNamedPack *pack, const char *name, const char *object,
	const char *field, uint32_t *value)
{
	const SparkJsonDocument *document;
	int32_t member, inner, token;
	if ( pack == 0 || pack->private_state == 0 || name == 0 || object == 0 || field == 0 || value == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	document = &pack->private_state->document;
	member = SparkNamedPackTensor(pack, name);
	if ( member < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	inner = SparkJsonFindObjectMember(document, member, object);
	token = inner < 0 ? -1 : SparkJsonFindObjectMember(document, inner, field);
	if ( token < 0 || SparkJsonGetUInt32(document, token, value) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SPARK_STATUS_OK);
}
