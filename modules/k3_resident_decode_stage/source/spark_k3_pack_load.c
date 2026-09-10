#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t SparkK3PackReadU64Le(const uint8_t *p)
{
	return((uint64_t)p[0] | ((uint64_t)p[1] << 8u) | ((uint64_t)p[2] << 16u) |
		((uint64_t)p[3] << 24u) | ((uint64_t)p[4] << 32u) |
		((uint64_t)p[5] << 40u) | ((uint64_t)p[6] << 48u) |
		((uint64_t)p[7] << 56u));
}

static uint32_t SparkK3PackReadU32Le(const uint8_t *p)
{
	return((uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
		((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u));
}

static SparkStatus SparkK3PackReadConfig(const SparkJsonDocument *document,
	int32_t config_token, SparkK3PackConfig *config)
{
	static const char *const names[] =
	{
		"hidden","layers","first_layer","total_layers","experts","top_k",
		"latent","intermediate","group","vocab","kda_heads","kda_head",
		"heads","kv_lora","rope","v_head","nope","shared","q_lora"
	};
	uint32_t values[19];
	int32_t token;
	for ( uint32_t index = 0u; index < 19u; index++ )
	{
		token = SparkJsonFindObjectMember(document, config_token, names[index]);
		if ( token < 0 ||
			SparkJsonGetUInt32(document, token, &values[index]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	config->hidden = values[0];
	config->layers = values[1];
	config->first_layer = values[2];
	config->total_layers = values[3];
	config->experts = values[4];
	config->top_k = values[5];
	config->latent = values[6];
	config->intermediate = values[7];
	config->group = values[8];
	config->vocab = values[9];
	config->kda_heads = values[10];
	config->kda_head = values[11];
	config->heads = values[12];
	config->kv_lora = values[13];
	config->rope = values[14];
	config->v_head = values[15];
	config->nope = values[16];
	config->shared = values[17];
	config->q_lora = values[18];
	return(SPARK_STATUS_OK);
}

struct SparkK3PackPrivate
{
	SparkJsonDocument document;
};

SparkStatus SparkK3PackOpen(const char *path, SparkK3Pack *pack)
{
	SparkStatus status;
	struct stat stat_buffer;
	uint8_t header[16];
	uint8_t *manifest_text;
	uint32_t magic, version, manifest_bytes;
	uint64_t window_bytes, payload_base;
	int32_t fd, root, config_token;
	if ( pack == 0 || path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pack, 0, sizeof(*pack));
	pack->private_state = (struct SparkK3PackPrivate *)calloc(1u,
		sizeof(*pack->private_state));
	if ( pack->private_state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkJsonDocumentReset(&pack->private_state->document);
	fd = open(path, O_RDONLY);
	if ( fd < 0 )
	{
		SparkK3PackClose(pack);
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( fstat(fd, &stat_buffer) != 0 )
	{
		(void)close(fd);
		SparkK3PackClose(pack);
		return(SPARK_STATUS_IO_ERROR);
	}
	pack->file_bytes = (uint64_t)stat_buffer.st_size;
	if ( pack->file_bytes < 16u )
	{
		(void)close(fd);
		SparkK3PackClose(pack);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( pread(fd, header, 16u, 0) != 16 )
	{
		(void)close(fd);
		SparkK3PackClose(pack);
		return(SPARK_STATUS_IO_ERROR);
	}
	magic = SparkK3PackReadU32Le(header);
	version = SparkK3PackReadU32Le(header + 4u);
	manifest_bytes = (uint32_t)SparkK3PackReadU64Le(header + 8u);
	if ( magic != SPARK_K3_PACK_MAGIC ||
		version != SPARK_K3_PACK_FORMAT_VERSION ||
		16ull + (uint64_t)manifest_bytes > pack->file_bytes )
	{
		(void)close(fd);
		SparkK3PackClose(pack);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	window_bytes = 16ull + (uint64_t)manifest_bytes;
	manifest_text = (uint8_t *)malloc((size_t)window_bytes);
	if ( manifest_text == 0 )
	{
		(void)close(fd);
		SparkK3PackClose(pack);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( pread(fd, manifest_text, (size_t)window_bytes, 0) != (ssize_t)window_bytes )
	{
		free(manifest_text);
		(void)close(fd);
		SparkK3PackClose(pack);
		return(SPARK_STATUS_IO_ERROR);
	}
	(void)close(fd);
	pack->version = version;
	payload_base = window_bytes;
	payload_base += (SPARK_K3_PACK_ALIGNMENT -
		(payload_base % SPARK_K3_PACK_ALIGNMENT)) % SPARK_K3_PACK_ALIGNMENT;
	pack->payload_base = payload_base;
	status = SparkJsonParseText((const char *)manifest_text + 16u,
		(size_t)manifest_bytes, &pack->private_state->document);
	free(manifest_text);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3PackClose(pack);
		SPARK_RETURN(status);
	}
	root = SparkJsonGetRootToken(&pack->private_state->document);
	config_token = SparkJsonFindObjectMember(&pack->private_state->document, root, "config");
	status = config_token < 0 ? SPARK_STATUS_VALIDATION_FAILED :
		SparkK3PackReadConfig(&pack->private_state->document, config_token, &pack->config);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3PackClose(pack);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkK3PackClose(SparkK3Pack *pack)
{
	if ( pack == 0 )
		return;
	if ( pack->private_state != 0 )
	{
		SparkJsonDocumentDestroy(&pack->private_state->document);
		free(pack->private_state);
		pack->private_state = 0;
	}
}

SparkStatus SparkK3PackLoadEntry(SparkK3Pack *pack, const char *name,
	SparkK3PackEntry *entry)
{
	int32_t root, tensors_token, member_token, field_token, shape_token,
		element_token;
	uint64_t wide_value;
	if ( pack == 0 || name == 0 || entry == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(entry, 0, sizeof(*entry));
	snprintf(entry->name, sizeof(entry->name), "%s", name);
	root = SparkJsonGetRootToken(&pack->private_state->document);
	tensors_token = SparkJsonFindObjectMember(&pack->private_state->document, root, "tensors");
	if ( tensors_token < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	member_token = SparkJsonFindObjectMember(&pack->private_state->document, tensors_token, name);
	if ( member_token < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	field_token = SparkJsonFindObjectMember(&pack->private_state->document, member_token, "offset");
	if ( field_token < 0 ||
		SparkJsonGetUInt64(&pack->private_state->document, field_token, &wide_value) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	entry->payload_offset = wide_value;
	field_token = SparkJsonFindObjectMember(&pack->private_state->document, member_token, "bytes");
	if ( field_token < 0 ||
		SparkJsonGetUInt64(&pack->private_state->document, field_token, &wide_value) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	entry->bytes = wide_value;
	field_token = SparkJsonFindObjectMember(&pack->private_state->document, member_token, "kind");
	if ( field_token < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkJsonStringEquals(&pack->private_state->document, field_token, "bf16") )
		entry->kind = SPARK_K3_PACK_KIND_BF16;
	else if ( SparkJsonStringEquals(&pack->private_state->document, field_token, "f32") )
		entry->kind = SPARK_K3_PACK_KIND_F32;
	else if ( SparkJsonStringEquals(&pack->private_state->document, field_token, "mxfp4_ws_interleaved_v1") )
		entry->kind = SPARK_K3_PACK_KIND_MXFP4_WS_INTERLEAVED_V1;
	else
		return(SPARK_STATUS_VALIDATION_FAILED);
	shape_token = SparkJsonFindObjectMember(&pack->private_state->document, member_token, "shape");
	if ( shape_token < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	entry->shape_count = SparkJsonGetArrayElementCount(&pack->private_state->document, shape_token);
	if ( entry->shape_count > 4u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	for ( uint32_t i = 0u; i < entry->shape_count; i++ )
	{
		element_token = SparkJsonGetArrayElement(&pack->private_state->document, shape_token, i);
		if ( element_token < 0 ||
			SparkJsonGetUInt32(&pack->private_state->document, element_token, &entry->shape[i]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( pack->payload_base + entry->payload_offset + entry->bytes >
		pack->file_bytes )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3PackLoadInterleaveTileK(SparkK3Pack *pack,
	const char *name, uint32_t *tile_k)
{
	int32_t root, tensors_token, member_token, interleave_token, field_token;
	uint32_t value;
	if ( pack == 0 || name == 0 || tile_k == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	root = SparkJsonGetRootToken(&pack->private_state->document);
	tensors_token = SparkJsonFindObjectMember(&pack->private_state->document, root, "tensors");
	if ( tensors_token < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	member_token = SparkJsonFindObjectMember(&pack->private_state->document, tensors_token, name);
	if ( member_token < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	interleave_token = SparkJsonFindObjectMember(&pack->private_state->document, member_token, "interleave");
	if ( interleave_token < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	field_token = SparkJsonFindObjectMember(&pack->private_state->document, interleave_token, "tile_k");
	if ( field_token < 0 ||
		SparkJsonGetUInt32(&pack->private_state->document, field_token, &value) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*tile_k = value;
	return(SPARK_STATUS_OK);
}
