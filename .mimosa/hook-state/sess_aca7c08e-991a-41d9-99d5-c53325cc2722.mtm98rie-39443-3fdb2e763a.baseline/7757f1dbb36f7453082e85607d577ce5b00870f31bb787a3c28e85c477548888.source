#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_k3_dspark_pack.h"
#include "sparkpipe/spark_json.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spark_k3_dspark_format.h"

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
	uint32_t magic, version, manifest_bytes;
	uint64_t payload_base;
	int32_t root, config_token;
	if ( pack == 0 || path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pack, 0, sizeof(*pack));
	pack->fd = -1;
	pack->private_state = (struct SparkK3PackPrivate *)calloc(1u,
		sizeof(*pack->private_state));
	if ( pack->private_state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkJsonDocumentReset(&pack->private_state->document);
	pack->fd = open(path, O_RDONLY);
	if ( pack->fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( fstat(pack->fd, &stat_buffer) != 0 )
	{
		SparkK3PackClose(pack);
		return(SPARK_STATUS_IO_ERROR);
	}
	pack->file_bytes = (uint64_t)stat_buffer.st_size;
	if ( pack->file_bytes < 16u )
	{
		SparkK3PackClose(pack);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	pack->mapping = (uint8_t *)mmap(0, (size_t)pack->file_bytes, PROT_READ,
		MAP_PRIVATE, pack->fd, 0);
	if ( pack->mapping == MAP_FAILED )
	{
		pack->mapping = 0;
		SparkK3PackClose(pack);
		return(SPARK_STATUS_IO_ERROR);
	}
	magic = SparkK3PackReadU32Le(pack->mapping);
	version = SparkK3PackReadU32Le(pack->mapping + 4u);
	manifest_bytes = (uint32_t)SparkK3PackReadU64Le(pack->mapping + 8u);
	if ( magic != SPARK_K3_PACK_MAGIC ||
		version != SPARK_K3_PACK_FORMAT_VERSION ||
		16ull + (uint64_t)manifest_bytes > pack->file_bytes )
	{
		SparkK3PackClose(pack);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	pack->version = version;
	payload_base = 16ull + (uint64_t)manifest_bytes;
	payload_base += (SPARK_K3_PACK_ALIGNMENT -
		(payload_base % SPARK_K3_PACK_ALIGNMENT)) % SPARK_K3_PACK_ALIGNMENT;
	pack->payload_base = payload_base;
	status = SparkJsonParseText((const char *)pack->mapping + 16u,
		(size_t)manifest_bytes, &pack->private_state->document);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3PackClose(pack);
		return(status);
	}
	root = SparkJsonGetRootToken(&pack->private_state->document);
	config_token = SparkJsonFindObjectMember(&pack->private_state->document, root, "config");
	status = config_token < 0 ? SPARK_STATUS_VALIDATION_FAILED :
		SparkK3PackReadConfig(&pack->private_state->document, config_token, &pack->config);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3PackClose(pack);
		return(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkK3PackClose(SparkK3Pack *pack)
{
	if ( pack == 0 )
		return;
	if ( pack->mapping != 0 )
		(void)munmap(pack->mapping, (size_t)pack->file_bytes);
	pack->mapping = 0;
	if ( pack->fd >= 0 )
		(void)close(pack->fd);
	pack->fd = -1;
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

const void *SparkK3PackPayload(const SparkK3Pack *pack,
	const SparkK3PackEntry *entry)
{
	if ( pack == 0 || entry == 0 || pack->mapping == 0 )
		return(0);
	return(pack->mapping + pack->payload_base + entry->payload_offset);
}


typedef struct SparkK3DsparkEntryRaw
{
	uint32_t kind;
	uint32_t layer;
	uint32_t format;
	uint32_t rows;
	uint32_t columns;
	uint64_t payload_offset;
	uint64_t payload_bytes;
} SparkK3DsparkEntryRaw;

static uint32_t SparkK3DsparkEntryU32(const uint8_t *base, uint32_t index)
{
	return(SparkK3PackReadU32Le(base + index * 4u));
}

static void SparkK3DsparkRefuse(char *refusal, uint32_t refusal_bytes,
	const char *text)
{
	if ( refusal != 0 && refusal_bytes != 0 )
		(void)snprintf(refusal, refusal_bytes, "%s", text);
}

static SparkStatus SparkK3DsparkCheckEntry(const SparkK3DsparkPack *pack,
	const SparkK3DsparkEntryRaw *entry, uint32_t entry_index,
	char *refusal, uint32_t refusal_bytes)
{
	uint32_t rows, columns;
	char text[SPARK_K3_DSPARK_MAX_REFUSAL_BYTES];
	if ( entry->format != SPARK_K3_DSPARK_WEIGHT_BF16 )
	{
		(void)snprintf(text,sizeof(text),"tensor %u: format %u is not BF16(0)",
			entry_index,entry->format);
		SparkK3DsparkRefuse(refusal,refusal_bytes,text);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->layer == 0xFFFFFFFFu )
	{
		if ( entry->kind <= SPARK_K3_DSPARK_TENSOR_FFN_DOWN ||
			entry->kind == SPARK_K3_DSPARK_TENSOR_RESERVED_14 )
		{
			(void)snprintf(text,sizeof(text),"tensor %u: kind %u is per-layer, "
				"entry is global",entry_index,entry->kind);
			SparkK3DsparkRefuse(refusal,refusal_bytes,text);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	else
	{
		if ( entry->kind > SPARK_K3_DSPARK_TENSOR_FFN_DOWN ||
			entry->layer >= pack->layer_count )
		{
			(void)snprintf(text,sizeof(text),"tensor %u: kind %u layer %u "
				"outside the per-layer range",entry_index,entry->kind,
				entry->layer);
			SparkK3DsparkRefuse(refusal,refusal_bytes,text);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	SparkK3DsparkKindShape(entry->kind,&rows,&columns);
	if ( entry->rows != rows || entry->columns != columns )
	{
		(void)snprintf(text,sizeof(text),"tensor %u kind %u: shape [%u,%u] "
			"vs format [%u,%u]",entry_index,entry->kind,entry->rows,
			entry->columns,rows,columns);
		SparkK3DsparkRefuse(refusal,refusal_bytes,text);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->payload_offset % SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT != 0u ||
		entry->payload_bytes != (uint64_t)rows * columns * 2ull ||
		entry->payload_offset + entry->payload_bytes > pack->file_bytes )
	{
		(void)snprintf(text,sizeof(text),"tensor %u kind %u: payload "
			"alignment/bounds inconsistent",entry_index,entry->kind);
		SparkK3DsparkRefuse(refusal,refusal_bytes,text);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3DsparkPackBind(const char *path, SparkK3DsparkPack *pack,
	char *refusal, uint32_t refusal_bytes)
{
	struct stat stat_buffer;
	uint32_t magic, version, header_bytes, entry_bytes, index, tap_index;
	uint8_t *ext;
	const uint8_t *cursor;
	char text[SPARK_K3_DSPARK_MAX_REFUSAL_BYTES];
	SparkK3DsparkEntryRaw entry;
	SparkStatus status;
	if ( pack == 0 || path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pack, 0, sizeof(*pack));
	pack->fd = -1;
	if ( refusal != 0 && refusal_bytes != 0 )
		refusal[0] = '\0';
	pack->fd = open(path, O_RDONLY);
	if ( pack->fd < 0 )
	{
		SparkK3DsparkRefuse(refusal,refusal_bytes,"drafter pack: open failed");
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( fstat(pack->fd, &stat_buffer) != 0 )
	{
		SparkK3DsparkPackRelease(pack);
		SparkK3DsparkRefuse(refusal,refusal_bytes,"drafter pack: fstat failed");
		return(SPARK_STATUS_IO_ERROR);
	}
	pack->file_bytes = (uint64_t)stat_buffer.st_size;
	if ( pack->file_bytes < SPARK_K3_DSPARK_HEADER_BYTES )
	{
		SparkK3DsparkPackRelease(pack);
		SparkK3DsparkRefuse(refusal,refusal_bytes,"drafter pack: shorter than a header");
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	pack->mapping = (uint8_t *)mmap(0, (size_t)pack->file_bytes, PROT_READ,
		MAP_PRIVATE, pack->fd, 0);
	if ( pack->mapping == MAP_FAILED )
	{
		pack->mapping = 0;
		SparkK3DsparkPackRelease(pack);
		SparkK3DsparkRefuse(refusal,refusal_bytes,"drafter pack: mmap failed");
		return(SPARK_STATUS_IO_ERROR);
	}
	magic = SparkK3PackReadU32Le(pack->mapping);
	version = SparkK3PackReadU32Le(pack->mapping + 4u);
	header_bytes = SparkK3PackReadU32Le(pack->mapping + 8u);
	entry_bytes = SparkK3PackReadU32Le(pack->mapping + 12u);
	pack->tensor_count = SparkK3PackReadU32Le(pack->mapping + 16u);
	if ( magic != SPARK_K3_DSPARK_MAGIC )
	{
		SparkK3DsparkPackRelease(pack);
		SparkK3DsparkRefuse(refusal,refusal_bytes,"drafter pack: magic is not K3DS");
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( version != SPARK_K3_DSPARK_FORMAT_VERSION ||
		header_bytes != SPARK_K3_DSPARK_HEADER_BYTES ||
		entry_bytes != SPARK_K3_DSPARK_ENTRY_BYTES )
	{
		SparkK3DsparkPackRelease(pack);
		SparkK3DsparkRefuse(refusal,refusal_bytes,"drafter pack: header layout mismatch");
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( pack->file_bytes != SparkK3PackReadU64Le(pack->mapping + 112u) )
	{
		SparkK3DsparkPackRelease(pack);
		SparkK3DsparkRefuse(refusal,refusal_bytes,
			"drafter pack: actual size != header file_bytes");
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	pack->hidden = SparkK3PackReadU32Le(pack->mapping + 20u);
	pack->layer_count = SparkK3PackReadU32Le(pack->mapping + 24u);
	pack->query_heads = SparkK3PackReadU32Le(pack->mapping + 64u);
	pack->kv_heads = SparkK3PackReadU32Le(pack->mapping + 68u);
	pack->head_dim = SparkK3PackReadU32Le(pack->mapping + 72u);
	pack->ffn_dimension = SparkK3PackReadU32Le(pack->mapping + 80u);
	pack->vocab = SparkK3PackReadU32Le(pack->mapping + 84u);
	pack->block_size = SparkK3PackReadU32Le(pack->mapping + 88u);
	pack->draft_token_count = pack->block_size >= 1u ? pack->block_size - 1u : 0u;
	ext = pack->mapping + SPARK_K3_DSPARK_CORE_HEADER_BYTES;
	for ( tap_index = 0u; tap_index < 5u; tap_index++ )
		pack->target_tap_layers[tap_index] =
			SparkK3DsparkEntryU32(ext, SPARK_K3_DSPARK_EXT_TAP_LAYER_0 + tap_index);
	pack->markov_rank = SparkK3DsparkEntryU32(ext, SPARK_K3_DSPARK_EXT_MARKOV_RANK);
	pack->mask_token_id = SparkK3DsparkEntryU32(ext, SPARK_K3_DSPARK_EXT_MASK_TOKEN_ID);
	pack->sliding_window = SparkK3DsparkEntryU32(ext, SPARK_K3_DSPARK_EXT_SLIDING_WINDOW);
	pack->flags = SparkK3DsparkEntryU32(ext, SPARK_K3_DSPARK_EXT_FLAGS);
	pack->confidence_input_dimension =
		SparkK3DsparkEntryU32(ext, SPARK_K3_DSPARK_EXT_CONFIDENCE_INPUT_DIM);
	#define SPARK_K3_DSPARK_BIND_CHECK(field, expected, label) \
		do { if ( (field) != (expected) ) { \
			(void)snprintf(text,sizeof(text),"drafter pack: %s %u != pinned %u", \
				label,(uint32_t)(field),(uint32_t)(expected)); \
			SparkK3DsparkRefuse(refusal,refusal_bytes,text); \
			SparkK3DsparkPackRelease(pack); \
			return(SPARK_STATUS_VALIDATION_FAILED); } } while (0)
	SPARK_K3_DSPARK_BIND_CHECK(pack->hidden, K3_HIDDEN, "hidden");
	SPARK_K3_DSPARK_BIND_CHECK(pack->layer_count, SPARK_K3_DSPARK_LAYER_COUNT, "layer_count");
	SPARK_K3_DSPARK_BIND_CHECK(pack->query_heads, SPARK_K3_DSPARK_ATTN_QUERY_HEADS, "query_heads");
	SPARK_K3_DSPARK_BIND_CHECK(pack->kv_heads, SPARK_K3_DSPARK_ATTN_KV_HEADS, "kv_heads");
	SPARK_K3_DSPARK_BIND_CHECK(pack->head_dim, SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION, "head_dim");
	SPARK_K3_DSPARK_BIND_CHECK(pack->ffn_dimension, SPARK_K3_DSPARK_FFN_INTERMEDIATE, "ffn");
	SPARK_K3_DSPARK_BIND_CHECK(pack->vocab, SPARK_K3_DSPARK_VOCAB, "vocab");
	SPARK_K3_DSPARK_BIND_CHECK(pack->block_size, SPARK_K3_DSPARK_BLOCK_SIZE, "block_size");
	SPARK_K3_DSPARK_BIND_CHECK(pack->markov_rank, SPARK_K3_DSPARK_MARKOV_RANK, "markov_rank");
	SPARK_K3_DSPARK_BIND_CHECK(pack->mask_token_id, SPARK_K3_DSPARK_MASK_TOKEN_ID, "mask_token_id");
	SPARK_K3_DSPARK_BIND_CHECK(pack->confidence_input_dimension,
		SPARK_K3_DSPARK_CONFIDENCE_INPUT_DIMENSION, "confidence_input_dim");
	SPARK_K3_DSPARK_BIND_CHECK(pack->target_tap_layers[0], SPARK_K3_DSPARK_TARGET_TAP_LAYER_0, "tap_layer_0");
	SPARK_K3_DSPARK_BIND_CHECK(pack->target_tap_layers[1], SPARK_K3_DSPARK_TARGET_TAP_LAYER_1, "tap_layer_1");
	SPARK_K3_DSPARK_BIND_CHECK(pack->target_tap_layers[2], SPARK_K3_DSPARK_TARGET_TAP_LAYER_2, "tap_layer_2");
	SPARK_K3_DSPARK_BIND_CHECK(pack->target_tap_layers[3], SPARK_K3_DSPARK_TARGET_TAP_LAYER_3, "tap_layer_3");
	SPARK_K3_DSPARK_BIND_CHECK(pack->target_tap_layers[4], SPARK_K3_DSPARK_TARGET_TAP_LAYER_4, "tap_layer_4");
	#undef SPARK_K3_DSPARK_BIND_CHECK
	if ( (pack->flags & SPARK_K3_DSPARK_FLAGS_REQUIRED) !=
		SPARK_K3_DSPARK_FLAGS_REQUIRED )
	{
		(void)snprintf(text,sizeof(text),"drafter pack: flags %#x missing one of "
			"embed|lm_head|confidence+markov (%#x)",pack->flags,
			SPARK_K3_DSPARK_FLAGS_REQUIRED);
		SparkK3DsparkRefuse(refusal,refusal_bytes,text);
		SparkK3DsparkPackRelease(pack);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( pack->tensor_count != SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT )
	{
		(void)snprintf(text,sizeof(text),"drafter pack: tensor_count %u != "
			"pinned inventory %u",pack->tensor_count,
			SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT);
		SparkK3DsparkRefuse(refusal,refusal_bytes,text);
		SparkK3DsparkPackRelease(pack);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	pack->entries = pack->mapping + SPARK_K3_DSPARK_HEADER_BYTES;
	cursor = pack->entries;
	for ( index = 0u; index < pack->tensor_count; index++ )
	{
		entry.kind = SparkK3PackReadU32Le(cursor);
		entry.layer = SparkK3PackReadU32Le(cursor + 4u);
		entry.format = SparkK3PackReadU32Le(cursor + 8u);
		entry.rows = SparkK3PackReadU32Le(cursor + 12u);
		entry.columns = SparkK3PackReadU32Le(cursor + 16u);
		entry.payload_offset = SparkK3PackReadU64Le(cursor + 24u);
		entry.payload_bytes = SparkK3PackReadU64Le(cursor + 32u);
		status = SparkK3DsparkCheckEntry(pack,&entry,index,refusal,refusal_bytes);
		if ( status != SPARK_STATUS_OK )
		{
			SparkK3DsparkPackRelease(pack);
			return(status);
		}
		cursor += SPARK_K3_DSPARK_ENTRY_BYTES;
	}
	return(SPARK_STATUS_OK);
}

void SparkK3DsparkPackRelease(SparkK3DsparkPack *pack)
{
	if ( pack == 0 )
		return;
	if ( pack->mapping != 0 )
		(void)munmap(pack->mapping, (size_t)pack->file_bytes);
	pack->mapping = 0;
	if ( pack->fd >= 0 )
		(void)close(pack->fd);
	pack->fd = -1;
}

SparkStatus SparkK3DsparkPackPayload(const SparkK3DsparkPack *pack,
	uint32_t kind, uint32_t layer, const void **payload, uint64_t *bytes)
{
	const uint8_t *cursor;
	uint32_t index;
	if ( pack == 0 || pack->mapping == 0 || payload == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	cursor = pack->entries;
	for ( index = 0u; index < pack->tensor_count; index++ )
	{
		if ( SparkK3PackReadU32Le(cursor) == kind &&
			SparkK3PackReadU32Le(cursor + 4u) == layer )
		{
			*payload = pack->mapping + SparkK3PackReadU64Le(cursor + 24u);
			if ( bytes != 0 )
				*bytes = SparkK3PackReadU64Le(cursor + 32u);
			return(SPARK_STATUS_OK);
		}
		cursor += SPARK_K3_DSPARK_ENTRY_BYTES;
	}
	return(SPARK_STATUS_NOT_FOUND);
}
