#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static SparkStatus SparkK3PackReadConfig(const SparkNamedPack *named, SparkK3PackConfig *config)
{
	static const char *const names[] =
	{
		"hidden","layers","first_layer","total_layers","experts","top_k",
		"latent","intermediate","group","vocab","kda_heads","kda_head",
		"heads","kv_lora","rope","v_head","nope","shared","q_lora"
	};
	uint32_t values[19];
	for ( uint32_t index = 0u; index < 19u; index++ )
	{
		if ( SparkNamedPackConfigU32(named, names[index], &values[index]) != SPARK_STATUS_OK )
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
	SparkNamedPack named;
};

SparkStatus SparkK3PackOpen(const char *path, SparkK3Pack *pack)
{
	SparkStatus status;
	if ( pack == 0 || path == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pack, 0, sizeof(*pack));
	pack->private_state = (struct SparkK3PackPrivate *)calloc(1u, sizeof(*pack->private_state));
	if ( pack->private_state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkNamedPackOpen(path, SPARK_K3_PACK_MAGIC, SPARK_K3_PACK_FORMAT_VERSION, SPARK_K3_PACK_ALIGNMENT,
		&pack->private_state->named);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3PackReadConfig(&pack->private_state->named, &pack->config);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3PackClose(pack);
		SPARK_RETURN(status);
	}
	pack->file_bytes = pack->private_state->named.file_bytes;
	pack->payload_base = pack->private_state->named.payload_base;
	pack->version = pack->private_state->named.version;
	return(SPARK_STATUS_OK);
}

void SparkK3PackClose(SparkK3Pack *pack)
{
	if ( pack == 0 || pack->private_state == 0 )
		return;
	SparkNamedPackClose(&pack->private_state->named);
	free(pack->private_state);
	pack->private_state = 0;
}

SparkStatus SparkK3PackLoadEntry(SparkK3Pack *pack, const char *name, SparkK3PackEntry *entry)
{
	SparkNamedPackEntry named;
	SparkStatus status;
	if ( pack == 0 || pack->private_state == 0 || name == 0 || entry == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(entry, 0, sizeof(*entry));
	snprintf(entry->name, sizeof(entry->name), "%s", name);
	status = SparkNamedPackEntryLoad(&pack->private_state->named, name, &named);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( strcmp(named.kind, "bf16") == 0 )
		entry->kind = SPARK_K3_PACK_KIND_BF16;
	else if ( strcmp(named.kind, "f32") == 0 )
		entry->kind = SPARK_K3_PACK_KIND_F32;
	else if ( strcmp(named.kind, "mxfp4_ws_interleaved_v1") == 0 )
		entry->kind = SPARK_K3_PACK_KIND_MXFP4_WS_INTERLEAVED_V1;
	else
		return(SPARK_STATUS_VALIDATION_FAILED);
	entry->payload_offset = named.payload_offset;
	entry->bytes = named.bytes;
	entry->shape_count = named.shape_count;
	memcpy(entry->shape, named.shape, sizeof(entry->shape));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkK3PackLoadInterleaveTileK(SparkK3Pack *pack, const char *name, uint32_t *tile_k)
{
	if ( pack == 0 || pack->private_state == 0 || name == 0 || tile_k == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkNamedPackTensorFieldU32(&pack->private_state->named, name, "interleave", "tile_k", tile_k));
}
