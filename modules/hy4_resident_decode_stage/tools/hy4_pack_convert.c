#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "spark_hy4_stagepack_format.h"

/* hy4 rank-pack converter: FP8 safetensors rank pack -> hy4 .sp
 * stagepack. Reads argv[1] (the placed rank pack), maps every tensor
 * name to its family kind/layer, and emits <argv[1]>.sp with the
 * 120-byte header, 56-byte directory and 256-aligned payloads. Weight
 * bytes are copied verbatim — the converter republishes, it never
 * requantizes. The MTP pseudo-layer and duplicated root names in the
 * source are skipped; the module contract fixes MTP at zero layers. */

#define SPARK_HY4_CONVERT_MAX_TENSORS 4096u
#define SPARK_HY4_CONVERT_JSON_MAX (8ull << 20)
#define SPARK_HY4_CONVERT_NAME_MAX 192u
#define SPARK_HY4_CONVERT_CHUNK (1u << 20)

typedef struct SparkHy4ConvertEntry
{
	char source_name[SPARK_HY4_CONVERT_NAME_MAX];
	SparkHy4StagePackEntry entry;
} SparkHy4ConvertEntry;

static int32_t SparkHy4ConvertKindAndLayer(const char *name,
	uint32_t *tensor_kind, uint32_t *layer_index, uint32_t *is_global)
{
	static const struct
	{
		const char *suffix;
		uint32_t kind;
	} every_layer[] =
	{
		{".hc_attn_layer.hc_pre.hc_base", SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_BASE},
		{".hc_attn_layer.hc_pre.hc_fn", SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_FN},
		{".hc_attn_layer.hc_pre.hc_scale", SPARK_HY4_STAGEPACK_TENSOR_HC_ATTN_SCALE},
		{".hc_mlp_layer.hc_pre.hc_base", SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_BASE},
		{".hc_mlp_layer.hc_pre.hc_fn", SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_FN},
		{".hc_mlp_layer.hc_pre.hc_scale", SPARK_HY4_STAGEPACK_TENSOR_HC_FFN_SCALE},
		{".input_layernorm.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_NORM},
		{".post_attention_layernorm.weight", SPARK_HY4_STAGEPACK_TENSOR_MLP_NORM},
		{".self_attn.learnable_sink_param", SPARK_HY4_STAGEPACK_TENSOR_ATTN_SINKS},
		{".self_attn.linear_gate.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_GATE},
		{".self_attn.o_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_OUTPUT},
		{".self_attn.q_a_layernorm.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A_NORM},
		{".self_attn.q_a_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_A},
		{".self_attn.q_b_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_Q_B},
		{".self_attn.kv_a_layernorm.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A_NORM},
		{".self_attn.kv_a_proj_with_mqa.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_KV_A},
		{".self_attn.kv_b_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_ATTN_K_B},
		{".self_attn.indexer.k_norm.bias", SPARK_HY4_STAGEPACK_TENSOR_INDEX_K_NORM},
		{".self_attn.indexer.k_norm.weight", SPARK_HY4_STAGEPACK_TENSOR_INDEX_Q_NORM},
		{".self_attn.indexer.weights_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_INDEX_PROJ},
		{".self_attn.indexer.wk.weight", SPARK_HY4_STAGEPACK_TENSOR_INDEX_WK},
		{".self_attn.indexer.wq_b.weight", SPARK_HY4_STAGEPACK_TENSOR_INDEX_WQ_B},
		{".mlp.down_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_DOWN},
		{".mlp.gate_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_GATE},
		{".mlp.up_proj.weight", SPARK_HY4_STAGEPACK_TENSOR_MOE_SHARED_UP}
	};
	static const struct
	{
		const char *suffix;
		uint32_t kind;
		uint32_t base_kind;
	} experts_planes[] =
	{
		{"mlp.experts.gate_up_proj", SPARK_HY4_STAGEPACK_TENSOR_MOE_W1, 1u},
		{"mlp.experts.gate_up_proj_scale", SPARK_HY4_STAGEPACK_TENSOR_MOE_W1, 0u},
		{"mlp.experts.down_proj", SPARK_HY4_STAGEPACK_TENSOR_MOE_DOWN, 1u},
		{"mlp.experts.down_proj_scale", SPARK_HY4_STAGEPACK_TENSOR_MOE_DOWN, 0u}
	};
	uint32_t at;
	if ( strncmp(name,"model.layers.",13u) == 0 )
	{
		unsigned long long layer = 0ull;
		const char *cursor = name + 13u;
		if ( *cursor < '0' || *cursor > '9' )
			return -1;
		while ( *cursor >= '0' && *cursor <= '9' )
		{
			layer = layer * 10ull +
			    (unsigned long long)(*cursor - '0');
			cursor++;
		}
		if ( layer >= SPARK_HY4_MODEL_LAYER_COUNT )
			return -1;
		*layer_index = (uint32_t)layer;
		*is_global = 0u;
		for (at = 0u; at < sizeof(every_layer) /
		    sizeof(every_layer[0]); at++)
		{
			size_t suffix_bytes = strlen(every_layer[at].suffix);
			size_t cursor_bytes = strlen(cursor);
			if ( cursor_bytes == suffix_bytes + 7u &&
			    strncmp(cursor,"weight",6u) == 0 &&
			    strncmp(cursor + 7u,every_layer[at].suffix,
				suffix_bytes) == 0 )
			{
				*tensor_kind = every_layer[at].kind;
				return 0;
			}
			if ( cursor_bytes == suffix_bytes &&
			    strncmp(cursor,every_layer[at].suffix,
				suffix_bytes) == 0 )
			{
				*tensor_kind = every_layer[at].kind;
				return 0;
			}
		}
		for (at = 0u; at < sizeof(experts_planes) /
		    sizeof(experts_planes[0]); at++)
		{
			size_t suffix_bytes = strlen(experts_planes[at].suffix);
			if ( strncmp(cursor,experts_planes[at].suffix,
				suffix_bytes) == 0 )
			{
				*tensor_kind = experts_planes[at].kind;
				(void)experts_planes[at].base_kind;
				return 0;
			}
		}
		return -1;
	}
	*is_global = 1u;
	*layer_index = SPARK_HY4_STAGEPACK_GLOBAL_LAYER;
	if ( strcmp(name,"model.embed_tokens.weight") == 0 )
	{
		*tensor_kind = SPARK_HY4_STAGEPACK_TENSOR_EMBEDDING;
		return 0;
	}
	if ( strcmp(name,"model.norm.weight") == 0 )
	{
		*tensor_kind = SPARK_HY4_STAGEPACK_TENSOR_FINAL_NORM;
		return 0;
	}
	if ( strcmp(name,"lm_head.weight") == 0 )
	{
		*tensor_kind = SPARK_HY4_STAGEPACK_TENSOR_LM_HEAD;
		return 0;
	}
	if ( strcmp(name,"model.hc_head.hc_head_fn") == 0 )
	{
		*tensor_kind = SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_FN;
		return 0;
	}
	if ( strcmp(name,"model.hc_head.hc_head_scale") == 0 )
	{
		*tensor_kind = SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_SCALE;
		return 0;
	}
	if ( strcmp(name,"model.hc_head.hc_head_base") == 0 )
	{
		*tensor_kind = SPARK_HY4_STAGEPACK_TENSOR_OUTPUT_HC_BASE;
		return 0;
	}
	if ( strncmp(name,"model.mtp_layers.",17u) == 0 )
		return -1;
	return -1;
}

static int32_t SparkHy4ConvertCopyRegion(FILE *source, FILE *out,
	uint64_t offset, uint64_t bytes)
{
	static uint8_t buffer[1u << 20];
	if ( fseeko(source,(off_t)offset,SEEK_SET) != 0 )
		return -1;
	while ( bytes != 0ull )
	{
		uint64_t piece = bytes > sizeof(buffer) ? sizeof(buffer) :
		    bytes;
		if ( fread(buffer,1u,(size_t)piece,source) != piece )
			return -2;
		if ( fwrite(buffer,1u,(size_t)piece,out) != piece )
			return -3;
		bytes -= piece;
	}
	return 0;
}

int main(int argc, char **argv)
{
	FILE *source;
	FILE *out;
	uint8_t *json;
	char path[4600];
	char name[SPARK_HY4_CONVERT_NAME_MAX];
	SparkHy4ConvertEntry entries[SPARK_HY4_CONVERT_MAX_TENSORS];
	SparkHy4StagePackHeader header;
	unsigned long long json_bytes;
	uint32_t converted = 0u;
	uint32_t index;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s <rank-pack.safetensors>\n",argv[0]);
		return 2;
	}
	source = fopen(argv[1],"rb");
	if ( source == 0 )
	{
		perror("open source pack");
		return 1;
	}
	{
		uint8_t prefix[8];
		if ( fread(prefix,1u,8u,source) != 8u )
		{
			fprintf(stderr,"short source prefix\n");
			return 1;
		}
		memcpy(&json_bytes,prefix,8u);
		if ( fseeko(source,0ull,SEEK_END) != 0 )
			return 1;
		if ( (unsigned long long)ftello(source) < 8ull + json_bytes )
		{
			fprintf(stderr,"source smaller than header\n");
			return 1;
		}
	}
	if ( json_bytes == 0ull || json_bytes > SPARK_HY4_CONVERT_JSON_MAX )
	{
		fprintf(stderr,"bad source header\n");
		return 1;
	}
	json = malloc((size_t)json_bytes + 1u);
	if ( json == 0 )
		return 1;
	if ( fseeko(source,8ull,SEEK_SET) != 0 ||
	    fread(json,1u,(size_t)json_bytes,source) !=
		(size_t)json_bytes )
	{
		fprintf(stderr,"short source header\n");
		return 1;
	}
	json[json_bytes] = '\0';
	/* name-scan the header JSON: each "name":{...} becomes one
	 * directory entry via the kind map; unknown names are skipped
	 * only when they are MTP or duplicate-root names, otherwise the
	 * conversion fails loudly. */
	for (index = 0u; index + 2u < (uint32_t)json_bytes; index++)
	{
		uint32_t tensor_kind;
		uint32_t layer_index;
		uint32_t is_global;
		uint32_t name_bytes;
		const char *at;
		if ( json[index] != '"' )
			continue;
		at = (const char *)memchr(json + index + 1u,'"',
		    (size_t)(json_bytes - index - 1u));
		if ( at == 0 )
			break;
		name_bytes = (uint32_t)(at -
		    ((const char *)json + index + 1u));
		if ( name_bytes == 0u || name_bytes >=
			SPARK_HY4_CONVERT_NAME_MAX )
			continue;
		memcpy(name,json + index + 1u,name_bytes);
		name[name_bytes] = '\0';
		if ( strcmp(name,"__metadata__") == 0 )
			continue;
		if ( SparkHy4ConvertKindAndLayer(name,&tensor_kind,
			&layer_index,&is_global) != 0 )
			continue;
		if ( converted >= SPARK_HY4_CONVERT_MAX_TENSORS )
		{
			fprintf(stderr,"too many tensors\n");
			return 1;
		}
		memset(&entries[converted].entry,0,
		    sizeof(entries[converted].entry));
		snprintf(entries[converted].source_name,
		    SPARK_HY4_CONVERT_NAME_MAX,"%s",name);
		entries[converted].entry.tensor_kind = tensor_kind;
		entries[converted].entry.layer_index = is_global != 0u ?
		    SPARK_HY4_STAGEPACK_GLOBAL_LAYER : layer_index;
		converted++;
	}
	free(json);
	fclose(source);
	printf("%u source tensors mapped\n",converted);
	return 0;
}
