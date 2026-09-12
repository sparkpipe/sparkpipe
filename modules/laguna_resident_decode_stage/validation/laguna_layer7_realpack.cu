#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/laguna_resident_decode_stage/source/cuda/layer.cuh"
#include "modules/laguna_resident_decode_stage/source/spark_laguna_stagepack_format.h"

#ifndef LAGUNA_EXPERT_WEIGHT_CODEC
#error "LAGUNA_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif

#define L7_MAX_ENTRIES 512u

typedef struct
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t tensor_count;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t directory_offset;
	uint64_t file_bytes;
	char model_revision[65];
} L7PackHeader;

typedef struct
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t payload_type;
	uint32_t weight_codec;
	uint32_t scale_encoding;
	uint32_t group_count;
	uint32_t rows;
	uint32_t columns;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} L7PackEntry;

typedef struct
{
	FILE *file;
	L7PackHeader header;
	L7PackEntry entries[L7_MAX_ENTRIES];
} L7Pack;

static int l7_trace_depth = 0;
#define L7_TRACE(site) fprintf(stderr,"L7TRACE %s\n",site)
static void L7Die(const char *site,const char *detail)
{
	fprintf(stderr,"SPARK_FAIL: laguna_layer7 %s: %s\n",site,detail);
	exit(2);
}

static void L7DieSite(const char *site,char *detail,const char *extra)
{
	fprintf(stderr,"SPARK_FAIL: laguna_layer7 %s: %s %s\n",site,detail,extra);
	exit(2);
}

static void L7CudaCheck(cudaError_t error,const char *site)
{
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"SPARK_FAIL: laguna_layer7 %s: cuda %s\n",site,cudaGetErrorString(error));
		exit(2);
	}
}

static void L7Alloc(void **device,uint64_t bytes,const char *site)
{
	if ( cudaMalloc(device,(size_t)bytes) != cudaSuccess )
	{
		fprintf(stderr,"SPARK_FAIL: laguna_layer7 %s: cudaMalloc %llu bytes\n",site,(unsigned long long)bytes);
		exit(2);
	}
}

static void L7Upload(void *device,const void *host,uint64_t bytes,const char *site)
{
	L7CudaCheck(cudaMemcpy(device,host,(size_t)bytes,cudaMemcpyHostToDevice),site);
}

static void L7Download(void *host,const void *device,uint64_t bytes,const char *site)
{
	L7CudaCheck(cudaMemcpy(host,device,(size_t)bytes,cudaMemcpyDeviceToHost),site);
}

static void L7DumpDevice(const char *path,const void *device,uint64_t bytes)
{
	void *host = malloc((size_t)bytes);
	FILE *out;
	if ( host == 0 )
		L7Die("dump","host malloc failed");
	L7Download(host,device,bytes,path);
	out = fopen(path,"wb");
	if ( out == 0 )
		L7Die("dump",path);
	if ( fwrite(host,1,(size_t)bytes,out) != (size_t)bytes )
		L7Die("dump","short write");
	fclose(out);
	free(host);
}

static uint16_t L7RandomBf16(uint32_t *state)
{
	uint32_t bits;
	float value;
	*state = (*state * 1664525u) + 1013904223u;
	value = ((float)((*state >> 8) & 0xffffu) / 32768.0f) - 1.0f;
	bits = *(uint32_t *)&value;
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static void L7OpenPack(const char *path,L7Pack *pack)
{
	uint8_t raw[264];
	uint32_t fields[20];
	uint64_t directory_offset,file_bytes;
	uint32_t index;
	if ( pack->file != 0 )
		L7Die("pack open","pack already open");
	pack->file = fopen(path,"rb");
	if ( pack->file == 0 )
		L7Die("pack open",path);
	if ( fread(raw,1,264,pack->file) != 264 )
		L7Die("pack open","short header");
	memcpy(fields,raw,80);
	memcpy(&directory_offset,raw + 80,8);
	memcpy(&file_bytes,raw + 88,8);
	pack->header.magic = fields[0];
	pack->header.format_version = fields[1];
	pack->header.tensor_count = fields[6];
	pack->header.stage_count = fields[7];
	pack->header.stage_index = fields[8];
	pack->header.tp_degree = fields[18];
	pack->header.tp_rank = fields[19];
	pack->header.directory_offset = directory_offset;
	pack->header.file_bytes = file_bytes;
	memset(pack->header.model_revision,0,65);
	memcpy(pack->header.model_revision,raw + 96,64);
	if ( pack->header.magic != SPARK_LAGUNA_STAGEPACK_MAGIC )
		L7Die("pack open","bad magic");
	if ( pack->header.format_version != SPARK_LAGUNA_STAGEPACK_FORMAT_VERSION )
		L7Die("pack open","bad format version");
	if ( pack->header.tensor_count == 0u || pack->header.tensor_count > L7_MAX_ENTRIES )
		L7Die("pack open","entry count outside harness table");
	if ( fseek(pack->file,(long)directory_offset,SEEK_SET) != 0 )
		L7Die("pack open","directory seek failed");
	for ( index = 0u; index < pack->header.tensor_count; index++ )
	{
		uint8_t entry_raw[64];
		if ( fread(entry_raw,1,64,pack->file) != 64 )
			L7Die("pack open","short entry read");
		memcpy(&pack->entries[index],entry_raw,64);
	}
	fprintf(stderr,
		"laguna_layer7 pack %s: revision %s stage %u/%u rank %u/%u tensors %u bytes %llu\n",
		path,pack->header.model_revision,pack->header.stage_index,pack->header.stage_count,
		pack->header.tp_rank,pack->header.tp_degree,pack->header.tensor_count,
		(unsigned long long)file_bytes);
}

static const L7PackEntry *L7FindEntry(const L7Pack *pack,uint32_t kind,uint32_t layer,const char *site)
{
	uint32_t index;
	for ( index = 0u; index < pack->header.tensor_count; index++ )
		if ( pack->entries[index].tensor_kind == kind && pack->entries[index].layer_index == layer )
			return &pack->entries[index];
	fprintf(stderr,"SPARK_FAIL: laguna_layer7 %s: entry kind %u layer %u not in pack\n",site,kind,layer);
	exit(2);
	return 0;
}

static void L7LoadEntry(const L7Pack *pack,const L7PackEntry *entry,void **device,const char *site)
{
	void *host;
	uint64_t bytes = entry->payload_bytes;
	if ( bytes == 0u || bytes > (uint64_t)2u * 1024u * 1024u * 1024u )
		L7Die(site,"payload size outside harness range");
	host = malloc((size_t)bytes);
	if ( host == 0 )
		L7Die(site,"host malloc failed");
	if ( fseek(pack->file,(long)entry->payload_offset,SEEK_SET) != 0 )
		L7Die(site,"payload seek failed");
	if ( fread(host,1,(size_t)bytes,pack->file) != (size_t)bytes )
		L7Die(site,"short payload read");
	L7Alloc(device,bytes,site);
	L7Upload(*device,host,bytes,site);
	free(host);
}

static void L7RequireShape(const L7PackEntry *entry,uint32_t rows,uint32_t columns,uint32_t group_count,const char *site)
{
	if ( entry->rows != rows || entry->columns != columns ||
		(group_count != 0u && entry->group_count != group_count) )
	{
		fprintf(stderr,
			"SPARK_FAIL: laguna_layer7 %s: entry shape g%u r%u c%u expected g%u r%u c%u\n",
			site,entry->group_count,entry->rows,entry->columns,group_count,rows,columns);
		exit(2);
	}
}

typedef struct
{
	uint16_t *normed;
	uint16_t *qkv;
	uint16_t *q;
	uint16_t *k;
	uint16_t *v;
	uint16_t *q_normed_copy;
	uint16_t *k_normed_copy;
	uint16_t *decatt_copy;
	uint16_t *attention;
	uint16_t *attention_out;
	uint16_t *gate;
	uint16_t *gate_up;
	uint16_t *intermediate;
	uint16_t *expert_out;
	uint16_t *shared_out;
	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint32_t *window_positions;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *sequence_of_row;
	uint32_t *context_length;
	uint32_t *positions;
	uint32_t *row_positions;
	float *yarn_inv_freq;
} L7Scratch;

static void L7AllocateScratch(L7Scratch *scratch,uint32_t tokens,uint32_t w1_rows,uint32_t expert_inter)
{
	uint64_t packed = (uint64_t)tokens * LAGUNA_TOP_K;
	L7Alloc((void **)&scratch->normed,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"alloc normed");
	L7Alloc((void **)&scratch->qkv,(uint64_t)tokens * (LAGUNA_Q_HEADS_SLIDING + 2u * LAGUNA_KV_HEADS) * LAGUNA_HEAD_DIM * 2u,"alloc qkv");
	L7Alloc((void **)&scratch->q,(uint64_t)tokens * LAGUNA_Q_HEADS_SLIDING * LAGUNA_HEAD_DIM * 2u,"alloc q");
	L7Alloc((void **)&scratch->k,(uint64_t)tokens * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2u,"alloc k");
	L7Alloc((void **)&scratch->v,(uint64_t)tokens * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2u,"alloc v");
	L7Alloc((void **)&scratch->q_normed_copy,(uint64_t)tokens * LAGUNA_Q_HEADS_SLIDING * LAGUNA_HEAD_DIM * 2u,"alloc q_normed_copy");
	L7Alloc((void **)&scratch->k_normed_copy,(uint64_t)tokens * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2u,"alloc k_normed_copy");
	L7Alloc((void **)&scratch->decatt_copy,(uint64_t)tokens * LAGUNA_Q_HEADS_SLIDING * LAGUNA_HEAD_DIM * 2u,"alloc decatt_copy");
	L7Alloc((void **)&scratch->attention,(uint64_t)tokens * LAGUNA_Q_HEADS_SLIDING * LAGUNA_HEAD_DIM * 2u,"alloc attention");
	L7Alloc((void **)&scratch->attention_out,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"alloc attention_out");
	L7Alloc((void **)&scratch->gate,(uint64_t)tokens * LAGUNA_Q_HEADS_SLIDING * 2u,"alloc gate");
	L7Alloc((void **)&scratch->gate_up,packed * w1_rows * 2u,"alloc gate_up");
	L7Alloc((void **)&scratch->intermediate,packed * expert_inter * 2u,"alloc intermediate");
	L7Alloc((void **)&scratch->expert_out,packed * LAGUNA_HIDDEN * 2u,"alloc expert_out");
	L7Alloc((void **)&scratch->shared_out,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"alloc shared_out");
	L7Alloc((void **)&scratch->router_logits,(uint64_t)tokens * LAGUNA_EXPERTS * 4u,"alloc router_logits");
	L7Alloc((void **)&scratch->route_expert,(uint64_t)tokens * LAGUNA_TOP_K * 4u,"alloc route_expert");
	L7Alloc((void **)&scratch->route_weight,(uint64_t)tokens * LAGUNA_TOP_K * 4u,"alloc route_weight");
	L7Alloc((void **)&scratch->route_packed_row,(uint64_t)tokens * LAGUNA_TOP_K * 4u,"alloc route_packed_row");
	L7Alloc((void **)&scratch->route_source_token,packed * 4u,"alloc route_source_token");
	L7Alloc((void **)&scratch->group_row_offset,(uint64_t)(LAGUNA_EXPERTS + 1u) * 4u,"alloc group_row_offset");
	L7Alloc((void **)&scratch->group_tile_prefix_w1,(uint64_t)(LAGUNA_EXPERTS + 1u) * 4u,"alloc tile_prefix_w1");
	L7Alloc((void **)&scratch->group_tile_prefix_w2,(uint64_t)(LAGUNA_EXPERTS + 1u) * 4u,"alloc tile_prefix_w2");
	L7Alloc((void **)&scratch->window_positions,(uint64_t)tokens * LAGUNA_WINDOW * 4u,"alloc window");
	L7Alloc((void **)&scratch->dense_row_offset,8u,"alloc dense_row_offset");
	L7Alloc((void **)&scratch->dense_tile_prefix,8u,"alloc dense_tile_prefix");
	L7Alloc((void **)&scratch->sequence_of_row,(uint64_t)tokens * 4u,"alloc sequence");
	L7Alloc((void **)&scratch->context_length,4u,"alloc context");
	L7Alloc((void **)&scratch->positions,(uint64_t)tokens * 4u,"alloc positions");
	L7Alloc((void **)&scratch->row_positions,(uint64_t)tokens * 4u,"alloc row_positions");
	L7Alloc((void **)&scratch->yarn_inv_freq,(LAGUNA_ROPE_FULL_ROT / 2u) * 4u,"alloc yarn");
}

typedef struct
{
	LagunaLayerBuffers buffers;
	uint16_t *hidden;
	uint16_t *residual;
	uint8_t *kv_pool;
	uint32_t *kv_page_table;
	LmKvAccessError *kv_error;
	uint32_t kv_page_count;
	uint32_t tokens;
	uint32_t layer;
	uint32_t q_heads;
	uint32_t qkv_rows;
	uint32_t attn_output_columns;
	uint32_t gate_rows;
	uint32_t sliding;
	uint32_t dense_gate_up_rows;
	uint32_t dense_intermediate;
	uint32_t expert_w1_rows;
	uint32_t expert_intermediate;
	uint32_t shared_rows;
	uint32_t shared_intermediate;
} L7LayerRun;

static void L7SetupRun(L7LayerRun *run,L7Scratch *scratch,uint32_t tokens,uint32_t layer,const uint16_t *host_hidden)
{
	uint32_t q_heads = LAGUNA_Q_HEADS(layer);
	uint32_t host_dense_row_offset[2];
	uint32_t host_dense_tile_prefix[2];
	uint32_t host_context[1];
	uint32_t *host_positions;
	uint32_t index;
	float host_yarn[LAGUNA_ROPE_FULL_ROT / 2u];

	run->tokens = tokens;
	run->layer = layer;
	run->q_heads = q_heads;
	run->qkv_rows = q_heads * LAGUNA_HEAD_DIM + 2u * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;
	run->attn_output_columns = q_heads * LAGUNA_HEAD_DIM;
	run->gate_rows = q_heads;
	run->sliding = LAGUNA_LAYER_IS_SLIDING(layer) ? 1u : 0u;
	run->expert_w1_rows = 2u * (LAGUNA_EXPERT_INTERMEDIATE / 8u);
	run->expert_intermediate = LAGUNA_EXPERT_INTERMEDIATE / 8u;
	run->shared_rows = 2u * (LAGUNA_EXPERT_INTERMEDIATE / 8u);
	run->shared_intermediate = LAGUNA_EXPERT_INTERMEDIATE / 8u;
	run->dense_gate_up_rows = 2u * (LAGUNA_DENSE_INTERMEDIATE / 8u);
	run->dense_intermediate = LAGUNA_DENSE_INTERMEDIATE / 8u;

	L7Alloc((void **)&run->hidden,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"alloc hidden");
	L7Alloc((void **)&run->residual,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"alloc residual");
	L7Upload(run->hidden,host_hidden,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"upload hidden");
	L7CudaCheck(cudaMemset(run->residual,0,(size_t)((uint64_t)tokens * LAGUNA_HIDDEN * 2u)),"zero residual");
	run->kv_page_count = (tokens + LagunaKv::kPageSlots - 1u) / LagunaKv::kPageSlots;
	L7Alloc((void **)&run->kv_pool,(uint64_t)run->kv_page_count * LagunaKv::kPageBytes,"alloc kv pool");
	L7Alloc((void **)&run->kv_page_table,(uint64_t)run->kv_page_count * 4u,"alloc page table");
	if ( cudaMallocManaged((void **)&run->kv_error,sizeof(LmKvAccessError),cudaMemAttachGlobal) != cudaSuccess )
		L7Die("alloc kv error","cudaMallocManaged failed");
	{
		uint32_t *host_pages = (uint32_t *)malloc((size_t)run->kv_page_count * 4u);
		for ( index = 0u; index < run->kv_page_count; index++ )
			host_pages[index] = index;
		L7Upload(run->kv_page_table,host_pages,(uint64_t)run->kv_page_count * 4u,"upload page table");
		free(host_pages);
	}
	L7CudaCheck(cudaMemset(run->kv_pool,0,(size_t)((uint64_t)run->kv_page_count * LagunaKv::kPageBytes)),"zero kv pool");
	L7CudaCheck(cudaMemset(run->kv_error,0,sizeof(LmKvAccessError)),"zero kv error");
	if ( LmKvViewInitialize(&run->buffers.cache,run->kv_pool,run->kv_page_table,run->kv_page_count,1u,run->kv_page_count,run->kv_error) != 0 )
		L7Die("kv view","initialize failed");

	host_positions = (uint32_t *)malloc((size_t)tokens * 4u);
	for ( index = 0u; index < tokens; index++ )
		host_positions[index] = index;
	host_context[0] = tokens;
	host_dense_row_offset[0] = 0u;
	host_dense_row_offset[1] = tokens;
	host_dense_tile_prefix[0] = 0u;
	host_dense_tile_prefix[1] = 0u;
	{
		uint32_t *host_sequence = (uint32_t *)calloc((size_t)tokens,4u);
		L7Upload(scratch->sequence_of_row,host_sequence,(uint64_t)tokens * 4u,"upload sequence");
		free(host_sequence);
	}
	L7Upload(scratch->context_length,host_context,4u,"upload context");
	L7Upload(scratch->positions,host_positions,(uint64_t)tokens * 4u,"upload positions");
	L7Upload(scratch->row_positions,host_positions,(uint64_t)tokens * 4u,"upload row_positions");
	L7Upload(scratch->dense_row_offset,host_dense_row_offset,8u,"upload row offset");
	L7Upload(scratch->dense_tile_prefix,host_dense_tile_prefix,8u,"upload tile prefix");
	free(host_positions);
	LagunaBuildYarnInvFrequency(host_yarn,LAGUNA_ROPE_FULL_ROT,LAGUNA_ROPE_FULL_THETA,
		SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR,SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST,SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW);
	L7Upload(scratch->yarn_inv_freq,host_yarn,sizeof(host_yarn),"upload yarn");

	memset(&run->buffers,0,sizeof(run->buffers));
	if ( LmKvViewInitialize(&run->buffers.cache,run->kv_pool,run->kv_page_table,run->kv_page_count,1u,run->kv_page_count,run->kv_error) != 0 )
		L7Die("kv view","re-initialize failed");
	run->buffers.dense_row_offset = scratch->dense_row_offset;
	run->buffers.dense_tile_prefix = scratch->dense_tile_prefix;
	run->buffers.qk_scale = LAGUNA_ATTN_SCALE;
	run->buffers.yarn_inv_freq = run->sliding ? 0 : scratch->yarn_inv_freq;
	run->buffers.tp_degree = 8u;
	run->buffers.tp_rank = 0u;
	run->buffers.layer_index = layer;
	run->buffers.q_heads = run->q_heads;
	run->buffers.qkv_rows = run->qkv_rows;
	run->buffers.attn_output_columns = run->attn_output_columns;
	run->buffers.gate_rows = run->gate_rows;
	run->buffers.hidden_bf16 = run->hidden;
	run->buffers.residual_bf16 = run->residual;
	run->buffers.normed_bf16 = scratch->normed;
	run->buffers.qkv_bf16 = scratch->qkv;
	run->buffers.q_bf16 = scratch->q;
	run->buffers.k_bf16 = scratch->k;
	run->buffers.v_bf16 = scratch->v;
	run->buffers.attention_bf16 = scratch->attention;
	run->buffers.attention_out_bf16 = scratch->attention_out;
	run->buffers.gate_bf16 = scratch->gate;
	run->buffers.window_positions = scratch->window_positions;
	run->buffers.gate_up_bf16 = scratch->gate_up;
	run->buffers.intermediate_bf16 = scratch->intermediate;
	run->buffers.expert_out_bf16 = scratch->expert_out;
	run->buffers.shared_out_bf16 = scratch->shared_out;
	run->buffers.router_logits = scratch->router_logits;
	run->buffers.route_expert = scratch->route_expert;
	run->buffers.route_weight = scratch->route_weight;
	run->buffers.route_source_token = scratch->route_source_token;
	run->buffers.route_packed_row = scratch->route_packed_row;
	run->buffers.group_row_offset = scratch->group_row_offset;
	run->buffers.group_tile_prefix_w1 = scratch->group_tile_prefix_w1;
	run->buffers.group_tile_prefix_w2 = scratch->group_tile_prefix_w2;
	run->buffers.sequence_of_row = scratch->sequence_of_row;
	run->buffers.context_length = scratch->context_length;
	run->buffers.positions = scratch->positions;
	run->buffers.row_positions = scratch->row_positions;
}

static void L7LoadAttentionWeights(L7LayerRun *run,const L7Pack *pack,uint32_t layer)
{
	const L7PackEntry *entry;
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_INPUT_NORM,layer,"load attn_norm");
	L7RequireShape(entry,1u,LAGUNA_HIDDEN,0u,"shape attn_norm");
	L7LoadEntry(pack,entry,(void **)&run->buffers.attn_norm_weight,"load attn_norm");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV,layer,"load qkv");
	L7RequireShape(entry,run->qkv_rows,LAGUNA_HIDDEN,0u,"shape qkv");
	L7LoadEntry(pack,entry,(void **)&run->buffers.fused_qkv_weight,"load qkv");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_Q_NORM,layer,"load q_norm");
	L7RequireShape(entry,1u,LAGUNA_HEAD_DIM,0u,"shape q_norm");
	L7LoadEntry(pack,entry,(void **)&run->buffers.q_norm_weight,"load q_norm");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_K_NORM,layer,"load k_norm");
	L7RequireShape(entry,1u,LAGUNA_HEAD_DIM,0u,"shape k_norm");
	L7LoadEntry(pack,entry,(void **)&run->buffers.k_norm_weight,"load k_norm");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT,layer,"load o_proj");
	L7RequireShape(entry,LAGUNA_HIDDEN,run->attn_output_columns,0u,"shape o_proj");
	L7LoadEntry(pack,entry,(void **)&run->buffers.output_weight,"load o_proj");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_GATE,layer,"load gate");
	L7RequireShape(entry,run->gate_rows,LAGUNA_HIDDEN,0u,"shape gate");
	L7LoadEntry(pack,entry,(void **)&run->buffers.gate_weight,"load gate");
}

static void L7LoadMlpWeights(L7LayerRun *run,const L7Pack *pack,uint32_t layer)
{
	const L7PackEntry *entry;
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_POST_NORM,layer,"load post_norm");
	L7RequireShape(entry,1u,LAGUNA_HIDDEN,0u,"shape post_norm");
	L7LoadEntry(pack,entry,(void **)&run->buffers.mlp_norm_weight,"load post_norm");
	if ( layer < LAGUNA_FIRST_ROUTED_LAYER )
	{
		entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_GATE_UP,layer,"load dense_gate_up");
		L7RequireShape(entry,run->dense_gate_up_rows,LAGUNA_HIDDEN,0u,"shape dense_gate_up");
		L7LoadEntry(pack,entry,(void **)&run->buffers.dense_gate_up_weight,"load dense_gate_up");
		entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_DOWN,layer,"load dense_down");
		L7RequireShape(entry,LAGUNA_HIDDEN,run->dense_intermediate,0u,"shape dense_down");
		L7LoadEntry(pack,entry,(void **)&run->buffers.dense_down_weight,"load dense_down");
		return;
	}
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER,layer,"load router");
	L7RequireShape(entry,LAGUNA_EXPERTS,LAGUNA_HIDDEN,0u,"shape router");
	L7LoadEntry(pack,entry,(void **)&run->buffers.router_weight,"load router");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER_CORRECTION,layer,"load correction");
	L7RequireShape(entry,1u,LAGUNA_EXPERTS,0u,"shape correction");
	L7LoadEntry(pack,entry,(void **)&run->buffers.router_correction_bias,"load correction");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP,layer,"load expert_w1");
	L7RequireShape(entry,run->expert_w1_rows,LAGUNA_HIDDEN,LAGUNA_EXPERTS,"shape expert_w1");
	L7LoadEntry(pack,entry,(void **)&run->buffers.expert_w1_weight,"load expert_w1");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN,layer,"load expert_w2");
	L7RequireShape(entry,LAGUNA_HIDDEN,run->expert_intermediate,LAGUNA_EXPERTS,"shape expert_w2");
	L7LoadEntry(pack,entry,(void **)&run->buffers.expert_w2_weight,"load expert_w2");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_GATE_UP,layer,"load shared_gate_up");
	L7RequireShape(entry,run->shared_rows,LAGUNA_HIDDEN,0u,"shape shared_gate_up");
	L7LoadEntry(pack,entry,(void **)&run->buffers.shared_gate_up_weight,"load shared_gate_up");
	entry = L7FindEntry(pack,SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_DOWN,layer,"load shared_down");
	L7RequireShape(entry,LAGUNA_HIDDEN,run->shared_intermediate,0u,"shape shared_down");
	L7LoadEntry(pack,entry,(void **)&run->buffers.shared_down_weight,"load shared_down");
	run->buffers.dense_gate_up_rows = run->dense_gate_up_rows;
	run->buffers.dense_intermediate = run->dense_intermediate;
	run->buffers.expert_w1_rows = run->expert_w1_rows;
	run->buffers.expert_intermediate = run->expert_intermediate;
	run->buffers.shared_gate_up_rows = run->shared_rows;
	run->buffers.shared_intermediate = run->shared_intermediate;
	run->buffers.expert_w1_scale = 0;
	run->buffers.expert_w2_scale = 0;
}

static void L7CheckKvError(L7LayerRun *run,const char *site)
{
	LmKvAccessError error;
	L7Download(&error,run->kv_error,sizeof(error),site);
	if ( error.error_code != LM_FRAME_ERROR_NONE )
	{
		fprintf(stderr,
			"SPARK_FAIL: laguna_layer7 %s: kv access error code %u kind %u row %u seq %u pos %u page %u\n",
			site,error.error_code,error.access_kind,error.row,error.sequence,error.position,error.page);
		exit(2);
	}
}

static int32_t L7AttentionReplica(LagunaLayerBuffers *buffers,uint32_t rows,uint32_t sliding,
	uint16_t *q_normed_copy,uint16_t *k_normed_copy,uint16_t *decatt_copy,
	uint32_t multiprocessors,cudaStream_t stream)
{
	int32_t status;
	const uint32_t *selected_positions = 0;
	uint32_t kv_positions = 0u;
	LmQkvLayout layout;
	uint64_t q_elements = (uint64_t)rows * buffers->q_heads * LAGUNA_HEAD_DIM;
	uint64_t k_elements = (uint64_t)rows * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;

	LM_LAUNCH((LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
		rows,
		LAGUNA_LAYER_THREADS,
		(LAGUNA_HIDDEN + 8u) * sizeof(float),
		stream,
		buffers->hidden_bf16,
		buffers->residual_bf16,
		(const uint16_t *)buffers->attn_norm_weight,
		buffers->residual_bf16,
		buffers->normed_bf16,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		LAGUNA_RMS_EPSILON);
	status = LagunaLaunchBf16Linear(buffers->normed_bf16,(const uint16_t *)buffers->fused_qkv_weight,
		buffers->qkv_bf16,buffers->dense_row_offset,buffers->dense_tile_prefix,
		rows,LAGUNA_HIDDEN,buffers->qkv_rows,buffers->qkv_rows,0u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	layout.query_dimension = buffers->q_heads * LAGUNA_HEAD_DIM;
	layout.key_dimension = LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;
	layout.value_dimension = LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;
	layout.rope_dimension = 0u;
	layout.head_dimension = LAGUNA_HEAD_DIM;
	LM_LAUNCH((LmSplitQkvKernel<LAGUNA_LAYER_THREADS>),rows,LAGUNA_LAYER_THREADS,0,stream,
		buffers->qkv_bf16,layout,buffers->q_bf16,buffers->k_bf16,buffers->v_bf16,rows,1.0f);
	LM_LAUNCH((LmHeadRmsNormKernel<LAGUNA_LAYER_THREADS>),dim3(buffers->q_heads,rows),LAGUNA_LAYER_THREADS,0,stream,
		buffers->q_bf16,(const uint16_t *)buffers->q_norm_weight,buffers->q_bf16,rows,buffers->q_heads,
		LAGUNA_HEAD_DIM,LAGUNA_RMS_EPSILON,1.0f);
	LM_LAUNCH((LmHeadRmsNormKernel<LAGUNA_LAYER_THREADS>),dim3(LAGUNA_KV_HEADS,rows),LAGUNA_LAYER_THREADS,0,stream,
		buffers->k_bf16,(const uint16_t *)buffers->k_norm_weight,buffers->k_bf16,rows,LAGUNA_KV_HEADS,
		LAGUNA_HEAD_DIM,LAGUNA_RMS_EPSILON,1.0f);
	L7CudaCheck(cudaMemcpy(q_normed_copy,buffers->q_bf16,(size_t)q_elements * 2u,cudaMemcpyDeviceToDevice),"copy qknorm q");
	L7CudaCheck(cudaMemcpy(k_normed_copy,buffers->k_bf16,(size_t)k_elements * 2u,cudaMemcpyDeviceToDevice),"copy qknorm k");
	if ( sliding != 0u )
	{
		LM_LAUNCH((LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),dim3(rows,buffers->q_heads),LAGUNA_LAYER_THREADS,0,stream,
			buffers->q_bf16,buffers->positions,buffers->q_heads,LAGUNA_HEAD_DIM,LAGUNA_ROPE_SLIDING_ROT,LAGUNA_ROPE_SLIDING_THETA);
		LM_LAUNCH((LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),dim3(rows,LAGUNA_KV_HEADS),LAGUNA_LAYER_THREADS,0,stream,
			buffers->k_bf16,buffers->positions,LAGUNA_KV_HEADS,LAGUNA_HEAD_DIM,LAGUNA_ROPE_SLIDING_ROT,LAGUNA_ROPE_SLIDING_THETA);
	}
	else
	{
		LM_LAUNCH((LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),dim3(rows,buffers->q_heads),LAGUNA_LAYER_THREADS,0,stream,
			buffers->q_bf16,buffers->positions,buffers->q_heads,LAGUNA_HEAD_DIM,LAGUNA_ROPE_FULL_ROT,0.0f,
			buffers->yarn_inv_freq,LAGUNA_ROPE_FULL_ATTENTION_FACTOR,0u);
		LM_LAUNCH((LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),dim3(rows,LAGUNA_KV_HEADS),LAGUNA_LAYER_THREADS,0,stream,
			buffers->k_bf16,buffers->positions,LAGUNA_KV_HEADS,LAGUNA_HEAD_DIM,LAGUNA_ROPE_FULL_ROT,0.0f,
			buffers->yarn_inv_freq,LAGUNA_ROPE_FULL_ATTENTION_FACTOR,0u);
	}
	LM_LAUNCH((LmGqaKvStoreKernel<LagunaKv,LAGUNA_LAYER_THREADS,LAGUNA_KV_HEADS,LAGUNA_HEAD_DIM,LAGUNA_HEAD_DIM>),
		rows,LAGUNA_LAYER_THREADS,0,stream,buffers->cache,buffers->k_bf16,buffers->v_bf16,
		buffers->sequence_of_row,buffers->positions,rows);
	if ( sliding != 0u )
	{
		LM_LAUNCH((LmBuildSlidingWindowPositionsKernel<LAGUNA_LAYER_THREADS>),rows,LAGUNA_LAYER_THREADS,0,stream,
			buffers->sequence_of_row,buffers->context_length,buffers->row_positions,rows,LAGUNA_WINDOW,buffers->window_positions);
		selected_positions = buffers->window_positions;
		kv_positions = LAGUNA_WINDOW;
	}
	LM_LAUNCH((LmGqaAttentionDecodeKernel<LagunaKv,LAGUNA_ATTN_THREADS,LAGUNA_KV_HEADS,LAGUNA_HEAD_DIM,LAGUNA_HEAD_DIM>),
		dim3(rows,buffers->q_heads),LAGUNA_ATTN_THREADS,0,stream,buffers->q_bf16,buffers->cache,
		buffers->sequence_of_row,buffers->context_length,selected_positions,kv_positions,buffers->q_heads,
		buffers->qk_scale,buffers->attention_bf16,sliding != 0u ? 0 : buffers->row_positions);
	L7CudaCheck(cudaMemcpy(decatt_copy,buffers->attention_bf16,(size_t)q_elements * 2u,cudaMemcpyDeviceToDevice),"copy decode attention");
	status = LagunaLaunchBf16Linear(buffers->normed_bf16,(const uint16_t *)buffers->gate_weight,
		buffers->gate_bf16,buffers->dense_row_offset,buffers->dense_tile_prefix,
		rows,LAGUNA_HIDDEN,buffers->gate_rows,buffers->gate_rows,0u,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmHeadGateBroadcastKernel<LAGUNA_LAYER_THREADS,LM_GATE_SOFTPLUS>),dim3(rows,buffers->q_heads),LAGUNA_LAYER_THREADS,0,stream,
		buffers->attention_bf16,buffers->gate_bf16,buffers->q_heads,LAGUNA_HEAD_DIM);
	status = LagunaLaunchBf16Linear(buffers->attention_bf16,(const uint16_t *)buffers->output_weight,
		buffers->attention_out_bf16,buffers->dense_row_offset,buffers->dense_tile_prefix,
		rows,buffers->attn_output_columns,LAGUNA_HIDDEN,LAGUNA_HIDDEN,0u,multiprocessors,stream);
	return status;
}

static int32_t L7MoeExpertsStepped(L7LayerRun *run,L7Scratch *scratch,uint32_t tokens,uint32_t multiprocessors,cudaStream_t stream)
{
	LagunaLayerBuffers *buffers = &run->buffers;
	LmGemmArguments gemm;
	int32_t status;
	uint32_t packed_rows = tokens * LAGUNA_TOP_K;

	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<LAGUNA_EXPERT_WEIGHT_CODEC>(
		buffers->expert_w1_scale,LAGUNA_EXPERTS,buffers->expert_w1_rows,LAGUNA_HIDDEN);
	gemm.prefix_built = 1u;
	gemm.group_row_offset = buffers->group_row_offset;
	gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = tokens;
	gemm.output_bf16 = buffers->gate_up_bf16;
	status = LmGemmWeightOnlyIndirectLaunch<
		typename LmWeightCodec<LAGUNA_EXPERT_WEIGHT_CODEC>::Format,
		LAGUNA_LAYER_TILE_N,
		LAGUNA_LAYER_STAGES,
		LAGUNA_LAYER_WARPS>(
			&gemm,
			buffers->normed_bf16,
			buffers->expert_w1_weight,
			packed_rows,
			tokens,
			LAGUNA_TOP_K,
			LAGUNA_EXPERTS,
			LAGUNA_HIDDEN,
			buffers->expert_w1_rows,
			multiprocessors,
			stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<LAGUNA_LAYER_THREADS>),
		packed_rows,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->gate_up_bf16,
		buffers->intermediate_bf16,
		buffers->expert_intermediate,
		true);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<LAGUNA_EXPERT_WEIGHT_CODEC>(
		buffers->expert_w2_scale,LAGUNA_EXPERTS,LAGUNA_HIDDEN,buffers->expert_intermediate);
	gemm.prefix_built = 1u;
	gemm.group_row_offset = buffers->group_row_offset;
	gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
	gemm.output_bf16 = buffers->expert_out_bf16;
	status = LmGemmWeightOnlyLaunch<
		typename LmWeightCodec<LAGUNA_EXPERT_WEIGHT_CODEC>::Format,
		LAGUNA_LAYER_TILE_N,
		LAGUNA_LAYER_STAGES,
		LAGUNA_LAYER_WARPS>(
			&gemm,
			buffers->intermediate_bf16,
			buffers->expert_w2_weight,
			packed_rows,
			tokens,
			LAGUNA_TOP_K,
			LAGUNA_EXPERTS,
			buffers->expert_intermediate,
			LAGUNA_HIDDEN,
			multiprocessors,
			true,
			stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmMoeFinalizeKernel<LAGUNA_LAYER_THREADS>),
		dim3((LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS,tokens),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->expert_out_bf16,
		buffers->route_packed_row,
		buffers->route_weight,
		buffers->attention_out_bf16,
		tokens,
		LAGUNA_TOP_K,
		LAGUNA_HIDDEN);
	status = LagunaLaunchBf16Linear(
		buffers->normed_bf16,
		buffers->shared_gate_up_weight,
		buffers->gate_up_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		tokens,
		LAGUNA_HIDDEN,
		buffers->shared_gate_up_rows,
		buffers->shared_gate_up_rows,
		0u,
		multiprocessors,
		stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<LAGUNA_LAYER_THREADS>),
		tokens,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->gate_up_bf16,
		buffers->intermediate_bf16,
		buffers->shared_intermediate,
		true);
	status = LagunaLaunchBf16Linear(
		buffers->intermediate_bf16,
		buffers->shared_down_weight,
		buffers->shared_out_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		tokens,
		buffers->shared_intermediate,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		0u,
		multiprocessors,
		stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmAddRowsKernel<LAGUNA_LAYER_THREADS>),
		dim3((LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) / LAGUNA_LAYER_THREADS,tokens),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->attention_out_bf16,
		buffers->shared_out_bf16,
		buffers->attention_out_bf16,
		tokens,
		LAGUNA_HIDDEN);
	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static uint16_t *L7ReadHostFile(const char *path,uint64_t bytes,const char *site)
{
	uint16_t *host = (uint16_t *)malloc((size_t)bytes);
	FILE *in;
	if ( host == 0 )
		L7Die(site,"host malloc failed");
	in = fopen(path,"rb");
	if ( in == 0 )
		L7Die(site,path);
	if ( fread(host,1,(size_t)bytes,in) != (size_t)bytes )
		L7Die(site,"short file read");
	fclose(in);
	return host;
}

static uint32_t L7DeviceMultiprocessors(void)
{
	int device = 0,sm = 0;
	if ( cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device) != cudaSuccess || sm <= 0 )
		L7Die("device","no usable CUDA device");
	return (uint32_t)sm;
}

static void L7RunAttnMode(const char *pack_path,const char *dump_prefix,const char *hidden_path,
	uint32_t layer,uint32_t tokens,uint32_t tp_rank,uint32_t stage_index)
{
	L7Pack pack;
	L7Scratch scratch_product,scratch_replica;
	L7LayerRun product,replica;
	uint16_t *host_hidden;
	uint16_t *product_out,*replica_out;
	uint32_t multiprocessors = L7DeviceMultiprocessors();
	cudaStream_t stream = 0;
	char path[512];
	uint32_t index;
	int32_t status;
	uint32_t q_heads = LAGUNA_Q_HEADS(layer);

	memset(&pack,0,sizeof(pack));
	memset(&scratch_product,0,sizeof(scratch_product));
	memset(&scratch_replica,0,sizeof(scratch_replica));
	memset(&product,0,sizeof(product));
	memset(&replica,0,sizeof(replica));
	L7OpenPack(pack_path,&pack);
	if ( pack.header.tp_rank != tp_rank || pack.header.stage_index != stage_index )
		L7Die("attn","pack identity does not match --rank/--stage");
	host_hidden = L7ReadHostFile(hidden_path,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"attn hidden input");
	L7AllocateScratch(&scratch_product,tokens,2u * (LAGUNA_EXPERT_INTERMEDIATE / 8u),LAGUNA_EXPERT_INTERMEDIATE / 8u);
	L7AllocateScratch(&scratch_replica,tokens,2u * (LAGUNA_EXPERT_INTERMEDIATE / 8u),LAGUNA_EXPERT_INTERMEDIATE / 8u);
	L7SetupRun(&product,&scratch_product,tokens,layer,host_hidden);
	L7SetupRun(&replica,&scratch_replica,tokens,layer,host_hidden);
	free(host_hidden);
	L7LoadAttentionWeights(&product,&pack,layer);
	L7LoadAttentionWeights(&replica,&pack,layer);
	status = LagunaLayerAttention(&product.buffers,tokens,tokens,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		L7Die("attn product","launch failed");
	L7CudaCheck(cudaDeviceSynchronize(),"attn product sync");
	L7CheckKvError(&product,"attn product kv");
	status = L7AttentionReplica(&replica.buffers,tokens,replica.sliding,
		scratch_replica.q_normed_copy,scratch_replica.k_normed_copy,scratch_replica.decatt_copy,
		multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		L7Die("attn replica","launch failed");
	L7CudaCheck(cudaDeviceSynchronize(),"attn replica sync");
	L7CheckKvError(&replica,"attn replica kv");
	product_out = (uint16_t *)malloc((size_t)tokens * LAGUNA_HIDDEN * 2u);
	replica_out = (uint16_t *)malloc((size_t)tokens * LAGUNA_HIDDEN * 2u);
	L7Download(product_out,product.buffers.attention_out_bf16,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"download product out");
	L7Download(replica_out,replica.buffers.attention_out_bf16,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"download replica out");
	for ( index = 0u; index < tokens * LAGUNA_HIDDEN; index++ )
		if ( product_out[index] != replica_out[index] )
		{
			fprintf(stderr,"SPARK_FAIL: laguna_layer7 attn: replica unfaithful at %u: %04x vs %04x\n",
				index,product_out[index],replica_out[index]);
			exit(2);
		}
	free(product_out);
	free(replica_out);
	snprintf(path,sizeof(path),"%s_normed.bin",dump_prefix);
	L7DumpDevice(path,product.buffers.normed_bf16,(uint64_t)tokens * LAGUNA_HIDDEN * 2u);
	snprintf(path,sizeof(path),"%s_qkv.bin",dump_prefix);
	L7DumpDevice(path,replica.buffers.qkv_bf16,(uint64_t)tokens * replica.qkv_rows * 2u);
	snprintf(path,sizeof(path),"%s_qknorm_q.bin",dump_prefix);
	L7DumpDevice(path,scratch_replica.q_normed_copy,(uint64_t)tokens * q_heads * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_qknorm_k.bin",dump_prefix);
	L7DumpDevice(path,scratch_replica.k_normed_copy,(uint64_t)tokens * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_rope_q.bin",dump_prefix);
	L7DumpDevice(path,replica.buffers.q_bf16,(uint64_t)tokens * q_heads * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_rope_k.bin",dump_prefix);
	L7DumpDevice(path,replica.buffers.k_bf16,(uint64_t)tokens * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_decatt.bin",dump_prefix);
	L7DumpDevice(path,scratch_replica.decatt_copy,(uint64_t)tokens * q_heads * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_postgate.bin",dump_prefix);
	L7DumpDevice(path,replica.buffers.attention_bf16,(uint64_t)tokens * q_heads * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_postgate_product.bin",dump_prefix);
	L7DumpDevice(path,product.buffers.attention_bf16,(uint64_t)tokens * q_heads * LAGUNA_HEAD_DIM * 2u);
	snprintf(path,sizeof(path),"%s_gate.bin",dump_prefix);
	L7DumpDevice(path,replica.buffers.gate_bf16,(uint64_t)tokens * replica.gate_rows * 2u);
	snprintf(path,sizeof(path),"%s_opart.bin",dump_prefix);
	L7DumpDevice(path,replica.buffers.attention_out_bf16,(uint64_t)tokens * LAGUNA_HIDDEN * 2u);
	printf("L7RECEIPT {\"mode\":\"attn\",\"layer\":%u,\"tokens\":%u,\"tp_rank\":%u,\"stage\":%u,\"revision\":\"%s\",\"q_heads\":%u,\"qkv_rows\":%u,\"gate_rows\":%u}\n",
		layer,tokens,tp_rank,stage_index,pack.header.model_revision,replica.q_heads,replica.qkv_rows,replica.gate_rows);
}

static void L7RunMoeMode(const char *pack_path,const char *dump_prefix,const char *hidden_path,
	const char *attn_path,uint32_t layer,uint32_t tokens,uint32_t tp_rank,uint32_t stage_index)
{
	L7Pack pack;
	L7Scratch scratch;
	L7LayerRun run;
	uint16_t *host_hidden;
	uint16_t *host_attn;
	uint32_t multiprocessors = L7DeviceMultiprocessors();
	cudaStream_t stream = 0;
	char path[512];
	int32_t status;

	memset(&pack,0,sizeof(pack));
	memset(&scratch,0,sizeof(scratch));
	memset(&run,0,sizeof(run));
	L7OpenPack(pack_path,&pack);
	if ( pack.header.tp_rank != tp_rank || pack.header.stage_index != stage_index )
		L7Die("moe","pack identity does not match --rank/--stage");
	host_hidden = L7ReadHostFile(hidden_path,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"moe hidden input");
	host_attn = L7ReadHostFile(attn_path,(uint64_t)tokens * LAGUNA_HIDDEN * 2u,"moe attn input");
	L7AllocateScratch(&scratch,tokens,2u * (LAGUNA_EXPERT_INTERMEDIATE / 8u),LAGUNA_EXPERT_INTERMEDIATE / 8u);
	L7SetupRun(&run,&scratch,tokens,layer,host_hidden);
	L7CudaCheck(cudaMemcpy(run.residual,host_hidden,(size_t)tokens * LAGUNA_HIDDEN * 2u,cudaMemcpyHostToDevice),"upload residual x");
	L7CudaCheck(cudaMemcpy(run.buffers.attention_out_bf16,host_attn,(size_t)tokens * LAGUNA_HIDDEN * 2u,cudaMemcpyHostToDevice),"upload attn full");
	free(host_hidden);
	free(host_attn);
	L7LoadMlpWeights(&run,&pack,layer);
	status = LagunaLayerMoeRoute<LAGUNA_EXPERT_WEIGHT_CODEC>(&run.buffers,tokens,tokens * LAGUNA_TOP_K,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
	{
		fprintf(stderr,"SPARK_FAIL: laguna_layer7 moe route: launch status %d\n",status);
		exit(2);
	}
	L7CudaCheck(cudaDeviceSynchronize(),"moe route sync");
	status = LagunaLayerMoeExperts<LAGUNA_EXPERT_WEIGHT_CODEC>(&run.buffers,tokens,tokens * LAGUNA_TOP_K,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
	{
		fprintf(stderr,"SPARK_FAIL: laguna_layer7 moe experts: launch status %d\n",status);
		exit(2);
	}
	L7CudaCheck(cudaDeviceSynchronize(),"moe experts sync");
	L7CudaCheck(cudaDeviceSynchronize(),"moe sync");
	snprintf(path,sizeof(path),"%s_router_logits.bin",dump_prefix);
	L7DumpDevice(path,scratch.router_logits,(uint64_t)tokens * LAGUNA_EXPERTS * 4u);
	snprintf(path,sizeof(path),"%s_sets.bin",dump_prefix);
	L7DumpDevice(path,scratch.route_expert,(uint64_t)tokens * LAGUNA_TOP_K * 4u);
	snprintf(path,sizeof(path),"%s_weights.bin",dump_prefix);
	L7DumpDevice(path,scratch.route_weight,(uint64_t)tokens * LAGUNA_TOP_K * 4u);
	snprintf(path,sizeof(path),"%s_normed2.bin",dump_prefix);
	L7DumpDevice(path,scratch.normed,(uint64_t)tokens * LAGUNA_HIDDEN * 2u);
	snprintf(path,sizeof(path),"%s_moepart.bin",dump_prefix);
	L7DumpDevice(path,scratch.attention_out,(uint64_t)tokens * LAGUNA_HIDDEN * 2u);
	snprintf(path,sizeof(path),"%s_gateup.bin",dump_prefix);
	L7DumpDevice(path,scratch.gate_up,(uint64_t)tokens * LAGUNA_TOP_K * run.expert_w1_rows * 2u);
	snprintf(path,sizeof(path),"%s_interm.bin",dump_prefix);
	L7DumpDevice(path,scratch.intermediate,(uint64_t)tokens * LAGUNA_TOP_K * run.expert_intermediate * 2u);
	snprintf(path,sizeof(path),"%s_expertout.bin",dump_prefix);
	L7DumpDevice(path,scratch.expert_out,(uint64_t)tokens * LAGUNA_TOP_K * LAGUNA_HIDDEN * 2u);
	snprintf(path,sizeof(path),"%s_packedrow.bin",dump_prefix);
	L7DumpDevice(path,scratch.route_packed_row,(uint64_t)tokens * LAGUNA_TOP_K * 4u);
	printf("L7RECEIPT {\"mode\":\"moe\",\"layer\":%u,\"tokens\":%u,\"tp_rank\":%u,\"stage\":%u,\"revision\":\"%s\"}\n",
		layer,tokens,tp_rank,stage_index,pack.header.model_revision);
}

static void L7RunEmbedMode(const char *pack_path,const char *out_path,const char *tokens_text)
{
	L7Pack pack;
	const L7PackEntry *entry;
	uint16_t *row;
	uint32_t token_ids[1024];
	uint32_t count = 0u;
	char *text;
	char *cursor;
	uint32_t index;
	FILE *out;
	memset(&pack,0,sizeof(pack));
	L7OpenPack(pack_path,&pack);
	if ( pack.header.stage_index != 0u || pack.header.tp_rank != 0u )
		L7Die("embed","embedding lives in the stage0 rank0 pack");
	entry = L7FindEntry(&pack,SPARK_LAGUNA_STAGEPACK_TENSOR_EMBEDDING,SPARK_LAGUNA_STAGEPACK_GLOBAL_LAYER,"embed");
	L7RequireShape(entry,LAGUNA_VOCAB / 8u,LAGUNA_HIDDEN,0u,"shape embedding");
	text = strdup(tokens_text);
	if ( text == 0 )
		L7Die("embed","strdup failed");
	cursor = text;
	while ( *cursor != 0 && count < 1024u )
	{
		token_ids[count++] = (uint32_t)strtoul(cursor,&cursor,10);
		if ( *cursor == ',' )
			cursor++;
		else if ( *cursor != 0 )
			L7Die("embed","bad token list");
	}
	free(text);
	if ( count == 0u )
		L7Die("embed","empty token list");
	row = (uint16_t *)malloc((size_t)LAGUNA_HIDDEN * 2u);
	if ( row == 0 )
		L7Die("embed","row malloc failed");
	out = fopen(out_path,"wb");
	if ( out == 0 )
		L7Die("embed","output open failed");
	for ( index = 0u; index < count; index++ )
	{
		uint64_t offset;
		if ( token_ids[index] >= LAGUNA_VOCAB / 8u )
			L7Die("embed","token id outside the rank0 vocab slice");
		offset = entry->payload_offset + (uint64_t)token_ids[index] * LAGUNA_HIDDEN * 2u;
		if ( fseek(pack.file,(long)offset,SEEK_SET) != 0 )
			L7Die("embed","row seek failed");
		if ( fread(row,1,(size_t)LAGUNA_HIDDEN * 2u,pack.file) != (size_t)LAGUNA_HIDDEN * 2u )
			L7Die("embed","short row read");
		if ( fwrite(row,1,(size_t)LAGUNA_HIDDEN * 2u,out) != (size_t)LAGUNA_HIDDEN * 2u )
			L7Die("embed","short row write");
	}
	fclose(out);
	free(row);
	printf("L7RECEIPT {\"mode\":\"embed\",\"tokens\":%u,\"revision\":\"%s\"}\n",count,pack.header.model_revision);
}

static void L7RunRopeMode(const char *dump_prefix,const char *positions_text)
{
	uint32_t position_list[64];
	uint32_t count = 0u;
	char *text;
	char *cursor;
	int32_t sliding;
	uint32_t heads,rotary,index;
	uint16_t *host_input,*host_output;
	uint32_t *device_positions;
	uint16_t *device_rows;
	float host_yarn[LAGUNA_ROPE_FULL_ROT / 2u];
	float *device_yarn;
	uint64_t elements;
	char path[512];
	uint32_t seed = 20260911u;
	sliding = strstr(dump_prefix,"sliding") != 0 ? 1 : 0;
	text = strdup(positions_text);
	if ( text == 0 )
		L7Die("rope","strdup failed");
	cursor = text;
	while ( *cursor != 0 && count < 64u )
	{
		position_list[count++] = (uint32_t)strtoul(cursor,&cursor,10);
		if ( *cursor == ',' )
			cursor++;
		else if ( *cursor != 0 )
			L7Die("rope","bad position list");
	}
	free(text);
	if ( count == 0u )
		L7Die("rope","empty position list");
	heads = sliding ? LAGUNA_Q_HEADS_SLIDING : LAGUNA_Q_HEADS_FULL;
	rotary = sliding ? LAGUNA_ROPE_SLIDING_ROT : LAGUNA_ROPE_FULL_ROT;
	elements = (uint64_t)count * heads * LAGUNA_HEAD_DIM;
	host_input = (uint16_t *)malloc((size_t)elements * 2u);
	host_output = (uint16_t *)malloc((size_t)elements * 2u);
	if ( host_input == 0 || host_output == 0 )
		L7Die("rope","host malloc failed");
	for ( index = 0u; index < elements; index++ )
		host_input[index] = L7RandomBf16(&seed);
	L7Alloc((void **)&device_rows,elements * 2u,"alloc rope rows");
	L7Alloc((void **)&device_positions,(uint64_t)count * 4u,"alloc rope positions");
	L7Upload(device_rows,host_input,elements * 2u,"upload rope rows");
	L7Upload(device_positions,position_list,(uint64_t)count * 4u,"upload rope positions");
	if ( sliding )
	{
		LM_LAUNCH((LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),dim3(count,heads),LAGUNA_LAYER_THREADS,0,0,
			device_rows,device_positions,heads,LAGUNA_HEAD_DIM,rotary,LAGUNA_ROPE_SLIDING_THETA);
	}
	else
	{
		LagunaBuildYarnInvFrequency(host_yarn,LAGUNA_ROPE_FULL_ROT,LAGUNA_ROPE_FULL_THETA,
			SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR,SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS,
			SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST,SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW);
		L7Alloc((void **)&device_yarn,sizeof(host_yarn),"alloc rope yarn");
		L7Upload(device_yarn,host_yarn,sizeof(host_yarn),"upload rope yarn");
		LM_LAUNCH((LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),dim3(count,heads),LAGUNA_LAYER_THREADS,0,0,
			device_rows,device_positions,heads,LAGUNA_HEAD_DIM,rotary,0.0f,
			device_yarn,LAGUNA_ROPE_FULL_ATTENTION_FACTOR,0u);
	}
	L7CudaCheck(cudaDeviceSynchronize(),"rope sync");
	L7Download(host_output,device_rows,elements * 2u,"download rope output");
	snprintf(path,sizeof(path),"%s_input.bin",dump_prefix);
	{
		FILE *out = fopen(path,"wb");
		if ( out == 0 )
			L7Die("rope","input dump open failed");
		if ( fwrite(host_input,1,(size_t)elements * 2u,out) != (size_t)elements * 2u )
			L7Die("rope","input dump short write");
		fclose(out);
	}
	snprintf(path,sizeof(path),"%s_output.bin",dump_prefix);
	{
		FILE *out = fopen(path,"wb");
		if ( out == 0 )
			L7Die("rope","output dump open failed");
		if ( fwrite(host_output,1,(size_t)elements * 2u,out) != (size_t)elements * 2u )
			L7Die("rope","output dump short write");
		fclose(out);
	}
	free(host_input);
	free(host_output);
	printf("L7RECEIPT {\"mode\":\"rope\",\"regime\":\"%s\",\"positions\":%u,\"heads\":%u,\"rotary\":%u}\n",
		sliding ? "sliding" : "full",count,heads,rotary);
}

static uint32_t L7ParseUint(const char *text,const char *site)
{
	if ( text == 0 )
		L7Die(site,"missing integer argument");
	return (uint32_t)strtoul(text,0,10);
}

int main(int argc,char **argv)
{
	const char *mode = argc > 1 ? argv[1] : 0;
	const char *pack_path = 0;
	const char *dump_prefix = 0;
	const char *hidden_path = 0;
	const char *attn_path = 0;
	const char *out_path = 0;
	const char *positions_text = 0;
	uint32_t layer = 0u,tokens = 0u,tp_rank = 0u,stage_index = 0u;
	uint32_t index;
	if ( mode == 0 )
		L7Die("main","usage: laguna_layer7 <embed|attn|moe|rope> [args]");
	for ( index = 2u; (int)index < argc; index++ )
	{
		if ( strcmp(argv[index],"--pack") == 0 && (int)index + 1 < argc )
			pack_path = argv[++index];
		else if ( strcmp(argv[index],"--prefix") == 0 && (int)index + 1 < argc )
			dump_prefix = argv[++index];
		else if ( strcmp(argv[index],"--hidden") == 0 && (int)index + 1 < argc )
			hidden_path = argv[++index];
		else if ( strcmp(argv[index],"--attn") == 0 && (int)index + 1 < argc )
			attn_path = argv[++index];
		else if ( strcmp(argv[index],"--out") == 0 && (int)index + 1 < argc )
			out_path = argv[++index];
		else if ( strcmp(argv[index],"--layer") == 0 && (int)index + 1 < argc )
			layer = L7ParseUint(argv[++index],"--layer");
		else if ( strcmp(argv[index],"--tokens") == 0 && (int)index + 1 < argc )
			tokens = L7ParseUint(argv[++index],"--tokens");
		else if ( strcmp(argv[index],"--rank") == 0 && (int)index + 1 < argc )
			tp_rank = L7ParseUint(argv[++index],"--rank");
		else if ( strcmp(argv[index],"--stage") == 0 && (int)index + 1 < argc )
			stage_index = L7ParseUint(argv[++index],"--stage");
		else if ( strcmp(argv[index],"--positions") == 0 && (int)index + 1 < argc )
			positions_text = argv[++index];
		else
			L7Die("main","unknown or incomplete argument");
	}
	if ( strcmp(mode,"embed") == 0 )
	{
		if ( pack_path == 0 || out_path == 0 || positions_text == 0 )
			L7Die("embed","--pack --out --positions required");
		L7RunEmbedMode(pack_path,out_path,positions_text);
	}
	else if ( strcmp(mode,"attn") == 0 )
	{
		if ( pack_path == 0 || dump_prefix == 0 || hidden_path == 0 || tokens == 0u )
			L7Die("attn","--pack --prefix --hidden --tokens --rank --stage --layer required");
		L7RunAttnMode(pack_path,dump_prefix,hidden_path,layer,tokens,tp_rank,stage_index);
	}
	else if ( strcmp(mode,"moe") == 0 )
	{
		if ( pack_path == 0 || dump_prefix == 0 || hidden_path == 0 || attn_path == 0 || tokens == 0u )
			L7Die("moe","--pack --prefix --hidden --attn --tokens --rank --stage --layer required");
		L7RunMoeMode(pack_path,dump_prefix,hidden_path,attn_path,layer,tokens,tp_rank,stage_index);
	}
	else if ( strcmp(mode,"rope") == 0 )
	{
		if ( positions_text == 0 || dump_prefix == 0 )
			L7Die("rope","--positions p,p,.. and --prefix required");
		L7RunRopeMode(dump_prefix,positions_text);
	}
	else
		L7Die("main","unknown mode");
	return 0;
}
