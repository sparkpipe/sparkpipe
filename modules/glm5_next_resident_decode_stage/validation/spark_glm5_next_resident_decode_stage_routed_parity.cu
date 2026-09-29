#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "modules/glm5_next_resident_decode_stage/source/cuda/unity.cu"
#include "spark_glm5_next_stagepack_format.h"

#define PARITY_PATHS 4u
#define PARITY_ROW_SETS_MAX 16u
#define PARITY_LAYERS_MAX 8u

enum { PARITY_NATURAL = 0, PARITY_SKINNY = 1, PARITY_GROUPED = 2, PARITY_GEMM = 3 };

static const char *const PARITY_PATH_NAMES[PARITY_PATHS] = {"natural","skinny","grouped","gemm"};

typedef struct ParityEntry
{
	SparkGlm5NextStagePackEntry entry;
	void *payload;
	void *scale;
} ParityEntry;

typedef struct ParityWave
{
	uint32_t rows;
	uint32_t packed;
	uint16_t *x;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *group_row_offset;
	uint32_t *tile_prefix_w1;
	uint32_t *tile_prefix_w2;
	uint16_t *gate_up;
	uint16_t *intermediate;
	uint16_t *expert_out;
	uint16_t *out;
} ParityWave;

static uint64_t parity_state;
static uint32_t parity_runs_written;

static uint32_t ParityNext(void)
{
	parity_state ^= parity_state << 13;
	parity_state ^= parity_state >> 7;
	parity_state ^= parity_state << 17;
	return((uint32_t)(parity_state >> 11));
}

static float ParityNormal(void)
{
	float u = ((float)(ParityNext() & 0xffffffu) + 0.5f) / 16777216.0f;
	float v = ((float)(ParityNext() & 0xffffffu) + 0.5f) / 16777216.0f;
	return(sqrtf(-2.0f * logf(u)) * cosf(6.28318530718f * v));
}

static uint16_t ParityBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	bits += 0x7fffu + ((bits >> 16) & 1u);
	return((uint16_t)(bits >> 16));
}

static int ParityFail(const char *what,long long value)
{
	fprintf(stderr,"ROUTED-PARITY-FAIL %s %lld\n",what,value);
	return(1);
}

static int ParityCuda(cudaError_t error,const char *what)
{
	if ( error == cudaSuccess )
		return(0);
	fprintf(stderr,"ROUTED-PARITY-FAIL cuda %s: %s\n",what,cudaGetErrorString(error));
	return(1);
}

static int ParityRead(int fd,uint64_t offset,uint64_t bytes,void **device)
{
	uint8_t *host = (uint8_t *)malloc(bytes);
	uint64_t done = 0u;
	ssize_t moved;
	if ( host == 0 )
		return(ParityFail("host allocation bytes",(long long)bytes));
	while ( done < bytes )
	{
		moved = pread(fd,host + done,(size_t)(bytes - done),(off_t)(offset + done));
		if ( moved < 0 && errno == EINTR )
			continue;
		if ( moved <= 0 )
		{
			free(host);
			return(ParityFail("pread offset",(long long)(offset + done)));
		}
		done += (uint64_t)moved;
	}
	if ( ParityCuda(cudaMalloc(device,bytes),"malloc weights") != 0 || ParityCuda(cudaMemcpy(*device,host,bytes,cudaMemcpyHostToDevice),"copy weights") != 0 )
	{
		free(host);
		return(1);
	}
	free(host);
	return(0);
}

static int ParityLoadLayer(int fd,const SparkGlm5NextStagePackHeader *header,uint32_t layer,ParityEntry *up,ParityEntry *down)
{
	SparkGlm5NextStagePackEntry entry;
	uint32_t index,found = 0u;
	memset(up,0,sizeof(*up));
	memset(down,0,sizeof(*down));
	for (index=0u; index<header->tensor_count; index++)
	{
		if ( pread(fd,&entry,sizeof(entry),(off_t)(header->directory_offset + (uint64_t)index * sizeof(entry))) != (ssize_t)sizeof(entry) )
			return(ParityFail("directory entry",index));
		if ( entry.layer_index != layer || (entry.tensor_kind != SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE && entry.tensor_kind != SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN) )
			continue;
		if ( entry.weight_codec != GLM5_NEXT_EXPERT_WEIGHT_CODEC || entry.scale_encoding != SparkWeightCodecScaleEncoding(entry.weight_codec) || entry.group_count != GLM5_NEXT_EXPERTS )
			return(ParityFail("entry codec does not match the build codec",entry.weight_codec));
		ParityEntry *target = entry.tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE ? up : down;
		target->entry = entry;
		if ( ParityRead(fd,entry.payload_offset,entry.payload_bytes,&target->payload) != 0 )
			return(1);
		if ( entry.scale_bytes != 0u && ParityRead(fd,entry.scale_offset,entry.scale_bytes,&target->scale) != 0 )
			return(1);
		found++;
	}
	if ( found != 2u || up->entry.columns != GLM5_NEXT_HIDDEN || down->entry.rows != GLM5_NEXT_HIDDEN || up->entry.rows != 2u * down->entry.columns )
		return(ParityFail("layer expert entries missing or misshaped",layer));
	return(0);
}

static void ParityFreeLayer(ParityEntry *entry)
{
	(void)cudaFree(entry->payload);
	(void)cudaFree(entry->scale);
	entry->payload = 0;
	entry->scale = 0;
}

static int ParityAllocateWave(ParityWave *wave,uint32_t rows,uint32_t w1_rows,uint32_t intermediate)
{
	uint32_t packed = rows * GLM5_NEXT_TOP_K;
	memset(wave,0,sizeof(*wave));
	wave->rows = rows;
	wave->packed = packed;
	return(ParityCuda(cudaMalloc((void **)&wave->x,(size_t)rows * GLM5_NEXT_HIDDEN * 2u),"x") ||
		ParityCuda(cudaMalloc((void **)&wave->route_expert,(size_t)packed * 4u),"route_expert") ||
		ParityCuda(cudaMalloc((void **)&wave->route_weight,(size_t)packed * 4u),"route_weight") ||
		ParityCuda(cudaMalloc((void **)&wave->route_packed_row,(size_t)packed * 4u),"route_packed_row") ||
		ParityCuda(cudaMalloc((void **)&wave->route_source_token,(size_t)packed * 4u),"route_source_token") ||
		ParityCuda(cudaMalloc((void **)&wave->group_row_offset,(size_t)(GLM5_NEXT_EXPERTS + 1u) * 4u),"group_row_offset") ||
		ParityCuda(cudaMalloc((void **)&wave->tile_prefix_w1,(size_t)(GLM5_NEXT_EXPERTS + 1u) * 64u * 4u),"tile_prefix_w1") ||
		ParityCuda(cudaMalloc((void **)&wave->tile_prefix_w2,(size_t)(GLM5_NEXT_EXPERTS + 1u) * 64u * 4u),"tile_prefix_w2") ||
		ParityCuda(cudaMalloc((void **)&wave->gate_up,(size_t)packed * w1_rows * 2u),"gate_up") ||
		ParityCuda(cudaMalloc((void **)&wave->intermediate,(size_t)packed * intermediate * 2u),"intermediate") ||
		ParityCuda(cudaMalloc((void **)&wave->expert_out,(size_t)packed * GLM5_NEXT_HIDDEN * 2u),"expert_out") ||
		ParityCuda(cudaMalloc((void **)&wave->out,(size_t)rows * GLM5_NEXT_HIDDEN * 2u),"out"));
}

static void ParityFreeWave(ParityWave *wave)
{
	(void)cudaFree(wave->x); (void)cudaFree(wave->route_expert); (void)cudaFree(wave->route_weight);
	(void)cudaFree(wave->route_packed_row); (void)cudaFree(wave->route_source_token); (void)cudaFree(wave->group_row_offset);
	(void)cudaFree(wave->tile_prefix_w1); (void)cudaFree(wave->tile_prefix_w2); (void)cudaFree(wave->gate_up);
	(void)cudaFree(wave->intermediate); (void)cudaFree(wave->expert_out); (void)cudaFree(wave->out);
}

static int32_t ParityUp(uint32_t path,const ParityEntry *up,const ParityWave *wave,uint32_t sms,cudaStream_t stream)
{
	using Format = typename LmWeightCodec<GLM5_NEXT_EXPERT_WEIGHT_CODEC>::Format;
	LmGemmArguments gemm;
	LmScaleTensor scale = LmWeightCodecScaleTensor<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(up->scale,GLM5_NEXT_EXPERTS,up->entry.rows,GLM5_NEXT_HIDDEN);
	if ( path == PARITY_SKINNY )
		return(LmSkinnyExperts<Format>(up->payload,scale,wave->x,wave->gate_up,wave->route_expert,wave->route_packed_row,wave->packed,GLM5_NEXT_TOP_K,0u,GLM5_NEXT_HIDDEN,up->entry.rows,stream));
	if ( path == PARITY_GROUPED )
		return(LmSkinnyGroupedExperts<Format>(up->payload,scale,wave->x,wave->gate_up,wave->group_row_offset,wave->route_source_token,GLM5_NEXT_EXPERTS,wave->packed,0u,GLM5_NEXT_HIDDEN,up->entry.rows,stream));
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = scale;
	gemm.prefix_built = 1u;
	gemm.group_row_offset = wave->group_row_offset;
	gemm.group_tile_prefix = wave->tile_prefix_w1;
	gemm.source_row_map = wave->route_source_token;
	gemm.source_row_count = wave->rows;
	gemm.output_bf16 = wave->gate_up;
	return(LmGemmWeightOnlyIndirectLaunch<Format,GLM5_NEXT_LAYER_TILE_N,GLM5_NEXT_LAYER_STAGES,GLM5_NEXT_LAYER_WARPS>(&gemm,wave->x,up->payload,wave->packed,wave->rows,GLM5_NEXT_TOP_K,GLM5_NEXT_EXPERTS,GLM5_NEXT_HIDDEN,up->entry.rows,sms,stream));
}

static int32_t ParityDown(uint32_t path,const ParityEntry *down,const ParityWave *wave,uint32_t sms,cudaStream_t stream)
{
	using Format = typename LmWeightCodec<GLM5_NEXT_EXPERT_WEIGHT_CODEC>::Format;
	LmGemmArguments gemm;
	uint32_t intermediate = down->entry.columns;
	LmScaleTensor scale = LmWeightCodecScaleTensor<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(down->scale,GLM5_NEXT_EXPERTS,GLM5_NEXT_HIDDEN,intermediate);
	if ( path == PARITY_SKINNY )
		return(LmSkinnyExperts<Format>(down->payload,scale,wave->intermediate,wave->expert_out,wave->route_expert,wave->route_packed_row,wave->packed,GLM5_NEXT_TOP_K,1u,intermediate,GLM5_NEXT_HIDDEN,stream));
	if ( path == PARITY_GROUPED )
		return(LmSkinnyGroupedExperts<Format>(down->payload,scale,wave->intermediate,wave->expert_out,wave->group_row_offset,wave->route_source_token,GLM5_NEXT_EXPERTS,wave->packed,1u,intermediate,GLM5_NEXT_HIDDEN,stream));
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = scale;
	gemm.prefix_built = 1u;
	gemm.group_row_offset = wave->group_row_offset;
	gemm.group_tile_prefix = wave->tile_prefix_w2;
	gemm.output_bf16 = wave->expert_out;
	return(LmGemmWeightOnlyLaunch<Format,GLM5_NEXT_LAYER_TILE_N,GLM5_NEXT_LAYER_STAGES,GLM5_NEXT_LAYER_WARPS>(&gemm,wave->intermediate,down->payload,wave->packed,wave->rows,GLM5_NEXT_TOP_K,GLM5_NEXT_EXPERTS,intermediate,GLM5_NEXT_HIDDEN,sms,true,stream));
}

static int32_t ParityNatural(const ParityEntry *up,const ParityEntry *down,const ParityWave *wave,uint32_t sms,cudaStream_t stream)
{
	Glm5NextLayerBuffers buffers;
	int32_t status;
	memset(&buffers,0,sizeof(buffers));
	buffers.expert_w1_weight = up->payload;
	buffers.expert_w1_scale = up->scale;
	buffers.expert_w2_weight = down->payload;
	buffers.expert_w2_scale = down->scale;
	buffers.expert_w1_rows = up->entry.rows;
	buffers.expert_intermediate = down->entry.columns;
	buffers.normed_bf16 = wave->x;
	buffers.gate_up_bf16 = wave->gate_up;
	buffers.intermediate_bf16 = wave->intermediate;
	buffers.expert_out_bf16 = wave->expert_out;
	buffers.route_expert = wave->route_expert;
	buffers.route_weight = wave->route_weight;
	buffers.route_packed_row = wave->route_packed_row;
	buffers.route_source_token = wave->route_source_token;
	buffers.group_row_offset = wave->group_row_offset;
	buffers.group_tile_prefix_w1 = wave->tile_prefix_w1;
	buffers.group_tile_prefix_w2 = wave->tile_prefix_w2;
	status = Glm5NextLayerMoeUp<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,wave->rows,wave->packed,sms,stream);
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextLayerMoeDown<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,wave->rows,wave->packed,sms,stream);
	return(status);
}

static int ParityWrite(const char *directory,const char *name,const void *device,uint64_t bytes,uint32_t from_device)
{
	char path[1024];
	void *host = (void *)device;
	FILE *file;
	int bad;
	snprintf(path,sizeof(path),"%s/%s",directory,name);
	if ( from_device != 0u )
	{
		host = malloc(bytes);
		if ( host == 0 || ParityCuda(cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost),name) != 0 )
			return(1);
	}
	file = fopen(path,"wb");
	bad = file == 0 || fwrite(host,1u,bytes,file) != bytes || fclose(file) != 0;
	if ( from_device != 0u )
		free(host);
	return(bad ? ParityFail("write",0) : 0);
}

static int ParityRun(const char *out_root,uint32_t layer,const ParityEntry *up,const ParityEntry *down,uint32_t rows,uint32_t sms,cudaStream_t stream,FILE *meta)
{
	ParityWave wave;
	char directory[1024];
	uint16_t *x_host;
	uint32_t *expert_host,index,pick,row,path,natural_path = PARITY_PATHS,natural_equal = 0u;
	float *weight_host;
	int32_t status[PARITY_PATHS];
	uint16_t *outputs[PARITY_PATHS];
	uint64_t out_bytes = (uint64_t)rows * GLM5_NEXT_HIDDEN * 2u;
	uint32_t pool[64],pool_size = rows == 1u ? GLM5_NEXT_TOP_K : (rows * 2u < 64u ? rows * 2u : 64u);
	if ( pool_size < GLM5_NEXT_TOP_K )
		pool_size = GLM5_NEXT_TOP_K;
	if ( ParityAllocateWave(&wave,rows,up->entry.rows,down->entry.columns) != 0 )
		return(1);
	x_host = (uint16_t *)malloc(out_bytes);
	expert_host = (uint32_t *)malloc((size_t)wave.packed * 4u);
	weight_host = (float *)malloc((size_t)wave.packed * 4u);
	for (index=0u; index<pool_size; index++)
	{
		do
		{
			pick = ParityNext() % GLM5_NEXT_EXPERTS;
			for (row=0u; row<index && pool[row] != pick; row++);
		} while ( row != index );
		pool[index] = pick;
	}
	for (index=0u; index<rows * GLM5_NEXT_HIDDEN; index++)
		x_host[index] = ParityBf16(ParityNormal());
	for (row=0u; row<rows; row++)
		for (index=0u; index<GLM5_NEXT_TOP_K; index++)
		{
			uint32_t k;
			do
			{
				pick = pool[ParityNext() % pool_size];
				for (k=0u; k<index && expert_host[row * GLM5_NEXT_TOP_K + k] != pick; k++);
			} while ( k != index );
			expert_host[row * GLM5_NEXT_TOP_K + index] = pick;
			weight_host[row * GLM5_NEXT_TOP_K + index] = 0.05f + (float)(ParityNext() % 1000u) / 2000.0f;
		}
	if ( ParityCuda(cudaMemcpy(wave.x,x_host,out_bytes,cudaMemcpyHostToDevice),"x") || ParityCuda(cudaMemcpy(wave.route_expert,expert_host,(size_t)wave.packed * 4u,cudaMemcpyHostToDevice),"experts") || ParityCuda(cudaMemcpy(wave.route_weight,weight_host,(size_t)wave.packed * 4u,cudaMemcpyHostToDevice),"weights") )
		return(1);
	if ( LmRouteBuild<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_EXPERTS>(wave.route_expert,rows,wave.packed,GLM5_NEXT_TOP_K,wave.group_row_offset,wave.route_packed_row,wave.route_source_token,up->entry.rows,GLM5_NEXT_HIDDEN,GLM5_NEXT_LAYER_TILE_N,wave.tile_prefix_w1,wave.tile_prefix_w2,stream) != LM_LAUNCH_OK )
		return(ParityFail("route build rows",rows));
	snprintf(directory,sizeof(directory),"%s/layer%02u_rows%03u",out_root,layer,rows);
	if ( mkdir(directory,0755) != 0 && errno != EEXIST )
		return(ParityFail("mkdir",errno));
	for (path=0u; path<PARITY_PATHS; path++)
	{
		outputs[path] = (uint16_t *)malloc(out_bytes);
		status[path] = path == PARITY_NATURAL ? ParityNatural(up,down,&wave,sms,stream) : ParityUp(path,up,&wave,sms,stream);
		if ( path != PARITY_NATURAL && status[path] == LM_LAUNCH_OK )
		{
			LM_LAUNCH((LmClampedUpGateKernel<GLM5_NEXT_LAYER_THREADS>),wave.packed,GLM5_NEXT_LAYER_THREADS,0,stream,wave.gate_up,wave.intermediate,down->entry.columns,SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT);
			status[path] = ParityDown(path,down,&wave,sms,stream);
		}
		if ( status[path] == LM_LAUNCH_OK )
		{
			const dim3 grid((GLM5_NEXT_HIDDEN + GLM5_NEXT_LAYER_THREADS - 1u) / GLM5_NEXT_LAYER_THREADS,rows);
			LM_LAUNCH((LmMoeFinalizeKernel<GLM5_NEXT_LAYER_THREADS>),grid,GLM5_NEXT_LAYER_THREADS,0,stream,wave.expert_out,wave.route_packed_row,wave.route_weight,wave.out,rows,GLM5_NEXT_TOP_K,GLM5_NEXT_HIDDEN);
			if ( ParityCuda(cudaStreamSynchronize(stream),PARITY_PATH_NAMES[path]) != 0 || ParityCuda(cudaMemcpy(outputs[path],wave.out,out_bytes,cudaMemcpyDeviceToHost),"out") != 0 )
				return(1);
			char name[64];
			snprintf(name,sizeof(name),"out_%s.bf16",PARITY_PATH_NAMES[path]);
			if ( ParityWrite(directory,name,outputs[path],out_bytes,0u) != 0 )
				return(1);
		}
		else if ( ParityCuda(cudaStreamSynchronize(stream),"after refusal") != 0 )
			return(1);
	}
	for (path=PARITY_SKINNY; path<PARITY_PATHS && natural_path == PARITY_PATHS; path++)
		if ( status[path] == LM_LAUNCH_OK )
			natural_path = path;
	natural_equal = status[PARITY_NATURAL] == LM_LAUNCH_OK && natural_path < PARITY_PATHS && memcmp(outputs[PARITY_NATURAL],outputs[natural_path],out_bytes) == 0 ? 1u : 0u;
	if ( ParityWrite(directory,"x.bf16",x_host,out_bytes,0u) || ParityWrite(directory,"route_expert.u32",expert_host,(uint64_t)wave.packed * 4u,0u) || ParityWrite(directory,"route_weight.f32",weight_host,(uint64_t)wave.packed * 4u,0u) )
		return(1);
	fprintf(meta,"%s{\"layer\":%u,\"rows\":%u,\"dir\":\"%s\",\"status\":{\"natural\":%d,\"skinny\":%d,\"grouped\":%d,\"gemm\":%d},\"natural_path\":\"%s\",\"natural_equals_path\":%s}",
		parity_runs_written++ != 0u ? ",\n" : "",layer,rows,directory,status[0],status[1],status[2],status[3],
		natural_path < PARITY_PATHS ? PARITY_PATH_NAMES[natural_path] : "none",natural_equal != 0u ? "true" : "false");
	printf("ROUTED-PARITY layer=%u rows=%u natural=%s skinny=%d grouped=%d gemm=%d natural_bitwise_equals_path=%u\n",layer,rows,natural_path < PARITY_PATHS ? PARITY_PATH_NAMES[natural_path] : "none",status[1],status[2],status[3],natural_equal);
	for (path=0u; path<PARITY_PATHS; path++)
		free(outputs[path]);
	free(x_host);
	free(expert_host);
	free(weight_host);
	ParityFreeWave(&wave);
	return(natural_equal != 0u ? 0 : ParityFail("natural dispatch differs from the first accepted path rows",rows));
}

int main(int argc,char **argv)
{
	SparkGlm5NextStagePackHeader header;
	ParityEntry up,down;
	uint32_t layers[PARITY_LAYERS_MAX],row_sets[PARITY_ROW_SETS_MAX],layer_count = 0u,row_count = 0u,index,set;
	cudaStream_t stream;
	char meta_path[1024],*cursor,*token;
	FILE *meta;
	int fd,sms = 0,failed = 0;
	if ( argc != 6 )
	{
		fprintf(stderr,"usage: %s <pack.sp> <layers,comma> <rows,comma> <seed> <out-dir>\n",argv[0]);
		return(2);
	}
	for (cursor=argv[2]; (token = strtok(cursor,",")) != 0 && layer_count < PARITY_LAYERS_MAX; cursor = 0)
		layers[layer_count++] = (uint32_t)strtoul(token,0,10);
	for (cursor=argv[3]; (token = strtok(cursor,",")) != 0 && row_count < PARITY_ROW_SETS_MAX; cursor = 0)
		row_sets[row_count++] = (uint32_t)strtoul(token,0,10);
	parity_state = strtoull(argv[4],0,0) | 1u;
	if ( mkdir(argv[5],0755) != 0 && errno != EEXIST )
		return(ParityFail("mkdir out",errno));
	fd = open(argv[1],O_RDONLY);
	if ( fd < 0 || pread(fd,&header,sizeof(header),0) != (ssize_t)sizeof(header) || header.magic != SPARK_GLM5_NEXT_STAGEPACK_MAGIC || header.directory_entry_bytes != sizeof(SparkGlm5NextStagePackEntry) )
		return(ParityFail("pack header",fd));
	if ( ParityCuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream") || ParityCuda(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0),"sms") )
		return(1);
	snprintf(meta_path,sizeof(meta_path),"%s/meta.json",argv[5]);
	meta = fopen(meta_path,"w");
	if ( meta == 0 )
		return(ParityFail("meta",errno));
	fprintf(meta,"{\"codec\":%u,\"codec_name\":\"%s\",\"pack\":\"%s\",\"header_expert_codec\":%u,\"tp_rank_header_flags\":%u,\"seed\":\"%s\",\"runs\":[\n",
		(unsigned)GLM5_NEXT_EXPERT_WEIGHT_CODEC,GLM5_NEXT_EXPERT_CODEC_NAME,argv[1],header.expert_weight_codec,header.flags,argv[4]);
	for (index=0u; index<layer_count && failed == 0; index++)
	{
		if ( ParityLoadLayer(fd,&header,layers[index],&up,&down) != 0 )
		{
			failed = 1;
			break;
		}
		for (set=0u; set<row_count && failed == 0; set++)
			failed = ParityRun(argv[5],layers[index],&up,&down,row_sets[set],(uint32_t)sms,stream,meta);
		ParityFreeLayer(&up);
		ParityFreeLayer(&down);
	}
	fprintf(meta,"\n]}\n");
	fclose(meta);
	close(fd);
	(void)cudaStreamDestroy(stream);
	if ( failed == 0 )
		printf("ROUTED-PARITY-DUMPS-DONE codec=%s layers=%u row_sets=%u out=%s\n",GLM5_NEXT_EXPERT_CODEC_NAME,layer_count,row_count,argv[5]);
	return(failed);
}
