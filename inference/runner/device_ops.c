#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "inference/runner/device_ops.h"

#define OPS_DECODE_PARTITION_KEYS 128u
#define OPS_PREFILL_PARTITION_KEYS 1024u
#define OPS_DECODE_ROWS_PER_SEQUENCE 8u
#define OPS_GEMM_ROWS 128u
#define OPS_GEMM_OUTPUTS 64u
#define OPS_LIBRARY_NAME "sparkpipe_kernels.metallib"

static const char *const ops_kernel_names[SPARK_OPS_KERNEL_COUNT] =
{
	"spark_gather_rows",
	"spark_rmsnorm_residual",
	"spark_rmsnorm_gate",
	"spark_head_rmsnorm_rope",
	"spark_gemv_r1",
	"spark_gemv_r2",
	"spark_gemv_r4",
	"spark_gemv_r8",
	"spark_gemv_f32_r1",
	"spark_gemv_f32_r2",
	"spark_gemv_f32_r4",
	"spark_gemv_f32_r8",
	"spark_gemv_fp8_r1",
	"spark_gemv_fp8_r2",
	"spark_gemv_fp8_r4",
	"spark_gemv_fp8_r8",
	"spark_dequant_fp8",
	"spark_gemm",
	"spark_gemm_f32",
	"spark_gemm_rows16",
	"spark_gemm_rows32",
	"spark_gemm_reduce",
	"spark_causal_conv_silu",
	"spark_gated_delta_prepare",
	"spark_gated_delta_rule",
	"spark_kv_store_paged",
	"spark_attention_decode_d256",
	"spark_attention_decode_d512",
	"spark_attention_decode_d128",
	"spark_attention_decode_d192v128",
	"spark_attention_decode_d576v512",
	"spark_attention_prefill_blocks_d256",
	"spark_attention_prefill_blocks_d512",
	"spark_attention_prefill_blocks_d128",
	"spark_attention_prefill_blocks_d192v128",
	"spark_attention_prefill_blocks_d576v512",
	"spark_attention_combine_d256",
	"spark_attention_combine_d512",
	"spark_attention_combine_d128",
	"spark_attention_combine_d192v128",
	"spark_attention_combine_d576v512",
	"spark_silu_mul",
	"spark_gelu_tanh_mul",
	"spark_silu_mul_routed",
	"spark_gelu_tanh_mul_routed",
	"spark_rmsnorm_branches",
	"spark_rmsnorm_scaled",
	"spark_scale_rows",
	"spark_softcap_rows",
	"spark_route_topk",
	"spark_route_pinned",
	"spark_route_combine",
	"spark_gemv_routed",
	"spark_gemv_mxfp4_routed",
	"spark_add_rows",
	"spark_logprob_rows",
	"spark_stream_read",
	"spark_silu_rounded_mul",
	"spark_route_combine_weighted",
	"spark_route_sigmoid_topk",
	"spark_rmsnorm_bf16",
	"spark_hyper_expand",
	"spark_group_rmsnorm",
	"spark_hyper_gate",
	"spark_hyper_mix",
	"spark_hyper_inject_weights",
	"spark_hyper_inject",
	"spark_hyper_key_gate",
	"spark_hyper_add",
	"spark_gemv_nvfp4_routed",
	"spark_route_combine_shared",
	"spark_sigmoid_gate_rows",
	"spark_ngram_hash",
	"spark_ngram_gather_fp8",
	"spark_ngram_history",
	"spark_route_local",
	"spark_rank_sum_bf16",
	"spark_rank_max_u64",
	"spark_rank_rows",
	"spark_head_gate",
	"spark_channel_delta_prepare",
	"spark_silu_clamp_mul",
	"spark_silu_clamp_mul_routed",
	"spark_silu_gate_limit_mul",
	"spark_silu_gate_limit_mul_routed",
	"spark_gemv_mxfp8_r1",
	"spark_gemv_mxfp8_r2",
	"spark_gemv_mxfp8_r4",
	"spark_gemv_mxfp8_r8",
	"spark_dequant_mxfp8",
	"spark_gemv_mxfp8_routed",
	"spark_hc_mix8",
	"spark_hc_mix4",
	"spark_hc_post",
	"spark_hc_site",
	"spark_hc_post_comb",
	"spark_stream_mean",
	"spark_gemv_fp8_routed",
	"spark_situ_mul",
	"spark_attnres_score",
	"spark_attnres_mix",
	"spark_gemm_fp8_blocks",
	"spark_gemm_mxfp8_blocks",
	"spark_route_sqrt_softplus_topk",
	"spark_block_qdq",
	"spark_kv_compress",
	"spark_kv_compress_save",
	"spark_compressed_positions",
	"spark_hc_site_lagged",
	"spark_hc_post_comb_f32",
	"spark_hc_collapse",
	"spark_stream_key_gate",
	"spark_kv_store_paged_skip",
	"spark_attention_decode_extra_d512",
	"spark_attention_combine_extra_d512",
	"spark_ngram_hash_mapped",
	"spark_ngram_gather_e8m0",
	"spark_ngram_history_mapped",
	"spark_poly_norm_mul",
	"spark_differential_gate",
	"spark_hc_site_clamped"
};

static SparkStatus OpsRefuse(const char *operation, const char *reason)
{
	fprintf(stderr,"device_ops: %s refused: %s\n",operation,reason);
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus OpsNeed(const char *operation, const char *name, SparkTensor tensor, uint64_t bytes)
{
	if ( tensor.range.buffer == 0 || tensor.bytes < bytes || bytes == 0u )
	{
		fprintf(stderr,"device_ops: %s refused: tensor %s holds %llu bytes, the shape needs %llu\n",operation,name,
			(unsigned long long)tensor.bytes,(unsigned long long)bytes);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static uint64_t OpsRowsBytes(uint32_t rows, uint32_t stride, uint32_t width, uint32_t element)
{
	return(((uint64_t)(rows - 1u) * stride + width) * element);
}

SparkTensor SparkTensorAt(const SparkDeviceBuffer *buffer, uint64_t offset, uint64_t bytes)
{
	SparkTensor tensor;
	tensor.range.buffer = buffer;
	tensor.range.offset = offset;
	tensor.bytes = bytes;
	return(tensor);
}

SparkTensor SparkTensorSlice(SparkTensor tensor, uint64_t offset, uint64_t bytes)
{
	SparkTensor slice = tensor;
	slice.range.offset += offset;
	slice.bytes = offset <= tensor.bytes && bytes <= tensor.bytes - offset ? bytes : 0u;
	return(slice);
}

static SparkDeviceBinding OpsBind(SparkTensor tensor, uint32_t access)
{
	SparkDeviceBinding binding;
	binding.range = tensor.range;
	binding.bytes = tensor.bytes;
	binding.access = access;
	return(binding);
}

static SparkStatus OpsLaunch(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkOpsKernel kernel, uint32_t gx, uint32_t gy, uint32_t gz,
	uint32_t threads, const void *params, uint32_t params_bytes, const SparkDeviceBinding *bindings, uint32_t count)
{
	SparkDeviceLaunch launch;
	SparkStatus status;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = gx;
	launch.groups.y = gy;
	launch.groups.z = gz;
	launch.threads.x = threads;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = params;
	launch.constant_bytes = params_bytes;
	launch.bindings = bindings;
	launch.binding_count = count;
	status = SparkDeviceKernelLaunch(queue,ops->kernels[kernel],&launch);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"device_ops: %s launch failed: %s\n",ops_kernel_names[kernel],SparkDeviceLastError()->text);
	return(status);
}

SparkStatus SparkDeviceOpsOpen(SparkDeviceOps *ops, const char *library_path)
{
	SparkStatus status;
	uint32_t index;
	memset(ops,0,sizeof(*ops));
	status = SparkDeviceLibraryLoad(&ops->library,library_path,"device ops");
	for ( index = 0u; status == SPARK_STATUS_OK && index < SPARK_OPS_KERNEL_COUNT; index++ )
		status = SparkDeviceKernelCreate(&ops->kernels[index],ops->library,ops_kernel_names[index]);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAllocate(&ops->error,SPARK_DEVICE_SPACE_SHARED_COHERENT,16u,"device ops error word");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAllocate(&ops->partial,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,SPARK_OPS_PARTIAL_BYTES,"device ops split partials");
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"device_ops: cannot open %s: %s %s\n",library_path,SparkDeviceLastError()->operation,SparkDeviceLastError()->text);
		SparkDeviceOpsClose(ops);
	}
	return(status);
}

void SparkDeviceOpsClose(SparkDeviceOps *ops)
{
	uint32_t index;
	if ( ops->error.handle != 0u )
		(void)SparkDeviceFree(&ops->error);
	if ( ops->partial.handle != 0u )
		(void)SparkDeviceFree(&ops->partial);
	for ( index = 0u; index < SPARK_OPS_KERNEL_COUNT; index++ )
		if ( ops->kernels[index] != 0 )
			(void)SparkDeviceKernelDestroy(ops->kernels[index]);
	if ( ops->library != 0 )
		(void)SparkDeviceLibraryUnload(ops->library);
	memset(ops,0,sizeof(*ops));
}

static pthread_mutex_t ops_shared_mutex = PTHREAD_MUTEX_INITIALIZER;
static SparkDeviceOps ops_shared;
static uint32_t ops_shared_users;
static SparkDeviceOps ops_lanes[SPARK_OPS_MAX_LANES];
static uint32_t ops_lane_users[SPARK_OPS_MAX_LANES];

static SparkStatus OpsSharedAcquireLocked(void)
{
	char path[PATH_MAX];
	Dl_info info;
	const char *slash;
	SparkStatus status = SPARK_STATUS_OK;
	if ( ops_shared_users == 0u )
	{
		if ( dladdr((const void *)SparkDeviceInitialize,&info) == 0 || info.dli_fname == 0 || (slash = strrchr(info.dli_fname,'/')) == 0 ||
			(size_t)(slash - info.dli_fname) + sizeof(OPS_LIBRARY_NAME) + 1u >= sizeof(path) )
			status = OpsRefuse("acquire","the device library's directory cannot be found, so its kernel library cannot be located");
		else
		{
			memcpy(path,info.dli_fname,(size_t)(slash - info.dli_fname) + 1u);
			memcpy(path + (slash - info.dli_fname) + 1,OPS_LIBRARY_NAME,sizeof(OPS_LIBRARY_NAME));
			status = SparkDeviceOpsOpen(&ops_shared,path);
		}
	}
	if ( status == SPARK_STATUS_OK )
		ops_shared_users++;
	return(status);
}

static void OpsSharedReleaseLocked(void)
{
	if ( ops_shared_users != 0u && --ops_shared_users == 0u )
		SparkDeviceOpsClose(&ops_shared);
}

static void OpsLaneFree(SparkDeviceOps *lane)
{
	if ( lane->error.handle != 0u )
		(void)SparkDeviceFree(&lane->error);
	if ( lane->partial.handle != 0u )
		(void)SparkDeviceFree(&lane->partial);
	memset(lane,0,sizeof(*lane));
}

SparkStatus SparkDeviceOpsAcquire(SparkDeviceOps **ops)
{
	return(SparkDeviceOpsAcquireLane(ops,0u));
}

SparkStatus SparkDeviceOpsAcquireLane(SparkDeviceOps **ops, uint32_t lane)
{
	SparkDeviceOps *instance = lane < SPARK_OPS_MAX_LANES ? &ops_lanes[lane] : 0;
	SparkStatus status;
	*ops = 0;
	if ( instance == 0 )
		return(OpsRefuse("acquire","an ops lane is below SPARK_OPS_MAX_LANES"));
	pthread_mutex_lock(&ops_shared_mutex);
	status = OpsSharedAcquireLocked();
	if ( status == SPARK_STATUS_OK && lane == 0u )
		*ops = &ops_shared;
	else if ( status == SPARK_STATUS_OK && ops_lane_users[lane] != 0u )
	{
		ops_lane_users[lane]++;
		*ops = instance;
	}
	else if ( status == SPARK_STATUS_OK )
	{
		memset(instance,0,sizeof(*instance));
		instance->library = ops_shared.library;
		memcpy(instance->kernels,ops_shared.kernels,sizeof(instance->kernels));
		instance->lane = lane;
		status = SparkDeviceAllocate(&instance->error,SPARK_DEVICE_SPACE_SHARED_COHERENT,16u,"device ops lane error word");
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceAllocate(&instance->partial,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,SPARK_OPS_PARTIAL_BYTES,"device ops lane split partials");
		if ( status == SPARK_STATUS_OK )
		{
			ops_lane_users[lane] = 1u;
			*ops = instance;
		}
		else
		{
			OpsLaneFree(instance);
			OpsSharedReleaseLocked();
		}
	}
	pthread_mutex_unlock(&ops_shared_mutex);
	return(status);
}

void SparkDeviceOpsRelease(SparkDeviceOps *ops)
{
	pthread_mutex_lock(&ops_shared_mutex);
	if ( ops != 0 && ops != &ops_shared && ops >= ops_lanes && ops < ops_lanes + SPARK_OPS_MAX_LANES && ops_lane_users[ops->lane] != 0u &&
		--ops_lane_users[ops->lane] == 0u )
		OpsLaneFree(ops);
	if ( ops == &ops_shared || (ops >= ops_lanes && ops < ops_lanes + SPARK_OPS_MAX_LANES) )
		OpsSharedReleaseLocked();
	pthread_mutex_unlock(&ops_shared_mutex);
}

SparkStatus SparkDeviceOpsTakeError(SparkDeviceOps *ops, uint32_t *error)
{
	uint32_t *word = (uint32_t *)ops->error.host;
	*error = __atomic_exchange_n(word,0u,__ATOMIC_SEQ_CST);
	return(*error == 0u ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

static SparkTensor OpsErrorWord(SparkDeviceOps *ops)
{
	return(SparkTensorAt(&ops->error,0u,4u));
}

static SparkStatus OpsGather(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows, uint32_t first_row, uint32_t sliced)
{
	SparkGatherParams params;
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( rows == 0u || width == 0u || width % 8u != 0u )
		return(OpsRefuse("gather","rows and a width that is a multiple of 8 are required"));
	status = OpsNeed("gather","table",table,(uint64_t)table_rows * width * 2u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gather","ids",ids,(uint64_t)rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gather","output",output,(uint64_t)rows * width * 2u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.width = width;
	params.table_rows = table_rows;
	params.first_row = first_row;
	params.sliced = sliced;
	bindings[0] = OpsBind(table,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(ids,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_GATHER,(width / 8u + 255u) / 256u,rows,1u,256u,&params,sizeof(params),bindings,4u));
}

SparkStatus SparkOpsGather(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows)
{
	return(OpsGather(ops,queue,table,ids,output,rows,width,table_rows,0u,0u));
}

SparkStatus SparkOpsGatherSlice(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows, uint32_t first_row)
{
	return(OpsGather(ops,queue,table,ids,output,rows,width,table_rows,first_row,1u));
}

SparkStatus SparkOpsRouteLocal(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor experts, const SparkRouteParams *params)
{
	const uint64_t routes = (uint64_t)params->rows * params->top_k;
	SparkDeviceBinding binding;
	SparkStatus status;
	if ( params->rows == 0u || params->top_k == 0u || params->local_experts == 0u || params->first_expert + params->local_experts > params->experts )
		return(OpsRefuse("route_local","rows, a top-k and a local expert range inside the layer's experts are required"));
	status = OpsNeed("route_local","experts",experts,routes * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	binding = OpsBind(experts,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ROUTE_LOCAL,(uint32_t)((routes + 255u) / 256u),1u,1u,256u,params,sizeof(*params),&binding,1u));
}

static SparkStatus OpsRankCombine(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkOpsKernel kernel, const char *operation, SparkTensor lanes,
	SparkTensor full, uint32_t ranks, uint64_t elements, uint32_t element_bytes)
{
	SparkRankParams params;
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( ranks == 0u || elements == 0u || elements > UINT32_MAX )
		return(OpsRefuse(operation,"ranks and at most 2^32 elements are required"));
	status = OpsNeed(operation,"lanes",lanes,(uint64_t)ranks * elements * element_bytes);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed(operation,"full",full,elements * element_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&params,0,sizeof(params));
	params.elements = elements;
	params.ranks = ranks;
	bindings[0] = OpsBind(lanes,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(full,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,kernel,(uint32_t)((elements + 255u) / 256u),1u,1u,256u,&params,sizeof(params),bindings,2u));
}

SparkStatus SparkOpsRankSumBf16(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor lanes, SparkTensor full, uint32_t ranks, uint64_t elements)
{
	return(OpsRankCombine(ops,queue,SPARK_OPS_RANK_SUM_BF16,"rank_sum_bf16",lanes,full,ranks,elements,2u));
}

SparkStatus SparkOpsRankMaxU64(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor lanes, SparkTensor full, uint32_t ranks, uint64_t elements)
{
	return(OpsRankCombine(ops,queue,SPARK_OPS_RANK_MAX_U64,"rank_max_u64",lanes,full,ranks,elements,8u));
}

SparkStatus SparkOpsRankRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor gathered, SparkTensor rows, uint32_t ranks, uint32_t row_count,
	uint32_t width)
{
	const uint64_t bytes = (uint64_t)ranks * row_count * width * sizeof(uint16_t);
	SparkRankParams params;
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( ranks == 0u || row_count == 0u || width == 0u || (uint64_t)ranks * width > UINT32_MAX )
		return(OpsRefuse("rank_rows","ranks, rows and a width are required"));
	status = OpsNeed("rank_rows","gathered",gathered,bytes);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rank_rows","rows",rows,bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&params,0,sizeof(params));
	params.elements = (uint64_t)row_count * width;
	params.ranks = ranks;
	params.rows = row_count;
	params.width = width;
	bindings[0] = OpsBind(gathered,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(rows,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_RANK_ROWS,(ranks * width + 255u) / 256u,row_count,1u,256u,&params,sizeof(params),bindings,2u));
}

static uint32_t OpsThreads(uint32_t width, uint32_t limit)
{
	uint32_t threads = (width + 31u) / 32u * 32u;
	return(threads > limit ? limit : threads);
}

SparkStatus SparkOpsHeadGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, SparkTensor logits, const SparkHeadGateParams *params)
{
	const uint64_t gate_width = params->gate_head_stride != 0u ? (uint64_t)(params->heads - 1u) * params->gate_head_stride + params->gate_offset +
		params->head_dimension : params->heads;
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->heads == 0u || params->head_dimension == 0u || params->gate_stride < gate_width ||
		(params->gate_head_stride != 0u && params->gate_head_stride < params->head_dimension) ||
		(params->activation != SPARK_HEAD_GATE_SOFTPLUS && params->activation != SPARK_HEAD_GATE_SIGMOID_BF16 && params->activation != SPARK_HEAD_GATE_SIGMOID) )
		return(OpsRefuse("head_gate","rows, heads, a head width, a gate row holding every head's gate and a known activation are required"));
	status = OpsNeed("head_gate","values",values,(uint64_t)params->rows * params->heads * params->head_dimension * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("head_gate","logits",logits,OpsRowsBytes(params->rows,params->gate_stride,(uint32_t)gate_width,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(values,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(logits,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_HEAD_GATE,params->heads,params->rows,1u,OpsThreads(params->head_dimension,256u),params,sizeof(*params),
		bindings,2u));
}

SparkStatus SparkOpsRmsNormResidual(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, const SparkTensor *addend,
	SparkTensor weight, SparkTensor output, const SparkNormParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || (addend != 0) != (params->has_addend != 0u) ||
		(params->write_sum != 0u && params->has_addend == 0u) )
		return(OpsRefuse("rmsnorm_residual","rows, a dimension and a residual whenever the sum is written are required"));
	status = OpsNeed("rmsnorm_residual","hidden",hidden,OpsRowsBytes(params->rows,params->input_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK && addend != 0 )
		status = OpsNeed("rmsnorm_residual","addend",*addend,OpsRowsBytes(params->rows,params->input_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK && params->weightless == 0u )
		status = OpsNeed("rmsnorm_residual","weight",weight,(uint64_t)params->dimension * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_residual","output",output,OpsRowsBytes(params->rows,params->output_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(hidden,params->write_sum != 0u ? SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE : SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(addend != 0 ? *addend : hidden,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(params->weightless != 0u ? hidden : weight,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_RMSNORM_RESIDUAL,params->rows,1u,1u,OpsThreads(params->dimension,1024u),params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsRmsNormGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor gate, SparkTensor weight,
	SparkTensor output, const SparkGatedNormParams *params)
{
	const uint32_t width = params->heads * params->head_dimension;
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->heads == 0u || params->head_dimension == 0u )
		return(OpsRefuse("rmsnorm_gate","rows, heads and a head dimension are required"));
	status = OpsNeed("rmsnorm_gate","input",input,OpsRowsBytes(params->rows,params->input_stride,width,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_gate","gate",gate,OpsRowsBytes(params->rows,params->gate_stride,params->gate_offset + width,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_gate","weight",weight,(uint64_t)params->head_dimension * (params->weight_f32 != 0u ? 4u : 2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_gate","output",output,OpsRowsBytes(params->rows,params->output_stride,width,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(gate,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_RMSNORM_GATE,params->heads,params->rows,1u,OpsThreads(params->head_dimension,1024u),params,sizeof(*params),
		bindings,4u));
}

SparkStatus SparkOpsHeadRmsNormRope(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor positions,
	SparkTensor output, const SparkHeadRopeParams *params)
{
	SparkHeadRopeParams launch = *params;
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->heads == 0u || params->head_dimension == 0u || params->head_dimension > 1024u ||
		params->rope_dimension % 2u != 0u || params->rope_dimension > params->head_dimension ||
		params->rope_dimension / 2u > SPARK_KERNEL_ROPE_PAIRS || (params->output_head_stride != 0u && params->output_head_stride < params->head_dimension) )
		return(OpsRefuse("head_rmsnorm_rope","heads of at most 1024 dimensions, an even rotary width inside the head and output heads that do not overlap are required"));
	if ( launch.scale == 0.0f )
		launch.scale = 1.0f;
	status = OpsNeed("head_rmsnorm_rope","input",input,
		OpsRowsBytes(params->rows,params->input_stride,params->input_offset + (params->heads - 1u) * params->input_head_stride + params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK && params->has_weight != 0u )
		status = OpsNeed("head_rmsnorm_rope","weight",weight,(uint64_t)params->head_dimension * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("head_rmsnorm_rope","positions",positions,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("head_rmsnorm_rope","output",output,OpsRowsBytes(params->rows,params->output_stride,
			(params->heads - 1u) * (params->output_head_stride != 0u ? params->output_head_stride : params->head_dimension) + params->head_dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(params->has_weight != 0u ? weight : positions,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(positions,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HEAD_RMSNORM_ROPE,params->heads,params->rows,1u,OpsThreads(params->head_dimension,1024u),&launch,
		sizeof(launch),bindings,4u));
}

static SparkOpsKernel OpsGemvKernel(const SparkGemvParams *params, uint32_t rows)
{
	const uint32_t slot = rows == 8u ? 3u : rows == 4u ? 2u : rows == 2u ? 1u : 0u;
	if ( params->output_f32 != 0u )
		return((SparkOpsKernel)(SPARK_OPS_GEMV_F32_R1 + slot));
	return((SparkOpsKernel)(SPARK_OPS_GEMV_R1 + slot));
}

static SparkStatus OpsLinearRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkDeviceBinding *bindings, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, depth = SPARK_GEMM_ROWS_DEPTH;
	const uint32_t tiles = (N + SPARK_GEMM_ROWS_OUTPUTS - 1u) / SPARK_GEMM_ROWS_OUTPUTS, tile_rows = params->rows <= 16u ? 16u : 32u;
	SparkGemvParams launch = *params;
	SparkDeviceBinding staged[3];
	uint32_t splits = (SPARK_OPS_SPLIT_TARGET_GROUPS + tiles - 1u) / tiles;
	SparkStatus status;
	if ( splits > K / depth )
		splits = K / depth;
	launch.split_span = (K / depth + splits - 1u) / splits * depth;
	launch.splits = (K + launch.split_span - 1u) / launch.split_span;
	staged[0] = bindings[0];
	staged[1] = bindings[1];
	staged[2] = bindings[2];
	if ( launch.splits > 1u )
	{
		const uint64_t bytes = (uint64_t)launch.splits * params->rows * N * sizeof(float);
		if ( bytes > ops->partial.bytes )
			return(OpsRefuse("linear","the split partials of this shape do not fit the ops partial buffer"));
		if ( ops->partial_queue != 0 && ops->partial_queue != queue )
			return(OpsRefuse("linear","the split partials belong to another queue; split projections run on one queue per ops instance"));
		ops->partial_queue = queue;
		staged[2] = OpsBind(SparkTensorAt(&ops->partial,0u,bytes),SPARK_DEVICE_BIND_WRITE);
	}
	status = OpsLaunch(ops,queue,tile_rows == 16u ? SPARK_OPS_GEMM_ROWS16 : SPARK_OPS_GEMM_ROWS32,(params->rows + tile_rows - 1u) / tile_rows,tiles,
		launch.splits,SPARK_GEMM_ROWS_THREADS,&launch,sizeof(launch),staged,3u);
	if ( status != SPARK_STATUS_OK || launch.splits == 1u )
		return(status);
	staged[0] = OpsBind(SparkTensorAt(&ops->partial,0u,(uint64_t)launch.splits * params->rows * N * sizeof(float)),SPARK_DEVICE_BIND_READ);
	staged[1] = bindings[2];
	return(OpsLaunch(ops,queue,SPARK_OPS_GEMM_REDUCE,(N + 255u) / 256u,params->rows,1u,256u,&launch,sizeof(launch),staged,2u));
}

SparkStatus SparkOpsLinear(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension;
	const uint32_t element = params->output_f32 != 0u ? (uint32_t)sizeof(float) : (uint32_t)sizeof(uint16_t);
	SparkDeviceBinding bindings[3];
	SparkGemvParams chunk;
	SparkStatus status;
	uint32_t done;
	const uint32_t gemv = (params->projection == SPARK_PROJECTION_GEMV && (params->rows <= SPARK_OPS_GEMV_MAX_ROWS || params->row_exact != 0u)) ||
		K % SPARK_GEMM_ROWS_DEPTH != 0u ? 1u : 0u;
	if ( params->rows == 0u || K == 0u || N == 0u || K % 8u != 0u || params->input_stride % 8u != 0u || params->input_stride < K ||
		(params->projection != SPARK_PROJECTION_GEMV && params->projection != SPARK_PROJECTION_ROW_TILES) ||
		(params->projection == SPARK_PROJECTION_ROW_TILES && params->row_exact != 0u && params->rows > SPARK_OPS_DECODE_MAX_ROWS) )
		return(OpsRefuse("linear","a known projection family, an input width that is a multiple of 8, a stride that covers it, "
			"and row-exact frames of at most the decode rows on row tiles"));
	status = OpsNeed("linear","input",x,OpsRowsBytes(params->rows,params->input_stride,K,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear","weight",w,(uint64_t)N * K * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear","output",y,OpsRowsBytes(params->rows,params->output_stride,N,element));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
	if ( gemv == 0u && params->rows > SPARK_OPS_DECODE_MAX_ROWS )
		return(OpsLaunch(ops,queue,params->output_f32 != 0u ? SPARK_OPS_GEMM_F32 : SPARK_OPS_GEMM,(N + OPS_GEMM_OUTPUTS - 1u) / OPS_GEMM_OUTPUTS,
			(params->rows + OPS_GEMM_ROWS - 1u) / OPS_GEMM_ROWS,1u,128u,params,sizeof(*params),bindings,3u));
	if ( gemv == 0u )
		return(OpsLinearRows(ops,queue,bindings,params));
	for ( done = 0u; status == SPARK_STATUS_OK && done < params->rows; done += chunk.rows )
	{
		const uint32_t left = params->rows - done;
		chunk = *params;
		chunk.rows = left >= SPARK_OPS_GEMV_PASS_ROWS ? SPARK_OPS_GEMV_PASS_ROWS : left >= 4u ? 4u : left >= 2u ? 2u : 1u;
		bindings[0] = OpsBind(SparkTensorSlice(x,(uint64_t)done * params->input_stride * sizeof(uint16_t),
			OpsRowsBytes(chunk.rows,params->input_stride,K,2u)),SPARK_DEVICE_BIND_READ);
		bindings[2] = OpsBind(SparkTensorSlice(y,(uint64_t)done * params->output_stride * element,OpsRowsBytes(chunk.rows,params->output_stride,N,element)),
			SPARK_DEVICE_BIND_WRITE);
		const uint32_t outputs = SPARK_GEMV_ROW_OUTPUTS(chunk.rows);
		status = OpsLaunch(ops,queue,OpsGemvKernel(params,chunk.rows),(N + outputs - 1u) / outputs,1u,1u,128u,&chunk,sizeof(chunk),bindings,3u);
	}
	return(status);
}

SparkStatus SparkOpsLinearFp8(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor dequant,
	SparkTensor y, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, mx = params->scale_group == SPARK_MXFP8_GROUP ? 1u : 0u;
	const uint32_t scale_columns = mx != 0u ? K / SPARK_MXFP8_GROUP : (K + SPARK_FP8_SCALE_K - 1u) / SPARK_FP8_SCALE_K;
	SparkDeviceBinding bindings[5];
	SparkGemvParams chunk;
	SparkStatus status;
	uint32_t done;
	if ( params->rows == 0u || K == 0u || N == 0u || K % (mx != 0u ? SPARK_MXFP8_GROUP : SPARK_FP8_CHUNK) != 0u || params->input_stride % 8u != 0u ||
		params->input_stride < K || params->output_f32 != 0u || params->scale_stride < scale_columns || (params->scale_group != 0u && mx == 0u) )
		return(OpsRefuse("linear_fp8","an input width that is a multiple of 16 (32 for MX scales), a stride that covers it, BF16 output and a scale row "
			"per 128 inputs in f32 or per 32 inputs in UE8M0"));
	status = OpsNeed("linear_fp8","input",x,OpsRowsBytes(params->rows,params->input_stride,K,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_fp8","weight",w,(uint64_t)N * K);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_fp8","scale",scale,(uint64_t)N * params->scale_stride * (mx != 0u ? 1u : sizeof(float)));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_fp8","output",y,OpsRowsBytes(params->rows,params->output_stride,N,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( params->rows > SPARK_OPS_FP8_GEMV_ROWS && params->row_exact == 0u && K % (mx != 0u ? SPARK_MXFP8_GROUP : SPARK_FP8_SCALE_K) == 0u )
	{
		status = OpsNeed("linear_fp8","dequant",dequant,(uint64_t)N * K * sizeof(uint16_t));
		if ( status != SPARK_STATUS_OK )
			return(status);
		bindings[0] = OpsBind(w,SPARK_DEVICE_BIND_READ);
		bindings[1] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
		bindings[2] = OpsBind(dequant,SPARK_DEVICE_BIND_WRITE);
		status = OpsLaunch(ops,queue,mx != 0u ? SPARK_OPS_DEQUANT_MXFP8 : SPARK_OPS_DEQUANT_FP8,(K / 4u + 255u) / 256u,N,1u,256u,params,sizeof(*params),
			bindings,3u);
		if ( status != SPARK_STATUS_OK )
			return(status);
		bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
		bindings[1] = OpsBind(dequant,SPARK_DEVICE_BIND_READ);
		bindings[2] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
		bindings[3] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
		return(OpsLaunch(ops,queue,mx != 0u ? SPARK_OPS_GEMM_MXFP8_BLOCKS : SPARK_OPS_GEMM_FP8_BLOCKS,(N + OPS_GEMM_OUTPUTS - 1u) / OPS_GEMM_OUTPUTS,
			(params->rows + OPS_GEMM_ROWS - 1u) / OPS_GEMM_ROWS,1u,128u,params,sizeof(*params),bindings,4u));
	}
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	for ( done = 0u; status == SPARK_STATUS_OK && done < params->rows; done += chunk.rows )
	{
		const uint32_t left = params->rows - done;
		chunk = *params;
		chunk.rows = left >= SPARK_OPS_GEMV_PASS_ROWS ? SPARK_OPS_GEMV_PASS_ROWS : left >= 4u ? 4u : left >= 2u ? 2u : 1u;
		bindings[0] = OpsBind(SparkTensorSlice(x,(uint64_t)done * params->input_stride * sizeof(uint16_t),
			OpsRowsBytes(chunk.rows,params->input_stride,K,2u)),SPARK_DEVICE_BIND_READ);
		bindings[3] = OpsBind(SparkTensorSlice(y,(uint64_t)done * params->output_stride * sizeof(uint16_t),OpsRowsBytes(chunk.rows,params->output_stride,N,2u)),
			SPARK_DEVICE_BIND_WRITE);
		const uint32_t outputs = SPARK_GEMV_ROW_OUTPUTS(chunk.rows);
		const uint32_t slot = chunk.rows == 8u ? 3u : chunk.rows == 4u ? 2u : chunk.rows == 2u ? 1u : 0u;
		status = OpsLaunch(ops,queue,(SparkOpsKernel)((mx != 0u ? SPARK_OPS_GEMV_MXFP8_R1 : SPARK_OPS_GEMV_FP8_R1) + slot),(N + outputs - 1u) / outputs,1u,1u,
			128u,&chunk,sizeof(chunk),bindings,4u);
	}
	return(status);
}

SparkStatus SparkOpsCausalConv(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor window, const SparkOpsSequences *sequences,
	SparkTensor input, SparkTensor weight, SparkTensor output, const SparkConvParams *params)
{
	SparkDeviceBinding bindings[6];
	SparkStatus status;
	if ( params->channels == 0u || params->sequences == 0u || params->kernel_taps == 0u || params->kernel_taps > 8u ||
		SPARK_CONV_SPAN(params->kernel_taps,params->dilation) > SPARK_CONV_SPAN_MAX || params->sequences != sequences->sequences ||
		params->commit != sequences->commit )
		return(OpsRefuse("causal_conv","channels, one to eight taps spanning at most 16 entries and matching sequence tables are required"));
	status = OpsNeed("causal_conv","state_index",sequences->state_index,(uint64_t)params->sequences * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("causal_conv","row_begin",sequences->row_begin,((uint64_t)params->sequences + 1u) * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("causal_conv","weight",weight,(uint64_t)params->channels * params->kernel_taps * (params->weight_f32 != 0u ? 4u : 2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("causal_conv","window",window,(uint64_t)params->channels * SPARK_CONV_SPAN(params->kernel_taps,params->dilation) * 2u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(window,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(sequences->state_index,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(sequences->row_begin,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_CAUSAL_CONV,(params->channels + 255u) / 256u,params->sequences,1u,256u,params,sizeof(*params),bindings,6u));
}

uint64_t SparkOpsDeltaScratchBytes(uint32_t rows, uint32_t key_heads, uint32_t value_heads, uint32_t key_dimension)
{
	return(((uint64_t)rows * key_heads * key_dimension * 2u + (uint64_t)rows * value_heads * 2u) * 4u);
}

SparkStatus SparkOpsGatedDelta(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor mix, SparkTensor gates, SparkTensor log_scale, SparkTensor bias, SparkTensor scratch, SparkTensor output,
	const SparkDeltaParams *params)
{
	const uint64_t vectors = (uint64_t)params->rows * params->key_heads * params->key_dimension * 4u;
	const uint64_t scalars = (uint64_t)params->rows * params->value_heads * 4u;
	SparkTensor query, key, decay, beta;
	SparkDeviceBinding bindings[10];
	SparkStatus status;
	if ( params->rows == 0u || params->key_heads == 0u || params->value_heads % params->key_heads != 0u ||
		params->key_dimension != SPARK_OPS_DELTA_KEY_DIM || params->value_dimension % 4u != 0u || params->value_dimension == 0u ||
		params->sequences != sequences->sequences || params->commit != sequences->commit || params->value_heads / params->key_heads > 128u )
		return(OpsRefuse("gated_delta","128-dimensional keys, value heads in whole groups per key head and matching sequence tables are required"));
	status = OpsNeed("gated_delta","scratch",scratch,SparkOpsDeltaScratchBytes(params->rows,params->key_heads,params->value_heads,params->key_dimension));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_delta","state",state,(uint64_t)params->value_heads * params->key_dimension * params->value_dimension * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_delta","log_scale",log_scale,(uint64_t)params->value_heads * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_delta","bias",bias,(uint64_t)params->value_heads * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_delta","output",output,OpsRowsBytes(params->rows,params->output_stride,params->value_heads * params->value_dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	query = SparkTensorSlice(scratch,0u,vectors);
	key = SparkTensorSlice(scratch,vectors,vectors);
	decay = SparkTensorSlice(scratch,2u * vectors,scalars);
	beta = SparkTensorSlice(scratch,2u * vectors + scalars,scalars);
	bindings[0] = OpsBind(mix,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(gates,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(log_scale,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(bias,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(query,SPARK_DEVICE_BIND_WRITE);
	bindings[5] = OpsBind(key,SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(decay,SPARK_DEVICE_BIND_WRITE);
	bindings[7] = OpsBind(beta,SPARK_DEVICE_BIND_WRITE);
	status = OpsLaunch(ops,queue,SPARK_OPS_DELTA_PREPARE,params->key_heads,params->rows,1u,params->key_dimension,params,sizeof(*params),bindings,8u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(state,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(sequences->state_index,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(sequences->row_begin,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(key,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(decay,SPARK_DEVICE_BIND_READ);
	bindings[6] = OpsBind(beta,SPARK_DEVICE_BIND_READ);
	bindings[7] = OpsBind(mix,SPARK_DEVICE_BIND_READ);
	bindings[8] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_DELTA_RULE,params->value_dimension / 4u,params->value_heads,params->sequences,128u,params,sizeof(*params),
		bindings,9u));
}

uint64_t SparkOpsChannelDeltaScratchBytes(uint32_t rows, uint32_t heads, uint32_t key_dimension)
{
	return(((uint64_t)rows * heads * key_dimension * 3u + (uint64_t)rows * heads) * 4u);
}

SparkStatus SparkOpsChannelDelta(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor mix, SparkTensor decay_logits, SparkTensor beta_logits, SparkTensor log_scale, SparkTensor bias, SparkTensor scratch, SparkTensor output,
	const SparkDeltaParams *params)
{
	const uint64_t vectors = (uint64_t)params->rows * params->key_heads * params->key_dimension * 4u;
	SparkDeltaParams launch = *params;
	SparkTensor query, key, decay, beta;
	SparkDeviceBinding bindings[10];
	SparkStatus status;
	if ( params->rows == 0u || params->key_heads == 0u || params->value_heads != params->key_heads || params->key_dimension != SPARK_OPS_DELTA_KEY_DIM ||
		params->value_dimension % 4u != 0u || params->value_dimension == 0u || params->sequences != sequences->sequences ||
		params->commit != sequences->commit || params->gate_stride < params->key_heads * params->key_dimension || params->beta_stride < params->key_heads )
		return(OpsRefuse("channel_delta","128-dimensional keys, one value head per key head, per-channel decay rows and matching sequence tables are required"));
	launch.channel_decay = 1u;
	status = OpsNeed("channel_delta","scratch",scratch,SparkOpsChannelDeltaScratchBytes(params->rows,params->key_heads,params->key_dimension));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("channel_delta","state",state,(uint64_t)params->value_heads * params->key_dimension * params->value_dimension * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("channel_delta","decay_logits",decay_logits,OpsRowsBytes(params->rows,params->gate_stride,params->key_heads * params->key_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("channel_delta","beta_logits",beta_logits,OpsRowsBytes(params->rows,params->beta_stride,params->key_heads,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("channel_delta","log_scale",log_scale,(uint64_t)params->key_heads * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("channel_delta","bias",bias,(uint64_t)params->key_heads * params->key_dimension * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("channel_delta","output",output,OpsRowsBytes(params->rows,params->output_stride,params->value_heads * params->value_dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	query = SparkTensorSlice(scratch,0u,vectors);
	key = SparkTensorSlice(scratch,vectors,vectors);
	decay = SparkTensorSlice(scratch,2u * vectors,vectors);
	beta = SparkTensorSlice(scratch,3u * vectors,(uint64_t)params->rows * params->key_heads * 4u);
	bindings[0] = OpsBind(mix,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(decay_logits,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(beta_logits,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(log_scale,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(bias,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(query,SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(key,SPARK_DEVICE_BIND_WRITE);
	bindings[7] = OpsBind(decay,SPARK_DEVICE_BIND_WRITE);
	bindings[8] = OpsBind(beta,SPARK_DEVICE_BIND_WRITE);
	status = OpsLaunch(ops,queue,SPARK_OPS_CHANNEL_DELTA_PREPARE,params->key_heads,params->rows,1u,params->key_dimension,&launch,sizeof(launch),bindings,9u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(state,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(sequences->state_index,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(sequences->row_begin,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(key,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(decay,SPARK_DEVICE_BIND_READ);
	bindings[6] = OpsBind(beta,SPARK_DEVICE_BIND_READ);
	bindings[7] = OpsBind(mix,SPARK_DEVICE_BIND_READ);
	bindings[8] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_DELTA_RULE,params->value_dimension / 4u,params->value_heads,params->sequences,128u,&launch,sizeof(launch),
		bindings,9u));
}

SparkStatus SparkOpsKvStore(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor key, SparkTensor value,
	const SparkKvParams *params)
{
	SparkDeviceBinding bindings[7];
	SparkStatus status;
	if ( params->rows == 0u || params->kv_heads != paged->kv_heads || params->page_slots != paged->page_slots ||
		params->pages_per_sequence != paged->pages_per_sequence || params->pool_pages != paged->pool_pages || params->head_dimension == 0u )
		return(OpsRefuse("kv_store","the store geometry must match the paged pool"));
	const uint32_t values = params->value_dimension != 0u ? params->value_dimension : params->head_dimension;
	status = OpsNeed("kv_store","pool",paged->pool,(uint64_t)paged->pool_pages * paged->page_slots * paged->kv_heads * (params->head_dimension + values) * 2u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_store","key",key,OpsRowsBytes(params->rows,params->key_stride,params->kv_heads * params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_store","value",value,OpsRowsBytes(params->rows,params->value_stride,params->value_offset + params->kv_heads * values,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(key,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(value,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->sequence_of_row,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(paged->page_table,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(paged->pool,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,params->skip != 0u ? SPARK_OPS_KV_STORE_SKIP : SPARK_OPS_KV_STORE,params->rows,params->kv_heads,1u,
		OpsThreads(params->head_dimension,1024u),params,sizeof(*params),bindings,7u));
}

static uint32_t OpsAttentionWidth(const SparkAttentionParams *params)
{
	return(params->head_dimension == SPARK_OPS_ATTENTION_WIDE_HEAD_DIM ? 1u : params->head_dimension == SPARK_OPS_ATTENTION_NARROW_HEAD_DIM ? 2u :
		params->head_dimension == SPARK_OPS_ATTENTION_SPLIT_KEY_DIM ? 3u : params->head_dimension == SPARK_OPS_ATTENTION_LATENT_KEY_DIM ? 4u : 0u);
}

static uint32_t OpsAttentionValueDimension(const SparkAttentionParams *params)
{
	return(params->value_dimension != 0u ? params->value_dimension : params->head_dimension);
}

static uint32_t OpsAttentionShapeKnown(const SparkAttentionParams *params)
{
	const uint32_t key = params->head_dimension, value = OpsAttentionValueDimension(params);
	if ( key == SPARK_OPS_ATTENTION_SPLIT_KEY_DIM )
		return(value == SPARK_OPS_ATTENTION_SPLIT_VALUE_DIM ? 1u : 0u);
	if ( key == SPARK_OPS_ATTENTION_LATENT_KEY_DIM )
		return(value == SPARK_OPS_ATTENTION_LATENT_VALUE_DIM ? 1u : 0u);
	return(value == key && (key == SPARK_OPS_ATTENTION_HEAD_DIM || key == SPARK_OPS_ATTENTION_WIDE_HEAD_DIM || key == SPARK_OPS_ATTENTION_NARROW_HEAD_DIM) ? 1u : 0u);
}

static SparkStatus OpsAttentionPrefill(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkDeviceBinding *bindings,
	const SparkAttentionParams *params, const uint32_t *host_sequence_of_row)
{
	SparkAttentionParams segment = *params;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t begin, end;
	for ( begin = 0u; status == SPARK_STATUS_OK && begin < params->rows; begin = end )
	{
		for ( end = begin + 1u; end < params->rows && host_sequence_of_row[end] == host_sequence_of_row[begin]; end++ )
			continue;
		const uint32_t block_rows = SPARK_ATTENTION_BLOCK_ROWS(params->head_dimension);
		segment.row_offset = begin;
		segment.rows = end;
		status = OpsLaunch(ops,queue,(SparkOpsKernel)(SPARK_OPS_ATTENTION_PREFILL + OpsAttentionWidth(params)),params->partitions,params->heads,
			(end - begin + block_rows - 1u) / block_rows,128u,&segment,sizeof(segment),bindings,7u);
	}
	return(status);
}

uint64_t SparkOpsAttentionScratchBytes(uint32_t rows, uint32_t heads, uint32_t head_dimension, uint32_t positions)
{
	const uint64_t decode = (uint64_t)rows * ((positions + OPS_DECODE_PARTITION_KEYS - 1u) / OPS_DECODE_PARTITION_KEYS);
	const uint64_t prefill = (uint64_t)rows * ((positions + OPS_PREFILL_PARTITION_KEYS - 1u) / OPS_PREFILL_PARTITION_KEYS);
	return((decode > prefill ? decode : prefill) * heads * (head_dimension + 2u) * sizeof(float));
}

SparkStatus SparkOpsAttention(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor query, SparkTensor gate,
	const SparkTensor *sinks, SparkTensor partial, SparkTensor output, const SparkAttentionParams *params, const uint32_t *host_positions,
	const uint32_t *host_sequence_of_row)
{
	SparkAttentionParams launch = *params;
	SparkDeviceBinding bindings[7];
	SparkStatus status;
	uint32_t row, longest = 0u, run = 0u, widest = 0u, decode;
	if ( launch.group == 0u && params->kv_heads != 0u && params->heads % params->kv_heads == 0u )
		launch.group = params->heads / params->kv_heads;
	if ( params->rows == 0u || params->heads == 0u || params->kv_heads != paged->kv_heads || launch.group == 0u ||
		(uint64_t)params->first_head + params->heads > (uint64_t)launch.group * params->kv_heads || OpsAttentionShapeKnown(params) == 0u || (sinks != 0) != (params->has_sinks != 0u) ||
		params->page_slots != paged->page_slots ||
		params->pages_per_sequence != paged->pages_per_sequence || params->pool_pages != paged->pool_pages || host_positions == 0 ||
		host_sequence_of_row == 0 )
		return(OpsRefuse("attention","128-, 256- or 512-dimensional heads (or 192-wide keys with 128-wide values, or 576-wide latent keys with 512-wide values), whole query groups per key head, "
			"a sink tensor exactly when the parameters ask for sinks and the host row tables are required"));
	for ( row = 0u; row < params->rows; row++ )
	{
		run = row != 0u && host_sequence_of_row[row] == host_sequence_of_row[row - 1u] ? run + 1u : 1u;
		if ( run > widest )
			widest = run;
		if ( host_positions[row] + 1u > longest )
			longest = host_positions[row] + 1u;
	}
	if ( (uint64_t)longest > (uint64_t)params->pages_per_sequence * params->page_slots )
		return(OpsRefuse("attention","every row's position must lie inside its sequence's pages"));
	decode = widest <= OPS_DECODE_ROWS_PER_SEQUENCE || params->row_exact != 0u ? 1u : 0u;
	launch.partition_keys = decode != 0u ? OPS_DECODE_PARTITION_KEYS : OPS_PREFILL_PARTITION_KEYS;
	launch.partitions = (longest + launch.partition_keys - 1u) / launch.partition_keys;
	launch.row_offset = 0u;
	status = OpsNeed("attention","partial",partial,(uint64_t)params->rows * params->heads * launch.partitions * (OpsAttentionValueDimension(params) + 2u) * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention","query",query,OpsRowsBytes(params->rows,params->query_stride,params->heads * params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK && params->output_gate != 0u )
		status = OpsNeed("attention","gate",gate,
			OpsRowsBytes(params->rows,params->gate_stride,params->gate_offset + (params->heads - 1u) * params->gate_head_stride + params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention","output",output,OpsRowsBytes(params->rows,params->output_stride,params->heads * OpsAttentionValueDimension(params),2u));
	if ( status == SPARK_STATUS_OK && sinks != 0 )
		status = OpsNeed("attention","sinks",*sinks,(uint64_t)params->heads * sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(paged->pool,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->page_table,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(paged->sequence_of_row,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(partial,SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	if ( decode != 0u )
		status = OpsLaunch(ops,queue,(SparkOpsKernel)(SPARK_OPS_ATTENTION_DECODE + OpsAttentionWidth(params)),launch.partitions,params->heads,params->rows,256u,
			&launch,sizeof(launch),bindings,7u);
	else
		status = OpsAttentionPrefill(ops,queue,bindings,&launch,host_sequence_of_row);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(partial,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(params->output_gate != 0u ? gate : query,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	bindings[4] = OpsBind(sinks != 0 ? *sinks : partial,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,(SparkOpsKernel)(SPARK_OPS_ATTENTION_COMBINE + OpsAttentionWidth(params)),params->heads,params->rows,1u,256u,&launch,
		sizeof(launch),bindings,5u));
}

SparkStatus SparkOpsAttentionExtra(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, const SparkOpsPaged *extra, SparkTensor query,
	const SparkTensor *sinks, SparkTensor partial, SparkTensor output, const SparkAttentionParams *params, const uint32_t *host_positions,
	const uint32_t *host_extra_positions)
{
	SparkAttentionParams launch = *params;
	SparkDeviceBinding bindings[7];
	SparkStatus status;
	uint32_t row, longest = 0u, extra_longest = 0u, primary;
	if ( launch.group == 0u && params->kv_heads != 0u && params->heads % params->kv_heads == 0u )
		launch.group = params->heads / params->kv_heads;
	if ( params->rows == 0u || params->heads == 0u || params->head_dimension != SPARK_OPS_ATTENTION_WIDE_HEAD_DIM ||
		OpsAttentionValueDimension(params) != SPARK_OPS_ATTENTION_WIDE_HEAD_DIM || params->output_gate != 0u || launch.group == 0u ||
		(uint64_t)params->first_head + params->heads > (uint64_t)launch.group * params->kv_heads || (sinks != 0) != (params->has_sinks != 0u) ||
		params->kv_heads != paged->kv_heads || extra->kv_heads != paged->kv_heads || params->page_slots != paged->page_slots ||
		extra->page_slots != paged->page_slots || params->pages_per_sequence != paged->pages_per_sequence || extra->pages_per_sequence != paged->pages_per_sequence ||
		params->pool_pages != paged->pool_pages || extra->pool_pages != paged->pool_pages || host_positions == 0 || host_extra_positions == 0 )
		return(OpsRefuse("attention_extra","512-dimensional heads without an output gate, whole query groups, one page geometry for both key ranges, "
			"a sink tensor exactly when the parameters ask for sinks and the host row tables of both ranges"));
	for ( row = 0u; row < params->rows; row++ )
	{
		if ( host_positions[row] + 1u > longest )
			longest = host_positions[row] + 1u;
		if ( host_extra_positions[row] + 1u > extra_longest )
			extra_longest = host_extra_positions[row] + 1u;
	}
	if ( (uint64_t)longest > (uint64_t)params->pages_per_sequence * params->page_slots ||
		(uint64_t)extra_longest > (uint64_t)params->pages_per_sequence * params->page_slots )
		return(OpsRefuse("attention_extra","every row's keys must lie inside its sequence's pages"));
	launch.partition_keys = OPS_DECODE_PARTITION_KEYS;
	primary = (longest + launch.partition_keys - 1u) / launch.partition_keys;
	launch.extra_partitions = (extra_longest + launch.partition_keys - 1u) / launch.partition_keys;
	launch.partitions = primary + launch.extra_partitions;
	launch.partition_base = 0u;
	launch.row_offset = 0u;
	status = OpsNeed("attention_extra","partial",partial,(uint64_t)params->rows * params->heads * launch.partitions * (SPARK_OPS_ATTENTION_WIDE_HEAD_DIM + 2u) *
		sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention_extra","query",query,OpsRowsBytes(params->rows,params->query_stride,params->heads * params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention_extra","output",output,OpsRowsBytes(params->rows,params->output_stride,params->heads * params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK && sinks != 0 )
		status = OpsNeed("attention_extra","sinks",*sinks,(uint64_t)params->heads * sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(paged->pool,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->page_table,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(paged->sequence_of_row,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(partial,SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	status = OpsLaunch(ops,queue,SPARK_OPS_ATTENTION_DECODE_EXTRA,primary,params->heads,params->rows,256u,&launch,sizeof(launch),bindings,7u);
	if ( status == SPARK_STATUS_OK && launch.extra_partitions != 0u )
	{
		SparkAttentionParams second = launch;
		second.partition_base = primary;
		second.window = 0u;
		bindings[1] = OpsBind(extra->pool,SPARK_DEVICE_BIND_READ);
		bindings[4] = OpsBind(extra->positions,SPARK_DEVICE_BIND_READ);
		status = OpsLaunch(ops,queue,SPARK_OPS_ATTENTION_DECODE_EXTRA,launch.extra_partitions,params->heads,params->rows,256u,&second,sizeof(second),bindings,7u);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(partial,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	bindings[4] = OpsBind(sinks != 0 ? *sinks : partial,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(extra->positions,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_ATTENTION_COMBINE_EXTRA,params->heads,params->rows,1u,256u,&launch,sizeof(launch),bindings,6u));
}

SparkStatus SparkOpsAddRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor addend, uint32_t rows,
	uint32_t dimension, uint32_t hidden_stride, uint32_t addend_stride)
{
	SparkSiluMulParams params;
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( rows == 0u || dimension == 0u || hidden_stride < dimension || addend_stride < dimension )
		return(OpsRefuse("add_rows","rows and strides that hold the dimension are required"));
	status = OpsNeed("add_rows","hidden",hidden,OpsRowsBytes(rows,hidden_stride,dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("add_rows","addend",addend,OpsRowsBytes(rows,addend_stride,dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	params.rows = rows;
	params.dimension = dimension;
	params.input_stride = hidden_stride;
	params.output_stride = addend_stride;
	bindings[0] = OpsBind(hidden,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(addend,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_ADD_ROWS,(dimension + 255u) / 256u,rows,1u,256u,&params,sizeof(params),bindings,2u));
}

SparkStatus SparkOpsGatedMul(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor output, const SparkSiluMulParams *params,
	uint32_t activation)
{
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || params->input_stride < 2u * params->dimension ||
		(activation != SPARK_GATED_SILU && activation != SPARK_GATED_GELU_TANH && activation != SPARK_GATED_SILU_ROUNDED && activation != SPARK_GATED_SILU_CLAMP &&
		activation != SPARK_GATED_SILU_GATE_LIMIT && activation != SPARK_GATED_SITU) ||
		((activation == SPARK_GATED_SILU_CLAMP || activation == SPARK_GATED_SILU_GATE_LIMIT || activation == SPARK_GATED_SITU) && !(params->limit > 0.0f)) )
		return(OpsRefuse("gated_mul","rows, an input stride that holds the gate and up halves, a known activation and its positive limit are required"));
	status = OpsNeed("gated_mul","input",input,OpsRowsBytes(params->rows,params->input_stride,2u * params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_mul","output",output,OpsRowsBytes(params->rows,params->output_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,activation == SPARK_GATED_GELU_TANH ? SPARK_OPS_GELU_TANH_MUL : activation == SPARK_GATED_SILU_ROUNDED ? SPARK_OPS_SILU_ROUNDED_MUL :
		activation == SPARK_GATED_SILU_CLAMP ? SPARK_OPS_SILU_CLAMP_MUL : activation == SPARK_GATED_SILU_GATE_LIMIT ? SPARK_OPS_SILU_GATE_LIMIT_MUL :
		activation == SPARK_GATED_SITU ? SPARK_OPS_SITU_MUL : SPARK_OPS_SILU_MUL,(params->dimension + 255u) / 256u,
		params->rows,1u,256u,params,sizeof(*params),bindings,2u));
}

SparkStatus SparkOpsGatedMulRouted(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weights, SparkTensor output,
	const SparkSiluMulParams *params, uint32_t activation)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || params->input_stride < 2u * params->dimension ||
		(activation != SPARK_GATED_SILU && activation != SPARK_GATED_GELU_TANH && activation != SPARK_GATED_SILU_CLAMP && activation != SPARK_GATED_SILU_GATE_LIMIT) ||
		((activation == SPARK_GATED_SILU_CLAMP || activation == SPARK_GATED_SILU_GATE_LIMIT) && !(params->limit > 0.0f)) )
		return(OpsRefuse("gated_mul_routed","routes, an input stride that holds the gate and up halves and a known activation are required"));
	status = OpsNeed("gated_mul_routed","input",input,OpsRowsBytes(params->rows,params->input_stride,2u * params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_mul_routed","weights",weights,(uint64_t)params->rows * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("gated_mul_routed","output",output,OpsRowsBytes(params->rows,params->output_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(weights,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,activation == SPARK_GATED_GELU_TANH ? SPARK_OPS_GELU_TANH_MUL_ROUTED : activation == SPARK_GATED_SILU_CLAMP ? SPARK_OPS_SILU_CLAMP_MUL_ROUTED :
		activation == SPARK_GATED_SILU_GATE_LIMIT ? SPARK_OPS_SILU_GATE_LIMIT_MUL_ROUTED : SPARK_OPS_SILU_MUL_ROUTED,
		(params->dimension + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,3u));
}

SparkStatus SparkOpsRmsNormBranches(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor first, const SparkTensor *second,
	const SparkTensor *first_weight, const SparkTensor *second_weight, SparkTensor output_weight, const SparkTensor *scalar, const SparkBranchNormParams *params)
{
	const uint64_t weight_bytes = (uint64_t)params->dimension * sizeof(float);
	SparkDeviceBinding bindings[7];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || params->dimension > SPARK_BRANCH_NORM_MAX_DIMENSION ||
		(params->branches != 1u && params->branches != 2u) || (params->branches == 2u && (second == 0 || first_weight == 0 || second_weight == 0)) ||
		(params->has_scalar != 0u) != (scalar != 0) )
		return(OpsRefuse("rmsnorm_branches","one branch, or two with their weights, a dimension within the kernel and a scalar exactly when it is applied"));
	status = OpsNeed("rmsnorm_branches","hidden",hidden,OpsRowsBytes(params->rows,params->hidden_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_branches","first",first,OpsRowsBytes(params->rows,params->branch_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK && second != 0 )
		status = OpsNeed("rmsnorm_branches","second",*second,OpsRowsBytes(params->rows,params->branch_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK && first_weight != 0 )
		status = OpsNeed("rmsnorm_branches","first weight",*first_weight,weight_bytes);
	if ( status == SPARK_STATUS_OK && second_weight != 0 )
		status = OpsNeed("rmsnorm_branches","second weight",*second_weight,weight_bytes);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_branches","output weight",output_weight,weight_bytes);
	if ( status == SPARK_STATUS_OK && scalar != 0 )
		status = OpsNeed("rmsnorm_branches","scalar",*scalar,sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(hidden,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(first,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(second != 0 ? *second : first,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(first_weight != 0 ? *first_weight : output_weight,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(second_weight != 0 ? *second_weight : output_weight,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(output_weight,SPARK_DEVICE_BIND_READ);
	bindings[6] = OpsBind(scalar != 0 ? *scalar : output_weight,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_RMSNORM_BRANCHES,params->rows,1u,1u,OpsThreads(params->dimension,1024u),params,sizeof(*params),bindings,7u));
}

SparkStatus SparkOpsRmsNormScaled(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor scale, SparkTensor output,
	const SparkBranchNormParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u )
		return(OpsRefuse("rmsnorm_scaled","rows and a dimension are required"));
	status = OpsNeed("rmsnorm_scaled","input",input,OpsRowsBytes(params->rows,params->branch_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_scaled","scale",scale,(uint64_t)params->dimension * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_scaled","output",output,OpsRowsBytes(params->rows,params->hidden_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_RMSNORM_SCALED,params->rows,1u,1u,OpsThreads(params->dimension,1024u),params,sizeof(*params),bindings,3u));
}

static SparkStatus OpsScale(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkOpsKernel kernel, SparkTensor values, uint64_t count, uint32_t element,
	float scale, float multiply, const char *operation)
{
	SparkScaleParams params;
	SparkDeviceBinding bindings[1];
	SparkStatus status = count != 0u ? OpsNeed(operation,"values",values,count * element) : OpsRefuse(operation,"a count is required");
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&params,0,sizeof(params));
	params.count = count;
	params.scale = scale;
	params.multiply = multiply;
	bindings[0] = OpsBind(values,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,kernel,(uint32_t)((count + 255u) / 256u),1u,1u,256u,&params,sizeof(params),bindings,1u));
}

SparkStatus SparkOpsScaleRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, uint64_t count, float scale)
{
	return(OpsScale(ops,queue,SPARK_OPS_SCALE_ROWS,values,count,(uint32_t)sizeof(uint16_t),scale,0.0f,"scale_rows"));
}

SparkStatus SparkOpsSoftcap(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, uint64_t count, float cap, float multiply)
{
	return(OpsScale(ops,queue,SPARK_OPS_SOFTCAP_ROWS,values,count,(uint32_t)sizeof(float),cap,multiply,"softcap"));
}

SparkStatus SparkOpsRouteTopk(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor expert_scale, const SparkTensor *pinned,
	SparkTensor experts, SparkTensor weights, const SparkRouteParams *params)
{
	SparkDeviceBinding bindings[6];
	SparkStatus status;
	if ( params->rows == 0u || params->experts == 0u || params->experts > SPARK_ROUTE_MAX_EXPERTS || params->top_k == 0u ||
		params->top_k > SPARK_ROUTE_MAX_TOP_K || params->top_k > params->experts || params->logit_stride < params->experts )
		return(OpsRefuse("route_topk","rows, at most 1,024 experts, a top-k of at most 16 within them and a logit stride that holds them"));
	status = OpsNeed("route_topk","logits",logits,OpsRowsBytes(params->rows,params->logit_stride,params->experts,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_topk","expert scale",expert_scale,(uint64_t)params->experts * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_topk","experts",experts,(uint64_t)params->rows * params->top_k * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_topk","weights",weights,(uint64_t)params->rows * params->top_k * sizeof(float));
	if ( status == SPARK_STATUS_OK && pinned != 0 )
		status = OpsNeed("route_topk","pinned experts",*pinned,(uint64_t)params->rows * params->top_k * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(logits,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(expert_scale,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(experts,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(weights,SPARK_DEVICE_BIND_WRITE);
	if ( pinned != 0 )
		bindings[4] = OpsBind(*pinned,SPARK_DEVICE_BIND_READ);
	bindings[pinned != 0 ? 5u : 4u] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,pinned != 0 ? SPARK_OPS_ROUTE_PINNED : SPARK_OPS_ROUTE_TOPK,params->rows,1u,1u,OpsThreads(params->experts,256u),params,
		sizeof(*params),bindings,pinned != 0 ? 6u : 5u));
}

SparkStatus SparkOpsRouteCombineWeighted(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor weights, SparkTensor output,
	const SparkRouteParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->top_k == 0u || params->top_k > SPARK_ROUTE_MAX_TOP_K || params->width == 0u )
		return(OpsRefuse("route_combine_weighted","rows, a top-k of at most 16 and a width are required"));
	status = OpsNeed("route_combine_weighted","expert outputs",expert_out,(uint64_t)params->rows * params->top_k * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_combine_weighted","weights",weights,(uint64_t)params->rows * params->top_k * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_combine_weighted","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(expert_out,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(weights,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ROUTE_COMBINE_WEIGHTED,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,3u));
}

SparkStatus SparkOpsRouteCombineShared(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor shared, SparkTensor output,
	const SparkRouteParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->top_k == 0u || params->top_k > SPARK_ROUTE_MAX_TOP_K || params->width == 0u )
		return(OpsRefuse("route_combine_shared","rows, a top-k of at most 16 and a width are required"));
	status = OpsNeed("route_combine_shared","expert outputs",expert_out,(uint64_t)params->rows * params->top_k * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_combine_shared","shared",shared,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_combine_shared","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(expert_out,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(shared,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ROUTE_COMBINE_SHARED,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,3u));
}

SparkStatus SparkOpsSigmoidGateRows(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, SparkTensor logits, const SparkRouteParams *params)
{
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->logit_stride == 0u )
		return(OpsRefuse("sigmoid_gate_rows","rows, a width and a logit stride are required"));
	status = OpsNeed("sigmoid_gate_rows","values",values,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("sigmoid_gate_rows","logits",logits,OpsRowsBytes(params->rows,params->logit_stride,1u,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(values,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(logits,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_SIGMOID_GATE_ROWS,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,2u));
}

SparkStatus SparkOpsRouteSigmoid(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor bias, SparkTensor experts, SparkTensor weights,
	const SparkRouteParams *params)
{
	SparkRouteParams launch = *params;
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->experts == 0u || params->experts > SPARK_ROUTE_MAX_EXPERTS || params->top_k == 0u ||
		params->top_k > SPARK_ROUTE_MAX_TOP_K || params->top_k > params->experts || params->logit_stride < params->experts ||
		(params->groups > 1u && (params->groups > SPARK_ROUTE_MAX_GROUPS || params->experts % params->groups != 0u || params->experts / params->groups < 2u ||
		params->topk_groups == 0u || params->topk_groups > params->groups || params->top_k > params->topk_groups * (params->experts / params->groups))) )
		return(OpsRefuse("route_sigmoid","rows, at most 1,024 experts, a top-k of at most 16 within them, a logit stride that holds them, and expert groups "
			"of at least two that tile the experts and leave the open groups enough candidates"));
	if ( launch.scale == 0.0f )
		launch.scale = 1.0f;
	status = OpsNeed("route_sigmoid","logits",logits,OpsRowsBytes(params->rows,params->logit_stride,params->experts,4u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_sigmoid","bias",bias,(uint64_t)params->experts * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_sigmoid","experts",experts,(uint64_t)params->rows * params->top_k * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_sigmoid","weights",weights,(uint64_t)params->rows * params->top_k * sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(logits,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(bias,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(experts,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(weights,SPARK_DEVICE_BIND_WRITE);
	if ( params->score != SPARK_ROUTE_SCORE_SIGMOID && params->score != SPARK_ROUTE_SCORE_SQRT_SOFTPLUS )
		return(OpsRefuse("route_sigmoid","the expert score is a sigmoid or the square root of a softplus"));
	return(OpsLaunch(ops,queue,params->score == SPARK_ROUTE_SCORE_SQRT_SOFTPLUS ? SPARK_OPS_ROUTE_SQRT_SOFTPLUS : SPARK_OPS_ROUTE_SIGMOID,params->rows,1u,1u,256u,
		&launch,sizeof(launch),bindings,4u));
}

SparkStatus SparkOpsRmsNormBf16(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor output,
	const SparkNormParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || params->has_addend != 0u )
		return(OpsRefuse("rmsnorm_bf16","rows and a dimension, and no residual, are required"));
	status = OpsNeed("rmsnorm_bf16","input",input,OpsRowsBytes(params->rows,params->input_stride,params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_bf16","weight",weight,(uint64_t)params->dimension * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_bf16","output",output,OpsRowsBytes(params->rows,params->output_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_RMSNORM_BF16,params->rows,1u,1u,OpsThreads(params->dimension,1024u),params,sizeof(*params),bindings,3u));
}

static SparkStatus OpsAttnResShape(const char *operation, const SparkAttnResParams *params, SparkTensor bank, SparkTensor partial, SparkTensor scores)
{
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || params->sources == 0u || params->sources > SPARK_ATTNRES_MAX_SOURCES ||
		params->score_stride < params->sources )
		return(OpsRefuse(operation,"rows, a width and one to sixteen sources with a score row that holds them are required"));
	status = params->sources > 1u ? OpsNeed(operation,"bank",bank,(uint64_t)(params->sources - 1u) * params->rows * params->dimension * sizeof(uint16_t)) :
		SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed(operation,"partial",partial,(uint64_t)params->rows * params->dimension * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed(operation,"scores",scores,(uint64_t)params->rows * params->score_stride * sizeof(float));
	return(status);
}

SparkStatus SparkOpsAttnResScore(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor bank, SparkTensor partial, SparkTensor weight, SparkTensor scores,
	const SparkAttnResParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status = OpsAttnResShape("attnres_score",params,bank,partial,scores);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attnres_score","weight",weight,(uint64_t)params->dimension * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(params->sources > 1u ? bank : partial,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(partial,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(scores,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ATTNRES_SCORE,params->rows,params->sources,1u,OpsThreads(params->dimension,1024u),params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsAttnResMix(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor bank, SparkTensor partial, SparkTensor scores, SparkTensor output,
	const SparkAttnResParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status = OpsAttnResShape("attnres_mix",params,bank,partial,scores);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attnres_mix","output",output,(uint64_t)params->rows * params->dimension * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(params->sources > 1u ? bank : partial,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(partial,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(scores,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ATTNRES_MIX,(params->dimension + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,4u));
}

static SparkStatus OpsHyperShape(const char *operation, const SparkHyperParams *params)
{
	if ( params->rows == 0u || params->streams == 0u || params->streams > SPARK_HYPER_MAX_STREAMS || params->width == 0u )
		return(OpsRefuse(operation,"rows, one to eight streams and a width are required"));
	return(SPARK_STATUS_OK);
}

static uint64_t OpsHyperBytes(const SparkHyperParams *params, uint32_t streams, uint32_t element)
{
	return((uint64_t)params->rows * streams * params->width * element);
}

SparkStatus SparkOpsHyperExpand(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor streams, const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[2];
	SparkStatus status = OpsHyperShape("hyper_expand",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_expand","hidden",hidden,OpsHyperBytes(params,1u,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_expand","streams",streams,OpsHyperBytes(params,params->streams,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(hidden,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(streams,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_EXPAND,(params->streams * params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),
		bindings,2u));
}

SparkStatus SparkOpsGroupRmsNorm(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor weight, SparkTensor output,
	const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status = OpsHyperShape("group_rmsnorm",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("group_rmsnorm","input",input,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("group_rmsnorm","weight",weight,(uint64_t)params->streams * params->width * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("group_rmsnorm","output",output,OpsHyperBytes(params,params->streams,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_GROUP_RMSNORM,params->streams,params->rows,1u,OpsThreads(params->width,1024u),params,sizeof(*params),
		bindings,3u));
}

SparkStatus SparkOpsHyperGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor values, const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[1];
	SparkStatus status = OpsHyperShape("hyper_gate",params);
	if ( status == SPARK_STATUS_OK && params->divisor == 0.0f )
		status = OpsRefuse("hyper_gate","a nonzero divisor is required");
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_gate","values",values,OpsHyperBytes(params,1u,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(values,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_GATE,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,1u));
}

SparkStatus SparkOpsHyperMix(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor up, SparkTensor normed, SparkTensor mixed,
	const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status = OpsHyperShape("hyper_mix",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_mix","up",up,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_mix","normed",normed,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_mix","mixed",mixed,OpsHyperBytes(params,1u,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(up,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(normed,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(mixed,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_MIX,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,3u));
}

SparkStatus SparkOpsHcMix(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor post, SparkTensor mixed, const SparkHcParams *params)
{
	const uint64_t flat = (uint64_t)params->streams * params->width;
	SparkDeviceBinding bindings[6];
	SparkStatus status = SPARK_STATUS_OK;
	if ( params->rows == 0u || params->width == 0u || params->streams == 0u || params->streams > SPARK_HC_MAX_STREAMS ||
		(params->mixes != 2u * params->streams && params->mixes != params->streams) || (params->mixes != 8u && params->mixes != 4u) )
		return(OpsRefuse("hc_mix","rows, a width and up to four streams with one mix per stream, or two (pre and post), eight or four in all"));
	status = OpsNeed("hc_mix","streams",streams,(uint64_t)params->rows * flat * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_mix","fn",fn,(uint64_t)params->mixes * flat * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_mix","base",base,(uint64_t)params->mixes * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_mix","scale",scale,(params->mixes > params->streams ? 2u : 1u) * sizeof(float));
	if ( status == SPARK_STATUS_OK && params->mixes > params->streams )
		status = OpsNeed("hc_mix","post",post,(uint64_t)params->rows * params->streams * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_mix","mixed",mixed,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(fn,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(base,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(params->mixes > params->streams ? post : base,params->mixes > params->streams ? SPARK_DEVICE_BIND_WRITE : SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(mixed,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,params->mixes == 8u ? SPARK_OPS_HC_MIX8 : SPARK_OPS_HC_MIX4,params->rows,1u,1u,256u,params,sizeof(*params),bindings,6u));
}

SparkStatus SparkOpsHcPost(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor post, SparkTensor output, const SparkHcParams *params)
{
	const uint64_t flat = (uint64_t)params->streams * params->width;
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->streams == 0u || params->streams > SPARK_HC_MAX_STREAMS )
		return(OpsRefuse("hc_post","rows, a width and up to four streams"));
	status = OpsNeed("hc_post","streams",streams,(uint64_t)params->rows * flat * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_post","post",post,(uint64_t)params->rows * params->streams * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_post","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(post,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_HC_POST,(uint32_t)((flat + 255u) / 256u),params->rows,1u,256u,params,sizeof(*params),bindings,3u));
}

static SparkStatus OpsHcSite(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkOpsKernel kernel, SparkTensor streams, SparkTensor fn, SparkTensor base,
	SparkTensor scale, const SparkTensor *pre, SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params)
{
	const uint64_t flat = (uint64_t)SPARK_HC_MAX_STREAMS * params->width;
	SparkDeviceBinding bindings[8];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->streams != SPARK_HC_MAX_STREAMS || params->mixes != SPARK_HC_COMB_MIXES )
		return(OpsRefuse("hc_site","rows, a width and four streams with 24 mixes (pre, post and the 4 x 4 comb)"));
	status = OpsNeed("hc_site","streams",streams,(uint64_t)params->rows * flat * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_site","fn",fn,(uint64_t)SPARK_HC_COMB_MIXES * flat * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_site","base",base,SPARK_HC_COMB_MIXES * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_site","scale",scale,3u * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_site","post",post,(uint64_t)params->rows * SPARK_HC_MAX_STREAMS * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_site","comb",comb,(uint64_t)params->rows * SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_site","mixed",mixed,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && pre != 0 )
		status = OpsNeed("hc_site","pre",*pre,(uint64_t)params->rows * SPARK_HC_MAX_STREAMS * sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(fn,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(base,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(post,SPARK_DEVICE_BIND_WRITE);
	bindings[5] = OpsBind(comb,SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(mixed,SPARK_DEVICE_BIND_WRITE);
	if ( pre != 0 )
		bindings[7] = OpsBind(*pre,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,kernel,params->rows,1u,1u,256u,params,sizeof(*params),bindings,pre != 0 ? 8u : 7u));
}

SparkStatus SparkOpsHcSite(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params)
{
	return(OpsHcSite(ops,queue,SPARK_OPS_HC_SITE,streams,fn,base,scale,0,post,comb,mixed,params));
}

SparkStatus SparkOpsHcSiteLagged(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor pre, SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params)
{
	return(OpsHcSite(ops,queue,SPARK_OPS_HC_SITE_LAGGED,streams,fn,base,scale,&pre,post,comb,mixed,params));
}

SparkStatus SparkOpsHcSiteClamped(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor fn, SparkTensor base, SparkTensor scale,
	SparkTensor post, SparkTensor comb, SparkTensor mixed, const SparkHcParams *params)
{
	if ( params->iterations == 0u || !(params->gate_clamp > 0.0f) || !(params->residual_clamp > 0.0f) || !(params->sum_floor > 0.0f) )
		return(OpsRefuse("hc_site_clamped","Sinkhorn passes, positive gate and residual clamps and a positive sum floor"));
	return(OpsHcSite(ops,queue,SPARK_OPS_HC_SITE_CLAMPED,streams,fn,base,scale,0,post,comb,mixed,params));
}

static SparkStatus OpsHcPostComb(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkOpsKernel kernel, SparkTensor streams, SparkTensor post, SparkTensor comb,
	SparkTensor output, const SparkHcParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->streams != SPARK_HC_MAX_STREAMS )
		return(OpsRefuse("hc_post_comb","rows, a width and four streams"));
	status = OpsNeed("hc_post_comb","streams",streams,(uint64_t)params->rows * SPARK_HC_MAX_STREAMS * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_post_comb","post",post,(uint64_t)params->rows * SPARK_HC_MAX_STREAMS * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_post_comb","comb",comb,(uint64_t)params->rows * SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_post_comb","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(post,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(comb,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,kernel,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsHcPostComb(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor post, SparkTensor comb, SparkTensor output,
	const SparkHcParams *params)
{
	return(OpsHcPostComb(ops,queue,SPARK_OPS_HC_POST_COMB,streams,post,comb,output,params));
}

SparkStatus SparkOpsHcPostCombF32(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor post, SparkTensor comb, SparkTensor output,
	const SparkHcParams *params)
{
	return(OpsHcPostComb(ops,queue,SPARK_OPS_HC_POST_COMB_F32,streams,post,comb,output,params));
}

SparkStatus SparkOpsHcCollapse(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor pre, SparkTensor output, const SparkHcParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->streams == 0u || params->streams > SPARK_HC_MAX_STREAMS )
		return(OpsRefuse("hc_collapse","rows, a width and up to four streams"));
	status = OpsNeed("hc_collapse","streams",streams,(uint64_t)params->rows * params->streams * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_collapse","pre",pre,(uint64_t)params->rows * params->streams * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hc_collapse","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(pre,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HC_COLLAPSE,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,3u));
}

SparkStatus SparkOpsStreamKeyGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor kv, SparkTensor query_weight,
	SparkTensor key_weight, const SparkGateParams *params)
{
	const uint64_t width = params->width;
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->streams == 0u || params->streams > SPARK_HYPER_MAX_STREAMS )
		return(OpsRefuse("stream_key_gate","rows, a width and up to eight streams"));
	status = OpsNeed("stream_key_gate","streams",streams,(uint64_t)params->rows * params->streams * width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("stream_key_gate","kv",kv,(uint64_t)params->rows * (params->streams + 1u) * width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("stream_key_gate","query weight",query_weight,(uint64_t)params->streams * width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("stream_key_gate","key weight",key_weight,(uint64_t)params->streams * width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(kv,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(query_weight,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(key_weight,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_STREAM_KEY_GATE,params->streams,params->rows,1u,OpsThreads(params->width,1024u),params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsBlockQdq(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor output, const SparkQdqParams *params)
{
	const uint32_t block = SPARK_QDQ_BLOCK(params->format), blocks = params->width / block;
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->width % block != 0u || params->input_stride < params->width ||
		params->output_stride < params->width || params->format > SPARK_QDQ_FP4_POW2 )
		return(OpsRefuse("block_qdq","rows of whole blocks (32 values for FP8 and power-of-two FP4, 16 for FP4 with an E4M3 scale) inside their strides"));
	status = OpsNeed("block_qdq","input",input,OpsRowsBytes(params->rows,params->input_stride,params->width,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("block_qdq","output",output,OpsRowsBytes(params->rows,params->output_stride,params->width,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_BLOCK_QDQ,(blocks + 63u) / 64u,params->rows,1u,64u,params,sizeof(*params),bindings,2u));
}

SparkStatus SparkOpsKvCompress(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor kv, SparkTensor score, SparkTensor state, const SparkOpsSequences *sequences,
	SparkTensor positions, SparkTensor pooled, const SparkCompressParams *params)
{
	SparkDeviceBinding bindings[7];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->sequences == 0u || params->sequences != sequences->sequences ||
		(params->ratio != 1u && params->ratio != 2u) || (params->ratio == 2u && params->state_stride < 2u * params->width) )
		return(OpsRefuse("kv_compress","rows, a width, the step's sequences and a ratio of one or two, a pair's carried state holding two rows"));
	status = OpsNeed("kv_compress","kv",kv,(uint64_t)params->rows * params->width * sizeof(float));
	if ( status == SPARK_STATUS_OK && params->ratio == 2u )
		status = OpsNeed("kv_compress","score",score,(uint64_t)params->rows * params->width * sizeof(float));
	if ( status == SPARK_STATUS_OK && params->ratio == 2u )
		status = OpsNeed("kv_compress","state",state,(uint64_t)params->state_stride * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_compress","positions",positions,(uint64_t)params->rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_compress","row_begin",sequences->row_begin,((uint64_t)params->sequences + 1u) * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_compress","state_index",sequences->state_index,(uint64_t)params->sequences * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_compress","pooled",pooled,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(kv,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(params->ratio == 2u ? score : kv,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(params->ratio == 2u ? state : kv,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(positions,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(sequences->row_begin,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(sequences->state_index,SPARK_DEVICE_BIND_READ);
	bindings[6] = OpsBind(pooled,SPARK_DEVICE_BIND_WRITE);
	status = OpsLaunch(ops,queue,SPARK_OPS_KV_COMPRESS,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,7u);
	if ( status != SPARK_STATUS_OK || params->ratio == 1u || params->commit == 0u )
		return(status);
	bindings[2] = OpsBind(state,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_KV_COMPRESS_SAVE,(params->width + 255u) / 256u,params->sequences,1u,256u,params,sizeof(*params),bindings,6u));
}

SparkStatus SparkOpsCompressedPositions(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor positions, SparkTensor store, SparkTensor rope, SparkTensor last,
	const SparkCompressParams *params)
{
	const uint64_t bytes = (uint64_t)params->rows * sizeof(uint32_t);
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->ratio == 0u )
		return(OpsRefuse("compressed_positions","rows and a compression ratio"));
	status = OpsNeed("compressed_positions","positions",positions,bytes);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("compressed_positions","store",store,bytes);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("compressed_positions","rope",rope,bytes);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("compressed_positions","last",last,bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(positions,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(store,SPARK_DEVICE_BIND_WRITE);
	bindings[2] = OpsBind(rope,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(last,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_COMPRESSED_POSITIONS,(params->rows + 63u) / 64u,1u,1u,64u,params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsStreamMean(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor output, const SparkHcParams *params)
{
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->streams == 0u || params->streams > SPARK_HC_MAX_STREAMS )
		return(OpsRefuse("stream_mean","rows, a width and up to four streams"));
	status = OpsNeed("stream_mean","streams",streams,(uint64_t)params->rows * params->streams * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("stream_mean","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_STREAM_MEAN,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,2u));
}

SparkStatus SparkOpsHyperInjectWeights(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor normed, SparkTensor weight, SparkTensor inject,
	const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status = OpsHyperShape("hyper_inject_weights",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_inject_weights","normed",normed,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_inject_weights","weight",weight,(uint64_t)params->streams * params->streams * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_inject_weights","inject",inject,(uint64_t)params->rows * params->streams * sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(normed,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(inject,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_INJECT_WEIGHTS,params->rows,1u,1u,256u,params,sizeof(*params),bindings,3u));
}

SparkStatus SparkOpsHyperInject(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor output, SparkTensor inject,
	const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status = OpsHyperShape("hyper_inject",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_inject","streams",streams,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_inject","output",output,OpsHyperBytes(params,1u,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_inject","inject",inject,(uint64_t)params->rows * params->streams * sizeof(float));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(output,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(inject,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_INJECT,(params->streams * params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),
		bindings,3u));
}

SparkStatus SparkOpsHyperKeyGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor key, SparkTensor query, SparkTensor value, SparkTensor gated,
	const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status = OpsHyperShape("hyper_key_gate",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_key_gate","key",key,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_key_gate","query",query,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_key_gate","value",value,OpsHyperBytes(params,1u,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_key_gate","gated",gated,OpsHyperBytes(params,params->streams,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(key,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(value,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(gated,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_KEY_GATE,params->streams,params->rows,1u,OpsThreads(params->width,1024u),params,sizeof(*params),
		bindings,4u));
}

SparkStatus SparkOpsHyperAdd(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor streams, SparkTensor gated, SparkTensor conv,
	const SparkHyperParams *params)
{
	SparkDeviceBinding bindings[3];
	SparkStatus status = OpsHyperShape("hyper_add",params);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_add","streams",streams,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_add","gated",gated,OpsHyperBytes(params,params->streams,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("hyper_add","conv",conv,OpsHyperBytes(params,params->streams,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(streams,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[1] = OpsBind(gated,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(conv,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_HYPER_ADD,(params->streams * params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),
		bindings,3u));
}

SparkStatus SparkOpsRouteCombine(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor expert_out, SparkTensor output,
	const SparkRouteParams *params)
{
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->top_k == 0u || params->top_k > SPARK_ROUTE_MAX_TOP_K || params->width == 0u )
		return(OpsRefuse("route_combine","rows, a top-k of at most 16 and a width are required"));
	status = OpsNeed("route_combine","expert outputs",expert_out,(uint64_t)params->rows * params->top_k * params->width * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("route_combine","output",output,(uint64_t)params->rows * params->width * sizeof(uint16_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(expert_out,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ROUTE_COMBINE,(params->width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,2u));
}

SparkStatus SparkOpsLinearRouted(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y, SparkTensor experts,
	uint32_t expert_count, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, sources = (params->rows + params->route_divisor - 1u) /
		(params->route_divisor != 0u ? params->route_divisor : 1u);
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || K == 0u || N == 0u || K % 8u != 0u || params->input_stride % 8u != 0u || params->input_stride < K ||
		params->route_divisor == 0u || params->output_f32 != 0u || expert_count == 0u )
		return(OpsRefuse("linear_routed","routes, an input width that is a multiple of 8, a source row per route divisor and BF16 output are required"));
	status = OpsNeed("linear_routed","input",x,OpsRowsBytes(sources,params->input_stride,K,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed","weight",w,(uint64_t)expert_count * N * K * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed","output",y,OpsRowsBytes(params->rows,params->output_stride,N,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed","experts",experts,(uint64_t)params->rows * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(experts,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_GEMV_ROUTED,(N + SPARK_GEMV_OUTPUTS - 1u) / SPARK_GEMV_OUTPUTS,params->rows,1u,128u,params,sizeof(*params),
		bindings,4u));
}

SparkStatus SparkOpsLinearRoutedFp8(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor y,
	SparkTensor experts, uint32_t expert_count, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, sources = (params->rows + params->route_divisor - 1u) /
		(params->route_divisor != 0u ? params->route_divisor : 1u), mx = params->scale_group == SPARK_MXFP8_GROUP ? 1u : 0u;
	const uint32_t group = mx != 0u ? SPARK_MXFP8_GROUP : SPARK_FP8_SCALE_K;
	SparkDeviceBinding bindings[5];
	SparkGemvParams launch;
	SparkStatus status;
	if ( params->rows == 0u || K == 0u || N == 0u || K % group != 0u || params->input_stride % 8u != 0u || params->input_stride < K ||
		params->route_divisor == 0u || params->output_f32 != 0u || expert_count == 0u || (params->scale_group != 0u && mx == 0u) )
		return(OpsRefuse("linear_routed_fp8","routes, an input width of whole scale groups (32 for MXFP8, 128 for FP8), a source row per route divisor and BF16 output are required"));
	status = OpsNeed("linear_routed_fp8","input",x,OpsRowsBytes(sources,params->input_stride,K,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_fp8","weight",w,(uint64_t)expert_count * N * K);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_fp8","scale",scale,(uint64_t)expert_count * N * (K / group) * (mx != 0u ? 1u : sizeof(float)));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_fp8","output",y,OpsRowsBytes(params->rows,params->output_stride,N,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_fp8","experts",experts,(uint64_t)params->rows * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	launch = *params;
	launch.scale_stride = K / group;
	bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
	bindings[4] = OpsBind(experts,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,mx != 0u ? SPARK_OPS_GEMV_MXFP8_ROUTED : SPARK_OPS_GEMV_FP8_ROUTED,(N + SPARK_GEMV_OUTPUTS - 1u) / SPARK_GEMV_OUTPUTS,
		params->rows,1u,128u,&launch,sizeof(launch),bindings,5u));
}

SparkStatus SparkOpsLinearRoutedMxfp4(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor y,
	SparkTensor experts, uint32_t expert_count, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, sources = (params->rows + params->route_divisor - 1u) /
		(params->route_divisor != 0u ? params->route_divisor : 1u);
	SparkDeviceBinding bindings[5];
	SparkStatus status;
	if ( params->rows == 0u || K == 0u || N == 0u || K % SPARK_MXFP4_GROUP != 0u || params->input_stride % 8u != 0u || params->input_stride < K ||
		params->route_divisor == 0u || params->output_f32 != 0u || expert_count == 0u )
		return(OpsRefuse("linear_routed_mxfp4","routes, an input width that is a multiple of 32, a source row per route divisor and BF16 output are required"));
	status = OpsNeed("linear_routed_mxfp4","input",x,OpsRowsBytes(sources,params->input_stride,K,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_mxfp4","weight",w,(uint64_t)expert_count * N * (K / 2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_mxfp4","scale",scale,(uint64_t)expert_count * N * (K / SPARK_MXFP4_GROUP));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_mxfp4","output",y,OpsRowsBytes(params->rows,params->output_stride,N,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_mxfp4","experts",experts,(uint64_t)params->rows * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
	bindings[4] = OpsBind(experts,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_GEMV_MXFP4_ROUTED,(N + SPARK_GEMV_OUTPUTS - 1u) / SPARK_GEMV_OUTPUTS,params->rows,1u,128u,params,sizeof(*params),
		bindings,5u));
}

static SparkStatus OpsNgramShape(const char *operation, const SparkNgramParams *params, const SparkOpsSequences *sequences)
{
	if ( params->sequences == 0u || params->orders == 0u || params->orders > SPARK_NGRAM_MAX_ORDERS || params->heads == 0u ||
		params->history_stride < params->orders || (sequences != 0 && (params->sequences != sequences->sequences || params->commit != sequences->commit)) )
		return(OpsRefuse(operation,"sequences, one to three orders, heads, a history that holds them and matching sequence tables are required"));
	return(SPARK_STATUS_OK);
}

static SparkStatus OpsNgramHash(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, const SparkTensor *token_map, SparkTensor positions,
	const SparkOpsSequences *sequences, SparkTensor history, SparkTensor multipliers, SparkTensor primes, SparkTensor offsets, SparkTensor ids,
	const SparkNgramParams *params)
{
	const uint64_t columns = (uint64_t)params->orders * params->heads;
	SparkDeviceBinding bindings[10];
	SparkStatus status = OpsNgramShape("ngram_hash",params,sequences);
	if ( status == SPARK_STATUS_OK && params->rows == 0u )
		status = OpsRefuse("ngram_hash","rows are required");
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","token ids",token_ids,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","positions",positions,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","row_begin",sequences->row_begin,((uint64_t)params->sequences + 1u) * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","state_index",sequences->state_index,(uint64_t)params->sequences * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","multipliers",multipliers,((uint64_t)params->orders + 1u) * 8u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","primes",primes,columns * 8u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","offsets",offsets,columns * 8u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_hash","ids",ids,(uint64_t)params->rows * columns * 8u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(token_ids,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(positions,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(sequences->row_begin,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(sequences->state_index,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(history,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(multipliers,SPARK_DEVICE_BIND_READ);
	bindings[6] = OpsBind(primes,SPARK_DEVICE_BIND_READ);
	bindings[7] = OpsBind(offsets,SPARK_DEVICE_BIND_READ);
	bindings[8] = OpsBind(ids,SPARK_DEVICE_BIND_WRITE);
	if ( token_map != 0 )
		bindings[9] = OpsBind(*token_map,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,token_map != 0 ? SPARK_OPS_NGRAM_HASH_MAPPED : SPARK_OPS_NGRAM_HASH,(params->rows + 63u) / 64u,1u,1u,64u,params,sizeof(*params),
		bindings,token_map != 0 ? 10u : 9u));
}

SparkStatus SparkOpsNgramHash(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, SparkTensor positions, const SparkOpsSequences *sequences,
	SparkTensor history, SparkTensor multipliers, SparkTensor primes, SparkTensor offsets, SparkTensor ids, const SparkNgramParams *params)
{
	return(OpsNgramHash(ops,queue,token_ids,0,positions,sequences,history,multipliers,primes,offsets,ids,params));
}

SparkStatus SparkOpsNgramHashMapped(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, SparkTensor token_map, SparkTensor positions,
	const SparkOpsSequences *sequences, SparkTensor history, SparkTensor multipliers, SparkTensor primes, SparkTensor offsets, SparkTensor ids,
	const SparkNgramParams *params)
{
	return(OpsNgramHash(ops,queue,token_ids,&token_map,positions,sequences,history,multipliers,primes,offsets,ids,params));
}

static SparkStatus OpsNgramGather(SparkDeviceOps *ops, SparkDeviceQueue *queue, uint32_t block_scaled, SparkTensor ids, SparkTensor payload, SparkTensor scale,
	SparkTensor embed, const SparkNgramParams *params)
{
	const uint64_t columns = (uint64_t)params->orders * params->heads;
	SparkDeviceBinding bindings[4];
	SparkStatus status = OpsNgramShape("ngram_gather_fp8",params,0);
	if ( status == SPARK_STATUS_OK && (params->rows == 0u || params->head_dimension == 0u || params->table_rows == 0u) )
		status = OpsRefuse("ngram_gather_fp8","rows, a head dimension and table rows are required");
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_gather_fp8","ids",ids,(uint64_t)params->rows * columns * 8u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_gather_fp8","payload",payload,params->table_rows * params->head_dimension);
	if ( status == SPARK_STATUS_OK && block_scaled != 0u && params->head_dimension % SPARK_MXFP8_GROUP != 0u )
		status = OpsRefuse("ngram_gather_fp8","block scales cover whole 32-wide groups of a head");
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_gather_fp8","scale",scale,block_scaled != 0u ? params->table_rows * (params->head_dimension / SPARK_MXFP8_GROUP) : 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_gather_fp8","embed",embed,(uint64_t)params->rows * columns * params->head_dimension * 2u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(ids,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(payload,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(embed,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,block_scaled != 0u ? SPARK_OPS_NGRAM_GATHER_E8M0 : SPARK_OPS_NGRAM_GATHER_FP8,params->rows,(uint32_t)columns,1u,
		OpsThreads(params->head_dimension,256u),params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsNgramGatherFp8(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor ids, SparkTensor payload, SparkTensor scale, SparkTensor embed,
	const SparkNgramParams *params)
{
	return(OpsNgramGather(ops,queue,0u,ids,payload,scale,embed,params));
}

SparkStatus SparkOpsNgramGatherE8m0(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor ids, SparkTensor payload, SparkTensor scale, SparkTensor embed,
	const SparkNgramParams *params)
{
	return(OpsNgramGather(ops,queue,1u,ids,payload,scale,embed,params));
}

static SparkStatus OpsNgramHistory(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, const SparkTensor *row_map, const SparkTensor *token_map,
	const SparkOpsSequences *sequences, SparkTensor history, const SparkNgramParams *params)
{
	SparkNgramParams mapped = *params;
	SparkDeviceBinding bindings[6];
	SparkStatus status = OpsNgramShape("ngram_history",params,sequences);
	if ( status == SPARK_STATUS_OK && params->rows == 0u )
		status = OpsRefuse("ngram_history","rows are required");
	if ( status == SPARK_STATUS_OK && row_map == 0 )
		status = OpsNeed("ngram_history","token ids",token_ids,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK && row_map != 0 )
		status = OpsNeed("ngram_history","row map",*row_map,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_history","row_begin",sequences->row_begin,((uint64_t)params->sequences + 1u) * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("ngram_history","state_index",sequences->state_index,(uint64_t)params->sequences * 4u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(token_ids,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(sequences->row_begin,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(sequences->state_index,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(history,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[4] = OpsBind(row_map != 0 ? *row_map : token_ids,SPARK_DEVICE_BIND_READ);
	mapped.mapped = row_map != 0 ? 1u : 0u;
	if ( token_map != 0 )
		bindings[5] = OpsBind(*token_map,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,token_map != 0 ? SPARK_OPS_NGRAM_HISTORY_MAPPED : SPARK_OPS_NGRAM_HISTORY,(params->sequences + 63u) / 64u,1u,1u,64u,&mapped,
		sizeof(mapped),bindings,token_map != 0 ? 6u : 5u));
}

SparkStatus SparkOpsNgramHistory(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, const SparkTensor *row_map, const SparkOpsSequences *sequences,
	SparkTensor history, const SparkNgramParams *params)
{
	return(OpsNgramHistory(ops,queue,token_ids,row_map,0,sequences,history,params));
}

SparkStatus SparkOpsNgramHistoryMapped(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor token_ids, SparkTensor token_map, const SparkOpsSequences *sequences,
	SparkTensor history, const SparkNgramParams *params)
{
	return(OpsNgramHistory(ops,queue,token_ids,0,&token_map,sequences,history,params));
}

SparkStatus SparkOpsLinearRoutedNvfp4(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor scale, SparkTensor global_scale,
	SparkTensor y, SparkTensor experts, uint32_t expert_count, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, sources = (params->rows + params->route_divisor - 1u) /
		(params->route_divisor != 0u ? params->route_divisor : 1u);
	const uint32_t neurons = 4u * (32u / SPARK_NVFP4_LANES(K / SPARK_NVFP4_CHUNK));
	SparkDeviceBinding bindings[6];
	SparkStatus status;
	if ( params->rows == 0u || K == 0u || N == 0u || K % SPARK_NVFP4_CHUNK != 0u || params->input_stride % 8u != 0u || params->input_stride < K ||
		params->route_divisor == 0u || params->output_f32 != 0u || expert_count == 0u )
		return(OpsRefuse("linear_routed_nvfp4","routes, an input width that is a multiple of 32, a source row per route divisor and BF16 output are required"));
	status = OpsNeed("linear_routed_nvfp4","input",x,OpsRowsBytes(sources,params->input_stride,K,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_nvfp4","weight",w,(uint64_t)expert_count * N * (K / 2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_nvfp4","scale",scale,(uint64_t)expert_count * N * (K / SPARK_NVFP4_GROUP));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_nvfp4","global scale",global_scale,(uint64_t)expert_count * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_nvfp4","output",y,OpsRowsBytes(params->rows,params->output_stride,N,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear_routed_nvfp4","experts",experts,(uint64_t)params->rows * sizeof(uint32_t));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(scale,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(global_scale,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
	bindings[5] = OpsBind(experts,SPARK_DEVICE_BIND_READ);
	return(OpsLaunch(ops,queue,SPARK_OPS_GEMV_NVFP4_ROUTED,(N + neurons - 1u) / neurons,params->rows,1u,128u,params,sizeof(*params),bindings,6u));
}

SparkStatus SparkOpsLogprob(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor logits, SparkTensor targets, SparkTensor result,
	SparkTensor argmax, const SparkLogprobParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->vocabulary == 0u || params->logit_stride < params->vocabulary )
		return(OpsRefuse("logprob","rows and a logit stride that holds the vocabulary are required"));
	status = OpsNeed("logprob","logits",logits,OpsRowsBytes(params->rows,params->logit_stride,params->vocabulary,4u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("logprob","targets",targets,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("logprob","result",result,(uint64_t)params->rows * 16u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("logprob","argmax",argmax,(uint64_t)params->rows * 4u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(logits,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(targets,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(result,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(argmax,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_LOGPROB,params->rows,1u,1u,1024u,params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsStreamRead(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor words, SparkTensor sink, uint32_t groups)
{
	SparkStreamParams params;
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( groups == 0u || words.bytes < 16u )
		return(OpsRefuse("stream_read","groups and at least one 16-byte word are required"));
	status = OpsNeed("stream_read","sink",sink,(uint64_t)groups * 4u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&params,0,sizeof(params));
	params.words = words.bytes / 16u;
	params.groups = groups;
	bindings[0] = OpsBind(words,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(sink,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_STREAM_READ,groups,1u,1u,1024u,&params,sizeof(params),bindings,2u));
}

SparkStatus SparkOpsPolyNormMul(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor gate_up, SparkTensor params_f32, const SparkTensor *param_of_row,
	SparkTensor output, const SparkPolyNormParams *params)
{
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->width == 0u || params->input_stride < 2u * params->width || params->output_stride < params->width ||
		params->param_count == 0u || (params->indexed != 0u) != (param_of_row != 0) || !(params->clamp > 0.0f) || !(params->epsilon > 0.0f) ||
		(params->routed != 0u && !(params->bias_clamp > 0.0f)) )
		return(OpsRefuse("poly_norm_mul","rows, a width, strides that hold the gate and up halves, parameter sets, a row index exactly when indexed, "
			"positive clamps and a positive epsilon"));
	status = OpsNeed("poly_norm_mul","gate_up",gate_up,OpsRowsBytes(params->rows,params->input_stride,2u * params->width,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("poly_norm_mul","params",params_f32,(uint64_t)params->param_count * 4u * sizeof(float));
	if ( status == SPARK_STATUS_OK && param_of_row != 0 )
		status = OpsNeed("poly_norm_mul","param_of_row",*param_of_row,(uint64_t)params->rows * sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("poly_norm_mul","output",output,OpsRowsBytes(params->rows,params->output_stride,params->width,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(gate_up,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(params_f32,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(param_of_row != 0 ? *param_of_row : params_f32,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_POLY_NORM_MUL,params->rows,1u,1u,256u,params,sizeof(*params),bindings,4u));
}

SparkStatus SparkOpsDifferentialGate(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor heads, SparkTensor lambda, SparkTensor gate, SparkTensor output,
	const SparkDifferentialGateParams *params)
{
	const uint32_t width = params->groups * params->signal * params->dimension, signals = params->groups * params->signal;
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->groups == 0u || params->signal == 0u || params->dimension == 0u ||
		params->heads_stride < params->groups * (params->signal + 1u) * params->dimension || params->lambda_stride < params->lambda_offset + signals ||
		params->gate_stride < width || params->output_stride < width )
		return(OpsRefuse("differential_gate","rows, groups of signal heads and one noise head, and strides that hold every head, lambda and gate"));
	status = OpsNeed("differential_gate","heads",heads,OpsRowsBytes(params->rows,params->heads_stride,params->groups * (params->signal + 1u) * params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("differential_gate","lambda",lambda,OpsRowsBytes(params->rows,params->lambda_stride,params->lambda_offset + signals,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("differential_gate","gate",gate,OpsRowsBytes(params->rows,params->gate_stride,width,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("differential_gate","output",output,OpsRowsBytes(params->rows,params->output_stride,width,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(heads,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(lambda,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(gate,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_DIFFERENTIAL_GATE,(width + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,4u));
}
