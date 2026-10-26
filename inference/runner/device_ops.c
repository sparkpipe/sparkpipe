#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "inference/runner/device_ops.h"

#define OPS_DECODE_PARTITION_KEYS 512u
#define OPS_PREFILL_PARTITION_KEYS 1024u
#define OPS_PREFILL_ROWS_PER_GROUP 8u
#define OPS_GEMV_OUTPUTS_PER_GROUP 16u
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
	"spark_gemv_gated_r1",
	"spark_gemv_gated_r2",
	"spark_gemv_gated_r4",
	"spark_gemv_gated_r8",
	"spark_gemm",
	"spark_gemm_f32",
	"spark_causal_conv_silu",
	"spark_gated_delta_prepare",
	"spark_gated_delta_rule",
	"spark_kv_store_paged",
	"spark_attention_decode",
	"spark_attention_prefill",
	"spark_attention_combine",
	"spark_silu_mul",
	"spark_add_rows",
	"spark_logprob_rows",
	"spark_stream_read"
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

SparkStatus SparkDeviceOpsAcquire(SparkDeviceOps **ops)
{
	char path[PATH_MAX];
	Dl_info info;
	const char *slash;
	SparkStatus status = SPARK_STATUS_OK;
	*ops = 0;
	pthread_mutex_lock(&ops_shared_mutex);
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
	{
		ops_shared_users++;
		*ops = &ops_shared;
	}
	pthread_mutex_unlock(&ops_shared_mutex);
	return(status);
}

void SparkDeviceOpsRelease(SparkDeviceOps *ops)
{
	pthread_mutex_lock(&ops_shared_mutex);
	if ( ops == &ops_shared && ops_shared_users != 0u && --ops_shared_users == 0u )
		SparkDeviceOpsClose(&ops_shared);
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

SparkStatus SparkOpsGather(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor table, SparkTensor ids, SparkTensor output,
	uint32_t rows, uint32_t width, uint32_t table_rows)
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
	bindings[0] = OpsBind(table,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(ids,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	bindings[3] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_GATHER,(width / 8u + 255u) / 256u,rows,1u,256u,&params,sizeof(params),bindings,4u));
}

static uint32_t OpsThreads(uint32_t width, uint32_t limit)
{
	uint32_t threads = (width + 31u) / 32u * 32u;
	return(threads > limit ? limit : threads);
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
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_residual","weight",weight,(uint64_t)params->dimension * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("rmsnorm_residual","output",output,OpsRowsBytes(params->rows,params->output_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(hidden,params->write_sum != 0u ? SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE : SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(addend != 0 ? *addend : hidden,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
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
		status = OpsNeed("rmsnorm_gate","weight",weight,(uint64_t)params->head_dimension * 2u);
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
	SparkDeviceBinding bindings[4];
	SparkStatus status;
	if ( params->rows == 0u || params->heads == 0u || params->head_dimension == 0u || params->head_dimension > 1024u ||
		params->rope_dimension % 2u != 0u || params->rope_dimension > params->head_dimension ||
		params->rope_dimension / 2u > SPARK_KERNEL_ROPE_PAIRS )
		return(OpsRefuse("head_rmsnorm_rope","heads of at most 1024 dimensions and an even rotary width inside the head are required"));
	status = OpsNeed("head_rmsnorm_rope","input",input,
		OpsRowsBytes(params->rows,params->input_stride,params->input_offset + (params->heads - 1u) * params->input_head_stride + params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("head_rmsnorm_rope","weight",weight,(uint64_t)params->head_dimension * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("head_rmsnorm_rope","positions",positions,(uint64_t)params->rows * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("head_rmsnorm_rope","output",output,OpsRowsBytes(params->rows,params->output_stride,params->heads * params->head_dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(weight,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(positions,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_HEAD_RMSNORM_ROPE,params->heads,params->rows,1u,OpsThreads(params->head_dimension,1024u),params,
		sizeof(*params),bindings,4u));
}

static SparkOpsKernel OpsGemvKernel(const SparkGemvParams *params, uint32_t rows)
{
	const uint32_t slot = rows == 8u ? 3u : rows == 4u ? 2u : rows == 2u ? 1u : 0u;
	if ( params->gated_input != 0u )
		return((SparkOpsKernel)(SPARK_OPS_GEMV_GATED_R1 + slot));
	if ( params->output_f32 != 0u )
		return((SparkOpsKernel)(SPARK_OPS_GEMV_F32_R1 + slot));
	return((SparkOpsKernel)(SPARK_OPS_GEMV_R1 + slot));
}

SparkStatus SparkOpsLinear(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y, const SparkGemvParams *params)
{
	const uint32_t K = params->input_dimension, N = params->output_dimension, element = params->output_f32 != 0u ? 4u : 2u;
	const uint32_t input_width = params->gated_input != 0u ? 2u * K : K;
	SparkDeviceBinding bindings[3];
	SparkGemvParams chunk;
	SparkStatus status;
	uint32_t done;
	if ( params->rows == 0u || K == 0u || N == 0u || K % 8u != 0u || params->input_stride % 8u != 0u || params->input_stride < input_width ||
		(params->rows > SPARK_OPS_GEMV_MAX_ROWS && (K % 32u != 0u || params->gated_input != 0u)) || (params->output_f32 != 0u && params->gated_input != 0u) )
		return(OpsRefuse("linear","an input width that is a multiple of 8 (32 above 8 rows), a stride that covers it, and a gated input only at 8 rows or fewer"));
	status = OpsNeed("linear","input",x,OpsRowsBytes(params->rows,params->input_stride,input_width,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear","weight",w,(uint64_t)N * K * 2u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("linear","output",y,OpsRowsBytes(params->rows,params->output_stride,N,element));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[1] = OpsBind(w,SPARK_DEVICE_BIND_READ);
	if ( params->rows > SPARK_OPS_GEMV_MAX_ROWS )
	{
		bindings[0] = OpsBind(x,SPARK_DEVICE_BIND_READ);
		bindings[2] = OpsBind(y,SPARK_DEVICE_BIND_WRITE);
		return(OpsLaunch(ops,queue,params->output_f32 != 0u ? SPARK_OPS_GEMM_F32 : SPARK_OPS_GEMM,(N + OPS_GEMM_OUTPUTS - 1u) / OPS_GEMM_OUTPUTS,
			(params->rows + OPS_GEMM_ROWS - 1u) / OPS_GEMM_ROWS,1u,128u,params,sizeof(*params),bindings,3u));
	}
	for ( done = 0u; status == SPARK_STATUS_OK && done < params->rows; done += chunk.rows )
	{
		const uint32_t left = params->rows - done;
		chunk = *params;
		chunk.rows = left >= 8u ? 8u : left >= 4u ? 4u : left >= 2u ? 2u : 1u;
		bindings[0] = OpsBind(SparkTensorSlice(x,(uint64_t)done * params->input_stride * 2u,OpsRowsBytes(chunk.rows,params->input_stride,input_width,2u)),
			SPARK_DEVICE_BIND_READ);
		bindings[2] = OpsBind(SparkTensorSlice(y,(uint64_t)done * params->output_stride * element,OpsRowsBytes(chunk.rows,params->output_stride,N,element)),
			SPARK_DEVICE_BIND_WRITE);
		status = OpsLaunch(ops,queue,OpsGemvKernel(params,chunk.rows),(N + OPS_GEMV_OUTPUTS_PER_GROUP - 1u) / OPS_GEMV_OUTPUTS_PER_GROUP,1u,1u,128u,
			&chunk,sizeof(chunk),bindings,3u);
	}
	return(status);
}

SparkStatus SparkOpsCausalConv(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor window, const SparkOpsSequences *sequences,
	SparkTensor input, SparkTensor weight, SparkTensor output, const SparkConvParams *params)
{
	SparkDeviceBinding bindings[6];
	SparkStatus status;
	if ( params->channels == 0u || params->sequences == 0u || params->kernel_taps == 0u || params->kernel_taps > 8u ||
		params->sequences != sequences->sequences || params->commit != sequences->commit )
		return(OpsRefuse("causal_conv","channels, one to eight taps and matching sequence tables are required"));
	status = OpsNeed("causal_conv","state_index",sequences->state_index,(uint64_t)params->sequences * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("causal_conv","row_begin",sequences->row_begin,((uint64_t)params->sequences + 1u) * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("causal_conv","weight",weight,(uint64_t)params->channels * params->kernel_taps * 2u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("causal_conv","window",window,(uint64_t)params->channels * params->kernel_taps * 2u);
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

SparkStatus SparkOpsKvStore(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor key, SparkTensor value,
	const SparkKvParams *params)
{
	SparkDeviceBinding bindings[7];
	SparkStatus status;
	if ( params->rows == 0u || params->kv_heads != paged->kv_heads || params->page_slots != paged->page_slots ||
		params->pages_per_sequence != paged->pages_per_sequence || params->pool_pages != paged->pool_pages || params->head_dimension == 0u )
		return(OpsRefuse("kv_store","the store geometry must match the paged pool"));
	status = OpsNeed("kv_store","pool",paged->pool,(uint64_t)paged->pool_pages * paged->page_slots * 2u * paged->kv_heads * params->head_dimension * 2u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_store","key",key,OpsRowsBytes(params->rows,params->key_stride,params->kv_heads * params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("kv_store","value",value,OpsRowsBytes(params->rows,params->value_stride,params->value_offset + params->kv_heads * params->head_dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(key,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(value,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->sequence_of_row,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(paged->page_table,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(paged->pool,SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	bindings[6] = OpsBind(OpsErrorWord(ops),SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_KV_STORE,params->rows,params->kv_heads,1u,OpsThreads(params->head_dimension,1024u),params,sizeof(*params),
		bindings,7u));
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
		segment.row_offset = begin;
		segment.rows = end;
		status = OpsLaunch(ops,queue,SPARK_OPS_ATTENTION_PREFILL,params->partitions,params->heads,
			(end - begin + OPS_PREFILL_ROWS_PER_GROUP - 1u) / OPS_PREFILL_ROWS_PER_GROUP,256u,&segment,sizeof(segment),bindings,6u);
	}
	return(status);
}

SparkStatus SparkOpsAttention(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor query, SparkTensor gate,
	SparkTensor partial, SparkTensor output, const SparkAttentionParams *params, const uint32_t *host_positions, const uint32_t *host_sequence_of_row)
{
	SparkAttentionParams launch = *params;
	SparkDeviceBinding bindings[6];
	SparkStatus status;
	uint32_t row, longest = 0u;
	if ( params->rows == 0u || params->heads == 0u || params->kv_heads != paged->kv_heads || params->heads % params->kv_heads != 0u ||
		params->head_dimension != SPARK_OPS_ATTENTION_HEAD_DIM || params->page_slots != paged->page_slots ||
		params->pages_per_sequence != paged->pages_per_sequence || params->pool_pages != paged->pool_pages || host_positions == 0 ||
		host_sequence_of_row == 0 )
		return(OpsRefuse("attention","256-dimensional heads, whole query groups per key head and the host row tables are required"));
	for ( row = 0u; row < params->rows; row++ )
		if ( host_positions[row] + 1u > longest )
			longest = host_positions[row] + 1u;
	launch.partition_keys = params->rows <= SPARK_OPS_GEMV_MAX_ROWS ? OPS_DECODE_PARTITION_KEYS : OPS_PREFILL_PARTITION_KEYS;
	launch.partitions = (longest + launch.partition_keys - 1u) / launch.partition_keys;
	launch.row_offset = 0u;
	status = OpsNeed("attention","partial",partial,(uint64_t)params->rows * params->heads * launch.partitions * SPARK_OPS_PARTIAL_FLOATS * 4u);
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention","query",query,OpsRowsBytes(params->rows,params->query_stride,params->heads * params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention","gate",gate,
			OpsRowsBytes(params->rows,params->gate_stride,params->gate_offset + (params->heads - 1u) * params->gate_head_stride + params->head_dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("attention","output",output,OpsRowsBytes(params->rows,params->output_stride,params->heads * params->head_dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(query,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(paged->pool,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->page_table,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(paged->sequence_of_row,SPARK_DEVICE_BIND_READ);
	bindings[4] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[5] = OpsBind(partial,SPARK_DEVICE_BIND_WRITE);
	if ( params->rows <= SPARK_OPS_GEMV_MAX_ROWS )
		status = OpsLaunch(ops,queue,SPARK_OPS_ATTENTION_DECODE,launch.partitions,params->heads,params->rows,256u,&launch,sizeof(launch),bindings,6u);
	else
		status = OpsAttentionPrefill(ops,queue,bindings,&launch,host_sequence_of_row);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(partial,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(gate,SPARK_DEVICE_BIND_READ);
	bindings[2] = OpsBind(paged->positions,SPARK_DEVICE_BIND_READ);
	bindings[3] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_ATTENTION_COMBINE,params->heads,params->rows,1u,256u,&launch,sizeof(launch),bindings,4u));
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

SparkStatus SparkOpsSiluMul(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor input, SparkTensor output, const SparkSiluMulParams *params)
{
	SparkDeviceBinding bindings[2];
	SparkStatus status;
	if ( params->rows == 0u || params->dimension == 0u || params->input_stride < 2u * params->dimension )
		return(OpsRefuse("silu_mul","rows and an input stride that holds the gate and up halves are required"));
	status = OpsNeed("silu_mul","input",input,OpsRowsBytes(params->rows,params->input_stride,2u * params->dimension,2u));
	if ( status == SPARK_STATUS_OK )
		status = OpsNeed("silu_mul","output",output,OpsRowsBytes(params->rows,params->output_stride,params->dimension,2u));
	if ( status != SPARK_STATUS_OK )
		return(status);
	bindings[0] = OpsBind(input,SPARK_DEVICE_BIND_READ);
	bindings[1] = OpsBind(output,SPARK_DEVICE_BIND_WRITE);
	return(OpsLaunch(ops,queue,SPARK_OPS_SILU_MUL,(params->dimension + 255u) / 256u,params->rows,1u,256u,params,sizeof(*params),bindings,2u));
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
