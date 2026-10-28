#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_json.h"
#include "tools/token_prompts.h"

#define REF_LAYERS 64u
#define REF_HIDDEN 5120u
#define REF_VOCAB 248320u
#define REF_FFN 17408u
#define REF_GATE_UP (2u * REF_FFN)
#define REF_KEY_HEADS 16u
#define REF_VALUE_HEADS 48u
#define REF_DIM 128u
#define REF_QK (REF_KEY_HEADS * REF_DIM)
#define REF_V (REF_VALUE_HEADS * REF_DIM)
#define REF_CHANNELS (2u * REF_QK + REF_V)
#define REF_HEADS 24u
#define REF_KV_HEADS 4u
#define REF_HEAD_DIM 256u
#define REF_FP8_BLOCK 128u
#define REF_MAX_SHARDS 128u
#define REF_Q (REF_HEADS * REF_HEAD_DIM)
#define REF_GATED_Q (2u * REF_Q)
#define REF_KV (REF_KV_HEADS * REF_HEAD_DIM)
#define REF_ROPE 64u
#define REF_THETA 10000000.0
#define REF_EPSILON 1e-6f
#define REF_CONV 4u
#define REF_MAX_CONTEXT 4096u
#define REF_SPLIT 1024u
#define REF_PREFIX "model.language_model."

typedef struct RefShard
{
	char name[96];
	uint8_t *base;
	uint64_t bytes;
	SparkJsonDocument header;
	uint64_t data;
} RefShard;

typedef struct RefCheckpoint
{
	char root[1024];
	SparkJsonDocument index;
	RefShard shards[REF_MAX_SHARDS];
	uint32_t shard_count;
} RefCheckpoint;

typedef struct RefLayer
{
	uint32_t attention;
	float *input_norm, *post_norm;
	float *gate, *up, *down;
	float *qkv, *z, *b, *a, *conv, *a_log, *dt_bias, *gdn_norm, *gdn_out;
	float *q, *k, *v, *o, *q_norm, *k_norm;
} RefLayer;

typedef struct RefModel
{
	RefLayer layers[REF_LAYERS];
	float *final_norm;
	float *head;
	const uint16_t *embed;
} RefModel;

typedef struct RefState
{
	float *gdn[REF_LAYERS];
	float *window[REF_LAYERS];
	float *keys[REF_LAYERS];
	float *values[REF_LAYERS];
	uint32_t length;
} RefState;

static float Bf16(uint16_t bits)
{
	uint32_t word = (uint32_t)bits << 16;
	float value;
	memcpy(&value,&word,sizeof(value));
	return(value);
}

static uint32_t ref_round_activations = 1u;

static float RoundBf16(float value)
{
	uint32_t bits;
	if ( ref_round_activations == 0u )
		return(value);
	memcpy(&bits,&value,sizeof(bits));
	if ( (bits & 0x7fffffffu) > 0x7f800000u )
		return(value);
	bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
	memcpy(&value,&bits,sizeof(value));
	return(value);
}

static float Silu(float x)
{
	return((float)((double)x / (1.0 + exp(-(double)x))));
}

static float Sigmoid(float x)
{
	return((float)(1.0 / (1.0 + exp(-(double)x))));
}

static double Seconds(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec + now.tv_nsec / 1e9);
}

static void Die(const char *what, const char *detail)
{
	fprintf(stderr,"qwen38_27b_reference: %s%s%s\n",what,detail != 0 ? ": " : "",detail != 0 ? detail : "");
	exit(1);
}

static RefShard *RefOpenShard(RefCheckpoint *checkpoint, const char *name)
{
	char path[2048];
	struct stat info;
	uint32_t index;
	uint64_t header_bytes;
	int fd;
	RefShard *shard;
	for ( index = 0u; index < checkpoint->shard_count; index++ )
		if ( strcmp(checkpoint->shards[index].name,name) == 0 )
			return(&checkpoint->shards[index]);
	if ( checkpoint->shard_count == REF_MAX_SHARDS )
		Die("too many shards",name);
	shard = &checkpoint->shards[checkpoint->shard_count++];
	snprintf(shard->name,sizeof(shard->name),"%s",name);
	snprintf(path,sizeof(path),"%s/%s",checkpoint->root,name);
	fd = open(path,O_RDONLY);
	if ( fd < 0 || fstat(fd,&info) != 0 )
		Die("cannot open shard",path);
	shard->bytes = (uint64_t)info.st_size;
	shard->base = (uint8_t *)mmap(0,(size_t)shard->bytes,PROT_READ,MAP_SHARED,fd,0);
	close(fd);
	if ( shard->base == MAP_FAILED )
		Die("cannot map shard",path);
	memcpy(&header_bytes,shard->base,sizeof(header_bytes));
	SparkJsonDocumentReset(&shard->header);
	if ( SparkJsonParseText((const char *)shard->base + 8,(size_t)header_bytes,&shard->header) != SPARK_STATUS_OK )
		Die("cannot parse shard header",path);
	shard->data = 8u + header_bytes;
	return(shard);
}

static const uint8_t *RefTensorAs(RefCheckpoint *checkpoint, const char *name, const char *dtype, uint64_t elements, uint32_t element_bytes)
{
	const int32_t map = SparkJsonFindObjectMember(&checkpoint->index,SparkJsonGetRootToken(&checkpoint->index),"weight_map");
	const int32_t entry = map >= 0 ? SparkJsonFindObjectMember(&checkpoint->index,map,name) : -1;
	char *shard_name = 0;
	RefShard *shard;
	int32_t meta, kind, offsets, shape, dimension;
	uint64_t begin, end, count = 1u, value;
	uint32_t index;
	if ( entry < 0 || SparkJsonCopyString(&checkpoint->index,entry,&shard_name) != SPARK_STATUS_OK )
		Die("checkpoint has no tensor",name);
	shard = RefOpenShard(checkpoint,shard_name);
	free(shard_name);
	meta = SparkJsonFindObjectMember(&shard->header,SparkJsonGetRootToken(&shard->header),name);
	kind = meta >= 0 ? SparkJsonFindObjectMember(&shard->header,meta,"dtype") : -1;
	offsets = meta >= 0 ? SparkJsonFindObjectMember(&shard->header,meta,"data_offsets") : -1;
	shape = meta >= 0 ? SparkJsonFindObjectMember(&shard->header,meta,"shape") : -1;
	if ( kind < 0 || offsets < 0 || shape < 0 || !SparkJsonStringEquals(&shard->header,kind,dtype) ||
		SparkJsonGetUInt64(&shard->header,SparkJsonGetArrayElement(&shard->header,offsets,0u),&begin) != SPARK_STATUS_OK ||
		SparkJsonGetUInt64(&shard->header,SparkJsonGetArrayElement(&shard->header,offsets,1u),&end) != SPARK_STATUS_OK )
		Die("tensor has the wrong dtype or no offsets",name);
	for ( index = 0u; index < SparkJsonGetArrayElementCount(&shard->header,shape); index++ )
	{
		dimension = SparkJsonGetArrayElement(&shard->header,shape,index);
		if ( SparkJsonGetUInt64(&shard->header,dimension,&value) != SPARK_STATUS_OK )
			Die("tensor shape is unreadable",name);
		count *= value;
	}
	if ( count != elements || end - begin != elements * element_bytes || shard->data + end > shard->bytes )
		Die("tensor has the wrong size",name);
	return(shard->base + shard->data + begin);
}

static const uint8_t *RefTensor(RefCheckpoint *checkpoint, const char *name, const char *dtype, uint64_t elements)
{
	return(RefTensorAs(checkpoint,name,dtype,elements,2u));
}

static uint32_t RefTensorIsFp8(RefCheckpoint *checkpoint, const char *name)
{
	const int32_t map = SparkJsonFindObjectMember(&checkpoint->index,SparkJsonGetRootToken(&checkpoint->index),"weight_map");
	const int32_t entry = map >= 0 ? SparkJsonFindObjectMember(&checkpoint->index,map,name) : -1;
	char *shard_name = 0;
	RefShard *shard;
	int32_t meta, kind;
	if ( entry < 0 || SparkJsonCopyString(&checkpoint->index,entry,&shard_name) != SPARK_STATUS_OK )
		Die("checkpoint has no tensor",name);
	shard = RefOpenShard(checkpoint,shard_name);
	free(shard_name);
	meta = SparkJsonFindObjectMember(&shard->header,SparkJsonGetRootToken(&shard->header),name);
	kind = meta >= 0 ? SparkJsonFindObjectMember(&shard->header,meta,"dtype") : -1;
	return(kind >= 0 && SparkJsonStringEquals(&shard->header,kind,"F8_E4M3") ? 1u : 0u);
}

static float E4m3(uint8_t code)
{
	const uint32_t exponent = (code >> 3u) & 15u, mantissa = code & 7u;
	const float magnitude = exponent == 0u ? ldexpf((float)mantissa,-9) : ldexpf(1.0f + (float)mantissa / 8.0f,(int)exponent - 7);
	return((code & 0x80u) != 0u ? -magnitude : magnitude);
}

static float *RefLoad(RefCheckpoint *checkpoint, const char *name, uint64_t elements, float add)
{
	const uint16_t *raw = (const uint16_t *)RefTensor(checkpoint,name,"BF16",elements);
	float *out = (float *)malloc((size_t)elements * sizeof(float));
	if ( out == 0 )
		Die("out of memory loading",name);
	dispatch_apply((size_t)((elements + 65535u) / 65536u),dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t block)
	{
		const uint64_t first = (uint64_t)block * 65536u, last = first + 65536u < elements ? first + 65536u : elements;
		for ( uint64_t index = first; index < last; index++ )
			out[index] = Bf16(raw[index]) + add;
	});
	return(out);
}

static float *RefLayerLoad(RefCheckpoint *checkpoint, uint32_t layer, const char *name, uint64_t elements, float add)
{
	char key[256];
	snprintf(key,sizeof(key),REF_PREFIX "layers.%u.%s",layer,name);
	return(RefLoad(checkpoint,key,elements,add));
}

static float *RefLayerLinear(RefCheckpoint *checkpoint, uint32_t layer, const char *name, uint32_t rows, uint32_t columns)
{
	char key[256], scale_key[288];
	const uint8_t *raw;
	const uint16_t *scale;
	float *out;
	const uint32_t scale_columns = (columns + REF_FP8_BLOCK - 1u) / REF_FP8_BLOCK;
	snprintf(key,sizeof(key),REF_PREFIX "layers.%u.%s.weight",layer,name);
	if ( RefTensorIsFp8(checkpoint,key) == 0u )
		return(RefLoad(checkpoint,key,(uint64_t)rows * columns,0.0f));
	snprintf(scale_key,sizeof(scale_key),"%s_scale_inv",key);
	raw = RefTensorAs(checkpoint,key,"F8_E4M3",(uint64_t)rows * columns,1u);
	scale = (const uint16_t *)RefTensor(checkpoint,scale_key,"BF16",(uint64_t)((rows + REF_FP8_BLOCK - 1u) / REF_FP8_BLOCK) * scale_columns);
	out = (float *)malloc((size_t)rows * columns * sizeof(float));
	if ( out == 0 )
		Die("out of memory loading",key);
	dispatch_apply((size_t)rows,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t row)
	{
		for ( uint32_t column = 0u; column < columns; column++ )
			out[row * columns + column] = E4m3(raw[row * columns + column]) *
				Bf16(scale[(row / REF_FP8_BLOCK) * scale_columns + column / REF_FP8_BLOCK]);
	});
	return(out);
}

static void RefLoadModel(RefCheckpoint *checkpoint, RefModel *model)
{
	uint32_t index;
	for ( index = 0u; index < REF_LAYERS; index++ )
	{
		RefLayer *layer = &model->layers[index];
		layer->attention = index % 4u == 3u;
		layer->input_norm = RefLayerLoad(checkpoint,index,"input_layernorm.weight",REF_HIDDEN,1.0f);
		layer->post_norm = RefLayerLoad(checkpoint,index,"post_attention_layernorm.weight",REF_HIDDEN,1.0f);
		layer->gate = RefLayerLinear(checkpoint,index,"mlp.gate_proj",REF_FFN,REF_HIDDEN);
		layer->up = RefLayerLinear(checkpoint,index,"mlp.up_proj",REF_FFN,REF_HIDDEN);
		layer->down = RefLayerLinear(checkpoint,index,"mlp.down_proj",REF_HIDDEN,REF_FFN);
		if ( layer->attention != 0u )
		{
			layer->q = RefLayerLinear(checkpoint,index,"self_attn.q_proj",2u * REF_Q,REF_HIDDEN);
			layer->k = RefLayerLinear(checkpoint,index,"self_attn.k_proj",REF_KV,REF_HIDDEN);
			layer->v = RefLayerLinear(checkpoint,index,"self_attn.v_proj",REF_KV,REF_HIDDEN);
			layer->o = RefLayerLinear(checkpoint,index,"self_attn.o_proj",REF_HIDDEN,REF_Q);
			layer->q_norm = RefLayerLoad(checkpoint,index,"self_attn.q_norm.weight",REF_HEAD_DIM,1.0f);
			layer->k_norm = RefLayerLoad(checkpoint,index,"self_attn.k_norm.weight",REF_HEAD_DIM,1.0f);
		}
		else
		{
			layer->qkv = RefLayerLinear(checkpoint,index,"linear_attn.in_proj_qkv",REF_CHANNELS,REF_HIDDEN);
			layer->z = RefLayerLinear(checkpoint,index,"linear_attn.in_proj_z",REF_V,REF_HIDDEN);
			layer->b = RefLayerLoad(checkpoint,index,"linear_attn.in_proj_b.weight",(uint64_t)REF_VALUE_HEADS * REF_HIDDEN,0.0f);
			layer->a = RefLayerLoad(checkpoint,index,"linear_attn.in_proj_a.weight",(uint64_t)REF_VALUE_HEADS * REF_HIDDEN,0.0f);
			layer->conv = RefLayerLoad(checkpoint,index,"linear_attn.conv1d.weight",(uint64_t)REF_CHANNELS * REF_CONV,0.0f);
			layer->a_log = RefLayerLoad(checkpoint,index,"linear_attn.A_log",REF_VALUE_HEADS,0.0f);
			layer->dt_bias = RefLayerLoad(checkpoint,index,"linear_attn.dt_bias",REF_VALUE_HEADS,0.0f);
			layer->gdn_norm = RefLayerLoad(checkpoint,index,"linear_attn.norm.weight",REF_DIM,0.0f);
			layer->gdn_out = RefLayerLinear(checkpoint,index,"linear_attn.out_proj",REF_HIDDEN,REF_V);
		}
		if ( index % 8u == 7u )
			fprintf(stderr,"loaded %u layers\n",index + 1u);
	}
	model->final_norm = RefLoad(checkpoint,REF_PREFIX "norm.weight",REF_HIDDEN,1.0f);
	model->head = RefLoad(checkpoint,"lm_head.weight",(uint64_t)REF_VOCAB * REF_HIDDEN,0.0f);
	model->embed = (const uint16_t *)RefTensor(checkpoint,REF_PREFIX "embed_tokens.weight","BF16",(uint64_t)REF_VOCAB * REF_HIDDEN);
}

static void RefGemm(uint32_t variant, const float *x, uint32_t rows, uint32_t input, const float *w, uint32_t output, float *y, uint32_t y_stride)
{
	uint32_t first, count;
	if ( variant == 0u )
	{
		cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,(int)rows,(int)output,(int)input,1.0f,x,(int)input,w,(int)input,0.0f,y,(int)y_stride);
		return;
	}
	for ( first = (input - 1u) / REF_SPLIT * REF_SPLIT, count = 0u;; first -= REF_SPLIT, count++ )
	{
		const uint32_t width = input - first < REF_SPLIT ? input - first : REF_SPLIT;
		cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,(int)rows,(int)output,(int)width,1.0f,x + first,(int)input,w + first,(int)input,
			count == 0u ? 0.0f : 1.0f,y,(int)y_stride);
		if ( first == 0u )
			break;
	}
}

static float RefSum(uint32_t variant, const float *values, uint32_t count, uint32_t stride)
{
	float total = 0.0f;
	uint32_t index;
	for ( index = 0u; index < count; index++ )
		total += values[(uint64_t)(variant == 0u ? index : count - 1u - index) * stride];
	return(total);
}

static void RefRound(float *values, uint64_t count)
{
	uint64_t index;
	for ( index = 0u; index < count; index++ )
		values[index] = RoundBf16(values[index]);
}

static void RefNorm(uint32_t variant, const float *x, uint32_t rows, const float *weight, float *y)
{
	dispatch_apply(rows,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t row)
	{
		const float *in = x + row * REF_HIDDEN;
		float squares[REF_HIDDEN];
		for ( uint32_t i = 0u; i < REF_HIDDEN; i++ )
			squares[i] = in[i] * in[i];
		const float scale = (float)(1.0 / sqrt((double)(RefSum(variant,squares,REF_HIDDEN,1u) / (float)REF_HIDDEN + REF_EPSILON)));
		for ( uint32_t i = 0u; i < REF_HIDDEN; i++ )
			y[row * REF_HIDDEN + i] = RoundBf16(in[i] * scale * weight[i]);
	});
}

static void RefLinearRows(uint32_t variant, const float *x, uint32_t rows, uint32_t input, const float *w, uint32_t output, float *y)
{
	RefGemm(variant,x,rows,input,w,output,y,output);
	RefRound(y,(uint64_t)rows * output);
}

static void RefGdn(uint32_t variant, const RefLayer *layer, uint32_t index, RefState *state, const float *x, uint32_t rows, float *partial)
{
	float *mix = (float *)malloc((size_t)rows * REF_CHANNELS * sizeof(float));
	float *z = (float *)malloc((size_t)rows * REF_V * sizeof(float));
	float *b = (float *)malloc((size_t)rows * REF_VALUE_HEADS * sizeof(float));
	float *a = (float *)malloc((size_t)rows * REF_VALUE_HEADS * sizeof(float));
	float *q = (float *)malloc((size_t)rows * REF_QK * sizeof(float));
	float *k = (float *)malloc((size_t)rows * REF_QK * sizeof(float));
	float *core = (float *)malloc((size_t)rows * REF_V * sizeof(float));
	float *gated = (float *)malloc((size_t)rows * REF_V * sizeof(float));
	float *window = state->window[index], *s = state->gdn[index];
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->qkv,REF_CHANNELS,mix);
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->z,REF_V,z);
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->b,REF_VALUE_HEADS,b);
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->a,REF_VALUE_HEADS,a);
	dispatch_apply(REF_CHANNELS,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t channel)
	{
		float *taps = window + channel * REF_CONV;
		for ( uint32_t row = 0u; row < rows; row++ )
		{
			float total = 0.0f;
			for ( uint32_t tap = 0u; tap + 1u < REF_CONV; tap++ )
				taps[tap] = taps[tap + 1u];
			taps[REF_CONV - 1u] = mix[(uint64_t)row * REF_CHANNELS + channel];
			for ( uint32_t tap = 0u; tap < REF_CONV; tap++ )
				total += taps[variant == 0u ? tap : REF_CONV - 1u - tap] * layer->conv[channel * REF_CONV + (variant == 0u ? tap : REF_CONV - 1u - tap)];
			mix[(uint64_t)row * REF_CHANNELS + channel] = RoundBf16(Silu(RoundBf16(total)));
		}
	});
	dispatch_apply((size_t)rows * REF_KEY_HEADS,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t item)
	{
		const uint32_t row = (uint32_t)(item / REF_KEY_HEADS), head = (uint32_t)(item % REF_KEY_HEADS);
		const float *qin = mix + (uint64_t)row * REF_CHANNELS + head * REF_DIM, *kin = qin + REF_QK;
		float qs[REF_DIM], ks[REF_DIM];
		for ( uint32_t i = 0u; i < REF_DIM; i++ )
		{
			qs[i] = qin[i] * qin[i];
			ks[i] = kin[i] * kin[i];
		}
		const float qscale = (float)(1.0 / sqrt((double)(RefSum(variant,qs,REF_DIM,1u) + REF_EPSILON)) / sqrt((double)REF_DIM));
		const float kscale = (float)(1.0 / sqrt((double)(RefSum(variant,ks,REF_DIM,1u) + REF_EPSILON)));
		for ( uint32_t i = 0u; i < REF_DIM; i++ )
		{
			q[(uint64_t)row * REF_QK + head * REF_DIM + i] = qin[i] * qscale;
			k[(uint64_t)row * REF_QK + head * REF_DIM + i] = kin[i] * kscale;
		}
	});
	dispatch_apply((size_t)REF_VALUE_HEADS * REF_DIM,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t item)
	{
		const uint32_t head = (uint32_t)(item / REF_DIM), column = (uint32_t)(item % REF_DIM), key_head = head / 3u;
		float cell[REF_DIM], products[REF_DIM];
		for ( uint32_t i = 0u; i < REF_DIM; i++ )
			cell[i] = s[((uint64_t)head * REF_DIM + i) * REF_DIM + column];
		for ( uint32_t row = 0u; row < rows; row++ )
		{
			const float *kv = k + (uint64_t)row * REF_QK + key_head * REF_DIM, *qv = q + (uint64_t)row * REF_QK + key_head * REF_DIM;
			const float shifted = a[(uint64_t)row * REF_VALUE_HEADS + head] + layer->dt_bias[head];
			const double softplus = shifted > 20.0f ? (double)shifted : log1p(exp((double)shifted));
			const float decay = (float)exp(-exp((double)layer->a_log[head]) * softplus);
			const float beta = Sigmoid(b[(uint64_t)row * REF_VALUE_HEADS + head]);
			const float value = mix[(uint64_t)row * REF_CHANNELS + 2u * REF_QK + head * REF_DIM + column];
			for ( uint32_t i = 0u; i < REF_DIM; i++ )
			{
				cell[i] *= decay;
				products[i] = cell[i] * kv[i];
			}
			const float coefficient = beta * (value - RefSum(variant,products,REF_DIM,1u));
			for ( uint32_t i = 0u; i < REF_DIM; i++ )
			{
				cell[i] += coefficient * kv[i];
				products[i] = cell[i] * qv[i];
			}
			core[(uint64_t)row * REF_V + head * REF_DIM + column] = RoundBf16(RefSum(variant,products,REF_DIM,1u));
		}
		for ( uint32_t i = 0u; i < REF_DIM; i++ )
			s[((uint64_t)head * REF_DIM + i) * REF_DIM + column] = cell[i];
	});
	dispatch_apply((size_t)rows * REF_VALUE_HEADS,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t item)
	{
		const uint32_t row = (uint32_t)(item / REF_VALUE_HEADS), head = (uint32_t)(item % REF_VALUE_HEADS);
		const float *o = core + (uint64_t)row * REF_V + head * REF_DIM;
		float squares[REF_DIM];
		for ( uint32_t i = 0u; i < REF_DIM; i++ )
			squares[i] = o[i] * o[i];
		const float scale = (float)(1.0 / sqrt((double)(RefSum(variant,squares,REF_DIM,1u) / (float)REF_DIM + REF_EPSILON)));
		for ( uint32_t i = 0u; i < REF_DIM; i++ )
		{
			const float normed = RoundBf16(RoundBf16(o[i] * scale) * layer->gdn_norm[i]);
			gated[(uint64_t)row * REF_V + head * REF_DIM + i] = RoundBf16(normed * Silu(z[(uint64_t)row * REF_V + head * REF_DIM + i]));
		}
	});
	RefLinearRows(variant,gated,rows,REF_V,layer->gdn_out,REF_HIDDEN,partial);
	free(mix);
	free(z);
	free(b);
	free(a);
	free(q);
	free(k);
	free(core);
	free(gated);
}

static void RefHeadNormRope(uint32_t variant, const float *in, uint32_t in_stride, uint32_t head_stride, uint32_t heads, const float *weight,
	uint32_t first_position, uint32_t rows, float *out)
{
	dispatch_apply((size_t)rows * heads,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t item)
	{
		const uint32_t row = (uint32_t)(item / heads), head = (uint32_t)(item % heads);
		const float *x = in + (uint64_t)row * in_stride + head * head_stride;
		float squares[REF_HEAD_DIM], normed[REF_HEAD_DIM];
		for ( uint32_t i = 0u; i < REF_HEAD_DIM; i++ )
			squares[i] = x[i] * x[i];
		const float scale = (float)(1.0 / sqrt((double)(RefSum(variant,squares,REF_HEAD_DIM,1u) / (float)REF_HEAD_DIM + REF_EPSILON)));
		for ( uint32_t i = 0u; i < REF_HEAD_DIM; i++ )
			normed[i] = RoundBf16(x[i] * scale * weight[i]);
		for ( uint32_t i = 0u; i < REF_HEAD_DIM; i++ )
		{
			float value = normed[i];
			if ( i < REF_ROPE )
			{
				const uint32_t pair = i % (REF_ROPE / 2u);
				const float frequency = (float)(1.0 / pow(REF_THETA,(double)(2u * pair) / (double)REF_ROPE));
				const float angle = (float)(first_position + row) * frequency;
				const float partner = i < REF_ROPE / 2u ? -normed[i + REF_ROPE / 2u] : normed[i - REF_ROPE / 2u];
				value = (float)((double)normed[i] * cos((double)angle) + (double)partner * sin((double)angle));
			}
			out[((uint64_t)row * heads + head) * REF_HEAD_DIM + i] = RoundBf16(value);
		}
	});
}

static void RefAttention(uint32_t variant, const RefLayer *layer, uint32_t index, RefState *state, const float *x, uint32_t rows, float *partial)
{
	const uint32_t first = state->length, length = first + rows;
	float *gq = (float *)malloc((size_t)rows * REF_GATED_Q * sizeof(float));
	float *kraw = (float *)malloc((size_t)rows * REF_KV * sizeof(float));
	float *vraw = (float *)malloc((size_t)rows * REF_KV * sizeof(float));
	float *query = (float *)malloc((size_t)rows * REF_Q * sizeof(float));
	float *gated = (float *)malloc((size_t)rows * REF_Q * sizeof(float));
	float *keys = state->keys[index], *values = state->values[index];
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->q,2u * REF_Q,gq);
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->k,REF_KV,kraw);
	RefLinearRows(variant,x,rows,REF_HIDDEN,layer->v,REF_KV,vraw);
	RefHeadNormRope(variant,gq,2u * REF_Q,2u * REF_HEAD_DIM,REF_HEADS,layer->q_norm,first,rows,query);
	RefHeadNormRope(variant,kraw,REF_KV,REF_HEAD_DIM,REF_KV_HEADS,layer->k_norm,first,rows,keys + (uint64_t)first * REF_KV);
	memcpy(values + (uint64_t)first * REF_KV,vraw,(size_t)rows * REF_KV * sizeof(float));
	dispatch_apply(REF_HEADS,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t head)
	{
		const uint32_t kv = (uint32_t)head / (REF_HEADS / REF_KV_HEADS);
		float *scores = (float *)malloc((size_t)rows * length * sizeof(float));
		float *out = (float *)malloc((size_t)rows * REF_HEAD_DIM * sizeof(float));
		cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,(int)rows,(int)length,(int)REF_HEAD_DIM,1.0f / 16.0f,query + head * REF_HEAD_DIM,(int)REF_Q,
			keys + kv * REF_HEAD_DIM,(int)REF_KV,0.0f,scores,(int)length);
		for ( uint32_t row = 0u; row < rows; row++ )
		{
			float *line = scores + (uint64_t)row * length, best = -INFINITY;
			const uint32_t visible = first + row + 1u;
			for ( uint32_t j = 0u; j < visible; j++ )
				best = line[j] > best ? line[j] : best;
			for ( uint32_t j = 0u; j < length; j++ )
				line[j] = j < visible ? (float)exp((double)(line[j] - best)) : 0.0f;
			const float total = RefSum(variant,line,visible,1u);
			for ( uint32_t j = 0u; j < visible; j++ )
				line[j] /= total;
		}
		cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,(int)rows,(int)REF_HEAD_DIM,(int)length,1.0f,scores,(int)length,
			values + kv * REF_HEAD_DIM,(int)REF_KV,0.0f,out,(int)REF_HEAD_DIM);
		for ( uint32_t row = 0u; row < rows; row++ )
			for ( uint32_t d = 0u; d < REF_HEAD_DIM; d++ )
			{
				const float g = gq[(uint64_t)row * 2u * REF_Q + head * 2u * REF_HEAD_DIM + REF_HEAD_DIM + d];
				gated[(uint64_t)row * REF_Q + head * REF_HEAD_DIM + d] = RoundBf16(RoundBf16(out[(uint64_t)row * REF_HEAD_DIM + d]) * Sigmoid(g));
			}
		free(scores);
		free(out);
	});
	RefLinearRows(variant,gated,rows,REF_Q,layer->o,REF_HIDDEN,partial);
	free(gq);
	free(kraw);
	free(vraw);
	free(query);
	free(gated);
}

static void RefAdd(float *hidden, const float *partial, uint64_t count)
{
	uint64_t index;
	for ( index = 0u; index < count; index++ )
		hidden[index] = RoundBf16(hidden[index] + partial[index]);
}

static void RefForward(uint32_t variant, const RefModel *model, RefState *state, const uint32_t *tokens, uint32_t rows, float *logits)
{
	float *hidden = (float *)malloc((size_t)rows * REF_HIDDEN * sizeof(float));
	float *normed = (float *)malloc((size_t)rows * REF_HIDDEN * sizeof(float));
	float *partial = (float *)malloc((size_t)rows * REF_HIDDEN * sizeof(float));
	float *gate_up = (float *)malloc((size_t)rows * REF_GATE_UP * sizeof(float));
	float *act = (float *)malloc((size_t)rows * REF_FFN * sizeof(float));
	uint32_t row, index, attention = 0u, recurrent = 0u;
	for ( row = 0u; row < rows; row++ )
		for ( index = 0u; index < REF_HIDDEN; index++ )
			hidden[(uint64_t)row * REF_HIDDEN + index] = Bf16(model->embed[(uint64_t)tokens[row] * REF_HIDDEN + index]);
	for ( index = 0u; index < REF_LAYERS; index++ )
	{
		const RefLayer *layer = &model->layers[index];
		RefNorm(variant,hidden,rows,layer->input_norm,normed);
		if ( layer->attention != 0u )
			RefAttention(variant,layer,attention++,state,normed,rows,partial);
		else
			RefGdn(variant,layer,recurrent++,state,normed,rows,partial);
		RefAdd(hidden,partial,(uint64_t)rows * REF_HIDDEN);
		RefNorm(variant,hidden,rows,layer->post_norm,normed);
		RefGemm(variant,normed,rows,REF_HIDDEN,layer->gate,REF_FFN,gate_up,2u * REF_FFN);
		RefGemm(variant,normed,rows,REF_HIDDEN,layer->up,REF_FFN,gate_up + REF_FFN,2u * REF_FFN);
		RefRound(gate_up,(uint64_t)rows * 2u * REF_FFN);
		for ( row = 0u; row < rows; row++ )
			for ( uint32_t i = 0u; i < REF_FFN; i++ )
				act[(uint64_t)row * REF_FFN + i] = RoundBf16(Silu(gate_up[(uint64_t)row * 2u * REF_FFN + i]) * gate_up[(uint64_t)row * 2u * REF_FFN + REF_FFN + i]);
		RefLinearRows(variant,act,rows,REF_FFN,layer->down,REF_HIDDEN,partial);
		RefAdd(hidden,partial,(uint64_t)rows * REF_HIDDEN);
	}
	RefNorm(variant,hidden,rows,model->final_norm,normed);
	RefGemm(variant,normed,rows,REF_HIDDEN,model->head,REF_VOCAB,logits,REF_VOCAB);
	state->length += rows;
	free(hidden);
	free(normed);
	free(partial);
	free(gate_up);
	free(act);
}

static void RefStateCreate(RefState *state)
{
	uint32_t index;
	memset(state,0,sizeof(*state));
	for ( index = 0u; index < REF_LAYERS; index++ )
	{
		state->gdn[index] = (float *)calloc((size_t)REF_VALUE_HEADS * REF_DIM * REF_DIM,sizeof(float));
		state->window[index] = (float *)calloc((size_t)REF_CHANNELS * REF_CONV,sizeof(float));
		state->keys[index] = (float *)calloc((size_t)REF_MAX_CONTEXT * REF_KV,sizeof(float));
		state->values[index] = (float *)calloc((size_t)REF_MAX_CONTEXT * REF_KV,sizeof(float));
		if ( state->gdn[index] == 0 || state->window[index] == 0 || state->keys[index] == 0 || state->values[index] == 0 )
			Die("out of memory for the reference state",0);
	}
}

static void RefStateReset(RefState *state)
{
	uint32_t index;
	for ( index = 0u; index < REF_LAYERS; index++ )
	{
		memset(state->gdn[index],0,(size_t)REF_VALUE_HEADS * REF_DIM * REF_DIM * sizeof(float));
		memset(state->window[index],0,(size_t)REF_CHANNELS * REF_CONV * sizeof(float));
	}
	state->length = 0u;
}

static uint32_t RefArgmax(const float *logits)
{
	uint32_t index, best = 0u;
	for ( index = 1u; index < REF_VOCAB; index++ )
		if ( logits[index] > logits[best] )
			best = index;
	return(best);
}

static double RefLogprob(const float *logits, uint32_t token)
{
	double best = -INFINITY, total = 0.0;
	uint32_t index;
	for ( index = 0u; index < REF_VOCAB; index++ )
		best = logits[index] > best ? logits[index] : best;
	for ( index = 0u; index < REF_VOCAB; index++ )
		total += exp((double)logits[index] - best);
	return((double)logits[token] - best - log(total));
}

static void RefPrompt(const RefModel *model, RefState *state, const SparkTokenPrompt *prompt, const char *out, uint32_t new_tokens, FILE *greedy, uint32_t first)
{
	float *logits = (float *)malloc((size_t)prompt->count * REF_VOCAB * sizeof(float));
	uint32_t variant, *generated = (uint32_t *)calloc(new_tokens + 1u,sizeof(uint32_t));
	char path[4096];
	if ( logits == 0 || generated == 0 || prompt->count + new_tokens > REF_MAX_CONTEXT || prompt->count < 2u )
		Die("a prompt does not fit the reference",prompt->name);
	for ( variant = 0u; variant < 2u; variant++ )
	{
		const double start = Seconds();
		FILE *file;
		uint32_t index;
		RefStateReset(state);
		RefForward(variant,model,state,prompt->tokens,prompt->count,logits);
		snprintf(path,sizeof(path),"%s/%s.ref_%c.logits",out,prompt->name,variant == 0u ? 'a' : 'b');
		file = fopen(path,"wb");
		if ( file == 0 || fwrite(logits,sizeof(float),(size_t)(prompt->count - 1u) * REF_VOCAB,file) != (size_t)(prompt->count - 1u) * REF_VOCAB ||
			fclose(file) != 0 )
			Die("cannot write",path);
		generated[0] = RefArgmax(logits + (uint64_t)(prompt->count - 1u) * REF_VOCAB);
		fprintf(stderr,"%s variant %c: %u teacher-forced positions in %.1f s\n",prompt->name,variant == 0u ? 'a' : 'b',prompt->count - 1u,Seconds() - start);
		fprintf(greedy,"%s{\"name\":\"%s\",\"variant\":\"%c\",\"prompt_tokens\":%u,\"generated\":[",first != 0u && variant == 0u ? "" : ",\n",prompt->name,
			variant == 0u ? 'a' : 'b',prompt->count);
		for ( index = 0u; index < new_tokens; index++ )
		{
			fprintf(greedy,"%s%u",index != 0u ? "," : "",generated[index]);
			RefForward(variant,model,state,&generated[index],1u,logits);
			generated[index + 1u] = RefArgmax(logits);
		}
		fprintf(greedy,"],\"seconds\":%.1f}",Seconds() - start);
		fflush(greedy);
		fprintf(stderr,"%s variant %c: %u greedy tokens, %.1f s total\n",prompt->name,variant == 0u ? 'a' : 'b',new_tokens,Seconds() - start);
		(void)RefLogprob;
	}
	free(logits);
	free(generated);
}

int main(int argc, char **argv)
{
	RefCheckpoint checkpoint;
	RefModel model;
	RefState state;
	SparkTokenPrompts prompts;
	char path[2048];
	FILE *greedy;
	uint32_t index, new_tokens;
	double start = Seconds();
	if ( argc != 5 && !(argc == 6 && strcmp(argv[5],"fp32-activations") == 0) )
	{
		fprintf(stderr,"usage: %s CHECKPOINT_DIR PROMPTS.json OUT_DIR NEW_TOKENS [fp32-activations]\n",argv[0]);
		return(2);
	}
	memset(&checkpoint,0,sizeof(checkpoint));
	snprintf(checkpoint.root,sizeof(checkpoint.root),"%s",argv[1]);
	snprintf(path,sizeof(path),"%s/model.safetensors.index.json",argv[1]);
	SparkJsonDocumentReset(&checkpoint.index);
	if ( SparkJsonLoadFile(path,&checkpoint.index) != SPARK_STATUS_OK )
		Die("cannot read the checkpoint index",path);
	if ( SparkTokenPromptsLoad(argv[2],&prompts) != SPARK_STATUS_OK )
		return(1);
	new_tokens = (uint32_t)strtoul(argv[4],0,10);
	mkdir(argv[3],0755);
	RefLoadModel(&checkpoint,&model);
	RefStateCreate(&state);
	fprintf(stderr,"weights converted to f32 in %.1f s\n",Seconds() - start);
	if ( argc == 6 )
	{
		float *logits = (float *)malloc((size_t)REF_MAX_CONTEXT * REF_VOCAB * sizeof(float));
		ref_round_activations = 0u;
		for ( index = 0u; logits != 0 && index < prompts.count; index++ )
		{
			const SparkTokenPrompt *prompt = &prompts.prompts[index];
			FILE *file;
			RefStateReset(&state);
			RefForward(0u,&model,&state,prompt->tokens,prompt->count,logits);
			snprintf(path,sizeof(path),"%s/%s.ref_c.logits",argv[3],prompt->name);
			file = fopen(path,"wb");
			if ( file == 0 || fwrite(logits,sizeof(float),(size_t)(prompt->count - 1u) * REF_VOCAB,file) != (size_t)(prompt->count - 1u) * REF_VOCAB ||
				fclose(file) != 0 )
				Die("cannot write",path);
			fprintf(stderr,"%s: FP32-activation forward written\n",prompt->name);
		}
		free(logits);
		return(0);
	}
	snprintf(path,sizeof(path),"%s/greedy.ref.json",argv[3]);
	greedy = fopen(path,"w");
	if ( greedy == 0 )
		Die("cannot write",path);
	fprintf(greedy,"{\"new_tokens\":%u,\"split_k\":%u,\"prompts\":[\n",new_tokens,REF_SPLIT);
	for ( index = 0u; index < prompts.count; index++ )
		RefPrompt(&model,&state,&prompts.prompts[index],argv[3],new_tokens,greedy,index == 0u);
	fprintf(greedy,"\n]}\n");
	fclose(greedy);
	SparkTokenPromptsFree(&prompts);
	fprintf(stderr,"reference done in %.1f s\n",Seconds() - start);
	return(0);
}
