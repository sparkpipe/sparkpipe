#pragma once

typedef struct SPARK_FAMILY(ValDevice)
{
	SPARK_FAMILY(GdnLayerWeights) gdn_weights;
	SPARK_FAMILY(AttnLayerWeights) attn_weights;
	SPARK_FAMILY(GdnStatePool) pool;
	uint16_t *conv_weight;
	float *a_log;
	float *dt_bias;
	uint16_t *gdn_norm_weight;
	uint16_t *q_norm_weight;
	uint16_t *k_norm_weight;
	float *state;
	uint16_t *conv_tail;
	uint32_t *cold;
	uint32_t *lane_indices;
	uint16_t *qkv;
	uint16_t *conv_out;
	uint16_t *core_out;
	uint16_t *z_bf16;
	uint16_t *gated_out;
	uint16_t *ba_bf16;
	float *log_decay;
	float *beta;
	float *chunk_qn;
	float *chunk_kn;
	float *chunk_cum_g;
	float *chunk_decay;
	float *chunk_attn;
	float *chunk_w;
	float *chunk_kg;
} SPARK_FAMILY(ValDevice);

static int SPARK_FAMILY(ValDeviceSetup)(SPARK_FAMILY(ValDevice) *device)
{
	uint64_t state_elements = 2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION);
	uint64_t vector_floats = (uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION);
	uint64_t matrix_floats = (uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS) * SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS);
	uint32_t tokens = SPARK_FAMILY_CONST(VALIDATION_CHUNK_TOKENS);
	cudaError_t error;
	memset(device,0,sizeof(*device));
	error = cudaMalloc((void **)&device->conv_weight,(uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->a_log,SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->dt_bias,SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->gdn_norm_weight,SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->q_norm_weight,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->k_norm_weight,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->state,state_elements * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->conv_tail,2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * (SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL) - 1u) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->cold,2 * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->lane_indices,2 * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->qkv,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->conv_out,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->core_out,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->z_bf16,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->gated_out,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->ba_bf16,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->log_decay,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->beta,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_qn,vector_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_kn,vector_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_cum_g,(uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS) * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_decay,matrix_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_attn,matrix_floats * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_w,(uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION) * sizeof(float));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device->chunk_kg,vector_floats * sizeof(float));
	if (error != cudaSuccess)
		return(SPARK_FAMILY(ValCuda)(error,"device_alloc"));
	device->pool.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION);
	device->pool.lane_capacity = 2u;
	device->pool.gdn_layer_count = 1u;
	device->pool.state_f32 = device->state;
	device->pool.state_layer_stride_elements = (uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION);
	device->pool.state_lane_stride_elements = device->pool.state_layer_stride_elements;
	device->pool.conv_tail_bf16 = device->conv_tail;
	device->pool.conv_tail_layer_stride_elements = (uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * (SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL) - 1u);
	device->pool.conv_tail_lane_stride_elements = device->pool.conv_tail_layer_stride_elements;
	device->pool.state_cold_by_row = device->cold;
	device->gdn_weights.conv_weight_bf16 = device->conv_weight;
	device->gdn_weights.a_log_f32 = device->a_log;
	device->gdn_weights.dt_bias_f32 = device->dt_bias;
	device->gdn_weights.gdn_norm_weight_bf16 = device->gdn_norm_weight;
	device->attn_weights.query_norm_weight_bf16 = device->q_norm_weight;
	device->attn_weights.key_norm_weight_bf16 = device->k_norm_weight;
	return(0);
}
