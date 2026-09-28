#pragma once

static int SPARK_FAMILY(ValCheckDecayBeta)(SPARK_FAMILY(ValDevice) *device)
{
	uint16_t host_ba[SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * 2u];
	float host_a[SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)],host_bias[SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)];
	float host_decay[SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)];
	float host_beta[SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)];
	float ref_decay[SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)];
	float ref_beta[SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)];
	SPARK_FAMILY(ValMetrics) metrics;
	uint32_t row,head;
	uint64_t index;
	float shifted;
	cudaError_t error;
	SPARK_FAMILY(ValRandomState) = 11u;
	SPARK_FAMILY(ValFillBf16)(host_ba,0,SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * 2u,2.0f);
	for (head = 0u; head < SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head++)
	{
		host_a[head] = SPARK_FAMILY(ValUniform)(1.0f);
		host_bias[head] = SPARK_FAMILY(ValUniform)(1.0f);
	}
	error = cudaMemcpy(device->ba_bf16,host_ba,sizeof(host_ba),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->a_log,host_a,sizeof(host_a),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->dt_bias,host_bias,sizeof(host_bias),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SPARK_FAMILY_CONST(VALIDATION_GDN_LAUNCH)(SPARK_FAMILY(LaunchDecayBeta),cudaStreamPerThread,device->ba_bf16,device->ba_bf16 + SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT),&device->gdn_weights,device->log_decay,device->beta,SPARK_FAMILY_CONST(VALIDATION_ROWS));
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(host_decay,device->log_decay,sizeof(host_decay),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(host_beta,device->beta,sizeof(host_beta),cudaMemcpyDeviceToHost);
	if (SPARK_FAMILY(ValCuda)(error,"decay_beta") != 0)
		return(1);
	for (row = 0u; row < SPARK_FAMILY_CONST(VALIDATION_ROWS); row++)
		for (head = 0u; head < SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head++)
		{
			index = ((uint64_t)row * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)) + head;
			shifted = SPARK_FAMILY(ValFromBf16)(host_ba[index]) + host_bias[head];
			ref_decay[index] = -expf(host_a[head]) * (shifted > 20.0f ? shifted : logf(1.0f + expf(shifted)));
			ref_beta[index] = 1.0f / (1.0f + expf(-SPARK_FAMILY(ValFromBf16)(host_ba[SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) + index])));
		}
	SPARK_FAMILY(ValMeasure)(&metrics,host_decay,ref_decay,SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT));
	if (SPARK_FAMILY(ValReport)("decay_gate",&metrics,1e-5,0.99999999) != 0)
		return(1);
	SPARK_FAMILY(ValMeasure)(&metrics,host_beta,ref_beta,SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT));
	return(SPARK_FAMILY(ValReport)("write_gate",&metrics,1e-5,0.99999999));
}

static int SPARK_FAMILY(ValCheckConv)(SPARK_FAMILY(ValDevice) *device)
{
	uint16_t *host_qkv,*host_out;
	float *exact,*expected,*actual,*tails;
	uint32_t token,row;
	uint32_t cold[2] = {1u,1u};
	uint32_t lanes[2] = {0u,1u};
	const uint64_t frame_elements = (uint64_t)SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS);
	uint64_t channel;
	cudaError_t error;
	SPARK_FAMILY(ValMetrics) metrics;
	host_qkv = (uint16_t *)calloc(frame_elements,sizeof(uint16_t));
	host_out = (uint16_t *)calloc(frame_elements,sizeof(uint16_t));
	exact = (float *)calloc(frame_elements,sizeof(float));
	expected = (float *)calloc(frame_elements,sizeof(float));
	actual = (float *)calloc(frame_elements,sizeof(float));
	tails = (float *)calloc(frame_elements * 3u,sizeof(float));
	if (host_qkv == 0 || host_out == 0 || exact == 0 || expected == 0 || actual == 0 || tails == 0)
		return(SPARK_FAMILY(ValFail)("conv_update","host_alloc"));
	SPARK_FAMILY(ValRandomState) = 23u;
	SPARK_FAMILY(ValFillBf16)(host_qkv,exact,frame_elements,1.0f);
	{
		uint16_t *weight_packed = (uint16_t *)calloc((uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL),sizeof(uint16_t));
		float *weight_exact = (float *)calloc((uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL),sizeof(float));
		if (weight_packed == 0 || weight_exact == 0)
			return(SPARK_FAMILY(ValFail)("conv_update","weight_alloc"));
		SPARK_FAMILY(ValFillBf16)(weight_packed,weight_exact,(uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL),0.5f);
		error = cudaMemcpy(device->conv_weight,weight_packed,(uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * SPARK_FAMILY_CONST(MODEL_GDN_CONV_KERNEL) * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->qkv,host_qkv,frame_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->lane_indices,lanes,sizeof(lanes),cudaMemcpyHostToDevice);
		for (token = 0u; token < 2u && error == cudaSuccess; token++)
		{
			if (token != 0u)
			{
				cold[0] = 0u; cold[1] = 0u;
				error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
			}
			if (error == cudaSuccess)
				error = SPARK_FAMILY_CONST(VALIDATION_GDN_LAUNCH)(SPARK_FAMILY(LaunchConvUpdate),cudaStreamPerThread,device->qkv + (uint64_t)token * 2u * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS),&device->gdn_weights,device->conv_out + (uint64_t)token * 2u * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS),&device->pool,device->lane_indices,2u,0u);
			if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
		}
		if (error == cudaSuccess)
			error = cudaMemcpy(host_out,device->conv_out,frame_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (SPARK_FAMILY(ValCuda)(error,"conv_update") != 0)
			return(1);
		for (row = 0u; row < 2u; row++)
			for (channel = 0u; channel < SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS); channel++)
			{
				float input[2],output[2],tail[3];
				uint64_t base = ((uint64_t)row * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS)) + channel;
				input[0] = exact[base];
				input[1] = exact[2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) + base];
				tail[0] = tail[1] = tail[2] = 0.0f;
				{
					uint32_t t,token_index;
					for (token_index = 0u; token_index < 2u; token_index++)
					{
						float window[4],accumulator = 0.0f;
						window[0] = token_index >= 3u ? input[token_index - 3u] : tail[token_index];
						window[1] = token_index >= 2u ? input[token_index - 2u] : tail[token_index + 1u];
						window[2] = token_index >= 1u ? input[token_index - 1u] : tail[token_index + 2u];
						window[3] = input[token_index];
						for (t = 0u; t < 4u; t++)
							accumulator += window[t] * weight_exact[(channel * 4u) + t];
						output[token_index] = SPARK_FAMILY(ValSilu)(accumulator);
					}
					tail[2] = input[1];
					tail[1] = input[0];
					tail[0] = 0.0f;
					(void)tail;
				}
				expected[(uint64_t)0u * 2u * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) + base] = output[0];
				expected[(uint64_t)1u * 2u * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) + base] = output[1];
			}
		for (token = 0u; token < 2u; token++)
			for (uint64_t i = 0u; i < 2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS); i++)
				actual[(uint64_t)token * 2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) + i] =
					SPARK_FAMILY(ValFromBf16)(host_out[(uint64_t)token * 2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) + i]);
		free(weight_packed);
		free(weight_exact);
	}
	SPARK_FAMILY(ValMeasure)(&metrics,actual,expected,frame_elements);
	free(host_qkv); free(host_out); free(exact); free(expected); free(actual); free(tails);
	return(SPARK_FAMILY(ValReport)("conv_update",&metrics,5e-3,0.99999));
}

static int SPARK_FAMILY(ValCheckGdnStep)(SPARK_FAMILY(ValDevice) *device)
{
	uint64_t state_elements = (uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION);
	uint16_t *host_conv = (uint16_t *)calloc(2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS),sizeof(uint16_t));
	float *exact = (float *)calloc(2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS),sizeof(float));
	float *state_host = (float *)calloc(2ull * state_elements,sizeof(float));
	float *state_reference = (float *)calloc(2ull * state_elements,sizeof(float));
	float *oracle_out = (float *)calloc(2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),sizeof(float));
	float *actual = (float *)calloc(2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),sizeof(float));
	uint16_t *core_packed = (uint16_t *)calloc(2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),sizeof(uint16_t));
	uint32_t lanes[2] = {0u,1u};
	uint32_t cold[2] = {1u,0u};
	SPARK_FAMILY(ValMetrics) metrics;
	uint32_t row,head;
	cudaError_t error;
	if (host_conv == 0 || exact == 0 || state_host == 0 || state_reference == 0 || oracle_out == 0 || actual == 0 || core_packed == 0)
		return(SPARK_FAMILY(ValFail)("gdn_step","host_alloc"));
	SPARK_FAMILY(ValRandomState) = 37u;
	SPARK_FAMILY(ValFillBf16)(host_conv,exact,2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS),1.0f);
	{
		uint64_t index;
		for (index = 0u; index < state_elements; index++)
			state_host[state_elements + index] = SPARK_FAMILY(ValUniform)(0.25f);
		memcpy(state_reference,state_host,2ull * state_elements * sizeof(float));
	}
	{
		float host_log_decay[2 * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)],host_beta[2 * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)];
		uint32_t head_index;
		for (head_index = 0u; head_index < 2u * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head_index++)
		{
			host_log_decay[head_index] = SPARK_FAMILY(ValUniform)(0.5f) - 0.5f;
			host_beta[head_index] = 0.25f + fabsf(SPARK_FAMILY(ValUniform)(0.5f));
		}
		error = cudaMemcpy(device->state,state_host,2ull * state_elements * sizeof(float),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->qkv,host_conv,2ull * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS) * sizeof(uint16_t),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->log_decay,host_log_decay,sizeof(host_log_decay),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->beta,host_beta,sizeof(host_beta),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->cold,cold,sizeof(cold),cudaMemcpyHostToDevice);
		if (error == cudaSuccess) error = cudaMemcpy(device->lane_indices,lanes,sizeof(lanes),cudaMemcpyHostToDevice);
		if (error == cudaSuccess)
			error = SPARK_FAMILY_CONST(VALIDATION_GDN_LAUNCH)(SPARK_FAMILY(LaunchGdnStep),cudaStreamPerThread,device->qkv,device->log_decay,device->beta,&device->pool,device->core_out,device->lane_indices,2u,0u);
		if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
		if (error == cudaSuccess) error = cudaMemcpy(core_packed,device->core_out,2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION) * sizeof(uint16_t),cudaMemcpyDeviceToHost);
		if (error == cudaSuccess) error = cudaMemcpy(state_host,device->state,2ull * state_elements * sizeof(float),cudaMemcpyDeviceToHost);
		if (SPARK_FAMILY(ValCuda)(error,"gdn_step") != 0)
			return(1);
		for (row = 0u; row < 2u; row++)
			for (head = 0u; head < SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head++)
			{
				uint32_t key_head = head / SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD);
				const float *q = exact + ((uint64_t)row * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS)) + ((uint64_t)key_head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION));
				const float *k = q + SPARK_FAMILY_CONST(MODEL_GDN_QK_DIMENSION);
				const float *v = exact + ((uint64_t)row * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS)) + (2ull * SPARK_FAMILY_CONST(MODEL_GDN_QK_DIMENSION)) + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION));
				float g = cold[row] != 0u ? -30.0f : host_log_decay[(row * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)) + head];
				float beta = host_beta[(row * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)) + head];
				SPARK_FAMILY(ValGdnRecurrence)(q,k,v,&g,&beta,
					state_reference + ((uint64_t)row * state_elements) + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),
					oracle_out + ((uint64_t)row * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)) + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),1u);
			}
	}
	{
		uint64_t index;
		for (index = 0u; index < 2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION); index++)
			actual[index] = SPARK_FAMILY(ValFromBf16)(core_packed[index]);
	}
	SPARK_FAMILY(ValMeasure)(&metrics,actual,oracle_out,2ull * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION));
	if (SPARK_FAMILY(ValReport)("gdn_step_output",&metrics,5e-3,0.99999) != 0)
		return(1);
	SPARK_FAMILY(ValMeasure)(&metrics,state_host,state_reference,2ull * state_elements);
	free(host_conv); free(exact); free(state_host); free(state_reference);
	free(oracle_out); free(actual); free(core_packed);
	return(SPARK_FAMILY(ValReport)("gdn_step_state",&metrics,1e-3,0.999999));
}

static int SPARK_FAMILY(ValCheckGatedNorm)(SPARK_FAMILY(ValDevice) *device)
{
	uint64_t elements = (uint64_t)SPARK_FAMILY_CONST(VALIDATION_ROWS) * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION);
	uint16_t *packed = (uint16_t *)calloc(elements * sizeof(uint16_t),sizeof(uint16_t));
	float *exact = (float *)calloc(elements * sizeof(uint16_t),sizeof(float));
	float *expected = (float *)calloc(elements,sizeof(float));
	float *actual = (float *)calloc(elements,sizeof(float));
	uint16_t *norm_packed = (uint16_t *)calloc(SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),sizeof(uint16_t));
	float *norm_exact = (float *)calloc(SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),sizeof(float));
	SPARK_FAMILY(ValMetrics) metrics;
	uint32_t row,head,element;
	cudaError_t error;
	if (packed == 0 || exact == 0 || expected == 0 || actual == 0 || norm_packed == 0 || norm_exact == 0)
		return(SPARK_FAMILY(ValFail)("gated_norm","host_alloc"));
	SPARK_FAMILY(ValRandomState) = 51u;
	SPARK_FAMILY(ValFillBf16)(packed,exact,elements,1.0f);
	SPARK_FAMILY(ValFillBf16)(packed + elements,exact + elements,elements,1.0f);
	SPARK_FAMILY(ValFillBf16)(norm_packed,norm_exact,SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),0.5f);
	error = cudaMemcpy(device->core_out,packed,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->z_bf16,packed + elements,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->gdn_norm_weight,norm_packed,SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION) * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SPARK_FAMILY_CONST(VALIDATION_GDN_LAUNCH)(SPARK_FAMILY(LaunchGatedNorm),cudaStreamPerThread,device->core_out,device->z_bf16,&device->gdn_weights,device->gated_out,SPARK_FAMILY_CONST(VALIDATION_ROWS),SPARK_FAMILY_CONST(MODEL_RMS_NORM_EPSILON));
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(packed,device->gated_out,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SPARK_FAMILY(ValCuda)(error,"gated_norm") != 0)
		return(1);
	for (row = 0u; row < SPARK_FAMILY_CONST(VALIDATION_ROWS); row++)
		for (head = 0u; head < SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head++)
		{
			uint64_t base = ((uint64_t)row * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION)) + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION));
			float variance = 0.0f,inverse;
			for (element = 0u; element < SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION); element++)
				variance += exact[base + element] * exact[base + element];
			inverse = 1.0f / sqrtf((variance / (float)SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)) + SPARK_FAMILY_CONST(MODEL_RMS_NORM_EPSILON));
			for (element = 0u; element < SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION); element++)
				expected[base + element] = exact[base + element] * inverse * norm_exact[element] *
					SPARK_FAMILY(ValSilu)(exact[elements + base + element]);
		}
	for (row = 0u; row < elements; row++)
		actual[row] = SPARK_FAMILY(ValFromBf16)(packed[row]);
	SPARK_FAMILY(ValMeasure)(&metrics,actual,expected,elements);
	free(packed); free(exact); free(expected); free(actual); free(norm_packed); free(norm_exact);
	return(SPARK_FAMILY(ValReport)("gated_norm",&metrics,5e-3,0.99999));
}

static int SPARK_FAMILY(ValCheckAttention)(SPARK_FAMILY(ValDevice) *device)
{
	const uint32_t tokens = SPARK_FAMILY_CONST(VALIDATION_ATTN_TOKENS);
	const uint64_t fused_elements = (uint64_t)tokens * 2u * SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION);
	const uint64_t kv_elements = (uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_ATTN_KV_DIMENSION);
	const uint64_t cache_elements = (uint64_t)SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) * SPARK_FAMILY_CONST(MODEL_ATTN_CACHE_TOKEN_ELEMENTS);
	uint16_t *q_packed = (uint16_t *)calloc(fused_elements,sizeof(uint16_t));
	uint16_t *k_packed = (uint16_t *)calloc(kv_elements,sizeof(uint16_t));
	uint16_t *v_packed = (uint16_t *)calloc(kv_elements,sizeof(uint16_t));
	float *q_exact = (float *)calloc(fused_elements,sizeof(float));
	float *k_exact = (float *)calloc(kv_elements,sizeof(float));
	float *v_exact = (float *)calloc(kv_elements,sizeof(float));
	float *expected = (float *)calloc(SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION),sizeof(float));
	float *actual = (float *)calloc(SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION),sizeof(float));
	uint16_t *out_packed = (uint16_t *)calloc(SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION),sizeof(uint16_t));
	uint16_t *norm_packed = (uint16_t *)calloc(2u * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),sizeof(uint16_t));
	float *q_norm_exact = (float *)calloc(SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),sizeof(float));
	float *k_norm_exact = (float *)calloc(SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),sizeof(float));
	uint32_t slot_mapping[SPARK_FAMILY_CONST(VALIDATION_ATTN_TOKENS)];
	uint64_t positions[SPARK_FAMILY_CONST(VALIDATION_ATTN_TOKENS)];
	uint32_t block_indices[1] = {0u};
	uint32_t block_counts[1] = {1u};
	uint32_t lane_zero[1] = {0u};
	uint32_t context[1] = {SPARK_FAMILY_CONST(VALIDATION_ATTN_TOKENS)};
	uint32_t token;
	cudaError_t error;
	SPARK_FAMILY(ValMetrics) metrics;
	SPARK_FAMILY(KvBlockTableView) table;
	void *kv_cache = 0;
	uint16_t *device_q = 0,*device_k = 0,*device_v = 0,*device_out = 0;
	uint32_t *device_slots = 0,*device_blocks = 0,*device_counts = 0,*device_lane = 0,*device_context = 0;
	uint64_t *device_positions = 0;
	if (q_packed == 0 || k_packed == 0 || v_packed == 0 || q_exact == 0 || k_exact == 0 ||
		v_exact == 0 || expected == 0 || actual == 0 || out_packed == 0 || norm_packed == 0 ||
		q_norm_exact == 0 || k_norm_exact == 0)
		return(SPARK_FAMILY(ValFail)("attn_decode","host_alloc"));
	SPARK_FAMILY(ValRandomState) = 67u;
	SPARK_FAMILY(ValFillBf16)(q_packed,q_exact,fused_elements,0.5f);
	SPARK_FAMILY(ValFillBf16)(k_packed,k_exact,kv_elements,0.5f);
	SPARK_FAMILY(ValFillBf16)(v_packed,v_exact,kv_elements,0.5f);
	SPARK_FAMILY(ValFillBf16)(norm_packed,q_norm_exact,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),0.5f);
	SPARK_FAMILY(ValFillBf16)(norm_packed + SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),k_norm_exact,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),0.5f);
	for (token = 0u; token < tokens; token++)
	{
		slot_mapping[token] = token;
		positions[token] = token;
	}
	error = cudaMalloc(&kv_cache,cache_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMemset(kv_cache,0,cache_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_q,fused_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_k,kv_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_v,kv_elements * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_out,SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION) * sizeof(uint16_t));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_slots,sizeof(slot_mapping));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_positions,sizeof(positions));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_blocks,sizeof(block_indices));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_counts,sizeof(block_counts));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_lane,sizeof(lane_zero));
	if (error == cudaSuccess) error = cudaMalloc((void **)&device_context,sizeof(context));
	if (error == cudaSuccess) error = cudaMemcpy(device_q,q_packed,fused_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_k,k_packed,kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_v,v_packed,kv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_slots,slot_mapping,sizeof(slot_mapping),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_positions,positions,sizeof(positions),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_blocks,block_indices,sizeof(block_indices),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_counts,block_counts,sizeof(block_counts),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_lane,lane_zero,sizeof(lane_zero),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device_context,context,sizeof(context),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->q_norm_weight,norm_packed,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION) * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->k_norm_weight,norm_packed + SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION) * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess)
		error = SPARK_FAMILY_CONST(VALIDATION_ATTN_PREPARE_LAUNCH)(SPARK_FAMILY(LaunchAttnPrepare),cudaStreamPerThread,device_q,device_k,device_v,&device->attn_weights,kv_cache,device_slots,device_positions,tokens,0u,cache_elements,cache_elements,SPARK_FAMILY_CONST(MODEL_RMS_NORM_EPSILON));
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION);
	table.descriptor_bytes = sizeof(table);
	table.block_token_count = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	table.lane_count = 1u;
	table.lane_stride = 1u;
	table.lane_capacity = 1u;
	table.physical_block_indices = device_blocks;
	table.lane_physical_block_counts = device_counts;
	table.host_physical_block_indices = block_indices;
	table.host_lane_physical_block_counts = block_counts;
	if (error == cudaSuccess)
		error = SPARK_FAMILY_CONST(VALIDATION_ATTN_DECODE_LAUNCH)(SPARK_FAMILY(LaunchAttnDecode),cudaStreamPerThread,device_q + ((uint64_t)(tokens - 1u) * 2u * SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION)),kv_cache,&table,device_lane,device_context,device_out,1u,0u,cache_elements,cache_elements);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(out_packed,device_out,SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION) * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (SPARK_FAMILY(ValCuda)(error,"attn_decode") != 0)
		return(1);
	{
		float *k_cache = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),sizeof(float));
		float *v_cache = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),sizeof(float));
		uint32_t kv_head,element;
		if (k_cache == 0 || v_cache == 0)
			return(SPARK_FAMILY(ValFail)("attn_decode","oracle_alloc"));
		for (kv_head = 0u; kv_head < SPARK_FAMILY_CONST(MODEL_ATTN_KV_HEAD_COUNT); kv_head++)
		{
			for (token = 0u; token < tokens; token++)
			{
				const float *k_row = k_exact + (((uint64_t)token * SPARK_FAMILY_CONST(MODEL_ATTN_KV_DIMENSION)) + ((uint64_t)kv_head * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)));
				float *cache_row = k_cache + ((uint64_t)token * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION));
				SPARK_FAMILY(ValRmsNorm)(k_row,k_norm_exact,cache_row,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),SPARK_FAMILY_CONST(MODEL_RMS_NORM_EPSILON));
				SPARK_FAMILY(ValRope)(cache_row,SPARK_FAMILY_CONST(MODEL_ATTN_ROPE_DIMENSION),token,SPARK_FAMILY_CONST(MODEL_ATTN_ROPE_THETA));
				for (element = 0u; element < SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION); element++)
					v_cache[((uint64_t)token * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)) + element] =
						v_exact[((uint64_t)token * SPARK_FAMILY_CONST(MODEL_ATTN_KV_DIMENSION)) + ((uint64_t)kv_head * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)) + element];
			}
			SPARK_FAMILY(ValAttention)(
				q_exact + (((uint64_t)(tokens - 1u) * 2u * SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION)) + ((uint64_t)kv_head * SPARK_FAMILY_CONST(VALIDATION_ATTN_GROUP) * 2u * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION))),
				k_cache,v_cache,q_norm_exact,
				expected + ((uint64_t)kv_head * SPARK_FAMILY_CONST(VALIDATION_ATTN_GROUP) * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)),
				SPARK_FAMILY_CONST(VALIDATION_ATTN_GROUP),tokens,SPARK_FAMILY_CONST(MODEL_RMS_NORM_EPSILON));
		}
		free(k_cache);
		free(v_cache);
	}
	for (token = 0u; token < SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION); token++)
		actual[token] = SPARK_FAMILY(ValFromBf16)(out_packed[token]);
	SPARK_FAMILY(ValMeasure)(&metrics,actual,expected,SPARK_FAMILY_CONST(MODEL_ATTN_QUERY_DIMENSION));
	cudaFree(kv_cache); cudaFree(device_q); cudaFree(device_k); cudaFree(device_v);
	cudaFree(device_out); cudaFree(device_slots); cudaFree(device_positions);
	cudaFree(device_blocks); cudaFree(device_counts); cudaFree(device_lane); cudaFree(device_context);
	free(q_packed); free(k_packed); free(v_packed); free(q_exact); free(k_exact);
	free(v_exact); free(expected); free(actual); free(out_packed); free(norm_packed);
	free(q_norm_exact); free(k_norm_exact);
	return(SPARK_FAMILY(ValReport)("attn_decode",&metrics,5e-3,0.99999));
}

static int SPARK_FAMILY(ValCheckGdnChunk)(SPARK_FAMILY(ValDevice) *device)
{
	const uint32_t tokens = SPARK_FAMILY_CONST(VALIDATION_CHUNK_TOKENS);
	const uint64_t conv_elements = (uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS);
	const uint64_t out_elements = (uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION);
	const uint64_t state_elements = (uint64_t)SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION);
	uint16_t *conv_packed = (uint16_t *)calloc(conv_elements,sizeof(uint16_t));
	float *exact = (float *)calloc(conv_elements,sizeof(float));
	uint16_t *out_packed = (uint16_t *)calloc(out_elements,sizeof(uint16_t));
	float *actual = (float *)calloc(out_elements,sizeof(float));
	float *expected = (float *)calloc(out_elements,sizeof(float));
	float *state_device = (float *)calloc(state_elements,sizeof(float));
	float *state_oracle = (float *)calloc(state_elements,sizeof(float));
	float *log_decay = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT),sizeof(float));
	float *beta = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT),sizeof(float));
	uint32_t head,token;
	uint64_t index;
	cudaError_t error;
	SPARK_FAMILY(ValMetrics) metrics;
	if (conv_packed == 0 || exact == 0 || out_packed == 0 || actual == 0 || expected == 0 ||
		state_device == 0 || state_oracle == 0 || log_decay == 0 || beta == 0)
		return(SPARK_FAMILY(ValFail)("gdn_chunk","host_alloc"));
	SPARK_FAMILY(ValRandomState) = 83u;
	SPARK_FAMILY(ValFillBf16)(conv_packed,exact,conv_elements,1.0f);
	for (index = 0u; index < (uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); index++)
	{
		log_decay[index] = SPARK_FAMILY(ValUniform)(0.25f) - 0.25f;
		beta[index] = 0.25f + fabsf(SPARK_FAMILY(ValUniform)(0.5f));
	}
	error = cudaMemset(device->state,0,state_elements * sizeof(float));
	if (error == cudaSuccess) error = cudaMemcpy(device->qkv,conv_packed,conv_elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->log_decay,log_decay,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(float),cudaMemcpyHostToDevice);
	if (error == cudaSuccess) error = cudaMemcpy(device->beta,beta,(uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT) * sizeof(float),cudaMemcpyHostToDevice);
	for (uint32_t base = 0u; base < tokens && error == cudaSuccess; base += SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS))
		error = SPARK_FAMILY_CONST(VALIDATION_GDN_LAUNCH)(SPARK_FAMILY(LaunchGdnChunk),cudaStreamPerThread,
			device->qkv + ((uint64_t)base * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS)),
			device->log_decay + ((uint64_t)base * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)),
			device->beta + ((uint64_t)base * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)),
			device->chunk_qn,device->chunk_kn,device->chunk_cum_g,device->chunk_decay,
			device->chunk_attn,device->chunk_w,device->chunk_kg,
			&device->pool,device->core_out + ((uint64_t)base * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION)),
			0u,SPARK_FAMILY_CONST(MODEL_GDN_CHUNK_TOKENS),0u);
	if (error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
	if (error == cudaSuccess) error = cudaMemcpy(out_packed,device->core_out,out_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if (error == cudaSuccess) error = cudaMemcpy(state_device,device->state,state_elements * sizeof(float),cudaMemcpyDeviceToHost);
	if (SPARK_FAMILY(ValCuda)(error,"gdn_chunk") != 0)
		return(1);
	for (head = 0u; head < SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head++)
	{
		uint32_t key_head = head / SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEADS_PER_KEY_HEAD);
		float *q_head = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION),sizeof(float));
		float *k_head = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION),sizeof(float));
		float *v_head = (float *)calloc((uint64_t)tokens * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION),sizeof(float));
		float *g_head = (float *)calloc(tokens,sizeof(float));
		float *b_head = (float *)calloc(tokens,sizeof(float));
		if (q_head == 0 || k_head == 0 || v_head == 0 || g_head == 0 || b_head == 0)
			return(SPARK_FAMILY(ValFail)("gdn_chunk","oracle_alloc"));
		for (token = 0u; token < tokens; token++)
		{
			uint64_t base = (uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_CONV_CHANNELS);
			memcpy(q_head + ((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION)),exact + base + ((uint64_t)key_head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION)),SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * sizeof(float));
			memcpy(k_head + ((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION)),exact + base + SPARK_FAMILY_CONST(MODEL_GDN_QK_DIMENSION) + ((uint64_t)key_head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION)),SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * sizeof(float));
			memcpy(v_head + ((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),exact + base + (2ull * SPARK_FAMILY_CONST(MODEL_GDN_QK_DIMENSION)) + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION) * sizeof(float));
			g_head[token] = log_decay[((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)) + head];
			b_head[token] = beta[((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT)) + head];
		}
		SPARK_FAMILY(ValGdnRecurrence)(q_head,k_head,v_head,g_head,b_head,
			state_oracle + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_KEY_DIMENSION) * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),
			expected + ((uint64_t)head * tokens * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),tokens);
		free(q_head); free(k_head); free(v_head); free(g_head); free(b_head);
	}
	{
		float *expected_ordered = (float *)calloc(out_elements,sizeof(float));
		if (expected_ordered == 0)
			return(SPARK_FAMILY(ValFail)("gdn_chunk","order_alloc"));
		for (token = 0u; token < tokens; token++)
			for (head = 0u; head < SPARK_FAMILY_CONST(MODEL_GDN_VALUE_HEAD_COUNT); head++)
				memcpy(expected_ordered + ((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_VALUE_DIMENSION)) + ((uint64_t)head * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),
					expected + ((uint64_t)head * tokens * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)) + ((uint64_t)token * SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION)),
					SPARK_FAMILY_CONST(MODEL_GDN_HEAD_VALUE_DIMENSION) * sizeof(float));
		for (index = 0u; index < out_elements; index++)
			actual[index] = SPARK_FAMILY(ValFromBf16)(out_packed[index]);
		SPARK_FAMILY(ValMeasure)(&metrics,actual,expected_ordered,out_elements);
		free(expected_ordered);
	}
	if (SPARK_FAMILY(ValReport)("gdn_chunk_output",&metrics,2e-2,0.999) != 0)
		return(1);
	SPARK_FAMILY(ValMeasure)(&metrics,state_device,state_oracle,state_elements);
	free(conv_packed); free(exact); free(out_packed); free(actual); free(expected);
	free(state_device); free(state_oracle); free(log_decay); free(beta);
	return(SPARK_FAMILY(ValReport)("gdn_chunk_state",&metrics,2e-2,0.999));
}
