
#if defined(SPARK_DSV4_MODEL_BUILD) && (SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0u)

static SparkStatus SparkDsv4ModuleRunDsparkDraft(
	SparkDsv4ModuleState *state,uint32_t slot,uint32_t draft_count)
{
	const uint32_t tap_count = SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT;
	const uint32_t hidden = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	const uint32_t kv_dim = SPARK_DSPARK_DRAFT_HEAD_DIMENSION;
	const uint32_t window_tokens = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
	uint32_t layer;
	(void)slot;
	(void)draft_count;

	SparkDsv4DSparkLaunchMeanReduction(state->stream,
		state->dspark_tap_bf16   ,
		state->dspark_capture_bf16   ,
		tap_count, SPARK_DSV4_MODEL_HC_STREAM_COUNT, hidden,
		state->multiprocessor_count);

	SparkDsv4ModuleLaunchDenseFp8Linear(state->stream,
		&state->mtp.main_proj, state->dspark_capture_bf16,
		state->dspark_main_bf16, 1u   );
	SparkDsv4ModuleLaunchRmsNorm(state->stream,
		state->mtp.main_norm_weight_bf16, state->dspark_main_bf16,
		state->dspark_main_bf16, 1u, hidden);

	SparkDsv4ModuleLaunchEmbeddingGather(state->stream,
		state->dspark_draft_ids   , state->dspark_x_bf16   ,
		draft_count);
	SparkDsv4LaunchExpandStreams(state->stream,
		state->dspark_x_bf16, state->dspark_draft_streams   ,
		draft_count, SPARK_DSV4_MODEL_HC_STREAM_COUNT, hidden,
		state->multiprocessor_count);

	for (layer = 0u; layer < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; layer++)
	{
		SparkDsv4DSparkLaunchMainKvWrite(state->stream,
			state->dspark_main_kv_bf16   ,
			state->dspark_main_kv_bf16   ,
			kv_dim, window_tokens,
			state->dspark_lane_position[slot] % window_tokens);

		SparkDsv4LaunchDsparkAttention(state->stream,
			state->dspark_q_attn_bf16, state->dspark_main_kv_bf16,
			0u   , slot,
			state->dspark_draft_kv_bf16, state->mtp_layers[layer].attn_sink_f32,
			1.0f / sqrtf((float)kv_dim), state->dspark_o_ranks_bf16,
			draft_count, SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT, kv_dim,
			window_tokens);

		SparkDsv4ModuleContinueDsparkLayer(state, slot, layer);
	}

	SparkDsv4LaunchDsparkMarkovBiasAccum(state->stream,
		state->dspark_logits_bf16, state->mtp.markov_w2.payload,
		state->dspark_markov_embed_bf16, state->dspark_logits_f32,
		state->vocab_offset, state->vocab_shard_count,
		SPARK_DSPARK_MARKOV_RANK, 0u   , state->multiprocessor_count);
	SparkDsv4LaunchDsparkArgmax(state->stream, state->dspark_logits_f32,
		state->vocab_shard_count, state->vocab_offset,
		state->dspark_draft_token_ids, state->dspark_scores_f32);
	SparkDsv4DSparkLaunchConfidence(state->stream,
		state->dspark_confidence_features   ,
		state->mtp.confidence_proj.payload, 0.0f   ,
		state->dspark_confidence_f32, draft_count);

	return(SPARK_STATUS_OK);
}

#endif
