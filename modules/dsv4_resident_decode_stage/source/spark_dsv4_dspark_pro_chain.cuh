/*
 * DSV4 Pro DSpark draft chain - GA 0813 (mtp.0/1/2, block 5, markov 512).
 *
 * This is the module-side sequence that runs AFTER the final main-layer head
 * emission on decode frames. The draft runs REPLICATED full-width on every
 * rank (tools/dsv4_tp16_stagepack.py: MTP rows replicate, zero draft
 * collectives), so the attn-side / ffn-side "reduce" steps below are LOCAL
 * (identity) - there is no cross-rank collective on the draft path.
 *
 * Launcher surface (landed in spark_dsv4_dspark_pro_kernels.cuh + reused
 * Flash launchers):
 *   SparkDsv4DSparkLaunchMeanReduction   (3-tap hc mean capture)
 *   SparkDsv4DSparkLaunchMainKvWrite     (rolling 128x512 main-KV window)
 *   SparkDsv4DSparkLaunchConfidence      (7680->1 sigmoid)
 *   SparkDsv4LaunchDsparkAttention       (reused Flash; Pro shapes)
 *   SparkDsv4LaunchDsparkMarkovBiasAccum (reused Flash; markov 512)
 *   SparkDsv4LaunchDsparkArgmax          (reused Flash; vocab shard)
 *
 * The main_proj / main_norm / wkv / kv_norm / gate-route / routed-expert /
 * shared / hc machinery are the EXISTING main-model launches (reused, no new
 * math) - the draft mHC block mirrors SparkDsv4ModuleContinueLayers on
 * state->mtp_layers[i].
 */

#if defined(SPARK_DSV4_MODEL_BUILD) && (SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0u)

/*
 * Per-slot draft forward. state is the module state; slot is the resident
 * slot; the buffers are the per-slot dspark_* allocations from the module
 * (dspark_capture_bf16[3], dspark_main_bf16, dspark_main_kv_bf16[3],
 * dspark_draft_q/kv/o, dspark_logits_f32, dspark_draft_ids).
 *
 * Sequence (dsv4_pro_ga_migration.md:126-163):
 *   1. capture taps 58-60 hc means -> main_proj(concat)->main_norm -> main_bf16
 *   2. draft ids [accepted, noise x4] -> embed -> 5x7168 -> 4 hc streams
 *   3. per draft layer i in {0,1,2}: main_kv write; draft attn; FFN(top-6 mtp)
 *   4. mtp.2 hc_head -> norm -> shared screened argmax -> markov bias -> confidence
 * Acceptance/verification stays client-side (compare drafts vs the main model).
 */
static SparkStatus SparkDsv4ModuleRunDsparkDraft(
	SparkDsv4ModuleState *state,uint32_t slot,uint32_t draft_count)
{
	const uint32_t tap_count = SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT; /* 3 */
	const uint32_t hidden = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;            /* 7168 */
	const uint32_t kv_dim = SPARK_DSPARK_DRAFT_HEAD_DIMENSION;           /* 512 */
	const uint32_t window_tokens = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;/* 128 */
	uint32_t layer;
	(void)slot;
	(void)draft_count;

	/* 1. tap capture: mean over the 4 hc streams for layers 58,59,60. */
	SparkDsv4DSparkLaunchMeanReduction(state->stream,
		state->dspark_tap_bf16 /* [3][4][7168] post-layer hc outputs */,
		state->dspark_capture_bf16 /* [3][7168] */,
		tap_count, SPARK_DSV4_MODEL_HC_STREAM_COUNT, hidden,
		state->multiprocessor_count);

	/* main_proj: dense FP8 linear (3*7168 -> 7168) over concat(captures),
	 * then RMS main_norm -> dspark_main_bf16. REUSE the existing FP8 linear +
	 * RMS-norm launches on state->mtp.main_proj / main_norm_weight_bf16. */
	SparkDsv4ModuleLaunchDenseFp8Linear(state->stream,
		&state->mtp.main_proj, state->dspark_capture_bf16,
		state->dspark_main_bf16, 1u /* rows */);
	SparkDsv4ModuleLaunchRmsNorm(state->stream,
		state->mtp.main_norm_weight_bf16, state->dspark_main_bf16,
		state->dspark_main_bf16, 1u, hidden);

	/* 2. draft ids [accepted, noise x4] -> embedding gather -> 5 x 7168,
	 *    replicate to 4 hc streams. REUSE the existing embedding gather +
	 *    SparkDsv4DsparkExpandStreams (Flash) for the stream expansion. */
	SparkDsv4ModuleLaunchEmbeddingGather(state->stream,
		state->dspark_draft_ids /* [5] */, state->dspark_x_bf16 /* [5][7168] */,
		draft_count);
	SparkDsv4LaunchExpandStreams(state->stream,
		state->dspark_x_bf16, state->dspark_draft_streams /* [5][4][7168] */,
		draft_count, SPARK_DSV4_MODEL_HC_STREAM_COUNT, hidden,
		state->multiprocessor_count);

	/* 3. the three draft layers (mtp.0/1/2), mirroring the main layer loop. */
	for (layer = 0u; layer < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; layer++)
	{
		/* a. main-stream KV: kv_norm(wkv(main_bf16)) -> rolling window slot. */
		SparkDsv4DSparkLaunchMainKvWrite(state->stream,
			state->dspark_main_kv_bf16 /* [3][128][512] per-layer window */,
			state->dspark_main_kv_bf16 /* [3][128][512] per-layer window */,
			kv_dim, window_tokens,
			state->dspark_lane_position[slot] % window_tokens);

		/* b. draft attention over main-KV window + causal draft KV. Reuse the
		 *    Flash SparkDsv4LaunchDsparkAttention with the Pro shapes
		 *    (head_count = pinned draft heads, head_dim = 512, window 128,
		 *    block 5). attn-side reduce is LOCAL (replicated draft). */
		SparkDsv4LaunchDsparkAttention(state->stream,
			state->dspark_q_attn_bf16, state->dspark_main_kv_bf16,
			0u /* lane_stride: single lane */, slot,
			state->dspark_draft_kv_bf16, state->mtp_layers[layer].attn_sink_f32,
			1.0f / sqrtf((float)kv_dim), state->dspark_o_ranks_bf16,
			draft_count, SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT, kv_dim,
			window_tokens);

		/* c. FFN: bias-gate top-6 routed mtp experts + shared, via the existing
		 *    continuation machinery on state->mtp_layers[layer]. ffn-side reduce
		 *    is LOCAL (replicated draft). */
		SparkDsv4ModuleContinueDsparkLayer(state, slot, layer);
	}

	/* 4. draft head: mtp.2 hc_head -> norm -> shared screened argmax (5 rows,
	 *    vocab-sharded) -> U64 maxloc -> 5 draft tokens; then markov bias +
	 *    confidence. Reuse the Flash MarkovBiasAccum + Argmax + Confidence. */
	SparkDsv4LaunchDsparkMarkovBiasAccum(state->stream,
		state->dspark_logits_bf16, state->mtp.markov_w2.payload,
		state->dspark_markov_embed_bf16, state->dspark_logits_f32,
		state->vocab_offset, state->vocab_shard_count,
		SPARK_DSPARK_MARKOV_RANK, 0u /* position */, state->multiprocessor_count);
	SparkDsv4LaunchDsparkArgmax(state->stream, state->dspark_logits_f32,
		state->vocab_shard_count, state->vocab_offset,
		state->dspark_draft_token_ids, state->dspark_scores_f32);
	SparkDsv4DSparkLaunchConfidence(state->stream,
		state->dspark_confidence_features /* [7680] = hidden|markov_embed */,
		state->mtp.confidence_proj.payload, 0.0f /* + bias from mtp */,
		state->dspark_confidence_f32, draft_count);

	/* The module only emits drafts + confidence; the acceptance/verification
	 * policy (compare draft logits vs the main model per position) is
	 * client-side. */
	return(SPARK_STATUS_OK);
}

#endif /* SPARK_DSV4_MODEL_BUILD && SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0 */
