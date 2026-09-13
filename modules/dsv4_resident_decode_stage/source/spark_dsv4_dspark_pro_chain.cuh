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
 *   SparkDsv4DSparkLaunchConfidence      ((hidden+markov)->1 sigmoid, per row)
 *   SparkDsv4LaunchDsparkAttention       (reused Flash; Pro shapes)
 *   SparkDsv4LaunchDsparkMarkovBiasAccum (reused Flash; markov 512)
 *   SparkDsv4LaunchDsparkArgmax          (reused Flash; vocab shard)
 *
 * Guard: SPARK_DSV4_PRO_BUILD is the compile flag Makefile.pro sets on every
 * Pro archive (-DSPARK_DSV4_PRO_BUILD=1, host and CUDA TUs alike). These
 * bodies used to gate on SPARK_DSV4_MODEL_BUILD, which no build ever
 * defined, so both Pro dspark headers compiled to nothing.
 *
 * The chain is expressed over an EXPLICIT buffer surface rather than the
 * host-side module state struct (that type is private to the module .c and
 * invisible here): the module drives each stage below with its own dspark_*
 * allocations. Every launch result is captured and propagated - a failed
 * launch aborts the chain with its cudaError_t instead of being discarded.
 *
 * Sequence (dsv4_pro_ga_migration.md:126-163):
 *   Prelude    1+2. tap capture -> main_proj(concat) -> main_norm -> main_x;
 *                  draft ids [anchor, noise x4] -> embed -> expand x streams
 *   LayerStep  3.   kv_norm(wkv(main)) row -> rolling window slot (per
 *                  mtp layer), then draft attention over window + causal
 *                  draft KV (FFN/hc continuation stays in the module loop,
 *                  which owns the state-local machinery)
 *   DraftHead  4.   markov bias + greedy argmax PER draft row (all
 *                  draft_count tokens, not just position 0), then the
 *                  confidence sigmoid over [rows x (hidden+markov)] features
 * Acceptance/verification stays client-side (compare drafts vs the main model).
 */

#if defined(SPARK_DSV4_PRO_BUILD) && \
	defined(SPARK_DSV4_MODEL_MTP_LAYER_COUNT) && \
	(SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0u)

/*
 * Stages 1+2: capture the target-layer hc means, project them through
 * main_proj + main_norm into main_x, then gather the draft ids (anchor +
 * noise) and replicate the embeddings across the hyper-connection streams.
 */
extern "C" cudaError_t SparkDsv4DsparkProChainPrelude(
	cudaStream_t stream,
	const void *taps_bf16,             /* [taps][hc][hidden] post-layer taps */
	void *tap_capture_bf16,            /* [taps][hidden] hc means */
	const SparkDsv4LinearView *main_proj, /* dense FP8 (taps*hidden -> hidden) */
	const void *main_norm_weight_bf16, /* [hidden] */
	void *main_x_bf16,                 /* [hidden] projection scratch */
	const uint32_t *draft_ids_u32,     /* [draft_count] anchor + noise ids */
	const void *token_embedding_bf16,  /* [vocab][hidden] */
	void *draft_x_bf16,                /* [draft_count][hidden] */
	void *draft_streams_bf16,          /* [draft_count][hc][hidden] */
	uint32_t draft_count,uint32_t multiprocessor_count)
{
	const uint32_t tap_count = SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT;
	const uint32_t hidden = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	const uint32_t hc_stream_count = SPARK_DSV4_MODEL_HC_STREAM_COUNT;
	cudaError_t error;
	if ( stream == 0 || taps_bf16 == 0 || tap_capture_bf16 == 0 ||
		main_proj == 0 || main_proj->payload == 0 ||
		main_norm_weight_bf16 == 0 || main_x_bf16 == 0 ||
		draft_ids_u32 == 0 || token_embedding_bf16 == 0 ||
		draft_x_bf16 == 0 || draft_streams_bf16 == 0 ||
		draft_count == 0u || multiprocessor_count == 0u )
		return(cudaErrorInvalidValue);
	error = SparkDsv4DSparkLaunchMeanReduction(stream,taps_bf16,
		tap_capture_bf16,tap_count,hc_stream_count,hidden,
		multiprocessor_count);
	if ( error != cudaSuccess )
		return(error);
	error = SparkDsv4LaunchLinear(stream,main_proj,tap_capture_bf16,
		main_x_bf16,1u /* rows */);
	if ( error != cudaSuccess )
		return(error);
	error = SparkDsv4LaunchRmsNorm(stream,main_x_bf16,
		main_norm_weight_bf16,main_x_bf16,1u,hidden,
		SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(error);
	error = SparkDsv4LaunchEmbeddingGather(stream,draft_ids_u32,
		token_embedding_bf16,draft_x_bf16,draft_count,hidden);
	if ( error != cudaSuccess )
		return(error);
	return(SparkDsv4LaunchExpandStreams(stream,draft_x_bf16,
		draft_streams_bf16,draft_count,hc_stream_count,hidden,
		multiprocessor_count));
}

/*
 * Stage 3, one mtp layer per call: write the freshly computed main-stream KV
 * row (kv_norm(wkv(main))) into THIS layer's rolling window slot, then run
 * the draft attention over the window ring plus the causal draft KV.
 */
extern "C" cudaError_t SparkDsv4DsparkProChainLayerStep(
	cudaStream_t stream,
	const void *main_kv_row_bf16,      /* [kv_dim] fresh kv_norm(wkv(main)) */
	void *main_kv_windows_bf16,        /* [mtp_layers][window][kv_dim] */
	uint32_t layer_index,
	const void *q_attn_bf16,           /* [draft_count][heads][kv_dim] */
	const void *draft_kv_bf16,         /* [draft_count][kv_dim] block kv */
	const float *attn_sink_f32,        /* [heads] this layer's sinks */
	void *attn_out_bf16,               /* [draft_count][heads][kv_dim] */
	uint32_t draft_count,uint32_t head_count,uint64_t seq_pos)
{
	const uint32_t kv_dim = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	const uint32_t window_tokens = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
	uint16_t *window_layer;
	if ( stream == 0 || main_kv_row_bf16 == 0 || main_kv_windows_bf16 == 0 ||
		q_attn_bf16 == 0 || draft_kv_bf16 == 0 || attn_sink_f32 == 0 ||
		attn_out_bf16 == 0 || draft_count == 0u || head_count == 0u ||
		layer_index >= SPARK_DSV4_MODEL_MTP_LAYER_COUNT )
		return(cudaErrorInvalidValue);
	/* The fresh KV row and the rolling window are distinct buffers: the old
	 * call passed the window as both source and destination, a self-copy
	 * that rewrote the slot with stale bytes. Refuse aliasing loudly. */
	if ( main_kv_row_bf16 == main_kv_windows_bf16 )
		return(cudaErrorInvalidValue);
	window_layer = (uint16_t *)main_kv_windows_bf16 +
		(uint64_t)layer_index * window_tokens * kv_dim;
	/* single-lane layout: lane_stride 0, lane_index 0; seq_pos picks the
	 * modulo slot inside the launcher's kernel. */
	cudaError_t error = SparkDsv4DSparkLaunchMainKvWrite(stream,
		main_kv_row_bf16,window_layer,kv_dim,window_tokens,
		(uint32_t)(seq_pos % window_tokens));
	if ( error != cudaSuccess )
		return(error);
	return(SparkDsv4LaunchDsparkAttention(stream,q_attn_bf16,window_layer,
		0ull /* lane_stride: single lane */,0u /* lane_index */,
		draft_kv_bf16,attn_sink_f32,1.0f / sqrtf((float)kv_dim),
		attn_out_bf16,draft_count,head_count,kv_dim,window_tokens));
}

/*
 * Stage 4: the draft head tail. Markov bias accumulation and greedy argmax
 * run for EVERY draft row - one token per row, draft_count tokens total
 * (the old code ran position 0 only and emitted a single draft) - then the
 * confidence head scores the assembled feature rows.
 */
extern "C" cudaError_t SparkDsv4DsparkProChainDraftHead(
	cudaStream_t stream,
	const void *logits_bf16,           /* [rows][shard] draft head logits */
	const void *markov_w2_bf16,        /* [vocab][markov_rank] */
	const void *markov_embed_bf16,     /* [markov_rank] */
	float *logits_f32,                 /* [rows][shard] biased logits */
	uint32_t vocab_row_start,uint32_t vocab_shard_rows,
	uint32_t *draft_token_ids,         /* [rows] emitted tokens */
	float *draft_scores_f32,           /* [rows] argmax scores */
	const void *confidence_features_bf16, /* [rows][hidden+markov_rank] FULL */
	const void *confidence_weight_bf16,float confidence_bias,
	float *confidence_f32,             /* [rows] */
	uint32_t draft_count,uint32_t multiprocessor_count)
{
	const uint32_t markov_rank = SPARK_DSV4_MODEL_DSPARK_MARKOV_RANK;
	const uint32_t confidence_dimension =
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION + markov_rank;
	uint32_t row;
	cudaError_t error;
	if ( stream == 0 || logits_bf16 == 0 || markov_w2_bf16 == 0 ||
		markov_embed_bf16 == 0 || logits_f32 == 0 ||
		draft_token_ids == 0 || draft_scores_f32 == 0 ||
		confidence_features_bf16 == 0 || confidence_weight_bf16 == 0 ||
		confidence_f32 == 0 || draft_count == 0u ||
		vocab_shard_rows == 0u || markov_rank == 0u ||
		multiprocessor_count == 0u )
		return(cudaErrorInvalidValue);
	for (row = 0u; row < draft_count; row++)
	{
		error = SparkDsv4LaunchDsparkMarkovBiasAccum(stream,logits_bf16,
			markov_w2_bf16,markov_embed_bf16,logits_f32,vocab_row_start,
			vocab_shard_rows,markov_rank,row,multiprocessor_count);
		if ( error != cudaSuccess )
			return(error);
		error = SparkDsv4LaunchDsparkArgmax(stream,
			logits_f32 + (uint64_t)row * vocab_shard_rows,vocab_shard_rows,
			vocab_row_start,draft_token_ids + row,draft_scores_f32 + row);
		if ( error != cudaSuccess )
			return(error);
	}
	/* The confidence kernel indexes features as [row][dimension]; callers
	 * must supply draft_count full rows of hidden|markov_embed. A single-row
	 * [dimension] buffer launched with rows > 1 reads past the allocation -
	 * that out-of-bounds access is why the dimension travels explicitly. */
	return(SparkDsv4DSparkLaunchConfidence(stream,confidence_features_bf16,
		confidence_weight_bf16,confidence_bias,confidence_f32,
		confidence_dimension,draft_count));
}

#endif /* SPARK_DSV4_PRO_BUILD && SPARK_DSV4_MODEL_MTP_LAYER_COUNT > 0 */
