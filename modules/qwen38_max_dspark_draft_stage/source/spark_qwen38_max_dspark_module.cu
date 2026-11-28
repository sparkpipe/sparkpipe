#include "inference/kernels/dspark_standard_transformer.cuh"

struct Q38MConfig
{
	static constexpr uint32_t HIDDEN = 8192u;
	static constexpr uint32_t TAPS = 6u;
	static constexpr uint32_t LAYERS = 5u;
	static constexpr uint32_t HEADS = 64u;
	static constexpr uint32_t KV_HEADS = 8u;
	static constexpr uint32_t HEAD_DIM = 128u;
	static constexpr uint32_t HEAD_THREADS = 128u;
	static constexpr uint32_t MLP = 16384u;
	static constexpr uint32_t BLOCK = 7u;
	static constexpr uint32_t SLIDING = 8192u;
	static constexpr uint32_t WINDOW = SLIDING - BLOCK;
	static constexpr uint32_t VOCAB = 248320u;
	static constexpr uint32_t DRAFT_VOCAB = VOCAB;
	static constexpr uint32_t MARKOV = 256u;
	static constexpr uint32_t CANDIDATES = 2048u;
	static constexpr uint32_t SAMPLED = 32u;
	static constexpr uint32_t BRANCHES = 2u;
	static constexpr uint32_t BRANCH_LEN = 4u;
	static constexpr uint32_t INGEST_ROWS = 256u;
	static constexpr uint32_t SPLIT_KEYS = 256u;
	static constexpr uint32_t ATTN_TILE = 32u;
	static constexpr uint32_t ATTN_Q_GROUP = 4u;
	static constexpr uint32_t SEQUENCES_MAX = 256u;
	static constexpr uint32_t PAGE_POSITIONS = 1u;
	static constexpr uint32_t PAGES_MAX = 1u;
	static constexpr uint32_t SPLITS_MAX = 1u;
	static constexpr uint32_t SPLIT_ROWS_MIN = 1u;
	static constexpr uint32_t RUNS_MAX = 1u;
	static constexpr bool PAGED = false;
	static constexpr bool ROUND_RMS = true;
	static constexpr bool USE_ROPE_GAIN = true;
	static constexpr bool CAUSAL = false;
	static constexpr DsparkNormMode HEAD_NORM_MODE = DSPARK_NORM_BF16_BOTH;
	static constexpr DsparkAttnKind ATTN_KIND = DSPARK_ATTN_RING_SPLIT_GQA128;
	static constexpr const char *LOG_TAG = "qwen38-max dspark";
	static constexpr const char *INTERFACE_NAME = "qwen38-max-dspark";
	static constexpr const char *BAD_PACK_FMT = "lane %s: the pack %s is not a Qwen3.8-Max DSpark pack this module reads";
	static constexpr const char *NAME_FC = "fc.weight";
	static constexpr const char *NAME_HIDDEN_NORM = "hidden_norm.weight";
	static constexpr const char *NAME_NORM = "norm.weight";
	static constexpr const char *NAME_MARKOV_W1 = "markov_head.markov_w1.weight";
	static constexpr const char *NAME_MARKOV_W2 = "markov_head.markov_w2.weight";
	static constexpr const char *NAME_CONF_W = "confidence_head.proj.weight";
	static constexpr const char *NAME_CONF_B = "confidence_head.proj.bias";
	static constexpr const char *LAYER_PREFIX_FMT = "layers.%u.%s";

	static SparkStatus CheckShape(DsparkStandardEngine<Q38MConfig>::State *state, const SparkJsonDocument *document)
	{
		const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
		const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
		const int32_t dflash = DsparkMember(document, checkpoint, "dflash_config");
		const int32_t rope = DsparkMember(document, checkpoint, "rope_parameters");
		const int32_t taps = DsparkMember(document, dflash, "target_layer_ids");
		const DsparkExpectedU64 expected[] = {
			{checkpoint, "block_size", BLOCK}, {checkpoint, "hidden_size", HIDDEN}, {checkpoint, "intermediate_size", MLP},
			{checkpoint, "num_hidden_layers", LAYERS}, {checkpoint, "num_attention_heads", HEADS}, {checkpoint, "num_key_value_heads", KV_HEADS},
			{checkpoint, "head_dim", HEAD_DIM}, {checkpoint, "vocab_size", VOCAB}, {checkpoint, "markov_rank", MARKOV},
		};
		const DsparkExpectedBool flags[] = {
			{checkpoint, "enable_confidence_head", true},
			{checkpoint, "confidence_head_with_markov", true},
			{checkpoint, "use_sliding_window", false},
		};
		uint64_t value = 0u;
		float theta = 0.0f, factor = 0.0f, original = 0.0f, fast = 0.0f, slow = 0.0f, attention_factor = 0.0f;
		int32_t token;
		SparkStatus status = DsparkCheckU64Table(document, LOG_TAG, expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
		if ( status == SPARK_STATUS_OK )
			status = DsparkCheckBoolTable(document, LOG_TAG, flags, (uint32_t)(sizeof(flags) / sizeof(flags[0])));
		if ( status != SPARK_STATUS_OK )
			return status;
		token = DsparkMember(document, checkpoint, "markov_head_type");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, dflash, "projector_type");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "dspark") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, dflash, "attention_mode");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "gqa") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( taps < 0 || SparkJsonGetArrayElementCount(document, taps) != TAPS )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, dflash, "mask_token_id");
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= VOCAB )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->mask_token = (uint32_t)value;
		token = DsparkMember(document, checkpoint, "rms_norm_eps");
		if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK || state->epsilon <= 0.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "rope_type");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "yarn") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( DsparkMember(document, rope, "rope_theta") < 0 || SparkJsonGetFloat(document, DsparkMember(document, rope, "rope_theta"), &theta) != SPARK_STATUS_OK ||
			theta <= 0.0f || DsparkMember(document, rope, "factor") < 0 || SparkJsonGetFloat(document, DsparkMember(document, rope, "factor"), &factor) != SPARK_STATUS_OK ||
			factor <= 0.0f || DsparkMember(document, rope, "original_max_position_embeddings") < 0 ||
			SparkJsonGetFloat(document, DsparkMember(document, rope, "original_max_position_embeddings"), &original) != SPARK_STATUS_OK || original <= 0.0f ||
			DsparkMember(document, rope, "beta_fast") < 0 || SparkJsonGetFloat(document, DsparkMember(document, rope, "beta_fast"), &fast) != SPARK_STATUS_OK ||
			fast <= 0.0f || DsparkMember(document, rope, "beta_slow") < 0 ||
			SparkJsonGetFloat(document, DsparkMember(document, rope, "beta_slow"), &slow) != SPARK_STATUS_OK || slow <= 0.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "attention_factor");
		if ( token >= 0 && SparkJsonGetFloat(document, token, &attention_factor) == SPARK_STATUS_OK && attention_factor > 0.0f )
			state->rope_gain = attention_factor;
		else
			state->rope_gain = (float)(factor > 1.0f ? 0.1 * log((double)factor) + 1.0 : 1.0);
		state->softmax_scale = (float)(1.0 / sqrt((double)HEAD_DIM));
		if ( SparkGpuMalloc((void **)&state->rope_inverse, (HEAD_DIM / 2u) * sizeof(float)) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		LM_LAUNCH((DsparkYarnTableKernel<HEAD_DIM / 2u, HEAD_DIM>), 1u, DSPARK_THREADS, 0, state->stream, state->rope_inverse, (double)theta,
			(double)factor, (double)original, (double)fast, (double)slow);
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}
};

using Q38MEngine = DsparkStandardEngine<Q38MConfig>;

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return Q38MEngine::Interface();
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_max_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	return Q38MEngine::CacheRing(opaque, sequence_id, layer, keys, values, positions);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_max_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	return Q38MEngine::ForwardDiag(opaque, sequence_id, anchor, position, y, elapsed_ns);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_max_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	return Q38MEngine::TraceDiag(opaque, sequence_id, anchor, position, layers, hidden);
}
