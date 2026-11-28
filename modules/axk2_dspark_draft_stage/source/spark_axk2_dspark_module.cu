#include "inference/kernels/dspark_standard_transformer.cuh"

struct Axk2Config
{
	static constexpr uint32_t HIDDEN = 7168u;
	static constexpr uint32_t TAPS = 3u;
	static constexpr uint32_t LAYERS = 3u;
	static constexpr uint32_t HEADS = 64u;
	static constexpr uint32_t KV_HEADS = 64u;
	static constexpr uint32_t HEAD_DIM = 112u;
	static constexpr uint32_t HEAD_THREADS = 128u;
	static constexpr uint32_t MLP = 18432u;
	static constexpr uint32_t BLOCK = 5u;
	static constexpr uint32_t SLIDING = 128u;
	static constexpr uint32_t WINDOW = SLIDING - BLOCK;
	static constexpr uint32_t VOCAB = 163840u;
	static constexpr uint32_t DRAFT_VOCAB = 32768u;
	static constexpr uint32_t MARKOV = 256u;
	static constexpr uint32_t CANDIDATES = 2048u;
	static constexpr uint32_t SAMPLED = 32u;
	static constexpr uint32_t BRANCHES = 2u;
	static constexpr uint32_t BRANCH_LEN = 4u;
	static constexpr uint32_t INGEST_ROWS = 256u;
	static constexpr uint32_t SPLIT_KEYS = 0u;
	static constexpr uint32_t ATTN_TILE = 32u;
	static constexpr uint32_t ATTN_Q_GROUP = 1u;
	static constexpr uint32_t SEQUENCES_MAX = 256u;
	static constexpr uint32_t PAGE_POSITIONS = 1u;
	static constexpr uint32_t PAGES_MAX = 1u;
	static constexpr uint32_t SPLITS_MAX = 1u;
	static constexpr uint32_t SPLIT_ROWS_MIN = 1u;
	static constexpr uint32_t RUNS_MAX = 1u;
	static constexpr bool PAGED = false;
	static constexpr bool ROUND_RMS = true;
	static constexpr bool USE_ROPE_GAIN = true;
	static constexpr bool CAUSAL = true;
	static constexpr DsparkNormMode HEAD_NORM_MODE = DSPARK_NORM_BF16_BOTH;
	static constexpr DsparkAttnKind ATTN_KIND = DSPARK_ATTN_RING_SINGLE_PASS;
	static constexpr const char *LOG_TAG = "axk2 dspark";
	static constexpr const char *INTERFACE_NAME = "axk2-dspark";
	static constexpr const char *BAD_PACK_FMT = "lane %s: the pack %s is not an A.X-K2 DSpark pack this module reads";
	static constexpr const char *NAME_FC = "fc.weight";
	static constexpr const char *NAME_HIDDEN_NORM = "hidden_norm.weight";
	static constexpr const char *NAME_NORM = "norm.weight";
	static constexpr const char *NAME_MARKOV_W1 = "markov_head.markov_w1.weight";
	static constexpr const char *NAME_MARKOV_W2 = "markov_head.markov_w2.weight";
	static constexpr const char *NAME_CONF_W = "confidence_head.proj.weight";
	static constexpr const char *NAME_CONF_B = "confidence_head.proj.bias";
	static constexpr const char *LAYER_PREFIX_FMT = "layers.%u.%s";

	static SparkStatus CheckShape(DsparkStandardEngine<Axk2Config>::State *state, const SparkJsonDocument *document)
	{
		const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
		const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
		const int32_t transformer = DsparkMember(document, checkpoint, "transformer_layer_config");
		const int32_t rope = DsparkMember(document, transformer, "rope_parameters");
		const int32_t aux = DsparkMember(document, checkpoint, "aux_hidden_state_layer_ids");
		const int32_t layer_types = DsparkMember(document, transformer, "layer_types");
		const DsparkExpectedU64 expected[] = {
			{checkpoint, "block_size", BLOCK}, {checkpoint, "draft_vocab_size", DRAFT_VOCAB}, {checkpoint, "markov_rank", MARKOV},
			{transformer, "hidden_size", HIDDEN}, {transformer, "intermediate_size", MLP}, {transformer, "num_hidden_layers", LAYERS},
			{transformer, "num_attention_heads", HEADS}, {transformer, "num_key_value_heads", KV_HEADS}, {transformer, "head_dim", HEAD_DIM},
			{transformer, "sliding_window", SLIDING}, {transformer, "vocab_size", VOCAB},
		};
		const DsparkExpectedBool flags[] = {
			{checkpoint, "sample_from_anchor", true}, {checkpoint, "enable_confidence_head", true}, {checkpoint, "confidence_head_with_markov", true},
			{checkpoint, "sliding_window_non_causal", false}, {transformer, "use_sliding_window", true},
		};
		uint64_t value = 0u, original = 0u;
		float theta = 0.0f, factor = 1.0f, beta_fast = 32.0f, beta_slow = 1.0f, mscale = 1.0f, mscale_all = 1.0f;
		int32_t token;
		SparkStatus status = DsparkCheckU64Table(document, LOG_TAG, expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
		if ( status == SPARK_STATUS_OK )
			status = DsparkCheckBoolTable(document, LOG_TAG, flags, (uint32_t)(sizeof(flags) / sizeof(flags[0])));
		if ( status != SPARK_STATUS_OK )
			return status;
		token = DsparkMember(document, checkpoint, "markov_head_type");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( aux < 0 || SparkJsonGetArrayElementCount(document, aux) != TAPS )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( layer_types < 0 || SparkJsonGetArrayElementCount(document, layer_types) != LAYERS )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for ( uint32_t layer = 0u; layer < LAYERS; layer++ )
		{
			const int32_t elem = SparkJsonGetArrayElement(document, layer_types, layer);
			if ( elem < 0 || !SparkJsonStringEquals(document, elem, "sliding_attention") )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		token = DsparkMember(document, checkpoint, "mask_token_id");
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= VOCAB )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->mask_token = (uint32_t)value;
		token = DsparkMember(document, transformer, "rms_norm_eps");
		if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "rope_type");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "yarn") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "rope_theta");
		if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "factor");
		if ( token < 0 || SparkJsonGetFloat(document, token, &factor) != SPARK_STATUS_OK || factor < 1.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "original_max_position_embeddings");
		if ( token < 0 || SparkJsonGetUInt64(document, token, &original) != SPARK_STATUS_OK || original == 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "beta_fast");
		if ( token < 0 || SparkJsonGetFloat(document, token, &beta_fast) != SPARK_STATUS_OK || beta_fast <= 0.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "beta_slow");
		if ( token < 0 || SparkJsonGetFloat(document, token, &beta_slow) != SPARK_STATUS_OK || beta_slow <= 0.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "mscale");
		if ( token >= 0 && SparkJsonGetFloat(document, token, &mscale) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "mscale_all_dim");
		if ( token >= 0 && SparkJsonGetFloat(document, token, &mscale_all) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->rope_gain = (float)(DsparkYarnScale((double)factor, (double)mscale) / DsparkYarnScale((double)factor, (double)mscale_all));
		state->softmax_scale = (float)(1.0 / sqrt((double)HEAD_DIM));
		if ( SparkGpuMalloc((void **)&state->rope_inverse, (HEAD_DIM / 2u) * sizeof(float)) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		LM_LAUNCH((DsparkYarnTableKernel<HEAD_DIM / 2u, HEAD_DIM>), 1u, DSPARK_THREADS, 0, state->stream, state->rope_inverse, (double)theta,
			(double)factor, (double)original, (double)beta_fast, (double)beta_slow);
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}
};

using Axk2Engine = DsparkStandardEngine<Axk2Config>;

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return Axk2Engine::Interface();
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_axk2_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	return Axk2Engine::CacheRing(opaque, sequence_id, layer, keys, values, positions);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_axk2_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	return Axk2Engine::ForwardDiag(opaque, sequence_id, anchor, position, y, elapsed_ns);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_axk2_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	return Axk2Engine::TraceDiag(opaque, sequence_id, anchor, position, layers, hidden);
}
