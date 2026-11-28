#include "sparkpipe/spark_glm52_model.h"
#include "inference/kernels/dspark_standard_transformer.cuh"

struct G53Config
{
	static constexpr uint32_t HIDDEN = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	static constexpr uint32_t TAPS = 5u;
	static constexpr uint32_t LAYERS = 3u;
	static constexpr uint32_t HEADS = 64u;
	static constexpr uint32_t KV_HEADS = 64u;
	static constexpr uint32_t HEAD_DIM = 64u;
	static constexpr uint32_t HEAD_THREADS = 64u;
	static constexpr uint32_t MLP = SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	static constexpr uint32_t BLOCK = 8u;
	static constexpr uint32_t SLIDING = 2048u;
	static constexpr uint32_t WINDOW = SLIDING - BLOCK;
	static constexpr uint32_t VOCAB = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	static constexpr uint32_t DRAFT_VOCAB = VOCAB;
	static constexpr uint32_t MARKOV = 256u;
	static constexpr uint32_t CANDIDATES = 2048u;
	static constexpr uint32_t SAMPLED = 32u;
	static constexpr uint32_t BRANCHES = 2u;
	static constexpr uint32_t BRANCH_LEN = 4u;
	static constexpr uint32_t INGEST_ROWS = 256u;
	static constexpr uint32_t SPLIT_KEYS = 256u;
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
	static constexpr bool USE_ROPE_GAIN = false;
	static constexpr bool CAUSAL = true;
	static constexpr DsparkNormMode HEAD_NORM_MODE = DSPARK_NORM_BF16_BOTH;
	static constexpr DsparkAttnKind ATTN_KIND = DSPARK_ATTN_RING_SPLIT_MHA64;
	static constexpr const char *LOG_TAG = "glm53 dspark";
	static constexpr const char *INTERFACE_NAME = "glm53-dspark";
	static constexpr const char *BAD_PACK_FMT = "lane %s: the pack %s is not a GLM-5.3 DSpark pack this module reads";
	static constexpr const char *NAME_FC = "fc.weight";
	static constexpr const char *NAME_HIDDEN_NORM = "hidden_norm.weight";
	static constexpr const char *NAME_NORM = "norm.weight";
	static constexpr const char *NAME_MARKOV_W1 = "markov_head.markov_w1.weight";
	static constexpr const char *NAME_MARKOV_W2 = "markov_head.markov_w2.weight";
	static constexpr const char *NAME_CONF_W = "confidence_head.proj.weight";
	static constexpr const char *NAME_CONF_B = "confidence_head.proj.bias";
	static constexpr const char *LAYER_PREFIX_FMT = "layers.%u.%s";

	static SparkStatus CheckShape(DsparkStandardEngine<G53Config>::State *state, const SparkJsonDocument *document)
	{
		const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
		const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
		const int32_t transformer = DsparkMember(document, checkpoint, "transformer_layer_config");
		const int32_t rope = DsparkMember(document, transformer, "rope_parameters");
		const int32_t aux = DsparkMember(document, checkpoint, "aux_hidden_state_layer_ids");
		const DsparkExpectedU64 expected[] = {
			{checkpoint, "block_size", BLOCK}, {checkpoint, "draft_vocab_size", VOCAB}, {checkpoint, "markov_rank", MARKOV},
			{transformer, "hidden_size", HIDDEN}, {transformer, "intermediate_size", MLP}, {transformer, "num_hidden_layers", LAYERS},
			{transformer, "num_attention_heads", HEADS}, {transformer, "num_key_value_heads", KV_HEADS}, {transformer, "head_dim", HEAD_DIM},
			{transformer, "sliding_window", SLIDING}, {transformer, "vocab_size", VOCAB},
		};
		const DsparkExpectedBool flags[] = {
			{checkpoint, "sample_from_anchor", true}, {checkpoint, "enable_confidence_head", true},
			{checkpoint, "confidence_head_with_markov", true}, {checkpoint, "sliding_window_non_causal", false},
		};
		uint64_t value = 0u;
		float theta = 0.0f;
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
		token = DsparkMember(document, checkpoint, "mask_token_id");
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= VOCAB )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->mask_token = (uint32_t)value;
		token = DsparkMember(document, transformer, "rms_norm_eps");
		if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		token = DsparkMember(document, rope, "rope_theta");
		if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK || theta <= 0.0f )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->softmax_scale = (float)(1.0 / sqrt((double)HEAD_DIM));
		if ( SparkGpuMalloc((void **)&state->rope_inverse, (HEAD_DIM / 2u) * sizeof(float)) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		LM_LAUNCH((DsparkRopeTableKernel<HEAD_DIM / 2u, HEAD_DIM>), 1u, DSPARK_THREADS, 0, state->stream, state->rope_inverse, theta);
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}
};

using G53Engine = DsparkStandardEngine<G53Config>;

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return G53Engine::Interface();
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	return G53Engine::CacheRing(opaque, sequence_id, layer, keys, values, positions);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	return G53Engine::ForwardDiag(opaque, sequence_id, anchor, position, y, elapsed_ns);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_glm53_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	return G53Engine::TraceDiag(opaque, sequence_id, anchor, position, layers, hidden);
}
