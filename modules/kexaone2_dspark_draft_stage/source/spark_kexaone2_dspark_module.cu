#include "inference/kernels/dspark_standard_transformer.cuh"

struct KX2Config
{
	static constexpr uint32_t HIDDEN = 6144u;
	static constexpr uint32_t TAPS = 5u;
	static constexpr uint32_t LAYERS = 5u;
	static constexpr uint32_t TARGET_LAYERS = 78u;
	static constexpr uint32_t HEADS = 64u;
	static constexpr uint32_t KV_HEADS = 8u;
	static constexpr uint32_t HEAD_DIM = 128u;
	static constexpr uint32_t HEAD_THREADS = 128u;
	static constexpr uint32_t MLP = 18432u;
	static constexpr uint32_t BLOCK = 7u;
	static constexpr uint32_t SLIDING = 1u;
	static constexpr uint32_t WINDOW = 1u;
	static constexpr uint32_t VOCAB = 153600u;
	static constexpr uint32_t DRAFT_VOCAB = VOCAB;
	static constexpr uint32_t MARKOV = 256u;
	static constexpr uint32_t CANDIDATES = 2048u;
	static constexpr uint32_t SAMPLED = 32u;
	static constexpr uint32_t BRANCHES = 2u;
	static constexpr uint32_t BRANCH_LEN = 4u;
	static constexpr uint32_t INGEST_ROWS = 256u;
	static constexpr uint32_t SPLIT_KEYS = 0u;
	static constexpr uint32_t ATTN_TILE = 32u;
	static constexpr uint32_t ATTN_Q_GROUP = 4u;
	static constexpr uint32_t SEQUENCES_MAX = 64u;
	static constexpr uint32_t PAGE_POSITIONS = 1024u;
	static constexpr uint32_t PAGES_MAX = 1024u;
	static constexpr uint32_t SPLITS_MAX = 16u;
	static constexpr uint32_t SPLIT_ROWS_MIN = 256u;
	static constexpr uint32_t RUNS_MAX = 256u;
	static constexpr bool PAGED = true;
	static constexpr bool ROUND_RMS = false;
	static constexpr bool USE_ROPE_GAIN = false;
	static constexpr bool CAUSAL = false;
	static constexpr DsparkNormMode HEAD_NORM_MODE = DSPARK_NORM_BF16_OUT;
	static constexpr DsparkAttnKind ATTN_KIND = DSPARK_ATTN_PAGED_GQA128;
	static constexpr const char *LOG_TAG = "kexaone2 dspark";
	static constexpr const char *INTERFACE_NAME = "kexaone2-dspark";
	static constexpr const char *BAD_PACK_FMT = "lane %s: the pack %s is not a K-EXAONE 2.0 DSpark pack this module reads";
	static constexpr const char *NAME_FC = "mtp.fc.weight";
	static constexpr const char *NAME_HIDDEN_NORM = "mtp.hidden_norm.weight";
	static constexpr const char *NAME_NORM = "mtp.norm.weight";
	static constexpr const char *NAME_MARKOV_W1 = "mtp.markov_head.markov_w1.weight";
	static constexpr const char *NAME_MARKOV_W2 = "mtp.markov_head.markov_w2.weight";
	static constexpr const char *NAME_CONF_W = "mtp.confidence_head.proj.weight";
	static constexpr const char *NAME_CONF_B = "mtp.confidence_head.proj.bias";
	static constexpr const char *LAYER_PREFIX_FMT = "mtp.layers.%u.%s";

	static SparkStatus CheckShape(DsparkStandardEngine<KX2Config>::State *state, const SparkJsonDocument *document)
	{
		const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
		const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
		const int32_t rope = DsparkMember(document, checkpoint, "rope_parameters");
		const int32_t taps = DsparkMember(document, checkpoint, "dspark_target_layer_ids");
		const int32_t layer_types = DsparkMember(document, checkpoint, "mtp_layer_types");
		const DsparkExpectedU64 expected[] = {
			{checkpoint, "dspark_block_size", BLOCK}, {checkpoint, "dspark_num_hidden_layers", LAYERS},
			{checkpoint, "dspark_num_target_layers", TARGET_LAYERS}, {checkpoint, "dspark_markov_rank", MARKOV},
			{checkpoint, "hidden_size", HIDDEN}, {checkpoint, "intermediate_size", MLP}, {checkpoint, "num_attention_heads", HEADS},
			{checkpoint, "num_key_value_heads", KV_HEADS}, {checkpoint, "head_dim", HEAD_DIM}, {checkpoint, "vocab_size", VOCAB},
		};
		const DsparkExpectedBool flags[] = {
			{checkpoint, "dspark_enable_confidence_head", true},
			{checkpoint, "dspark_confidence_head_with_markov", true},
		};
		uint64_t value = 0u;
		float theta = 0.0f;
		int32_t token;
		SparkStatus status = DsparkCheckU64Table(document, LOG_TAG, expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
		if ( status == SPARK_STATUS_OK )
			status = DsparkCheckBoolTable(document, LOG_TAG, flags, (uint32_t)(sizeof(flags) / sizeof(flags[0])));
		if ( status != SPARK_STATUS_OK )
			return status;
		token = DsparkMember(document, checkpoint, "dspark_markov_head_type");
		if ( token < 0 || !SparkJsonStringEquals(document, token, "vanilla") )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( taps < 0 || SparkJsonGetArrayElementCount(document, taps) != TAPS )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( layer_types < 0 || SparkJsonGetArrayElementCount(document, layer_types) != LAYERS )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for ( uint32_t layer = 0u; layer < LAYERS; layer++ )
		{
			const int32_t elem = SparkJsonGetArrayElement(document, layer_types, layer);
			if ( elem < 0 || !SparkJsonStringEquals(document, elem, "full_attention") )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		token = DsparkMember(document, checkpoint, "dspark_noise_token_id");
		if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= VOCAB )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->mask_token = (uint32_t)value;
		token = DsparkMember(document, checkpoint, "rms_norm_eps");
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

using KX2Engine = DsparkStandardEngine<KX2Config>;

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return KX2Engine::Interface();
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_kexaone2_dspark_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint64_t first, uint32_t count, uint16_t *keys, uint16_t *values)
{
	return KX2Engine::CachePaged(opaque, sequence_id, layer, first, count, keys, values);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_kexaone2_dspark_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
{
	return KX2Engine::ForwardDiag(opaque, sequence_id, anchor, position, y, elapsed_ns);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_kexaone2_dspark_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	return KX2Engine::TraceDiag(opaque, sequence_id, anchor, position, layers, hidden);
}
