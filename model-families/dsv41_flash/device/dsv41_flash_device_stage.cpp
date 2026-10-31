#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_dsv41_flash_device_stage.h"
#include "sparkpipe/spark_dsv41_flash_family.h"
#include "sparkpipe/spark_rope_plan.h"

#define DD_SLICES 64u
#define DD_ALIGNMENT 128u
#define DD_ENGRAM_FILE "dsv41_flash.engram.pack"

template<uint32_t TP>
struct D41Device
{
	static constexpr uint32_t kHidden = SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION;
	static constexpr uint32_t kLayers = SPARK_DSV41_FLASH_MODEL_LAYER_COUNT;
	static constexpr uint32_t kVocab = SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT;
	static constexpr uint32_t kHeads = SPARK_DSV41_FLASH_MODEL_ATTENTION_HEAD_COUNT / TP;
	static constexpr uint32_t kHeadDim = SPARK_DSV41_FLASH_MODEL_HEAD_DIMENSION;
	static constexpr uint32_t kRope = SPARK_DSV41_FLASH_MODEL_QK_ROPE_HEAD_DIMENSION;
	static constexpr uint32_t kNope = kHeadDim - kRope;
	static constexpr uint32_t kQueryWidth = kHeads * kHeadDim;
	static constexpr uint32_t kQLora = SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK;
	static constexpr uint32_t kOLora = SPARK_DSV41_FLASH_MODEL_OUTPUT_LORA_RANK;
	static constexpr uint32_t kGroups = SPARK_DSV41_FLASH_MODEL_OUTPUT_GROUP_COUNT / TP;
	static constexpr uint32_t kGroupInput = SPARK_DSV41_FLASH_MODEL_OUTPUT_GROUP_INPUT_DIMENSION;
	static constexpr uint32_t kOutputWidth = kGroups * kOLora;
	static constexpr uint32_t kWindow = SPARK_DSV41_FLASH_MODEL_SLIDING_WINDOW;
	static constexpr uint32_t kExperts = SPARK_DSV41_FLASH_MODEL_MOE_ROUTED_EXPERT_COUNT;
	static constexpr uint32_t kLocalExperts = kExperts / TP;
	static constexpr uint32_t kTopK = SPARK_DSV41_FLASH_MODEL_MOE_EXPERTS_PER_TOKEN;
	static constexpr uint32_t kMoeInter = SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION;
	static constexpr uint32_t kSharedRows = kMoeInter / TP;
	static constexpr uint32_t kStreams = SPARK_DSV41_FLASH_MODEL_HC_MULT;
	static constexpr uint32_t kResidual = kStreams * kHidden;
	static constexpr uint32_t kOrders = SPARK_DSV41_FLASH_MODEL_ENGRAM_MAX_NGRAM_SIZE - 1u;
	static constexpr uint32_t kEngramHeads = SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_COUNT;
	static constexpr uint32_t kEngramColumns = kOrders * kEngramHeads;
	static constexpr uint32_t kEngramDim = SPARK_DSV41_FLASH_MODEL_ENGRAM_HEAD_DIMENSION;
	static constexpr uint32_t kEngramScales = kEngramDim / SPARK_MXFP8_GROUP;
	static constexpr uint32_t kEngramEmbed = kEngramColumns * kEngramDim;
	static constexpr uint32_t kEngramKv = (kStreams + 1u) * kHidden;
	static constexpr uint32_t kEngramKvLocal = kEngramKv / TP;
	static constexpr uint32_t kEngrams = SPARK_DSV41_FLASH_MODEL_ENGRAM_MODULE_COUNT;
	static constexpr uint32_t kHistory = kOrders + 1u;
	static constexpr uint32_t kCaches = SPARK_DSV41_FLASH_MODEL_KV_SOURCE_COUNT;
	static constexpr uint32_t kPairCaches = kCaches - 1u;
	static constexpr uint32_t kDenseKeys = SPARK_DSV41_FLASH_MODEL_INDEX_TOP_K;
	static constexpr float kEpsilon = SPARK_DSV41_FLASH_MODEL_RMS_NORM_EPSILON;

	static_assert(TP != 0u && SPARK_DSV41_FLASH_MODEL_OUTPUT_GROUP_COUNT % TP == 0u, "a rank holds whole output groups");
	static_assert(kSharedRows % SPARK_MXFP8_GROUP == 0u && kEngramKvLocal % SPARK_MXFP8_GROUP == 0u, "rank slices hold whole 32-wide scale groups");

	struct Hyper
	{
		SparkTensor fn;
		SparkTensor base;
		SparkTensor scale;
	};

	struct Layer
	{
		Hyper hyper[2];
		SparkTensor norm[2];
		SparkDeviceWeight q_a;
		SparkDeviceWeight q_b;
		SparkDeviceWeight kv;
		SparkDeviceWeight o_a;
		SparkDeviceWeight o_b;
		SparkTensor q_norm;
		SparkTensor kv_norm;
		SparkTensor sinks;
		SparkTensor c_kv;
		SparkTensor c_gate;
		SparkTensor c_norm;
		SparkTensor router;
		SparkTensor router_bias;
		SparkDeviceExpertMatrix gate;
		SparkDeviceExpertMatrix up;
		SparkDeviceExpertMatrix down;
		SparkDeviceWeight shared_gate_up;
		SparkDeviceWeight shared_down;
		SparkDeviceWeight engram_kv;
		SparkTensor engram_query;
		SparkTensor engram_key;
	};

	struct EngramTable
	{
		const uint8_t *payload;
		const uint8_t *scale;
		uint64_t rows;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		Layer layers[kLayers];
		SparkTensor final_norm;
		SparkTensor token_map;
		SparkTensor multipliers;
		SparkTensor primes;
		SparkTensor offsets;
		SparkHeadRopeParams query_rope[2];
		SparkHeadRopeParams key_rope[2];
		SparkHeadRopeParams output_rope[2];
		SparkTensor streams;
		SparkTensor pre;
		SparkTensor post;
		SparkTensor comb;
		SparkTensor mixed;
		SparkTensor normed;
		SparkTensor quantized;
		SparkTensor partial;
		SparkTensor shared_out;
		SparkTensor router_logits;
		SparkTensor q_latent;
		SparkTensor q_normed;
		SparkTensor query;
		SparkTensor kv;
		SparkTensor attended;
		SparkTensor records;
		SparkTensor wo_a_out;
		SparkTensor compressed_kv;
		SparkTensor compressed_score;
		SparkTensor pooled;
		SparkTensor latent;
		SparkTensor store[2];
		SparkTensor rope[2];
		SparkTensor last[2];
		SparkTensor engram_embed;
		SparkTensor engram_local;
		SparkTensor engram_lanes;
		SparkTensor engram_gathered;
		SparkDeviceFfnScratch ffn;
		SparkDeviceExpertsScratch experts;
		SparkDeviceBuffer engram_ids;
		SparkDeviceBuffer engram_payload;
		SparkDeviceBuffer engram_scale;
		SparkDeviceBuffer engram_local_ids;
		SparkDeviceBuffer pre_init;
		uint32_t *host_last;
		uint32_t engram_pad;
		uint32_t engram_rows[kEngrams];
		EngramTable engram[kEngrams];
		SparkNamedPack engram_pack;
		uint32_t engram_open;
		int engram_fd;
		void *engram_map;
		uint64_t engram_bytes;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
	};

	static uint64_t Max(uint64_t a, uint64_t b)
	{
		return a > b ? a : b;
	}

	static uint32_t RatioSlot(uint32_t ratio)
	{
		return ratio == 2u ? 0u : 1u;
	}

	static uint32_t EngramLayer(uint32_t which)
	{
		return which == 0u ? SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_0 : SPARK_DSV41_FLASH_MODEL_ENGRAM_LAYER_1;
	}

	static void RopeTable(float *table, uint32_t yarn)
	{
		if ( yarn != 0u )
		{
			SparkRopeYarnRampTable(table, kRope, SPARK_DSV41_FLASH_MODEL_COMPRESS_ROPE_THETA, SPARK_DSV41_FLASH_MODEL_YARN_FACTOR,
				(float)SPARK_DSV41_FLASH_MODEL_YARN_BETA_FAST, (float)SPARK_DSV41_FLASH_MODEL_YARN_BETA_SLOW, SPARK_DSV41_FLASH_MODEL_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS);
			return;
		}
		for ( uint32_t pair = 0u; pair < kRope / 2u; pair++ )
			table[pair] = 1.0f / powf(SPARK_DSV41_FLASH_MODEL_ROPE_THETA, (float)(2u * pair) / (float)kRope);
	}

	static SparkHeadRopeParams TailRope(uint32_t heads, const float *table, float sign)
	{
		SparkHeadRopeParams params;
		memset(&params, 0, sizeof(params));
		params.heads = heads;
		params.head_dimension = kRope;
		params.input_stride = params.output_stride = heads * kHeadDim;
		params.input_offset = kNope;
		params.input_head_stride = params.output_head_stride = kHeadDim;
		params.rope_dimension = kRope;
		params.plain = 1u;
		params.scale = 1.0f;
		params.interleaved = 1u;
		for ( uint32_t pair = 0u; pair < kRope / 2u; pair++ )
			params.inverse_frequency[pair] = sign * table[pair];
		return params;
	}

	static void Shapes(State *state)
	{
		SparkDeviceStageShape *shape = &state->shape;
		memset(shape, 0, sizeof(*shape));
		shape->tag = "sparkpipe_dsv41_flash_device";
		shape->hidden = kHidden;
		shape->vocab = kVocab;
		shape->layers = kLayers;
		shape->kv_layers = kLayers + kCaches;
		shape->kv_heads = 1u;
		shape->kv_head_dimension = kHeadDim;
		shape->pack_alignment = DD_ALIGNMENT;
		shape->epsilon = kEpsilon;
		shape->projection = SPARK_PROJECTION_GEMV;
		shape->prenormed_head = 1u;
		for ( uint32_t yarn = 0u; yarn < 2u; yarn++ )
		{
			float table[SPARK_KERNEL_ROPE_PAIRS];
			RopeTable(table, yarn);
			state->query_rope[yarn] = TailRope(kHeads, table, 1.0f);
			state->key_rope[yarn] = TailRope(1u, table, 1.0f);
			state->output_rope[yarn] = TailRope(kHeads, table, -1.0f);
		}
	}

	static uint64_t RecordBytes(const State *state)
	{
		const uint32_t positions = state->stage.max_positions < kDenseKeys ? state->stage.max_positions : kDenseKeys;
		const uint64_t partitions = 2u * (((uint64_t)positions + 127u) / 128u);
		return Max(SparkOpsAttentionScratchBytes(state->stage.max_rows, kHeads, kHeadDim, positions),
			(uint64_t)state->stage.max_rows * kHeads * partitions * (kHeadDim + 2u) * sizeof(float));
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = kHidden, routes = rows * kTopK, table = rows * sizeof(uint32_t);
		const uint64_t dequant = Max(Max((uint64_t)kQLora * kHidden, (uint64_t)kQueryWidth * kQLora), Max((uint64_t)kHidden * kOutputWidth,
			Max((uint64_t)2u * kSharedRows * kHidden, (uint64_t)kEngramKvLocal * kEngramEmbed)));
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * kResidual * 2u, &state->streams },
			{ rows * kStreams * sizeof(float), &state->pre },
			{ rows * kStreams * sizeof(float), &state->post },
			{ rows * kStreams * kStreams * sizeof(float), &state->comb },
			{ rows * hidden * 2u, &state->mixed },
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->quantized },
			{ rows * hidden * 2u, &state->partial },
			{ rows * hidden * 2u, &state->shared_out },
			{ rows * kExperts * sizeof(float), &state->router_logits },
			{ rows * kQLora * 2u, &state->q_latent },
			{ rows * kQLora * 2u, &state->q_normed },
			{ rows * kQueryWidth * 2u, &state->query },
			{ rows * kHeadDim * 2u, &state->kv },
			{ rows * kQueryWidth * 2u, &state->attended },
			{ RecordBytes(state), &state->records },
			{ rows * kOutputWidth * 2u, &state->wo_a_out },
			{ rows * kHeadDim * sizeof(float), &state->compressed_kv },
			{ rows * kHeadDim * sizeof(float), &state->compressed_score },
			{ rows * kHeadDim * 2u, &state->pooled },
			{ rows * kHeadDim * 2u, &state->latent },
			{ table, &state->store[0] },
			{ table, &state->store[1] },
			{ table, &state->rope[0] },
			{ table, &state->rope[1] },
			{ table, &state->last[0] },
			{ table, &state->last[1] },
			{ rows * kEngramEmbed * 2u, &state->engram_embed },
			{ rows * kEngramKvLocal * 2u, &state->engram_local },
			{ TP * rows * kEngramKvLocal * 2u, &state->engram_lanes },
			{ rows * kEngramKv * 2u, &state->engram_gathered },
			{ rows * 2u * kSharedRows * 2u, &state->ffn.gate_up },
			{ rows * kSharedRows * 2u, &state->ffn.act },
			{ routes * sizeof(uint32_t), &state->experts.experts },
			{ routes * sizeof(float), &state->experts.weights },
			{ routes * 2u * kMoeInter * 2u, &state->experts.gate_up },
			{ routes * kMoeInter * 2u, &state->experts.act },
			{ routes * hidden * 2u, &state->experts.out },
			{ (routes + 1u) * hidden * 2u, &state->experts.sorted },
			{ dequant * 2u, &state->stage.dequant }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[DD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->engram_fd = -1;
		Shapes(state);
		const SparkDeviceStageRecurrentPlan pools[] = { { kPairCaches, 2u * kHeadDim * sizeof(float) }, { 1u, kHistory * sizeof(int32_t) } };
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", kHidden }, { "layers", kLayers }, { "vocab", kVocab }, { "vocab_rows", kVocab / TP }, { "heads", kHeads },
			{ "experts", kExperts }, { "local_experts", kLocalExperts }, { "top_k", kTopK }, { "shared_rows", kSharedRows }
		};
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK && (SparkNamedPackConfigU32(&state->stage.pack, "engram_pad", &state->engram_pad) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->stage.pack, "engram_rows_0", &state->engram_rows[0]) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->stage.pack, "engram_rows_1", &state->engram_rows[1]) != SPARK_STATUS_OK) )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks the engram pad id or table rows");
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), pools,
				sizeof(pools) / sizeof(pools[0]), request->configuration->state_budget_bytes);
		if ( status == SPARK_STATUS_OK )
			status = OpenEngram(state, request->configuration->rank_pack_path);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus OpenEngram(State *state, const char *rank_pack_path)
	{
		char path[1024], name[64];
		const char *slash = strrchr(rank_pack_path, '/');
		const int directory = slash != 0 ? (int)(slash - rank_pack_path) + 1 : 0;
		uint32_t rows[kEngrams] = { 0u, 0u };
		if ( snprintf(path, sizeof(path), "%.*s%s", directory, rank_pack_path, DD_ENGRAM_FILE) >= (int)sizeof(path) )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "the engram table path is too long");
		if ( SparkNamedPackOpen(path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, DD_ALIGNMENT, &state->engram_pack) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "%s: cannot open the engram tables %s beside the rank pack\n", state->shape.tag, path);
			return SPARK_STATUS_NOT_FOUND;
		}
		state->engram_open = 1u;
		state->engram_bytes = state->engram_pack.file_bytes;
		if ( SparkNamedPackConfigU32(&state->engram_pack, "engram_rows_0", &rows[0]) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->engram_pack, "engram_rows_1", &rows[1]) != SPARK_STATUS_OK ||
			rows[0] != state->engram_rows[0] || rows[1] != state->engram_rows[1] )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_TARGET_MISMATCH, "the engram tables are not this checkpoint's two engram modules");
		state->engram_fd = open(path, O_RDONLY);
		if ( state->engram_fd < 0 )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_IO_ERROR, "cannot read the engram tables");
		state->engram_map = mmap(0, (size_t)state->engram_bytes, PROT_READ, MAP_SHARED, state->engram_fd, 0);
		if ( state->engram_map == MAP_FAILED )
		{
			state->engram_map = 0;
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_IO_ERROR, "cannot map the engram tables");
		}
		for ( uint32_t which = 0u; which < kEngrams; which++ )
		{
			SparkNamedPackEntry payload, scale;
			snprintf(name, sizeof(name), "engram.%u.payload", which);
			if ( SparkNamedPackEntryLoad(&state->engram_pack, name, &payload) != SPARK_STATUS_OK || strcmp(payload.kind, "fp8") != 0 ||
				payload.bytes != (uint64_t)rows[which] * kEngramDim )
				return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_SCHEMA_ERROR, "an engram table lacks its fp8 rows");
			snprintf(name, sizeof(name), "engram.%u.scale", which);
			if ( SparkNamedPackEntryLoad(&state->engram_pack, name, &scale) != SPARK_STATUS_OK || strcmp(scale.kind, "e8m0") != 0 ||
				scale.bytes != (uint64_t)rows[which] * kEngramScales )
				return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_SCHEMA_ERROR, "an engram table lacks its e8m0 scales");
			state->engram[which].payload = (const uint8_t *)state->engram_map + state->engram_pack.payload_base + payload.payload_offset;
			state->engram[which].scale = (const uint8_t *)state->engram_map + state->engram_pack.payload_base + scale.payload_offset;
			state->engram[which].rows = rows[which];
		}
		return SPARK_STATUS_OK;
	}

	static void Close(void *model)
	{
		State *state = (State *)model;
		if ( state == 0 )
			return;
		if ( state->engram_map != 0 )
			munmap(state->engram_map, (size_t)state->engram_bytes);
		if ( state->engram_fd >= 0 )
			close(state->engram_fd);
		if ( state->engram_open != 0u )
			SparkNamedPackClose(&state->engram_pack);
		if ( state->engram_ids.handle != 0u )
			(void)SparkDeviceFree(&state->engram_ids);
		if ( state->engram_payload.handle != 0u )
			(void)SparkDeviceFree(&state->engram_payload);
		if ( state->engram_scale.handle != 0u )
			(void)SparkDeviceFree(&state->engram_scale);
		if ( state->engram_local_ids.handle != 0u )
			(void)SparkDeviceFree(&state->engram_local_ids);
		if ( state->pre_init.handle != 0u )
			(void)SparkDeviceFree(&state->pre_init);
		free(state->host_last);
		SparkDeviceStageClose(model);
	}

	static SparkStatus BindHyper(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *prefix, Hyper *hyper)
	{
		char name[48];
		snprintf(name, sizeof(name), "%s_fn", prefix);
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32",
			(uint64_t)SPARK_HC_COMB_MIXES * kResidual * sizeof(float), &hyper->fn);
		snprintf(name, sizeof(name), "%s_base", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", SPARK_HC_COMB_MIXES * sizeof(float), &hyper->base);
		snprintf(name, sizeof(name), "%s_scale", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", 3u * sizeof(float), &hyper->scale);
		return status;
	}

	static SparkStatus BindExpert(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t rows,
		uint64_t columns, SparkDeviceExpertMatrix *matrix)
	{
		char scale_name[48];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "mxfp4_e2m1", kLocalExperts * rows * (columns / 2u),
			&matrix->values);
		matrix->codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
		snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, scale_name, "e8m0", kLocalExperts * rows * (columns / SPARK_MXFP4_GROUP),
				&matrix->scale);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageNoteRouted(&state->stage, matrix->values.bytes + matrix->scale.bytes, kLocalExperts, kTopK);
		return status;
	}

	static SparkStatus BindVector(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t count,
		SparkTensor *tensor)
	{
		return SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", count * sizeof(float), tensor);
	}

	static SparkStatus BindAttention(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, Layer *layer)
	{
		SparkStatus status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "q_a", kQLora, kHidden, &layer->q_a);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "q_norm", kQLora, &layer->q_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "q_b", kQueryWidth, kQLora, &layer->q_b);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "kv", kHeadDim, kHidden, &layer->kv);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "kv_norm", kHeadDim, &layer->kv_norm);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "sinks", kHeads, &layer->sinks);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "o_a", kOutputWidth, kGroupInput, &layer->o_a);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "o_b", kHidden, kOutputWidth, &layer->o_b);
		if ( status != SPARK_STATUS_OK || D41KvSource(index) == 0u )
			return status;
		status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "c_kv", "bf16", (uint64_t)kHeadDim * kHidden * 2u, &layer->c_kv);
		if ( status == SPARK_STATUS_OK && D41Ratio(index) == 2u )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "c_gate", "bf16", (uint64_t)kHeadDim * kHidden * 2u, &layer->c_gate);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "c_norm", kHeadDim, &layer->c_norm);
		return status;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = BindHyper(state, weights, index, "attn_hc", &layer->hyper[D41_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = BindHyper(state, weights, index, "ffn_hc", &layer->hyper[D41_FFN_PART]);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "attn_norm", kHidden, &layer->norm[D41_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "ffn_norm", kHidden, &layer->norm[D41_FFN_PART]);
		if ( status == SPARK_STATUS_OK )
			status = BindAttention(state, weights, index, layer);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router", "bf16", (uint64_t)kExperts * kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = BindVector(state, weights, index, "router_bias", kExperts, &layer->router_bias);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w1", kMoeInter, kHidden, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w3", kMoeInter, kHidden, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w2", kHidden, kMoeInter, &layer->down);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "shared_gate_up", 2u * kSharedRows, kHidden, &layer->shared_gate_up);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "shared_down", kHidden, kSharedRows, &layer->shared_down);
		if ( status != SPARK_STATUS_OK || D41EngramIndex(index) < 0 )
			return status;
		status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "engram_kv", kEngramKvLocal, kEngramEmbed, &layer->engram_kv);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "engram_query", "bf16", (uint64_t)kStreams * kHidden * 2u, &layer->engram_query);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "engram_key", "bf16", (uint64_t)kStreams * kHidden * 2u, &layer->engram_key);
		return status;
	}

	static SparkStatus AllocateShared(SparkDeviceBuffer *buffer, uint64_t bytes, const char *owner)
	{
		const SparkStatus status = SparkDeviceAllocate(buffer, SPARK_DEVICE_SPACE_SHARED_COHERENT, bytes, owner);
		if ( status == SPARK_STATUS_OK )
			memset(buffer->host, 0, (size_t)bytes);
		return status;
	}

	static SparkStatus AllocateHost(State *state)
	{
		const uint64_t rows = state->stage.max_rows, slots = rows * kEngramColumns;
		SparkStatus status = AllocateShared(&state->engram_ids, kEngrams * slots * sizeof(int64_t), "dsv41 engram ids");
		if ( status == SPARK_STATUS_OK )
			status = AllocateShared(&state->engram_payload, kEngrams * slots * kEngramDim, "dsv41 engram rows");
		if ( status == SPARK_STATUS_OK )
			status = AllocateShared(&state->engram_scale, kEngrams * slots * kEngramScales, "dsv41 engram scales");
		if ( status == SPARK_STATUS_OK )
			status = AllocateShared(&state->engram_local_ids, slots * sizeof(int64_t), "dsv41 engram row ids");
		if ( status == SPARK_STATUS_OK )
			status = AllocateShared(&state->pre_init, rows * kStreams * sizeof(float), "dsv41 first collapse");
		if ( status != SPARK_STATUS_OK )
			return SparkDeviceStageRefuse(&state->stage, status, "cannot allocate the engram staging and the first collapse weights");
		for ( uint64_t slot = 0u; slot < slots; slot++ )
			((int64_t *)state->engram_local_ids.host)[slot] = (int64_t)slot;
		for ( uint64_t row = 0u; row < rows; row++ )
			((float *)state->pre_init.host)[row * kStreams] = 1.0f;
		state->host_last = (uint32_t *)calloc((size_t)rows, sizeof(uint32_t));
		return state->host_last != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[DD_SLICES];
		SparkStatus status;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		status = SparkDeviceStageTensor(&state->stage, weights, "final_norm", "f32", kHidden * sizeof(float), &state->final_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, "engram.token_map", "i32", (uint64_t)kVocab * sizeof(int32_t), &state->token_map);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, "engram.multipliers", "i64", (uint64_t)kEngrams * (kOrders + 1u) * sizeof(int64_t),
				&state->multipliers);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, "engram.primes", "i64", (uint64_t)kEngrams * kEngramColumns * sizeof(int64_t), &state->primes);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, "engram.offsets", "i64", (uint64_t)kEngrams * kEngramColumns * sizeof(int64_t),
				&state->offsets);
		for ( uint32_t index = 0u; index < kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayer(state, weights, index);
		if ( status == SPARK_STATUS_OK )
			status = AllocateHost(state);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkDeviceStageAllocate(&state->stage, slices, ScratchSlices(state, slices));
	}

	static SparkHcParams HcParams(const Frame &frame)
	{
		SparkHcParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.streams = kStreams;
		params.width = kHidden;
		params.mixes = SPARK_HC_COMB_MIXES;
		params.norm_epsilon = kEpsilon;
		params.hc_epsilon = SPARK_DSV41_FLASH_MODEL_HC_EPS;
		params.post_magnitude = 2.0f;
		params.iterations = SPARK_DSV41_FLASH_MODEL_HC_SINKHORN_ITERATIONS;
		return params;
	}

	static SparkStatus Linear(State *state, const Frame &frame, SparkTensor x, uint32_t x_stride, const SparkDeviceWeight &w, SparkTensor y, uint32_t y_stride,
		uint32_t input, uint32_t output)
	{
		return SparkDeviceLinearWeight(&state->stage, frame.queue, x, x_stride, &w, y, y_stride, frame.t.rows, input, output);
	}

	static SparkStatus LinearF32(State *state, const Frame &frame, SparkTensor x, SparkTensor w, SparkTensor y, uint32_t output)
	{
		SparkGemvParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.input_dimension = params.input_stride = kHidden;
		params.output_dimension = params.output_stride = output;
		params.output_f32 = 1u;
		params.projection = state->stage.shape->projection;
		params.row_exact = state->stage.step_exact;
		return SparkOpsLinear(state->stage.ops, frame.queue, x, w, y, &params);
	}

	static SparkStatus Norm(State *state, const Frame &frame, SparkTensor input, SparkTensor weight, SparkTensor output, uint32_t width)
	{
		SparkNormParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.dimension = params.input_stride = params.output_stride = width;
		params.epsilon = kEpsilon;
		return SparkOpsRmsNormResidual(state->stage.ops, frame.queue, input, 0, weight, output, &params);
	}

	static SparkStatus Qdq(State *state, const Frame &frame, SparkTensor input, SparkTensor output, uint32_t width, uint32_t format)
	{
		SparkQdqParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.width = params.input_stride = params.output_stride = width;
		params.format = format;
		return SparkOpsBlockQdq(state->stage.ops, frame.queue, input, output, &params);
	}

	static SparkStatus Tail(State *state, const Frame &frame, const SparkHeadRopeParams &base, SparkTensor rows, SparkTensor positions)
	{
		SparkHeadRopeParams params = base;
		params.rows = frame.t.rows;
		return SparkOpsHeadRmsNormRope(state->stage.ops, frame.queue, rows, positions, positions,
			SparkTensorSlice(rows, kNope * sizeof(uint16_t), rows.bytes - kNope * sizeof(uint16_t)), &params);
	}

	static SparkStatus Store(State *state, const Frame &frame, uint32_t kv_layer, SparkTensor rows, SparkTensor positions, uint32_t skip)
	{
		SparkOpsPaged paged = SparkDeviceStagePaged(&state->stage, &frame.t, kv_layer);
		SparkKvParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.kv_heads = 1u;
		params.head_dimension = kHeadDim;
		params.page_slots = paged.page_slots;
		params.pages_per_sequence = paged.pages_per_sequence;
		params.key_stride = params.value_stride = kHeadDim;
		params.pool_pages = paged.pool_pages;
		params.skip = skip;
		paged.positions = positions;
		return SparkOpsKvStore(state->stage.ops, frame.queue, &paged, rows, rows, &params);
	}

	static SparkTensor EngramSlice(const SparkDeviceBuffer *buffer, uint32_t which, uint64_t per_row, uint64_t max_rows, uint32_t rows)
	{
		return SparkTensorAt(buffer, (uint64_t)which * max_rows * per_row, (uint64_t)rows * per_row);
	}

	static SparkNgramParams NgramParams(const State *state, const Frame &frame)
	{
		SparkNgramParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.sequences = frame.t.sequences.sequences;
		params.orders = kOrders;
		params.heads = kEngramHeads;
		params.pad_token = (int32_t)state->engram_pad;
		params.block_token = -1;
		params.block_from = 0u;
		params.history_stride = kHistory;
		params.head_dimension = kEngramDim;
		params.commit = frame.t.sequences.commit;
		params.table_rows = (uint64_t)frame.t.rows * kEngramColumns;
		return params;
	}

	static SparkStatus EngramRows(State *state, Frame &frame)
	{
		const uint32_t rows = frame.t.rows;
		const uint64_t max_rows = state->stage.max_rows;
		const SparkNgramParams params = NgramParams(state, frame);
		const SparkTensor history = SparkDeviceStageRecurrent(&state->stage, 1u, 0u);
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t which = 0u; which < kEngrams && status == SPARK_STATUS_OK; which++ )
			status = SparkOpsNgramHashMapped(state->stage.ops, frame.queue, frame.t.token_ids, state->token_map, frame.t.positions, &frame.t.sequences, history,
				SparkTensorSlice(state->multipliers, (uint64_t)which * (kOrders + 1u) * sizeof(int64_t), (kOrders + 1u) * sizeof(int64_t)),
				SparkTensorSlice(state->primes, (uint64_t)which * kEngramColumns * sizeof(int64_t), kEngramColumns * sizeof(int64_t)),
				SparkTensorSlice(state->offsets, (uint64_t)which * kEngramColumns * sizeof(int64_t), kEngramColumns * sizeof(int64_t)),
				EngramSlice(&state->engram_ids, which, kEngramColumns * sizeof(int64_t), max_rows, rows), &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceQueueWait(frame.queue, SPARK_DEVICE_WAIT_FOREVER);
		for ( uint32_t which = 0u; which < kEngrams && status == SPARK_STATUS_OK; which++ )
		{
			const int64_t *ids = (const int64_t *)state->engram_ids.host + (uint64_t)which * max_rows * kEngramColumns;
			uint8_t *payload = (uint8_t *)state->engram_payload.host + (uint64_t)which * max_rows * kEngramColumns * kEngramDim;
			uint8_t *scale = (uint8_t *)state->engram_scale.host + (uint64_t)which * max_rows * kEngramColumns * kEngramScales;
			const EngramTable &engram = state->engram[which];
			for ( uint64_t slot = 0u; slot < (uint64_t)rows * kEngramColumns; slot++ )
				if ( ids[slot] >= 0 && (uint64_t)ids[slot] < engram.rows )
				{
					memcpy(payload + slot * kEngramDim, engram.payload + (uint64_t)ids[slot] * kEngramDim, kEngramDim);
					memcpy(scale + slot * kEngramScales, engram.scale + (uint64_t)ids[slot] * kEngramScales, kEngramScales);
				}
				else
					status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INTERNAL_ERROR, "an engram hash left its module's table");
		}
		return status;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &frame->t);
		for ( uint32_t row = 0u; status == SPARK_STATUS_OK && row < frame->t.rows; row++ )
		{
			if ( frame->t.host_positions[row] >= kDenseKeys )
				status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_CAPACITY_EXCEEDED,
					"the sparse-attention indexer is not built: a row past position 511 sees more compressed keys than the indexer keeps, where attention is no longer dense");
			else
				state->host_last[row] = (frame->t.host_positions[row] + 1u) / 2u - 1u;
		}
		if ( status == SPARK_STATUS_OK && frame->t.token_ids.bytes == 0u )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "the engram hash needs the step's token ids");
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkHyperParams expand;
		memset(&expand, 0, sizeof(expand));
		expand.rows = frame->t.rows;
		expand.streams = kStreams;
		expand.width = kHidden;
		status = SparkOpsHyperExpand(state->stage.ops, frame->queue, frame->t.hidden, state->streams, &expand);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceCopy(frame->queue, state->pre.range, SparkTensorAt(&state->pre_init, 0u, 0u).range,
				(uint64_t)frame->t.rows * kStreams * sizeof(float));
		for ( uint32_t ratio = 1u; status == SPARK_STATUS_OK && ratio <= 2u; ratio++ )
		{
			SparkCompressParams params;
			memset(&params, 0, sizeof(params));
			params.rows = frame->t.rows;
			params.ratio = ratio;
			status = SparkOpsCompressedPositions(state->stage.ops, frame->queue, frame->t.positions, state->store[RatioSlot(ratio)], state->rope[RatioSlot(ratio)],
				state->last[RatioSlot(ratio)], &params);
		}
		if ( status == SPARK_STATUS_OK )
			status = EngramRows(state, *frame);
		return status;
	}

	static SparkStatus LayerRows(State *state, Frame &frame, uint32_t index)
	{
		(void)state;
		(void)frame;
		(void)index;
		return SPARK_STATUS_OK;
	}

	static bool HasRows(const Frame &frame)
	{
		return frame.t.rows != 0u;
	}

	static void LayerEnd(State *state, Frame &frame, uint32_t index, SparkStatus status)
	{
		(void)frame;
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "%s: layer %u failed status=%d\n", state->shape.tag, index, (int)status);
	}

	static SparkStatus Engram(State *state, Frame &frame, uint32_t which)
	{
		const Layer &layer = state->layers[EngramLayer(which)];
		const uint32_t rows = frame.t.rows;
		const uint64_t max_rows = state->stage.max_rows;
		const SparkNgramParams params = NgramParams(state, frame);
		SparkGateParams gate;
		SparkTensor kv = state->engram_local;
		SparkStatus status = SparkOpsNgramGatherE8m0(state->stage.ops, frame.queue, SparkTensorAt(&state->engram_local_ids, 0u, params.table_rows * sizeof(int64_t)),
			EngramSlice(&state->engram_payload, which, kEngramColumns * kEngramDim, max_rows, rows),
			EngramSlice(&state->engram_scale, which, kEngramColumns * kEngramScales, max_rows, rows), state->engram_embed, &params);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->engram_embed, state->engram_embed, kEngramEmbed, SPARK_QDQ_FP8);
		if ( status == SPARK_STATUS_OK )
			status = Linear(state, frame, state->engram_embed, kEngramEmbed, layer.engram_kv, state->engram_local, kEngramKvLocal, kEngramEmbed, kEngramKvLocal);
		if ( status == SPARK_STATUS_OK && TP > 1u )
		{
			kv = state->engram_gathered;
			status = SparkDeviceStageGatherRows(&state->stage, frame.queue, state->engram_local, state->engram_lanes, kv, rows, kEngramKvLocal);
		}
		memset(&gate, 0, sizeof(gate));
		gate.rows = rows;
		gate.streams = kStreams;
		gate.width = kHidden;
		gate.epsilon = kEpsilon;
		gate.floor = 1e-6f;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsStreamKeyGate(state->stage.ops, frame.queue, state->streams, kv, layer.engram_query, layer.engram_key, &gate);
		return status;
	}

	static SparkStatus HyperMix(State *state, Frame &frame, uint32_t index, D41Part part)
	{
		const Hyper &hyper = state->layers[index].hyper[part];
		const SparkHcParams params = HcParams(frame);
		return SparkOpsHcSiteLagged(state->stage.ops, frame.queue, state->streams, hyper.fn, hyper.base, hyper.scale, state->pre, state->post, state->comb,
			state->mixed, &params);
	}

	static SparkStatus CollapseNorm(State *state, Frame &frame, uint32_t index, D41Part part)
	{
		SparkStatus status = Norm(state, frame, state->mixed, state->layers[index].norm[part], state->normed, kHidden);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->normed, state->quantized, kHidden, SPARK_QDQ_FP8);
		return status;
	}

	static SparkStatus Compress(State *state, Frame &frame, uint32_t index)
	{
		const Layer &layer = state->layers[index];
		const uint32_t ratio = D41Ratio(index), cache = D41CacheIndex(index), slot = RatioSlot(ratio);
		SparkCompressParams params;
		SparkStatus status = LinearF32(state, frame, state->normed, layer.c_kv, state->compressed_kv, kHeadDim);
		if ( status == SPARK_STATUS_OK && ratio == 2u )
			status = LinearF32(state, frame, state->normed, layer.c_gate, state->compressed_score, kHeadDim);
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.sequences = frame.t.sequences.sequences;
		params.width = kHeadDim;
		params.ratio = ratio;
		params.state_stride = 2u * kHeadDim;
		params.commit = frame.t.sequences.commit;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsKvCompress(state->stage.ops, frame.queue, state->compressed_kv, state->compressed_score,
				ratio == 2u ? SparkDeviceStageRecurrent(&state->stage, 0u, cache) : state->compressed_kv, &frame.t.sequences, frame.t.positions, state->pooled, &params);
		if ( status == SPARK_STATUS_OK )
			status = Norm(state, frame, state->pooled, layer.c_norm, state->latent, kHeadDim);
		if ( status == SPARK_STATUS_OK )
			status = Tail(state, frame, state->key_rope[1], state->latent, state->rope[slot]);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->latent, state->latent, kHeadDim, SPARK_QDQ_FP4);
		if ( status == SPARK_STATUS_OK )
			status = Store(state, frame, kLayers + cache, state->latent, state->store[slot], 1u);
		return status;
	}

	static SparkStatus Attend(State *state, Frame &frame, uint32_t index)
	{
		const Layer &layer = state->layers[index];
		const uint32_t ratio = D41Ratio(index);
		SparkOpsPaged paged = SparkDeviceStagePaged(&state->stage, &frame.t, index);
		SparkAttentionParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.heads = kHeads;
		params.kv_heads = 1u;
		params.head_dimension = kHeadDim;
		params.page_slots = paged.page_slots;
		params.pages_per_sequence = paged.pages_per_sequence;
		params.pool_pages = paged.pool_pages;
		params.query_stride = params.output_stride = kQueryWidth;
		params.scale = 1.0f / sqrtf((float)kHeadDim);
		params.has_sinks = 1u;
		params.window = kWindow;
		if ( ratio == 0u )
			return SparkOpsAttention(state->stage.ops, frame.queue, &paged, state->query, state->query, &layer.sinks, state->records, state->attended, &params,
				frame.t.host_positions, frame.t.host_sequence_of_row);
		SparkOpsPaged extra = SparkDeviceStagePaged(&state->stage, &frame.t, kLayers + D41CacheIndex(index));
		extra.positions = state->last[RatioSlot(ratio)];
		return SparkOpsAttentionExtra(state->stage.ops, frame.queue, &paged, &extra, state->query, &layer.sinks, state->records, state->attended, &params,
			frame.t.host_positions, ratio == 2u ? state->host_last : frame.t.host_positions);
	}

	static SparkStatus OutputGroups(State *state, Frame &frame, const Layer &layer)
	{
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t group = 0u; group < kGroups && status == SPARK_STATUS_OK; group++ )
		{
			SparkDeviceWeight part = layer.o_a;
			const uint64_t values = (uint64_t)kOLora * kGroupInput, scales = values / SPARK_MXFP8_GROUP;
			part.values = SparkTensorSlice(layer.o_a.values, group * values, values);
			part.scale = SparkTensorSlice(layer.o_a.scale, group * scales, scales);
			status = Linear(state, frame, SparkTensorSlice(state->attended, (uint64_t)group * kGroupInput * sizeof(uint16_t),
				state->attended.bytes - (uint64_t)group * kGroupInput * sizeof(uint16_t)), kQueryWidth, part,
				SparkTensorSlice(state->wo_a_out, (uint64_t)group * kOLora * sizeof(uint16_t), state->wo_a_out.bytes - (uint64_t)group * kOLora * sizeof(uint16_t)),
				kOutputWidth, kGroupInput, kOLora);
		}
		return status;
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t index)
	{
		const Layer &layer = state->layers[index];
		const uint32_t table = D41Ratio(index) > 0u ? 1u : 0u;
		SparkStatus status = Linear(state, frame, state->quantized, kHidden, layer.q_a, state->q_latent, kQLora, kHidden, kQLora);
		if ( status == SPARK_STATUS_OK )
			status = Norm(state, frame, state->q_latent, layer.q_norm, state->q_normed, kQLora);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->q_normed, state->q_normed, kQLora, SPARK_QDQ_FP8);
		if ( status == SPARK_STATUS_OK )
			status = Linear(state, frame, state->q_normed, kQLora, layer.q_b, state->query, kQueryWidth, kQLora, kQueryWidth);
		if ( status == SPARK_STATUS_OK )
			status = Tail(state, frame, state->query_rope[table], state->query, frame.t.positions);
		if ( status == SPARK_STATUS_OK )
			status = Linear(state, frame, state->quantized, kHidden, layer.kv, state->kv, kHeadDim, kHidden, kHeadDim);
		if ( status == SPARK_STATUS_OK )
			status = Norm(state, frame, state->kv, layer.kv_norm, state->kv, kHeadDim);
		if ( status == SPARK_STATUS_OK )
			status = Tail(state, frame, state->key_rope[table], state->kv, frame.t.positions);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->kv, state->kv, kHeadDim, SPARK_QDQ_FP8);
		if ( status == SPARK_STATUS_OK )
			status = Store(state, frame, index, state->kv, frame.t.positions, 0u);
		if ( status == SPARK_STATUS_OK && D41KvSource(index) != 0u )
			status = Compress(state, frame, index);
		if ( status == SPARK_STATUS_OK )
			status = Attend(state, frame, index);
		if ( status == SPARK_STATUS_OK )
			status = Tail(state, frame, state->output_rope[table], state->attended, frame.t.positions);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->attended, state->attended, kQueryWidth, SPARK_QDQ_FP8);
		if ( status == SPARK_STATUS_OK )
			status = OutputGroups(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->wo_a_out, state->wo_a_out, kOutputWidth, SPARK_QDQ_FP8);
		if ( status == SPARK_STATUS_OK )
			status = Linear(state, frame, state->wo_a_out, kOutputWidth, layer.o_b, state->partial, kHidden, kOutputWidth, kHidden);
		return status;
	}

	static SparkStatus AttentionReduce(State *state, Frame &frame, uint32_t index)
	{
		(void)index;
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus HyperPost(State *state, Frame &frame, D41Part part)
	{
		const SparkHcParams params = HcParams(frame);
		(void)part;
		return SparkOpsHcPostCombF32(state->stage.ops, frame.queue, state->streams, state->post, state->comb, state->partial, &params);
	}

	static SparkStatus SharedExpert(State *state, Frame &frame, const Layer &layer)
	{
		SparkSiluMulParams gated;
		SparkStatus status = Linear(state, frame, state->quantized, kHidden, layer.shared_gate_up, state->ffn.gate_up, 2u * kSharedRows, kHidden, 2u * kSharedRows);
		memset(&gated, 0, sizeof(gated));
		gated.rows = frame.t.rows;
		gated.dimension = kSharedRows;
		gated.input_stride = 2u * kSharedRows;
		gated.output_stride = kSharedRows;
		gated.limit = SPARK_DSV41_FLASH_MODEL_SWIGLU_LIMIT;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsGatedMul(state->stage.ops, frame.queue, state->ffn.gate_up, state->ffn.act, &gated, SPARK_GATED_SILU_GATE_LIMIT);
		if ( status == SPARK_STATUS_OK )
			status = Qdq(state, frame, state->ffn.act, state->ffn.act, kSharedRows, SPARK_QDQ_FP8);
		if ( status == SPARK_STATUS_OK )
			status = Linear(state, frame, state->ffn.act, kSharedRows, layer.shared_down, state->shared_out, kHidden, kSharedRows, kHidden);
		return status;
	}

	static SparkStatus Moe(State *state, Frame &frame, uint32_t index)
	{
		const Layer &layer = state->layers[index];
		const uint32_t rows = frame.t.rows;
		const SparkDeviceRoutedFfnShape ffn = { kLocalExperts, kTopK, kMoeInter, SPARK_GATED_SILU_GATE_LIMIT, 1u, SPARK_DSV41_FLASH_MODEL_SWIGLU_LIMIT, 0.0f, 0u };
		SparkRouteParams route;
		SparkStatus status = LinearF32(state, frame, state->normed, layer.router, state->router_logits, kExperts);
		memset(&route, 0, sizeof(route));
		route.rows = rows;
		route.experts = kExperts;
		route.top_k = kTopK;
		route.width = kHidden;
		route.logit_stride = kExperts;
		route.scale = SPARK_DSV41_FLASH_MODEL_MOE_ROUTED_SCALING_FACTOR;
		route.tree_sum = 1u;
		route.score = SPARK_ROUTE_SCORE_SQRT_SOFTPLUS;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, frame.queue, state->router_logits, layer.router_bias, state->experts.experts, state->experts.weights,
				&route);
		route.local_experts = kLocalExperts;
		route.first_expert = state->stage.tp_rank * kLocalExperts;
		if ( status == SPARK_STATUS_OK && TP > 1u )
			status = SparkOpsRouteLocal(state->stage.ops, frame.queue, state->experts.experts, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfnQdq(&state->stage, frame.queue, &ffn, rows, &layer.gate, &layer.up, &layer.down, &state->experts, state->quantized);
		if ( status == SPARK_STATUS_OK )
			status = SharedExpert(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineShared(state->stage.ops, frame.queue, state->experts.out, state->shared_out, state->partial, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, rows);
		return status;
	}

	static SparkStatus Tap(State *state, Frame &frame, uint32_t index)
	{
		(void)state;
		(void)frame;
		(void)index;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Finish(State *state, Frame &frame, uint32_t index)
	{
		SparkDeviceStageLayerDone(&state->stage, index);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, state->streams, kResidual, index);
	}

	static SparkStatus End(State *state, Frame &frame, SparkStatus status)
	{
		const SparkHcParams params = HcParams(frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkOpsHcCollapse(state->stage.ops, frame.queue, state->streams, state->pre, state->mixed, &params);
		if ( status == SPARK_STATUS_OK )
			status = Norm(state, frame, state->mixed, state->final_norm, frame.t.hidden, kHidden);
		if ( status == SPARK_STATUS_OK )
		{
			const SparkNgramParams history = NgramParams(state, frame);
			status = SparkOpsNgramHistoryMapped(state->stage.ops, frame.queue, frame.t.token_ids, state->token_map, &frame.t.sequences,
				SparkDeviceStageRecurrent(&state->stage, 1u, 0u), &history);
		}
		return status;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return D41Schedule<D41Device>::Step((State *)model, step, queue);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			"sparkpipe_dsv41_flash_device",
			"deepseek-v4.1-flash",
			"official",
			Open,
			SparkDeviceStageManifestCheck,
			SparkDeviceStageBindWeightd,
			Close,
			Step,
			SparkDeviceStageSideband,
			SparkDeviceStageAttachKv,
			SparkDeviceStageRecurrentCopy,
			SparkDeviceStageResetSlot,
			SparkDeviceStageProbeBuffers,
			SparkDeviceStageReport,
			0,
			SparkDeviceStageLane,
			0,
			0,
			BindDevice,
			0
		};
		return &interface_table;
	}
};

const SparkStageRunnerModelInterface *SparkDsv41FlashDeviceStageModelAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return D41Device<1u>::Interface();
	case 2u:
		return D41Device<2u>::Interface();
	case 4u:
		return D41Device<4u>::Interface();
	case 8u:
		return D41Device<8u>::Interface();
	default:
		return 0;
	}
}
