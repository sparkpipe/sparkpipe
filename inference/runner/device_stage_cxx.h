#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_error_site.h"

struct SparkDeviceHyper
{
	SparkTensor fn;
	SparkTensor base;
	SparkTensor scale;
};

static inline uint64_t SparkDeviceStageMax(uint64_t a, uint64_t b)
{
	return a > b ? a : b;
}

template<size_t N>
static inline uint32_t SparkDeviceStageAppendSlices(SparkDeviceStageSlice *slices, uint32_t count, const SparkDeviceStageSlice (&plan)[N])
{
	for ( size_t index = 0u; index < N; index++ )
		slices[count++] = plan[index];
	return count;
}

static inline uint32_t SparkDeviceStageAppendExpertsScratch(const SparkDeviceStage *stage, const SparkDeviceExpertsShape *shape,
	SparkDeviceExpertsScratch *scratch, SparkDeviceStageSlice *slices, uint32_t count)
{
	SparkTensor *targets[] =
	{
		&scratch->routed, &scratch->logits, &scratch->experts, &scratch->weights,
		&scratch->normed, &scratch->gate_up, &scratch->act, &scratch->out, &scratch->sorted
	};
	for ( uint32_t index = 0u; index < (uint32_t)(sizeof(targets) / sizeof(targets[0])); index++ )
	{
		slices[count].bytes = SparkDeviceExpertsScratchBytes(stage, shape, index);
		slices[count++].target = targets[index];
	}
	return count;
}

static inline uint32_t SparkDeviceStageAppendRoutedScratch(SparkDeviceExpertsScratch *scratch, uint64_t routes, uint64_t inter, uint64_t width,
	SparkDeviceStageSlice *slices, uint32_t count)
{
	const SparkDeviceStageSlice plan[] =
	{
		{ routes * sizeof(uint32_t), &scratch->experts },
		{ routes * sizeof(float), &scratch->weights },
		{ routes * 2u * inter * 2u, &scratch->gate_up },
		{ routes * inter * 2u, &scratch->act },
		{ routes * width * 2u, &scratch->out },
		{ (routes + 1u) * width * 2u, &scratch->sorted }
	};
	return SparkDeviceStageAppendSlices(slices, count, plan);
}

static inline uint32_t SparkDeviceStageAppendChannelDeltaScratch(const SparkDeviceStage *stage, uint64_t width, uint32_t heads, uint32_t dim,
	uint64_t low_rank, SparkDeviceChannelDeltaScratch *scratch, SparkDeviceStageSlice *slices, uint32_t count)
{
	const uint64_t rows = stage->max_rows;
	const SparkDeviceStageSlice plan[] =
	{
		{ rows * 3u * width * 2u, &scratch->qkv },
		{ rows * heads * 2u, &scratch->beta },
		{ rows * low_rank * 2u, &scratch->low },
		{ rows * width * 2u, &scratch->decay },
		{ rows * width * 2u, &scratch->gate },
		{ SparkOpsChannelDeltaScratchBytes(stage->max_rows, heads, dim), &scratch->delta },
		{ rows * width * 2u, &scratch->core },
		{ rows * width * 2u, &scratch->gated }
	};
	return SparkDeviceStageAppendSlices(slices, count, plan);
}

static inline SparkStatus SparkDeviceStageCheckWeights(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights)
{
	if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != stage->pack_bytes )
		return SparkDeviceStageRefuse(stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
	return SPARK_STATUS_OK;
}

static inline SparkStatus SparkDeviceStageBindHyper(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, const char *prefix,
	const char *sep, uint32_t mixes, uint32_t residual, uint32_t scales, SparkDeviceHyper *hyper)
{
	char name[64];
	snprintf(name, sizeof(name), "%s%sfn", prefix, sep);
	SparkStatus status = SparkDeviceStageTensor(stage, weights, name, "f32", (uint64_t)mixes * residual * sizeof(float), &hyper->fn);
	snprintf(name, sizeof(name), "%s%sbase", prefix, sep);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageTensor(stage, weights, name, "f32", (uint64_t)mixes * sizeof(float), &hyper->base);
	snprintf(name, sizeof(name), "%s%sscale", prefix, sep);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageTensor(stage, weights, name, "f32", (uint64_t)scales * sizeof(float), &hyper->scale);
	return status;
}

static inline SparkStatus SparkDeviceStageBindLayerHyper(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t index,
	const char *prefix, const char *sep, uint32_t mixes, uint32_t residual, uint32_t scales, SparkDeviceHyper *hyper)
{
	char full[64];
	snprintf(full, sizeof(full), "layers.%u.%s", index, prefix);
	return SparkDeviceStageBindHyper(stage, weights, full, sep, mixes, residual, scales, hyper);
}

static inline SparkStatus SparkDeviceStageBindExpertMatrix(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t index,
	const char *name, const char *kind, const char *scale_sep, const char *scale_kind, uint32_t codec, uint32_t scale_group,
	uint32_t experts, uint32_t top_k, uint64_t values_bytes, uint64_t scale_bytes, SparkDeviceExpertMatrix *matrix)
{
	char scale_name[64];
	SparkStatus status = SparkDeviceStageLayerTensor(stage, weights, index, name, kind, values_bytes, &matrix->values);
	matrix->codec = codec;
	matrix->scale_group = scale_group;
	snprintf(scale_name, sizeof(scale_name), "%s%sscale", name, scale_sep);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage, weights, index, scale_name, scale_kind, scale_bytes, &matrix->scale);
	if ( status == SPARK_STATUS_OK )
		SparkDeviceStageNoteRouted(stage, matrix->values.bytes + matrix->scale.bytes, experts, top_k);
	return status;
}

static inline SparkStatus SparkDeviceStageBindExpertMxfp4(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t index,
	const char *name, const char *scale_sep, uint32_t experts, uint32_t top_k, uint64_t rows, uint64_t columns, uint32_t group,
	SparkDeviceExpertMatrix *matrix)
{
	return SparkDeviceStageBindExpertMatrix(stage, weights, index, name, "mxfp4_e2m1", scale_sep, "e8m0", SPARK_WEIGHT_CODEC_MXFP4_E2M1, 0u,
		experts, top_k, (uint64_t)experts * rows * (columns / 2u), (uint64_t)experts * rows * (columns / group), matrix);
}

static inline SparkStatus SparkDeviceStageBindExpertFp8(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t index,
	const char *name, uint32_t experts, uint32_t top_k, uint64_t rows, uint64_t columns, uint32_t scale_block, SparkDeviceExpertMatrix *matrix)
{
	return SparkDeviceStageBindExpertMatrix(stage, weights, index, name, "fp8", "_", "f32", SPARK_WEIGHT_CODEC_FP8_E4M3, 0u,
		experts, top_k, (uint64_t)experts * rows * columns, (uint64_t)experts * rows * (columns / scale_block) * sizeof(float), matrix);
}

static inline SparkStatus SparkDeviceStageBindExpertMxfp8(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t index,
	const char *name, uint32_t experts, uint32_t top_k, uint64_t rows, uint64_t columns, uint32_t scale_group, SparkDeviceExpertMatrix *matrix)
{
	return SparkDeviceStageBindExpertMatrix(stage, weights, index, name, "fp8", "_", "u8", SPARK_WEIGHT_CODEC_FP8_E4M3, scale_group,
		experts, top_k, (uint64_t)experts * rows * columns, (uint64_t)experts * rows * (columns / scale_group), matrix);
}

static inline SparkStatus SparkDeviceStageLinearF32(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor x, SparkTensor w, SparkTensor y,
	uint32_t rows, uint32_t input_dim, uint32_t output_dim, uint32_t output_f32 = 1u)
{
	SparkGemvParams params;
	memset(&params, 0, sizeof(params));
	params.rows = rows;
	params.input_dimension = params.input_stride = input_dim;
	params.output_dimension = params.output_stride = output_dim;
	params.output_f32 = output_f32;
	params.projection = stage->shape->projection;
	params.row_exact = stage->step_exact;
	return SparkOpsLinear(stage->ops, queue, x, w, y, &params);
}

static inline SparkNormParams SparkDeviceStageNormParams(uint32_t rows, uint32_t width, float epsilon, uint32_t round_first = 0u,
	uint32_t has_addend = 0u, uint32_t weightless = 0u)
{
	SparkNormParams norm;
	memset(&norm, 0, sizeof(norm));
	norm.rows = rows;
	norm.dimension = norm.input_stride = norm.output_stride = width;
	norm.has_addend = has_addend;
	norm.epsilon = epsilon;
	norm.weightless = weightless;
	norm.round_first = round_first;
	return norm;
}

static inline SparkStatus SparkDeviceStageHyperExpand(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor hidden, SparkTensor streams,
	uint32_t rows, uint32_t stream_count, uint32_t width)
{
	SparkHyperParams params;
	memset(&params, 0, sizeof(params));
	params.rows = rows;
	params.streams = stream_count;
	params.width = width;
	return SparkOpsHyperExpand(stage->ops, queue, hidden, streams, &params);
}

static inline SparkStatus SparkDeviceStageRouteSigmoid(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor normed, SparkTensor router,
	SparkTensor bias, SparkTensor logits, SparkDeviceExpertsScratch *scratch, uint32_t rows, uint32_t hidden, uint32_t experts,
	uint32_t local_experts, uint32_t top_k, uint32_t tp_degree, float scale, uint32_t tree_sum, uint32_t groups, uint32_t topk_groups,
	uint32_t score, uint32_t combine_width, SparkRouteParams *out_route)
{
	SparkRouteParams route;
	SparkStatus status = SparkDeviceStageLinearF32(stage, queue, normed, router, logits, rows, hidden, experts);
	memset(&route, 0, sizeof(route));
	route.rows = rows;
	route.experts = experts;
	route.top_k = top_k;
	route.width = combine_width;
	route.logit_stride = experts;
	route.scale = scale;
	route.tree_sum = tree_sum;
	route.groups = groups;
	route.topk_groups = topk_groups;
	route.score = score;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRouteSigmoid(stage->ops, queue, logits, bias, scratch->experts, scratch->weights, &route);
	route.local_experts = local_experts;
	route.first_expert = stage->tp_rank * local_experts;
	if ( status == SPARK_STATUS_OK && tp_degree > 1u && local_experts != experts )
		status = SparkOpsRouteLocal(stage->ops, queue, scratch->experts, &route);
	if ( out_route != 0 )
		*out_route = route;
	return status;
}

static inline SparkStatus SparkDeviceStageKeepRelocateOnly(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *queue)
{
	SparkDeviceStage *stage = (SparkDeviceStage *)model;
	SparkDeviceKeepTensors keep;
	SparkStatus status = SparkDeviceStageKeepTables(stage, step, kept, &keep);
	if ( status == SPARK_STATUS_OK && step->tree != 0u )
		status = SparkDeviceStageKeepRelocate(stage, (SparkDeviceQueue *)queue, step, &keep);
	return status;
}

template<class ShapeFn>
static inline SparkStatus SparkDeviceStageKeepAttentionKv(SparkDeviceStage *stage, void *queue, const SparkStageRunnerStep *step,
	const uint32_t *kept, uint32_t layers, ShapeFn shape_of)
{
	SparkDeviceKeepTensors keep;
	SparkDeviceStepTensors frame;
	SparkStatus status = SparkDeviceStageKeepTables(stage, step, kept, &keep);
	if ( status != SPARK_STATUS_OK || step->tree == 0u || keep.moves == 0u )
		return status;
	status = SparkDeviceStageStepTables(stage, step, &frame);
	for ( uint32_t index = 0u; index < layers && status == SPARK_STATUS_OK; index++ )
	{
		const SparkDeviceAttentionShape &a = shape_of(index);
		SparkOpsPaged paged = SparkDeviceStagePaged(stage, &frame, index);
		SparkKvParams move = { keep.moves, a.kv_heads, a.head_dimension, paged.page_slots, paged.pages_per_sequence, 0u, 0u, 0u, paged.pool_pages, a.value_dimension, 0u, 0u };
		paged.kv_heads = a.kv_heads;
		status = SparkOpsKvMove(stage->ops, (SparkDeviceQueue *)queue, &paged, keep.move_sources, keep.move_destinations, &move);
	}
	return status;
}

static inline SparkStatus SparkDeviceStageKeepIndexKv(SparkDeviceStage *stage, void *queue, const SparkStageRunnerStep *step,
	const uint32_t *kept, uint32_t index_layers, uint32_t index_dim)
{
	SparkDeviceKeepTensors keep;
	SparkDeviceStepTensors frame;
	SparkStatus status = SparkDeviceStageKeepTables(stage, step, kept, &keep);
	if ( status != SPARK_STATUS_OK || step->tree == 0u || keep.moves == 0u )
		return status;
	status = SparkDeviceStageKeepRelocate(stage, (SparkDeviceQueue *)queue, step, &keep);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageStepTables(stage, step, &frame);
	for ( uint32_t index = 0u; index < index_layers && status == SPARK_STATUS_OK; index++ )
	{
		SparkOpsPaged paged = SparkDeviceStagePaged(stage, &frame, 0u);
		SparkKvParams move = { keep.moves, 1u, index_dim / 2u, paged.page_slots, paged.pages_per_sequence, 0u, 0u, 0u, paged.pool_pages, index_dim / 2u, 0u, 0u };
		paged.pool = SparkDeviceStageIndexCache(stage, index).pool;
		paged.kv_heads = 1u;
		status = SparkOpsKvMove(stage->ops, (SparkDeviceQueue *)queue, &paged, keep.move_sources, keep.move_destinations, &move);
	}
	return status;
}

template<class F>
struct SparkDeviceStageModel
{
	static constexpr bool kHasKeepRows = false;
	static constexpr bool kHasCustomClose = false;
	static constexpr bool kTreeVerify = false;
	static constexpr uint32_t kMaxSlices = 48u;

	template<class S>
	static const auto &LayerOf(S *state, uint32_t index)
	{
		return state->layers[index];
	}

	template<class S, class Frame>
	static SparkStatus Begin(S *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		return SparkDeviceStageStepTables(&state->stage, step, &frame->t);
	}

	template<class S, class Frame>
	static SparkStatus Check(S *state, const Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	template<class S, class Frame>
	static SparkStatus RmsNorm(S *state, const Frame &frame, SparkTensor weight)
	{
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, 0, weight, state->normed);
	}

	template<class S, class Frame>
	static SparkStatus AddRmsNorm(S *state, const Frame &frame, SparkTensor weight)
	{
		const SparkTensor addend = state->partial;
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, &addend, weight, state->normed);
	}

	template<class S, class Frame>
	static SparkStatus AllReduce(S *state, const Frame &frame)
	{
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	template<class S, class Frame>
	static SparkStatus AddRows(S *state, const Frame &frame)
	{
		const uint32_t hidden = state->stage.shape->hidden;
		return SparkOpsAddRows(state->stage.ops, frame.queue, frame.t.hidden, state->partial, frame.t.rows, hidden, hidden, hidden);
	}

	template<class S, class Frame>
	static SparkStatus LayerDone(S *state, const Frame &frame, uint32_t index)
	{
		SparkDeviceStageLayerDone(&state->stage, index);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, frame.t.hidden, state->stage.shape->hidden, index);
	}

	template<class S>
	static void PreInit(S *state)
	{
		(void)state;
	}

	template<class S>
	static uint32_t RecurrentPools(const S *state, SparkDeviceStageRecurrentPlan *pools)
	{
		(void)state;
		(void)pools;
		return 0u;
	}

	template<class S>
	static SparkStatus AfterReserve(S *state, const SparkStageRunnerModelOpen *request)
	{
		(void)state;
		(void)request;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static void AfterAllocate(S *state)
	{
		(void)state;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		using State = typename F::State;
		SparkDeviceStageSlice slices[F::kMaxSlices];
		SparkDeviceStageRecurrentPlan pools[SPARK_DEVICE_STAGE_MAX_RECURRENT_POOLS];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		F::PreInit(state);
		F::Shapes(state);
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK )
			status = F::Configure(state, request);
		if ( status == SPARK_STATUS_OK )
		{
			const uint32_t pool_count = F::RecurrentPools(state, pools);
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, F::ScratchSlices(state, slices)),
				pool_count != 0u ? pools : 0, pool_count, request->configuration->state_budget_bytes);
		}
		if ( status == SPARK_STATUS_OK )
			status = F::AfterReserve(state, request);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkDeviceStageGeometry(&state->stage, geometry);
		if constexpr ( F::kTreeVerify )
			geometry->tree_verify = status == SPARK_STATUS_OK ? 1u : 0u;
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		using State = typename F::State;
		State *state = (State *)model;
		SparkDeviceStageSlice slices[F::kMaxSlices];
		SparkStatus status = SparkDeviceStageCheckWeights(&state->stage, weights);
		(void)geometry;
		if ( status == SPARK_STATUS_OK )
			status = F::BindWeights(state, weights);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkDeviceStageAllocate(&state->stage, slices, F::ScratchSlices(state, slices));
		if ( status == SPARK_STATUS_OK )
			F::AfterAllocate(state);
		return status;
	}

	static void (*CloseEntry(void))(void *)
	{
		if constexpr ( F::kHasCustomClose )
			return F::Close;
		else
			return SparkDeviceStageClose;
	}

	static SparkStatus (*KeepRowsEntry(void))(void *, const SparkStageRunnerStep *, const uint32_t *, void *)
	{
		if constexpr ( F::kHasKeepRows )
			return F::KeepRows;
		else
			return 0;
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			F::Tag(),
			F::Name(),
			"official",
			F::Open,
			SparkDeviceStageManifestCheck,
			SparkDeviceStageBindWeightd,
			CloseEntry(),
			F::Step,
			SparkDeviceStageSideband,
			SparkDeviceStageAttachKv,
			SparkDeviceStageRecurrentCopy,
			SparkDeviceStageResetSlot,
			SparkDeviceStageProbeBuffers,
			SparkDeviceStageReport,
			0,
			SparkDeviceStageLane,
			KeepRowsEntry(),
			0,
			F::BindDevice,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0u,
			0u,
			0
		};
		return &interface_table;
	}
};

template<template<class> class Device, template<uint32_t> class P>
static inline const SparkStageRunnerModelInterface *SparkDeviceStageAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return Device<P<1u>>::Interface();
	case 2u:
		return Device<P<2u>>::Interface();
	case 4u:
		return Device<P<4u>>::Interface();
	case 8u:
		return Device<P<8u>>::Interface();
	default:
		return 0;
	}
}

template<template<uint32_t> class Device>
static inline const SparkStageRunnerModelInterface *SparkDeviceStageTpAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return Device<1u>::Interface();
	case 2u:
		return Device<2u>::Interface();
	case 4u:
		return Device<4u>::Interface();
	case 8u:
		return Device<8u>::Interface();
	default:
		return 0;
	}
}
