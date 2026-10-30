#pragma once

#include <stdint.h>

#include "inference/llms/kimi_k3/config.h"

#define K3_COLLECTIVE_MLA_QUERY 4u
#define K3_COLLECTIVE_MLA_PARTIALS 5u
#define K3_COLLECTIVE_MLA_DOWN 6u
#define K3_COLLECTIVE_MLA_KEYS 7u
#define K3_COLLECTIVE_FINISH 0x100u
#define K3_COLLECTIVE_BEGIN 0x200u
#define K3_COLLECTIVE_ROWS 0x400u

enum K3Stream : uint32_t
{
	K3_STREAM_MAIN = 0u,
	K3_STREAM_A = 1u,
	K3_STREAM_B = 2u,
	K3_STREAM_NONE = 3u
};

enum K3Lane : uint32_t
{
	K3_LANE_A = 0u,
	K3_LANE_B = 1u
};

enum K3AttnResSite : uint32_t
{
	K3_ATTNRES_ATTENTION = 0u,
	K3_ATTNRES_MLP = 1u,
	K3_ATTNRES_OUT = 2u
};

struct K3Layout
{
	static constexpr bool Kda(uint32_t layer)
	{
		return K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT;
	}
	static constexpr bool Mla(uint32_t layer)
	{
		return K3_LAYER_KIND(layer) == LM_LAYER_LATENT;
	}
	static constexpr bool Routed(uint32_t layer)
	{
		return layer >= K3_FIRST_ROUTED_LAYER;
	}
	static constexpr bool Boundary(uint32_t layer)
	{
		return (layer % K3_ATTNRES_BLOCK_SIZE) == 0u;
	}
	static constexpr uint32_t Bank(uint32_t layer)
	{
		return layer / K3_ATTNRES_BLOCK_SIZE;
	}
	static constexpr uint32_t AttentionSources(uint32_t layer)
	{
		return ((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u;
	}
	static constexpr uint32_t MlpSources(uint32_t layer)
	{
		return (layer / K3_ATTNRES_BLOCK_SIZE) + 2u;
	}
	static constexpr uint32_t OutSources(void)
	{
		return ((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u;
	}
};

template<class B>
struct K3Schedule
{
	using L = K3Layout;
	using Context = typename B::Context;

	static int32_t MlaGather(Context &c, uint32_t layer, uint32_t accumulate)
	{
		uint32_t window;
		int32_t status = B::MlaQuery(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status == B::kOk )
			status = B::MlaGatherPack(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		B::SetGatherWindow(c, K3_LANE_A, 0u);
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS);
		status = B::MlaGate(c, K3_LANE_A, K3_STREAM_MAIN);
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS | K3_COLLECTIVE_FINISH);
		if ( B::MlaPasses(c, K3_LANE_A) <= 1u )
		{
			if ( status == B::kOk && B::GatherExpanded(c, K3_LANE_A) != 0u )
				status = B::MlaExpandValues(c, K3_LANE_A, 0u, K3_STREAM_MAIN);
			if ( status == B::kOk )
				status = B::MlaGatherPartials(c, K3_LANE_A, 0u, K3_STREAM_MAIN);
			if ( status == B::kOk )
				status = B::MlaGatherMerge(c, K3_LANE_A, K3_STREAM_MAIN);
		}
		else
		{
			for ( window = 0u; window < B::MlaPasses(c, K3_LANE_A) && status == B::kOk; window++ )
			{
				if ( window > 0u && B::GatherWindows(c, K3_LANE_A) > 1u )
				{
					B::SetGatherWindow(c, K3_LANE_A, window);
					B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS);
					B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS | K3_COLLECTIVE_FINISH);
				}
				if ( B::GatherExpanded(c, K3_LANE_A) != 0u )
					status = B::MlaExpandValues(c, K3_LANE_A, window, K3_STREAM_MAIN);
				if ( status == B::kOk )
					status = B::MlaGatherPartials(c, K3_LANE_A, window, K3_STREAM_MAIN);
				if ( status == B::kOk )
					status = B::MlaShardFold(c, K3_LANE_A, window, K3_STREAM_MAIN);
			}
			if ( status == B::kOk )
				status = B::MlaShardFinish(c, K3_LANE_A, K3_STREAM_MAIN);
		}
		if ( status != B::kOk )
			return status;
		return B::MlaOutput(c, K3_LANE_A, accumulate, K3_STREAM_MAIN);
	}

	static int32_t MlaShard(Context &c, uint32_t layer, uint32_t accumulate)
	{
		int32_t status;
		if ( !B::HasCollective(c) )
			return B::kShapeError;
		if ( B::ShardGather(c, K3_LANE_A) != 0u )
			return MlaGather(c, layer, accumulate);
		if ( B::MlaDownSliced(c, K3_LANE_A) )
		{
			status = B::MlaDownSlice(c, K3_LANE_A, K3_STREAM_MAIN);
			if ( status != B::kOk )
				return status;
			B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_DOWN);
		}
		else
			status = B::MlaDown(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		status = B::MlaUp(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_QUERY);
		status = B::MlaGate(c, K3_LANE_A, K3_STREAM_MAIN);
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_QUERY | K3_COLLECTIVE_FINISH);
		if ( status != B::kOk )
			return status;
		status = B::MlaShardPartials(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_PARTIALS);
		status = B::MlaShardMerge(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		return B::MlaOutput(c, K3_LANE_A, accumulate, K3_STREAM_MAIN);
	}

	static int32_t Attention(Context &c, uint32_t layer, uint32_t accumulate)
	{
		if ( L::Kda(layer) )
			return B::Kda(c, K3_LANE_A, B::Sequences(c), accumulate, K3_STREAM_MAIN);
		if ( L::Mla(layer) )
		{
			if constexpr ( B::kShardedMla )
				if ( B::KvSharded(c, K3_LANE_A) )
					return MlaShard(c, layer, accumulate);
			return B::MlaLocal(c, K3_LANE_A, accumulate, K3_STREAM_MAIN);
		}
		return B::kShapeError;
	}

	static int32_t CellsMoe(Context &c, uint32_t layer)
	{
		const uint32_t rows = B::Rows(c, K3_LANE_A), packed = B::PackedRows(c);
		int32_t status;
		if ( !B::HasCollective(c) || B::TpSharded(c, K3_LANE_A) == 0u )
			return B::kShapeError;
		status = B::LatentMoe(c, K3_LANE_A, rows, packed, 0u, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 2u);
		status = B::SharedUp(c, K3_LANE_A, K3_STREAM_MAIN);
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 2u | K3_COLLECTIVE_FINISH);
		if ( status == B::kOk )
			status = B::MoeSelect(c, K3_LANE_A, rows, packed, K3_STREAM_MAIN);
		if ( status == B::kOk && B::HasLazy(c) )
			status = B::Acquire(c, K3_LANE_A, layer);
		if ( status == B::kOk )
			status = B::LatentMoe(c, K3_LANE_A, rows, packed, 1u, K3_STREAM_MAIN);
		if ( status != B::kOk )
			return status;
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 3u);
		status = B::SharedDown(c, K3_LANE_A, K3_STREAM_MAIN);
		B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 3u | K3_COLLECTIVE_FINISH);
		if ( status != B::kOk )
			return status;
		return B::LatentMoe(c, K3_LANE_A, rows, packed, 2u, K3_STREAM_MAIN);
	}

	static int32_t LazyMoe(Context &c, uint32_t layer)
	{
		int32_t status = B::MoeRoute(c, K3_LANE_A, K3_STREAM_MAIN);
		if ( status == B::kOk )
			status = B::Acquire(c, K3_LANE_A, layer);
		if ( status == B::kOk )
			status = B::MoeWeighted(c, K3_LANE_A, 0u, K3_STREAM_MAIN);
		if ( status == B::kOk && B::HasCollective(c) )
			B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 2u);
		if ( status == B::kOk )
			status = B::MoeWeighted(c, K3_LANE_A, 1u, K3_STREAM_MAIN);
		if ( status == B::kOk && B::TpSharded(c, K3_LANE_A) != 0u )
		{
			if ( !B::HasCollective(c) )
				status = B::kShapeError;
			else
			{
				B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 3u);
				status = B::MoeWeighted(c, K3_LANE_A, 2u, K3_STREAM_MAIN);
			}
		}
		return status;
	}

	static int32_t ResidentMoe(Context &c, uint32_t layer)
	{
		const uint32_t rows = B::Rows(c, K3_LANE_A), packed = B::PackedRows(c);
		int32_t status = B::LatentMoe(c, K3_LANE_A, rows, packed, 0u, K3_STREAM_MAIN);
		if ( status == B::kOk && B::HasCollective(c) )
			B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 2u);
		if ( status == B::kOk )
			status = B::LatentMoe(c, K3_LANE_A, rows, packed, 1u, K3_STREAM_MAIN);
		if ( status == B::kOk && B::TpSharded(c, K3_LANE_A) != 0u )
		{
			if ( !B::HasCollective(c) )
				status = B::kShapeError;
			else
			{
				B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, 3u);
				status = B::LatentMoe(c, K3_LANE_A, rows, packed, 2u, K3_STREAM_MAIN);
			}
		}
		return status;
	}

	static int32_t Mlp(Context &c, uint32_t layer)
	{
		if ( !L::Routed(layer) )
			return B::Dense(c, K3_LANE_A, K3_STREAM_MAIN);
		if constexpr ( B::kExpertCells )
			if ( B::ExpertCells(c, K3_LANE_A) )
				return CellsMoe(c, layer);
		if constexpr ( B::kLazyExperts )
			if ( B::HasLazy(c) )
				return LazyMoe(c, layer);
		return ResidentMoe(c, layer);
	}

	static void PreAttention(Context &c, uint32_t lane, uint32_t layer, uint32_t stream, bool residual)
	{
		if ( layer > 0u && residual )
			B::AttnRes(c, lane, K3_ATTNRES_ATTENTION, L::AttentionSources(layer), stream);
		if ( L::Boundary(layer) )
		{
			if ( layer == 0u )
				B::PartialSet(c, lane, stream);
			B::BankStore(c, lane, L::Bank(layer), stream);
		}
	}

	static int32_t Slice(Context &c, uint32_t first_layer, uint32_t layer_count)
	{
		uint32_t offset, layer;
		int32_t status;
		for ( offset = 0u; offset < layer_count; ++offset )
		{
			layer = first_layer + offset;
			if ( layer >= K3_LAYERS )
				return B::Failure(layer, "shape", B::kShapeError);
			B::Bind(c, K3_LANE_A, offset, layer);
			PreAttention(c, K3_LANE_A, layer, K3_STREAM_MAIN, true);
			status = Attention(c, layer, L::Boundary(layer) ? 0u : 1u);
			if ( status != B::kOk )
				return B::Failure(layer, "attention", status);
			if ( L::Boundary(layer) )
				B::PartialSetUnsharded(c, K3_LANE_A, K3_STREAM_MAIN);
			if ( B::HasCollective(c) )
			{
				status = B::MlpPrefetchRound(c, layer);
				if ( status != B::kOk )
					return B::Failure(layer, "attention-round", status);
			}
			B::AttnRes(c, K3_LANE_A, K3_ATTNRES_MLP, L::MlpSources(layer), K3_STREAM_MAIN);
			status = Mlp(c, layer);
			if ( status != B::kOk )
				return B::Failure(layer, !L::Routed(layer) ? "dense" : B::HasLazy(c) ? "routed-lazy" : "routed", status);
			if ( B::HasCollective(c) )
			{
				status = B::AttentionPrefetchRound(c, offset, layer, layer_count);
				if ( status != B::kOk )
					return B::Failure(layer, "mlp-round", status);
			}
			B::Release(c, layer);
			B::Tap(c, K3_LANE_A, layer, 0u, K3_STREAM_MAIN);
		}
		if ( first_layer + layer_count == K3_LAYERS )
		{
			B::AttnRes(c, K3_LANE_A, K3_ATTNRES_OUT, L::OutSources(), K3_STREAM_MAIN);
			return B::kOk;
		}
		B::CopyRows(c, K3_LANE_A, K3_STREAM_MAIN);
		return B::kOk;
	}

	static int32_t PairMlaWindows(Context &c, uint32_t layer)
	{
		const uint32_t windows = B::MlaPasses(c, K3_LANE_A), landing = B::GatherWindows(c, K3_LANE_A) > 1u ? 1u : 0u;
		uint32_t window;
		int32_t status = B::kOk;
		if ( windows < 2u || B::MlaPasses(c, K3_LANE_B) != windows || B::GatherWindows(c, K3_LANE_B) != B::GatherWindows(c, K3_LANE_A) ||
			B::GatherExpanded(c, K3_LANE_B) != B::GatherExpanded(c, K3_LANE_A) )
			return B::kShapeError;
		if ( landing != 0u )
		{
			B::SetGatherWindow(c, K3_LANE_A, 1u);
			B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS);
			B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS | K3_COLLECTIVE_FINISH);
		}
		for ( window = 0u; window < windows && status == B::kOk; window++ )
		{
			if ( B::GatherExpanded(c, K3_LANE_A) != 0u )
				status = B::MlaExpandValues(c, K3_LANE_A, window, K3_STREAM_A);
			if ( status == B::kOk )
				status = B::MlaGatherPartials(c, K3_LANE_A, window, K3_STREAM_A);
			if ( status == B::kOk )
				status = B::MlaShardFold(c, K3_LANE_A, window, K3_STREAM_A);
			if ( status == B::kOk )
				status = B::Fence(c, K3_STREAM_B, K3_STREAM_A);
			if ( status == B::kOk )
				status = B::MlaGatherPartials(c, K3_LANE_B, window, K3_STREAM_B);
			if ( status == B::kOk )
				status = B::MlaShardFold(c, K3_LANE_B, window, K3_STREAM_B);
			if ( status == B::kOk && window + 1u < windows )
			{
				status = B::Fence(c, K3_STREAM_A, K3_STREAM_MAIN);
				if ( status == B::kOk )
					status = B::Fence(c, K3_STREAM_A, K3_STREAM_B);
			}
			if ( status == B::kOk && landing != 0u && window + 2u < windows )
			{
				status = B::Fence(c, K3_STREAM_MAIN, K3_STREAM_B);
				B::SetGatherWindow(c, K3_LANE_A, window + 2u);
				B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS);
				B::Collective(c, K3_LANE_A, K3_STREAM_MAIN, layer, K3_COLLECTIVE_MLA_KEYS | K3_COLLECTIVE_FINISH);
			}
		}
		return status;
	}

	static int32_t SlicePair(Context &c, uint32_t first_layer, uint32_t layer_count)
	{
		const uint32_t row_shard = B::RowShard(c, first_layer, layer_count);
		const uint32_t rows_a = B::Rows(c, K3_LANE_A), rows_b = B::Rows(c, K3_LANE_B), total = rows_a + rows_b;
		uint32_t offset, layer = first_layer;
		int32_t status = B::kOk;
		if ( !B::PairReady(c) )
			return B::kShapeError;
#define K3_PAIR_DO(call) do { if ( status == B::kOk ) status = (call); } while ( 0 )
		K3_PAIR_DO(B::Fence(c, K3_STREAM_A, K3_STREAM_MAIN));
		K3_PAIR_DO(B::Fence(c, K3_STREAM_B, K3_STREAM_MAIN));
		for ( offset = 0u; offset < layer_count && status == B::kOk; ++offset )
		{
			layer = first_layer + offset;
			if ( layer >= K3_LAYERS )
				return B::Failure(layer, "shape", B::kShapeError);
			B::Bind(c, K3_LANE_A, offset, layer);
			B::Bind(c, K3_LANE_B, offset, layer);
			PreAttention(c, K3_LANE_A, layer, K3_STREAM_A, row_shard == 0u);
			if ( L::Kda(layer) )
				K3_PAIR_DO(B::Kda(c, K3_LANE_A, 1u, 0u, K3_STREAM_A));
			else
				K3_PAIR_DO(B::MlaQuery(c, K3_LANE_A, K3_STREAM_A));
			K3_PAIR_DO(B::Fence(c, K3_STREAM_B, K3_STREAM_A));
			if ( L::Kda(layer) )
			{
				K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_A, layer, row_shard, 0u));
				PreAttention(c, K3_LANE_B, layer, K3_STREAM_B, row_shard == 0u);
				K3_PAIR_DO(B::Kda(c, K3_LANE_B, 1u, 0u, K3_STREAM_B));
			}
			else
			{
				PreAttention(c, K3_LANE_B, layer, K3_STREAM_B, row_shard == 0u);
				K3_PAIR_DO(B::MlaQuery(c, K3_LANE_B, K3_STREAM_B));
				K3_PAIR_DO(B::MlaGatherPack(c, K3_LANE_A, K3_STREAM_B));
				B::SetGatherWindow(c, K3_LANE_A, 0u);
				if ( B::GatherPrefetched(c, K3_LANE_A) != 0u )
					K3_PAIR_DO(B::Fence(c, K3_STREAM_MAIN, K3_STREAM_B));
				else
					K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_B, K3_STREAM_NONE, layer, K3_COLLECTIVE_MLA_KEYS, 1u));
				B::SetGatherPrefetched(c, K3_LANE_A, 0u);
				K3_PAIR_DO(B::MlaGate(c, K3_LANE_A, K3_STREAM_A));
				K3_PAIR_DO(B::Fence(c, K3_STREAM_A, K3_STREAM_MAIN));
				K3_PAIR_DO(B::MlaGate(c, K3_LANE_B, K3_STREAM_B));
				if ( B::MlaPasses(c, K3_LANE_A) <= 1u )
				{
					K3_PAIR_DO(B::Fence(c, K3_STREAM_B, K3_STREAM_MAIN));
					if ( B::GatherExpanded(c, K3_LANE_A) != 0u )
						K3_PAIR_DO(B::MlaExpandValues(c, K3_LANE_A, 0u, K3_STREAM_A));
					K3_PAIR_DO(B::MlaGatherPartials(c, K3_LANE_A, 0u, K3_STREAM_A));
					K3_PAIR_DO(B::MlaGatherMerge(c, K3_LANE_A, K3_STREAM_A));
					K3_PAIR_DO(B::MlaOutput(c, K3_LANE_A, 0u, K3_STREAM_A));
					K3_PAIR_DO(B::Fence(c, K3_STREAM_B, K3_STREAM_A));
					K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_A, layer, row_shard, 0u));
					K3_PAIR_DO(B::MlaGatherPartials(c, K3_LANE_B, 0u, K3_STREAM_B));
					K3_PAIR_DO(B::MlaGatherMerge(c, K3_LANE_B, K3_STREAM_B));
					K3_PAIR_DO(B::MlaOutput(c, K3_LANE_B, 0u, K3_STREAM_B));
				}
				else
				{
					K3_PAIR_DO(PairMlaWindows(c, layer));
					K3_PAIR_DO(B::MlaShardFinish(c, K3_LANE_A, K3_STREAM_A));
					K3_PAIR_DO(B::MlaOutput(c, K3_LANE_A, 0u, K3_STREAM_A));
					K3_PAIR_DO(B::Fence(c, K3_STREAM_B, K3_STREAM_A));
					K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_A, layer, row_shard, 0u));
					K3_PAIR_DO(B::MlaShardFinish(c, K3_LANE_B, K3_STREAM_B));
					K3_PAIR_DO(B::MlaOutput(c, K3_LANE_B, 0u, K3_STREAM_B));
				}
			}
			K3_PAIR_DO(B::PairRound(c, K3_LANE_B, K3_STREAM_B, K3_STREAM_B, layer, row_shard, 0u));
			if ( status != B::kOk )
				return B::Failure(layer, "pair-attention", status);
			B::SetNextAttnRes(c, K3_LANE_A, offset, layer_count);
			B::SetNextAttnRes(c, K3_LANE_B, offset, layer_count);
			if ( row_shard == 0u )
				B::AttnRes(c, K3_LANE_A, K3_ATTNRES_MLP, L::MlpSources(layer), K3_STREAM_A);
			if ( !L::Routed(layer) )
			{
				K3_PAIR_DO(B::Dense(c, K3_LANE_A, K3_STREAM_A));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_A, layer, 1u | row_shard, 0u));
				if ( row_shard == 0u )
					B::AttnRes(c, K3_LANE_B, K3_ATTNRES_MLP, L::MlpSources(layer), K3_STREAM_B);
				K3_PAIR_DO(B::Dense(c, K3_LANE_B, K3_STREAM_B));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_B, K3_STREAM_B, K3_STREAM_B, layer, 1u | row_shard, 0u));
				if ( status != B::kOk )
					return B::Failure(layer, "pair-dense", status);
			}
			else
			{
				K3_PAIR_DO(B::LatentMoe(c, K3_LANE_A, rows_a, rows_a * K3_TOP_K, 0u, K3_STREAM_A));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_NONE, layer, 2u, 1u));
				K3_PAIR_DO(B::SharedUp(c, K3_LANE_A, K3_STREAM_A));
				K3_PAIR_DO(B::Fence(c, K3_STREAM_A, K3_STREAM_MAIN));
				if ( row_shard == 0u )
					B::AttnRes(c, K3_LANE_B, K3_ATTNRES_MLP, L::MlpSources(layer), K3_STREAM_B);
				K3_PAIR_DO(B::LatentMoe(c, K3_LANE_B, rows_b, rows_b * K3_TOP_K, 0u, K3_STREAM_B));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_B, K3_STREAM_B, K3_STREAM_NONE, layer, 2u, 1u));
				K3_PAIR_DO(B::SharedUp(c, K3_LANE_B, K3_STREAM_B));
				K3_PAIR_DO(B::Fence(c, K3_STREAM_A, K3_STREAM_MAIN));
				K3_PAIR_DO(B::Fence(c, K3_STREAM_A, K3_STREAM_B));
				if ( status == B::kOk && offset + 1u < layer_count && L::Mla(layer + 1u) && B::ShardGather(c, K3_LANE_A) != 0u &&
					B::GatherWindows(c, K3_LANE_A) > 1u && B::GatherHistory(c, K3_LANE_A) != 0u )
				{
					K3_PAIR_DO(B::MlaPrefetchPack(c, K3_LANE_A, layer + 1u, K3_STREAM_B));
					B::SetGatherWindow(c, K3_LANE_A, 0u);
					K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_B, K3_STREAM_NONE, layer + 1u, K3_COLLECTIVE_MLA_KEYS, 1u));
					if ( status == B::kOk )
						B::SetGatherPrefetched(c, K3_LANE_A, 1u);
				}
				K3_PAIR_DO(B::MoeSelect(c, K3_LANE_A, total, total * K3_TOP_K, K3_STREAM_A));
				if ( status == B::kOk && B::HasLazy(c) )
					status = B::Acquire(c, K3_LANE_A, layer);
				K3_PAIR_DO(B::LatentMoe(c, K3_LANE_A, total, total * K3_TOP_K, 1u, K3_STREAM_A));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_A, layer, 3u, 1u));
				K3_PAIR_DO(B::SharedDown(c, K3_LANE_B, K3_STREAM_B));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_B, K3_STREAM_A, K3_STREAM_B, layer, 3u, 1u));
				K3_PAIR_DO(B::SharedDown(c, K3_LANE_A, K3_STREAM_A));
				K3_PAIR_DO(B::LatentMoe(c, K3_LANE_A, rows_a, rows_a * K3_TOP_K, 2u, K3_STREAM_A));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_A, K3_STREAM_A, K3_STREAM_A, layer, 1u | row_shard, 0u));
				K3_PAIR_DO(B::LatentMoe(c, K3_LANE_B, rows_b, rows_b * K3_TOP_K, 2u, K3_STREAM_B));
				K3_PAIR_DO(B::PairRound(c, K3_LANE_B, K3_STREAM_B, K3_STREAM_B, layer, 1u | row_shard, 0u));
				if ( status != B::kOk )
					return B::Failure(layer, "pair-routed", status);
			}
			B::Release(c, layer);
			if ( row_shard == 0u )
			{
				B::Tap(c, K3_LANE_A, layer, 0u, K3_STREAM_A);
				B::Tap(c, K3_LANE_B, layer, rows_a, K3_STREAM_B);
			}
		}
		if ( status != B::kOk )
			return B::Failure(layer, "pair", status);
		if ( first_layer + layer_count == K3_LAYERS )
		{
			if ( row_shard == 0u )
			{
				B::AttnRes(c, K3_LANE_A, K3_ATTNRES_OUT, L::OutSources(), K3_STREAM_A);
				B::AttnRes(c, K3_LANE_B, K3_ATTNRES_OUT, L::OutSources(), K3_STREAM_B);
			}
		}
		else
		{
			B::CopyRows(c, K3_LANE_A, K3_STREAM_A);
			B::CopyRows(c, K3_LANE_B, K3_STREAM_B);
		}
		K3_PAIR_DO(B::Fence(c, K3_STREAM_MAIN, K3_STREAM_A));
		K3_PAIR_DO(B::Fence(c, K3_STREAM_MAIN, K3_STREAM_B));
#undef K3_PAIR_DO
		return status;
	}

	static int32_t SliceHalf(Context &c, uint32_t layer, uint32_t phase)
	{
		const uint32_t rows = B::Rows(c, K3_LANE_A), packed = B::PackedRows(c);
		int32_t status;
		if ( phase == 0u )
		{
			B::Bind(c, K3_LANE_A, 0u, layer);
			PreAttention(c, K3_LANE_A, layer, K3_STREAM_MAIN, true);
			status = Attention(c, layer, L::Boundary(layer) ? 0u : 1u);
			if ( status != B::kOk )
				return status;
			if ( L::Boundary(layer) )
				B::PartialSetUnsharded(c, K3_LANE_A, K3_STREAM_MAIN);
			return B::kOk;
		}
		if ( !L::Routed(layer) )
		{
			B::AttnRes(c, K3_LANE_A, K3_ATTNRES_MLP, L::MlpSources(layer), K3_STREAM_MAIN);
			return B::Dense(c, K3_LANE_A, K3_STREAM_MAIN);
		}
		if ( phase == 1u )
		{
			B::AttnRes(c, K3_LANE_A, K3_ATTNRES_MLP, L::MlpSources(layer), K3_STREAM_MAIN);
			return B::LatentMoe(c, K3_LANE_A, rows, packed, 0u, K3_STREAM_MAIN);
		}
		if ( phase == 2u )
			return B::LatentMoe(c, K3_LANE_A, rows, packed, 1u, K3_STREAM_MAIN);
		return B::LatentMoe(c, K3_LANE_A, rows, packed, 2u, K3_STREAM_MAIN);
	}
};
