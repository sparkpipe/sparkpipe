#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"

#define RANKS 16u
#define EXPERTS 384u
#define LOCAL (EXPERTS / RANKS)
#define TOPK 6u
#define HIDDEN 5120u
#define RECORD_BYTES ((uint64_t)HIDDEN + HIDDEN / 32u + 8u * TOPK)

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static uint32_t state_word = 20261116u;

static uint32_t Next(void)
{
	state_word ^= state_word << 13u;
	state_word ^= state_word >> 17u;
	state_word ^= state_word << 5u;
	return state_word;
}

static float FromBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static uint16_t ToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

template <typename T> static T *Device(const std::vector<T> &host)
{
	T *device = 0;
	CUDA(cudaMalloc(&device, host.size() * sizeof(T) + 16u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

template <typename T> static std::vector<T> Host(const T *device, size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

#define COPIES 2u
#define HELD ((1u + COPIES) * LOCAL)

static std::vector<uint16_t> expert_table;

static uint16_t ExpertOut(uint32_t token, uint32_t expert, uint32_t element)
{
	if ( expert_table.empty() )
	{
		expert_table.resize((size_t)EXPERTS * HIDDEN);
		for ( auto &value : expert_table )
			value = ToBf16((float)(Next() % 8192u) / 512.0f - 8.0f);
	}
	const uint32_t shifted = element + (uint32_t)(((uint64_t)token * 97u) % HIDDEN);
	return expert_table[(size_t)expert * HIDDEN + (shifted >= HIDDEN ? shifted - HIDDEN : shifted)];
}

static uint32_t Case(uint32_t own, uint32_t rows, uint32_t hot, uint32_t shuffled, uint32_t mode)
{
	std::vector<uint8_t> owner(EXPERTS), place(EXPERTS), backup(COPIES * EXPERTS);
	std::vector<uint16_t> expert_at(RANKS * LOCAL);
	std::vector<std::vector<uint8_t>> group(RANKS, std::vector<uint8_t>(EXPERTS, (uint8_t)HELD));
	for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
	{
		const uint32_t at = shuffled != 0u ? (expert * 7u + 5u) % EXPERTS : expert;
		owner[expert] = (uint8_t)(at / LOCAL);
		place[expert] = (uint8_t)(at % LOCAL);
		backup[expert] = (uint8_t)((owner[expert] + 1u + place[expert] % (RANKS - 1u)) % RANKS);
		backup[EXPERTS + expert] = (uint8_t)((owner[expert] + RANKS / 2u + place[expert] % (RANKS / 2u)) % RANKS);
		expert_at[owner[expert] * LOCAL + place[expert]] = (uint16_t)expert;
	}
	{
		uint32_t placed[COPIES][RANKS] = {{0}};
		for ( uint32_t copy = 0u; copy < COPIES; ++copy )
			for ( uint32_t home = 0u; home < RANKS; ++home )
				for ( uint32_t index = 0u; index < LOCAL; ++index )
				{
					const uint32_t expert = expert_at[home * LOCAL + index], holder = backup[copy * EXPERTS + expert];
					REQUIRE(holder != home && placed[copy][holder] < LOCAL && (copy == 0u || holder != backup[expert]));
					group[holder][expert] = (uint8_t)(LOCAL + copy * LOCAL + placed[copy][holder]++);
				}
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
			for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
				if ( owner[expert] == rank )
					group[rank][expert] = place[expert];
	}
	uint8_t *owner_d = Device(owner), *backup_d = Device(backup);
	const uint32_t shuffle = (own + RANKS - 1u) / RANKS;
	typedef D41ExchangeCounts<RANKS, EXPERTS> Counts;
	typedef D41ExchangeTotals<RANKS, EXPERTS> Totals;
	std::vector<uint32_t> live(RANKS);
	std::vector<std::vector<uint32_t>> owned_experts(RANKS);
	std::vector<std::vector<float>> owned_weights(RANKS);
	std::vector<std::vector<uint8_t>> owned_codes(RANKS), owned_scales(RANKS), moved(RANKS);
	std::vector<Counts> parts(RANKS);
	for ( uint32_t source = 0u; source < RANKS; ++source )
	{
		live[source] = D41ShuffleLive(source, rows, own);
		owned_experts[source].resize((size_t)own * TOPK);
		owned_weights[source].resize((size_t)own * TOPK);
		owned_codes[source].resize((size_t)own * HIDDEN);
		owned_scales[source].resize((size_t)own * (HIDDEN / 32u));
		for ( uint32_t row = 0u; row < own; ++row )
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				uint32_t expert, taken;
				do
				{
					expert = hot != 0u && Next() % 2u == 0u ? Next() % (2u * LOCAL) : Next() % EXPERTS;
					taken = 0u;
					for ( uint32_t j = 0u; j < k; ++j )
						taken |= owned_experts[source][(size_t)row * TOPK + j] == expert ? 1u : 0u;
				}
				while ( taken != 0u );
				owned_experts[source][(size_t)row * TOPK + k] = expert;
				owned_weights[source][(size_t)row * TOPK + k] = (float)(Next() % 1000u) / 997.0f;
			}
		for ( auto &byte : owned_codes[source] )
			byte = (uint8_t)(Next() % 0x7fu);
		for ( auto &byte : owned_scales[source] )
			byte = (uint8_t)(120u + Next() % 12u);
		uint32_t *experts_d = Device(owned_experts[source]);
		float *weights_d = Device(owned_weights[source]);
		uint8_t *codes_d = Device(owned_codes[source]), *scales_d = Device(owned_scales[source]), *send_d = 0;
		Counts *counts_d = 0;
		CUDA(cudaMalloc(&counts_d, sizeof(Counts)));
		CUDA(cudaMemset(counts_d, 0xa5, sizeof(Counts)));
		D41ExchangeCountsKernel<RANKS, EXPERTS><<<1u, 1024u>>>(experts_d, live[source], source * own, TOPK, owner_d, 1u, counts_d);
		CUDA(cudaGetLastError());
		CUDA(cudaMemcpy(&parts[source], counts_d, sizeof(Counts), cudaMemcpyDeviceToHost));
		std::vector<uint32_t> pairs(RANKS * RANKS, 0u), routes(EXPERTS, 0u), sole(RANKS * EXPERTS, 0u);
		for ( uint32_t row = 0u; row < live[source]; ++row )
		{
			const uint32_t target = (source * own + row) % RANKS;
			uint32_t mask = 0u, twice = 0u;
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				const uint32_t home = 1u << owner[owned_experts[source][(size_t)row * TOPK + k]];
				twice |= mask & home;
				mask |= home;
			}
			for ( uint32_t rank = 0u; rank < RANKS; ++rank )
				pairs[target * RANKS + rank] += (mask >> rank) & 1u;
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				const uint32_t expert = owned_experts[source][(size_t)row * TOPK + k];
				routes[expert]++;
				sole[target * EXPERTS + expert] += ((twice >> owner[expert]) & 1u) == 0u ? 1u : 0u;
			}
		}
		for ( uint32_t index = 0u; index < RANKS * RANKS; ++index )
			REQUIRE(parts[source].pair[index] == pairs[index]);
		for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
			REQUIRE(parts[source].routes[expert] == routes[expert]);
		for ( uint32_t index = 0u; index < RANKS * EXPERTS; ++index )
			REQUIRE(parts[source].sole[index] == sole[index]);
		CUDA(cudaMalloc(&send_d, (size_t)RANKS * shuffle * RECORD_BYTES));
		CUDA(cudaMemset(send_d, 0, (size_t)RANKS * shuffle * RECORD_BYTES));
		if ( live[source] != 0u )
			D41ShufflePackKernel<<<live[source], 256u>>>(codes_d, scales_d, experts_d, weights_d, source * own, RANKS, shuffle, HIDDEN, TOPK, RECORD_BYTES,
				send_d);
		CUDA(cudaGetLastError());
		moved[source] = Host(send_d, (size_t)RANKS * shuffle * RECORD_BYTES);
		cudaFree(experts_d); cudaFree(counts_d); cudaFree(weights_d); cudaFree(codes_d); cudaFree(scales_d); cudaFree(send_d);
	}
	Totals totals;
	{
		Counts *gathered_d = 0;
		Totals *totals_d = 0;
		CUDA(cudaMalloc(&gathered_d, RANKS * sizeof(Counts)));
		CUDA(cudaMalloc(&totals_d, sizeof(Totals)));
		CUDA(cudaMemcpy(gathered_d, parts.data(), RANKS * sizeof(Counts), cudaMemcpyHostToDevice));
		D41ExchangeTotalsKernel<RANKS, EXPERTS><<<7u, 1024u>>>((const uint8_t *)gathered_d, sizeof(Counts), 1u, totals_d);
		CUDA(cudaGetLastError());
		CUDA(cudaMemcpy(&totals, totals_d, sizeof(Totals), cudaMemcpyDeviceToHost));
		for ( uint32_t index = 0u; index < RANKS * RANKS + EXPERTS + RANKS * EXPERTS; ++index )
		{
			uint32_t sum = 0u;
			for ( uint32_t sender = 0u; sender < RANKS; ++sender )
				sum += index < RANKS * RANKS ? parts[sender].pair[index] : index < RANKS * RANKS + EXPERTS ? parts[sender].routes[index - RANKS * RANKS] :
					parts[sender].sole[index - RANKS * RANKS - EXPERTS];
			REQUIRE(sum == (index < RANKS * RANKS ? totals.pair[index] : index < RANKS * RANKS + EXPERTS ? totals.routes[index - RANKS * RANKS] :
				(uint32_t)totals.sole[index - RANKS * RANKS - EXPERTS]));
		}
		cudaFree(gathered_d); cudaFree(totals_d);
	}
	std::vector<D41ReceivedRows<RANKS>> moe_in(RANKS);
	std::vector<uint32_t> moe_rows(RANKS);
	std::vector<std::vector<uint32_t>> experts(RANKS), slots(RANKS), token(RANKS);
	std::vector<std::vector<float>> weights(RANKS);
	std::vector<std::vector<uint8_t>> codes(RANKS), scales(RANKS);
	for ( uint32_t target = 0u; target < RANKS; ++target )
	{
		std::vector<uint8_t> received((size_t)RANKS * shuffle * RECORD_BYTES);
		moe_in[target].first[0] = 0u;
		for ( uint32_t source = 0u; source < RANKS; ++source )
		{
			memcpy(received.data() + (size_t)source * shuffle * RECORD_BYTES, moved[source].data() + (size_t)target * shuffle * RECORD_BYTES,
				(size_t)shuffle * RECORD_BYTES);
			moe_in[target].first[source + 1u] = moe_in[target].first[source] + D41ShuffleCount(source, target, rows, own, RANKS);
		}
		const uint32_t count = moe_rows[target] = moe_in[target].first[RANKS], spaces = count != 0u ? count : 1u;
		uint8_t *received_d = Device(received), *codes_d = 0, *scales_d = 0;
		uint32_t *experts_d = 0;
		float *weights_d = 0;
		CUDA(cudaMalloc(&codes_d, (size_t)spaces * HIDDEN));
		CUDA(cudaMalloc(&scales_d, (size_t)spaces * (HIDDEN / 32u)));
		CUDA(cudaMalloc(&experts_d, (size_t)spaces * TOPK * sizeof(uint32_t)));
		CUDA(cudaMalloc(&weights_d, (size_t)spaces * TOPK * sizeof(float)));
		if ( count != 0u )
			D41ShuffleUnpackKernel<RANKS><<<dim3(2u, count), 256u>>>(received_d, moe_in[target], shuffle, HIDDEN, TOPK, RECORD_BYTES, codes_d, scales_d,
				experts_d, weights_d);
		CUDA(cudaGetLastError());
		codes[target] = Host(codes_d, (size_t)count * HIDDEN);
		scales[target] = Host(scales_d, (size_t)count * (HIDDEN / 32u));
		experts[target] = Host(experts_d, (size_t)count * TOPK);
		weights[target] = Host(weights_d, (size_t)count * TOPK);
		token[target].assign(count, UINT32_MAX);
		uint32_t seen = 0u;
		for ( uint32_t source = 0u; source < RANKS; ++source )
			for ( uint32_t row = 0u; row < live[source]; ++row )
			{
				if ( (source * own + row) % RANKS != target )
					continue;
				const size_t q = (size_t)moe_in[target].first[source] + row / RANKS;
				REQUIRE(q < moe_in[target].first[source + 1u]);
				++seen;
				token[target][q] = source * own + row;
				REQUIRE(memcmp(codes[target].data() + q * HIDDEN, owned_codes[source].data() + (size_t)row * HIDDEN, HIDDEN) == 0);
				REQUIRE(memcmp(scales[target].data() + q * (HIDDEN / 32u), owned_scales[source].data() + (size_t)row * (HIDDEN / 32u), HIDDEN / 32u) == 0);
				for ( uint32_t k = 0u; k < TOPK; ++k )
				{
					REQUIRE(experts[target][q * TOPK + k] == owned_experts[source][(size_t)row * TOPK + k]);
					REQUIRE(weights[target][q * TOPK + k] == owned_weights[source][(size_t)row * TOPK + k]);
				}
			}
		REQUIRE(seen == count);
		cudaFree(received_d); cudaFree(codes_d); cudaFree(scales_d); cudaFree(experts_d); cudaFree(weights_d);
	}
	std::vector<uint16_t> quota((size_t)COPIES * RANKS * EXPERTS, 0u);
	std::vector<uint32_t> counts(RANKS * RANKS);
	uint32_t offload = 0u, limit = RANKS * shuffle;
	if ( mode == 1u )
	{
		offload = D41BackupPlan<RANKS, EXPERTS, LOCAL, COPIES>(&totals, owner.data(), backup.data(), expert_at.data(), limit, quota.data(), counts.data());
		for ( uint32_t index = 0u; index < RANKS * RANKS && offload != 0u; ++index )
			REQUIRE(counts[index] <= limit);
	}
	else
	{
		memcpy(counts.data(), totals.pair, sizeof(totals.pair));
		for ( uint32_t target = 0u; target < RANKS && mode == 2u; ++target )
			for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
			{
				uint32_t left = totals.sole[target * EXPERTS + expert];
				for ( uint32_t copy = 0u; copy < COPIES; ++copy )
				{
					const uint32_t take = Next() % 3u == 0u ? left : Next() % (left + 1u);
					quota[((size_t)copy * RANKS + target) * EXPERTS + expert] = (uint16_t)take;
					counts[target * RANKS + owner[expert]] -= take;
					counts[target * RANKS + backup[copy * EXPERTS + expert]] += take;
					offload |= take != 0u ? 1u : 0u;
					left -= take;
				}
			}
	}
	uint64_t offloaded = 0u;
	for ( uint32_t index = 0u; index < RANKS * EXPERTS; ++index )
	{
		uint32_t planned = 0u;
		for ( uint32_t copy = 0u; copy < COPIES; ++copy )
			planned += quota[(size_t)copy * RANKS * EXPERTS + index];
		REQUIRE(planned <= totals.sole[index]);
		offloaded += planned;
	}
	uint32_t capacity = 0u;
	for ( uint32_t value : counts )
		capacity = value > capacity ? value : capacity;
	capacity = capacity != 0u ? (capacity + 7u) / 8u * 8u : 8u;
	REQUIRE(mode == 2u || capacity <= RANKS * shuffle || capacity == 8u);
	std::vector<uint32_t *> slot_d(RANKS);
	for ( uint32_t source = 0u; source < RANKS; ++source )
	{
		const uint32_t routed = moe_rows[source], spaces = routed != 0u ? routed : 1u;
		uint32_t *experts_d = Device(experts[source].empty() ? std::vector<uint32_t>(TOPK, 0u) : experts[source]), *count_d = 0;
		std::vector<uint16_t> own_quota((size_t)COPIES * EXPERTS);
		for ( uint32_t copy = 0u; copy < COPIES; ++copy )
			std::copy(quota.begin() + ((size_t)copy * RANKS + source) * EXPERTS, quota.begin() + ((size_t)copy * RANKS + source + 1u) * EXPERTS,
				own_quota.begin() + (size_t)copy * EXPERTS);
		uint16_t *quota_d = Device(own_quota);
		CUDA(cudaMalloc(&slot_d[source], (size_t)spaces * RANKS * sizeof(uint32_t)));
		CUDA(cudaMalloc(&count_d, RANKS * sizeof(uint32_t)));
		D41ExchangeSlotsKernel<RANKS, EXPERTS, TOPK, COPIES><<<1u, D41K_SLOT_WARPS * 32u>>>(experts_d, routed, owner_d, backup_d, offload != 0u ? quota_d : 0,
			capacity, slot_d[source], count_d);
		CUDA(cudaGetLastError());
		slots[source] = Host(slot_d[source], (size_t)routed * RANKS);
		const std::vector<uint32_t> count = Host(count_d, RANKS);
		std::vector<uint32_t> taken(EXPERTS, 0u);
		std::vector<std::vector<uint8_t>> used(RANKS, std::vector<uint8_t>(capacity, 0u));
		for ( uint32_t row = 0u; row < routed; ++row )
		{
			uint32_t mask = 0u, twice = 0u, target[RANKS];
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				const uint32_t home = 1u << owner[experts[source][(size_t)row * TOPK + k]];
				twice |= mask & home;
				mask |= home;
			}
			for ( uint32_t rank = 0u; rank < RANKS; ++rank )
				target[rank] = rank;
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				const uint32_t expert = experts[source][(size_t)row * TOPK + k];
				if ( ((twice >> owner[expert]) & 1u) == 0u && offload != 0u )
				{
					const uint32_t rank_in_expert = taken[expert]++;
					uint32_t bound = 0u;
					for ( uint32_t copy = 0u; copy < COPIES; ++copy )
					{
						const uint32_t start = bound;
						bound += quota[((size_t)copy * RANKS + source) * EXPERTS + expert];
						if ( rank_in_expert >= start && rank_in_expert < bound )
							target[owner[expert]] = backup[copy * EXPERTS + expert];
					}
				}
			}
			for ( uint32_t home = 0u; home < RANKS; ++home )
			{
				const uint32_t at = slots[source][(size_t)row * RANKS + home];
				if ( ((mask >> home) & 1u) == 0u )
				{
					REQUIRE(at == UINT32_MAX);
					continue;
				}
				REQUIRE(at != UINT32_MAX && at / capacity == target[home] && at % capacity < counts[source * RANKS + target[home]]);
				REQUIRE(used[at / capacity][at % capacity] == 0u);
				used[at / capacity][at % capacity] = 1u;
			}
		}
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
			REQUIRE(count[rank] == counts[source * RANKS + rank]);
		cudaFree(experts_d); cudaFree(count_d); cudaFree(quota_d);
	}
	std::vector<std::vector<uint8_t>> sent(RANKS);
	for ( uint32_t source = 0u; source < RANKS; ++source )
	{
		const uint32_t routed = moe_rows[source];
		uint32_t *experts_d = Device(experts[source].empty() ? std::vector<uint32_t>(TOPK, 0u) : experts[source]);
		float *weights_d = Device(weights[source].empty() ? std::vector<float>(TOPK, 0.0f) : weights[source]);
		uint8_t *codes_d = Device(codes[source].empty() ? std::vector<uint8_t>(HIDDEN, 0u) : codes[source]);
		uint8_t *scales_d = Device(scales[source].empty() ? std::vector<uint8_t>(HIDDEN / 32u, 0u) : scales[source]), *send_d = 0;
		CUDA(cudaMalloc(&send_d, (size_t)RANKS * capacity * RECORD_BYTES));
		CUDA(cudaMemset(send_d, 0, (size_t)RANKS * capacity * RECORD_BYTES));
		if ( routed != 0u )
			D41ExchangePackKernel<<<dim3(routed, RANKS), 256u>>>(codes_d, scales_d, experts_d, weights_d, slot_d[source], owner_d, RANKS, capacity, HIDDEN,
				TOPK, EXPERTS, RECORD_BYTES, send_d);
		CUDA(cudaGetLastError());
		sent[source] = Host(send_d, (size_t)RANKS * capacity * RECORD_BYTES);
		cudaFree(experts_d); cudaFree(weights_d); cudaFree(codes_d); cudaFree(scales_d); cudaFree(send_d);
	}
	std::vector<std::vector<uint16_t>> partial(RANKS);
	uint64_t finalized_wrong = 0u, backup_records = 0u;
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		std::vector<uint8_t> received((size_t)RANKS * capacity * RECORD_BYTES);
		for ( uint32_t source = 0u; source < RANKS; ++source )
			memcpy(received.data() + (size_t)source * capacity * RECORD_BYTES, sent[source].data() + (size_t)rank * capacity * RECORD_BYTES,
				(size_t)capacity * RECORD_BYTES);
		D41ReceivedRows<RANKS> rows_in;
		rows_in.first[0] = 0u;
		for ( uint32_t source = 0u; source < RANKS; ++source )
			rows_in.first[source + 1u] = rows_in.first[source] + counts[(size_t)source * RANKS + rank];
		const uint32_t received_rows = rows_in.first[RANKS], spaces = received_rows != 0u ? received_rows : 1u;
		std::vector<uint32_t> origin_source(spaces, UINT32_MAX), origin_row(spaces, UINT32_MAX), origin_home(spaces, UINT32_MAX);
		for ( uint32_t source = 0u; source < RANKS; ++source )
			for ( uint32_t row = 0u; row < moe_rows[source]; ++row )
				for ( uint32_t home = 0u; home < RANKS; ++home )
				{
					const uint32_t at = slots[source][(size_t)row * RANKS + home];
					if ( at == UINT32_MAX || at / capacity != rank )
						continue;
					const size_t q = (size_t)rows_in.first[source] + at % capacity;
					REQUIRE(q < rows_in.first[source + 1u] && origin_row[q] == UINT32_MAX);
					origin_source[q] = source;
					origin_row[q] = row;
					origin_home[q] = home;
				}
		uint8_t *received_d = Device(received), *group_d = Device(group[rank]);
		uint8_t *codes_d = 0, *scales_d = 0;
		uint16_t *packed_d = 0, *final_d = 0;
		uint32_t *route_d = 0, *packed_row_d = 0, *global_d = 0;
		float *weight_d = 0;
		CUDA(cudaMalloc(&codes_d, (size_t)spaces * HIDDEN));
		CUDA(cudaMalloc(&scales_d, (size_t)spaces * (HIDDEN / 32u)));
		CUDA(cudaMalloc(&route_d, (size_t)spaces * TOPK * sizeof(uint32_t)));
		CUDA(cudaMalloc(&weight_d, (size_t)spaces * TOPK * sizeof(float)));
		CUDA(cudaMalloc(&global_d, (size_t)spaces * TOPK * sizeof(uint32_t)));
		if ( received_rows != 0u )
			D41ExchangeUnpackKernel<RANKS><<<dim3((HIDDEN / 16u + 255u) / 256u, received_rows), 256u>>>(received_d, rows_in, capacity, HIDDEN, TOPK,
				RECORD_BYTES, codes_d, scales_d, route_d, weight_d);
		CUDA(cudaGetLastError());
		const std::vector<uint8_t> unpacked_codes = Host(codes_d, (size_t)received_rows * HIDDEN);
		const std::vector<uint8_t> unpacked_scales = Host(scales_d, (size_t)received_rows * (HIDDEN / 32u));
		const std::vector<uint32_t> route = Host(route_d, (size_t)received_rows * TOPK);
		const std::vector<float> weight = Host(weight_d, (size_t)received_rows * TOPK);
		if ( received_rows != 0u )
			D41RouteLocalKernel<<<(received_rows * TOPK + 255u) / 256u, 256u>>>(route_d, weight_d, global_d, received_rows * TOPK, group_d, LOCAL, HELD,
				EXPERTS);
		CUDA(cudaGetLastError());
		const std::vector<uint32_t> local = Host(route_d, (size_t)received_rows * TOPK), global = Host(global_d, (size_t)received_rows * TOPK);
		const std::vector<float> kept = Host(weight_d, (size_t)received_rows * TOPK);
		for ( uint32_t q = 0u; q < received_rows; ++q )
		{
			const uint32_t source = origin_source[q], row = origin_row[q], home = origin_home[q];
			REQUIRE(row != UINT32_MAX);
			backup_records += home != rank ? 1u : 0u;
			REQUIRE(memcmp(unpacked_codes.data() + (size_t)q * HIDDEN, codes[source].data() + (size_t)row * HIDDEN, HIDDEN) == 0);
			REQUIRE(memcmp(unpacked_scales.data() + (size_t)q * (HIDDEN / 32u), scales[source].data() + (size_t)row * (HIDDEN / 32u), HIDDEN / 32u) == 0);
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				const size_t index = (size_t)q * TOPK + k;
				const uint32_t expert = experts[source][(size_t)row * TOPK + k];
				const uint32_t mine = home == rank ? (owner[expert] == rank ? 1u : 0u) : (owner[expert] == home ? 1u : 0u);
				REQUIRE(route[index] == (home == rank ? expert : owner[expert] == home ? (expert | D41K_BACKUP_FLAG) : EXPERTS));
				REQUIRE(weight[index] == weights[source][(size_t)row * TOPK + k]);
				REQUIRE(global[index] == (home == rank || owner[expert] == home ? expert : EXPERTS));
				REQUIRE(local[index] == (mine != 0u ? group[rank][expert] : HELD));
				REQUIRE(mine == 0u || (home == rank ? local[index] < LOCAL : local[index] >= LOCAL && local[index] < HELD &&
					backup[((local[index] - LOCAL) / LOCAL) * EXPERTS + expert] == rank));
				REQUIRE(kept[index] == (mine != 0u ? weight[index] : 0.0f));
			}
		}
		std::vector<uint16_t> packed((size_t)spaces * TOPK * HIDDEN);
		std::vector<uint32_t> packed_row((size_t)spaces * TOPK);
		for ( size_t index = 0u; index < (size_t)received_rows * TOPK; ++index )
			packed_row[index] = (uint32_t)(((size_t)received_rows * TOPK - 1u - index) % ((size_t)received_rows * TOPK));
		for ( size_t index = 0u; index < packed.size(); ++index )
			packed[index] = ToBf16((float)(Next() % 8192u) / 512.0f - 8.0f);
		for ( uint32_t q = 0u; q < received_rows; ++q )
			for ( uint32_t k = 0u; k < TOPK; ++k )
				if ( local[(size_t)q * TOPK + k] < HELD )
					for ( uint32_t element = 0u; element < HIDDEN; ++element )
						packed[(size_t)packed_row[(size_t)q * TOPK + k] * HIDDEN + element] =
							ExpertOut(token[origin_source[q]][origin_row[q]], global[(size_t)q * TOPK + k], element);
		packed_d = Device(packed);
		packed_row_d = Device(packed_row);
		CUDA(cudaMalloc(&final_d, (size_t)RANKS * capacity * HIDDEN * sizeof(uint16_t)));
		CUDA(cudaMemset(final_d, 0, (size_t)RANKS * capacity * HIDDEN * sizeof(uint16_t)));
		std::vector<uint32_t> places(received_rows + 1u);
		for ( uint32_t q = 0u; q <= received_rows; ++q )
			places[q] = q < received_rows ? received_rows - 1u - q : received_rows + 7u;
		uint32_t *places_d = Device(places);
		D41RecordPlaceKernel<RANKS><<<(received_rows + 1u + 255u) / 256u, 256u>>>(places_d, received_rows + 1u, rows_in, capacity);
		CUDA(cudaGetLastError());
		const std::vector<uint32_t> placed = Host(places_d, (size_t)received_rows + 1u);
		cudaFree(places_d);
		for ( uint32_t q = 0u; q <= received_rows; ++q )
		{
			const uint32_t row = places[q];
			uint32_t source = 0u;
			while ( source < RANKS && row >= rows_in.first[source + 1u] )
				source++;
			REQUIRE(placed[q] == (row < received_rows ? source * capacity + row - rows_in.first[source] : row));
		}
		std::vector<uint32_t> output_row((size_t)spaces * TOPK, UINT32_MAX);
		uint32_t *output_row_d = Device(output_row);
		if ( received_rows != 0u )
			D41FinalizeMapKernel<RANKS><<<(received_rows + 255u) / 256u, 256u>>>(route_d, packed_row_d, rows_in, capacity, TOPK, HELD, received_rows, 0x80000000u,
				output_row_d);
		CUDA(cudaGetLastError());
		output_row = Host(output_row_d, (size_t)spaces * TOPK);
		std::vector<uint16_t> written((size_t)RANKS * capacity * HIDDEN, 0u);
		for ( uint32_t q = 0u; q < received_rows; ++q )
		{
			uint32_t local_routes = 0u;
			for ( uint32_t k = 0u; k < TOPK; ++k )
				local_routes += local[(size_t)q * TOPK + k] < HELD ? 1u : 0u;
			for ( uint32_t k = 0u; k < TOPK; ++k )
			{
				const size_t index = (size_t)q * TOPK + k;
				if ( local[index] >= HELD )
					continue;
				const uint32_t to = output_row[packed_row[index]];
				uint32_t source = 0u;
				while ( q >= rows_in.first[source + 1u] )
					source++;
				const uint32_t record = source * capacity + q - rows_in.first[source];
				REQUIRE(local_routes == 1u ? to == (0x80000000u | record) : to == packed_row[index]);
				if ( local_routes == 1u )
					memcpy(written.data() + (size_t)record * HIDDEN, packed.data() + (size_t)packed_row[index] * HIDDEN, HIDDEN * sizeof(uint16_t));
			}
		}
		CUDA(cudaMemcpy(final_d, written.data(), written.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
		if ( received_rows != 0u )
			D41ExchangeFinalizeKernel<RANKS><<<dim3((HIDDEN / 8u + 255u) / 256u, received_rows), 256u>>>(packed_d, packed_row_d, route_d, rows_in, capacity,
				TOPK, HELD, HIDDEN, 1u, final_d);
		CUDA(cudaGetLastError());
		cudaFree(output_row_d);
		partial[rank] = Host(final_d, (size_t)RANKS * capacity * HIDDEN);
		for ( uint32_t source = 0u; source < RANKS; ++source )
			for ( uint32_t index = 0u; index < capacity; ++index )
			{
				const size_t padded = (size_t)source * capacity + index;
				const uint32_t real = index < counts[(size_t)source * RANKS + rank] ? 1u : 0u;
				for ( uint32_t element = 0u; element < HIDDEN; ++element )
				{
					uint16_t expected = 0u;
					if ( real != 0u )
					{
						const size_t q = (size_t)rows_in.first[source] + index;
						float total = 0.0f;
						uint32_t local_routes = 0u, only = 0u;
						for ( uint32_t k = 0u; k < TOPK; ++k )
							if ( local[q * TOPK + k] < HELD )
							{
								total += FromBf16(packed[(size_t)packed_row[q * TOPK + k] * HIDDEN + element]);
								local_routes++;
								only = packed_row[q * TOPK + k];
							}
						expected = local_routes == 1u ? packed[(size_t)only * HIDDEN + element] : ToBf16(total);
					}
					finalized_wrong += partial[rank][padded * HIDDEN + element] != expected ? 1u : 0u;
				}
			}
		cudaFree(received_d); cudaFree(group_d); cudaFree(codes_d); cudaFree(scales_d); cudaFree(route_d); cudaFree(weight_d); cudaFree(global_d);
		cudaFree(packed_d); cudaFree(packed_row_d); cudaFree(final_d);
	}
	std::vector<std::vector<uint16_t>> summed(RANKS);
	for ( uint32_t target = 0u; target < RANKS; ++target )
	{
		std::vector<uint16_t> landing((size_t)RANKS * capacity * HIDDEN);
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
			memcpy(landing.data() + (size_t)rank * capacity * HIDDEN, partial[rank].data() + (size_t)target * capacity * HIDDEN,
				(size_t)capacity * HIDDEN * sizeof(uint16_t));
		uint16_t *landing_d = Device(landing), *back_d = 0;
		CUDA(cudaMalloc(&back_d, (size_t)RANKS * shuffle * HIDDEN * sizeof(uint16_t)));
		CUDA(cudaMemset(back_d, 0, (size_t)RANKS * shuffle * HIDDEN * sizeof(uint16_t)));
		if ( moe_rows[target] != 0u )
			D41ShuffleSumKernel<RANKS><<<dim3((HIDDEN / 8u + 255u) / 256u, moe_rows[target]), 256u>>>(landing_d, slot_d[target], moe_in[target], shuffle,
				HIDDEN, back_d);
		CUDA(cudaGetLastError());
		summed[target] = Host(back_d, (size_t)RANKS * shuffle * HIDDEN);
		cudaFree(landing_d); cudaFree(back_d);
	}
	uint64_t wrong = 0u;
	for ( uint32_t source = 0u; source < RANKS; ++source )
	{
		std::vector<uint16_t> back((size_t)RANKS * shuffle * HIDDEN), shared((size_t)own * HIDDEN);
		for ( uint32_t target = 0u; target < RANKS; ++target )
			memcpy(back.data() + (size_t)target * shuffle * HIDDEN, summed[target].data() + (size_t)source * shuffle * HIDDEN,
				(size_t)shuffle * HIDDEN * sizeof(uint16_t));
		for ( auto &value : shared )
			value = ToBf16((float)(Next() % 4096u) / 1024.0f - 2.0f);
		uint16_t *back_d = Device(back), *shared_d = Device(shared), *output_d = 0;
		CUDA(cudaMalloc(&output_d, (size_t)own * HIDDEN * sizeof(uint16_t)));
		D41ShuffleFinishKernel<<<dim3((HIDDEN / 8u + 255u) / 256u, own), 256u>>>(back_d, shared_d, live[source], source * own, RANKS, shuffle, HIDDEN,
			output_d);
		CUDA(cudaGetLastError());
		const std::vector<uint16_t> output = Host(output_d, (size_t)own * HIDDEN);
		std::vector<float> rank_sum(RANKS * HIDDEN);
		for ( uint32_t row = 0u; row < own; ++row )
		{
			const uint32_t *row_experts = owned_experts[source].data() + (size_t)row * TOPK, global_row = source * own + row;
			std::fill(rank_sum.begin(), rank_sum.end(), 0.0f);
			uint32_t mask = 0u;
			if ( row < live[source] )
				for ( uint32_t k = 0u; k < TOPK; ++k )
				{
					const uint32_t home = owner[row_experts[k]];
					mask |= 1u << home;
					for ( uint32_t element = 0u; element < HIDDEN; ++element )
						rank_sum[(size_t)home * HIDDEN + element] += FromBf16(ExpertOut(global_row, row_experts[k], element));
				}
			for ( uint32_t element = 0u; element < HIDDEN; ++element )
			{
				uint16_t expected = 0u;
				if ( row < live[source] )
				{
					float total = 0.0f;
					for ( uint32_t home = 0u; home < RANKS; ++home )
						if ( ((mask >> home) & 1u) != 0u )
							total += FromBf16(ToBf16(rank_sum[(size_t)home * HIDDEN + element]));
					expected = ToBf16(FromBf16(ToBf16(total)) + FromBf16(shared[(size_t)row * HIDDEN + element]));
				}
				wrong += output[(size_t)row * HIDDEN + element] != expected ? 1u : 0u;
			}
		}
		cudaFree(back_d); cudaFree(shared_d); cudaFree(output_d);
	}
	for ( uint32_t source = 0u; source < RANKS; ++source )
		cudaFree(slot_d[source]);
	cudaFree(owner_d);
	cudaFree(backup_d);
	printf("exchange own %u rows %u%s%s %s: shuffle %u, capacity %u, %llu routes offloaded in %llu backup records, %llu finalized and %llu combined "
		"values differ\n", own, rows, hot != 0u ? " (two hot ranks)" : "", shuffled != 0u ? " (placed experts)" : "",
		mode == 0u ? "no backups" : mode == 1u ? "planned backups" : "random backups", shuffle, capacity, (unsigned long long)offloaded,
		(unsigned long long)backup_records, (unsigned long long)finalized_wrong, (unsigned long long)wrong);
	return wrong != 0u || finalized_wrong != 0u || backup_records != offloaded || (mode == 1u && hot != 0u && shuffled == 0u && rows != 0u && offloaded == 0u) ? 1u : 0u;
}

static uint32_t EngramCase(uint32_t rows, uint32_t own, uint64_t table_rows, uint32_t invalid)
{
	const uint32_t columns = 24u, dim = D41K_ENGRAM_DIM, record = D41K_ENGRAM_RECORD_BYTES;
	const uint64_t part = (table_rows + RANKS - 1u) / RANKS;
	std::vector<int64_t> ids((size_t)rows * columns);
	uint64_t wrong = 0u, invalid_ids = 0u;
	for ( size_t entry = 0u; entry < ids.size(); ++entry )
	{
		ids[entry] = (int64_t)(((uint64_t)Next() << 20u ^ Next()) % table_rows);
		if ( invalid != 0u && Next() % 16u == 0u )
		{
			ids[entry] = Next() % 2u == 0u ? -1 - (int64_t)(Next() % 5u) : (int64_t)(table_rows + Next() % 7u);
			invalid_ids++;
		}
	}
	int64_t *ids_d = Device(ids);
	uint32_t *slot_d = 0, *counts_d = 0;
	CUDA(cudaMalloc(&slot_d, ids.size() * sizeof(uint32_t)));
	CUDA(cudaMalloc(&counts_d, RANKS * RANKS * sizeof(uint32_t)));
	D41EngramSlotsKernel<RANKS><<<RANKS, 1024u>>>(ids_d, rows, own, columns, part, table_rows, slot_d, counts_d);
	CUDA(cudaGetLastError());
	const std::vector<uint32_t> counts = Host(counts_d, RANKS * RANKS);
	std::vector<uint32_t> expected_counts(RANKS * RANKS, 0u);
	for ( uint32_t row = 0u; row < rows; ++row )
		for ( uint32_t column = 0u; column < columns; ++column )
		{
			const int64_t id = ids[(size_t)row * columns + column];
			if ( id >= 0 && (uint64_t)id < table_rows )
				expected_counts[(size_t)((uint64_t)id / part) * RANKS + row / own]++;
		}
	uint32_t capacity = 0u;
	for ( uint32_t index = 0u; index < RANKS * RANKS; ++index )
	{
		wrong += counts[index] != expected_counts[index] ? 1u : 0u;
		capacity = counts[index] > capacity ? counts[index] : capacity;
	}
	capacity = capacity != 0u ? (capacity + 7u) / 8u * 8u : 8u;
	std::vector<uint8_t *> payload_d(RANKS), scale_d(RANKS), send_d(RANKS);
	std::vector<uint16_t *> dense_d(RANKS);
	std::vector<uint64_t> local_rows(RANKS);
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		local_rows[rank] = (uint64_t)rank * part >= table_rows ? 0u : (table_rows - (uint64_t)rank * part < part ? table_rows - (uint64_t)rank * part : part);
		std::vector<uint8_t> payload((size_t)local_rows[rank] * dim + 16u), scale((size_t)local_rows[rank] * (dim / 32u) + 16u);
		for ( uint8_t &code : payload )
		{
			code = (uint8_t)Next();
			code = (code & 0x7fu) == 0x7fu ? (uint8_t)(code ^ 1u) : code;
		}
		for ( uint8_t &exponent : scale )
			exponent = (uint8_t)(112u + Next() % 24u);
		payload_d[rank] = Device(payload);
		scale_d[rank] = Device(scale);
		CUDA(cudaMalloc(&send_d[rank], (size_t)RANKS * capacity * record));
		CUDA(cudaMemset(send_d[rank], 0xa5, (size_t)RANKS * capacity * record));
		D41EngramPackKernel<<<rows, 256u>>>(ids_d, slot_d, payload_d[rank], scale_d[rank], own, columns, (uint64_t)rank * part, local_rows[rank],
			capacity, send_d[rank]);
		CUDA(cudaGetLastError());
		CUDA(cudaMalloc(&dense_d[rank], (size_t)rows * columns * dim * sizeof(uint16_t)));
		LmNgramGatherE8m0Kernel<<<dim3(rows, columns), 256u>>>(ids_d, payload_d[rank], scale_d[rank], dense_d[rank], (uint64_t)rank * part,
			local_rows[rank], columns, dim);
		CUDA(cudaGetLastError());
	}
	std::vector<std::vector<uint16_t>> dense(RANKS);
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		dense[rank] = Host(dense_d[rank], (size_t)rows * columns * dim);
	uint8_t *received_d = 0;
	uint16_t *embed_d = 0;
	CUDA(cudaMalloc(&received_d, (size_t)RANKS * capacity * record));
	CUDA(cudaMalloc(&embed_d, (size_t)own * columns * dim * sizeof(uint16_t)));
	for ( uint32_t owner = 0u; owner < RANKS; ++owner )
	{
		for ( uint32_t source = 0u; source < RANKS; ++source )
			CUDA(cudaMemcpy(received_d + (size_t)source * capacity * record, send_d[source] + (size_t)owner * capacity * record, (size_t)capacity * record,
				cudaMemcpyDeviceToDevice));
		CUDA(cudaMemset(embed_d, 0x5a, (size_t)own * columns * dim * sizeof(uint16_t)));
		D41EngramUnpackKernel<<<dim3(own, columns), 256u>>>(ids_d, slot_d, received_d, rows, owner * own, columns, part, table_rows, capacity, embed_d);
		CUDA(cudaGetLastError());
		const std::vector<uint16_t> embed = Host(embed_d, (size_t)own * columns * dim);
		for ( uint32_t local = 0u; local < own; ++local )
			for ( uint32_t column = 0u; column < columns; ++column )
			{
				const uint32_t row = owner * own + local;
				const int64_t id = row < rows ? ids[(size_t)row * columns + column] : -1;
				const uint32_t source = id >= 0 && (uint64_t)id < table_rows ? (uint32_t)((uint64_t)id / part) : UINT32_MAX;
				for ( uint32_t element = 0u; element < dim; ++element )
				{
					const uint16_t expected = source != UINT32_MAX ? dense[source][((size_t)row * columns + column) * dim + element] : 0u;
					wrong += embed[((size_t)local * columns + column) * dim + element] != expected ? 1u : 0u;
				}
			}
	}
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		cudaFree(payload_d[rank]); cudaFree(scale_d[rank]); cudaFree(send_d[rank]); cudaFree(dense_d[rank]);
	}
	cudaFree(received_d); cudaFree(embed_d); cudaFree(ids_d); cudaFree(slot_d); cudaFree(counts_d);
	printf("engram exchange rows %u own %u table %llu%s: capacity %u of %u entries a pair at most, %llu values differ\n", rows, own,
		(unsigned long long)table_rows, invalid != 0u ? " (with ids no rank holds)" : "", capacity, own * columns, (unsigned long long)wrong);
	return wrong != 0u || (invalid_ids == 0u && invalid != 0u) ? 1u : 0u;
}

static uint32_t WindowCase(uint32_t rows, uint32_t own, uint32_t edge)
{
	const uint32_t dim = D41K_LATENT, window = 128u;
	std::vector<uint16_t> full((size_t)own * RANKS * dim);
	for ( uint16_t &value : full )
		value = (uint16_t)Next();
	std::vector<uint16_t> edges((size_t)RANKS * edge * dim);
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		memcpy(&edges[(size_t)rank * edge * dim], &full[((size_t)rank * own + own - edge) * dim], (size_t)edge * dim * sizeof(uint16_t));
	uint16_t *edges_d = Device(edges), *window_d = 0;
	CUDA(cudaMalloc(&window_d, full.size() * sizeof(uint16_t)));
	uint64_t wrong = 0u;
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		std::vector<uint16_t> own_kv(&full[(size_t)rank * own * dim], &full[(size_t)(rank + 1u) * own * dim]);
		uint16_t *own_d = Device(own_kv);
		CUDA(cudaMemset(window_d, 0xee, full.size() * sizeof(uint16_t)));
		D41WindowAssembleKernel<<<own + RANKS * edge, 64u>>>(own_d, edges_d, window_d, own, edge, rank * own);
		CUDA(cudaGetLastError());
		const std::vector<uint16_t> window_kv = Host(window_d, full.size());
		const uint32_t first = rank * own, from = first >= window - 1u ? first - (window - 1u) : 0u, to = first + own < rows ? first + own : rows;
		for ( uint32_t row = from; row < to; ++row )
			wrong += memcmp(&window_kv[(size_t)row * dim], &full[(size_t)row * dim], dim * sizeof(uint16_t)) != 0 ? 1u : 0u;
		for ( uint32_t row = rows - window; row < rows; ++row )
			wrong += memcmp(&window_kv[(size_t)row * dim], &full[(size_t)row * dim], dim * sizeof(uint16_t)) != 0 ? 1u : 0u;
		cudaFree(own_d);
	}
	cudaFree(edges_d);
	cudaFree(window_d);
	printf("window edges rows %u own %u edge %u: %llu window rows differ from the full gather\n", rows, own, edge, (unsigned long long)wrong);
	return wrong != 0u ? 1u : 0u;
}

static uint32_t TailCase(uint32_t rows, uint32_t first)
{
	const uint32_t width = 64u, vectors = width / 8u, own = (rows + 127u) / 128u * 8u, tail_rows = rows - first;
	const uint32_t tail_own = (tail_rows + 127u) / 128u * 8u;
	std::vector<uint16_t> full((size_t)rows * width);
	std::vector<uint32_t> weights((size_t)rows * D41K_HC);
	for ( uint16_t &value : full )
		value = (uint16_t)Next();
	for ( uint32_t &value : weights )
		value = Next();
	uint32_t capacity = 0u, low;
	for ( uint32_t source = 0u; source < RANKS; ++source )
		for ( uint32_t target = 0u; target < RANKS; ++target )
		{
			const uint32_t count = D41TailSpan(source, target, rows, first, own, tail_own, &low);
			capacity = count > capacity ? count : capacity;
		}
	std::vector<std::vector<uint4>> sent(RANKS), sent_pre(RANKS);
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		std::vector<uint16_t> streams((size_t)rows * width, 0xeeeeu);
		std::vector<uint32_t> pre((size_t)rows * D41K_HC, 0xeeeeeeeeu);
		const uint32_t from = rank * own < rows ? rank * own : rows, to = (rank + 1u) * own < rows ? (rank + 1u) * own : rows;
		for ( uint32_t row = from; row < to; ++row )
		{
			memcpy(&streams[(size_t)row * width], &full[(size_t)row * width], width * sizeof(uint16_t));
			memcpy(&pre[(size_t)row * D41K_HC], &weights[(size_t)row * D41K_HC], D41K_HC * sizeof(uint32_t));
		}
		uint16_t *streams_d = Device(streams);
		uint32_t *pre_d = Device(pre);
		uint4 *send_d = 0, *send_pre_d = 0;
		CUDA(cudaMalloc(&send_d, (size_t)RANKS * capacity * vectors * sizeof(uint4)));
		CUDA(cudaMalloc(&send_pre_d, (size_t)RANKS * capacity * sizeof(uint4)));
		CUDA(cudaMemset(send_d, 0xcd, (size_t)RANKS * capacity * vectors * sizeof(uint4)));
		CUDA(cudaMemset(send_pre_d, 0xcd, (size_t)RANKS * capacity * sizeof(uint4)));
		D41TailPackKernel<<<dim3(2u, capacity, RANKS), 64u>>>(streams_d, (const float *)pre_d, rank, rows, first, own, tail_own, capacity, width, send_d,
			send_pre_d);
		CUDA(cudaGetLastError());
		sent[rank] = Host(send_d, (size_t)RANKS * capacity * vectors);
		sent_pre[rank] = Host(send_pre_d, (size_t)RANKS * capacity);
		cudaFree(streams_d);
		cudaFree(pre_d);
		cudaFree(send_d);
		cudaFree(send_pre_d);
	}
	uint64_t wrong = 0u;
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		std::vector<uint4> received((size_t)RANKS * capacity * vectors), received_pre((size_t)RANKS * capacity);
		for ( uint32_t source = 0u; source < RANKS; ++source )
		{
			memcpy(&received[(size_t)source * capacity * vectors], &sent[source][(size_t)rank * capacity * vectors], (size_t)capacity * vectors * sizeof(uint4));
			memcpy(&received_pre[(size_t)source * capacity], &sent_pre[source][(size_t)rank * capacity], (size_t)capacity * sizeof(uint4));
		}
		uint4 *received_d = Device(received), *received_pre_d = Device(received_pre);
		uint16_t *streams_d = 0;
		uint32_t *pre_d = 0;
		CUDA(cudaMalloc(&streams_d, (size_t)RANKS * tail_own * width * sizeof(uint16_t)));
		CUDA(cudaMalloc(&pre_d, (size_t)RANKS * tail_own * D41K_HC * sizeof(uint32_t)));
		CUDA(cudaMemset(streams_d, 0xab, (size_t)RANKS * tail_own * width * sizeof(uint16_t)));
		CUDA(cudaMemset(pre_d, 0xab, (size_t)RANKS * tail_own * D41K_HC * sizeof(uint32_t)));
		D41TailUnpackKernel<<<dim3(2u, capacity, RANKS), 64u>>>(received_d, received_pre_d, rank, rows, first, own, tail_own, capacity, width, streams_d,
			(float *)pre_d);
		CUDA(cudaGetLastError());
		const std::vector<uint16_t> streams = Host(streams_d, (size_t)RANKS * tail_own * width);
		const std::vector<uint32_t> pre = Host(pre_d, (size_t)RANKS * tail_own * D41K_HC);
		const uint32_t from = rank * tail_own < tail_rows ? rank * tail_own : tail_rows;
		const uint32_t to = (rank + 1u) * tail_own < tail_rows ? (rank + 1u) * tail_own : tail_rows;
		for ( uint32_t row = from; row < to; ++row )
			wrong += memcmp(&streams[(size_t)row * width], &full[(size_t)(first + row) * width], width * sizeof(uint16_t)) != 0 ||
				memcmp(&pre[(size_t)row * D41K_HC], &weights[(size_t)(first + row) * D41K_HC], D41K_HC * sizeof(uint32_t)) != 0 ? 1u : 0u;
		cudaFree(received_d);
		cudaFree(received_pre_d);
		cudaFree(streams_d);
		cudaFree(pre_d);
	}
	printf("tail rows %u first %u own %u tail own %u capacity %u: %llu tail rows differ from the owners' rows\n", rows, first, own, tail_own, capacity,
		(unsigned long long)wrong);
	return wrong != 0u || capacity > tail_own ? 1u : 0u;
}

int main(void)
{
	uint32_t failures = 0u;
	for ( uint32_t mode = 0u; mode < 3u; ++mode )
	{
		failures += Case(128u, 2048u, 0u, 0u, mode);
		failures += Case(128u, 1997u, 0u, 0u, mode);
		failures += Case(1024u, 16384u, 0u, 0u, mode);
		failures += Case(1024u, 16384u, 1u, 0u, mode);
		failures += Case(1024u, 16383u, 1u, 1u, mode);
		failures += Case(440u, 7000u, 0u, 1u, mode);
		failures += Case(160u, 2541u, 1u, 1u, mode);
		failures += Case(1544u, 24000u, 0u, 1u, mode);
		failures += Case(64u, 0u, 0u, 0u, mode);
	}
	failures += EngramCase(16384u, 1024u, 16u * 20011u + 5u, 1u);
	failures += EngramCase(8191u, 512u, 16u * 20011u, 0u);
	failures += EngramCase(4223u, 264u, 16u * 50u - 7u, 1u);
	failures += WindowCase(16384u, 1024u, 256u);
	failures += WindowCase(16383u, 1024u, 256u);
	failures += WindowCase(8191u, 512u, 256u);
	failures += WindowCase(2049u, 136u, 136u);
	failures += WindowCase(2048u, 128u, 128u);
	failures += TailCase(16384u, 16384u - 2541u);
	failures += TailCase(16383u, 16383u - 2541u);
	failures += TailCase(8192u, 8192u - 2541u);
	failures += TailCase(7000u, 7000u - 2541u);
	failures += TailCase(2600u, 59u);
	failures += TailCase(4096u, 4096u - 300u);
	if ( failures != 0u )
	{
		fprintf(stderr, "FAIL dsv41 exchange: %u cases differ\n", failures);
		return 1;
	}
	printf("PASS dsv41 exchange: every owned row reaches the MoE owner its row index names, modulo the ranks, once, with its codes, scales, experts "
		"and weights, and the owners' counts add up to the MoE owners' own; every MoE row's group of routes into a rank reaches that rank, or the "
		"rank holding one of the two backup copies of the group's only expert as planned, once, by contiguous or placed experts, in row order, and is localized "
		"to the rank's own or backup slots, with its codes, scales, experts and weights, unpacked source by source with no padding rows; each "
		"rank's record with one local route gets that output written straight to its place in the padded return and the finalize sums the other records' local expert outputs in route order into theirs; the MoE "
		"owner sums the groups' partials in rank order and rounds once, and the row's owner adds the shared expert as the publisher does, the "
		"same bits with planned, random or no backups; rows past the live count are zero; every engram row a rank holds reaches the owner of the token row that looks it up, "
		"once, and decodes to the BF16 the holder's own lookup gives, ids no rank holds and padding rows give zero; each rank's own rows, the window "
		"before them and the step's last window assemble from the ranks' edge rows as the full gather gives them; every tail row's streams and stream "
		"weights reach the rank that owns it in the tail's own split, from the rank that owned it in the step, once\n");
	return 0;
}
