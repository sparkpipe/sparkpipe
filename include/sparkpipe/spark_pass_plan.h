#pragma once
#include <stdint.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
#define SPARK_PASS_PLAN_MAX_RANKS 16u
typedef struct SparkPassPlan
{
	uint32_t passes;
	uint32_t reserved0;
	uint64_t pass_share;
	uint64_t last_tokens;
	uint64_t padded_keys;
} SparkPassPlan;
typedef struct SparkPassBalance
{
	uint32_t ranks;
	uint32_t help_active;
	uint64_t help_capacity;
	double node_speed[SPARK_PASS_PLAN_MAX_RANKS];
	uint32_t help[SPARK_PASS_PLAN_MAX_RANKS * SPARK_PASS_PLAN_MAX_RANKS];
} SparkPassBalance;
static inline uint32_t SparkPassNodeRows(uint64_t tokens, uint32_t ranks, uint32_t row_align)
{
	const uint64_t rows = (tokens + ranks - 1u) / ranks;
	return (uint32_t)((rows + row_align - 1u) / row_align * row_align);
}
static inline int SparkPassPlanTokens(uint64_t first_position, uint64_t token_count, uint32_t ranks, uint32_t node_rows, uint32_t row_align, SparkPassPlan *plan)
{
	const uint64_t pass_max = (uint64_t)ranks * node_rows, unit = (uint64_t)ranks * row_align;
	uint64_t passes;
	memset(plan, 0, sizeof(*plan));
	if ( token_count == 0u || ranks == 0u || ranks > SPARK_PASS_PLAN_MAX_RANKS || row_align == 0u || node_rows == 0u || node_rows % row_align != 0u )
		return -1;
	passes = (token_count + pass_max - 1u) / pass_max;
	plan->pass_share = (token_count + unit - 1u) / unit;
	plan->pass_share = (plan->pass_share + passes - 1u) / passes * unit;
	passes = (token_count + plan->pass_share - 1u) / plan->pass_share;
	if ( passes > UINT32_MAX )
		return -1;
	plan->passes = (uint32_t)passes;
	plan->last_tokens = token_count - (passes - 1u) * plan->pass_share;
	plan->padded_keys = first_position + token_count - plan->last_tokens + (uint64_t)ranks * SparkPassNodeRows(plan->last_tokens, ranks, row_align);
	return plan->padded_keys > UINT32_MAX ? -1 : 0;
}
static inline double SparkPassVisibleKeys(uint64_t first_key, uint64_t pass_tokens, uint64_t first, uint64_t end)
{
	const uint64_t real_end = end < pass_tokens ? end : pass_tokens;
	double keys = 0.0;
	if ( first < real_end )
		keys += (double)(real_end - first) * (double)(first_key + 1u) + ((double)first + (double)real_end - 1.0) * (double)(real_end - first) / 2.0;
	if ( end > (first > pass_tokens ? first : pass_tokens) )
		keys += (double)(end - (first > pass_tokens ? first : pass_tokens)) * (double)(first_key + pass_tokens);
	return keys;
}
static inline double SparkPassOwnAttention(const SparkPassBalance *balance, uint32_t node, uint64_t first_key, uint32_t rows, uint64_t pass_tokens)
{
	double keys = 0.0;
	uint32_t slice;
	for (slice = 0u; slice < balance->ranks; slice++)
		keys += SparkPassVisibleKeys(first_key, pass_tokens, (uint64_t)slice * rows,
			(uint64_t)slice * rows + rows - (balance->help_active != 0u && slice != node ? balance->help[node * balance->ranks + slice] : 0u));
	return keys;
}
static inline void SparkPassPlanHelp(SparkPassBalance *balance, uint64_t first_key, uint32_t rows, uint64_t pass_tokens, double min_share)
{
	const uint32_t ranks = balance->ranks;
	const double work = SparkPassVisibleKeys(first_key, pass_tokens, 0u, (uint64_t)ranks * rows);
	double speed = 0.0, balanced, excess[SPARK_PASS_PLAN_MAX_RANKS], room[SPARK_PASS_PLAN_MAX_RANKS], room_total = 0.0;
	uint32_t node, helper;
	memset(balance->help, 0, sizeof(balance->help));
	balance->help_active = 0u;
	if ( ranks == 0u || ranks > SPARK_PASS_PLAN_MAX_RANKS )
		return;
	for (node = 0u; node < ranks; node++)
	{
		if ( !(balance->node_speed[node] > 0.0) )
			return;
		speed += balance->node_speed[node];
	}
	balanced = (double)ranks * work / speed;
	for (node = 0u; node < ranks; node++)
	{
		excess[node] = work - balanced * balance->node_speed[node];
		room[node] = excess[node] < 0.0 ? -excess[node] : 0.0;
		room_total += room[node];
	}
	for (node = 0u; node < ranks && room_total > 0.0; node++)
	{
		if ( excess[node] <= min_share * work )
			continue;
		for (helper = 0u; helper < ranks; helper++)
		{
			const double tail = SparkPassVisibleKeys(first_key, pass_tokens, (uint64_t)(helper + 1u) * rows - 1u, (uint64_t)(helper + 1u) * rows);
			const double share = room[helper] > 0.0 ? excess[node] * room[helper] / room_total : 0.0;
			const double take = tail > 0.0 ? share / tail : 0.0;
			balance->help[node * ranks + helper] = take >= (double)rows ? rows : (uint32_t)take;
		}
	}
	for (helper = 0u; helper < ranks; helper++)
	{
		uint64_t taken = 0u;
		for (node = 0u; node < ranks; node++)
			taken += balance->help[node * ranks + helper];
		for (node = 0u; node < ranks && taken > balance->help_capacity; node++)
			balance->help[node * ranks + helper] = (uint32_t)((uint64_t)balance->help[node * ranks + helper] * balance->help_capacity / taken);
		for (node = 0u; node < ranks; node++)
			balance->help_active |= balance->help[node * ranks + helper] != 0u ? 1u : 0u;
	}
}
#ifdef __cplusplus
}
#endif
