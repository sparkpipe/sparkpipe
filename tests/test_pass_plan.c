#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_pass_plan.h"

static int failures;

static void check(int condition, const char *what, unsigned long long a, unsigned long long b)
{
	if ( condition )
		return;
	printf("FAIL %s (%llu, %llu)\n", what, a, b);
	failures++;
}

static void plan_case(uint64_t first, uint64_t tokens, uint32_t node_rows)
{
	const uint32_t ranks = 16u, align = 128u;
	const uint64_t pass_max = (uint64_t)ranks * node_rows;
	SparkPassPlan plan;
	const int result = SparkPassPlanTokens(first, tokens, ranks, node_rows, align, &plan);
	check(result == 0, "a valid prompt plans", tokens, node_rows);
	if ( result != 0 )
		return;
	check(plan.passes == (tokens + pass_max - 1u) / pass_max, "the plan uses the fewest passes the node rows allow", tokens, plan.passes);
	check(plan.pass_share % ((uint64_t)ranks * align) == 0u && plan.pass_share <= pass_max, "a pass share is whole aligned rows on every rank and fits the node rows",
		tokens, plan.pass_share);
	check((uint64_t)(plan.passes - 1u) * plan.pass_share < tokens && tokens <= (uint64_t)plan.passes * plan.pass_share, "the passes cover the prompt and none is empty",
		tokens, plan.pass_share);
	check(plan.last_tokens == tokens - (uint64_t)(plan.passes - 1u) * plan.pass_share && plan.last_tokens != 0u && plan.last_tokens <= plan.pass_share,
		"the last pass holds the remainder", tokens, plan.last_tokens);
	check(SparkPassNodeRows(plan.last_tokens, ranks, align) <= node_rows, "the last pass fits the node rows", tokens, plan.last_tokens);
	check(plan.padded_keys == first + tokens - plan.last_tokens + (uint64_t)ranks * SparkPassNodeRows(plan.last_tokens, ranks, align) && plan.padded_keys >= first + tokens,
		"the padded key count covers the last pass's padding", tokens, plan.padded_keys);
	check(plan.last_tokens + (uint64_t)(plan.passes - 1u) * ranks * align >= tokens / plan.passes,
		"passes share the prompt evenly instead of leaving a sliver for the last one", tokens, plan.last_tokens);
}

static double brute_keys(uint64_t first_key, uint64_t pass_tokens, uint64_t first, uint64_t end)
{
	double keys = 0.0;
	for (uint64_t row = first; row < end; row++)
		keys += row < pass_tokens ? (double)(first_key + row + 1u) : (double)(first_key + pass_tokens);
	return keys;
}

static uint64_t next_random(uint64_t *state)
{
	*state = *state * 6364136223846793005ull + 1442695040888963407ull;
	return *state >> 33u;
}

static void balance_cases(void)
{
	SparkPassBalance balance;
	const uint32_t rows = 16384u;
	const uint64_t first_key = 0u, pass_tokens = 16u * rows - 300u;
	uint32_t node, helper;
	uint64_t taken;
	double before, after;
	memset(&balance, 0, sizeof(balance));
	balance.ranks = 16u;
	balance.help_capacity = 16u * rows / 4u;
	SparkPassPlanHelp(&balance, first_key, rows, pass_tokens, 0.005);
	check(balance.help_active == 0u, "no speeds measured yet means no help", 0u, 0u);
	for (node = 0u; node < 16u; node++)
		balance.node_speed[node] = 1.0;
	SparkPassPlanHelp(&balance, first_key, rows, pass_tokens, 0.005);
	check(balance.help_active == 0u, "equal speeds need no help", 0u, 0u);
	balance.node_speed[9] = 0.999;
	SparkPassPlanHelp(&balance, first_key, rows, pass_tokens, 0.005);
	check(balance.help_active == 0u, "a node within the minimum share of the balanced time keeps its work", 9u, 0u);
	balance.node_speed[9] = 0.8;
	before = SparkPassOwnAttention(&balance, 9u, first_key, rows, pass_tokens);
	SparkPassPlanHelp(&balance, first_key, rows, pass_tokens, 0.005);
	check(balance.help_active != 0u, "a slow node gets help", 9u, 0u);
	for (node = 0u; node < 16u; node++)
		for (helper = 0u; helper < 16u; helper++)
			check(node == 9u || balance.help[node * 16u + helper] == 0u, "only the slow node gives work away", node, helper);
	check(balance.help[9u * 16u + 9u] == 0u, "a node does not help itself", 9u, 9u);
	after = SparkPassOwnAttention(&balance, 9u, first_key, rows, pass_tokens);
	check(after < before, "the slow node's own attention shrinks", (unsigned long long)before, (unsigned long long)after);
	balance.help_capacity = 100u;
	SparkPassPlanHelp(&balance, first_key, rows, pass_tokens, 0.005);
	for (helper = 0u; helper < 16u; helper++)
	{
		taken = 0u;
		for (node = 0u; node < 16u; node++)
			taken += balance.help[node * 16u + helper];
		check(taken <= balance.help_capacity, "a helper takes no more rows than its staging holds", helper, taken);
	}
}

int main(void)
{
	static const uint32_t node_rows[] = {2048u, 16384u, 32768u};
	SparkPassPlan plan;
	uint64_t state = 0x5eedull, tokens;
	check(SparkPassNodeRows(1u, 16u, 128u) == 128u && SparkPassNodeRows(16u * 128u, 16u, 128u) == 128u && SparkPassNodeRows(16u * 128u + 1u, 16u, 128u) == 256u,
		"node rows round each rank's share up to the alignment", 0u, 0u);
	check(SparkPassPlanTokens(0u, 0u, 16u, 16384u, 128u, &plan) != 0, "an empty prompt does not plan", 0u, 0u);
	check(SparkPassPlanTokens(0u, 1000u, 16u, 1000u, 128u, &plan) != 0, "node rows off the alignment do not plan", 0u, 0u);
	check(SparkPassPlanTokens(0u, 1000u, 17u, 16384u, 128u, &plan) != 0, "more ranks than the plan holds do not plan", 0u, 0u);
	for (uint32_t index = 0u; index < sizeof(node_rows) / sizeof(node_rows[0]); index++)
	{
		const uint64_t pass_max = 16u * (uint64_t)node_rows[index];
		const uint64_t edges[] = {1u, 2u, 2047u, 2048u, 2049u, pass_max - 1u, pass_max, pass_max + 1u, 2u * pass_max - 1u, 2u * pass_max, 2u * pass_max + 1u, 990881u, 258339u};
		for (uint32_t edge = 0u; edge < sizeof(edges) / sizeof(edges[0]); edge++)
			plan_case(0u, edges[edge], node_rows[index]);
		for (tokens = 1u; tokens < 2100000u; tokens += 1u + next_random(&state) % 7919u)
			plan_case(next_random(&state) % 4u == 0u ? next_random(&state) % 300000u : 0u, tokens, node_rows[index]);
	}
	for (uint32_t trial = 0u; trial < 2000u; trial++)
	{
		const uint64_t first_key = next_random(&state) % 5000u, pass_tokens = 1u + next_random(&state) % 3000u;
		const uint64_t first = next_random(&state) % 3500u, end = first + next_random(&state) % 500u;
		const double formula = SparkPassVisibleKeys(first_key, pass_tokens, first, end), brute = brute_keys(first_key, pass_tokens, first, end);
		check(formula == brute, "visible keys match a row-by-row count, padding included", (unsigned long long)formula, (unsigned long long)brute);
	}
	balance_cases();
	if ( failures != 0 )
	{
		printf("FAIL test_pass_plan: %d checks failed\n", failures);
		return 1;
	}
	printf("PASS test_pass_plan: passes cover every prompt evenly in the fewest passes, padded keys cover the last pass, visible keys match a direct count, and the balancer only moves a slow node's rows within staging capacity\n");
	return 0;
}
