/* Pin and self-test the model-neutral serial-TP replay harness.
 *
 * Host paths only: bf16 conversion, the four collective emulations, the
 * budget-checked serial sweep, and the golden compare are all exercised here
 * with no CUDA. A caller wires the CUDA side through its module API and the
 * TP_STANDALONE-style collective no-op (see docs/serial_tp_replay.md); this
 * test proves the host half the caller builds on.
 *
 * Returns 0 on pass, non-zero on the first failure.
 */
#include "serial_tp_replay.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PIN(expr) _Static_assert((expr), #expr)

/* ABI + budget pins */
PIN(SPARK_SERIAL_TP_REPLAY_ABI_VERSION == 1u);
PIN(SPARK_SERIAL_TP_REPLAY_DEFAULT_DEVICE_BUDGET_BYTES == (108ull << 30));
PIN(SPARK_SERIAL_TP_COLLECTIVE_ALL_REDUCE_SUM_BF16 == 0u);
PIN(SPARK_SERIAL_TP_COLLECTIVE_REDUCE_SCATTER_BF16 == 1u);
PIN(SPARK_SERIAL_TP_COLLECTIVE_U64_MAXLOC == 2u);
PIN(SPARK_SERIAL_TP_COLLECTIVE_IDENTITY == 3u);

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

/* bf16 round-trip through fp32 must be exact for bf16-representable values. */
static void test_bf16_roundtrip(void)
{
	uint16_t code;
	for (code = 0u; code < 0x8000u; code += 97u)
	{
		uint16_t back = spark_serial_tp_f32_to_bf16(
			spark_serial_tp_bf16_to_f32(code));
		if (back != code)
		{
			fprintf(stderr, "FAIL bf16 roundtrip 0x%04x -> 0x%04x\n", code, back);
			failures++;
			return;
		}
	}
	CHECK(spark_serial_tp_bf16_to_f32(0x3f80) == 1.0f);          /* 1.0 */
	CHECK(spark_serial_tp_bf16_to_f32(0x4000) == 2.0f);          /* 2.0 */
	CHECK(spark_serial_tp_bf16_to_f32(0xc000) == -2.0f);         /* -2.0 */
}

/* all-reduce sum: rank r contributes (r+1) * value; sum = tp*(tp+1)/2 * value. */
static void test_all_reduce_sum(void)
{
	enum { TP = 4, N = 8 };
	uint16_t partials[TP * N], out[N];
	uint32_t r; uint64_t i;
	for (r = 0u; r < TP; ++r)
		for (i = 0u; i < N; ++i)
			partials[(r * N) + i] = spark_serial_tp_f32_to_bf16(
				(float)(r + 1u) * (float)(i + 1u));
	spark_serial_tp_all_reduce_sum_bf16(partials, TP, N, out);
	for (i = 0u; i < N; ++i)
		CHECK(spark_serial_tp_bf16_to_f32(out[i]) ==
			(float)(TP * (TP + 1u) / 2u) * (float)(i + 1u));
}

/* reduce-scatter: rank r gets the r-th slice summed over all ranks. */
static void test_reduce_scatter(void)
{
	enum { TP = 4, N = 8 };
	uint16_t partials[TP * N], out[2];
	uint32_t r; uint64_t i;
	for (r = 0u; r < TP; ++r)
		for (i = 0u; i < N; ++i)
			partials[(r * N) + i] = spark_serial_tp_f32_to_bf16((float)(i + 1u));
	/* every rank contributes the same [1..8]; rank 1's slice is elements 2,3 */
	spark_serial_tp_reduce_scatter_bf16(partials, TP, N, 1u, out);
	CHECK(spark_serial_tp_bf16_to_f32(out[0]) == 3.0f * (float)TP);
	CHECK(spark_serial_tp_bf16_to_f32(out[1]) == 4.0f * (float)TP);
}

/* u64 maxloc: max value and the lowest rank achieving it. */
static void test_u64_maxloc(void)
{
	enum { TP = 3, N = 4 };
	uint64_t partials[TP * N] = {
		1, 9, 9, 4,
		9, 5, 2, 9,
		3, 6, 9, 8 };
	uint64_t values[N]; uint32_t ranks[N];
	spark_serial_tp_u64_maxloc(partials, TP, N, values, ranks);
	CHECK(values[0] == 9ull && ranks[0] == 1u);
	CHECK(values[1] == 9ull && ranks[1] == 0u);
	CHECK(values[2] == 9ull && ranks[2] == 0u);
	CHECK(values[3] == 9ull && ranks[3] == 1u);
}

/* ---- fake 2-rank row-parallel stage driving the sweep + compare ---- */
struct FakeStage
{
	uint32_t rank;              /* rank passed to run_rank */
	uint32_t call_count;        /* ranks run so far */
	uint32_t fail_on_rank;      /* UINT32_MAX = never */
	uint64_t shard_bytes;
	int load_fail;
	int run_fail;
};

static uint64_t fake_shard_bytes(uint32_t rank, void *context)
{
	(void)rank;
	return ((struct FakeStage *)context)->shard_bytes;
}

static int fake_load(uint32_t rank, void *context)
{
	struct FakeStage *s = (struct FakeStage *)context;
	if (s->load_fail)
		return -1;
	s->rank = rank;
	return 0;
}

static int fake_free(uint32_t rank, void *context)
{
	(void)rank;
	(void)context;
	return 0;
}

static int fake_run(uint32_t rank, const uint16_t *input_bf16,
	uint16_t *partial_out, uint64_t input_elements, uint64_t partial_elements,
	void *context)
{
	struct FakeStage *s = (struct FakeStage *)context;
	uint64_t i;
	(void)input_elements;
	s->call_count++;
	if (s->run_fail || rank == s->fail_on_rank)
		return -1;
	/* partial[i] = input[i] * (rank + 1) */
	for (i = 0u; i < partial_elements; ++i)
	{
		float v = input_bf16 != 0
			? spark_serial_tp_bf16_to_f32(input_bf16[i])
			: 0.0f;
		partial_out[i] = spark_serial_tp_f32_to_bf16(v * (float)(rank + 1u));
	}
	return 0;
}

static void test_sweep_and_compare(void)
{
	enum { TP = 2, N = 4 };
	struct FakeStage s;
	SparkSerialTpRankHooks hooks;
	SparkSerialTpDeviceBudget budget;
	uint16_t partials[TP * N], reduced[N], golden[N], input[N];
	uint64_t i;

	memset(&s, 0, sizeof(s));
	s.shard_bytes = 1u << 30;       /* 1 GiB fake shard */
	s.fail_on_rank = 0xffffffffu;
	hooks.load_shard = fake_load;
	hooks.free_shard = fake_free;
	hooks.shard_device_bytes = fake_shard_bytes;
	hooks.run_rank = fake_run;
	budget.cap_bytes = SPARK_SERIAL_TP_REPLAY_DEFAULT_DEVICE_BUDGET_BYTES;
	budget.held_bytes = 0u;
	budget.peak_held_bytes = 0u;

	for (i = 0u; i < N; ++i)
		input[i] = spark_serial_tp_f32_to_bf16((float)(i + 1u));

	/* first sweep from input: partials[r][i] = input[i] * (r+1) */
	CHECK(spark_serial_tp_sweep(TP, N, input, N, partials, &hooks, &s, &budget) == 0);
	CHECK(s.call_count == TP);
	CHECK(budget.held_bytes == 0u);
	CHECK(budget.peak_held_bytes == s.shard_bytes);
	/* all-reduce: input[i] * (1 + 2) = 3 * input[i] */
	spark_serial_tp_all_reduce_sum_bf16(partials, TP, N, reduced);
	for (i = 0u; i < N; ++i)
	{
		golden[i] = spark_serial_tp_f32_to_bf16(3.0f * (float)(i + 1u));
		CHECK(spark_serial_tp_bf16_to_f32(reduced[i]) ==
			spark_serial_tp_bf16_to_f32(golden[i]));
	}
	CHECK(spark_serial_tp_compare_exact(reduced, N, golden) == 0);

	/* budget overflow: a shard larger than the cap is rejected */
	budget.held_bytes = 0u;
	budget.peak_held_bytes = 0u;
	budget.cap_bytes = 1u << 20;    /* 1 MiB < 1 GiB shard */
	CHECK(spark_serial_tp_sweep(TP, N, input, N, partials, &hooks, &s, &budget) == -3);
	CHECK(budget.held_bytes == 0u);
}

static void test_hash(void)
{
	uint16_t a[4] = { 1, 2, 3, 4 };
	uint16_t b[4] = { 1, 2, 3, 5 };
	CHECK(spark_serial_tp_hash_elements(a, 4, sizeof(uint16_t)) ==
		spark_serial_tp_hash_elements(a, 4, sizeof(uint16_t)));
	CHECK(spark_serial_tp_hash_elements(a, 4, sizeof(uint16_t)) !=
		spark_serial_tp_hash_elements(b, 4, sizeof(uint16_t)));
}

int main(void)
{
	test_bf16_roundtrip();
	test_all_reduce_sum();
	test_reduce_scatter();
	test_u64_maxloc();
	test_sweep_and_compare();
	test_hash();
	if (failures != 0)
	{
		fprintf(stderr, "serial_tp_replay: %d failure(s)\n", failures);
		return 1;
	}
	printf("serial_tp_replay: pass\n");
	return 0;
}
