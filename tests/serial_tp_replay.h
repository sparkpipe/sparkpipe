#ifndef SPARK_SERIAL_TP_REPLAY_H
#define SPARK_SERIAL_TP_REPLAY_H
/*
 * Serial-TP replay harness: run one tensor-parallel shard at a time on ONE
 * device, emulate the model's TP collectives host-side, and compare the final
 * result against a golden reference. This establishes CORRECTNESS while the
 * fleet is down (slow, but fits one node's memory budget).
 *
 * Model-neutral by construction: geometry, shard load/run/free, the collective
 * plan, and the golden are ALL injected by the caller. Nothing here names a
 * model, a layer, or a dtype beyond bf16 (the shared transport width). CUDA
 * lives entirely behind the caller's hooks, so the host paths below compile
 * and run without nvcc; a caller wires the CUDA side through its module API
 * with the TP_STANDALONE-style collective no-op (the reduces become no-ops and
 * this harness performs them host-side instead).
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SERIAL_TP_REPLAY_ABI_VERSION 1u
/* Default device budget: 108 GiB cgroup. Callers may lower it per deployment. */
#define SPARK_SERIAL_TP_REPLAY_DEFAULT_DEVICE_BUDGET_BYTES (108ull << 30)

/* One collective emulated host-side between rank sweeps. */
typedef enum SparkSerialTpCollectiveKind
{
	SPARK_SERIAL_TP_COLLECTIVE_ALL_REDUCE_SUM_BF16 = 0,
	SPARK_SERIAL_TP_COLLECTIVE_REDUCE_SCATTER_BF16 = 1,
	SPARK_SERIAL_TP_COLLECTIVE_U64_MAXLOC = 2,
	SPARK_SERIAL_TP_COLLECTIVE_IDENTITY = 3
} SparkSerialTpCollectiveKind;

/* Serial shard-residency accounting. held_bytes must never exceed cap_bytes. */
typedef struct SparkSerialTpDeviceBudget
{
	uint64_t cap_bytes;
	uint64_t held_bytes;
	uint64_t peak_held_bytes;
} SparkSerialTpDeviceBudget;

/* Caller-injected rank hooks. All pointers are stable for the sweep's duration.
 * run_rank writes rank r's PARTIAL (un-reduced) output into partial_out, which
 * is partial_elements bf16. load_shard/free_shard bracket run_rank so only one
 * shard is resident at a time; shard_device_bytes drives the budget check. */
typedef struct SparkSerialTpRankHooks
{
	int (*load_shard)(uint32_t rank, void *context);
	int (*free_shard)(uint32_t rank, void *context);
	uint64_t (*shard_device_bytes)(uint32_t rank, void *context);
	int (*run_rank)(uint32_t rank, const uint16_t *input_bf16,
		uint16_t *partial_out_bf16, uint64_t input_elements,
		uint64_t partial_elements, void *context);
} SparkSerialTpRankHooks;

/* Final-output golden compare: return 0 on match, non-zero otherwise. */
typedef int (*SparkSerialTpGoldenCompare)(const uint16_t *output_bf16,
	uint64_t element_count, void *context);

/*
 * One serial sweep: for rank 0..tp_degree-1, load the shard (budget-checked),
 * run it over input_bf16, capture the partial into partials_bf16, and free the
 * shard. partials_bf16 is caller-owned host memory of
 * tp_degree * partial_elements bf16. input_bf16 may be NULL (treated as zeros)
 * for the first sweep. Returns 0 on success, non-zero on any rank failure or a
 * budget overflow.
 */
int spark_serial_tp_sweep(
	uint32_t tp_degree,
	uint64_t partial_elements,
	const uint16_t *input_bf16,
	uint64_t input_elements,
	uint16_t *partials_bf16,
	const SparkSerialTpRankHooks *hooks,
	void *context,
	SparkSerialTpDeviceBudget *budget);

/* ---- host collective emulation (pure C, deterministic) ---- */

/* out[i] = sum over ranks of partials[r*n + i], bf16 in fp32 sum, RNE round. */
void spark_serial_tp_all_reduce_sum_bf16(const uint16_t *partials, uint32_t tp,
	uint64_t n, uint16_t *out);

/* rank r's slice of the reduce-scatter: out[j] = sum over ranks of
 * partials[s*n + r*(n/tp) + j]. Requires n % tp == 0. */
void spark_serial_tp_reduce_scatter_bf16(const uint16_t *partials, uint32_t tp,
	uint64_t n, uint32_t rank, uint16_t *out);

/* u64 maxloc: out_values[i] = max over ranks of partials[r*n + i];
 * out_ranks[i] = lowest rank achieving it. */
void spark_serial_tp_u64_maxloc(const uint64_t *partials, uint32_t tp,
	uint64_t n, uint64_t *out_values, uint32_t *out_ranks);

/* ---- bf16 <-> fp32 host conversion (round-to-nearest-even) ---- */
float spark_serial_tp_bf16_to_f32(uint16_t b);
uint16_t spark_serial_tp_f32_to_bf16(float f);

/* ---- golden compare helpers ---- */

/* Exact token-id / element compare (memcmp); returns 0 on match. */
int spark_serial_tp_compare_exact(const uint16_t *output_bf16,
	uint64_t element_count, void *context /* golden uint16_t* */);

/* Deterministic FNV-1a 64 hash over element bytes; used when the golden is a
 * logits hash rather than exact token ids. */
uint64_t spark_serial_tp_hash_elements(const void *data, uint64_t element_count,
	uint32_t element_bytes);

#ifdef __cplusplus
}
#endif

#endif /* SPARK_SERIAL_TP_REPLAY_H */
