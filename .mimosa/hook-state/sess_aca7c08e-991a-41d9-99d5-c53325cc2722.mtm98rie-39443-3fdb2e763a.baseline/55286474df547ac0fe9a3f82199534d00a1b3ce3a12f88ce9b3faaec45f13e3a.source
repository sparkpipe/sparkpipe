#ifndef SPARK_SERIAL_TP_REPLAY_H
#define SPARK_SERIAL_TP_REPLAY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SERIAL_TP_REPLAY_ABI_VERSION 1u
#define SPARK_SERIAL_TP_REPLAY_DEFAULT_DEVICE_BUDGET_BYTES (108ull << 30)

typedef enum SparkSerialTpCollectiveKind
{
	SPARK_SERIAL_TP_COLLECTIVE_ALL_REDUCE_SUM_BF16 = 0,
	SPARK_SERIAL_TP_COLLECTIVE_REDUCE_SCATTER_BF16 = 1,
	SPARK_SERIAL_TP_COLLECTIVE_U64_MAXLOC = 2,
	SPARK_SERIAL_TP_COLLECTIVE_IDENTITY = 3
} SparkSerialTpCollectiveKind;

typedef struct SparkSerialTpDeviceBudget
{
	uint64_t cap_bytes;
	uint64_t held_bytes;
	uint64_t peak_held_bytes;
} SparkSerialTpDeviceBudget;

typedef struct SparkSerialTpRankHooks
{
	int (*load_shard)(uint32_t rank, void *context);
	int (*free_shard)(uint32_t rank, void *context);
	uint64_t (*shard_device_bytes)(uint32_t rank, void *context);
	int (*run_rank)(uint32_t rank, const uint16_t *input_bf16,
		uint16_t *partial_out_bf16, uint64_t input_elements,
		uint64_t partial_elements, void *context);
} SparkSerialTpRankHooks;

typedef int (*SparkSerialTpGoldenCompare)(const uint16_t *output_bf16,
	uint64_t element_count, void *context);

int spark_serial_tp_sweep(
	uint32_t tp_degree,
	uint64_t partial_elements,
	const uint16_t *input_bf16,
	uint64_t input_elements,
	uint16_t *partials_bf16,
	const SparkSerialTpRankHooks *hooks,
	void *context,
	SparkSerialTpDeviceBudget *budget);


void spark_serial_tp_all_reduce_sum_bf16(const uint16_t *partials, uint32_t tp,
	uint64_t n, uint16_t *out);

void spark_serial_tp_reduce_scatter_bf16(const uint16_t *partials, uint32_t tp,
	uint64_t n, uint32_t rank, uint16_t *out);

void spark_serial_tp_u64_maxloc(const uint64_t *partials, uint32_t tp,
	uint64_t n, uint64_t *out_values, uint32_t *out_ranks);

float spark_serial_tp_bf16_to_f32(uint16_t b);
uint16_t spark_serial_tp_f32_to_bf16(float f);


int spark_serial_tp_compare_exact(const uint16_t *output_bf16,
	uint64_t element_count, void *context   );

uint64_t spark_serial_tp_hash_elements(const void *data, uint64_t element_count,
	uint32_t element_bytes);

#ifdef __cplusplus
}
#endif

#endif
