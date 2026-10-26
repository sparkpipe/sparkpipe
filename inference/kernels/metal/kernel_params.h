#ifndef SPARKPIPE_INFERENCE_KERNELS_METAL_KERNEL_PARAMS_H
#define SPARKPIPE_INFERENCE_KERNELS_METAL_KERNEL_PARAMS_H

#define SPARK_KERNEL_ROPE_PAIRS 64u
#define SPARK_KERNEL_GEMV_ROWS 8u
#define SPARK_KERNEL_SEQUENCE_NONE 0xffffffffu

typedef struct SparkGatherParams
{
	uint32_t rows;
	uint32_t width;
	uint32_t table_rows;
	uint32_t pad;
} SparkGatherParams;

typedef struct SparkNormParams
{
	uint32_t rows;
	uint32_t dimension;
	uint32_t input_stride;
	uint32_t output_stride;
	uint32_t has_addend;
	uint32_t write_sum;
	float epsilon;
	uint32_t pad;
} SparkNormParams;

typedef struct SparkGatedNormParams
{
	uint32_t rows;
	uint32_t heads;
	uint32_t head_dimension;
	uint32_t input_stride;
	uint32_t gate_stride;
	uint32_t gate_offset;
	uint32_t output_stride;
	float epsilon;
} SparkGatedNormParams;

typedef struct SparkHeadRopeParams
{
	uint32_t rows;
	uint32_t heads;
	uint32_t head_dimension;
	uint32_t input_stride;
	uint32_t input_offset;
	uint32_t input_head_stride;
	uint32_t output_stride;
	uint32_t rope_dimension;
	float epsilon;
	uint32_t pad[3];
	float inverse_frequency[SPARK_KERNEL_ROPE_PAIRS];
} SparkHeadRopeParams;

typedef struct SparkGemvParams
{
	uint32_t rows;
	uint32_t input_dimension;
	uint32_t output_dimension;
	uint32_t input_stride;
	uint32_t output_stride;
	uint32_t output_f32;
	uint32_t gated_input;
	uint32_t pad;
} SparkGemvParams;

typedef struct SparkGemmParams
{
	uint32_t rows;
	uint32_t input_dimension;
	uint32_t output_dimension;
	uint32_t input_stride;
	uint32_t output_stride;
	uint32_t output_f32;
	uint32_t gated_input;
	uint32_t pad;
} SparkGemmParams;

typedef struct SparkConvParams
{
	uint32_t channels;
	uint32_t row_stride;
	uint32_t sequences;
	uint32_t kernel_taps;
	uint32_t commit;
	uint32_t pad[3];
} SparkConvParams;

typedef struct SparkDeltaParams
{
	uint32_t rows;
	uint32_t sequences;
	uint32_t key_heads;
	uint32_t value_heads;
	uint32_t key_dimension;
	uint32_t value_dimension;
	uint32_t mix_stride;
	uint32_t query_offset;
	uint32_t key_offset;
	uint32_t value_offset;
	uint32_t gate_stride;
	uint32_t beta_offset;
	uint32_t decay_offset;
	uint32_t output_stride;
	uint32_t commit;
	float epsilon;
} SparkDeltaParams;

typedef struct SparkKvParams
{
	uint32_t rows;
	uint32_t kv_heads;
	uint32_t head_dimension;
	uint32_t page_slots;
	uint32_t pages_per_sequence;
	uint32_t key_stride;
	uint32_t value_stride;
	uint32_t value_offset;
	uint32_t pool_pages;
	uint32_t pad[3];
} SparkKvParams;

typedef struct SparkAttentionParams
{
	uint32_t rows;
	uint32_t heads;
	uint32_t kv_heads;
	uint32_t head_dimension;
	uint32_t page_slots;
	uint32_t pages_per_sequence;
	uint32_t pool_pages;
	uint32_t query_stride;
	uint32_t gate_stride;
	uint32_t gate_offset;
	uint32_t gate_head_stride;
	uint32_t output_stride;
	uint32_t partitions;
	uint32_t partition_keys;
	float scale;
	uint32_t row_offset;
} SparkAttentionParams;

typedef struct SparkSiluMulParams
{
	uint32_t rows;
	uint32_t dimension;
	uint32_t input_stride;
	uint32_t output_stride;
} SparkSiluMulParams;

typedef struct SparkLogprobParams
{
	uint32_t rows;
	uint32_t vocabulary;
	uint32_t logit_stride;
	uint32_t pad;
} SparkLogprobParams;

typedef struct SparkStreamParams
{
	uint64_t words;
	uint32_t groups;
	uint32_t pad;
} SparkStreamParams;

#endif
