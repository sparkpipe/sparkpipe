#ifndef SPARKPIPE_INFERENCE_KERNELS_METAL_KERNEL_PARAMS_H
#define SPARKPIPE_INFERENCE_KERNELS_METAL_KERNEL_PARAMS_H

#define SPARK_KERNEL_ROPE_PAIRS 256u
#define SPARK_GEMV_OUTPUTS 4u
#define SPARK_PROJECTION_GEMV 0u
#define SPARK_PROJECTION_ROW_TILES 1u
#define SPARK_GEMV_ROW_OUTPUTS(rows) ((rows) >= 8u ? 1u : (rows) >= 4u ? 2u : SPARK_GEMV_OUTPUTS)
#define SPARK_ATTENTION_BLOCK_ROWS(dim) ((dim) > 256u ? 16u : 32u)
#define SPARK_GEMM_ROWS_OUTPUTS 128u
#define SPARK_GEMM_ROWS_DEPTH 64u
#define SPARK_GEMM_ROWS_THREADS 128u
#define SPARK_KERNEL_SEQUENCE_NONE 0xffffffffu
#define SPARK_BRANCH_NORM_MAX_DIMENSION 7680u
#define SPARK_ROUTE_MAX_EXPERTS 512u
#define SPARK_ROUTE_MAX_TOP_K 16u
#define SPARK_GATED_SILU 0u
#define SPARK_GATED_GELU_TANH 1u
#define SPARK_GATED_SILU_ROUNDED 2u
#define SPARK_FP8_SCALE_K 128u
#define SPARK_FP8_CHUNK 16u
#define SPARK_MXFP4_GROUP 32u
#define SPARK_NVFP4_GROUP 16u
#define SPARK_NVFP4_CHUNK 32u
#define SPARK_NVFP4_LANES(chunks) ((chunks) >= 32u ? 32u : (chunks) >= 16u ? 16u : 8u)
#define SPARK_ATTENTION_GATE_SIGMOID 1u
#define SPARK_ATTENTION_GATE_SIGMOID_BF16 2u
#define SPARK_HYPER_MAX_STREAMS 8u

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
	uint32_t weightless;
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

typedef struct SparkBranchNormParams
{
	uint32_t rows;
	uint32_t dimension;
	uint32_t branch_stride;
	uint32_t hidden_stride;
	uint32_t branches;
	uint32_t has_scalar;
	float epsilon;
	float output_scale;
} SparkBranchNormParams;

typedef struct SparkRouteParams
{
	uint32_t rows;
	uint32_t experts;
	uint32_t top_k;
	uint32_t width;
	uint32_t logit_stride;
	uint32_t pad[3];
} SparkRouteParams;

typedef struct SparkScaleParams
{
	uint64_t count;
	float scale;
	float multiply;
} SparkScaleParams;

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
	uint32_t has_weight;
	uint32_t rope_rounded_terms;
	float multiply;
	uint32_t plain;
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
	uint32_t projection;
	uint32_t splits;
	uint32_t split_span;
	uint32_t row_exact;
	uint32_t route_divisor;
	uint32_t scale_stride;
} SparkGemvParams;

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
	uint32_t value_dimension;
	uint32_t pad[2];
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
	uint32_t row_exact;
	uint32_t window;
	uint32_t output_gate;
	uint32_t value_dimension;
	uint32_t has_sinks;
	uint32_t pad[2];
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

typedef struct SparkHyperParams
{
	uint32_t rows;
	uint32_t streams;
	uint32_t width;
	float epsilon;
	float divisor;
	uint32_t pad[3];
} SparkHyperParams;

#endif
