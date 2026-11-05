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
#define SPARK_ROUTE_MAX_EXPERTS 1024u
#define SPARK_ROUTE_MAX_TOP_K 16u
#define SPARK_ROUTE_MAX_GROUPS 32u
#define SPARK_ROUTE_SKIP 0xffffffffu
#define SPARK_GATED_SILU 0u
#define SPARK_GATED_GELU_TANH 1u
#define SPARK_GATED_SILU_ROUNDED 2u
#define SPARK_GATED_SILU_CLAMP 3u
#define SPARK_GATED_SILU_GATE_LIMIT 4u
#define SPARK_GATED_SITU 5u
#define SPARK_HEAD_GATE_SOFTPLUS 0u
#define SPARK_HEAD_GATE_SIGMOID_BF16 1u
#define SPARK_HEAD_GATE_SIGMOID 2u
#define SPARK_FP8_SCALE_K 128u
#define SPARK_FP8_CHUNK 16u
#define SPARK_MXFP4_GROUP 32u
#define SPARK_MXFP8_GROUP 32u
#define SPARK_NVFP4_GROUP 16u
#define SPARK_NVFP4_CHUNK 32u
#define SPARK_NVFP4_LANES(chunks) ((chunks) >= 32u ? 32u : (chunks) >= 16u ? 16u : 8u)
#define SPARK_ATTENTION_GATE_SIGMOID 1u
#define SPARK_ATTENTION_GATE_SIGMOID_BF16 2u
#define SPARK_HYPER_MAX_STREAMS 8u
#define SPARK_CONV_SPAN_MAX 16u
#define SPARK_NGRAM_MAX_ORDERS 3u
#define SPARK_CONV_SPAN(taps, dilation) (((taps) - 1u) * ((dilation) != 0u ? (dilation) : 1u) + 1u)

typedef struct SparkGatherParams
{
	uint32_t rows;
	uint32_t width;
	uint32_t table_rows;
	uint32_t first_row;
	uint32_t sliced;
	uint32_t pad[3];
} SparkGatherParams;

typedef struct SparkRankParams
{
	uint64_t elements;
	uint32_t ranks;
	uint32_t rows;
	uint32_t width;
	uint32_t pad[3];
} SparkRankParams;

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

#define SPARK_ATTNRES_MAX_SOURCES 16u

typedef struct SparkAttnResParams
{
	uint32_t rows;
	uint32_t sources;
	uint32_t dimension;
	uint32_t score_stride;
	float epsilon;
	uint32_t pad[3];
} SparkAttnResParams;

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
	uint32_t sigmoid_gate;
	uint32_t weight_f32;
	uint32_t round_norm;
	uint32_t pad;
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
	uint32_t first_expert;
	uint32_t local_experts;
	float scale;
	uint32_t round_logits;
	uint32_t groups;
	uint32_t topk_groups;
	uint32_t tree_sum;
	uint32_t score;
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
	float scale;
	uint32_t interleaved;
	uint32_t output_head_stride;
	float inverse_frequency[SPARK_KERNEL_ROPE_PAIRS];
} SparkHeadRopeParams;

typedef struct SparkHeadGateParams
{
	uint32_t rows;
	uint32_t heads;
	uint32_t head_dimension;
	uint32_t gate_stride;
	uint32_t activation;
	uint32_t gate_head_stride;
	uint32_t gate_offset;
	uint32_t pad;
} SparkHeadGateParams;

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
	uint32_t scale_group;
} SparkGemvParams;

typedef struct SparkConvParams
{
	uint32_t channels;
	uint32_t row_stride;
	uint32_t sequences;
	uint32_t kernel_taps;
	uint32_t commit;
	uint32_t dilation;
	uint32_t weight_f32;
	uint32_t pad;
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
	uint32_t channel_decay;
	uint32_t beta_stride;
	float minimum_log_decay;
	uint32_t beta_mode;
	float prenorm_epsilon;
} SparkDeltaParams;

#define SPARK_DELTA_BETA_SIGMOID 0u
#define SPARK_DELTA_BETA_DOUBLED 1u
#define SPARK_DELTA_BETA_ROUNDED 2u

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
	uint32_t skip;
	uint32_t pad;
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
	uint32_t first_head;
	uint32_t group;
	uint32_t partition_base;
	uint32_t extra_partitions;
	uint32_t pad[3];
} SparkAttentionParams;

typedef struct SparkSiluMulParams
{
	uint32_t rows;
	uint32_t dimension;
	uint32_t input_stride;
	uint32_t output_stride;
	float limit;
	float up_limit;
	uint32_t pad[2];
} SparkSiluMulParams;

typedef struct SparkPolyNormParams
{
	uint32_t rows;
	uint32_t width;
	uint32_t input_stride;
	uint32_t output_stride;
	uint32_t param_count;
	uint32_t indexed;
	uint32_t routed;
	uint32_t pad;
	float clamp;
	float bias_clamp;
	float output_scale;
	float epsilon;
} SparkPolyNormParams;

typedef struct SparkDifferentialGateParams
{
	uint32_t rows;
	uint32_t groups;
	uint32_t signal;
	uint32_t dimension;
	uint32_t heads_stride;
	uint32_t lambda_stride;
	uint32_t lambda_offset;
	uint32_t gate_stride;
	uint32_t output_stride;
	uint32_t pad[3];
} SparkDifferentialGateParams;

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

typedef struct SparkNgramParams
{
	uint32_t rows;
	uint32_t sequences;
	uint32_t orders;
	uint32_t heads;
	int32_t pad_token;
	int32_t block_token;
	uint32_t block_from;
	uint32_t history_stride;
	uint32_t head_dimension;
	uint32_t commit;
	uint32_t mapped;
	uint32_t pad;
	uint64_t first_row;
	uint64_t table_rows;
} SparkNgramParams;

typedef struct SparkHyperParams
{
	uint32_t rows;
	uint32_t streams;
	uint32_t width;
	float epsilon;
	float divisor;
	uint32_t pad[3];
} SparkHyperParams;

#define SPARK_HC_MAX_STREAMS 4u

typedef struct SparkHcParams
{
	uint32_t rows;
	uint32_t streams;
	uint32_t width;
	uint32_t mixes;
	float norm_epsilon;
	float hc_epsilon;
	float post_magnitude;
	uint32_t iterations;
	float gate_clamp;
	float residual_clamp;
	float sum_floor;
	uint32_t pad;
} SparkHcParams;

#define SPARK_HC_COMB_MIXES 24u

#define SPARK_QDQ_FP8 0u
#define SPARK_QDQ_FP4 1u
#define SPARK_QDQ_FP4_POW2 2u
#define SPARK_QDQ_BLOCK(format) ((format) == SPARK_QDQ_FP4 ? 16u : 32u)
#define SPARK_ROUTE_SCORE_SIGMOID 0u
#define SPARK_ROUTE_SCORE_SQRT_SOFTPLUS 1u
#define SPARK_KV_SKIP 0xffffffffu

typedef struct SparkQdqParams
{
	uint32_t rows;
	uint32_t width;
	uint32_t input_stride;
	uint32_t output_stride;
	uint32_t format;
	uint32_t pad[3];
} SparkQdqParams;

typedef struct SparkCompressParams
{
	uint32_t rows;
	uint32_t sequences;
	uint32_t width;
	uint32_t ratio;
	uint32_t state_stride;
	uint32_t commit;
	uint32_t pad[2];
} SparkCompressParams;

typedef struct SparkGateParams
{
	uint32_t rows;
	uint32_t streams;
	uint32_t width;
	float epsilon;
	float floor;
	uint32_t pad[3];
} SparkGateParams;

#endif
