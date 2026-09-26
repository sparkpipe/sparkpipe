typedef struct SPARK_ABI_TYPE(GdnLayerWeights)
{
	SPARK_ABI_TYPE(LinearView) qkv;
	SPARK_ABI_TYPE(LinearView) gate;
	SPARK_ABI_TYPE(LinearView) beta;
	SPARK_ABI_TYPE(LinearView) decay;
	SPARK_ABI_TYPE(LinearView) output;
	const void *conv_weight_bf16;
	const float *a_log_f32;
	const float *dt_bias_f32;
	const void *gdn_norm_weight_bf16;
} SPARK_ABI_TYPE(GdnLayerWeights);

typedef struct SPARK_ABI_TYPE(AttnLayerWeights)
{
	SPARK_ABI_TYPE(LinearView) query;
	SPARK_ABI_TYPE(LinearView) key;
	SPARK_ABI_TYPE(LinearView) value;
	SPARK_ABI_TYPE(LinearView) output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SPARK_ABI_TYPE(AttnLayerWeights);
