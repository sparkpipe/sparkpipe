typedef struct SPARK_ABI_TYPE(MoeWeights)
{
	SPARK_ABI_TYPE(LinearView) gate;
	SPARK_ABI_TYPE(LinearView) experts_w1;
	SPARK_ABI_TYPE(LinearView) experts_w3;
	SPARK_ABI_TYPE(LinearView) experts_w2;
	SPARK_ABI_TYPE(LinearView) shared_gate;
	SPARK_ABI_TYPE(LinearView) shared_up;
	SPARK_ABI_TYPE(LinearView) shared_down;
	const void *shared_gate_weight_bf16;
	uint64_t experts_w1_payload_offset;
	uint64_t experts_w1_scale_offset;
	uint64_t experts_w3_payload_offset;
	uint64_t experts_w3_scale_offset;
	uint64_t experts_w2_payload_offset;
	uint64_t experts_w2_scale_offset;
} SPARK_ABI_TYPE(MoeWeights);
