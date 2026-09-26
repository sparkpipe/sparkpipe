typedef struct SPARK_ABI_TYPE(PrefillFrameView)
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
} SPARK_ABI_TYPE(PrefillFrameView);
