typedef struct SPARK_ABI_TYPE(MtpDraftView)
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t draft_token_count;
	uint64_t base_position;
	uint64_t sequence_id;
	const uint32_t *row_token_ids;
} SPARK_ABI_TYPE(MtpDraftView);

typedef struct SPARK_ABI_TYPE(GdnSnapshotView)
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t snapshot_index;
	uint32_t reserved0;
} SPARK_ABI_TYPE(GdnSnapshotView);
