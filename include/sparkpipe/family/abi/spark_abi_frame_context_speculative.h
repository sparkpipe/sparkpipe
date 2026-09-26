typedef SparkStatus (*SPARK_ABI_TYPE(HiddenTransportPostReceiveFunction))(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SPARK_ABI_TYPE(HiddenTransportSendFunction))(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

typedef struct SPARK_ABI_TYPE(ResidentDecodeStageFrameContext)
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SPARK_ABI_TYPE(KvBlockTableView) *kv_block_table;
	const SPARK_ABI_TYPE(DecodeBatchView) *decode_batch;
	const SPARK_ABI_TYPE(PrefillFrameView) *prefill_frame;
	const SPARK_ABI_TYPE(MtpDraftView) *mtp_draft;
	const SPARK_ABI_TYPE(GdnSnapshotView) *gdn_snapshot;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SPARK_ABI_TYPE(HiddenTransportPostReceiveFunction) hidden_input_post_receive_function;
	SPARK_ABI_TYPE(HiddenTransportSendFunction) hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SPARK_ABI_TYPE(ResidentDecodeStageFrameContext);
