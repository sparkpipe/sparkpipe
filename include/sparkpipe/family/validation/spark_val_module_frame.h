#pragma once

typedef struct SPARK_FAMILY(ValCapture)
{
	uint16_t hidden[SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS) * SPARK_FAMILY_CONST(VALIDATION_HIDDEN_WIDTH)];
	uint32_t sends;
} SPARK_FAMILY(ValCapture);

static SparkStatus SPARK_FAMILY(ValCaptureSend)(SparkHiddenTransportSession *session, const SparkHiddenTransportPacket *packet)
{
	SPARK_FAMILY(ValCapture) *capture = (SPARK_FAMILY(ValCapture) *)session;
	uint64_t bytes = (uint64_t)packet->active_sequence_count * SPARK_FAMILY_CONST(VALIDATION_HIDDEN_WIDTH) * 2u;
	if (packet->hidden_bf16 == 0 || packet->active_sequence_count > SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS) ||
		packet->hidden_dimension != SPARK_FAMILY_CONST(VALIDATION_HIDDEN_WIDTH))
		return(SPARK_STATUS_VALIDATION_FAILED);
	if (cudaMemcpy(capture->hidden,packet->hidden_bf16,bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
		return(SPARK_STATUS_IO_ERROR);
	capture->sends++;
	return(SPARK_STATUS_OK);
}

typedef struct SPARK_FAMILY(ValModule)
{
	void *state;
	uint32_t token_ids[SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS)];
	uint32_t output_token_ids[SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS)];
	uint32_t head_stage;
	uint32_t lanes[SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS)];
	uint64_t positions[SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS)];
	uint64_t sequence_ids[SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS)];
	uint32_t host_blocks[SPARK_FAMILY_CONST(VALIDATION_KV_LANES)];
	uint32_t host_counts[SPARK_FAMILY_CONST(VALIDATION_KV_LANES)];
	uint32_t *device_blocks;
	uint32_t *device_counts;
	SPARK_FAMILY(KvBlockTableView) table;
	SPARK_FAMILY(DecodeBatchView) decode_batch;
	SPARK_FAMILY(PrefillFrameView) prefill_view;
	SPARK_FAMILY(ResidentDecodeStageFrameContext) context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SPARK_FAMILY(ValCapture) capture;
} SPARK_FAMILY(ValModule);

static int SPARK_FAMILY(ValModuleExecute)(SPARK_FAMILY(ValModule) *module, uint32_t prefill, uint32_t rows, uint32_t lane, uint32_t draft_count, const SPARK_FAMILY(MtpDraftView) *draft_view)
{
	SparkStatus status;
	memset(&module->context,0,sizeof(module->context));
	memset(&module->frame,0,sizeof(module->frame));
	module->context.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION);
	module->context.descriptor_bytes = sizeof(module->context);
	module->context.flags =
		SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) |
		(module->head_stage == 0u
			? SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT)
			: 0u) |
		(prefill != 0u
			? SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW)
			: SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW)) |
		(draft_count != 0u ? SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) : 0u);
	module->context.mtp_draft = draft_view;
	module->context.kv_block_table = &module->table;
	if (prefill != 0u)
	{
		module->prefill_view.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION);
		module->prefill_view.descriptor_bytes = sizeof(module->prefill_view);
		module->prefill_view.lane_index = lane;
		module->prefill_view.token_count = rows;
		module->prefill_view.base_position = module->positions[0];
		module->prefill_view.sequence_id = module->sequence_ids[0];
		module->context.prefill_frame = &module->prefill_view;
	}
	else
	{
		module->decode_batch.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION);
		module->decode_batch.descriptor_bytes = sizeof(module->decode_batch);
		module->decode_batch.row_count = rows;
		module->decode_batch.row_lane_indices = module->lanes;
		module->decode_batch.row_positions = module->positions;
		module->decode_batch.row_sequence_ids = module->sequence_ids;
		module->context.decode_batch = &module->decode_batch;
	}
	module->context.hidden_output_transport_session = module->head_stage != 0u ? 0 : (SparkHiddenTransportSession *)&module->capture;
	module->context.hidden_output_send_function = module->head_stage != 0u ? 0 : SPARK_FAMILY(ValCaptureSend);
	module->frame.program_id = 1u;
	module->frame.tokens_per_sequence = 1u;
	module->frame.request_id = 1u;
	module->frame.sequence_id = module->sequence_ids[0];
	module->frame.sequence_position = module->positions[0];
	module->frame.active_slot_count = prefill != 0u ? 1u : rows;
	module->frame.new_token_count = rows;
	module->frame.flags = prefill != 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	module->frame.execution_stream = (void *)cudaStreamPerThread;
	module->frame.buffers = module->buffers;
	module->frame.buffer_count = module->head_stage != 0u ? 2u : 1u;
	module->frame.user_context = &module->context;
	memset(module->buffers,0,sizeof(module->buffers));
	module->buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
	module->buffers[0].address = module->token_ids;
	module->buffers[0].bytes = rows * sizeof(uint32_t);
	if ( module->head_stage != 0u )
	{
		module->buffers[1].slot = 1u;
		module->buffers[1].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		module->buffers[1].address = module->output_token_ids;
		module->buffers[1].bytes = sizeof(module->output_token_ids);
	}
	status = SPARK_FAMILY(ResidentDecodeStageExecute)(module->state,&module->frame);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_validation failure=module_execute prefill=%u rows=%u status=%d\n",prefill,rows,(int)status);
		return(1);
	}
	return(0);
}

static int SPARK_FAMILY(ValCheckMtpDraft)(SPARK_FAMILY(ValModule) *module)
{
	SPARK_FAMILY(MtpDraftView) draft_view;
	uint32_t draft;
	module->token_ids[0] = 4242u;
	module->lanes[0] = 0u;
	module->positions[0] = SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS) + 1u;
	module->sequence_ids[0] = 1u;
	memset(&draft_view,0,sizeof(draft_view));
	draft_view.abi_version = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION);
	draft_view.descriptor_bytes = sizeof(draft_view);
	draft_view.lane_index = 0u;
	draft_view.draft_token_count = 2u;
	draft_view.base_position = (uint64_t)SPARK_FAMILY_CONST(VALIDATION_PREFILL_TOKENS) + 2u;
	draft_view.sequence_id = 1u;
	draft_view.row_token_ids = module->token_ids;
	if (SPARK_FAMILY(ValModuleExecute)(module,0u,1u,0u,2u,&draft_view) != 0)
		return(1);
	for (draft = 0u; draft < 2u; draft++)
	{
		if (module->output_token_ids[1u + draft] >= SPARK_FAMILY_CONST(MODEL_VOCAB_COUNT))
			return(SPARK_FAMILY(ValFail)("module_mtp_draft","out_of_vocab"));
	}
	printf(SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_validation check=module_mtp_draft in_vocab=1 drafts=[%u,%u]\n",module->output_token_ids[1],module->output_token_ids[2]);
	return(0);
}
