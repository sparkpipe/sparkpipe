#include <string.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_runner.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_row_layout.h"

#define SPARK_DSV4_STAGE_RUNNER_MAX_DRIVER_BUFFERS 2u

static _Thread_local uint32_t SparkDsv4StageRunnerLaneOrdinals[
	SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT];
static _Thread_local uint32_t SparkDsv4StageRunnerOccurrences[
	SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
static _Thread_local uint32_t SparkDsv4StageRunnerLastRows[
	SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

static SparkStatus SparkDsv4StageRunnerValidateConfiguration(
    const SparkDsv4StageRunnerConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_DSV4_STAGE_RUNNER_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES ||
        configuration->reserved0 != 0u ||
		(configuration->flags & ~SPARK_DSV4_STAGE_RUNNER_KNOWN_FLAGS) != 0u ||
        configuration->stage_count == 0u ||
        configuration->stage_count >
            SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT ||
        configuration->stage_index >= configuration->stage_count ||
        configuration->max_active_sequence_count == 0u ||
		configuration->max_active_sequence_count >
			SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ||
		configuration->max_input_row_count < configuration->max_active_sequence_count ||
		configuration->max_input_row_count >
			SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		configuration->resident_sequence_capacity < configuration->max_active_sequence_count ||
		configuration->resident_sequence_capacity >
			SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT ||
        configuration->driver_interface == 0 ||
        configuration->driver_instance == 0 ||
        configuration->program == 0 ||
        configuration->program->submit == 0 ||
        configuration->execution_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration->stage_index != 0u) !=
            ((configuration->flags &
                SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY) != 0u) ||
        (configuration->stage_index + 1u < configuration->stage_count) !=
            ((configuration->flags &
                SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY) != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration->flags &
            SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION) != 0u &&
        configuration->driver_interface->admit == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
	return SPARK_STATUS_OK;
}

static SparkStatus SparkDsv4StageRunnerValidatePrefillRows(
	const SparkDsv4StageRunner *runner,
	const SparkDsv4StageRunnerDispatch *dispatch)
{
	SparkRowLayoutDirectLaneContext direct;
	SparkStatus status;
	status = SparkRowLayoutDirectLaneMapInitialize(&direct,SparkDsv4StageRunnerLaneOrdinals,runner->resident_sequence_capacity,dispatch->row_lane_indices,dispatch->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkRowLayoutValidateRoundMajor(dispatch->row_count,dispatch->active_sequence_count,dispatch->row_lane_indices,SparkRowLayoutDirectLaneOrdinal,&direct,SparkDsv4StageRunnerOccurrences,SparkDsv4StageRunnerLastRows));
}

static SparkStatus SparkDsv4StageRunnerValidateEmitRows(
	const SparkDsv4StageRunnerDispatch *dispatch,
	uint32_t is_prefill)
{
	uint8_t seen[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t index,lane,previous,row;
	if ( is_prefill == 0u )
		return(dispatch->emit_count == 0u && dispatch->emit_row_indices == 0 && dispatch->emit_lane_indices == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( dispatch->emit_count > dispatch->active_sequence_count || ((dispatch->emit_count != 0u) != (dispatch->emit_row_indices != 0)) || ((dispatch->emit_count != 0u) != (dispatch->emit_lane_indices != 0)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	previous = UINT32_MAX;
	for (index=0u; index<dispatch->emit_count; index++)
	{
		row = dispatch->emit_row_indices[index];
		lane = dispatch->emit_lane_indices[index];
		if ( row >= dispatch->row_count || lane >= dispatch->active_sequence_count || seen[lane] != 0u || (index != 0u && row <= previous) || row != SparkDsv4StageRunnerLastRows[lane] )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		previous = row;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4StageRunnerValidateDispatchShape(
    const SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerDispatch *dispatch)
{
    uint32_t is_prefill;
    if (runner == 0 || dispatch == 0 ||
        dispatch->abi_version != SPARK_DSV4_STAGE_RUNNER_ABI_VERSION ||
        dispatch->descriptor_bytes != SPARK_DSV4_STAGE_RUNNER_DISPATCH_BYTES ||
        (dispatch->flags & ~SPARK_DSV4_STAGE_RUNNER_DISPATCH_KNOWN_FLAGS) != 0u ||
        dispatch->active_sequence_count == 0u ||
        dispatch->active_sequence_count > runner->max_active_sequence_count ||
		dispatch->row_count == 0u ||
		dispatch->row_count > runner->max_input_row_count ||
        dispatch->lane_count == 0u ||
        dispatch->lane_count > runner->max_active_sequence_count ||
        dispatch->row_lane_indices == 0 ||
        dispatch->row_positions == 0 ||
        dispatch->row_sequence_ids == 0 ||
        dispatch->request_id == 0u ||
        dispatch->sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    is_prefill = (dispatch->flags &
        SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL) != 0u ? 1u : 0u;
	if (is_prefill != 0u)
	{
		if (dispatch->new_token_count != dispatch->row_count ||
			SparkDsv4StageRunnerValidatePrefillRows(runner,dispatch) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (dispatch->new_token_count != dispatch->active_sequence_count ||
        dispatch->row_count != dispatch->active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
	return SparkDsv4StageRunnerValidateEmitRows(dispatch,is_prefill);
}

static SparkStatus SparkDsv4StageRunnerValidateDispatchBoundaries(
	const SparkDsv4StageRunner *runner,
	const SparkDsv4StageRunnerDispatch *dispatch)
{
	uint64_t hidden_bytes;
    if (dispatch->token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (runner->owns_final_head != 0u && dispatch->output_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((runner->stage_index != 0u &&
         (dispatch->hidden_input_bf16 == 0 ||
          dispatch->hidden_input_bytes == 0u)) ||
        (runner->stage_index == 0u &&
         (dispatch->hidden_input_bf16 != 0 ||
          dispatch->hidden_input_bytes != 0u)) ||
        (runner->owns_final_head == 0u &&
         (dispatch->hidden_output_bf16 == 0 ||
          dispatch->hidden_output_bytes == 0u)) ||
        (runner->owns_final_head != 0u &&
         (dispatch->hidden_output_bf16 != 0 ||
          dispatch->hidden_output_bytes != 0u)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    hidden_bytes = (uint64_t)dispatch->row_count *
        SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
        SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
    if ((runner->stage_index != 0u &&
         hidden_bytes > dispatch->hidden_input_bytes) ||
        (runner->owns_final_head == 0u &&
         hidden_bytes > dispatch->hidden_output_bytes))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDsv4StageRunnerValidateDispatch(
	const SparkDsv4StageRunner *runner,
	const SparkDsv4StageRunnerDispatch *dispatch)
{
	SparkStatus status;
	status = SparkDsv4StageRunnerValidateDispatchShape(runner,dispatch);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkDsv4StageRunnerValidateDispatchBoundaries(runner,dispatch);
}

static void SparkDsv4StageRunnerBuildFrame(
    const SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerDispatch *dispatch,
    SparkDsv4ResidentDecodeStageFrameContext *context,
    SparkDsv4PrefillBatchView *prefill_batch,
    SparkDsv4DecodeBatchView *decode_batch,
    SparkModelDriverBuffer *buffers,
    SparkModelDriverFrame *frame)
{
    uint32_t is_prefill;
    uint32_t buffer_count;
    uint32_t output_buffer_index;

    is_prefill = (dispatch->flags &
        SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL) != 0u ? 1u : 0u;
    memset(context, 0, sizeof(*context));
    context->abi_version =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
    context->descriptor_bytes = sizeof(*context);
    context->flags = is_prefill != 0u ?
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW :
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
    if (runner->stage_index != 0u)
    {
        context->flags |=
            SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER;
        context->hidden_input_bf16 = dispatch->hidden_input_bf16;
        context->hidden_input_bytes = dispatch->hidden_input_bytes;
    }
    if (runner->owns_final_head == 0u)
    {
        context->flags |=
            SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER;
        context->hidden_output_bf16 = dispatch->hidden_output_bf16;
        context->hidden_output_bytes = dispatch->hidden_output_bytes;
    }
    if (is_prefill != 0u)
    {
        memset(prefill_batch, 0, sizeof(*prefill_batch));
        prefill_batch->abi_version =
            SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION;
        prefill_batch->descriptor_bytes = (uint32_t)sizeof(*prefill_batch);
		prefill_batch->row_count = dispatch->row_count;
		prefill_batch->active_sequence_count = dispatch->active_sequence_count;
		prefill_batch->emit_count = dispatch->emit_count;
		prefill_batch->token_ids = dispatch->token_ids;
        prefill_batch->row_lane_indices = dispatch->row_lane_indices;
        prefill_batch->row_positions = dispatch->row_positions;
		prefill_batch->row_sequence_ids = dispatch->row_sequence_ids;
		prefill_batch->emit_row_indices = dispatch->emit_row_indices;
		prefill_batch->emit_lane_indices = dispatch->emit_lane_indices;
        context->prefill_batch = prefill_batch;
    }
    else
    {
        memset(decode_batch, 0, sizeof(*decode_batch));
        decode_batch->abi_version =
            SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
        decode_batch->descriptor_bytes = (uint32_t)sizeof(*decode_batch);
        decode_batch->row_count = dispatch->row_count;
        decode_batch->row_lane_indices = dispatch->row_lane_indices;
        decode_batch->row_positions = dispatch->row_positions;
        decode_batch->row_sequence_ids = dispatch->row_sequence_ids;
        context->decode_batch = decode_batch;
    }
    memset(buffers, 0,
        sizeof(*buffers) * SPARK_DSV4_STAGE_RUNNER_MAX_DRIVER_BUFFERS);
    buffer_count = 0u;
    buffers[buffer_count].slot = buffer_count;
    buffers[buffer_count].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
    buffers[buffer_count].address = (void *)dispatch->token_ids;
    buffers[buffer_count].bytes =
        (uint64_t)dispatch->row_count * sizeof(uint32_t);
    buffer_count += 1u;
    if (runner->owns_final_head != 0u)
    {
        output_buffer_index = buffer_count;
        buffers[output_buffer_index].slot = output_buffer_index;
        buffers[output_buffer_index].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		buffers[output_buffer_index].address = dispatch->output_token_ids;
		buffers[output_buffer_index].bytes =
			(uint64_t)dispatch->active_sequence_count * sizeof(uint32_t);
        buffer_count += 1u;
    }
    memset(frame, 0, sizeof(*frame));
    frame->request_id = dispatch->request_id;
    frame->sequence_id = dispatch->sequence_id;
    frame->sequence_position = dispatch->sequence_position;
    frame->deadline_time_ns = dispatch->deadline_time_ns;
    frame->active_slot_count = dispatch->active_sequence_count;
    frame->new_token_count = dispatch->new_token_count;
    frame->priority = dispatch->priority;
    frame->flags = is_prefill != 0u ?
        SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
    frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    frame->program_id = runner->program->program_id;
	frame->execution_stream = runner->execution_stream;
    frame->buffers = buffers;
    frame->buffer_count = buffer_count;
    frame->user_context = context;
    frame->residency = dispatch->residency;
    frame->completion_function = dispatch->completion_function;
    frame->completion_context = dispatch->completion_context;
}

static SparkStatus SparkDsv4StageRunnerAdmit(
    SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerDispatch *dispatch,
    SparkModelDriverFrame *frame)
{
    SparkModelDriverAdmissionRequest request;
    SparkModelDriverAdmissionDecision decision;
    SparkStatus status;

    if ((runner->flags &
            SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    memset(&request, 0, sizeof(request));
    request.descriptor_bytes = sizeof(request);
    request.program_id = runner->program->program_id;
    request.request_id = dispatch->request_id;
    request.sequence_id = dispatch->sequence_id;
    request.sequence_position = dispatch->sequence_position;
    request.deadline_time_ns = dispatch->deadline_time_ns;
    request.active_slot_count = dispatch->active_sequence_count;
    request.new_token_count = dispatch->new_token_count;
    request.priority = dispatch->priority;
    request.frame_flags = frame->flags;
    request.residency = dispatch->residency;
	SparkModelDriverInitializeAdmissionDecision(&decision);
    status = runner->driver_interface->admit(
        runner->driver_instance, &request, &decision);
    if (status != SPARK_STATUS_OK)
    {
        runner->stats.last_status = (uint32_t)status;
        return status;
    }
	if (SparkModelDriverAdmissionDecisionIsValid(&decision) == 0u)
	{
		runner->stats.last_status = SPARK_STATUS_ABI_MISMATCH;
		return SPARK_STATUS_ABI_MISMATCH;
	}
	runner->stats.last_admission_rejection = decision.rejection_reason;
    if (decision.accepted == 0u)
    {
        runner->stats.rejected_count += 1u;
        return decision.rejection_reason ==
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY ?
            SPARK_STATUS_BUSY : SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    runner->stats.admitted_count += 1u;
	return SparkModelDriverApplyAdmissionDecision(&decision, frame);
}

SparkStatus SparkDsv4StageRunnerInitialize(
    SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerConfiguration *configuration)
{
    SparkStatus status;

    if (runner == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkDsv4StageRunnerValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    memset(runner, 0, sizeof(*runner));
    runner->abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
    runner->descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_BYTES;
    runner->flags = configuration->flags;
    runner->stage_index = configuration->stage_index;
    runner->stage_count = configuration->stage_count;
	runner->max_active_sequence_count =
		configuration->max_active_sequence_count;
	runner->max_input_row_count = configuration->max_input_row_count;
	runner->resident_sequence_capacity = configuration->resident_sequence_capacity;
    runner->owns_embedding = configuration->stage_index == 0u ? 1u : 0u;
    runner->owns_final_head = configuration->stage_index + 1u ==
        configuration->stage_count ? 1u : 0u;
    runner->driver_interface = configuration->driver_interface;
    runner->driver_instance = configuration->driver_instance;
    runner->program = configuration->program;
    runner->execution_stream = configuration->execution_stream;
    runner->stats.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
    runner->stats.descriptor_bytes =
        (uint32_t)sizeof(SparkDsv4StageRunnerStats);
    return SPARK_STATUS_OK;
}

SparkStatus SparkDsv4StageRunnerSubmit(
    SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerDispatch *dispatch)
{
    SparkDsv4ResidentDecodeStageFrameContext context;
    SparkDsv4PrefillBatchView prefill_batch;
    SparkDsv4DecodeBatchView decode_batch;
    SparkModelDriverBuffer buffers[2];
    SparkModelDriverFrame frame;
    SparkStatus status;

    status = SparkDsv4StageRunnerValidateDispatch(runner, dispatch);
    if (status != SPARK_STATUS_OK)
    {
        if (runner != 0)
        {
            runner->stats.last_status = (uint32_t)status;
            runner->stats.rejected_count += 1u;
        }
        return status;
    }
    SparkDsv4StageRunnerBuildFrame(
        runner, dispatch, &context, &prefill_batch, &decode_batch,
        buffers, &frame);
    status = SparkDsv4StageRunnerAdmit(runner, dispatch, &frame);
    if (status != SPARK_STATUS_OK)
    {
        runner->stats.last_status = (uint32_t)status;
        return status;
    }
    status = runner->program->submit(runner->driver_instance, &frame);
    runner->stats.last_status = (uint32_t)status;
    if (status == SPARK_STATUS_OK)
    {
        runner->stats.submitted_count += 1u;
    }
    else
    {
        runner->stats.submit_failed_count += 1u;
    }
    return status;
}

SparkStatus SparkDsv4StageRunnerGetStats(
    const SparkDsv4StageRunner *runner,
    SparkDsv4StageRunnerStats *stats_out)
{
    if (runner == 0 || stats_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *stats_out = runner->stats;
    return SPARK_STATUS_OK;
}
