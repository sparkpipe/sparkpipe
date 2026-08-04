#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_mtp_tree.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_model_driver_support.h"

static SparkStatus SparkGlm52ProductionRunnerValidateProgram(
    const SparkModelDriverProgramDescriptor *program)
{
    uint32_t missing_flags;

    if ( program == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( program->submit == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    missing_flags =
        (SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_REQUIRED_PROGRAM_FLAGS &
         ~program->flags);
    if ( missing_flags != 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ProductionRunnerValidateConfiguration(
    const SparkResidentDecodeStageProductionRunnerConfiguration *configuration)
{
    SparkStatus status;

    if ( configuration == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( configuration->abi_version !=
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION )
        return SPARK_STATUS_ABI_MISMATCH;
    if ( configuration->descriptor_bytes !=
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES )
        return SPARK_STATUS_ABI_MISMATCH;
    if ( (configuration->flags &
        ~SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_KNOWN_FLAGS) != 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( configuration->driver_interface == 0 ||
        configuration->driver_instance == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( configuration->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION )
    {
        if ( configuration->driver_interface->admit == 0 )
            return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ProductionRunnerValidateProgram(configuration->program);
    if ( status != SPARK_STATUS_OK )
        return status;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ProductionRunnerValidateDispatchShape(
    const SparkModelDriverProgramDescriptor *program,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch)
{
    const SparkModelDriverProgramProfile *profile;
    uint64_t execution_row_count;
    uint32_t prefill;

    if ( dispatch == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( dispatch->abi_version !=
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION )
        return SPARK_STATUS_ABI_MISMATCH;
    if ( dispatch->descriptor_bytes !=
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES )
        return SPARK_STATUS_ABI_MISMATCH;
    if ( (dispatch->flags &
        ~SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_KNOWN_FLAGS) != 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( dispatch->active_sequence_count == 0u ||
        dispatch->new_token_count == 0u ||
        dispatch->kv_block_table == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( dispatch->logical_lane_count == 0u ||
        dispatch->rows_per_lane == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    execution_row_count =
        (uint64_t)dispatch->logical_lane_count *
        (uint64_t)dispatch->rows_per_lane;
    prefill = (dispatch->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL) != 0u;
    if ( execution_row_count != dispatch->active_sequence_count ||
        dispatch->prefill_view != 0 ||
        (prefill != 0u &&
         (dispatch->rows_per_lane != dispatch->new_token_count ||
          dispatch->new_token_count >
            SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH)) ||
        (prefill == 0u &&
         dispatch->rows_per_lane >
            SPARK_GLM52_MODEL_MAX_SPECULATIVE_ROWS_PER_LANE) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( (dispatch->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u &&
        (prefill != 0u ||
         dispatch->rows_per_lane !=
            SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( dispatch->pipeline_slot >=
        SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT )
        return SPARK_STATUS_INVALID_ARGUMENT;
    profile = program->profile;
    if ( profile != 0 && profile->max_active_slots != 0u &&
        dispatch->active_sequence_count > profile->max_active_slots )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    if ( profile != 0 && profile->max_new_tokens != 0u &&
        prefill == 0u &&
        dispatch->new_token_count > profile->max_new_tokens )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ProductionRunnerValidateDispatch(
    const SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch)
{
    SparkStatus status;

    status = SparkGlm52ProductionRunnerValidateDispatchShape(
        runner->program,
        dispatch);
    if ( status != SPARK_STATUS_OK )
        return status;
    if ( (runner->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT) != 0u &&
        dispatch->hidden_input_transport_session == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( (runner->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT) != 0u &&
        dispatch->hidden_output_transport_session == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

static void SparkGlm52ProductionRunnerBuildFrameContext(
    const SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch,
    SparkResidentDecodeStageFrameContext *frame_context)
{
    memset(frame_context, 0, sizeof(*frame_context));
    frame_context->abi_version =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
    frame_context->descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_DESCRIPTOR_BYTES;
    frame_context->flags =
        SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE;
    frame_context->logical_lane_count = dispatch->logical_lane_count;
    frame_context->rows_per_lane = dispatch->rows_per_lane;
    if ( (dispatch->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL) != 0u )
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME;
    }
    else if ( dispatch->rows_per_lane > 1u )
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_LAYER_MAJOR_SPECULATIVE_VERIFY;
    }
    if ( (dispatch->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u )
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_TREE_VERIFY;
    }
    if ((dispatch->flags &
            SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED) != 0u)
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_PRERECEIVED;
    }
    frame_context->kv_block_table = dispatch->kv_block_table;
    frame_context->hidden_input_transport_session =
        dispatch->hidden_input_transport_session;
    frame_context->hidden_output_transport_session =
        dispatch->hidden_output_transport_session;
    frame_context->hidden_input_packet = dispatch->hidden_input_packet;
    frame_context->hidden_output_packet = dispatch->hidden_output_packet;
    if ( dispatch->mtp_draft_token_budgets != 0 )
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS;
        frame_context->mtp_draft_token_budgets =
            dispatch->mtp_draft_token_budgets;
    }
    if ( dispatch->dspark_hidden_tap_plan != 0 )
    {
        uint32_t tap_index;
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MODEL_HIDDEN_TAPS;
        frame_context->model_hidden_tap_plan =
            dispatch->dspark_hidden_tap_plan;
        if ( dispatch->dspark_hidden_tap_outputs_bf16 != 0 )
        {
            for ( tap_index = 0u; tap_index < SPARK_GLM52_DSPARK_AUX_LAYER_COUNT; ++tap_index )
                frame_context->model_hidden_tap_output_bf16[tap_index] =
                    dispatch->dspark_hidden_tap_outputs_bf16[tap_index];
        }
        frame_context->model_hidden_tap_lane_stride_bytes =
            dispatch->dspark_hidden_tap_lane_stride_bytes;
    }
    if ( dispatch->hidden_input_transport_session != 0 )
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT;
        frame_context->hidden_input_post_receive_function =
            SparkHiddenTransportPostReceive;
    }
    if ( dispatch->hidden_output_transport_session != 0 )
    {
        frame_context->flags |=
            SPARK_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
        frame_context->hidden_output_send_function = SparkHiddenTransportSend;
    }
    (void)runner;
}

static void SparkGlm52ProductionRunnerBuildAdmissionRequest(
    const SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch,
    SparkModelDriverAdmissionRequest *request)
{
    memset(request, 0, sizeof(*request));
    request->descriptor_bytes =
        ((uint32_t)sizeof(SparkModelDriverAdmissionRequest));
    request->program_id = runner->program_id;
    request->request_id = dispatch->request_id;
    request->sequence_id = dispatch->sequence_id;
    request->sequence_position = dispatch->sequence_position;
    request->deadline_time_ns = dispatch->deadline_time_ns;
    request->active_slot_count = dispatch->active_sequence_count;
    request->new_token_count = dispatch->new_token_count;
    request->priority = dispatch->priority;
    request->frame_flags = dispatch->flags;
    request->residency = dispatch->residency;
}

static void SparkGlm52ProductionRunnerBuildFrame(
    const SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch,
    const SparkResidentDecodeStageFrameContext *frame_context,
    SparkModelDriverFrame *frame)
{
    memset(frame, 0, sizeof(*frame));
    frame->request_id = dispatch->request_id;
    frame->sequence_id = dispatch->sequence_id;
    frame->sequence_position = dispatch->sequence_position;
    frame->deadline_time_ns = dispatch->deadline_time_ns;
    frame->active_slot_count = dispatch->active_sequence_count;
    frame->new_token_count = dispatch->new_token_count;
    frame->priority = dispatch->priority;
    frame->flags = dispatch->flags;
    frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    frame->program_id = runner->program_id;
    frame->execution_stream = runner->execution_stream;
    frame->residency = dispatch->residency;
    frame->user_context = (void *)frame_context;
    frame->completion_function = dispatch->completion_function;
    frame->completion_context = dispatch->completion_context;
    frame->scalar[SPARK_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX] =
        dispatch->pipeline_slot;
}

static SparkStatus SparkGlm52ProductionRunnerAdmit(
    SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch,
    SparkModelDriverFrame *frame)
{
    SparkModelDriverAdmissionRequest request;
    SparkModelDriverAdmissionDecision decision;
    SparkStatus status;

    if ( (runner->flags &
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION) == 0u )
        return SPARK_STATUS_OK;
    SparkGlm52ProductionRunnerBuildAdmissionRequest(runner, dispatch, &request);
	SparkModelDriverInitializeAdmissionDecision(&decision);
    status = runner->driver_interface->admit(
        runner->driver_instance,
        &request,
        &decision);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( SparkModelDriverAdmissionDecisionIsValid(&decision) == 0u )
		return SPARK_STATUS_ABI_MISMATCH;
    if ( decision.accepted == 0u )
    {
        runner->stats.last_admission_rejection = decision.rejection_reason;
        runner->stats.rejected_count += 1u;
        fprintf(
            stderr,
            "production_runner_admit_reject reason=%u request=%llu sequence=%llu position=%llu active=%u tokens=%u flags=0x%08x available=%u pressure=%u\n",
            decision.rejection_reason,
            (unsigned long long)request.request_id,
            (unsigned long long)request.sequence_id,
            (unsigned long long)request.sequence_position,
            request.active_slot_count,
            request.new_token_count,
            request.frame_flags,
            decision.available_dispatch_slot_count,
            decision.private_queue_pressure);
        return SPARK_STATUS_BUSY;
	}
	runner->stats.admitted_count += 1u;
	return SparkModelDriverApplyAdmissionDecision(&decision, frame);
}

SparkStatus SparkResidentDecodeStageProductionRunnerInitialize(
    SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerConfiguration *configuration)
{
    SparkStatus status;

    if ( runner == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52ProductionRunnerValidateConfiguration(configuration);
    if ( status != SPARK_STATUS_OK )
        return status;
    memset(runner, 0, sizeof(*runner));
    runner->abi_version =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    runner->descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_BYTES;
    runner->flags = configuration->flags;
    runner->program_id = configuration->program->program_id;
    runner->driver_interface = configuration->driver_interface;
    runner->driver_instance = configuration->driver_instance;
    runner->program = configuration->program;
    runner->execution_stream = configuration->execution_stream;
    runner->stats.abi_version =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    runner->stats.descriptor_bytes =
        SPARK_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_STATS_BYTES;
    return SPARK_STATUS_OK;
}

SparkStatus SparkResidentDecodeStageProductionRunnerSubmit(
    SparkResidentDecodeStageProductionRunner *runner,
    const SparkResidentDecodeStageProductionRunnerDispatch *dispatch)
{
    SparkResidentDecodeStageFrameContext frame_context;
    SparkModelDriverFrame frame;
    SparkStatus status;

    if ( runner == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    status = SparkGlm52ProductionRunnerValidateDispatch(runner, dispatch);
    if ( status != SPARK_STATUS_OK )
    {
        runner->stats.last_status = (uint32_t)status;
        runner->stats.rejected_count += 1u;
        fprintf(
            stderr,
            "production_runner_validate_reject status=%d flags=0x%08x active=%u tokens=%u slot=%u kv=%p in_session=%p out_session=%p prefill_view=%p runner_flags=0x%08x\n",
            (int32_t)status,
            dispatch != 0 ? dispatch->flags : 0u,
            dispatch != 0 ? dispatch->active_sequence_count : 0u,
            dispatch != 0 ? dispatch->new_token_count : 0u,
            dispatch != 0 ? dispatch->pipeline_slot : 0u,
            dispatch != 0 ? (const void *)dispatch->kv_block_table : 0,
            dispatch != 0 ? (void *)dispatch->hidden_input_transport_session : 0,
            dispatch != 0 ? (void *)dispatch->hidden_output_transport_session : 0,
            dispatch != 0 ? (const void *)dispatch->prefill_view : 0,
            runner->flags);
        return status;
    }
    SparkGlm52ProductionRunnerBuildFrameContext(runner, dispatch, &frame_context);
    SparkGlm52ProductionRunnerBuildFrame(runner, dispatch, &frame_context, &frame);
    status = SparkGlm52ProductionRunnerAdmit(runner, dispatch, &frame);
    if ( status != SPARK_STATUS_OK )
    {
        runner->stats.last_status = (uint32_t)status;
        return status;
    }
    status = runner->program->submit(runner->driver_instance, &frame);
    runner->stats.last_status = (uint32_t)status;
    if ( status == SPARK_STATUS_OK )
        runner->stats.submitted_count += 1u;
    else
    {
        runner->stats.submit_failed_count += 1u;
        if ( status != SPARK_STATUS_BUSY )
            fprintf(
                stderr,
                "production_runner_submit_failed status=%d frame_flags=0x%08x active=%u tokens=%u slot=%u position=%llu frame_context=%p\n",
                (int32_t)status,
                frame.flags,
                frame.active_slot_count,
                frame.new_token_count,
                frame.driver_dispatch_slot,
                (unsigned long long)frame.sequence_position,
                frame.user_context);
    }
    return status;
}

SparkStatus SparkResidentDecodeStageProductionRunnerProgress(
    SparkResidentDecodeStageProductionRunner *runner)
{
    SparkModelDriverRuntimeSnapshot snapshot;

    if ( runner == 0 ||
        runner->driver_interface == 0 ||
        runner->driver_interface->snapshot == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(&snapshot,0,sizeof(snapshot));
    return runner->driver_interface->snapshot(
        runner->driver_instance,
        runner->program_id,
        &snapshot);
}

SparkStatus SparkResidentDecodeStageProductionRunnerWaitIdle(
    SparkResidentDecodeStageProductionRunner *runner,
    uint32_t max_poll_count)
{
    SparkModelDriverRuntimeSnapshot snapshot;
    SparkStatus status;
    struct timespec sleep_interval;
    uint32_t poll_index;

    if ( runner == 0 ||
        runner->driver_interface == 0 ||
        runner->driver_interface->snapshot == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    for (poll_index = 0u; poll_index < max_poll_count; ++poll_index)
    {
        memset(&snapshot,0,sizeof(snapshot));
        status = runner->driver_interface->snapshot(
            runner->driver_instance,
            runner->program_id,
            &snapshot);
        if (status != SPARK_STATUS_OK)
            return status;
        if (snapshot.active_submission_count == 0u &&
            snapshot.completed_count >= snapshot.submitted_count)
            return SPARK_STATUS_OK;
        sleep_interval.tv_sec = 0;
        sleep_interval.tv_nsec = 200000;
        (void)nanosleep(&sleep_interval,0);
    }
    return SPARK_STATUS_BUSY;
}

SparkStatus SparkResidentDecodeStageProductionRunnerGetStats(
    const SparkResidentDecodeStageProductionRunner *runner,
    SparkResidentDecodeStageProductionRunnerStats *stats_out)
{
    if ( runner == 0 || stats_out == 0 )
        return SPARK_STATUS_INVALID_ARGUMENT;
    *stats_out = runner->stats;
    return SPARK_STATUS_OK;
}
