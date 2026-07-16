#!/usr/bin/env python3
from pathlib import Path


def test_final_stage_has_hidden_only_builtin_launcher(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    for bucket in (16, 32, 64, 128, 256, 512, 1024):
        needle = (
            "SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_ENTRY"
            f"(stage_index, SPARK_GLM52_STAGE_PLAN_BUCKET_B{bucket}, final_token_stage)")
        assert needle in source
    assert "SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(12u, 0u)" in source
    assert "SPARK_GLM52_EXACT_PP13_BUILTIN_AOT_LAUNCHER_BUCKETS(12u, 1u)" in source


def test_exact_pp13_final_stage_can_run_hidden_only(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    rejected_contract = (
        "final_token_stage !=\n"
        "            (exact_stage_slice_plan->first_layer_index + 6u =="
    )
    allowed_contract = (
        "(final_token_stage != 0u &&\n"
        "         exact_stage_slice_plan->first_layer_index + 6u !="
    )
    assert rejected_contract not in source
    assert allowed_contract in source


def test_pp13_rank_capacity_is_not_fixed_batch(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    reserved_block_start = source.index("node->reserved_execution_flags =")
    reserved_block_end = source.index("if ((state->rank_plan.flags &", reserved_block_start)
    reserved_block = source[reserved_block_start:reserved_block_end]
    assert "SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH" not in reserved_block


def test_pp13_builder_uses_compressed_absorbed_mla(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    wire_start = source.index("static void SparkGlm52Pp13BuilderWireLayer(")
    wire_end = source.index(
        "static void SparkGlm52Pp13BuilderConfigureMtpLayer(", wire_start)
    wire_body = source[wire_start:wire_end]
    assert "node->key_nope_cache_bf16 = 0;" in wire_body
    assert "node->value_cache_bf16 = 0;" in wire_body
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_"
            "ABSORBED_LATENT" in wire_body)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_"
            "TILED_ONLINE_SOFTMAX" not in wire_body)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_"
            "TILED_ONLINE_ATTENTION" not in wire_body)
    assert "void *key_nope_cache;" not in source
    assert "void *value_cache;" not in source
    assert "ALLOC_FIELD(key_nope_cache," not in source
    assert "ALLOC_FIELD(value_cache," not in source
    assert "ZERO_FIELD(key_nope_cache," not in source
    assert "ZERO_FIELD(value_cache," not in source


def test_pp13_rank_enables_read_only_dsa_fragment_prefetch(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    assert "SparkGlm52Pp13BuilderWireDsaFragmentPrefetch" in source
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_"
            "READ_ONLY_REQUIRED_CAPABILITIES" in source)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_"
            "PAYLOAD_FLAG_L2_PREFETCH_ONLY" in source)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_"
            "DSA_KV_FRAGMENT_TRANSPORT" in source)
    assert "layer->node.dsa_kv_fragment_prefetch_plan = &layer->dsa_prefetch_plan;" in source
    assert ("state->rank_plan.execution_row_capacity" in source)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_BLOCK_COUNT" in source)
    assert "(void *)layer->dsa_selection_event" in source
    assert "(void *)layer->dsa_prefetch_event" in source
    assert "(void *)state->dsa_selection_event" not in source
    assert "(void *)state->dsa_prefetch_event" not in source


def test_pp13_bulk_prefill_has_one_embedding_kernel(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    assert source.count(
        "__global__ static void SparkGlm52Pp13BuilderGatherPrefillEmbeddingKernel(") == 1
    assert "work_packet->prefill_token_ids[" in source
    assert "output_bf16_words[word_index] = 0u;" in source


def test_rdma_transport_is_multilane_and_event_driven(root: Path) -> None:
    source = (root / "modules" / "hidden_transport_spark_host_rdma_verbs.cu").read_text(
        encoding="utf-8")
    send_start = source.index("static SparkStatus SparkHiddenSparkHostRdmaSend(")
    destroy_start = source.index("static void SparkHiddenSparkHostRdmaDestroyState(", send_start)
    send_body = source[send_start:destroy_start]
    lane_post_start = source.index(
        "static SparkStatus SparkHiddenSparkHostRdmaPostLaneWrites(")
    lane_post_end = source.index(
        "static SparkStatus SparkHiddenSparkHostRdmaPostPacketWrites(",
        lane_post_start)
    lane_post_body = source[lane_post_start:lane_post_end]
    scalar_prepare_start = source.index(
        "static SparkStatus SparkHiddenSparkHostRdmaPrepareInflightSend(")
    scalar_prepare_end = source.index(
        "static SparkStatus SparkHiddenSparkHostRdmaSend(",
        scalar_prepare_start)
    scalar_prepare_body = source[scalar_prepare_start:scalar_prepare_end]
    assert "SparkMemlinkBuildTransferPartition(" in source
    assert "cudaEventRecord(" in send_body
    assert "cudaEventQuery(" in send_body
    assert "cudaLaunchHostFunc(" in send_body
    assert "cudaStreamSynchronize(" not in send_body
    assert "SparkHiddenSparkHostRdmaWaitForPostedWrites" not in source
    assert "ibv_poll_cq(" not in lane_post_body
    assert "IBV_SEND_SIGNALED" in lane_post_body
    assert "outstanding_send_wr_counts" in lane_post_body
    assert "SparkHiddenSparkHostRdmaStagePacket(" in scalar_prepare_body
    assert "SparkHiddenSparkHostRdmaGetCachedMemoryRegion(" not in scalar_prepare_body
    assert ("return status == SPARK_STATUS_OK ? SPARK_STATUS_OK : "
            "SPARK_STATUS_BUSY;" in scalar_prepare_body)
    assert "cudaHostAlloc(" in source
    assert "ibv_reg_mr(" in source
    assert "SparkHiddenSparkHostRdmaRetireCompletedSends(state);" in source
    assert ("SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION" in
            (root / "include" / "sparkpipe" /
             "spark_hidden_transport.h").read_text(encoding="utf-8"))
    assert ("transport_interface.post_receive_batch =\n"
            "        SparkHiddenSparkHostRdmaPostReceiveBatch;" in source)
    assert ("transport_interface.send_batch = "
            "SparkHiddenSparkHostRdmaSendBatch;" in source)


def test_prebound_linear_plan_accepts_smaller_active_count(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageMaybeLaunchPreboundLinearPlan(")
    end = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchMtpDraft(",
        start)
    function_body = source[start:end]
    assert "linear_plan_active_mismatch" not in function_body
    assert "active_sequence_count != linear_plan->maximum_active_sequence_count" not in function_body


def test_fp8_linear_plans_require_scaled_gemm_backend(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchBlackwellBuiltInQuantizedTensorCoreLinearPlan(")
    end = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchLinear(",
        start)
    function_body = source[start:end]
    fp8_start = function_body.index(
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3")
    fp8_end = function_body.index("grid = dim3(", fp8_start)
    fp8_branch = function_body[fp8_start:fp8_end]
    assert "if (backend == 0)" in fp8_branch
    assert "return SPARK_STATUS_MODULE_NOT_VALIDATED;" in fp8_branch
    assert "LaunchFp8E4m3ActivationWeightLinearScaledGemmBackend" in fp8_branch
    assert "SupportedQuantizedBf16WmmaLinearKernel" not in fp8_branch


def test_pp13_builder_binds_all_fp8_linear_plans(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index("static SparkStatus SparkGlm52Pp13BuilderBindLayerPlans(")
    end = source.index("static SparkStatus SparkGlm52Pp13BuilderBindFp8Moe(", start)
    function_body = source[start:end]
    regular_bind = function_body.index(
        "SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedRegularLinearPlans(")
    fp8_bind = function_body.index(
        "SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearPlansScaledGemmBackend(")
    assert fp8_bind > regular_bind
    assert "&state->fp8_scaled_gemm_backend" in function_body[fp8_bind:]


def test_bulk_prefill_progresses_runner_after_each_chunk(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    progress_call = "SparkGlm52ResidentDecodeStageProductionRunnerProgress("
    runner_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderRunPrefillFrame(")
    prefill_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrefill(", runner_start)
    decode_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderDecode(", prefill_start)
    runner_body = source[runner_start:prefill_start]
    prefill_body = source[prefill_start:decode_start]
    assert progress_call in runner_body
    assert "SparkGlm52Pp13BuilderRunPrefillFrame(" in prefill_body
    assert "SPARK_GLM52_PP13_BUILDER_MAX_PREFILL_TOKENS" in prefill_body
    assert "token_offset += chunk_token_count" in prefill_body
    assert "state->prefill_frame_view" in source
    assert "false &&" not in source


def test_bulk_prefill_validates_the_runtime_view_stride(root: Path) -> None:
    builder = (root / "modules" / "glm52_resident_decode_stage" / "source" /
               "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                   encoding="utf-8")
    cuda = (root / "modules" / "glm52_resident_decode_stage" / "source" /
            "spark_glm52_sm121_required_decode_stage.cu").read_text(
                encoding="utf-8")
    wire_start = builder.index(
        "static void SparkGlm52Pp13BuilderWireLayerSerialPrefillPlan(")
    wire_end = builder.index(
        "static void SparkGlm52Pp13BuilderWireLayer(", wire_start)
    wire_body = builder[wire_start:wire_end]
    validate_start = cuda.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageValidatePagedChunkPrefillPlan(")
    validate_end = cuda.index(
        "static __device__ __forceinline__ uint32_t "
        "SparkGlm52ResidentDecodeStageResolvePagedPrefillCacheSlot(",
        validate_start)
    validate_body = cuda[validate_start:validate_end]
    assert (
        "SPARK_GLM52_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_RUNTIME_PREFILL_VIEW"
        in wire_body)
    runtime_stride = (
        "prompt_token_stride = prefill_frame_view != 0\n"
        "        ? prefill_frame_view->prompt_token_stride")
    static_stride = ": paged_prefill_plan->prompt_token_stride != 0u"
    assert runtime_stride in validate_body
    assert validate_body.index(runtime_stride) < validate_body.index(
        static_stride)


def test_decode_uses_one_tree_aware_work_packet_path(root: Path) -> None:
    builder = (root / "modules" / "glm52_resident_decode_stage" / "source" /
               "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                   encoding="utf-8")
    service = (root / "src" /
               "spark_glm52_pp13_service_backend.c").read_text(
                   encoding="utf-8")
    decode_start = builder.index(
        "static SparkStatus SparkGlm52Pp13BuilderDecode(")
    decode_end = builder.index(
        "static SparkStatus SparkGlm52Pp13BuilderTakeDsparkDraft(",
        decode_start)
    decode_body = builder[decode_start:decode_end]
    assert "SparkGlm52Pp13WorkControlBuildDecodePacket(" in decode_body
    assert "SparkGlm52Pp13BuilderSubmitWork(" in decode_body
    assert "SparkGlm52Pp13BuilderPrepareDeviceKvView(" not in decode_body
    assert "SparkGlm52ResidentDecodeStageProductionRunnerSubmit(" not in decode_body
    submit_start = builder.index(
        "static SparkStatus SparkGlm52Pp13BuilderSubmitWork(")
    submit_end = builder.index(
        "static uint64_t SparkGlm52Pp13BuilderProbeFnv64(", submit_start)
    submit_body = builder[submit_start:submit_end]
    assert ("work_packet->execution_batch_bucket >\n"
            "\t\t\tSPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET" in submit_body)
    assert ("work_packet->execution_batch_bucket >\n"
            "\t\t\tstate->rank_plan.logical_lane_capacity" not in submit_body)

    packet_start = service.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendBuildDecodeWorkPacket(")
    packet_end = service.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendBuildPrefillWorkPacket(",
        packet_start)
    packet_body = service[packet_start:packet_end]
    assert "SparkGlm52Pp13WorkControlBuildDecodePacketRange(" in packet_body
    assert "packet->flags =" not in packet_body
    assert "packet->rows_per_lane =" not in packet_body


def test_prefill_probe_hashes_the_exact_stage_input(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    probe_start = source.index(
        "static void SparkGlm52Pp13BuilderMaybeProbePrefillInputHidden(")
    prepare_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPreparePrefillFrame(",
        probe_start)
    probe_body = source[probe_start:prepare_start]
    prefill_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrefill(", prepare_start)
    decode_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderDecode(", prefill_start)
    prefill_body = source[prefill_start:decode_start]
    assert "SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES" in probe_body
    assert "SparkGlm52Pp13BuilderProbeFnv64(" in probe_body
    assert "SparkGlm52Pp13BuilderMaybeProbePrefillInputHidden(" in prefill_body


def test_fp8_phase_probe_targets_the_first_divergent_layer(root: Path) -> None:
    cuda_source = (root / "modules" / "glm52_resident_decode_stage" /
                   "source" /
                   "spark_glm52_sm121_required_decode_stage.cu").read_text(
                       encoding="utf-8")
    builder_source = (root / "modules" / "glm52_resident_decode_stage" /
                      "source" /
                      "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                          encoding="utf-8")
    probe_start = cuda_source.index(
        "static void SparkGlm52ResidentDecodeStageDeviceHashProbe(")
    probe_end = cuda_source.index(
        "static bool SparkGlm52ResidentDecodeStageFp8KvCachePlanIsUsableCuda(",
        probe_start)
    assert "node_context->layer_index != 0u" in cuda_source[
        probe_start:probe_end]
    assert "fp8_layer0_attention_probe" in builder_source


def test_fp8_validator_preserves_quantized_dense_execution(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" /
              "validation" /
              "spark_glm52_resident_decode_stage_cuda_validation.cu").read_text(
                  encoding="utf-8")
    start = source.index("static bool SparkValidationBindRequiredLinearPlans(")
    end = source.index(
        "static bool SparkValidationInitializeDenseLayerCacheAliases(", start)
    function_body = source[start:end]
    assert function_body.count("use_quantized_dense_plans == 0u &&") == 2


def test_speculative_verify_exposes_the_full_verifier_vector(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderFinalizePackedMtpVerify(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderEmitPackedDsparkVerifyCompletions(",
        start)
    function_body = source[start:end]
    assert "completion.token_count = work_packet->rows_per_lane;" in function_body
    assert ("completion.token_count = accepted_draft_count + 1u;" not in
            function_body)
    assert "execution_row_base + token_index" in function_body

    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderEmitPackedDsparkVerifyCompletions(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrepareDsparkStages(",
        start)
    function_body = source[start:end]
    assert "completion.token_count = work_packet->rows_per_lane;" in function_body
    assert ("completion.token_count = accepted_draft_count + 1u;" not in
            function_body)
    assert "execution_row_base + token_index" in function_body


def test_target_and_mtp_heads_use_distinct_fail_closed_tensor_core_gemms(
        root: Path) -> None:
    builder_source = (root / "modules" / "glm52_resident_decode_stage" /
                      "source" /
                      "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                          encoding="utf-8")
    stage_source = (root / "modules" / "glm52_resident_decode_stage" /
                    "source" /
                    "spark_glm52_sm121_required_decode_stage.cu").read_text(
                        encoding="utf-8")
    assert "MTP_HEAD_CHUNK_LANE_CAPACITY" not in builder_source
    start = builder_source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedyRows(")
    end = builder_source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedy(",
        start)
    row_function_body = builder_source[start:end]
    assert "cublasGemmEx(" in row_function_body
    assert "CUBLAS_GEMM_DEFAULT_TENSOR_OP" in row_function_body
    assert "row_count" in row_function_body
    start = end
    end = builder_source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpFullVocabGreedy(",
        start)
    function_body = builder_source[start:end]
    assert "active_sequence_count" in function_body
    assert "row_offset" in function_body
    assert "full_vocab_head_row_capacity" in function_body
    start = end
    end = builder_source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchTensorCoreFinalTokenHead(",
        start)
    mtp_head_body = builder_source[start:end]
    assert ("SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationWeightLinearScaledGemmBackend("
            in mtp_head_body)
    assert ("SparkGlm52Pp13BuilderMtpFullVocabTop2Kernel<uint16_t><<<" in
            mtp_head_body)
    assert "SparkGlm52Pp13BuilderMtpFullVocabAlternateKernel" not in builder_source
    assert "SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedy(" not in mtp_head_body
    start = builder_source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(")
    end = builder_source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", start)
    function_body = builder_source[start:end]
    assert "SparkGlm52Pp13BuilderLaunchMtpFullVocabGreedy(" in function_body
    assert "SparkGlm52Sm121RequiredDecodeStageLaunchFullVocabGreedy(" not in function_body
    assert "SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET" in builder_source
    assert ("state->exact_plan.final_token_launch_function =" in
            builder_source)
    assert "backend=cublas_bf16_tensor_core" in builder_source
    assert "backend=fp8_e4m3_block128_tensor_core" in builder_source
    assert "target_verifier_backend=cublas_bf16_tensor_core" in builder_source
    assert "SparkGlm52Pp13BuilderQuantizeMtpDraftHeadKernel<<<" in builder_source
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3" in
            builder_source)
    assert "state->mtp_draft_head_ready == 0u" in builder_source
    assert "fail_closed=1" in builder_source
    head_start = stage_source.index(
        "static SparkStatus "
        "SparkGlm52ResidentDecodeStageLaunchBuiltInFullVocabGreedyFinalTokenEpilogue(")
    head_end = stage_source.index(
        "extern \"C\" SparkStatus "
        "SparkGlm52Sm121RequiredDecodeStageLaunchFullVocabGreedy(", head_start)
    head_body = stage_source[head_start:head_end]
    assert "exact_stage_slice_plan->final_token_launch_function" in head_body
    assert "final_token_launch_function(" in head_body


def test_mtp_previous_target_position_contracts_are_explicit(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    model_header = (root / "include" / "sparkpipe" /
                    "spark_glm52_model.h").read_text(encoding="utf-8")
    assert ("#define SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA 1u" in
            model_header)
    assert ("#define SPARK_GLM52_MODEL_MTP_EXECUTION_POSITION_OFFSET 0u" in
            model_header)
    metadata_start = source.index(
        "__global__ static void SparkGlm52Pp13BuilderMtpMetadataKernel(")
    metadata_end = source.index(
        "__global__ static void SparkGlm52Pp13BuilderMtpStoreKernel(",
        metadata_start)
    metadata_body = source[metadata_start:metadata_end]
    assert "SPARK_GLM52_MODEL_MTP_EXECUTION_POSITION_OFFSET" in metadata_body
    load_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpPreviousTargets(")
    load_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderStoreMtpPreviousTarget(",
        load_start)
    load_body = source[load_start:load_end]
    assert "uint32_t previous_position_delta" in load_body
    assert "SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA" in load_body
    assert "+\n\t\t\t\tprevious_position_delta !=" in load_body
    assert "host_mtp_previous_positions[request_slot_index] + 1u" not in load_body
    assert source.count(
        "SPARK_GLM52_PP13_BUILDER_MTP_TARGET_PRECEDES_INPUT_POSITION") == 2
    assert "SPARK_GLM52_PP13_BUILDER_MTP_TARGET_SAME_INPUT_POSITION" not in source
    draft_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(")
    draft_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", draft_start)
    draft_body = source[draft_start:draft_end]
    assert "state->mtp_use_previous_for_draft" not in source
    assert "state->mtp_previous_target_hidden" not in draft_body
    assert "base_slot->normalized_hidden_bf16" in draft_body
    assert ("state,token_ids,base_slot->positions,hidden_bf16,draft_index" in
            draft_body)


def test_mtp_linear_plans_use_logical_rows(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    stage_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrepareStageLinearPlanRows(")
    mtp_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows(",
        stage_start)
    stage_body = source[stage_start:mtp_start]
    assert "state->mtp_layer.linear_binding" not in stage_body

    draft_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(")
    draft_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", draft_start)
    draft_body = source[draft_start:draft_end]
    prepare = "SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows("
    draft_loop = "for (draft_index = 0u;"
    assert draft_body.index(prepare) < draft_body.index(draft_loop)
    assert "state,active_sequence_count);" in draft_body


def test_mtp_draft_plan_builds_asymmetric_top2_tree(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", start)
    body = source[start:end]
    assert "state->host_mtp_draft_budgets[lane_index]" in body
    assert ("draft_token_count != "
            "SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT" in body)
    assert ("draft_index < "
            "SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT" in body)
    assert "alternate_required = draft_index == 0u ? 0u : 1u;" in body
    assert "SparkGlm52Pp13BuilderLaunchMtpFullVocabGreedy(" in body
    assert "SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX" in body
    assert "SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX" in body


def test_mtp_runtime_depth_specializes_and_caches_cuda_graphs(root: Path) -> None:
    builder = (root / "modules" / "glm52_resident_decode_stage" / "source" /
               "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                   encoding="utf-8")
    required = (root / "modules" / "glm52_resident_decode_stage" / "source" /
                "spark_glm52_sm121_required_decode_stage.cu").read_text(
                    encoding="utf-8")
    header = (root / "modules" / "glm52_resident_decode_stage" / "include" /
              "sparkpipe" /
              "spark_glm52_resident_decode_stage_firmware.h").read_text(
                  encoding="utf-8")
    assert ("mtp_draft_token_count == 0u\n"
            "\t\t\t? 0u : "
            "SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;" in builder)
    assert ("graph_draft_token_count == 0u\n"
            "\t\t\t? 0u : "
            "SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;" in builder)
    assert ("node_context->mtp_draft_plan->graph_draft_token_count" in
            required)
    assert "SparkGlm52ResidentDecodeStageFindCachedGraph(" in required
    assert "SparkGlm52ResidentDecodeStageRetainCurrentGraph(" in required
    assert "SparkGlm52ResidentDecodeStageDestroyGraphCache(" in required
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_GRAPH_VARIANT_COUNT" in
            header)
    assert ("SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_GRAPH_SPARE_ENTRY_COUNT" in
            header)


def test_mtp_runtime_failures_name_the_failing_phase(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpLayer(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(", start)
    body = source[start:end]
    for phase in (
            "mtp_metadata",
            "mtp_fusion",
            "mtp_eh_projection",
            "mtp_required_layer",
            "mtp_prepare_linear_rows",
            "mtp_full_vocab_greedy",
            "mtp_store_draft"):
        assert '"' + phase + '"' in body


def test_mtp_gpu_profile_is_graph_compatible_and_explicitly_enabled(
        root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    assert 'getenv("SPARKPIPE_MTP_GPU_PROFILE")' in source
    assert "SparkGlm52Pp13BuilderMtpGpuProfileKernel<<<1u,1u,0u,stream>>>" in source
    assert 'asm volatile("mov.u64 %0, %%globaltimer;"' in source
    assert "cudaEvent" not in source[source.index(
        "static SparkStatus SparkGlm52Pp13BuilderMarkMtpGpuProfile("):
        source.index(
            "static SparkStatus SparkGlm52Pp13BuilderReportMtpGpuProfile(")]
    for phase in (
            "MTP_GPU_PROFILE_START",
            "MTP_GPU_PROFILE_METADATA",
            "MTP_GPU_PROFILE_FUSION",
            "MTP_GPU_PROFILE_EH_PROJECTION",
            "MTP_GPU_PROFILE_REQUIRED_LAYER",
            "MTP_GPU_PROFILE_VOCAB_HEAD",
            "MTP_GPU_PROFILE_STORE"):
        assert phase in source
    assert 'state,"decode",work_packet->lane_count,maximum_draft_count' in source
    assert "state,profile_kind,lane_count,draft_token_count" in source
    assert ("SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,"
            '"verify_followup"' in source)
    assert ("*draft_token_count_out =\n"
            "\t\t\tSPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;" in source)
    assert "SparkGlm52Pp13BuilderPrepareSerialVerifierMtpDraft" not in source
    assert "SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT" not in source
    assert "graph_compatible=1" in source


def test_mtp_tree_copies_branch_history_before_attention(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    layer_start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchLayerBody(")
    layer_end = source.index(
        "static SparkStatus SparkGlm52Sm121RequiredDecodeStageSubmit(",
        layer_start)
    body = source[layer_start:layer_end]
    clone = "SparkGlm52ResidentDecodeStageMaybeCloneMtpTreeKvBlocks("
    prepare = "SparkGlm52ResidentDecodeStagePrepareKernel<<<"
    patch = "SparkGlm52ResidentDecodeStageMaybePatchMtpTreeAncestors("
    attention = "SparkGlm52ResidentDecodeStageLaunchAbsorbedLatentAttention("
    assert body.index(clone) < body.index(prepare)
    assert body.index(prepare) < body.index(patch)
    assert body.index(patch) < body.index(attention)
    assert "SparkGlm52ResidentDecodeStageMtpTreeCloneBlocksKernel" in source
    assert "SparkGlm52ResidentDecodeStageMtpTreePatchAncestorsKernel" in source
    assert "SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT" in source
    assert "SPARK_GLM52_MODEL_MTP_TREE_ANCESTOR_COPY_COUNT" in source


def test_mtp_tree_rebases_private_draft_cache_before_followup(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    finalize_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderFinalizePackedMtpVerify(")
    finalize_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLoadMtpPreviousTargets(",
        finalize_start)
    finalize = source[finalize_start:finalize_end]
    assert finalize.index("SparkGlm52Pp13BuilderCompactSpeculativeKvRows(") < \
        finalize.index("SparkGlm52Pp13BuilderRebaseMtpTreeCache(")
    assert finalize.index("SparkGlm52Pp13BuilderRebaseMtpTreeCache(") < \
        finalize.index("SparkGlm52Pp13BuilderLaunchVerifierMtpDraft(")
    for function_name in (
            "SparkGlm52Pp13BuilderLaunchMtpKvRollback(",
            "SparkGlm52Pp13BuilderLaunchMtpKvPromotion("):
        start = source.index("static SparkStatus " + function_name)
        end = source.index("\nstatic ", start + 1)
        assert "state->mtp_layer" not in source[start:end]


def test_mtp_retry_cleanup_preserves_resolution_receipt(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    first = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderDiscardMtpKvTransactions(")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderDiscardMtpKvTransactions(",
        first + 1)
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderRecordMtpKvTransactions(",
        start)
    body = source[start:end]
    non_release = body.index(
        "\tfor (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)",
        body.index("\t\treturn SPARK_STATUS_OK;\n\t}"))
    retry_cleanup = body[non_release:]
    assert "SparkGlm52Pp13BuilderClearActiveMtpKvTransaction(" in retry_cleanup
    assert "memset(transaction,0,sizeof(*transaction));" not in retry_cleanup


def test_mtp_transaction_uses_expanded_execution_row(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderRecordMtpKvTransactions(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderLaunchMtpKvRollback(",
        start)
    body = source[start:end]
    assert "execution_row_index = lane_index;" in body
    assert ("state->host_lane_physical_block_counts[execution_row_index]"
            in body)
    assert ("((uint64_t)execution_row_index * state->kv_state.lane_stride)"
            in body)
    assert "state->host_lane_physical_block_counts[lane_index]" not in body


def test_mtp_kv_resolution_scratch_is_shared_and_execution_row_sized(
        root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    shared_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderInitializeSharedBuffers(")
    shared_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderValidateConfiguration(",
        shared_start)
    shared = source[shared_start:shared_end]
    mtp_start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderInitializeMtp(")
    mtp_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderBuildLayer(",
        mtp_start)
    mtp = source[mtp_start:mtp_end]
    assert "max_active = state->rank_plan.execution_row_capacity;" in shared
    assert "(void **)&state->device_mtp_request_slot_indices" in shared
    assert "max_active * sizeof(uint32_t)" in shared
    assert "state->host_mtp_request_slot_indices =" in shared
    assert "(size_t)(max_active * sizeof(uint32_t))" in shared
    assert "state->host_mtp_request_slot_indices == 0" in shared
    assert "device_mtp_request_slot_indices" not in mtp
    assert "host_mtp_request_slot_indices" not in mtp


def test_mtp_serial_train_continuation_keeps_transaction_open(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static uint32_t SparkGlm52Pp13BuilderContinuesMtpKvTransaction(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderApplyMtpKvResolutions(",
        start)
    helper = source[start:end]
    assert "work_packet->speculative_token_index == 0u" in helper
    assert "SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) == 0u" in helper
    assert "transaction->request_id != lane->request_id" in helper
    assert "transaction->sequence_id != lane->sequence_id" in helper
    assert "transaction->proposed_token_count != lane->speculative_token_count" in helper
    assert "transaction->base_position == base_position" in helper
    apply_end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderBuildResidentKvTable(", end)
    apply_body = source[end:apply_end]
    assert ("SparkGlm52Pp13BuilderContinuesMtpKvTransaction(\n"
            "\t\t\t\t\twork_packet,lane,transaction) == 0u" in apply_body)


def test_attached_resident_decode_preserves_mtp_resolution(root: Path) -> None:
    backend = (root / "src" / "spark_glm52_pp13_service_backend.c").read_text(
        encoding="utf-8")
    daemon = (root / "tools" / "sparkpipe_glm52_cuda_residentd.c").read_text(
        encoding="utf-8")
    backend_start = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendBuildDecodeResidentPayload(")
    backend_end = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendSubmitDecodeToResident(",
        backend_start)
    backend_body = backend[backend_start:backend_end]
    daemon_start = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdBuildDecodeWorkPacket(")
    daemon_end = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdHandleSubmitPrefill(",
        daemon_start)
    daemon_body = daemon[daemon_start:daemon_end]
    for field in ("mtp_resolution_proposed_token_count",
                  "mtp_resolution_accepted_token_count"):
        assert f"target_lane->{field}" in backend_body
        assert f"lane->{field}" in backend_body
        assert f"target_lane->{field}" in daemon_body
        assert f"source_lane->{field}" in daemon_body
    assert "SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE" in daemon_body


def test_attached_prefill_uses_nonblocking_ordered_forwarding(root: Path) -> None:
    backend = (root / "src" /
               "spark_glm52_pp13_service_backend.c").read_text(
                   encoding="utf-8")
    daemon = (root / "tools" /
              "sparkpipe_glm52_cuda_residentd.c").read_text(
                  encoding="utf-8")
    assert (
        "#define SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS \\\n"
        "\tSPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS"
        in backend)
    assert "SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_TOKENS 512u" not in backend
    submit_start = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendSubmitPrefillToResident(")
    submit_end = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendBuildDecodeResidentPayload(",
        submit_start)
    submit_body = backend[submit_start:submit_end]
    forward = "SparkGlm52Pp13ServiceBackendForwardPrefillPacket("
    local = "SparkGlm52Pp13ServiceBackendSubmitPrefillPacket("
    assert submit_body.index(forward) < submit_body.index(local)
    assert "retry_count" not in submit_body
    assert "nanosleep" not in submit_body
    forward_start = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendForwardPrefillPacket(")
    forward_end = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendSubmitPrefillPacket(",
        forward_start)
    forward_body = backend[forward_start:forward_end]
    assert "SparkGlm52Pp13ServiceBackendEnqueueWorkPacket(" in forward_body
    assert "SparkGlm52Pp13ServiceBackendPumpWorkOutput(state);" in forward_body
    assert "SparkGlm52Pp13ServiceBackendDrainWorkOutput" not in backend
    assert "poll(&descriptor,1u,-1)" not in backend
    defer_start = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdHandleSubmitPrefill(")
    defer_end = daemon.index(
        "static uint32_t SparkGlm52CudaResidentdPumpPendingPrefill(",
        defer_start)
    defer_body = daemon[defer_start:defer_end]
    assert "packet->new_token_count != 1u" not in defer_body
    assert (
        "(packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u"
        in defer_body)
    assert "runtime->pending_prefill_work.packet = *packet;" in defer_body
    assert "runtime->pending_prefill_work.client_fd = client_fd;" in defer_body
    assert "runtime->pending_prefill_active = 1u;" in defer_body
    assert "return SPARK_STATUS_OK;" in defer_body
    pump_start = defer_end
    pump_end = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdHandleSubmitDecode(",
        pump_start)
    pump_body = daemon[pump_start:pump_end]
    assert "SparkGlm52CudaResidentdEnqueueWork(" in pump_body
    assert "if (status == SPARK_STATUS_BUSY)\n        return 0u;" in pump_body
    assert "deferred_prefill_work_failed" in pump_body
    main_start = daemon.index("int main(int argc, char **argv)")
    main_body = daemon[main_start:]
    assert "SparkGlm52CudaResidentdPumpPendingPrefill(" in main_body
    assert "active_client_fd" not in daemon
    poll_start = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdBuildPollFds(")
    poll_end = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdProgressTransport(",
        poll_start)
    poll_body = daemon[poll_start:poll_end]
    assert "SPARK_GLM52_CUDA_RESIDENTD_OUTPUT_CONTROL_LIMIT" in poll_body
    assert "POLLOUT" in poll_body
    assert "runtime->work_queue_count <" not in poll_body
    assert "SparkGlm52CudaResidentdWriteFull" not in daemon
    assert "SPARK_GLM52_CUDA_RESIDENTD_CONTROL_WRITE_TIMEOUT_MS" not in daemon
    assert "SparkGlm52CudaResidentdFlushClientOutput(" in daemon
    completion_start = daemon.index(
        "static void SparkGlm52CudaResidentdCompletion(")
    completion_end = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdOpenListenSocket(",
        completion_start)
    completion_body = daemon[completion_start:completion_end]
    assert "SparkGlm52CudaResidentdQueueCompletion(runtime,&message);" in (
        completion_body)
    assert "write(" not in completion_body
    assert "send(" not in completion_body
    connect_start = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendConnectCudaResident(")
    connect_end = backend.index(
        "static SparkStatus SparkGlm52Pp13ServiceBackendEnsureCudaResident(",
        connect_start)
    connect_body = backend[connect_start:connect_end]
    assert "hello.control_generation = state->session_id_base;" in connect_body
    generation_start = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdAdvanceControlGeneration(")
    generation_end = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdFillStats(",
        generation_start)
    generation_body = daemon[generation_start:generation_end]
    assert "control_generation < previous_generation" in generation_body
    assert "runtime->work_queue_head = 0u;" in generation_body
    assert "runtime->work_queue_count = 0u;" in generation_body
    assert "runtime->work_queue_error_count += dropped_work;" in generation_body
    assert "SparkGlm52CudaResidentdResetControlRuntime(" in generation_body
    assert "reset_control_generation(" in daemon
    assert "SparkHiddenTransportClose(runtime->input_transport_session);" in daemon
    assert "SparkGlm52CudaResidentdCreateDriverInstance(" in daemon
    enqueue_start = daemon.index(
        "static SparkStatus SparkGlm52CudaResidentdEnqueueWork(")
    enqueue_end = daemon.index(
        "static void SparkGlm52CudaResidentdPopQueuedWork(",enqueue_start)
    enqueue_body = daemon[enqueue_start:enqueue_end]
    assert "packet->control_generation != 0u" in enqueue_body
    assert "SparkGlm52CudaResidentdAdvanceControlGeneration(" in enqueue_body


def test_layer_body_failures_are_never_silent(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.rindex(
        "static SparkStatus SparkGlm52ResidentDecodeStageTraceLayerBodyStatus(")
    end = source.index("\n}\n", start)
    body = source[start:end]
    assert "getenv" not in body
    assert "layer_body_failed" in body


def test_plain_wide_decode_bypasses_dspark_finalizer(root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderStartPendingWorkFinalizer(")
    end = source.index(
        "static SparkStatus SparkGlm52Pp13BuilderProgress(",
        start)
    function_body = source[start:end]
    wide_finalize = "SparkGlm52Pp13BuilderEmitWideDecodeCompletions("
    dspark_guard = (
        "SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) != 0u")
    dspark_prepare = "SparkGlm52Pp13BuilderPrepareDsparkStages("
    assert wide_finalize in function_body
    assert dspark_guard in function_body
    assert dspark_prepare in function_body
    assert function_body.index(wide_finalize) < function_body.index(
        dspark_guard)
    assert function_body.index(dspark_guard) < function_body.index(
        dspark_prepare)


def test_resident_block_stride_is_independent_of_the_physical_pool(
        root: Path) -> None:
    builder = (root / "modules" / "glm52_resident_decode_stage" / "source" /
               "spark_glm52_pp13_node_context_builder_cuda.cu").read_text(
                   encoding="utf-8")
    module = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_resident_decode_stage_module.c").read_text(
                  encoding="utf-8")
    assignment = (
        "node->max_blocks_per_sequence =\n"
        "\t\tSPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;")
    assert builder.count(assignment) == 2
    assert "SparkGlm52Pp13BuilderResidentMaxBlocksPerSequence" not in builder
    assert ("node_context->max_blocks_per_sequence >\n"
            "            node_context->kv_block_count") not in module
    assert "physical_block_index >= node_context->kv_block_count" in module


def test_service_backend_namespaces_ids_per_live_session(root: Path) -> None:
    source = (root / "src" /
              "spark_glm52_pp13_service_backend.c").read_text(
                  encoding="utf-8")
    assert "state->request_api.next_sequence_id = state->session_id_base" in source
    assert "service_configuration.request_id_base = state->session_id_base" in source
    assert "state->cuda_resident_next_sequence_number = state->session_id_base" in source


def test_final_event_pump_detects_disconnect_before_send(root: Path) -> None:
    source = (root / "tools" /
              "sparkpipe_glm52_pp13_rank_daemon.c").read_text(
                  encoding="utf-8")
    start = source.index("static uint32_t SparkGlm52Pp13DaemonPumpFinalEvents(")
    end = source.index("static SparkStatus SparkGlm52Pp13DaemonInitialize(", start)
    function_body = source[start:end]
    receive = "SparkGlm52Pp13DaemonPumpFinalEventReceive(runtime)"
    send = "SparkGlm52Pp13DaemonPumpFinalEventSend(runtime)"
    assert function_body.index(receive) < function_body.index(send)


def test_rank_queue_does_not_overtake_a_deferred_sequence_position(
        root: Path) -> None:
    source = (root / "tools" /
              "sparkpipe_glm52_pp13_rank_daemon.c").read_text(
                  encoding="utf-8")
    start = source.index("static uint32_t SparkGlm52Pp13DaemonPumpQueuedWork(")
    end = source.index("static void SparkGlm52Pp13DaemonHandleWork(", start)
    function_body = source[start:end]
    predecessor = "SparkGlm52Pp13DaemonHasQueuedDependency(runtime,packet)"
    forward = "SparkGlm52Pp13DaemonForwardWork(runtime,packet)"
    submit = "SparkGlm52Pp13DaemonSubmitWork(runtime,packet)"
    assert predecessor in source
    assert function_body.index(predecessor) < function_body.index(forward)
    assert function_body.index(predecessor) < function_body.index(submit)
    forward_wait = function_body.index(
        "SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING_FORWARD")
    submit_wait = function_body.index(
        "SPARK_GLM52_PP13_DAEMON_WORK_STATE_WAITING_SUBMIT")
    assert forward_wait < submit_wait
    assert "return 0u;" in function_body[forward_wait:submit_wait]


def test_short_context_bypasses_indexshare_for_exact_prefix_attention(
        root: Path) -> None:
    source = (root / "modules" / "glm52_resident_decode_stage" / "source" /
              "spark_glm52_sm121_required_decode_stage.cu").read_text(
                  encoding="utf-8")
    start = source.index(
        "static SparkStatus SparkGlm52ResidentDecodeStageLaunchSparseIndexSelection(")
    end = source.index(
        "static uint32_t SparkGlm52ResidentDecodeStageFp8AmaxProbeEnabled(",
        start)
    function_body = source[start:end]
    prefix = "SparkGlm52ResidentDecodeStageLaunchContextPrefixSparseIndices("
    shared = "SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED"
    assert "pipeline_slot->dsa_candidate_count <=" in function_body
    assert function_body.index(prefix) < function_body.index(shared)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    test_final_stage_has_hidden_only_builtin_launcher(root)
    test_exact_pp13_final_stage_can_run_hidden_only(root)
    test_pp13_rank_capacity_is_not_fixed_batch(root)
    test_pp13_rank_enables_read_only_dsa_fragment_prefetch(root)
    test_rdma_transport_is_multilane_and_event_driven(root)
    test_prebound_linear_plan_accepts_smaller_active_count(root)
    test_fp8_linear_plans_require_scaled_gemm_backend(root)
    test_pp13_builder_binds_all_fp8_linear_plans(root)
    test_bulk_prefill_progresses_runner_after_each_chunk(root)
    test_bulk_prefill_validates_the_runtime_view_stride(root)
    test_prefill_probe_hashes_the_exact_stage_input(root)
    test_fp8_phase_probe_targets_the_first_divergent_layer(root)
    test_fp8_validator_preserves_quantized_dense_execution(root)
    test_speculative_verify_exposes_the_full_verifier_vector(root)
    test_target_and_mtp_heads_use_distinct_fail_closed_tensor_core_gemms(root)
    test_pp13_builder_uses_compressed_absorbed_mla(root)
    test_pp13_bulk_prefill_has_one_embedding_kernel(root)
    test_mtp_previous_target_position_contracts_are_explicit(root)
    test_mtp_linear_plans_use_logical_rows(root)
    test_mtp_draft_plan_builds_asymmetric_top2_tree(root)
    test_mtp_runtime_depth_specializes_and_caches_cuda_graphs(root)
    test_mtp_runtime_failures_name_the_failing_phase(root)
    test_mtp_gpu_profile_is_graph_compatible_and_explicitly_enabled(root)
    test_mtp_tree_copies_branch_history_before_attention(root)
    test_mtp_tree_rebases_private_draft_cache_before_followup(root)
    test_mtp_transaction_uses_expanded_execution_row(root)
    test_layer_body_failures_are_never_silent(root)
    test_plain_wide_decode_bypasses_dspark_finalizer(root)
    test_resident_block_stride_is_independent_of_the_physical_pool(root)
    test_service_backend_namespaces_ids_per_live_session(root)
    test_final_event_pump_detects_disconnect_before_send(root)
    test_rank_queue_does_not_overtake_a_deferred_sequence_position(root)
    test_short_context_bypasses_indexshare_for_exact_prefix_attention(root)
    test_mtp_retry_cleanup_preserves_resolution_receipt(root)
    test_mtp_serial_train_continuation_keeps_transaction_open(root)
    test_attached_resident_decode_preserves_mtp_resolution(root)
    test_attached_prefill_uses_nonblocking_ordered_forwarding(root)


if __name__ == "__main__":
    main()
