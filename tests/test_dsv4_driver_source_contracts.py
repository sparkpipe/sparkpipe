#!/usr/bin/env python3
"""Pin DSV4 Flash GA to its authoritative native-driver contract."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
	return (ROOT / relative).read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
	if needle not in text:
		raise SystemExit(f"missing {label}: {needle}")


def reject(text: str, needle: str, label: str) -> None:
	if needle in text:
		raise SystemExit(f"forbidden {label}: {needle}")


def function_body(text: str, name: str) -> str:
	start = text.index(name)
	brace = text.index("{", start)
	depth = 1
	index = brace + 1
	while depth != 0:
		if text[index] == "{":
			depth += 1
		elif text[index] == "}":
			depth -= 1
		index += 1
	return text[brace:index]


def main() -> None:
	legacy = (
		"inference/llms/deepseek_v4",
		"inference/llms/deepseek_v4_pro",
		"model-families/dsv4/include/sparkpipe/spark_dsv4_flash_model.h",
	)
	for relative in legacy:
		if (ROOT / relative).exists():
			raise SystemExit(f"obsolete DSV4 implementation remains: {relative}")
	contract = json.loads(read("model_contracts/dsv4_flash_authoritative.json"))
	if contract["model_id"] != "deepseek-ai/DeepSeek-V4-Flash-0731":
		raise SystemExit("DSV4 Flash source model is not GA")
	if contract["source_revision"] != "7872f01b1d1fe23eabc4c98b48bffcef5a386062":
		raise SystemExit("DSV4 Flash source revision is not exact")
	if contract["model"]["mtp_layer_count"] != 0:
		raise SystemExit("DSV4 Flash GA baseline MTP layer count is not zero")
	if contract["dspark"]["layer_count"] != 3:
		raise SystemExit("DSV4 Flash GA checkpoint DSpark count is not exact")
	if contract["runtime"]["packed_mtp_layer_count"] != 0:
		raise SystemExit("DSV4 Flash GA baseline must not pack DSpark tensors")
	if contract["precision"]["routed_expert_weight_codec"] != "mxfp4_e2m1":
		raise SystemExit("DSV4 Flash package does not declare its exact expert codec")
	if (contract["precision"]["dynamic_activation_format"], contract["precision"]["scale_format"]) != ("fp8_e4m3", "ue8m0"):
		raise SystemExit("DSV4 Flash package does not declare its trained activation codec")
	header = read("model-families/dsv4/include/sparkpipe/spark_dsv4_model.h")
	dtype = read("inference/kernels/dtype.cuh")
	activation = read("inference/kernels/activation.cuh")
	common = read("model-families/common/include/sparkpipe/spark_lm_kernels.cuh")
	row_compaction = read("model-families/common/include/sparkpipe/spark_row_compaction.cuh")
	row_layout = read("include/sparkpipe/spark_row_layout.h")
	module = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c")
	pool_layout = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_pool_layout.h")
	paged_cache = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_paged_cache.c")
	generic_cache = "\n".join(read(relative) for relative in (
		"cache/kv_cache.c",
		"cache/kv_page_cache.c",
		"cache/kv_page_store.c",
		"cache/prefix_cache.c",
		"runtime/model_batch_engine.c",
		"runtime/model_serving_adapter.c",
	))
	firmware = read("modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h")
	cuda = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu")
	stagepack = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h")
	adapter = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c")
	runner = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_stage_runner.c")
	validator = read("modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu")
	validator_script = read("modules/dsv4_resident_decode_stage/validation/validate_dsv4_resident_decode_stage_cuda.sh")
	module_makefile = read("modules/dsv4_resident_decode_stage/Makefile")
	fixture_verifier = read("tools/verify_dsv4_ga_reference_fixture.py")
	fixture_generator = read("tools/generate_dsv4_ga_reference_fixture.sh")
	driver_smoke = read("tools/sparkpipe_dsv4_driver_cuda_smoke.c")
	require(header, "Generated from the exact source revision", "generated model contract")
	require(header, "SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_MXFP4_E2M1", "package expert codec")
	require(header, "SPARK_DSV4_MODEL_ACTIVATION_CODEC SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0", "package activation codec")
	require(header, "SPARK_DSV4_MODEL_OUTPUT_COMPOSITION_ACTIVATION_CODEC SPARK_ACTIVATION_CODEC_NONE", "output-composition activation codec")
	require(dtype, "return((uint8_t)encoded);", "scalar E4M3 low-byte result")
	require(activation, "LmActivationStageFp8Qdq", "shared in-tile activation codec")
	require(common, "SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC>", "dense activation codec propagation")
	require(common, "entry / SPARK_LM_TILE_N][entry % SPARK_LM_TILE_N]", "software-pipelined output tile row stride")
	reject(common, "entry / SPARK_LM_TILE][entry % SPARK_LM_TILE_N]", "software-pipelined output tile row alias")
	require(header, "SPARK_DSV4_MODEL_LAYER_KIND_INVALID UINT32_MAX", "invalid layer sentinel")
	reject(header, "return(SPARK_DSV4_MODEL_LAYER_KIND_SWA);\n}", "out-of-range SWA fallback")
	require(module, "SparkDsv4ModelLayerKind(layer_index)", "model-owned layer dispatch")
	require(module, "kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA", "CSA indexer dispatch")
	require(module, "kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA", "compressed attention dispatch")
	require(module, "SparkDsv4LaunchSparseAttn", "sparse attention execution")
	require(module, "SparkDsv4LaunchHcSplitSinkhorn", "inference mHC Sinkhorn")
	require(module, "SparkDsv4LaunchHcMixSplitKSinkhorn", "split-K inference mHC")
	require(module, "metadata_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT", "frame-sized metadata ownership")
	require(module, "uint64_t rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT", "frame-sized bulk scratch ownership")
	require(module, "head_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT", "active-width final-head scratch")
	reject(module, "head_rows = state->resident_sequence_capacity", "resident-width per-submit scratch")
	require(module, "lane_states[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT]", "resident-width lane state")
	require(module, "cudaMemcpy2DAsync(slot->input_token_ids", "single frame metadata staging")
	require(module, "SparkDsv4ModuleStageFrameRows(state,slot,frame,context)", "metadata staged before bulk dispatch")
	reject(module, "SparkDsv4ModulePrefillWaveNeedsHead", "quadratic prefill terminal-row scan")
	require(module, "SparkDsv4ModuleRunPrefillHead", "single compact prefill head")
	prefill_head = function_body(module, "SparkDsv4ModuleRunPrefillHead(")
	require(prefill_head, "SparkDsv4LaunchGatherHeadRows", "compact head-row gather")
	require(prefill_head, "SparkDsv4ModuleProjectHead(state,slot,slot->residual_bf16,slot->slot_counts,prefill->emit_count)", "emit-count head projection")
	require(prefill_head, "SparkDsv4LaunchScatterHeadTokens", "lane-slot token scatter")
	reject(prefill_head, "for (", "repeated per-wave head launch")
	require(row_compaction, "SparkGatherBf16RowsKernel", "shared bf16 row gather")
	require(row_compaction, "SparkScatterU32RowsKernel", "shared u32 row scatter")
	require(row_layout, "SparkRowLayoutValidateRoundMajor", "shared linear row-layout validator")
	require(row_layout, "SparkRowLayoutRoundMajorWaveRowCount", "shared linear wave sizing")
	require(module, "prefill->emit_row_indices[index] != last_rows[lane]", "terminal-only prefill emit validation")
	reject(module, "SparkDsv4ValidateRoundMajorPrefillRows", "model-specific row-layout validator")
	require(module, "SparkDsv4ModuleRunCausalAttention", "causal bulk-prefill attention")
	require(module, "slot->row_lane_indices + first_row", "causal attention row slicing")
	require(module, "state->topk_column_count,sink_f32", "per-wave attention sink")
	require(module, "SparkDsv4ModuleStartLayers(continuation)", "asynchronous TP layer entrypoint")
	require(module, "SparkDsv4ModuleContinueLayers", "collective-owned TP continuation")
	reject(module, "SparkDsv4ModuleRunPrefillWave", "full-layer prefill wave replay")
	require(cuda, "SparkDsv4RopeKernel", "checkpoint interleaved RoPE")
	require(cuda, "LmE4m3ToFloat(LmFloatToE4m3(value))", "shared exact E4M3 quantize-dequantize")
	reject(cuda, "SparkDsv4EncodeE4m3", "duplicate DSV4 E4M3 codec")
	require(cuda, "SparkLmHostLaunchSm121StridedDecodeLinear<", "shape-aware output-composition linear")
	require(common, "SparkLmHostLaunchSm121NativeLinear<", "B8/B1024 native output-composition linear")
	require(common, "(uint64_t)row * output_row_stride + output_offset +", "B>1 output-composition row stride")
	attention = function_body(module, "SparkDsv4ModuleRunAttention(")
	require(attention, "SparkDsv4LaunchLinear(stream,&layer->attn.wo_b",
		"column-parallel WO full-hidden partial")
	reject(attention, "state->tp_rank * local_hidden",
		"diagonal WO rank-row write")
	require(module,
		"SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16",
		"full-hidden TP sum collective")
	require(module, "submission.local_device = device_bf16;",
		"in-place full-hidden TP reduction")
	require(cuda, "SparkDsv4LaunchIndexerScore", "lightning indexer")
	require(cuda, "SparkDsv4BuildAttentionIndicesKernel", "device attention-index assembly")
	require(cuda, "SparkDsv4AttentionWindowSlot(position,column", "chronological ring attention indices")
	require(cuda, "SparkDsv4CacheScatterKernel", "device KV cache scatter")
	require(cuda, "pooled = 0.0f;", "initialized non-emitting compressor row")
	require(cuda, "if ( row_lane_indices[previous] == lane )", "single compressor CTA per lane")
	require(cuda, "for (; row<row_count; row++)", "ordered same-lane compressor rows")
	require(module, "channels = weights->overlap * cache_width;", "HCA single-width compressor channels")
	require(module, "overlapped = weights->overlap > 1u ? 1u : 0u;", "CSA-only compressor overlap")
	require(paged_cache, "SparkDsv4PagedPoolBuildLayout(configuration->first_layer_index", "DSV4 page geometry")
	require(paged_cache, "SparkKvPageCacheBeginLaneTransaction", "generic transactional page lifecycle binding")
	require(paged_cache, "SparkKvPageCacheRollbackLaneTransaction", "generic page rollback binding")
	require(module, "SPARK_DSV4_PREPARED_CACHE_COMMITTED", "prepared-cache committed state")
	require(module, "SPARK_DSV4_PREPARED_CACHE_ADOPTING", "prepared-cache in-flight ownership state")
	require(module, "SparkKvCacheArenaReserveUnassignedResidentBlocks", "atomic PREPARE capacity reservation")
	require(module, "SparkKvCacheArenaConsumeUnassignedResidentBlocks", "COMMIT ownership adoption before mutable allocation")
	require(module, "prepared->state = SPARK_DSV4_PREPARED_CACHE_ADOPTING", "exact committed-to-adopting transition")
	require(module, "SparkDsv4ModuleCacheAdmissionRequestMatches(prepared", "full prepared-cache identity validation")
	require(module, "async->prepared_cache_admission_index", "terminal callback retains prepared-cache ownership")
	terminal_cache_release = function_body(module,
		"static SparkStatus SparkDsv4ModuleReleaseCommittedCacheAdmission(")
	require(terminal_cache_release,
		"prepared->state == SPARK_DSV4_PREPARED_CACHE_COMMITTED",
		"pre-adoption cleanup committed-state guard")
	require(terminal_cache_release,
		"SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,&request)",
		"pre-adoption cleanup exact request match")
	require(terminal_cache_release,
		"SparkDsv4ModuleClearCacheAdmission(state,prepared,1u)",
		"pre-adoption cleanup capacity release")
	execute_entry = function_body(module,
		"SparkStatus SparkDsv4ResidentDecodeStageExecute(")
	require(execute_entry, "status != SPARK_STATUS_BUSY",
		"BUSY retains committed cache admission for FIFO retry")
	require(execute_entry, "status != SPARK_STATUS_PENDING",
		"PENDING retains committed cache admission for FIFO retry")
	require(execute_entry, "SparkDsv4ModuleReleaseCommittedCacheAdmission(",
		"terminal pre-adoption cache cleanup")
	require(module, "SparkKvPageStoreInitialize", "generic page-store binding")
	require(generic_cache, "SparkKvPageStoreWorkerMain", "generic asynchronous page worker")
	require(generic_cache, "SPARK_KV_PAGE_STORE_JOB_QUEUED", "generic page transfer queue")
	require(generic_cache, "result = SPARK_STATUS_BUSY", "generic multi-page prefetch progress")
	require(module, "cudaStreamQuery((cudaStream_t)state->kv_page_store_stream)", "DSV4 transfer completion query")
	reject(module, "cudaStreamSynchronize((cudaStream_t)state->kv_page_store_stream)", "blocking DSV4 page transfer")
	require(module, "state->logical_page_capacity = host_services->kv_logical_page_capacity", "generic logical-page budget binding")
	require(module, "state->physical_page_capacity = host_services->kv_physical_page_capacity", "generic physical-page budget binding")
	reject(module + paged_cache, "SparkPrefixCache", "model-specific prefix scheduling")
	reject(adapter, "physical_page_capacity = state->max_active_sequence_count", "model-specific physical-page sizing policy")
	reject(adapter, "logical_page_capacity = (uint64_t)state->resident_sequence_capacity", "model-specific logical-page sizing policy")
	reject(adapter, "node_context.logical_page_capacity", "cache policy in DSV4 node context")
	reject(adapter, "node_context.physical_page_capacity", "cache policy in DSV4 node context")
	require(adapter, "request.kv_logical_page_capacity =", "generic logical-page budget forwarding")
	require(adapter, "request.kv_physical_page_capacity =", "generic physical-page budget forwarding")
	require(generic_cache, "SparkModelBatchSchedulerPlanCacheBoundLaneCount", "generic cache-aware batch planner")
	require(generic_cache, "runtime_limits->kv_physical_page_capacity", "generic runtime physical-page validation")
	for model_name in ("dsv4", "deepseek", "glm52", "spark_glm", "qwen36", "spark_qwen"):
		reject(generic_cache.lower(), model_name, "model name in generic cache runtime")
	require(pool_layout, "SparkDsv4PoolCacheLaneElements(max_sequence_positions,kind)", "per-layer compressed cache sizing")
	require(pool_layout, "overlap * ratio * overlap * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION", "ratio-specific compressor state sizing")
	require(module, "state->cache_lane_stride_by_layer[layer_index]", "per-layer cache addressing")
	require(module, "SparkDsv4LaunchInitializePages", "generation-safe page initialization")
	require(cuda, "parent == UINT32_MAX ? 0u : page_pool[source + word]", "zero-or-parent page initialization")
	require(cuda, "score_base[score_spans[span_index].offset_words + word] = -INFINITY", "negative-infinity page score initialization")
	reject(cuda, "SparkDsv4ResetCompressorStateKernel", "obsolete dense-lane reset kernel")
	reject(module, "cudaMemsetAsync(state->compress_score_state_f32 + offset,0", "zero compressor score reset")
	reject(module, "cudaMemsetAsync(state->index_score_state_f32 + offset,0", "zero indexer score reset")
	require(stagepack, "SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1", "checkpoint MXFP4 expert payload")
	require(header, "SPARK_DSV4_MODEL_MTP_LAYER_COUNT 0u", "zero runtime MTP layers")
	require(header, "SPARK_DSV4_MODEL_CHECKPOINT_DSPARK_LAYER_COUNT 3u", "checkpoint DSpark layers")
	require(stagepack, "SPARK_DSV4_STAGEPACK_FORMAT_VERSION 3u", "GA stage-pack format")
	reject(module, "spark_glm", "GLM driver coupling")
	reject(adapter, "spark_glm", "GLM adapter coupling")
	require(adapter, "SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION", "asynchronous adapter completion capability")
	require(adapter, ".expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC", "adapter codec binding")
	require(adapter, "dispatch.token_ids = submission->token_ids;", "all-stage token routing input")
	require(runner, "buffers[buffer_count].address = (void *)dispatch->token_ids;", "all-stage token driver buffer")
	require(module, "slot->host_input_token_ids[row] = token_ids[row];", "all-stage token staging")
	require(module, "token_ids = (const uint32_t *)frame->buffers[0].address;", "decode token routing input")
	require(module, "frame->buffers[1u].address", "final token output publication")
	reject(module, "state->owns_embedding != 0u ? 1u : 0u", "embedding-dependent output slot")
	require(driver_smoke, "dispatch.output_token_ids =", "final-head driver smoke output")
	reject(adapter + runner + module, "stage_index == 0u ? submission->token_ids", "stage-zero-only token routing")
	moe = function_body(module, "SparkDsv4ModuleRunMoeRouted(")
	if moe.count("SparkDsv4LaunchFusedExpertW13Act(") != 1 or moe.count("SparkDsv4LaunchExpertDown(") != 1:
		raise SystemExit("DSV4 routed MoE must issue exactly one fused W13 activation and one native W2")
	require(moe, "SparkDsv4LaunchMoePairReduce(stream", "column-parallel routed full-hidden partial")
	reject(moe, "SparkDsv4LaunchMoePairReduceStrided", "diagonal routed rank-row reduction")
	moe_full = function_body(module, "SparkDsv4ModuleRunMoe(")
	require(moe_full, "SparkDsv4LaunchLinear(stream,&moe->shared_w2", "column-parallel shared-W2 full-hidden partial")
	reject(moe_full, "state->tp_rank * hidden_dimension", "diagonal shared-W2 rank-row write")
	reject(moe, "SparkDsv4LaunchExpertUp(", "split routed W1/W3 launch")
	reject(moe, "SparkDsv4LaunchSwigluClamp(", "separate routed SwiGLU launch")
	require(cuda, "LmWeightCodec<SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC>::Format", "compile-time package codec")
	require(cuda, "SparkLmHostLaunchSm121FusedExpertW13", "native fused routed W13")
	require(cuda, "SparkLmHostLaunchSm121ExpertW2", "native packed routed W2")
	require(common, "LmMmaMxf8Mxf4(gate_total", "native MXFP8-by-MXFP4 routed W1")
	require(common, "LmMmaMxf8Mxf4(up_total", "native MXFP8-by-MXFP4 routed W3")
	require(common, "LmMmaMxf8Mxf4(total", "native MXFP8-by-MXFP4 routed W2")
	reject(function_body(cuda, "SparkDsv4LaunchExpertDown("), "LmGemmWeightOnlyLaunch", "BF16-dequant routed W2 success path")
	reject(cuda, "SparkLmExpertTileAllKernel", "legacy runtime-format expert kernel")
	reject(moe, "cudaStreamSynchronize", "routed MoE synchronization")
	reject(moe, "for (expert", "per-expert host dispatch")
	reject(module, "SparkDsv4ModuleHostTopkFill", "host-built attention indices")
	reject(module, "host_topk_indices", "resident host attention-index matrix")
	require(module, "destination->request_generation = context->request_generation;", "synthesized cache lane request generation")
	require(module, "destination->step_generation = context->step_generation;", "synthesized cache lane step generation")
	for name in (
		"SPARK_DSV4_STAGE_COUNT",
		"SPARK_DSV4_STAGE_INDEX",
		"SPARK_DSV4_STAGE_FIRST_LAYER",
		"SPARK_DSV4_STAGE_LAYER_COUNT",
		"SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES",
		"SPARK_DSV4_STAGE_MAX_SEQ",
		"SPARK_DSV4_STAGE_PIPELINE_SLOTS",
	):
		require(validator, name, "stage-specific CUDA validation configuration")
	require(firmware, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 13u", "PP13 module slot capacity")
	require(firmware, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT 16384u", "resident lane capacity")
	require(firmware, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PHYSICAL_PAGE_COUNT 16384u", "B1024 pipeline page ceiling")
	reject(firmware, "uint32_t logical_page_capacity;", "generic logical-page policy in DSV4 node context")
	reject(firmware, "uint32_t physical_page_capacity;", "generic physical-page policy in DSV4 node context")
	require(adapter, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT", "PP13 adapter slot capacity")
	require(adapter, ".max_resident_sequence_count = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT", "adapter resident lane capacity")
	require(validator, "(SPARK_BATCH_BUCKET < 8u ? SPARK_BATCH_BUCKET : 8u)", "bucket-safe CUDA validation width")
	require(validator, "SPARK_BATCH_BUCKET < SPARK_DSV4_VALIDATION_REFERENCE_FIXTURE_ROW_COUNT", "bucket-safe retained-reference prefix")
	require(validator, "SparkDsv4ValidationReadPrefix", "verified full fixture with bucket prefix")
	require(validator, "frame->batch.row_count = SPARK_DSV4_VALIDATION_ROW_COUNT;", "CUDA decode batch")
	reject(validator, "SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT - 1u", "non-native DSV4 head validation width")
	require(validator, "SparkDsv4ValidationRequireCuda(error,\"head_cuda\")", "CUDA head failure detail")
	require(validator, "frame->lanes[row] = row;", "CUDA distinct resident lanes")
	require(validator, "frame->frame.completion_function = SparkDsv4ValidationCompletion", "validator external completion")
	require(driver_smoke, "node_context.linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC", "driver smoke linear codec binding")
	require(driver_smoke, "node_context.expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC", "driver smoke expert codec binding")
	require(driver_smoke, "node_context.kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC", "driver smoke KV codec binding")
	reject(validator, "node_context->stage_count = 2u", "hardcoded validator topology")
	reject(validator + validator_script + module, "ALLOW_UNQUALIFIED", "runtime qualification bypass")
	require(validator_script, "-lcuda", "CUDA Driver API validator link")
	require(validator_script, 'batch_bucket="${SPARK_MODULE_BATCH_BUCKET:-}"', "published batch variant identity")
	require(validator_script, '"-DSPARK_BATCH_BUCKET=${batch_bucket}"', "matching validator batch geometry")
	require(validator_script, "qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128", "retained GA stage-0 fixture")
	require(validator_script, '"${SPARK_DSV4_STAGE_INDEX:-}" == "0"', "reference stage index gate")
	require(validator_script, '"${SPARK_DSV4_STAGE_FIRST_LAYER:-}" == "0"', "reference first-layer gate")
	require(validator_script, '"${SPARK_DSV4_STAGE_LAYER_COUNT:-}" == "3"', "reference layer-count gate")
	require(validator_script, "unset SPARK_DSV4_REFERENCE_TOKEN_PATH", "caller-independent reference token selection")
	require(validator_script, "unset SPARK_DSV4_REFERENCE_OUTPUT_PATH", "caller-independent reference output selection")
	require(validator_script, 'python3 "${reference_verifier}"', "strict reference fixture verification")
	require(validator_script, 'export SPARK_DSV4_REFERENCE_TOKEN_PATH="${reference_fixture_directory}/prompt_tokens.u32le"', "verified reference token export")
	require(validator_script, 'export SPARK_DSV4_REFERENCE_OUTPUT_PATH="${reference_fixture_directory}/after_layer_2.bf16le"', "verified reference output export")
	require(validator_script, "SPARK_DSV4_REFERENCE_MANIFEST_SHA256", "pinned reference manifest input")
	require(validator, "mode->use_reference != exact_reference_slice", "C-side exact stage-0 reference enforcement")
	require(validator, "SPARK_DSV4_VALIDATION_REFERENCE_MAX_ROW_RELATIVE_L2", "true per-row reference guard")
	require(module_makefile, "override DSV4_GA_STAGE0_REFERENCE_MANIFEST_SHA256 :=", "retained reference digest binding")
	require(module_makefile, "SPARK_DSV4_REFERENCE_MANIFEST_SHA256=$(DSV4_GA_STAGE0_REFERENCE_MANIFEST_SHA256)", "reference digest runtime configuration")
	require(module_makefile, "SPARK_DSV4_CUDA_VALIDATOR_SHA256=$(DSV4_CUDA_VALIDATOR_SHA256)", "CUDA validator digest configuration")
	require(module_makefile, "SPARK_DSV4_REFERENCE_VERIFIER_SHA256=$(DSV4_REFERENCE_VERIFIER_SHA256)", "reference verifier digest configuration")
	require(validator_script, 'require_source_digest "${SPARK_DSV4_CUDA_VALIDATOR_SHA256:-}"', "CUDA validator source binding")
	require(validator_script, 'require_source_digest "${SPARK_DSV4_REFERENCE_VERIFIER_SHA256:-}"', "reference verifier source binding")
	digest_line = next(line for line in module_makefile.splitlines() if line.startswith("override DSV4_GA_STAGE0_REFERENCE_MANIFEST_SHA256 :="))
	pinned_digest = digest_line.split(":=", 1)[1].strip()
	fixture_manifest = ROOT / "qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128/manifest.json"
	if fixture_manifest.is_file():
		actual_digest = hashlib.sha256(fixture_manifest.read_bytes()).hexdigest()
		if pinned_digest != actual_digest:
			raise SystemExit("DSV4 retained reference manifest digest is not exact")
	elif pinned_digest != "":
		raise SystemExit("DSV4 reference digest is pinned without a retained fixture")
	require(fixture_verifier, 'REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"', "fixture GA revision")
	require(fixture_verifier, 'VECTOR_PATH = "after_layer_2.bf16le"', "fixture final stage-0 vector")
	require(fixture_verifier, "fixture directory entries are not exact", "closed fixture inventory")
	require(fixture_generator, "for run in first second", "fresh-process reference reproduction")
	require(fixture_generator, 'cmp "${scratch_directory}/first/${artifact}"', "byte-exact reference reproduction gate")
	print("PASS DSV4 active-module source contracts")


if __name__ == "__main__":
	main()
