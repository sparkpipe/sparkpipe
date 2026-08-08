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
	module = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c")
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
	require(module, "metadata_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT", "frame-sized metadata ownership")
	require(module, "uint64_t rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT", "frame-sized bulk scratch ownership")
	require(module, "cudaMemcpy2DAsync(slot->input_token_ids", "single frame metadata staging")
	require(module, "SparkDsv4ModuleStageFrameRows(state,slot,frame,context)", "metadata staged before bulk dispatch")
	require(module, "SparkDsv4ModulePrefillWaveNeedsHead", "prefill final-head wave pruning")
	require(module, "SparkDsv4ModuleRunPrefillHeads", "prefill final-head wave slicing")
	require(module, "SparkDsv4ModuleRunCausalAttention", "causal bulk-prefill attention")
	require(module, "slot->row_lane_indices + first_row", "causal attention row slicing")
	require(module, "state->topk_column_count,sink_f32", "per-wave attention sink")
	require(module, "SparkDsv4ModuleRunLayers(state,slot,input_streams_bf16,output_streams_bf16,prefill,rows)", "single bulk layer traversal")
	reject(module, "SparkDsv4ModuleRunPrefillWave", "full-layer prefill wave replay")
	require(cuda, "SparkDsv4RopeKernel", "checkpoint interleaved RoPE")
	require(cuda, "LmE4m3ToFloat(LmFloatToE4m3(value))", "shared exact E4M3 quantize-dequantize")
	reject(cuda, "SparkDsv4EncodeE4m3", "duplicate DSV4 E4M3 codec")
	require(cuda, "SparkDsv4StridedLinearKernel<SPARK_DSV4_MODEL_OUTPUT_COMPOSITION_ACTIVATION_CODEC>", "model-declared output-composition activation")
	require(cuda, "SparkDsv4LaunchIndexerScore", "lightning indexer")
	require(cuda, "SparkDsv4BuildAttentionIndicesKernel", "device attention-index assembly")
	require(cuda, "SparkDsv4AttentionWindowSlot(position,column", "chronological ring attention indices")
	require(cuda, "SparkDsv4CacheScatterKernel", "device KV cache scatter")
	require(cuda, "pooled = 0.0f;", "initialized non-emitting compressor row")
	require(cuda, "scores[element] = -INFINITY;", "negative-infinity compressor score reset")
	require(cuda, "if ( row_lane_indices[previous] == lane )", "single compressor CTA per lane")
	require(cuda, "for (; row<row_count; row++)", "ordered same-lane compressor rows")
	require(module, "channels = weights->overlap * cache_width;", "HCA single-width compressor channels")
	require(module, "overlapped = weights->overlap > 1u ? 1u : 0u;", "CSA-only compressor overlap")
	require(module, "SparkDsv4LaunchResetCompressorState(stream,state->compress_kv_state_f32 + offset,state->compress_score_state_f32 + offset,state->compress_state_lane_stride)", "compressor state reset")
	require(module, "SparkDsv4LaunchResetCompressorState(stream,state->index_kv_state_f32 + offset,state->index_score_state_f32 + offset,state->index_state_lane_stride)", "indexer state reset")
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
	if moe.count("SparkDsv4LaunchExpertUp(") != 2 or moe.count("SparkDsv4LaunchExpertDown(") != 1:
		raise SystemExit("DSV4 routed MoE must issue exactly W1, W3, and W2 grouped GEMMs")
	require(cuda, "LmWeightCodec<SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC>::Format", "compile-time package codec")
	require(cuda, "LmGemmWeightOnlyIndirectLaunch<SparkDsv4ExpertWeightFormat", "indirect grouped expert up GEMM")
	require(cuda, "SPARK_DSV4_EXPERT_WARPS,SPARK_DSV4_MODEL_ACTIVATION_CODEC>", "grouped GEMM activation codec")
	require(cuda, "LmGemmWeightOnlyLaunch<SparkDsv4ExpertWeightFormat", "grouped expert down GEMM")
	reject(cuda, "SparkLmExpertTileAllKernel", "legacy runtime-format expert kernel")
	reject(moe, "cudaStreamSynchronize", "routed MoE synchronization")
	reject(moe, "for (expert", "per-expert host dispatch")
	reject(module, "SparkDsv4ModuleHostTopkFill", "host-built attention indices")
	reject(module, "host_topk_indices", "resident host attention-index matrix")
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
	require(adapter, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT", "PP13 adapter slot capacity")
	require(validator, "#define SPARK_DSV4_VALIDATION_ROW_COUNT 8u", "B8 CUDA validation width")
	require(validator, "frame->batch.row_count = SPARK_DSV4_VALIDATION_ROW_COUNT;", "B8 CUDA decode batch")
	require(validator, "frame->lanes[row] = row;", "B8 CUDA distinct resident lanes")
	require(validator, "frame->frame.completion_function = SparkDsv4ValidationCompletion", "validator external completion")
	require(driver_smoke, "node_context.linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC", "driver smoke linear codec binding")
	require(driver_smoke, "node_context.expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC", "driver smoke expert codec binding")
	require(driver_smoke, "node_context.kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC", "driver smoke KV codec binding")
	reject(validator, "node_context->stage_count = 2u", "hardcoded validator topology")
	reject(validator + validator_script + module, "ALLOW_UNQUALIFIED", "runtime qualification bypass")
	require(validator_script, "-lcuda", "CUDA Driver API validator link")
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
