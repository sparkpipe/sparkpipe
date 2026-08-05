#!/usr/bin/env python3
"""Pin DSV4 Flash to its one authoritative contract and active module."""

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
	if contract["source_revision"] != "60d8d70770c6776ff598c94bb586a859a38244f1":
		raise SystemExit("DSV4 Flash source revision is not exact")
	if contract["precision"]["routed_expert_weight_codec"] != "mxfp4_e2m1":
		raise SystemExit("DSV4 Flash package does not declare its exact expert codec")
	header = read("model-families/dsv4/include/sparkpipe/spark_dsv4_model.h")
	module = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c")
	cuda = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu")
	stagepack = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h")
	adapter = read("modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c")
	require(header, "Generated from the exact source revision", "generated model contract")
	require(header, "SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC SPARK_WEIGHT_CODEC_MXFP4_E2M1", "package expert codec")
	require(header, "SPARK_DSV4_MODEL_LAYER_KIND_INVALID UINT32_MAX", "invalid layer sentinel")
	reject(header, "return(SPARK_DSV4_MODEL_LAYER_KIND_SWA);\n}", "out-of-range SWA fallback")
	require(module, "SparkDsv4ModelLayerKind(layer_index)", "model-owned layer dispatch")
	require(module, "kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA", "CSA indexer dispatch")
	require(module, "kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA", "compressed attention dispatch")
	require(module, "SparkDsv4LaunchSparseAttn", "sparse attention execution")
	require(module, "SparkDsv4LaunchHcSplitSinkhorn", "inference mHC Sinkhorn")
	require(cuda, "SparkDsv4RopeKernel", "checkpoint interleaved RoPE")
	require(cuda, "SparkDsv4LaunchIndexerScore", "lightning indexer")
	require(cuda, "SparkDsv4BuildAttentionIndicesKernel", "device attention-index assembly")
	require(cuda, "SparkDsv4CacheScatterKernel", "device KV cache scatter")
	require(stagepack, "SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1", "checkpoint MXFP4 expert payload")
	require(adapter, ".expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC", "adapter codec binding")
	moe = function_body(module, "SparkDsv4ModuleRunMoeRouted(")
	if moe.count("SparkDsv4LaunchExpertUp(") != 2 or moe.count("SparkDsv4LaunchExpertDown(") != 1:
		raise SystemExit("DSV4 routed MoE must issue exactly W1, W3, and W2 grouped GEMMs")
	require(cuda, "LmWeightCodec<SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC>::Format", "compile-time package codec")
	require(cuda, "LmGemmWeightOnlyIndirectLaunch<SparkDsv4ExpertWeightFormat", "indirect grouped expert up GEMM")
	require(cuda, "LmGemmWeightOnlyLaunch<SparkDsv4ExpertWeightFormat", "grouped expert down GEMM")
	reject(cuda, "SparkLmExpertTileAllKernel", "legacy runtime-format expert kernel")
	reject(moe, "cudaStreamSynchronize", "routed MoE synchronization")
	reject(moe, "for (expert", "per-expert host dispatch")
	reject(module, "SparkDsv4ModuleHostTopkFill", "host-built attention indices")
	reject(module, "host_topk_indices", "resident host attention-index matrix")
	print("PASS DSV4 active-module source contracts")


if __name__ == "__main__":
	main()
