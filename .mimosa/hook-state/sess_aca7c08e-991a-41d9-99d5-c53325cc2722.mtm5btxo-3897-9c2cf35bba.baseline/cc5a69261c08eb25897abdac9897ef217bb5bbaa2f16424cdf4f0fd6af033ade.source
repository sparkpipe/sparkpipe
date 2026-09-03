#!/usr/bin/env python3
"""Every claimed model family has one geometry source and, when active, one module."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
	"glm52": {
		"config": "modules/glm52_resident_decode_stage/source/cuda/config.h",
		"host": "model-families/glm52/include/sparkpipe/spark_glm52_model.h",
		"module": "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c",
		"pairs": [],
	},
	"k3": {
		"config": "inference/llms/kimi_k3/config.h",
		"host": "model-families/k3/include/sparkpipe/spark_k3_model.h",
		"module": None,
		"pairs": [
			("K3_HIDDEN", "SPARK_K3_MODEL_HIDDEN_DIMENSION"),
			("K3_LAYERS", "SPARK_K3_MODEL_LAYER_COUNT"),
			("K3_VOCAB", "SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT"),
			("K3_MAX_CONTEXT", "SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS"),
			("K3_FIRST_ROUTED_LAYER", "SPARK_K3_MODEL_FIRST_ROUTED_LAYER"),
			("K3_EXPERTS", "SPARK_K3_MODEL_MOE_EXPERT_COUNT"),
			("K3_TOP_K", "SPARK_K3_MODEL_MOE_TOP_K"),
			("K3_SHARED_EXPERTS", "SPARK_K3_MODEL_MOE_SHARED_EXPERT_COUNT"),
			("K3_EXPERT_INTERMEDIATE", "SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION"),
			("K3_ROUTED_EXPERT_HIDDEN", "SPARK_K3_MODEL_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION"),
			("K3_DENSE_INTERMEDIATE", "SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION"),
			("K3_MLA_HEADS", "SPARK_K3_MODEL_MLA_HEAD_COUNT"),
			("K3_KV_LORA_RANK", "SPARK_K3_MODEL_MLA_LATENT_DIMENSION"),
			("K3_Q_LORA_RANK", "SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION"),
			("K3_QK_NOPE_DIM", "SPARK_K3_MODEL_MLA_QK_NOPE_HEAD_DIMENSION"),
			("K3_QK_UNROTATED_DIM", "SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION"),
			("K3_V_HEAD_DIM", "SPARK_K3_MODEL_MLA_VALUE_HEAD_DIMENSION"),
			("K3_MLA_USE_NOPE", "SPARK_K3_MODEL_MLA_USE_NOPE"),
			("K3_MLA_OUTPUT_GATE", "SPARK_K3_MODEL_MLA_OUTPUT_GATE"),
			("K3_KDA_HEADS", "SPARK_K3_MODEL_KDA_HEAD_COUNT"),
			("K3_KDA_KEY_DIM", "SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION"),
			("K3_KDA_VALUE_DIM", "SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION"),
			("K3_KDA_CONV_KERNEL", "SPARK_K3_MODEL_KDA_CONV_KERNEL"),
			("K3_KDA_FULL_RANK_GATE", "SPARK_K3_MODEL_KDA_FULL_RANK_GATE"),
			("K3_KDA_A_LOG_SOURCE_HEADS", "SPARK_K3_MODEL_KDA_A_LOG_SOURCE_HEAD_COUNT"),
			("K3_KDA_QK_L2NORM", "SPARK_K3_MODEL_KDA_QK_L2NORM"),
			("K3_ATTNRES_BLOCK_SIZE", "SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS"),
			("K3_ATTNRES_MAX_SOURCES", "SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS"),
			("K3_MXFP4_GROUP", "SPARK_K3_MODEL_MXFP4_GROUP_SIZE"),
		],
	},
	"mimo25": {
		"config": "inference/llms/mimo_2_5/config.h",
		"host": "model-families/mimo25/include/sparkpipe/spark_mimo25_model.h",
		"module": None,
		"pairs": [],
	},
	"qwen38_27b": {
		"config": "inference/llms/qwen_3_6/config.h",
		"host": "model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_model.h",
		"module": None,
		"pairs": [
			("QWEN38_27B_HIDDEN", "SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION"),
			("QWEN38_27B_LAYERS", "SPARK_QWEN38_27B_MODEL_LAYER_COUNT"),
			("QWEN38_27B_VOCAB", "SPARK_QWEN38_27B_MODEL_VOCAB_COUNT"),
			("QWEN38_27B_HEAD_DIM", "SPARK_QWEN38_27B_MODEL_HEAD_DIMENSION"),
			("QWEN38_27B_ATTN_HEADS", "SPARK_QWEN38_27B_MODEL_ATTENTION_HEAD_COUNT"),
			("QWEN38_27B_KV_HEADS", "SPARK_QWEN38_27B_MODEL_KV_HEAD_COUNT"),
			("QWEN38_27B_ROPE_DIM", "SPARK_QWEN38_27B_MODEL_ATTN_ROPE_DIMENSION"),
			("QWEN38_27B_FFN_INTERMEDIATE", "SPARK_QWEN38_27B_MODEL_FFN_INTERMEDIATE_DIMENSION"),
			("QWEN38_27B_MTP_LAYERS", "SPARK_QWEN38_27B_MODEL_MTP_LAYER_COUNT"),
			("QWEN38_27B_ATTENTION_PERIOD", "SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD"),
			("QWEN38_27B_FULL_PHASE", "SPARK_QWEN38_27B_MODEL_FULL_ATTENTION_PHASE"),
			("QWEN38_27B_GDN_KEY_HEADS", "SPARK_QWEN38_27B_MODEL_GDN_KEY_HEAD_COUNT"),
			("QWEN38_27B_GDN_VALUE_HEADS", "SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT"),
			("QWEN38_27B_GDN_KEY_DIM", "SPARK_QWEN38_27B_MODEL_GDN_HEAD_KEY_DIMENSION"),
			("QWEN38_27B_GDN_VALUE_DIM", "SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION"),
			("QWEN38_27B_GDN_CONV_KERNEL", "SPARK_QWEN38_27B_MODEL_GDN_CONV_KERNEL"),
		],
	},
	"dsv4": {
		"config": "model_contracts/dsv4_flash_authoritative.json",
		"host": "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h",
		"module": "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c",
		"pairs": [],
	},
}


def defines(path: Path) -> dict[str, str]:
	return dict(re.findall(r"#define (\w+) (\d+)u", path.read_text(errors="surrogateescape")))


def main() -> int:
	legacy_paths = (
		ROOT / "inference/llms/glm5_2",
		ROOT / "inference/llms/deepseek_v4",
		ROOT / "inference/llms/deepseek_v4_pro",
	)
	for legacy_path in legacy_paths:
		if legacy_path.exists():
			raise AssertionError(f"legacy model implementation remains: {legacy_path}")
	problems = 0
	for family,spec in FAMILIES.items():
		for label in ("config", "host"):
			path = ROOT / spec[label]
			if not path.is_file():
				print(f"FAIL {family}: missing {label} ({spec[label]})")
				problems += 1
		module = spec["module"]
		if module is not None and not (ROOT / module).is_file():
			print(f"FAIL {family}: missing active model module ({module})")
			problems += 1
		config = ROOT / spec["config"]
		host = ROOT / spec["host"]
		if config.suffix == ".h" and config.is_file() and host.is_file():
			firmware = defines(config)
			host_defines = defines(host)
			for firmware_key,host_key in spec["pairs"]:
				if firmware.get(firmware_key) != host_defines.get(host_key):
					print(f"FAIL {family}: {firmware_key}={firmware.get(firmware_key)} vs {host_key}={host_defines.get(host_key)}")
					problems += 1
	print(f"{len(FAMILIES)} families checked, {problems} problems")
	return(1 if problems != 0 else 0)


if __name__ == "__main__":
	sys.exit(main())
