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
		"host": "model-families/k3/include/sparkpipe/spark_k3_kv_geometry.h",
		"module": None,
		"pairs": [],
	},
	"mimo25": {
		"config": "inference/llms/mimo_2_5/config.h",
		"host": "model-families/mimo25/include/sparkpipe/spark_mimo25_model.h",
		"module": None,
		"pairs": [],
	},
	"qwen36": {
		"config": "inference/llms/qwen_3_6/config.h",
		"host": "model-families/qwen36/include/sparkpipe/spark_qwen36_model.h",
		"module": None,
		"pairs": [
			("QWEN36_HIDDEN", "SPARK_QWEN36_MODEL_HIDDEN_DIMENSION"),
			("QWEN36_LAYERS", "SPARK_QWEN36_MODEL_LAYER_COUNT"),
			("QWEN36_VOCAB", "SPARK_QWEN36_MODEL_VOCAB_COUNT"),
			("QWEN36_HEAD_DIM", "SPARK_QWEN36_MODEL_HEAD_DIMENSION"),
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
