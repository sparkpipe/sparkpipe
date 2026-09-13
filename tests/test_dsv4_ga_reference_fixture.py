#!/usr/bin/env python3
"""Exercise the retained DSV4 GA fixture verifier without checkpoint files."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
TOKEN_IDS = [0,3476,477,18068,260,3375,35312,3417,16,38074,13254,16,455,4087,3287,2231,1605,270,21361,8786,9045,16,128803,79418,2317,566,8130,345,14866,3312,2019,16,983,1142,469,1142,554,6242,260,31191,603,19905,418,270,4031,2455,2562,1167,1479,270,6074,15398,344,10097,16,2052,270,15398,344,1353,4521,538,260,2395,2740,294,18885,6243,14,20430,418,270,19904,50098,5898,1789,638,1341,294,6319,2562,3737,603,25529,223,18,855,270,2019,344,7681,1202,270,10844,22283,339,671,2019,109029,260,716,15,10554,30347,112566,1936,14327,436,304,270,489,5927,7104,339,9945,1137,9854,69,201,223,19,28,1823,11006,334,30557,32684,16617]

import verify_dsv4_ga_reference_fixture as verifier  # noqa: E402


def sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(fixture: Path, document: dict[str, Any]) -> str:
	path = fixture / "manifest.json"
	path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return sha256(path)


def build_fixture(fixture: Path) -> tuple[dict[str, Any], str]:
	token_ids = TOKEN_IDS
	token_path = fixture / verifier.TOKEN_PATH
	vector_path = fixture / verifier.VECTOR_PATH
	token_path.write_bytes(b"".join(token.to_bytes(4, "little") for token in token_ids))
	vector_path.write_bytes(b"\x80\x3f" + bytes(verifier.VECTOR_BYTES - 2))
	document: dict[str, Any] = {
		"format": verifier.FORMAT,
		"checkpoint": {
			"model": verifier.MODEL,
			"revision": verifier.REVISION,
			"index_sha256": verifier.INDEX_SHA256,
			"config_sha256": verifier.CONFIG_SHA256,
			"tokenizer_sha256": verifier.TOKENIZER_SHA256,
		},
		"generator": {
			"path": "dsv4_ga_reference_vector.py",
			"sha256": sha256(ROOT / "tools" / "dsv4_ga_reference_vector.py"),
		},
		"reference": {
			"model_py_sha256": verifier.REFERENCE_MODEL_SHA256,
			"kernel_py_sha256": verifier.REFERENCE_KERNEL_SHA256,
			"config_sha256": verifier.REFERENCE_CONFIG_SHA256,
			"sparse_attention_fallback": "torch-global-softmax-bf16-probability-v1",
			"hadamard_fallback": "normalized-sylvester-hadamard-v1",
			"fp8_gemm_fallback": "torch-block128-fp32-accumulation-v1",
			"torch": "test",
			"python": "test",
			"cuda": "test",
			"device": "test",
			"device_capability": [12, 1],
		},
		"input": {
			"batch_json_sha256": verifier.REFERENCE_BATCH_JSON_SHA256,
			"source_request_id": 76000,
			"source_sequence_id": 76000,
			"token_artifact": {
				"path": verifier.TOKEN_PATH,
				"sha256": sha256(token_path),
				"bytes": verifier.TOKEN_BYTES,
				"dtype": "uint32_le",
				"shape": [1, verifier.TOKEN_COUNT],
			},
			"token_ids": [token_ids],
			"positions": [list(range(verifier.TOKEN_COUNT))],
			"validation_sequence_ids": [1],
			"tensor_order": "batch,sequence,hc,hidden",
		},
		"layer_range": {"first": 0, "count": 3},
		"loaded_parameter_count": 1,
		"loaded_parameter_names_sha256": "6" * 64,
		"consumed_checkpoint_tensor_count": 1,
		"consumed_checkpoint_tensor_names_sha256": "7" * 64,
		"source_shards": [
			{"path": path, "bytes": identity[0], "sha256": identity[1]}
			for path, identity in verifier.SOURCE_SHARDS.items()
		],
		"vectors": [
			{
				"path": verifier.VECTOR_PATH,
				"sha256": sha256(vector_path),
				"bytes": verifier.VECTOR_BYTES,
				"shape": [1, verifier.TOKEN_COUNT, verifier.HC_STREAM_COUNT, verifier.HIDDEN_DIMENSION],
				"dtype": "torch.bfloat16",
				"min": 0.0,
				"max": 1.0,
				"mean": 1.0 / verifier.VECTOR_ELEMENTS,
				"l2": 1.0,
				"nonfinite": 0,
			}
		],
	}
	return document, write_manifest(fixture, document)


def expect_error(action: Any, label: str) -> None:
	try:
		action()
	except verifier.FixtureError:
		return
	raise SystemExit(f"verifier accepted {label}")


def main() -> int:
	with tempfile.TemporaryDirectory(prefix="sparkpipe-dsv4-reference-") as directory:
		fixture = Path(directory)
		document, manifest_sha256 = build_fixture(fixture)
		verifier.verify_fixture(fixture, manifest_sha256)
		vector_path = fixture / verifier.VECTOR_PATH
		with vector_path.open("r+b") as handle:
			handle.write(b"\x01")
		expect_error(lambda: verifier.verify_fixture(fixture, manifest_sha256), "corrupt vector payload")
		vector_path.write_bytes(b"\x80\x3f" + bytes(verifier.VECTOR_BYTES - 2))
		document["vectors"][0]["mean"] = 0.0
		dishonest_statistics_sha256 = write_manifest(fixture, document)
		expect_error(lambda: verifier.verify_fixture(fixture, dishonest_statistics_sha256), "dishonest vector statistics")
		document["vectors"][0]["mean"] = 1.0 / verifier.VECTOR_ELEMENTS
		document["checkpoint"]["revision"] = "0" * 40
		wrong_identity_sha256 = write_manifest(fixture, document)
		expect_error(lambda: verifier.verify_fixture(fixture, wrong_identity_sha256), "wrong checkpoint identity")
		document["checkpoint"]["revision"] = verifier.REVISION
		manifest_sha256 = write_manifest(fixture, document)
		extra = fixture / "unexpected"
		extra.write_bytes(b"x")
		expect_error(lambda: verifier.verify_fixture(fixture, manifest_sha256), "unexpected fixture entry")
	retained_fixture = ROOT / "qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128"
	if retained_fixture.is_dir():
		makefile = (ROOT / "modules/dsv4_resident_decode_stage/Makefile").read_text(encoding="utf-8")
		digest_line = next(line for line in makefile.splitlines() if line.startswith("override DSV4_GA_STAGE0_REFERENCE_MANIFEST_SHA256 :="))
		retained_digest = digest_line.split(":=", 1)[1].strip()
		verifier.verify_fixture(retained_fixture, retained_digest)
	print("PASS DSV4 GA reference fixture verifier")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
