#!/usr/bin/env python3
"""Verify the retained DeepSeek V4 GA stage-0 numerical fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import struct
import sys
from pathlib import Path
from typing import Any

from dsv4_ga_reference_profiles import PROFILES


FORMAT = "sparkpipe-dsv4-ga-reference-vectors-v1"
TOKEN_PATH = "prompt_tokens.u32le"
VECTOR_PATH = "after_layer_2.bf16le"
MANIFEST_MAX_BYTES = 256 * 1024
SHA256_HEX_LENGTH = 64
PROFILE = PROFILES["flash"]


def token_bytes() -> int:
	return PROFILE["token_count"] * 4


def vector_bytes() -> int:
	return PROFILE["token_count"] * PROFILE["hc_stream_count"] * PROFILE["hidden_dimension"] * 2


def vector_elements() -> int:
	return PROFILE["token_count"] * PROFILE["hc_stream_count"] * PROFILE["hidden_dimension"]


class FixtureError(RuntimeError):
	"""A retained fixture failed an identity or schema check."""


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("fixture_directory", type=Path)
	parser.add_argument("expected_manifest_sha256")
	parser.add_argument(
		"--profile",
		choices=sorted(PROFILES),
		default="flash",
		help="checkpoint identity/geometry profile the fixture was generated against",
	)
	return parser.parse_args()


def require(condition: bool, message: str) -> None:
	if not condition:
		raise FixtureError(message)


def require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
	require(isinstance(value, dict), f"{label} must be an object")
	require(set(value) == keys, f"{label} keys are not exact")
	return value


def require_integer(value: Any, expected: int, label: str) -> None:
	require(not isinstance(value, bool) and isinstance(value, int), f"{label} must be an integer")
	require(value == expected, f"{label} is not exact")


def require_sha256(value: Any, label: str) -> str:
	require(isinstance(value, str), f"{label} must be a string")
	require(len(value) == SHA256_HEX_LENGTH, f"{label} length is invalid")
	require(all(character in "0123456789abcdef" for character in value), f"{label} is not lowercase SHA-256")
	return value


def require_string(value: Any, expected: str | None, label: str) -> str:
	require(isinstance(value, str) and value != "", f"{label} must be a non-empty string")
	if expected is not None:
		require(value == expected, f"{label} is not exact")
	return value


def require_finite_number(value: Any, label: str) -> float:
	require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} must be numeric")
	converted = float(value)
	require(math.isfinite(converted), f"{label} must be finite")
	return converted


def require_regular_file(path: Path, expected_bytes: int | None, label: str) -> None:
	try:
		metadata = path.lstat()
	except OSError as error:
		raise FixtureError(f"{label} is unavailable: {error}") from error
	require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
	require(not path.is_symlink(), f"{label} must not be a symlink")
	if expected_bytes is not None:
		require(metadata.st_size == expected_bytes, f"{label} byte count is not exact")


def sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		while chunk := handle.read(1024 * 1024):
			digest.update(chunk)
	return digest.hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for key, value in pairs:
		if key in result:
			raise FixtureError(f"manifest contains duplicate key: {key}")
		result[key] = value
	return result


def reject_json_constant(value: str) -> Any:
	raise FixtureError(f"manifest contains non-JSON numeric constant: {value}")


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
	require_regular_file(path, None, "manifest")
	size = path.stat().st_size
	require(0 < size <= MANIFEST_MAX_BYTES, "manifest byte count is invalid")
	require(sha256_file(path) == expected_sha256, "manifest SHA-256 mismatch")
	try:
		text = path.read_text(encoding="utf-8")
		document = json.loads(
			text,
			object_pairs_hook=unique_object,
			parse_constant=reject_json_constant,
		)
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise FixtureError(f"manifest is not strict UTF-8 JSON: {error}") from error
	return require_exact_keys(
		document,
		{
			"format",
			"checkpoint",
			"generator",
			"reference",
			"input",
			"layer_range",
			"loaded_parameter_count",
			"loaded_parameter_names_sha256",
			"consumed_checkpoint_tensor_count",
			"consumed_checkpoint_tensor_names_sha256",
			"source_shards",
			"vectors",
		},
		"manifest",
	)


def verify_checkpoint(value: Any) -> None:
	checkpoint = require_exact_keys(
		value,
		{"model", "revision", "index_sha256", "config_sha256", "tokenizer_sha256"},
		"checkpoint",
	)
	require_string(checkpoint["model"], PROFILE["model"], "checkpoint.model")
	require_string(checkpoint["revision"], PROFILE["revision"], "checkpoint.revision")
	require_string(checkpoint["index_sha256"], PROFILE["index_sha256"], "checkpoint.index_sha256")
	require_string(checkpoint["config_sha256"], PROFILE["config_sha256"], "checkpoint.config_sha256")
	require_string(checkpoint["tokenizer_sha256"], PROFILE["tokenizer_sha256"], "checkpoint.tokenizer_sha256")


def verify_provenance(generator_value: Any, reference_value: Any) -> None:
	generator = require_exact_keys(generator_value, {"path", "sha256"}, "generator")
	require_string(generator["path"], "dsv4_ga_reference_vector.py", "generator.path")
	generator_digest = require_sha256(generator["sha256"], "generator.sha256")
	generator_path = Path(__file__).with_name("dsv4_ga_reference_vector.py")
	require_regular_file(generator_path, None, "reference generator")
	require(sha256_file(generator_path) == generator_digest, "generator SHA-256 does not match the repository tool")
	reference = require_exact_keys(
		reference_value,
		{
			"model_py_sha256",
			"kernel_py_sha256",
			"config_sha256",
			"sparse_attention_fallback",
			"hadamard_fallback",
			"fp8_gemm_fallback",
			"torch",
			"python",
			"cuda",
			"device",
			"device_capability",
		},
		"reference",
	)
	require_string(reference["model_py_sha256"], PROFILE["reference_model_sha256"], "reference.model_py_sha256")
	require_string(reference["kernel_py_sha256"], PROFILE["reference_kernel_sha256"], "reference.kernel_py_sha256")
	require_string(reference["config_sha256"], PROFILE["reference_config_sha256"], "reference.config_sha256")
	require_string(reference["sparse_attention_fallback"], "torch-global-softmax-bf16-probability-v1", "reference.sparse_attention_fallback")
	require_string(reference["hadamard_fallback"], "normalized-sylvester-hadamard-v1", "reference.hadamard_fallback")
	require_string(reference["fp8_gemm_fallback"], "torch-block128-fp32-accumulation-v1", "reference.fp8_gemm_fallback")
	for field in ("torch", "python", "cuda", "device"):
		require_string(reference[field], None, f"reference.{field}")
	require(reference["device_capability"] == [12, 1], "reference.device_capability is not SM121")


def verify_token_record(value: Any, fixture: Path, token_ids: Any) -> None:
	record = require_exact_keys(value, {"path", "sha256", "bytes", "dtype", "shape"}, "input.token_artifact")
	require_string(record["path"], TOKEN_PATH, "input.token_artifact.path")
	require_integer(record["bytes"], token_bytes(), "input.token_artifact.bytes")
	require_string(record["dtype"], "uint32_le", "input.token_artifact.dtype")
	require(record["shape"] == [1, PROFILE["token_count"]], "input.token_artifact.shape is not exact")
	digest = require_string(record["sha256"], PROFILE["token_payload_sha256"], "input.token_artifact.sha256")
	path = fixture / TOKEN_PATH
	require_regular_file(path, token_bytes(), "token artifact")
	require(sha256_file(path) == digest, "token artifact SHA-256 mismatch")
	require(isinstance(token_ids, list) and len(token_ids) == 1, "input.token_ids must have one row")
	row = token_ids[0]
	require(isinstance(row, list) and len(row) == PROFILE["token_count"], "input.token_ids row is not B1x128")
	for token in row:
		require(not isinstance(token, bool) and isinstance(token, int), "input token must be an integer")
		require(0 <= token < PROFILE["vocabulary_size"], "input token is outside the GA vocabulary")
	with path.open("rb") as handle:
		payload_tokens = [value[0] for value in struct.iter_unpack("<I", handle.read())]
	require(payload_tokens == row, "token artifact does not match manifest token_ids")


def verify_input(value: Any, fixture: Path) -> None:
	input_record = require_exact_keys(
		value,
		{"batch_json_sha256", "source_request_id", "source_sequence_id", "token_artifact", "token_ids", "positions", "validation_sequence_ids", "tensor_order"},
		"input",
	)
	require_string(input_record["batch_json_sha256"], PROFILE["batch_json_sha256"], "input.batch_json_sha256")
	require_integer(input_record["source_request_id"], 76000, "input.source_request_id")
	require_integer(input_record["source_sequence_id"], 76000, "input.source_sequence_id")
	require(input_record["positions"] == [list(range(PROFILE["token_count"]))], "input.positions is not exact")
	require(input_record["validation_sequence_ids"] == [1], "input.validation_sequence_ids is not exact")
	require_string(input_record["tensor_order"], "batch,sequence,hc,hidden", "input.tensor_order")
	verify_token_record(input_record["token_artifact"], fixture, input_record["token_ids"])


def verify_source_shards(value: Any) -> None:
	require(isinstance(value, list) and len(value) == len(PROFILE["source_shards"]), "source_shards inventory is not exact")
	seen: set[str] = set()
	for index, item in enumerate(value):
		record = require_exact_keys(item, {"path", "bytes", "sha256"}, f"source_shards[{index}]")
		path = require_string(record["path"], None, f"source_shards[{index}].path")
		require(path in PROFILE["source_shards"] and path not in seen, f"source_shards[{index}].path is not exact")
		expected_bytes, expected_digest = PROFILE["source_shards"][path]
		require_integer(record["bytes"], expected_bytes, f"source_shards[{index}].bytes")
		require_string(record["sha256"], expected_digest, f"source_shards[{index}].sha256")
		seen.add(path)
	require(seen == set(PROFILE["source_shards"]), "source_shards inventory is incomplete")


def bf16_to_float(value: int) -> float:
	return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def vector_payload_statistics(path: Path) -> tuple[float, float, float, float, int, int]:
	minimum = math.inf
	maximum = -math.inf
	total = 0.0
	total_squared = 0.0
	nonfinite = 0
	nonzero = 0
	with path.open("rb") as handle:
		payload = handle.read()
	for (bits,) in struct.iter_unpack("<H", payload):
		value = bf16_to_float(bits)
		if not math.isfinite(value):
			nonfinite += 1
			continue
		minimum = min(minimum, value)
		maximum = max(maximum, value)
		total += value
		total_squared += value * value
		nonzero += value != 0.0
	require(nonfinite == 0, "vector payload contains non-finite BF16 values")
	require(nonzero > 0, "vector payload is all zero")
	return minimum, maximum, total / vector_elements(), math.sqrt(total_squared), nonfinite, nonzero


def require_close(actual: float, claimed: float, label: str) -> None:
	tolerance = max(1e-9, abs(actual) * 1e-5)
	require(abs(actual - claimed) <= tolerance, f"{label} does not match the vector payload")


def verify_vector(value: Any, fixture: Path) -> None:
	record = require_exact_keys(
		value,
		{"path", "sha256", "bytes", "shape", "dtype", "min", "max", "mean", "l2", "nonfinite"},
		"vectors[0]",
	)
	require_string(record["path"], VECTOR_PATH, "vectors[0].path")
	require_integer(record["bytes"], vector_bytes(), "vectors[0].bytes")
	require(record["shape"] == [1, PROFILE["token_count"], PROFILE["hc_stream_count"], PROFILE["hidden_dimension"]], "vectors[0].shape is not exact")
	require_string(record["dtype"], "torch.bfloat16", "vectors[0].dtype")
	require_integer(record["nonfinite"], 0, "vectors[0].nonfinite")
	claimed_minimum = require_finite_number(record["min"], "vectors[0].min")
	claimed_maximum = require_finite_number(record["max"], "vectors[0].max")
	claimed_mean = require_finite_number(record["mean"], "vectors[0].mean")
	claimed_l2 = require_finite_number(record["l2"], "vectors[0].l2")
	require(claimed_minimum <= claimed_maximum, "vectors[0] range is invalid")
	require(claimed_l2 > 0.0, "vectors[0].l2 must be positive")
	digest = require_sha256(record["sha256"], "vectors[0].sha256")
	path = fixture / VECTOR_PATH
	require_regular_file(path, vector_bytes(), "vector artifact")
	require(sha256_file(path) == digest, "vector artifact SHA-256 mismatch")
	minimum, maximum, mean, l2, nonfinite, _ = vector_payload_statistics(path)
	require(minimum == claimed_minimum, "vectors[0].min does not match the vector payload")
	require(maximum == claimed_maximum, "vectors[0].max does not match the vector payload")
	require_integer(record["nonfinite"], nonfinite, "vectors[0].nonfinite")
	require_close(mean, claimed_mean, "vectors[0].mean")
	require_close(l2, claimed_l2, "vectors[0].l2")


def verify_fixture(fixture: Path, expected_manifest_sha256: str) -> None:
	expected_digest = require_sha256(expected_manifest_sha256, "expected manifest SHA-256")
	require(fixture.is_dir() and not fixture.is_symlink(), "fixture directory is unavailable")
	expected_names = {"manifest.json", TOKEN_PATH, VECTOR_PATH}
	require({path.name for path in fixture.iterdir()} == expected_names, "fixture directory entries are not exact")
	manifest = load_manifest(fixture / "manifest.json", expected_digest)
	require_string(manifest["format"], FORMAT, "format")
	verify_checkpoint(manifest["checkpoint"])
	verify_provenance(manifest["generator"], manifest["reference"])
	verify_input(manifest["input"], fixture)
	layer_range = require_exact_keys(manifest["layer_range"], {"first", "count"}, "layer_range")
	require_integer(layer_range["first"], 0, "layer_range.first")
	require_integer(layer_range["count"], 3, "layer_range.count")
	loaded_count = manifest["loaded_parameter_count"]
	require(not isinstance(loaded_count, bool) and isinstance(loaded_count, int) and loaded_count > 0, "loaded_parameter_count is invalid")
	require_sha256(manifest["loaded_parameter_names_sha256"], "loaded_parameter_names_sha256")
	consumed_count = manifest["consumed_checkpoint_tensor_count"]
	require(not isinstance(consumed_count, bool) and isinstance(consumed_count, int) and consumed_count >= loaded_count, "consumed_checkpoint_tensor_count is invalid")
	require_sha256(manifest["consumed_checkpoint_tensor_names_sha256"], "consumed_checkpoint_tensor_names_sha256")
	verify_source_shards(manifest["source_shards"])
	require(isinstance(manifest["vectors"], list) and len(manifest["vectors"]) == 1, "vectors must contain one boundary tensor")
	verify_vector(manifest["vectors"][0], fixture)


def main() -> int:
	global PROFILE
	options = parse_args()
	PROFILE = PROFILES[options.profile]
	if PROFILE["token_payload_sha256"] is None:
		print("DSV4 GA reference fixture invalid: profile token payload is not established", file=sys.stderr)
		return 1
	try:
		verify_fixture(options.fixture_directory.absolute(), options.expected_manifest_sha256)
	except FixtureError as error:
		print(f"DSV4 GA reference fixture invalid: {error}", file=sys.stderr)
		return 1
	print(f"PASS DSV4 GA reference fixture manifest={options.expected_manifest_sha256}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
