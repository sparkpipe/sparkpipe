#!/usr/bin/env python3
"""Pin the DSV4 head-scale seeding contract and the slice-independent oracle.

Two invariants are enforced here:

1. hc_head_scale_value / mtp_hc_head_scale_value are seeded from the pack's
   1x1 HC_HEAD_SCALE tensors with one synchronous device readback at bind
   time. The pack payloads live in device memory, so no code path may
   dereference them on the host; launches consume the host mirror by value.
2. The CUDA validator's numerical oracle gate is slice-independent: any
   (stage, first layer, layer count) inside the model runs the comparison,
   deeper slices consuming the fixture's previous boundary as hidden input.
   Stage 0 layers 0..2 stays the mandated configuration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = (
	ROOT / "modules/dsv4_resident_decode_stage/source"
	/ "spark_dsv4_resident_decode_stage_module.c"
)
VALIDATOR = (
	ROOT / "modules/dsv4_resident_decode_stage/validation"
	/ "spark_dsv4_resident_decode_stage_cuda_validation.cu"
)
DRIVER = (
	ROOT / "modules/dsv4_resident_decode_stage/validation"
	/ "validate_dsv4_resident_decode_stage_cuda.sh"
)
sys.path.insert(0, str(ROOT / "tools"))
import verify_dsv4_ga_reference_fixture as verifier  # noqa: E402
_ga_fixture_spec = importlib.util.spec_from_file_location(
	"dsv4_ga_fixture_contract", ROOT / "tests/test_dsv4_ga_reference_fixture.py")
_ga_fixture = importlib.util.module_from_spec(_ga_fixture_spec)
_ga_fixture_spec.loader.exec_module(_ga_fixture)


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


def check_head_scale_seeding() -> None:
	module = MODULE.read_text(encoding="utf-8")
	helper = function_body(
		module,
		"static SparkStatus SparkDsv4ModuleSeedHeadScaleValue("
		"const float *device_scalar,float *host_value)",
	)
	bind = function_body(module, "static SparkStatus SparkDsv4ModuleBindGlobal")
	coverage = function_body(module, "static SparkStatus SparkDsv4ModuleVerifyCoverage")
	# Both scalars are seeded from the pack tensor through the device-safe
	# readback helper at bind time.
	require(helper, "cudaMemcpyDeviceToHost", "bind-time readback copy")
	require(
		bind,
		"SparkDsv4ModuleSeedHeadScaleValue((const float *)payload,"
		"&state->hc_head_scale_value)",
		"global head scale seed",
	)
	require(
		bind,
		"SparkDsv4ModuleSeedHeadScaleValue((const float *)payload,"
		"&state->mtp_hc_head_scale_value)",
		"mtp head scale seed",
	)
	# A failed readback must fail the pack load, not arm a zero scale.
	require(
		bind,
		"if ( status != SPARK_STATUS_OK )\n\t\t\treturn(status);",
		"readback failure propagation",
	)
	# Launches and gates consume host mirrors only.
	require(
		module,
		"slot->mixes_f32,state->hc_head_scale_value,",
		"target head launch uses host mirror",
	)
	require(
		module,
		"slot->mixes_f32,state->mtp_hc_head_scale_value,",
		"dspark head launch uses host mirror",
	)
	require(
		coverage,
		"isfinite(state->hc_head_scale_value) == 0",
		"global head scale gate",
	)
	require(
		coverage,
		"isfinite(state->mtp_hc_head_scale_value) == 0",
		"mtp head scale gate",
	)
	# The pack tensor itself stays coverage-mandated for every rank.
	require(
		module,
		"tensor <= SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE",
		"HC_HEAD_SCALE coverage mandate",
	)
	# No host-side dereference of the device-resident scale payloads.
	reject(module, "((const float *)payload)[0]", "host dereference of pack payload")
	reject(module, "hc_head_scale_f32[0]", "host index of device scale tensor")


def check_mtp_seed_placement() -> None:
	module = MODULE.read_text(encoding="utf-8")
	bind = function_body(module, "static SparkStatus SparkDsv4ModuleBindGlobal")
	coverage = function_body(module, "static SparkStatus SparkDsv4ModuleVerifyCoverage")
	global_case = bind.index("case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE:")
	global_seed = bind.index(
		"SparkDsv4ModuleSeedHeadScaleValue((const float *)payload,"
		"&state->hc_head_scale_value)")
	mtp_case = bind.index("case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE:")
	mtp_seed = bind.index(
		"SparkDsv4ModuleSeedHeadScaleValue((const float *)payload,"
		"&state->mtp_hc_head_scale_value)")
	assert global_case < global_seed < mtp_case < mtp_seed, (
		"head-scale seeds escaped their binder cases")
	require(
		coverage,
		"for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ; "
		"tensor <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ; tensor++)",
		"MTP extras coverage mandate",
	)
	mtp_block = coverage.index("SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u")
	extras_gate = coverage.index("state->mtp.hc_head_scale_f32 == 0")
	value_gate = coverage.index("isfinite(state->mtp_hc_head_scale_value) == 0")
	assert mtp_block < extras_gate < value_gate, (
		"MTP scale gates must sit inside the MTP>0 coverage block")
	require(coverage, "pack_mtp_head_scale_invalid",
			"distinct refusal message for the MTP mirror")


def check_mtp_variant_chain() -> None:
	internal_header = (
		ROOT / "modules/dsv4_resident_decode_stage/source"
		/ "spark_dsv4_resident_decode_stage_internal.h"
	).read_text(encoding="utf-8")
	require(
		internal_header,
		"SPARK_DSV4_GEOM_ROW(SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE,"
		"SPARK_DSV4_STAGEPACK_CLASS_GLOBAL,",
		"MTP scale classified as a global record",
	)
	flash_packer = (ROOT / "tools/dsv4_stagepack.py").read_text(encoding="utf-8")
	require(
		flash_packer,
		"add_record(records, KIND_MTP_HC_HEAD_SCALE, GLOBAL_LAYER, "
		"WEIGHT_F32, 1, 1,",
		"Flash packer emits the draft scale as GLOBAL 1x1 F32",
	)
	pro_packer = (ROOT / "tools/dsv4_pro_stagepack.py").read_text(encoding="utf-8")
	require(
		pro_packer,
		"flash.KIND_MTP_HC_HEAD_SCALE,",
		"Pro packer references the draft scale kind",
	)
	require(
		pro_packer,
		'("mtp.2.hc_head_scale",))',
		"Pro packer sources it from mtp.2.hc_head_scale",
	)
	# Cross-boundary numeric pin: the Python packer's wire constants and the
	# C loader's tensor-kind enum must agree value-for-value. A renumber on
	# either side would otherwise surface only as refused packs at load
	# time - or worse, as silently misrouted records - instead of here.
	c_mtp_kinds = dict(re.findall(
		r"SPARK_DSV4_STAGEPACK_TENSOR_(MTP_[A-Z_0-9]+) = (\d+),",
		internal_header))
	py_mtp_kinds = dict(re.findall(
		r"KIND_(MTP_[A-Z_0-9]+) = (\d+)", flash_packer))
	assert set(c_mtp_kinds) == set(py_mtp_kinds), (c_mtp_kinds, py_mtp_kinds)
	for kind_name, c_value in c_mtp_kinds.items():
		assert int(c_value) == int(py_mtp_kinds[kind_name]), kind_name
	assert int(c_mtp_kinds["MTP_HC_HEAD_SCALE"]) == 46
	c_global = re.search(
		r"#define SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ([A-Z0-9_]+)",
		internal_header)
	# The C header may spell the sentinel as UINT32_MAX; resolve that one
	# macro by hand so both spellings pin to the same wire value.
	assert c_global is not None, "GLOBAL_LAYER define missing"
	global_layer_text = c_global.group(1)
	global_layer_value = (0xFFFFFFFF if global_layer_text == "UINT32_MAX"
		else int(global_layer_text, 0))
	assert global_layer_value == 0xFFFFFFFF, global_layer_text
	assert "GLOBAL_LAYER = 0xFFFFFFFF" in flash_packer
	c_weight_f32 = re.search(
		r"#define SPARK_DSV4_STAGEPACK_WEIGHT_F32 (\d+)u", internal_header)
	assert c_weight_f32 is not None and int(c_weight_f32.group(1)) == 1
	assert re.search(r"WEIGHT_F32 = 1\b", flash_packer)
	c_kind_count = re.search(
		r"SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT = (\d+)", internal_header)
	assert c_kind_count is not None and int(c_kind_count.group(1)) == 50
	assert max(int(v) for v in py_mtp_kinds.values()) == 49


def check_head_scale_seed_semantics() -> None:
	"""Compile and run the seed/gate math against the repository CUDA stubs."""
	probe_source = r'''#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

typedef int SparkStatus;
#define SPARK_STATUS_OK 0
#define SPARK_STATUS_INVALID_ARGUMENT 7
#define SPARK_DSV4_MODULE_TAG "p0a_probe"
static SparkStatus SparkStageModuleCudaStatus(const char *tag,cudaError_t error,const char *label);
static SparkStatus SparkDsv4ModuleSeedHeadScaleValue(const float *device_scalar,float *host_value)
{
	cudaError_t error;
	if ( device_scalar == 0 || host_value == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*host_value = 0.0f;
	error = cudaMemcpy(host_value,device_scalar,sizeof(*host_value),cudaMemcpyDeviceToHost);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"head_scale_readback"));
}
static SparkStatus SparkStageModuleCudaStatus(const char *tag,cudaError_t error,const char *label)
{
	(void)tag;(void)label;
	return error == cudaSuccess ? SPARK_STATUS_OK : 61;
}
/* Mirror of VerifyCoverage's MTP>0 value gate. */
static int mtp_head_scale_gate(float mirror)
{
	if ( !isfinite(mirror) || mirror == 0.0f )
		return(1);
	return(0);
}
int main(void)
{
	float scalar = 0.68915f,mirror = 12345.0f;
	float mtp_mirror,mtp_payload = 0.68916f,bad_payload;
	if ( SparkDsv4ModuleSeedHeadScaleValue(&scalar,&mirror) != SPARK_STATUS_OK ) return 1;
	if ( mirror != scalar ) return 2;
	if ( !isfinite(mirror) || mirror == 0.0f ) return 3;
	if ( SparkDsv4ModuleSeedHeadScaleValue(0,&mirror) != SPARK_STATUS_INVALID_ARGUMENT ) return 4;
	/* MTP>0 flow: an unseeded (calloc-zero) mirror must refuse, a seeded
	 * one must pass, and a NaN payload must seed fine but then refuse. */
	mtp_mirror = 0.0f;
	if ( mtp_head_scale_gate(mtp_mirror) != 1 ) return 5;
	if ( SparkDsv4ModuleSeedHeadScaleValue(&mtp_payload,&mtp_mirror) != SPARK_STATUS_OK ) return 6;
	if ( mtp_mirror != mtp_payload ) return 7;
	if ( mtp_head_scale_gate(mtp_mirror) != 0 ) return 8;
	bad_payload = NAN;
	if ( SparkDsv4ModuleSeedHeadScaleValue(&bad_payload,&mtp_mirror) != SPARK_STATUS_OK ) return 9;
	if ( mtp_head_scale_gate(mtp_mirror) != 1 ) return 10;
	printf("PASS p0a probe\\n");
	return 0;
}
'''
	with tempfile.TemporaryDirectory(prefix="p0a-probe-") as temporary:
		root = Path(temporary)
		source = root / "probe.c"
		binary = root / "probe"
		source.write_text(probe_source, encoding="utf-8")
		compile_command = [
			*__import__("shlex").split(__import__("os").environ.get("CC", "cc")),
			"-std=c11", "-Wall", "-Wextra", "-Werror",
			"-Itests/cuda_stub",
			str(source), str(ROOT / "tests/cuda_stub/cuda_runtime_stub.c"),
			"-lm", "-o", str(binary),
		]
		result = subprocess.run(compile_command, cwd=ROOT, text=True, capture_output=True, check=False)
		if result.returncode != 0:
			print(result.stdout)
			print(result.stderr, file=sys.stderr)
			raise SystemExit("P0-A probe failed to compile")
		run = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
		if run.returncode != 0 or "PASS p0a probe" not in run.stdout:
			raise SystemExit(f"P0-A probe failed: rc={run.returncode} out={run.stdout!r}")


def check_pro_variant_seam() -> None:
	"""Pro geometry must seed identically: compile the module TU and run
	the executable bind/coverage seam under -DSPARK_DSV4_PRO_BUILD=1
	(61-layer alias of the shared model header, MTP count unchanged).
	Consolidation candidate: the auto-stub link loop below duplicates
	tests/test_dsv4_head_scale_value.py and should move into a shared
	helper once that guard's ownership settles.
	"""
	cc = os.environ.get("CC", "cc")
	pro_syntax_flags = [
		"-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
		"-D_POSIX_C_SOURCE=200809L", "-D_FILE_OFFSET_BITS=64",
		"-DSPARK_BATCH_BUCKET=1024u", "-DSPARK_DSV4_PRO_BUILD=1",
		"-I.", "-Iinclude", "-Isrc", "-Itests/cuda_stub",
		"-Imodel-families/common/include",
		"-Imodel-families/dsv4/include",
		"-Imodules/dsv4_resident_decode_stage/include",
		"-Imodules/dsv4_resident_decode_stage/source",
		"-include", "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h",
	]
	syntax_result = subprocess.run(
		shlex.split(cc) + pro_syntax_flags + [str(MODULE)],
		cwd=ROOT, text=True, capture_output=True, check=False)
	if syntax_result.returncode != 0:
		print(syntax_result.stdout)
		print(syntax_result.stderr, file=sys.stderr)
		raise SystemExit("PRO-variant module syntax failed")

	with tempfile.TemporaryDirectory(prefix="p0a-pro-") as temporary:
		root = Path(temporary)
		objects = []
		for source, extra, suffix in (
			(ROOT / "tests/test_dsv4_head_scale_value.c",
			 ["-DSPARK_DSV4_PRO_BUILD=1"], "test"),
			(ROOT / "tests/cuda_stub/cuda_runtime_stub.c", [], "stub"),
			(ROOT / "runtime/stage_module_common.c", [], "common"),
		):
			obj = root / ("pro_seam_" + suffix + ".o")
			build = subprocess.run(
				shlex.split(cc) + [
					"-std=c11", "-Wall", "-Wextra", "-Werror", "-O1",
					"-D_POSIX_C_SOURCE=200809L", "-D_FILE_OFFSET_BITS=64",
					"-DSPARK_BATCH_BUCKET=1024u", *extra,
					"-include",
					"model-families/dsv4/include/sparkpipe/spark_dsv4_model.h",
					"-I.", "-Iinclude", "-Isrc", "-Itests/cuda_stub",
					"-Imodel-families/common/include",
					"-Imodel-families/dsv4/include",
					"-Imodules/dsv4_resident_decode_stage/include",
					"-Imodules/dsv4_resident_decode_stage/source",
					"-c", str(source), "-o", str(obj),
				],
				cwd=ROOT, text=True, capture_output=True, check=False)
			if build.returncode != 0:
				print(build.stdout)
				print(build.stderr, file=sys.stderr)
				raise SystemExit("PRO seam build failed: " + suffix)
			objects.append(obj)
		link_base = shlex.split(cc) + [str(obj) for obj in objects] + [
			"-lm", "-lpthread"]
		binary = root / "pro_seam.bin"
		linked = False
		for _round in range(3):
			link = subprocess.run(link_base + ["-o", str(binary)], cwd=ROOT,
				text=True, capture_output=True, check=False)
			if link.returncode == 0:
				linked = True
				break
			undefined = sorted(set(re.findall(
				'"_?([A-Za-z_][A-Za-z0-9_]*)", referenced from',
				link.stderr)))
			unexpected = [s for s in undefined
				if not s.startswith(("Spark", "cuda"))]
			if not undefined or unexpected:
				raise SystemExit(
					f"PRO seam link failed: {unexpected or 'unparsed'}")
			stub_c = root / "link_stubs.c"
			stub_c.write_text("".join(
				f"long {symbol}(void) {{ return 0L; }}\n"
				for symbol in undefined), encoding="utf-8")
			stub_o = root / "link_stubs.o"
			subprocess.run(shlex.split(cc) + ["-std=c11", "-O0", "-c",
				str(stub_c), "-o", str(stub_o)], cwd=ROOT, check=True)
			link_base.append(str(stub_o))
		if not linked:
			raise SystemExit("PRO seam link unresolved")
		executed = subprocess.run([str(binary)], cwd=ROOT, text=True,
			capture_output=True, check=False)
		if executed.returncode != 0 or \
				"PASS dsv4 hc_head_scale_value" not in executed.stdout:
			raise SystemExit(
				f"PRO seam execution failed: {executed.stdout!r}")


def check_oracle_gate_wiring() -> None:
	validator = VALIDATOR.read_text(encoding="utf-8")
	run_reference = function_body(validator, "static int SparkDsv4ValidationRunReference")
	require_mode = function_body(validator, "static int SparkDsv4ValidationRequireMode")
	load_mode = function_body(validator, "static int SparkDsv4ValidationLoadMode")
	# The 3-layer-only refusal is gone; slice bounds replace it.
	reject(run_reference, "layer_count != 3u", "hard-coded 3-layer oracle pin")
	reject(run_reference, "reference_stage0_slice", "legacy stage-0 oracle refusal")
	require(run_reference, "reference_slice_range", "oracle slice bounds")
	require(run_reference, "hidden_path", "boundary hidden input parameter")
	require(
		run_reference,
		"SparkDsv4ValidationBuildReferenceFrame(&frame,token_path,hidden_path)",
		"frame builder consumes boundary input",
	)
	# Any valid slice may enable the oracle; stage 0 layers 0..2 must still run it.
	require(
		require_mode,
		"node_context->first_layer_index + node_context->layer_count "
		"> SPARK_DSV4_MODEL_LAYER_COUNT",
		"slice upper bound",
	)
	require(
		require_mode,
		"exact_reference_slice != 0 && mode->use_reference == 0",
		"stage-0 three-layer oracle mandate",
	)
	require(
		require_mode,
		"reference_hidden_configuration=invalid",
		"boundary input pairing gate",
	)
	require(
		load_mode,
		'getenv("SPARK_DSV4_REFERENCE_INPUT_HIDDEN_PATH")',
		"boundary input environment",
	)
	require(
		validator,
		"SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER",
		"hidden input context flag",
	)
	# The validation driver addresses per-slice fixtures and verifies identity.
	driver = DRIVER.read_text(encoding="utf-8")
	require(driver, "after_layer_${last_layer}.bf16le", "slice-addressed golden output")
	require(driver, "SPARK_DSV4_REFERENCE_INPUT_HIDDEN_PATH", "driver boundary export")
	require(driver, "--expect-layer-first", "fixture range expectation")


def build_range_fixture(directory: Path, first: int, count: int) -> tuple[dict, str]:
	vector_name = f"after_layer_{first + count - 1}.bf16le"
	token_ids = _ga_fixture.TOKEN_IDS
	token_payload = b"".join(token.to_bytes(4, "little") for token in token_ids)
	(directory / verifier.TOKEN_PATH).write_bytes(token_payload)
	vector_payload = b"\x80\x3f" + bytes(verifier.VECTOR_BYTES - 2)
	(directory / vector_name).write_bytes(vector_payload)
	document = {
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
			"sha256": hashlib.sha256(
				(ROOT / "tools/dsv4_ga_reference_vector.py").read_bytes()
			).hexdigest(),
		},
		"reference": {
			"model_py_sha256": verifier.REFERENCE_MODEL_SHA256,
			"kernel_py_sha256": verifier.REFERENCE_KERNEL_SHA256,
			"config_sha256": verifier.REFERENCE_CONFIG_SHA256,
			"sparse_attention_fallback": "torch-global-softmax-bf16-probability-v1",
			"hadamard_fallback": "normalized-sylvester-hadamard-v1",
			"fp8_gemm_fallback": "torch-block128-fp32-accumulation-v1",
			"torch": "test", "python": "test", "cuda": "test", "device": "test",
			"device_capability": [12, 1],
		},
		"input": {
			"batch_json_sha256": verifier.REFERENCE_BATCH_JSON_SHA256,
			"source_request_id": 76000,
			"source_sequence_id": 76000,
			"token_artifact": {
				"path": verifier.TOKEN_PATH,
				"sha256": hashlib.sha256(token_payload).hexdigest(),
				"bytes": verifier.TOKEN_BYTES,
				"dtype": "uint32_le",
				"shape": [1, verifier.TOKEN_COUNT],
			},
			"token_ids": [token_ids],
			"positions": [list(range(verifier.TOKEN_COUNT))],
			"validation_sequence_ids": [1],
			"tensor_order": "batch,sequence,hc,hidden",
		},
		"layer_range": {"first": first, "count": count},
		"loaded_parameter_count": 1,
		"loaded_parameter_names_sha256": "6" * 64,
		"consumed_checkpoint_tensor_count": 1,
		"consumed_checkpoint_tensor_names_sha256": "7" * 64,
		"source_shards": [
			{"path": path, "bytes": identity[0], "sha256": identity[1]}
			for path, identity in verifier.SOURCE_SHARDS.items()
		],
		"vectors": [{
			"path": vector_name,
			"sha256": hashlib.sha256(vector_payload).hexdigest(),
			"bytes": verifier.VECTOR_BYTES,
			"shape": [1, verifier.TOKEN_COUNT, verifier.HC_STREAM_COUNT, verifier.HIDDEN_DIMENSION],
			"dtype": "torch.bfloat16",
			"min": 0.0, "max": 1.0,
			"mean": 1.0 / verifier.VECTOR_ELEMENTS,
			"l2": 1.0,
			"nonfinite": 0,
		}],
	}
	manifest_path = directory / "manifest.json"
	manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return document, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def expect_fixture_error(action, label: str) -> None:
	try:
		action()
	except verifier.FixtureError:
		return
	raise SystemExit(f"verifier accepted {label}")


def check_verifier_range_support() -> None:
	with tempfile.TemporaryDirectory(prefix="dsv4-range-") as temporary:
		fixture = Path(temporary)
		_, manifest_sha256 = build_range_fixture(fixture, 3, 11)
		verifier.verify_fixture(
			fixture, manifest_sha256, expect_layer_first=3, expect_layer_count=11)
		verifier.verify_fixture(fixture, manifest_sha256)
		expect_fixture_error(
			lambda: verifier.verify_fixture(
				fixture, manifest_sha256, expect_layer_first=0),
			"wrong expected first layer",
		)
		expect_fixture_error(
			lambda: verifier.verify_fixture(
				fixture, manifest_sha256, expect_layer_count=3),
			"wrong expected layer count",
		)
		(fixture / "after_layer_13.bf16le").unlink()
		expect_fixture_error(
			lambda: verifier.verify_fixture(fixture, manifest_sha256),
			"missing range boundary artifact",
		)


def main() -> int:
	check_head_scale_seeding()
	check_mtp_seed_placement()
	check_mtp_variant_chain()
	check_head_scale_seed_semantics()
	check_pro_variant_seam()
	check_oracle_gate_wiring()
	check_verifier_range_support()
	print("PASS DSV4 head scale seeding and slice-independent oracle gate")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
