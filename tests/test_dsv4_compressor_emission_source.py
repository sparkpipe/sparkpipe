#!/usr/bin/env python3
"""Fail closed on DSV4's device-predicated compressor emission path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"
MODULE = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c"
COMMON = ROOT / "model-families/common/include/sparkpipe/spark_lm_kernels.cuh"
PROBE = ROOT / "tools/hardware/spark_dsv4_compressor_emission_bitwise.cu"
MAKEFILE = ROOT / "Makefile"


def body(source: str, name: str) -> str:
	start = source.index(name)
	brace = source.index("{", start)
	depth = 1
	index = brace + 1
	while depth != 0:
		if source[index] == "{":
			depth += 1
		elif source[index] == "}":
			depth -= 1
		index += 1
	return source[brace:index]


def ordered(source: str, names: tuple[str, ...]) -> None:
	positions = [source.index(name) for name in names]
	if positions != sorted(positions):
		raise AssertionError(f"operation order changed: {names}")


def main() -> int:
	cuda = CUDA.read_text(encoding="utf-8")
	module = MODULE.read_text(encoding="utf-8")
	common = COMMON.read_text(encoding="utf-8")
	probe = PROBE.read_text(encoding="utf-8")
	makefile = MAKEFILE.read_text(encoding="utf-8")
	post = body(module, "SparkDsv4ModuleRunCompressorPost(")
	if post.count("SparkDsv4LaunchKvEmission(") != 1:
		raise AssertionError("compressor post must launch one fused emission kernel")
	for legacy in ("SparkDsv4LaunchRmsNorm(", "SparkDsv4LaunchRope(",
		"SparkDsv4LaunchHadamard(", "SparkDsv4LaunchQuantSim(",
		"SparkDsv4LaunchCacheScatter("):
		if legacy in post:
			raise AssertionError(f"legacy post-emission launch remains: {legacy}")
	fused = body(cuda, "SparkDsv4KvEmissionKernel(")
	predicate = fused.index("ratio != 0u && emitted[row] == 0u")
	if predicate > fused.index("SparkLmRmsNormRow("):
		raise AssertionError("boundary predicate does not guard emission work")
	ordered(fused,("SparkLmRmsNormRow(","SparkDsv4RopeRow(",
		"SparkDsv4HadamardRow(","SparkDsv4QuantSimGroup(",
		"SparkDsv4CacheScatterRow("))
	if "__syncthreads();" not in fused:
		raise AssertionError("fused BF16 stages lack device ordering barriers")
	if "SparkLmRmsNormRow(" not in body(common,"SparkLmRmsNormKernel("):
		raise AssertionError("standalone and fused RMS paths do not share math")
	for kernel,helper in (("SparkDsv4RopeKernel(","SparkDsv4RopeRow("),
		("SparkDsv4HadamardKernel(","SparkDsv4HadamardRow("),
		("SparkDsv4QuantSimKernel(","SparkDsv4QuantSimGroup("),
		("SparkDsv4CacheScatterKernel(","SparkDsv4CacheScatterRow(")):
		if helper not in body(cuda,kernel):
			raise AssertionError(f"standalone {kernel} does not share {helper}")
	for case in ("swa-always","csa-attention","csa-index","hca-attention"):
		if case not in probe:
			raise AssertionError(f"bitwise probe missing {case}")
	if probe.count("memcmp(") != 2:
		raise AssertionError("bitwise probe must compare emission and cache bytes")
	if "build/spark_dsv4_compressor_emission_bitwise:" not in makefile:
		raise AssertionError("bitwise CUDA probe has no build target")
	print("PASS DSV4 boundary-predicated compressor emission source contract")
	return(0)


if __name__ == "__main__":
	raise SystemExit(main())
