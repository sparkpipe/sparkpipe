#!/usr/bin/env python3
"""Fail closed on the exact row-generic DSV4 indexer post fusion."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu"
MODULE = ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c"
VALIDATOR = ROOT / "modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu"


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


def require(source: str, needle: str, message: str) -> None:
	if needle not in source:
		raise AssertionError(f"missing {message}: {needle}")


def forbid(source: str, needle: str, message: str) -> None:
	if needle in source:
		raise AssertionError(f"forbidden {message}: {needle}")


def main() -> int:
	cuda = CUDA.read_text(encoding="utf-8")
	module = MODULE.read_text(encoding="utf-8")
	validator = VALIDATOR.read_text(encoding="utf-8")
	kernel = body(cuda, "SparkDsv4IndexerPostKernel(")
	rope = kernel.index("SparkDsv4RopeRow(")
	first_boundary = kernel.index("__syncthreads();", rope)
	hadamard = kernel.index("SparkDsv4HadamardRow(", first_boundary)
	second_boundary = kernel.index("__syncthreads();", hadamard)
	quant = kernel.index("SparkDsv4QuantSimGroup(", second_boundary)
	if not rope < first_boundary < hadamard < second_boundary < quant:
		raise AssertionError("indexer fusion lost a BF16 store/reload boundary")
	require(kernel, "row = blockIdx.x,head = blockIdx.y",
		"row-generic B1-B1024 grid")
	require(kernel, "SPARK_DSV4_MODEL_FP4_MAX,1u",
		"checkpoint-declared FP4 quantization simulation")
	require(cuda, "uint32_t rope_dim,uint32_t quant_block,uint32_t inverse)",
		"runtime RoPE-direction kernel contract")
	require(kernel, "head_dim,rope_dim,inverse",
		"runtime RoPE direction reaches the shared primitive")
	forbid(kernel, "head_dim,rope_dim,0u",
		"constant-folded RoPE direction")
	launcher = body(cuda, "SparkDsv4LaunchIndexerPost(")
	require(launcher, "SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT",
		"B1-B1024 row gate")
	require(launcher, "dim3 grid(row_count,head_count)",
		"single row-by-head launch geometry")
	require(launcher, "uint32_t inverse = 0u;",
		"forward direction passed through the kernel ABI")
	require(launcher, "quant_block,\n\t\tinverse",
		"runtime direction launch argument")
	forbid(launcher, "#if", "compile-time feature fork")
	production = body(module, "SparkDsv4ModuleRunIndexerCore(")
	require(production, "SparkDsv4LaunchIndexerPost(",
		"one-launch production indexer post-processing")
	for legacy in ("SparkDsv4LaunchRope(", "SparkDsv4LaunchHadamard(",
		"SparkDsv4LaunchQuantSim("):
		forbid(production, legacy, "sequential production indexer post launch")
	case = body(validator, "SparkDsv4ValidationIndexerPostCase(")
	for call in ("SparkDsv4LaunchRope(", "SparkDsv4LaunchHadamard(",
		"SparkDsv4LaunchQuantSim(", "SparkDsv4LaunchIndexerPost("):
		require(case, call, "byte-exact hardware comparison")
	require(validator, "seeds[] = {1u,9u,53u}",
		"known-sensitive deterministic hardware seeds")
	require(validator, "262144u", "long-position range-reduction fixture")
	require(validator, "SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA",
		"GA compressed-frequency table")
	require(body(validator, "SparkDsv4ValidationPostFusions("),
		"SparkDsv4ValidationIndexerPostCase(&buffers,",
		"mandatory indexer post hardware cases")
	print("PASS DSV4 exact row-generic indexer post fusion source contract")
	return(0)


if __name__ == "__main__":
	raise SystemExit(main())
