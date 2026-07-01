#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


HIDDEN_DIMENSION = 6144
BF16_BYTES = 2
LAYER_COUNT = 78
FIRST_ROUTED_LAYER = 3
MAX_ROUTED_LAYERS_PER_STAGE = 8
DEFAULT_BUCKETS = (8, 16, 32, 64)
DEFAULT_STAGES = ("11:8", "19:8", "27:8", "35:8", "43:8", "51:8", "59:8", "67:8", "75:3")
DEFAULT_AOT_OUTPUT_DIR = Path("build/glm52_b12x_aot")
DEFAULT_MODULE_ARCHIVE = Path("build/modules/glm52_resident_decode_stage/libglm52_resident_decode_stage.a")
DEFAULT_DRIVER_SO = ""
DEFAULT_B12X_ADAPTER_ARCHIVE = Path("build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a")
DEFAULT_B12X_BACKEND_ARCHIVE = Path("build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a")
DEFAULT_B12X_KERNEL_TABLE_ARCHIVE = Path("build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a")
ROUTED_PASS_RE = re.compile(
    r"routed_pipeline_from_hidden=1.*?first_routed_layer=(?P<first>\d+).*?"
    r"routed_chain_layers=(?P<count>\d+).*?total_submissions=(?P<submissions>\d+).*?"
    r"total_us=(?P<total>[0-9.]+).*?maximum_us=(?P<maximum>[0-9.]+).*?"
    r"limit_us=(?P<limit>[0-9.]+).*?graph_captures=(?P<captures>\d+).*?"
    r"graph_replays=(?P<replays>\d+)"
)
DENSE_PREFIX_PASS_RE = re.compile(
    r"dense_prefix_routed_pipeline=1.*?dense_chain_layers=(?P<dense>\d+).*?"
    r"first_routed_layer=(?P<first>\d+).*?routed_chain_layers=(?P<count>\d+).*?"
    r"total_submissions=(?P<submissions>\d+).*?total_us=(?P<total>[0-9.]+).*?"
    r"maximum_us=(?P<maximum>[0-9.]+).*?limit_us=(?P<limit>[0-9.]+).*?"
    r"graph_captures=(?P<captures>\d+).*?graph_replays=(?P<replays>\d+)"
)
DENSE_CHAIN_PASS_RE = re.compile(
    r"dense_chain_layers=(?P<dense>\d+).*?"
    r"dense_chain_submissions=(?P<submissions>\d+).*?"
    r"dense_chain_total_us=(?P<total>[0-9.]+).*?"
    r"maximum_us=(?P<maximum>[0-9.]+).*?limit_us=(?P<limit>[0-9.]+).*?"
    r"graph_captures=(?P<captures>\d+).*?graph_replays=(?P<replays>\d+)"
)


class SweepFailure(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_csv_u32(text: str) -> List[int]:
    values: List[int] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise SweepFailure(f"invalid positive integer: {item}")
        values.append(value)
    if not values:
        raise SweepFailure("empty integer list")
    return values


def parse_stage(text: str) -> Tuple[int, int]:
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise SweepFailure(f"stage must be FIRST:COUNT, got {text}")
    first = int(parts[0])
    count = int(parts[1])
    stage_validator_parameters(first, count)
    return first, count


def stage_validator_parameters(first_layer: int, layer_count: int) -> Tuple[int, int, bool]:
    if layer_count <= 0:
        raise SweepFailure(f"invalid stage {first_layer}:{layer_count}")
    if first_layer < 0 or first_layer >= LAYER_COUNT:
        raise SweepFailure(f"invalid stage {first_layer}:{layer_count}")
    if layer_count > LAYER_COUNT - first_layer:
        raise SweepFailure(f"stage exceeds GLM52 layer count {first_layer}:{layer_count}")
    if first_layer == 0:
        routed_count = layer_count - FIRST_ROUTED_LAYER
        if routed_count < 0 or routed_count > MAX_ROUTED_LAYERS_PER_STAGE:
            raise SweepFailure(f"invalid dense-prefix stage {first_layer}:{layer_count}")
        return FIRST_ROUTED_LAYER, routed_count, True
    if first_layer < FIRST_ROUTED_LAYER or layer_count > MAX_ROUTED_LAYERS_PER_STAGE:
        raise SweepFailure(f"invalid routed stage {first_layer}:{layer_count}")
    return first_layer, layer_count, False


def parse_stages(values: Sequence[str]) -> List[Tuple[int, int]]:
    result = [parse_stage(value) for value in values]
    if not result:
        raise SweepFailure("at least one stage is required")
    return result


def ensure_batch_hidden(base_hidden: Path, output_dir: Path, batch: int) -> Path:
    vector_bytes = HIDDEN_DIMENSION * BF16_BYTES
    target_bytes = vector_bytes * batch
    payload = base_hidden.read_bytes()
    if len(payload) == target_bytes:
        return base_hidden
    if len(payload) != vector_bytes:
        raise SweepFailure(
            f"{base_hidden} must contain either one hidden vector ({vector_bytes} bytes) "
            f"or B hidden vectors ({target_bytes} bytes)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"input_hidden_b{batch}.bf16"
    output_path.write_bytes(payload * batch)
    return output_path


def resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SweepFailure(f"missing {label}: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SweepFailure(f"missing {label}: {path}")


def read_runtime_link_args(path: Path) -> List[str]:
    require_nonempty_file(path, "B12x runtime link args")
    return shlex.split(path.read_text(encoding="utf-8"))


def resolve_link_arg_file(root: Path, link_arg: str) -> Path | None:
    if link_arg.startswith("-"):
        return None
    candidate = Path(link_arg)
    if candidate.suffix not in {".a", ".so", ".o"}:
        return None
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def required_cuda_link_args(root: Path, args: argparse.Namespace) -> List[str]:
    if args.required_cuda_link_args:
        return shlex.split(args.required_cuda_link_args)
    aot_output_dir = resolve_path(root, Path(args.aot_output_dir))
    archive_paths = [
        resolve_path(root, DEFAULT_B12X_ADAPTER_ARCHIVE),
        resolve_path(root, DEFAULT_B12X_BACKEND_ARCHIVE),
        resolve_path(root, DEFAULT_B12X_KERNEL_TABLE_ARCHIVE),
    ]
    for archive_path in archive_paths:
        require_nonempty_file(archive_path, "B12x CUDA archive")
    runtime_link_args = read_runtime_link_args(
        aot_output_dir / "generated" / "runtime_link_args.txt"
    )
    return [str(archive_path) for archive_path in archive_paths] + runtime_link_args


def library_path_from_link_args(root: Path, link_args: Sequence[str]) -> str:
    directories: List[str] = []
    seen: set[str] = set()
    for link_arg in link_args:
        candidate = resolve_link_arg_file(root, link_arg)
        if candidate is None or candidate.suffix != ".so":
            continue
        directory = str(candidate.parent)
        if directory not in seen:
            seen.add(directory)
            directories.append(directory)
    return ":".join(directories)


def validator_inputs(
    root: Path,
    args: argparse.Namespace,
    required_link_args: Sequence[str],
) -> List[Path]:
    inputs = [
        root / "modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu",
        args.module_archive,
        root / "build/libsparkpipe_common.a",
        root / "build/libsparkpipe_runtime.a",
        root / "build/libsparkpipe_compiler.a",
    ]
    for link_arg in required_link_args:
        candidate = resolve_link_arg_file(root, link_arg)
        if candidate is not None:
            inputs.append(candidate)
    return inputs


def validator_metadata(
    compile_command: Sequence[str],
    input_paths: Sequence[Path],
) -> Dict[str, Any]:
    return {
        "compile_command": list(compile_command),
        "inputs": [
            {
                "path": str(input_path),
                "mtime_ns": input_path.stat().st_mtime_ns,
                "size": input_path.stat().st_size,
            }
            for input_path in input_paths
        ],
    }


def validator_is_current(
    validator_path: Path,
    metadata_path: Path,
    expected_metadata: Dict[str, Any],
) -> bool:
    if not validator_path.is_file() or validator_path.stat().st_size == 0:
        return False
    if not metadata_path.is_file():
        return False
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return actual_metadata == expected_metadata


def ensure_cached_validator(
    root: Path,
    args: argparse.Namespace,
    batch: int,
    required_link_args: Sequence[str],
) -> Path:
    cache_dir = resolve_path(root, args.validator_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    validator_path = cache_dir / f"glm52_resident_decode_stage_validator_b{batch}"
    metadata_path = validator_path.with_suffix(".json")
    make_command = [
        "make",
        "build/libsparkpipe_common.a",
        "build/libsparkpipe_runtime.a",
        "build/libsparkpipe_compiler.a",
    ]
    make_result = subprocess.run(
        make_command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if make_result.returncode != 0:
        raise SweepFailure(
            "failed to build base SparkPipe archives for validator:\n"
            + make_result.stdout
            + make_result.stderr
        )
    input_paths = validator_inputs(root, args, required_link_args)
    for input_path in input_paths:
        require_nonempty_file(input_path, "validator input")
    module_dir = root / "modules/glm52_resident_decode_stage"
    compile_command = [
        args.nvcc,
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        f"-arch={args.cuda_arch}",
        f"-DSPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT={batch}u",
        f"-I{root / 'include'}",
        f"-I{module_dir / 'include'}",
        f"-I{module_dir / 'source'}",
        str(module_dir / "validation/spark_glm52_resident_decode_stage_cuda_validation.cu"),
        str(args.module_archive),
        str(root / "build/libsparkpipe_runtime.a"),
        str(root / "build/libsparkpipe_compiler.a"),
        str(root / "build/libsparkpipe_common.a"),
        *required_link_args,
        "-lcublasLt",
        "-lcublas",
        "-ldl",
        "-o",
        str(validator_path),
    ]
    metadata = validator_metadata(compile_command, input_paths)
    if not args.force_validator_rebuild and validator_is_current(
        validator_path,
        metadata_path,
        metadata,
    ):
        return validator_path
    compile_result = subprocess.run(
        compile_command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    compile_log = cache_dir / f"glm52_resident_decode_stage_validator_b{batch}.build.log"
    compile_log.write_text(compile_result.stdout + compile_result.stderr, encoding="utf-8")
    if compile_result.returncode != 0:
        raise SweepFailure(f"validator compile failed; see {compile_log}")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return validator_path


def direct_validator_environment(
    root: Path,
    args: argparse.Namespace,
    batch: int,
    first_layer: int,
    layer_count: int,
    input_hidden: Path,
    output_hidden: Path,
) -> Dict[str, str]:
    env = os.environ.copy()
    routed_first_layer, routed_layer_count, dense_prefix = stage_validator_parameters(
        first_layer,
        layer_count,
    )
    library_path = library_path_from_link_args(root, args.required_cuda_link_args_list)
    if library_path:
        env["LD_LIBRARY_PATH"] = (
            library_path
            if not env.get("LD_LIBRARY_PATH")
            else library_path + ":" + env["LD_LIBRARY_PATH"]
        )
    env["GLM52_MODEL_DIR"] = args.model_dir
    env["GLM52_ALLOW_REMOTE_MODEL_DIR"] = os.environ.get("GLM52_ALLOW_REMOTE_MODEL_DIR", "0")
    env["GLM52_B12X_MOE_PACK_DIR"] = str(args.b12x_moe_pack_dir)
    if not (dense_prefix and routed_layer_count == 0):
        env["GLM52_ROUTED_CHAIN_FIRST_LAYER_INDEX"] = str(routed_first_layer)
        env["GLM52_ROUTED_CHAIN_LAYER_COUNT"] = str(routed_layer_count)
    env["GLM52_ENABLE_CUDA_GRAPH_REPLAY"] = "1" if args.graph else "0"
    env["GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT"] = str(batch)
    env["GLM52_PRODUCTION_TIMING"] = "0" if args.validation_timing else "1"
    if dense_prefix:
        env["GLM52_INPUT_TOKEN_ID"] = os.environ.get("GLM52_INPUT_TOKEN_ID", "1000")
        if routed_layer_count == 0:
            env["GLM52_CHAIN_DENSE_LAYERS"] = "1"
            if not args.validation_timing:
                env["GLM52_DENSE_CHAIN_CURRENT_TOKEN_ONLY"] = "1"
        else:
            env["GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK"] = "1"
            if not args.validation_timing:
                env["GLM52_DENSE_PREFIX_CURRENT_TOKEN_ONLY"] = "1"
    else:
        env["GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16"] = "1"
        env["GLM52_PIPELINE_INPUT_HIDDEN_BF16"] = str(input_hidden)
    env["GLM52_PIPELINE_OUTPUT_HIDDEN_BF16"] = str(output_hidden)
    env["GLM52_REQUIRED_CUDA_LINK_ARGS"] = " ".join(args.required_cuda_link_args_list)
    return env


def parse_result(log_text: str) -> Dict[str, Any]:
    match = None
    dense_prefix = False
    for candidate in ROUTED_PASS_RE.finditer(log_text):
        match = candidate
    for candidate in DENSE_PREFIX_PASS_RE.finditer(log_text):
        match = candidate
        dense_prefix = True
    for candidate in DENSE_CHAIN_PASS_RE.finditer(log_text):
        match = candidate
        dense_prefix = True
    if match is None:
        raise SweepFailure("validator did not emit stage timing")
    total_us = float(match.group("total"))
    maximum_us = float(match.group("maximum"))
    submissions = int(match.group("submissions"))
    if total_us <= 0.0 or maximum_us <= 0.0 or submissions <= 0:
        raise SweepFailure("validator timing output was invalid")
    if dense_prefix:
        dense_count = int(match.group("dense"))
        first_layer = 0
        routed_count = int(match.groupdict().get("count", 0) or 0)
        layer_count = dense_count + routed_count
    else:
        first_layer = int(match.group("first"))
        layer_count = int(match.group("count"))
    return {
        "first_layer": first_layer,
        "layer_count": layer_count,
        "submissions": submissions,
        "total_us": total_us,
        "maximum_us": maximum_us,
        "limit_us": float(match.group("limit")),
        "graph_captures": int(match.group("captures")),
        "graph_replays": int(match.group("replays")),
    }


def build_package_command(
    args: argparse.Namespace,
    batch: int,
    first_layer: int,
    layer_count: int,
    input_hidden: Path,
    output_hidden: Path,
) -> List[str]:
    routed_first_layer, routed_layer_count, dense_prefix = stage_validator_parameters(
        first_layer,
        layer_count,
    )
    command = [
        "make",
        "glm52_resident_decode_stage_firmware_package",
        f"MAX_STAGE_MICROSECONDS={args.max_stage_us}",
        (
            "GLM52_VALIDATION_MODE=dense_to_layer3_routed"
            if dense_prefix
            else "GLM52_VALIDATION_MODE=routed_from_hidden"
        ),
        f"GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT={batch}",
        f"GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX={routed_first_layer}",
        f"GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT={routed_layer_count}",
        f"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16={output_hidden}",
        f"GLM52_ENABLE_CUDA_GRAPH_REPLAY={1 if args.graph else 0}",
    ]
    if dense_prefix:
        command.append(f"GLM52_VALIDATION_INPUT_TOKEN_ID={os.environ.get('GLM52_INPUT_TOKEN_ID', '1000')}")
    else:
        command.append(f"GLM52_PIPELINE_INPUT_HIDDEN_BF16={input_hidden}")
    if args.model_dir:
        command.append(f"GLM52_MODEL_DIR={args.model_dir}")
    if args.cuda_arch:
        command.append(f"CUDA_ARCH={args.cuda_arch}")
    if args.nvcc:
        command.append(f"NVCC={args.nvcc}")
    if args.aot_env:
        command.append(f"SPARKPIPE_B12X_AOT_ENV={args.aot_env}")
    if args.aot_output_dir:
        command.append(f"B12X_AOT_OUTPUT_DIR={args.aot_output_dir}")
    if args.b12x_moe_pack_dir:
        command.append(f"B12X_MOE_PACK_OUTPUT_DIR={args.b12x_moe_pack_dir}")
    if args.b12x_moe_pack_layers:
        command.append(f"B12X_MOE_PACK_LAYERS={args.b12x_moe_pack_layers}")
    command.append(
        "B12X_MOE_PACK_REQUIRE_REUSE=0"
        if args.allow_pack_build
        else "B12X_MOE_PACK_REQUIRE_REUSE=1"
    )
    if args.verify_reused_sha256:
        command.append("B12X_MOE_PACK_VERIFY_REUSED_SHA256=1")
    return command


def build_direct_command(
    validator_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    command = [str(validator_path), str(args.max_stage_us)]
    if args.driver_so:
        command.append(str(args.driver_so))
    return command


def run_one(
    root: Path,
    args: argparse.Namespace,
    batch: int,
    first_layer: int,
    layer_count: int,
    input_hidden: Path,
    attempt_index: int,
    warmup: bool,
) -> Dict[str, Any]:
    attempt_label = f"w{attempt_index}" if warmup else f"r{attempt_index}"
    output_hidden = (
        args.output_dir /
        f"output_hidden_f{first_layer}_c{layer_count}_b{batch}_{attempt_label}.bf16"
    )
    if args.package_each_run:
        command = build_package_command(args, batch, first_layer, layer_count, input_hidden, output_hidden)
        env = os.environ.copy()
        execution_mode = "package"
        validator_path = ""
    else:
        validator = ensure_cached_validator(root, args, batch, args.required_cuda_link_args_list)
        command = build_direct_command(validator, args)
        env = direct_validator_environment(
            root,
            args,
            batch,
            first_layer,
            layer_count,
            input_hidden,
            output_hidden,
        )
        execution_mode = "direct_cached_validator"
        validator_path = str(validator)
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_text = completed.stdout + completed.stderr
    record: Dict[str, Any] = {
        "batch_size": batch,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "attempt_index": attempt_index,
        "warmup": warmup,
        "graph_requested": bool(args.graph),
        "execution_mode": execution_mode,
        "validator_path": validator_path,
        "command": command,
        "returncode": completed.returncode,
        "log_path": str(
            args.output_dir /
            f"log_f{first_layer}_c{layer_count}_b{batch}_{attempt_label}.txt"
        ),
    }
    Path(record["log_path"]).write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        record["status"] = "fail"
        record["error"] = f"validator exited {completed.returncode}"
        return record
    try:
        record.update(parse_result(log_text))
    except SweepFailure as exc:
        record["status"] = "fail"
        record["error"] = str(exc)
        return record
    stage_ms = record["total_us"] / 1000.0
    record["status"] = "pass"
    record["stage_ms"] = stage_ms
    record["per_layer_ms"] = stage_ms / float(record["layer_count"])
    record["filled_pipeline_tok_s"] = (float(batch) * 1000.0) / stage_ms
    return record


def summarize_attempts(
    args: argparse.Namespace,
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    measured = [
        attempt for attempt in attempts
        if attempt.get("status") == "pass" and not attempt.get("warmup", False)
    ]
    if not measured:
        failed = attempts[-1].copy()
        failed["attempt_count"] = len(attempts)
        failed["warmup_runs"] = args.warmup_runs
        failed["measure_runs"] = args.measure_runs
        failed["best_attempt_index"] = ""
        return failed
    best = min(measured, key=lambda item: float(item["total_us"])).copy()
    best["attempt_count"] = len(attempts)
    best["warmup_runs"] = args.warmup_runs
    best["measure_runs"] = args.measure_runs
    best["best_attempt_index"] = best["attempt_index"]
    return best


def write_reports(
    output_dir: Path,
    records: List[Dict[str, Any]],
    attempt_records: List[Dict[str, Any]],
) -> None:
    json_path = output_dir / "glm52_stage_bucket_sweep.json"
    tsv_path = output_dir / "glm52_stage_bucket_sweep.tsv"
    json_path.write_text(
        json.dumps(
            {"records": records, "attempt_records": attempt_records},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    columns = [
        "status",
        "batch_size",
        "first_layer",
        "layer_count",
        "stage_ms",
        "per_layer_ms",
        "filled_pipeline_tok_s",
        "total_us",
        "maximum_us",
        "submissions",
        "graph_captures",
        "graph_replays",
        "attempt_count",
        "warmup_runs",
        "measure_runs",
        "best_attempt_index",
        "execution_mode",
        "validator_path",
        "error",
        "log_path",
    ]
    lines = ["\t".join(columns)]
    for record in records:
        lines.append("\t".join(str(record.get(column, "")) for column in columns))
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Run GLM52 hidden-stage bucket throughput sweep")
    parser.add_argument("--buckets", default=",".join(str(item) for item in DEFAULT_BUCKETS))
    parser.add_argument("--stage", action="append", default=None, help="Routed stage as FIRST:COUNT; repeatable")
    parser.add_argument("--input-hidden", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/glm52_stage_bucket_sweep"), type=Path)
    parser.add_argument("--max-stage-us", default="1000000")
    parser.add_argument("--model-dir", default=os.environ.get("GLM52_MODEL_DIR", ""))
    parser.add_argument("--cuda-arch", default="sm_121a")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--aot-env", default=os.environ.get("SPARKPIPE_B12X_AOT_ENV", ""))
    parser.add_argument(
        "--aot-output-dir",
        default=os.environ.get("B12X_AOT_OUTPUT_DIR", str(DEFAULT_AOT_OUTPUT_DIR)),
    )
    parser.add_argument("--b12x-moe-pack-dir", default=os.environ.get("B12X_MOE_PACK_OUTPUT_DIR", ""))
    parser.add_argument("--b12x-moe-pack-layers", default=os.environ.get("B12X_MOE_PACK_LAYERS", ""))
    parser.add_argument("--module-archive", default=DEFAULT_MODULE_ARCHIVE, type=Path)
    parser.add_argument("--driver-so", default=os.environ.get("GLM52_STAGE_SWEEP_DRIVER_SO", DEFAULT_DRIVER_SO))
    parser.add_argument("--validator-cache-dir", default=None, type=Path)
    parser.add_argument("--required-cuda-link-args", default=os.environ.get("GLM52_REQUIRED_CUDA_LINK_ARGS", ""))
    parser.add_argument("--force-validator-rebuild", action="store_true")
    parser.add_argument("--package-each-run", action="store_true")
    parser.add_argument("--allow-pack-build", action="store_true")
    parser.add_argument("--require-pack-reuse", action="store_true")
    parser.add_argument("--verify-reused-sha256", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--validation-timing",
        action="store_true",
        help="Measure legacy validation/reference timing instead of production decode timing",
    )
    parser.add_argument("--warmup-runs", default=0, type=int)
    parser.add_argument("--measure-runs", default=1, type=int)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_pack_build and args.require_pack_reuse:
        raise SweepFailure("--allow-pack-build conflicts with --require-pack-reuse")

    root = repo_root()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validator_cache_dir is None:
        args.validator_cache_dir = args.output_dir / "validators"
    args.module_archive = resolve_path(root, args.module_archive)
    if args.driver_so:
        args.driver_so = resolve_path(root, Path(args.driver_so))
    else:
        args.driver_so = None
    args.input_hidden = resolve_path(root, args.input_hidden)
    if args.b12x_moe_pack_dir:
        args.b12x_moe_pack_dir = resolve_path(root, Path(args.b12x_moe_pack_dir))
    if not args.package_each_run:
        require_nonempty_file(args.module_archive, "resident decode-stage module archive")
        if args.driver_so is not None:
            require_nonempty_file(args.driver_so, "resident decode-stage driver shared object")
        if not args.model_dir:
            raise SweepFailure("set --model-dir or GLM52_MODEL_DIR for direct cached validation")
        if not args.b12x_moe_pack_dir:
            raise SweepFailure("set --b12x-moe-pack-dir for direct cached validation")
        require_directory(Path(args.b12x_moe_pack_dir), "B12x MoE pack directory")
        args.required_cuda_link_args_list = required_cuda_link_args(root, args)
    else:
        args.required_cuda_link_args_list = []
    if not args.input_hidden.exists():
        raise SweepFailure(f"missing input hidden file: {args.input_hidden}")
    if args.warmup_runs < 0 or args.measure_runs <= 0:
        raise SweepFailure("--warmup-runs must be non-negative and --measure-runs must be positive")
    buckets = parse_csv_u32(args.buckets)
    stages = parse_stages(args.stage if args.stage is not None else DEFAULT_STAGES)
    records: List[Dict[str, Any]] = []
    attempt_records: List[Dict[str, Any]] = []
    for batch in buckets:
        input_hidden = ensure_batch_hidden(args.input_hidden, args.output_dir, batch)
        for first_layer, layer_count in stages:
            attempts: List[Dict[str, Any]] = []
            for attempt_index in range(args.warmup_runs):
                attempt = run_one(
                    root,
                    args,
                    batch,
                    first_layer,
                    layer_count,
                    input_hidden,
                    attempt_index,
                    True,
                )
                attempts.append(attempt)
                attempt_records.append(attempt)
                print(
                    f"bucket={batch} stage={first_layer}:{layer_count} "
                    f"warmup={attempt_index} status={attempt['status']} "
                    f"tok_s={attempt.get('filled_pipeline_tok_s', '')} "
                    f"log={attempt['log_path']}",
                    flush=True,
                )
                if attempt["status"] != "pass" and not args.keep_going:
                    record = summarize_attempts(args, attempts)
                    records.append(record)
                    write_reports(args.output_dir, records, attempt_records)
                    return 1
            for attempt_index in range(args.measure_runs):
                attempt = run_one(
                    root,
                    args,
                    batch,
                    first_layer,
                    layer_count,
                    input_hidden,
                    attempt_index,
                    False,
                )
                attempts.append(attempt)
                attempt_records.append(attempt)
                print(
                    f"bucket={batch} stage={first_layer}:{layer_count} "
                    f"measure={attempt_index} status={attempt['status']} "
                    f"tok_s={attempt.get('filled_pipeline_tok_s', '')} "
                    f"log={attempt['log_path']}",
                    flush=True,
                )
                if attempt["status"] != "pass" and not args.keep_going:
                    break
            record = summarize_attempts(args, attempts)
            records.append(record)
            print(
                f"bucket={batch} stage={first_layer}:{layer_count} "
                f"status={record['status']} tok_s={record.get('filled_pipeline_tok_s', '')} "
                f"log={record['log_path']}",
                flush=True,
            )
            if record["status"] != "pass" and not args.keep_going:
                write_reports(args.output_dir, records, attempt_records)
                return 1
    write_reports(args.output_dir, records, attempt_records)
    return 0 if all(record["status"] == "pass" for record in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SweepFailure as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
