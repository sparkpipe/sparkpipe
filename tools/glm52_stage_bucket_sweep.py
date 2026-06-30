#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


HIDDEN_DIMENSION = 6144
BF16_BYTES = 2
DEFAULT_BUCKETS = (8, 16, 32, 64)
DEFAULT_STAGES = ("11:8", "19:8", "27:8", "35:8", "43:8", "51:8", "59:8", "67:8", "75:3")
PASS_RE = re.compile(
    r"routed_pipeline_from_hidden=1.*?first_routed_layer=(?P<first>\d+).*?"
    r"routed_chain_layers=(?P<count>\d+).*?total_submissions=(?P<submissions>\d+).*?"
    r"total_us=(?P<total>[0-9.]+).*?maximum_us=(?P<maximum>[0-9.]+).*?"
    r"limit_us=(?P<limit>[0-9.]+).*?graph_captures=(?P<captures>\d+).*?"
    r"graph_replays=(?P<replays>\d+)"
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
    if first < 3 or count <= 0 or count > 8:
        raise SweepFailure(f"invalid routed stage {text}")
    return first, count


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


def parse_result(log_text: str) -> Dict[str, Any]:
    match = None
    for candidate in PASS_RE.finditer(log_text):
        match = candidate
    if match is None:
        raise SweepFailure("validator did not emit routed_pipeline_from_hidden timing")
    total_us = float(match.group("total"))
    maximum_us = float(match.group("maximum"))
    submissions = int(match.group("submissions"))
    if total_us <= 0.0 or maximum_us <= 0.0 or submissions <= 0:
        raise SweepFailure("validator timing output was invalid")
    return {
        "first_layer": int(match.group("first")),
        "layer_count": int(match.group("count")),
        "submissions": submissions,
        "total_us": total_us,
        "maximum_us": maximum_us,
        "limit_us": float(match.group("limit")),
        "graph_captures": int(match.group("captures")),
        "graph_replays": int(match.group("replays")),
    }


def build_command(
    args: argparse.Namespace,
    batch: int,
    first_layer: int,
    layer_count: int,
    input_hidden: Path,
    output_hidden: Path,
) -> List[str]:
    command = [
        "make",
        "glm52_resident_decode_stage_firmware_package",
        f"MAX_STAGE_MICROSECONDS={args.max_stage_us}",
        "GLM52_VALIDATION_MODE=routed_from_hidden",
        f"GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT={batch}",
        f"GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX={first_layer}",
        f"GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT={layer_count}",
        f"GLM52_PIPELINE_INPUT_HIDDEN_BF16={input_hidden}",
        f"GLM52_PIPELINE_OUTPUT_HIDDEN_BF16={output_hidden}",
        f"GLM52_ENABLE_CUDA_GRAPH_REPLAY={1 if args.graph else 0}",
    ]
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


def run_one(
    root: Path,
    args: argparse.Namespace,
    batch: int,
    first_layer: int,
    layer_count: int,
    input_hidden: Path,
) -> Dict[str, Any]:
    output_hidden = args.output_dir / f"output_hidden_f{first_layer}_c{layer_count}_b{batch}.bf16"
    command = build_command(args, batch, first_layer, layer_count, input_hidden, output_hidden)
    env = os.environ.copy()
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
        "graph_requested": bool(args.graph),
        "command": command,
        "returncode": completed.returncode,
        "log_path": str(args.output_dir / f"log_f{first_layer}_c{layer_count}_b{batch}.txt"),
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


def write_reports(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    json_path = output_dir / "glm52_stage_bucket_sweep.json"
    tsv_path = output_dir / "glm52_stage_bucket_sweep.tsv"
    json_path.write_text(json.dumps({"records": records}, indent=2, sort_keys=True), encoding="utf-8")
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
    parser.add_argument("--aot-output-dir", default=os.environ.get("B12X_AOT_OUTPUT_DIR", ""))
    parser.add_argument("--b12x-moe-pack-dir", default=os.environ.get("B12X_MOE_PACK_OUTPUT_DIR", ""))
    parser.add_argument("--b12x-moe-pack-layers", default=os.environ.get("B12X_MOE_PACK_LAYERS", ""))
    parser.add_argument("--allow-pack-build", action="store_true")
    parser.add_argument("--require-pack-reuse", action="store_true")
    parser.add_argument("--verify-reused-sha256", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_pack_build and args.require_pack_reuse:
        raise SweepFailure("--allow-pack-build conflicts with --require-pack-reuse")

    root = repo_root()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_hidden.exists():
        raise SweepFailure(f"missing input hidden file: {args.input_hidden}")
    buckets = parse_csv_u32(args.buckets)
    stages = parse_stages(args.stage if args.stage is not None else DEFAULT_STAGES)
    records: List[Dict[str, Any]] = []
    for batch in buckets:
        input_hidden = ensure_batch_hidden(args.input_hidden, args.output_dir, batch)
        for first_layer, layer_count in stages:
            record = run_one(root, args, batch, first_layer, layer_count, input_hidden)
            records.append(record)
            print(
                f"bucket={batch} stage={first_layer}:{layer_count} "
                f"status={record['status']} tok_s={record.get('filled_pipeline_tok_s', '')} "
                f"log={record['log_path']}",
                flush=True,
            )
            if record["status"] != "pass" and not args.keep_going:
                write_reports(args.output_dir, records)
                return 1
    write_reports(args.output_dir, records)
    return 0 if all(record["status"] == "pass" for record in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SweepFailure as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
