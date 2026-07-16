#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def load_tool(repository: Path):
    path = repository / "tools" / "glm52_nvfp4_artifact_preflight.py"
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(
        "glm52_nvfp4_artifact_preflight",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load NVFP4 artifact preflight")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def write_at(file_descriptor: int, offset: int, payload: bytes) -> None:
    written = os.pwrite(file_descriptor, payload, offset)
    if written != len(payload):
        raise RuntimeError("short sparse fixture write")


def create_stagepack(
    tool,
    root: Path,
    rank: int,
    layer: int,
    source_hash: str,
) -> None:
    common = tool.stagepack
    file_name = f"stage_{rank:02d}_non_moe.spstage"
    path = root / file_name
    tensor_map = {}
    offset = 0
    selected = set(common.required_layer_tensors(layer))
    for name, (dtype, shape) in sorted(
        common.required_stage_tensors(rank).items()
    ):
        offset = common.align_up(offset, common.STAGE_REGION_ALIGNMENT)
        byte_count = common.tensor_bytes(dtype, shape)
        tensor_map[name] = {
            "file": file_name,
            "offset": offset,
            "bytes": byte_count,
            "dtype": dtype,
            "shape": list(shape),
            "source_shard": "fixture.safetensors",
        }
        offset += byte_count
    file_descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(file_descriptor, offset)
        for name in selected:
            record = tensor_map[name]
            width = min(record["bytes"], 64)
            for sample_index, relative_offset in enumerate(
                common.sample_offsets(record["bytes"], 64)
            ):
                payload = bytes(
                    ((value + sample_index + len(name)) % 251) + 1
                    for value in range(width)
                )
                write_at(
                    file_descriptor,
                    record["offset"] + relative_offset,
                    payload,
                )
    finally:
        os.close(file_descriptor)
    index = {
        "format": common.STAGEPACK_FORMAT,
        "topology": common.STAGEPACK_TOPOLOGY,
        "model_quantization": tool.NVFP4_QUANTIZATION,
        "non_expert_weight_dtype": common.STAGEPACK_DTYPE,
        "stage_count": common.STAGE_COUNT,
        "layers_per_stage": common.LAYERS_PER_STAGE,
        "source_model_index_sha256": source_hash,
        "stages": {
            str(rank): {
                "file": file_name,
                "first_layer": rank * common.LAYERS_PER_STAGE,
                "layer_count": common.LAYERS_PER_STAGE,
                "tensor_count": len(tensor_map),
            },
        },
        "tensor_map": tensor_map,
    }
    (root / common.STAGEPACK_INDEX).write_text(
        json.dumps(index),
        encoding="utf-8",
    )


def create_aot_bundle(tool, generated: Path, maximum_tokens: int) -> Path:
    objects = generated / "objects"
    objects.mkdir(parents=True)
    object_name = "spark_glm52_b12x_static_t1024_fixture.o"
    object_path = objects / object_name
    object_path.write_bytes(b"fixture-aot-object")
    function_name = "spark_glm52_b12x_static_t1024_fixture"
    manifest = {
        "record_schema": tool.AOT_SCHEMA,
        "required_module": tool.b12x.REQUIRED_MODULE,
        "required_arch": tool.b12x.REQUIRED_ARCH,
        "runtime_language": "c_cuda_tvm_ffi",
        "compile_time_languages": [
            "python",
            "torch",
            "flashinfer",
            "cutlass_cute_dsl",
        ],
        "fallback_allowed": False,
        "runtime_backend_selection": "forbidden",
        "production_backend_policy": "exact_static_buckets_only",
        "runtime_bucket_decomposition": "forbidden",
        "runtime_diagnostic_routing_mutation": "forbidden",
        "deterministic_fc2_finalize": True,
        "route_scatter_output": True,
        "route_slice_output": True,
        "shape": {
            "hidden_dimension": tool.b12x.HIDDEN_DIMENSION,
            "intermediate_dimension": tool.b12x.INTERMEDIATE_DIMENSION,
            "expert_count": tool.b12x.EXPERT_COUNT,
            "top_k": tool.b12x.TOP_K,
            "fused_w1_rows": (
                tool.b12x.W1_COMPONENT_COUNT *
                tool.b12x.INTERMEDIATE_DIMENSION
            ),
        },
        "maximum_token_count": maximum_tokens,
        "buckets": [{
            "token_upper_bound": maximum_tokens,
            "backend_kind": "static",
            "function_name": function_name,
            "max_active_clusters": 1,
            "static_mma_tile_m": 128,
            "static_mma_tile_n": 128,
            "route_output_slice_count": (
                tool.b12x.INTERMEDIATE_DIMENSION + 127
            ) // 128,
            "avg_us": 1,
            "p95_us": 1,
            "routed_rows_capacity": maximum_tokens * tool.b12x.TOP_K,
            "max_rows": maximum_tokens * tool.b12x.TOP_K,
            "physical_tile_capacity": 0,
            "task_capacity": 0,
        }],
        "exported_objects": {
            function_name: {
                "kind": "static",
                "token_count": maximum_tokens,
                "object": object_name,
                "cache_key_repr": "fixture",
                "sha256": tool.stagepack.sha256_file(object_path),
            },
        },
    }
    manifest_hash = tool.hash_aot_manifest(manifest)
    manifest["manifest_hash_sha256"] = manifest_hash
    manifest["manifest_hash_low64"] = tool.low64_from_sha256(manifest_hash)
    manifest_path = generated / "aot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (generated / tool.GENERATED_SOURCE).write_text(
        f"{function_name}\n0x{manifest['manifest_hash_low64']:016x}\n",
        encoding="utf-8",
    )
    (generated / tool.GENERATED_FLAGS).write_text(
        "TVM_FFI_CFLAGS := -Ifixture\n"
        "TVM_FFI_LDFLAGS := -Lfixture\n"
        "TVM_FFI_LIBS := -ltvm_ffi\n"
        "CUTE_TVM_FFI_RUNTIME_LIBS := fixture\n",
        encoding="utf-8",
    )
    runtime_library = generated / "libtvm_ffi.so"
    runtime_library.write_bytes(b"fixture-runtime")
    (generated / tool.RUNTIME_LINK_ARGS).write_text(
        str(runtime_library) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_region_samples(
    tool,
    file_descriptor: int,
    regions: list[dict[str, int]],
) -> None:
    for region_index in (
        tool.b12x.REGION_W1_WEIGHT,
        tool.b12x.REGION_W1_SCALE,
        tool.b12x.REGION_W2_WEIGHT,
        tool.b12x.REGION_W2_SCALE,
    ):
        region = regions[region_index]
        for sample_index, relative_offset in enumerate(
            tool.stagepack.sample_offsets(region["bytes"], 64)
        ):
            payload = bytes(
                ((value + sample_index + region_index) % 251) + 1
                for value in range(64)
            )
            write_at(
                file_descriptor,
                region["offset"] + relative_offset,
                payload,
            )
    ones = tool.struct.pack(
        "<" + ("f" * tool.b12x.EXPERT_COUNT),
        *([1.0] * tool.b12x.EXPERT_COUNT),
    )
    for region_index in (
        tool.b12x.REGION_W1_ALPHA,
        tool.b12x.REGION_FC2_INPUT_SCALE,
        tool.b12x.REGION_W2_ALPHA,
    ):
        write_at(
            file_descriptor,
            regions[region_index]["offset"],
            ones,
        )


def create_pack(
    tool,
    root: Path,
    layer: int,
    aot_manifest: Path,
    maximum_tokens: int,
) -> tuple[Path, dict[str, object]]:
    aot_document = json.loads(aot_manifest.read_text(encoding="utf-8"))
    kernel_hash = aot_document["manifest_hash_low64"]
    aot_file_sha256 = tool.stagepack.sha256_file(aot_manifest)
    qualification_hash = tool.low64_from_sha256(aot_file_sha256)
    qualified_us = 1
    regions = tool.b12x.reserve_regions()
    pack_hash = tool.b12x.pack_metadata_hash_low64(
        layer,
        regions,
        maximum_tokens,
        qualified_us,
        qualification_hash,
        kernel_hash,
    )
    file_name = f"glm52_layer_{layer:04d}_b12x_moe.spb12x"
    path = root / file_name
    file_bytes = regions[-1]["offset"] + regions[-1]["bytes"]
    file_descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(file_descriptor, file_bytes)
        write_at(
            file_descriptor,
            0,
            tool.b12x.pack_header(
                layer,
                maximum_tokens,
                qualified_us,
                qualification_hash,
                kernel_hash,
                pack_hash,
                regions,
            ),
        )
        write_region_samples(tool, file_descriptor, regions)
    finally:
        os.close(file_descriptor)
    return path, {
        "path": str(path),
        "layer_index": layer,
        "bytes": file_bytes,
        "sha256": "b" * 64,
        "pack_hash_low64": pack_hash,
        "kernel_manifest_hash_low64": kernel_hash,
        "qualification_record_hash_low64": qualification_hash,
        "qualified_maximum_microseconds": qualified_us,
        "regions": regions,
        "reused": False,
    }


def run_preflight(
    script: Path,
    stagepack_root: Path,
    nvfp4_root: Path,
    aot_manifest: Path,
    rank: int,
    layer: int,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(script),
            "--rank",
            str(rank),
            "--layer",
            str(layer),
            "--stagepack-root",
            str(stagepack_root),
            "--nvfp4-pack-root",
            str(nvfp4_root),
            "--aot-manifest",
            str(aot_manifest),
            "--sample-bytes",
            "64",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "tools" / "glm52_nvfp4_artifact_preflight.py"
    tool = load_tool(repository)
    rank = 2
    layer = 12
    maximum_tokens = 1024
    with tempfile.TemporaryDirectory(
        prefix="sparkpipe_nvfp4_artifact_preflight_",
    ) as directory:
        root = Path(directory)
        model_dir = root / "model"
        stagepack_root = root / "stagepack"
        nvfp4_root = root / "nvfp4"
        generated = root / "generated"
        model_dir.mkdir()
        stagepack_root.mkdir()
        nvfp4_root.mkdir()
        generated.mkdir()
        source_index = model_dir / tool.b12x.SOURCE_INDEX_FILE
        source_index.write_text(
            json.dumps({"weight_map": {"fixture": "fixture.safetensors"}}),
            encoding="utf-8",
        )
        source_hash = tool.stagepack.sha256_file(source_index)
        create_stagepack(
            tool,
            stagepack_root,
            rank,
            layer,
            source_hash,
        )
        aot_manifest = create_aot_bundle(
            tool,
            generated,
            maximum_tokens,
        )
        micro_document = json.loads(
            aot_manifest.read_text(encoding="utf-8")
        )
        micro_document["buckets"][0]["backend_kind"] = "micro"
        try:
            tool.validate_aot_buckets(micro_document, maximum_tokens)
        except tool.PreflightFailure as error:
            assert "requires exact static AOT buckets" in str(error)
        else:
            raise AssertionError("micro AOT bucket unexpectedly passed preflight")
        pack_path, record = create_pack(
            tool,
            nvfp4_root,
            layer,
            aot_manifest,
            maximum_tokens,
        )
        aot_document = json.loads(aot_manifest.read_text(encoding="utf-8"))
        resident = tool.b12x.build_resident_manifest(
            model_dir,
            aot_manifest,
            maximum_tokens,
            aot_document["manifest_hash_low64"],
            [record],
        )
        resident_path = nvfp4_root / tool.NVFP4_MANIFEST
        resident_path.write_text(json.dumps(resident), encoding="utf-8")
        result = run_preflight(
            script,
            stagepack_root,
            nvfp4_root,
            aot_manifest,
            rank,
            layer,
        )
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt["status"] == "ok"
        assert receipt["maximum_token_count"] == maximum_tokens
        assert receipt["required_execution_rows"] == 1024
        assert receipt["aot"]["object_count"] == 1
        assert [item["layer"] for item in receipt["layers"]] == [layer]
        mtp = run_preflight(
            script,
            stagepack_root,
            nvfp4_root,
            aot_manifest,
            rank,
            layer,
            "--mtp",
        )
        assert mtp.returncode != 0
        assert "required=7168 available=1024" in mtp.stderr
        resident["scale2_baked_into_block_scales"] = False
        resident_path.write_text(json.dumps(resident), encoding="utf-8")
        stale_scale_contract = run_preflight(
            script,
            stagepack_root,
            nvfp4_root,
            aot_manifest,
            rank,
            layer,
        )
        assert stale_scale_contract.returncode != 0
        assert "scale2_baked_into_block_scales mismatch" in (
            stale_scale_contract.stderr
        )
        resident["scale2_baked_into_block_scales"] = True
        resident_path.write_text(json.dumps(resident), encoding="utf-8")
        fields, regions = tool.parse_pack_header(
            pack_path.read_bytes()[:tool.b12x.HEADER_BYTES]
        )
        wrong_kernel_hash = fields[19] + 1
        file_descriptor = os.open(pack_path, os.O_WRONLY)
        try:
            write_at(
                file_descriptor,
                0,
                tool.b12x.pack_header(
                    layer,
                    maximum_tokens,
                    fields[17],
                    fields[18],
                    wrong_kernel_hash,
                    fields[20],
                    regions,
                ),
            )
        finally:
            os.close(file_descriptor)
        wrong_kernel = run_preflight(
            script,
            stagepack_root,
            nvfp4_root,
            aot_manifest,
            rank,
            layer,
        )
        assert wrong_kernel.returncode != 0
        assert "kernel manifest hash mismatch" in wrong_kernel.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
