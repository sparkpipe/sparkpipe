#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def write_aot_manifest(module, path: Path, maximum_token_count: int = 1024):
    document = {
        "record_schema": module.AOT_MANIFEST_SCHEMA,
        "required_module": module.REQUIRED_MODULE,
        "required_arch": module.REQUIRED_ARCH,
        "fallback_allowed": False,
        "runtime_backend_selection": "forbidden",
        "maximum_token_count": maximum_token_count,
        "buckets": [{
            "token_upper_bound": maximum_token_count,
            "p95_us": 17,
        }],
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    digest = module.sha256_text(encoded)
    document["manifest_hash_sha256"] = digest
    document["manifest_hash_low64"] = module.low64_from_hex(digest)
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def load_pack_module():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "glm52_b12x_resident_pack.py"
    spec = importlib.util.spec_from_file_location(
        "glm52_b12x_resident_pack",
        tool_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load B12x resident pack tool")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tool_path.parent))
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def main() -> int:
    module = load_pack_module()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_dir = root / "model"
        model_dir.mkdir()
        source_index = model_dir / module.SOURCE_INDEX_FILE
        source_index.write_text(
            json.dumps({"weight_map": {"tensor": "model.safetensors"}}),
            encoding="utf-8",
        )
        aot_manifest = root / "aot_manifest.json"
        aot_document = write_aot_manifest(module, aot_manifest)
        assert module.read_aot_manifest(aot_manifest) == (
            1024,
            17,
            aot_document["manifest_hash_low64"],
        )
        records = [{
            "layer_index": 3,
            "path": str(root / "glm52_layer_0003_b12x_moe.spb12x"),
            "bytes": 1234,
            "sha256": "a" * 64,
        }]
        manifest = module.build_resident_manifest(
            model_dir,
            aot_manifest,
            1024,
            0x1234,
            records,
        )
        assert manifest["record_schema"] == module.MANIFEST_SCHEMA
        assert manifest["source_model_index_sha256"] == module.sha256_file(
            source_index
        )
        assert manifest["aot_manifest_sha256"] == module.sha256_file(
            aot_manifest
        )
        assert manifest["pack_magic"] == "SPARKGLM52B12X"
        assert manifest["pack_extension"] == ".spb12x"
        assert manifest["pack_abi_version"] == 3
        assert manifest["maximum_token_count"] == 1024
        assert manifest["quant_mode"] == module.QUANT_MODE_NVFP4
        assert manifest["output_dtype_name"] == "BF16"
        assert manifest["scale2_baked_into_block_scales"] is True
        assert manifest["w1_alpha"] == "ones_fp32_by_expert"
        assert manifest["w2_alpha"] == "ones_fp32_by_expert"
        assert manifest["fc2_input_scale"] == "ones_fp32_by_expert"
        assert manifest["fallback_allowed"] is False
        assert manifest["runtime_backend_selection"] == "forbidden"
        assert manifest["packs"] == records
        stale_aot_manifest = root / "stale_aot_manifest.json"
        stale_document = dict(aot_document)
        stale_document["maximum_token_count"] = 2048
        stale_aot_manifest.write_text(
            json.dumps(stale_document),
            encoding="utf-8",
        )
        try:
            module.read_aot_manifest(stale_aot_manifest)
        except module.PackFailure as error:
            assert "content hash" in str(error)
        else:
            raise AssertionError("stale AOT manifest hash was accepted")
        fallback_aot_manifest = root / "fallback_aot_manifest.json"
        fallback_document = dict(aot_document)
        fallback_document["fallback_allowed"] = True
        encoded = json.dumps(
            {
                key: value
                for key, value in fallback_document.items()
                if key not in ("manifest_hash_sha256", "manifest_hash_low64")
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        fallback_digest = module.sha256_text(encoded)
        fallback_document["manifest_hash_sha256"] = fallback_digest
        fallback_document["manifest_hash_low64"] = module.low64_from_hex(
            fallback_digest
        )
        fallback_aot_manifest.write_text(
            json.dumps(fallback_document),
            encoding="utf-8",
        )
        try:
            module.read_aot_manifest(fallback_aot_manifest)
        except module.PackFailure as error:
            assert "permits fallback" in str(error)
        else:
            raise AssertionError("fallback-enabled AOT manifest was accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
