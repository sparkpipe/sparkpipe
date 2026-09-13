#!/usr/bin/env python3
"""Host gate: mint a tiny SYNTHETIC GLM5.2 drafter pack end-to-end.

No GPU, no torch, no real checkpoint: a dwarf-geometry drafter fixture runs
the exact production mint path (tools/glm52_dspark_stagepack.mint) - shard
consolidation, manifest field pinning via glm52_dspark_manifest, bytes+sha256
receipts - and the verifier that both the backend's Initialize contract and
tools/glm52_gen_deployment.py apply. Negative cases prove staleness,
truncation and shape drift fail loudly. The python tensor table is pinned
against the backend C tables' shape count (9 fixed + 11 x draft layers = 64
roles, matching SPARK_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT).
"""

import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "glm52_dspark_stagepack.py"
DEPLOY_TOOL = ROOT / "tools" / "glm52_gen_deployment.py"
MODEL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"

DWARF_CONTRACT = {
    "hidden_dimension": 64,
    "output_vocab_count": 128,
    "maximum_context_tokens": 4096,
    "rms_norm_epsilon": 1e-5,
    "rope_theta": 10000.0,
    "dspark": {
        "aux_layer_ids": [1, 2],
        "block_size": 8,
        "draft_layer_count": 2,
        "draft_attention_head_count": 4,
        "draft_kv_head_count": 4,
        "draft_head_dimension": 16,
        "draft_intermediate_dimension": 32,
        "markov_rank": 8,
        "mask_token_id": 12,
        "max_anchors": 64,
        "maximum_speculative_token_count": 7,
    },
}


def load_tool():
    spec = importlib.util.spec_from_file_location("glm52_dspark_stagepack",
                                                  TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so gen_deployment's lazy
    # `from glm52_dspark_stagepack import ...` reuses THIS instance and the
    # geometry patch below is observed by the audited code path.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_safetensors(path: Path, tensors: dict) -> None:
    header = {}
    blobs = []
    offset = 0
    for name in sorted(tensors):
        shape, payload = tensors[name]
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + len(payload)]}
        blobs.append(payload)
        offset += len(payload)
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for blob in blobs:
            handle.write(blob)


def dwarf_checkpoint(directory: Path, tool, sharded: bool,
                     mutate=None) -> Path:
    """Synthetic drafter checkpoint at dwarf geometry; every tensor gets
    distinct pseudo-random bf16 bytes so consolidation is checkable."""
    geometry = tool.resolve_geometry(DWARF_CONTRACT)
    tensors = {}
    for index, (kind, name, rows, columns) in enumerate(
            tool.tensor_specs(geometry)):
        payload = bytes((index * 31 + row) % 256
                        for row in range(rows * columns * 2))
        tensors[name] = ((rows, columns), payload)
    if mutate is not None:
        mutate(tensors)
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids":
            DWARF_CONTRACT["dspark"]["aux_layer_ids"],
        "block_size": DWARF_CONTRACT["dspark"]["block_size"],
        "dtype": "bfloat16",
        "draft_vocab_size": DWARF_CONTRACT["output_vocab_count"],
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": DWARF_CONTRACT["dspark"]["markov_rank"],
        "max_anchors": DWARF_CONTRACT["dspark"]["max_anchors"],
        "speculators_config": {
            "algorithm": "dspark",
            "proposal_methods": [{
                "proposal_type": "greedy",
                "speculative_tokens":
                    DWARF_CONTRACT["dspark"]["maximum_speculative_token_count"],
                "verifier_accept_k": 1,
            }],
            "verifier": {"name_or_path": "zai-org/GLM-5.2-FP8"},
        },
        "transformer_layer_config": {
            "hidden_size": DWARF_CONTRACT["hidden_dimension"],
            "intermediate_size":
                DWARF_CONTRACT["dspark"]["draft_intermediate_dimension"],
            "num_hidden_layers": DWARF_CONTRACT["dspark"]["draft_layer_count"],
            "num_attention_heads":
                DWARF_CONTRACT["dspark"]["draft_attention_head_count"],
            "num_key_value_heads":
                DWARF_CONTRACT["dspark"]["draft_kv_head_count"],
            "head_dim": DWARF_CONTRACT["dspark"]["draft_head_dimension"],
            "vocab_size": DWARF_CONTRACT["output_vocab_count"],
            "max_position_embeddings": DWARF_CONTRACT["maximum_context_tokens"],
            "rms_norm_eps": DWARF_CONTRACT["rms_norm_epsilon"],
            "rope_parameters": {"rope_theta": DWARF_CONTRACT["rope_theta"]},
        },
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    if not sharded:
        write_safetensors(directory / "model.safetensors", tensors)
        return directory
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2],
              "model-00002-of-00002.safetensors": names[1::2]}
    weight_map = {}
    for shard, shard_names in shards.items():
        write_safetensors(directory / shard,
                          {name: tensors[name] for name in shard_names})
        for name in shard_names:
            weight_map[name] = shard
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return directory


def source_payloads(checkpoint: Path) -> dict:
    """name -> payload bytes, read back through every checkpoint shard."""
    payloads = {}
    index = checkpoint / "model.safetensors.index.json"
    paths = ([checkpoint / "model.safetensors"] if not index.is_file()
             else sorted({checkpoint / shard for shard in
                          json.loads(index.read_text())["weight_map"].values()}))
    for path in paths:
        with open(path, "rb") as handle:
            (header_bytes,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_bytes))
            base = 8 + header_bytes
            for name, record in header.items():
                if name == "__metadata__":
                    continue
                begin, end = record["data_offsets"]
                handle.seek(base + begin)
                payloads[name] = handle.read(end - begin)
    return payloads


def main() -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    tool = load_tool()

    import glm52_dspark_manifest as manifest_module
    saved = {name: getattr(manifest_module, name)
             for name in ("MODEL_CONTRACT", "DSPARK_CONTRACT", "AUX_LAYERS",
                          "MAX_SPECULATIVE_TOKENS")}
    manifest_module.MODEL_CONTRACT = DWARF_CONTRACT
    manifest_module.DSPARK_CONTRACT = DWARF_CONTRACT["dspark"]
    manifest_module.AUX_LAYERS = list(
        DWARF_CONTRACT["dspark"]["aux_layer_ids"])
    manifest_module.MAX_SPECULATIVE_TOKENS = (
        DWARF_CONTRACT["dspark"]["maximum_speculative_token_count"])
    try:
        run_gate(tool)
    finally:
        for name, value in saved.items():
            setattr(manifest_module, name, value)
    return 0


def run_gate(tool) -> None:
    geometry = tool.resolve_geometry(DWARF_CONTRACT)

    # 1. The python spec table mirrors the backend C tables: at the real
    #    geometry the role count equals SPARK_DSPARK_DRAFT_BACKEND_WEIGHT_COUNT.
    real_geometry = tool.resolve_geometry()
    assert len(tool.tensor_specs(real_geometry)) == 64, \
        "python spec table != backend weight-role count (64)"
    assert len(tool.tensor_specs(geometry)) == \
        9 + 11 * DWARF_CONTRACT["dspark"]["draft_layer_count"]
    fixed_names = {name for kind, name, _, _ in tool.tensor_specs(real_geometry)
                   if kind == "fixed"}
    assert fixed_names == {
        "embed_tokens.weight", "fc.weight", "hidden_norm.weight",
        "norm.weight", "lm_head.weight", "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight", "confidence_head.proj.weight",
        "confidence_head.proj.bias"}, fixed_names

    with tempfile.TemporaryDirectory(prefix="glm52_dspark_gate.") as tmp:
        tmp = Path(tmp)

        # 2. Sharded checkpoint -> consolidated pack, end to end.
        checkpoint = dwarf_checkpoint(tmp / "ckpt_sharded", tool,
                                      sharded=True)
        pack = tmp / "packs" / "glm52_dspark_drafter"
        receipt = tool.mint(checkpoint, pack, MODEL_REVISION,
                            contract=DWARF_CONTRACT)
        assert receipt["verified"]["ok"] is True
        assert receipt["inventory"]["consolidated"] is True
        assert receipt["inventory"]["tensor_count"] == \
            9 + 11 * DWARF_CONTRACT["dspark"]["draft_layer_count"]
        assert (pack / "manifest.json").is_file()
        # Byte-for-byte round trip through the consolidation.
        packed = tool.SafetensorsFile(pack / "model.safetensors")
        sources = source_payloads(checkpoint)
        for _, name, _, _ in tool.tensor_specs(geometry):
            located = next((candidate for candidate in (name, "model." + name)
                            if candidate in packed.header), None)
            assert located is not None, name
            _, _, start, count = packed.entry(located)
            with open(pack / "model.safetensors", "rb") as handle:
                handle.seek(start)
                assert handle.read(count) == sources[locate_source(
                    located, sources)], name
        # Independent re-verification of the minted pack.
        tool.verify_pack_directory(pack, geometry)

        # 3. Single-file checkpoint -> byte-copied pack.
        single_ckpt = dwarf_checkpoint(tmp / "ckpt_single", tool,
                                       sharded=False)
        single_pack = tmp / "pack_single"
        receipt_single = tool.mint(single_ckpt, single_pack, MODEL_REVISION,
                                   contract=DWARF_CONTRACT)
        assert receipt_single["inventory"]["consolidated"] is False
        assert (single_pack / "model.safetensors").read_bytes() == \
            (single_ckpt / "model.safetensors").read_bytes()
        tool.verify_pack_directory(single_pack, geometry)

        # 4. Staleness fails loudly: flip one payload byte post-mint.
        tampered = tmp / "pack_tampered"
        shutil.copytree(pack, tampered)
        data = bytearray((tampered / "model.safetensors").read_bytes())
        data[-1] ^= 0xFF
        (tampered / "model.safetensors").write_bytes(bytes(data))
        try:
            tool.verify_pack_directory(tampered, geometry)
            raise AssertionError("tampered pack passed verification")
        except tool.PackFailure as error:
            assert "sha256" in str(error), error

        # 5. Shape drift in the checkpoint refuses to mint.
        def bend(tensors):
            name = "layers.0.self_attn.q_proj.weight"
            shape, _ = tensors[name]
            tensors[name] = (shape, tensors[name][1][:-2])

        bad_ckpt = dwarf_checkpoint(tmp / "ckpt_bad", tool, sharded=True,
                                    mutate=bend)
        try:
            tool.mint(bad_ckpt, tmp / "pack_bad", MODEL_REVISION,
                      contract=DWARF_CONTRACT)
            raise AssertionError("shape-drifted checkpoint minted")
        except tool.PackFailure:
            pass

        # 6. gen_deployment's consistency pass: a missing pack refuses
        #    loudly through the real CLI (6a); accept + hash-staleness are
        #    proven in-process at dwarf geometry (6b/6c) because the
        #    production CLI pins REAL contract dims; 6d proves the
        #    production CLI refuses even a fresh dwarf pack.
        deploy_out = tmp / "deploy"
        result = subprocess.run(
            [sys.executable, str(DEPLOY_TOOL), str(deploy_out),
             "--pipeline", "tp1"], capture_output=True, text=True)
        assert result.returncode != 0, \
            "tp1 generation passed WITHOUT the referenced drafter pack"
        assert "drafter pack is missing" in (result.stderr + result.stdout), \
            (result.stderr, result.stdout)

        import glm52_gen_deployment as deploy
        deploy_config = {"dspark_pack_path": "packs/glm52_dspark_drafter",
                         "speculation_enabled": True}
        deploy_pack = deploy_out / "packs" / "glm52_dspark_drafter"
        shutil.copytree(pack, deploy_pack)
        saved_resolver = tool.resolve_geometry
        tool.resolve_geometry = lambda contract=None: dict(geometry)
        try:
            # 6b. A fresh pack at the audited geometry passes the audit.
            deploy.audit_dspark_reference(str(deploy_out), deploy_config)
            # 6c. Same-length content tampering is still caught - by the
            #     sha256 receipt, not merely the byte count.
            data = bytearray((deploy_pack / "config.json").read_bytes())
            data[0] = data[0] ^ 0x20
            (deploy_pack / "config.json").write_bytes(bytes(data))
            assert len(data) == (deploy_pack / "config.json").stat().st_size
            try:
                deploy.audit_dspark_reference(str(deploy_out), deploy_config)
                raise AssertionError("stale drafter pack passed the audit")
            except tool.PackFailure as error:
                assert "sha256" in str(error), error
        finally:
            tool.resolve_geometry = saved_resolver

        # 6d. The production CLI audits at REAL contract dims: even a fresh,
        #     internally-consistent dwarf pack is refused there - a servable
        #     pack must carry the compiled geometry.
        shutil.rmtree(deploy_pack)
        shutil.copytree(pack, deploy_pack)
        result = subprocess.run(
            [sys.executable, str(DEPLOY_TOOL), str(deploy_out),
             "--pipeline", "tp1"], capture_output=True, text=True)
        assert result.returncode != 0, \
            "production deployment accepted a dwarf-geometry drafter pack"
        assert "REFUSED" in (result.stderr + result.stdout), \
            (result.stderr, result.stdout)

    print("test_glm52_dspark_stagepack: mint/verify/consistency PASS")


def locate_source(packed_name: str, sources: dict) -> str:
    if packed_name in sources:
        return packed_name
    prefixless = packed_name[len("model."):] if packed_name.startswith(
        "model.") else packed_name
    if prefixless in sources:
        return prefixless
    raise KeyError(packed_name)


if __name__ == "__main__":
    raise SystemExit(main())
