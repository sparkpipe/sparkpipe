#!/usr/bin/env python3
"""Mint the GLM-5.2 DFlash2 drafter artifact pack (setup time, never serving).

The resident stage's DFlash2 speculator consumes THREE artifacts through its
process environment (SPARK_GLM52_DSPARK_{MANIFEST,CONFIG,SAFETENSORS}):
a manifest JSON pinning bytes+sha256 receipts, the drafter config.json the
manifest records, and ONE consolidated BF16 model.safetensors. The backend
(modules/glm52_dspark_draft_backend) re-validates every one of those receipts
at Initialize and the epoch-3 validator re-checks them on hardware, so this
tool emits them together and re-verifies the result before declaring success
- a stale or half-copied pack cannot pass silently.

This mirrors tools/qwen36_dspark_stagepack.py (that family loads a wire pack;
the GLM5.2 backend consumes the raw triple instead - same receipt discipline,
different container). Drafter checkpoint input may be a single model-
safetensors or HF-sharded (model.safetensors.index.json); shards are
consolidated losslessly, byte for byte, without torch. Non-model logic
(failure type, alignment, contract loading, manifest field pinning) is shared
with glm52_resident_pack_common.py / glm52_dspark_manifest.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from glm52_dspark_manifest import (  # noqa: E402
    FORMAT as MANIFEST_FORMAT,
    build_manifest,
    sha256_file,
)
from glm52_resident_pack_common import PackFailure  # noqa: E402

try:
    from glm52_model_contract import load_model_contract  # noqa: E402
except ImportError:  # pragma: no cover - direct-script fallback
    from glm52_model_contract import load_model_contract  # type: ignore

RECEIPT_KIND = "sparkpipe.glm52.dflash2-stagepack-receipt.v1"
SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "manifest.json"

# Fixed ABI constants of the manifest contract block, mirroring
# SparkGlm52DsparkManifestValidateContract in the backend verbatim.
CONTRACT_ABI_VERSION = 2
VERIFIER_HIDDEN_DTYPE_BF16 = 1
DRAFT_DTYPE_BF16 = 1


def resolve_geometry(contract: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The drafter geometry table, resolved from the model contract.

    Single source for both the tensor inventory below and the manifest
    cross-checks; tests inject a dwarf table to exercise the identical code
    path at negligible size (such a pack is a schema fixture, never servable
    - the backend pins these numbers against its compiled constants).
    """
    contract = contract if contract is not None else load_model_contract()
    dspark = contract["dspark"]
    return {
        "hidden_dimension": contract["hidden_dimension"],
        "vocabulary_size": contract["output_vocab_count"],
        "draft_layer_count": dspark["draft_layer_count"],
        "attention_head_count": dspark["draft_attention_head_count"],
        "kv_head_count": dspark["draft_kv_head_count"],
        "head_dimension": dspark["draft_head_dimension"],
        "intermediate_dimension": dspark["draft_intermediate_dimension"],
        "markov_rank": dspark["markov_rank"],
        "max_anchors": dspark["max_anchors"],
        "block_size": dspark["block_size"],
        "maximum_speculative_token_count":
            dspark["maximum_speculative_token_count"],
        "aux_hidden_state_layer_ids": list(dspark["aux_layer_ids"]),
    }


def derived_dimensions(geometry: Dict[str, Any]) -> Dict[str, int]:
    """Dimensions the backend derives from the base table."""
    return {
        "attention_dimension": geometry["attention_head_count"]
        * geometry["head_dimension"],
        "fused_input_dimension": len(geometry["aux_hidden_state_layer_ids"])
        * geometry["hidden_dimension"],
        "confidence_dimension": geometry["hidden_dimension"]
        + geometry["markov_rank"],
    }


def tensor_specs(geometry: Dict[str, Any]) -> List[Tuple[str, str, int, int]]:
    """The exact inventory the backend uploads, in its declaration order:
    (kind, name, rows, columns) where kind is "fixed" or "layer".

    Mirror of SparkGlm52DsparkLoadFixedTensors / LoadLayerTensors; the host
    gate pins this table against those C tables so drift fails in CI rather
    than on a GB10 at pack time. The backend also accepts each fixed name
    under a redundant "model." prefix; both spellings validate here.
    """
    dims = derived_dimensions(geometry)
    hidden = geometry["hidden_dimension"]
    vocab = geometry["vocabulary_size"]
    markov = geometry["markov_rank"]
    attn = dims["attention_dimension"]
    head_dim = geometry["head_dimension"]
    intermediate = geometry["intermediate_dimension"]
    fixed = [
        ("embed_tokens.weight", vocab, hidden),
        ("fc.weight", hidden, dims["fused_input_dimension"]),
        ("hidden_norm.weight", hidden, 1),
        ("norm.weight", hidden, 1),
        ("lm_head.weight", vocab, hidden),
        ("markov_head.markov_w1.weight", vocab, markov),
        ("markov_head.markov_w2.weight", vocab, markov),
        ("confidence_head.proj.weight", 1, dims["confidence_dimension"]),
        ("confidence_head.proj.bias", 1, 1),
    ]
    per_layer = [
        ("input_layernorm.weight", hidden, 1),
        ("self_attn.q_proj.weight", attn, hidden),
        ("self_attn.k_proj.weight", attn, hidden),
        ("self_attn.v_proj.weight", attn, hidden),
        ("self_attn.q_norm.weight", head_dim, 1),
        ("self_attn.k_norm.weight", head_dim, 1),
        ("self_attn.o_proj.weight", hidden, attn),
        ("post_attention_layernorm.weight", hidden, 1),
        ("mlp.gate_proj.weight", intermediate, hidden),
        ("mlp.up_proj.weight", intermediate, hidden),
        ("mlp.down_proj.weight", hidden, intermediate),
    ]
    specs = [("fixed", name, rows, columns) for name, rows, columns in fixed]
    for layer in range(geometry["draft_layer_count"]):
        for name, rows, columns in per_layer:
            specs.append(("layer", "layers.%d.%s" % (layer, name), rows, columns))
    return specs


class SafetensorsFile:
    """Minimal safetensors reader over one file: header map + payload region."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise PackFailure("safetensors too small: %s" % path)
            (header_bytes,) = struct.unpack("<Q", raw)
            header_raw = handle.read(header_bytes)
            if len(header_raw) != header_bytes:
                raise PackFailure("truncated safetensors header: %s" % path)
        try:
            self.header = json.loads(header_raw)
        except ValueError as error:
            raise PackFailure("invalid safetensors JSON header: %s" % error)
        self.header.pop("__metadata__", None)
        self.data_offset = 8 + header_bytes
        self.file_bytes = path.stat().st_size

    def names(self) -> List[str]:
        return list(self.header)

    def entry(self, name: str) -> Tuple[str, Tuple[int, ...], int, int]:
        """(dtype, shape, absolute data start, byte count)."""
        record = self.header.get(name)
        if not isinstance(record, dict):
            raise PackFailure("missing safetensors tensor: %s" % name)
        try:
            dtype = record["dtype"]
            shape = tuple(int(value) for value in record["shape"])
            begin, end = (int(value) for value in record["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise PackFailure("malformed safetensors entry %s: %s"
                              % (name, error))
        if end < begin or self.data_offset + end > self.file_bytes:
            raise PackFailure("safetensors range out of file: %s" % name)
        return dtype, shape, self.data_offset + begin, end - begin


def open_checkpoint_tensor_map(checkpoint: Path) -> Tuple[Dict[str, Path], Path]:
    """Resolve the checkpoint's tensor->shard map (single file or sharded)."""
    single = checkpoint / SAFETENSORS_NAME
    index = checkpoint / (SAFETENSORS_NAME + ".index.json")
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = {name: checkpoint / shard for name, shard in weight_map.items()}
        missing = sorted({str(shard) for shard in shards.values()
                          if not shard.is_file()})
        if missing:
            raise PackFailure("missing safetensors shards: %s" % missing)
        return shards, index
    if single.is_file():
        return {}, single
    raise PackFailure("drafter checkpoint has no model.safetensors: %s"
                      % checkpoint)


class TensorPlan:
    """One spec tensor located inside its checkpoint shard."""

    def __init__(self, kind: str, name: str, shard: Path, start: int,
                 count: int, dtype: str, shape: Tuple[int, ...]):
        self.kind = kind
        self.name = name
        self.shard = shard
        self.start = start
        self.count = count
        self.dtype = dtype
        self.shape = shape


def locate_spec_tensors(
    shards: Dict[str, Path], single: Path, geometry: Dict[str, Any]
) -> Tuple[List[TensorPlan], List[str]]:
    """Validate every spec tensor against the checkpoint headers.

    Returns ([TensorPlan...], extra_names). A spec tensor may sit under its
    bare name (layer tensors always) or under the backend's optional
    "model." prefix (fixed tensors); dtype must be BF16 and the stored shape
    must equal the spec exactly - the same acceptance rules the C backend
    applies at Initialize, enforced here at mint time instead.
    """
    shard_paths = sorted(set(shards.values()))
    headers = ([SafetensorsFile(single)] if not shard_paths
               else [SafetensorsFile(path) for path in shard_paths])
    plans: List[TensorPlan] = []
    claimed: set = set()
    for kind, name, rows, columns in tensor_specs(geometry):
        candidates = (name, "model." + name) if kind == "fixed" else (name,)
        located: Optional[Tuple[str, SafetensorsFile]] = None
        for candidate in candidates:
            for header in headers:
                if candidate in header.header:
                    located = (candidate, header)
                    break
            if located is not None:
                break
        if located is None:
            raise PackFailure("missing drafter tensor: %s" % name)
        candidate, header = located
        dtype, shape, start, count = header.entry(candidate)
        if dtype != "BF16":
            raise PackFailure("drafter tensor %s must be BF16, got %s"
                              % (candidate, dtype))
        if tuple(shape) != (rows, columns):
            raise PackFailure("drafter tensor %s shape %r != spec [%d,%d]"
                              % (candidate, list(shape), rows, columns))
        if count != rows * columns * 2:
            raise PackFailure("drafter tensor %s byte count mismatch" % candidate)
        plans.append(TensorPlan(kind, candidate, header.path, start, count,
                                dtype, shape))
        claimed.add(candidate)
    extras: List[str] = []
    for header in headers:
        extras.extend(sorted(set(header.names()) - claimed))
    return plans, extras


def consolidate(
    shards: Dict[str, Path], single: Path, output: Path,
    geometry: Dict[str, Any]
) -> Dict[str, Any]:
    """Emit ONE consolidated model.safetensors; returns the inventory.

    Single-file checkpoints whose inventory already validates are copied
    byte for byte (the backend mmaps the file, so bytes are the contract).
    Sharded checkpoints are merged without torch: the new header assigns
    contiguous offsets in spec order, every payload range is copied
    verbatim from its shard, and unclaimed extra tensors ride along so the
    consolidation stays lossless.
    """
    plans, extras = locate_spec_tensors(shards, single, geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not shards:
        shutil.copyfile(single, output)
        return {"tensor_count": len(plans), "extras": extras,
                "consolidated": False}

    shard_paths = sorted(set(shards.values()))
    headers = [SafetensorsFile(path) for path in shard_paths]

    header: Dict[str, Any] = {}
    copies: List[Tuple[int, int, int]] = []  # (header_index, src_start, count)
    cursor = 0

    def claim(name: str, dtype: str, shape: Tuple[int, ...], count: int,
              header_index: int, start: int) -> None:
        nonlocal cursor
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [cursor, cursor + count]}
        copies.append((header_index, start, count))
        cursor += count

    for plan in plans:
        header_index = shard_paths.index(plan.shard)
        claim(plan.name, plan.dtype, plan.shape, plan.count, header_index,
              plan.start)
    claimed = set(header)
    for index, source in enumerate(headers):
        for name in sorted(source.names()):
            if name in claimed:
                continue
            dtype, shape, start, count = source.entry(name)
            claim(name, dtype, shape, count, index, start)

    header_raw = json.dumps(header).encode("utf-8")
    temp = output.with_name(output.name + ".consolidating")
    handles = [open(path, "rb") for path in shard_paths]
    try:
        with open(temp, "wb") as out:
            out.write(struct.pack("<Q", len(header_raw)))
            out.write(header_raw)
            for header_index, start, count in copies:
                handles[header_index].seek(start)
                payload = handles[header_index].read(count)
                if len(payload) != count:
                    raise PackFailure("short read consolidating %s" % output)
                out.write(payload)
        os.replace(temp, output)
    finally:
        for handle in handles:
            handle.close()
    return {"tensor_count": len(plans), "extras": extras, "consolidated": True}


def require_revision(model_revision: str) -> str:
    """The backend refuses any manifest revision that is not 40 lowercase
    hex characters; refuse here, at mint time, with the same rule."""
    if len(model_revision) != 40 or any(
            character not in "0123456789abcdef"
            for character in model_revision):
        raise PackFailure("--model-revision must be 40 lowercase hex "
                          "characters, got %r" % model_revision)
    return model_revision


def mint(checkpoint: Path, output: Path, model_revision: str,
         contract: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mint <output>/{config.json,model.safetensors,manifest.json} + receipt.

    The manifest is built FROM the emitted files, so its bytes+sha256
    receipts pin exactly what a consumer will hash again.
    """
    geometry = resolve_geometry(contract)
    require_revision(model_revision)
    config_source = checkpoint / CONFIG_NAME
    if not config_source.is_file():
        raise PackFailure("missing drafter config: %s" % config_source)

    shards, single = open_checkpoint_tensor_map(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_source, output / CONFIG_NAME)
    inventory = consolidate(shards, single, output / SAFETENSORS_NAME,
                            geometry)

    # Field pinning (architectures, block_size, draft dims, verifier...) is
    # glm52_dspark_manifest's job; running it over the OUTPUT dir makes the
    # manifest describe the minted bytes, never the upstream ones.
    manifest = build_manifest(output, model_revision)
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    verification = verify_pack_directory(output, geometry)
    receipt = {
        "kind": RECEIPT_KIND,
        "tool": "tools/glm52_dspark_stagepack.py",
        "model_revision": model_revision,
        "checkpoint": str(checkpoint),
        "geometry": geometry,
        "inventory": inventory,
        "files": {
            CONFIG_NAME: manifest["config_json"],
            SAFETENSORS_NAME: manifest["model_safetensors"],
            MANIFEST_NAME: {
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
        },
        "verified": verification,
    }
    (output / "mint_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return receipt


def verify_pack_directory(
    pack_dir: Path, geometry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Re-validate a minted pack exactly where its consumers will.

    Mirrors SparkGlm52DsparkValidateArtifactManifest (format id, revision
    syntax, verifier block, contract block, per-file bytes+sha256 records)
    plus the safetensors inventory check against the backend's upload
    table. Raises PackFailure loudly on any stale, truncated, or tampered
    file; gen_deployment calls this on every referenced dspark_pack_path.
    """
    geometry = (geometry if geometry is not None else resolve_geometry())
    manifest_path = pack_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PackFailure("drafter pack missing manifest: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise PackFailure("drafter pack %s: %s" % (pack_dir, message))

    require(manifest.get("format") == MANIFEST_FORMAT,
            "format=%r != %r" % (manifest.get("format"), MANIFEST_FORMAT))
    revision = manifest.get("model_revision")
    require(isinstance(revision, str) and len(revision) == 40
            and all(character in "0123456789abcdef"
                    for character in revision),
            "model_revision must be 40 lowercase hex chars")
    verifier = manifest.get("verifier_contract")
    require(isinstance(verifier, dict), "missing verifier_contract block")
    require(verifier.get("quantization_independent") is True,
            "verifier_contract.quantization_independent must be true")
    require(verifier.get("hidden_dtype") == "bf16",
            "verifier_contract.hidden_dtype must be bf16")
    require(verifier.get("hidden_dimension") == geometry["hidden_dimension"],
            "verifier_contract.hidden_dimension mismatch")
    require(verifier.get("vocabulary_size") == geometry["vocabulary_size"],
            "verifier_contract.vocabulary_size mismatch")
    contract_block = manifest.get("contract")
    require(isinstance(contract_block, dict), "missing contract block")
    expected_contract = {
        "abi_version": CONTRACT_ABI_VERSION,
        "verifier_hidden_dtype": VERIFIER_HIDDEN_DTYPE_BF16,
        "draft_dtype": DRAFT_DTYPE_BF16,
        "draft_layer_count": geometry["draft_layer_count"],
        "block_size": geometry["block_size"],
        "hidden_dimension": geometry["hidden_dimension"],
        "intermediate_dimension": geometry["intermediate_dimension"],
        "attention_head_count": geometry["attention_head_count"],
        "kv_head_count": geometry["kv_head_count"],
        "head_dimension": geometry["head_dimension"],
        "vocab_size": geometry["vocabulary_size"],
        "draft_vocab_size": geometry["vocabulary_size"],
        "markov_rank": geometry["markov_rank"],
        "max_anchors": geometry["max_anchors"],
        "maximum_speculative_token_count":
            geometry["maximum_speculative_token_count"],
        "verifier_accept_k": 1,
        "enable_confidence_head": 1,
        "confidence_head_with_markov": 1,
    }
    for name, value in expected_contract.items():
        require(contract_block.get(name) == value,
                "contract.%s=%r != %r"
                % (name, contract_block.get(name), value))

    for record, filename in (("config_json", CONFIG_NAME),
                             ("model_safetensors", SAFETENSORS_NAME)):
        entry = manifest.get(record)
        require(isinstance(entry, dict), "missing manifest record %s" % record)
        require(entry.get("path") == filename,
                "%s.path=%r != %r" % (record, entry.get("path"), filename))
        file_path = pack_dir / filename
        require(file_path.is_file(), "missing file: %s" % file_path)
        actual_bytes = file_path.stat().st_size
        require(entry.get("bytes") == actual_bytes,
                "%s.bytes=%r != on-disk %d"
                % (record, entry.get("bytes"), actual_bytes))
        actual_sha256 = sha256_file(file_path)
        require(entry.get("sha256") == actual_sha256,
                "%s.sha256 stale: manifest %s != actual %s"
                % (record, entry.get("sha256"), actual_sha256))

    safetensors = SafetensorsFile(pack_dir / SAFETENSORS_NAME)
    seen: set = set()
    for _, name, rows, columns in tensor_specs(geometry):
        located = next((candidate for candidate in (name, "model." + name)
                        if candidate in safetensors.header), None)
        require(located is not None, "safetensors missing tensor %s" % name)
        dtype, shape, _, count = safetensors.entry(located)
        require(dtype == "BF16",
                "tensor %s dtype %s != BF16" % (located, dtype))
        require(tuple(shape) == (rows, columns),
                "tensor %s shape %r != [%d,%d]"
                % (located, list(shape), rows, columns))
        require(count == rows * columns * 2,
                "tensor %s byte count mismatch" % located)
        seen.add(located)
    extras = sorted(set(safetensors.names()) - seen)
    return {"ok": True, "tensors": len(tensor_specs(geometry)),
            "extras": extras}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="drafter checkpoint dir (config.json plus a "
                             "[possibly sharded] model.safetensors)")
    parser.add_argument("--output", required=True, type=Path,
                        help="pack dir to mint (config.json, model."
                             "safetensors, manifest.json)")
    parser.add_argument("--model-revision", required=True,
                        help="40 lowercase hex chars; pinned into the manifest")
    args = parser.parse_args()
    try:
        receipt = mint(args.checkpoint, args.output, args.model_revision)
    except PackFailure as error:
        print("glm52_dspark_stagepack failed: %s" % error, file=sys.stderr)
        return 1
    print("glm52_dspark_stagepack minted %s tensors=%d extras=%d "
          "consolidated=%s safetensors_gib=%.2f"
          % (args.output, receipt["inventory"]["tensor_count"],
             len(receipt["inventory"]["extras"]),
             receipt["inventory"]["consolidated"],
             receipt["files"][SAFETENSORS_NAME]["bytes"] / 2 ** 30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
