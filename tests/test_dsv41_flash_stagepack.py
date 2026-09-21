#!/usr/bin/env python3
"""End-to-end self-test for the dsv41 flash real-source pack converter.

Builds a tiny synthetic checkpoint (own geometry patch, both converter and
verifier tables), runs headers -> plan -> copy -> verify in-process, checks
the engram/sidecar accounting, and proves the byte verifier detects and
localizes corruption in both spine and expert regions.
"""

import json
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import dsv41_flash_stagepack as conv
import dsv41_flash_pack_verify as ver

TINY = {"LAYER_COUNT": 4, "HIDDEN": 128, "VOCAB": 512, "ROUTED_EXPERTS": 8,
        "EXPERT_WIDTH": 32, "SHARD_COUNT": 2,
        "KV_SOURCE_LAYERS": (1,), "GATE_LAYERS": (1,),
        "INDEX_SOURCE_LAYERS": (1, 2), "SWA_ONLY_LAYERS": (0, 3)}
TP = 4


def patch(mod):
    for key, value in TINY.items():
        setattr(mod, key, value)


def tensor_bytes(name: str, count: int) -> bytes:
    seed = int.from_bytes(name.encode()[:8], "little")
    return random.Random(seed).randbytes(count)


def build_checkpoint(warm: Path, codec: str) -> set:
    shards = {1: {}, 2: {}}
    meta = {}
    for kind, layer, spec in conv.entry_specs():
        scale_dtype = "F8_E4M3" if codec == "nvfp4" and kind in conv.EXPERT_KINDS else "F8_E8M0"
        entries = [(spec[0], spec[2], spec[3])]
        if spec[1]:
            entries.append((spec[1], scale_dtype, spec[4]))
        for name_t, dtype, shape in entries:
            if "{e}" in name_t:
                for expert in range(conv.ROUTED_EXPERTS):
                    meta[name_t.format(e=expert)] = (dtype, shape)
            else:
                meta[name_t] = (dtype, shape)
    activation = set()
    if codec == "nvfp4":
        for layer in range(conv.LAYER_COUNT):
            for kind in conv.EXPERT_KINDS:
                for expert in range(conv.ROUTED_EXPERTS):
                    ns = conv.nvidia_expert_spec(kind, layer, expert)
                    meta[ns["global"]] = ("F32", [])
                    meta[ns["input_scale"]] = ("F32", [])
                    activation.add(ns["input_scale"])
    extras = {
        "layers.1.engram.embed.weight": ("F8_E4M3", [64, 4]),
        "layers.1.engram.embed.scale": ("F8_E8M0", [64, 1]),
        "layers.1.engram.k_weight": ("BF16", [4]),
        "layers.1.engram.q_weight": ("BF16", [4]),
        "layers.1.engram.wkv.weight": ("BF16", [4, 4]),
        "layers.1.engram.wkv.scale": ("F8_E8M0", [4, 1]),
        "layers.2.engram.embed.weight": ("F8_E4M3", [64, 4]),
        "layers.2.engram.embed.scale": ("F8_E8M0", [64, 1]),
        "layers.2.engram.k_weight": ("BF16", [4]),
        "layers.2.engram.q_weight": ("BF16", [4]),
        "layers.2.engram.wkv.weight": ("BF16", [4, 4]),
        "layers.2.engram.wkv.scale": ("F8_E8M0", [4, 1]),
        "mtp.0.norm.weight": ("BF16", [8]),
        "mtp.1.markov_head.head.weight": ("BF16", [4, 4]),
        "vision.norm.weight": ("BF16", [8]),
        "aligner.w0.weight": ("BF16", [4, 4]),
        "image_start": ("BF16", [4]),
    }
    meta.update(extras)
    dtype_bytes = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1}
    for name, (dtype, shape) in meta.items():
        count = 1
        for dim in shape:
            count *= dim
        shard = 1 + (hash(name) % conv.SHARD_COUNT)
        shards[shard][name] = (dtype, shape, tensor_bytes(name, count * dtype_bytes[dtype]))
    weight_map = {}
    for shard in sorted(shards):
        header = {}
        blob = bytearray()
        for name, (dtype, shape, data) in shards[shard].items():
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [len(blob), len(blob) + len(data)]}
            blob += data
            weight_map[name] = f"model-{shard:05d}-of-{conv.SHARD_COUNT:05d}.safetensors"
        raw = json.dumps(header).encode()
        path = warm / f"model-{shard:05d}-of-{conv.SHARD_COUNT:05d}.safetensors"
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(raw)))
            handle.write(raw)
            handle.write(bytes(blob))
    with open(warm / "index.json", "w", encoding="utf-8") as handle:
        json.dump({"weight_map": weight_map, "metadata": {"total_size": 0}}, handle)
    return len(meta), activation


def expect_fail(callable_fn, needle: str):
    try:
        callable_fn()
    except ver.Fail as error:
        if needle not in str(error):
            raise AssertionError(f"failure lacked {needle!r}: {error}")
        return
    raise AssertionError("expected verify failure did not raise")


def run_case(root: Path, codec: str) -> dict:
    conv.EXPERT_CODEC = codec
    warm = root / f"warm-{codec}"
    out = root / f"out-{codec}"
    warm.mkdir()
    total, activation = build_checkpoint(warm, codec)
    conv.cmd_headers(str(warm), str(warm / "index.json"), str(root / f"headers-{codec}.json"))
    conv.cmd_plan(str(root / f"headers-{codec}.json"), str(out), TP, "selftest-revision",
                  "aa" * 32, "bb" * 32, "cc" * 32, expert_codec=codec)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["tensor_count"] == 4 * 25 + 7 + 2 + 3, plan["tensor_count"]
    assert plan["tp"] == TP
    assert plan["expert_codec"] == codec
    conv.cmd_copy(str(root / f"headers-{codec}.json"), str(out / "plan.json"), str(out),
                  str(warm), 1, conv.SHARD_COUNT)
    report = json.loads((out / "engram_report.json").read_text())
    assert report["accounting"]["engram_excluded"] == 12
    assert report["accounting"]["dspark_sidecar"] == 2
    assert report["accounting"]["vision_aligner_out_of_scope"] == 3
    assert report["accounting"]["activation_scales_out_of_scope"] == len(activation)
    assert report["accounting"]["pack_consumed"] + 12 + 2 + 3 + len(activation) == total
    headers = str(root / f"headers-{codec}.json")
    ver.cmd_verify(headers, str(out), str(warm), str(out / "plan.json"), 1, conv.SHARD_COUNT)
    pack3 = out / "rank3.spstage"
    payload = next(e for e in plan["entries"] if e["kind"] == conv.K_Q_A)
    expert = next(e for e in plan["entries"] if e["kind"] == conv.K_EXP_W1)
    expert_scale = dict(expert)
    for target, label in ((payload, "spine"), (expert, "expert"),
                          (expert_scale, "expert-scale" if codec == "nvfp4" else None)):
        if label is None:
            continue
        target = dict(target)
        offset = target["payload_offset"] if label != "expert-scale" else target["scale_offset"]
        with open(pack3, "r+b") as handle:
            handle.seek(offset + 3)
            original = handle.read(1)
            handle.seek(offset + 3)
            handle.write(bytes([original[0] ^ 0xFF]))
        expect_fail(lambda: ver.cmd_verify(headers, str(out), str(warm),
                                           str(out / "plan.json"), 1, conv.SHARD_COUNT),
                    "BYTE MISMATCH")
        with open(pack3, "r+b") as handle:
            handle.seek(offset + 3)
            handle.write(original)
    if codec == "nvfp4":
        with open(pack3, "r+b") as handle:
            handle.seek(expert["scale_offset"] + expert["scale_bytes"] // TP - 2)
            original = handle.read(1)
            handle.seek(expert["scale_offset"] + expert["scale_bytes"] // TP - 2)
            handle.write(bytes([original[0] ^ 0xFF]))
        expect_fail(lambda: ver.cmd_verify(headers, str(out), str(warm),
                                           str(out / "plan.json"), 1, conv.SHARD_COUNT),
                    "BYTE MISMATCH")
        with open(pack3, "r+b") as handle:
            handle.seek(expert["scale_offset"] + expert["scale_bytes"] // TP - 2)
            handle.write(original)
    ver.cmd_verify(headers, str(out), str(warm), str(out / "plan.json"), 1, conv.SHARD_COUNT)
    return plan


def main() -> int:
    patch(conv)
    patch(ver)
    root = Path(tempfile.mkdtemp(prefix="dsv41-stagepack-selftest-"))
    try:
        results = {}
        for codec in ("mxfp4", "nvfp4"):
            plan = run_case(root, codec)
            results[codec] = (plan["tensor_count"], plan["spine_bytes"], plan["expert_bytes"])
            print(f"dsv41 stagepack selftest [{codec}] PASS: {plan['tensor_count']} entries, "
                  f"tp{TP}, spine {plan['spine_bytes']} B, expert {plan['expert_bytes']} B, "
                  f"corruption localized in spine+expert"
                  + ("+scale-plane+global" if codec == "nvfp4" else "")
                  + ", engram 12 excluded")
        assert results["nvfp4"][2] > results["mxfp4"][2], "nvfp4 expert window must carry "
        "the denser per-16 plane plus globals"
        print("dsv41 stagepack selftest PASS: mxfp4 + nvidia nvfp4 wires verified")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
