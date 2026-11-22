#!/usr/bin/env python3
"""tools/drafter_pack.py on small synthetic checkpoints of each drafter's shape.

For every drafter it checks that each packed tensor is the source's bytes exactly, that the target's embedding, head
and norm are packed under target.* names with their source recorded, that the left-out tensors and version scalars
are where the receipt says, and that the pack refuses: a tensor no rule names, a layer past the config's count, a
dtype the drafter does not pack, a missing target tensor, a drafter tensor that the checkpoint's index places in a file
that is not there, a source whose .sha256 sidecar or --expect-sha256 disagrees, and a --target given to the drafter
whose target lives in its own checkpoint.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "drafter_pack.py"
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "I64": 8}
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def write_safetensors(path, tensors):
    header, blobs, cursor = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        count = 1
        for extent in shape:
            count *= extent
        if data is None:
            data = os.urandom(count * DTYPE_BYTES[dtype])
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + len(data)]}
        blobs.append(data)
        cursor += len(data)
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return {name: blob for name, blob in zip(tensors, blobs)}


def read_pack(path):
    data = path.read_bytes()
    magic, version, manifest_bytes = struct.unpack_from("<IIQ", data, 0)
    manifest = json.loads(data[16:16 + manifest_bytes])
    base = 16 + manifest_bytes
    base += (-base) % 128
    return magic, version, manifest, {name: data[base + entry["offset"]:base + entry["offset"] + entry["bytes"]]
                                      for name, entry in manifest["tensors"].items()}


def run(*arguments):
    return subprocess.run([sys.executable, str(TOOL)] + [str(argument) for argument in arguments],
                          capture_output=True, text=True, timeout=120)


def k31_checkpoint(directory, extra=None, layer=1):
    (directory / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": ["K3DSparkModel"], "num_hidden_layers": 2,
                                                       "hidden_size": 8, "target_layer_ids": [1, 2, 3, 4, 5]}))
    tensors = {"context_norm.weight": ("BF16", [8], None), "context_proj.weight": ("BF16", [8, 40], None),
               "final_norm.weight": ("BF16", [8], None), "layers.0.input_layernorm.weight": ("BF16", [8], None),
               f"layers.{layer}.self_attn.q_a_proj.weight": ("BF16", [4, 8], None),
               "layers.1.mlp.down_proj.weight": ("BF16", [8, 16], None),
               "markov_head.markov_w1.weight": ("BF16", [32, 4], None), "markov_head.markov_w2.weight": ("BF16", [32, 4], None),
               "confidence_head.proj.weight": ("BF16", [1, 12], None), "confidence_head.proj.bias": ("BF16", [1], None)}
    tensors.update(extra or {})
    return write_safetensors(directory / "model.safetensors", tensors)


def k31_target(path, drop=None):
    tensors = {"language_model.lm_head.weight": ("BF16", [32, 8], None),
               "language_model.model.embed_tokens.weight": ("BF16", [32, 8], None),
               "language_model.model.norm.weight": ("BF16", [8], None),
               "language_model.model.output_attn_res_norm.weight": ("BF16", [8], None),
               "language_model.model.output_attn_res_proj.weight": ("BF16", [1, 8], None)}
    tensors.pop(drop, None)
    return write_safetensors(path, tensors)


def test_k31(scratch):
    checkpoint, target, out = scratch / "k31", scratch / "k31-target" / "model-00094.safetensors", scratch / "k31-out"
    sources = k31_checkpoint(checkpoint)
    sources.update(k31_target(target))
    result = run("--drafter", "k31dspark", "--checkpoint", checkpoint, "--target", target, "--revision", "r1", "--out", out)
    check(result.returncode == 0, f"k31dspark packs: {result.stderr}")
    pack = out / "k31dspark.bf16.tp1.rank0.sp"
    magic, version, manifest, payloads = read_pack(pack)
    check(magic == 0x4B505244 and version == 1 and manifest["format"]["drafter"] == "k31dspark", "k31dspark: header and format")
    check(set(payloads) == {name for name in sources if not name.startswith("language_model.")} |
          {"target.embed", "target.head", "target.norm"}, f"k31dspark: packed names {sorted(payloads)}")
    for name, blob in payloads.items():
        source = manifest["tensors"][name]["source"]
        check(blob == sources[source], f"k31dspark: {name} is not {source} byte for byte")
    check(manifest["tensors"]["target.head"]["source"] == "language_model.lm_head.weight" and
          manifest["tensors"]["context_proj.weight"]["shape"] == [8, 40], "k31dspark: target source and shape recorded")
    receipt = json.loads((out / "k31dspark.bf16.tp1.rank0.sp.receipt.json").read_text())
    check(len(receipt["skipped"]) == 2 and receipt["tensors"] == len(payloads) and len(receipt["sources"]) == 2,
          "k31dspark: the receipt names the left-out tensors and both sources")
    check((out / "k31dspark.bf16.tp1.rank0.sp.sha256").read_text().split()[0] == receipt["sha256"], "k31dspark: sidecar")
    unknown = scratch / "k31-unknown"
    k31_checkpoint(unknown, {"extra.weight": ("BF16", [2], None)})
    result = run("--drafter", "k31dspark", "--checkpoint", unknown, "--target", target, "--revision", "r1", "--out", scratch / "o1")
    check(result.returncode == 1 and "extra.weight is neither packed nor left out" in result.stderr, "k31dspark: an unnamed tensor refuses")
    deep = scratch / "k31-deep"
    k31_checkpoint(deep, layer=2)
    result = run("--drafter", "k31dspark", "--checkpoint", deep, "--target", target, "--revision", "r1", "--out", scratch / "o2")
    check(result.returncode == 1 and "layer 2 is past num_hidden_layers = 2" in result.stderr, "k31dspark: a layer past the count refuses")
    wide = scratch / "k31-f32"
    k31_checkpoint(wide, {"final_norm.weight": ("F32", [8], None)})
    result = run("--drafter", "k31dspark", "--checkpoint", wide, "--target", target, "--revision", "r1", "--out", scratch / "o3")
    check(result.returncode == 1 and "dtype F32 is not one this drafter packs" in result.stderr, "k31dspark: an unpacked dtype refuses")
    short = scratch / "k31-short" / "model-00094.safetensors"
    k31_target(short, drop="language_model.lm_head.weight")
    result = run("--drafter", "k31dspark", "--checkpoint", checkpoint, "--target", short, "--revision", "r1", "--out", scratch / "o4")
    check(result.returncode == 1 and "target tensors missing: language_model.lm_head.weight" in result.stderr,
          "k31dspark: a missing target tensor refuses")
    (checkpoint / "model.safetensors.sha256").write_text("0" * 64 + "  model.safetensors\n")
    result = run("--drafter", "k31dspark", "--checkpoint", checkpoint, "--target", target, "--revision", "r1", "--out", scratch / "o5")
    check(result.returncode == 1 and "its receipt says" in result.stderr, "k31dspark: a sidecar that disagrees refuses")
    (checkpoint / "model.safetensors.sha256").unlink()
    result = run("--drafter", "k31dspark", "--checkpoint", checkpoint, "--target", target, "--revision", "r1", "--out", scratch / "o6",
                 "--expect-sha256", f"{target.name}={'1' * 64}")
    check(result.returncode == 1 and "its receipt says" in result.stderr, "k31dspark: an --expect-sha256 that disagrees refuses")


def test_dsv41(scratch):
    checkpoint, out = scratch / "dsv41", scratch / "dsv41-out"
    (checkpoint / "inference").mkdir(parents=True)
    (checkpoint / "inference" / "config.json").write_text(json.dumps({"hidden_size": 8, "n_routed_experts": 2}))
    first = write_safetensors(checkpoint / "model-00002-of-00048.safetensors",
                              {"embed.weight": ("BF16", [32, 8], None), "image_start": ("BF16", [8], None)})
    second = write_safetensors(checkpoint / "model-00043-of-00048.safetensors",
                               {"head.weight": ("BF16", [32, 8], None), "norm.weight": ("BF16", [8], None),
                                "mtp.0.attn.wq_a.weight": ("F8_E4M3", [4, 8], None), "mtp.0.attn.wq_a.scale": ("F8_E8M0", [1, 1], None),
                                "mtp.0.attn.attn_sink": ("F32", [2], None), "mtp.0.ffn.experts.0.w1.weight": ("I8", [4, 4], None),
                                "mtp.0.ffn.experts.0.w1.scale": ("F8_E8M0", [4, 1], None), "mtp.0.norm.weight": ("BF16", [8], None)})
    index = {"weight_map": {name: path for path, names in (("model-00002-of-00048.safetensors", first),
                                                         ("model-00043-of-00048.safetensors", second)) for name in names}}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    result = run("--drafter", "dsv41dspark", "--checkpoint", checkpoint, "--revision", "r2", "--out", out)
    check(result.returncode == 0, f"dsv41dspark packs: {result.stderr}")
    _, _, manifest, payloads = read_pack(out / "dsv41dspark.mxfp4.tp1.rank0.sp")
    sources = dict(first, **second)
    check("image_start" not in payloads and "target.embed" in payloads and "embed.weight" not in payloads, "dsv41dspark: names")
    kinds = {name: entry["kind"] for name, entry in manifest["tensors"].items()}
    check(kinds["mtp.0.attn.wq_a.weight"] == "fp8" and kinds["mtp.0.attn.wq_a.scale"] == "e8m0" and
          kinds["mtp.0.ffn.experts.0.w1.weight"] == "mxfp4_e2m1" and kinds["mtp.0.attn.attn_sink"] == "f32",
          f"dsv41dspark: kinds {kinds}")
    for name, blob in payloads.items():
        check(blob == sources[manifest["tensors"][name]["source"]], f"dsv41dspark: {name} byte for byte")
    result = run("--drafter", "dsv41dspark", "--checkpoint", checkpoint, "--target", checkpoint / "model-00002-of-00048.safetensors",
                 "--revision", "r2", "--out", scratch / "o7")
    check(result.returncode == 1 and "takes no --target" in result.stderr, "dsv41dspark: a --target refuses")
    index["weight_map"]["mtp.0.ffn.experts.1.w1.weight"] = "model-00044-of-00048.safetensors"
    index["weight_map"]["mtp.0.ffn.experts.1.w1.scale"] = "model-00044-of-00048.safetensors"
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    result = run("--drafter", "dsv41dspark", "--checkpoint", checkpoint, "--revision", "r2", "--out", scratch / "o8")
    check(result.returncode == 1 and "mtp.0.ffn.experts.1.w1.scale is in model-00044-of-00048.safetensors, which is not in"
          in result.stderr and "(2 drafter tensors are absent)" in result.stderr, f"dsv41dspark: a file the index names is absent: {result.stderr}")


def test_qwen(scratch):
    checkpoint, target, out = scratch / "dspine", scratch / "qwen-head" / "head.safetensors", scratch / "dspine-out"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["DFlashDraftModel"], "num_hidden_layers": 1}))
    sources = write_safetensors(checkpoint / "model.safetensors",
                                {"draft_block_causal_version": ("I64", [], struct.pack("<q", 2)),
                                 "selected_prefix_bridge.terminal_disabled_version": ("I64", [], struct.pack("<q", 1)),
                                 "fc.weight": ("BF16", [8, 40], None), "hidden_norm.weight": ("BF16", [8], None),
                                 "norm.weight": ("BF16", [8], None), "layers.0.self_attn.q_proj.weight": ("BF16", [8, 8], None),
                                 "selected_prefix_bridge.codebook": ("BF16", [32, 4], None),
                                 "selected_prefix_bridge.gate.bias": ("BF16", [4], None)})
    sources.update(write_safetensors(target, {"model.language_model.embed_tokens.weight": ("BF16", [32, 8], None),
                                              "lm_head.weight": ("BF16", [32, 8], None),
                                              "model.language_model.norm.weight": ("F32", [8], None)}))
    result = run("--drafter", "qwen27bdspine", "--checkpoint", checkpoint, "--target", target, "--revision", "r3", "--out", out)
    check(result.returncode == 0, f"qwen27bdspine packs: {result.stderr}")
    _, _, manifest, payloads = read_pack(out / "qwen27bdspine.bf16.tp1.rank0.sp")
    check(manifest["config"]["scalars"] == {"draft_block_causal_version": 2, "selected_prefix_bridge.terminal_disabled_version": 1}
          and "draft_block_causal_version" not in payloads, "qwen27bdspine: version scalars go into the config")
    check(manifest["tensors"]["target.norm"]["kind"] == "f32", "qwen27bdspine: the target's F32 norm keeps its dtype")
    for name, blob in payloads.items():
        check(blob == sources[manifest["tensors"][name]["source"]], f"qwen27bdspine: {name} byte for byte")


def main():
    with tempfile.TemporaryDirectory(prefix="drafter-pack-") as directory:
        scratch = Path(directory)
        test_k31(scratch)
        test_dsv41(scratch)
        test_qwen(scratch)
    for failure in FAILURES:
        print(f"FAIL {failure}")
    if FAILURES:
        return 1
    print("PASS drafter packs: byte-exact tensors under their names, target.* for the target's embedding, head and norm, "
          "version scalars in the config, and refusals for unnamed tensors, layers past the count, unpacked dtypes, "
          "missing target tensors, drafter files the index names but the checkpoint lacks, disagreeing receipts and a "
          "stray --target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
