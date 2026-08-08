#!/usr/bin/env python3
"""Round-trip the qwen36 stage pack converter against a synthetic checkpoint.

Builds a sparse fake checkpoint with the real tensor names and shapes (the
payload bytes are mostly holes - a 55 GB checkpoint that writes in a moment),
converts a two-GDN-layer slice, and parses the pack back against the format
header's rules. The checks that matter are the ones a shape table cannot see:

  * in_proj_b lands in GDN_BETA and in_proj_a in GDN_DECAY - the beta/decay
    swap is the highest-risk mapping in the converter
  * A_log and dt_bias arrive BF16 and leave F32, value-exact
  * the vision tower in the index is never referenced
  * a missing shard and a wrong shape are hard failures, not silent packs
  * the C synthesizer's own pack passes the same verifier (the two writers
    share no code, so agreement is evidence, not tautology)
"""
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import qwen36_stagepack as packer  # noqa: E402

SYNTHESIZER = ROOT / "modules/qwen36_resident_decode_stage/tools/qwen36_pack_synthesize.c"


def bf16(value: float) -> bytes:
    """Little-endian bf16 of an exactly-representable value (top half of f32)."""
    return struct.pack(">f", value)[:2][::-1]


def build_fake_checkpoint(directory: Path) -> dict:
    """A sparse one-shard checkpoint covering two GDN layers, with markers."""
    tensors = {}
    for layer in (1, 2):
        prefix = f"model.language_model.layers.{layer}."
        for name, shape in {
            "input_layernorm.weight": [5120],
            "post_attention_layernorm.weight": [5120],
            "mlp.gate_proj.weight": [17408, 5120],
            "mlp.up_proj.weight": [17408, 5120],
            "mlp.down_proj.weight": [5120, 17408],
            "linear_attn.in_proj_qkv.weight": [10240, 5120],
            "linear_attn.in_proj_z.weight": [6144, 5120],
            "linear_attn.in_proj_b.weight": [48, 5120],
            "linear_attn.in_proj_a.weight": [48, 5120],
            "linear_attn.out_proj.weight": [5120, 6144],
            "linear_attn.conv1d.weight": [10240, 1, 4],
            "linear_attn.A_log": [48],
            "linear_attn.dt_bias": [48],
            "linear_attn.norm.weight": [128],
        }.items():
            tensors[prefix + name] = shape
    # Present in the index, never referenced: the converter must not touch it.
    tensors["model.visual.patch_embed.proj.weight"] = [1152, 3, 16, 16]

    header = {}
    cursor = 0
    for name, shape in tensors.items():
        elements = math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [cursor, cursor + elements * 2]}
        cursor += elements * 2
    header_json = json.dumps(header).encode()
    shard = directory / "model-00001-of-00001.safetensors"
    data_start = 8 + len(header_json)
    with shard.open("wb") as file:
        file.write(struct.pack("<Q", len(header_json)))
        file.write(header_json)
        file.truncate(data_start + cursor)
        # Markers: beta rows are 0x11..., decay rows 0x22..., and A_log /
        # dt_bias carry known values for the f32 upcast check.
        def mark(name, fill):
            start = data_start + header[name]["data_offsets"][0]
            file.seek(start)
            file.write(fill * 64)
        for layer in (1, 2):
            prefix = f"model.language_model.layers.{layer}."
            mark(prefix + "linear_attn.in_proj_b.weight", b"\x11")
            mark(prefix + "linear_attn.in_proj_a.weight", b"\x22")
            a_log = b"".join(bf16(0.5 * (index + 1)) for index in range(48))
            dt_bias = b"".join(bf16(-0.25 * index) for index in range(48))
            file.seek(data_start + header[prefix + "linear_attn.A_log"]["data_offsets"][0])
            file.write(a_log)
            file.seek(data_start + header[prefix + "linear_attn.dt_bias"]["data_offsets"][0])
            file.write(dt_bias)

    index = {"metadata": {"total_size": cursor},
             "weight_map": {name: shard.name for name in tensors}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    config = {"text_config": {
        "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16,
        "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "intermediate_size": 17408, "vocab_size": 248320,
        "full_attention_interval": 4, "attn_output_gate": True,
        "tie_word_embeddings": False,
    }}
    (directory / "config.json").write_text(json.dumps(config))
    return {"header": header, "data_start": data_start}


def read_pack_entry(pack: Path, want_kind: int, want_layer: int) -> tuple:
    with pack.open("rb") as file:
        fields = packer.HEADER_STRUCT.unpack(file.read(packer.HEADER_BYTES))
        tensor_count = fields[4]
        directory = file.read(tensor_count * packer.ENTRY_BYTES)
    for index in range(tensor_count):
        entry = packer.ENTRY_STRUCT.unpack_from(directory, index * packer.ENTRY_BYTES)
        if entry[0] == want_kind and entry[1] == want_layer:
            return entry
    raise AssertionError(f"kind {want_kind} layer {want_layer} not in pack")


def main() -> int:
    failures = 0

    def check(condition, label):
        nonlocal failures
        if not condition:
            print(f"  FAIL {label}")
            failures += 1

    # The tool's wire constants must be the format header's, restated.
    header_text = (ROOT / "modules/qwen36_resident_decode_stage/source/spark_qwen36_stagepack_format.h").read_text()
    import re
    magic = re.search(r"STAGEPACK_MAGIC (0x[0-9a-fA-F]+)u", header_text)
    version = re.search(r"STAGEPACK_FORMAT_VERSION (\d+)u", header_text)
    check(int(magic.group(1), 16) == packer.MAGIC, "magic drifted from the format header")
    check(int(version.group(1)) == packer.FORMAT_VERSION, "format version drifted from the header")

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        build_fake_checkpoint(work)
        pack = work / "stage.qwen36sp"
        receipt = {}
        packer.convert(work, pack, 1, 2, receipt, dry_run=False)
        result = packer.verify(pack)
        check(result["tensor_count"] == 28, f"slice tensor count {result['tensor_count']}")

        entry = read_pack_entry(pack, packer.KIND_GDN_BETA, 1)
        with pack.open("rb") as file:
            file.seek(entry[6])
            check(file.read(64) == b"\x11" * 64, "in_proj_b did not land in GDN_BETA")
        entry = read_pack_entry(pack, packer.KIND_GDN_DECAY, 1)
        with pack.open("rb") as file:
            file.seek(entry[6])
            check(file.read(64) == b"\x22" * 64, "in_proj_a did not land in GDN_DECAY")
        entry = read_pack_entry(pack, packer.KIND_GDN_A_LOG, 2)
        check(entry[2] == packer.WEIGHT_F32 and entry[7] == 48 * 4,
              "A_log is not f32 in the pack")
        with pack.open("rb") as file:
            file.seek(entry[6])
            values = struct.unpack("<48f", file.read(48 * 4))
        check(all(values[i] == 0.5 * (i + 1) for i in range(48)),
              "A_log bf16->f32 upcast lost values")
        entry = read_pack_entry(pack, packer.KIND_GDN_DT_BIAS, 2)
        with pack.open("rb") as file:
            file.seek(entry[6])
            values = struct.unpack("<48f", file.read(48 * 4))
        check(all(values[i] == -0.25 * i for i in range(48)),
              "dt_bias bf16->f32 upcast lost values")

        # The last stage's inventory: MTP globals, the MTP layer at its
        # marker, and the second embedding copy - planned, not written.
        plan = packer.build_inventory(62, 2)
        check(len(plan) == 43, f"last-stage inventory {len(plan)} tensors, expected 43")
        mtp = [ref for ref in plan if ref.layer == packer.MTP_LAYER]
        check(len(mtp) == 11, f"MTP layer tensors {len(mtp)}, expected 11")
        embeddings = [ref for ref in plan if ref.kind == packer.KIND_EMBEDDING]
        check(len(embeddings) == 1, "head stage must carry exactly one embedding copy")
        check(any(ref.name == "mtp.fc.weight" for ref in plan), "MTP fc missing from the plan")

        # Failure paths: a missing shard and a wrong shape must stop the pack.
        (work / "model-00001-of-00001.safetensors").rename(work / "renamed.safetensors")
        try:
            packer.convert(work, work / "x.qwen36sp", 1, 2, {}, dry_run=False)
            check(False, "missing shard did not fail")
        except packer.PackFailure:
            pass
        (work / "renamed.safetensors").rename(work / "model-00001-of-00001.safetensors")
        # Wrong-shape probe: claim the beta projection is 96 rows.
        bad = work / "bad"
        bad.mkdir()
        build_fake_checkpoint(bad)
        with (bad / "model-00001-of-00001.safetensors").open("rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            shard_header = json.loads(file.read(length))
        key = "model.language_model.layers.1.linear_attn.in_proj_b.weight"
        shard_header[key]["shape"] = [96, 5120]
        encoded = json.dumps(shard_header).encode()
        with (bad / "model-00001-of-00001.safetensors").open("r+b") as file:
            file.write(struct.pack("<Q", len(encoded)))
            file.write(encoded)
        try:
            packer.convert(bad, bad / "x.qwen36sp", 1, 2, {}, dry_run=False)
            check(False, "wrong shape did not fail")
        except packer.PackFailure:
            pass

    # The C synthesizer's pack passes the same verifier.
    synthesizer_bin = Path("/tmp/qwen36_pack_synthesize_test")
    build = subprocess.run(
        ["cc", "-std=c11", "-O1",
         f"-I{ROOT}/include", f"-I{ROOT}/model-families/common/include",
         f"-I{ROOT}/model-families/qwen36/include",
         f"-I{ROOT}/modules/qwen36_resident_decode_stage/include",
         f"-I{ROOT}/modules/qwen36_resident_decode_stage/source",
         str(SYNTHESIZER), "-o", str(synthesizer_bin)],
        capture_output=True, text=True)
    check(build.returncode == 0, f"synthesizer build: {build.stderr[:200]}")
    if build.returncode == 0:
        synth_pack = Path("/tmp/qwen36_synth_slice.qwen36sp")
        run = subprocess.run(
            [str(synthesizer_bin), "--output", str(synth_pack),
             "--first-layer", "1", "--layer-count", "2", "--bf16"],
            capture_output=True, text=True)
        check(run.returncode == 0, f"synthesizer run: {run.stderr[:200]}")
        if run.returncode == 0:
            result = packer.verify(synth_pack)
            check(result["tensor_count"] == 28, "synthesizer pack failed the python verifier")

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("PASS qwen36 stage pack converter round-trip: checkpoint mapping, "
          "f32 upcast, inventory, failure paths, and the C synthesizer all agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
