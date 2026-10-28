#!/usr/bin/env python3
"""k3_pack_append apply: existing tensors and expert ranges keep their bytes, the added tensors land intact, and the digest matches."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from k3_pack import ALIGN, KIND_BF16, MAGIC, VERSION  # noqa: E402
from spark_named_pack import PayloadWriter, assemble  # noqa: E402

EXPERTS_MAGIC = 0x58504557


def build(path, tensors):
    payload = Path(str(path) + ".payload")
    writer = PayloadWriter(payload, json.dumps({"test": str(path)}), align=ALIGN)
    for name, data in tensors:
        writer.add(name, data, KIND_BF16, [len(data) // 2])
    writer.close()
    assemble(path, payload, MAGIC, VERSION, ALIGN, {"name": "test"}, {"layers": 1}, writer.manifest)
    return writer.manifest


def base_of(path):
    with open(path, "rb") as handle:
        _, _, length = struct.unpack("<IIQ", handle.read(16))
        manifest = json.loads(handle.read(length))
    base = 16 + length
    return manifest, base + (-base) % ALIGN


def main():
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        pack, additions = directory / "k3.stage0.rank00.pack", directory / "k3.additions.rank00.pack"
        old = [("model.layers.0.attn_norm_weight", bytes(range(256)) * 3), ("model.layers.1.expert_w1_weight", bytes(reversed(range(256))) * 40)]
        manifest = build(pack, old)
        added = [(f"model.layers.1.mla_q_b_weight", b"\x11\x22" * 700), (f"model.layers.1.mla_kv_b_key_full_weight", b"\x33\x44" * 900),
                 (f"model.layers.1.mla_kv_b_value_full_weight", b"\x55\x66" * 300)]
        build(additions, added)
        _, old_base = base_of(pack)
        expert = manifest["model.layers.1.expert_w1_weight"]
        record = struct.pack("<IIIIQQ", 1, 0, 1, 0, old_base + expert["offset"] + 256, 2048) + hashlib.md5(b"x").digest()
        Path(str(pack) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, 2, 1, 0) + record)
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack_append.py"), "apply", "--pack", str(pack), "--additions", str(additions)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print("FAIL apply:", result.stderr.strip()[-400:])
            raise SystemExit(1)
        staged = Path(str(pack) + ".append")
        new_manifest, new_base = base_of(staged)
        blob = staged.read_bytes()
        for name, data in old + added:
            entry = new_manifest["tensors"].get(name)
            if entry is None or blob[new_base + entry["offset"]:new_base + entry["offset"] + entry["bytes"]] != data:
                failures.append(f"{name} does not read back its bytes from the appended pack")
        for name, _ in old:
            if new_manifest["tensors"][name]["offset"] != manifest[name]["offset"]:
                failures.append(f"{name} moved inside the payload")
        experts = Path(str(staged) + ".experts").read_bytes()
        offset, length = struct.unpack_from("<QQ", experts, 16 + 16)
        original = pack.read_bytes()
        if blob[offset:offset + length] != original[old_base + expert["offset"] + 256:old_base + expert["offset"] + 256 + length]:
            failures.append("the expert range does not point at the same bytes after the manifest grew")
        if Path(str(staged) + ".sha256").read_text().split()[0] != hashlib.sha256(blob).hexdigest():
            failures.append("the digest file does not match the appended pack")
        again = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack_append.py"), "apply", "--pack", str(staged), "--additions", str(additions)],
                               capture_output=True, text=True, timeout=120)
        if again.returncode == 0:
            failures.append("a pack that already carries the tensors must be refused")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS k3_pack_append: existing tensors keep their payload offsets and bytes, expert ranges follow the grown manifest, the added tensors read back, the digest matches, and a second append is refused")


if __name__ == "__main__":
    main()
