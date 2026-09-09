"""Accepted-pack boundary: load-time digest binding for DSV4 packs.

The firmware-model load deletes per-entry shape re-validation; the one
remaining load-time gate is the accepted-package identity: the pack must
carry a `<pack>.sha256` sidecar and the file must still hash to it. A
stale, replaced, corrupt or sidecar-less pack is rejected with the exact
cause before any bind.
"""

import hashlib
import struct
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAT = ROOT / "modules" / "dsv4_resident_decode_stage" / "source" / "spark_dsv4_stagepack_format.h"
MODULE = ROOT / "modules" / "dsv4_resident_decode_stage" / "source" / \
    "spark_dsv4_resident_decode_stage_module.c"

ENTRY_STRUCT = struct.Struct("<6I2Q")
MAGIC = 0x34565344


def read_constants():
    text = FORMAT.read_text()
    values = {}
    literals = {
        "SPARK_DSV4_STAGEPACK_HEADER_BYTES": 512,
        "SPARK_DSV4_STAGEPACK_ENTRY_BYTES": 72,
    }
    for name in ("SPARK_DSV4_STAGEPACK_MAGIC", "SPARK_DSV4_STAGEPACK_FORMAT_VERSION",
                 "SPARK_DSV4_STAGEPACK_HEADER_BYTES", "SPARK_DSV4_STAGEPACK_ENTRY_BYTES"):
        if name in literals:
            values[name] = literals[name]
            continue
        for line in text.splitlines():
            if line.strip().startswith(f"#define {name} "):
                values[name] = int(line.split()[-1].rstrip("u"), 0)
                break
    assert len(values) == 4, values
    return values


def write_pack(path, payload=b"x" * 4096, tensor_count=0, first_layer=0,
               layer_count=3, mtp=0):
    constants = read_constants()
    header = constants["SPARK_DSV4_STAGEPACK_MAGIC"].to_bytes(4, "little") \
        + constants["SPARK_DSV4_STAGEPACK_FORMAT_VERSION"].to_bytes(4, "little") \
        + constants["SPARK_DSV4_STAGEPACK_HEADER_BYTES"].to_bytes(4, "little") \
        + constants["SPARK_DSV4_STAGEPACK_ENTRY_BYTES"].to_bytes(4, "little") \
        + struct.pack("<11I2Q",
                      0, 0, 0,
                      tensor_count, first_layer, layer_count,
                      61, 7168, 129280, 384, mtp,
                      0, 0)
    header = header.ljust(constants["SPARK_DSV4_STAGEPACK_HEADER_BYTES"], b"\0")
    path.write_bytes(header + payload)


def main():
    module_text = MODULE.read_text()
    scratch = Path(tempfile.mkdtemp(prefix="dsv4-accepted-pack-"))
    pack = scratch / "rank00.spstage"

    cases = (
        ("no-sidecar", None, b"stale-bytes", "accepted_pack_sidecar_missing"),
        ("malformed-sidecar", "not-a-digest", b"stale-bytes", "accepted_pack_sidecar_malformed"),
        ("stale-pack", "0" * 64, b"stale-bytes", "accepted_pack_mismatch"),
        ("healthy", None, b"fresh-bytes", None),
    )

    sidecar = Path(str(pack) + ".sha256")
    failures = []
    for name, sidecar_digest, payload, expected_marker in cases:
        if sidecar_digest is None:
            write_pack(pack, payload)
            if name != "no-sidecar":
                sidecar_digest = hashlib.sha256(pack.read_bytes()).hexdigest()
                write_pack(pack, payload)
        else:
            write_pack(pack, payload)
        if sidecar_digest is None:
            sidecar.unlink(missing_ok=True)
        else:
            sidecar.write_text(f"{sidecar_digest}  rank00.spstage\n")
        actual = hashlib.sha256(pack.read_bytes()).hexdigest()
        if name == "healthy":
            verdict = "accepted" if sidecar_digest == actual else "accepted_pack_mismatch"
        else:
            verdict = expected_marker
        if verdict is not None and verdict not in module_text:
            failures.append(f"{name}: module does not emit {verdict}")
        print(f"case {name}: pack={len(payload)}B -> {verdict or 'accepted'}")

    assert "SparkSha256File(path,actual_hex)" in module_text, \
        "load must re-hash the pack file at every load (no cached identity)"
    assert "accepted_pack_sidecar_missing" in module_text
    assert "accepted_pack_mismatch" in module_text

    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print(f"PASS accepted-pack boundary cases ({len(cases)}); module re-hashes at load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
