#!/usr/bin/env python3
import random
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glm52_resident_stagepack as packer
import glm52_spine_source_verify as verifier

GLOBAL = 0xFFFFFFFF
SPINE = [(packer.K_EMBEDDING, GLOBAL, packer.PAYLOAD_BF16, packer.CODEC_BF16, packer.SCALE_NONE, 1, 4, 32, 256, 0),
         (packer.K_ATTN_NORM, 3, packer.PAYLOAD_BF16, packer.CODEC_BF16, packer.SCALE_NONE, 1, 1, 64, 128, 0),
         (packer.K_ROUTER_CORRECTION, 3, packer.PAYLOAD_F32, packer.CODEC_NONE, packer.SCALE_NONE, 1, 1, 16, 64, 0)]
EXPERTS = [(packer.K_EXPERT_UP_GATE, 3, packer.PAYLOAD_PACKED_WEIGHT, packer.CODEC_FP8, packer.SCALE_F32, 2, 4, 128, 1024, 32),
           (packer.K_EXPERT_DOWN, 3, packer.PAYLOAD_PACKED_WEIGHT, packer.CODEC_FP8, packer.SCALE_F32, 2, 4, 128, 1024, 32)]


def source_bytes(rows):
    rng = random.Random(11)
    return {(row[0], row[1]): (bytes(rng.getrandbits(8) for _ in range(row[8])),
                               bytes(rng.getrandbits(8) for _ in range(row[9]))) for row in rows}


def plan_for(rows, data, drop=None):
    plan = []
    for row in rows:
        if (row[0], row[1]) == drop:
            continue
        entry = packer.Entry(*row[:8])
        payload, scale = data[(row[0], row[1])]
        produce_payload = (lambda blob=payload: iter((blob[:len(blob) // 2], blob[len(blob) // 2:])))
        produce_scale = (lambda blob=scale: iter((blob,))) if scale else None
        plan.append(packer.PlanItem(entry, produce_payload, produce_scale, len(payload), len(scale), [f"t{row[0]}"]))
    return plan


def write_pack(path, rows, data, tp=(16, 5), flip=None):
    directory_offset = 256
    cursor = directory_offset + len(rows) * 64
    placed = []
    for row in rows:
        payload_offset = (cursor + 255) & ~255
        cursor = payload_offset + row[8]
        scale_offset = 0
        if row[9]:
            scale_offset = (cursor + 255) & ~255
            cursor = scale_offset + row[9]
        placed.append(row[:8] + (payload_offset, row[8], scale_offset, row[9]))
    header = struct.pack(verifier.HEADER_FORMAT, 0x32534C47, 3, 264, 64, 1, 0, len(rows), 16, 5, 0, 78, 78, 6144,
                         154880, 256, 1, 1, 1, tp[0], tp[1], directory_offset, cursor,
                         b"b4734de4".ljust(65, b"\0"), bytes(32), bytes([7]) * 32, bytes([8]) * 32)
    blob = bytearray(cursor)
    blob[:len(header)] = header
    for index, row in enumerate(placed):
        struct.pack_into(verifier.ENTRY_FORMAT, blob, directory_offset + index * 64, *row)
        payload, scale = data[(row[0], row[1])]
        blob[row[8]:row[8] + row[9]] = payload
        blob[row[10]:row[10] + row[11]] = scale
        if flip == (row[0], row[1]):
            blob[row[8] + row[9] - 1] ^= 0x01
    path.write_bytes(bytes(blob))


def main():
    failures = []
    rows = SPINE + EXPERTS
    data = source_bytes(rows)
    other = source_bytes(rows)
    other = {key: (bytes(reversed(value[0])), value[1]) for key, value in other.items()}
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        good = base / "good.glm52sp"
        write_pack(good, rows, data)
        report = verifier.verify(plan_for(rows, data), good, 16, 5)
        if report["result"] != "PASS" or report["spine_entries"] != 3 or report["expert_entries_skipped"] != 2:
            failures.append(f"a byte-exact pack did not pass: {report['failures']}")
        if report["header"]["stage_count"] != 16 or report["header"]["model_revision"] != "b4734de4":
            failures.append(f"header fields not reported: {report['header']}")
        expert_data = dict(data)
        expert_data.update({key: other[key] for key in ((22, 3), (23, 3))})
        grafted = base / "grafted.glm52sp"
        write_pack(grafted, rows, expert_data)
        if verifier.verify(plan_for(rows, data), grafted, 16, 5)["result"] != "PASS":
            failures.append("replaced expert entries failed a spine-only verify")
        for key in ((packer.K_EMBEDDING, GLOBAL), (packer.K_ROUTER_CORRECTION, 3)):
            flipped = base / f"flip-{key[0]}.glm52sp"
            write_pack(flipped, rows, data, flip=key)
            report = verifier.verify(plan_for(rows, data), flipped, 16, 5)
            if report["result"] != "FAIL" or not any(f"kind={key[0]}" in failure and "differs from the source" in failure
                                                     for failure in report["failures"]):
                failures.append(f"a flipped byte in kind {key[0]} was not reported: {report['failures']}")
        report = verifier.verify(plan_for(rows, data, drop=(packer.K_ATTN_NORM, 3)), good, 16, 5)
        if report["result"] != "FAIL" or not any("not in the packer's plan" in failure for failure in report["failures"]):
            failures.append("a pack entry absent from the plan was not reported")
        missing = base / "missing.glm52sp"
        write_pack(missing, [row for row in rows if row[0] != packer.K_ATTN_NORM], data)
        report = verifier.verify(plan_for(rows, data), missing, 16, 5)
        if not any("missing from the pack" in failure for failure in report["failures"]):
            failures.append("a planned spine entry missing from the pack was not reported")
        reshaped = [row if row[0] != packer.K_ATTN_NORM else row[:6] + (2, 32) + row[8:] for row in rows]
        report = verifier.verify(plan_for(reshaped, data), good, 16, 5)
        if not any("geometry differs" in failure for failure in report["failures"]):
            failures.append("a directory geometry mismatch was not reported")
        report = verifier.verify(plan_for(rows, data), good, 16, 4)
        if not any("verifying tp16 rank 4" in failure for failure in report["failures"]):
            failures.append("a pack of another rank was not reported")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS glm52 spine source verify: byte-exact spine passes, expert entries skipped, flipped bytes, "
          "missing/extra entries, geometry and rank mismatches reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
