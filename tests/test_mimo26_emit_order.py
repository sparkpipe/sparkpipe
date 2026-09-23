#!/usr/bin/env python3
"""mimo26 emission span order: record-major is the verify-clean default.

The file-major emitter appends each plane's spans in shard-visit order, so a
multi-span record whose tensors sit in several shards gets its plane bytes
permuted against the plan order the verifier uses (the production incident:
mimo26pro rank6 and mimo26flash rank0 failed --assemble's byte-compare at
expert entries). This test reshards the synthetic checkpoint so expert
tensors interleave across shards, then proves:

  * default emission (record-major) round-trips emit -> assemble -> verify
  * file-major emission produces planes whose verify FAILS
  * mimo26_reorder_planes.py repairs those planes locally and verify passes
"""
from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import mimo26_stagepack as packer  # noqa: E402
sys.path.insert(0, str(ROOT / "tools" / "stagepack_gap"))
import mimo26_reorder_planes as repair  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
import test_mimo26_stagepack as fixture  # noqa: E402

PATTERN = bytes(range(251, 256)) * 8192


def reshard(checkpoint: Path, buckets: int) -> None:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    original = checkpoint / "model_pp0_ep0_shard0.safetensors"
    with original.open("rb") as file:
        header_len = struct.unpack("<Q", file.read(8))[0]
        original_header = json.loads(file.read(header_len))
    names = sorted(weight_map)
    assignment = {}
    groups = {f"model_pp0_ep{bucket}_shard0.safetensors": []
              for bucket in range(buckets)}
    import re
    for position, name in enumerate(names):
        expert = re.search(r"\.experts\.(\d+)\.", name)
        bucket = int(expert.group(1)) % buckets if expert else position % buckets
        shard = f"model_pp0_ep{bucket}_shard0.safetensors"
        groups[shard].append(name)
        assignment[name] = shard
    original.unlink()
    import random
    for shard, shard_names in groups.items():
        header = {}
        cursor = 0
        for name in shard_names:
            meta = original_header[name]
            dtype, shape = meta["dtype"], meta["shape"]
            elements = math.prod(shape)
            per = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}[dtype]
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [cursor, cursor + elements * per]}
            cursor += elements * per
        header_json = json.dumps(header).encode()
        with (checkpoint / shard).open("wb") as file:
            file.write(struct.pack("<Q", len(header_json)))
            file.write(header_json)
            for name in shard_names:
                meta = original_header[name]
                elements = math.prod(meta["shape"])
                per = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}[meta["dtype"]]
                file.write(random.Random(name).randbytes(elements * per))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": assignment}))


def permuted(source, records) -> bool:
    for record in records:
        if len(record.spans) < 2:
            continue
        shards = [source.weight_map[span.name] for span in record.spans]
        if sorted(set(shards)) != [shards[0]] * len(shards):
            return True
    return False


def main() -> int:
    packer.ARMS["pro"] = dict(fixture.MINI)
    tmp = Path(tempfile.mkdtemp())
    checkpoint = tmp / "ckpt"
    checkpoint.mkdir()
    fixture.build_checkpoint(checkpoint)
    reshard(checkpoint, 3)
    source = packer.SafetensorsSource(checkpoint)
    packer.check_source("pro", source)
    records = packer.build_plan("pro", source.config, 2, 0)
    assert permuted(source, records), "fixture failed to interleave shards"

    good = tmp / "record-major"
    args = fixture.Args(checkpoint=str(checkpoint), tp=2, rank=0,
                        out=str(good / "rank0.sp"),
                        stage_dir=str(good / "stage"))
    args.emit = True
    assert packer.do_emit(args) == 0
    args.emit, args.assemble = False, True
    assert packer.do_assemble(args) == 0
    args.assemble, args.verify = False, True
    assert packer.do_verify(args) == 0

    bad = tmp / "file-major"
    args = fixture.Args(checkpoint=str(checkpoint), tp=2, rank=0,
                        out=str(bad / "rank0.sp"),
                        stage_dir=str(bad / "stage"))
    stage_dir = bad / "stage"
    stage_dir.mkdir(parents=True)
    reader = packer.SourceReader(source)
    packer.emit_file_major(source, records, stage_dir, reader)
    args.assemble = True
    assert packer.do_assemble(args) == 0
    args.assemble, args.verify = False, True
    try:
        packer.do_verify(args)
        raise AssertionError("file-major planes verified; expected permutation")
    except packer.PackFailure:
        pass

    layout = repair.append_layout(source, records)
    fixed, skipped = repair.reorder(stage_dir, layout, source)
    assert fixed > 0, "repair found nothing to fix; fixture lost the permutation"
    args.verify, args.assemble = False, True
    assert packer.do_assemble(args) == 0
    args.assemble, args.verify = False, True
    assert packer.do_verify(args) == 0
    print(f"emit-order: record-major clean, file-major permuted {fixed} planes, "
          f"repair restored verify ({skipped} already ordered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
