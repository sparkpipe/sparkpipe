#!/usr/bin/env python3
"""ling_stagepack --resume: an interrupted emit continues from the journal.

The packer plan is built synthetically (deterministic producers, alternating
scale segments) so emit() can be driven to a mid-stream failure, resumed, and
compared byte-for-byte against a straight-through build. Corruption inside a
journaled region must cut the trusted prefix and still finish; a journal that
disagrees with the plan geometry must fail loudly.
"""

import hashlib
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import ling_stagepack as ling


REVISION = "r" * 40
CONTRACT_SHA = bytes(range(32))


def synthetic_plan(items: int, fail_at: int = -1):
    plan = []
    for index in range(items):
        entry = ling.Entry(kind=ling.K_DENSE_GATE_UP + index % 3, layer=index,
                           payload_type=1, weight_codec=ling.CODEC_BF16,
                           scale_encoding=0, group_count=1,
                           rows=64 + index, columns=8)
        entry.payload_bytes = 4096 + index * 16
        if index % 2:
            entry.scale_bytes = 128
        payload = random.Random(index).randbytes(entry.payload_bytes)
        scale = random.Random(index + 1000).randbytes(entry.scale_bytes)

        def produce(data=payload, fail=(index == fail_at)):
            step = 512
            for offset in range(0, len(data), step):
                if fail and offset >= len(data) // 2:
                    raise RuntimeError("simulated kill mid-region")
                yield data[offset:offset + step]

        def produce_scale(data=scale):
            yield data

        plan.append(ling.PlanItem(entry, produce,
                                  produce_scale if entry.scale_bytes else None))
    return SimpleNamespace(plan=plan, tp_degree=16, tp_rank=7,
                           expert_codec=ling.CODEC_BF16)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh(tmp: Path, name: str) -> Path:
    out = tmp / name
    out.mkdir()
    return out / "lingfin.bf16.tp16.rank7.sp"


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        reference_dir = fresh(tmp, "ref")
        ling.emit(synthetic_plan(6), reference_dir, REVISION, CONTRACT_SHA)
        reference = sha(reference_dir)

        target = fresh(tmp, "resume")
        try:
            ling.emit(synthetic_plan(6, fail_at=4), target, REVISION, CONTRACT_SHA,
                      resume=True)
            raise AssertionError("interrupted emit unexpectedly succeeded")
        except RuntimeError:
            pass
        partial = target.with_name(target.name + ".partial")
        journal = target.with_name(target.name + ".journal.jsonl")
        assert target.exists() is False
        assert partial.exists() and journal.exists()
        kept = [line for line in journal.read_text().splitlines() if line]
        assert len(kept) == 4, f"journal kept {len(kept)} regions, want 4"

        ling.emit(synthetic_plan(6), target, REVISION, CONTRACT_SHA, resume=True)
        assert target.exists()
        assert not partial.exists() and not journal.exists()
        assert sha(target) == reference, "resumed pack differs from reference"

        corrupt_target = fresh(tmp, "corrupt")
        try:
            ling.emit(synthetic_plan(6, fail_at=5), corrupt_target, REVISION,
                      CONTRACT_SHA, resume=True)
            raise AssertionError("interrupted emit unexpectedly succeeded")
        except RuntimeError:
            pass
        with corrupt_target.with_name(corrupt_target.name + ".partial").open("r+b") as handle:
            handle.seek(2048)
            handle.write(b"\xff" * 16)
        ling.emit(synthetic_plan(6), corrupt_target, REVISION, CONTRACT_SHA,
                  resume=True)
        assert sha(corrupt_target) == reference, "corruption repair differs"

        drift_target = fresh(tmp, "drift")
        try:
            ling.emit(synthetic_plan(6, fail_at=3), drift_target, REVISION,
                      CONTRACT_SHA, resume=True)
        except RuntimeError:
            pass
        drift = synthetic_plan(6)
        drift.plan[1].entry.payload_bytes += 8
        try:
            ling.emit(drift, drift_target, REVISION, CONTRACT_SHA, resume=True)
            raise AssertionError("plan drift resumed instead of failing")
        except ling.PackFailure:
            pass

        torn_target = fresh(tmp, "torn")
        try:
            ling.emit(synthetic_plan(6, fail_at=3), torn_target, REVISION,
                      CONTRACT_SHA, resume=True)
        except RuntimeError:
            pass
        with torn_target.with_name(torn_target.name + ".journal.jsonl").open("a") as handle:
            handle.write('{"index": 3, "kind": 9')
        ling.emit(synthetic_plan(6), torn_target, REVISION, CONTRACT_SHA,
                  resume=True)
        assert sha(torn_target) == reference, "torn journal recovery differs"
    finally:
        shutil.rmtree(tmp)
    print("resume: interrupted, corrupted, drifted and torn cases all close")
    return 0


if __name__ == "__main__":
    sys.exit(main())
