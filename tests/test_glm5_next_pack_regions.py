#!/usr/bin/env python3
"""Streaming region oracle rejects wrong, truncated and oversized planes."""
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from glm5_next_pack_verify import verify_region, check_stage_header


def main():
    for stage, first, count in ((0, 0, 12), (1, 12, 11), (2, 23, 11), (3, 34, 11)):
        args = SimpleNamespace(stage_count=4, stage_index=stage,
                               first_layer=first, layer_count=count, mtp=False,
                               mtp_only=False)
        header = dict(stage_count=4, stage_index=stage, first_layer=first,
                      layer_count=count, total_layers=45, flags=0)
        check_stage_header(header, args)
        for key in header:
            corrupted = dict(header)
            corrupted[key] += 1
            try:
                check_stage_header(corrupted, args)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"incorrect {key} accepted")
    mtp_args = SimpleNamespace(stage_count=1, stage_index=0, first_layer=0,
                               layer_count=45, mtp=False, mtp_only=True)
    mtp_header = dict(stage_count=1, stage_index=0, first_layer=45,
                      layer_count=1, total_layers=45, flags=1)
    check_stage_header(mtp_header, mtp_args)
    for key in mtp_header:
        corrupted = dict(mtp_header)
        corrupted[key] += 1
        try:
            check_stage_header(corrupted, mtp_args)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"incorrect MTP-only {key} accepted")
    full_header = dict(stage_count=1, stage_index=0, first_layer=0,
                       layer_count=45, total_layers=45, flags=0)
    try:
        check_stage_header(full_header, mtp_args)
    except SystemExit:
        pass
    else:
        raise AssertionError("a full pack header passed as the MTP sidecar")
    data = bytes(range(251)) * 40000
    expected = hashlib.sha256(data).hexdigest()
    assert verify_region(b"prefix" + data, 6, len(data), iter([data]), "large") == expected
    assert verify_region(data, 0, len(data), iter([data[:17], data[17:]]), "split") == expected
    assert verify_region(b"", 0, 0, iter(()), "empty") == hashlib.sha256(b"").hexdigest()
    for actual, chunks in ((data, [data[:-1]]), (data, [data, b"x"]),
                           (data[:-1] + b"x", [data]), (data[:-1], [data])):
        try:
            verify_region(actual, 0, len(data), iter(chunks), "expected failure")
        except SystemExit:
            pass
        else:
            raise AssertionError("bad region accepted")
    script = Path(__file__).resolve().parents[1] / "tools/glm5_next_pack_verify.py"
    result = subprocess.run([sys.executable, str(script), "--pack", "unused",
                             "--source", "unused", "--tp-rank", "0",
                             "--all-tensors", "--skip-spot"], capture_output=True)
    assert result.returncode != 0 and b"not allowed" in result.stderr
    for extra in (["--mtp"], ["--stage-count", "4"], ["--first-layer", "3"], ["--layer-count", "1"]):
        result = subprocess.run([sys.executable, str(script), "--pack", "unused",
                                 "--source", "unused", "--tp-rank", "0",
                                 "--mtp-only"] + extra, capture_output=True)
        assert result.returncode != 0 and b"--mtp-only takes no" in result.stderr, extra
    print("PASS streaming checkpoint regions, exclusive verification scope and the MTP-only sidecar header")


if __name__ == "__main__":
    main()
