#!/usr/bin/env python3
"""Statistical deep-parity scorer: for every logged round with a ctxrun dump,
run the numpy reference under several context conventions and compare
per-position hit rates against the device's own drafts and the round truth.

Inputs: the residentd log (qwen36_spec_diag lines) + /tmp/ctxrun_<n>.meta and
ctxrun_<n>_taps.bin written by SPARK_QWEN36_DFLASH2_CTX_DUMP=1. Run n feeds
round n (run 1 = the bootstrap decode; the meta's drafts must equal the
round's drafts[1:8] - asserted).

Two truth alignments are scored because the block's emission semantics is the
open question: NEXT (draft d ~ token@base+d+1 = emitted[d]) vs OWN (draft d
~ token@base+d = emitted[d-1]). The alignment reproducing the device's known
~0.6 position-1 rate identifies the semantics; ref-vs-device agreement then
localizes any forward defect, and convention deltas localize conditioning
defects.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/spark2/sparkpipe/tools")
import qwen36_dflash2_deep_parity as DP  # noqa: E402
import qwen36_dspark_reference as R  # noqa: E402

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen38-tp1-parity.log"

# cache the safetensors loads: run_reference reloads per call otherwise
_w = {}
_orig_drafter, _orig_shared = R.load_drafter, R.load_target_shared
R.load_drafter = lambda: _w.setdefault("d", _orig_drafter())
R.load_target_shared = lambda: _w.setdefault("s", _orig_shared())


def parse_rounds(path):
    rounds = []
    line_re = re.compile(
        r"qwen36_spec_diag C0=(\d+) accepted=(\d+) drafts=\[([0-9,]+)\] emitted=\[([0-9,]+)\]"
    )
    for line in open(path, errors="replace"):
        m = line_re.search(line)
        if m:
            drafts = [int(t) for t in m.group(3).split(",")]
            emitted = [int(t) for t in m.group(4).split(",")]
            rounds.append({"c0": int(m.group(1)), "acc": int(m.group(2)), "drafts": drafts, "emitted": emitted})
    return rounds


def parse_run(n):
    text = open(f"/tmp/ctxrun_{n}.meta").read()
    m = re.search(r"drafts=([0-9 ]+)", text)
    drafts = [int(t) for t in m.group(1).split()]
    meta = {}
    for field in text[: m.start()].split():
        k, _, v = field.partition("=")
        meta[k] = int(v)
    meta["drafts"] = drafts
    return meta


def hits(drafts, truth, offset):
    if offset == 0:
        pairs = list(zip(drafts, truth))
    else:
        pairs = list(zip(drafts, truth[offset:]))
    return [1 if a == b else 0 for a, b in pairs]


def main():
    rounds = parse_rounds(LOG)
    conv_filter = sys.argv[2] if len(sys.argv) > 2 else "all"
    max_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 6
    taps_cache = {}

    def taps_for(n, lo, count):
        if n not in taps_cache:
            path = f"/tmp/ctxrun_{n}_taps.bin"
            raw = np.fromfile(path, dtype=np.uint16)
            taps_cache[n] = R.bf16_to_f32(raw).reshape(-1, R.TAPS, R.HIDDEN)
        return taps_cache[n]

    matched = 0
    stats = {name: {"next": [0] * 7, "own": [0] * 7, "n": [0] * 7}
             for name in ("device", "ref_dev", "ref_devpg", "ref_old1", "ref_top1")}
    agree = [0] * 7
    for n in range(1, min(len(rounds) - 1, 79) + 1):
        if matched >= max_rounds:
            break
        if not os.path.exists(f"/tmp/ctxrun_{n}.meta"):
            continue
        run = parse_run(n)
        rnd = rounds[n] if n < len(rounds) else None  # run n feeds round n+1
        if rnd is None:
            continue
        # original remap restored: draft_ids[i+1] = walk output i
        if run["drafts"] != rnd["drafts"][1:8]:
            print(f"run {n}: drafts mismatch vs round {n+1} - skipping")
            continue
        matched += 1
        base, window, lo_dump, anchor = run["base"], run["window"], run["lo"], run["anchor"]
        taps = taps_for(n, lo_dump, 0)
        truth = rnd["emitted"]
        for name in stats:
            for pos in range(7):
                stats[name]["n"][pos] += 1
        for pos, (d, t) in enumerate(zip(run["drafts"], truth)):
            stats["device"]["next"][pos] += d == t
        for pos, (d, t) in enumerate(zip(run["drafts"], truth[1:])):
            stats["device"]["own"][pos] += d == t
        convs = {
            "ref_dev": (base - window, base, base),
            "ref_old_nogp": (base - 1 - window, base - 1, base - 1),
            "ref_old1": (base - 1, base, base),
        }
        if conv_filter != "all":
            convs = {k: v for k, v in convs.items() if k == conv_filter}
        for name, (lo, hi, bb) in convs.items():
            ref, top1 = DP.run_reference(taps, lo_dump, lo, hi, anchor, bb)
            for pos, (d, t) in enumerate(zip(ref, truth)):
                stats[name]["next"][pos] += d == t
            for pos, (d, t) in enumerate(zip(ref, truth[1:])):
                stats[name]["own"][pos] += d == t
            if name == "ref_dev":
                for pos, (a, b) in enumerate(zip(ref, run["drafts"])):
                    agree[pos] += a == b
            if name == "ref_dev":  # top1 baseline under the same forward
                for pos, (d, t) in enumerate(zip(top1, truth)):
                    stats["ref_top1"]["next"][pos] += d == t
                for pos, (d, t) in enumerate(zip(top1, truth[1:])):
                    stats["ref_top1"]["own"][pos] += d == t
    print(f"matched rounds: {matched}")
    for align in ("next", "own"):
        print(f"== truth alignment: {align} ==")
        for name in ("device", "ref_dev", "ref_devpg", "ref_old1", "ref_top1"):
            rates = " ".join(f"{stats[name][align][p]}/{stats[name]['n'][p]}" for p in range(7))
            print(f"  {name:10s} {rates}")
    print("dev==ref_dev agreement:", " ".join(f"{a}/{matched}" for a in agree))


if __name__ == "__main__":
    main()
