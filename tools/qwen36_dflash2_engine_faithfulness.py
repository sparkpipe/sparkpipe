#!/usr/bin/env python3
"""Engine-faithfulness check: replay the numpy oracle on the ENGINE's own
dumped taps (ctxrun_N from SPARK_QWEN38_27B_DFLASH2_CTX_DUMP on spark2) and
compare its 7 drafts against the engine's drafts (ctxrun_N.meta).

Agreement => the engine's block forward + selection is faithful to the
validated reference forward; the acceptance gap then lives in the inputs
(the fp8 target's taps/emissions vs the bf16 teacher the drafter was
distilled against), not the engine.

Run on spark3 (needs the drafter safetensors + the parity tooling):
  qwen38_27b_dflash2_engine_faithfulness.py <first> <last>
with /tmp/ctxrun_*.{bin,meta} present.
"""
import re
import sys

import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location("dp", "/tmp/qwen38_27b_dflash2_deep_parity.py")
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)
R = dp.R

first, last = int(sys.argv[1]), int(sys.argv[2])
agree_by_pos = [0] * 7
total = 0
perfect = 0
for n in range(first, last + 1):
    try:
        meta_line = open(f"/tmp/ctxrun_{n}.meta").read().strip()
        m = re.search(r"lo=(\d+) base=(\d+) window=(\d+) anchor=(\d+) drafts=([\d ]+)", meta_line)
        lo, base, window, anchor = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        dev = [int(x) for x in m.group(5).split()]
    except FileNotFoundError:
        continue
    nb_base = lo
    nb = base + 8 - lo
    taps = R.bf16_to_f32(np.fromfile(f"/tmp/ctxrun_{n}_taps.bin", dtype=np.uint16)).reshape(nb, R.TAPS, R.HIDDEN)
    if base - window < nb_base:
        continue
    walk, top1 = dp.run_reference(taps, nb_base, base - window, base, anchor, base)
    # the engine selects the per-mask-row ARGMAX (the serving semantics);
    # run_reference's first return is the (obsolete) codebook walk
    ref = top1
    total += 1
    ok = all(a == b for a, b in zip(ref, dev))
    perfect += ok
    for i, (a, b) in enumerate(zip(ref, dev)):
        agree_by_pos[i] += a == b
    if total <= 6 or not ok:
        mark = "OK " if ok else "DIFF"
        print(f"run {n}: {mark} engine={dev} oracle_argmax={ref} walk={walk}")
print(f"\nrounds compared: {total}  perfect: {perfect}")
print("per-position agreement:", " ".join(f"{a}/{total}" for a in agree_by_pos))
