#!/usr/bin/env python3
"""Compare two GLM-5.3 Flash serving archives written by glm53_flash_golden.py.

Prints, per prompt and section (b1, b4, reuse), whether the greedy tokens are
identical and the first divergence; for logprob runs, whether the replay's
own B4 and reuse logprobs equal its B1 bit for bit. Exit status 0 only when
every compared token list is identical.
"""

from __future__ import annotations

import json
import sys


def divergence(a: list[int], b: list[int]) -> int:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return -1 if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: glm53_flash_golden_compare.py GOLDEN.json REPLAY.json")
    golden = json.load(open(sys.argv[1]))
    replay = json.load(open(sys.argv[2]))
    failures = 0
    print(f"golden={golden['label']} replay={replay['label']}")
    for section in ("b1", "b4", "reuse"):
        for name, want in golden[section].items():
            got = replay[section].get(name, {})
            if "tokens" not in want or "tokens" not in got:
                print(f"{section:5s} {name:22s} MISSING golden_error={want.get('error')} replay_error={got.get('error')}")
                failures += 1
                continue
            at = divergence(want["tokens"], got["tokens"])
            same = at < 0
            failures += 0 if same else 1
            note = "" if same else f" first_divergence={at} golden={want['tokens'][at:at + 4]} replay={got['tokens'][at:at + 4]}"
            print(f"{section:5s} {name:22s} tokens {'IDENTICAL' if same else 'DIFFER'} n={len(want['tokens'])}/{len(got['tokens'])}{note}")
    if replay.get("logprobs"):
        for section, pairs in (("b4", [(n, n) for n in replay["b4"]]), ("reuse", [("reuse_weightd_q2", None), ("doc8k_weightd_again", "doc8k_weightd")])):
            for name, base in pairs:
                if base is None:
                    continue
                a, b = replay[section].get(name, {}), replay["b1"].get(base, {})
                print(f"self  {section}/{name} vs b1/{base}: tokens {'same' if a.get('tokens') == b.get('tokens') else 'DIFFER'}, "
                      f"logprobs {'bit-identical' if a.get('logprobs') == b.get('logprobs') else 'DIFFER'}")
    print(f"token lists differing: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
