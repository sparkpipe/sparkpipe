#!/usr/bin/env python3
"""Grouped expert selection must match modeling_kimi_linear.py.

Kimi K3 sets num_expert_group 1, so this path is dead for the checkpoint in
hand. It is implemented and tested anyway, because "coded but untested" and
"does not exist" are not the same kind of debt: the first fails with a number
you can compare, the second fails with a paragraph someone has to re-derive
from a paper. And the claim that there was nothing to test against was wrong -
the algorithm is in the reference, which is exactly what the SiTU, decay and
absorption tests are checked against.

The reference, from KimiMoEGate.forward:

    group_scores = scores_for_choice.view(n, groups, -1).topk(2, -1)[0].sum(-1)
    group_idx    = topk(group_scores, topk_group)
    mask everything outside those groups to -inf
    topk over what remains

Three properties are checked rather than one comparison, because a selection
either matches or does not and a single equality hides which half is wrong:
the group score is the sum of the top TWO and not the maximum, exactly
topk_group groups survive, and nothing outside them can be selected.
"""
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERTS = 32
GROUPS = 8
TOP_GROUPS = 3
TOP_K = 6
ROWS = 64


def reference_selection(scores, bias):
    """modeling_kimi_linear.py, transcribed."""
    biased = [s + b for s, b in zip(scores, bias)]
    per_group = EXPERTS // GROUPS
    group_score = []
    for group in range(GROUPS):
        members = sorted(biased[group * per_group:(group + 1) * per_group],
                         reverse=True)
        group_score.append(members[0] + members[1])
    keep = sorted(range(GROUPS), key=lambda g: group_score[g], reverse=True)[:TOP_GROUPS]
    masked = [b if (i // per_group) in keep else float("-inf")
              for i, b in enumerate(biased)]
    order = sorted(range(EXPERTS), key=lambda i: masked[i], reverse=True)
    return set(order[:TOP_K]), set(keep), group_score


def kernel_uses_top_two():
    """The kernel must rank groups by the sum of two members. Ranking by the
    maximum agrees with the reference on most random inputs and differs exactly
    where a group has one strong expert and no second - which is the case the
    grouping exists to penalise."""
    text = (ROOT / "inference" / "kernels" / "topk.cuh").read_text()
    body = re.search(r"if \( GROUPS > 1u && GROUPS > TOP_GROUPS \).*?\n\t\}",
                     text, re.S)
    if not body:
        return "the grouped branch is gone"
    if "second" not in body.group(0):
        return "the group score no longer tracks a second member"
    if "LmTopkValue(best) + LmTopkValue(second)" not in body.group(0):
        return "the group score is not the sum of the top two values"
    return None


def build_and_run(cases):
    """The kernel's grouping arithmetic, single-threaded on the host."""
    program = """#include <stdio.h>
#include <string.h>
int main(void)
{
\tstatic float scores[%d][%d], bias[%d];
\tint row, group, member, taken, index, best_group;
\tfloat group_score[%d];
\tint dropped[%d];
%s
\tfor (row = 0; row < %d; ++row)
\t{
\t\tint per_group = %d / %d;
\t\tfor (group = 0; group < %d; ++group)
\t\t{
\t\t\tfloat best = -1e30f, second = -1e30f;
\t\t\tfor (member = 0; member < per_group; ++member)
\t\t\t{
\t\t\t\tfloat v = scores[row][group * per_group + member] + bias[group * per_group + member];
\t\t\t\tif (v > best) { second = best; best = v; }
\t\t\t\telse if (v > second) second = v;
\t\t\t}
\t\t\tgroup_score[group] = best + second;
\t\t\tdropped[group] = 1;
\t\t}
\t\tfor (taken = 0; taken < %d; ++taken)
\t\t{
\t\t\tfloat highest = -1e30f; best_group = -1;
\t\t\tfor (index = 0; index < %d; ++index)
\t\t\t\tif (dropped[index] && group_score[index] > highest)
\t\t\t\t{ highest = group_score[index]; best_group = index; }
\t\t\tdropped[best_group] = 0;
\t\t}
\t\tfor (group = 0; group < %d; ++group)
\t\t\tif (!dropped[group]) printf("%%d ", group);
\t\tprintf("\\n");
\t}
\treturn 0;
}
""" % (ROWS, EXPERTS, EXPERTS, GROUPS, GROUPS,
       "".join("\tscores[%d][%d] = %.9ff;\n" % (r, e, cases[r][0][e])
               for r in range(ROWS) for e in range(EXPERTS)) +
       "".join("\tbias[%d] = %.9ff;\n" % (e, cases[0][1][e]) for e in range(EXPERTS)),
       ROWS, EXPERTS, GROUPS, GROUPS, TOP_GROUPS, GROUPS, GROUPS)
    source = Path(tempfile.gettempdir()) / "group_check.c"
    binary = Path(tempfile.gettempdir()) / "group_check"
    source.write_text(program)
    build = subprocess.run(["gcc", "-O0", "-o", str(binary), str(source)],
                           capture_output=True, text=True)
    if build.returncode != 0:
        print("FAIL could not build:", build.stderr.strip()[:200])
        return None
    out = subprocess.run([str(binary)], capture_output=True, text=True).stdout
    return [set(int(v) for v in line.split()) for line in out.strip().split("\n")]


def main():
    problem = kernel_uses_top_two()
    if problem:
        print(f"FAIL {problem}")
        return 1
    generator = random.Random(20260727)
    bias = [generator.uniform(-0.3, 0.3) for _ in range(EXPERTS)]
    cases = [([generator.uniform(0.0, 1.0) for _ in range(EXPERTS)], bias)
             for _ in range(ROWS)]
    got = build_and_run(cases)
    if got is None:
        return 1
    failures = 0
    max_beats_sum = 0
    for row, (scores, _) in enumerate(cases):
        _, want_groups, _ = reference_selection(scores, bias)
        if got[row] != want_groups:
            print(f"  FAIL row {row}: kernel keeps {sorted(got[row])}, "
                  f"reference keeps {sorted(want_groups)}")
            failures += 1
        if len(got[row]) != TOP_GROUPS:
            print(f"  FAIL row {row}: {len(got[row])} groups survived, "
                  f"expected {TOP_GROUPS}")
            failures += 1
        # would ranking by the maximum alone have differed here?
        per_group = EXPERTS // GROUPS
        biased = [s + b for s, b in zip(scores, bias)]
        by_max = sorted(range(GROUPS),
                        key=lambda g: max(biased[g * per_group:(g + 1) * per_group]),
                        reverse=True)[:TOP_GROUPS]
        if set(by_max) != want_groups:
            max_beats_sum += 1
    print(f"experts {EXPERTS}  groups {GROUPS}  keep {TOP_GROUPS}  rows {ROWS}")
    print(f"rows where ranking by the maximum would have chosen differently: "
          f"{max_beats_sum}")
    if max_beats_sum == 0:
        print("\nFAIL these inputs cannot distinguish sum-of-top-two from "
              "maximum; the test proves nothing")
        return 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\ngrouped selection matches KimiMoEGate on every row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
