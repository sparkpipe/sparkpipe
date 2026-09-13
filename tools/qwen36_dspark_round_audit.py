#!/usr/bin/env python3
"""Round audit: which COMPONENT of which speculative round first commits a non-golden token.

The instrumented adapter prints, per round, one qwen36_spec_diag line (lane,
base_position, C0, accepted, the first drafts and emitted ids) and one round_commit
line (accepted, min_accepted, credited, the absolute positions the round commits).
A spec run's token_ids and the no-spec golden supply what those rounds actually
committed. Together they pin the failure to a named component, because a round
commits exactly (min_accepted + 3) tokens in a fixed order:

    offset 0                      C0                 the decode frame's emission
    offset 1 .. min_accepted      accepted drafts    draft_ids[1..min_accepted]
    offset min_accepted + 1       correction         emitted_ids[min_accepted]
    offset min_accepted + 2       replay emission    the replay frame's last row

Each offset accuses a different mechanism:

    C0              the decode frame's own head - upstream of speculation entirely
    accepted draft  the acceptance test passed a draft the target would not emit,
                    i.e. the verify frame's emitted ids are wrong (its state) or
                    the comparison is misaligned
    correction      the verify's own emission at the first rejected row is wrong
    replay emission the ONLY committed token nothing cross-checks: produced by a
                    freshly restored and re-walked state, and it becomes the next
                    round's C0

usage: qwen36_dspark_round_audit.py WINDOW_LOG SPEC_JSON GOLDEN_JSON [--base N]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIAG = re.compile(
    r"qwen36_spec_diag lane=(\d+) base_position=(\d+) C0=(\d+) accepted=(\d+) "
    r"drafts=\[([0-9,]+)\] emitted=\[([0-9,]+)\]")
COMMIT = re.compile(
    r"qwen36_spec round_commit lane=(\d+) base_position=(\d+) accepted=(\d+) "
    r"min_accepted=(\d+) credited=(\d+)(?: ceiling=(\d+))? positions=\[(\d+)\.\.(\d+)\]")


def parse_rounds(path: Path):
    """Pair each diag with the round_commit that follows it, by base position."""
    diags, commits = {}, []
    for line in path.read_text(errors="replace").splitlines():
        match = DIAG.search(line)
        if match is not None:
            diags[int(match.group(2))] = {
                "lane": int(match.group(1)), "c0": int(match.group(3)),
                "accepted": int(match.group(4)),
                "drafts": [int(value) for value in match.group(5).split(",")],
                "emitted": [int(value) for value in match.group(6).split(",")]}
            continue
        match = COMMIT.search(line)
        if match is not None:
            commits.append({
                "lane": int(match.group(1)), "base": int(match.group(2)),
                "accepted": int(match.group(3)), "min_accepted": int(match.group(4)),
                "credited": int(match.group(5)),
                "ceiling": int(match.group(6)) if match.group(6) else None,
                "first": int(match.group(7)), "last": int(match.group(8))})
    for commit in commits:
        commit["diag"] = diags.get(commit["base"])
    return commits


def component(offset: int, min_accepted: int):
    if offset == 0:
        return "C0 (decode frame head)"
    if offset <= min_accepted:
        return f"accepted draft #{offset} (verify emitted / acceptance test)"
    if offset == min_accepted + 1:
        return "correction (verify emission at the rejected row)"
    if offset == min_accepted + 2:
        return "REPLAY EMISSION (the round's only unchecked token)"
    return f"offset {offset} BEYOND credited - accounting overrun"


def main() -> int:
    arguments = [value for value in sys.argv[1:] if not value.startswith("--")]
    base_override = None
    for index, value in enumerate(sys.argv[1:]):
        if value == "--base":
            base_override = int(sys.argv[index + 2])
    if len(arguments) < 3:
        raise SystemExit(__doc__)
    rounds = parse_rounds(Path(arguments[0]))
    spec = json.load(open(arguments[1]))["token_ids"]
    golden = json.load(open(arguments[2]))["token_ids"]
    if not rounds:
        raise SystemExit(f"no round_commit lines in {arguments[0]}")
    origin = base_override if base_override is not None else rounds[0]["base"]
    print(f"rounds        = {len(rounds)}  positions {rounds[0]['first']}..{rounds[-1]['last']}")
    print(f"token index 0 = absolute position {origin}; spec {len(spec)} tokens, "
          f"golden {len(golden)} tokens")
    stream_first = next((index for index, (a, b) in enumerate(zip(spec, golden)) if a != b), None)
    print(f"stream diverges at token index {stream_first} "
          f"(absolute {None if stream_first is None else origin + stream_first})")

    # LADDER CHECK: the rounds must tile the committed stream with no gap or overlap.
    # It starts at the FIRST ROUND's position, not at the stream origin: index 0 of the
    # stream is the prefill's own emission, which no speculative round commits (that
    # one-token offset is exactly what makes a naive base_position - prompt_length
    # alignment accuse the wrong component).
    expected = rounds[0]["first"]
    ladder_ok = True
    for entry in rounds:
        if entry["first"] != expected:
            print(f"  LADDER BREAK: round at base {entry['base']} starts at {entry['first']}, "
                  f"expected {expected}")
            ladder_ok = False
        if entry["last"] - entry["first"] + 1 != entry["credited"]:
            print(f"  SPAN MISMATCH at base {entry['base']}: credited {entry['credited']} "
                  f"but positions span {entry['last'] - entry['first'] + 1}")
            ladder_ok = False
        if entry["credited"] != entry["min_accepted"] + 3:
            print(f"  CREDIT MISMATCH at base {entry['base']}: credited {entry['credited']} "
                  f"!= min_accepted {entry['min_accepted']} + 3")
            ladder_ok = False
        expected = entry["last"] + 1
    print(f"ladder        = {'contiguous, credited == min_accepted + 3 everywhere' if ladder_ok else 'BROKEN (see above)'}")

    print()
    header = (f"{'base':>6} {'acc':>4} {'min':>4} {'cred':>5} {'positions':>13} "
              f"{'C0':>7} {'first bad':>10} {'component':>52}")
    print(header)
    first_bad_round = None
    for entry in rounds:
        bad_offset = None
        for offset in range(entry["credited"]):
            index = entry["first"] + offset - origin
            if 0 <= index < min(len(spec), len(golden)) and spec[index] != golden[index]:
                bad_offset = offset
                break
        name = "-" if bad_offset is None else component(bad_offset, entry["min_accepted"])
        print(f"{entry['base']:>6} {entry['accepted']:>4} {entry['min_accepted']:>4} "
              f"{entry['credited']:>5} {f'{entry[chr(102)+chr(105)+chr(114)+chr(115)+chr(116)]}..{entry[chr(108)+chr(97)+chr(115)+chr(116)]}':>13} "
              f"{entry['diag']['c0'] if entry['diag'] else 0:>7} "
              f"{('-' if bad_offset is None else str(bad_offset)):>10} {name:>52}")
        if bad_offset is not None and first_bad_round is None:
            first_bad_round = (entry, bad_offset)
    if first_bad_round is None:
        print("\nno round commits a non-golden token in this window")
        return 0

    entry, offset = first_bad_round
    index = entry["first"] + offset - origin
    print()
    print("FIRST NON-GOLDEN COMMITTED TOKEN")
    print(f"  round base_position {entry['base']}, offset {offset} -> absolute position "
          f"{entry['first'] + offset}, golden index {index}")
    print(f"  committed {spec[index]}  golden {golden[index]}")
    print(f"  accepted={entry['accepted']} min_accepted={entry['min_accepted']} "
          f"credited={entry['credited']}"
          + (f" ceiling={entry['ceiling']}" if entry["ceiling"] is not None else ""))
    print(f"  COMPONENT: {component(offset, entry['min_accepted'])}")
    if entry["diag"] is not None:
        diag = entry["diag"]
        print(f"  diag: C0={diag['c0']} drafts={diag['drafts']} emitted={diag['emitted']}")
        # The acceptance test is emitted[j] == draft_ids[j+1]; show it against the golden
        # so a wrongly accepted draft is visible as "target said X, golden says Y".
        for j in range(min(len(diag["emitted"]), len(diag["drafts"]) - 1)):
            position = entry["first"] + 1 + j
            golden_index = position - origin
            golden_token = golden[golden_index] if 0 <= golden_index < len(golden) else None
            print(f"    row {j}: verify emitted {diag['emitted'][j]} vs draft {diag['drafts'][j + 1]}"
                  f" -> {'accept' if diag['emitted'][j] == diag['drafts'][j + 1] else 'reject'}"
                  f"; golden at position {position} is {golden_token}")
    print()
    print("  what each component would mean:")
    print("    C0              the decode frame's head - speculation is not the cause")
    print("    accepted draft  the verify frame's emitted ids are wrong (its state) or the")
    print("                    comparison is misaligned - a draft was accepted that the")
    print("                    target would not have emitted")
    print("    correction      the verify's emission at the first rejected row is wrong")
    print("    replay emission the only committed token nothing cross-checks, produced on a")
    print("                    freshly restored state and used as the next round's C0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
