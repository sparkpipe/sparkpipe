"""The K3 engine keeps the contract the slice kernels enforce.

The sequence-run kernels demand rows sorted by sequence with positions
ascending in a run; the MLA path demands context_length count every stored
row; the sampler demands logits only where a forward actually predicts the
next token. The engine is the one component that can violate all of that at
once while every kernel stays correct, so this gate drives the host harness
through three scenarios and holds every printed plan to the contract.

Main scenario - two requests (prompt 5 and prompt 3 under a four-row budget
and two slots) plus a third that must queue for a freed slot:

  chunked prefill: prompt 5 splits 4 + 0-with-decode (the final prompt token
  belongs to decode), and the chunk never asks for logits
  continuous batching: decode rows and a prefill chunk share a step
  positions ascend within every run and resume where the last chunk stopped
  context_length is always the run's last position + 1
  a finished request's slot is reused by the queued one
  EOS ends a request under budget and its output stops there

Serving-bug scenario (the main-8 audit's K3-006..009, on the same engine):

  K3-006 a one-token prompt opens in DECODE: one row at position 0 with
         context 1 and logits on the row - no zero-row prefill, no
         context underflow
  K3-007 four finished records on a four-record engine must submit again;
         the fifth submit hits the capacity wall
  K3-008 a duplicated commit, a slot poked out of bounds and a stale step
         all fail closed with K3_ENGINE_ERR_STATE, and the machine commits
         the current step right after
  K3-009 a draft whose middle accepted token is EOS truncates there: the
         rest of the block and the bonus never reach the output

Fairness scenario (K3-010): three requests on a three-slot engine with a
two-row budget, so j and k's decode rows fill every pass - l's prefill must
still advance within K3_ENGINE_PREFILL_PERIOD passes of admission.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_steps(text):
    steps = []
    for block in re.split(r"(?=step \d+ )", text):
        m = re.match(r"step (\d+) rows (\d+) sequences (\d+)", block)
        if not m:
            continue
        seqs = []
        for sm in re.finditer(
                r"seq \d+ slot (\d+) request (\d+) run (\d+)\.\.(\d+) "
                r"context (\d+) logits (-?\d+)((?:\n    row [^\n]*)*)", block):
            rows = [tuple(int(x) for x in rm.groups()) for rm in re.finditer(
                r"row (\d+) token (\d+) position (\d+) slot (\d+)", sm.group(7))]
            seqs.append(dict(slot=int(sm.group(1)), request=int(sm.group(2)),
                             begin=int(sm.group(3)), end=int(sm.group(4)),
                             context=int(sm.group(5)), logits=int(sm.group(6)),
                             rows=rows))
        nexts = {int(vm.group(1)): int(vm.group(2)) - 1 for vm in
                 re.finditer(r"verify_next (\d+) (\d+)", block)}
        steps.append(dict(index=int(m.group(1)), rows=int(m.group(2)),
                          seqs=seqs, verify_next=nexts))
    return steps


def check_contract(steps, label):
    """The run contract every scenario must hold: contiguous runs, positions
    ascending by one and resuming where the request stopped, context_length
    counting every stored row, and no run planned with zero rows (K3-006)."""
    failures = 0
    last_position = {}
    for step in steps:
        cursor = 0
        for seq in step["seqs"]:
            if seq["begin"] != cursor:
                print(f"  FAIL {label} step {step['index']}: runs are not contiguous")
                failures += 1
            cursor = seq["end"]
            positions = [r[2] for r in seq["rows"]]
            if not positions:
                print(f"  FAIL {label} step {step['index']}: a run was planned "
                      "with zero rows")
                failures += 1
            if positions != sorted(positions) or (
                    len(positions) > 1
                    and positions != list(range(positions[0], positions[0] + len(positions)))):
                print(f"  FAIL {label} step {step['index']}: positions do not ascend by one")
                failures += 1
            if any(r[3] != seq["slot"] for r in seq["rows"]):
                print(f"  FAIL {label} step {step['index']}: a row's slot differs from its run's")
                failures += 1
            if positions and seq["context"] != positions[-1] + 1:
                print(f"  FAIL {label} step {step['index']}: context {seq['context']} "
                      f"is not last position + 1")
                failures += 1
            key = seq["request"]
            if positions and key in last_position and positions[0] != last_position[key] + 1:
                print(f"  FAIL {label} request {key}: resumed at {positions[0]} "
                      f"after {last_position[key]}")
                failures += 1
            if positions:
                last_position[key] = positions[-1]
            if seq["logits"] >= 0 and not (seq["begin"] <= seq["logits"] < seq["end"]):
                print(f"  FAIL {label} step {step['index']}: logits row outside the run")
                failures += 1
        # a verify resolution re-plans the first rejected position - the
        # bonus landed there without a forward - so resumption restarts one
        # short of it, exactly as the harness declared at commit time
        for key, value in step["verify_next"].items():
            last_position[key] = value
        if cursor != step["rows"]:
            print(f"  FAIL {label} step {step['index']}: runs do not cover the rows")
            failures += 1
    return failures


def main():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "k3_engine_host"
        build = subprocess.run(
            ["cc", "-std=c11", "-O1", "-I", str(ROOT),
             str(ROOT / "tests" / "host_cuda" / "k3_engine_host.c"),
             "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build:", build.stderr[:400])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    if run.returncode != 0:
        print(f"FAIL the engine faulted (returncode {run.returncode})")
        print(run.stdout[-800:])
        return 1
    text = run.stdout
    failures = 0
    # the harness prints one transcript per scenario, split by its markers
    parts = re.split(r"^scenario (\S+)\s*$", text, flags=re.M)
    main_text = parts[0]
    sections = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    for label in ("serving_bugs", "fairness"):
        if label not in sections:
            print(f"  FAIL the harness printed no {label} scenario")
            failures += 1
            sections[label] = ""
    steps = parse_steps(main_text)
    bug_steps = parse_steps(sections["serving_bugs"])
    fair_steps = parse_steps(sections["fairness"])
    failures += check_contract(steps, "main")
    failures += check_contract(bug_steps, "serving_bugs")
    failures += check_contract(fair_steps, "fairness")

    # request 1: prompt 5, budget 4 -> first chunk is exactly 4 rows, no logits
    first = steps[0]["seqs"][0]
    if not (first["request"] == 1 and first["end"] - first["begin"] == 4
            and first["logits"] < 0):
        print("  FAIL prompt 5 under budget 4 must open with a 4-row chunk "
              "asking for no logits")
        failures += 1
    # some step must mix a decode row with a prefill chunk
    if not any(any(s["logits"] >= 0 for s in st["seqs"])
               and any(s["logits"] < 0 and s["end"] - s["begin"] > 0 for s in st["seqs"])
               for st in steps):
        print("  FAIL no step mixed decode with a prefill chunk")
        failures += 1
    # request 3 queued, then reused a slot some finished request abandoned
    slots_of_3 = {s["slot"] for st in steps for s in st["seqs"] if s["request"] == 3}
    slots_of_12 = {s["slot"] for st in steps for s in st["seqs"] if s["request"] in (1, 2)}
    if not slots_of_3 or not slots_of_3 <= slots_of_12:
        print("  FAIL the queued request did not reuse a freed slot")
        failures += 1
    if "out_b " not in text or re.search(r"out_b (\d+) 7\b", text) is None:
        print("  FAIL EOS did not end request b's output at its second token")
        failures += 1

    # the verify lane: a verify-only step of one four-row run, positions
    # ascending from the last committed token, logits at the head; resolving
    # it (2 accepted + bonus) must appear as three consecutive outputs.
    vm = re.search(r"verify step (\d+)", main_text)
    if vm is None:
        print("  FAIL no verify step was planned after the draft")
        failures += 1
    else:
        vstep = next(st for st in steps if st["index"] == int(vm.group(1)))
        run = vstep["seqs"][0]
        if len(vstep["seqs"]) != 1 or run["end"] - run["begin"] != 4:
            print("  FAIL the verify step is not one four-row run")
            failures += 1
        if run["logits"] != run["begin"]:
            print("  FAIL verify logits are not at the run's head")
            failures += 1
        if [r[1] for r in run["rows"]][1:] != [201, 202, 203]:
            print("  FAIL the verify rows do not carry the draft")
            failures += 1
    if re.search(r"out_a 100 101 201 202 ", text) is None:
        print("  FAIL acceptance did not land the drafts in the output")
        failures += 1

    # acceptance counters (SURVEY_K3 #10): the main scenario's verify commit
    # proposed three drafts and emitted two; K3-009's commit proposed three
    # and emitted two (301 then the EOS draft) - cumulative 6/4.
    drafts = re.findall(r"^drafts (\d+) (\d+)$", text, re.M)
    if drafts != [("3", "2"), ("6", "4")]:
        print(f"  FAIL draft acceptance counters are {drafts}, expected "
              f"[('3', '2'), ('6', '4')]")
        failures += 1

    # K3-006: the one-token prompt (request 4) opened in DECODE - a single
    # row at position 0, context 1, logits on that row - and ended on EOS
    d_runs = [s for st in bug_steps for s in st["seqs"] if s["request"] == 4]
    if not d_runs or not (d_runs[0]["rows"][0][2] == 0
                          and d_runs[0]["context"] == 1
                          and d_runs[0]["logits"] == d_runs[0]["begin"]
                          and len(d_runs[0]["rows"]) == 1):
        print("  FAIL K3-006: the one-token prompt did not open as one "
              "decode row at position 0 with context 1")
        failures += 1
    if re.search(r"out_d 42 7\b", text) is None:
        print("  FAIL K3-006: the one-token prompt did not decode to EOS")
        failures += 1
    # K3-007: eight submits on a four-record engine - the finished records
    # came back to FREE - and the ninth hit the capacity wall
    if re.search(r"reuse e 5 f 6 g 7 h 8 full -71\b", text) is None:
        print("  FAIL K3-007: finished request records did not return to FREE")
        failures += 1
    # K3-008: duplicated, out-of-bounds and stale commits all fail closed
    for marker in ("commit_dup -73", "commit_oob -73", "commit_stale -73"):
        if marker not in text:
            print(f"  FAIL K3-008: {marker.split()[0]} did not fail closed")
            failures += 1
    if re.search(r"out_f \d+ \d+", text) is None:
        print("  FAIL K3-008: the engine did not keep committing after the "
              "rejected commits")
        failures += 1
    # K3-009: EOS inside the accepted drafts truncated the block - out_e
    # stops at 301, 7 and the bonus (999) never lands anywhere
    if re.search(r"out_e 301 7 0 0\b", text) is None or "999" in text:
        print("  FAIL K3-009: EOS inside the accepted drafts did not "
              "truncate the block and skip the bonus")
        failures += 1

    # K3-010: with j and k's decode rows filling the two-row budget, l's
    # prefill still advanced within one fairness period of the crowding
    l_runs = [(st["index"], s) for st in fair_steps for s in st["seqs"]
              if s["request"] == 3]
    crowded = any(st["rows"] == 2 and len(st["seqs"]) == 2
                  and all(s["logits"] >= 0 for s in st["seqs"])
                  for st in fair_steps)
    if not crowded:
        print("  FAIL K3-010: no step had full decode lanes crowding the budget")
        failures += 1
    if not l_runs or l_runs[0][0] > 4 or l_runs[0][1]["logits"] >= 0:
        print("  FAIL K3-010: the crowded prefiller did not advance within "
              "one fairness period")
        failures += 1

    print(f"steps {len(steps)} + {len(bug_steps)} + {len(fair_steps)}, "
          "requests finished 3 + 5")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe engine batches, chunks, mixes, ends, recycles, fails closed "
          "and shares - and never breaks the run contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
