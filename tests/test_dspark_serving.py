#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Ring:
    def __init__(self, torch, window):
        self.keys = torch.zeros(3, window, 4, dtype=torch.bfloat16)
        self.positions = torch.full((window,), -1, dtype=torch.int64)


class StubDrafter:
    window = 8
    max_drafts = 5
    device = "cpu"

    def __init__(self, torch):
        self.torch = torch
        self.ingested = []
        self.drafted = []

    def context(self):
        return Ring(self.torch, self.window)

    def ingest(self, ring, positions, taps):
        slots = self.torch.tensor([p % self.window for p in positions], dtype=self.torch.int64)
        ring.keys.index_copy_(1, slots, self.torch.full((3, len(positions), 4), float(positions[-1] if positions else 0), dtype=self.torch.bfloat16))
        ring.positions.index_copy_(0, slots, self.torch.tensor(positions, dtype=self.torch.int64))
        self.ingested.append(list(positions))

    def draft(self, ring, anchor, position, rule):
        self.drafted.append((anchor, position, rule))
        return [anchor + 1, anchor + 2, anchor + 3, anchor + 4, anchor + 5, anchor + 6], [0.9] * 6


def ring_failures(draftd, torch, np, drafter, ring_class):
    failures = []
    ring = ring_class(drafter)
    ring.write(100, np.arange(1000, 1006, dtype=np.uint32), [None] * 6)
    if not ring.held(np.arange(100, 106)).all() or ring.held(np.arange(98, 100)).any():
        failures.append("a ring holds exactly the positions it was written")
    ring.write(103, np.array([2003, 2004], dtype=np.uint32), [None] * 2)
    if ring.held(np.array([105])).any() or not ring.held(np.arange(100, 105)).all() or list(ring.tokens[np.arange(103, 105) % 8]) != [2003, 2004]:
        failures.append("rewriting from position 103 must drop every later slot (105) and keep the earlier ones")
    if int(ring.ring.positions[105 % 8]) != -1:
        failures.append("a dropped slot must also be cleared in the drafter's ring, or the drafter attends a stale key")
    ring.write(200, np.arange(3000, 3012, dtype=np.uint32), [None] * 12)
    if drafter.ingested[-1] != list(range(204, 212)) or not ring.held(np.arange(204, 212)).all() or ring.held(np.arange(200, 204)).any():
        failures.append(f"a write longer than the window keeps only its last {drafter.window} positions: ingested {drafter.ingested[-1]}")

    donor = ring_class(drafter)
    donor.write(40, np.arange(500, 508, dtype=np.uint32), [None] * 8)
    taker = ring_class(drafter)
    taker.copy_from(donor, np.arange(42, 46))
    if not taker.held(np.arange(42, 46)).all() or taker.held(np.array([41, 46])).any() or list(taker.tokens[np.arange(42, 46) % 8]) != [502, 503, 504, 505]:
        failures.append("a donor copy brings exactly the requested positions and their tokens")
    if not torch.equal(taker.ring.keys[:, np.arange(42, 46) % 8], donor.ring.keys[:, np.arange(42, 46) % 8]):
        failures.append("a donor copy brings the donor's keys for those positions")

    return failures


class GateDrafter:
    block = 5
    window = 4
    tap_names = ("o36",)
    dim = 2
    device = "cpu"

    def context(self):
        return {}


def gate_run(score, torch, tokens, wrong_at, minimum, sampled=False, eager_wrong_at=()):
    drafter = GateDrafter()

    def eager(context, token, position, rule):
        drafts = list(tokens[position + 1: position + 1 + drafter.block]) + [0] * max(0, position + 1 + drafter.block - len(tokens))
        return ([drafts[0] + 2] + drafts[1:] if position in eager_wrong_at else drafts), [0.5] * drafter.block

    def fast(context, token, position, rule):
        drafts, confidence = eager(context, token, position, rule)
        return ([drafts[0] + 1] + drafts[1:] if position in wrong_at else drafts), confidence

    hello = {"entries": ["o36"], "hidden": 2}
    streams = {(0, 7, 0): {"positions": {p: (None, None, token, 0x2 if p >= 5 else 0) for p, token in enumerate(tokens)}}}
    score.requests = lambda taps, _filter: (hello, streams)
    score.load_rows = lambda hello, records, columns: torch.zeros(len(records), 1, 2)
    score.drafters = lambda args: (drafter, [("eager", lambda *a: None, eager), ("fast", lambda *a: None, fast)])
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "gate.json"
        sys.argv = ["dspark_score.py", "--checkpoint", "-", "--taps", "-", "--gate", "--gate-anchors", str(minimum), "--out", str(out)] + (["--temperature", "1.0"] if sampled else [])
        status = score.main()
        return status, json.loads(out.read_text())


def gate_failures(torch):
    sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))
    spec = importlib.util.spec_from_file_location("dspark_score", ROOT / "tools" / "spec_offline" / "dspark_score.py")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    failures = []
    tokens = list(range(100, 130))
    status, report = gate_run(score, torch, tokens, set(), 20)
    if status != 0 or report["gate"]["first_draft_agreement"] != 1.0 or report["gate"]["anchors"] != 24 or report["eager"]["tokens_per_round"] != report["fast"]["tokens_per_round"]:
        failures.append(f"identical drafters on 24 anchors pass the gate: {status} {report.get('gate')}")
    status, report = gate_run(score, torch, tokens, {9}, 20)
    if status != 1 or not any("agreement" in reason for reason in report["gate"]["failed"]) or report["fast"]["tokens_per_round"] >= report["eager"]["tokens_per_round"]:
        failures.append(f"one first-draft disagreement in 24 anchors (95.8%) fails the 99% agreement bar: {status} {report.get('gate')}")
    status, report = gate_run(score, torch, tokens, {9}, 20, sampled=True, eager_wrong_at={9})
    if status != 0 or report["gate"]["first_draft_agreement"] >= 1.0 or report["gate"]["tokens_per_round_gap"] != 0.0 or report["gate"]["failed"]:
        failures.append(f"a sampled run reports a first-draft disagreement where both drafts miss, but gates only on tokens per round: {status} {report.get('gate')}")
    status, report = gate_run(score, torch, tokens, {9}, 20, eager_wrong_at={9})
    if status != 1 or report["gate"]["failed"] != ["first-draft agreement 0.9583, below 0.99"]:
        failures.append(f"the same disagreement fails a greedy run: {status} {report.get('gate')}")
    status, report = gate_run(score, torch, tokens, set(range(5, 15)), 20, sampled=True)
    if status != 1 or not any("tokens/round gap" in reason for reason in report["gate"]["failed"]):
        failures.append(f"a sampled run whose fast drafter loses ten first drafts in 24 fails on the tokens per round gap: {status} {report.get('gate')}")
    status, report = gate_run(score, torch, tokens, set(), 25)
    if status != 1 or report["gate"]["failed"] != ["24 anchors, below 25"]:
        failures.append(f"fewer anchors than the gate needs fail it even when the drafters agree: {status} {report.get('gate')}")
    return failures


def serving(draftd, drafter, contexts):
    server = draftd.Server.__new__(draftd.Server)
    server.latent, server.dspark, server.drafter, server.max_drafts, server.contexts = False, True, drafter, drafter.max_drafts, contexts
    server.stats = {"gaps": 0, "unavailable": 0, "drafted": 0, "draft_us": 0, "donor_fills": 0}
    server.timing = {"compute": 0.0, "context": 0}
    return server


def main():
    if importlib.util.find_spec("torch") is None:
        print("SKIP test_dspark_serving (needs torch)")
        return 0
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("block_draftd", ROOT / "tools" / "block_draftd.py")
    draftd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draftd)
    drafter = StubDrafter(torch)
    failures = ring_failures(draftd, torch, np, drafter, draftd.RingContext)

    class LeakyRing(draftd.RingContext):
        def write(self, first, tokens, taps):
            kept = self.ring.positions.clone()
            super().write(first, tokens, taps)
            cleared = (self.ring.positions == -1) & (kept != -1)
            self.ring.positions[cleared] = kept[cleared]

    if not any("stale key" in failure for failure in ring_failures(draftd, torch, np, StubDrafter(torch), LeakyRing)):
        failures.append("negative control: a ring that keeps stale slots in the drafter's table must fail the ring checks")

    if draftd.dspark_rule(drafter, (5, 0.0, 0.9, 4)) is not None:
        failures.append("greedy sampling (inverse temperature 0) drafts with no rule")
    seed, temperature, nucleus, top_k = draftd.dspark_rule(drafter, ((1 << 64) - 3, 2.0, 1.0, 7))
    if int(seed) != -3 or float(temperature) != 2.0 or float(nucleus) != 0.0 or int(top_k) != 7:
        failures.append("a seed at or above 2^63 wraps to its signed value, and nucleus 1.0 means no nucleus cut")
    if float(draftd.dspark_rule(drafter, (9, 1.0, 0.95, 0))[2]) != np.float32(0.95):
        failures.append("a nucleus below 1.0 passes through")

    committed = np.arange(0, 61, dtype=np.uint32) + 7000
    anchor, position = int(committed[60]), 60
    server = serving(draftd, drafter, OrderedDict())
    direct = draftd.RingContext(drafter)
    direct.write(52, committed[52:60], [None] * 8)
    server.contexts[11] = direct
    result = server.draft((11, anchor, position, committed, 4, (0, 0.0, 1.0, 0)))
    if result is None or result[0] != [7061, 7062, 7063, 7064] or drafter.drafted[-1][:2] != (anchor, position) or drafter.drafted[-1][2] is not None:
        failures.append(f"a held, matching window drafts depth-limited from the anchor: {result}")
    result = server.draft((11, anchor, position, committed, 9, (3, 1.0, 0.9, 0)))
    if result is None or len(result[0]) != drafter.max_drafts or drafter.drafted[-1][2] is None:
        failures.append("depth is capped at the drafter's five drafts, and sampled requests carry their rule")
    wrong = committed.copy()
    wrong[57] += 1
    before = len(drafter.drafted)
    if server.draft((11, int(wrong[60]), position, wrong, 4, (0, 0.0, 1.0, 0))) is not None or server.stats["unavailable"] != 1 or len(drafter.drafted) != before:
        failures.append("a committed token that differs from the ring's must answer unavailable without drafting")

    taker = draftd.RingContext(drafter)
    taker.write(56, committed[56:60], [None] * 4)
    server.contexts[12] = taker
    result = server.draft((12, anchor, position, committed, 4, (0, 0.0, 1.0, 0)))
    if result is None or server.stats["donor_fills"] != 1 or server.stats["gaps"] != 0 or not taker.held(np.arange(52, 60)).all():
        failures.append(f"a window missing positions another sequence holds with the same tokens is filled from that donor: {server.stats}")

    server = serving(draftd, drafter, OrderedDict())
    stranger = draftd.RingContext(drafter)
    stranger.write(52, committed[52:56] + 1, [None] * 4)
    short = draftd.RingContext(drafter)
    short.write(56, committed[56:60], [None] * 4)
    server.contexts[13], server.contexts[14] = stranger, short
    result = server.draft((14, anchor, position, committed, 4, (0, 0.0, 1.0, 0)))
    if result is None or server.stats["donor_fills"] != 0 or server.stats["gaps"] != 1 or short.held(np.arange(52, 56)).any():
        failures.append(f"a donor with other tokens is never copied; the window shrinks past the gap and still drafts: {server.stats}")

    failures += gate_failures(torch)

    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS DSpark serving path: the ring drops stale slots in both its tables, keeps the last window of a long write, takes donor slots with their keys "
          "and tokens; the sampling rule wraps seeds and maps nucleus 1.0; drafts are depth-capped; a token mismatch answers unavailable without drafting; donors fill only matching positions and a gap shrinks the window; "
          "the negative control (stale drafter slots) fails; the fast/eager gate passes agreeing drafters and fails a greedy disagreement, a sampled tokens-per-round gap or too few anchors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
