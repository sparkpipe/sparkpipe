from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .streams import Stream

INDEX_TOP_K = 2048
ROWS_MIN = 2
ROWS_MAX = 8
LOOKUP_MIN_MATCH = 3
LOOKUP_MAX_MATCH = 8
MAX_POSITIONS = 1048576
PREFILL_WAVE_ROWS = 128
REGIME_UNSPLIT, REGIME_SPLIT, REGIME_SELECTED = 0, 1, 2
MASK64 = (1 << 64) - 1

Drafter = Callable[[int, int], list[int] | None]


def graph_regime(context: int, split: int) -> int:
    if context > INDEX_TOP_K:
        return REGIME_SELECTED
    bound = min(context, INDEX_TOP_K)
    return REGIME_SPLIT if split != 0 and bound >= split else REGIME_UNSPLIT


def rows_fit(position: int, rows: int, split: int, max_positions: int = MAX_POSITIONS) -> int:
    if rows == 0 or position >= max_positions:
        return 0
    regime = graph_regime(position + 1, split)
    fit = 1
    while fit < rows and position + fit < max_positions and graph_regime(position + fit + 1, split) == regime:
        fit += 1
    return fit


def verify_depth(budget: int, produced: int, rows_max: int, position: int, split: int, max_positions: int = MAX_POSITIONS) -> int:
    if rows_max < ROWS_MIN or produced >= budget or budget - produced < 2:
        return 0
    depth = min(budget - produced - 1, rows_max - 1)
    fit = rows_fit(position, depth + 1, split, max_positions)
    return 0 if fit == 0 else fit - 1


def depth_cap_next(cap: int, proposed: int, accepted: int, cap_max: int) -> int:
    if cap_max == 0:
        return 0
    if cap == 0 or cap > cap_max:
        cap = cap_max
    if proposed != 0 and accepted >= proposed:
        following = cap_max if cap > cap_max // 2 else 2 * cap
    else:
        following = accepted + 1
    return min(following, cap_max)


def resolve_chain(draft: list[int], truth_after_anchor: list[int]) -> tuple[int, int]:
    accepted = 0
    while accepted < len(draft) and draft[accepted] == truth_after_anchor[accepted]:
        accepted += 1
    return accepted, accepted + 1


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


class LookupHistory:
    def __init__(self, min_match: int = LOOKUP_MIN_MATCH, max_match: int = LOOKUP_MAX_MATCH):
        self.history: list[int] = []
        self.min_match = min_match
        self.max_match = max_match

    def observe(self, position: int, tokens) -> None:
        if position > len(self.history):
            raise ValueError("lookup history has a gap")
        del self.history[position:]
        self.history.extend(tokens)

    def find(self, anchor: int) -> tuple[int, int]:
        history = self.history
        if anchor >= len(history):
            return 0, 0
        best_length = best_end = 0
        end = anchor
        while end > 0 and best_length < self.max_match:
            length = 0
            while length < self.max_match and length < end and history[end - 1 - length] == history[anchor - length]:
                length += 1
            if length > best_length:
                best_length, best_end = length, end - 1
            end -= 1
        return (best_length, best_end) if best_length >= self.min_match else (0, 0)

    def draft(self, anchor: int, requested: int) -> list[int] | None:
        length, source_end = self.find(anchor)
        if length == 0:
            return None
        tokens = []
        for count in range(requested):
            source = source_end + 1 + count
            if source > anchor:
                break
            tokens.append(self.history[source])
        return tokens or None


def synthetic_drafter(truth: tuple[int, ...], accept_milli: int, seed: int, vocab: int) -> Drafter:
    def draft(anchor: int, requested: int) -> list[int] | None:
        tokens = []
        for index in range(requested):
            position = anchor + 1 + index
            if position >= len(truth):
                break
            expected = truth[position]
            draw = mix64((seed ^ ((position << 8) & MASK64) ^ index) & MASK64) % 1000
            tokens.append(expected if draw < accept_milli else (expected + 1) % vocab)
        return tokens or None
    return draft


@dataclass
class ChainCounts:
    frames: int = 0
    verify_frames: int = 0
    plain_frames: int = 0
    plain_frame_tokens: int = 0
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    plain_steps: int = 0
    tokens: int = 0
    rows: list[int] = field(default_factory=lambda: [0] * (ROWS_MAX + 1))
    committed: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"frames": self.frames, "verify_frames": self.verify_frames, "plain_frames": self.plain_frames,
                "plain_frame_tokens": self.plain_frame_tokens, "rounds": self.rounds, "proposed": self.proposed,
                "accepted": self.accepted, "plain_steps": self.plain_steps, "tokens": self.tokens, "rows": list(self.rows)}


def frame_steps(frame: int, block: int, committed: int, length: int) -> int:
    steps = min(frame, length - committed)
    block_remaining = block - ((committed - 1) % block)
    return min(steps, block_remaining)


def simulate_chain(stream: Stream, drafter: str, vocab: int, rows: int = 8, frame: int = 8, block: int = 64, split: int = 64,
                   fixed_depth: bool = False, seed: int = 1, tap_drafter: Drafter | None = None) -> ChainCounts:
    if rows < ROWS_MIN or rows > ROWS_MAX or frame < 1 or frame > 32 or block < 1:
        raise ValueError("rows must be 2..8, frame 1..32, block >= 1")
    truth = stream.tokens
    prompt, length = stream.prompt, stream.length
    lookup = LookupHistory()
    lookup.observe(0, truth[:prompt + 1])
    available = prompt <= PREFILL_WAVE_ROWS
    if drafter == "lookup":
        function: Drafter = lookup.draft
        gated = False
    elif drafter.startswith("synthetic:"):
        function = synthetic_drafter(truth, int(drafter.split(":", 1)[1]), seed, vocab)
        gated = True
    elif drafter == "tap":
        if tap_drafter is None:
            raise ValueError("drafter 'tap' needs tap_drafter")
        function = tap_drafter
        gated = True
    else:
        raise ValueError(f"unknown chain drafter {drafter}")
    counts = ChainCounts()
    counts.committed = list(truth[:prompt + 1])
    cap = rows - 1
    committed = prompt + 1
    counts.tokens = length - prompt
    while committed < length:
        budget = frame_steps(frame, block, committed, length)
        produced = 0
        counts.frames += 1
        while True:
            anchor = committed + produced - 1
            depth = verify_depth(budget, produced, rows, anchor, split)
            if not fixed_depth and depth > cap:
                depth = cap
            draft = None
            if depth != 0 and (available or not gated):
                draft = function(anchor, depth)
            if not draft:
                if produced == 0:
                    counts.plain_frames += 1
                    counts.plain_frame_tokens += budget
                    lookup.observe(committed, truth[committed:committed + budget])
                    counts.committed.extend(truth[committed:committed + budget])
                    produced = budget
                    break
                lookup.observe(committed + produced, truth[committed + produced:committed + produced + 1])
                counts.committed.append(truth[committed + produced])
                counts.plain_steps += 1
                produced += 1
                if produced >= budget:
                    break
                continue
            if produced == 0:
                counts.verify_frames += 1
            verifier = list(truth[anchor + 1:anchor + 2 + len(draft)])
            accepted, committed_count = resolve_chain(draft, verifier)
            cap = depth_cap_next(cap, len(draft), accepted, rows - 1)
            lookup.observe(anchor + 1, truth[anchor + 1:anchor + 1 + committed_count])
            counts.committed.extend(truth[anchor + 1:anchor + 1 + committed_count])
            counts.rounds += 1
            counts.proposed += len(draft)
            counts.accepted += accepted
            counts.rows[len(draft) + 1] += 1
            produced += committed_count
            if produced >= budget:
                break
        committed += produced
        available = True
    return counts
