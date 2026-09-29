from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .drafters import Drafter, chain_from
from .streams import Stream
from .tree import Node, Shape, build_tree, build_trie, path_tokens, resolve_tree

DEPTH_MAX = 7


@dataclass
class PositionTable:
    depth: int
    reached: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    positions: int = 0
    top1_agree: int = 0

    def __post_init__(self):
        self.reached = [0] * self.depth
        self.accepted = [0] * self.depth

    def acceptance(self) -> list[float | None]:
        return [self.accepted[i] / self.reached[i] if self.reached[i] else None for i in range(self.depth)]

    def tau(self, k: int) -> float:
        rates = self.acceptance()
        total = reach = 1.0
        for i in range(k):
            rate = rates[i]
            if rate is None:
                break
            reach *= rate
            total += reach
        return total

    def as_dict(self) -> dict:
        return {"positions": self.positions, "reached": list(self.reached), "accepted": list(self.accepted),
                "acceptance_per_position": self.acceptance(), "top1_agreement": self.top1_agree / self.positions if self.positions else None,
                "tau": {str(k): round(self.tau(k), 4) for k in range(1, self.depth + 1)}}


def anchors(stream: Stream) -> range:
    return range(stream.prompt - 1, stream.length - 1)


def position_table(stream: Stream, drafter: Drafter, taps: np.ndarray | None, depth: int = DEPTH_MAX) -> PositionTable:
    table = PositionTable(depth)
    drafter.reset(stream.tokens, stream.prompt, taps)
    for anchor in anchors(stream):
        remaining = stream.length - 1 - anchor
        requested = min(depth, remaining)
        draft = chain_from(drafter, anchor, requested)
        table.positions += 1
        if draft and draft[0] == stream.tokens[anchor + 1]:
            table.top1_agree += 1
        matched = True
        for index in range(requested):
            table.reached[index] += 1
            matched = matched and index < len(draft) and draft[index] == stream.tokens[anchor + 1 + index]
            if matched:
                table.accepted[index] += 1
            else:
                break
        drafter.observe(anchor + 1, stream.tokens[anchor + 1])
    return table


@dataclass
class RoundResult:
    rounds: int = 0
    proposed_rows: int = 0
    accepted: int = 0
    committed: int = 0
    accepted_histogram: list[int] = field(default_factory=list)
    stream_exact: bool = True

    def tokens_per_round(self) -> float:
        return self.committed / self.rounds if self.rounds else 0.0

    def as_dict(self) -> dict:
        return {"rounds": self.rounds, "rows": self.proposed_rows, "accepted": self.accepted, "committed": self.committed,
                "tokens_per_round": round(self.tokens_per_round(), 4), "accepted_histogram": list(self.accepted_histogram),
                "stream_exact": self.stream_exact}


def round_replay(stream: Stream, drafter: Drafter, taps: np.ndarray | None, shape: Shape, members: list[Drafter] | None = None) -> RoundResult:
    result = RoundResult(accepted_histogram=[0] * (shape.rows + 1))
    committed = list(stream.tokens[:stream.prompt + 1])
    drafter.reset(stream.tokens, stream.prompt, taps)
    for member in members or []:
        member.reset(stream.tokens, stream.prompt, taps)
    anchor = stream.prompt
    while anchor < stream.length - 1:
        remaining = stream.length - 1 - anchor
        max_depth = min(shape.depth if shape.kind != "trie" else shape.rows - 1, remaining)
        if shape.kind == "trie":
            nodes: list[Node] = build_trie(members or [drafter], anchor, shape.rows, max_depth)
        else:
            nodes = build_tree(drafter, anchor, shape, max_depth)
        truth_after = list(stream.tokens[anchor + 1:anchor + 1 + remaining])
        if nodes:
            accepted, committed_count, _, best_node = resolve_tree(nodes, truth_after)
            committed_count = min(committed_count, remaining)
            round_tokens = path_tokens(nodes, best_node)[:committed_count]
        else:
            accepted, committed_count, round_tokens = 0, 1, []
        round_tokens = round_tokens + truth_after[len(round_tokens):committed_count]
        result.rounds += 1
        result.proposed_rows += len(nodes) + 1
        result.accepted += accepted
        result.committed += committed_count
        result.accepted_histogram[min(accepted, shape.rows)] += 1
        for offset, token in enumerate(round_tokens):
            committed.append(token)
            drafter.observe(anchor + 1 + offset, token)
            for member in members or []:
                member.observe(anchor + 1 + offset, token)
        anchor += committed_count
    result.stream_exact = tuple(committed) == stream.tokens
    return result


def agreement(stream: Stream, candidate: Drafter, reference: Drafter, taps: np.ndarray | None) -> dict:
    candidate.reset(stream.tokens, stream.prompt, taps)
    reference.reset(stream.tokens, stream.prompt, taps)
    agree = total = 0
    for anchor in anchors(stream):
        left = candidate.expand(anchor, (), 1)
        right = reference.expand(anchor, (), 1)
        total += 1
        agree += int(bool(left) and bool(right) and left[0] == right[0])
        candidate.observe(anchor + 1, stream.tokens[anchor + 1])
        reference.observe(anchor + 1, stream.tokens[anchor + 1])
    return {"positions": total, "agree": agree, "agreement": agree / total if total else None}
