from __future__ import annotations

import re
from dataclasses import dataclass

from .drafters import Drafter

CHAIN = re.compile(r"^chain-k(\d+)$")
TREE = re.compile(r"^tree-top(\d+)d(\d+)-r(\d+)$")
TRIE = re.compile(r"^trie-([a-z0-9]+(?:\+[a-z0-9]+)+)-r(\d+)$")
ROOT = -1


@dataclass(frozen=True)
class Shape:
    kind: str
    depth: int
    width: int
    rows: int
    members: tuple[str, ...]

    @property
    def text(self) -> str:
        if self.kind == "chain":
            return f"chain-k{self.depth}"
        if self.kind == "tree":
            return f"tree-top{self.width}d{self.depth}-r{self.rows}"
        return f"trie-{'+'.join(self.members)}-r{self.rows}"


def parse_shape(text: str) -> Shape:
    if text == "adaptive":
        return Shape("adaptive", 7, 1, 8, ())
    match = CHAIN.match(text)
    if match:
        depth = int(match.group(1))
        if depth > 31:
            raise ValueError(f"{text}: chain depth above 31")
        return Shape("chain", depth, 1, depth + 1, ())
    match = TREE.match(text)
    if match:
        width, depth, rows = (int(group) for group in match.groups())
        if width < 2 or depth < 1 or rows < 2:
            raise ValueError(f"{text}: a tree needs top>=2, d>=1 and r>=2")
        return Shape("tree", depth, width, rows, ())
    match = TRIE.match(text)
    if match:
        members = tuple(match.group(1).split("+"))
        rows = int(match.group(2))
        if len(set(members)) != len(members) or rows < 2:
            raise ValueError(f"{text}: trie members must be distinct and r>=2")
        return Shape("trie", rows - 1, 1, rows, members)
    raise ValueError(f"unknown shape {text}; use chain-kN, adaptive, tree-topBdD-rR or trie-a+b-rR")


@dataclass
class Node:
    token: int
    parent: int
    depth: int


def build_tree(drafter: Drafter, anchor: int, shape: Shape, max_depth: int) -> list[Node]:
    nodes: list[Node] = []
    frontier: list[tuple[int, tuple[int, ...]]] = [(ROOT, ())]
    budget = shape.rows - 1
    depth = 0
    while frontier and depth < max_depth and len(nodes) < budget:
        following: list[tuple[int, tuple[int, ...]]] = []
        width = shape.width if (shape.kind == "tree" and depth < shape.depth) else 1
        if shape.kind == "chain" and depth >= shape.depth:
            break
        for parent, prefix in frontier:
            for token in drafter.expand(anchor, prefix, width):
                if len(nodes) >= budget:
                    break
                nodes.append(Node(token, parent, depth + 1))
                following.append((len(nodes) - 1, prefix + (token,)))
        frontier = following
        depth += 1
    return nodes


def build_trie(drafters: list[Drafter], anchor: int, rows: int, max_depth: int) -> list[Node]:
    nodes: list[Node] = []
    index: dict[tuple[int, ...], int] = {(): ROOT}
    budget = rows - 1
    prefixes: list[tuple[int, ...] | None] = [() for _ in drafters]
    for depth in range(max_depth):
        for member, drafter in enumerate(drafters):
            prefix = prefixes[member]
            if prefix is None:
                continue
            choices = drafter.expand(anchor, prefix, 1)
            if not choices:
                prefixes[member] = None
                continue
            prefix = prefix + (choices[0],)
            if prefix not in index:
                if len(nodes) >= budget:
                    return nodes
                nodes.append(Node(choices[0], index[prefix[:-1]], depth + 1))
                index[prefix] = len(nodes) - 1
            prefixes[member] = prefix
    return nodes


def resolve_tree(nodes: list[Node], truth_after_anchor: list[int]) -> tuple[int, int, int]:
    accepted_depth = [0] * len(nodes)
    best_depth = 0
    accepted_nodes = 0
    for index, node in enumerate(nodes):
        parent_depth = 0 if node.parent == ROOT else accepted_depth[node.parent]
        if (node.parent == ROOT or parent_depth != 0) and parent_depth < len(truth_after_anchor) and node.token == truth_after_anchor[parent_depth]:
            accepted_depth[index] = parent_depth + 1
            accepted_nodes += 1
            best_depth = max(best_depth, accepted_depth[index])
    return best_depth, best_depth + 1, accepted_nodes


def resolve_tree_rows(tokens: list[int], parents: list[int], verifier: list[int]) -> tuple[int, int]:
    accepted_depth = [0] * len(tokens)
    best_depth = 0
    best_node = ROOT
    for index, (token, parent) in enumerate(zip(tokens, parents)):
        parent_depth = 0 if parent == ROOT else accepted_depth[parent]
        row = 0 if parent == ROOT else parent + 1
        if (parent == ROOT or parent_depth != 0) and token == verifier[row]:
            accepted_depth[index] = parent_depth + 1
            if accepted_depth[index] > best_depth:
                best_depth, best_node = accepted_depth[index], index
    bonus_row = 0 if best_node == ROOT else best_node + 1
    committed = best_depth + 1 if bonus_row < len(verifier) else best_depth
    return best_depth, committed
