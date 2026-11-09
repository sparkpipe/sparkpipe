#!/usr/bin/env python3
"""The draft server's DFT3 node table: a trunk, then the drafter's runner-up tokens as one-node
branches ranked by score, within the request's node budget and never a duplicate sibling."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))

from dft3_tree import ROOT_PARENT, draft_nodes  # noqa: E402

SOURCE = 2


def layout_accepts(nodes):
    first_child = {}
    branch = {0: 0}
    for row in range(1, len(nodes)):
        parent = nodes[row][1]
        if parent >= row or nodes[row][2] != nodes[parent][2] + 1:
            return False
        for sibling in range(parent + 1, row):
            if nodes[sibling][1] == parent and nodes[sibling][0] == nodes[row][0]:
                return False
        if parent not in first_child:
            if parent != row - 1:
                return False
            first_child[parent] = row
            branch[row] = branch[parent]
        else:
            if branch[parent] != 0:
                return False
            branch[row] = max(branch.values()) + 1
    return nodes[0][1] == ROOT_PARENT


class Dft3TreeNodesTest(unittest.TestCase):
    def test_chain_without_alternates(self):
        nodes = draft_nodes(7, [1, 2, 3], [0.9, 0.8, 0.7], 16, SOURCE)
        self.assertEqual(nodes, [(7, ROOT_PARENT, 0, SOURCE, 1.0), (1, 0, 1, SOURCE, 0.9), (2, 1, 2, SOURCE, 0.8), (3, 2, 3, SOURCE, 0.7)])
        self.assertEqual(draft_nodes(7, [1, 2, 3], [0.9, 0.8, 0.7], 16, SOURCE, [4, 5, 6], [0.1, 0.2, 0.3], 0), nodes)
        self.assertEqual(draft_nodes(7, [1, 2, 3], None, 16, SOURCE), [(7, ROOT_PARENT, 0, SOURCE, 1.0), (1, 0, 1, SOURCE, 0.0), (2, 1, 2, SOURCE, 0.0), (3, 2, 3, SOURCE, 0.0)])

    def test_alternates_become_one_node_branches_best_first(self):
        nodes = draft_nodes(7, [1, 2, 3], [0.9, 0.8, 0.7], 16, SOURCE, [4, 5, 6], [0.1, 0.3, 0.2], 2)
        self.assertEqual(nodes[:4], [(7, ROOT_PARENT, 0, SOURCE, 1.0), (1, 0, 1, SOURCE, 0.9), (2, 1, 2, SOURCE, 0.8), (3, 2, 3, SOURCE, 0.7)])
        self.assertEqual(nodes[4:], [(5, 1, 2, SOURCE, 0.3), (6, 2, 3, SOURCE, 0.2)])
        self.assertTrue(layout_accepts(nodes))

    def test_duplicate_and_weak_alternates_are_skipped(self):
        nodes = draft_nodes(7, [1, 2, 3], [0.9, 0.8, 0.7], 16, SOURCE, [1, 5, 6], [0.5, 0.3, 0.01], 4, min_score=0.05)
        self.assertEqual(nodes[4:], [(5, 1, 2, SOURCE, 0.3)])
        self.assertTrue(layout_accepts(nodes))

    def test_node_budget_caps_the_trunk_then_the_branches(self):
        nodes = draft_nodes(7, [1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6], 3, SOURCE, [5, 6, 8, 9], [0.4, 0.4, 0.4, 0.4], 4)
        self.assertEqual(len(nodes), 3)
        self.assertEqual([n[0] for n in nodes], [7, 1, 2])
        nodes = draft_nodes(7, [1, 2], [0.9, 0.8], 4, SOURCE, [5, 6], [0.4, 0.5], 4)
        self.assertEqual([n[0] for n in nodes], [7, 1, 2, 6])
        self.assertTrue(layout_accepts(nodes))

    def test_a_second_child_of_a_branch_is_never_emitted(self):
        for trial in range(50):
            drafts = [(trial * 7 + k) % 101 + 1 for k in range(6)]
            alternates = [(trial * 13 + k * 3) % 101 + 1 for k in range(6)]
            scores = [((trial + k) % 10) / 10.0 for k in range(6)]
            self.assertTrue(layout_accepts(draft_nodes(3, drafts, [0.5] * 6, 12, SOURCE, alternates, scores, 3)), f"trial {trial}")


if __name__ == "__main__":
    unittest.main(verbosity=1)
