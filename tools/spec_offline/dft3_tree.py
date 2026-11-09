"""DFT3 node tables: a trunk and the drafter's runner-up tokens as one-node branches.

A node is (token, parent_index, depth, source_bit, score). Node 0 is the anchor; trunk node
d sits at index d with parent d - 1; an alternate at step s is a child of trunk node s at
depth s + 1 and starts its own branch, so the table lists the trunk first and then the
branches, as SparkSpeculationTreeLayoutBuild requires.
"""

ROOT_PARENT = 0xFFFFFFFF


def draft_nodes(anchor, drafts, confidence, max_nodes, source_bit, alternates=None, alternate_scores=None, max_branches=0, min_score=0.0):
    nodes = [(int(anchor), ROOT_PARENT, 0, source_bit, 1.0)]
    trunk = list(drafts[:max(0, max_nodes - 1)])
    for index, token in enumerate(trunk):
        nodes.append((int(token), index, index + 1, source_bit, float(confidence[index]) if confidence else 0.0))
    if not alternates or max_branches <= 0:
        return nodes
    ranked = []
    for step in range(min(len(trunk), len(alternates))):
        score = float(alternate_scores[step]) if alternate_scores else 0.0
        if int(alternates[step]) != int(trunk[step]) and score >= min_score:
            ranked.append((score, step))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    for score, step in ranked[:max_branches]:
        if len(nodes) >= max_nodes:
            break
        nodes.append((int(alternates[step]), step, step + 1, source_bit, score))
    return nodes
