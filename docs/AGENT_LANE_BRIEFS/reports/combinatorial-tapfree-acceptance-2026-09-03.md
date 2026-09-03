# Combinatorial tap-free speculator acceptance (glm5.3 corpus)

Date: 2026-09-03. Harness: /home/spec/draft_service/acceptance_combinatorial.py on the RTX 5090
(34.5 s for the full grid, no subsampling). Corpus: bench_corpus.json (92 ds4_eval quality
fixtures, 20,738 anchors, anchor >= 16, depth cap 16, branching k=4). Acceptance = longest tree
path matching the recorded continuation; exact for greedy targets, a proxy if fixtures were
sampled. All three speculators exploit within-sequence repetition only, so absolute numbers
reflect corpus repetitiveness.

## Singles: chain (k=1, B=16) vs tree (k=4, best B=64)

| source | cov | chain acc | tree acc | acc given proposal (tree) | P(>=8) |
|---|---|---|---|---|---|
| bigram | .569 | 0.51 | 1.00 | 1.76 | .022 |
| ngram3 (3->2 backoff) | .317 | 0.77 | 1.22 | 3.84 | .062 |
| suffix_match (LLMA, 16->2) | .317 | 1.10 | 1.25 | 3.93 | .062 |

## Combos (k=4, B=64 unless noted)

| sources | acc B=8 | acc B=64 | P(>=2) | P(>=8) | unique wins |
|---|---|---|---|---|---|
| bigram+ngram3 | 0.89 | 1.43 | .218 | .064 | 0-4 |
| bigram+suffix | 1.01 | 1.51 | .219 | .066 | 3-6 |
| ngram3+suffix | 0.87 | 1.31 | .168 | .066 | 0-4 |
| all three | 1.01 | 1.53 | .217 | .070 | 6-7 |

Unique wins = anchors (of 20,738) where the combo beats EVERY constituent at the same budget.

## What the data says

- Composition buys ~22-43% over the best single source, and the mechanism is coverage union
  (per-anchor oracle-max), not cross-source synergy: merged tree == per-anchor max of
  constituents on 99.6% of anchors. Still a real win — you get the max without knowing which
  source is right.
- Coverage is the binding constraint: bigram fires on 57% of anchors, ngram3/suffix on 31.7%
  (identical sets — both bottom out at "this bigram context occurred before"). Combos inherit
  bigram's coverage; no combo extends it.
- Width buys depth mostly for bigram (chain 0.51 -> tree 1.00); suffix_match's top-1 chain is
  usually right (1.10 -> 1.25). Singles saturate at B=32-64; combos still gain at 64.
- ngram3 is subsumed by suffix_match (their B=8 combo == suffix alone). If one gets cut, it is
  ngram3.
- Drafter wall time <= 0.94 ms/anchor for the three-source combo, pure numpy — irrelevant next
  to a target forward.

## Consequence for the farm

suffix_match earns a speculator slot (tap-free, cheap, strongest single, decorrelated mechanism
from bigram). Proposed: farm bit 16 (SPEC_SUFFIX), controllable via the speculator mask like the
rest. Tap-conditioned sources (DFlash2, and MTP on-target) enter the same measurement once the
tap-extraction seam lands.
