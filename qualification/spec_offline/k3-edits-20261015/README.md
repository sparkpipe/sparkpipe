# K3 code-edit corpus: offline draft acceptance (2026-10-15)

Ten repository files (90-490 lines: Python, C, CUDA headers), each with one
localized change (rename the first defined function that is also used, at its
definition and every use), in two reply styles: the whole updated file
(`editfull`) and SEARCH/REPLACE blocks (`editsr`). Built with
`tools/spec_edit_corpus.py build` (`corpus.json`), rendered with K3's chat
template (thinking off), generated greedy on the production K3 engine
(release 1d89e0377 plus the sampled-decode change, 2026-10-15) with
`tools/spec_edit_corpus.py generate` (`streams/*.u32`, `generation.json`:
17,905 output tokens, every reply ended on its own stop token). Replayed with
`python3 -m tools.spec_offline.t0 acceptance|rounds --model k3` (the engine's
lookup rules: suffix match of 3-8 tokens over the request's own history).

| Class | Drafter | Tokens per round, k=1 / 3 / 7 | Rows per round at k=7 |
| --- | --- | --- | --- |
| editfull (15,902 positions) | lookup | 1.95 / 3.76 / 7.01 | 7.62 |
| editfull | suffix | 1.99 / 3.90 / 7.56 | 7.73 |
| editsr (2,003 positions) | lookup | 1.70 / 2.57 / 3.44 | 4.60 |
| editsr | suffix | 1.78 / 2.84 / 3.99 | 5.75 |

Every replay kept the committed stream identical to the no-speculation
stream (asserted by the tool).

Pricing (estimate, not measured): K3 TP16 has no verify wave yet, so the
cost of a verify frame of r rows is taken from batched decode of r
sequences on the same fleet (r=1 62.5 ms, 2 87 ms, 4 111 ms, 8 138 ms;
2026-10-07/08). Whole-file edits: about 7.0-7.6 tokens per 135-137 ms
round, 52-55 tok/s against 16 without speculation (3.2-3.4x).
SEARCH/REPLACE edits: about 3.4-4.0 tokens per 115-123 ms round, 30-32 tok/s
(1.9-2.0x). Break-even at 8 rows is about 2.2 tokens per round. Not
included: per-round host and draft overhead (about 2-3 ms), sampled
traffic (chat runs at temperature 1, which needs a sampling-correct verify
rule), and the earlier finding that prose, chat and code generation stay
below no-speculation with these drafters (lanes/spec-offline.md, GLM Flash).
