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

## Long frames and the measured wave cost (2026-10-15)

`rounds-long-*.json`: the same replay with chain depths up to 255
(`tools/spec_offline/tree.py` now accepts depths up to 255).

| k | editfull lookup / suffix | editsr lookup / suffix |
| ---: | --- | --- |
| 15 | 12.7 / 14.2 | 4.1 / 4.9 |
| 31 | 21.4 / 25.1 | 4.5 / 5.6 |
| 63 | 34.1 / 40.9 | 4.7 / 5.9 |
| 127 | 47.7 / 59.3 | 4.8 / 5.9 |
| 255 | 61.6 / 76.4 | 4.8 / 5.9 |

`wave-cost.jsonl`: the time to the first token of a request that adds r
random tokens to a cached 2,049-token prefix, max_tokens 1, through each
model's current prefill path, n=5 (K3) and n=1 (GLM Full), in one exclusive
hold each:

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K3 ms | 124 | 132 | 158 | 230 | 271 | 340 | 529 | 672 | 1021 |
| GLM Full ms | 57 | 67 | 83 | 122 | 182 | 234 | | | |

GLM Full requests of 64 and more new tokens failed in that probe (cause
not yet known). Priced with these costs, whole-file edits commit about
34-41 tokens per 529 ms round at k=63 and 48-59 per 672 ms at k=127 on
K3, about 65-88 tok/s (estimate); SEARCH/REPLACE edits need frames sized
to the match.
