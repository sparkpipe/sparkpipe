# GLM Full long frames: holds 3 and 4 (2026-10-15)

Hold 3 (`../hold3/`, 13:42-14:11Z): tip `a397d67b0` (arm T), the tip without
the scatter choice `0d9319a79` (N), control `d2e0fda7b` (C, verify removed
from the descriptor). Order T1 CL TL NL T2; L runs use the long-context corpus
`edit_corpus_long.json`.

Hold 4 (this directory, 14:11-14:33Z): node adapter and residentd of
`a397d67b0` on every arm; hub API `a397d67b0` (T) against `2b2fa2a44` (P, the
background-only frame cap). Order P1 T3 PL P2.

Each run follows a KV snapshot and prefix-index wipe of lane 6.

- `generation-<run>.json`: per task output tokens, seconds and time to first
  token on the 20-task edit corpus.
- `cmp.txt`: the engine's last `batch_speculation` line per run (frames,
  drafts, kept drafts, chain and single costs, frames per row bucket), the
  decode tok/s per edit style and the stream comparison against T3.
- `hold.log`: arm hashes and the run sequence.
