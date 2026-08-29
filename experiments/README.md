# experiments/ — the R&D carve-out (operator directive 2026-08-30)

Code here is EXCLUDED from the code-size ratchet: this is where the
speculator-tournament hill-climb lives (tree-verify prototypes,
agreement-matrix replays, bandit routers) while we learn what to build.

Rules of the carve-out:
- Operational rigor is NOT carved out: TERM-only kills, spawn-captured
  pids, receipts for every claim, no silent fallbacks — the carve-out
  is about the code-size metric, not about discipline.
- Promotion: when results say what the production shape is, the code
  moves OUT of experiments/ into the real tree (modules/, runtime/...)
  and counts as authored code from that day, passing the full merge
  gates (ratchet, cyclomatic, Solutions/Codesize^2, DRY).
- Nothing here ships: no serving path may link into experiments/.
