# GLM Full long verify frames on code edits (2026-10-15)

Hold 1 of the long-frame speculation work, frames inside one cache block.
Arms: N = spec build with verify removed from the GLM descriptor
(`02bd01e7c`), S = `a161d0de7`. Run order N1 S1 S2 N2 on lane 6, each after a
KV snapshot and prefix-index wipe, 2026-10-15 11:27-12:15Z.

- `generation-<arm>.json`: per task prompt and output tokens, finish reason,
  total seconds, time to first token and decode tok/s, from
  `tools/spec_edit_corpus.py generate` on the 20-task corpus
  (`qualification/spec_offline/k3-edits-20261015/corpus.json`).
- `cmp.txt`: engine verify counters per arm, golden and greedy-prompt
  comparisons against N1, stream identity and per-style speed.
- `hold.log`: binary hashes per arm and activation times.
