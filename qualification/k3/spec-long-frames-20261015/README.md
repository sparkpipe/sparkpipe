# K3 long verify frames on code edits (2026-10-15)

Hold 2 of the long-frame speculation work, K3 part (12:55-13:41Z). Arms:
C = `36f24360e` (verify removed from the K3 descriptor), P = `1134081e3`.
Order C1 P1 P2 C2 on k3-lanes/lane8, each after a KV snapshot and prefix-index
wipe.

- `generation-k3-<arm>.json`: per task tokens, seconds, time to first token
  and decode tok/s on the 20-task corpus.
- `sampled.jsonl`: the story prompt per arm, greedy x4, T=1 top-p 0.95 seeds
  101-104, T=1 seeds 201-202, with tokens and tok/s.
- `cmp.txt`: engine verify counters and stream comparisons of the whole hold
  (GLM and K3).
