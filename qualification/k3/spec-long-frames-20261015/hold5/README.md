# K3 long frames: hold 5 (2026-10-15, 14:34-14:57Z)

Node adapter (`libk3_tp16_serving_adapter.so`) and residentd of `a397d67b0` on
every arm, hub K3 adapter of `a397d67b0`. Hub API `a397d67b0` (KT, frames of
a priority-0 request capped at eight chained tokens) against `2b2fa2a44` (KP,
the cap only below a served foreground priority). Order KP1 KT1 KP2 on
k3-lanes/lane8, each after a KV snapshot and prefix-index wipe.

- `generation-<run>.json`: per task output tokens, seconds and time to first
  token on the 20-task edit corpus.
- `cmp.txt`: the engine's last `batch_speculation` line per run, decode tok/s
  per edit style, and the stream comparison against KT1 (editsr_09 differs).
- `hold.log`: arm hashes and the run sequence.
