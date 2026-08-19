# DFlash2 27B bench plan (the race protocol)

Pins the measurement protocol for the DFlash2-vs-gym-5.3 race so every
number is comparable, hardware-measured, and ladder-consistent. Applies to
both the W-series completion bench and the k-sweep.

## Axes (all hardware-measured on spark3, no estimates)

- no-spec baseline: the FP8/mixed 29.9GB pack, greedy, B1 = **8.02 tok/s**
  (HWM 8.03; O128 hash 80e8fb9d...). Re-measure on the same binary as the
  spec run, same session.
- spec: DFlash2 k (num_speculative_tokens), same pack, greedy target,
  greedy draft. Report spec tok/s, accepted/k per step, acceptance length
  (L = mean accepted drafts per step), first-position acceptance rate.
- correctness gate BEFORE speed: the parity harness (oracles
  assert_allclose) and the no-spec O128 hash must reproduce on the same
  binary; a spec number from a binary whose no-spec repro failed is
  fantasy and gets retracted.

## k-sweep

k = {4, 7, 8, 12} - do NOT inherit 7. The v1 curve (n=2 -> 2.00x @ 91.1%,
n=8 -> 3.36x @ 63.1%, n=12 -> 3.64x @ 50.5%) is the reference shape;
DFlash2's selector favors larger k, but B1 verify cost caps the win.
Report the full sweep table; the HWM cell is the best (k, tok/s) that
beats the no-spec baseline on the SAME prompt set.

## Prompt set (upstream-comparable, not O128-only)

Acceptance on the 128-token devcycle prompt (our DSpark L=0.735) is NOT
comparable to upstream's dataset means (DSpark 3.62 / DFlash2 4.80 on
GSM8K/MATH/HumanEval/MBPP/MT-Bench). Bench on MT-Bench-class chat turns
(>= 5 turns, a few hundred tokens each) so acceptance length is
meaningful; keep the O128 run as a sanity pin only.

## Report format (every bench post)

`DFlash2 k=<k> | spec <tok/s> | no-spec <tok/s> | L <x> | accepted/k <y> |
first-pos <z>% | hash <ok/FAIL> | prompt set <name> | verdict
keep/restore`

Verdict rule: keep the dspark/dflash2 config live ONLY if spec > no-spec
on the same prompt set with the hash gate green; otherwise restore the
BEFORE config (BF16 + MTP D=2) and record why.

## Race notes

- gym 5.3 owns spark2; our lane owns spark3. Never touch the other's box.
- The PR (sparkpipe/sparkpipe#670) is public - both sides can read the
  other's state. Our edge is the landed machinery (packer, oracles,
  port rails); theirs is a fresh start.
- Numbers posted to docs/PERF_DASHBOARD.md follow the existing cadence
  (every 15min or on a new HWM).
