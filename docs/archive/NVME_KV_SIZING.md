# NVMe bandwidth sizing for the JIT KV tier

The decision on the table: the internal NVMe is ~2x the external drive. If
5 GB/s per node covers the KV tier's steady-state demand, a 4-8 TB external
NVMe becomes the KV backing store and the internal drive keeps the weight
packs and the swap warm path.

**Every number below is an estimate, not a measurement.** The model and all
constants live in `tools/nvme_kv_estimate.py`; every table here is its
output, and `tests/test_nvme_kv_estimate.py` holds the arithmetic to the
roadmap it derives from. Re-run with measured constants when the ring
reports them — the assumed ones are flagged below and at the top of the
script. Method and per-model geometry come from the per-batch section of
`docs/PERF_ROADMAP_2026-08-01.md:236-522`; nothing is fitted.

## Assumed constants (unmeasured; the verdicts' sensitivity is stated)

```text
internal NVMe     5-7 GB/s per node    assumed; pinned nowhere in the tree
                                       (PERF_ROADMAP_2026-08-01.md:496-498)
external NVMe     2.5-3.5 GB/s/node    the user's "external is 2x slower"
resident KV       32 GB per node       assumed: the ~29 GiB production GPU
                                       pool (GLM52_B1024_JIT_KV_INTEGRATION.md
                                       :53-58) plus host headroom. THE
                                       sensitive constant: every spill
                                       crossover moves linearly with it.
ring              13 nodes, 2,840 GB/s aggregate at the 80% target
                                       (PERF_ROADMAP_2026-08-01.md:32-33)
```

The 16-node TP16/PP16 end state scales resident capacity to 512 GB and
divides ring demand by 16; the per-node verdicts below move ~20% in the
favourable direction and no classification flips. `--nodes 16` re-prices.

## Demand side: KV bytes per decode step, and when the tier is touched at all

A decode step re-reads every active sequence's KV (minus what token
selection prunes), so KV traffic scales with **batch x context**. Per-seq
per-step KV read, from the contracts via the roadmap:

```text
model       stored B/token   read per step                    source
K3          27,648           27,648 x ctx (full MLA re-read)  roadmap:292-294
GLM 5.2     89,856           89,856 x min(ctx, 2048 DSA sel.) roadmap:315-318
Qwen 3.6    65,536           65,536 x ctx (16 full-attn L)    roadmap:337-339
DSv4 Pro     8,919            8,919 x ctx; past ~128K slots    roadmap:365-369
                             only 1,152 selected slots read
DSv4 Flash   6,082            6,082 x ctx                     roadmap:386-389
MiMo 2.5    17,920           17,920 x ctx + 26.8 MB SWA fixed roadmap:401-406
```

Recurrent state (K3 KDA 455 MB fp32 / 227.5 MB bf16 per seq; Qwen GDN
26 MB) is read AND written every step. It is never tier material — paging
it would cost more than the KV it frees — so it is charged against the
resident budget for every active sequence first, and a batch whose state
alone exceeds the budget is infeasible at any drive speed.

Placement is per sequence, which is what the JIT tier's admission control
actually does (`GLM52_B1024_JIT_KV_INTEGRATION.md:62-66`): a sequence's
blocks are all upstairs or all on the drive. The working set exceeds
GPU+host — and the tier enters the decode path — at `B > floor(resident /
per-seq footprint)`:

```text
spill crossover B (resident 416 GB ring, fp32 K3 state)
model       2K ctx      128K ctx     1M ctx
K3          >1024*      ~110         ~14
GLM 5.2     2,260       35           4
Qwen 3.6    3,104       48           6
DSv4 Pro    22,700      355          44
DSv4 Flash  33,000      521          65
MiMo 2.5    11,300      175          22
*state, not KV, binds first: 1024 x 455 MB = 466 GB > 416 GB. With the
 roadmap's own bf16-state lever (PERF_ROADMAP_2026-08-01.md:454-471) K3
 B1024 x 2K is fully resident; bf16 is a B1024 prerequisite either way.
```

The step time is the roadmap's per-batch step bytes (fixed stream +
coverage x expert pool + per-seq state/KV, context-adjusted) divided by
2,840 GB/s. The estimator reproduces every 2K step-byte table in the
roadmap to rounding, and its no-residency mode reproduces the roadmap's
own prefetch-budget table (9.9 / 50.1 / 166.9 / 23.0 / 64.4 / 63.7 GB/s,
PERF_ROADMAP_2026-08-01.md:501-511) — the test asserts both. Pricing step
time at the 80% memory roofline is conservative for B256+: the 6.5 TFLOPS
compute wall makes those steps longer, which lowers demand.

## Supply side

Internal 5-7 GB/s per node (65-91 GB/s ring), external 2.5-3.5 GB/s
(32.5-45.5 GB/s ring). The JIT lookahead hides LATENCY — a read issued
`transfer_steps` early lands before its layer — but it cannot hide
BANDWIDTH: required bandwidth = spill bytes per step / step time, and when
demand exceeds supply the step stalls no matter how far ahead the planner
looks. That is why this is a bandwidth sizing, not a lookahead sizing.

## The verdict table

Steady-state NVMe read demand, per node, at the 80%-target step rate.
"resident" = zero demand, the working set fits GPU+host. Classes:
external-ok (<= 3.5), internal-5 (<= 5.0, the user's threshold),
internal-only (<= 7.0), infeasible (> 7.0).

```text
2K chat: every model, B1 through B1024 -> resident (0.00 GB/s per node).
    Sole exception: K3 B1024 fp32 state exceeds the resident budget by
    itself; with bf16 state it is resident too. Chat never touches the
    drive in steady state. The tier's 2K job is churn and resume, not
    bandwidth.

128K agent:
model       B8         B64          B256         B1024        cap B @5 GB/s
K3          resident   resident     50.9 infeas. state-over*  111 (118 bf16)
GLM 5.2     resident   1.74 ext-ok  11.1 infeas. 42.1 infeas. 130
Qwen 3.6    resident   49.8 infeas. 173  infeas. 206  infeas. 49
DSv4 Pro    resident   resident     resident     85.0 infeas. 379
DSv4 Flash  resident   resident     resident     90.7 infeas. 537
MiMo 2.5    resident   resident     46.4 infeas. 163  infeas. 182

1M worst case:
model       B8         B64          B256         B1024        cap B @5 GB/s
K3          resident   107  infeas. 169  infeas. state-over*  14
GLM 5.2     0.84 ext   3.61 int-5   12.6 infeas. 43.5 infeas. 95
Qwen 3.6    50.0 infeas. 196 infeas. 213  infeas. 217  infeas. 6
DSv4 Pro    resident   0.08 ext-ok  0.59 ext-ok  2.67 ext-ok  1,899
DSv4 Flash  resident   resident     149  infeas. 200  infeas. 66
MiMo 2.5    resident   117  infeas. 188  infeas. 211  infeas. 22
*K3 B1024: fp32 state alone is 466 GB against the 416 GB assumed budget.
```

Read of the table:

- **At 2K the 5 GB/s question is vacuous** — demand is zero through B1024
  on every model. Dedicating the external drive costs chat nothing.
- **At 128K the answer is yes under admission control.** GLM at the B64
  production point needs 1.74 GB/s per node — the external drive covers it
  with 2x headroom. Every model has an admission cap (last column) at
  which 5 GB/s suffices; Pro and Flash's caps are above the B256
  production batch, K3/Qwen's are below it. The cap is a scheduler
  number, not a hardware one: admit past it and the step stalls.
- **At 1M only the selection models survive batch.** DSv4 Pro — compressed
  storage plus top-k selection past 128K — serves B1024 x 1M at 2.67 GB/s
  per node: external-ok at the largest batch and context on the board.
  GLM's DSA cap holds B64 to 3.61. The full-re-read models (K3 MLA, Qwen,
  Flash, MiMo) are all infeasible at batch: a spilled sequence re-reads
  its whole context every step, and no drive fixes that.

Mitigations where the table says infeasible, in the order they pay:

1. **Admission control against the caps** — the JIT posture the B1024
   integration doc already mandates; the caps above are its numbers.
2. **fp8 KV** — halves stored and read bytes, doubling every cap and every
   capacity figure; the format option already exists
   (`K3_KV_BITS`, `inference/llms/kimi_k3/layer.cuh:31`).
3. **Token selection for the full-re-read models** — GLM's DSA cap and
   Pro's top-k are exactly what makes their 1M rows survivable; K3's MLA
   and Qwen's full attention have no such pruning, and their 1M rows are
   the price.
4. **Prefix dedup** — the tier is content-addressed (`content_hash`), so a
   shared system/tooling prefix is stored and read once per ring, not once
   per sequence. Agent workloads with large shared prefixes move their
   effective context — and their cap — proportionally.
5. **K3 bf16 state** — not a KV lever but the residency one: it is what
   puts K3 B1024 x 2K back inside the budget at all.
6. A smaller lookahead is NOT a mitigation: it trades the one resource the
   tier has in surplus (time) against the one it lacks (bandwidth).

## Capacity: 1 TB internal vs 4-8 TB external

Whole sequences per node's drive, state included, from the estimator
(reproduces the roadmap's capacity table, PERF_ROADMAP_2026-08-01.md
:481-489, to its stated formula):

```text
model       ctx      1 TB     4 TB     8 TB
K3          8K       1,467    5,869    11,738
K3          128K       245      980     1,961
K3          1M          33      135       271
GLM 5.2     8K       1,358    5,434    10,868
GLM 5.2     128K        84      339       679
GLM 5.2     1M          10       42        84
Qwen 3.6    8K       1,776    7,106    14,212
Qwen 3.6    128K       116      464       928
Qwen 3.6    1M          14       58       116
DSv4 Pro    8K      13,686   54,746   109,492
DSv4 Pro    128K       855    3,421     6,843
DSv4 Pro    1M         106      427       855
DSv4 Flash  8K      19,950   79,802   159,604
DSv4 Flash  128K     1,253    5,015    10,031
DSv4 Flash  1M         156      627     1,254
MiMo 2.5    8K       5,757   23,028    46,056
MiMo 2.5    128K       420    1,683     3,367
MiMo 2.5    1M          53      212       425
```

Honest deltas against the roadmap's table: it prints 1,359 / 85 for the
GLM 8K/128K rows (rounds where this floor-truncates), 20,427 / 1,284 / 160
for Flash (2-3% high against its own 6,082 B/token) and 29 for K3 @ 1M
(this model: 33). The formula is the roadmap's own; its Flash and K3-1M
cells do not reproduce under it. Estimates at this precision are +/-3%
either way; the 4x/8x scaling between columns is the content of the table.

The capacity story for the decision: at 128K agent context the 1 TB
internal drive holds 10-855 sequences per node depending on model — for
GLM that is 84, i.e. roughly one full B64 cohort plus churn headroom, and
the eviction clock will be turning constantly. The 4-8 TB external drive
turns "one cohort fits" into "the day's cold set fits", which is the
difference between the tier being a resume device and being a recompute
avoidance device.

## The resume-after-topology-switch guarantee

The guarantee on the books: model swap = drain + parallel NVMe load of
~100 GB/node in 14-20 s at 5-7 GB/s local NVMe, target <20 s
(`docs/techdebt.md:128-135`). Two findings:

- **The weight-pack warm path must stay on the internal drive.** The same
  ~100 GB/node at external 2.5-3.5 GB/s is 29-40 s: the <20 s guarantee
  breaks. This constrains the decision — the external drive can own KV,
  not the packs.
- **1 TB of eviction pressure does not change the KV side of the
  guarantee; bandwidth does.** The KV a paused cohort needs back after a
  TP16/PP16 switch is its working set, <= 32 GB/node by the resident
  budget above: 4.6-6.4 s at internal speeds, 9-13 s at external. Both
  fit the 20 s budget alongside the pack load if the two drives load in
  parallel (packs internal, KV external); serialised on one external
  drive, 100 + 32 GB is 38-53 s and the guarantee is gone. Eviction
  pressure enters only through a pin failure: the tier holds the paused
  working set with ~30x headroom at 1 TB, and the eviction clock skips
  pinned blocks (`cache/nvme_tier.c:333-336`), so admission that pins
  before the drain keeps every resume byte on-tier. The realistic failure
  is not eviction but absence: a cold sequence that was never published
  before the switch is a recompute, which at 128K of context is the one
  outcome the 20 s budget cannot absorb.

## Standing uncertainties

- The 32 GB/node resident budget is assumed. Halving it roughly halves
  every spill crossover and admission cap; doubling it turns most 128K
  rows resident. It is the first constant to measure.
- Internal 5-7 / external 2.5-3.5 GB/s are assumed (the roadmap flags the
  internal figure as PENDING). The verdict classes are stated against both
  ends of both ranges; the GLM 1M B64 row (3.61) is the only one that
  flips class inside the external range.
- Step times are priced at the 80% memory roofline. The compute wall
  (PERF_ROADMAP_2026-08-01.md:417-452) lengthens B256+ steps and lowers
  demand; Qwen's B1024 rows are unreachable as steps regardless (its
  subsection there), so its demand figures are upper bounds on a step that
  runs slower.
- Selection is modelled as a clean cap/cliff. If GLM's DSA or Pro's top-k
  selects CHURNING slot sets across steps, the effective read from the
  tier is worse than the capped figure — the selection index's own
  re-reads are unpriced (the roadmap's standing uncertainty,
  PERF_ROADMAP_2026-08-01.md:858-860).
- Flash's two sliding-window layers (bounded at 128 slots) are priced as
  full-context reads; at 1M this overstates its read by <2%.
- Uniform-routing coverage overestimates the expert stream at B8-B128
  (upper bound, roadmap:248-254), which shortens those steps and therefore
  RAISES estimated demand — conservative in the safe direction.
