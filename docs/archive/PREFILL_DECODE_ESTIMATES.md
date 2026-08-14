# Prefill/decode estimates — six models, two topologies, five batch points

Every table below is the verbatim output of `tools/perf_estimate.py`
(2026-08-01). The estimator REUSES the roadmap's byte law by importing
`tools/nvme_kv_estimate.py` — same per-model geometry, same coverage
function, same `step_gb` — and adds the terms the roadmap prices only in
prose: launch overhead (eager vs CUDA-graph replay), the 6.5 TFLOPS compute
wall, TP collectives and PP stage transport per topology, and a
chunked-prefill model. `tests/test_perf_estimate.py` holds it to the
roadmap's per-batch tables and crossover batches; it is wired into the
Makefile's PYTHON_TESTS and `tools/gates.sh`. **Every number here is an
estimate; the unmeasured constants are flagged in the file header and
repeated below.**

## The model in one paragraph

Decode step time = max(bandwidth term, compute term) + launch overlay +
min(TP collective, PP transport). Bandwidth term = step bytes (fixed
stream + coverage × expert pool + per-seq state/KV, the roadmap's
per-batch law, plus the shared-expert stream the coverage law omits — see
`Perf.shared_gb`) over nodes × 273 GB/s × eta 0.80. Compute term = batch ×
per-token FLOPs over the nodes' rate — either every GEMM class at the
measured 6.5 TFLOPS/node QKVO wall (default; pessimistic for FP8/FP4
experts, roadmap:862-864) or per-precision peaks. Launch overlay = static
LM_LAUNCH count × launch_ns, or (stages + 1) × launch_ns under graph
replay. TP collective = 2 all-reduces/layer of B × hidden BF16 at the
fabric wire rate + per-AR latency floor. PP transport = (stages − 1) ×
(per-row payload / wire + hop floor). The recipe column picks whichever
overlay is cheaper; PP wins everywhere on these numbers, which is the
roadmap's own conclusion (TP16_DUAL_RAIL_SPEED.md:86-88 — TP is a
throughput topology, PP keeps single-stream latency).

## Assumed constants (unmeasured; sensitivity stated where it matters)

```text
launch_ns        2 us default (range 2-5)   PENDING — calibration doc:155-157
eta_bw           0.80                        measured, 3 derivations (calib:87-97)
QKVO wall        6.5 TFLOPS/node             measured on glm52 (calib:66-73);
                                             applied to ALL GEMM classes =
                                             pessimism for FP8/FP4 experts
ring AR latency  29 us/AR                    ASSUMPTION: hop floor, pipelined
switch AR/hop    15 us                       ASSUMPTION (TP16 doc:36-37 says
                                             "assumed" too)
prefill MFU      0.45 (range 0.35-0.55)      ASSUMPTION, nothing in-tree
chunk            256 tokens                  glm52.json:36
K3 launches      1,974 static vs 3,276 roadmap hand count — PENDING gap,
                                             roadmap:845-846 carries +-10%
SM specs         64K regs, 2048 threads/SM   ASSUMPTIONS (occupancy only);
                                             128 KB smem is layout.cuh:21
ptxas table      build server /opt/sparkpipe-topo/build/cuda13_sm121a_gate/
                 logs/*.ptxas.txt, CUDA 13.3, sm_121a, production flags
```

## Decode — 13-node ring (2,839 GB/s @ eta 0.80), eager launches, 2K ctx

```text
model           B  tok/s/seq   agg tok/s recipe binding    top-3 bottlenecks (share)
k3              1       19.1        19.1     PP bandwidth  bandwidth 92%, launch 8%, compute 5%
k3              8       8.73        69.8     PP bandwidth  bandwidth 95%, compute 17%, launch 3%
k3             64       2.38         152     PP bandwidth  bandwidth 97%, compute 37%, transport 2%
k3            256       1.50         384     PP bandwidth  bandwidth 95%, compute 94%, transport 5%
k3           1024       0.38         388     PP compute    compute 95%, bandwidth 34%, transport 5%

glm52           1       48.8        48.8     PP bandwidth  bandwidth 91%, launch 8%, compute 4%
glm52           8       14.3         115     PP bandwidth  bandwidth 97%, compute 10%, launch 2%
glm52          64       4.20         269     PP bandwidth  bandwidth 99%, compute 24%, launch 1%
glm52         256       3.48         891     PP bandwidth  bandwidth 98%, compute 80%, transport 1%
glm52        1024       1.07       1,099     PP compute    compute 98%, bandwidth 36%, transport 1%

qwen36          1       52.3        52.3     PP bandwidth  bandwidth 93%, launch 5%, compute 3%
qwen36          8       50.9         407     PP bandwidth  bandwidth 93%, compute 24%, launch 5%
qwen36         64       25.0       1,599     PP compute    compute 95%, bandwidth 55%, launch 3%
qwen36        256       6.41       1,641     PP compute    compute 98%, bandwidth 22%, transport 2%
qwen36       1024       1.61       1,652     PP compute    compute 98%, bandwidth 14%, transport 2%

dsv4_pro        1       45.8        45.8     PP bandwidth  bandwidth 88%, launch 11%, compute 7%
dsv4_pro        8       20.1         161     PP bandwidth  bandwidth 94%, compute 26%, launch 5%
dsv4_pro       64       5.23         335     PP bandwidth  bandwidth 98%, compute 54%, launch 1%
dsv4_pro      256       2.37         607     PP compute    compute 99%, bandwidth 67%, transport 1%
dsv4_pro     1024       0.60         610     PP compute    compute 99%, bandwidth 18%, transport 1%

dsv4_flash      1        142         142     PP bandwidth  bandwidth 72%, launch 23%, compute 6%
dsv4_flash      8       69.7         558     PP bandwidth  bandwidth 86%, compute 22%, launch 11%
dsv4_flash     64       22.4       1,434     PP bandwidth  bandwidth 94%, compute 56%, launch 4%
dsv4_flash    256       9.59       2,455     PP compute    compute 96%, bandwidth 52%, transport 2%
dsv4_flash   1024       2.43       2,491     PP compute    compute 98%, bandwidth 14%, transport 2%

mimo25          1        141         141     PP bandwidth  bandwidth 76%, launch 19%, transport 5%
mimo25          8       36.2         289     PP bandwidth  bandwidth 94%, compute 10%, launch 5%
mimo25         64       10.2         653     PP bandwidth  bandwidth 98%, compute 22%, launch 1%
mimo25        256       8.47       2,167     PP bandwidth  bandwidth 97%, compute 74%, transport 2%
mimo25       1024       2.78       2,845     PP compute    compute 97%, bandwidth 37%, transport 2%
```

## Decode — 16-node dual switch (3,494 GB/s @ eta 0.80), eager, 2K ctx

```text
model           B  tok/s/seq   agg tok/s recipe binding    top-3 bottlenecks (share)
k3              1       23.1        23.1     PP bandwidth  bandwidth 90%, launch 9%, compute 5%
k3              8       10.7        85.5     PP bandwidth  bandwidth 95%, compute 17%, launch 4%
k3             64       2.94         188     PP bandwidth  bandwidth 97%, compute 37%, transport 2%
k3            256       1.86         477     PP bandwidth  bandwidth 96%, compute 95%, transport 4%
k3           1024       0.47         483     PP compute    compute 96%, bandwidth 35%, transport 4%

glm52           1       59.3        59.3     PP bandwidth  bandwidth 89%, launch 9%, compute 4%
glm52           8       17.6         141     PP bandwidth  bandwidth 97%, compute 10%, launch 3%
glm52          64       5.16         330     PP bandwidth  bandwidth 99%, compute 24%, launch 1%
glm52         256       4.29       1,098     PP bandwidth  bandwidth 98%, compute 80%, transport 1%
glm52        1024       1.33       1,357     PP compute    compute 99%, bandwidth 36%, transport 1%

qwen36          1       63.8        63.8     PP bandwidth  bandwidth 92%, launch 7%, compute 3%
qwen36          8       62.1         497     PP bandwidth  bandwidth 92%, compute 24%, launch 6%
qwen36         64       30.7       1,967     PP compute    compute 95%, bandwidth 55%, launch 3%
qwen36        256       7.91       2,025     PP compute    compute 98%, bandwidth 22%, transport 1%
qwen36       1024       1.99       2,041     PP compute    compute 98%, bandwidth 14%, transport 1%

dsv4_pro        1       55.2        55.2     PP bandwidth  bandwidth 86%, launch 13%, compute 7%
dsv4_pro        8       24.5         196     PP bandwidth  bandwidth 94%, compute 26%, launch 6%
dsv4_pro       64       6.43         411     PP bandwidth  bandwidth 98%, compute 54%, launch 1%
dsv4_pro      256       2.92         748     PP compute    compute 99%, bandwidth 67%, transport 1%
dsv4_pro     1024       0.73         752     PP compute    compute 99%, bandwidth 18%, transport 1%

dsv4_flash      1        168         168     PP bandwidth  bandwidth 69%, launch 28%, compute 5%
dsv4_flash      8       84.1         673     PP bandwidth  bandwidth 84%, compute 21%, launch 14%
dsv4_flash     64       27.5       1,757     PP bandwidth  bandwidth 94%, compute 56%, launch 5%
dsv4_flash    256       11.8       3,026     PP compute    compute 96%, bandwidth 52%, launch 2%
dsv4_flash   1024       3.00       3,077     PP compute    compute 98%, bandwidth 14%, transport 2%

mimo25          1        168         168     PP bandwidth  bandwidth 74%, launch 23%, compute 5%
mimo25          8       44.2         353     PP bandwidth  bandwidth 93%, compute 10%, launch 6%
mimo25         64       12.5         803     PP bandwidth  bandwidth 98%, compute 22%, launch 2%
mimo25        256       10.4       2,672     PP bandwidth  bandwidth 97%, compute 74%, transport 2%
mimo25       1024       3.43       3,517     PP compute    compute 98%, bandwidth 37%, transport 2%
```

(Caveat on PP16: 16 stages exceed SPARK_STAGE_PLAN_MAX_STAGE_COUNT (13) —
the engine constant must be lifted before the 16-node ring loads a PP16
plan; noted in the recipes themselves, e.g.
examples/recipes/mimo25.PP16.*.json. The dual16 PP numbers above assume
that lift.)

## Decode at 128K context (KV term scaled; eager)

```text
model       topo          B1         B8        B64       B256       B1024   (tok/s/seq, ring13)
k3          ring13       18.6       8.03       2.00       1.01       0.38
glm52       ring13       48.8       14.3       4.20       3.48       1.07
qwen36      ring13       45.2       23.0       4.66       1.25       0.32
dsv4_pro    ring13       44.9       18.9       4.61       2.37       0.60
dsv4_flash  ring13        137       60.4       16.0       7.78       2.43
mimo25      ring13        127       29.3       6.66       3.06       1.03
```

GLM's DSA cap (2,048 selected slots) makes its 128K row identical to 2K —
the sparse attention paying for itself. Pro's compressed KV + selection
past 128K slots holds its row near-2K as well. Qwen collapses hardest
(52.3 → 45.2 at B1, and 4.7x worse at B8): its 16 full-attention layers
re-read 65.5 KB/token of KV with no pruning. `--context` re-prices any
length.

## (a) Does anything prevent 80% MBU at B1? Yes — launch overhead; then the hop floor

Effective MBU = ideal step time at 100% of raw aggregate / modelled step:

```text
model       ring13 eager   ring13 graph   dual16 eager   dual16 graph
k3            0.733          0.792          0.721          0.793
glm52         0.725          0.784          0.716          0.786
qwen36        0.742          0.783          0.736          0.786
dsv4_pro      0.702          0.784          0.687          0.786
dsv4_flash    0.573          0.743          0.549          0.751
mimo25        0.608          0.747          0.588          0.755
```

Three findings:

1. **Eager launches alone cost every model its 80% target at B1** — 6-15
   points of MBU for the mid models, ~20 for Flash/MiMo (whose 5 ms steps
   cannot amortise 820/673 launches × 2 µs = 1.6/1.3 ms). At the top of
   the launch_ns range (5 µs) eager MBU drops to 0.39-0.69: the D10 graph
   work is not optional for any B1 target. This is the roadmap's D1/D10
   finding (roadmap:589-613) quantified per model.
2. **With graph replay the launch term collapses to ~14/17 launches per
   step and every model returns to 0.74-0.79.** K3 lands at 0.792 —
   effectively the target. The remaining gap is NOT launches: it is the
   PP stage-boundary latency, 12 × 29 µs hop floor = 348 µs per token on
   the ring, which is 1-2% of the heavy models' steps but 6-7% of
   Flash/MiMo's. That is exactly the D5/D6 attack surface (bounded ack
   window, pre-advertised RDMA slots — roadmap:654-671). The B1 campaign
   order the numbers imply: graphs first (13-25 points), hop-floor work
   second (the last 1-5).
3. **DSv4 Pro at B1 on the ring: 45.8 tok/s eager, 51.1 with graphs** —
   the quoted 50 tok/s target sits precisely on the far side of the graph
   work, matching the roadmap's "ACHIEVABLE ONLY AT THE 80% GOAL"
   (roadmap:164-167). Graph-mode B1 for all models on ring13: K3 20.6,
   GLM 52.7, Qwen 55.1, Pro 51.1, Flash 185, MiMo 173 — i.e. the memory
   roofline minus the hop-floor tax.

## (b) Where each model crosses from bandwidth- to compute-bound

```text
model       crossover @6.5 TF wall   @unit peaks        roadmap:423-430 says
k3          B258                     never (<B8192)      B258 / never
glm52       B320                     never               B320 / never
qwen36      B34                      B300                B34  / B294
dsv4_pro    B165                     B8161               B165 / never(ish)
dsv4_flash  B130                     never               B130 / never
mimo25      B341                     never               B341 / never
```

The scan reproduces the roadmap's wall crossovers exactly (the test
asserts this). Read: **B64 is the last batch where every model except
qwen is safely bandwidth-bound even at the measured wall; B256 is past
the wall crossover for Pro, Flash and (marginally) K3** — the WMMA
retile is the B256+ P0, as the roadmap orders it. Qwen is the outlier:
compute-bound from B34 at the wall and from ~B300 even at impossible-
perfect BF16 efficiency; its 80%-of-bandwidth target at batch is only
coherent after FP8-FFN quantization (roadmap:350-361). If FP8/FP4 expert
GEMMs hold even ~40% of peak (PENDING, calib:158-159), Pro/Flash/MiMo's
real crossovers move past B1024 and their B1024 rows in the decode table
are pessimistic by up to the compute/bandwidth ratio shown.

## (c) Fabric share per topology — the dual switch is never the binding constraint

Share of step time, graph mode (TP collective | PP transport):

```text
model       ring13 B1      ring13 B64     ring13 B1024    dual16 B1024
k3          11.9% | 1.0%    7.4% | 2.0%    15.5% | 4.8%    10.4% | 3.8%
glm52       25.4% | 1.9%    9.6% | 0.5%    31.7% | 1.3%    21.2% | 1.0%
qwen36      21.5% | 2.0%   41.3% | 2.5%    32.6% | 1.7%    21.8% | 1.3%
dsv4_pro    19.4% | 1.8%   10.6% | 0.7%    16.0% | 0.9%    10.7% | 0.7%
dsv4_flash  48.0% | 6.6%   21.3% | 2.0%    26.6% | 2.1%    17.8% | 1.6%
mimo25      50.2% | 6.2%   10.6% | 0.9%    33.9% | 2.3%    22.7% | 1.8%
```

- **PP transport is 0.4-6.6% everywhere** — never close to binding. K3's
  126 KiB AttnRes stage payload (config.h:205-213) costs it the largest
  transport share at B1024 (4.8% ring) and it still doesn't bind.
- **TP collective is the second-largest additive term at B1 for the light
  models** (48-50% on the ring for Flash/MiMo — 2 × layers × 29 µs of
  latency floor against a ~5 ms step) which is precisely why the recipe
  picker puts every model on PP at B1. At B64-B1024 the collective's wire
  term grows with B: qwen's B×hidden payload makes TP worst at B64
  (41.3% of step on the ring) — the TP16 doc's "dense qwen is deeply
  comm-bound at batch" (TP16_DUAL_RAIL_SPEED.md:26-28), reproduced.
- **Is the dual-switch fabric ever binding at B1024? No.** Worst fabric
  share on dual16 at B1024 is 22.7% (mimo25 TP); since the model takes
  max(bandwidth, compute) + overlay, a fabric term would have to exceed
  the whole memory/compute term to bind, and it is 3-10x short of that
  for every model. What the dual rail buys is halving the largest
  additive tax (ring→dual at B1024: glm52 TP 31.7%→21.2%, mimo25
  33.9%→22.7%), not removing a bound. The binding constraints at B1024
  remain compute (every model, at the wall) and K3's fp32 state.

## (d) Prefill — and whether single-node 128K is viable

tok/s per sequence, MFU 0.45 (±0.10 moves every number ~±20%), eager
launches, PP overlay. Chunked at 256 tokens; each chunk re-streams the
weights, which is why the bandwidth-bound models are flat in context —
the per-chunk stream is context-independent and the attention quadratic
only overtakes it for the widest-attention models.

```text
model       ctx     ring13   dual16   1 node    128K single-node time   binding (ring13 @128K)
k3          2K        442      550       36                             bandwidth
k3          32K       442      550       36                             bandwidth
k3          128K      442      550       36      3,618 s (1.0 h)        bandwidth
glm52       2K        945    1,165       74                             bandwidth
glm52       32K       929    1,145       73                             bandwidth
glm52       128K      420      517       33      4,026 s (1.1 h)        compute (quadratic)
qwen36      2K      3,411    4,222      276                             compute
qwen36      32K     3,228    3,994      261                             compute
qwen36      128K    2,754    3,405      221        593 s (10 min)       compute
dsv4_pro    2K        887    1,093       70                             bandwidth
dsv4_pro    32K       839    1,034       66                             bandwidth
dsv4_pro    128K      323      398       25      5,232 s (1.5 h)        compute (quadratic)
dsv4_flash  2K      4,521    5,580      373                             bandwidth
dsv4_flash  32K     3,169    3,908      256                             compute
dsv4_flash  128K      982    1,210       77      1,709 s (28 min)       compute
mimo25      2K      2,278    2,809      181                             bandwidth
mimo25      32K     2,278    2,809      181                             bandwidth
mimo25      128K    2,278    2,809      181        724 s (12 min)       bandwidth
```

**Single-node 128K prefill is not viable for K3, GLM, or Pro** (1-1.5
hours per sequence — below any interactive or batch-SLO reading), and is
merely tolerable for Flash (28 min), MiMo (12 min) and Qwen (10 min).
Prefill must be ring-distributed for the big three; at ring scale a 128K
prompt is 4-6 minutes for GLM/Pro, 5 for K3. The mechanism differs by
model: K3 is pinned by the per-chunk expert-pool re-stream (1.4 TB pool
× 512 chunks — chunked prefill of a giant MoE is a weight-streaming
problem, and the fix is bigger chunks or persistent experts, not more
FLOPs), while GLM/Pro cross to compute-bound at 128K on the attention
quadratic (their MLA width × ctx² term reaches 49/58 PFLOP). Qwen's
advantage is real but narrow: dense 27B has no pool to re-stream and a
small 16-layer quadratic.

**Chunked-prefill × decode interaction.** Each prefill chunk pays the
same launch overlay as a decode step (515-1,974 launches eager), so an
eager 128K prefill carries 512 × 1-4 ms of pure launch overhead — 0.5-2 s
per prompt, second-order against the times above but first-order for
2K-8K chat prompts (where it is 4-16 ms against a 2-10 ms prefill). The
decode side is the roadmap's standing item: multi-chunk decode is
fail-closed today (AUDIT F1), so a prefill chunk interleaves as a full
step on every stage; the planner treats chunk and decode steps as the
same shape class for graph capture. Nothing in the numbers changes the
F1 priority — it says the interleave tax is launches, not bandwidth.

## Occupancy sanity check (sm_121a ptxas, production flags)

SM assumptions: 65,536 registers/SM and 2,048 threads/SM (ASSUMED —
standard NVIDIA values, not GB10-confirmed), 131,072 B smem/SM (NOT
assumed: `inference/kernels/layout.cuh:21`), 256-thread blocks
(`runtime/launch.h:201`). Kernel data: the build server's
`logs/inference__llms__*__unity.ptxas.txt`; GEMM dynamic smem recomputed
from `LmGemmSharedBytes` (inference/kernels/gemm.cuh:40-47).

```text
k3    GEMM Bf16→Mxfp4 M64   109 regs  25,104 B smem -> 2 blk/SM (25%, regs-bound)
k3    DeltaRule<256,128,128> 94 regs  67,200 B smem -> 1 blk/SM (12%, smem-bound)
k3    GEMM Bf16 M64          87 regs  49,168 B smem -> 2 blk/SM (25%, regs-bound)
glm52 GEMM Bf16→Fp8 M64     107 regs  32,784 B smem -> 2 blk/SM (25%, regs-bound)
glm52 GEMM Bf16→Fp8 M32      72 regs  24,592 B smem -> 3 blk/SM (38%, regs-bound)
qwen36 GEMM Bf16 M32         64 regs  40,976 B smem -> 3 blk/SM (38%, smem-bound)
dsv4* GEMM Fp8 M64 K128     123 regs  49,168 B smem -> 2 blk/SM (25%, regs-bound)
dsv4* GEMM Fp8→Mxfp4 M32     83 regs  25,616 B smem -> 3 blk/SM (38%, regs-bound)
mimo25 GEMM Int7 M64 K256    95 regs  86,032 B smem -> 1 blk/SM (12%, smem-bound)
```

Three readings:

- **No kernel fails to launch on occupancy grounds**: every top-3 kernel
  gets at least one 256-thread block per SM, zero spills reported by
  ptxas for all of them. The "full SM occupancy" question answers itself
  in the negative for the M64 GEMMs — 109-123 registers means 2
  blocks/SM = 25% thread occupancy — but these are bandwidth-bound
  decode GEMMs; 2 blocks × 2-stage TMA pipelines is plausibly enough to
  saturate LPDDR (PLAUSIBLE, not measured — this is exactly the
  calibration doc's per-kernel-class eta_bw PENDING, calib:148-154).
- **The delta rule is smem-bound at exactly 1 block/SM**: 94 regs would
  allow 2, but 64 KiB dynamic + 1,664 B static = 67,200 B and two of
  those (134,400 B) exceed the 131,072 B SM budget. That is by design —
  the kernel is persistent within a launch (linear_attn.cuh:268-276) and
  wants one block per SM — but it depends on the >48 KiB dynamic-smem
  opt-in succeeding on sm_121 (kimi_k3/layer.cuh:405-412). Whether GB10
  grants ≥67,200 B per block is UNVERIFIED and is a hardware-week check,
  not an estimate: if the opt-in ceiling is lower the KDA/GDN path does
  not launch at all.
- **The M32 tile variants recover 3 blocks/SM (38%)** — the B1-B8 regime
  uses them (tile_m is selected per token bucket, gemm.cuh:30-33), so the
  small-batch launches are also the higher-occupancy ones. Convenient,
  not accidental.

## Standing uncertainties

- launch_ns (2-5 µs) moves every eager B1 number ±(launches × 3 µs):
  K3 ±30% of its launch term, Flash/MiMo ±20 points of eager MBU. The
  graph-mode conclusions are insensitive to it (14-17 launches either
  way). PENDING per the calibration doc.
- The K3 launch count: 1,974 static LM_LAUNCH vs the roadmap's 3,276
  hand count (D1 counts logical kernels the wrappers may fuse or loop).
  Both carried; `--launches-multiplier 1.66` reprices K3 at the roadmap
  figure (eager MBU 0.733 → 0.698 — the finding's direction does not
  change, its magnitude does).
- The wall applied to FP8/FP4 expert GEMMs is pessimism (roadmap:862-864);
  Pro/Flash/MiMo B256+ rows move up if expert GEMMs beat 6.5 TFLOPS/node.
- Prefill MFU 0.35-0.55 is asserted, not measured; prefill tok/s scales
  ~linearly with it inside the compute-bound cells.
- The ring TP collective assumes hop-pipelined ring AR at one 29 µs
  floor per AR; a store-and-forward ring AR would multiply the latency
  term by ~2(N-1) and make TP13 untenable at B1 — one more reason the
  recipe picker says PP. Measure before quoting TP13 B1 numbers.
- GLM's prefill quadratic assumes full attention during prefill (DSA
  selection is modelled as decode-time only). If DSA prunes prefill too,
  GLM's 128K prefill row improves toward the flat 945/1,165 band.
- Occupancy SM constants (65,536 regs, 2,048 threads) are assumptions;
  the smem figures and register counts are ptxas fact. The delta rule's
  1-block/SM conclusion holds under any plausible reg/thread constant
  because smem binds first.
