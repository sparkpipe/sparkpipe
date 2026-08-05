# Performance roadmap, 2026-08-01 — roofline model and the attack list

Pre-hardware. Three days before ring access. This document builds the
bandwidth roofline the quoted targets are measured against, prices every
statically identifiable defect between the current code and that roofline,
and orders the fixes. Every code claim carries a file:line. Every number
derived here is reproducible from `inference/llms/*/config.h`,
`model_contracts/*.json`, and the arithmetic inline; nothing is fitted.

The quoted targets, B1 decode on the 13-spark ring:

```text
DeepSeek V4 (Pro class, 4-bit experts)   ~50 tok/s
GLM 5.2 (FP8 experts, BF16 rest)         ~25 tok/s
Kimi K3 (MXFP4 experts, BF16 rest)       ~12 tok/s
2TB+ models generally                    ~12 tok/s
```

The stated engineering goal is 80%+ of the bandwidth roofline. eta_bw ~0.80
is also this repo's own calibrated memory-path efficiency
(docs/GB10_CUDA_COST_MODEL_CALIBRATION.md:87-97, three independent
derivations), so "target = 80% of roofline" and "target = roofline at the
calibrated eta" are the same statement. That is not a coincidence to lean
on; it is a coincidence to verify on hardware.

## The machine, as this repo encodes it

```text
per node:   273 GB/s LPDDR5X unified, 128 GB capacity, 48 SM sm_121a,
            ~31 dense BF16 TFLOPS, ~250 FP8 TFLOPS, 2x 100 GbE
            (README.md:50-56, GB10_CUDA_COST_MODEL_CALIBRATION.md:16-26)
13-ring:    3.549 TB/s aggregate local bus; 1.66 TB aggregate memory
effective:  eta_bw 0.80 -> 218 GB/s per node, 2.84 TB/s aggregate
sustained:  the topology study's planning floor is MBU 0.55, net 0.8
            (docs/BANDWIDTH_LEDGER.md:119)
hop floor:  29 us/hop software floor, measured
            (docs/AUDIT_2026-07-29_BANDWIDTH_PATH.md:65,
             docs/HANDOFF_DECODE_PERFORMANCE.md:236)
balance:    ~915 FLOP/byte — every decode kernel here is memory-bound
            (GB10_CUDA_COST_MODEL_CALIBRATION.md:24-26)
```

The roofline law is the ledger's own: tokens/s = streamed bytes/s divided
by bytes-per-token (docs/BANDWIDTH_LEDGER.md:3-8). At B1 there is no
amortisation; the whole active-weight stream is paid every token. At B>1
the fixed stream amortises 1/B but the routed-expert stream anti-amortises
along the coverage S-curve — the per-batch section below prices both.

## Per-model bytes/token at B1, and the implied ceiling

TP1 view (the whole model read once per token); under TP each rank streams
its shard, so the ring aggregate applies when the cohort pipeline keeps all
13 buses saturated — the condition audit F1-F3 exist to protect
(docs/AUDIT_2026-07-29_BANDWIDTH_PATH.md:38-78).

### Kimi K3 — contract recipe (BF16 non-expert, MXFP4 routed experts)

Geometry: `inference/llms/kimi_k3/config.h:76-96` (hidden 7168, 93 layers =
69 KDA + 24 MLA, vocab 163840, 896 experts top-16 + 2 shared, latent 3584,
expert intermediate 3072); precision:
`model_contracts/k3_authoritative.json:74-85`.

```text
KDA attention      30.61 B params x 69 layers x 2 B   61.2 GB
MLA attention       5.57 B params x 24 layers x 2 B   11.1 GB
shared experts + latent projections + router
                   17.5  B params            x 2 B   35.0 GB
lm head (163840 x 7168 x 2 B, per STEP)               2.35 GB
routed experts     16 x 92 layers x 33.0 M x 4.25/8   25.8 GB
                   (17.55 MB/expert MXFP4+E8M0)
KDA state R+W      69 layers x 6.59 MB x 2  (below)   0.91 GB
MLA KV read        24 x 1152 B x ctx  (2K ctx)        0.06 GB
                                                   ---------
total                                             ~136.4 GB/token
```

Split cross-checked against `tools/k3_param_budget.py` (104.62 B active
params enumerated vs 104.2 B published, 0.4%). Note the budget script's own
headline — 72.1 GB/token — assumes FP8 attention (`k3_param_budget.py:61`,
`# fp8`), which the authoritative contract does not grant: non-expert
weights are BF16 (`k3_authoritative.json:79`). Both recipes are carried
below; the contract one is the honest ceiling.

```text
13-ring ceiling, BF16 recipe (136.4 GB):  26.0 tok/s @100%  20.8 @80%  14.3 @55%
13-ring ceiling, FP8-attention (73.0 GB): 48.6 tok/s @100%  38.9 @80%  26.8 @55%
single node, BF16 recipe:                  2.0 tok/s @100%
```

**Target 12 tok/s: ACHIEVABLE with margin.** It needs 46% of the 13-ring
aggregate at the contract's BF16 recipe, 25% at the FP8-attention recipe.
The target is not the risk; the MBU is.

### GLM 5.2 — contract recipe (FP8 E4M3 experts, BF16 rest)

Geometry: `model_contracts/glm52.json` (78 layers, hidden 6144, first
routed layer 3, 256 experts top-8, expert intermediate 2048, dense 12288,
heads 64, latent 512+64, vocab 154880).

```text
MLA attention      ~165 M params/layer x 78 x 2 B     25.7 GB
routed experts     8 x 75 layers x 37.75 MB FP8       22.6 GB
dense FFN, layers 0-2  3 x 226.5 M x 2 B               1.4 GB
lm head (154880 x 6144 x 2 B, per STEP)                1.9 GB
router + DSA index weights                            ~1.2 GB
KV read            78 x 1152 B x min(ctx, 2048 sel.)   0.2 GB
                                                   ---------
total                                              ~53.0 GB/token
```

Consistent with the 12-node model's "~70 GB BF16-heavy / ~25-35 GB
FP8/NVFP4" bands (docs/GLM52_12X_SPARK_PERFORMANCE_MODEL.md:61-66) once
their NVFP4-expert assumption is swapped for the contract's FP8.

```text
13-ring ceiling:   67.0 tok/s @100%  53.6 @80%  36.9 @55%
single node:        5.2 tok/s @100%   4.1 @80%
```

**Target 25 tok/s: ACHIEVABLE.** 37% of the 13-ring aggregate. Even a
6-node partition at the calibrated eta reaches it.

### DeepSeek V4 Flash — contract recipe (FP4 experts, FP8 non-expert)

Geometry: `model_contracts/dsv4_flash_authoritative.json` (43 layers,
hidden 4096, 64 heads at head_dim 512, 256 experts top-6 + 1 shared,
expert intermediate 2048, vocab 129280).

```text
attention          ~195 M params/layer x 43 x 1 B FP8  8.4 GB
routed+shared experts  7 x 43 x 12.6 MB FP4            3.8 GB
lm head (129280 x 4096 x 2 B BF16)                     1.1 GB
indexer + misc                                         ~1 GB
                                                   ---------
total                                              ~14.7 GB/token
```

```text
13-ring ceiling:   241 tok/s @100%  193 @80%  133 @55%
single node:       18.6 tok/s @100%  14.9 @80%
```

### DeepSeek V4 Pro — the model the 50 tok/s quote actually fits

Geometry: `model_contracts/dsv4_pro_authoritative.json` (61 layers, hidden
7168, 128 heads at head_dim 512, 384 experts top-6 + 1 shared, expert
intermediate 3072 — a ~1.6T model).

```text
attention          ~632 M params/layer x 61 x 1 B FP8 38.5 GB
routed+shared experts  7 x 61 x 33.0 MB FP4          14.1 GB
lm head (129280 x 7168 x 2 B BF16)                    1.9 GB
                                                   ---------
total                                              ~54.5 GB/token
```

```text
13-ring ceiling:    65 tok/s @100%  52 @80%  36 @55%
single node:         5.0 tok/s @100%
```

**Target 50 tok/s: ACHIEVABLE ONLY AT THE 80% GOAL.** It needs 77% of the
13-ring aggregate — there is no slack for a second defect class once MBU
slips. Pro is the tightest target on the board, and it is why the launch
and host-overhead items below are P0 rather than polish: at 50 tok/s the
per-token budget is 20 ms across the whole ring, and the current launch
tax alone is 15-25% of it.

## The K3 state correction — the ledger's 192x, and the corrected ceiling

`docs/BANDWIDTH_LEDGER.md:23-24` prices KDA state at "32 KB R + 32 KB W per
sequence per KDA layer (69 layers: 4.4 MB R+W per seq)". That figure is one
128x128 bf16 plane — one head, wrong precision. The code's own constants
say otherwise:

```text
K3_KDA_STATE_SLOT_BYTES = 96 heads x 128 x 128 x 4 B fp32 = 6,291,456 B
    (inference/llms/kimi_k3/config.h:303-304, element bytes :293)
K3_KDA_CONV_WINDOW_BYTES = 294,912 B   (config.h:294-296)
per layer per sequence: 6.59 MB  (config.h:305)
```

6,291,456 / 32,768 = **exactly 192x**. The ledger undercounts the recurrent
state stream by 192: the true figure is 69 layers x 6.59 MB x 2 (read +
write) = **909 MB per sequence per token**, not 4.4 MB. K3_SPEED.md:39-41
already carries the corrected order (895 MB, slot only) — the ledger entry
is the stale one, and its "the recurrence's floor, already minimal" verdict
survives only because the code, not the ledger, allocates the state.

What the correction does and does not change:

- **B1 ceiling: unchanged for practical purposes.** +0.91 GB on a 136.4 GB
  stream is 0.7%. The 26.0 tok/s BF16-recipe ceiling stands; **12 tok/s K3
  remains achievable — the math above is the corrected math.**
- **Batch ceiling: this is where it bites.** State scales with B while
  weights amortise: B16 = 14.5 GB, B64 = 58.2 GB, B256 = 232.7 GB of state
  R+W per step (against a weight pass that shrinks per-token toward the
  full-pool sweep). CORRECTED 2026-08-01: an earlier revision of this
  bullet called the state ~40% of the B64 step; that priced the step as
  fixed-weights + state and ignored expert coverage growth. The per-batch
  section below puts the B64 step at 1,157.9 GB and the state at 5% of it
  — first-order only from B256 (13%), dominance-adjacent only at B1024
  (36%). The bf16-state option the README prices at admission
  (README.md:81-84) is the lever, and it is a numerics question
  (K3_SPEED.md:56-60), not a systems one.
- **Prefill and replay:** the fold path re-streams the same slots; the
  correction quadruples nothing there because the fold was already priced
  from the code constants, not the ledger.

## The "100 tok/s DSv4 with 4-bit experts" claim — sanity check

Plainly: **no single GB10 node can do this, and the claim is only coherent
as a cluster or concurrency number.**

- Single-node ceiling for DSv4 Flash at B1 is 18.6 tok/s at 100% MBU (its
  14.7 GB/token against 273 GB/s). Public single-Spark reports for this
  exact model class land at ~12-15 tok/s quantized — i.e. 65-80% MBU,
  which cross-validates this repo's eta rather than contradicting it.
- The public ~70 tok/s figures for DSv4 Flash on 2x DGX Spark are
  client-side decode *concurrency* (200 output tokens per request, batched)
  — aggregate throughput under batch, where weight bytes amortise across
  the cohort. Batched aggregate is not B1 single-stream.
- At B1 single-stream, 100 tok/s of Flash requires 1.47 TB/s of weight
  stream: **>= 6-7 GB10 nodes at the calibrated 80% eta.** On THIS 13-ring
  the Flash ceiling is ~190-240 tok/s, so 100 tok/s B1 is reachable here at
  ~42-53% MBU — but that is a statement about a 13-node aggregate, not
  about anyone's single node.
- For DSv4 Pro (54.5 GB/token), 100 tok/s needs 5.45 TB/s — **>= 24 nodes
  at 80% eta. Not achievable on this ring at any MBU** (ceiling 65). If the
  quoted rig ran Pro-class weights at 100 tok/s it was either batched
  concurrency or a 2-dozen-node fabric.

The user's own target (~50 tok/s B1) is the Pro-class 80%-eta ceiling, and
is the right kind of quote: a ring-aggregate single-stream number.

## Per-batch roofline — B1, B8, B64, B256, B1024 (2026-08-01 extension)

The targets are now 80% of roofline at every batch, not only B1. This
section prices the decode step at each batch for all six models. Method,
so every figure is reproducible:

- Step bytes = fixed stream (attention + dense FFN + shared experts + head,
  read once per step) + expert stream (coverage(B) x full routed pool) +
  B x (recurrent state R+W + KV read + residual-bank reads). Reference
  context 2K tokens; KV terms scale linearly with context and the formula
  is given so any context can be re-derived.
- Expert coverage under uniform routing: cov(B) = 1 - (1 - 1/E)^(B*k).
  Uniform routing OVERESTIMATES coverage against measured skew: glm52
  measured 87% at B128 and 98% at B256 (GB10_CUDA_COST_MODEL_CALIBRATION.md
  :62-64) where the uniform law gives 98.2% and 99.97%. Skewed routing
  streams FEWER distinct experts, so the expert-stream figures at B8-B128
  are upper bounds; skew also concentrates rows per hot expert, which helps
  the grouped GEMM. Real route histograms are standing measurement #9
  below.
- Rates: 3.549 TB/s ring aggregate at 100%, 2.84 TB/s at the 80% target
  (the "The machine" block above). tok/s/seq = 2840 / step bytes; aggregate
  = B x that.

### The anti-amortisation law, and the dense-equivalent crossover

The B1 tables above amortise nothing. The naive expectation for B>1 is
that weight bytes amortise 1/B across rows. That is true ONLY for the
fixed stream. The routed-expert stream does the opposite until the pool
saturates: each added token routes to k fresh draws, so the stream GROWS
with coverage — K3 streams 25.8 GB of experts at B1 and 192.7 GB at B8
for the same 8 tokens. MoE batching is an S-curve: per-token expert bytes
fall only after coverage saturates. The dense-equivalent crossover — the
batch where every expert is hot and the stream is the full pool sweep —
at 99% coverage (draws = 4.6 x E):

```text
model      E    k   B(99% cov)   full-pool expert stream per step
K3         896  16   ~B258       896 x 17.55 MB x 92 L   = 1,446.7 GB
GLM 5.2    256   8   ~B147       256 x 37.75 MB x 75 L   =   724.8 GB
DSv4 Pro   384   6   ~B295       384 x 33.0  MB x 61 L   =   773.2 GB
DSv4 Flash 256   6   ~B196       256 x 12.6  MB x 43 L   =   138.7 GB
MiMo 2.5   256   8   ~B147       256 x 25.17 MB x 47 L   =   302.8 GB
```

At B1024 every MoE model here is past crossover: rows per hot expert are
K3 18, GLM 32, Pro 16, Flash 24, MiMo 32 — decode MoE is batched dense
GEMM territory, and the expert problem is a scheduling problem, not a
sparsity problem. B64 is the worst regime per byte: coverage 63-87%
(stream most of the pool) at ~1 row per hot expert (amortise nothing).
This is exactly the regime the expert-queue amortization exists to rescue
(docs/BATCHPLANE_MODEL_RECONCILIATION.md:10-25), and its routed-queue
depth is the ring measurement that proves or kills it.

### K3 (BF16 non-expert, MXFP4 experts; 69 KDA + 24 MLA)

Fixed stream 109.65 GB (the B1 table above). Pool 1,446.7 GB. Per-seq per
step: KDA state R+W 0.909 GB (69 x 6.59 MB x 2, config.h:303-305), MLA KV
0.0566 GB at 2K (24 x 1152 B x ctx, kv_lora 512 + unrotated 64 at 16 bits,
k3_authoritative.json mla/cache), AttnRes bank reads ~0.013 GB (2 sites x
93 layers x ~5 of 9 candidate reps x 14,336 B, config.h:178-205; the bank
is per-in-flight-row activation state, not a context-growing cache — its
capacity cost is the 126 KB stage payload per row, config.h:208-213).

```text
B       fixed   experts  state    KV+AR   step GB   tok/s/seq  agg tok/s
1       109.7     25.8     0.9      0.07     136.4     20.8        20.8
8       109.7    192.7     7.3      0.6      310.2      9.2        73.3
64      109.7    985.6    58.2      4.5    1,157.9     2.45       157.0
256     109.7  1,431.8   232.7     17.9    1,792.0     1.58       405.7
1024    109.7  1,446.7   930.8     71.6    2,558.7     1.11     1,136.6
```

Dominant term: B1 fixed weights (80%); B8-B256 the expert coverage stream
(62-85%); B1024 the fp32 state is 36% of step bytes — the only batch where
the state term is first-order. Launch tax (D1, 6.6-16.5 ms/step) is 12-34%
of the B1 step, ~1% of the B1024 step: graphs are a B1-B8 fix.

### GLM 5.2 (FP8 experts, BF16 rest; no recurrent state)

Fixed 30.2 GB. Pool 724.8 GB. KV 0.184 GB/seq/step at 2K (78 x 1152 B x
min(ctx, 2048 selected), glm52.json dsa_selected_token_count). The DSA
index cache (32 heads x 128 dim) is a second per-slot read not priced
here — second-order at 2K, and PENDING a geometry confirmation.

```text
B       fixed   experts   KV      step GB   tok/s/seq  agg tok/s
1        30.2     22.6     0.2       53.0     53.6        53.6
8        30.2    160.3     1.5      192.3     14.8       118.2
64       30.2    626.7    11.8      669.1      4.24      271.7
256      30.2    724.5    47.1      801.9      3.54      906.7
1024     30.2    724.8   188.4      943.4      3.01    3,082.5
```

No state term; the expert stream dominates B8-B1024 (78-83%). At B1024 KV
is 20% of step bytes.

### Qwen 3.6 (dense 27B, all-BF16 — bind.cu:133 LmBf16Format; 48 GDN + 16 full-attention layers)

Fixed 50.2 GB — the whole dense weight read every step, no expert sparsity
(16 attn layers x 0.341 B + 48 GDN layers x 0.383 B params at 2 B, plus
the 2.54 GB head; geometry config.h:31-67). GDN state is ALREADY bf16:
0.54 MB/seq/layer (config.h:65-67, the *2u element), 0.052 GB/seq/step
R+W across 48 layers. KV 0.131 GB/seq/step at 2K (16 layers x 4 KV heads
x 256 dim x K+V x 2 B, config.h:47-49).

```text
B       fixed   state    KV      step GB   tok/s/seq  agg tok/s   compute roof
1        50.2     0.05     0.1       50.4     56.4        56.4
8        50.2     0.4      1.0       51.7     55.0       439.6
64       50.2     3.3      8.6       62.1     45.7     2,926.8
256      50.2    13.3     34.4       97.8     29.0     7,431.1    8,028 @peak BF16
1024     50.2    53.1   137.4      240.8     11.8    12,078.1    8,028 @peak / 1,683 @6.5 TF
```

**Qwen is the model the 80%-of-bandwidth target cannot be promised to at
B1024.** 50.2 GFLOP/token against 50.2 GB fixed is 1,049 FLOP/byte on the
weight stream; GB10's BF16 balance point is ~114 FLOP/byte (31 TFLOPS /
273 GB/s — the calibration doc's 915 is the FP8 balance,
GB10_CUDA_COST_MODEL_CALIBRATION.md:21-26). Decode turns compute-bound at
~B294 even at 100% of the BF16 peak, and at ~B34 at the measured 6.5 TFLOPS
QKVO wall. At B1024 the honest ceiling is 8,028 tok/s aggregate at
impossibly-perfect BF16 efficiency — 66% of the 80%-MBU target — and 1,683
at the measured wall. The lever is the one the calibration doc already
names: dense-FFN quantization (GB10_CUDA_COST_MODEL_CALIBRATION.md:125-127).
FP8 FFN halves the dominant byte term AND moves 50 GFLOP/token onto the
250 TFLOPS unit, where 12,078 tok/s needs only 19% of FP8 peak.

### DeepSeek V4 Pro (FP4 experts, FP8 non-expert, compressed KV)

Fixed 40.4 GB. Pool 773.2 GB. KV: the compression ladder
(dsv4_pro_authoritative.json compression_ratios — 62 entries for 61 layers,
30 at 4:1, 31 at 128:1, trailing 0 unverified) stores ~8.9 KB/token, not
61 x 1152 B; the step read at 2K with top-k 1024 + window 128 selection is
0.018 GB/seq (selection binds only past ~128K compressed slots).

```text
B       fixed   experts   KV      step GB   tok/s/seq  agg tok/s
1        40.4     14.1    0.02       54.5     52.1        52.1
8        40.4     91.2     0.1      131.5     21.6       172.8
64       40.4    489.5     1.2      530.6      5.35      342.6
256      40.4    759.1     4.7      804.0      3.53      904.3
1024     40.4    773.2    18.7      832.1      3.41    3,495.0
```

Per-seq tok/s holds 3.4-3.5 from B256 to B1024 — the flattest batch curve
on the board, because the pool is small (773 GB) and KV is compressed
away. The 50 tok/s B1 target analysis above stands unchanged.

### DeepSeek V4 Flash (FP4 experts, FP8 non-expert)

Fixed 10.5 GB. Pool 138.7 GB. KV ~6.1 KB/token stored (2 sliding-window
layers bounded at 128 slots, 21 layers at 4:1, 20 at 128:1 — 44 entries
for 43 layers, UNVERIFIED in deepseek_v4/config.h); step read 0.012 GB/seq
at 2K. B1 reproduces the 14.7 GB above (the ~1 GB "indexer + misc" lump
carries this KV read).

```text
B       fixed   experts   KV      step GB   tok/s/seq  agg tok/s
1        10.5      3.8    0.01       14.3    198.4       198.4
8        10.5     23.7     0.1        34.4     82.7       661.3
64       10.5    107.8     0.8       119.1     23.8     1,525.6
256      10.5    138.4     3.2       152.1     18.7     4,781.6
1024     10.5    138.7    12.7       162.0     17.5    17,956.1
```

### MiMo 2.5 (FP8 linears — bind.cu:123 LmFp8; per-head KV, no latent)

Fixed 5.9 GB. Pool 302.8 GB (47 routed layers, MIMO25_FIRST_ROUTED_LAYER
is a GUESS, config.h:81). KV: 7 full layers x 2,560 B/slot (config.h:98-99)
plus 41 SWA layers bounded at 128 slots x 5,120 B: 0.064 GB/seq/step at 2K,
of which 26.8 MB is the fixed SWA window re-read.

```text
B       fixed   experts   KV      step GB   tok/s/seq  agg tok/s
1         5.9      9.5    0.06       15.4    184.7       184.7
8         5.9     67.0     0.5        73.5     38.7       309.3
64        5.9    262.0     4.1       271.9     10.4       668.5
256       5.9    302.5    16.3       324.8      8.7     2,238.3
1024      5.9    302.8    65.1       373.7      7.6     7,781.2
```

### The compute wall per batch — where bytes stop being the bound

Per-token FLOPs by precision (2 x active params), the ring aggregate
roofs, and the batch at which compute time overtakes memory time:

```text
model  BF16 GF  FP8 GF  FP4 GF | roof @peak   roof @6.5 TF wall | crossover @wall  @peak
K3      110       0      97    |   3,474              408        |   B258          never (<B2048)
GLM      30.5    45.2     0    |  11,162            1,116        |   B320          never
Qwen     50.2     0       0    |   8,028            1,683        |   B34           B294
Pro       3.7    77      56.4  |  24,067              616        |   B165          never
Flash     1.1    16.8    15.2  |  97,683            2,553        |   B130          never
MiMo      1.3    27.6     0    |  86,067            2,928        |   B341          never
```

Roofs are aggregate tok/s (per-row time is batch-independent: B cancels).
Peak = 13 x (31 BF16 / 250 FP8 / 500 dense FP4) TFLOPS. Wall = every GEMM
class at the measured 6.5 TFLOPS QKVO figure (GB10_CUDA_COST_MODEL_
CALIBRATION.md:66-73) — right for BF16 projections, pessimistic for the
FP8/FP4 expert GEMMs whose efficiency is PENDING (calibration doc:158-159);
if FP8 GEMMs hold even 40% of FP8 peak, Pro and Flash stay memory-bound
past B1024. Three readings:

- **B256 is the last batch where bandwidth alone sets every MoE ceiling.**
  At the wall, Pro and Flash have already crossed (B165, B130); K3 crosses
  at B258. This matches the measured glm regime map — QKVO compute-bound
  from B128 (calibration doc:46-48) — and promotes WMMA retiling from
  "measurement #10" to the B256+ P0.
- **At B1024 the wall caps K3 at 408 tok/s against the 1,137 memory
  roofline** — a 2.8x gap. Closing it needs BF16-class GEMMs at ~9.6
  TFLOPS/node (31% of peak), 1.5x past the measured wall. GLM needs ~8.1.
  These are retile targets, not hopes: 2.6% of peak is a tiling defect,
  not a silicon limit.
- **Qwen at B256+ is compute-bound at ANY achievable BF16 efficiency**;
  its B1024 row in the byte table is unreachable as stated (see its
  subsection). Its 80% target is only coherent after FP8 FFN quantization.

### The bf16-state option, priced per batch (K3 only — Qwen's GDN state is already bf16)

Halving the KDA state term (0.909 -> 0.455 GB/seq/step; the numerics
question is K3_SPEED.md:56-60, priced at admission in README.md:81-84):

```text
B       fp32 step GB   bf16 step GB   agg tok/s gain
1           136.4          135.9        +0.3%    noise — do not touch numerics for B1
64        1,157.9        1,128.8        +2.5%
256       1,792.0        1,675.7        +6.9%    406 -> 434
1024      2,558.7        2,093.3       +22.2%  1,137 -> 1,389 (memory side; the
                                                 compute wall still caps at ~408-690
                                                 until the WMMA retile lands)
```

Verdict: bf16 state is a B256+ lever and a B1024 prerequisite, inert at
B1-B8. Sequence it after the numerics review and before the B1024 push,
not in the B1 campaign.

### KV capacity and the NVMe budget

Per-token KV (whole model, unsharded; each pipeline rank holds only its
own layers' share — the 12-rank GLM shard of 7.5 KB/token is why the
production pool of 65,536 blocks x 64 slots lands at 29 GiB,
docs/BATCHPLANE_MODEL_RECONCILIATION.md:43-47), per-seq fixed state, and
what 1 TB of NVMe per node holds (ring aggregate is 13 x):

```text
model   KV B/token   fixed/seq      seqs @ 8K ctx   seqs @ 128K ctx   seqs @ 1M ctx
K3        27,648     455 MB state      1,467             245                29
GLM       89,856       —               1,359              85                10
Qwen      65,536      26 MB state      1,777             116                14
Pro        8,919       —              13,687             855               106
Flash      6,082       0.3 MB SWA     20,427           1,284               160
MiMo      17,920      26.9 MB SWA      5,758             421                52
```

Cross-check: GLM at 13,312 requests x 4,096 tokens x 89,856 B / 13 nodes =
377 GB/node — the reconciliation doc's independently derived "370 to 429
GiB per node" (BATCHPLANE_MODEL_RECONCILIATION.md:72-74). The accounting
agrees to 2%.

Prefetch budget. NVMe read bandwidth is not pinned anywhere in this tree —
assume ~5-7 GB/s per node (Gen4/Gen5 x4 class), 65-91 GB/s ring aggregate,
explicitly PENDING until the ring measures it. KV demand if the working
set were served from NVMe live, at the 80%-target rates above:

```text
GLM  B1     9.9 GB/s   feasible per-node, tight
GLM  B64   50.1 GB/s   ring-marginal: needs 55-77% of aggregate NVMe with
                       perfect prefetch-ahead — no headroom for churn
GLM  B256 166.9 GB/s   impossible: 2-3x the whole ring's NVMe
K3   B256  23.0 GB/s   feasible (compressed-context MLA is cheap to re-read)
K3   B1024 64.4 GB/s   ring-marginal
Pro  B1024 63.7 GB/s   ring-marginal at 2K; at 32K ctx the read is
                       0.28 GB/seq/step x 3,495 = ~960 GB/s — impossible
Qwen B1024 >1 TB/s     impossible by an order of magnitude
```

Verdict: **NVMe cannot be in the decode KV path at B256+ for any model,
and at B64 it is already marginal for the latent-cache models.** The
working set must be GPU/host-resident — Pro at B1024 x 32K ctx is 282 GB
ring-wide (22 GB/node), which fits host memory — with NVMe as overflow
for cold sequences under admission control, exactly the JIT-paging posture
the B1024 integration doc already mandates
(docs/GLM52_B1024_JIT_KV_INTEGRATION.md:62-66: report kv_nvme_read_bytes
per generated token or the throughput claim is invalid). The conclusion is
robust to the unpinned NVMe constant: 2x the assumed bandwidth moves only
the B64 GLM row from "marginal" to "feasible".

### Per-batch attack priorities

```text
B1     1. D2 tensor-map cache (3-13%, certain, days)
       2. D10/D1 CUDA graphs + fusion (12-34% of every model's step)
       3. D6 pre-advertised RDMA slots (~0.35 ms/tok)
       4. D3 warp-reduce delta norm (hours)
       State dtype: inert. NVMe: inert. Compute: irrelevant at B1.
B8     1. graphs/fusion still pay (launch tax is 6-15% of the ~2x-longer step)
       2. expert-queue depth and grouped-MoE plumbing: the coverage stream
          is now 62-83% of MoE step bytes at ~1 row/expert — the worst
          amortisation regime on the curve (PHASE6_REMAINING_WORK.md:142-150)
       3. split-KV decode for CTA parallelism (PHASE6_REMAINING_WORK.md:152)
B64    1. device-side grouped MoE + route-aware scheduling (85% of bytes)
       2. D5 bounded ack window + D7 packet slab (packet rates, fill/drain)
       3. Qwen: compute-bound at the wall from here — WMMA retile or FP8 FFN
       4. KV residency policy: NVMe overflow must be admission-controlled
B256   1. WMMA RETILING — Pro/Flash are past the wall crossover (B165/B130),
          K3 crosses at B258; this is the batch where the 6.5 TFLOPS figure
          stops being a glm52 footnote and becomes the ceiling
       2. K3 bf16 state: +6.9%, first batch where it pays
       3. NVMe: settled — resident working set, JIT overflow (above)
       4. D9 indirect-A gather: still <0.5% of step bytes; keep it last
B1024  1. compute efficiency, nothing else first: at the wall every model
          is 1.4-7x below its memory roofline; retile targets ~8-10 TFLOPS
          /node BF16-class for K3/GLM, FP8-class efficiency measurement for
          Pro/Flash (calibration PENDING list)
       2. K3 bf16 state: +22.2% memory headroom, a prerequisite once
          compute is fixed
       3. full-pool expert streaming as batched dense GEMM, weight-
          stationary (18-32 rows/expert — the grouped-MoE continuation,
          PHASE6_REMAINING_WORK.md:142-150)
       4. KV admission control against the capacity table above
```

### Corrections this analysis makes to earlier numbers in this document

- **The "K3 state correction" section above says "at B64 the state is ~40%
  of the step's bytes at the BF16 recipe".** That figure priced the B64
  step as fixed-weights + state and ignored expert coverage growth. With
  coverage the B64 step is 1,157.9 GB and the state is 5% of it. The state
  reaches first-order (13%) only at B256 and dominance-adjacent (36%) only
  at B1024. The bullet's own lever conclusion (bf16 state matters at
  batch, not at B1) survives; its magnitude was wrong.
- **"At B1 there is no amortisation" (The roofline law, above) is
  incomplete for B>1:** the fixed stream amortises 1/B but the
  routed-expert stream anti-amortises along the coverage S-curve until
  ~B150-B300 depending on the model. Every B8-B64 aggregate figure in the
  per-model tables is dominated by this term.
- **Flash B1 is 14.3 GB here vs 14.7 GB in the B1 table** — the difference
  is the ~1 GB "indexer + misc" lump, which the decomposed table prices
  explicitly (KV read at 2K included). No behavioural change.
- The launch tax's relative weight is batch-dependent: D1/D10 are B1-B8
  items. At B256-B1024 the same 6.6-16.5 ms is 1-4% of the step, and the
  attack list re-orders accordingly (above). The B1-ordered list below is
  unaffected.

## The static defect inventory, priced

Budgets used below: K3 at the 12 tok/s target = 83 ms/token; at the 20.8
tok/s 80%-eta ceiling = 48 ms. DSv4 Pro at 50 tok/s = 20 ms. GLM at 25
tok/s = 40 ms. Launch cost 2-5 us (GB10 launch/plan floor,
GB10_CUDA_COST_MODEL_CALIBRATION.md:36-44; the calibration doc flags the
exact constant as PENDING).

### D1. Launch count — 20 launches per KDA layer, ~3,300 per K3 token

Counted from `inference/llms/kimi_k3/layer.cuh`: the KDA path
(`K3LayerKda`, layer.cuh:407-507) issues 20 launches steady-state — 1 norm,
3 q/k/v projections, 3 convolutions, 2 L2 norms, 2 decay projections +
decay kernel, beta projection + sigmoid, 1 delta rule, 1 output norm, 2
gate projections, 1 output gate, 1 output projection. The LatentMoE
(`K3LayerLatentMoe`, layer.cuh:592-726) adds 15; MLA (`K3LayerMla`,
layer.cuh:515-589) runs 13 + the same 15. Plus ~2 AttnRes retrievals per
layer and the head: **~3,300 launches per K3 decode token**.

Cost: 6.6-16.5 ms/token of launch overhead if the host cannot stay ahead —
12-34% of the K3 budget, and against DSv4 Pro's 20 ms budget the ~1,100
launch equivalent is 3.3-5.5 ms = 16-27%. NOTE: the ledger's S5 says "~700
per K3 decode token" (docs/BANDWIDTH_LEDGER.md:71-77) — that figure is
stale against the current layer file; re-derive before quoting it.

Fix direction: CUDA graphs per (rows, layer-range) shape — the ledger's S5,
and the plumbing already exists: `cuda_graph_exec_cache` slots
(`include/sparkpipe/spark_resident_decode_stage.h:790`) and capture/replay
counters (`inference/stage/module.c:1590-1591`); capture itself does not
(docs/AUDIT_2026-07-29_BANDWIDTH_PATH.md:52-59). Kernel fusion (conv + L2
norm + decay + beta into the projection epilogues, the conv triple into
one launch) cuts the count ~2x and multiplies with graphs.

### D2. Per-launch tensor-map encodes on the host hot path

Every GEMM launch re-encodes both TMA descriptors:
`LmGemmLaunchAsymmetric` calls `LmGemmEncodeMapsSplit` unconditionally
(`runtime/gemm.cuh:297-312`), two `cuTensorMapEncodeTiled`-class driver
calls per launch. At ~1,300 K3 GEMM launches/token that is ~2,600 host-side
driver encodes per token; at ~0.5-1 us each, 1.3-2.6 ms of pure host time
per token — 3-13% of budget, and it serialises with launch submission.

Weight pointers and shapes are stable for the life of a bind: a descriptor
cache keyed by (pointer, geometry) hits ~100%. Effort: days. Certainty:
high — the encode inputs are launch-invariant by inspection.

### D3. Serial thread-0 norm inside the delta rule

`LmDeltaRuleKernel` does the in-kernel q/k L2 norm as a serial 128-element
reduce on thread 0, bracketed by two `__syncthreads()`
(`inference/kernels/linear_attn.cuh:325-340` — the comment says "Serial
reduce by thread 0" and justifies it by KEY_DIM=128). Every other lane of
the block idles through it, once per row per head, on all 96-head x
69-layer blocks. A warp shuffle reduce over 128 elements is the same five
instructions the file's own `LmBlockSum` uses elsewhere
(`inference/kernels/norm.cuh:29-46`). Microseconds per launch, but it sits
inside the one kernel that owns the state slot; at 69 layers x 96 heads of
blocks it is pure occupancy waste. Effort: hours. The same pattern in the
fold path (linear_attn.cuh:225-231) is verify-only, lower priority.

### D4. Host-staged F32 all-reduce

`ring/transport/tp_collective.c` runs the TP all-reduce over TCP sockets
from host memory in f32 (ledger S4, docs/BANDWIDTH_LEDGER.md:64-69).
Decode payloads are bf16 device tensors: the path pays device->host staging
plus 2x wire width. At decode the AR is latency-bound (tolerable); at TP
prefill the format tax alone is ~0.4 ms/token. Fix is named: bf16 sum
variant plus device staging on the RDMA tier. Effort: medium; hardware to
qualify.

### D5. Stop-and-wait acknowledgement RTT

Forwarding retains each packet until a matching ack and the window is
deliberately stop-and-wait (docs/PHASE6_TRANSACTIONAL_COMPLETION_OWNERSHIP.md:60-62;
the replacement is already specified at docs/PHASE6_REMAINING_WORK.md:111-120).
Cost: head-of-line blocking per hop — one 29 us floor per in-flight packet
per stage boundary. At B1 with 13 cohorts in flight the pipeline hides most
of it; at fill/drain and at high batch it caps packets/s directly. PHASE6's
own note: "cannot be the final B256-B1024 protocol"
(docs/PHASE6_REMAINING_WORK.md:168-169).

### D6. RECEIVE_READY RTT per payload

The RDMA transport negotiates every payload with a control round trip:
`SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY`
(`ring/transport/rdma.cu:65`), sent per packet (rdma.cu:2124) and consumed
at rdma.cu:2061. That is one extra network RTT (~29 us floor) before every
payload write on every hop — ~0.35 ms/token across 12 hops at B1, worse per
prefill chunk. The fix is already written down twice: pre-advertise
persistent receive slots (docs/PHASE4_REMAINING_WORK.md:131-133,
docs/PHASE3_REMAINING_WORK.md:48). Effort: low-medium. Certainty: high.

### D7. Per-packet malloc

Every queued work packet takes a fresh heap copy:
`slot->packet_bytes = calloc(1u, packet->descriptor_bytes)`
(`node/backend.c:1994`), with the matching free on retire. Microseconds of
allocator time plus jitter per packet per hop; at B256-B1024 packet rates
this is a latency-variance source, not a throughput wall. The bounded slab
with slot generations is already specced
(docs/PHASE6_REMAINING_WORK.md:119-120). Effort: hours.

### D8. Full-vocab head streaming every step

The head reads the whole vocab shard per token: K3 2.35 GB
(`K3Head`, layer.cuh:758-767; geometry config.h:76-78), GLM 1.9 GB
(the kernel header prices it itself, `inference/kernels/head.cuh:11-16`).
At TP13 that is ~180 MB/rank for K3 — 1.7% of the 48 ms budget; at a
hypothetical TP16 it is the ledger's second-largest tensor (S7,
docs/BANDWIDTH_LEDGER.md:84-89). Verdict stands: architectural; exact
sampling admits no cheat. Mitigations already in-tree: DSpark amortises one
head read across accepted tokens (ledger S7), and the restricted-vocab form
is exact where a grammar applies (`GLM52_RESTRICTED_VOCAB`,
modules/glm52_resident_decode_stage/source/cuda/config.h:27-30; `K3Head` already takes `token_ids`,
layer.cuh:758).

### D9. MoE gather double-touch

`LmGatherRowsKernel` copies each routed row (latent R + packed W,
layer.cuh:629-633), then the weight-only GEMM reads the packed copy again
(layer.cuh:647-651): 2R+1W where a gather-aware A-load pays 1R (ledger S1,
docs/BANDWIDTH_LEDGER.md:36-46). At B1 this is 16 rows x 3584 x 2 B x 2 =
229 KB/layer — noise. At B16 it is ~150 MB/token across the MoE stack; at
prefill width it matches the amortised weight stream. Fix: indirect-A
variant of the weight-only GEMM. Effort: high; TMA-vs-cp.async occupancy is
an open hardware question (ledger S1 "UPDATE B-16s"). Prioritise for batch,
not for the B1 targets.

### D10. No CUDA graphs

Covered as D1's fix; listed because the audit names it its own finding
(F2, docs/AUDIT_2026-07-29_BANDWIDTH_PATH.md:52-59). The engine's step
planner was built so step shapes repeat exactly — capture is unblocked
design-wise (docs/BANDWIDTH_LEDGER.md:75-77).

## Prioritised attack list — (impact x certainty) / effort

Ordered for the B1 targets on the 13-ring. "Recovery" is against the
relevant target budget; all entries are host-verifiable except where noted.

```text
#  item                          effort   recovery                    why this order
1  D2 tensor-map descriptor cache days    1.3-2.6 ms/tok (3-13%)     certain, cheap,
                                                                        unblocks D10
2  D10/D1 CUDA graph capture     weeks   6.6-16.5 ms/tok (12-34%)   the K3 12->20+
     (per step-shape, engine level)                                 tok/s gap itself;
                                                                    Pro needs it for 50
3  D1 fusion: conv/L2/decay/beta, 1-2 wk  ~2x fewer KDA launches    multiplies with #2;
     conv triple -> one launch                                     verifiable on host
4  D6 pre-advertised RDMA slots  days    ~0.35 ms/tok B1,           already specced
                                         per-chunk at prefill       (PHASE4:131-133)
5  D3 warp-reduce delta norm     hours   intra-kernel occupancy     trivial, safe
6  D4 bf16 device-staged AR      medium  ~0.4 ms/tok TP prefill;    hardware to qualify
                                         decode latency             (ledger S4)
7  D5 bounded ack window         medium  batch + fill/drain         already specced
                                                                    (PHASE6:111-120)
8  D7 packet slab + generations  hours   jitter, not throughput     trivial, safe
9  D9 indirect-A gather GEMM     high    ~150 MB/tok at B16;        batch-only; open
                                         ~10 GB/chunk prefill       TMA occupancy Q
10 D8 restricted-vocab head      days    up to 2.35 GB/step where   exact; applies only
     where grammars apply                  a grammar exists         to constrained decodes
```

Items 1-5 close the gap to the quoted B1 targets on every model. Items 6-10
are the batch and prefill campaign. Nothing above requires a guessed
constant; every entry's cost is derivable from the cited code.

## vLLM/SGLang technique map — what this tree has, plans, or lacks

```text
technique              vLLM/SGLang reference        sparkpipe status
CUDA graphs            piecewise (vLLM), full-graph  PLANNED — slots + counters exist,
                       decode graphs (SGLang)        capture absent (D10; stage.h:790)
fused/grouped MoE      fused_moe, FlashInfer grouped PLANNED — device-side expert-major
                       GEMM                         descriptors, weight-stationary
                                                    continuation (PHASE6_REMAINING_
                                                    WORK.md:142-150, PHASE4:109-123)
persistent decode      FlashInfer persistent         GAP — not in the tree. The delta
                       scheduling; megakernel decode rule is persistent within a launch
                                                    (linear_attn.cuh:268-276) but the
                                                    step is not one persistent kernel;
                                                    graphs (#2) are the chosen route.
chunked prefill        standard in both              PARTIAL — glm52.json:36 caps
                                                    dispatch at 256 tokens; the ledger
                                                    prices chunked TP prefill; but the
                                                    multi-chunk DECODE path is fail-
                                                    closed (AUDIT F1, backend.c
                                                    DecodeInner chunk_count != 1)
radix/prefix cache     radix tree (SGLang), APC      PRESENT — prefix reuse default-on
                       (vLLM)                         (README.md:152-156), hash-
                                                    consistent prefix cache in the
                                                    mandatory contract (must_work_
                                                    targets.json:26)
split-KV decode        FlashDecoding, split-KV       PLANNED — "split-key grouped
                       attention                      attention so KV reuse does not
                                                    collapse B1 CTA parallelism"
                                                    (PHASE6_REMAINING_WORK.md:152)
speculative decode     MTP (vLLM), EAGLE (SGLang)    PRESENT — MTP + DSpark block 8
                                                    (glm52.json dspark/mtp fields;
                                                    speculation/ tree; ledger S7)
quantised KV           FP8 KV in both                PRESENT as format option —
                                                    LmKvLatent parameterised on bits
                                                    (layer.cuh:31; K3_KV_BITS)
```

No technique the quoted targets depend on is missing from the plan; the two
that are missing from the code (graphs, device-side grouped MoE) are the
two the attack list puts first.

## What must be measured on hardware — the honest list

Everything below is unverifiable from this machine, in the order bring-up
should run it. The instruments already exist; the receipts do not.

1. **eta_bw per kernel class.** The 0.80 figure is calibrated on glm
   buckets (GB10_CUDA_COST_MODEL_CALIBRATION.md:87-97); a WMMA tile, a
   memcpy-bound scatter, and a head reduce hit different fractions of peak.
   The whole roadmap is priced at 0.80; if the KDA delta rule or the MXFP4
   decode-in-load path lands at 0.6, every ceiling here drops 25%. This is
   measurement #1 because it re-prices everything else.
2. **The launch_ns constant.** PENDING in the calibration doc
   (GB10_CUDA_COST_MODEL_CALIBRATION.md:155-157). D1/D10's recovery is
   6.6-16.5 ms at 2-5 us; the real number picks how much fusion (#3) is
   needed on top of graphs.
3. **Phase-clock gaps between slice completion and next launch, per
   stage** — the audit's named first measurement
   (AUDIT_2026-07-29_BANDWIDTH_PATH.md:56-59), using the per-slot
   `phase_clock_cycles` instrument already in the tree.
4. **Transport overlap discipline.** Whether cohort N+1's receive is posted
   while N computes (AUDIT F3); the 29 us/hop floor is measured, the
   overlap is not (HANDOFF_DECODE_PERFORMANCE.md:236).
5. **The chunk path.** Multi-chunk decode is written and gated off
   (AUDIT F1); first hardware task is validating it and learning the real
   per-node row ceiling.
6. **TMA vs cp.async occupancy for the gather** — gates D9's fix shape
   (ledger S1).
7. **L2 vs DRAM on weight re-reads** — one Nsight counter, gates M-tile
   depth in the projection kernels (HANDOFF_DECODE_PERFORMANCE.md:215-218).
8. **Memory latency microbenchmark** — sets pipeline depth for every
   kernel; nothing in the repo measures it
   (HANDOFF_DECODE_PERFORMANCE.md:231-233).
9. **Real route-skew histograms** — replace the uniform-routing expert
   estimates the ledger's batch law uses
   (docs/BANDWIDTH_LEDGER.md:114, PHASE6_REMAINING_WORK.md:148).
10. **The QKVO WMMA 6.5 TFLOPS wall.** Measured on glm
    (GB10_CUDA_COST_MODEL_CALIBRATION.md:66-73); if the family kernels
    inherit the tiling, K3 and qwen (hidden 7168/5120) hit it harder.
    Retiling is a compute fix, invisible to this byte model — which is
    exactly why it needs a profile, not an estimate.

## Standing uncertainties in this document

- DSv4 attention parameter counts are enumerated from contract geometry,
  not from a checkpoint tensor list; the grouped low-rank output projection
  (output_lora_rank, output_group_count) is approximated as full-width.
  +-15% on the DSv4 attention line moves the Pro ceiling by ~4 tok/s.
- The MXFP4 byte rate uses 4.25 bits/weight (group-32 E8M0); the measured
  scale-plane compressibility (docs/BANDWIDTH_LEDGER.md:151-194) is a
  capacity/bandwidth upside not priced here.
- K3_SPEED.md's B1 68 tok/s (K3_SPEED.md:45-53) implies ~52 GB/token —
  reachable only under the FP8-attention recipe, not the BF16 contract
  recipe. Treated here as the optimistic recipe, not an error; the target
  clears either way.
- The launch counts are static LM_LAUNCH enumeration; the driver may elide
  or batch some. +-10% does not change any ranking.
- Per-batch section: expert coverage assumes uniform routing; measured
  glm52 skew (87% at B128 vs 98% uniform, calibration doc:62-64) means the
  B8-B128 expert-stream figures are upper bounds. Real route histograms
  are standing measurement #9.
- Per-batch section: the DSv4 compression-ratio tables are off-by-one
  against the layer counts (62 entries for 61 Pro layers, 44 for 43 Flash
  layers, trailing 0s; deepseek_v4/config.h flags this UNVERIFIED). The KV
  bytes/token move by at most one layer's share, +-15%.
- Per-batch section: the per-node NVMe read rate (~5-7 GB/s assumed) is
  pinned nowhere in this tree; the KV residency verdicts are stated to be
  robust to 2x in either direction, but the B64 GLM "marginal" row flips
  to "feasible" at the high end.
- Per-batch section: GLM's DSA index cache read (32 heads x 128 dim per
  slot, glm52.json) is unpriced; at 2K context it is second-order beside
  the 1152 B latent read, at 1M context it is not.
- Per-batch section: the 6.5 TFLOPS wall applied to FP8/FP4 expert GEMMs
  is a pessimism, not a measurement — their efficiency is on the
  calibration doc's PENDING list, and Pro/Flash's B165/B130 compute
  crossovers move past B1024 if FP8 GEMMs hold ~40% of FP8 peak.
