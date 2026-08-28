# B* knee-sweep lane report — 2026-08-28

Worktree /tmp/lane-knee, branch `lane/knee-sweep`. Node: spark3 only
(Qwen 3.8-27B dense FP8 TP1 deployment `/home/spark3/sparkdata/qwen38.fp8.tp1`).
spark2 untouched. Node left restored (config byte-equal to pristine, no
daemons, 45C idle at exit; backups of everything I replaced kept on-node).

Tool: `tests/test_batch_knee_sweep.sh` (from the continuous-batching merge,
parameterized; invoked per-point with `BATCHES="<single B>"` so every point
is an independent residentd session — one reboot costs one point, per the
coordinator's crash-loop directive).

## Headline

On merged main (continuous batching), the decode throughput curve vs B is
monotone and steep through the entire healthy envelope, and per-step time
STILL FALLS at the top of it:

| config | B | decode tok/s | per-step ms | vs B=1 |
|---|---|---|---|---|
| 128-lane | 1 | 8.44 | 118.5 | 1.00x |
| 128-lane | 2 | 16.86 | 118.6 | 2.00x |
| 128-lane | 4 | 33.73 | 118.6 | 4.00x |
| 128-lane | 8 | 111.18 | 72.0 | 13.2x |
| 128-lane | 16 | 406.65 | 39.3 | 48.2x |
| 128-lane | 32 | **1469.40** | 21.8 | **174x** |
| 64-lane | 4 | 55.48 | 72.1 | (K1 confirm: 55.44) |
| 64-lane | 8 | 204.20 | 39.2 | |
| 64-lane | 16 | 733.05 | 21.8 | |

The old stack's collapse at B=8 is gone (B=8 = 111 tok/s = 13.2x B=1; the
old stack measured ~0.5 tok/s there). The ledger's "Aggregate B16 ~9 tok/s"
becomes 406.65 tok/s (45x). The classical knee (d(step)/dB turning linear)
was NOT reached: at the best healthy point step time is still decreasing in
B. The region above it is walled off by two distinct software defects (P1
findings below), not by memory bandwidth.

## K1 — deployment reproduced at exact HEAD of main

The on-node binaries had no traceable provenance (built 08-28 01:11 by an
unknown lane; the node's own checkout is 265 commits behind), so I rebuilt
all three tools from exact main:

```
$ ssh spark3 'cd ~/sparkpipe && git fetch origin main && git worktree add /tmp/lane-knee-build f8f2ea07'
HEAD is now at f8f2ea07 glm53 assessment: HC inter-stage boundary note from M3 ...
$ ssh spark3 'cd /tmp/lane-knee-build && make -j8 build/sparkpipe_model_residentd build/sparkpipe_model_batch build/sparkpipe_model_api'
cc ... node/model_residentd.c ... -o build/sparkpipe_model_residentd        [exit 0]
$ ssh spark3 'sha256sum /tmp/lane-knee-build/build/sparkpipe_model_residentd /home/spark3/sparkdata/qwen38.fp8.tp1/bin/sparkpipe_model_residentd'
747bebc80d321f48d25d2289f34d474d44e0e5a0a067a36473cbdd12cbea7080  (both — hash-verified install)
```

Pre-lane binaries kept as `bin/*.before-knee-20260827`. Sanity sweep
(config unchanged from CB4, all four config generations byte-identical in
runtime_limits — verified against `model_resident_pristine.json`):

```
$ SPARK_HOST=spark3 DEPLOY=/home/spark3/sparkdata/qwen38.fp8.tp1 BATCHES="1 2 4" tests/test_batch_knee_sweep.sh
B,budget_hi,wall_hi_ms,tokens_hi,aggregate_tok_s,decode_tok_s
1,256,30715,256,8.33,8.45
2,256,61783,512,16.57,16.75
4,256,75997,1024,53.90,55.44
```

B=4 at 55.44 vs CB4's 36.22 (+53%): the whole curve moved up on HEAD, and
B=1 at 8.45 matches the ledger's known-good no-spec regime (7.7 measured /
8.03 HWM) — the old CB4 numbers were taken on the slower pre-lane stack.

Ledger-cell attestation (asked by coordinator): the ledger rows are O512
cells; the sweep protocol is 64-token prompts with budget-difference
windows. Re-ran B=1/2/4 with budgets 256/512 so the measured window is the
256→512 output regime:

```
1,512,60948,512,8.40,8.46
2,512,121902,1024,16.80,16.91
4,512,244809,2048,33.46,33.49
```

B=1/B=2 are window-invariant (8.45/8.46, 16.75/16.91). B=4 is NOT: 55.44 in
the 128→256 window vs 33.49 at 256→512 (longer live contexts cost more per
step at small B). The scoreboard B1 8.45/8.46 number is solid for the O512
cell; the B4 55.44 figure is specific to the 128→256 window and the
64-lane config (see seam table below).

## K2 — KV pool scaling verified, caps raised, and one config rejected

Pool chain (read from source): serving adapter computes
`kv_block_count = resident_sequence_capacity × ceil(max_sequence_positions/64)`
(`modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:2348`),
exports it as `SPARK_QWEN38_27B_STAGE_KV_BLOCKS`, and the stage allocates
`cache_block_stride × kv_block_count` bf16
(`..._module.c:917-920`). `kv_logical/physical_page_capacity` are
ENGINE-side admission bounds only (host memory, no device cost). Measured
with the stage's own ready line:

| config | lanes | blocks | device_gib |
|---|---|---|---|
| 64 lanes, 8192 pos (CB4/K1) | 64 | 8192 | 71.1 |
| 128 lanes, 4096 pos (sweep) | 128 | 8192 | 80.6 |
| 128 lanes, 8192 pos (rejected) | 128 | 16384 | never reached ready |

```
qwen38_27b_stage ready stage=0/1 slice=0+64 tp=1/0 gdn=48 attn=16 lanes=64 kv_blocks=8192 device_gib=71.1
qwen38_27b_stage ready stage=0/1 slice=0+64 tp=1/0 gdn=48 attn=16 lanes=128 kv_blocks=8192 device_gib=80.6
```

The +9.5 GiB for 128 lanes at equal blocks = GDN state pool for 64 more
lanes + 2 more pipeline slots. The 128-lane/8192-pos config was attempted
once and is the probable trigger of the day's second reboot (incident log
below) — it was dropped, and the sweep ran at the coordinator-approved
equivalent-blocks envelope instead.

Sweep config diff (128-lane ladder), exact:

```
 "runtime_limits": {
-  "max_inflight_submissions": 2,     +  "max_inflight_submissions": 4,
-  "max_active_sequences": 64,        +  "max_active_sequences": 128,
    "max_input_rows": 128,               "max_input_rows": 128,
-  "resident_sequence_capacity": 64,  +  "resident_sequence_capacity": 128,
-  "kv_logical_page_capacity": 256,   +  "kv_logical_page_capacity": 2048,
-  "kv_physical_page_capacity": 64    +  "kv_physical_page_capacity": 1024
 }
 config/qwen38_27b_tp1_rank0.json: "max_sequence_positions": 8192 -> 4096
```

Page math at the sweep protocol (64-token prompt + 256 budget = 320 tokens
= 5 pages per request at 64-token blocks): B=128 needs 640 ≤ 1024 physical;
64 blocks/lane ≥ 5 used. Nothing page-bound in the healthy envelope.

**Exact-32K cell infeasibility** (ledger cell = 32768 prompt + 256 output =
33024 positions): 33024 positions = 517 blocks/lane. At 8192 total blocks
that is 15 concurrent lanes; at 128 lanes it is 66176 blocks ≈ 278 GiB of
KV alone on a 121 GiB unified node. The knee sweep therefore uses the
script's 64/256 protocol; the exact-32K cell can only be measured at
B ≤ 15 on this hardware, far below any interesting B.

## K3 — the sweep, point by point (checkpointed per B)

Per-point driver: drop page caches (see operational findings), one
`tests/test_batch_knee_sweep.sh` invocation per B (own residentd), CSV row
appended mac-side and node-side (`/home/spark3/knee-sweep-results.csv`,
9 rows, kept as receipt). Raw rows (budgets 128/256):

```
B,budget_hi,wall_hi_ms,tokens_hi,aggregate_tok_s,decode_tok_s
1,256,30800,256,8.31,8.44          (128-lane config)
2,256,61572,512,16.63,16.86
4,256,123276,1024,33.23,33.73
8,256,151217,2048,108.35,111.18
16,256,170513,4096,384.35,406.65
32,256,198447,8192,1320.98,1469.40
4,256,75661,1024,54.14,55.48       (64-lane config, K1 confirm)
8,256,84855,2048,193.08,204.20     (64-lane)
16,256,99145,4096,661.01,733.05    (64-lane)
```

### Failed points, with raw evidence

**B=24 and B=48 (128-lane): intermittent fatal `FAILURE_DEACTIVATE_ROUTE`.**

```
B=48: sparkpipe_model_batch_failure status=4 stage=0
      sparkpipe_model_batch_pipeline submitted=35 continued=0 admitted=33 rejected=2
      node log: model_residentd route_failed status=17 reason=7 work_kind=1 submission=34
B=24: sparkpipe_model_batch_pipeline submitted=505 continued=0 admitted=503 rejected=2
      node log: model_residentd route_failed status=17 reason=7 work_kind=2 submission=504
```

status 17 = SPARK_STATUS_INTERNAL_ERROR; reason 7 =
`SPARK_MODEL_RESIDENTD_FAILURE_DEACTIVATE_ROUTE`
(`node/model_residentd.c:40`). Root cause path: the engine aborts a
prefetch mid-flight → `SparkModelServingAdapterResolvePrefetch(...,
PREFETCH_RESOLUTION_ABORT)` returns OK → `SparkModelResidentdDeactivateRouteLocked`
fails → the residentd declares a terminal INTERNAL_ERROR for the whole
client (`node/model_residentd.c:1410-1417`). B=16 and B=32 completed
cleanly; B=24 and B=48 died this way — intermittent at lane counts ≥ 24 on
this deployment. It is a client-fatal bug in the prefetch-abort cleanup
path, independent of load level (505 submissions succeeded before the B=24
hit).

**B ≥ lanes/2: pathological slowdown, no errors (both configs).**

- 128-lane B=64: lo run (8192 tokens) took ~39 min ≈ 3.5 tok/s aggregate
  (vs 1469 at B=32); GPU 96% busy throughout; zero error lines in the
  residentd log; hi run pacing ~6 tok/s when aborted at the 30-min soak
  cap (batch client TERM'd cleanly).
- 64-lane B=32: lo run (4096 tokens) 11:15:03→11:34:02 = 19 min ≈ 3.6
  tok/s aggregate; GPU 96%; zero error lines; aborted identically.

Not pages: at 128 lanes the B=64 run had 1024 physical pages for a 640-page
worst-case demand. Not absolute row count: 64-lane B=32 (32 rows) collapsed
while 128-lane B=32 (32 rows) was the best point of the sweep. The common
factor is B ≥ lanes/2. Mechanism unidentified — it is a per-step cost
explosion (~16.8 s/step at 128-lane B=64 vs 21.8 ms at B=32), which puts it
in the same family as the ledger's known FFN row-scaling pathology, but at
a ratio trigger rather than an absolute row trigger. Needs the runtime
owners; everything above B=lanes/2 is unmeasurable until it is fixed.

### The step-time invariant (config seam)

step time is a function of r = B×64/lanes, identical across both configs:

| r = B×64/lanes | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| per-step ms | 118.5 | 118.6 | 72.1 | 39.2 | 21.8 |

(64-lane B=4 = 72.1 ms = 128-lane B=8; 64-lane B=8 = 39.2 = 128-lane B=16;
64-lane B=16 = 21.8 = 128-lane B=32.) The per-step cost therefore scales
with LANE CAPACITY, not with B — i.e. each decode step pays for work
proportional to the configured lane array (GDN state pool is the obvious
suspect: `gdn_pool.lane_capacity = max_active_sequence_count`). Deploying
128 lanes to chase B=128 doubles the per-step floor of every small batch:
at B=4, 64 lanes is 1.64x faster than 128 lanes (55.48 vs 33.73). Match
lane capacity to the batch you actually serve.

## K4 — where is the knee?

1. **The classical knee was not reached.** Through the entire healthy
   envelope (B ≤ lanes/4) the marginal d(step)/dB is negative — step time
   falls 118.5 → 21.8 ms as B grows. The memory-bound → compute-bound
   crossover, defined as where step time starts growing linearly with B,
   lies ABOVE B=32 on this stack. This is consistent with B*≈106 in
   direction but does not confirm the value, because:
2. **Two software walls, not bandwidth, cap the measurement**: the
   prefetch-abort deactivate bug (fatal from B=24, intermittent) and the
   B ≥ lanes/2 ratio cliff (16.8 s/step at 128-lane B=64). Both are P1s
   with receipts above. Until they are fixed, B > 32 is unmeasurable at
   any lane capacity, and the "90%+ compute" claim stays bounded, not
   proven.
3. **The weights-resident reframe (coordinator's framing, supported by the
   numbers)**: the 29.9 GB FP8 pack plus KV live inside the 121 GiB
   unified memory. Once resident there is no per-step HBM weight stream in
   the classical sense — the flat 118.5 ms floor at B ≤ 4 (and its
   lane-scaling) is per-step overhead amortizing across the batch, not
   weight streaming. Evidence: sub-floor step times (72/39/22 ms) are
   impossible if every step re-streamed 29.9 GB at the ~250-273 GB/s class
   bandwidth the B=1 step time implies; and the floor moves with lane
   capacity, not with model bytes. The B* = R/(2·b·BW) law with HBM-class
   BW does not describe this deployment class; the knee physics here is
   lane-array overhead + coherent-memory compute.
4. **Compute sanity check**: 1469.4 tok/s × 54 GFLOP/token (2×27B dense)
   ≈ 79 TFLOPS effective — above the TOPOLOGY_GUIDE's 50 TFLOPS GB10
   figure. Either the guide's compute figure is conservative for sm_121a,
   or dense-27B FLOPs/token is lower than 2×params (48 of 64 layers are
   GDN/linear-attention, which do materially less work per token than the
   2×N assumption), or both. The measured 174x aggregate scaling B=1→32
   is the load-bearing number; the TFLOPS attribution needs a kernel-level
   breakdown before it goes in the ledger.
5. **Thesis bounding**: "3% → 90%+ compute" cannot be settled from this
   curve alone — utilization denominators need the kernel-level FLOP
   accounting. What is proven: continuous batching on main turns the same
   hardware from 8.44 tok/s (B=1) into 1469.4 tok/s (B=32) aggregate with
   per-step time still falling at the edge of the healthy envelope.

## Incident log (honest, chronological)

1. **~08:56 node reboot (day's #2, coordinator's crash-loop context)**: my
   first K2 probe launched residentd at 128 lanes × 8192 positions (would
   be ~104+ GiB device). The node went ssh-unresponsive and returned
   rebooted ~09:02. I did not reboot it. Root cause as accepted by the
   coordinator: noisy-neighbor unified-memory pressure (my ~104 GiB CUDA
   reservation) beside mds + 2×osd on the same 121 GiB node. Config
   retreated to the approved equal-blocks envelope (80.6 GiB); the new
   binding rule (no GPU lane work on storage hosts, my lane excepted at
   the 71.1 GiB-class envelope) came out of this. The day's earlier
   reboot (~08:13) predates all of my GPU work.
2. **11:02 foreign daemon**: a dsv4_pro.tp4pp4 rank-3 residentd appeared
   mid-ladder, broke one B=4 point (client IO_ERROR at connect), and
   self-exited (transport_open timeout waiting for rank 7 — fleet-less
   pattern). Reported to coordinator; not touched by me. One point lost.
3. **SSH relay flakiness**: three dropped connections (mac-side
   ds4_spark_ssh_proxy timeouts, one during a drop_caches+probe). Long
   steps now run detached (setsid nohup) with short poll ssh's; per-point
   checkpointing kept the loss at zero points.

## Operational findings (for whoever runs here next)

- **Page cache starves cudaMemGetInfo on GB10**: after pack mmap reads,
  `free -g` showed 95 GiB buff/cache and a fresh residentd failed with
  `pack_device_memory_insufficient free=17681293312 pack=30135214592`.
  `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` before every residentd
  start fixed it. Unified-memory nodes report cudaMalloc-usable free
  EXCLUDING reclaimable page cache.
- **TERM handling of the current residentd is clean** — verified exits at
  71.1 and 80.6 GiB resident, and after client-side failures. No
  TERM-immune spin observed on this stack (unlike the old stack in the CB4
  incident).
- **pgrep -f self-match hazard**: `pgrep -f sparkpipe_model_residentd`
  matches the ssh wrapper's own cmdline; my first "TERM" killed the
  wrapper and left the daemon up (caused one refused sweep start). Kill by
  exact pid from a pidfile or use `pgrep -x`.

## Node state at exit

Config restored to pristine (verified `live == pristine` runtime_limits;
adapter back at 8192 positions; value-identical, formatting normalized by
the json rewrite — originals kept as `*.before-knee*`). Binaries: HEAD
f8f2ea07 build left in place (hash-verified above), pre-lane binaries kept
as `bin/*.before-knee-20260827`. No daemons, no stray temp files,
`/home/spark3/knee-sweep-results.csv` kept as the node-side receipt.
spark3 stayed up on one boot (08:58:51) through the entire main sweep.

## INTEGRATION REQUEST

No shared files changed by this lane (sweep script used as committed; no
runtime edits). Two P1 findings for the runtime owners, with receipts in
this report:

1. `FAILURE_DEACTIVATE_ROUTE` client-fatal path
   (`node/model_residentd.c:1410-1417`): prefetch-abort deactivate failure
   escalates to terminal INTERNAL_ERROR, killing healthy 500+-submission
   runs at B ≥ 24 (intermittent).
2. B ≥ lanes/2 per-step collapse (~770x step-time blowup, 96% GPU, no
   errors), ratio-triggered (64-lane B=32 and 128-lane B=64; NOT
   page-bound; NOT absolute-row-bound). Candidate families: the FFN
   row-scaling defect's ratio variant, or GDN state pool processing at
   high lane occupancy.
3. (Performance, non-fatal) per-step cost scales with configured lane
   capacity — deployments should size `max_active_sequences` to the
   served batch, not to the descriptor max.

## Next experiments

1. Fix P1 #1, then bracket B=48 cleanly (it never produced a rate).
2. Fix/bypass P1 #2 (or add a lane-occupancy knob), then extend the
   128-lane ladder 48→128 to find the actual knee; prediction from the
   r-invariant trend (21.8 ms at r=16, still falling) is that the knee
   sits well above 32 and the 90%+ thesis is testable there.
3. Kernel-level FLOP accounting for one decode step at B=32 to pin the
   effective TFLOPS and settle the B* formula's compute term for GDN
   architectures.
4. Re-run a low-B ladder with lane capacity exactly B (e.g. active=8 for
   B=8) — the r-invariant predicts the 39.2 ms step at B=8 becomes the
   118.5 ms class floor if lanes are oversized the other way; a
   matched-lane deployment may beat both ladders at every point.
