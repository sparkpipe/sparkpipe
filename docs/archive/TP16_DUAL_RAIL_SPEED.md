# TP16 on dual-rail 2x100GbE: topology confirmation and the speed tables

## The topology, confirmed arithmetically
Two switches, each 4x 400G broken out 4-way: 16x 100G endpoints per
switch - exactly one per Spark per switch, all 16 nodes, no port to
spare and none missing. Each Spark's two ports land on two fully
independent switched fabrics: **200 Gb/s (25 GB/s) per node**, double
the single-rail 12.5 GB/s, with zero cross-rail contention and whole-
switch failure isolation. (The specific switch SKU is newer than
anything verifiable offline; the port arithmetic above is what the
plan depends on and it is exact.)

The 50/50 split is not a hashing hope: sparkpipe owns the transport,
so each transfer is striped deterministically - even chunks rail A,
odd chunks rail B - giving 50/50 by construction, not by ECMP luck.
This is a named transport work item: rail-striped sends with per-rail
completion, and per-rail byte counters so the split is measured, not
assumed.

## Does it double throughput? The honest answer
Network throughput: yes, exactly 2x. End-to-end decode: it doubles
the *communication term*, so the gain is largest where that term
binds. TP16 decode has two per-layer allreduces of B x hidden
activations; at B=64 that is 320 MB (K3) / 230 MB (GLM) / 157 MB
(qwen) TX per node per step - 26/18/13 ms on one rail, halved on two,
against weight-stream times that range from 12 ms (qwen) to ~350 ms
(K3). Dense bf16 qwen, whose weights stream in 12 ms, is deeply
comm-bound at batch and gains the most; the MoE giants are weight-
stream-bound and gain ~10-17% at B256. Prefill (not tabled) is
activation-heavy and comm-bound almost everywhere - it approaches the
full 2x.

## The tables
tok/s as serial-overlap bounds: left assumes compute then communicate,
right assumes perfect overlap; reality lands between and the gap is
what per-layer comm/compute pipelining is worth. 16 nodes, 4.37 TB/s
aggregate LPDDR, 15 us allreduce latency floor per hop-pair assumed.

### K3 - MXFP4 experts, bf16 everything else (1.55 TB, fits 2.05 TB with ~500 GB headroom)
| rail | ctx | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| 1 | 4k  | 31-34 | 135-146 | 219-242 | 497-624 |
| 1 | 64k | 30-34 | 128-138 | 201-221 | 416-502 |
| 2 | 4k  | 31-34 | 138-146 | 229-242 | **552-624** |
| 2 | 64k | 31-34 | 131-138 | 210-221 | 454-502 |

The bf16 rest (attention, projections, embeddings: ~102 GB/step)
dominates small batches - hence B1 ~31 tok/s against the ~68 of the
all-MXFP4 estimate in K3_SPEED.md; keeping attention at bf16 costs
single-stream latency and buys numerics. The fp32 KDA state term and
the expert-union curve carry over unchanged.

### GLM 5.2 - fp8 MoE, bf16 rest (751 GB total)
| rail | ctx | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| 1 | 4k   | 72-89 | 201-220 | 368-418 | 990-1401 |
| 1 | 128k | 72-89 | 201-220 | 368-418 | 990-1401 |
| 2 | 4k   | 73-89 | 207-220 | 389-418 | **1154-1401** |
| 2 | 128k | 73-89 | 207-220 | 389-418 | 1154-1401 |

DSA makes context length disappear from the table: reads are capped at
the candidate set (~2k rows) regardless of 4k or 128k - the sparse
attention paying for itself exactly as designed.

### Qwen 3.6 27B - bf16 dense (54.5 GB)
| rail | ctx | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| 1 | 4k  | 68-80 | 855-1172 | 2021-3727 | 3066-4899 |
| 1 | 16k | 67-79 | 738-964  | 1472-2209 | 1959-3263 |
| 2 | 4k  | 69-80 | 933-1172 | **2521-3727** | **4388-8192** |
| 2 | 16k | 68-79 | 796-964  | 1721-2209 | 2426-3263 |

The dual-rail poster child: +25% at B64 serial, +43% at B256 serial,
and in the overlapped bound B256 goes 4899 -> 8192 because comm is
the binding term outright. Full-KV reads (64 KB per context token per
sequence across the 16 full layers) are what bends the 16k rows.

## What this changes on the board
- Transport: rail-striped dual-path sends + per-rail counters (new).
- TP16 shard geometry (PR #503 lineage) becomes the primary layout
  for qwen-class dense models; PP remains right for the MoE giants
  where weight residency, not comm, is the constraint - and the
  serial-overlap gap says per-layer comm/compute overlap is worth up
  to ~40% at large batch, making it a first-class hardware-week
  measurement alongside F1-F3.
- B=1 on TP16 pays ~2.8 ms/step of allreduce latency floors (186
  hops-pairs); single-stream latency still favors PP. TP16 is a
  throughput topology.
