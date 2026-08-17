# DSV4 Pro — P1/P2 execution results (spark0 + spark2, 2026-08-17)

GO executed under the 108G cgroup on both idle hosts. Solo residentd boot (P1)
+ two-rank bring-up (P2). No decode, no downloads. Raw receipts on-host under
`/tmp/dsv4pro-p1-{spark0,spark2}-*.result`.

## Headline numbers

| Host (rank) | Boot | Transport | Memory peak (cgroup) | Verdict |
|---|---|---|---|---|
| spark0 (rank 0, PP stage 0) | **OK** | `fabric_ready` @ ~5 s (rocep1s0f1, gid 3) | **68.7 MiB** (flat, 130 s) | residentd does **not** eagerly load the 99 GiB pack |
| spark2 (rank 2, PP stage 0) | **FAILED** | `fabric_invalid matching_active_100g_count=0` | n/a (unit exited @ 10 s) | **both 100G + 200G RoCE ports DOWN** |

**The 99 GiB-vs-108G answer:** the residentd boots at ~69 MiB — the 92.76 GiB
rank pack is **mmap'd lazily**, never faulted during boot/transport-open/boundary
wait. The 108G cap is therefore **not exercised at boot**; the 99 GiB resident
only materializes at first decode (pack pages faulted by the weight-read kernels),
which a solo PP/TP shard cannot reach. So the cgroup kill risk the user worried
about is a **decode-phase** risk, not a boot-phase risk, and remains unmeasured
until the full ring decodes.

## P1 details

### spark0 (rank 0) — boot success after fixing two stale config paths
- Timings: `fabric_ready` (transport open) at ~5 s; `rank0_to_rank4_hidden`
  boundary connect waited 120 s (peer absent) -> `initialize=busy status=15
  phase=transport_open` -> clean exit at ~130 s. Matches the sparkb R7 solo-boot
  pattern (`tools/devcycle/dsv4_pro_single_spark_receipts.md:105-132`).
- Memory: peak **72,056,832 B (68.7 MiB)**; `memory.events` oom_kill=0. Far below
  MemoryHigh=100G and MemoryMax=108G.

### spark2 (rank 2) — fabric down (NOT a config issue)
- `hidden_spark_rdma_fabric_invalid matching_active_100g_count=0` +
  `hidden_spark_rdma_discover_failed port=1 status=14` +
  `initialize=route_not_found status=14 phase=transport_open rank=2 stage=2`.
- Confirmed at the link layer: `enp1s0f0np0` (200G) and `enp1s0f1np1` (100G)
  are both **DOWN** on spark2; only 10GbE + WiFi are up. (spark0's same two ports
  are UP.) spark2 is "idle" because its RoCE NIC has no link — it would not join
  the 16-rank ring as-is.

## P2 (two-rank bring-up) — BLOCKED
rank 0 (spark0) + rank 2 (spark2) cannot open the RDMA transport/collective
because spark2's 100G+200G ports are down. No collective smoke possible. To run
P2, spark2's NIC link must be restored (cable/switch/port), or P2 moves to a
different host pair (e.g. spark0 + spark1).

## Critical production bug found (blocks the 16-rank ring on ALL hosts)
`model_resident.json` (regenerated 2026-08-17 04:07) carries two stale absolute
paths that do not match the deployed directory names, so **residentd fails at
boot with `deployment=io_error`**:
1. `nodes[].runtime_root` = `.../sparkdata/dsv4_pro.fp8.tp4_pp4.b1024` — actual
   dir is `dsv4_pro.tp4pp4` (verified by `newfstatat` ENOENT).
2. `nodes[].kv_backing_directory` = `.../kvcache/dsv4_pro/tp4_pp4.bf16` — actual
   dir is `tp4pp4.bf16` (verified by `newfstatat` ENOENT).
Both were fixed in place on spark0/spark2 with sed for this measurement; the fix
(`dsv4_pro.fp8.tp4_pp4.b1024 -> dsv4_pro.tp4pp4` and
`tp4_pp4.bf16 -> tp4pp4.bf16`) must be propagated to the config generator and all
16 hosts before the ring run, or the ring will not boot.

## What I did (local, reversible)
- No downloads. On spark0+spark2: created then removed a symlink probe; sed-fixed
  the two config paths (see above); booted `sparkpipe_model_residentd` under a
  transient `systemd-run` unit with `MemoryHigh=100G/MemoryMax=108G/MemorySwapMax=0`
  (mirroring the production `sparkpipe_model_residentd.service`); stopped + reset
  all units afterward. Transient units are cleaned up; the sed edits remain on the
  two hosts (they are the correct production fix, flagged for propagation).
