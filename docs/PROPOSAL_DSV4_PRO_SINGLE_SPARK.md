# DSV4 Pro — idle spark0/spark2 scout: inventory, size check, single-spark proposal

Read-only scout of spark0 (aitopatom-9ab9) and spark2 (aitopatom-931a) on
2026-08-17, via `ssh -o BatchMode=yes`. No downloads, no launches — this is the
inventory + proposal only, per instruction. All sizes are measured on-host.

---

## 1. Inventory — what Pro artifacts exist on spark0 / spark2 RIGHT NOW

| Artifact | spark0 | spark2 | Notes |
|---|---|---|---|
| **TP4xPP4 rank pack** | `dsv4_pro_tp4_pp4_stage.spstage` = **99,603,890,892 B = 92.76 GiB** | same = **92.76 GiB** | `/home/{host}/sparkdata/dsv4_pro.tp4pp4/packs/`; this is the PP+TP shard, NOT a tp16 shard |
| Serving adapter | `libdsv4_pro_tp4_pp4_serving_adapter.so` (597 KB) | same | built Aug 17 14:23 |
| Transport lib | `libhidden_transport_spark_host_rdma_verbs.so` (201 KB) | same | built Aug 17 14:23 |
| Model driver | `model_driver.so` (7.29 MB) | same | built Aug 17 14:23 |
| residentd / batch | `sparkpipe_model_residentd` (474 KB), `sparkpipe_model_batch` (734 KB) | same | built Aug 17 14:23 |
| Stage / resident config | `dsv4_pro_tp4_pp4_stage.json`, `model_resident.json` | same | Aug 17 04:07/04:09 |
| KV backing dir | `kvcache/dsv4_pro/tp4pp4.bf16` (empty) | same (empty) | + `pp13.bf16` (flash) |
| **TP16 PRO shards** | **none** | **none** | only `dsv4_flash.fp8.tp16.*` (Flash) exist |
| Full pack / val slices | none | none | full pack lives on sparkb/spark3, not here |
| Pro model source (extnvme) | **none** (only Qwen) | **Flash-0731 only** (no Pro-0813) | Pro GA source is on spark3 |
| Single-spark validator / build tree | none | none | `/tmp/dsv4pro-validator*` + `/tmp/sparkpipe-pro-dev` live on sparkb |
| GPU | NVIDIA GB10 (idle) | NVIDIA GB10 (idle) | present, not running |
| Host RAM | 119 GiB (13 used / 106 avail) | 119 GiB (3 used / 116 avail) | GB10 unified memory |
| cgroup | `MemoryHigh=100G`, `MemoryMax=108G` (=115,964,116,992 B), `MemorySwapMax=0` | same | `/etc/systemd/system/sparkpipe_model_residentd.service` |

**Bottom line:** both hosts carry a healthy, current **TP4xPP4** deployment
(rank pack + adapter + driver + residentd + config). They carry **no** tp16
shard, **no** Pro model source, **no** validator, and **no** val-slice — so any
"build a TP2/slice pack" or "toggle FP8-KV" work must fetch artifacts from
sparkb/spark3 first.

---

## 2. Size check vs the 108 GiB guardrail

Reference sizes:

- Authoritative full pack: **864,875,157,944 B = 805.4 GiB** (61 layers + MTP).
- Measured TP4xPP4 rank pack (this host): **92.76 GiB**.
- **Hypothetical single TP16 rank shard** (full model at 1/16 width, MTP also
  sharded): ~805.4/16 ≈ **~50 GiB** — would fit comfortably under 108 GiB.
  **It does not exist on disk** and cannot be built here (no Pro source / full
  pack on either host).

Why the TP4xPP4 rank pack is ~2x a TP16 shard: every rank pack **replicates the
full GA MTP/DSpark block** (3 draft layers + main_proj + markov + confidence +
hc_head) into each rank (`tools/dsv4_pro_stagepack.py:175-214`; receipts
"the module replicates the complete draft block into every rank pack"). That
~40 GiB replication is what pushes the single rank from ~51 GiB (backbone slice)
to 92.76 GiB.

Guardrail arithmetic for the **existing** rank pack:

| Component | Size |
|---|---|
| Rank pack weights (resident, GB10 unified mem) | ~92.76 GiB |
| + runtime / CUDA context / activations (1 slot) | ~2-4 GiB |
| + KV physical pool | **production config = 16,384 pages x ~4.3 MB ≈ 70 GiB (would blow the cap if pre-allocated)**; validation-sized 1,024 pages ≈ 4.3 GiB |
| **Total** | **~99 GiB (min pool) → ~166 GiB (production pool)** |
| cgroup | MemoryHigh=100G, MemoryMax=108G |

**Verdict:** the single rank pack itself (92.76 GiB) is under the user's ~100 GiB
threshold, but the **resident total is right at the edge** — ~99 GiB with a
minimal pool (fits, but is already past MemoryHigh=100G and only ~9 GiB under
MemoryMax), and it does **not** fit with the production 16,384-page pool. This is
exactly the question an empirical boot answers (see P1). A true TP16 shard
(~50 GiB) would be safe, but does not exist.

---

## 3. Proposal — highest-value use of 1-2 sparks (no download)

### P1 (recommended, zero download): solo residentd boot + cgroup memory-footprint profile
Boot the **existing** TP4xPP4 residentd in solo mode on spark0 (rank 0) — and
optionally spark2 (rank 2) — under the systemd unit (which enforces
MemoryMax=108G), re-running the sparkb "R7 solo boot" pattern
(`tools/devcycle/dsv4_pro_single_spark_receipts.md:105-132`).

- **What it validates:** (a) the open size question *empirically* — peak
  `memory.current` during pack load + module init vs MemoryHigh/MemoryMax, i.e.
  does the 92.76 GiB rank pack actually load under 108G, and how much headroom
  remains for a KV pool; (b) per-rank **pack-load + module-init + transport-open
  wall time** (a load-time datum that bounds the weight-bound claim's I/O leg,
  though NOT the per-token decode); (c) health of these two hosts' adapter/driver/
  pack after the Aug 17 rebuild (both hosts' binaries were rebuilt 14:23 today).
- **Not** a token run: a single PP/TP shard cannot decode end-to-end (needs its
  3 TP peers + 3 PP stages), so it will reach peer-connect and busy-timeout (120 s)
  cleanly — which is the expected, useful outcome.

### P2 (optional second spark): two-rank transport/collective bring-up (rank 0 + rank 2)
Boot spark0 (rank 0) + spark2 (rank 2) together. Both are PP stage 0 (TP group
{0,1,2,3}), so this exercises the **TP collective credit handshake + hidden
transport open between two live ranks** instead of all-absent peers.

- **What it validates:** the F1/F2/F3 risk surface (CUDA-graph prewarm, fail-stop
  collective, 120 s handshake window) on the real fabric, and that two ranks'
  RDMA QPs/credits establish before the full ring. Boot still stalls at the
  missing ranks 1,3 (by design); no decode.
- **Δ value vs P1:** de-risks the collective path; P1 alone is sufficient to
  answer the size question.

### Candidate verdicts (the three the user named)
1. **TP2 shard run of the GA DSpark draft path** — **not runnable today**: the
   DSpark draft execution is not implemented (`spark_dsv4_resident_decode_stage_module.c:47-48`
   "refused until a native pass lands"; no `SparkDsv4DSparkLaunch*` kernels in the
   tree), a true TP2 shard (~805/2 ≈ 402 GiB) exceeds 108 GiB, and no TP2/slice
   pack or validator exists on these hosts. Blocked on DSpark kernels + a slice
   pack + validator build (all downloads/builds you deferred).
2. **Per-rank decode timing profile (weight-bound claim)** — **partial**: a single
   shard cannot decode, so per-token expert-streaming timing needs the full ring.
   P1's pack-load + init timing is the only per-rank timing obtainable on 1-2 sparks.
3. **FP8 KV codec selectability toggle** — **not runnable here**: FP8 KV is a
   build-time flag (`Makefile.pro:14-20`) + pack-header change; needs a module
   rebuild (source tree is on sparkb) and a `--kv-codec fp8_e4m3` pack re-gen.
   Blocked on a build + pack, which you deferred.

---

## 4. What I need from you
- **Go signal** to execute P1 (solo boot + memory profile on spark0, optionally
  spark2) — the only zero-download, high-value option. I will NOT download or
  launch until you say so.
- If you also want P2 (two-rank transport bring-up), confirm both hosts are free
  to stay in a 120 s busy-timeout state (they are idle, so no eviction conflict).
- For the DSpark-draft or FP8-KV candidates, authorize a fetch of the val-slice
  pack + validator from sparkb/spark3 (and/or a DSpark kernel build) — that is a
  separate, larger step and is currently out of scope for "no download".
