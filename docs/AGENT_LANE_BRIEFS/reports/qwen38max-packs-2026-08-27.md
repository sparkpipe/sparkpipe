# qwen38max lane report — 2026-08-27

Lane: Qwen 3.8 Max pack building (branch `lane/qwen38max`, worktree
`/tmp/lane-qwen38max`). Nodes: sparkb, sparke, sparkc, sparkd.
Model: Qwen/Qwen3.8-2.4T-A95B (qwen3_5_moe arch, hidden 8192, 92 layers
3:1 GDN:full, MoE 512+1 top-10 moe_int 2048, vocab 248320, MTP-1, FP8
experts with 128x128 F32 scale_inv, everything else BF16).

## Milestone status

| Milestone | Status |
|---|---|
| P1 source verify | DONE — cold source verified against contract |
| P2 packer check/fix | DONE — 4 bugs fixed; TP4xPP4 semantics pinned by the module (see finding F1) |
| P3 build 4 stage packs (16 rank deployments) | RUNNING on spark0 (local NVMe scratch), see live section |
| P3b pack verifier gate | TOOL BUILT + self-tested (PASS good, FAIL corrupt); full packs verify when builds land |
| P4 deploy | BLOCKED by memory arithmetic (F2) — deploy tooling shipped; packs parked on build node |
| P5 smoke | PASS at smoke scale — module builds + loads packs (synth AND real checkpoint slice) |

## P1 — source identity (commands + raw output)

Cold source `/mnt/cold-raid6/models/qwen3.8-2.4t-a95b-fp8` on spark0
(warm copy to `/mnt/model-warm/qwen3.8-2.4t-a95b-fp8` still IN FLIGHT —
3 concurrent model copies share the Ceph mount, ~50-60 MB/s aggregate,
ETA ~10h from 04:19; `.COPY-DONE` polled every 5 min in background).

```
HASH_VERIFY.json: {"bytes_checked": 2496154395007, "files_checked": 224,
 "files_total": 224,
 "manifest_sha256": "8825c2e711da1194066f85c6d689053ece88597bc1a0f850b65841d42e1441b3",
 "state": "verified"}
```
manifest_sha256 matches `model_contracts/qwen38_authoritative.json`
`sources.checkpoint_config` EXACTLY. Every contract geometry key matches
live config.json (hidden 8192, 92 layers, 64 q heads, 4 kv, head 256,
GDN 16/128/128/128, conv 4, experts 512, top-10, moe_int 2048, vocab
248320, interval 4, output gate true, tie false, architectures
["Qwen3_5MoeForCausalLM"], max_position_embeddings 262144). Index holds
287,119 tensors (512 experts x 3 weights + 512 scale_inv per layer x 92
+ shared/router/attention/MTP/head).

## F1 — TOPOLOGY FINDING: the module consumes FULL-WIDTH stage packs; TP is kernel-time

The brief's "TP-shard the attention/MoE dims 4 ways per rank pack" is NOT
what the qwen38_max runtime consumes. Evidence (all in-tree, main):

* `modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c`
  `SparkQwen38MaxModuleValidateEntry` rejects any entry whose
  rows/columns differ from the FULL-width shape table
  (`pack_entry_invalid kind=... layer=...`); there are no TP fields in
  the wire header (`spark_qwen38_max_stagepack_format.h`: 26xu32+2xu64,
  no tp_degree/tp_rank).
* The TP kernels slice AT RUN TIME from the resident full-width buffers:
  `SparkQwen38MaxLaunchGroupedExpertLinear` offsets the expert payload by
  `tp_rank * experts_per_rank * payload_stride` (cuda.cu ~1769), and the
  attention kernels index heads/KV by `tp_rank` with the paged cache
  holding exactly 1/tp_degree of the KV heads (cuda.cu 242-250).
  Pre-sharded rank packs would mis-index the expert buffer at execution.
* Therefore: world_rank = pp_stage*4 + tp_rank, and the 4 ranks of a TP
  group load the SAME stage-pack file. "16 rank packs" = 4 distinct
  files x 4 mounts. `tools/qwen38_tp4pp4_packs.py` emits that manifest.

Also note the kernel TP cap: `SparkQwen38LaunchAttnPrepare/Decode` reject
tp_degree where `ATTN_KV_HEAD_COUNT % tp_degree != 0` — with 4 KV heads
TP is capped at 4 in-tree. TP8/TP16 ("expand to 8/16 later") needs KV
head slicing rework first, not just packs.

## F2 — DEPLOYMENT BLOCKER: stage packs exceed node memory by 4.8x-5.1x

Exact pack sizes (dry-run vs live checkpoint, tools/qwen38_stagepack.py):

```
slice=0+23  tensors=423 file_bytes=619404266240 file_gib=576.87
slice=23+23 tensors=419 file_bytes=615297878528 file_gib=573.04
slice=46+23 tensors=419 file_bytes=615297878528 file_gib=573.04
slice=69+23 tensors=442 file_bytes=650427388672 file_gib=605.76  (embed+head+MTP)
```

The loader (`SparkStageModuleLoadDeviceRegion`) makes every entry
resident on the device: one rank of a TP4 stage needs the full 573-606
GiB pack (kernel-time slicing reads its quarter from the whole). sparkb
`free -g`: **119 GB total** unified memory (GB10). Even a hypothetical
rank-sharded pack (experts/4) would be ~148 GiB — still over. The B1
anchors in the brief (1.29 tok/s, ~39 agg @B256 TP4xPP4) are DSV4
measurements (qualification/dsv4/...), not this model; the 2.4T model
cannot fit TP4xPP4 on 16x119GB at FP8.

Paths forward (coordinator decisions, all module work not pack work):
1. TP4xPP8 = 32 ranks (12+11-12 layers/stage ~= 77-85 GiB/rank at FP8)
   — fits 119 GB with KV headroom, needs 32 sparks and the module's
   stage env (SPARK_QWEN38_MAX_STAGE_COUNT supports 32).
2. MXFP4 expert requant (halves expert bytes; wire format 3 + kernels
   exist) + a sharded-pack loader (real per-rank packs, module reads its
   slice) — biggest engineering, keeps 16 ranks.
3. Bigger-memory nodes for 8 of the 16 ranks. Not available in-cluster.

## P2 — packer fixes (all committed on lane/qwen38max)

`tools/qwen38_stagepack.py`:
* removed the dead MXFP4 quantization path — `convert()` called an
  UNDEFINED `copy_mxfp4_tensor` (NameError landmine); contract pins
  experts "kept AS SHIPPED" FP8. MXFP4_GROUP kept (wire header field).
* `copy_fp8_experts` now streams per expert (was a 8 GiB in-RAM
  bytearray per tensor); F32 scale plane buffered (2 MiB) and appended
  after the payload, matching the wire layout.
* receipt sha256 computed hash-while-write (was: full re-read of the
  finished pack — brutal on contended warm storage).
* expert-0 shape checks now verify shapes (was dtype-only).
* receipt `weight_formats.routed_experts` corrected mxfp4_e2m1 ->
  fp8_e4m3_f32b128_scale_inv; docstring corrected (source is the vendor
  FP8 per-expert release, not BF16; TP semantics documented).

New tools:
* `tools/qwen38_tp4pp4_packs.py` — builds the 4 stage packs (max 2
  concurrent, cluster rule) + writes `manifest.json` with the 16-rank
  table (world_rank, pp_stage, tp_rank, host, pack, sha256) and the
  F1 note. Dry-run against the live checkpoint PASSes for all 4 slices.
* `tools/qwen38_pack_verify.py` — THE missing verifier (audit finding):
  (1) structure: header vs firmware constants, inventory count, per-entry
  shape/format/scale/alignment/bounds, no duplicate/missing (kind,layer);
  (2) content: sha256 per entry region vs the byte stream re-derived from
  the checkpoint (BF16 pass-through, BF16->F32 widening, FP8 expert
  stacking + scale planes);
  (3) receipt cross-check (+ optional --recompute-file-hash).
* `tools/qwen38_tp4pp4_deploy.sh` — per-host deploy from the manifest,
  no hardcoded nodes (`--manifest --host [--target-dir] [--dry-run]`).

## Module build + smoke (P5)

The brief's "build issues" reproduced and fixed (commits on lane):
* `.cu` rename drift: `SparkQwen38LinearView` -> `SparkQwen38MaxLinearView`,
  `SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_*` -> `..._MAX_...`,
  launchers `SparkQwen38Launch*` -> `SparkQwen38MaxLaunch*` (the module
  .c externs), `SparkQwen38ConfigureCudaKernels` -> Max,
  `SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS` -> MAX.
* `SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE>` -> 3-arg form
  with `SPARK_LM_CTA_WARPS` (template grew a CTA_WARPS param; 27b module
  is the precedent).

Result: `make -C modules/qwen38_max_resident_decode_stage archive ...`
BUILDS clean on sparkb (nvcc sm_121a, driver 580.159.03) — only benign
set-but-unused warnings.

Smoke ("module ready" equivalent), via `tests/test_qwen38_pack_load.c`
(fixed its own rename drift + made the slice env-overridable; NOTE this
test is an ORPHAN — no Makefile rule builds it; Makefile is out of my
write set — see integration requests):

```
$ ./build/qwen38_max_pack_synthesize --first-layer 1 --layer-count 1 --output synth1.spstage
  (19 tensors, 24.9 GiB, FP8 experts, scale_group_size=128)  [synth tool bug fixed: it wrote 0]
$ ./build/test_qwen38_pack_load synth1.spstage
qwen38_stage initialize ok slice=1+1 gdn=1 attn=0 owns_embedding=0 owns_head=0
initialize status=0 state=0xf746d1e1b010
destroy ok                                   # 4.1s, 24.9 GiB device load
$ ./build/test_qwen38_pack_load gdn1.spstage   # REAL checkpoint slice [0,1), verifier-PASS
qwen38_stage initialize ok slice=0+1 gdn=1 attn=0 owns_embedding=1 owns_head=0
initialize status=0                          # 4.7s, 28.7 GiB real weights loaded
$ SPARK_..._FIRST_LAYER=3 ... test_qwen38_pack_load synth3.spstage
qwen38_stage initialize ok slice=3+1 gdn=0 attn=1 ...   # attention binding path
$ SPARK_..._FIRST_LAYER=91 SPARK_..._STAGE_INDEX=3 ... test_qwen38_pack_load synth91.spstage
qwen38_stage initialize ok slice=91+1 attn=1 owns_head=1 # MTP + head ownership path
```

Full residentd serving remains blocked: `make publish` hard-requires a
GPU_VALIDATOR (`modules/resident_decode_stage_rules.mk:186-188`) and
qwen38_max has NO validator harness (every other family has
`validation/validate_<family>_resident_decode_stage_cuda.sh`). Deferred
per brief; the pack verifier is the gate for packs.

## Verifier self-test (real checkpoint, 1 layer)

```
$ python3 tools/qwen38_pack_verify.py --pack gdn1.spstage --checkpoint <cold> --receipt ...
content 20/20 tensors
qwen38_pack_verify pack=gdn1.spstage verdict=PASS file_gib=28.71 errors=0     (10m1s)

# negative test: flip one byte inside expert-1 payload of MOE_W1 in a copy
qwen38_pack_verify pack=gdn1-corrupt.spstage verdict=FAIL errors=2
  kind=6 layer=0x0 name=model.layers.0.mlp.experts.{e}.gate_proj.weight: pack bytes != source bytes
```
First verifier revision FAILED the good pack (per-expert interleaved
expected stream) — fixed to all-payloads-then-all-scales; the manual
per-component digests proved the PACK was correct both times.

## P3 live status (updated as builds land)

* Building on spark0 (reads verified cold RAID, writes spark0 local NVMe
  `/home/spark0/qwen38max-lane/packs/` — warm Ceph writes measured
  26 MB/s under the 3-copy contention, i.e. ~26 h; local NVMe + rsync
  off node keeps spark0 root-FS headroom safe).
* 2 concurrent builds (cluster rule). Observed combined ~94 MB/s cold
  read under contention. NOTE: the DSV4 lane's packer is running on
  spark0 concurrently, reading the same RAID.
* After each stage: verify (verifier gate) -> rsync to its designated
  node -> delete local copy (keeps spark0 root at <= ~2.3 TiB used).

[final numbers appended below when builds+verify complete]

## Integration requests (coordinator)

1. WIRE TEST: `tests/test_qwen38_pack_load.c` is an orphan (no
   `build/test_qwen38_pack_load` rule in Makefile TEST_NAMES). Link line
   that works: cc tests/test_qwen38_pack_load.c
   build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a
   -lcudart -lstdc++ -lpthread -lm -ldl (needs the module archive built
   first). Request a rule; Makefile is out of my write set.
2. SHARED WIRING: none needed for packs. The module fixes above are all
   inside modules/qwen38_max_resident_decode_stage/ (my write set).
3. DECISION: deployment topology for this model on 16x119GB GB10 (see
   F2 options 1-3). Packs for TP4xPP4 cannot be loaded by any in-cluster
   node at FP8; this is physics, not tooling.
4. DECISION: qwen38_max GPU validator harness (needed for `make
   publish`/`validate` and any decode-parity work). I can build it next
   lane block if assigned (27b's validation/ is the template).
5. KV-head split: TP is capped at 4 by `ATTN_KV_HEAD_COUNT %
   tp_degree` in the attn launchers; TP8/16 needs kernel work.

## Next experiments

* Land P3 packs + P3b verify PASS x4 (running).
* Park packs on sparkb/e/c/d (one per stage) so the artifacts are
  node-local for whichever topology the coordinator picks.
* If assigned: qwen38_max GPU validator harness (27b template), then
  TP4xPP8 mid-tier GPU validation of the GDN/attn/MoE kernels.
