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

## F3 — HARD CONSTRAINT: 16 ranks at FP8 is impossible; MXFP4 is mandatory for any 16-rank plan

Total routed-expert params: 92 layers x 512 experts x 3 x 2048 x 8192 =
2,362,230,128,640 (~2.36T). FP8 = 1 B/param => 2.36 TB, and per rank that
is total/ranks — INVARIANT under any PP/TP split:

  16 ranks, FP8:      147.6 GiB expert payload/rank  > 119 GB node  => IMPOSSIBLE
  16 ranks, MXFP4:     73.8 GiB payload + 4.6 GiB scales = 78.4 GiB/rank experts
  32 ranks, FP8:       73.8 GiB/rank experts — fits, but the cluster has
                       15 hosts (spark0-15 minus spark1-restarting; spark2
                       released per coordinator) — 32 ranks need 32 hosts: OUT.

So "16 ranks" and "FP8" cannot both hold. The real sprint decision is:
MXFP4 sharded packs at TP4xPP4 on 16 hosts (when spark1 returns). Budget
per rank at TP4xPP4+MXFP4: 78.4 experts + ~6.2 sharded non-expert +
2-8 embed/head/MTP share + 10-20 KV/GDN-state planner pools ~= 97-113 GiB
vs 119 GB — FITS but the KV planner must be sized to the remaining margin.

## Sharded-pack sprint scope (coordinator request)

(4) CAN THE PACKER ALONE EMIT RANK PACKS THE CURRENT MODULE LOADS? NO —
three independent mechanisms refuse it:
  i.  `SparkQwen38MaxModuleValidateEntry` (module.c:362) compares every
      entry's rows/columns against the FULL-width shape table; any shard
      fails `pack_entry_invalid`.
  ii. The grouped-expert kernels index rank slices OUT of the full buffer:
      `payload = weight_payload + tp_rank * experts_per_rank * payload_stride`
      (cuda.cu:1769-1772, 1832-1834) — rank>0 reads past a sharded
      allocation.
  iii. The generic projection launcher `SparkQwen38MaxLaunchLinear(view,...)`
      (module.c:1379, call sites 1527-1531) has NO tp args: qkv/gate/beta/
      decay/attn q·k·v·o/shared-expert consume FULL-width views, and
      `SparkQwen38MaxModuleVerifyCoverage` (module.c:515-549) requires
      every kind bit per layer — a rank missing "the other ranks' share"
      fails coverage. `SparkQwen38MaxLaunchGdnStep` (module.c:1382) and
      `SparkQwen38MaxLaunchMoeRoute` (module.c:1389) are likewise
      tp-free (GDN core walks all heads; route builds the GLOBAL 512-expert
      prefix table).
Conclusion: module changes are on the critical path; this cannot be a
packer-only fix.

(a) WIRE FORMAT — the 120-byte header is FULL (`<26I2Q>` = 26xu32 + 2xu64,
static-asserted in spark_qwen38_max_stagepack_format.h:123-126). Add
tp_degree + tp_rank (+2 reserved u32) => header 128 bytes, FORMAT_VERSION 2,
bump SPARK_QWEN38_MAX_STAGEPACK_{HEADER_BYTES,FORMAT_VERSION}, extend
ExpectedGeometry + HeaderMatches. Touches: format .h, packer, verifier,
synthesize tool. ~60-80 lines total.

(b) LOADER — shape table gains a tp axis: every Sliced kind scales its
rank-local extent, replicated kinds stay full-width:
  expert-sharded (rows/4): MOE_W1/W3/DOWN (512->128 experts; scale planes
    slice on the same expert-major boundaries — clean).
  head-sharded: GDN_GATE (16384->4096 rows), GDN_BETA/DECAY (128->32),
    GDN_A_LOG/DT_BIAS (128->32), GDN_NORM (128->32 cols),
    ATTN_QUERY (32768->8192 rows, head-contiguous fused query|gate),
    ATTN_KEY/VALUE (1024->256), ATTN_Q/K_NORM ([1,256]->[16,256] composed).
  input-dim sharded (cols/4): GDN_OUTPUT (16384->4096 cols),
    ATTN_OUTPUT (16384->4096 cols).
  composed (non-contiguous in source): GDN_QKV rows = q512|k512|v4096 per
    rank (packer gathers q/k/v independently; conv weight GDN_CONV_WEIGHT
    must be sliced with the SAME channel order, 20480->5120 rows).
  replicated: MOE_GATE (global route), SHARED_* (decide: replicate or
    +all-reduce), EMBEDDING, LM_HEAD (+ 4-bit head shadow), MTP globals.
Functions: SparkQwen38MaxStagePackShape{EveryLayer,Gdn,Attn} gain (tp_degree)
or new tp-aware wrappers; ValidateEntry passes the rank shape; the
natural-format exception must also accept MXFP4_E2M1 for the three expert
kinds (currently BF16-only exception, module.c:366-370). ~150-250 lines
(module.c + format .h).

(c) KERNELS — mixed:
  already rank-aware: AttnPrepare/AttnDecode slice heads/KV by tp_rank
    (cuda.cu 242-278, 415-481) — with sharded views they need LOCAL head
    indexing (drop the tp_rank*local_heads base) while keeping the
    tp_degree=1 path byte-identical (~30-60 lines).
  already parameterized: GroupedExpert Linear/TileLinear compute rank
    offsets from the GLOBAL prefix arrays — with sharded buffers the base
    offset becomes 0 but the route/prefix tables stay global (router is
    replicated) — small (~20-40 lines) + an MXFP4 grouped path (see d).
  needs real work: GdnStep/GdnChunk head loops + state-pool sizing /4
    (~100-200 lines); one additional residual all-reduce if GDN output is
    sharded (the TP collective wiring exists — module.c 1088-1103).
  MXFP4 grouped experts: SparkLmDotRowMxfp4<32> exists
    (model-families/common/include/sparkpipe/spark_lm_kernels.cuh:807-833)
    and the MMA path is proven in inference/kernels/mma.cuh
    (LM_MMA4_MXFP4_GROUP 32, nibble-packed e2m1 + ue8m0/32); the grouped
    expert launchers gate on `weight_format != FP8_E4M3_F32B128`
    (cuda.cu:1756,1814) and their scale-stride math assumes 128-block F32 —
    add the group-32 E8M0 path (~100-150 lines).
Total kernels: ~300-500 lines, one file (spark_qwen38_max_resident_decode_stage_cuda.cu).

(d) MXFP4 EXPERT PACKING — the wire side is already specified in the
qwen38_max format (WEIGHT_FORMAT_MXFP4_E2M1 = 3; payload elements/2,
E8M0 scale plane elements/32 — format .h PayloadBytes/ScaleBytes handle
it today). The packer needs a vectorized requantizer F8_E4M3(block
128x128 scale_inv) -> MXFP4(group 32 E8M0): dequant block -> per-32-group
amax -> e8m0 code -> LUT nibble; pure numpy, ~100-150 lines, I/O-bound
(~2.4 TB pass, ~1-2 h on spark0). QUALITY GATE: this CHANGES the model
(contract precision policy says FP8 "kept AS SHIPPED") — require a
kernel-cosine/decode parity gate vs the FP8 reference before any serving
claim; the family has no validator harness yet, which gates everything.

(e) ESTIMATE — files: spark_qwen38_max_stagepack_format.h,
spark_qwen38_max_resident_decode_stage_module.c,
spark_qwen38_max_resident_decode_stage_cuda.cu,
modules/.../tools/qwen38_max_pack_synthesize.c, tools/qwen38_stagepack.py
(or sibling sharded packer), tools/qwen38_pack_verify.py, NEW
validation/validate_qwen38_max_resident_decode_stage_cuda.sh (port of the
27b harness — REQUIRED: `make publish` fails without GPU_VALIDATOR, and
decode parity is the quality gate for the requant). LOC: format ~70,
loader ~200, kernels ~400, packer ~300, verifier ~100, synth/test ~100,
validator port ~1 day of adaptation: ~1,150-1,600 lines + harness.
Calendar, single agent: packer+verifier 1-1.5 d; module loader/format
1 d; kernels 1.5-2 d; validator port + GPU parity 1-1.5 d => 4-6 days
honest. "1-2 days" only covers the packer side or skips GPU validation
(which the build chain's publish gate forbids). Prereq decision: accept
the MXFP4 requant quality trade (or land the validator first and measure
before committing).

## Node plan (coordinator updates, 2026-08-27)

1. First correction: sparkc/sparkd are contended (GLM+K3+QwenMax) — lane
   node is sparkb ONLY. Applied: nothing was ever parked on c/d; the
   verify-park pipeline was retargeted before any pack landed.
2. Second revision asked for immediate deploy of "~144 GB rank packs"
   (2.3T/16) to spark0-15. NOT EXECUTABLE as stated — see F1/F2: rank
   packs of that shape do not exist (loader is full-width-only) and the
   real 573-606 GiB stage packs cannot be resident on 119 GB nodes.
   Answered with options (a) TP4xPP8/32 ranks — fits, no module change;
   (b) sharded-pack module work; (c) stage-unsupported-packs-on-ack.
   Also: 16 ranks need 16 hosts; spark2=prod and spark1=restarting leave
   14 — two short for the 16-rank plan regardless.
3. Deploy target dir per coordinator: /home/<host>/sparkdata/qwen38max.tp4pp4/packs/.
   (Pipeline v2 parks stages 0-2 on sparkb there after verifier PASS;
   stage 3 stays on the build host spark0 so sparkb stays <=75% disk.
   spark0 locals are deleted after each park size-check.)

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
