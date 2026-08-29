# K3 TP4xPP4 fleet — launch state (lane/k3-finish, 2026-08-29/30)

Live copy of this file sits at
`/home/<host>/sparkdata/k3.mxfp4.tp4pp4/LAUNCH-STATE.md` on every staged
rank. Any coordinator/agent can take over mid-bring-up from this file.
Update 2026-08-29 ~22:00 node time (lane/k3-finish): ALL 16 runtime trees
re-staged UNIFORMLY from branch tip; see the status table and "blockers"
for what moved.

## Topology (deployment contract: ALL 16 RANKS, ONE WAVE)

K3 rides the hybrid TP4xPP4 contract. The adapter descriptor
(`modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c`)
hard-requires it and the residentd enforces three independent locks:

1. `stage_count = 16` — `SparkModelResidentDeploymentValidateForAdapter`
   (runtime/model_resident_deployment.c:451) rejects any deployment whose
   `node_count != 16` (TARGET_MISMATCH). A 4-rank slice deployment JSON
   cannot even load.
2. `rank_index == stage_index == linear rank 0..15` — hybrid geometry
   (runtime/pipeline_runtime.c:131-135). PP stage = rank/4, TP rank = rank%4.
3. Pipeline neighbors: rank r's hidden-transport previous rank is r-4
   (node/model_residentd.c:1196-1211) — rank 12 RECEIVES from rank 8
   (spark8). Live receipt (/tmp/k3-residentd-rank12.log, sparkc):
   `hidden_spark_rdma_open_timeout route=rank8_to_rank12_hidden
   role=receiver port=62712 waited_ms=120000` then
   `initialize=busy status=15 phase=transport_open` — a stage-3-only
   fleet never reaches its ready line.

Runner side agrees: `stage_count==4` path binds the PP4 stage tables
(layers 0-23 for stage 0 — a stage-3 pack fails the placement check), and
the derive/PP1 path sets `owns_embedding=1`, which a stage-3 pack cannot
satisfy (the embed tensor lives in stage-0 packs). Decode needs all 93
layers: all 16 ranks, all 4 stage pack sets.

| rank | host | stage | pack (runtime_root=/home/<host>/sparkdata/k3.mxfp4.tp4pp4) | status 2026-08-29 ~22:00 (lane/k3-finish) |
|---|---|---|---|---|
| 0-3 | spark0-3 | 0 | k3.stage0.rank00-03.pack | stage pack COMPLETE + structural PASS (552 tensors, 393,525,084,800 B); slicing on sparke; deploy next (k3_deploy_stage_par.sh 0) |
| 4-7 | spark4-7 | 1 | k3.stage1.rank00-03.pack | stage pack structural PASS; rank packs cross-verify PASS (in flight 21:52+); deploy next (k3_deploy_stage_par.sh 1) |
| 8-11 | spark8-b | 2 | k3.stage2.rank00-03.pack | build RUNNING on sparke under tools/k3_keepalive.sh (journal-resume supervisor; 179/537 tensors at 22:00); then slice+verify+deploy |
| 12-15 | sparkc-f | 3 | k3.stage3.rank00-03.pack | DEPLOYED, sha256-verified vs the P2 verifier receipts |

Stage packs verified before any slice/deploy: `k3_verify_pack.py`
structural PASS on k3_stage_0_24.pack and k3_stage_24_23.pack
(/home/sparke/k3build/verify_stage{0,1}.log); every rank pack is
cross-verified against its stage pack before deploy
(verify_stage{1,0,2}_ranks.log).

Binaries+configs, ALL 16 NODES (re-staged 2026-08-29 ~21:55, uniform):
built from branch tip (lane/k3-finish @ a12af96e = main b993b7d + lane
commits; k3 module moved 4 commits past the cb3797a the old staging used,
incl. the bind required-flags bitwise fix) on sparke
/home/sparke/k3finish-src: residentd 487,080 B, libk3_serving_adapter.so
2,956,200 B, hidden_transport.so (libhidden_transport_spark_host_rdma_verbs)
203,360 B; libnccl.so.2 per node = bf23e731... (fleet-identical).
CONFIG FIX THAT MATTERS: the checked-in
modules/k3_resident_decode_stage/configs/model_resident.json was STALE
(PP-stage-duplicated stage_index {0,0,0,0,1,...} and kv 4096/4096 — both
rejected by the deployment validators). Regenerated from
tools/k3_gen_deployment.sh (unique linear stage_index 0-15, kv 0/0),
pinned by tests/test_k3_deployment_config.py, and now deployed on all 16.
Staging recipe: `bash tools/k3_stage_runtime.sh sparke
/home/sparke/k3finish-src/build` (run on sparke from the checkout).

## Memory envelope (110 GiB operator ceiling)

Per node: weights 98,119,908,864 B = 91.4 GiB (pack mmap, registered)
+ KDA state pools 17 layers x 16 seq x 6,586,368 B = 1.79 GiB
+ KDA conv windows ~77 MB + MLA KV pool (kv_pages=64) 0.45-0.62 GiB
+ scratch ~3 MB + NCCL/transport buffers + CUDA context ≈ 95-97 GiB TOTAL.
Fits alone with ~13 GiB margin. Does NOT fit with any other model's rank
co-resident (glm5_next rank = 21.7 GiB → ~117 GiB = over the ceiling and
the ~114 GiB NVRM kill line): the K3 window is EXCLUSIVE fleet-wide —
the glm5_next TP16 fleet must be paused (fleet_swap semantics) for the
window and restored after.

MLA KV pool arithmetic (kv_pages=64, generator k3_gen_adapter_configs.sh):
page = 64 slots x 1152 B (MLA latent row, kv_lora_rank 512 + unrotated 64,
BF16) = 73,728 B = 64 positions; 64 pages = 4,096 positions/sequence.
Device cost = mla_count x kv_pages x 73,728 B x 16 sequences, with stage
MLA counts 6/5/6/7 → 0.45 GiB (stage-1) to 0.62 GiB (stage-3) per rank.
Covers the quality fixtures 5x (max 769 tokens). The smoke value
kv_pages=2 held only 128 positions and overflowed on every real prompt.
The 224K-token fixtures need kv_pages=3500 ≈ 25 GiB/rank at 16 sequences
— re-plan memory (or drop sequence occupancy) before that step.

kv note: the descriptor carries no JIT_KV, so the deployment's
runtime_limits MUST be kv_logical/physical_page_capacity = 0/0
(runtime/model_serving_adapter.c:174-177 rejects nonzero) — the module
sizes its pools itself; the adapter.json "kv_pages" field is the knob.

## Ports (K3 block)

control endpoint 21480/rank; hidden transport control base 62700 (+rank;
rank 12 listens 62712); NCCL device collective 64620; host tp_collective
61620+rank (XOR-1 partners only); adapter config peer lists are stage-local.

## Bring-up (when all 16 packs are deployed)

From a checkout of lane/k3-finish (or main once merged). The check now
ALSO enforces the memory envelope (refuses any node with <100 GiB
available; one rank needs ~96-97 GiB of the 110 GiB ceiling):

    bash tools/k3_fleet_wave.sh check    # every rank: binaries, configs,
                                         # pack present, no foreign daemon,
                                         # >=100G available
    bash tools/k3_fleet_wave.sh launch   # 45s TIME_WAIT sleep, then ALL 16
                                         # ranks in ONE wave, pids captured
                                         # per node in residentd.pid
    bash tools/k3_fleet_wave.sh status   # ready-line poll (15 min deadline)
    bash tools/k3_fleet_wave.sh stop     # TERM own captured pids only

Wave rules (glm53 all-16 lessons): ONE simultaneous wave — staggered
launches die on the 180s hidden-transport connect window; TERM-sweep own
pids first; never `pkill -f`; ready line is `model_residentd ready`.
Reserve the window in the queue first:
`python3 tools/spark_queue.py reserve --node sparkX --holder lane-k3-finish`
for all 16 (the glm5_next lane's reservations must be released first —
one wave owner).

## Current blockers to the first K3 fleet number

1. Stage packs (lane/k3-finish, 2026-08-30): ALL FOUR STAGES DEPLOYED.
   stage-0 (spark0-3, k3_stage_0_24.deploys/ 00:41), stage-1 (spark4-7,
   k3_stage_24_23.deploys/ 22:11), stage-2 (spark8-b,
   k3_stage_47_23.deploys/ 03:4x — receipts:
   rank00 c2ee5a33d89f72f9bd75e3b1845d4b238dc7a4e1b7bac9effb8e5027ffb4a514,
   rank01 8ce3ba3b758c83a25c0b22bd65d4a029a587905aab650326a84a7c231d70ad4f,
   rank02 25e376c06d37b8611b41908fb078f4f5fc39baf6df1e64ed8448ed9111e85007,
   rank03 fe455ac7db5dbd1140efa6752de6241055a9da2a882c2b5a6d1f8f8738b1f157),
   stage-3 (sparkc-f, from the P2 verifier receipts). Cross-verified
   before deploy: all four 47_23 rank packs PASS vs the stage pack
   (534 tensors, 23 layers, verify_stage2_ranks.log, CHAIN complete
   03:33+). The build can no longer die silently:
   /home/sparke/k3build/keepalive.sh (committed as tools/k3_keepalive.sh)
   restarts the journal-resuming packer if it dies and logs every restart
   to keepalive.log; it exits DONE when the pack completes. If sparke
   rebooted: re-run
   `cd /home/sparke/k3build && nohup bash keepalive.sh 47 23 /home/sparke/k3build/k3_stage_47_23.pack >> keepalive.log 2>&1 &`.
   STAGE TABLE WARNING (the 48_23 defect, caught 2026-08-30): the runner's
   PP4 stage table is {first: 0,24,47,70} x {count: 24,23,23,23}
   (spark_k3_resident_decode_stage_runner.cu K3RunnerFirstLayer/
   K3RunnerLayerCount). Stage-2 is 47_23, NOT 48_23 — a 48_23 pack fails
   SparkK3ModuleInitialize's first_layer check on ranks 8-11 and would
   have silently dropped layer 47 while duplicating 70. The 47_23 pack
   (388,835,087,872 B, keepalive DONE 00:46:43 node time) replaced it;
   its four rank packs sliced and cross-verified (534 tensors, 23 layers,
   verify_stage2_ranks.log). The wrong 48_23 pack is left on sparke
   (~389 G reclaimable — coordinator's call).
2. Exclusive fleet window: glm5_next TP16 fleet holds all 16 nodes;
   glm5-dsa is the wave owner for glm5_next (the lane-glm5attractor x16
   hold transferred to them). K3 cannot coexist (memory above). Windows
   are taken ONLY via the queue's 90-min rolling reservations — when the
   glm5 lease lapses AND no residentd of any family is live
   (`k3_fleet_wave.sh check` is the census): reserve all 16 for
   lane-k3-finish, then `check` (must be fully green incl. memory), then
   `launch`. NOTE: spark2/spark3 are the active-MDS hosts per the storage
   rule — check `ceph fs status` before lighting; if the MacStudio MDS
   relocation has not landed, the operator owes an envelope exception for
   those two ranks (the 96-97 GiB envelope itself leaves ~13 GiB, the
   rule exists because MDS+OSD share the same unified memory).
3. First number once ready: B1 decode through the fleet, then COMPSEC-17
   through the live /v1 endpoint (quality gate; fixtures for glm5.3-flash
   are pre-tokenized at 93a3a0b — the K3 tokenizer one-shot needs running
   on spark1 per the same pattern before COMPSEC can fire). The /v1 front
   is STAGED: bin/sparkpipe_model_api on spark0 (built from lane tip,
   774,216 B, 2026-08-30 03:40); start after FLEET READY:
   `cd /home/spark0/sparkdata/k3.mxfp4.tp4pp4 && nohup
   ./bin/sparkpipe_model_api --deployment config/model_resident.json
   --runtime-root $PWD --port 8433 > api.log 2>&1 < /dev/null &`.
   K3 quality fixtures are pre-tokenized in-tree:
   qualification/ds4_eval/quality-fixtures-kimi-k3.json (92 cases, 17
   COMPSEC, max 769 tokens — all fit kv_pages=64 with 4.5x room).

## TP4 equivalence across a real stage (staged, runs in the window)

The full 93-layer stage-0 pack cannot register on any 119 GiB node
(cudaHostRegister OOM at 103 GB — session-2 receipt), so the equivalence
runs on the k3_stage_0_4.slice (first=0 count=4, 55,337,455,488 B full +
four 13,890,672,768 B rank packs, staged on sparke 02:40-02:52): five
single-spark tp1 steps of one token, full[k] vs Σ rank[k] via
tests/test_k3_runner_step.cu --dump, checked by
tools/k3_tp4_equivalence_check.py. GOTCHA (live-verified): the slice pack
must run with --pp1 — the PP4 stage tables VALIDATION_FAIL a 0_4 pack
(INIT FAIL 9); --pp1 takes the slice bounds from the pack manifest.
Run INSIDE the exclusive window while the K3 fleet is down (the full
slice registers ~52 GiB on sparke).
