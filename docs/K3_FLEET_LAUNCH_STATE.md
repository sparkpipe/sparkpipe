# K3 TP4xPP4 fleet — launch state (lane/k3-fleet, 2026-08-28/29)

Live copy of this file sits at
`/home/<host>/sparkdata/k3.mxfp4.tp4pp4/LAUNCH-STATE.md` on every staged
rank. Any coordinator/agent can take over mid-bring-up from this file.

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

| rank | host | stage | pack (runtime_root=/home/<host>/sparkdata/k3.mxfp4.tp4pp4) | status 2026-08-29 |
|---|---|---|---|---|
| 0-3 | spark0-3 | 0 | k3.stage0.rank00-03.pack | MISSING (build mid-journal on sparke) |
| 4-7 | spark4-7 | 1 | k3.stage1.rank00-03.pack | BUILT+SLICED on sparke:/home/sparke/k3build (97,651,455,232 B each), NOT deployed |
| 8-11 | spark8-b | 2 | k3.stage2.rank00-03.pack | NOT BUILT |
| 12-15 | sparkc-f | 3 | k3.stage3.rank00-03.pack | DEPLOYED, sha256-verified vs the P2 verifier receipts |

Staged binaries+configs (residentd, libk3_serving_adapter.so,
hidden_transport.so, libnccl.so.2, model_resident.json 16-node kv=0/0,
per-host adapter.json): ranks 12-15 DONE (built from git cb3797a =
lane/k3-fleet, /home/<host>/k3fleet-src on each node). Ranks 0-11: run
the same staging once their packs land (tools/k3_stage_runtime.sh +
tools/k3_gen_deployment.sh + tools/k3_gen_adapter_configs.sh).

## Memory envelope (110 GiB operator ceiling)

Per node: weights 98,119,908,864 B = 91.4 GiB (pack mmap, registered)
+ KDA state pools 17 layers x 16 seq x 6,291,456 B = 1.59 GiB
+ KDA conv windows ~77 MB + MLA KV pool (kv_pages=2) < 1 MB + scratch ~3 MB
+ NCCL/transport buffers + CUDA context ≈ 95-97 GiB TOTAL.
Fits alone with ~13 GiB margin. Does NOT fit with any other model's rank
co-resident (glm5_next rank = 21.7 GiB → ~117 GiB = over the ceiling and
the ~114 GiB NVRM kill line): the K3 window is EXCLUSIVE fleet-wide —
the glm5_next TP16 fleet must be paused (fleet_swap semantics) for the
window and restored after.

kv note: the descriptor carries no JIT_KV, so the deployment's
runtime_limits MUST be kv_logical/physical_page_capacity = 0/0
(runtime/model_serving_adapter.c:174-177 rejects nonzero) — the module
sizes its pools itself; the adapter.json "kv_pages" field is the knob.

## Ports (K3 block)

control endpoint 21480/rank; hidden transport control base 62700 (+rank;
rank 12 listens 62712); NCCL device collective 64620; host tp_collective
61620+rank (XOR-1 partners only); adapter config peer lists are stage-local.

## Bring-up (when all 16 packs are deployed)

From a checkout of lane/k3-fleet (or main once merged):

    bash tools/k3_fleet_wave.sh check    # every rank: binaries, configs,
                                         # pack present, no foreign daemon
    bash tools/k3_fleet_wave.sh launch   # 45s TIME_WAIT sleep, then ALL 16
                                         # ranks in ONE wave, pids captured
                                         # per node in residentd.pid
    bash tools/k3_fleet_wave.sh status   # ready-line poll (15 min deadline)
    bash tools/k3_fleet_wave.sh stop     # TERM own captured pids only

Wave rules (glm53 all-16 lessons): ONE simultaneous wave — staggered
launches die on the 180s hidden-transport connect window; TERM-sweep own
pids first; never `pkill -f`; ready line is `model_residentd ready`.

## Current blockers to the first K3 fleet number

1. Stage-0/1/2 packs (pack lane): stage-1 DONE on sparke (deploy to
   spark4-7 is a copy+sha256); stage-2 not built; stage-0 journal at
   174.5 GB payload (mid layer 11), auto-resume loop NOT armed since the
   sparke reboot — /home/sparke/k3build/probe_and_resume.sh restarts it.
2. Exclusive fleet window: glm5_next TP16 fleet is up on all 16 nodes
   (active debug lane). K3 cannot coexist (memory above). Coordinate the
   pause via the lane reports / coordinator; restore after the window.
3. First number once ready: B1 decode through the fleet, then COMPSEC-17
   through the API if the API path is up (quality gate).
