# STEP 1 probe — 2026-08-25 ~17:45 UTC (lane G-glm52-bringup)

## Hosts (ssh BatchMode, all probes read-only)
| host | up | GPU | avail on ~/sparkdata | residentd | slot-A ports |
|---|---|---|---|---|---|
| spark8 | yes | GB10 | 2089G | none | free |
| spark9 | yes | GB10 | 2068G | none | free |
| sparka | yes | GB10 | 2004G | none | free |
| sparkb | yes | GB10 | 832G | none | free |
| spark1 | yes | GB10 | 1381G | none | free |
| sparkd | yes | GB10 | 2056G | none | free |
| sparke | yes | GB10 | 747G | none | free |
| sparkf | yes | GB10 | 2414G | none | free |
| sparkc | NO — "Connection timed out during banner exchange" | — | — | — | — |

(pgrep hits in raw output were the probe's own bash cmdline; no real
`bin/sparkpipe_model_residentd` anywhere. ss shows nothing listening on
19480-19487 / 63620-63627 / 607xx.)

## Packs (all exactly 102835957760 B)
- spark8 glm52_tp8_rank00.fp8.glms52sp Aug 15 17:22
- spark9 rank01, sparka rank02, sparkb rank03, sparkd rank05,
  sparke rank06 (nlink=2), sparkf rank07 — same timestamp
- **spark1 rank04 dated Aug 25 14:27** — shipped since the bring-up plan was
  written; tree fully seeded (bin/, lib/, config/glm52_stage.json,
  model_resident.json); ~/kvcache/glm52.tp8 created 14:38.

## Config spot-checks
- Every staged config: tp_collective.listen_port=63620 (uniform; field is
  unused by the glm52 backend — bind port = peer_ports[0]+rank per
  ring/transport/rdma.cu control_port_base+sink_rank and the schema check in
  spark_glm52_serving_adapter.c:454-460).
- peers on ALL ranks incl. spark1 = [spark8,spark9,sparka,sparkb,spark1,sparkd,sparke,sparkf], peer_ports 63620..63627.
- spark1: tp_rank=4, stage_pack_path=packs/glm52_tp8_rank04.fp8.glms52sp ✓
- model_resident.json nodes[4]: runtime_root=/home/spark1/sparkdata/glm52.tp8.fp8,
  transport_host=spark1, kv_backing=/home/spark1/kvcache/glm52.tp8 (8GiB),
  control_endpoint spark1:19484 ✓; 8 nodes total; coordinator_rank_index 0.
- bench_b1.json (spark8 only): request_capacity 16, stop_token_ids [154820],
  one request: 11 prompt tokens, output_token_budget **128** ← the O128 gate.

## Pending at close of step
sha256 gate running in background:
- src: sparke:/tmp/rank04_src.sha256
- dst: spark1:/tmp/rank04_dst.sha256
