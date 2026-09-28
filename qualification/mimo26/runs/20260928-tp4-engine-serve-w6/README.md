# MiMo-V2.6-Flash TP4 rank engine and serving, 2026-09-28 (lane mimo, w6)

Hosts spark4, spark6, spark7, spark8 (mesh ranks 4,6,7,8), weightd lane 9 on the
fleet weightd, SPARK_TP_WAIT_MODE=hardware. Repaired v2 rank packs only:
rank0 994f5da7, rank1 2e7b443e, rank2 3fad5b9d, rank3 ac660ea8. Every local
expert is pinned in the weightd pool (3008 experts per rank, 40.3 GB pool,
2.30 GB spine). All numbers are greedy, no speculation.

## Correctness

| check | path | result |
| --- | --- | --- |
| capital / code / science, 16 tokens each | `mimo26_tp_decode`, one row per step, eager | 48/48 equal the CPU reference (`tool/eager-*.log`) |
| capital, 16 tokens | same, graph replay (19 of 20 steps) | 16/16 (`tool/graph-capital.rank0.log`) |
| capital / code / science | `mimo26_tp_decode`, 8-row prefill + graph decode | 48/48 (`rows/graph-rows-*.log`) |
| COMPSEC-17, thinking off, MiMo chat template, max 512 | `mimo26_tp_generate`, one row per step | 17/17 (`tool/compsec17-summary.json`) |
| COMPSEC-17 | `mimo26_tp_generate`, 8-row prefill | 17/17; all 17 token streams byte-equal the one-row run (`rows/`) |
| COMPSEC-17 | `mimo26-api` (rtx5090:8439), prefix reuse off, one-row firmware | 17/17; byte-equal the tool run (`api/`) |
| capital / code / science, 16 tokens | `mimo26-api`, multi-row firmware | 48/48 |
| 4 concurrent requests, 32 tokens | `mimo26-api`, both firmwares | each equals its solo output |

## Performance (no speculation, measured inside perf_window.py)

API, multi-row firmware (`rows/api-perf.json`):

| metric | value |
| --- | --- |
| B1 decode, 46-token prompt, 128 tokens | 22.6 tok/s |
| B1 decode, 46-token prompt, 512 tokens | 21.4 tok/s |
| TTFT, 46-token prompt | 0.71 s |
| TTFT, 278-token prompt | 4.67 s |
| 4 streams x 256 tokens | 51.1 tok/s aggregate |

Earlier firmware, one row per step (`tool/perf-*.log` and the API run at 16:02Z):
B1 23.4 tok/s at 128 and 22.7 tok/s at 512 tokens; TTFT 1.84 s (46 tokens) and
11.4 s (278 tokens); the tool measured 23.2 tok/s graph and 22.3 tok/s eager at
512 tokens, and 35.7 ms (graph) / 38.6 ms (eager) per token at 16 tokens. The w5
validation tool took about 1.8 s per token.

Prompt-prefix reuse is off (`prefix_reuse: false`): the engine keeps its own
per-lane KV and cannot restore a shared prefix.

## Commands

    python3 tools/mimo26_tp_run.py \
      --hosts spark4,spark6,spark7,spark8 --mesh 4,6,7,8 --lane 9 --root '$HOME/sparkdata/mimo26w6' \
      --prompt capital --logs /tmp/x --binary bin/m26_decode_w6 --mode graph-rows
    python3 tools/mimo26_tp_run.py ... \
      --generate compsec/requests.txt --max-new 512 --out compsec/out --binary bin/m26_generate_w6 --mode graph
    python3 tools/mimo26_compsec17.py prepare|serve|grade ...
    MIMO_HOSTS=spark4,spark6,spark7,spark8 MIMO_MESH=4,6,7,8 MIMO_LANE=9 ... tools/mimo26_lane.sh setup|start|api|stop
