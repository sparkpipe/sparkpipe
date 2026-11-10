# Environment variables

What every SparkPipe process reads from its environment, as of 2026-11-10.
Configuration lives in the deployment, the stage configs and command-line
flags; the environment carries a handoff, operator conveniences and DEBUG
diagnostics, nothing that changes how a release build serves.

## The serving path

`sparkpipe_model_residentd` and the stage serving adapter it loads read two
variables, and residentd sets both itself before it loads the adapter:

| Variable | Set by | Meaning |
| --- | --- | --- |
| `SPARK_WEIGHTD_SOCKET` | residentd, from the deployment's `weightd.socket_path` | the node's weightd socket |
| `SPARK_WEIGHTD_PACK_SHA256` | residentd, from the root's one `packs/*.sha256` | the rank pack's digest |

The adapter reads them once at configure and passes the socket to the runner,
the device collective and the KV binding in their configurations. A lane's
mesh lane, rank map, expert pool cap and read-only arena share are stage
config members (`device_collective.mesh_lane`, `device_collective.mesh_ranks`,
`expert_pool_cap_bytes`, `weights_share`; [LANES.md](LANES.md)). The PP stage
boundary transport (`ring/transport/rdma.cu`), loaded only for `HYBRID_TP_PP`
adapters, reads the same socket handoff.

`sparkpipe_model_api` reads nothing from the environment.

## weightd

Every weightd setting is a command-line flag. The variable named beside a flag
applies only when the flag is absent ([WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md)).

| Flag | Variable | Default |
| --- | --- | --- |
| `--socket` | `SPARK_WEIGHTD_SOCKET` | `/tmp/spark_weightd.sock` |
| `--device-bytes-max` | `SPARK_WEIGHTD_DEVICE_BYTES_MAX` | 110 GiB |
| `--kv-reserve-bytes` | `SPARK_WEIGHTD_KV_RESERVE_BYTES` | 0 |
| `--load-pace-bytes-per-second` | `SPARK_WEIGHTD_LOAD_PACE_BYTES_PER_SECOND` | none: a load beside a serving lane is refused |
| `--kv-shared-window-bytes` | `SPARK_WEIGHTD_KV_SHARED_WINDOW_BYTES` | 0 |
| `--mesh-dir` | `SPARK_WEIGHTD_MESH_DIR` | `/tmp/weightd-mesh` |
| `--latch-port` | `SPARK_WEIGHTD_LATCH_PORT` | 61900 |
| `--mesh-doorbell-cpu` | `SPARK_WEIGHTD_MESH_DOORBELL_CPU` | unpinned |

The fleet agent (`tools/fleet_node_agent.sh`) starts weightd with flags only:
the socket, `--device-bytes-max` and `--kv-reserve-bytes` from its
`WEIGHTD_DEVICE_BYTES_MAX` and `WEIGHTD_KV_RESERVE_BYTES` (which
`tests/test_lane_records.py` holds equal to `deployment/lanes.json`'s
`weightd` block) and the mesh identity. It passes none of the node's
environment to weightd, and it refuses a root whose environment files name a
`SPARK_` variable.

## Tools

| Variable | Read by |
| --- | --- |
| `SPARK_MODEL_BATCH_SEQUENTIAL` | `sparkpipe_model_batch`: run the requests one at a time |
| `SPARK_WEIGHTD_SOCKET`, `SPARK_WEIGHTD_PACK_SHA256`, `SPARK_WEIGHTD_IDENTITY_MODEL`, `SPARK_WEIGHTD_IDENTITY_REVISION`, `SPARK_WEIGHTD_ATTACH_LAZY` | `weightdctl` and the eager attach it uses (`SparkWeightdAttachPack`), and the weightd probes |
| `SPARK_MESH_PAIR_INTERFACE` | the fleet agent: the point-to-point verbs device (default `roceP2p1s0f0`) |
| `SPARK_LANES` | `tools/stage_lane.py` and `tools/lane_config.py`: a candidate lane file in place of `deployment/lanes.json` |

## DEBUG builds only

A release build does not read these; their execution shapes (layer dumps
turn graphs off) exist only in `DEBUG` builds.

| Variable | Effect |
| --- | --- |
| `SPARK_DEVICE_LAYER_DUMP` | the device-neutral runner's per-layer dump file |
| `SPARK_DSV41_FLASH_LAYER_DUMP` | DeepSeek V4.1 Flash's per-layer dump file |
| `SPARK_QWEN38_27B_LAYER_DUMP`, `SPARK_QWEN38_MAX_LAYER_DUMP`, `SPARK_QWEN38_FLASH_LAYER_DUMP` | the Qwen hybrid family's per-layer dump file |
| `SPARK_TP_ROUND_TRACE` | one line per device collective round |
| `SPARK_MODEL_BATCH_DECODE_AFTER_PREFILL` | `sparkpipe_model_batch` holds decode until every prefill ends |
| `SPARK_WEIGHTD_MESH_DEVICE_PROBE` | weightd registers each pool with the NIC once to check the address range |

## Retired

Nothing reads these. A node drop-in or script that still sets one sets
nothing; remove the line at the next maintenance window.

`SPARK_TP_WAIT_MODE` (the stage config's `device_collective.wait_mode`),
`SPARK_WEIGHTD_LANE` and `SPARK_TP_MESH_RANKS` (`device_collective.mesh_lane`
and `mesh_ranks`), `SPARK_WEIGHTD_EXPERT_POOL_BYTES` (`expert_pool_cap_bytes`),
`SPARK_WEIGHTD_SHARE` (`weights_share`), `SPARK_WEIGHTD_SPINE_BUDGET_BYTES`
(the runtime derives the spine from the pack), `SPARK_WEIGHTD_ATTACH`,
`SPARK_MODEL_API_LOG` (the API unit's own log redirection),
`SPARK_SHA256_FILE_PIPELINE`, `SPARK_WEIGHTD_KV_WRITE_BUDGET_BYTES_PER_DAY`,
`SPARK_MODEL_ID` and `SPARK_BATCH_INFLIGHT_BUDGET_NS` (the deployment's model
id and `request_inflight_budget_ms`), and the agent drop-in's `G5_*` lines.
