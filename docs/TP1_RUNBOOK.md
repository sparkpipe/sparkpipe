# Qwen 3.8 27B — TP1 (single-spark) runbook

Makes the checkpoint-pin -> TP1 run a single command. Companion to the TP1
serving-adapter build-flag switch in
`modules/qwen36_resident_decode_stage/source/spark_qwen36_serving_adapter.c`.
Prep only — no commits, no pushes. Uses the EXISTING qwen36 driver; the 3.8 27B
checkpoint (once the user pins its HF revision) drops into `--checkpoint`.

## 0. The one switch

`SPARK_QWEN36_SERVING_TP_DEGREE` (compile-time, overridable):
- 4 (default) = TP4 whole-stack build — UNCHANGED from the shipped adapter.
- 1 = TP1 single-rank full-width build.
- 0 = legacy PP layer-slice build (not shipped).

Everything downstream (adapter_id, descriptor stage_count, stage_layer_counts,
the env setup, OwnsEmbedding/FinalHead) derives from it. The TP4 default is
byte-for-byte the prior build.

## 1. Pack command (full-width TP1 pack)

`tools/qwen36_stagepack.py` already takes `--tp-degree 1 --tp-rank 0` by
default (= no column sharding, full-width), so no packer change is needed:

```sh
python3 tools/qwen36_stagepack.py \
  --checkpoint /path/to/Qwen/Qwen3.8-27B \
  --output /home/spark3/sparkdata/qwen38.bf16.tp1/packs/qwen38.tp1.qwen36sp \
  --first-layer 0 --layer-count 64
```

The pack header records tp_degree=1 tp_rank=0. The receipt captures
`source_index_sha256` + `source_config_sha256`, which is where the HF
revision belongs (record it there at re-pack time to close the open pin).

## 2. Adapter build command (TP1 variant)

```sh
make build/libqwen36_serving_adapter.so \
  QWEN36_SERVING_ADAPTER_FLAGS="-D_POSIX_C_SOURCE=200809L \
    -DQWEN36_MODEL_REVISION=\"bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1\" \
    -DQWEN36_CONTRACT_SHA256=\"$(sha256sum examples/model_descriptions/qwen36_resident_decode_stage_firmware.json | awk '{print $1}')\" \
    -DSPARK_QWEN36_SERVING_TP_DEGREE=1"
```

Omitting `-DSPARK_QWEN36_SERVING_TP_DEGREE=1` rebuilds the TP4 default. The
adapter id flips to `spark.qwen36.serving-adapter.tp1.v1` and the descriptor
stage_count to 1. The driver (.so) and module archive are unchanged.

## 3. Single-node deployment JSON shape

`node_count` MUST equal the adapter descriptor `stage_count` (=1 for TP1).
`coordinator_rank_index` = 0 and the single node is rank 0 / stage 0.

```json
{
  "schema_version": 2,
  "coordinator_rank_index": 0,
  "adapter": { "shared_object_path": "lib/model_serving_adapter.so" },
  "driver": { "shared_object_path": "lib/model_driver.so", "program_name": "resident_decode" },
  "transport": { "shared_object_path": "lib/hidden_transport.so", "mode": "host-rdma", "control_port_base": 58700 },
  "runtime_limits": {
    "max_inflight_submissions": 2, "max_active_sequences": 64,
    "max_input_rows": 128, "resident_sequence_capacity": 64,
    "kv_logical_page_capacity": 0, "kv_physical_page_capacity": 0
  },
  "nodes": [
    {
      "rank_index": 0, "stage_index": 0,
      "runtime_root": "/home/spark3/sparkdata/qwen38.bf16.tp1",
      "node_target": "cuda.sm121.qwen36.resident_decode_stage.bf16",
      "transport_host": "spark3-fabric",
      "adapter_configuration_path": "config/qwen36_tp1_rank0.json",
      "kv_backing_directory": null, "kv_backing_maximum_bytes": 0,
      "control_endpoint": { "kind": "tcp", "host": "spark3", "port": 17480 }
    }
  ]
}
```

## 4. Rank config (adapter_configuration_path)

```json
{
  "schema_version": 3,
  "model_revision": "bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1",
  "stage_pack_path": "packs/qwen38.tp1.qwen36sp",
  "max_sequence_positions": 8192
}
```

## 5. Launch + A/B (no TP_STANDALONE needed — TP1 has no collective)

residend env (`/etc/sparkpipe/residentd.env`):
```
RUNTIME_ROOT=/home/spark3/sparkdata/qwen38.bf16.tp1
RANK_INDEX=0
MODEL=qwen27b
HOST=spark3
SPARK_QWEN36_SERVING_SPECULATE=1
SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY=recover   # A: recover (default)
# SPARK_QWEN36_SERVING_SPEC_FIRST_DRAFT_POLICY=strict  # B: strict
```

Then: write the `10-user.conf` drop-in (User=<ssh user>), `systemctl
daemon-reload`, `systemctl start sparkpipe_model_residentd`, and run
`sparkpipe_model_batch --deployment config/model_resident.json --runtime-root $PWD
--batch /tmp/b1.json` (B1 batch = 1 request, prompt [0], output budget 32).

Acceptance signal (the landed speculation fix): under `recover` a
`qwen36_spec first_draft_miss` stderr line still yields `tokens_per_sequence > 1`
(chain continues on C0); under `strict` the same miss collapses to 1. The
completion `model_extension` (kind `0x5136`) carries `first_draft_miss_count`
+ `first_draft_policy`.

## Verification

- `clang -fsyntax-only` (both `-DSPARK_QWEN36_SERVING_TP_DEGREE=1` and default) — clean.
- `tests/test_dry_law.py`, `tests/test_code_size.py` (ceiling bumped +15),
  `tests/test_qwen36_stagepack.py` — green in-clone.
