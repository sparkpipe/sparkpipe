# k3_resident_decode_stage

Kimi K3 resident decode stage (rank-pack loader, stage runner and serving
adapter), which the root `Makefile` links into
`build/libk3_serving_adapter.$(SHARED_LIBRARY_EXT)` (TP4 x PP4,
`SPARK_K3_SERVING_TOPOLOGY=404`) and
`build/libk3_tp16_serving_adapter.$(SHARED_LIBRARY_EXT)` (TP16,
`SPARK_K3_SERVING_TOPOLOGY=16`).

## Implementation notes

### Two device collectives, one per reduce width

- The shared device collective (`ring/transport/tp_device_collective.c`)
  sizes an `ALL_REDUCE_SUM_BF16` as `active_sequence_count * row_elements`.
  A submission's `row_elements` may not exceed the collective's
  `local_hidden_dimension`, and 0 selects that width. `ALL_REDUCE_MAX_U64`
  moves one `uint64_t` per row regardless of width. The k3 runner zeroes
  every `SparkTpDeviceCollectiveSubmission`, so each of its BF16 sum reduces
  moves `rows` times the collective width from the submitted pointer. A
  wrong width is a wrong-extent reduce or an out-of-bounds access, not a
  slowdown.
- The runner therefore uses two collectives:
  - Hidden collective (`device_collective`, mesh band 0, width `K3_HIDDEN`):
    the in-place embedding reduce (`K3RunnerReduceBf16`), the per-layer
    phase 0 and phase 1 reduces, and the `ALL_REDUCE_MAX_U64` head argmax.
  - Wide collective (`device_collective_wide`, mesh band 1, width
    `K3_RUNNER_GATE_UP_WIDTH` = `K3_TOP_K * K3_EXPERT_INTERMEDIATE * 2`): the
    phase 2 fused gate_up reduce, whose row extent is not a multiple of
    `K3_HIDDEN`, and the phase 3 reduce of `shared_out_bf16`. The phase 3
    payload is `rows * K3_ROUTED_EXPERT_HIDDEN` elements packed at the start
    of `fused_device`; the reduce covers `rows * K3_RUNNER_GATE_UP_WIDTH`
    elements, which contain it.
  - Each collective has its own ordinal sequence (`tp_next_ordinal`,
    `tp_next_ordinal_wide`).
- Phase 1 on every layer after layer 0 packs two segments into
  `fused_device`: `hidden_bf16` at offset 0 and `shared_out_bf16` at offset
  `rows * K3_HIDDEN`. It issues one stream-ordered reduce per segment on the
  hidden collective, with consecutive ordinals, because a `2 * K3_HIDDEN` row
  cannot ride a `K3_HIDDEN`-wide collective.
- `SparkK3StageRunnerInitialize` fails with `SPARK_STATUS_INVALID_ARGUMENT`
  when `device_collective->local_hidden_dimension` is not `K3_HIDDEN` or
  `device_collective_wide->local_hidden_dimension` is not
  `K3_RUNNER_GATE_UP_WIDTH`.
- `SparkK3StageRunnerConfiguration.device_collective_wide` may be null at
  any TP degree. The serving adapter always sets it when it creates a device
  collective. With `tp_degree > 1`, a device collective, no wide collective
  and no host collective (`tp_collective`), `K3RunnerLayerCollective` skips
  the phase 2 and phase 3 reduces without reporting an error.

### Shared mesh lane (serving adapter)

- weightd gives a mesh lane a single owner connection.
  `SparkWeightdServerOnLaneAcquire` returns `SPARK_STATUS_NO_LANE` when the
  requested lane already has an owner, and `SPARK_STATUS_DUPLICATE` when the
  connection already owns a lane. Two collectives cannot each acquire the
  same lane.
- The adapter connects once to `SPARK_WEIGHTD_SOCKET` (unset:
  `SPARK_STATUS_UNSUPPORTED`; connect failure: `SPARK_STATUS_IO_ERROR`),
  acquires the lane named by the optional `SPARK_WEIGHTD_LANE` with
  `SparkWeightdClientLaneAcquire`, keeps the connection in `lane_client`, and
  passes it as `mesh_lane_client` in both collective configurations. Each
  collective binds its own band of that lane when it is created
  (`SparkWeightdClientLaneBind`). A failed acquire returns
  `SPARK_STATUS_NO_LANE`.
- The hidden collective's `local_hidden_dimension` comes from the adapter
  config's `hidden` value (default `SPARK_K3_MODEL_HIDDEN_DIMENSION`); the
  runner rejects anything other than `K3_HIDDEN`.
- The wide configuration is a copy of the hidden one with its own width
  (`SPARK_K3_MODEL_MOE_TOP_K * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION * 2`),
  `mesh_band_index` 1, `control_port_base` one below the hidden collective's,
  and `collective_identifier` equal to the hidden identifier XOR
  `0x0000800000000000`. It shares the topology and the lane owner.

### Lazy expert attach

- The runner sets `request.identity.arena_bytes` to the rank pack's file
  size. `SparkWeightdLazyPackCreateChecked` (`SPARK_STATUS_IO_ERROR`) and the
  daemon attach (`SparkWeightdServerAttachCold`, logged as
  `WDATTACH size-mismatch`, `SPARK_STATUS_INVALID_ARGUMENT`) both reject an
  identity whose `arena_bytes` differs from the pack file size.
- A lazy startup failure fails initialization; there is no eager fallback.
- `SparkK3ManifestCheck` walks the layers of the rank pack's slice,
  `[first_layer, first_layer + layers)`:
  - layers without expert tensors are skipped (layer 0 is dense;
    `K3_FIRST_ROUTED_LAYER` is 1);
  - `expert_w1_weight` and `expert_w2_weight` must be both present or both
    absent;
  - each tensor's byte size must divide by the expert count. The check is per
    tensor because w1 and w2 have different per-expert sizes;
  - each expert must have one group with exactly two ranges. Range 0 (w1) and
    range 1 (w2) each start at `payload_base + payload_offset + expert *
    per-expert bytes` of their own tensor and are one per-expert size long;
  - the number of groups walked must equal the manifest's `group_count`.
- During initialization the runner walks the same slice again, skips layers
  without expert tensors, and records `layer_w1_offset` and `layer_w2_offset`
  for each routed layer. The bound is the rank pack's `config.layers` (its
  stage slice), not `config.total_layers`. It fails with
  `SPARK_STATUS_PARSE_ERROR` if w1 and w2 presence differ for a layer or the
  slice has no routed layer.

### Stray accounting (diagnostic)

`SPARK_K3_STRAY_WSET` names a head working-set file of raw `uint32_t`
(layer, expert) pairs over the full-model key space. When it is set,
`SparkK3RunnerStrayAccount` checks every (layer, expert) key that
`SparkK3RunnerLazyAcquire` passes to `SparkWeightdMapAcquire` against the
file. Only this rank's PP stage can route, so only that slice of the file is
ever hit. When accounting is armed, `SparkK3StageRunnerDestroy` prints
`K3-STRAY-RECEIPT` (selections, strays, `stray_rate`, unique stray pairs,
head keys) as its first step, before any early return in teardown. If the
bit sets cannot be allocated or the file cannot be opened, accounting is
disabled with a log line. Compare `stray_rate` with the
`head_selection_coverage` that `tools/k3_smoke_experts.py` records in
`model-families/k3/smoke_experts.json`.

### Mesh kernels in the CUDA translation unit

`ring/transport/tp_device_collective.c` calls the mesh launchers
(`SparkTpLaunchMeshCopyDown`, `SparkTpLaunchMeshPublish`,
`SparkTpLaunchMeshWait`, `SparkTpLaunchMeshTree`, `SparkTpLaunchMeshHardware`
and others) as external symbols. They are defined in
`model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh`, so every
shared object that links the device collective must compile that header into
one of its own CUDA translation units.
`spark_k3_resident_decode_stage_cuda.cu` includes it. Serving adapters are
loaded with `dlopen(..., RTLD_NOW | RTLD_LOCAL)`, so a missing launcher fails
the load.
