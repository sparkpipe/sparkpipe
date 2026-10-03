# Ling direct development lane

`tools/ling_lane.sh` runs the ling TP16 driver on spark0..sparkf in one
weightd lane, attached to the running weightd, with one transient
`systemd-run --user` unit per node (`sp-ling-rd<lane>`). It replaces the
queue-bound lane-9 wrapper for direct development runs; the queue stays
for exclusive 16-node performance windows.

`tools/ling_lane.py` renders the deployment and the sixteen adapter
configurations. Everything depends on the lane id:

| plane | ports |
| --- | --- |
| residentd control | 23000 + 16·lane + rank |
| TP collective listen and peers | 53000 + 16·lane + rank |
| transport base (topology only) | 64000 + 16·lane |
| session matrices (topology only) | 24256 + 32·lane, +16 for hc |

The template porting checklist puts session blocks at 23168 + 64·lane,
which is disjoint from the control blocks only for lanes 0..9: lane 10's
control block 23160..23175 would overlap lane 0's session block. The ling
lanes therefore keep their session blocks in 24256..24767, above every
checklist block for sixteen lanes.

The collective identifier is lane-tagged and never zero. Runtime roots
are `/home/<host>/ling-lane<lane>/root`, with KV backing inside the root
and a finite cap. `tests/test_ling_lane.py` renders all sixteen lanes and
checks the following:

- ports and runtime roots follow the table;
- no port of any plane (control, collective, transport block, both
  session matrices) repeats across the sixteen lanes, and no ling session
  port falls in a checklist block of any lane;
- the adapter configuration has exactly the adapter's member set;
- the lane-9 configuration equals `tools/ling_multidev_lane.py` outside
  the collective;
- `--check` catches drift;
- the script refuses to run without its settings;
- `setup`, `start` and `stop` exit non-zero and name the host when any
  rank's remote step fails (exercised with a stub `ssh` that fails on one
  host).

Every setting is required and none has a default:

```sh
export LING_LANE=10 LING_CODEC=bf16 LING_WEIGHTD_SOCKET=/tmp/spark_weightd.sock
export LING_FIRMWARE=/path/to/fw-bf16 LING_EXPERT_POOL_BYTES=4294967296 LING_MEMORY_MAX=12G
tools/ling_lane.sh setup
tools/ling_lane.sh start
tools/ling_lane.sh status
export LING_API_HOST=rtx5090 LING_API_PORT=8437 LING_API_UNIT=ling3-api LING_API_BUILD=/home/spec/build-ling/api
export LING_API_TOKENIZER=/home/spec/build-ling/ling_tokenizer.json
tools/ling_lane.sh api
tools/ling_lane.sh decode 678,7706,300,11406,341 16
tools/ling_lane.sh stop
```

`setup` symlinks the rank's placed pack and `.experts` manifest and
writes the single `pack.sha256` sidecar. It then derives the spine
budget from the manifest with `tools/ling_multidev_lane.py
--spine-budget`. `LING_FIRMWARE` holds the aarch64 artifacts
`tools/ling_multidev_build.sh` produces (b128 variant archive, linked
driver, adapter, residentd, transport). `LING_API_BUILD` holds
`sparkpipe_model_api` and the ling adapter built on the API host.

The deployment names the publisher `tokenizer.json` (sha256 40fb9d7d…) so
the API accepts text prompts. Its `vocabulary_size` is 157153, the
tokenizer's highest id plus one, which the API checks; the model head is
padded to 157184. `api` copies `LING_API_TOKENIZER` into the API runtime
root and checks that digest.

Expected first tokens are the publisher-code reference in
`qualification/ling_reference/ling_hf_reference.json`.

## Serving on the fleet (2026-09-28)

The lane serves Ling-3.0-flash at TP16 bf16 beside GLM, one weightd lane,
API on the rtx5090. Receipts are in
`qualification/ling/runs/ling3-tp16-20260928-2a01137/`.

- Greedy tokens against the reference, 3 prompts x 16 tokens: count_up and
  python_fibonacci 16/16; capital_of_france 2/16, then " The" (468) where
  the reference has "\n" (363). The reference's own logits at that step
  are 15.125 and 15.0625, one bf16 ulp apart and inside its 0.5
  run-to-run spread, so this is a tie, not a driver defect.
- The driver applies the publisher serving SwiGLU limits on layers 34-41
  (`spark_ling_swiglu_limits.h`, both contracts carry the lists).
- The adapter is refused at load: the driver keeps no KDA state per cached
  prefix, and prefix reuse is required (I23). The deployment keeps one
  submission in flight (the TP chain keys the single device collective per
  chain).
- Memory per node is about 22 GiB: the weightd arena holds the whole
  15.7 GiB rank pack, and the residentd holds about 6.7 GiB of GPU memory.
  `LING_EXPERT_POOL_BYTES` is passed through but ling does not attach
  lazily.

Measured in the perf window (no speculation, final build c3148b3,
receipts in `qualification/ling/runs/ling3-tp16-20260928-c3148b3/`):

| case | result |
| --- | --- |
| B1 decode, 128 tokens | 54.5 tok/s |
| B1 decode, 512 tokens | 49.1 tok/s |
| TTFT, 298-token prompt | 4.75 s median |
| 8 streams x 128 tokens | 50.6 tok/s aggregate |
| six concurrent reference requests | each equals its solo tokens |
| COMPSEC-17, thinking off | 14/17 |

The earlier 2a01137 run (B1 52.3, 8 streams 52.7) predates the KDA state
index fix: its single-stream numbers stand, but its concurrent outputs
were wrong. Every wave holds one row
(`SparkLingRoundMajorWaveRows`), so 8 streams cost what 8 sequential B1
steps cost; see TECHDEBT. Spin waits beat hardware waits on this lane
(B1 55.3 against 50.6 tok/s,
`qualification/ling/runs/ling3-tp16-20260928-waitmode-ab/`).

`tools/api_serving_perf.py` is the streaming client that produced the
table:

```sh
python3 tools/api_serving_perf.py --endpoint http://127.0.0.1:8437 \
    --cases o128,o512,ttft230 --repeats 3 --label nospec --output perf.jsonl
python3 tools/api_serving_perf.py --endpoint http://127.0.0.1:8437 \
    --cases streams8x128 --repeats 1 --label nospec --output perf.jsonl
```

## Dense GEMM hang repro

`tools/dev/ling_gemm_repro.cu` launches the ling BF16 dense GEMM entry
`LingGemmBf16` over a sweep of shapes and row counts, and reports launches
that do not finish. `modules/ling_resident_decode_stage/source/cuda/unity.cu`
generates `LingGemmBf16` from `SPARK_FAMILY_BARE(GemmBf16)` in
`include/sparkpipe/family/glm/spark_glm_unity_gemm.cuh`. The repro has no
device code. Link it against the ling module archive
(`libling_resident_decode_stage_<codec>.a`), which defines `LingGemmBf16` and
`SparkLingConfigureCudaModule`.

The repro declares its own copy of `LmGemmArguments`, which must stay
layout-identical to the one in `inference/kernels/gemm.cuh`.

The repro takes no arguments. It runs as follows:

- `SparkLingConfigureCudaModule` supplies the SM count. It needs compute
  capability 12.1; if it fails, the repro prints `configure failed` and
  assumes 48.
- Shapes `(input, output)`: 2560x2560, 2560x16320, 4096x2560, 2560x12288,
  6144x2560, 2560x6144. Row counts: 1, 2, 3, 4, 8, 16, 17.
- Each run fills the activation with byte `0x3c` and the weight with `0x38`,
  zeroes the output, and sets one group (`group_row_offset = {0, rows}`,
  `group_tile_prefix = {0, ceil(output / 128)}`). It then calls
  `LingGemmBf16` ungrouped, with `group_count` 1.
- The repro polls `cudaStreamQuery` every 50 ms for up to 8 s and prints
  `rows=R in=I out=O PASS|HANG`. Output values are not checked.
- A run that hangs, is refused by `LingGemmBf16`, or fails a setup step
  (stream, allocation, fill or copy) is counted, and the repro calls
  `cudaDeviceReset()` before the next run. The final `hangs=N` counts all
  such runs, not only hangs.

Exit codes: 0 when every run completes, 1 otherwise.
