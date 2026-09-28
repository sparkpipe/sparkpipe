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
- The deployment sets `prefix_reuse` false (the driver keeps no KDA state
  per cached prefix) and one submission in flight (the TP chain keys the
  single device collective per chain).
- Memory per node is about 22 GiB: the weightd arena holds the whole
  15.7 GiB rank pack, and the residentd holds about 6.7 GiB of GPU memory.
  `LING_EXPERT_POOL_BYTES` is passed through but ling does not attach
  lazily.

Measured in the perf window (no speculation, firmware 8cc64a4):

| case | result |
| --- | --- |
| B1 decode, 128 tokens | 52.3 tok/s (18.6-22.3 ms/token) |
| B1 decode, 512 tokens | 43.1 tok/s (about 23 ms/token) |
| TTFT, 298-token prompt | 5.9 s median (5.1-8.2 s) |
| 8 streams x 128 tokens | 52.7 tok/s aggregate, 63.3 steady |
| COMPSEC-17, thinking off | 14/17 |

`tools/api_serving_perf.py` is the streaming client that produced the
table:

```sh
python3 tools/api_serving_perf.py --endpoint http://127.0.0.1:8437 \
    --cases o128,o512,ttft230 --repeats 3 --label nospec --output perf.jsonl
python3 tools/api_serving_perf.py --endpoint http://127.0.0.1:8437 \
    --cases streams8x128 --repeats 1 --label nospec --output perf.jsonl
```
