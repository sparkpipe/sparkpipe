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
| session matrices (topology only) | 23168 + 64·lane, +16 for hc |

The collective identifier is lane-tagged and never zero. Runtime roots
are `/home/<host>/ling-lane<lane>/root`, with KV backing inside the root
and a finite cap. `tests/test_ling_lane.py` renders all sixteen lanes and
checks the following:

- ports and runtime roots follow the table;
- no bound port repeats across lanes;
- the adapter configuration has exactly the adapter's member set;
- the lane-9 configuration equals `tools/ling_multidev_lane.py` outside
  the collective;
- `--check` catches drift;
- the script refuses to run without its settings.

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
