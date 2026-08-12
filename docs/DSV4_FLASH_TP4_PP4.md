# DeepSeek V4 Flash TP4 x PP4

This topology runs one DeepSeek V4 Flash model over sixteen Spark nodes.  It
uses tensor parallel width four inside each pipeline stage and four pipeline
stages.  Physical rank is:

```text
world_rank = pp_stage * 4 + tp_rank
```

The pipeline slices are layers 0--10, 11--21, 22--32, and 33--42.  TP4 is
local to each slice.  The four pipeline lanes are:

```text
0 -> 4 -> 8 -> 12
1 -> 5 -> 9 -> 13
2 -> 6 -> 10 -> 14
3 -> 7 -> 11 -> 15
```

No collective spans more than four nodes.  KV pages contain only the local
pipeline slice, but that slice's KV is replicated across its four TP ranks;
TP does not divide the KV byte ledger by four.

## B1024 admission

The serving adapter exposes four in-flight slots, one for each pipeline
microbatch.  At B1024 that is a maximum admitted cohort of 4,096 requests.
This is an admission/topology contract, not a measured concurrency result.

The native SM121 compute route accepts B1, B8, and B1024 exactly.  The B64 and
B256 rows retained below are recovered analytical design points; they do not
make those shapes legal native kernel launches.

## Stage-local graph islands

Production TP decode has no first-request capture and no eager fallback.  Each
pipeline slot owns a sealed array of exactly `2 * local_layer_count + 1`
prewarmed CUDA graph islands: one attention island and one FFN island per
local layer, plus one final post-FFN/head-or-boundary island.  The four PP
stages therefore require 23, 23, 23, and 21 graph entries per slot.  A
microbatch traversing all four stages launches 90 stage-local islands.

The stage JSON pins this as
`"cuda_graph_count_by_pp_stage": [23, 23, 23, 21]`.  Missing, short, or
incorrect counts fail initialization.  Every slot/island is captured and
instantiated before the module reports ready; prewarm records but does not
launch a graph, so it does not mutate KV state.  Runtime decode accepts only
the compiled exact width B1, B8, or B1024 and fails closed if the graph set is
not sealed or replay fails.

The separate 87-island roofline below is intentionally retained as the
recovered full-model TP4 (`2 * 43 + 1`) analytical estimate.  It is not the
aggregate launch count of the PP4 stage-local implementation, which is 90.

## Performance estimates (not measurements)

The following values are recovered analytical estimates.  No sixteen-node
run, CUDA compile receipt, SM121 PTX qualification, latency histogram, or
throughput measurement is present in the recovered archive.

| Batch per PP slot | Admitted requests | Single-request tok/s | Filled-pipeline tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 30.46 | 86.62 |
| 8 | 32 | 17.47 | 464 |
| 64 | 256 | 6.00 | 1,356 |
| 256 | 1,024 | 2.38 | 1,949 |
| 1,024 | 4,096 | 0.60 | 1,962 |

`tools/dsv4_tp4_pp4_perf_estimate.py` preserves this table and labels both
human and JSON output as estimates with `measured: false`.  The model treats KV
as replicated across the four TP ranks of its PP stage, places the singleton
final head on one rank of the final PP stage, retains replication for fixed/FP8
weights unless a tensor is explicitly TP-sharded, and includes host-RDMA memory traffic
at pipeline boundaries.  Single-request throughput is the inverse end-to-end
latency; filled throughput is `B / max(stage time)`.  The two columns therefore
must not be reconstructed by simply multiplying the single-request rate.

### Corrected B1 TP4 decode roofline (separate estimate)

The corrected single-sequence roofline is separate from the older table
above.  It prices 4.312 decimal GB of traffic at 65% of 273 GB/s, or an
effective 177.450 GB/s, giving a 24.300 ms rounded bandwidth term.  Four TP
phases per each of 43 layers at an estimated 50 microseconds per phase add
8.600 ms.  The recovered headline omitted the 0.261 ms estimate for launching
87 prewarmed compute-island graphs at 3 microseconds each; making it explicit
closes the arithmetic:

```text
24.300 ms bandwidth + 8.600 ms collectives + 0.261 ms graph launches
    = 33.161 ms (rounded)
1000 / 33.161 ms = 30.16 raw tokens/s
```

These are analytical roofline values, not measurements.  The recovered table's
B1 value (30.46 tok/s/sequence) is retained for provenance, while 30.16 tok/s
is the corrected raw B1 roofline pending live hardware measurement.  The
arithmetic is executable in `tools/dsv4_tp4_decode_roofline.py`.

## Build the rank-local stage packs

```sh
python3 tools/dsv4_tp4_pp4_stagepacks.py \
  --input-pack /data/dsv4_flash.full.spstage \
  --output-directory /data/dsv4_flash.tp4_pp4
```

Copy each generated `rankNN` pack to the corresponding node as:

```text
packs/dsv4_flash_tp4_pp4_stage.spstage
```

The generated manifest pins each physical rank's PP layer slice, TP rank,
payload byte count, and SHA-256 digest.

Build the serving adapter with:

```sh
make build/libdsv4_tp4_pp4_serving_adapter.so
```

The deployment configuration is
`examples/deployments/dsv4_flash_tp4_pp4_stage.json`; the host-RDMA deployment
spec is `examples/deployments/dsv4_flash_tp4_pp4_host_rdma.spec.json`.  Replace
the example host names and port bases with the installed fleet values before
generating a deployment.

The exact-width B1 correctness release uses
`examples/deployments/dsv4_flash_tp4_pp4_b1_host_rdma.spec.json` and
`examples/release/dsv4_tp4_pp4_b1_template/sparkpipe.json`. Its driver must be
compiled from
`examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json`,
which resolves the validated `.b1.v3` module instead of the unflagged B1024
module. It retains one
physical KV page and 32 logical page-table entries, enough to lazily advance a
single 4,096-position sequence without dense KV preallocation.

The source/host reconstruction prewarms and seals the stage-local graphs, but
CUDA 13 capture/instantiate/replay acceptance on SM121 has not been observed
in this archive.  Live compilation, graph replay against the GA fixture, and
sixteen-node performance qualification remain explicit release gates; this
document makes no measurement or hardware-qualification claim.

Future speculative-decoding experiments are deliberately excluded from this
topology's no-spec B1 qualification.  The public dSpark/MTP observations and
their validation gates are recorded in
`docs/DSPARK_FUTURE_OPTIMIZATION_LEDGER_2026-08-12.md`.
