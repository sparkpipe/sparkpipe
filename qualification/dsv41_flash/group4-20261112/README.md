# DeepSeek V4.1 Flash on a group of four Sparks: first lane, 2026-11-12

The model's sixteen-rank lane cut its context sixteen ways and held its
203 GB of engram tables in memory. Four Sparks cannot do the second, and the
first costs rounds a group of four does not need. This lane
(`dsv41_flash_tp4`, nodes 0-3, mesh lane 6) keeps the whole context on each
rank and reads engram rows from disk. It is a test lane; nothing is deployed
and no lane record names it.

The measure is a public four-Spark TP4 release of the same model (its
reference build of 2026-11-07, its page read again on 2026-11-12).

| | Reference, 4 Sparks | This lane, first pass | This lane, second pass | Sixteen-rank lane, same day |
| --- | ---: | ---: | ---: | ---: |
| One stream, no drafts, a token | 31.79 ms prose, 34.75 ms code | 32.2, 35.1, 34.5 ms | 32.2, 33.3, 34.2 ms | 21.5-21.8 ms |
| One stream with drafts | 82.6 tok/s | not run | not run | 62-64 tok/s |
| Eight streams, no drafts | | 96-112 tok/s | not run again | not run |
| Eight streams with drafts | 207.4 tok/s | not run | not run | 151-153 tok/s (2026-10-24) |
| Prefill, 8,192 tokens | | 1,742 tok/s | 2,374 tok/s | 2,189 tok/s |
| Prefill, 32,768 tokens | 8,508 tok/s | 1,897 tok/s | 2,651, 2,695 tok/s | 2,373 tok/s |
| Prefill, 131,072 tokens | 8,657 tok/s | 1,888 tok/s | 2,498 tok/s | 2,337 tok/s |
| Memory a node | 71.67 GiB of model, 10.5 GiB of KV | 83 GiB in use, 32 GiB free | 83 GiB in use, 32 GiB free | |

The second pass is the code of `003bd08b2`, measured the same day; its
section is at the end. The sections between describe the first pass.

- The reference's 21.05 ms is its mean step over twenty batch shapes; its
  single stream steps in 31.79 ms on prose and 34.75 ms on code. One stream
  on this lane is level with that. An earlier version of the plan compared
  21.05 ms with our single stream.
- Prefill was at 22% of the reference in the first pass and is at 31% in
  the second; decode with drafts is not wired on this lane. Both are below.
- The sixteen-rank 131,072-token run reused 8,191 cached tokens; its rate is
  over the 122,881 it computed.

## What ran

- Firmware: `sparkpipe_model_residentd` `2f1bcec58481…`, `weightd_warm`
  `38398f7d4ae0…` and the transport library `4f5e3d3f3934…` of the group
  lanes (`qualification/group-sizes-20261112/`); the adapter
  `libdsv41_flash_tp4_serving_adapter.so` built on the build host from the
  code of `179436031`, SHA-256
  `18355e0ca527c548e121c7bc6f6bd96cb74b2a24ce1cbe08a898376974453f9e`. A
  rebuild at that commit (`069ac54d30f1…`) has the same `.text`, `.rodata`
  and `.nv_fatbin` (`decfa631d2e1…`, `e1761d9e7584…`, `9cecaac568ca…`). The
  hub API is the x86 build of the same tree.
- Lane: 8 sequences, waves of at most 4,096 rows, a 262,144-token window
  (1,024 KV pages a sequence, at most 8,192 on the device), no drafter. Ready
  20 s after start. The production model was unloaded through the swap
  controller for the runs and loaded again after them.
- Packs: `tools/dsv41_flash_stagepack.py plan … 4` and `copy` on the hub
  from the publisher's checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
  `1bae0e1088993b9b82de08f59de5be5a34e42170`, 76,180,408,320 bytes a rank;
  the tool's checkpoint verify passes (304,721,245,952 bytes equal the
  checkpoint). Each pack directory holds a zero-range `.experts` manifest.

  | Rank | Node | Pack SHA-256 | Engram shard SHA-256 |
  | ---: | --- | --- | --- |
  | 0 | spark0 | `861a22076dca75c16f1052c086a86a8a8c52ff1fa37e9b5897003f97a3c2bd7e` | `74f093db598fedf63da57bbf150fec2cd98d26aac1a83e7469c11540a6d70b3a` |
  | 1 | spark1 | `1d7065731f01649849ffc4145b35174459026d3e9ac10c24fab6d6820948254b` | `af45b6037cefaeab56e811a66918088946a0e5de6ba38f088722149014bb62ca` |
  | 2 | spark2 | `10e918d2bb3161d52e89dc15556341f406a4cfdda06f47f3f86dc6f6f4afa5fb` | `e6ddeb893ad25553d7b96d533fd008144308365a215d1615ba353dc659878b91` |
  | 3 | spark3 | `26dbc045adf04c6a96fe0c8e392ea81ef540255ee1ee33ab53fb98e42832adba` | `e52585d0c18457cef6a1de7f92d4d8e878c44629b8dfd6df4eafb0c4c07414d4` |

- Engram shards: `tools/dsv41_flash_engram_shard.py --ranks 4 cut` and
  `verify`, 51.0 GB a rank, on disk only. The lane loads each shard's scales
  and small tensors (1.9 GiB) and reads table rows as steps need them.
- Requests: decode is three 256-token continuations of corpus text (two of
  prose, one of LaTeX), speculation off, timed from the stream; prefill is
  one uncached request with one output token; eight streams are eight
  summaries of C source at once, timed over the stretch all eight generate.

## Where a decode step goes

`tools/roofline_meter.py` on the lane's logs: 35.5 ms a token, of which
15.6 ms is the memory floor (3.8 GB a node a token at 243 GB/s), 3.9 ms the
132 rounds (27 to 48 µs each) and 15.8 ms the rank's other work. A CUPTI
trace of rank 0 (`decode-kernels.txt`, taken with the block-mask grid and
before the fold read its keys in batches) splits the rank's work for one
step:

| | ms a step |
| --- | ---: |
| Kernels that read weights (routed experts 6.2, linears 11.6, head 0.9) | 18.7 |
| Rounds (arrival, direct, publish) | 5.6 |
| Sliding-window fold | 4.3 |
| Hyper-connection mixing | 1.6 |
| Index scoring and top-k | 1.3 |
| Everything else | about 3 |

The first lane stepped in 35.0 to 37.7 ms. Its block-mask kernel ran one
block a row over the whole score stride, 2.1 ms a step with a 262,144-token
window; with a grid the step is 33.9 to 36.5 ms (the trace above). The
window fold waited on memory once a key; steps of at most 16 rows now use a
one-pass fold that reads keys in batches (`D41WindowFoldWholeKernel`, the
same bits as the sequential fold by `tests/test_dsv41_window_fold.cu`), and
the step is 32.2 to 35.1 ms. The fold has not been traced again. With a
16,384-token window the step is 3 ms shorter than with 262,144: a captured
step scores the whole window's stride.

## Where a prefill wave goes

A traced 1,024-row wave (`prefill-kernels.txt`, 728 ms under the trace, 21
to 24 layers a wave because rows above the last KV layer are skipped):

| | ms a wave | |
| --- | ---: | --- |
| Routed experts | 157 | every wave reads all 384 experts of a layer once, 1.7 GB a layer a rank: a cost a wave, not a row |
| Sliding-window fold | 100 | 16 heads by 128 keys a row, no tensor cores |
| Rounds | about 150 | 3,532 arrival waits a wave: a round's rows go in about 44 pieces, each waited for |
| Dense linears | 62 | about 40 TFLOPS |
| Hyper-connection mixing and post | 41 | |
| Norms, MoE assembly, merges, the rest | about 70 | |

Waves of 4,096 rows instead of 1,024 spread the expert read over four times
the rows: 2.0 to 2.1 s a wave against 4 × 0.63 s, 1,897 against 1,622 tok/s
at 32,768 tokens. The one-pass fold is slower than the chunked one on large
waves (2.2 to 2.3 s a wave), so large waves keep the chunked fold. A
checkpoint every 16,384 tokens adds about 0.6 s to its wave, and the last
wave of a prompt about 0.7 s for the layers the others skip.

## Eight streams

All eight generating: 96 to 112 tok/s, a step of 71.6 to 83.0 ms; two
streams: 51 to 52.5 tok/s, 38 to 39 ms a step. Each stream routes to its own
six experts a layer, so a step of eight reads up to eight times the expert
bytes of a step of one.

## Against the sixteen-rank lane

Both lanes scored the same 2,112 positions of 66 corpus documents (32 a
document from token 256, `probe.index.json`, tokens SHA-256
`4e74b70c284c…`), one request at a time, no speculation
(`four-vs-sixteen.json`; score files `tp16.corpus.jsonl` `0826bdd32e50…` and
`tp4.corpus.jsonl` `57deda063373…` on the hub).

| | Sixteen ranks | Four ranks | |
| --- | ---: | ---: | --- |
| Next token right | 79.12% | 79.17% | 20 positions only sixteen predicts, 21 only four (p 1.0) |
| NLL of the corpus token | 0.627 nats | +1.17% [+0.40, +1.98] | over the 2,039 positions both list it at |
| Same best token | | 96.64% [95.83, 97.35] | |
| KL lower bound | | 0.0094 [0.0082, 0.0107] | |

| | Positions | NLL, four relative to sixteen |
| --- | ---: | --- |
| Code | 1,280 | +2.01% [+0.34, +3.73] |
| Prose, 2001 | 96 | +0.36% [-0.89, +1.14] |
| Prose, 2026-07 | 736 | +0.69% [-0.22, +1.65] |
| Tokens 256-383 | 528 | +1.92% |
| Tokens 384-511 | 528 | +1.61% |
| Tokens 512-639 | 528 | +1.36% |
| Tokens 640-767 | 528 | -0.21% |

- The four-rank lane predicts the next token as often as the sixteen-rank
  lane and gives the corpus token 1.17% more NLL. That is four times the
  FP8 variant's cost on Qwen 3.8 27B, and it is not explained.
- It is not one broken kernel. The same lane run with 16-row waves
  (`tp4r16.first.jsonl` `c597f2e82a58…`) differs from its own 4,096-row run
  by 1.18% NLL over the first eight positions of every document (0.73%
  against 1.92% above sixteen ranks) and by 2.22% at the first position
  (-0.02% against +2.25%). A layer dump of one 256-token prompt through
  both runs is bit-identical at layer 0 on most rows, and row 0 of layer
  1 differs in one element of 5,120; the residual streams are then 1.2% of
  their norm apart after layer 1, 10% after layer 6 and about 19% from
  layer 12 on, and the routed experts agree on 40 to 60% of the rows from
  layer 6 on. The model quantizes its activations to FP8 between
  projections, so one BF16 unit of difference becomes an eighth-step
  somewhere, and any two orders of the same arithmetic are two samples.
- So what is measured is two samples of a chaotic computation, and whether
  the four-rank large-wave sample is worse on average than the sixteen-rank
  one needs the publisher's code as the arbiter: the same positions through
  `tools/dsv41_official_harness.py` on this checkpoint. The repository's
  reference tokens and T1 fixtures are of an earlier checkpoint. Open.
- A group size is a variant offered with its figures
  (`docs/VARIANT_GATE.md`); these are this one's.

## Open

- Prefill, in the order the trace gives: the rounds (pieces of about 24
  rows, each waited for), the window fold as a matrix product over the
  wave's keys, the hyper-connection kernels, the dense linears, waves above
  4,096 rows (the framework's limit).
- Engram rows are fetched when a step needs them, by 32 reader threads
  behind a cache of 1,048,576 rows a table. A prompt's tokens are known
  before its waves run; nothing fetches ahead yet.
- A captured decode step scores and masks the whole window's stride. On
  each rank that is the whole context, where sixteen ranks each score a
  sixteenth; splitting the scoring by position again, with the KV whole, is
  not built.
- Drafts: no drafter is wired to this lane, so the reference's 82.6 and
  207.4 tok/s have no counterpart here yet.
- The sixteen-rank adapter built from this change keeps 288 of its
  kernels, gains three it does not call and has one changed, the block
  mask (`tools/kernel_codegen_diff.py` against its parent `ab5a93a7b`).
  The model's GPU tests pass on the build host (window fold, landing, MoE
  slices). The sixteen-rank lane has not been run on the fleet from this
  tree.
- The lane has no record, no swap entry and no release row.

## Second pass: prefill

Three changes to the rank's code, measured on the same lane shape (8
sequences, waves of at most 4,096 rows, 1,024 KV pages a sequence) with the
same requests. The adapter that ran is SHA-256
`e4906b24561220bf54a5f0f5fced9b16cf8503abfb2d3416e163928495477e55`; a
rebuild at `003bd08b2` (`55f2afa4b9d2…`) has the same `.text`, `.rodata`
and `.nv_fatbin` (`3dd709ba9f5d…`, `942fc89626c0…`, `357ab45c3623…`). The
other firmware is the first pass's.

| | 8,192 tokens | 32,768 tokens | 131,072 tokens | One stream |
| --- | ---: | ---: | ---: | ---: |
| First pass | 1,742 tok/s | 1,897 | 1,888 | 32.2, 35.1, 34.5 ms |
| Window fold as matrix products | 1,960 | 2,209, 2,352 | | |
| and MoE activations as FP8 codes | 2,053 | 2,323, 2,466 | 2,384 | 32.0, 33.3, 34.0 ms |
| and engram rows fetched ahead | 2,374 | 2,651, 2,695 | 2,498 | 32.2, 33.3, 34.2 ms |

- **Window fold.** A committed wave folds its 128 window keys as two matrix
  products over blocks of 128 rows on the stream GEMM (scores against the
  block's 255 keys, then values), and the merge takes the result as a
  second record source. The first pass's trace had the fold at 100 ms a
  1,024-row wave; a 4,096-row wave spends about 96 ms in it now, 39 of
  them in the merge that reads both sources.
  Steps of at most 16 rows and verify steps keep the sequential fold. The
  wave fold rounds each weight to BF16 against the window's maximum, the
  sequential fold against the running maximum; on the build host
  (`tests/test_dsv41_window_fold.cu`) their attention outputs differ by at
  most one BF16 step and 90% of the outputs are the same bits.
- **MoE activations.** A rank sent every peer its slice of each routed
  expert's activations in BF16, and the receiver quantized them to FP8
  before the down projection. It sends the FP8 codes and one exponent a
  block of 32 instead: the same values (`tests/test_dsv41_flash_moe_slices.cu`
  at four ranks, bit for bit), 3,064 elements a row a peer instead of
  4,736. A row now costs a rank 41 KB a layer on the wire instead of 51.
- **Engram rows.** A wave waited about 135 ms at each of the two engram
  layers for its 98,304 table rows to come off the disk, 16% of the wave
  with the GPU idle. The row ids depend only on the wave's tokens, so they
  are hashed when the wave begins and a thread reads both tables while the
  first layers run. The first table is still about 63 ms short at layer 1
  and the second is ready when its layer comes.

The lane repeats itself: the 2,112 scored positions below came back as the
same bytes (score file SHA-256 `3ca04f8ca886…`) from three starts of the
lane, with the fold alone, with the FP8 codes and with the fetch ahead. So
the last two changes do not move a log-prob.

### Where a 4,096-row wave goes now

A CUPTI trace of rank 0 over six waves of a 32,768-token prompt
(`prefill-kernels.second.txt`; a wave takes 1.55 s, the GPU is busy 88% of
it):

| | ms a wave | |
| --- | ---: | --- |
| Rounds | 386 | 10,950 pieces of 131 KB a wave, five kernels a piece; 192 ms of it waiting for arrivals |
| Routed experts | 267 | 48 calls of 5.5 ms: each reads every expert of a layer whatever the wave holds |
| Dense linears | 181 | |
| Hyper-connection mixing and post | 140 | |
| Window fold with its merge | 96 | |
| MoE assembly (send, assemble, finalize, gather, routes) | 74 | |
| Sparse attention | 55 | |
| Norms | 52 | |
| Index scoring and top-k | 48 | |
| Idle | 190 | 112 ms in gaps under 5 µs between 51,400 kernels, 74 ms waiting for the first engram table's rows |

The API's own step times for 4,096-row waves in the untraced runs: median
1.43 s, a tenth of them under 1.29 s and a tenth over 2.0 s (a checkpoint
every 16,384 tokens, the last wave of a prompt, a longer context).

### Tried and set aside

- Replaying a large round's kernels as a captured CUDA graph, and launching
  the per-piece kernels as dependent launches so each is scheduled while
  its predecessor runs: 2,401 to 2,545 and 2,396 to 2,557 tok/s at 32,768
  tokens against 2,323 to 2,466, about 3% each. The time between a round's
  kernels is not the host's launch cost. Neither is in the code.
- Mesh slots of 1 MiB instead of 256 KiB, so that a round goes in a quarter
  of the pieces: `SPARK_WEIGHTD_MESH_SLOT_ROWS` 32, 41,230 checks of
  `tests/test_weightd_mesh_mock.c` pass with the layout pinned anew. It
  cannot be measured on the fleet from a lane: a lane uses the node's
  released weightd, and the layout is that daemon's. It needs 1.2 GB more
  pinned memory a node and a release. Not in the code.

### Against the sixteen-rank lane, second pass

The same 2,112 positions (`four-vs-sixteen.second.json`; score file
`3ca04f8ca886…` against the sixteen-rank `0826bdd32e50…`):

| | Sixteen ranks | Four ranks, second pass | |
| --- | ---: | ---: | --- |
| Next token right | 79.12% | 79.40% | 14 positions only sixteen predicts, 20 only four (p 0.39) |
| NLL of the corpus token | 0.620 nats | -0.84% [-1.56, -0.11] | over the 2,036 positions both list it at |
| Same best token | | 97.02% [96.16, 97.77] | |
| KL lower bound | | 0.0091 [0.0078, 0.0104] | |

| | Positions | NLL, four relative to sixteen |
| --- | ---: | --- |
| Code | 1,280 | -1.37% [-2.48, -0.20] |
| Prose, 2001 | 96 | -0.66% [-1.68, +0.99] |
| Prose, 2026-07 | 736 | -0.49% [-1.50, +0.49] |

Against the first pass's own scores (`second-vs-first.json`) the corpus
token's NLL is 1.88% lower [-2.70, -1.07], on code 3.25% lower [-4.73,
-1.60]. Only the window fold of committed waves changed between the two
(the other two changes do not move a log-prob), so on this sample weights
rounded against the window's maximum serve the model better than weights
rounded against a running maximum, by more than the two folds differ on
random data. Why is not established. The first pass's 1.17% above sixteen
ranks was this fold; the sixteen-rank lane still folds each rank's eight
keys sequentially. The publisher's code remains the arbiter and has not
been run on these positions.

### Open after the second pass

- The rounds: 386 ms of a 1.55 s wave on the GPU and most of the gaps. A
  piece is half a mesh slot, 131 KB, and costs five kernels; a row sends
  41 KB a layer. Larger slots need a weightd release (above). The
  attention all-reduce could become a gather of the output projection's
  FP8 inputs (6 KB a row a layer instead of 15) if a rank held the whole
  second output projection for waves: a pack change, 1.3 GB a rank.
- Routed experts: 5.5 ms a call for 64 rows an expert in 128-row tiles.
  Waves of 16,384 rows, the reference's size, would fill the tiles; the
  framework's limit is 4,096.
- Dense linears, hyper-connection kernels, the merge's second read, the
  first engram table's last 63 ms, the checkpoint every 16,384 tokens.
- Everything in "Open" above that is not prefill: drafts, eight streams,
  the window's stride in a captured step, the sixteen-rank lane from this
  tree (its adapter keeps 290 of 298 kernels against `179436031`'s: the two
  MoE exchange kernels take two more arguments and four fold kernels are
  added that sixteen ranks do not call), the lane's record.
