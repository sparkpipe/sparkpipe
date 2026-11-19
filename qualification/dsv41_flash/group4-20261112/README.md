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
  released weightd, and the layout is that daemon's. It needs 1.15 GiB more
  pinned memory a node and a release. Not in this lane's code; it is its own
  change now, measured on two machines outside the fleet (third pass below).

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

## Third pass: where the rounds wait, 2026-11-13

### The second pass's trace, read by wave

Only the lower 20 layers run on every row of a wave. The upper 20 run in
the tail step that follows every 16,384 tokens, on 128 rows fewer a layer.
The table above is 1.55 s of the trace: a wave and its share of that tail.
A full 4,096-row wave alone takes 1,409 ms in the same trace (three waves,
46,909 kernels each; `prefill-rounds.third.txt` holds this reading and the
third pass's):

| | ms a wave | |
| --- | ---: | --- |
| Rounds | 450 | 69 runs of round kernels, 10,045 pieces of 131 KB a peer: 172 ms in arrival waits, 178 ms in the other round kernels, 99 ms between them |
| Routed experts | 236 | 40 calls of 5.9 ms |
| Dense linears | 196 | |
| Hyper-connection mixing and post | 126 | |
| KV write | 51 | 18 calls |
| Norms | 48 | |
| Sparse attention scoring | 46 | |
| Window merge | 36 | |
| Between kernels outside the rounds | 61 | |

What the rounds wait for, by the stage that ends before each run:

| stage before the run | runs a wave | ms a run | first piece waits | pieces |
| --- | ---: | ---: | ---: | ---: |
| MoE send (the all-to-all) | 20 | 7.0 | 0.18 ms | 192 |
| Output projection (the attention sum's two halves) | 22 | 6.3 | 0.28 ms | 181 |
| N-gram gather (the two table layers) | 2 | 50.5 | 44.2 ms | 178 |
| MoE finalize (the hidden-slice gather) | 20 | 2.9 | 0.24 ms | 80 |
| Index pack | 3 | 2.2 | 0.20 ms | 65 |

After each of the two table layers rank 0 has its rows and waits 44 ms for
the other ranks to read theirs: 88 ms a wave, on top of the 74 ms it waits
for its own first table. No other stage keeps a rank waiting for another.

### What a rank sends, and over which port

Byte counters of both 100G ports of ranks 0 and 1 around a 33,792-token
prefill: a rank sends 36.6 GB, 1.08 MB a token, 24.4 GB of it on one port
and 12.2 GB on the other. weightd puts a whole peer on one rail (a link takes
the second rail when the two ranks' parities differ), so in a group of four
two of a rank's three peers share a port.

`build/mesh_round_bench` on the same four Sparks under the released weightd
(256 KiB slots), rounds only, microseconds a round:

| exchange of a 4,096-row wave | a round | the rank waits for peers 1, 2, 3 |
| --- | ---: | --- |
| reduce-scatter, 5,120 wide | 1,998 | 547, 100, 536 |
| all-gather, 1,280 wide | 2,047 | 598, 153, 591 |
| all-to-all, 3,064 a peer | 4,879 | 1,057, 179, 1,048 |

The port with two peers runs at 10.2 to 10.5 GB/s in all three; peer 2,
alone on the other port, is done five times sooner. A layer's four rounds
add up to 11 ms in the bench, 220 ms a wave of 20 layers, against the
lane's 450 ms: the lane launches its round kernels one by one where the
bench replays a graph, and it waits for the table reads.

The 1 MiB slots, measured on two GB10 machines outside the fleet with a
private weightd each and one port (a group of two): reduce-scatter 2,404 to
1,805 microseconds, all-gather 1,214 to 933, all-to-all 2,978 to 2,137, a
decode-size round 12.0 either way; the port goes from 8.4-8.7 to 11.2-11.8
GB/s. On four Sparks the busier port is already at 10.2-10.5 GB/s, so
there the larger slot is bound by that port until each peer's traffic is
split over both.

### Table reads

A rank reads about a quarter of a wave's 98,304 rows a table, the columns
of its own shard. A reader thread took the row file's lock twice for every
row it read, some 49,000 times a table over 32 threads. A reader now takes
up to 64 rows a lock and the tables are read by 64 threads; a fetch smaller
than the reader count wakes one reader a row, as a decode step needs
(`runtime/row_file.c`, `tests/test_row_file.c`).

Each from a cold start of the lane, the same prompts:

| | 32,768 tokens | 65,536 tokens |
| --- | ---: | ---: |
| Second pass | 2,650 tok/s | 2,686 tok/s |
| Batches and 64 readers | 2,855 tok/s | 2,801 tok/s |

The three usual prompts after a cold start, and one stream, with the final
adapter (SHA-256 `23de7e9fac51fc048627a63853ee3ecd11fa1ecc83ad9fb51ed73a71f4a52c9f`, commit `880d3dd0a`; the other firmware is the first
pass's):

| | 8,192 tokens | 32,768 tokens | 131,072 tokens | One stream |
| --- | ---: | ---: | ---: | ---: |
| Second pass | 2,374 tok/s | 2,651, 2,695 | 2,498 | 32.2, 33.3, 34.2 ms |
| Third pass | 2,600 tok/s | 2,908 | 2,519 | 32.2 to 34.4 ms over two runs of the three prompts |

The 131,072-token prompt gains 0.8%, inside the spread of that prompt
between runs; why it gains less than the shorter prompts is not
established.

A trace of the final adapter (three full waves of a 32,768-token prompt):
a wave takes 1,294 ms where it took 1,409. The rounds are 381 ms (450), of
them 96 ms in arrival waits (172). The run after each table layer waits
3.9 ms for the other ranks (44). Between kernels the GPU is idle 111 ms a
wave (160).

The rows read are the same rows: with batches and 64 readers the 2,112
scored positions come back as the same bytes as the second pass's (score
file SHA-256 `3ca04f8ca886…`). The final adapter differs from the scored
one only in how many readers a small fetch wakes.

### Open after the third pass

- The rounds are 381 ms of the 1,294 ms wave: a piece takes 37.9
  microseconds, 18.2 in its kernels, 10.1 between them and 9.6 waiting,
  where the busier port needs 22 to 25 for its bytes. 1 MiB mesh slots are
  their own change and need a weightd release; splitting each peer over
  both rails is a weightd change after it; the attention sum as a gather of
  FP8 inputs is the pack change above.
- Routed experts, dense linears and hyper-connection mixing are 558 ms of
  the wave and untouched by any of this.

## Fourth pass: larger waves, 2026-11-13

The framework held a wave to 4,096 rows (`SPARK_STAGE_SERVING_MAX_ROWS`).
The limit is 16,384 now and a lane chooses its wave size (`STAGE_ROWS`).
Two of this module's kernels launched one grid row a route, 114,688 of them
at 16,384 rows where a launch allows 65,535; their routes now span two axes
of the grid (`D41ItemGrid`), with the same block order and the same bits
(`tests/test_dsv41_flash_moe_slices.cu` runs a 4,800-row wave at four
ranks).

One build for all three sizes (adapter SHA-256
`0bbf49986749d6e7585cc5b60abd2ddefae5419402e11627de3f88071c9b889b`, built
from the sources of commit `c3237d996`; a rebuild at that commit,
`4ad02f6030e7…`, has the same `.text`, `.rodata` and `.nv_fatbin`:
`9eb83a08edd3…`, `24f92de6714b…`, `fbf539d509db…`), a cold start of the
lane for each, the same requests:

| Waves of | 8,192 tokens | 32,768 tokens | 131,072 tokens | Free memory a node |
| --- | ---: | ---: | ---: | ---: |
| 4,096 rows | 2,584 tok/s | 2,900 | 2,527 | 31.5 GiB |
| 8,192 rows | 2,650 tok/s | 3,011 | 2,613 | 28.6 GiB |
| 16,384 rows | 2,653 tok/s | 3,117 | 2,693 | 23.0 GiB |

One stream decoding 400 tokens while another request prefills 65,536:

| Waves of | The prefill takes | Stalls of the stream over 1 s | Its longest | The stream takes |
| --- | ---: | ---: | ---: | ---: |
| 4,096 rows | 28.5 s | 16 | 2.5 s | 37.1 s |
| 8,192 rows | 25.7 s | 8 | 3.8 s | 36.0 s |
| 16,384 rows | 23.6 s | 4 | 6.3 s | 35.0 s |

- One stream alone is the same: 32.2, 34.2 and 34.5 ms a token with
  16,384-row waves, 32.2, 33.7 and 34.9 with 4,096.
- The answer to a 40,000-token prompt, 16 tokens with the five likeliest at
  each, is the same bytes at all three sizes (SHA-256 `b79b025295d2819f…`).
- The gain is the routed experts. A call reads every expert's weights
  whatever the wave holds, 3.3 ms, so 16,384 rows cost 13.7 ms a call where
  four waves of 4,096 cost four times 5.4: 681 ms against 1,066 ms a 16,384
  rows (`prefill-waves.fourth.txt`). Rounds, dense linears and mixing cost
  the same a row.
- The price is memory, 8.5 GiB a node at 16,384 rows for the wave's
  buffers, and the wait of a stream that decodes beside a prefill: a decode
  step waits for the wave in front of it, so the stream stalls four times
  for up to 6.3 s where it stalled sixteen times for up to 2.5 s.

Prefill is at 31% to 37% of the reference with 16,384-row waves.

### The checkpoint stride

Every 16,384 tokens the upper 20 layers run on the last rows, 2,541 at
layer 20 down to 128 at the top, so that the recurrent state can be saved
there (the module's `cache_checkpoint_token_count`). That tail takes 490 ms,
9% of a 16,384-row wave's cycle, 132 ms of it the experts' weights read for
a few rows. A 131,072-token prompt runs eight tails and needs the last. A
stride of 65,536 would save three in four; a later request that shares a
prefix would then resume up to 65,535 tokens before its first new token
instead of up to 16,383. Not changed.

## Fifth pass: two kernels a piece, 2026-11-13

A pipelined round (a gather, a reduce-scatter or an all-to-all of three or
more half slots) launched four or five kernels for each 131,072-byte piece:
a ship gate, a stage or a copy, a publish, an arrival wait and a combine,
about 190,000 kernels a 16,384-row wave. It now launches two: a one-warp
kernel that waits for the previous piece to ship and for the peers' piece to
land, and a wide kernel that stages the next piece, combines the landed one
and rings the doorbell. A first version did all of it in one wide kernel
whose blocks each waited; with sixteen ranks on one GPU, as in the probe,
the waiting blocks left no room for the other ranks' kernels, so the wait
stays in its own small kernel.

Checks on the build host: the GPU probe's ten modes (733 cases, sixteen
ranks included), `test_tp_allreduce_fuzz`, `test_tp_mesh_pipeline_boundaries`
and `test_tp_device_collective_mock`. On the lane the answer to the
40,000-token prompt is the same bytes as before (`b79b025295d2819f…`).

Same lane, same prompts, 16,384-row waves:

| | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| Fourth pass | 2,653 tok/s | 3,117 | 2,693 |
| Two kernels a piece | 2,725 tok/s | 3,221 | 2,772 |

That is 3%, where the kernel counts suggested 10%: a wave's first 16,384
rows took 4.80 to 4.83 s. A trace of the new build
(`prefill-rounds.fifth.txt`) has a piece of the large rounds at 30.7 us
where it was 37.9, and 13.8 us of that is the wait: the rounds moved from
the GPU's pace to the daemon's, which posts and completes three
131,072-byte writes a piece, two of them on one port. Rounds are still
1.47 s of a traced 5.55 s wave: the sum of the attention output 0.64 s
(0.35 s through the staged path, a whole slot a write, and 0.29 s through
the split sum), the experts' all-to-all 0.50 s, the experts' gather 0.21 s.
The 1 MiB slots (PR 1932,
weightd IPC ABI 15) cut the writes to a quarter and measured 23 to 28% a
round on the Lenovo pair; they need the daemon replaced on the nodes. After
the rounds, the same trace has the expert and dense multiplies at 1.47 s,
the sparse scoring and its top-k at 0.83 s and the stream mixing at 0.52 s.

## Sixth pass: row-wise work once per row, 2026-11-13

At four ranks every rank held every row and repeated all the work that
depends on one row alone: the stream mixing, the collapse and norm before
attention and before the experts, the stream update after them, the engram
gate, the index scoring with its top-k, and the q and kv projections. Four
ranks did the same arithmetic on the same 16,384 rows.

In a committed wave of 2,048 rows or more each rank now owns a quarter of
the rows and does that work for its own rows only:

- Stream mixing, collapse and norm, stream update and the engram gate run
  on the owner. The owner gathers the normed row to the group where a layer
  needs it on every rank (before the experts, and before attention on the
  compressor and index layers) and every rank quantizes the gathered rows
  itself.
- On the other layers the owner also projects q and kv and gathers the
  normed q latent and the window kv, 1,792 values a row instead of 4,096.
- The index scoring and top-k run on the owner, which gathers the 512
  selected positions a row.
- The sum of the attention output is a reduce-scatter that lands on the
  owner only, and each rank returns its slice of the expert output to the
  owner only.
- Layers 20 to 39 only ever run on a wave's last 2,541 rows or fewer, so the
  split covers layers 0 to 19 and the streams go back to every rank at layer
  20.
- A prompt of N tokens prefills N-1 rows, so a prompt's last wave is never
  a multiple of 32. The stage rounds the wave up to a multiple of 32 inside:
  the pad rows start as zeros, stay zeros, are never selected, and only the
  real rows leave the stage.

The answer to the 40,000-token prompt is the same bytes as before
(`b79b025295d2819f…`) at every step, on all three node groups.

Same prompts, 16,384-row waves, 1 MiB mesh slots (main of 2026-11-13 on the
released weightd), a 14 GiB state budget because the rank state grows to
13.5 GB:

| | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| Fifth pass on 131,072-byte slots | 2,725 tok/s | 3,221 | 2,772 |
| The same code on 1 MiB slots | 2,869 tok/s | 3,403 | 2,915 |
| Row split, first step | 2,899 tok/s | 3,608 | 3,482 |
| With own-row q and kv projections | 2,899 tok/s | 3,689 | 3,606 |
| With padded waves, nodes 0-3 | 3,321 tok/s | 4,115 | 3,834 |
| The same build, nodes 4-7 | 3,269 tok/s | 4,079 | 3,817 |
| The same build, nodes 8-b | 3,308 tok/s | 4,125 | 3,845 |

The gain grows with the context because the index scoring it no longer
repeats grows with the index. 8,192-row waves measure within 2% of
16,384-row waves (2,865 / 3,579 / 3,411 against 2,894 / 3,576 / 3,475 on
the first step), so the wave size is not a lever.

A traced 16,384-row wave is now 4.05 s where it was 5.55 s
(`prefill-rounds.sixth.txt`, rank 0, the second and third waves of a
65,536-token prompt):

- Rounds 1.13 s: the experts' all-to-all 395 ms, the attention
  reduce-scatter 235 ms, the gathers of the normed row 170 ms, the engram's
  rounds 88 ms, the gathers of q latent and kv 59 ms, the expert return
  57 ms, the hand-back of the streams 29 ms. A piece of a large round now
  carries about 2.5 MB out of a rank in 93 us, which is the two ports' rate,
  so what is left here is the bytes: the experts' all-to-all sends every
  route's activation slice and the shared expert's partial to every rank.
- Multiplies 1.43 s: the experts 0.68 s and the dense layers 0.75 s, at 40
  to 50 TFLOPS with the BF16 tensor-core form. The native MXFP8 and MXFP4
  forms are wrapped in `inference/kernels/mma.cuh` and no kernel uses them.
- The rest 1.5 s, the largest being the attention over the selected blocks
  194 ms, the merge of window and block attention 154 ms, the expert send
  packing 119 ms, quantize passes 98 ms, stream mixing 90 ms, sparse scoring
  84 ms, stream update 84 ms, activation assembly 83 ms, top-k 74 ms.

Seen once and not again: on the first run of the padded build, before the
pad rows were zeroed, rank 3's first decode graph capture failed
(`MESH-LAUNCH-FAIL cuda=901`) right after a padded first wave. Seven later
runs in the same order, four of them with zeroed pad rows, captured without
a fault.

## Seventh pass: the reference on our nodes, stage by stage, 2026-11-13

### The reference on nodes 0-3

The reference (build 1107, stock 4 KiB kernel, the same four Sparks and
switches as this lane) prefills cold prompts from the same corpus at:

| Prompt | tok/s |
| --- | ---: |
| 8,192 tokens | 7,410 and 7,637 |
| 16,384 tokens | 7,770 and 8,094 |
| 32,768 tokens | 5,807 on the first prompt of that length, then 8,467 |
| 65,536 tokens | 6,924, 8,490 and 8,417; 8,576 with every rank traced |
| 131,072 tokens | 8,407 |

With our kernel tracer loaded into its four containers, one 16,384-token step
(the second of a 65,536-token prompt) takes 1,816 ms on rank 0 and the GPU is
busy for 99.9% of it. A middle step runs only layers 0 to 19: its router runs
20 times. Layers 20 to 39 run in the last step only.

### Stage by stage

The same rows of the same prompt (positions 16,384 to 32,767 of a
65,536-token prompt), rank 0, ms per 16,384-token step:

| Stage | Ours, sixth pass | Ours, native MXFP8 | Reference |
| --- | ---: | ---: | ---: |
| Whole step | 4,003 | 3,828 | 1,816 |
| Rounds (all collective kernels) | 1,077 | 1,135 | 425 |
| Dense projections | 768 | 502 | 291 |
| Experts | 668 | 667 | 484 |
| Attention (block attention, window merge) | 367 | 363 | 311 |
| Separate quantize, norm, column and rope passes | 297 | 316 | 31 |
| Expert send, activation assembly, finalize, route build | 307 | 305 | 174 |
| Index scoring and top-k | 176 | 169 | 44 |
| Stream mixing, update, collapse | 182 | 175 | 140 |

What the columns hold:

- The reference exchanges FP8 codes where we exchange BF16: its WO-A output
  crosses as MXFP8 (2,112 bytes a row against our 10,240-byte BF16 partial),
  and its expert input crosses as MXFP8 with the routes taken on the owner.
  Its collectives also overlap its multiplies by about 110 ms a step; ours do
  not overlap.
- Its dense projections and experts multiply MXFP8 natively. Ours multiplied
  a BF16 copy of every FP8 weight with BF16 tensor-core instructions at 52 to
  69 TFLOPS.
- Our middle waves also run layers 20 to 39 on their last 2,541 rows. The
  engine asks every prefill wave for the state at the next 16,384-token cache
  point, and a resumable state at a cache point needs the decoder layers'
  windows there. By the per-layer kernel durations that tail is about
  300 ms of our 3,828 ms; 134 ms of our expert time is the tail alone. The
  reference keeps no state mid-prompt and skips those layers.
- Our expert kernel runs about 67 TFLOPS with BF16 instructions, close to the
  reference's FP8-by-FP4 kernel for the same work; the expert row's gap is
  mostly the tail.

### Native MXFP8 projections

Every input to the dense projections was already rounded to MXFP8 (E4M3
codes and one power-of-two scale per 32, `SparkMathFp8QdqScale`), and every
weight is MXFP8 in the pack. `inference/kernels/mx_gemm.cuh` adds a
quantizer that writes the codes and scales with that rounding and a GEMM on
the block-scaled `mma.sync ... kind::mxf8f6f4.block_scale` instruction. On
this toolchain the instruction takes its scale registers in braces with
16-bit byte and lane selectors, and lane 4g carries row g's scale with lane
4g+1 carrying row g+8's (`tests/cuda/mx_gemm_cuda.cu` checks the kernel that
relies on it).

Each 32-wide block's product goes into an FP32 total, as the BF16 path did.
Letting the instruction chain its own accumulator instead is faster by
under 3% and is up to 8.7 BF16 ulp off the fp64 sum on a cancelling output
(the BF16 path and the block-summed form stay within 0.51 ulp on the same
samples). At the lane's shapes the native kernel runs 100 to 121 TFLOPS
against 52 to 69:

| Projection (rows x K x N) | BF16 path | Native MXFP8 |
| --- | ---: | ---: |
| q_a, own rows (4,096 x 5,120 x 1,280) | 67.0 TFLOPS | 114.4 |
| kv_a, own rows (4,096 x 5,120 x 512) | 51.1 | 104.3 |
| q_b (16,384 x 1,280 x 8,192) | 64.5 | 118.5 |
| o_b (16,384 x 2,048 x 5,120) | 65.8 | 121.2 |
| shared w1 shape (16,384 x 5,120 x 576) | 57.6 | 105.0 |
| engram wkv (16,384 x 6,144 x 6,400) | 68.4 | 118.2 |

99.99% of the outputs are the same bits as the BF16 path's on random MXFP8
data, and the lane's 40,000-token answer is unchanged.

| Build | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| Sixth pass, nodes c-f, same hour | 3,296 tok/s | 4,118 | 3,839 |
| Native MXFP8, nodes 4-7 | 3,492 tok/s | 4,347 | 4,061 |
| Native MXFP8, nodes 8-b | 3,450 tok/s | 4,390 | 4,076 |
| Native MXFP8, nodes c-f | 3,513 tok/s | 4,389 | 4,074 |

### The shared expert as a route

Each rank sent every other rank a 2,560-byte BF16 partial of the shared
expert for every row: its K slice's contribution to the destination's output
columns. Every rank pack holds the whole shared down matrix, so the exchange
now carries the rank's shared activation slice instead (FP8 codes with one
scale per 32, like a routed expert's slice; 640 or 512 values, at most 660
bytes a row), and each rank computes the four K-slice partials for its own
output columns. `D41FinalizeSliceKernel` keeps the publisher's per-source
sums (each source rank's experts plus its shared partial, rounded to BF16,
then summed), so the outputs are the same bits. An exchange row is 4,240 bytes
a destination instead of 6,128.

The first build changed the 40,000-token answer: the send kernel's grid
rounds the item count up to whole 32,768-item spans, and the rounded-up items
ran the old shared-partial branch into the next destination's shared codes.
Waves of 16,384 rows (exactly three spans) and short waves were clean; a
7,230-row last wave and a wave resumed from a cached prefix were not. With
that branch given no rows the 40,000-token answer is `b79b025295d2819f…`
again and 1,000-, 3,000- and 20,000-token prompts answer as before.

| Build | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| Native MXFP8, nodes c-f, same hour | 3,514 tok/s | 4,383 | 4,077 |
| Shared expert as a route, nodes 4-7 | 3,636 tok/s | 4,511 | 4,224 |

### What is left, largest first

Against the reference's 1,816 ms step, by the table above:

1. Rounds: the attention sum still crosses as a 10,240-byte BF16 partial a
   row where the reference sends 2,112 bytes of MXFP8 (it needs every rank to
   hold the whole o_b; the pack holds a rank's columns), the expert input
   crosses as BF16 rows where FP8 codes and the owner's routes would carry
   half, and no round overlaps a multiply.
2. The quantize, norm, column and rope passes (316 ms against 31): the
   reference writes FP8 codes from the producing kernel.
3. Layers 20 to 39 on every wave's last 2,541 rows, about 300 ms a wave, for
   a resumable state at each 16,384-token cache point.
4. Index scoring and top-k (169 ms against 44), and the expert send,
   assembly and finalize kernels (305 ms against 174).

## Drafts, 2026-11-13

The stage taps hidden states and verifies drafted rows at any rank count.
What the lane lacked was configuration and a drafter; no code changed.

- A lane record gives each rank's adapter the taps (the outputs of layers 36
  to 38, sent by rank 3 to the hub's 100G address) and the API its draft
  server: five drafts a step, of which the engine's frame planner verifies
  as many as its measured costs justify. `lane-record.drafts.json` is the
  record that ran. It is not in `deployment/lanes.json`, so the swap
  controller does not know this lane yet.
- The drafter is the sixteen-rank lane's, DSpark for this checkpoint, as a
  second instance of the hub's drafter service on its own unit and ports.

One stream, eight 96-token prompts (three of LaTeX prose, five of code), 256
tokens each, greedy, the adapter of the fourth pass with 4,096-row waves:

| | All eight prompts | A prompt |
| --- | ---: | ---: |
| Without drafts | 30.3 tok/s | 28.8 to 31.3 |
| With drafts | 71.3 tok/s | 45.4 to 114.1 |

- The drafted text is the plain text on all eight prompts, and on seven
  streams at once.
- Over the session the engine verified 681 frames holding 3,605 drafted
  tokens and kept 2,482 of them.
- A plain step takes 32 ms. A step that kept one to three tokens took 50 to
  57 ms (median), one that kept five or six 60 to 62, nine to eleven 72 to
  82, eighteen 127, twenty-seven 150. A verify row costs about 5.5 ms: every
  rank reads its slice of six more experts for it, 1.06 GB a row a rank.
- Another stream costs what a draft row costs, so drafts stop paying as
  streams are added:

| Streams | Without drafts | With drafts |
| --- | ---: | ---: |
| 1 | 30.3 tok/s | 71.3 |
| 3 | 65.7 | 67.6 |
| 7 | 94.1 | 90.9 |

- Prefill with the taps on is unchanged: 2,655 and 2,904 tok/s at 8,192 and
  32,768 tokens.

The reference reports 82.6 tok/s for one stream with drafts and 207.4 for
eight, on its own prompts; these prompts are not its prompts. A row needs
4.25 GB of expert weights over the group whoever reads them, which this
lane reads at about four fifths of the nodes' memory bandwidth. How the
reference's eight-stream figure fits those bytes is not established here.


## Eighth pass: the owner's rows cross as MXFP8, 2026-11-14

The branch now carries main at a242ef38e (weightd IPC ABI 16, four mesh
lanes), so the lanes run on the released weightd, on mesh lane 3. Two
exchanges of a wave split by rows move fewer bytes, and the outputs keep
their bits:

- The expert input. Every rank ran the router over all rows from a gathered
  BF16 copy of the normed input (10,240 bytes a row). The owner now routes
  its own rows and the group gathers E4M3 codes with one power-of-two scale
  per 32 and the routes, 5,328 bytes a row. The decoded codes are the values
  the gathered copy's quantization gave, and the router reads the same rows.
- The attention output. Every rank multiplied its o_b K-slice for all rows
  and the group reduce-scattered BF16 partials (10,240 bytes a row). Each
  rank now sends each owner its rows' WO-A output as the MXFP8 codes its
  multiply used (2,112 bytes a row). The owner multiplies them by all four
  K-slices of o_b, gathered once from every rank's pack slice at the first
  wave (1.6 GiB a rank), and adds the four BF16 partials in FP32 in rank
  order with one round-to-nearest-even, which is the pipelined
  reduce-scatter's rule.

Four groups, two rounds with the variants swapped between groups; every run
answered the 40,000-token prompt with `b79b025295d2819f`:

| Build | Groups | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | --- | ---: | ---: | ---: |
| Seventh pass on the new weightd | 0-3, 4-7 | 3,627 tok/s | 4,530 | 4,230 |
| fp8 expert input | 4-7, 8-b | 3,655 | 4,639 | 4,324 |
| fp8 expert input and attention output to owners | 0-3, c-f | 3,785 | 4,755 | 4,435 |

The 8,192-token figure of the last row includes the one-time o_b gather.
The rank state grows by the gathered slices: the lane needs a 16 GiB state
budget where it needed 14.

### Where a middle wave goes now

The MXFP8 trace's middle 16,384-row wave (3,819 ms on rank 0, before these
two changes) by kernel: the experts' cell multiplies 667 ms, the round
kernels about 1,050 ms (572 waiting for peers, 372 copying pieces, 94
arrivals; about 10 GB a wave a rank, so bytes drive them), the MXFP8
multiplies 325 ms, latent attention 203 ms and the window merge 148 ms, the
remaining BF16 multiplies (router, compressor, index) 175 ms, the expert
send 119 ms, the stream mixing about 170 ms, separate quantization 88 ms.
Next, largest first:

1. The experts multiply BF16 copies of FP8 activations and FP4 weights. The
   native FP8-by-FP4 block-scaled instruction ran the dense projections at
   100 to 121 TFLOPS against 52 to 69.
2. The expert exchange (189 MB a layer a rank) sends every route's
   activation slice to every rank. Expert parallelism sends the input once to
   each rank that holds one of the row's experts and returns BF16 sums, and
   its per-rank sums are the publisher's own four-GPU arithmetic.
3. The expert send kernel, the separate quantization passes and the column
   copies.

### The down projection on fp8 by fp4, o_a without column copies, and the cache stride

`4247654d4` runs the experts' down projection on the block-scaled FP8-by-FP4
instruction from the exchanged codes, and lets each o_a group read its
columns in place and write into its columns of the WO-A row. The up
projection stays on the BF16 cells: on its gathered rows the native kernel
was no faster (11.9 against 12.0 ms), because the BF16 cell kernel already
runs at about 86 TFLOPS.

The engine runs layers 20 to 39 on a wave's last rows whenever the wave ends
at a cache point, every 16,384 tokens (the module's
`cache_checkpoint_token_count`). Engine builds with a stride of 65,536 and
with no stride, on the same ranks, measure what that costs. Four groups, two
rounds, every run answering the 40,000-token prompt with
`b79b025295d2819f`:

| Rank build | Engine stride | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | --- | ---: | ---: | ---: |
| eighth pass | 16,384 | 3,748 tok/s | 4,714 | 4,426 |
| with `4247654d4` | 16,384 | 3,872 | 4,954 | 4,601 |
| eighth pass | 65,536 | 3,754 | 5,112 | 5,096 |
| eighth pass | none | 3,769 | 5,120 | 5,224 |
| with `4247654d4` | none | 3,869 | 5,316 | 5,459 |

A coarser stride is a product choice: a later request that shares a prefix
resumes at the last cache point before it, so with no stride it resumes only
where an earlier prompt ended. Saving each cache point's layer-20 input rows
and running layers 20 to 39 on them only when a request resumes there would
keep the 16,384-token resume points at the no-stride speed.

### Engram rows: one read per distinct block

The 2026-11-14 trace of a middle wave waited 99 ms at the first Engram layer:
the rows of 16,384 tokens came from disk one 4 KiB read per id. `539e08834`
sorts a fetch's misses, merges rows that share or neighbour a block into
spans of up to 64 KiB, and reads each span once. Four groups on the same
engine (`hubbuild-rall`), two of each rank build, each answering the
40,000-token prompt with `b79b025295d2819f`:

| Rank build | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| before (`d41rel-cols`) | 3,851 / 3,871 tok/s | 5,267 / 5,300 | 5,433 / 5,458 |
| merged reads (`d41rel-eng`) | 3,900 / 3,885 | 5,365 / 5,421 | 5,501 / 5,481 |

One to two percent: the fetch starts when the wave starts, so part of it
still lands after the first Engram layer is reached. Hiding the rest needs
the next wave's tokens before that wave begins.

### Whole experts per rank, and Engram rows reduce-scattered to their owners

Two changes, measured together on 2026-11-14 with the same engine
(`hubbuild-rall`) and 16 GiB of rank state, one group each, every run
answering the 40,000-token prompt with `b79b025295d2819f` (the bits did not
move):

- `5a254425d` gives each rank whole experts when the pack says so (a new
  pack flag; `tools/dsv41_flash_stagepack.py plan ... whole`, `1a8ba58c5`).
  Rank r holds experts 96r to 96r + 95, runs its own routes from the gathered
  rows, adds its slice of the shared expert, rounds once, and one rank-order
  reduce-scatter (126 MB a rank at 16,384 rows) replaces the activation
  all-to-all (208 MB), its assembly and the slice exchange. The per-rank
  sums are the publisher's four-GPU arithmetic, so the bits are the same.
- `fb1dc9fc9` (with `4bce70d28`): an Engram layer reduce-scatters the
  embeddings to the row owners and each rank projects only its own rows with
  the whole key and value projection, instead of all-reducing every row and
  all-gathering 629 MB a rank of keys and values it then ignored.

| Rank build and pack | 8,192 tokens | 32,768 tokens | 131,072 tokens | One stream | Eight streams |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert slices | 3,901 tok/s | 5,324 | 5,482 | 33.2 / 35.4 / 35.4 ms a token | 92.9 tok/s |
| whole experts | 4,099 | 5,518 | 5,701 | 35.6 / 36.8 / 37.5 | 84.5 |
| expert slices, Engram to owners | 3,904 | 5,633 | 5,723 | 32.9 / 35.3 / 34.9 | 93.6 |
| whole experts, Engram to owners | 4,236 | 5,876 | 5,991 | 35.8 / 36.7 / 36.9 | 86.1 |

The Engram change is free: 4% on prefill and nothing on decode. Whole experts
trade: 4% on prefill, but a decoding step waits for the rank that holds the
most of the step's experts (2.8 of 6 on average against 1.5 slices), about
1.5 to 2.4 ms a token alone and 9% at eight streams. A pack of whole experts
is an option for prefill-heavy use; the slice pack stays the default.

### The window inside the sparse attention, and fewer casts (2026-11-15)

Against the reference measured on the same nodes (8,467 tok/s at 32,768
tokens, 8,407 at 131,072, 7,410 and 7,637 at 8,192; 1,816 ms for the step
covering tokens 16,384 to 32,767 of a 65,536-token prompt):

| Rank build | Pack | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | --- | ---: | ---: | ---: |
| before (`572fd7ca1`) | whole experts | 4,243 tok/s, 56% | 5,819, 69% | 6,035, 72% |
| fused attention (`1d77aca26`) | whole experts | 4,556, 61% | 6,616, 78% | 6,794, 81% |
| and codes for WO-A and the query gather (`85cdc722f`) | whole experts | 4,704, 63% | 6,676, 79% | 6,891, 82% |
| the same | expert slices | 4,384, 58% | 6,283, 74% | 6,484, 77% |

Every run answered the 40,000-token prompt with `b79b025295d2819f`.

- `1d77aca26` (`D41AttendKernel`, `dsv41_flash_attend.cuh`): a wave of more
  than 16 rows attends its 128-key window (from the wave or the ring) and its
  selected compressed rows in one online softmax, adds the sink and writes the
  output. It gives each row's 16 heads all eight consumer warps; the shared
  sparse kernel padded 16 heads to 32 queries and used four. The window's two
  matrix products, their FP32 scores and weights, the second set of records and
  the merge that read both sets back are gone. Weights are rounded against a
  running maximum taken a tile at a time, so the bits move: over the 2,112
  corpus positions next-token accuracy is 79.40% against 78.98%, the NLL
  change +0.4% [-0.5, +1.3], top-1 agreement 97.0%, the size of any two
  arithmetic variants' difference here.
- `30a3f644a`: the attention's epilogue undoes the rope and casts the WO-A
  input to MXFP8, byte for byte the rope kernel and quantizer it replaces.
- `85cdc722f`: the query latent is gathered as MXFP8 codes (1,320 bytes a row
  against 2,560) and multiplied from them.

The same step, rank 0, ms (trace of the fused-attention build):

| Stage | Ours | Reference | Ours, % of the reference's time |
| --- | ---: | ---: | ---: |
| Whole step | 2,244 | 1,816 | 124% |
| Rounds | 616 | 425 | 145% |
| Dense projections | 318 | 291 | 109% |
| Experts | 483 | 484 | 100% |
| Attention | 225 | 311 | 72% |
| Separate passes | 128 | 31 | 413% |
| Expert send and finalize | 155 | 174 | 89% |
| Index scoring and top-k | 81 | 44 | 184% |
| Mixing | 132 | 140 | 94% |

What is left: the rounds hold about 94 ms of waiting for the rank with the
most routes (whole experts only), about 95 ms of Engram reduce-scatter that
waits on the slowest rank's disk fetch, and the 110 ms the reference hides
under its multiplies; the GPU also idles about 64 ms at the first Engram layer.
The fetch of layer 1's rows takes 65 to 360 ms a wave on a rank and starts
when the wave starts; more readers and a fourfold row cache did not shorten it.
Hiding it needs the next wave's tokens during the current wave.

### Engram reads through io_uring, a top-k for long rows, ratio-2 rows sized by their keys (2026-11-15)

From here the 8,192-token figure is the second request after a lane starts:
the first one pays about 0.15 s of warm-up (4,746 tok/s against 5,104 to
5,363 warm on the same build). The reference warms up before it serves. Its
8,192-token figure is the mean of 7,410 and 7,637, 7,524 tok/s.

| Rank build | Pack | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | --- | ---: | ---: | ---: |
| `85cdc722f` | whole experts | 5,159 tok/s, 69% | 6,750, 80% | 6,902, 82% |
| Engram rows through io_uring | whole experts | 5,320, 71% | 7,005, 83% | 6,933, 82% |
| and the top-k and the ratio-2 rows | whole experts | 5,335, 71% | 7,091, 84% | 7,277, 87% |
| the same | expert slices | 5,006, 67% | 6,721, 79% | 6,857, 82% |

Each figure is the mean of two lanes on different nodes; every run answered the
40,000-token prompt with `b79b025295d2819f`.

- Engram rows: each of four readers owns an io_uring ring of 128 reads, refilled
  as reads complete (`runtime/row_file.c`; other systems keep the blocking
  readers). On the row file a Spark's NVMe gives 345K random 4 KiB reads a second
  to 64 blocking threads and 1.67M to 1.72M to four rings. The GPU's idle time in
  a 16,384-row wave fell from about 90 ms to about 30 ms.
- Top-k: a row of more than 16,384 scores finds its threshold with one or two
  histogram passes (eleven bits, counts gathered per warp), keeps the few rows of
  the threshold's bucket in shared memory and sorts them there
  (`LmTopkExactWideKernel`). It returns the same indices in the same order as the
  single-block kernel, ties and `-inf` included (168 device cases), 1.6 to 2 times
  faster on rows of 28,672 to 131,072 scores. Shorter rows keep the single-block
  kernel, which is faster while a row stays in L1.
- The ratio-2 index layers (2, 8 and 14) score half as many columns, one per
  compressed key rather than one per position, so half of each row is no longer
  `-inf`; never fewer than 512, so a row with fewer keys pads the same way.
  Such a layer scores all of a rank's rows in one call instead of four.

The step covering tokens 16,384 to 32,767 of a 65,536-token prompt, rank 0,
whole experts, ms:

| Stage | `85cdc722f` | Now | Reference | Now, % of the reference's time |
| --- | ---: | ---: | ---: | ---: |
| Whole step | 2,244 | 2,026 | 1,816 | 112% |
| Rounds | 616 | 503 | 425 | 118% |
| Experts | 483 | 482 | 484 | 100% |
| Dense projections | 318 | 323 | 291 | 111% |
| Attention | 225 | 245 | 311 | 79% |
| Expert send and finalize | 155 | 159 | 174 | 91% |
| Mixing | 132 | 136 | 140 | 97% |
| Separate passes | 128 | 85 | 31 | 274% |
| Index scoring and top-k | 81 | 46 | 44 | 105% |
| GPU idle | 90 | 31 | 2 | |

What is left in this step: the reference hides about 110 ms of its collectives
under its multiplies (it projects a slice of columns while the previous slice
is reduce-scattered), and ours do not overlap; 23 ms of the idle time is the
launch gap between the two kernels of each of 4,379 transport pieces; the
separate norm and cast passes; dense projections 11% slower. A prompt's last
wave also runs the upper 20 layers on its last 2,541 rows (fewer at each layer,
128 at the last), 662 ms at 65,536 tokens and about 410 ms at 8,192: 27% of an
8,192-token prefill.

### Bounded replay, measured and not in the code

The reference runs the upper 20 layers (20 to 39, which read layer 20's cache)
on a prompt's last 128 rows only, and each of those rows' windows starts at the
first of them. The publisher's inference code runs every row through every
layer, with a 128-key window per layer, so its final row depends on the last
2,541 rows of layer 20; ours computes exactly that. Measured with the window
floored at the first replayed row:

| Pack | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| whole experts, exact | 5,292 tok/s, 70% | 7,010, 83% | 6,930, 82% |
| whole experts, bounded replay | 6,450, 86% | 7,513, 89% | 7,152, 85% |
| expert slices, exact | 4,994, 66% | 6,626, 78% | 6,545, 78% |
| expert slices, bounded replay | 5,963, 79% | 7,164, 85% | 6,782, 81% |

(Builds before the top-k and ratio-2 change; the 40,000-token prompt's answer was
the same.) Against the exact computation over 2,112 corpus positions, each a fresh
prefill of 256 to 752 tokens: next-token accuracy 79.12% against 79.17%
(McNemar p = 1.0), NLL +0.27% [-0.11, +0.66], top-1 agreement 98.4%, KL at
least 0.0024 nats. It changes what the model computes, so it is an option to
choose, not a speed-up.

## Sixteen ranks, each owning its rows (2026-11-16)

`build/libdsv41_flash_rows16_serving_adapter.so` (`-DDSV41_FLASH_ROWS=1`, lane
`dsv41_flash_rows16`) runs the four-Spark design on all sixteen Sparks with
attention degree 1 and KV degree 1: every rank keeps every head's attention
weights (q_b, o_a, o_b, the indexer's q_b and weights and the sinks; pack layout
`rows`, flag 4) and the whole context's keys, and owns a sixteenth of each
step's rows through layer 19. The routed experts stay whole, 24 per rank. Only
the rows' KV latents cross for attention, the indexer runs on the owner's rows,
and the MoE exchange is the four-Spark one at sixteen ranks (MXFP8 rows
gathered, sums reduce-scattered to their owners). The tail (layers 20 to 39 on
the last 2,541 rows) splits each layer's attention over the sixteen ranks and
gathers the outputs; its MoE keeps the replicated routing and an all-reduce.
There is no fragmentation into groups of four: one sequence's keys live once
per rank and every rank serves every sequence.

| Build | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| reference, four Sparks | 7,524 tok/s | 8,467 | 8,407 |
| production sixteen ranks (heads and context split) | 2,605, 35% | 2,798, 33% | 2,761, 33% |
| rows owned, tail replicated | 7,979, 106% | 12,881, 152% | 14,261, 170% |
| rows owned, tail attention split | 8,515, 113% | 13,065, 154% | 14,505, 173% |
| rows owned, MoE rows sent to their experts' ranks, q_b scales fixed | 8,953, 119% | 14,516, 171% | 16,552, 197% |
| and layer 20 handed off from owned rows in chunks before the tail | 9,198, 122% | 14,423, 170% | 16,857, 201% |
| and experts placed by routed load | 8,878, 118% | 15,316, 181% | 18,125, 216% |

Teacher-forced next-token scores over 512 corpus positions (16 documents,
fresh prefills of 256 to 752 tokens) against the four-Spark build:

| Arm | Top-1 agreement | KL at least, nats | NLL change |
| --- | ---: | ---: | ---: |
| production sixteen ranks | 94.1% [92.2, 96.1] | 0.016 | -1.2% [-2.6, +0.3] |
| rows owned, before the q_b scale fix | 90.0% [88.1, 92.0] | 0.072 | +2.2% [-0.6, +5.1] |
| rows owned, fixed, MoE rows sent to their experts' ranks | 95.1% [93.6, 96.7] | 0.016 | -2.3% [-3.8, -1.0] |

The fixed build agrees with the four-Spark build as closely as production
does. Its 40,000- and 1,500-token answers are the four-Spark build's
(`b79b025295d2819f`, `9595ae42ab7d191d`); at 200 tokens it parts after nine
tokens, as a near tie allows.

Six things had to change on the way: each 16-head part's q_b multiply reads
its own rows' weight scales (they are stored per row; reading them as 32x32
blocks gave heads 16 to 63 another part's scales, KL 0.072), the sinks are
offset by the attention rank (every rank's offset is 0 here; before, ranks 1 to 15 read past the 64
sinks), the engram reduce-scatter's collective bound covers twice the padded
rows, the fused attention takes a live-row count so a rank's padding rows are
never attended, steps of 17 to 256 rows that are not split take the per-head
attention decode uses (the fused kernel reads 16 heads a row, and a rank here
holds 64), and a whole-expert pack carries the empty experts manifest.

What is left, largest first, for 16,384-row steps:

1. Done: each row now goes only to the ranks that hold its experts and comes
   back as one partial per rank, instead of every row to every rank and BF16
   sums reduce-scattered back. It measured 2-5% before the scale fix; a trace
   of the fixed build places the rest.

Rank 0's CUPTI trace of the fixed build on a 32,768-token prompt (run rw15):

| Stage | First chunk, 16,384 rows | Final chunk, with the tail |
| --- | ---: | ---: |
| Mesh rounds, nothing else on the GPU | 495 ms | 644 ms |
| Experts (up and down) | 118 | 144 |
| Expert routing, exchange packing, finalize | 82 | 106 |
| Dense projections | 94 | 120 |
| Fused attention and window | 45 | 81 |
| Stream mixing, norms, quantize, index | 68 | 142 |
| Whole step | 890 | 1,251 |

The rounds are 827 pieces of about 300 us of waiting each. Compute is about
395 ms a chunk, so the remaining gap to the 27,300 tok/s estimate is the
rounds: overlapping them with compute, and cutting the gathers below.
2. Done: experts placed by routed load. With rank r on experts [24r, 24r+24),
   one layer's busiest rank took 1.4 to 2.5 times the average rows (the 12
   busiest of 384 experts carry 29% of the routes, a different group hot in
   each layer), and the all-to-all pads every pair to the busiest. Placing each
   layer's experts busiest first on the least-loaded rank
   (`tools/dsv41_flash_expert_placement.py`, counts from the long corpus,
   `model-families/dsv41_flash/expert_placement_rows16.json`, written beside
   the packs as `expert_placement.bin`) cut the busiest rank to 1.41 times the
   average on chunks it was not built from, and the step 6-8%. The score
   against the four-Spark build is unchanged (94.1% top-1, KL 0.0145).
   Copying the busiest 16 experts to every rank, each row's owner computing
   them, would bring it to 1.34 and keep 29% of the routes off the wire.
3. Done for chunks before the tail: layer 20's keys are compressed from the
   owned rows' normalized hidden states, gathered (164 MB), instead of every
   row's four streams (671 MB); the inputs are the same bits. It saved about
   28 ms a chunk. The final chunk still gathers the streams for its tail.
4. Done: the compressed-key layers (2, 8, 14 and the layer-20 handoff) no
   longer gather every row's normalized hidden (164 MB a layer). Each owner
   projects its own rows' c_kv and c_score (5,120 to 512, FP32) and the ranks
   gather those (67 MB, 34 MB at layer 20, where only c_kv is used); the
   compressor, keys and index keys then run on every row as before. A row's
   projection is the same bits whatever the row count above 16 (the stream
   GEMM accumulates each output in a fixed K order, with no split-K).
   Attention builds its queries in a buffer of their own, so the owned rows'
   normalized hidden stays in place for the indexer; steps whose rows are not
   owned (the tail, short prefills) still read the full rows.
5. Done: the tail is owned like the rest. In the final chunk layer 20
   compresses its keys from the owned rows, as the chunks before it do, and
   the tail's rows (the last 2,541) move once, by two all-to-alls (the four
   streams, then the stream weights), from the ranks that owned them in the
   step to a split of their own, 160 rows a rank. Layers 20 to 39 then take
   the rows-owned path: each rank computes its rows, and the MoE exchange
   carries them to their experts' ranks. The tail before ran routing, the
   shared expert and the stream mixing for every tail row on every rank,
   summed the MoE by an all-reduce or, from layer 25 on, by an all-gather of
   every rank's partial for every tail row (about 1.8 GB a rank a final
   chunk), and gathered every row's four streams at layer 20 (480 MB a rank).
   It also added the shared expert inside one rank's partial before the
   ranks' sum; the exchange rounds the routed sum first and adds the shared
   expert after it, as the rest of the step and the publisher do. So the
   scores move (all 56 deep positions) and now equal, bit for bit, a build
   with no tail at all, every layer on every row: 56 of 56 deep positions and
   30 of 30 fresh 7,000-token prompts.
6. The mesh rounds still run with nothing else on the GPU; splitting a step
   into two halves on two streams would hide them behind compute. Built and
   exact, not faster while both halves share one collective stream: see
   below.
7. Done: the engram exchange. Each rank holds a sixteenth of each engram
   table by row range. Instead of reduce-scattering a [rows x 6,144] BF16
   buffer that is zero except the rows the rank holds (201 MB a table on a
   16,384-row step), every rank numbers the looked-up rows by holder and owner
   (every rank hashes every row, so the counts need no collective, only one
   read on the host at layer 1 for both tables), holders send each row's FP8
   codes and scales (272 bytes) to the row's owner, and the owner decodes them
   as the holder's own lookup does: about 6 MB a rank instead of 188 MB. Both
   tables' ids are hashed at layer 1 and the n-gram history is updated there
   instead of at the step's end.
8. Done: window keys. Each layer gathered every row's window key (16 MB) for
   attention that reads only the 127 rows before each owned block and, for
   the window ring, the step's last 128 rows. The ranks now gather their last
   min(own, 256) rows (4 MB); 256 covers a window plus the padding rows a
   rank's block can end with.

| Build | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| experts placed by routed load (above) | 8,878, 118% | 15,316, 181% | 18,125, 216% |
| and the engram exchange, owned key projections and window keys | 9,637, 128% | 16,202, 191% | 19,360, 230% |
| and the pipelined rounds' closing marker (2026-11-18, below) | 9,382, 125% | 16,414, 194% | 19,115, 227% |
| and the tail owned like the rest (2026-11-18) | 11,591, 154% | 17,704, 209% | 19,723, 235% |
| the marker build in two lanes on one collective stream (not on) | 9,275, 123% | 16,130, 191% | 19,158, 228% |
| and compact exchange rows, four piece-copy blocks an SM (2026-11-18, below) | 11,999, 159% | 18,621, 220% | 20,466, 243% |
| and rows re-shuffled to MoE owners by row index (2026-11-18, below) | 12,513, 166% | 19,363, 229% | 21,151, 252% |
| and the embedding gathered once, looked up locally (2026-11-18, below) | 12,784, 170% | 20,122, 238% | 21,817, 260% |
| and index scores with each block's key slots found once (2026-11-18, below) | 12,906, 172% | 20,163, 238% | 22,150, 263% |
| and expert gate and up projections on the publisher's FP4 arithmetic (2026-11-18, below) | 13,377, 178% | 21,090, 249% | 23,308, 277% |
| and the route build ranked by warp matches (2026-11-18, below) | 13,383, 178% | 21,187, 250% | 23,496, 279% |
| and the shared expert on a side stream during the MoE rounds (2026-11-18, below) | 13,613, 181% | 21,610, 255% | 23,612, 281% |
| and attention two head groups a pass (2026-11-18, below) | 13,649, 181% | 21,600, 255% | 24,202, 288% |
| and backup expert copies taking rows off the busiest expert rank (2026-11-18, below) | 14,411, 192% | 23,372, 276% | 25,990, 309% |
| and a leaner expert loop with the same bits (2026-11-18, below) | 14,370, 191% | 23,773, 281% | 26,343, 313% |
| and the plan's second pass shrinking the padded pair (2026-11-18, below) | 14,861, 198% | 23,733, 280% | 26,559, 316% |
| and the owner's eight o_a groups in one launch (2026-11-18, below) | 14,805, 197% | 24,030, 284% | 26,975, 321% |
| and a second backup copy of each expert (2026-11-18, below) | 15,189, 202% | 24,317, 287% | 27,367, 326% |
| and index scores from keys staged in shared memory (2026-11-18, below) | 15,064, 200% | 24,677, 291% | 28,019, 333% |
| and single-route records skipping the finalize, eight-row mHC partials (2026-11-18, below, level) | 14,920, 198% | 24,690, 292% | 28,001, 333% |
| and the gate and up experts reading the landing records (2026-11-18, below, level) | 14,920, 198% | 24,871, 294% | 28,012, 333% |
| and one decode graph for every prompt length (2026-11-18, below) | 15,074, 200% | 24,799, 293% | 28,317, 337% |
| and the one-row step spread over the ranks (2026-11-18, below) | 15,759, 209% | 25,366, 300% | 28,482, 339% |
| and the side-stream shared expert yielding SMs, graph slots per kind (2026-11-18, below) | 15,600, 207% | 25,313, 299% | 28,458, 339% |
| and the owned tail re-split at each indexer layer (2026-11-18, below) | 16,797, 223% | 26,154, 309% | 28,590, 340% |
| and the staged mHC Sinkhorn partials (2026-11-18, below) | 17,129, 228% | 26,594, 314% | 29,448, 350% |
| and the spread one-row step's window fold in clusters (2026-11-18, below, level) | 16,857, 224% | 26,931, 318% | 29,115, 346% |

All four changes give the same bits: teacher-forced scores at 56 positions
deep in a joined corpus (prefills of 6,000 to 12,000 tokens that resume from
cached prefixes, every final chunk running the tail) are identical to the
previous build's at every position, as are the 40,000-, 1,500- and 200-token
answers. A first version had the indexer read the owned rows' hidden even on
steps whose rows are not owned; the same check caught it (KL 0.0019, all 56
positions moved), and a single-variable bisect over builds placed it.

Two lanes, built and exact, not on. The runner splits a step of 4,096 rows
or more into halves on two streams, with their rounds in one fixed order on
every rank. DSV4.1 needed the second lane's arena sized and planned, the
per-slot state (window ring, compressor state, n-gram history), the verify
buffers and the constant layer-0 stream mix shared with the first lane (an
arena of its own held zeros for that mix, which gave the second half's rows a
zero layer-0 attention input: KL 0.040 at deep positions while the 40,000-token
answer still matched), and a row-state fence each layer.

With all that, two runs of the lane build still scored 2 to 5 of 30 fresh
7,000-token prompts differently from one lane, different prompts each run,
and timed out now and then. Every rank hashed its own rows after each block
and then each stage of the MoE; across runs the first difference was always
in the second lane's FFN, in the rows an owner received back from the
combine all-to-all, while every sender's expert output was the same bits,
and per peer and per 64-row chunk it was always the second-to-last chunk of
the four-chunk pipelined round. The fifth pass's pipelined round publishes
its last chunk before it reads the second-to-last one, so a fast rank can end
the round while a slow rank still has two chunks to read, and the fast
rank's next whole-slot round (here the other lane's dispatch all-to-all,
about 1 MB a peer) lands on one of them. On one lane the compute between
rounds hides it; with two lanes the other lane's round follows at once and
its kernels slow the reader. Each rank now publishes an 8-byte marker after
its last read and waits for every peer's marker before the round returns
(e223b2b25); the model test and a new probe mode, `--pipeline-boundary`,
reproduce the overwrite without the marker and pass with it. The lane build
then matched one lane bit for bit on 90 of 90 fresh prompts in three runs and
at all 56 deep positions, with no timeouts. The marker costs one small round
trip a pipelined round; one lane measured 9,382, 16,414 and 19,115 tok/s
after it against 9,637, 16,202 and 19,360 before (single runs).

Two lanes are no faster: both lanes' rounds share one collective stream, so a
lane's ready round waits behind the other lane's, 224 to 332 ms a 16,384-row
step with nothing else on the GPU, and the final chunk's 20-layer tail runs on
one lane alone (about 350 ms). A collective stream per lane and a tail split
over both lanes would take 200 to 260 ms a step and 150 to 250 ms a final
chunk. A stream per lane means two collectives in flight per engine, which
the rule in force since October forbids
(`docs/CX7_DMA_INVESTIGATION_2026-10.md`, section 15); lifting it is the
owner's decision.

Compact exchange rows and wider piece copies (2026-11-18). A CUPTI trace of
the owned-tail build (rank 0, 32,768-token prompt) put a non-final
16,384-row step at 741 ms: 290 ms of round kernels with no compute under
them (MoE return all-to-all 145, MoE send 68, the counts gather 32), experts
156, dense projections 85 and MoE glue 83. Two exact changes:

- The exchange pads every pair to the busiest pair. An expert rank unpacked,
  routed and finalized 16 x capacity rows, often three times its real rows.
  Unpack now writes only real rows, packed source by source; route build,
  experts and finalize see only those, and finalize writes each row to its
  place in the padded return. Finalize, unpack and the owner's combine move
  eight BF16 values a thread. Same sums in the same order.
- The pipelined piece kernel copied 15 MiB a step with one 128-thread block
  an SM; it now runs four (`SPARK_TP_MESH_PIECE_RESIDENCY`).

All 11 probe modes and the exchange test pass, and scores are identical at
56 deep and 30 fresh positions. Same session, the owned-tail build first:
11,733 / 15,788 / 19,194 tok/s; compact rows alone 12,068 / 18,260 /
20,303; both 12,116 / 18,623 / 20,555 and 11,882 / 18,619 / 20,376 (the
table row is the mean of the two).

Why the gain is small: routes dumped from all sixteen ranks for the 8K, 32K
and 131K prompts show the hottest expert of a layer used by 35% of a step's
rows on average and by up to 71%. Two costs follow from it.

1. Pair sizes. Each owner's 1,024 contiguous rows lean on a few ranks'
   experts: the busiest pair averages 821 rows against an average pair of
   336, and a padded all-to-all moves the busiest pair's size to every peer.
   Rows interleaved by 16 over the MoE owners bring the busiest pair to 523
   rows, 35% fewer pieces; that needs no pack change (next).
2. Expert compute. The busiest rank's experts take about 239 ms a step
   against 160 ms on average (rank 0's measured 156 ms fits 1.3 us a
   route-row), and the others wait at the next round. Fixed copies of hot
   experts do not fix it, because the hot experts change with the document:
   the corpus's top 16 cut the busiest pair only from 821 to 656 rows, and
   even the step's own top 8 on every rank leave the busiest rank at 1.36
   times the average. Two copies of every expert, the copy chosen per step,
   bring it to the average; that is exact only if owners rebuild each home
   rank's partial from per-route outputs, and it needs the second copies
   loaded.

Rows re-shuffled to MoE owners (2026-11-18). Each owner now sends its owned
rows' FP8 codes, scales, experts and weights to the MoE owner named by the
row's index modulo 16 (a uniform all-to-all, 64 rows a pair on a 16,384-row
step). The MoE owner runs the exchange for rows spread over the whole step,
sums the ranks' partials in rank order, rounds once and returns the result;
the row's owner adds the shared expert. The per-pair counts the capacity
needs travel in one all-gather before the re-shuffle (each owner counts, for
every MoE owner and expert rank, the rows it will contribute), so the host
reads them while the re-shuffle runs. Same sums, same roundings, same order.
Two extra rounds a layer (about 20 ms a step) against a busiest pair of 523
rows instead of 821. Exchange test (owner to MoE owner to expert rank and
back, nine cases) passes; scores identical at 56 deep and 30 fresh
positions. Eight runs: 12,338-12,691 / 19,068-19,578 / 21,036-21,257 tok/s;
the previous build 12,000-12,114 / 18,405-18,654 / 20,479-20,584 in three
runs the same session.

A trace of the first chunk shows where the rest goes: the MoE return round
fell from 259 to 224 ms a step and the send from 72 to 52 ms, but 175 ms of
the return is waiting for the busiest expert rank. Next: two copies of every
expert, the copy chosen per step for balance, and per-route returns so that
each MoE owner rebuilds every home rank's partial exactly (simulated: busiest
rank's experts 239 to 176 ms a step, 332 to 285 piece steps).

Seen five times in about 50 runs today, on every build: one rank's first
decode graph capture after a prefill fails (`cuda=901`) and the other ranks
time out. With the driver's error log on (`CUDA_LOG_FILE=stderr`) a failing
rank showed the cause: another thread's `cuStreamQuery` on the stream being
captured returned 900 (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`), and the
runner's next launch found the capture invalidated. The runner hands its
stream to the device API as an adopted queue; when a KV worker stamps or
copies into a buffer that queue has stamped, the device API asks whether the
adopted queue's work is done, which queries the runner's stream. The runner
now marks its adopted queue for the length of a capture
(`SparkDeviceQueueCapture`); while marked, the device API reads the queue as
busy without touching its stream, and waits poll instead of synchronizing.
`tests/test_device_capture_cuda.py` reproduces the invalidation with a raw
query from another thread and shows queries, timed waits and stamped fills
against a marked queue leave the capture whole.

Second expert copies (2026-11-18, not on). Each expert also held on a
second rank (9 GB a rank for layers 0-19, filled over the mesh at the first
long prefill), the copy chosen per step, routes sent and returned one by one
so that each MoE owner rebuilds every home rank's partial in route order
(exact: 56 deep and 30 fresh positions identical; the exchange test picks
copies at random). Balancing each MoE owner's routes per expert spread every
expert over both copies, so each rank read about twice the expert weights
and lost what balance gained (rank 0's experts 160 to 186 ms a step).
Balancing whole experts instead: 12,258 and 12,021 / 19,644 and 19,486 /
21,769 and 21,790 tok/s against 12,513 / 19,363 / 21,151 for the re-shuffle
build: +3% at 131K, -3% at 8K, for 9 GB a rank. Kept out; the code is on
branch `topology/dsv41-second-copies` (4e4c6be2d).

Embedding gathered once (2026-11-18). Each step began with the runner's
vocab-sharded embedding: every rank looked up the rows its shard holds and an
all-reduce summed 16,384 x 5,120 BF16 values, about 315 MB each way a rank,
20 ms a 16K step and one round in every decode step. In rows mode a rank
only ever reads its own rows. The model now asks the runner to gather the
shards once (`embed_gather`, stage runner model ABI 8; an all-gather of
1.3 GB a rank before the first step, outside any graph capture) and every
step looks rows up locally. The all-reduce summed one value with fifteen
zeros from +0 in FP32, so the local lookup writes +0 for -0 and nothing else
changes: scores identical at 56 deep and 30 fresh positions, the same 40K
answer. Three interleaved runs each: 12,784 / 20,122 / 21,817 tok/s against
12,697 / 19,287 / 21,258 for the previous build; a lane's first request pays
the gather once (about 0.3 s).

A trace of ranks 0 and 15 on the 131K prompt (both ranks' clocks agree within
a few ms) puts the rest of the waiting in three places: the MoE return round
waits for the busiest expert rank (70 to 120 ms a chunk), rank 0 waits 20 to
30 ms a chunk at the counts gather for later rows' heavier attention and
index scoring, and the final chunk's 2,541-row tail costs about 250 ms (index
scoring over the whole context 48 ms, weight-read-bound experts 42 ms,
rounds 100 ms). The index-score kernel ran at 20 to 30 TFLOPS on the long
calls (about 277 ms of rank 0's 131K prompt), and it spent most of that
finding keys rather than multiplying them (next).

Index scores with each block's key slots found once (2026-11-18). Every lane
of every warp mapped its key's position to a cache slot on every eight-key
tile: the shard arithmetic's runtime divisions (one of them 64-bit) and a
page-table read, repeated by the 32 lanes and eight row warps that share each
key. Staging a block's 256 keys in shared memory first (69.6 KB, so one block
an SM, and no loads in flight while it filled) was exact but 1 to 3% slower
in the lane. The scorer for blocks of eight rows now has one thread find
each of the block's 256 slots into a 2 KB table the eight row warps share
(the per-lane lookup stays for a block whose rows span two sequences), and
each warp loads the next tile's keys while it multiplies the current one.
Same MMAs in the same order, same epilogue:
`tests/test_index_score_slots_cuda.py` compares every score bit for bit with
the per-row scorer (nine cases, page and shard views, 1,280-byte slots, two
sequences in a block); offline 1.4 to 1.6 times faster (1,024 rows by 65,536
keys: 21.9 to 13.4 ms, 25 to 41 TFLOPS). Scores identical at 56 deep and 30
fresh positions; three interleaved runs each: 12,906 / 20,163 / 22,150 tok/s
against 12,738 / 19,747 / 21,708 for the previous build in the same session.

Expert gate and up projections on the publisher's FP4 arithmetic
(2026-11-18). The publisher's `fp4_gemm` (its `inference/kernel.py`) casts
the FP4 weights to FP8, multiplies each 32-wide K block of FP8 activations in
one tensor-core MMA from zero, scales the block's result by the activation
and weight E8M0 scales and adds it into an FP32 total. The down projection
has done the same on the block-scaled `kind::mxf8f6f4` instruction since 14
November; the gate and up projections (w13) still widened the codes to BF16
and chained one BF16 MMA accumulator across K, which rounds differently in
about 0.05% of outputs. `tests/test_cell_mx_cuda.py` now emulates the
publisher's GEMM on the device (a plain E4M3 MMA from zero per block, then
both scales and the add in FP32) and requires the block-scaled kernel's bits:
0 of 1,029,888 outputs differ, against 475 for the BF16 kernel. The expert
rank now keeps the exchange's codes and scales as they arrive (5,280 bytes a
row where the BF16 copy took 10,240) and runs w13 on the block-scaled
kernel, as w2 already did; steps that do not exchange (decode and short
steps) are unchanged. Scores move by design, by rounding-level amounts
carried through 43 layers and the expert routing: at the 56 deep positions
the truth token's logprob moves 0.038 on average (mean truth NLL 0.3796 to
0.3808), at the 30 fresh ones 0.088 (0.745 to 0.817, a few positions each
way); the top token changes at 4 and 3 near-tied positions, and the largest
moves are in tokens below logprob -9. These scores (runs zn18d1, zn18f1) are
the reference for later exact changes. Three interleaved runs each: 13,377 /
21,090 / 23,308 tok/s against 12,790 / 20,000 / 22,154 for the previous
build; the 40,000-token answer unchanged.

Route build ranked by warp matches (2026-11-18). `LmRouteBuild` runs as one
block of 256 threads, chunk by chunk in route order. Each thread used to
count the earlier routes of its chunk with the same expert one at a time (up
to 255 reads a route); a warp now ranks its 32 routes with one
`__match_any_sync`, adds the counts of the chunk's earlier warps from a
small table, and loads the next chunk's experts while it places the current
one. The packed order (by expert, then route order) is the same, which
`tests/test_route_build_scan.cu` checks route by route, now also on
26,652-row steps; 159,912 routes take 503 us instead of 1,604. An expert
rank of a full 16K step builds about 53,000 routes a layer, about 530 us
before. Three interleaved runs: 13,383 / 21,187 / 23,496 tok/s against
13,127 / 21,222 / 23,318 for the previous build (0.2% lower at 32K, within
the run-to-run spread); scores identical at 56 deep and 30 fresh positions.

Shared expert on a side stream (2026-11-18). Each owner's shared expert
(gate and up projections, SwiGLU and down projection, about 0.8 ms a layer
for 1,024 rows) ran on the step's stream between the dispatch round and the
routed experts, so the busiest expert rank started its experts that much
later. It now runs on a second stream, forked once the owner's rows are
quantized and joined just before the owner adds it to the MoE sum, so it
overlaps the counts, re-shuffle and dispatch rounds, which mostly wait on
the network. It is compute only: one collective is still in flight at a
time. Same kernels on the same inputs, so the same bits. Three interleaved
runs: 13,613 / 21,610 / 23,612 tok/s against 13,383 / 21,187 / 23,496 for
the route build and 13,127 / 21,222 / 23,318 for the build before it; scores
identical at 56 deep and 30 fresh positions.

Attention two head groups a pass (2026-11-18). Each owner ran the attention
kernel four times a layer, once for each group of 16 of its 64 heads, and
each pass re-read every row's selected and window keys (about 640 keys of 1
KB a row): the kernel moves 32 KB of keys a tile and spends about 2 us a
tile, so it was bound by the key reads, not the multiplies. Prefetching keys
two tiles ahead into L2 from the issuing warp made it slower (12,942 /
19,629 / 23,030 tok/s against 13,074 / 21,039 / 23,684 in one pair), because
that warp's own per-key slot lookups are on the path. The kernel now takes
two head groups a pass (`D41_ATTEND_TILES`): each consumer warp keeps one
16-head query tile over half the width, as before, and takes two 8-key
column tiles of the scores instead of one; the softmax runs its 16-lane
trees once a head group; each value block loaded for the second product
serves both groups. Every head's MMA chains and reduction trees are the ones
the one-group kernel runs, so `tests/test_dsv41_window_fold.py` compares two
groups a pass with one group a launch bit for bit (outputs and MXFP8 codes,
one and two sequences, 17 to 1,024 rows). Three interleaved runs: 13,649 /
21,600 / 24,202 tok/s against 13,614 / 21,450 / 23,768 for the previous
build; the gain is at long contexts, where every row reads its 512 selected
keys. Scores identical at 56 deep and 30 fresh positions.

Whole caches skip the shard arithmetic (2026-11-18). In rows mode every rank
holds the whole compressed cache (shard degree one), but each slot lookup
still ran the shard arithmetic: the owner check, the local position and the
slot in a page divide by the run-time grain and degree, and the page stride
is a 64-bit division. With degree one, `LmKvShardSlotRequired` now takes the
plain page lookup and `SparkKvShardLocalPosition` returns its argument; both
give the same address and run the same page-table checks. No measurable
change in the lane (13,682 / 21,435 / 24,143 tok/s with it, three runs);
scores identical.

Backup expert copies take rows off the busiest expert rank (2026-11-18). A
16K step's routed load is far from even: the hottest expert of a layer can
serve a third of the step's rows, so in the route dump the busiest rank's
experts ran about 8.0 ms a layer against 5.9 on average, every MoE owner
waited for it in the return round, and the dispatch and return rounds are
padded to the busiest pair of ranks (523 rows a pair against 336 on
average). Each expert of layers 0-19 now has one backup copy on rank (home +
1 + slot mod 15) mod 16, 24 experts a rank, 9.0 GB; the copies cross the
mesh once, from the home ranks' packed cells, on the first step that
exchanges (about 0.6 s, inside the lane's warm-up prefill).

A MoE owner may now send a row whose only route into a rank is one expert to
the rank holding that expert's backup copy instead. Dispatch and return
records are one per row and rank of the row's experts; a backup record
carries the row's codes and scales and only the flagged expert (the other
routes name no expert), the backup rank multiplies it with the copy's cells
on the same kernel (`LmCellMxExpertsSplit` reads groups past the split from
the copies), and its finalize rounds the one expert output to BF16 exactly
as the home rank's finalize would. The MoE owner still adds the ranks'
partials in rank order, reading each from whichever rank computed it, so the
sums are the same bits wherever a row went.

Every rank plans the moves from the same numbers. The counts round now
carries, besides the rows each owner sends each pair of ranks, the routes of
every expert and, for each MoE owner and expert, the rows whose only route
into the expert's rank is that expert (7,936 bytes a rank); the ranks sum
them on the device, and the host reads them once an event says the copy has
landed, without waiting for the re-shuffle round. The plan
(`D41BackupPlan`) balances a load model of 93 route units for an active
expert (one weight read, about 66 us, against about 0.71 us a route) plus
its routes: up to 64 times it takes the busiest rank and moves half the gap
to the backup rank of that rank's expert whose backup is least loaded,
split over the MoE owners in proportion to their candidate rows. Each MoE
owner sends the first such rows in row order up to its quota, and the plan's
pair counts size both rounds. On the route dump the plan takes the busiest
rank from 8.0 to 6.4 ms a layer and the padded pair from 523 to 433 rows.

`tests/test_dsv41_exchange.cu` runs every case with no backups, with the
plan and with random quotas up to every candidate row (up to 70,188
offloaded routes in a case); expert outputs there depend only on the token
and the expert, and the MoE owner's sums equal the sums computed without
backups at every element. Three interleaved runs: 14,411 / 23,372 / 25,990
tok/s against 13,664 / 21,866 / 24,158 for the previous build (+5.5%, +6.9%,
+7.6%); scores identical at 56 deep and 30 fresh positions, and the
40,000-token answer unchanged.

A second pass of the plan, moving single rows off the largest pair while the
backup's load stays under the new busiest load, takes the padded pair to 418
rows on the dump; at about 14 us a padded row a layer that is worth about
0.6%, under the run-to-run spread, so it is not in the code yet.

A leaner expert loop (2026-11-18). The block-scaled expert kernel
(`LmCellMxKernel`) issued 506 instructions for each 32 MMAs of a warp's 32-K
tile: besides the 128 FP32 adds the publisher's arithmetic needs (each
32-block an MMA from zero, added in FP32), it recomputed the tile's address
with a run-time division by the segment's tile count, rebuilt shared-memory
addresses from the shared window each tile, shifted and masked each row
chunk's scale byte out of its word, and spent about 16 instructions
expanding each word of FP4 weights. The loop now runs the four pipeline
stages unrolled, so each stage's slot and each MMA's scale byte are
constants (the instruction picks the byte from the word, loaded once per
four tiles), walks the weight cells by a tile stride from one pointer,
keeps 32-bit shared addresses, and expands a word of FP4 weights with two
shifts and masks and two byte permutes. 1,432 instructions per 128 MMAs,
11.2 an MMA against 15.8. The MMAs and the order of the FP32 adds are the
same, so `tests/cuda/cell_mx_cuda.cu` still finds the publisher's FP4 GEMM
bits in every output, with and without split copies.

Measured alone on one Spark with routes in the order the route build gives
them: w13 of an even rank (6,144 routes over 24 experts) 2.66 to 2.52 ms, w2
1.34 to 1.21 ms; a rank with a hot expert (8,000 routes, 2,400 on one
expert) 3.73 to 3.20 ms and 1.81 to 1.54 ms. Steps of a few routes an
expert are bound by reading the weights and do not change.

Two things tried and dropped on the way: passing the scale byte to the MMA
as a register (the compiler moved it into a uniform register around every
MMA and spilled), and staging activation rows interleaved four bytes at a
time so one 128-bit load gives the MMA's operand quad (it removed the
register moves but the four-byte copies cost more than they saved). A
register-only probe puts the exact pattern's ceiling at 182 TFLOPS
(chained accumulators 211); L2 delivers about 1.8 TB/s and DRAM 230 GB/s on
a Spark, so neither the expert kernel (~0.7 TB/s from L2) nor the dense
MXFP8 GEMM (~0.9 TB/s) is bandwidth-bound.

Three interleaved runs: 14,370 / 23,773 / 26,343 tok/s against 14,112 /
23,060 / 26,005 for the backup-copy build in the same session (its figures
above are an earlier session's); scores identical at 56 deep and 30 fresh
positions.

The plan's second pass shrinks the padded pair (2026-11-18). The dispatch
and return all-to-alls pad every pair of ranks to the busiest pair; a padded
row is 16 x (5,328 + 10,240) bytes a layer for every rank, about 10 us of
wire at the ~25 GB/s a Spark sends, while a route of expert work on the
busiest rank costs about 0.6 us. After the load balancing, the plan now
takes single rows off the largest pair (MoE owner, rank) to the backup rank
of one of that rank's experts, as long as the row's only route into the rank
is that expert, the backup's pair stays below the largest, and the backup's
load stays within two expert weight reads (186 route units) of the busiest
load the first pass left. On the route dump the padded pair goes from 433
(first pass only) to 394 rows; a slack of 0, 500 or 1,000 route units gave
418, 386 and 382 rows at a higher busiest load, and two weight reads had the
lowest modelled cost (0.6 us a route of the busiest load plus 10 us a padded
row). `tests/test_dsv41_exchange.cu` runs the plan with the pass in every
case (a full 16K step with uniform routing pads to 352 rows instead of 376).

Three interleaved runs: 14,861 / 23,733 / 26,559 tok/s against 14,451 /
23,433 / 26,140 for the expert-loop build in the same session; scores
identical at 56 deep and 30 fresh positions. This replaces the note above
that the second pass was not in the code.

The owner's eight o_a groups in one launch (2026-11-18). Each owner
multiplied its 1,024 rows by the eight o_a groups as eight MXFP8 GEMMs of
64 tiles each (4,096 to 1,024, 128 x 128 tiles), so each launch ran one full
wave on the 48 SMs and a second wave a third full. `LmMxGemmGrouped` runs up
to eight same-shape GEMMs in one persistent grid: a tile's group gives its
activation, weight and scale pointers and its output column offset, and
every output element is the same sum in the same order, so
`tests/cuda/mx_gemm_cuda.cu` compares a grouped launch with one launch a
group bit for bit (eight groups of 1,024 x 4,096 x 1,024, three groups of
300 rows, one group).

Three interleaved runs: 14,805 / 24,030 / 26,975 tok/s against 14,554 /
23,935 / 26,717 for the capacity-pass build in the same session; scores
identical at 56 deep and 30 fresh positions. On one Spark the 4,096-to-1,024
groups ran 0.109 ms each as eight launches. Two other things measured for
the dense MXFP8 GEMM and not taken: stages 64 wide in K (four or five
stages, or 128 x 256 tiles in three) were all slower than the 128 x 128 x
128 three-stage tile (65 to 87 TFLOPS against 107 on the owner shapes), and
the best existing tile per shape would save 2% of the dense time.

A second backup copy of each expert (2026-11-18). With one copy, all of a
hot expert's single-route rows that leave its rank go to the same backup
rank, which caps how far the plan can even out both the busiest rank's
experts and the busiest pair. Each expert of layers 0-19 now has a second
copy on rank (home + 8 + slot mod 8) mod 16: every rank holds 24 second
copies (three experts from each of eight home ranks) and never the first
copy of the same expert, so a rank holds 72 expert groups (its own 24, then
the two sets of copies, 18 GB of copies a rank in all; a lane run had at
least 36.7 GiB free on spark0 before). The copies cross the mesh in four
rounds a layer on the first exchanged step, half a rank's slots at a time
so no pair carries more than two experts a round.

The plan now chooses, for each move, the expert and the copy whose rank is
least loaded, splits each copy's rows over the MoE owners by the rows each
still has, and the capacity pass may take a row to either copy. A MoE owner
sends an expert's first quota of single-route rows in row order to the
first copy and the next quota to the second. On the route dump the
modelled MoE cost of a layer drops from 9.49 to 9.05 ms (busiest rank's
experts 5.54 to 5.28 ms, padded pair 395 to 378 rows).
`tests/test_dsv41_exchange.cu` places both copies and runs planned and
random quotas for each copy (up to 93K offloaded routes in a case); the
sums equal the sums without backups at every element.

Three interleaved runs: 15,189 / 24,317 / 27,367 tok/s against 14,635 /
24,088 / 27,009 for the grouped-o_a build in the same session; scores
identical at 56 deep and 30 fresh positions. The lane's first prompt pays
the fill once (an 8K warm-up prefill at about 2,500 tok/s instead of 5,600).


Two prefill lanes on two collective bands, measured and not kept
(2026-11-18). The step's rows were split into two lanes, the second lane's
rounds went to band 1 of the same mesh lane on its own stream (no turn
taking), each layer started lane 1 once lane 0 had written its cache rows,
and each lane had its own planning buffers and shared-expert stream. Three
interleaved pairs: 13,744 / 23,744 / 26,822 tok/s against 15,054 / 24,730 /
27,584 for the same build without lanes (-8.7 / -4.0 / -2.8%), scores
identical at 56 deep and 30 fresh positions. A traced 131K prompt on rank 0
shows why: the overlap hid 1.04 of 1.84 s of round waits (654 ms left
exposed against 1,024 without lanes), but the compute union grew from 3.19 to
3.84 s, because each lane reads every expert's and every projection's
weights again (about 60 ms a 16K step) and the kernels of the two lanes slow
each other. Layers 20-39 run only on the final rows, all in lane 1, so the
tail never overlaps. The patch is kept outside the tree; lanes would pay
only if both lanes shared one expert pass a layer.

Where a 16K step of the current build goes (rank 0, a 131K prompt, 599 ms a
step): round waits not under compute 128 ms, experts 111, projections 84,
attention 55, round copies 50, mHC 36, index scores growing from 3 to 39 ms
with the context. The MoE return round's first piece waits a median 1.1 ms
for the slowest rank's experts; every later piece takes the wire time of its
7.7 MB (about 340 us), so past that wait the rounds run at wire speed and the
copies hide under the wire. An 8K prompt is one prefill step (483 ms on rank
0: layers 0-19 299 ms, the tail 184 ms) plus a one-row step for the prompt's
last token, because the batch engine checkpoints the state at N - 1 tokens.
The tail, layers 20-39 on the last 2,541 rows, costs 184-213 ms a prompt
(38% of an 8K prompt, 16% of 32K, 4% of 131K); about 30 ms of it is waiting
for the rank that holds a hot expert, since layers 20-39 have no backup
copies.

Tried on one Spark and not kept, all bit-identical: the expert kernel's A
operand by ldmatrix with the B nibbles regrouped (1,052 instead of 1,478
instructions per 128 MMAs, no faster: 2.557 -> 2.547 ms), eight route slots
for every expert call (only a very hot tail expert gains, -8% on w13, +5% on
w2), and 64-wide K stages (w13 -8.5%, w2 +4.6%, tails +2-4%). Modelled on the
route dump and not built: experts placed by co-selection (a row touches 4.0
instead of 5.2 ranks, but the busiest rank rises 1-1.6 ms a layer and the
modelled MoE cost goes from 36.1 to 36.9 ms over four layers), and the
publisher's single FP32 sum with every route free to move (busiest rank 6.2
-> 6.0 ms, but per-route returns and unbalanced pairs raise the modelled
layer from 10.0 to 11.6 ms).

Index scores from keys staged in shared memory (2026-11-18). Each of the
block-slot scorer's eight row warps loaded every key of its eight-key tiles
with 4-byte loads that each touched eight key slots, so the scorer ran at 42
TFLOPS (1,024 rows by 32,768 keys in 6.53 ms on one Spark). When a block's
rows share one sequence the block now copies its keys 32 at a time into
padded shared rows (272 bytes, three stages, 16-byte copies through the slot
table it already builds) and each warp takes its key fragments by ldmatrix,
two k steps an instruction; the earlier whole-block staging (69.6 KB, no
copies in flight) was slower, this one keeps two stages in flight in 26 KB.
The MMAs, their order and the epilogue (now `LmIndexScoreStore`, shared by
every path) are unchanged, and a block spanning two sequences keeps the
per-warp lookups. `tests/cuda/index_score_slots_cuda.cu` still finds every
score of its nine cases equal to the per-row scorer. One Spark: 6.53 to
3.97 ms at 32,768 keys, 1.65 to 1.02 at 8,192, 1.04 to 0.64 for 160 rows
(69 TFLOPS). The scores grow with the context (3 ms in the first 16K step of
a 131K prompt, 39 ms in the eighth), so the gain grows with prompt length.
Three interleaved runs each: 15,064 / 24,677 / 28,019 tok/s against 14,827 /
24,574 / 27,508 for the backup-copy build in the same session; scores
identical at 56 deep and 30 fresh positions.

Records with one local route skip the finalize, and the mHC partials take
eight rows a block (2026-11-18). A dispatch record whose row routes to a
single expert of the receiving rank needs no sum, so the down multiply now
writes that record's BF16 output straight into its slot of the return buffer
(`LmCellMxCodes.output_rows`, a mapped row) and the finalize sums only the
records with two or more local routes, in the same route order. The
Sinkhorn partials re-read the mixer's FP32 weights for every block, so above
512 rows a block now takes eight rows instead of four.
`tests/cuda/cell_mx_cuda.cu` checks mapped rows and strided codes against
the publisher's bits, `tests/test_dsv41_exchange.cu` the finalize beside
mapped single-route records, and `tests/test_dsv41_hc_mix.cu` the eight-row
partials. Three interleaved runs each: 14,920 / 24,690 / 28,001 tok/s
against 14,908 / 24,461 / 28,036; scores identical at 56 deep and 30 fresh
positions. Level within the gate's spread; kept because it is exact and does
less work.

The gate and up experts read their codes from the landing records
(2026-11-18). The dispatch unpack copied each landed record's FP8 codes and
scales into a packed buffer before the w13 multiply; the multiply now
gathers them from the records (a code stride and a scale stride per record)
and the unpack writes only routes and weights. Three interleaved runs each:
14,920 / 24,871 / 28,012 against 14,885 / 24,783 / 28,080; identical scores.
Level, kept for the same reason.

The mHC mixes on a side stream beside the MoE rounds, measured and not kept
(2026-11-18): every L2-bound kernel ran slower beside the rounds' copies,
14,348 / 23,823 / 26,905 against 15,047 / 24,722 / 27,908, identical scores.

One decode graph for every prompt length (2026-11-18). A prompt's last token
reaches the stage as a one-row prefill step that carries the prompt's length
in `gather_context`, and the runner's graph identity compared that field, so
each prompt whose length differed from the previous one recaptured and
re-instantiated the 3,000-node one-row graph before its last token (6 to 15
ms of host time with every GPU idle), and the first decode step after it
recaptured again. DSV4.1 never reads the field, so its geometry now sets
`graph_ignores_gather` and the runner leaves both gather fields out of the
identity; `tests/test_graph_ignores_gather.py` fails if a family that sets
the flag reads either field. In a gate run the graphs captured fall from
five to three (the lane's first prompt, and the last prompt step and first
decode step of the request that asks for log-probabilities, which has its
own slot), and the last step of a 32K and a 131K prompt costs 56.3 and 58.0
ms of wall time instead of 70.4 and 69.3. Three interleaved runs each: 15,074 / 24,799 / 28,317 tok/s
against 14,956 / 25,024 / 28,292; the saving is inside the gate's spread at
the whole-prompt level. A prompt's last step and a decode step also differ
in `last_rows_only`, so a request that went on to decode recaptured at its
last prompt token and again at its first decode step whatever the gather
fields said; the runner's graph slots now key on that flag too, so each kind
keeps its own graph.

A one-row step spread over the ranks (2026-11-18). With every attention
weight on every rank, a one-row step (a prompt's last token, and every
decode step) ran all 64 heads and read q_b, o_a and o_b in full on all
sixteen ranks, 6.7 GB of dense FP8 weights a step, and rank 0 alone ran the
shared expert while the others waited for it at the MoE gather: 54 ms a
step, about a tenth of an 8K prompt. Each rank now computes the query and
attention of its own four heads, half of its head group's o_a rows and a
sixteenth of o_b's, from the same weights, inputs and kernels (the one-row
multiply's lane count and neurons per group depend only on K, so every
output is the same sum), and three all-gathers (merged heads, o_a, o_b)
rebuild the vectors every rank needs. q_a, kv_a, the indexer and the
selection stay whole on every rank. The last step of an 8K, 32K and 131K prompt takes 32.7, 35.2 and 36.2 ms on the
ranks instead of 53.4, 55.2 and 56.8, and a decode step 34.2 ms instead of
54.1 (decode on the sixteen ranks from about 18.5 to 29 tokens a second).
Three interleaved runs each: 15,759 / 25,366 / 28,482 tok/s against 14,994 /
25,094 / 28,120, the same generated answer, and scores identical at 56 deep
and 30 fresh positions, which score the last step's own logits. Spreading the shared
expert of a one-row step the same way (a sixteenth of its rows a rank, two
all-gathers, rank 0 adding the whole output where it did before) was exact
but level, 15,474 / 25,180 / 28,323 against 15,429 / 24,934 / 28,171 with
the same decode time: once rank 0 no longer waits on its heads, the ranks
holding the token's routed experts set the pace, so it is not kept.

The side-stream shared expert yields SMs at every tile (2026-11-18). The
shared expert's three MXFP8 multiplies run on a side stream beside the MoE
counts and shuffle rounds, and their persistent grid held every SM until a
whole multiply was done. The round's wait and copy kernels on the main
stream, which already has the higher priority, could not start until then:
on every rank of a traced 131K step the counts round's wait kernel started
about 220 us after its publish and the round took 460-810 us for 8 KB.
`LmMxGemmDenseGrid` keeps the dense multiply's tile choice but takes its own
grid limit, and the side stream now runs one tile a block, so the main
stream gets an SM at each tile boundary; `tests/cuda/mx_gemm_cuda.cu`
compares every tile shape between persistent blocks and one tile a block.
The runner also keys its graph slots on `last_rows_only`, so a request's
last prompt step and its decode steps keep separate graphs instead of
recapturing each other's. Three interleaved runs each: 15,600 / 25,313 /
28,458 tok/s against 15,474 / 25,203 / 28,313; scores identical at 56 deep
and 30 fresh positions. Within the gate's spread at the whole-prompt level.

The owned tail re-splits as it shrinks (2026-11-18). Layers 20-39 run only on
the prompt's last 2,541 rows, and each later layer needs 127 fewer of them,
but once those rows were split sixteen ways at layer 20 every later layer
kept 160 rows a rank: contiguous blocks cannot simply drop rows, since the
low ranks would idle while the high ranks keep full blocks. At each indexer
layer (24, 28, 32, 36), where every row's sparse selection is recomputed
anyway, the remaining rows are now split sixteen ways again by the first
split's pack, all-to-all and unpack, starting from the previous split, so a
rank's rows go 160, 128, 96, 64, 32. Each row's candidate-block mask moves
with it: only the row's owner at layer 20 computes it, and a long prompt's
later indexer layers apply it, so the first build, which moved only the
streams, changed 48 of the 56 deep scores of prompts from 18K tokens up (the
argmax stayed the same) while the 6K prompts and the 30 fresh positions
stayed identical. A mask row at the lane's context capacity is wider than
a collective row may be (the hidden width), so the second build's mask
all-to-all was refused at the first re-split of a 32K prompt; the masks now
travel as whole hidden-wide parts of a padded record, as the streams travel
as their four streams. `tests/test_dsv41_exchange.cu` re-splits rows and
padded masks from an owned layout at each step and checks every row and
mask lands with its new owner once. Three interleaved runs each: 16,797 /
26,154 / 28,590 tok/s against 15,755 / 25,604 / 28,284, the same generated
answer, and scores identical at 56 deep and 30 fresh positions.

Where the hub API's time between steps goes (2026-11-18, measured, nothing
changed). Outside the ranks' own step time a prompt spends 13.5 ms at 8K
(2.6%), 20 ms at 32K and 76 ms at 131K (1.7%). An instrumented API build
timed each step: between a chunk's last completion and the next chunk's
dispatch the engine itself takes 0.4-0.5 ms, building the submission
0.2-0.3 ms, but handing a 16,384-row chunk to the sixteen ranks takes
3.8-9.2 ms (a one-row step 0.1-0.2 ms, a request's first chunk 1.5 ms).
Only a request's first chunk prepares and commits; later chunks continue on
their lease. Each chunk is about 400 KB a rank (24 bytes a row: token, lane,
position, sequence), copied into each rank's queue and sent sixteen times;
one sequence's prefill rows could travel as their tokens alone, about 64 KB.

The mHC Sinkhorn partials stage their inputs (2026-11-18). The row-blocked
partials kernel kept its 200 accumulators (25 mix tasks for eight rows) on
the stack, an 800-byte frame a thread, and read the mixer's FP32 weights
from memory for every element. `LmHcSinkhornPartialStagedKernel` copies
each 256-element step of the mixer weights and the block's rows into shared
memory three steps ahead and splits the 25 tasks over two 256-thread
groups, so the accumulators stay in registers (128, no spills); thread `t`
of each group still sums elements `t`, `t + 256`, ... in order and the
warps reduce as before, so every partial is the same bits
(`tests/test_dsv41_hc_mix.cu`, at the model's width and at a width whose
last step is partial). On one Spark a call takes 0.29 ms at 1,024 rows
instead of 0.47, 0.19 at 512 instead of 0.26, and the same 0.08 at 160.
Three interleaved runs each: 17,129 / 26,594 / 29,448 tok/s against 16,413 /
26,025 / 28,565, the same generated answer, and scores identical at 56 deep
and 30 fresh positions.

The one-row window fold runs as an eight-block cluster (2026-11-18). A
one-row step (a prompt's last token, and every decode step) folds each
head's 128-key window into its attention record in
`D41WindowFoldWholeKernel`, which ran one block a head through the keys in
sequence: 63-76 us a layer, about 3 ms of a 30 ms step, on four blocks
once the heads are spread over the ranks. In a spread step each head now
runs as a cluster of eight blocks (`D41WindowFoldClusterKernel`): each block scores sixteen of the keys with the same
per-key reduction, the blocks read each other's scores from distributed
shared memory, and each folds its own 64 elements of the latent in key
order, so every record is the same bits (`tests/test_dsv41_window_fold.cu`,
whole-window cases from early positions to ring keys, against the
key-by-key fold). On one Spark the fold of one row and four heads takes 30
us instead of 63; with all 64 heads of a row the cluster form is slower (218
us against 69, 3.4 ms against 0.49 for sixteen rows), so steps that are not
spread keep the one-block-a-head kernel. Three
interleaved runs each: 16,857 / 26,931 / 29,115 tok/s against 16,993 /
26,576 / 29,415, level within the gate's spread (a prompt has one one-row
step), while the gate's decode request (a 200-token prompt and 16 tokens)
took 2.04 s against 2.08; scores identical at 56 deep and 30 fresh
positions. Kept because it is exact and shortens every decode step.

The prompt's last token runs in the last wave (2026-11-19). The engine
used to stop a prompt's last wave one token before its end, publish the
cache checkpoint there, and run the last token as a one-row step: about 33
ms of step and up to 9 ms of gap on an 8,192-token prompt. The last wave
now runs through the last token and publishes the checkpoint one token
before the end from inside it (`docs/INTERFACE_CONTRACTS.md`, "Folded
prompt checkpoint"); the rank saves the two pieces of DSV4.1's state that
differ there, the ratio-2 compressors' pending row and the n-gram history.
In this first build a prompt whose last token or the one before it ended a
256-token page kept the one-row step, so 8,192, 32,768 and 131,072 tokens
did not fold; the runs below use 8,100, 32,100 and 131,000.

Both arms run the same rank build and an engine with no cache stride, as
every figure since the cache-stride table; only the fold capability
differs:

| Arm | 8,100 tokens | 32,100 tokens | 131,000 tokens |
| --- | ---: | ---: | ---: |
| one-row last step | 16,740 tok/s | 26,397 | 28,716 |
| folded | 17,790 | 27,097 | 29,242 |
| one-row last step | 16,926 | 26,541 | 29,405 |
| folded | 17,801 | 27,003 | 29,764 |
| one-row last step | 16,803 | 26,530 | 29,341 |
| folded | 17,967 | 27,100 | 29,732 |
| mean, one-row | 16,823 | 26,489 | 29,154 |
| mean, folded | 17,853 | 27,067 | 29,579 |

That is +6.1%, +2.2% and +1.5%, and 237%, 320% and 352% of the four-Spark
reference. At 8,192 / 32,768 / 131,072 tokens the two arms are level
(16,887 / 26,607 / 29,133 against 16,639 / 26,333 / 29,426), and every run
answers the 40,000-token prompt with `b79b025295d2819f`.

The checkpoint is exact. With the capability off the build scores the deep
and fresh probes as `zn18d1` and `zn18f1` bit for bit (56 and 30
positions). Seven prompts sent twice in one session, the second time each
resuming at the checkpoint the first folded, score bit for bit as the same
prompts on the one-row engine. The first output token is computed in the
wave instead of a one-row step, as the publisher's forward pass does, so
its log-probs move: the seven prompts have the same best token and differ
by at most 0.75 in a log-prob; on the deep probes, whose 16-token
extensions now run as 17-row waves past the 16-row kernel switch, the best
token is the same at 56 of 56 and 29 of 30 positions and a log-prob moves
by at most 3.3.

Prompts that end on a page edge fold too (2026-11-19). When N - 1 ends a
page, that page is sealed as the checkpoint with its state and token N - 1
stays in the open page after it. When N ends a page, the full page is
sealed without state and the checkpoint goes onto a device copy of it. A
wave that ends on the 64th page names 64 block identities; the first build
held 63 and refused 32,768- and 131,072-token prompts. Same rank build, no
cache stride, fresh prefix `fk19`, three runs each:

| Arm | 8,192 tokens | 32,768 tokens | 131,072 tokens |
| --- | ---: | ---: | ---: |
| one-row last step | 17,213 tok/s | 26,445 | 29,412 |
| folded | 18,308 | 26,950 | 29,262 |
| one-row last step | 16,832 | 26,682 | 29,029 |
| folded | 17,464 | 27,162 | 29,755 |
| one-row last step | 17,252 | 26,517 | 28,892 |
| folded | 16,382 | 27,127 | 29,443 |
| mean, one-row | 17,099 | 26,548 | 29,111 |
| mean, folded | 17,385 | 27,080 | 29,487 |

That is +1.7%, +2.0% and +1.3% (the folded 8,192-token runs range from
16,382 to 18,308), and 231%, 320% and 351% of the four-Spark reference.

Exactness at the page edges, from repeats and from rank 0's saved
snapshots of both engines compared byte for byte:

- N - 1 ends a page (8,193 to 69,633 tokens): every saved page is the same
  as the one-row engine's, and the state records differ only in the window
  ring's slot for token N - 1, which a resume rewrites. Repeats that resume
  at a held checkpoint score bit for bit as the one-row engine. The store
  keeps the four newest states of a chain, so the two oldest checkpoints
  of the six-prompt chain were gone by the repeats; those two repeats
  folded from an earlier point, and their first tokens moved as every
  folded first token does.
- N ends a page (8,192 to 69,632 tokens): the checkpoint's state and its
  copied page differ only in token N - 1's slot, which a resume rewrites.
  The full page ending at N holds token N - 1's KV from the folded wave.
  The one-row engine seals that page only when the next request, resuming
  at N - 1, recomputes token N - 1 as the first row of its own wave, and
  the two computations differ in low bits. A prompt that extends past N
  reads the folded value. In the six-prompt chain the repeats of the
  first two prompts score bit for bit as the one-row engine; from the
  third on, whose cached chain passes through an earlier folded page, they
  differ in low bits (same best token 6 of 6, at most 1.41 in a log-prob).
  The cache keeps the first copy of a block it seals, so a chain can hold
  the folded value next to a checkpoint that a later wave computed with the
  recomputed one; the one-row engine has the same mix when a reply's
  one-row steps seal a block that a later prompt recomputes.
- Folded runs repeat bit for bit across lanes (two more lanes, `fm19`).

The engine of these runs has no cache stride, as the module now has (next
section).
The module drops the 16,384-token cache stride (2026-11-19). Every
16,384-token wave ended at a cache point and ran layers 20 to 39 on its
last rows so that the recurrent state could be saved there: 17,284 /
23,990 / 23,314 tok/s at 8,192 / 32,768 / 131,072 tokens on the sixteen
ranks, against 16,955 / 26,741 / 29,382 with no stride (the engine every
figure here since the cache-stride table has used). The stride is now
1,048,576 tokens, past any lane's context. A request still resumes where an
earlier prompt or reply ended and at the cache points a request declares;
it no longer resumes at a 16,384-token mark inside a prompt it shares only
in part.
