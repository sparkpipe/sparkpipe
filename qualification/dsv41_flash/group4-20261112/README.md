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
rows took 4.80 to 4.83 s. Removing the launches moved the rounds from the
GPU's pace to the daemon's: a piece now waits on weightd posting and
completing three 131,072-byte writes. The 1 MiB slots (PR 1932, weightd IPC
ABI 15) cut those writes to a quarter and measured 23 to 28% a round on the
Lenovo pair; they need the daemon replaced on the nodes.

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

