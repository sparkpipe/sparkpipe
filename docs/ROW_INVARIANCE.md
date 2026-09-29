# Row invariance (batch invariance)

A row's output bits must not depend on how many other rows share its wave or
which rows they are. Exact speculation (verify rows equal B1 rows) and
provider replay (a verifier replays one request alone) both need this. See
TECHDEBT.md, "Dynamic batching", for the rules adopted from TensorFold.

## Harness

`make test-glm5-next-row-invariance` builds and runs
`tests/test_glm5_next_row_invariance.cu` on a GB10 (sm_121a). It includes the
glm5_next resident stage and calls its production entries in rank geometry
(TP16: 4 MLA heads, 1/16 of the vocabulary, 1/16 of every expert and shared
intermediate):

| Family | Entry | Sites |
|---|---|---|
| `dense_linear` | `Glm5NextLaunchBf16LinearRows` for the nine per-rank dense shapes | output |
| `dense_mlp` | `Glm5NextLayerDenseMlp` | output |
| `moe` | `Glm5NextLayerMoeRoute` then `Glm5NextLayerMoeExperts` (FP8 experts, shared expert, finalize) | router logits, route (expert ids and weights), output |
| `attention` | `LmLatentAttentionHeadsLaunch` as the layer calls it: 16 sequences of four consecutive positions each (so a wave can hold a multi-row run of one sequence, as MTP verify and prefill do), contexts from under 64 to past 2,048 with per-row 2,051-slot selected lists, the context of each sequence set by its last row in the wave; every wave is also run with one, two and four heads per block and must give the same bytes | latent output |
| `head_greedy` | `SparkGlm5NextRunHead`, greedy rows (B1 takes the certified FP8 path) | maxloc word, token |
| `head_sampled` | `SparkGlm5NextRunHead`, every row sampled | maxloc word, token |
| `head_mixed` | `SparkGlm5NextRunHead`, half the rows sampled | maxloc word, token |
| `head_ties` | `SparkGlm5NextRunHead`, greedy, every head row duplicated so each argmax is an exact tie; the lower token id must win | maxloc word, token |

Each family has a pool of 64 rows with fixed random inputs. Every pool row is
first run alone (wave 1). Waves of 2, 8, 17 and 64 are then drawn from the
pool (three compositions each, one for 64) and every row's bytes are compared
with its wave-1 bytes. Each composition is run twice; a family that is not
repeatable fails outright, so a wave dependence is never confused with
nondeterminism.

Output: one line per cell, `ROWEQ cell=<family>.<site>@<wave>
changed_rows=<n>/<total> verdict=EQUAL|BREAK`. `--report` only prints. `--run`
fails when a cell breaks that is not in `roweq_known_breaks`, or when a listed
cell no longer breaks. The list may only shrink: a fix PR removes the cells it
repairs in the same change.

## Baseline (origin/main 09fdad6, sparkf, 48 SMs)

After the skinny-rows fix, 12 cells remain: attention and the greedy and
mixed head maxloc words. After the head fix, only the four attention cells
remain. After the attention fix, none remain (88 cells, 0 breaks). The
baseline attention family was the simpler decode-only version; the current
one also covers runs of one sequence and selected lists.


38 of 80 cells break:

- Dense projections, the dense MLP, the router logits, the route and the MoE
  output at 17 and 64 rows: above eight rows the skinny GEMV declines and the
  tensor-core GEMM runs with a different accumulation order. Waves of 2 and 8
  are equal.
- Attention at every wave above 1: a decode wave uses the all-heads kernel,
  one row the per-head kernel, with different partition counts.
- Greedy head maxloc at every wave above 1: B1 uses the certified FP8 screen
  and lane-split rescore, more rows the sequential-k rows kernel. The token was
  equal in every sampled composition; the score word was not, and TP16 compares
  scores across ranks.
- Greedy rows in a wave with a sampled row (`head_mixed`) move to the sampled
  kernel. Sampled rows alone are equal at every wave.

## Fixes

### Dense projections and router (skinny rows)

`LmSkinnyDenseRows` (inference/kernels/skinny.cuh) keeps the one-row skinny
arithmetic at every row count: lanes are chosen from the input width only, lane
`sub` sums chunks `sub + i*LANES` in ascending order, every chunk starts from
+0, and the lanes meet in the same XOR butterfly. Rows 1-8 still take
`LmSkinnyDense`. Above eight rows `LmSkinnyRowsKernel` gives each thread eight
rows and one, two or four neurons; the grouping only changes reuse, never the
per-element chain. There is no GEMM fallback: a shape it cannot take is an
error. glm5_next uses it through `Glm5NextLaunchBf16LinearRows` for all 19
dense sites, MTP `eh_proj` and the router. ling and laguna keep
`LaunchBf16Linear` until they qualify their own change.

The routed-expert GEMM fallbacks are removed: a mesh wave holds at most 128
rows, so at most 1,024 routed pairs, and the grouped skinny kernel takes
16 x 288 = 4,608 (a static assertion in the stage checks this).

Cost on sparkf (GB10, `make bench-glm5-next-batch`, TP16 rank geometry,
context 1,024, compute only, `--iterations 5 --copies 1`; the GPU was also
serving production, so steps vary by about 1 ms). Main was measured before and
after the branch:

| Rows | main step ms | branch step ms | change |
|---|---|---|---|
| 1 | 20.22 / 18.66 | 18.95 | unchanged path |
| 8 | 36.64 / 36.35 | 36.96 | unchanged path |
| 16 | 59.05 / 58.10 | 53.86 | -4.2 ms |
| 32 | 82.68 / 82.05 | 77.95 | -4.1 ms |
| 64 | 115.82 / 114.96 | 113.55 | -1.4 ms |
| 128 | 159.91 / 160.92 | 168.26 | +7.3 ms (about +57 us per prompt row) |

Decode waves of 16-64 rows get faster because the tensor-core GEMM wastes
most of its tile on the narrow projections (router, shared expert, indexer
keys). 128-row prefill waves pay for the wide projections (q_a, index_q,
attn_out, kda_qkv_beta). `make test-skinny-gemv` prints the per-shape table.

### Greedy head (exact rows)

B1 greedy takes the certified FP8 head: an FP8 screen with sound bounds, then
an exact BF16 rescore of the survivors (`SparkLmDotRowBf16`, a 32-lane
`fmaf` chain over pairs, then the warp reduction) and the lowest id on ties.
Because the bounds are sound, its token and score are the full-vocabulary
argmax of that rescore arithmetic. Waves of two or more greedy rows now
compute exactly that argmax without a screen: `SparkLmHeadExactRowsKernel`
stages eight rows of hidden state in shared memory, gives each warp four
vocabulary rows at a time, and keeps each (row, token) product in the same
lane order as the rescore. B1 is unchanged. Greedy rows in a wave with a
sampled row are computed the same way and replace the sampled kernel's greedy
result (`Glm5NextHeadGreedySelectKernel`). The one-row BF16 candidate kernel
now breaks ties by the lowest token id, like the rows kernel and the commit.

Cost (sparkf, per-rank vocabulary 9,680, random weights, median of 7 x 10
launches, RMSNorm included; B1 stays on the certified path at about 0.38 ms):

| Rows | old rows kernel ms | exact rows ms |
|---|---|---|
| 2 | 0.55 | 0.40 |
| 8 | 0.56 | 0.42 |
| 16 | 0.57 | 0.77 |
| 32 | 1.11 | 1.53 |
| 64 | 2.20 | 3.10 |

So waves of 2-8 rows get faster and a 64-row wave pays about 0.9 ms per step
on the rank that owns the head. Sixteen rows of hidden state would need
128 KB of shared memory, over the per-block limit; a K-slab version is the
next step if the 64-row cost matters.

### Attention (absolute key tiles)

`LmLatentAttentionHeadsKernel` now plans each row from the row alone:

- keys: `min(context_length[sequence], position + 1)`. A row takes its
  selected list only when that count exceeds `dense_limit` (2,048 for
  glm5_next) and the wave has a list, so a short row next to a long one stays
  dense;
- tiles: `ceil(keys / 256)`; tile `t` covers keys `[256 t, 256 t + 256)`; warp
  `w` takes keys `256 t + w, + 16, ...` in two-key steps;
- a one-tile row writes its output directly; a longer row writes one partial
  per tile, and `LmLatentAttentionTilesCombineKernel` merges tiles 0..P-1 in
  order;
- the grid is `(rows, tiles of the longest possible row, head groups)`. Blocks
  past a row's own tile count exit. How many heads a block handles (one, two
  or four) is picked from the wave size for speed; it never changes a head's
  arithmetic, and the harness checks that.

The per-head split kernel, `attention_decode_wave` and the split threshold's
numeric role are gone from glm5_next. `decode_split_context_threshold` still
shapes graph regimes and bounds, which no longer affect bits. A row whose key
count exceeds the planned tiles reports `PAGE_TABLE_OUT_OF_RANGE` through the
KV access error. The partials keep 16 tiles per row and head (the longest row,
a 2,051-slot list, needs 9).

This changes B1 attention numerics once. COMPSEC-17 and MTP parity must be
requalified on this reference.

Cost, per MLA layer (sparkf, 4 heads, all rows at the given context, median of
7 x 20 launches; the GPU was shared with another lane's run, so old numbers
varied between runs and the table gives the range):

| Rows | Context | old us | tiles us |
|---|---|---|---|
| 1 | 64 | 10-23 | 16 |
| 1 | 512 | 29-51 | 31 |
| 1 | 1,024 | 50-87 | 31 |
| 1 | 2,048 | 92-102 | 31 |
| 8 | 1,024 | 137-180 | 64-78 |
| 16 | 1,024 | 190-238 | 115-165 |
| 64 | 1,024 | 455-618 | 512-685 |
| 64 | 2,048 | 747-997 | 935-1,282 |
| 128 | 1,024 | 845-1,102 | 905-1,242 |

glm5_next has 11 MLA layers per rank. So B1 at 2,048 tokens saves about
0.7 ms per token and B1 below 128 tokens pays at most 0.07 ms; 8-16-row waves
save about 0.8 ms per step; 64-row waves at 2,048 tokens pay up to about 2 ms.
The next lever for large waves is occupancy: the four-head block uses 207
registers, so one block per SM.

## Full layer stack

`make test-glm5-next-row-invariance-stack` runs the 45-layer glm5_next stack
of `bench-glm5-next-batch` (TP16 rank geometry, synthetic weights, KV, index
and KDA state) with `--row-hash DIRECTORY`. The tool runs the wave once, then
runs every row alone with the KDA state restored. At 20 sites per layer it
writes a per-row 64-bit hash (`LmRowHashKernel`,
`inference/kernels/row_hash.cuh`: an order-free sum of SplitMix64 over
(site, word index, 16-bit word)). The sites are the q_a, q_b, kv_a, index
q/k/head and attention latent outputs; the KDA fused projections and state;
attention out; the HC hidden after attention and after the MLP; the router
logits, route and weights; shared and MLP out; and the head token and score.
`tools/row_hash_compare.py` joins the two traces on (step, layer, site,
sequence, position) and names the first divergence.

Results on sparkf with the whole stack of fixes: every entry is equal for 2,
8, 17 and 64 rows at 1,024 tokens (33,856 entries at 64 rows) and for 2 and 9
rows at 2,100 tokens (selected-list attention and the indexer). With the
attention fix removed, it reports `layer=3 site=attn_out` at 2 rows, and at
17 rows the divergence reaches every KDA and MLP site. So the stack harness
catches the breaks the kernel harness found.

## Whole stack cost

`bench-glm5-next-batch --iterations 5 --copies 1` on sparkf with the GPU
otherwise idle, main against this stack, two passes each. The numbers are
compute step ms in TP16 rank geometry. The bench has no certified head
payload, so its head column does not change.

| Rows | 1,024 tokens main | 1,024 tokens stack | 2,048 tokens main | 2,048 tokens stack |
|---|---|---|---|---|
| 1 | 18.4 / 18.5 | 18.0 / 17.7 | 19.4 / 20.9 | 18.6 / 20.6 |
| 8 | 35.5 / 35.9 | 35.6 / 34.8 | 37.7 / 36.2 | 35.7 / 35.7 |
| 16 | 59.3 / 59.3 | 51.6 / 53.2 | 58.1 / 57.5 | 53.3 / 56.3 |
| 32 | 79.6 / 80.5 | 76.0 / 75.8 | 81.9 / 82.1 | 80.3 / 76.9 |
| 64 | 114.2 / 113.6 | 110.9 / 110.9 | 119.1 / 118.8 | 118.3 / 125.7 |
| 128 | 157.5 / 158.5 | 169.3 / 168.3 | 166.0 / 167.8 | 220.5 / 184.7 |

Decode waves up to 64 rows are as fast or faster. 128-row waves (prefill
chunks) pay 7% at 1,024 tokens and more at 2,048 tokens, from the skinny order
on the wide projections and the attention tiles.

The tensor-core GEMM (`LmGemmLaunch`, BF16, the glm5_next tile) is row-invariant
across M on its own. At K x N of 4096 x 1536, 1536 x 4096, 1024 x 4096 and
4096 x 288, every row of M = 2, 9, 17, 64 and 128 equals the same row at M = 1,
bitwise (sparkf). So prompt rows could keep the tensor-core order and generated
rows the skinny order, and each row's bits would still depend only on the row
and its role. That would remove the 128-row prefill cost. It needs one of two
things: waves split by role, or two launches over row subsets. Generated-token
checkpoints would also have to stay out of prefix reuse for verified requests.

## Rules outside the kernels

TensorFold's rules (TECHDEBT.md, "Dynamic batching") beyond the kernel order,
and where each one is held:

- **Cross-GPU partial sums in rank order.** With `SPARK_TP_WAIT_MODE=hardware`
  (the fleet's serving mode) every BF16 sum takes the direct path, or
  reduce-scatter plus all-gather at 49,152 elements or more at degree 4 or
  more (`SparkTpMeshHardwareDirectRound`). Both combine through
  `SparkTpMeshHardwareDirectElement`: an FP32 sum from 0 over peers 0, 1, ...,
  degree - 1, truncated to BF16. RS+AG only changes which rank does the sum
  for a slice, and the gather copies the BF16 bits. So an element's sum is the
  same on both paths, and does not depend on how many rows share the payload.
  `make test-tp-mesh-reduction-order` checks this on one GPU with the real
  combine kernels: direct and RS+AG against a host rank-order sum at degrees
  4, 8 and 16 and 1-49 rows of 1,000-4,096 elements, each row reduced alone
  against the same row in the payload, and a negative control (the reversed
  peer order gives different bits for at least one element per case).
  The kv_shard attention merge (`LmLatentShardMergeKernel`) also folds the
  sources in rank order. Spin mode is not canonical: one-row payloads take the
  host round and larger ones the tree, whose fold order differs. Verified
  serving needs the hardware wait.
- **Router ties by token id.** Top-k already breaks score ties by the lowest
  expert id, one row per warp (`LmTopkWarpCandidate`). `LmRouteBuild` used to
  place each routed pair in its expert's group with `atomicAdd`, so the order
  of the rows inside a group depended on thread timing. It now places them in
  route order, which is (token id, k): the first warp walks the routes 32 at
  a time and ranks equal experts with `__match_any_sync`.
  `tests/test_route_build_scan.cu` requires that order; the old kernel fails
  it on the skewed case. The skinny expert kernels accumulate each row
  separately, so this did not change bits today; it makes the grouped layout
  a function of the batch alone, which a tensor-core grouped GEMM will need.
- **Load-time gate.** The glm5_next stage refuses to initialize when a
  multi-row wave differs from its rows run alone. After the weights load
  (and experts are pinned), `SparkGlm5NextRowGateCheck` runs
  `SparkGlm5NextLaunchCudaRowGate` on this rank's real weights: the MLA q_a
  projection of the first local MLA layer through the layer's own skinny
  multi / skinny rows path, and the first local routed layer's
  `Glm5NextLayerMoeRoute` and `Glm5NextLayerMoeExperts` (norm, router logits,
  top-k, route build, grouped or per-pair experts, shared expert, finalize).
  It runs 64 synthetic rows alone, then waves of 2, 8, 17 (twice, at rows 0
  and 40) and 64, and compares every row bitwise. It prints one line per rank,
  `ROW-GATE rank=.. q_a_layer=.. moe_layer=.. waves=5 rows=108
  checked=q_a:1,router:1,moe:1 mismatches=q_a:0,router:0,moe:0 launch=0
  verdict=EQUAL`, and fails initialization with `VALIDATION_FAILED` on any
  mismatch (`INTERNAL_ERROR` if a launch fails). It has no switch. It costs
  about 70 small launches at load and about 10 MB of scratch, freed before
  serving. Attention, KDA and the head need cache state and are covered by the
  harness and the stack test, not by the gate. The harness also calls the gate
  on synthetic weights (`ROWEQ load_gate`); with a mutation that picks 16
  lanes above 16 rows in `LmSkinnyRowsLanes`, the gate reports
  `mismatches=19/98/4` and the harness fails.

## Not yet covered

MTP drafting and verify through the stage, rows at different positions in one
stack wave, sampled rows in the stack, graph against eager replay, and the
spin-mode collectives. The TP16 check is COMPSEC-17 sequential against 17
concurrent on the fleet.
