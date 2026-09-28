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
| `attention` | the stage's decode dispatch: one row takes the per-head split kernel, a decode wave the all-heads kernel, split on the wave's longest context against threshold 64 | latent output |
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
remain.


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

## Not yet covered

KDA, the DSA indexer (pool scores and top-k), the HC site and post kernels,
MTP drafting, multi-row runs of one sequence (MTP verify, prefill chunks),
selected-list attention past 2,048 tokens, graph against eager replay, and the
TP16 collectives. Add a family when a fix touches one of them.
