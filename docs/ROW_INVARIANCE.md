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
| `dense_linear` | `Glm5NextLaunchBf16Linear` for the nine per-rank dense shapes | output |
| `dense_mlp` | `Glm5NextLayerDenseMlp` | output |
| `moe` | `Glm5NextLayerMoeRoute` then `Glm5NextLayerMoeExperts` (FP8 experts, shared expert, finalize) | router logits, route (expert ids and weights), output |
| `attention` | the stage's decode dispatch: one row takes the per-head split kernel, a decode wave the all-heads kernel, split on the wave's longest context against threshold 64 | latent output |
| `head_greedy` | `SparkGlm5NextRunHead`, greedy rows (B1 takes the certified FP8 path) | maxloc word, token |
| `head_sampled` | `SparkGlm5NextRunHead`, every row sampled | maxloc word, token |
| `head_mixed` | `SparkGlm5NextRunHead`, half the rows sampled | maxloc word, token |

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

## Not yet covered

KDA, the DSA indexer (pool scores and top-k), the HC site and post kernels,
MTP drafting, multi-row runs of one sequence (MTP verify, prefill chunks),
selected-list attention past 2,048 tokens, graph against eager replay, and the
TP16 collectives. Add a family when a fix touches one of them.
