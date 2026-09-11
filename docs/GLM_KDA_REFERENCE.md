# GLM Flash recurrent-state oracle correction

2026-09-08, based on main `52ce0e594c88e2ca934f762677311353f61f65fb`.

The C validator multiplied state by the retention vector and then multiplied
by retention again while computing the key prediction. KDA predicts from the
already-decayed state. The [upstream scalar reference](https://github.com/fla-org/flash-linear-attention/blob/14bef3d9ec98ee359c9a9c7860795d91675dd091/fla/ops/kda/naive.py#L54)
and this repository's Python vector oracle both apply decay once. The CUDA
kernel computes the prediction from old state times retention and applies the
decay once during its update; its implementation is unchanged here.

Extract the corrected scalar head recurrence into `spark_kda_reference.h` for
independent host validation across models. Key/value dimensions are parameters;
query/key normalization and query scaling remain the caller's responsibility.
Do not reuse production device code as its own oracle.

The regression starts with nonzero 2x2 state, unequal channel retention and
nonzero beta. It checks exact state and output against hand-calculated values
for two successive tokens, plus a rectangular 2x1 state. Injecting the previous
second decay makes the first assertion fail. Host regression, GLM oracle
selftest and complete validator syntax compilation pass. Merged-main GPU
validation remains pending.

Lesson: a random small-signal, zero-initial-state fixture can underweight a
recurrence error. Check state evolution directly with nonzero initial state,
then retain end-to-end numerical checks. Earlier component PASS results do not
establish that this flawed reference was correct, nor qualify full GLM serving.
