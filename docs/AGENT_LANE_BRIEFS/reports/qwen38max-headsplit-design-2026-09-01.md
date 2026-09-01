# qwen38max head-split increment — the output-projection constraint (2026-09-01)

Follow-on to merged #765 (kv-replication kernels). This is the decision
document for the head-split module wiring: what is trivial, what is
blocked, and the three ways through.

## What is already free

The merged kernels slice q/k/v projection weights and the KV cache at
runtime by rank (whole-head rows), and the module already passes
tp_degree/tp_rank into every attention launch. Row-sharded tensors —
attention Q/K/V projections, routed experts, the LM head — are
contiguous row cuts of the full-width resident buffers, so a rank's
slice is a plain pointer offset. No loader or pack change.

## The constraint: output projections cut along their INPUT

Attention o_proj is [H=8192, Q=16384] (BF16, 256 MiB full-width); GDN
out_proj is [H, V=16384]. With head-split, a rank holds only its
query/value-head slice of the INPUT columns ([H, Q/16]), and partial
outputs are summed by the TP all-reduce (the module already has
SparkQwen38MaxModuleTpAllReduceHidden; #754's validated flow places the
reduce right after the attention linear).

But a column slice of a row-major tensor is NOT a contiguous pointer
offset, and the family's LinearView (weight_payload +
input/output_dimension) cannot express a stride. The #754 branch never
hit this because its sharded packs stored each rank's column slice
contiguously — main's full-width packs do not.

## Three ways through (coordinator ruling requested on #765)

(a) **Load-time column slice (family-local, recommended).** The module
loader copies the rank's [H, Q_local] slice of o_proj/out_proj into a
rank-local buffer at load (strided pack read, contiguous resident).
No shared-kernel change; a linear view then works as-is; resident
memory drops by 15/16 of those two tensors (~240 MiB/rank at TP16).
This is the #754 loader's proven semantics, narrowed to two tensor
kinds.

(b) **Strided-view kernel (shared code).** Add input_offset +
input_stride to the shared SparkLm batched-linear path. Touches
kernels/ shared across families — coordinator review, wider blast
radius, no memory win (full-width o_proj stays resident).

(c) **Attention stays full-width at TP16 (current main behavior).**
Correct today; costs 16x duplicated attention compute and full-width
KV cache — precisely the waste head-split removes, so it caps the
decode-speed win this increment exists for.

## Sizing note (with (a))

At TP16 single-stage, per-rank resident: experts 512/16 x 3 tensors
MXFP4 ~ 90 GiB — still over the 110 GiB law with spine + KV cache
 once head-split shrinks it. So (a) alone does not make TP16
single-stage feasible; the expert slice must also be load-time (same
mechanism, row cut, trivially contiguous) — then experts ~5.7 GiB/rank
and everything fits with headroom. I.e. the real decision is load-time
slicing of {routed experts, o_proj, out_proj, lm_head?} vs PP×TP
hybrid topology. lm_head (vocab-row cut, 248320 x 8192 x 2 B = ~3.9
GiB) slices contiguously with an all-gather-style reduce at the head —
the relief valve from the TP16 report, free under (a)'s mechanism.
