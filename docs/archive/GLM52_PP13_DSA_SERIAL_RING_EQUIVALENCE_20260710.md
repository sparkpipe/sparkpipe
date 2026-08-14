# GLM-5.2 PP13 DSA serialized/ring equivalence, 2026-07-10

## Verdict

The live B1 FP8 ring and the corrected serialized stage0 path are byte-identical
for all five prompt rows. The previous serialized reference was invalid for DSA
index-sharing layers because its node contexts never received their real layer
indices.

The missing assignment made every full-indexer layer publish selected indices to
the layer-0 cache. Shared layer 3 read the untouched layer-2 cache, which was all
zeros. It therefore attended token 0 repeated 2,048 times. The live ring sets the
layer index and did not have this defect.

The correction is:

```c
node_context->layer_index = layer_index;
```

in `SparkValidationPrepareExactPp13StageSliceLayer`.

## Reproduction

Repository main before the correction:

```text
710c5a93c7e3339198a898a1057da4d09c30bf08
```

Input hidden sequence:

```text
five BF16 rows
hidden dimension 6144
token ids 45494 10397 13 10397 13
```

Validation shape:

```text
first layer 0
layer count 6
active sequences 1
FP8 attention and FP8 MoE packs
graph replay disabled
```

Before the correction, token 1 first diverged at layer 3:

```text
field                 serialized-invalid       live-ring
layer2 output         346ccaedfcd271c3         346ccaedfcd271c3
layer3 DSA indices    b9d103fd6854a325         nonzero
layer3 output         5dae16be454540c9         889da44cc1ad7cc5
```

`b9d103fd6854a325` is the FNV-1a hash of the all-zero 8,192-byte
selected-index payload.

After the correction, each serialized stage0 row matches the ring dump:

```text
row  serialized SHA-256                                                ring SHA-256
0    e8d769e1b93c74151e8d25061edfeeee40d075e47c108cde267b2b532d78c684  e8d769e1b93c74151e8d25061edfeeee40d075e47c108cde267b2b532d78c684
1    0d88b6bc3918b59e7aa1729b13e12a52b2b80d52b82a2530739b69523231bda7  0d88b6bc3918b59e7aa1729b13e12a52b2b80d52b82a2530739b69523231bda7
2    94fedf9b3c085c19d1e50dc4b866ba69e48f44dc64ce2321398ce0fcbddb629f  94fedf9b3c085c19d1e50dc4b866ba69e48f44dc64ce2321398ce0fcbddb629f
3    a86824a1d3c713ef1edc97e0f8478692eda340727b659840d733577becfc9982  a86824a1d3c713ef1edc97e0f8478692eda340727b659840d733577becfc9982
4    e69f78bd1e24add18bc8517255f14c9c72137879dab0ccc5aa87a4e920de50e4  e69f78bd1e24add18bc8517255f14c9c72137879dab0ccc5aa87a4e920de50e4
```

## Model-contract findings

The checkpoint configuration declares:

```text
rope_interleave=true
indexer_rope_interleave=true
rope_theta=8000000
rms_norm_eps=1e-5
qk_head_dim=256
index_head_dim=128
index_topk=2048
```

Interleaved RoPE and theta 8,000,000 are therefore confirmed. The generated
model contract now also owns attention scale `1/sqrt(256)`, model RMSNorm
epsilon `1e-5`, DSA key LayerNorm epsilon `1e-6`, DSA score scale
`1/sqrt(128)`, DSA index-sharing geometry, and routed-MoE scale `2.5`. The DSA
key LayerNorm remains independent from the model RMSNorm value.

## Corrected model-semantics receipt

After applying the generated model constants, the exact FP8 stage `0:6` loop
and built-in launcher produced the same five-row output:

```text
SHA-256 5a93fde2f080c2a8cf22fd30ef1e389a7f1b7b02fddd1fc2508bff972e458111
rows    5
bytes   61440
```

A second loop-path run was byte-identical. The run used the same token sequence,
FP8 stagepack, and FP8 MoE packs as the earlier comparison, with graph replay
disabled. This hash supersedes the pre-contract stage0 hashes above for future
ring comparisons.
