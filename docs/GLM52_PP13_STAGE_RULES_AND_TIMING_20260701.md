# GLM52 PP13 Stage Rules And Timing 2026-07-01

## Rule check

The live GLM 5.2 NVFP4 model config on `spark2` reports:

```text
num_hidden_layers = 78
first_k_dense_replace = 3
mlp_layer_types[0..2] = dense
mlp_layer_types[3..77] = sparse
moe_layer_freq = 1
index_share_for_mtp_iteration = true
num_nextn_predict_layers = 1
```

The current CUDA validator enforces:

```text
first routed layer >= 3
routed chain layer count = 1..8
first + count <= 78
```

I did not find an in-repo rule that routed stages must be exactly 4 or 8
layers. The old 8-layer split was a packaging/performance choice, not a
currently encoded legality rule. The proposed PP13 split below is therefore
valid under the current repo validator. It should still be treated as a
candidate until a production stage-plan checker encodes every GLM/MTP/driver
constraint, because final-token behavior is stricter than hidden-only timing.

## PP13 candidate

```text
stage 00: layers 0-5     dense 0-2 + routed 3-5
stage 01: layers 6-11
stage 02: layers 12-17
stage 03: layers 18-23
stage 04: layers 24-29
stage 05: layers 30-35
stage 06: layers 36-41
stage 07: layers 42-47
stage 08: layers 48-53
stage 09: layers 54-59
stage 10: layers 60-65
stage 11: layers 66-71
stage 12: layers 72-77
```

This is not yet balanced. Chained timings show the late routed layers are much
slower than the early routed layers, so a production PP13 split should give
fewer layers to the late range after the stage-slice runtime exists.

## Chained B64 timing

These timings came from `spark2`, local NVMe model files, cached validators,
real B12x resident packs, and chained hidden outputs. The original B64
hidden-only chain was run before the B-aware final-token validation fix; the
final `72-77` token stage was rerun after the fix.

```text
00-05 prefix:  50.471 ms
06-11 hidden:  46.947 ms
12-17 hidden:  45.002 ms
18-23 hidden:  48.055 ms
24-29 hidden:  49.294 ms
30-35 hidden:  53.859 ms
36-41 hidden:  59.106 ms
42-47 hidden:  72.135 ms
48-53 hidden:  84.188 ms
54-59 hidden:  96.356 ms
60-65 hidden:  91.690 ms
66-71 hidden:  95.458 ms
72-77 hidden:  96.159 ms
72-77 final: 112.159 ms, restricted_token=1205, lm_head_error=0.00001860
```

The uniform PP13 B64 hidden-only ceiling is about:

```text
64 * 1000 / 96.356 = 664 tok/s
```

Including the measured final-token stage as the slowest stage:

```text
64 * 1000 / 112.159 = 571 tok/s
```

## Chained B32 timing

```text
00-05 prefix:  39.691 ms
06-11 hidden:  35.142 ms
12-17 hidden:  33.496 ms
18-23 hidden:  36.755 ms
24-29 hidden:  38.837 ms
30-35 hidden:  41.760 ms
36-41 hidden:  46.160 ms
42-47 hidden:  56.283 ms
48-53 hidden:  60.862 ms
54-59 hidden:  69.126 ms
60-65 hidden:  72.429 ms
66-71 hidden:  81.610 ms
72-77 hidden:  73.314 ms
72-77 final: 112.656 ms, restricted_token=1048, lm_head_error=0.00000954
```

The uniform PP13 B32 hidden-only ceiling is about:

```text
32 * 1000 / 81.610 = 392 tok/s
```

Including the measured final-token stage as the slowest stage:

```text
32 * 1000 / 112.656 = 284 tok/s
```

## B-aware final-token validation fix

The B32/B64 final-token path previously failed with layer-77 router IDs turning
into zeros. The cause was not a stricter layer boundary. The validation harness
allocated final logits, selected-token, MTP draft, MTP target, MTP accept, and
MTP committed buffers as B1-sized arrays even when the validator was compiled
for B32 or B64.

After sizing those buffers by `SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT` and
checking aggregate MTP counter invariants, both final runs passed:

```text
B32 final: total_us=112655.999 graph_captures=6 graph_replays=6
B64 final: total_us=112159.487 graph_captures=6 graph_replays=6
```

The patched runs prove per-layer graph capture/replay is active in the validator.
They do not yet prove stage-slice graph replay, because each 6-layer stage still
reports `total_submissions=6`.

## Remaining advisor speedups

High-impact items still open:

```text
1. True stage-slice execution: one submit for a layer group, not one submit per layer.
2. Stage-slice CUDA graph capture/replay: one graph for layers first..first+count-1.
3. FP8/tensor-core projection plans for Q/KV/O instead of BF16 cuBLASLt plans.
4. Measured PP12/PP13/PP14 balancing from chained hidden timings, not equal layer counts.
5. Persistent hidden transport ABI for real ring execution.
6. Production bucket scheduler for B16/B32/B64.
7. Production stage-plan checker that encodes GLM config and firmware cut rules.
```

The next performance target should be items 1 and 2 together. Per-layer graph
capture is useful, but it still leaves six launches for a six-layer stage.
