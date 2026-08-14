# GLM52 Projection Tile Experiments 2026-07-03

Target:

```text
Spark2
SM121
PP13
B64 decode
stage-slice graph path
GLM52_EXACT_PP13_DEVICE_SYNC_TIMING=1
```

## Hypothesis: reuse quantized weight decode across B64 lanes

The built-in quantized BF16 WMMA Q/KV/O path originally used a 16x16 output tile
for every projection. At B64 this decoded the same quantized weight tile once
per 16-sequence row tile, so the attention output projection decoded each weight
tile four times per output tile.

Change:

```text
attention output projection only:
    B32 -> 32x16 sequence/output tile
    B64 -> 64x16 sequence/output tile

Q/KV projections:
    keep 16x16 tile
```

Measured result:

```text
stage 66:6 B64 graph path:
    before optimized selector: about 564-605 tok/s across repeated runs
    64x16 attention-output-only selector: 616.841 tok/s standalone profile

full PP13 B64 graph sweep:
    old slowest observed: about 564-565 tok/s
    64x16 attention-output-only slowest observed: 618.772 tok/s
```

Phase profile showed attention output projection falling from roughly 7.2-7.6 ms
per layer to roughly 5.8-6.9 ms per layer. Applying the wide tile to Q/KV
projections was rejected because it made the attention projection phase slower.

Decision:

```text
keep 64x16 / 32x16 only for attention output projection.
keep 16x16 for Q/KV projection shapes.
```

## Rejected: 64x32 attention output tile

Hypothesis:

```text
For B64 attention output projection, a 64x32 tile might reuse each input tile
across two output-column fragments and reduce per-block overhead.
```

Measured result:

```text
stage 42:6 B64 graph path:
    64x16 attention-output-only: 662.717 tok/s standalone profile
    64x32 attention-output-only: 566.795 tok/s standalone profile
```

Phase profile:

```text
64x16 attention output projection:
    roughly 5.8-6.7 ms/layer on stage 42:6

64x32 attention output projection:
    roughly 8.3-9.3 ms/layer on stage 42:6
```

Conclusion:

```text
64x32 loses. The extra accumulator/register/shared-memory pressure likely costs
more occupancy than the input-tile reuse saves. Do not use N32 for this built-in
WMMA fallback path.
```
