# Kimi K3 Weight-Only MXFP4 Contract

The K3 checkpoint quantizes selected `Linear` weights to symmetric group-32
MXFP4 with `uint8` E8M0 scales. Its checkpoint contract does not quantize input
or output activations.

## Expert data path

```text
BF16 activation
  x MXFP4 weight with E8M0 scale per group of 32
  -> weight tile decoded to BF16 registers
  -> BF16 MMA with FP32 accumulation
  -> BF16 output
```

The expert path uses an asymmetric GEMM. Activation rows remain BF16. Routed
tokens are gathered into expert-major BF16 order before the first expert GEMM;
the second expert activation is already expert-major after the activation
function.

## Weight and scale layout

Pack format V2 interleaves MXFP4 payload and E8M0 scales in one tensor per
expert GEMM. Sixteen 64-byte payload rows plus one 64-byte scale row describe
sixteen neurons for one 128-element K tile. One TMA box fetches payload and
scales together.

The consumer decodes E8M0 in the weight-load path. It does not pre-expand scale
bytes into a second FP32 plane. `w1` and `w3` are packed into the combined
gate/up layout; `w2` retains its declared orientation.

The complete container and offset contract is
[`K3_PACK_FORMAT_V2.md`](K3_PACK_FORMAT_V2.md).

## Qualification

The exact package must prove asymmetric `A=BF16, B=MXFP4` dispatch, E8M0 decode
against a trusted dequantized reference, route-gather bounds, complete
`w1/w3 -> activation -> w2` numerical comparison, packed-byte accounting, and
SM121 execution on real weights. Missing evidence fails publication; no
activation-quantized compatibility path is permitted.
