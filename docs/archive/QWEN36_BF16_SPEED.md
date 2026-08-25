# Qwen 3.6 27B, full BF16 on the sparkring - the speed model
Grounded in inference/llms/qwen_3_6/config.h. Dense hybrid: 64 layers,
16x (3 gated DeltaNet -> 1 gated full attention), hidden 5120, FFN
17408, GQA 24/4 heads x 256, GDN 16 key x 48 value heads x 128.

Parameters, counted from the projections: full-attention layer 372.2M,
DeltaNet layer 383.1M, embedding 1.271B (x2 untied), one MTP layer ->
**27.27B total, 54.5 GB in BF16**. Fits one GB10 with room; the whole
ring is a luxury, not a requirement.

Per-token streamed state on top of weights, per sequence:
- GDN: 6.46 MB/layer read+write x 48 layers = **310 MB/seq/token**
  (context-independent - the linear layers never grow). The state is one
  128x128 FP32 matrix per VALUE head - 48 of them, 3.0 MiB, plus the 80 KiB
  fused-QKV convolution window, read and written every token. This line used
  to say 51.9 MB, priced from a 16-head bf16 slot: wrong twice over. The
  delta rule's pool contract is fp32 (the bf16 sizing put half the heads past
  the end of the slot), and the recurrence holds one state per value head,
  not one per key head - q and k are repeated three ways, the GQA expansion
  the reference defines for GDN. The bf16-state option halves this line and
  is the biggest single lever the model has at B64+; it is a numerics
  question on a compounding recurrence, so it lands as a kernel variant with
  a precision contract, not as a smaller pool behind the fp32 kernel's back.
- Full-attention KV: 4 KB per context token per layer x 16 layers =
  **ctx x 64 KB/seq read per token** (BF16 KV, the growth term).

Decode step time = (54.5 GB + B x (310 MB + ctx x 64 KB)) / bandwidth.
At 273 GB/s per GB10:

| nodes (BW)      | ctx   | B=1  | B=8   | B=32   | B=64   |
|-----------------|-------|------|-------|--------|--------|
| 1  (0.27 TB/s)  | 1k    | 5.0  | 38    | 120    | 222    |
|                 | 4k    | 5.0  | 37    | 120    | 192    |
|                 | 16k   | 4.9  | 33    |  89    | 124    |
| 4  (1.09 TB/s)  | 4k    | 20   | 148   | 480    | 767    |
| 13 (3.55 TB/s)  | 1k    | 65   | 494   | 1707   | 2891   |
|                 | 4k    | 64   | 481   | 1560   | 2493   |
|                 | 16k   | 64   | 434   | 1159   | 1606   |

Readings:
- **Single-stream floor: 5 tok/s on one node, 65 tok/s across the
  ring** (200 ms vs 15 ms per token of pure weight streaming). BF16 is
  a bandwidth tax paid in latency; interactive single-user wants the
  ring or a quantized ladder rung. The state correction does not move
  this: 310 MB on a 54.5 GB stream is under 1 percent at B1.
- **The batch knee is where B x state rivals weights**: at 4k context
  each lane adds ~570 MB/step, so past B~95 on the ring the state
  stream overtakes the weights and per-lane throughput halves - the
  table's B64 column is still weights-dominated everywhere, but by
  2:1 rather than the 5:1 the old pricing claimed. This is the column
  the bf16-state option buys back.
- Long context bites only the 16 full-attention layers: 16k context
  costs 1 GB/seq/step - the 3:1 hybrid doing exactly its job, and
  16k/B64 on the ring still clears 1.6k tok/s.
- These are bus-saturation ceilings assuming the cohort-13 pipeline
  keeps the bus busy (audit F1-F3 are the risks to that assumption);
  compute rides under the weight stream at these batch sizes.

Debug-distance status: qwen has firmware config, host geometry,
doorway, null provider link, conformance rows, and - with the
uniform-estimated profile - an ADMITTED scheduler path proven by
test_uniform_profile_admit on qwen's exact dense geometry {64,64}.
Getting here surfaced and fixed two real balancer defects: the cut
rule that pinned the dense prefix to stage zero forbade every split in
a fully dense model, and the reachability probe indexed the maximum
layer count instead of the geometry's - an out-of-bounds read past the
VLA that glm had been passing on stack luck. Both are gated now. What
remains for qwen tokens is the same execute rung K3 waits on, plus the
chat surface.

Driver state, 2026-08-01 launch audit: the full-attention path was
wired to the MLA latent kernel over a cache no launch wrote into, and
the GDN state slot was half the stride the fp32 delta rule addresses -
both fixed, with per-head GQA store/decode kernels (kernels/gqa.cuh)
and a 48-slice state, all host-verified in
tests/host_cuda/qwen38_27b_layer_host.cu. Still open and named in
config.h: the GDN forget/write gates have no producer (the recurrent
layers read gate buffers nothing computes until the beta/alpha
projection is bound), the attention output gate, and mrope for
non-text input.
