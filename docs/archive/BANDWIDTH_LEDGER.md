# Bandwidth ledger

GB10 law: the machine is memory-bus-bound, so the implementation's job is to
move the fewest bytes and keep the bus saturated with them - every byte doing
maximum work. This ledger itemises where bytes go per step, names the code
that spends more than the arithmetic requires, and ranks the fixes. Exact
tonnage comes from `tools/k3_param_budget.py`; formulas here are per token,
per layer unless said otherwise. R = read, W = write, all through the bus.

## K3 decode, batch B, per token

The irreducible stream (per token, whole model, TP1 view; divide weight terms
by tp_degree under TP):

  non-expert weights      every bf16 projection + norms, read once per step,
                          amortised across B - the dominant term at B=1
  expert weights          top_k unique-expert inflation x (w1|w3 + w2)
                          MXFP4 payloads + E8M0 planes, amortised across the
                          rows that share an expert
  lm head                 vocab x hidden bf16 per STEP - 2.35 GB at B=1 TP1,
                          147 MB per rank at TP16; the single largest
                          non-expert tensor and it amortises only across B
  KDA state               32 KB R + 32 KB W per sequence per KDA layer
                          (69 layers: 4.4 MB R+W per seq) - the recurrence's
                          floor, already minimal
  conv windows            3 x kernel x dim x 2B per seq per KDA layer - noise
  MLA latent cache        context x (kv_lora + rope) x 2B read per seq per
                          MLA layer + one row W - the long-context term
  activations             hidden/latent rows between kernels; small at decode

## The suspects (correct code that spends extra bytes or launches)

Ranked by estimated cost x confidence. "Fix window" says what this container
can do versus what needs the sparks.

S1  GATHER DOUBLE-TOUCH (route expansion). LmGatherRowsKernel copies each
    routed row: latent R + packed W, then the expert GEMM reads packed again.
    2R + 1W per routed byte where a gather-aware A-load would pay 1R.
    Cost: packed_rows x 3584 x 2B extra R+W per MoE layer - at B=16 decode
    ~1.8 MB/layer, ~150 MB/token across 82 MoE layers; at prefill C=512 it
    is ~117 MB/layer, ~10 GB per chunk (~20 MB/token) - the same order as
    the amortised weight stream. Fix: an indirect-A variant of the
    weight-only GEMM (cp.async per-row gather instead of TMA on A).
    Window: design now, measure on hardware - TMA vs ldgsts occupancy is
    the open question. UPDATE B-16s: at small B the gather is latency-hidden
    behind weight reads; the prefill case is the one that pays.

S2  GLM52 LAYER AUDIT DEBT. The glm path has not had the K3-grade byte
    audit. Known from reading: Glm52BindLayer clears buffers with memsets on
    every bind and rebuilds per-layer state K3 caches; the layer's gather
    and prefix behaviour predates this branch's GEMM fixes (it inherits the
    dense-derive automatically, but nothing told its route path about
    prefix_built). Window: full audit is CPU work - do it this week, it is
    the other first-class model.

S3  ATTNRES PARTIAL TOUCHES. K3PartialAdd is a separate kernel: hidden R +
    partial R + partial W per module add, 4 touches per layer. The adds
    could ride the producing GEMM's epilogue (accumulate-into-partial),
    saving one full hidden-row R+W per module - ~57 KB x rows x 2 per
    layer, ~10 MB/token at B=1 across 93 layers. Real but small; epilogue
    fusion is invasive. Window: hardware week, after profiles say whether
    the bus or the SMs notice.

S4  HOST-STAGED ALL-REDUCE. SparkTpCollectiveAllReduceSumF32 is TCP
    from host memory in f32. Decode payloads are bf16 device tensors:
    that path pays device->host staging plus 2x wire width. At decode
    the AR is latency-bound (fine); at TP prefill the 2x wire width is
    ~0.4 ms/token of pure format. UPDATE phase6: the format tax is
    landed - SparkTpCollectiveAllReduceSumBf16 keeps the staging and
    the wire bf16 with f32 accumulate per doubling step (round-to-
    nearest-even, bitwise-identical across ranks), halving the D2H,
    wire, and H2D bytes; the wire kind rides the operation header so
    mixed-kind groups fail validation. What remains is the staging
    itself: two host-memory round trips per collective. The device-
    resident tier is the GPUDirect RDMA build of ring/transport/
    rdma.cu (SPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT=1), which today
    speaks the hidden-state transport ABI, not this collective's
    exchange protocol - retargeting it is hardware-week work, not a
    blind edit. Window: hardware.

S5  LAUNCH COUNT, REMAINDER. After 39b3b27 the prefix launches are gone;
    what remains is one launch per GEMM plus the fixed kernels - ~700 per
    K3 decode token. At ~3-5 us each that is 2-3 ms against an 85 ms B=1
    budget today, and the whole budget at TP16's ~5 ms target. Fix: CUDA
    graphs per (rows, layer-range) shape, captured at engine level.
    Window: hardware week; the engine's step planner was built to make
    step shapes repeat exactly so capture can work.

S6  VERIFY REPLAY STORES. Five LmCopyRowsKernel calls per KDA layer during
    verify - ~96 KB x rows per layer R+W. Verify-only, off the decode path,
    and the alternative (recompute at fold) re-reads weights instead.
    Verdict: correct spend, keep.

S7  HEAD EVERY STEP. The lm head reads its full vocab shard per step. At
    TP16 that is 147 MB/rank/step - second only to expert flow at small B.
    No cheat preserves exact sampling; DSpark amortises it across accepted
    tokens (one head read verifies eight positions), which is another
    reason speculation matters at B=1. Verdict: architectural, mitigated
    by DSpark.

## Prefill addendum

Chunked TP prefill's extra terms: the AR per layer (hidden x C x 2B x 2
transfers, S4's format tax on top), the gather at chunk width (S1's worst
case), and the MLA latent cache growing quadratically in reads across
chunks - the last is the DCP/position-sharding conversation, out of scope
for this ledger.

## Standing rule

A new kernel or path lands with its line in this ledger: what it reads,
what it writes, and why that is the minimum. The gates prove correctness;
this file is where "correct but slow" goes to be seen.


## Topology study: TP_g x PP_s on the two-plane fabric (model, 2026-07-29)

CORRECTION recorded: an earlier decode estimate amortized the expert sweep
over B under PP; the correct law is m = B/s - a stage's weight pass serves
only its microbatch. Large-B PP numbers shrink accordingly; TP's standing
improves everywhere once the switched fabric's ~20 us AR floor replaces the
ring's 870 us.

Laws: (1) touched experts/layer = 384*(1-(47/48)^m), m = B/s;
(2) AR cost per layer-pass = 2 ARs * (20 us + 2*m*14.3KB*(g-1)/g / 12.5 GB/s)
    over the switch; TP2 pairs ride adjacent ring links instead;
(3) per-node weight residency invariant in (g,s); only time moves.

Decode agg tok/s ideal/sustained (MBU .55, net .8, 2K ctx; 256K col fp8 KV):
  TP16     : B1 50/30  B8 111/63  B64 202/115  B1024 1028/649  B64@256K 174/98
  TP8xPP2  : B1 28/16  B8  99/56  B64 161/90   B1024  894/520  B64@256K 142/79
  TP4xPP4  : B1 15/9   B8  82/46  B64 138/76   B1024  581/324  B64@256K 124/68
  TP2xPP8  : B1  8/4   B8  61/34  B64 121/67   B1024  344/190  B64@256K 110/61
  PP16     : B1  4/2   B8  32/18  B64 106/58   B1024  223/123  B64@256K  97/54
Prefill agg tok/s (2048 chunks): PP16 11.8K > TP2xPP8 10.2K > TP4xPP4 6.1K
  > TP8xPP2 3.7K > TP16 2.1K. Decode monotone toward TP (switch flattened
  the AR tax); prefill monotone toward PP (chunk ARs are bandwidth).
DSpark: ~x1.9 effective at full sweep (verify rows ride paid weight reads);
  ~x1.0 at B=1 on top-8-of-384. Measured column replaces all of this at
  bring-up.


## Expert precombination study (measured, 2026-07-29)

Kimi-K3 HF weights, layer 5, w2, 16 experts, MXFP4 dequantized, sketch
rank 256/expert. Shared-basis hypothesis W_e ~ A C_e B + delta: REJECTED.
  shared MEAN energy: 6.3% (min 6.0%)
  shared COLUMN space: r=64 6.9% / r=128 13.1% / r=256 24.2% /
                       r=512 42.3% / r=1024 67.6%   (rows: same +1-2%)
  single-expert own spectrum: top-64 7.9%, top-512 46.1%
The cross-expert shared basis at r=512 captures LESS than one expert's
own top-512: column spaces are mutually ~random. MXFP4-native training
left no linear redundancy to precombine. Bytes floor stands at ~59
GB/token B=1 (moe_intermediate corrected 4096 -> 3072 from config.json;
expert = 35 MB; all decode estimates improve ~13%). Config also confirms
every serving-tier geometry number our drift gates hold: hidden 7168,
93 layers, MLA schedule [every 4th + last], kda 96x128, conv 4,
kv_lora 512, rope 64.


## Compression codec architecture (sparkdev K3 data, 2026-07-29)

Measured: MXFP4 nibbles 3.752/4 bits (zstd 5.81%); E8M0 scale plane 84.7%
compressible; whole shard 10.96%. The 160 GB is real but NOT uniform, and
WHERE decompression happens decides whether bandwidth wins or dies:

THE STAGING TRAP: weights are read every step. Storing compressed in
unified memory and decompressing LPDDR->LPDDR (nvCOMP pass, rotating
staging) costs read C + write U + read U per step ~= 2.9x layer traffic.
On a bus-bound machine that is a ~65% throughput cut to save capacity.
Staging decompress is NEVER for hot weights. (It IS free money for the
NVMe pack: model swap time drops ~11% at zero runtime cost.)

THE REGISTER-FEED RULE (refines the >1% rule): a plane earns a resident
codec iff (a) capacity saved > 1%, (b) TILE-ADDRESSABLE - independent
blocks matching GEMM tiles (128 x TILE_K) with an offset table, random
access preserved, and (c) decode <= ~8 ALU ops/byte fused into the
consume path. Then the bus reads COMPRESSED bytes and the win is
capacity AND bandwidth together.

VERDICTS ON THE MEASURED PLANES:
- Nibbles at 5.81%: fails (c) - entropy decode at fragment rate is not
  8 ops/byte. Not harvestable unless structure is found (see test spec).
- E8M0 scales at 84.7%: PASSES ALL THREE. Scales are 1/17 of expert
  bytes (~76 GB of ~1.3 TB); 84.7% structure means per-row exponent
  streams delta+pack to ~2-3 bits/group with a mode+exceptions decode of
  a few ops - and the consume path ALREADY has the per-thread scale
  cache to hang it on. Expected: ~60-65 GB capacity AND ~4-5% fewer
  expert bytes per full-sweep step = ~+4% ceiling throughput. This is
  the codec to build.

TEST SPEC FOR THE FULL-DATA SWEEP (what to measure so verdicts follow):
1. Everything BLOCK-WISE at 128x512-element tiles, not whole-tensor -
   random access is the constraint, and whole-tensor numbers flatter.
2. Byte-plane splits: bf16 attention/shared/dense tensors as sign / exp
   / mantissa planes (exponent planes of trained bf16 typically 60-80%
   compressible - the 0.3 TB non-expert tier may hide 30-60 GB).
3. E8M0 deltas two ways: within-row (neuron-adjacent groups) and
   CROSS-EXPERT (expert_i scale map minus expert_0's) - if experts share
   scale topology the plane crushes further.
4. fp32 tensors (A_log, dt_bias, norms) as planes; router weights;
   embedding table.
5. Report bits/element post-transform, not zstd ratio, so the in-kernel
   decode can be designed from the number.

External validation, same report: a 14-real-layer CPU traversal matched
our layer semantics exactly - MLA at 0-indexed 3/7/11, AttnRes
boundaries at 0 and 12, layer-12 bank advancing to two vectors, layer-1
top-16 routing equal to the retained reference. The host-gate trajectory
claims now have a checkpoint-backed witness.
