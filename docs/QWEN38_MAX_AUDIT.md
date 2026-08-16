# Qwen3.8-2.4T-A95B inference path audit

Scope: the resident decode stage module, the serving adapter and the
PP/TP communication plumbing, the inference/llms/qwen_3_8 driver family,
and the packer/loader. Correctness footguns are fixed in this branch;
performance and capacity findings are measured on a single GB10 spark
(no fleet window was needed for any of this).

## 1. Correctness footguns found and FIXED

1. **Adapter stage slicing was the qwen36 PP13 clone's.**
   `stage_layer_counts = {5,5,...,6,2}` summed to 64 layers while
   STAGE_COUNT was 16 world ranks; ranks 4-15 indexed past the array and
   every PP boundary used the world rank as the PP stage. Fixed:
   `{23,23,23,23}` per-PP-stage counts and `SparkQwen38ServingPpStageIndex
   = world_rank / 4` driving every boundary (hidden in/out, token in/out,
   first-layer computation).
2. **Adapter advertised BF16 expert weights** (`.expert_weight_codec`) while
   the pack stores FP8_E4M3 block-128. Fixed to SPARK_WEIGHT_CODEC_FP8_E4M3.
3. **Adapter context cap was the qwen36 8192** and would reject the model's
   own 262144-context stage config (LoadConfiguration treats the cap as a
   schema error). Fixed to SPARK_QWEN38_MODEL_MAXIMUM_CONTEXT_TOKENS.
4. **Adapter advertised 4 inflight submissions while the module executes
   every frame on ONE slot (slots[0]) with a per-frame stream sync.**
   Clamped to 1 until multi-slot pipelining lands.
5. **The module ignored the firmware's hidden-transport contract entirely.**
   The firmware header requires HIDDEN_INPUT/OUTPUT_TRANSPORT on PP
   boundaries and says the module refuses mismatches; Execute never read
   `frame->user_context`, so a real PP run would have decoded the previous
   token's stale slot buffers with no error. Wired: frame-context
   validation (flags vs stage position, with the unqualified-smoke escape),
   hidden consume into the slot buffer before the layer loop, hidden emit
   after the last layer, and the adapter's KV block table when present.
6. **Batch > 409 sequences silently skipped rows in every MoE group.**
   LmRouteBuild prices its tile prefixes at 32-row tiles past 409 rows
   while the grouped scalar expert kernel hardcodes 16-row tiles; a 512-row
   batch would compute 16 of every 32 rows. The module now refuses
   max_active_sequences > 409 loudly (the real fix is moving the MoE to the
   launch-planner GEMM, which shares one tile-M with the route build).
7. **Driver family: the GDN gated head norm launched 256-template threads at
   128 threads** — the upper warps returned before writing the reduction
   scratch, so LmBlockSum read garbage lanes. Fixed: full-CTA launch plus
   explicit scratch zeroing in both family norm kernels.
8. **Driver family: the shared-expert add gated the ROUTED sum**
   (sigmoid(gate)*(routed+shared)); the checkpoint's learned
   shared_expert_gate multiplies only the shared branch. Fixed to
   routed + sigmoid(gate)*shared (matches the module).

## 2. Correctness footguns found, NOT fixed (design work, fleet-validated)

1. **Single-slot, sync-per-frame execution.** Every Execute ends in
   cudaStreamSynchronize; the module owns one slot. The PP pipeline cannot
   overlap the next token's GDN with this token's MoE, and the adapter's
   max_inflight is clamped to 1 as a consequence. Fix: multi-slot
   stream-ordered dispatch + completion callbacks (the transport already
   has completion queues).
2. **The adapter's post_receive/send end in cudaStreamSynchronize** — every
   PP handoff stalls the whole rank per token. Fix: async send with a
   transport completion.
3. **No lane lifecycle.** Execute hardcodes row_cold = 0 and identity
   lanes; a fresh sequence in production must start cold (the GDN state
   and conv tail are zeroed at pool init, but nothing re-marks lanes cold
   after eviction/residency). Admit/residency are stubs.
4. **TP tensor sharding is absent.** The module has no TP-rank concept, no
   column/row-parallel dispatch, and no collective calls. The deployment
   configs, adapter mapping and hidden transport are now TP4xPP4-shaped;
   the per-layer all-reduce schedule is designed in section 5 and lands
   with the rank-local packs.
5. Dead code: `SparkQwen38LaunchFusedExpertW13Act` /
   `SparkQwen38LaunchExpertDown` are MXFP4-only launchers nothing calls
   (the FP8 path uses the grouped scalar linear). Left in place as the
   natural landing points for an FP8 sm121 expert path.

## 3. Measured performance (GB10, single spark, real FP8 packs)

Timing harness: N decode steps through the module, B sequences per step.

| slice | B | ms/step | tok/s |
|---|---|---|---|
| 1 GDN layer (24.9 GiB pack) | 1 | 8.45 | 118 |
| 1 GDN layer | 2 | 12.30 | 163 |
| 1 GDN layer | 4 | 19.94 | 201 |
| 1 GDN layer | 8 | 36.99 | 216 |
| GDN + attention (49.8 GiB pack) | 1 | 22.49 | 44 |

Reads:

- **Per-step time grows almost linearly with batch** (8.45 -> 36.99 ms over
  B=1..8, only 1.8x throughput gain): the MoE decode is scalar-compute
  bound, and the scalar path gets no tensor-core amortization.
- **Per token per layer the MoE streams ~500 MB of FP8 expert weights**
  (w1 17.2 GB + w2 8.6 GB per layer). At B=1 every byte is touched for one
  token: on GB10's LPDDR5X (~250 GB/s) that is a ~1.8-2 ms/layer floor
  before any arithmetic. 23 layers => >= 40 ms/token per stage; four PP
  stages => >= 160 ms/token, i.e. ~6 tok/s single stream, before the
  scalar compute, launch overhead and the per-frame sync.
- Extrapolated end-to-end single-sequence latency: 23 layers x
  (8.4..14 ms) = 190..320 ms/token/stage; PP4 = 0.8..1.3 s/token.
  **The current decode path is not serving-viable without either large
  microbatches or tensor-core expert kernels.**
- The dense linears (qkv/out, 134..168M MACs each) run the scalar kernel
  for B < 16; the tile path engages at B >= 16 and would remove most of
  the non-MoE scalar work.
- The sm121 tensor-core expert kernels exist in the common library
  (SparkLmSm121B1ExpertW13Kernel / SparkLmSm121ExpertW2Kernel with native
  B1/batched shapes) but accept only MXFP4_E2M1 weights. FP8_E4M3_F32B128
  (the vendor format, our quality-first choice) has NO tensor-core grouped
  path in common kernels — this is the single largest perf gap and a
  coordinated common-code change.

## 4. Capacity: the headline finding

The FP8 checkpoint is 2.31 TiB (experts 2.37 TB decimal + ~80-140 GB BF16
spine). On 16 ranks the minimum residency is **2.31 TiB / 16 = ~148 GB per
rank** (experts partitioned across TP AND layers across PP — any other
arrangement is worse: replicated experts = 593 GB/rank at TP4).

Each GB10 reports 119 GB unified memory with ~107 GB usable.
**148 GB > 107 GB: the fully-resident TP4xPP4 plan does not fit the fleet.**

Options (decision needed):
1. Requantize routed experts to MXFP4-E2M1 (~74 GB/rank, fits comfortably,
   and unlocks the existing sm121 expert kernels — a double win), at the
   cost of the quality-first vendor-FP8 stance. The module's firmware
   already accepts MXFP4 in the pack format, so this is a packer change.
2. More nodes: 24 ranks (TP4xPP6) ~99 GB/rank (marginal); 32 ranks ~74 GB.
3. Non-resident experts with an NVMe tier (the 3.6 TB NVMe holds the whole
   checkpoint per node already); needs a paging design the tree does not
   have today.

Note the redistribution already places the full checkpoint slices on the
internal NVMe of every node, so option 3's data placement is done.

## 5. Communications audit (TP4xPP4)

Per layer, per token, per rank, with a replicated router gate and fused
outputs (the recommended schedule):

| collective | bytes/token | count/layer |
|---|---|---|
| expert w2 partial-sum all-reduce | 16 KB (8192 bf16) | 1 |
| attention/GDN output all-reduce | 16 KB | 1 |
| router gate all-reduce | 2 KB (512 f32) | 1 — avoidable |

- The router gate can be REPLICATED (512x8192 BF16 = 8 MB/rank) so each
  rank computes the full gate locally and the 2 KB all-reduce disappears
  from the hot path. Recommended.
- With gate replication: 2 x 16 KB per layer, or fused into ONE 32 KB
  all-reduce per layer. 23 layers/stage = 23-46 collectives/token/stage.
- Latency model on 100GbE (recursive_doubling, 2 rounds, ~3-5 us/hop +
  ~1.3 us/KB): ~8-12 us per 32 KB collective -> ~0.2-0.55 ms/token/stage
  serialized. Overlapped with the next layer's compute via the transport's
  counter_rotating_split_ring this shrinks toward zero; serialized it is
  20-40% of the current scalar compute, negligible once the MoE moves to
  tensor cores.
- **PP handoffs**: 16 KB/token x 3 boundaries. The current adapter sends
  synchronously (cudaStreamSynchronize in post_receive/send), so each
  boundary adds a full stall per token on top of the transfer; the fix is
  async sends with transport completions (section 2.2).
- **Transport mode**: the qwen38 spec is host-rdma — every boundary and
  collective stages through host memory. dsv4 has gpudirect-rdma spec
  variants; a qwen38 gpudirect spec should be added once collectives land.
- **Rails/ports**: the stage config's dual-rail peer lists match
  examples/topologies/dual_switch_16node_production.json; control 22480,
  collectives 66620 (+rank), transport control 63700 match the fleet
  registry's qwen38max block. step_rail_indices [0,1,1] is inherited from
  the dsv4 template and should be re-derived once the collective schedule
  is real.
- **KV fabric traffic**: none in decode — every stage owns its own
  attention layers' KV locally (GDN layers have no cache). The stage KV
  client is compiled but unused (KV_PROVIDER none). Good.

## 6. Packer / loader notes

- Per-tensor cudaMalloc + serial staged H2D at load (~30 GB per rank, a
  few hundred allocations) — startup-only, acceptable; no async overlap
  today.
- The packer materializes each expert table as one Python bytearray
  (~17 GB for w1) and opens each expert's shard per tensor — one-time
  conversion cost, not runtime.
- The pack receipt still labels routed_experts "mxfp4_e2m1" although the
  pack carries the vendor FP8; cosmetic, fixed by the next packer touch.

## 7. Priority order for the next work

1. Decide the capacity question (section 4): MXFP4 requantization vs more
   nodes vs expert paging. This gates everything.
2. FP8 tensor-core expert kernels in common code (coordinated) — the
   single largest latency lever; or MXFP4 experts + the existing sm121
   kernels if option 1 wins.
3. Multi-slot, sync-free dispatch + async PP sends (removes the pipeline
   bubbles).
4. TP rank-local packs + the collective schedule above.
5. Fleet window: exact end-to-end verification.
