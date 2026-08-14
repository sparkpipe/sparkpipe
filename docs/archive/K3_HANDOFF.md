# Kimi K3 on SparkPipe — handoff

Branch `codex/dry-sweep`, PR #520. 40 gates, one of them red on purpose.

Nothing in this branch has ever run on a GPU. Everything below is verified by
compilation, by geometry, by arithmetic against published sources, or by
executing kernels on a CPU. The first GPU run will find things none of that can.

---

## 1. The one red gate, and the work it blocks

`tests/test_k3_quant_recipe.py` fails. It is right to.

**K3 is weight-only MXFP4.** `quantization_config` sets `input_activations:
null`. This tree quantises the expert *activations* to MXFP4 at three sites —
`layer.cuh` `K3Quantise<Format>` in the projection helper and both expert GEMMs
— because `LmGemmLaunch` takes one `Format` for both operands.

The correct path is `BF16 activation x streamed MXFP4 weight -> decode the
weight tile to BF16 registers -> BF16 MMA, FP32 accumulate -> BF16 out`.

### What it takes, in order

1. **Split `Format` into `FormatA` and `FormatB`** through
   `inference/kernels/gemm.cuh` (210 lines), `runtime/gemm.cuh` (143) and
   `inference/kernels/tile.cuh` (182). The structure is already there —
   `LmGemmSharedBytes` at gemm.cuh:43 already sizes the two tiles separately —
   but every `Format::` reference has to be attributed to an operand. About
   40-60 edits. Keep `LmGemmLaunch` as an explicit same-format wrapper so the
   other four models do not move.

2. **A BF16 route gather.** The first expert GEMM needs
   `latent_bf16 + route_source_token -> expert-major`, which the quantiser was
   doing implicitly. The second activation is already expert-major after SiTU.

3. **`const uint8_t *weight_scale_e8m0`, decoded in the load.**
   `LmE8m0ToFloat` exists in `formats/mxfp4.cuh` and is unused; the GEMM takes
   `const float *scale_b`. The audit's figure for pre-expanding to FP32 is
   17,547,264 -> 20,643,840 bytes per expert layer. **Verify that number before
   acting on it** — I did not.

4. **One scale layout, asserted.** SUPERSEDED by pack format V2
   (`docs/K3_PACK_FORMAT_V2.md`): the scale plane no longer exists as a
   plane. The packer interleaves payload and E8M0 scales per 16-neuron cell
   per 128-element k-tile — sixteen 64B payload rows plus one 64B scale row,
   zero padding — so one TMA box fetches a stage's weights and scales
   together and the far LDG stream this item was about to bless is never
   built.

5. **The packer.** EXISTS: `tools/k3_pack.py` emits pack format V2. It
   preserves the MXFP4 payload and U8 scales (now interleaved, item 4),
   concatenates `w1`/`w3` gate-first, keeps `w2` orientation, fuses the six
   KDA input projections into two single-shard-class tensors, validates 896
   experts and group 32, rejects E8M0 `0xff`, and fails loudly on a recipe
   mismatch. The consumption contract is `docs/K3_PACK_FORMAT_V2.md`.

Also owed: the two MLA folds. `test_mla_absorption.py` proves them at 5.9e-16
and prints them. `mla_kv_b_weight` sits in `K3LayerBuffers` with no call site,
which is the visible symptom.

---

## 2. What is done and verified

**Every kernel K3 launches** is implemented and checked against the source it
was written from — the config, `modeling_kimi_linear.py`, the tech report,
FlashKDA's `torch_ref.py`, or the SGLang bring-up post. Six gates execute real
code on a CPU:

| gate | what runs |
|---|---|
| `kda on host` | bounded decay -> delta rule, 1.9e-3 vs the recurrence; ReplaySSM fold byte-exact |
| `router on host` | sigmoid -> topk -> renormalise, exact vs `KimiMoEGate` |
| `layer on host` | conv+Swish, L2, SiTU, RMS norm, output gate, MoE finalize, AttnRes |
| `mla on host` | KV store + attention over a paged cache, two sequences, interleaved pages |
| `k3 layer on host` | the real `K3LayerLatentMoe`, GEMM recorded |
| `layer dataflow` | dead writes and packed-row route maps, statically |

**Constants**: every one read from the checkpoint. Five K2-lineage guesses were
wrong (shared experts 1->2, expert intermediate 2048->3072, dense 18432->33792,
routed scale 2.5->1.0, KDA heads 64->96).

**Corrections worth knowing about**: the delta rule predicts from the *decayed*
state; the decay adds `dt_bias` in fp32 before the per-head scale; the forget
gate is per-head-per-channel; the MLA latent is `kv_lora_rank` not the KDA head
dim; the KDA state is fp32 and carries its conv windows (54.2 MB at TP=8,
matching SGLang); K3 is NoPE throughout; the router bias selects but does not
weigh.

---

## 3. Open, not started

**AttnRes across a stage boundary.** The kernel and the layer wiring are done;
the bank is 143,360 bytes a token, 8.8 MB at B64 crossing a stage. Follow
SGLang: generate the block representation once at the boundary layer, share it,
carry only new blocks. `deepseek_v4`'s hyper-connections are the same class and
should land together.

**Speculation.** `LmReplayFoldKernel` is byte-exact against the decode state and
nothing drives it. `dspark.h` has the drafter's constants and records that the
verify-trim cost curve is a *staircase*, not a line — a planner modelling it as
linear cuts on a shelf and keeps on a riser. Trimming is break-even below batch
8.

**The layer harness stops at the stage boundary.** It reaches
`K3LayerLatentMoe`; nothing executes `ring/`.

---

## 4. Performance, from `docs/GB10_CUDA_COST_MODEL_CALIBRATION.md`

273 GB/s LPDDR5x at eta_bw 0.80 = **218 GB/s effective**, against a calibrated
6.5 TFLOP/s on the linear path. **30 FLOP per byte before compute can bind.**
On this machine a weight you avoid reading is worth about a thousand FLOPs you
add.

That settled the MLA absorption question — dropping value absorption saves
11.25 GB and 55 ms/token for 46 us of new arithmetic, and is *also* the only
correct option because the elementwise output gate does not commute with the
fold.

`tools/k3_param_budget.py` enumerates the active parameters: 104.62B against
Moonshot's published 104.2B. **Attention is 36.18B (35%), not the 42B a residual
method gives, and KDA is 85% of it** — so the weight target is KDA, not MLA.
Within a KDA layer, q/k/v is 60% and the full-rank output gate is 6.08B on its
own.

---

## 5. Traps this branch hit, so you do not

- **One `Format` for two operands.** Caused the INT7-everywhere defect and then
  the activation-quantisation defect. Both times the fix looked like picking a
  better `Format`.
- **A constant standing in for a nearby one.** Three times: qwen's KV heads, the
  MLA latent, the KDA pool stride. Now `slot_bytes` is a parameter and the
  geometry constants are named per model.
- **`if` where `if constexpr` is needed.** A runtime guard still instantiates
  the template below it.
- **An add is not a copy.** `LmAddRowsKernel(x, x, out)` is `2x`.
- **A format with `kScaleGroup == 0` reaching a quantiser** is a divide-by-zero
  sizing the grid. Now asserted at the kernel.
- **`g++` cannot parse `<<< >>>`.** Hence `LM_LAUNCH`, which is also why the
  launch gate is no longer a regex.
- **Warp shuffles at one thread.** The host shim must return 0 from
  `__shfl_down_sync` and declare one lane per warp, or `LmBlockSum` returns zero
  and an RMS norm is off by 225x.

---

## 6. How to run things

    sh tools/gates.sh                       all 40
    python3 tests/test_k3_layer_host.py     a real layer on a CPU
    python3 tools/k3_param_budget.py        where the 104B sits
    sh tools/build.sh                       all five models, sm_121a

Reference material used, none of it in the repo: `modeling_kimi_linear.py` and
`config.json` from the gated `moonshotai/Kimi-K3`; `k3_tech_report.pdf`;
FlashKDA's `tests/torch_ref.py`; the LMSYS bring-up post.
