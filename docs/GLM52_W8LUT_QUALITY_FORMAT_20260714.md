# GLM-5.2 W8LUT quality weight format — migration plan and candidate module

Date: 2026-07-14. Author: offline analysis + candidate code, not yet a published module.
Per the ledger rules in `docs/GLM52_MEASURED_STATUS.md`: everything below is offline
work — compilation, host tests, and checkpoint measurements. **No runtime status is
claimed.** Offline measurements are labeled with their exact inputs and are reproducible
from the shipped code.

Implementation update 2026-07-16: the independent `.spw8lut` artifact path,
resident binding, parallel router/route/tile preparation, and BF16 WMMA expert
kernel are implemented and host-qualified. Full-ring accuracy and throughput
remain hardware qualification work, not a production claim.

## 0. Summary

Replace the routed-expert quantization for layers 3–77 with **W8LUT v2**, a per-tensor
exponent-biased 1-3-4 minifloat: 8 bits per weight like the production FP8 path, but
**1.95× lower Frobenius error measured on real GLM-5.2 expert weights**, no block-scale
tensors, no sidecars, decode = one integer add, bit-deterministic by construction.
Alongside it: promote the trunk (MLA projections, shared experts, dense 0–2, DSA indexer)
and the layer-78 MTP experts to BF16, cap context at 256k, and evaluate FP8 KV latent.

This lands as a **third quantization variant** next to the two already declared in the
model description (`glm52_nvfp4_4bit`, `glm52_fp8_e4m3_8bit`):

```
glm52_w8lut_8bit  ->  SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT
```

NVFP4 remains the memory-pressure point; FP8 the current production point
(`glm52-fp8-main-*` releases); W8LUT is the quality-maximum point at FP8's footprint.

## 1. Falsification log (do not re-litigate)

A v1 design (top-15 (sign,exp) symbol table + 4-bit mantissa + exact escape sidecar) was
killed by measurement: on 210 real expert tensors the top-15 coverage of the 512-way
symbol space is 98.27%, not the 99.97% suggested by 256-way byte-split accounting →
escape rate 1.69% → ≈ +73 GB of sidecar across the checkpoint. The escaped values are the
smallest-magnitude tail, where relative precision is worthless. v2 clamps that tail
instead and spends nothing.

## 2. Measured quality (offline, real weights)

Input: `zai-org/GLM-5.2` (BF16 master) shard `model-00002-of-00282.safetensors`, all 210
routed-expert tensors (2.64B params, layer-10 region). Float64 accumulation. Comparison
recipe for the production path: e4m3 with 128×128 block amax scaling
(`fp8_scale_block: 128` in `model_contracts/glm52.json`), via ml_dtypes RNE casting.

| metric | e4m3 + block128 (production recipe) | W8LUT v2 |
|---|---|---|
| Frobenius rel error ‖ΔW‖/‖W‖ | 0.02646 | **0.01358** |
| max rel error, in-window | ≥ 1.0 (2,592 nonzeros flushed to exact 0 in 27 sampled tensors — block-scale underflow) | 0.0303 (≤ 2⁻⁵, exact bound) |
| below-window weights (absolute-clamped) | — | 2.346% of count, ~0.02pp of Frobenius |
| exact zeros in source | — | 0 in 2.64B weights |
| per-tensor metadata | [I/128,H/128] f32 scales | one u16 (`e0`) |

Reproduce: `tools/glm52_w8lut_codec.py` +
`realdata_result.json` (retained numbers).

## 3. Format (normative reference: `source/w8lut.h`, `source/w8lut_ref.c`)

Code byte `[7]=sign, [6:4]=eidx, [3:0]=man`; per tensor one parameter `e0`.

```
(c & 0x7F) == 0  ->  signed zero
else             ->  bf16 = sign<<15 | (e0 + eidx)<<7 | man<<3
```

Encode: RNE 7→4 mantissa bits via `(b + 3 + ((b>>3)&1)) & 0xFFF8` (the add carries into
the exponent bit-arithmetically, correct across the subnormal boundary).
`e0 = max exponent of the RNE-rounded tensor − 7`: the amax octave is always exactly
representable — **nothing ever clips above**, which is what removes e4m3's underflow /
overflow pathology. Below-window values round to nearest of {0, bottom grid}; the
value-space threshold is bit pattern `((e0−1)<<7)|0x08`. inf/nan in source is a
conversion error (−11/−12), measured absent in real weights.

## 4. Resident pack: `SPARKGLM52W8LUT` (delta from the FP8 pack)

Same one-time-packer discipline as `tools/glm52_fp8_resident_pack.py` (Python/Torch/
safetensors offline; serving loads packs from C only). Same header shape
(`Fp8MoePackHeader` fields, 512-byte header, 4096-aligned regions, per-layer file with
all 256 experts), with:

```
quant_mode   = 3   (QUANT_MODE_W8LUT; NVFP4=1, FP8_E4M3=2)
scale_layout = 2   (SCALE_LAYOUT_W8LUT_E0)
REGION_W1_WEIGHT    : u8 codes, fused up_gate, expert-major row-major  [E][2*I][H]
REGION_W1_SCALE_INV : u16 e0, per expert per component                 [E][2]
REGION_W2_WEIGHT    : u8 codes                                         [E][H][I]
REGION_W2_SCALE_INV : u16 e0                                           [E][1]
```

W8LUT is a physically separate pack family. Files use the `.spw8lut` extension,
`SPARKGLM52W8LUT` wire magic, and `w8lut_moe_pack_manifest.json` in a dedicated pack
directory. The packer refuses to replace an existing `.spw8lut` file or manifest and
refuses any destination containing `.spfp8`, B12x, or unrelated artifacts. It never
opens, edits, relocates, or aliases the working FP8 pack set.

Generation is explicit and fail-closed:

```
make -j glm52_w8lut_resident_pack \
    W8LUT_MODEL_DIR=/stable/bf16/zai-org/GLM-5.2 \
    W8LUT_MOE_PACK_OUTPUT_DIR=/stable/packs/glm52_w8lut_pp13_stage00 \
    W8LUT_MOE_PACK_LAYERS=3,4,5 \
    W8LUT_MOE_PACK_JOBS=3
```

Per-expert-per-component `e0` (each expert's up / gate / down slice gets its own window)
is strictly finer than the per-tensor windows measured in §2, so §2 is a lower bound on
quality. Conversion source is the **BF16 master** (`zai-org/GLM-5.2`), never the FP8
checkpoint — requantizing FP8 would bake e4m3 error into W8LUT. Packer verifies every
slice (decode == RNE reference in-window, re-encode idempotence) and writes a manifest.
`tools/glm52_w8lut_codec.py` `encode()` is the production resident-pack conversion core.

### Rank-local BF16 plus W8LUT conversion

`tools/glm52_w8lut_stage_pack_watch.py` derives the PP13 layer assignment from the
checked-in model contract. `--layers auto` maps rank 0 to routed layers 3–5, ranks
1–11 to six routed layers each, and rank 12 to layers 72–77 plus MTP layer 78. It
waits only for the exact source shards used by that rank's outputs, rather than every
file conservatively assigned by an older waterfall manifest.

When `--stage-packer` and `--stage-output-dir` are supplied, one invocation creates
both isolated pack families from the rank-local BF16 master shards:

```
python3 tools/glm52_w8lut_stage_pack_watch.py \
    --manifest /tmp/ds4_waterfall_manifest_glm52_bf16.json \
    --rank 2 \
    --model-dir /home/spark2/models/hf/zai-org/GLM-5.2 \
    --output-dir /home/spark2/sparkpipe_artifacts/glm52_w8lut_resident_moe_pp13_stage_v1 \
    --packer tools/glm52_w8lut_resident_pack.py \
    --stage-packer tools/glm52_stage_pack.py \
    --stage-output-dir /home/spark2/sparkpipe_artifacts/glm52_w8lut_bf16_pp13_stage_payload_v1 \
    --layers auto
```

The StagePack index identifies `model_quantization=w8lut`, records the BF16 source
index SHA-256, and rejects non-BF16 non-expert weights or FP8 scale tensors. Existing
FP8 StagePacks and `.spfp8` expert packs live in different directories and are not
opened or replaced.

This is the data-format prerequisite only. The current node-context builder still
binds FP8 attention, dense, shared-expert, and DSA projections in W8LUT mode. Loading
the new BF16 StagePacks into those projection plans remains a separate runtime change;
no mixed W8LUT-expert/BF16-trunk inference status is claimed here.

## 5. Trunk and MTP to BF16

| tensors | today | change |
|---|---|---|
| MLA q_a/q_b/kv_a_proj_with_mqa/kv_b/o, all layers | FP8 | BF16 (+12.14 GiB) |
| shared experts, all layers | FP8 | BF16 (+2.67 GiB) |
| dense MLP layers 0–2 | FP8 storage, BF16 use (`dense_*_weight_fp8` in `spark_glm52_pp13_node_context_builder_cuda.cu`, `dense_*_weight_bf16` in `spark_glm52_resident_decode_stage_module.c`) | BF16 native — the fp8→bf16 boundary and its nondeterminism class is deleted, not patched (+0.63 GiB) |
| DSA indexer wk/wq_b | FP8 | BF16 (+~0.5 GiB, packer logs exact) |
| **layer-78 MTP routed experts** | FP8 | **BF16 pack** (+9.0 GiB, +0.69 GiB/node) — ct decision 2026-07-14: buys speculative acceptance quality; MTP one-draft path is live and measured (`glm52-fp8-main-f5364187-b1-mtp1`) |

## 6. Footprint and traffic (derived from measured checkpoint census)

Checkpoint-level: 703.7 GiB (FP8 today) → ≈ 729 GiB → **56.1 GiB/node PP13 average**;
per-node actuals depend on the layer assignment (first/last ranks carry embed / the
1.772 GiB BF16 head). Decode weight traffic +3.6% → est. −3% decode. Prefill expert
GEMMs move from FP8 TC to BF16 MMA after expand → est. −10..25% prefill until an
optional requant path is justified by measurement. Estimates carry no status until a
matched release measurement per the ledger rules.

## 7. Kernels (candidate, in `source/w8lut_kernels.cu`)

`w8lut_expand` (bulk decode → BF16 buffer; prefill feeds the existing BF16 GEMM path)
and `w8lut_gemv` (fused decode+GEMV for decode: one block per output column, weight row
decoded to SMEM once and reused across M, fixed 256-thread strided partials + fixed
halving-tree reduction, no atomics, no split-K → bit-identical across runs and ranks).
At B1024 × top-8/256, M ≈ 32 rows/expert — weight-bandwidth-bound, decode is free.

**Status: written, NOT compiled by the author (no GPU in the authoring environment).**
`make -C modules/glm52_w8lut_quality_weights test_gpu` on a Spark runs G1 (device decode
bit-exact vs the C reference) and G2 (gemv bit-exact vs a CPU float mirror using the
identical accumulation order, `fmaf` matching nvcc's default contraction).

Production integration must follow the module rules (README §Rules, STATUS.md): a real
published link unit with ABI, target, entry symbols, and validation recipe — either a
new W8LUT MoE primitive module or a W8LUT mode of the B12x path if FlashInfer's template
admits a custom dequant hook. These kernels are the reference semantics and a starting
point, not a drop-in production path.

## 8. Context cap and KV

`model_contracts/glm52.json`: `maximum_context_tokens` 1048576 → 262144, with
`kv_pool_tokens` re-sized accordingly (see `GLM52_1M_CONTEXT_CAPACITY_20260706.md` for
the capacity model this reverses). DSA attends to `dsa_selected_token_count = 2048`
regardless of context, so ≥256k contexts pay storage and indexer triage for no added
attention budget. KV-latent FP8 is a separate decision: current latent cache dtype and
the JIT-KV interaction need confirming against `GLM52_B1024_JIT_KV_INTEGRATION.md` /
`sparkpipe_glm52_kv_jit_budget.c` before sizing; gate any change with §9's KL harness.

## 9. Gates

| gate | test | status |
|---|---|---|
| G0 | `make test_ref` — RNE goldens (carry, ties, subnormal boundary), inf/nan reject, window/threshold goldens, property maxrel ≤ 2⁻⁵, decode→re-encode idempotence | PASS, author machine (offline; no runtime status) |
| G0b | `make crosscheck` — numpy encoder ≡ C reference, bit-for-bit | PASS, author machine |
| G1/G2 | `make glm52_w8lut_quality_cuda_gate` on spark0, SM121, merged commit `07c4e77a` | PASS: 12,582,912-code expand bit-exact; GEMV mirror bit-exact; FP64 spot maxrel 4.16e-05 |
| G3 | per-layer ‖out−BF16ref‖: W8LUT strictly closer than FP8, every layer (`glm52_transformers_stage_reference.py` provides the reference harness pattern) | pending |
| G4 | `source/kl_eval.py`: mean KL(BF16 ‖ cand) over ≥2M teacher-forced tokens; accept KL_w8lut ≤ 0.5 × KL_fp8 (expected ~0.25×) | pending |
| G5 | matched release measurements per the ledger: decode B64/B1024, prefill, N-run + cross-rank bit-identity soak | pending |
| G6 | per-rank memory audit at B1024 / 256k | pending |

## 10. Open items

Current KV latent dtype + JIT-KV pool interaction; per-rank layer→pack assignment vs the
+1.8 GiB head/embed ranks; module publication route (new primitive vs B12x mode);
whether the dspark draft path shares any expert packs (assumed no — separate weights);
prefill:decode production mix to budget the prefill regression.
