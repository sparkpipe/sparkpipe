# T1 reference-decoder comparison contract (offline half)

The offline half of the T1 accuracy gate produces, per family, a pinned
numpy reference decode of a canonical prompt set from the warm checkpoint.
The serving-gated half runs the resident driver on the same prompt set and
compares against the committed fixtures with this contract. Nothing here
contacts the daemon, the fleet drivers, or any GPU.

## Fixture format (T1R1)

Files are `T1R1` containers: 4-byte magic `T1R1`, then a little-endian u64
length and a JSON metadata block, then one zlib-compressed blob per array,
each preceded by a little-endian u64 compressed length. The metadata block
carries per-array dtype, shape, uncompressed byte count, and sha256 of the
uncompressed bytes, so any byte flip fails closed at read time naming the
array. Writers are in `tools/t1_reference_common.py`
(`write_fixture`/`read_fixture`). Array order and the metadata are
canonically sorted, and two runs over the same checkpoint and prompts are
byte-identical (proved by `tests/test_t1_reference_decoder.py`).

## Array naming

- `prompt_token_ids` (i32), `generated_token_ids` (i32): the canonical
  greedy continuation, budget fixed in the committed `prompts.json`.
- `pos{P:04d}_layer{L:04d}_streams` (bf16, hidden-per-stream flat): the
  residual stream after layer L at prompt position P, captured at the
  anchor layers listed per prompt (`capture_layers`).
- `pos{P:04d}_layer{L:04d}_route_ids` (i32 top-k), `..._route_weights`
  (f32 top-k): the routed-expert decision, captured at every routed layer
  and position. Driver compare is exact on ids (same rounded scores must
  select the same experts under the lowest-index tie law) and banded on
  weights.
- `pos{P:04d}_head_top1_token` (i32), `..._head_top1_score` (f32): greedy
  head result, captured for the last prompt position and every generated
  position.

## Reference generation

`tools/t1_reference_decoder.py --family F --checkpoint DIR --header
model-families/F/include/sparkpipe/llm_defines.h --prompts prompts.json
--output OUT`. The engine consumes the family's `SPARK_LLM_*` header keys
and fails closed when a key disagrees with the checkpoint's `config.json`
(unambiguous keys hard-fail; keyings with known ambiguity, like glm5_next
`SPARK_LLM_MLA_V_HEAD_DIMENSION`, are recorded in the manifest as
`defines_config_mismatches` for adjudication). Each run writes
`MANIFEST.json` with the checkpoint config/index sha256, header sha256,
prompts sha256, per-fixture sha256, and generator versions.

## Driver-side comparison (serving-gated half)

The driver hook dumps the same array names (bf16 patterns for streams, the
integer routing decisions, head top-1) into a T1R1 file, then:

    tools/t1_reference_compare.py compare \
        --reference OUT/F/<prompt>.t1r --candidate DRIVER.t1r

Bands: routing ids and token ids exact; streams, logits scores, and route
weights pass at `rel <= 0.02` or `abs <= 1e-3` (bf16 cross-implementation
band; the operator may rule a tighter per-family band). The tool prints
every offending array with index and values, then `FIRST DIVERGENCE:
<array>` and exits 1. Exit 0 prints `RESULT: PASS`.

Negative control (runs anywhere, no fleet dependency): corrupt one array
of a copy and require the conviction,

    tools/t1_reference_compare.py corrupt-fixture --source F.t1r \
        --target NEG.t1r --array pos0001_layer0001_streams --offset 9
    tools/t1_reference_compare.py compare --reference F.t1r --candidate NEG.t1r

must exit 1 naming the corrupted array. `verify-manifest` re-checks every
committed fixture against its manifest sha256.

## Fixture quarantine

A family directory's `MANIFEST.json` may carry
`"quarantine": {"reason": ..., "fixtures": [...], "since": ...}`. A listed
fixture is refused everywhere: `read_fixture` raises, so `compare` and
every tool built on the reader fail, and `verify-manifest` exits 1 naming
the reason. A quarantine without a reason or fixture list, or one naming a
fixture absent from the manifest, is itself a failure. Remove the
quarantine only in the commit that regenerates the fixtures.
`tests/test_t1_reference_quarantine.py` pins this behaviour.

### e2m1 table (2026-09-28, #1288)

`t1_reference_common._E2M1_LUT` decoded e2m1 as
{0, .5, 2, 3, 4, 6, 8, 12}: every normal code 2x, the subnormal 0.5
correct. The OCP set is {0, .5, 1, 1.5, 2, 3, 4, 6}. Two generators had
absorbed the doubling as a 0.5 factor fitted against their fp8 twins,
which still left code 0.5 decoded as 0.25. With the corrected table the
fit to the twin is 1.0 without the factor (glm-5.3-flash-nvfp4-nvidia vs
bf16-official: 0.993 to 0.999 on dense and expert tensors; qwen3.8-27b
nvfp4a16 vs fp8: 0.994), so the factor is removed.

| family / arm | path | state |
|---|---|---|
| qwen38_max (nvfp4 experts) | `_E2M1_LUT` direct, no factor: experts 2x | fixtures quarantined; `model-families/qwen38_max/smoke_experts.json` derives from their route ids |
| qwen38_27b nvfp4a16 | `nvfp4_to_f32` x 0.5 / global | fixtures quarantined; top-level fp8 fixtures unaffected |
| glm53flash nvfp4 | `nvfp4_to_f32` x scale_2 x 0.5 | no committed fixtures; any nvfp4-arm fixture made before #1288 is invalid; bf16 and fp8 arms unaffected |
| mimo26 | written after the fix | valid |

Regenerate a quarantined set with `tools/t1_reference_decoder.py` on its
pinned checkpoint and drop the quarantine in the same commit.

## Position-0 anchor

For families with a committed checkpoint layer oracle, the generator's
position-0 behavior is cross-checked against that oracle on the fleet
before fixtures are trusted: glm5_next was verified against
`tools/glm5_next_checkpoint_layer_reference.py` (top-1 token equal, layer
output norms within 2.3 percent over 45 layers). A generator whose
position-0 anchor disagrees with the committed oracle is not fixture-grade.

## Wave-refs2 families (ling, gemma4 31b, laguna)

Engines: `tools/t1_reference_ling.py`, `tools/t1_reference_gemma4.py`,
`tools/t1_reference_laguna.py`, each with its `SPARK_LLM_*` pin under
`model-families/<family>/include/sparkpipe/llm_defines.h` and its canonical
prompt set under `qualification/t1_reference/<family>/prompts.json` (the
same two prompt texts as glm5_next, tokenized per family tokenizer).

- ling (BailingMoeV3 hybrid, 42 layers): KDA per the fleet-verified
  `SparkLingVal*` semantics (fused decay `exp(lower_bound * sigmoid(exp(A_log)
  * (f_proj + dt_bias)))`, per-key-channel decayed recurrent delta rule,
  grouped sigmoid router with expert bias 8 groups / 4 groups / top-8),
  MLA at layer index 5 mod 6 with absorbed latent attention and head-wise
  sigmoid gate. Generation sanity: greedy continuation of "The capital of
  France is" is " Paris". No independent publisher oracle runs offline
  (the pinned HF reference needs the fla KDA kernels); semantics were
  ported line-by-line from
  `modules/ling_resident_decode_stage/validation/spark_ling_resident_decode_stage_cuda_validation.cu`,
  which ACC-2 verified checkpoint-faithful at layers 0/21/41.
- gemma4 31b (60 layers, 5 sliding : 1 full): verified against the
  publishers' HF implementation on the fleet — the anchor oracle
  (`tools/t1_gemma4_anchor_check.py`) reproduces the committed
  `validation/gemma4_anchors` rope tables and keqv chain bitwise (0 ulp),
  and the HF cross-check (`tools/t1_gemma4_hf_check.py`) gives top-1
  agreement with equal scores on the first two decode positions and
  per-layer hidden-state agreement within 2.1 percent through layer 58;
  from the third position the greedy chains diverge on a near-tie, the
  documented cross-implementation chaos class (the fixtures pin the fleet
  conventions, which round attention scores and probs to bf16 where the
  publisher keeps them in f32). Layer semantics follow the publisher:
  attention output passes `post_attention_layernorm` before the residual
  add, per-layer `layer_scalar` scales the layer output, k_eq_v holds for
  full layers only (sliding layers have a real v_proj), and the final
  logit softcap is inert for greedy top-1.
- laguna-s 2.1 (48 layers, 3 sliding : 1 full): dual rope domains per the
  fleet plan — full layers use the yarn table (theta 5e5, factor 128,
  attention factor 1.4852030263919618, 64-dim partial rotation) and
  sliding layers theta 1e4 full-head rotation, half-split pairing,
  per-head q/k RMSNorms, softplus per-head output gating, sigmoid router
  with e_score_correction_bias top-10 with routed scaling 2.5, shared
  expert, dense layer 0. The checkpoint stores the bias as
  `mlp.experts.e_score_correction_bias` (the publisher remaps it onto the
  router at load). Before 2026-09-28 the engine looked only for
  `mlp.gate.e_score_correction_bias` and silently used zeros when it was
  absent. The engine now reads either name and refuses a layer that has
  neither. In revision 0f573140 all 47 bias vectors are exactly zero, so
  the committed fixtures do not change. The canonical T1 prompt set has no
  BOS token (id 2, which the tokenizer adds by default), and that is why
  its greedy text is degenerate (" isThe ofThe"). The fixtures remain a
  valid engine reference for those exact ids.

### laguna torch reference (publisher semantics, GPU)

`tools/laguna_reference_torch.py` is a second, independent laguna
reference. It follows `modeling_laguna.py` step by step in bf16 with the
publisher's casts: RMSNorm in f32 then cast, rope cos/sin cast to bf16, SDPA
attention, softplus gate in f32, bf16 router logits upcast before the sigmoid,
and experts accumulated in expert order. Tensors stream from the safetensors
shards through a GPU LRU of routed experts. It prefills the prompt, then
decodes with a KV cache.

```sh
python3 tools/laguna_reference_torch.py --checkpoint /mnt/model-warm/laguna-s-2.1 \
  --prompts model-families/laguna/reference_prompts.json --output out \
  --device cuda:0 --expert-cache 400
```

On the rtx5090 (torch 2.11, 32 GB) the three prompts x 16 tokens take
about 10 minutes. About 9-18 s per token is spent reading experts from the
ceph mount. The raw `out/reference.json` keeps routes, bf16 logits, timings
and the absolute checkpoint path. The committed
`model-families/laguna/reference_tokens.json` is derived from it by
`tools/laguna_reference_fixture.py`, which drops the per-run fields, rounds
the f32 top-2 logits to 4 places, decodes the generated text and records the
strict prefix of every prompt:

```sh
python3 tools/laguna_reference_fixture.py --raw out/reference.json \
  --tokenizer /mnt/model-warm/laguna-s-2.1/tokenizer.json \
  --checkpoint-label "poolside/Laguna-S-2.1 0f573140834b11cfac0c2af97a101a7a69a13e22 (bf16)" \
  --tie-margin 0.1 --output model-families/laguna/reference_tokens.json
```

Engine comparison rule: `strict_steps` is the index of the first step whose
f32 top-2 margin is below `tie_margin` (0.1). The engine must emit exactly
the reference tokens before that step, must emit one of the two f32
candidates at that step, and is not compared after it, because the
continuation depends on the tie. capital_of_france and count_up both stop at
step 13 (margins 0.006 and 0.058; at capital_of_france step 13 the bf16
argmax 340 is not the f32 argmax 22345). python_is_prime has no near tie
(smallest margin 0.73), so all 16 tokens are strict.

Tests:
- `tests/test_laguna_reference_fixture.py` (in `make test`, no torch) checks
  the derivation, its refusals (foreign generator, token/step disagreement, a
  token outside the f32 top-2, missing header fields) and that the committed
  fixture matches `reference_prompts.json` by sha256 and ids, follows
  `tie_margin`, and is byte-identical to the tool's encoding.
- `tests/test_t1_reference_engines.py` (in `make test`) checks the numpy
  engine's router bias on a single-file synthetic checkpoint: the bias is
  read under `mlp.experts.` and `mlp.gate.` with identical fixtures, a +4
  bias forces an expert into every route and a -4 bias excludes it, and a
  checkpoint without the bias is refused with an error naming
  `e_score_correction_bias`. The name lookup goes through
  `Safetensors.has`, which reads the shard header, so it works with and
  without `model.safetensors.index.json`.
- `tests/test_laguna_reference_torch.py` needs torch and is run by hand on a
  torch host. It checks that incremental decode equals full recompute
  across the sliding window, pins the sliding mask against the publisher
  rule (`key > query - window`: a query sees itself and the previous
  window-1 keys), checks the correction bias under both names and the
  refusal of a missing bias or attention sinks, and checks that the yarn
  table equals the numpy engine's.

Checkpoint staging note: the wave's fixture runs decoded from
byte-identical node-local copies of the warm checkpoints
(`/home/spark1/refs2_ckpt/...`), recorded as such in each manifest's
`checkpoint.path`; the copies' config/index sha256 in each manifest pin
content against the warm originals, and the gemma4 config sha equals the
contract freeze digest. Determinism receipts: two independent runs per
family are byte-identical per fixture; negative controls convict naming
the corrupted array; `verify-manifest` passes on both runs.
