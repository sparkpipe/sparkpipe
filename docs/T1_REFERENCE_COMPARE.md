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
  with e_score_correction_bias (zeros when absent) top-10 with routed
  scaling 2.5, shared expert, dense layer 0.

Checkpoint staging note: the wave's fixture runs decoded from
byte-identical node-local copies of the warm checkpoints
(`/home/spark1/refs2_ckpt/...`), recorded as such in each manifest's
`checkpoint.path`; the copies' config/index sha256 in each manifest pin
content against the warm originals, and the gemma4 config sha equals the
contract freeze digest. Determinism receipts: two independent runs per
family are byte-identical per fixture; negative controls convict naming
the corrupted array; `verify-manifest` passes on both runs.
