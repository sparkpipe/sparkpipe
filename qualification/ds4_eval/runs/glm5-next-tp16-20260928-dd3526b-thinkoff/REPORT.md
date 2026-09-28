# GLM 5.3 Flash TP16 — COMPSEC-17, thinking off, 2026-09-28

**Result: 14/17.** Wrong: compsec-079 (answered 9,11,13; expected 18-19),
compsec-080 (answered 0; expected 5-6), compsec-090 (answered 0; expected
12-13). Every case ended with a stop token; 16 of 17 answered in one
`Answer:` line of 5–9 tokens.

This is the first COMPSEC pass on record for SparkPipe-served GLM 5.3 Flash.
Earlier runs sent raw pre-tokenized prompts without the chat template and
graded the first line of a 256-token completion, which scores 0/17 on this
coherent model (the model continues the prompt text and reasons before
answering).

## Protocol

- Harness: `tools/glm5_next_compsec17.py --thinking off` (this PR), fixture
  `qualification/ds4_eval/quality-fixtures-glm5.3-flash.json`
  (sha256 `a888fa69…`, see `summary.json`).
- Prompt: the fixture question decoded to text and wrapped in the GLM chat
  template, `[gMASK]<sop><|user|>\n{question}<|assistant|>\n<think></think>\n`.
- Sampling: temperature 0, max 512 tokens, one request at a time.
- Grading: `qualification/ds4_eval/compare_runs.py` (last `Answer:` line after
  `</think>`; pass iff the line set is a non-empty subset of the expected lines).
- Endpoint: `g53-api` on the rtx5090, `http://127.0.0.1:8433/v1/completions`.

## Build and deployment identity

| Component | Identity |
| --- | --- |
| Engine source | main at PR #1243 (`dd3526b`); before bundles i21–i30 |
| residentd | `0afa3721f377e980b431d394ad544f198a839f18257ef9101d0e232f01feebaa` |
| model_driver.so | `0e15456abc55f1f8b0ec7f71995909f473cecf1a6d93841f470d6070da2f14f2` |
| serving adapter (ranks) | `90209fb4c8ba439ff6f7724e45b0d15647794bacaa9c6f795c1fbf760ca60125` |
| hidden_transport.so | `538aa5b1aed7a4270427425a25020a710ecc29d46fc9b11f24842f7367cef8f5` |
| weightd | `3da98597a88b60b5` (announced `core/WEIGHTSD_BIN`) |
| fleet agent | `a05e207588da2cef` |
| API (x86, built from `dd3526b`) | `2acff5135b69411aa5eb20e4f4e50c0a29717203b4a39026d6ea751abbb598b4` |
| API serving adapter (x86) | `e0dac318cb8bef804f41766c738906e3378bdc064c39108463dd468bf11abbdc` |
| tokenizer.json | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| rank 0 pack | `318dd18ad18dc748181f22687be2a482c53f76d13c23c3056895128fa550ac80` |
| Root | `glm53flash.fp8.tp16`, model `zai-org/GLM-5.3-Flash` revision `84c6a6aa9497188e15a635ba793b0f95a79b1033` |

Runtime configuration (fleet-agent drop-in `20-serving.conf`, sha256 prefix
`8324336487eecc38`): `G5_GRAPH_PATH=1`, `G5_PIN_EXPERTS=1`,
`SPARK_TP_WAIT_MODE=hardware`. The API ran with
`SPARK_MODEL_API_MAX_PREFILL_ROWS=8` as the workaround for single-sequence
prefill waves above 8 rows (fixed by PR #1255, not yet deployed).

## Consistency with the eager configuration

The same 17 prompts produced byte-identical completions earlier the same day
with the fleet in eager/spin mode (no expert pinning), so the graph path did
not change any token.
