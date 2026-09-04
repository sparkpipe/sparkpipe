# Tokenizer subsystem audit — multi-model + foreign-drafting fitness

Date: 2026-09-04. Scope: text/tokenizer.c, text/tokenizer_sidecar.c, all consumers. Read-only
audit against main at 16fb012. Verdict: the tokenizer CORE is solid (thread-safe, dual-format,
piece cache, 92-case GLM ground truth); every layer ABOVE it assumes one model, one tokenizer,
one id space — and that assumption is what foreign (cross-tokenizer) drafting collides with.

## What exists

- Embedded C byte-level BPE (text/tokenizer.c), three load formats (HF tokenizer.json, tiktoken
  ranks, compiled binary), sidecar wrapper content-sniffs the format. Caller-owned workspaces;
  the tokenizer itself is read-only shared state.
- Binding: per-deployment JSON member "tokenizer": {"path": ...} (model_resident_deployment.c),
  loaded by node/model_api.c. NOT bound in the model contract; a wrong-asset load starts cleanly
  (only prints the vocab count — no agreement check against the model's output vocab).
- Singleton: one tokenizer per model_api process (static file scope, model_api.c:66-69).
  Multi-model = process isolation + LiteLLM routing.
- Stop tokens: env SPARK_EOS_TOKEN_IDS at serving time; contracts carry eos_token_ids; the two
  are not tied together. Deployment generators do not emit the tokenizer member at all.
- Chat: /v1/chat/completions routes to the completion handler; messages requests fail with
  "prompt_token_ids required". The one chat template is GLM52-hardcoded and offline-only.

## Problems by severity (fix direction in one line)

1. No tokenizer↔model binding validation — wrong asset serves wrong ids silently. Fix: expected
   vocab count + asset sha256 in the deployment member, refuse start on mismatch.
2. The speculation stack hard-assumes a shared id space: bridge wire format carries bare u32
   token ids with no tokenizer identity; policy requires draft_vocab_size <= vocab_size and
   compares ids numerically. Fix: tokenizer identity on bridge records + a detok->retok mapping
   stage before verify + verify on text offsets, not id indices.
3. Cross-tokenizer text exchange is lossy at special tokens (added tokens decode to literal
   marker text; unknown ids hard-fail decode with no byte fallback). Fix: byte-level decode mode
   for specials, or explicit id->id mapping tables for the shared special subset.
4. Singleton sidecar blocks holding target + drafter tokenizers in one edge process. Fix: hang
   the sidecar off the deployment struct, pass by pointer.
5. Stop-token config drift (env vs contract). Fix: generate the deployment's stop list +
   tokenizer member from the model contract in the *_gen_deployment tools.
6. Chat endpoint is a stub. Fix: reject chat explicitly until templates are per-deployment data.
7. Silent pretokenizer-pattern fallback (only two GPT-4-style patterns recognized; K3's real
   pattern unimplemented). Fix: fail loudly on unrecognized Split pattern.
8. Compiled tokenizer format silently drops ignore_merges/rank_ordered_merges. Fix: format v3 or
   stop serving from compiled assets.
9. Test gaps: no chat-template test, no mismatch-startup test, no cross-tokenizer round-trip;
   only GLM has id-level ground truth.

## Consequence for the draft farm

Same-tokenizer drafters (every DFlash/DFlash2/DSpark checkpoint on the 5090 — they are trained
per target and share its vocab) need nothing from this list. A FOREIGN drafter (e.g. a Qwen LM
drafting GLM 5.3) needs items 2+3+4 as a prerequisite chain: tokenizer identity on the wire,
byte-safe cross-decode/encode, and a dual-tokenizer edge. Until then the farm's foreign bit
stays unassigned — honest state.
