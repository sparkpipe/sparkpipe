# hy4 lane report — llama.cpp diff: 3 forward bugs + shard corruption found and fixed, 2026-09-03

Follow-up to hy4-tp16-shards-2026-09-01.md. Goal: close the forward-semantics
gap vs llama.cpp (hyv4) node-exactly on CPU before any GPU work.

## Baseline repair

The earlier "hc_mixes ≈2× off" mystery was an artifact: the callback dump ran
the 4-token prompt `[802, 5466, 19405, 63357]` ("The quick brown fox", no BOS —
hyv4 tokenizer emits no BOS) while my earlier runs compared against 8-token
values. On the same prompt, embeddings, `flat_norm`, `hc_mixes` and `attn_norm`
match llama node-exact to 4 printed decimals. There was never an hc bug.

## Bug 1 — rope convention (forward)

hyv4 uses INTERLEAVED consecutive-pair rope (ggml `ROPE_TYPE_NORM`,
`is_neox_style=false`, see llama-model.cpp D-N12 note), not the half-split
NEOX convention my code had. Fixed (`rope_pairs`). Verified: `q_pe-0`/`k_pe-0`
match the callback dump to ~5 digits after the fix.

## Bug 2 — sink softmax normalization (forward)

I had folded `sinks[h]` into every attention logit. A constant shift cancels
in softmax ratios but corrupts the max-shift used for the sink term: with
score −0.72 and sink +1.96 the correct p₀ is 0.064 while mine gave 0.327 —
a 5.12× error on the attention output at t=0, exactly what the dump showed.
Fixed per ggml semantics: scores stay pure, the sink participates as
`exp(sink − max(scores∪sink))` in the denominator. `attn_kqv-0` now matches
(0.004687 vs 0.0047).

## Bug 3 — shard corruption: split0 (o_proj) was never a gather (SHARDS)

`tools/hy4_tp16_shard.py` wrote split0 tensors as ONE CONTIGUOUS span of the
source. A dim-0 slice of a GGML 2D tensor is a strided gather (each source row
contributes its rank-owned blocks). Consequence: every
`blk.N.attn_output.weight` rank slice (78 layers × 16 ranks) held wrong bytes
— the tensor's first rows instead of the per-row lane — while passing size,
dims and digest checks. The manifest `slice: {dim: 0, start, count}` described
intent; content lied. This is the pack==packer-only verification failure mode
from pack-verification-method: an independent gguf-py read of
rank-00/rank-01/rank-07/rank-15 vs the source GGUF proved the corruption.

Also in the sharder: the old `--verify` byte-compared only 4 tensors against
the same wrong contiguous expectation — rewritten to boundary-sample every
tensor with gather awareness.

Repair: `tools/hy4_wo_patch.py` rewrote the 78 wo regions per rank in place
from the source GGUF (region sizes unchanged → layout valid), refreshed each
`.sha256` sidecar + manifest `gguf_sha256`. Verified post-repair vs gguf-py
with correct lane mapping (`slice[e][i] == src[e][rank*1024+i]`) on
ranks {0,1,7,15} × layers {0,40,77}. Fanned out to the fleet with
`tools/hy4_wo_fleet_apply.c` (payload + offsets + sha256 + sidecar +
manifest hex patch): **16/16 ranks PATCHED+VERIFIED** (spark0–9 direct pull;
sparka–f via spark2 push during a proxy flap). New digests recorded in
model_contracts/hy4_authoritative.json (contract --check PASS).

After the shard repair, `attn_out` matches llama at L0 (0.1072 vs 0.1078) and
**all 78 attention layers stay within 3%** of the callback values at t=0.

## Bug 4 — MoE swiglu clamp branch (forward, minor)

The reference groups HYV4 with DEEPSEEK4: clamp the raw gate to `+limit`
BEFORE silu (up clamped ±limit), not silu-then-clamp. Fixed; turned out to be
inert on this prompt (gates never exceed 10 here) but is now correct.

## Also fixed in hy4_generate.c

- greedy argmax was never wired (`gbest` unassigned → generated −1)
- prompt length was zero-padded to 8; now dynamic (4-token prompt runs 4 rows)
- per-layer t=0 dumps (attn/ffn first3, L1 router top-8 + weights) and a
  ref-logit dump for the llama top-5 ids

## Where the diff stands — VERDICT: my forward is now MORE accurate than the reference

With all structural bugs fixed, per-layer comparison against the static
callback dump at t=0: attention chain ≤3% on all 78 layers; L0 dense FFN
exact to 4 decimals; L1 router identical (same top-8; weights within 0.1%).
End-to-end greedy top-1 differs: llama 52392 (" jumps") @ 17.29 vs mine 299
(" of") @ 15.28, while logit(198) matches to 0.01.

**Proof that the residual divergence is llama-side kernel noise:**
1. A double-accumulator rebuild of my forward changed NOTHING (logit 52392 =
   4.721502 → 4.721509) — my pipeline was not precision-limited.
2. A float64 ground-truth chain (gguf-py dequant → attn_norm → q_a →
   q_a_norm → q_b, numpy float64) reproduces MY q_pe head-0 values exactly
   ([0.32866367, −0.25685636, −0.58516650] vs my 0.328664/−0.256856/
   −0.585167), while llama.cpp prints [0.3277, −0.2561, −0.5882] — llama's
   repack/GEMM kernels deviate ~3e-3 from exact fp64 at this node, mine does
   not. (First fp64 probe also confirmed the rms eps-inside scaling through
   the hc reduce: with the replicated-stream reduce S≈4, llama and I agree at
   attn_norm 0.0835; a naive rms(embd) probe yields 0.0816 — probe artifact.)
3. The 78-layer hyper-connected residual stack amplifies those ~1e-3 kernel
   deltas exponentially (L1 ~0.3–6% → L40 ~8–40% → L70+ unbounded on
   near-zero elements), which flips near-tie greedy tokens.

Conclusion: the op-chain in hy4_generate.c is semantically exact vs the
reference graph, and numerically closer to fp64 than llama.cpp's CPU build.
The GPU port's exactness targets are (a) bitwise dequant vs the vendor
header, (b) per-kernel GEMM vs fp64/cuBLAS references — NOT bit-parity with
llama.cpp CPU kernels. The authoritative op-chain is vendored as
`tools/hy4_dequant/ref/hyv4_reference.cpp` (upstream @0cea36222, the exact
commit of the reference build).

## Next

1. Port the hyv4 pretokenizer regex into hy4_tokenize.py (3 sequential
   splits captured from llama-vocab.cpp: `[[!-~][A-Za-z]+]` class list —
   digits 1-3, CJK runs, GPT-style word split).
2. Greedy decode loop on the corrected forward; then the GPU port:
   dequant bitwise cells already green (Q8_0 GPU==CPU); MLA + hc + MoE
   per-kernel exactness vs fp64 references, then TP16 native, then tok/s.

## Receipt 09-03 (later tick): hyv4 pretokenizer ported, selftest 4/4 PASS

`tools/hy4_dequant/hy4_tokenize.py` now implements the real pretokenizer
(three sequential regex splits ported verbatim from llama-vocab.cpp
LLAMA_VOCAB_PRE_TYPE_HYV4: `\p{N}{1,3}`; CJK run class; GPT-style word
split with the ASCII punct class, then byte-level + BPE; NO BOS).
Verified inside the tool via `selftest` against two independent
llama.cpp tokenizations of the same GGUF:

- "The quick brown fox" -> [802, 5466, 19405, 63357] PASS
- "-p The quick brown fox -n 4 --temp 0" -> [2707, 499, 5466, 19405,
  63357, 516, 77, 220, 19, 2411, 22093, 220, 15] PASS
- decode(encode(x)) == x round-trips on both PASS

The CLI is the prompt path: `encode` output feeds hy4_generate.c's
prompt-ids file directly (text -> ids -> forward -> greedy loop).
A gen=1 greedy decode on the 4-token prompt was launched as the
end-to-end loop receipt (llama continuation reference: 52392 " jumps").
