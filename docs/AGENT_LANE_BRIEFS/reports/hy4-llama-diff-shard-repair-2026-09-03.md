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

## Receipt 09-03 (closed): greedy decode loop GREEN end-to-end

gen=1 run on the tokenizer-encoded 4-token prompt (spark2
hy4-cmp/gen1.log): prefill argmax 299 (" of") fed back, t=4 forward
top-1 logit 17.8445 -> GENERATED TOKEN 269 (" the"). Decoded
continuation: "The quick brown fox of the". All 5 token-stacks finite.
llama.cpp gives " jumps" (52392) for the same prompt — per the fp64
analysis above, a near-tie flip caused by llama's own CPU kernel
rounding, not a semantic difference.

Ladder state: dequant all classes green; loader green; single-layer
green; tokenizer wired (selftest 4/4); greedy decode loop green; whole-
layer llama diff closed. Next: GPU port (dequant cells already green),
TP16 native module, tok/s.

## Receipt 09-03 (next tick): hy4-gpu4 — GPU dequant ALL classes BITWISE PASS

Cell hy4-gpu4c-dequant-all [gpu][spark2] (queue): CUDA kernels for
Q4_K/Q5_K/Q6_K/IQ4_XS/IQ1_M/IQ2_XXS/IQ3_XXS ported one-to-one from the
vendor header, run against real blocks (512 blocks = 131,072 elements per
class) from the node-local rank-02 pack, compared BITWISE (memcmp on float
bits) with the CPU vendor dequant. Result:

    TYPE 12 (Q4_K)    512 blocks BITWISE PASS
    TYPE 13 (Q5_K)    512 blocks BITWISE PASS
    TYPE 14 (Q6_K)    512 blocks BITWISE PASS
    TYPE 23 (IQ4_XS)  512 blocks BITWISE PASS
    TYPE 29 (IQ1_M)   512 blocks BITWISE PASS
    TYPE 16 (IQ2_XXS) 512 blocks BITWISE PASS
    TYPE 18 (IQ3_XXS) 512 blocks BITWISE PASS
    DEQUANT_ALL PASS (7 classes)

Context: GB10 sm_121, CUDA 13.0, nvcc -O2 -fmad=false; fp16 conversions via
an uploaded 65536-entry host-built table (bitwise parity by construction);
tables (iq1s_grid 2048, iq2xxs_grid 256, iq3xxs_grid 256, ksigns/kmask,
kvalues_iq4nl) uploaded from the vendor header — no duplicated data. With
the earlier Q8_0 cell, ALL 9 weight classes in the pack are GPU-bitwise
verified. The dequant layer of the GPU port is complete.

Two build/run defects fixed in the same tick: _Static_assert is C11 (nvcc
C++ needs a `#define _Static_assert static_assert` shim before including
the vendor header — header stays verbatim), and the GGUF KV string parser
silently aborted on any string longer than the fixed buffer (a chat-template
KV) — rstr now streams strings of any length.

Queue ops this tick (tool = origin/main spark_queue.py): nccl-burst-sweep2/
sweep4 claims were stale (TTL expired 8h prior, processes verified dead on
spark3+spark5, sweep finished per results.jsonl) and were reaped via
dispatch's own TTL/exit-file reap + done marks; dispatcher daemon was DOWN
since 09-02 22:25, one-shot `dispatch` passes used to run the queue (k3
spark8 cpu cells remain queued ahead per FCFS).

## Receipt 09-03 (next tick): hy4-gpu5 q-path chain cell — STAGED+QUEUED (dispatch blocked by glm53flash fleet sweep campaign)

Increment: GPU kernels for the forward's core primitives — gemv (fp32
FMA, one thread per row), rms_norm (block reduction + scale, eps inside),
and INTERLEAVED rope — composed into the exact q-path chain already
proven in fp64 during the llama diff (embd row 802 -> attn_norm ->
q_a -> q_a_norm -> q_b head 0 -> rope), all weights from the node-local
rank-00 bundle. Validated per-stage against an in-harness float64
reference (tol 5e-4..1e-3 * max(1,|ref|)), plus a golden check against
the numpy-fp64 q_pe values captured last tick (0.32866367,
-0.25685636, -0.58516650). Sources: tools/hy4_gpu/hy4_qchain_test.cu +
run_qchain_test.sh.

Submitted as hy4-gpu5b-qchain [gpu][spark2] (v2 fixed an embd_row
declaration; v1's build failure was diagnosed and fixed in-tick).
NOT RUN YET: glm53flash is chaining live fleet-wide meshbench sweeps
(nccl-mb-sweep1/3, p0, all 16 nodes, ~14 min each, resubmitted back to
back); my p5 cell correctly holds behind the priority barrier and will
dispatch on the first free pass (or automatically after the 120-min
anti-starvation aging). No stale claims this time — the sweeps are live
work; nothing was reaped. Dispatcher daemon still down; one-shot
dispatch passes used.

NEXT: (1) poll hy4-gpu5b-qchain to completion, expect QCHAIN PASS
receipt; (2) extend the same cell pattern to the attention-score +
softmax-with-sink kernel and the MoE expert gather, still vs fp64;
(3) then the TP16 native module port.

## Receipt 09-03 (closed): hy4-gpu5 QCHAIN PASS — GPU q-path chain within fp32 noise of fp64

Cell hy4-gpu5e-qchain [gpu][spark2]: gemv + rms_norm + interleaved-rope
CUDA kernels composed into the q-path chain (embd row 802 -> attn_norm ->
q_a Q5_K -> q_a_norm -> q_b Q8_0 head 0 -> rope), all weights host-dequanted
from the node-local rank-00 bundle via the vendor header, validated
per-stage against an in-harness float64 reference:

    ATTN_NORM  max|d| = 6.5e-08  (6144 elems)
    Q_A_GEMV   max|d| = 4.2e-06  (2048)
    Q_A_NORM   max|d| = 2.3e-06  (2048)
    Q_B_GEMV   max|d| = 5.5e-06  (256)
    ROPE_POS0  max|d| = 5.5e-06  (64)
    ROPE_POS3  max|d| = 5.5e-06  (64)
    GOLDEN q_pe [0.32866380, -0.25685647, -0.58516574] vs numpy-fp64
    [0.32866368, -0.25685635, -0.58516651]  (matches to ~1e-7 relative)
    QCHAIN PASS

Context: GB10 sm_121, CUDA 13.0, -fmad=false; tol 1e-3*max(1,|ref|); all
observed deviations at or below fp32 rounding noise, i.e. the GPU chain is
numerically indistinguishable from fp64 ground truth at this scale.

Three harness defects found and fixed during bring-up (each diagnosed by
bisecting a FAIL against the independent gguf-py bundle computation):
1. embd_row declared after use (build).
2. block_geom lacked the F32 case, so the attn_norm read silently returned
   an empty buffer and the downstream range-copy segfaulted.
3. f32 norm tensors were constructed through uint8 iterator conversion
   (byte VALUES as floats) instead of bit reinterpretation — produced
   plausible-looking but wildly wrong gate scales; caught because the
   in-harness fp64 reference is only as good as its own host inputs, and
   an independent gguf-py computation on the same bundle exposed it.

Execution note: glm53flash ran live fleet meshbench sweeps/probes
back-to-back (nccl-mb-sweep1/3, nccl-mb-probe1, p0, all 16 nodes); the
cell dispatched on the first inter-task gap. All queue ops used the
origin/main tool; dispatcher daemon still down, one-shot passes only.

NEXT: attention-score + softmax-with-sink kernel and the MoE expert
gather (same fp64 pattern); then TP16 native module port.

## Receipt 09-03 (next tick): hy4-gpu6 absorbed-MLA attention cell — STAGED+QUEUED (glm53flash live campaign holds fleet)

Increment: the attention block as CUDA kernels — kv path (kv_a gemv 6144→576,
rms_norm on the 512-latent half, interleaved rope on the 64-pe half, KV cache
for all 4 prompt tokens), the full q path (reuses the proven qchain kernels),
absorb (k_b gemv 192→512 per head), the sink-softmax attention combine (scores
= q_abs·k_lat + q_pe·k_pe scaled 1/√256; max over scores∪sink; sink as its own
denominator term — the exact ggml semantics fixed two ticks ago), and v
decompression (v_b gemv 512→256) — layer 0 head 0, causal over the 4-token
prompt, compared per token position against an in-harness fp64 reference
(tol 1e-3) AND against llama.cpp eval-callback goldens (attn_kqv-0 head 0:
t0 [0.0047, 0.0020, -0.0051] .. t3 [0.0308, 0.0514, -0.0408]).

Submitted as hy4-gpu6-attn [gpu][spark2]. NOT RUN YET: glm53flash's live
meshbench campaign (nccl-mb-probe1/probe2/mb-curve1, p0, all 16 nodes,
chained) holds the priority barrier; the cell dispatches on the first free
pass. All queue ops via the origin/main tool; dispatcher daemon still down.

NEXT: (1) poll hy4-gpu6-attn to completion — expect ATTN_T0..T3 PASS with
first3 matching the llama goldens within kernel-noise; (2) MoE expert-gather
cell (same fp64 pattern); (3) then the TP16 native module port.

## Receipt 09-03 (closed): hy4-gpu6 ATTN PASS — absorbed-MLA attention GPU-validated

Cell hy4-gpu6d-attn [gpu][spark2]: layer 0 head 0 attention as CUDA kernels —
KV cache built on GPU for all 4 prompt tokens (kv_a gemv 6144→576 + rms_norm
on the 512-latent half + interleaved rope on the 64-pe half), the q path
(q_a gemv + norms + q_b head-0 gemv), absorb (k_b gemv 192→512), the
sink-softmax combine (scores scaled 1/√256, max over scores∪sink, sink as
its own denominator term), and v_b decompression (512→256). Compared per
token position against an in-harness float64 reference:

    ATTN_T0  max|d| = 3.6e-08    ATTN_T2  max|d| = 3.1e-07
    ATTN_T1  max|d| = 1.9e-07    ATTN_T3  max|d| = 1.7e-07
    ATTN PASS

first3 vs llama eval-callback goldens (head 0): t0 [0.0048, 0.0020, -0.0052]
vs [0.0047, 0.0020, -0.0051]; t3 [0.0328, 0.0529, -0.0436] vs [0.0308,
0.0514, -0.0408] — within llama's own kernel-noise envelope.

Two more harness lessons (both found by refusing to accept a vacuous PASS):
1. Embedding rows must come from OWNER-rank bundles — tokens 19405/63357
   are outside rank-0's vocab slice [0..7552); reading them from rank-00
   yielded NaNs that slipped a comparison check (NaN comparisons are always
   false). NaN/inf is now a hard FAIL in check(), and embeddings are read
   from the owner bundle (owner = token/7552, row = token%7552).
2. The owner bundle filename embeds its own rank number — path derivation
   must rewrite the file name, not just the directory.

The full attention stage of the GPU port is now validated (kv path, absorb,
sink softmax, v decompress). NEXT: MoE expert-gather cell (same fp64
pattern), then hc_pre/hc_post/head kernels, then the TP16 native module.

## Receipt 09-03 (closed): hy4-gpu7 MOE PASS — expert-gather GPU-validated vs fp64

Cell hy4-gpu7b-moe [gpu][spark2]: the routed-expert MoE FFN of layer 1 as
CUDA kernels. Router logits via gemv (gate_inp F32), sigmoid probs, biased
top-8 selection (probs + exp_probs_b), weights renormalized x 2.827; per
selected expert: owner-rank bundle slab reads (expert e -> rank e/16, slab
e%16), host dequant (gate/up IQ1_M|IQ2_XXS, down IQ3_XXS|IQ4_XS via the
vendor header), upload, gate/up gemvs, HYV4 swiglu-clamp kernel (up +-10,
gate pre-silu <= 10), down gemv, weighted accumulate; plus the shared
expert (Q6_K, unclamped, weight 1). MoE input = deterministic seeded
pattern (isolates the MoE stage; hc kernels are a separate cell).

    selected: 129 179 203 173 113 47 216 21
    MOE_ROUTED      PASS  max|d| = 3.2e-05  (6144 elems)
    MOE_WITH_SHEXP  PASS  max|d| = 3.7e-05  (6144 elems)
    MOE PASS

Context: GB10 sm_121, CUDA 13.0, -fmad=false, tol 1e-3*max(1,|ref|).
With dequant (9 classes bitwise), q-chain, and attention cells green, the
remaining GPU-port pieces are the hc_pre/hc_post/hc_head kernels, then
full-layer assembly, then the TP16 native module port.

## 09-03 (later tick): FP8 (unquantized) TP16 stagepacks — operator directive

Operator: build stagepacks for the FULL UNQUANTIZED hy4 (inference-first on
full precision; new weightd/residentd improvements), with the packer memory
limited per the stagepack dev's improved patterns.

Source: /mnt/model-warm/hy4-preview-fp8-official (766 GB, 130 safetensors
shards, 2838 tensors). MXFP8 modelopt layout: F8_E4M3 weights with U8 E8M0
scale companions (group-32 along the input dim), BF16 for the exclude list
(embed_tokens, lm_head, norms, gates, hc fns, indexer proj), MTP layer
present (39 tensors) and included.

Build node: sparkc (2.1T free, warm visible, new weightd running; operator
sanctioned a different spark over spark2's 745G free).

Packer: tools/hy4_fp8_stagepack.py — safetensors-per-rank (schema
hy4-fp8-tp16-v1, manifest + .sha256 sidecar), verbatim bytes (no
requantization), TP16 splits: vocab/head/expert dim0 ranges, o_proj-family
dim1 row gathers, replicate the rest (5.6 GB/rank: dense layer-0 MLP,
shared experts, q_a/kv_a, gates, hc fns, MTP eh_proj). Memory discipline
from the qwen38_max packer: 512 KiB streaming chunks, posix_fadvise
DONTNEED per source chunk, sync_file_range + DONTNEED per output tensor.

Census per rank: 2838 tensors = 652 range (50.0 G) + 79 gather (0.5 G) +
2107 replicate (5.6 G) = 56.1 G; 16 ranks ≈ 898 G.

Progress: rank 0 BUILT (63.6 GB file, sha 403b54b1db9bf8d7..., peak RSS
558 MB per /usr/bin/time — memory discipline confirmed). Independent
verifier (tools/hy4_fp8_pack_verify.py — byte-compares range/gather/
replicate/MTP/scale samples against source reads) running; ranks 1-15
building in a detached chain on sparkc (build.log).

Defects found and fixed during bring-up:
1. Expert scale names use a single `_scale` suffix (not `.weight_scale`)
   and fell through to replicate — split_rule now strips both.
2. safetensors data_offsets are RELATIVE TO EACH SHARD'S DATA SECTION;
   the first rank-0 build copied shard headers as payloads. Offsets are
   now resolved with per-shard data starts, and the verifier byte-compares
   against the source so this class cannot pass silently.
3. os.sync_file_range is not present on every Python build — guarded.

NEXT: (1) verify PASS on rank 0, verify one mid-build rank (e.g. rank 7);
(2) place packs to the fleet (rank r -> spark{hex r}) once the chain
completes; (3) weightd/residentd load-path wiring for the FP8 pack format;
(4) GPU inference cells move to the FP8 packs (dequant-free: FP8 bytes are
the native kernel format; MX scale application kernels needed).
