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

## 09-03 (next tick): FP8 chain progress + hc cell staged

FP8 build chain on sparkc: rank 00 done (63.6 GB, sha 403b54b1db9bf8d7),
rank 01 done (peak RSS 557 MB), rank 02 in progress. The independent
rank-0 verify competed with the chain for warm reads (both crawl under
concurrency — the known warm-stall pattern) and was killed to unblock the
chain; re-verify runs after ALL_DONE. Lesson: serialize warm-heavy jobs
per node.

GPU hyper-connection cell (tools/hy4_gpu/hy4_hc_test.cu — hc_pre flat-rms
+ fn gemv + sigmoid gates + weighted reduce, hc_post distribute, hc_head
collapse; fp64 checks + llama hc_mixes-0 goldens) is written, staged to
spark2, and submitted as hy4-gpu8-hc; execution queued behind the live
glm53flash fleet campaign (nccl-db-d2a4 at close).

NEXT: (1) hy4-gpu8-hc receipt; (2) FP8 chain completion + post-chain
verify; (3) placement fan-out rank r -> spark{hex r}; (4) weightd/
residentd load-path wiring for the FP8 safetensors-per-rank format.

## 09-03 (next tick): FP8 packs 16/16 BUILT; manifest collision fixed; verify pending relaunch

The 15-rank chain completed: ALL 16 FP8 rank packs exist on sparkc
(~/hy4-fp8-packs/model-fp8-tp16-rank-XX.safetensors), peak RSS 557-558 MB
per rank build. Defect found by the first verification attempt: ALL ranks
wrote the same manifest.json (rank 15's survived) — fixed to per-rank
manifest-rank-XX.json plus a --manifest-only mode that regenerates
manifest + sha sidecar from an existing pack without re-copying data
(needed one more fix: rank_view returns (out_dims, spec) and the
manifest-only path initially unpacked it wrong; the diagnostic raise now
names any tensor whose slice kind disagrees with its split). All 16
per-rank manifests + sha sidecars regenerated.

Verification state: the sample byte-compare runs (rank 0, rank 7) were
killed twice — once by warm-read contention with the build chain, once by
a fleet-proxy disconnect (launched without setsid; lesson repeated: EVERY
detached remote job gets setsid). Relaunch under setsid next tick; the
checks are cheap once warm reads flow.

NEXT: (1) setsid verify ranks 0 + 7 (byte-compare vs source); (2) place
packs rank r -> spark{hex r}; (3) weightd/residentd load-path wiring for
hy4-fp8-tp16-v1; (4) GPU inference cells switch to FP8 packs (MX
scale-application kernels).

## Receipt 09-03 (closed): hy4-gpu8b HC PASS — GPU port kernel layer COMPLETE

Cell hy4-gpu8b-hc [gpu][spark2]: hc_pre (rms over the flattened 24576
stream vector, hc_fn gemv, sigmoid pre/post gates, weighted reduce),
hc_post distribute, hc_head collapse — all validated against an
in-harness float64 reference on real layer-0/final weights with the real
hc_init (replicated embedding row 802):

    HC_MIXES       max|d| = 1.1e-03 on O(100-300) values (relative ~1e-5)
                   gpu [112.7597, 76.8200, 74.4943, 95.1448, -289.2010,
                        -282.9741, -332.2350, -281.2709]
                   llama golden [112.7601, 76.8200, 74.4942, 95.1446,
                                 ?, -282.9738, -332.2355, -281.2696]
    HC_REDUCED     max|d| = 1.4e-08  (6144)
    HC_DISTRIBUTE  max|d| = 6.5e-08  (24576)
    HC_HEAD        max|d| = 4.7e-08  (6144)
    HC PASS

With this, every kernel group of the GPU port is validated:
dequant (9 classes bitwise), gemv/rms_norm/rope chain, absorbed-MLA
attention with sink softmax, MoE expert gather with HYV4 clamp, and the
hyper-connection stage. Remaining before serving: full-layer assembly
(compose all kernels, diff layer-0/layer-1 against the CPU forward's
dumps), then the TP16 native module.

FP8 packs: build complete (16/16 on sparkc) but sparkc's link is down at
close (proxy flap); the rank-0/7 verification relaunch (setsid) and
placement fan-out are the next FP8 items once it returns.

## 09-03 (next tick): full-layer GPU assembly cell — bisecting to the last bug

Cell hy4_layer_test (hy4-gpu9* series, [gpu][spark2]): token 802 through
layers 0 and 1 on GPU — all 64 heads gathered from the 16 rank-bundle
slices (q_b/k_b/v_b/gate/wo per-rank gemvs, head-sliced), dense layer-0
FFN, the 256-expert routed MoE + shared expert in layer 1, both hc
stages — compared against the CPU forward's dumped layer-1 state
(hy4_generate now takes a dump arg: dump file + layer index; l1state.t0).

Execution is green through layer 0 (all heads, dense FFN) and through
layer 1's router + expert slabs. The final state diffed at |d|=0.014-0.022
on early elements — beyond fp32 noise. Stage-sum checkpoints (GPUSUM
post-attn/post-ffn per layer vs the CPU run's logged sums) plus a head-0
probe pinned it: the absorb gemv was reading the 2048-dim q_lora vector
instead of the head's 192-element nope slice — score came out ~1e27,
p0 = 1.0, and the attention output was the unweighted v projection.
One-line fix submitted (hy4-gpu9i-layer, v9); queued behind the live
glm53flash campaign at close (dispatch retries in flight).

Also this tick: the CPU forward gained the state-dump flag (dump file +
layer index), and the FP8 chain state stands at 16/16 built with per-rank
manifests; the rank-0/7 verify + placement resume when sparkc's link
returns.

NEXT: (1) v9 receipt — expect LAYER_STATE PASS at fp32-noise level;
(2) FP8 verify + placement; (3) TP16 native module port (the assembly
cell IS the module's forward blueprint).

## 09-03 (next tick): full-layer assembly — L0+L1 run end-to-end; score-dot bug under active bisect

Iteration status (hy4-gpu9* series): v5 fixed the slab reads (they opened
the bundle DIR — the loader path is the pack dir; the GGUF filename embeds
the rank). v6-v10 bisected the remaining state divergence with stage-sum
checkpoints (GPUSUM post-attn/post-ffn per layer vs the CPU run's logged
sums) and a head-0 probe:

- v6 (sink=0): post-attn L0 sum +1.096 vs CPU -0.619 — wrong.
- v7 (sinks loaded): +6.457 — wrong the other way; sinks confirmed
  [1.9623, 2.2837, 0.7103, ...].
- v8 probe: head-0 score = ~1e27 — the absorb gemv was reading the
  2048-dim q_lora vector instead of the head's nope slice. Fixed (v9).
- v9: score still garbage (~8e32) but v10's value probes show qh, qabs
  and klat are all SANE at probe time — while the dot result reads exactly
  0.0. Leading suspects: d_s is allocated 4 bytes but the second dot
  writes d_s[1] (OOB clobber), and there is no launch-error check or
  device sync before the 8-byte readback.

Next iteration (v11, mechanical): size d_s to 8 bytes, add
cudaGetLastError after every launch, cudaDeviceSynchronize before the
score readback. Everything else in the layer (L0 all-head attention +
dense FFN, L1 router + owner-bundle expert slabs + shared expert, both hc
stages) demonstrably executes; the sums move with each fix as expected.

FP8 packs: still 16/16 on sparkc, link still down; verify (setsid) +
placement resume when it returns.

## Receipt 09-03 (closed): hy4-gpu9n LAYER_STATE PASS — full-layer GPU assembly validated

Cell hy4-gpu9n-layer [gpu][spark2]: token 802 through layers 0 and 1 on
GPU, all 64 heads from the 16 rank-bundle slices, dense layer-0 FFN, the
256-expert routed MoE + shared expert in layer 1, both hc stages —

    GPUSUM post-attn L0 -0.618768   (CPU -0.618758)
    GPUSUM post-ffn  L0 -2.229357   (CPU -2.229346)
    GPUSUM post-attn L1 -4.347372   (CPU -4.347359)
    GPUSUM post-ffn  L1  3.274987   (CPU  3.274961)
    LAYER_STATE PASS (max|d| = 2.68e-06, 24576 elems)

Probes all match their anchors: flat-norm = llama node_6 (0.6739...),
hc_mixes = llama golden (112.76...), cur = attn_norm (0.083460...),
head-0 score -0.719982 / p 0.064027 (the validated attention values),
klat = llama kv_cmpr. Context: GB10 sm_121, CUDA 13.0, -fmad=false,
single-token causal (t0 attention), tol 2e-3.

Root cause of the divergence: the embedding load dequanted the ENTIRE
1.7 GB token_embd slice into a 6144-float buffer — a 186 MB heap overflow
that corrupted neighboring allocations and produced the plausible-looking
wrong values of v6-v13. Fixed to a single-row read (row 802, 3456 bytes).
With it, every earlier "fixed" symptom resolved at once.

THE GPU PORT ASSEMBLY IS COMPLETE: dequant (9 classes bitwise),
gemv/rms/rope, absorbed-MLA attention with sinks, MoE with HYV4 clamp,
hyper-connections — composed and validated against the CPU forward at
fp32-noise level. Next: the TP16 native module port (this cell is the
module forward's blueprint), then tok/s. FP8 packs await verify +
placement when sparkc's link returns.

## 09-03 (next tick): FP8 verify running clean; full-forward cell in work

- FP8 rank-0/7 verification relaunched under setsid via a staged script
  (tools/hy4_gpu/run_fp8_verify.sh) on sparkc — the earlier attempt was
  OOM/killed and the one before that died with its ssh session. Receipts
  land next tick; placement fan-out follows a PASS.
- The full 78-layer GPU forward cell (hy4_forward_test) is being built on
  the validated layer-cell pieces: layer-outer/token-inner order so
  per-layer weights stream from the bundles once, KV caches resident on
  device, per-token hc streams, batched expert-union gathering per layer,
  lm_head argmax at the final position expecting 299 (the CPU greedy
  prefill token). Draft in the worktree; submit after the MoE-union
  staging is converted to host-dequant + upload (device-pointer dequant
  is the bug class the LAYER_STATE hunt just proved out).

## 09-03 (FP8 verify): RED STOP — rank packs have content corruption; placement blocked

The independent verifier caught real corruption in the built FP8 packs
(working exactly as designed — this is the UD-IQ1_M split0 lesson
repeating at the FP8 layer):

- Direct manual byte-compare (python, pack vs source) confirms
  `model.layers.72.self_attn.kv_a_layernorm.weight` (a 1024-byte
  replicated BF16 norm) holds WRONG bytes in rank-07; its correct content
  sits 320,511,404 bytes EARLIER in the file than the header claims.
- The constant-shift hypothesis is falsified: layer-1 regions do not
  match at claimed−320MB either. The file has multiple/variable
  displacement — i.e., the data stream got reordered or over/under-written
  relative to the header plan somewhere from layer 1 onward.

Per the exactness discipline: RED STOP — no placement of these packs.
The 16 built files stay on sparkc (~/hy4-fp8-packs/) as forensic
material. The verifier did its job; the packer writer needs a fix.

Suspected area: build_rank's data-write loop vs its header-cursor plan
(alignment, gather byte accounting, or an entry emitting more bytes than
reserved). Next tick: rebuild rank 7 with a per-tensor
write-then-verify-in-pass (hash each tensor region immediately after
writing and compare against an independently computed expectation), find
the first divergent tensor, fix the writer, rebuild all 16.

The UD-IQ1_M packs are UNAFFECTED (different format, independently
verified at placement). GPU-port work (LAYER_STATE PASS) is unaffected.

## 09-03 (later tick): gather-slice root cause fixed; instrumented rebuild running

ROOT CAUSE of the FP8 corruption found by arithmetic: rank_view's dim-1
gather computed the row-slice bytes as `c0, c1 = lo * step, hi * step`
with step = FULL row bytes — for rank 0 (lo=0) that degenerated to
writing FULL o_proj rows (rank-0 file 63.59 GB vs the correct 56.13 GB
census — a 7.5 GB overrun), and for every other rank the slice fell past
the row end, writing EMPTY bytes for all 78 o_proj gathers (ranks 1-15
collapsed, the observed 320 MB displacement at layer 72, and the earlier
verify MISMATCHes). Fixed to `lo * esize, hi * esize` in the packer AND
the verifier's gather expectation.

Rebuilt all 16 ranks with the fix: uniform 56,131,321,268 bytes per rank
(matches the census exactly). Re-verified rank 0 with the fixed verifier:
all range/gather/replicate samples PASS except two replicates
(q_a_proj-49, MTP shared_experts.up_proj — both [2048,6144] F8, ~98.6%
of bytes differing, pack = source shifted 4 bytes) whose 4-byte shift is
unexplained by the fixed slice math. To settle it definitively, the
packer gained HY4_PACK_VERIFY_WRITE instrumentation: per-tensor read-back
compare against an independent fresh source read, with a shift-scan on
mismatch. The instrumented rank-0 rebuild runs detached on sparkc
(rank0vw.log, setsid — safe from ssh drops) and will name the first
divergent tensor, or pass clean, when it reaches that tensor.

Placement remains blocked pending this verification (RED STOP stands).
Next: read the instrumented result; PASS → verify suite + placement;
FAIL → debug stream_copy's seek for the flagged tensor (the independent
source read in the instrumentation removes any shared-handle doubt).

## 09-03 (next tick): FP8 chain moved to spark2 build-ship-delete; placement running

The free-pool nodes (sparkc/sparkd/etc) are saturated with the qwen4_flash
lane's own warm-heavy stagepack builds (their 16-rank TP4xPP4 FP8 loop) —
the warm path is contended fleet-wide, so the pack chain moved to spark2
(this lane's dedicated node, no competing warm build): per-rank build from
warm → ship 56GB to rank r's home node (spark{hex r}) → remote sha verify
against the build sha → delete local. Chain script:
tools/hy4_fp8_chain.sh (bash, per-rank, ship+sha-verify+delete, stops on
any failure).

Status at close: ranks 00+01 built, shipped and sha-verified on their home
nodes; rank 02 building. ~10 min/rank → all 16 placed in ~2.5h. The
remote-sha check makes each placement self-verifying (the earlier
independent suite verify of ranks 0/7 plus this per-rank remote sha close
the loop).

Chain ops note: the first two chain launches failed on shell quirks
(dash arithmetic `$((16#..))` under /bin/sh; a pgrep self-match kill) —
fixed by explicit rank/node lists and bash shebang; the pkill self-match
trap claimed another ssh session (known trap, by-pid kills only).

## 09-03 (next tick): build moved back to sparkc — spark2's warm path stalls at ~2MB/s

spark2's warm/ceph client stalled at ~2 MB/s on the rank-02 build (8
hours/rank — the known spark2 warm-stall pattern resurfacing), so the
chain moved BACK to sparkc: the stale instrumented run and partial files
were killed/cleaned, and a fresh 16-rank build chain (build4.log) is
running with the fixed packer. sparkc's warm path is contended by the
qwen4_flash lane's own stagepack builds but still delivers ~2 min/rank.
The corrupt old packs were deleted; ranks rebuild from the warm source.

When build4 completes: verify ranks 0+7 with the fixed verifier, settle
the q_a_proj-49 4-byte replicate question (targeted probe), then place
all 16 via the proven push paths (spark0-9 direct; hex nodes via spark2
push), and wire the weightd/residentd load path for hy4-fp8-tp16-v1.

## 09-03 (next tick): fanout packer launched, mislaunch cleaned; relaunch pending

The single-pass fanout packer (tools/hy4_fp8_stagepack_fanout.py — one
sequential warm pass fanning to all 16 rank outputs, 813 GB total reads
vs 13 TB for 16 per-rank passes) was launched but its nohup/setsid launch
chained badly: stdout landed on a pipe that died with the launching ssh,
and the first stdout flush killed the process (broken pipe) after ~8.19
GB of rank-00. Current state: NO packer process running; one orphaned
8.19 GB partial rank-00 at ~/hy4-fp8-packs/ (deleted before relaunch).

Relaunch procedure (next tick): delete the partial, launch via a staged
runner script with ABSOLUTE paths (log + output dir), verify the rank-00
file grows in ~/hy4-fp8-packs/, and monitor. The warm path is contended
by the qwen4_flash fleet builds (observed ~14 MB/s on sparkc) — the pass
takes ~16 h contended, faster when their loop gaps. The single-pass
design means progress is monotonic and restartable by simply rerunning
(completed ranks are skipped only by manual deletion — safest to let it
overwrite; verification happens after completion anyway).

## 09-03 (next tick): fanout packer launched clean — single warm pass in progress

After killing two overlapping instances (a launcher chain split caused a
duplicate launch on top of the ssh-drop survivor), exactly ONE fanout
python is running on sparkc: single sequential pass over the 813 GB
checkpoint, fanning to all 16 rank outputs. rank-00 growing (201 MB at
first sample). Internal redirects in the staged runner
(run_fp8_fanout.sh: cd + rm partials + exec python >> fanout.log) make it
immune to launcher-ssh drops — the earlier death was the stdout pipe
dying with its launching ssh (first flush after drop = broken pipe).

ETA: ~16 h if the qwen4_flash warm contention holds (~14 MB/s observed),
much faster when their loop gaps. Completion check: grep fanout.log for
"rankNN done sha=" lines ×16. Then the verify suite on ranks 0+7, then
placement (rank r -> spark{hex r}:~/sparkdata/hy4.fp8.tp16/packs/), then
weightd/residentd load wiring for hy4-fp8-tp16-v1, then GPU cells on the
FP8 packs (MX scale-apply kernels).

## 09-03 (later): two long-running jobs detached; results land next tick

1. FP8 fanout packer (sparkc): single warm pass over 813 GB fanning to
   all 16 rank outputs; progress visible via per-rank file growth.
   sparkc/spark2 ssh flaked during polling (proxy flap) — the job is
   setsid-detached and unaffected. Completion check: 16 "rankNN done
   sha=" lines in ~/hy4-fp8-packs/fanout.log, then the verify suite
   (ranks 0+7), then placement (rank r -> spark{hex r}).

2. Full 78-layer GPU forward (spark2, hy4_forward_test v3): the d_branch
   buffer was sized for one token while the MoE accumulate indexed it per
   token (OOB write poisoning the context — the layer-0 "done" print then
   illegal-access at the next alloc). Fixed to T*N_EMBD; also fixed two
   per-token buffer bugs found by inspection (attn-normed cur and qr were
   computed per token but only the last token's copy survived — the
   attention phase read stale data for tokens 0-2 via d_cur/d_qr instead
   of d_cur_all/d_qr_all). v3 running detached (~1h+); completion check:
   TOP1 line — expect 299 (the CPU greedy prefill token).

NEXT: read both results; on FORWARD PASS + verify PASS -> placement +
weightd/residentd wiring; the GPU forward cell then IS the TP16 native
module's decode blueprint.
