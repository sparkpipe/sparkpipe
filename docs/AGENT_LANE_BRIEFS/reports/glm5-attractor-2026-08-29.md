# glm5-attractor — the repeat-attractor root cause FOUND and FIXED (the routed MoE
# sum never reached the residual); KDA suspect (a) proven CLEAN by an independent
# checkpoint-semantics oracle; generation moved from `26517 x24` to word fragments

Lane `lane/glm5-attractor`, worktree /tmp/lane-glm5attractor, branch tip
`bec1bae` (pushed; two commits on d644c81). Fleet spark0..sparkf on the
FIXED driver sha `88b992791ba74661` (module artifact `b12ff541…`,
validator PASS 0 failures), fixed2 packs unchanged, api spark0:8433 UP
(reservations `lane-glm5attractor` on all 16, 90-min rolling). Every
claim below = command + raw output.

## How this serves the scoreboard

The residual defect after probe-fix was rank-invariant shared math. This
lane settled suspect (a) (KDA decay/norm-gate) CLEAN with a new
instrument, then found and fixed THE rank-invariant shared-math defect:
the MoE tail discarded the routed-expert sum and every MoE layer placed
exactly 16x the attention sublayer output as its "FFN" contribution.
Generation on the canonical cold fixture request went from
`[98218 x20, 93541, ...]` (attractor) to `way,ive      database:` —
word fragments, no attractor. NOT coherent yet: COMPSEC-17 / 92x / M5
remain blocked per the stop rule; the next localization is written below.

## 1. The instrument (commit 90b1d33)

- **G5N-VEC full-vector dumps** (`layer.cuh`, diag only, env
  `SPARK_GLM5_NEXT_PROBE_VEC`, rank 0, pass-capped
  `SPARK_GLM5_NEXT_PROBE_VEC_PASSES`, wave flag `--probe-vec`): hex rows
  of every layer-0 KDA stage buffer (collapsed/normed/fused_qkvb/latents/
  postconv/decay_logit/retention/write_gate/delta_out/gated/gate/
  out_partial + per-head f32 state) and the head input. Waves are
  SINGLE-ROW (verified: 4-token request = 7 waves; `G5N-DBG execute`
  carries pos), so pass p IS position p and the host can track the
  recurrence from zero.
- **`tools/glm5_next_kda_host_oracle.py`**: an independent host
  reimplementation of the KDA cell from the CHECKPOINT's semantics (fla
  KimiDeltaAttention reference math: `g = lower_bound *
  sigmoid(exp(A_log) * (f + dt_bias))` per (head, channel) — the fla
  `naive_kda_lowerbound_gate` form, NOT the softplus form; decay-before-
  predict delta rule with post-update read; q/k L2 per head; RMSNorm-
  then-sigmoid-gate). It consumes the dumps stage by stage, so any
  divergence convicts ONE kernel. The reference was fetched from fla
  main (fla-org/flash-linear-attention `fla/layers/kda.py`,
  `fla/ops/kda/{chunk,gate}.py`).

## 2. Suspect (a) KDA: CLEAN (receipt)

Cold probe-vec wave, fixture case 1 (recNu3MXkvWUzHZr9, 176 tokens) as
the first request, 199 waves dumped. Oracle verdict per stage
(`python3 /tmp/g5attr_oracle.py /tmp/g5attr_vec1.log /tmp/fixture_c1.json`):

    attn_norm   100.00% exact 0 ulp     (input_layernorm mapping CONFIRMED)
    fused_qkvb   99.94% exact max 1 ulp (q|k|v|beta GEMM + rank sections)
    q/k/v_postconv 100.00% exact        (conv taps, swish, window carry, L2)
    decay_logit 100.00% exact           (f_b_proj rows[0:512] slice RIGHT)
    retention   max_rel 7.1e-07         (__expf vs libm last-ulps only)
    write_gate  100.00% exact           (b section sigmoid)
    delta_out   100.00% exact           (the recurrence, per pass)
    state_h0-3  max_rel 2.2e-07         (fp32 accumulation-order noise)
    out_partial 100.00% exact           (o_proj col-slice GEMM)
    head argmax matches emitted token (modulo <1e-2 logit ties)

The KDA cell — decay mapping, dt_bias/A_log per-rank indexing, norm-gate
order, delta rule, state — is exactly the checkpoint's math. Suspect (a)
is CLOSED. (First oracle run showed q/k/decay divergences; all three were
oracle bugs — post-L2 dumps, a double bf16 cast, GEMM-fed sigmoid — fixed
in the oracle, then everything matched. The module was right each time.)

## 3. ROOT CAUSE: the routed-MoE sum never reached the residual (commit bec1bae)

With (a) clean, the MoE family was next. READING the MoE tail against its
glm52 donor found the deviation immediately:

    glm52 donor : finalize -> hidden ; AddRows(hidden, shared_out, hidden)
    glm5_next   : finalize -> hidden ; AddRows(attention_out, shared_out, hidden)

`hidden_bf16` IS the HC streams surface (BindLayer). So the routed sum
was clobbered by the add; `attention_out` still held the ALREADY-REDUCED
attention sublayer output; and REDUCE_MLP then allreduced sixteen
IDENTICAL copies of it.

RECEIPT (diag driver, cold wave, rank 0, the probe pair per layer):

    L42 first r0 post -0.108028 -0.410636 -0.227772 -0.224844
    L42 second r0 post -1.72851 -6.57042 -3.64447 -3.59763  = 16.000000x
    L44 first r0 post -0.0958237 0.299302 0.520482 0.153072
    L44 second r0 post -1.53324 4.78907 8.3282 2.44927     = 16.000000x

Every MoE layer (42 of 45) placed exactly 16x the attention output as
its FFN contribution; the FFN knowledge was absent from the model.
Properties that kept it hidden: rank-invariant (all ranks identical);
TP1-invisible (the reduce no-ops at tp_degree==1); the M3 tier walk
mirrors the chain and tier 1 covers only the dense layer 0; byte-level
pack audits don't see dataflow. Exactly the class the brief predicted.

FIX: the finalize lands the routed sum directly in `attention_out_bf16`
and the shared rank partial adds IN PLACE
(`attention_out = routed + shared`); the streams surface is no longer
touched by the MoE tail. Dense tail (layers 0-2) was already correct.

AFTER (cold wave 2, served:0, same fixture request):

    BEFORE: {"tokens":[98218,98218,...x20,93541,93541,98218,93541]}   ("alianalian…")
    AFTER:  {"tokens":[3117,11,533,220,220,220,220,4625,25,...]}
            = "way,ive      database:"          — words, punctuation, NO attractor
    r0 post ratio: GONE (L42 first (0.729,-0.204,0.921,0.604) vs
    second (-0.442,-0.030,0.078,0.122) — MLP output now flows)
    prefill head trajectory: 'sic' '（）' '26' ' called' ' the' ' in' '4' ' the'

## 4. What remains (evidence, not chase)

Still degenerate ("way,ive      database:", case 2 gives digits+spaces).
Next suspects, ordered:

1. **DSA core attention at real positions** (suspect (c), the last
   unverified compute family). NOTE found while reading:
   `Glm5NextLayerIndexer` skips indexer rope entirely ("NO rope on either
   half - NoPE model, and indexer_rope_interleave is not a porting
   dependency") while config carries `indexer_rope_interleave: true`.
   At context <= 2048 the pool selection is BYPASSED (early return), so
   the indexer is out of the loop for fixture-length prompts and the
   core nope-absorbed MLA + latent-cache read is the exposed surface.
   The instrument that worked here (donor diff) applies: dsv4's MLA/
   indexer vs this port, then G5N-VEC at the DSA site + oracle.
2. **mHC semantics** (comb/sinkhorn/post/pre and the UNWEIGHTED head
   mean) — no independent reference pulled yet; the L44 comb collapses
   to ~one-hot on stream 0 (0.9997/0.0009/0.0004/0.0038) — healthy or
   not is unverified.
3. **swiglu_limit 10.0 is NOT wired** (`LmSiluMulKernel` clamps nothing;
   `GLM5_NEXT_SWIGLU_LIMIT` defined-unused; dsv4/qwen38max clamp).
   Measured INERT on fixture traffic at layer 0 (max gate 2.59, max
   |up| 2.27 across 8x12288 channels): a latent deviation, not the
   residual defect. Fix opportunistically with the validator's recorded
   deviation.
4. **Cross-request state bleed** (closeout item 2) — still open: second
   request on a slot differs from the first. Canonical readings require
   a fresh wave (this report's receipts are all served:0 firsts).

## 5. Artifacts / fleet state

- Branch `lane/glm5-attractor` tips: `90b1d33` (instrument), `bec1bae`
  (fix). Ratchet 226696 -> 227169 -> 227209, exact + justified in-commit.
  Offline gates PASS (build-all, run-tests, package-manifest) at both.
- Driver `88b992791ba74661` deployed 16/16 (built on spark0 from the
  branch; module artifact `b12ff541e7301df0…`; validator PASS 0
  failures, tier1/probe0 + tier2a attention site + determinism).
  Diag (probe+vec) by env; the same binary serves clean.
- Fleet UP: 16/16 ready on fixed2 packs + fixed driver, api
  spark0:8433 healthy. Wave logs: /tmp/g5attr-wave{1,2}.log; vec dump
  log preserved at spark0:/tmp/g5attr_vec1.log (160 MB, 199 passes);
  oracle at tools/glm5_next_kda_host_oracle.py (also
  spark0:/tmp/g5attr_oracle.py).
- Canonical post-fix receipt (cold, served:0, fixture case 1,
  max_tokens 24): tokens [3117,11,533,220,220,220,220,4625,25,
  220,...] — archived here.
- Reservations `lane-glm5attractor` all 16, 90-min rolling (renewed
  18:11Z).

## Integration request

1. Merge `lane/glm5-attractor` (bec1bae): the MoE-tail fix is the
   cold-first-request root cause #2 — REQUIRED for any coherence.
2. The instrument (G5N-VEC + oracle + --probe-vec) is the template for
   the DSA-site localization; keep until coherence, then trim + ratchet.
3. After any repack: header provenance patch (unchanged from probe-fix).

## Next lane's exact steps

1. Donor-diff `Glm5NextLayerAttention` (nope-absorbed scoring, latent
   cache, qk_scale, v path) against dsv4 + glm52; then extend G5N-VEC
   to the DSA site at L3 (normed/q_compressed/qkv/latent cache row/
   scores/selected/attn partial) and run the same dump-fed oracle.
2. Pull the GLM 5.3 mHC reference (or dsv4's validator oracle) and
   arbitrate the comb/sinkhorn/head-mean semantics.
3. Wire swiglu_limit (fixtures keep |gate|,|up| << 10 — validator-safe).
4. State reset on slot re-acquire (closeout item 2) before any served
   quality number: only served:0 firsts are canonical today.
5. When coherent: COMPSEC-17 -> 92x -> M5 (harnesses staged since
   closeout: tools/glm5_next_compsec17.py, tools/glm5_next_m5_batch.py).
