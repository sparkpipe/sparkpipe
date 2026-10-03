# GLM-5.3 Flash quant A/B — W1 report

Campaign `flash-w1`, plan_sha256 `495032b0fb9a534146bbd5d2865cfb1031c6a949731b6ed2c8af55c3336d29c5`
(`qualification/ab/flash-w1/PLAN.json`, frozen by PR #1369). Window stamp
`w1-20260929T2200Z`, run 2026-09-29T22:00:31Z-23:46:12Z, firmware commit
`8058e4b3357442f2c1b2095ab5289d41a1054e9a`. All window steps (pre, open, A, B,
C, close) PASS; production's root files and greedy tokens were identical to
the pre-window baseline at close.

Source data: `/Users/mac/wf/quant-w1-prep/results/w1/w1-20260929T2200Z/`
(receipts, `aa.txt`, `window.log`) and `sparkf:~/build-quant-w1-prep/w1/w1-20260929T2200Z/`
(merged dumps and exports). Analysis ran on sparkf, CPU only, each step inside
`systemd-run --user --scope -p MemoryMax=8G`. No GPU jobs, no fleet writes. No
large files were copied off the fleet; this report and the small (~80-90 KB)
comparison JSONs it is built from stay on sparkf under
`~/build-quant-w1-prep/w1/w1-20260929T2200Z/analysis/`.

No performance numbers are in this report. W1 is an accuracy window only; no
roofline or tok/s figures were measured, and none are invented here
(`lanes/ROOFLINE_REPORTING.md`).

## Tool-chain finding: `ab_verdict.py`/`ab_report.py` cannot run on the W1 fleet receipts as shipped

`tools/ab_score_compare.py` ran cleanly on every W1 arm (below). Getting a
verdict and a rendered report hit a real gap between the frozen analysis
tools and what the W1 window actually writes:

- `tools/ab_verdict.py --aa AA.json ...` and `tools/ab_report.py` both consume
  receipts in the `sparkpipe-ab-receipt-v1` schema
  (`qualification/ab/receipt.schema.json`: top-level `plan_sha256`,
  `arm_digest`, `dumps.merged_sha256`, `requests.cached_prompt_tokens`,
  `topology`, `execution`, `cache`, ...).
- The W1 window's lane script (`/Users/mac/wf/quant-w1-prep/scripts/w1_lib.sh`,
  untracked) writes `fleet_receipt.json` in a different, ops-focused
  `sparkpipe-ab-w1-fleet-receipt-v1` schema (`format`, `window`, `slot`, `arm`,
  `run`, `utc_start/end`, `client`, `nodes`, ...). No tool in the repo or in the
  lane's `scripts/` converts one into the other, and `ab_receipt.py validate`
  would refuse a `fleet_receipt.json` outright (it is missing every field
  `sparkpipe-ab-receipt-v1` requires at the top level).
- I hand-assembled an `AA.json` (`sparkpipe-ab-aa-v1`) directly from verified
  evidence instead of fabricating one: `aa.txt`'s recorded
  `merged_sha256`/`generated_token_ids_sha256` for the F1/F1AA pair, cross-checked
  byte-for-byte against `F1-f1-ref/export.sha256` and `F1AA-f1-aa/export.sha256`
  on sparkf (both `e4c21df3efb3048403a68f222799b87356b3c5bd12a6a3e6f1e3d188fe6bb35f`).
- With that `AA.json`, the natural single invocation for all three W1 arms
  fails:

  ```
  $ python3 ab_verdict.py --plan PLAN.json --aa AA.json \
      --comparison flash.S1.e-fp8.k-bf16/bf16/fp32.d-none=cmp-F2-vs-F1.json \
      --comparison flash.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-none=cmp-F3-vs-F1.json \
      --comparison flash.S0.e-fp8.k-bf16/bf16/fp32.d-none=cmp-F0-vs-F2.json \
      --out VERDICTS-all.json
  ab_verdict: REFUSED: flash.S0.e-fp8.k-bf16/bf16/fp32.d-none: its reference dump is not one of the A/A pair
  ```

  Dropping F0 instead fails the complementary way, because `decide()` requires
  every non-reference, non-D plan arm to have a comparison in the same call:

  ```
  $ python3 ab_verdict.py --plan PLAN.json --aa AA.json \
      --comparison flash.S1.e-fp8.k-bf16/bf16/fp32.d-none=cmp-F2-vs-F1.json \
      --comparison flash.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-none=cmp-F3-vs-F1.json \
      --out VERDICTS-E.json
  ab_verdict: REFUSED: no comparison for plan arm flash.S0.e-fp8.k-bf16/bf16/fp32.d-none
  ```

  The root cause: F0's plan `compare_to` is F2 (the spine-isolating bridge),
  so its comparison's reference dump is `F2-f2-probe/bridge-reference.npz`,
  not F1's. `--aa` takes exactly one record and `decide()` insists on every
  plan arm at once, so no single `ab_verdict.py` call can cover a PLAN whose
  arms are anchored to two different reference dumps. `ab_report.py` has the
  same dependency on `sparkpipe-ab-receipt-v1` receipts (`ab_receipt.compare()`,
  `arm["dumps"]["merged_sha256"]`, `arm["topology"]`, ...), so it inherits the
  same blocker and was not run against the real receipts either.
- This looks like a genuine tooling/integration gap (a missing
  `fleet_receipt.json` -> `sparkpipe-ab-receipt-v1` converter, or a
  `ab_verdict.py` CLI limited to one reference-dump family per invocation) and
  not a small, obviously-correct one-line bug, so I did not patch
  `ab_verdict.py`/`ab_report.py`/`ab_receipt.py` blind. The lane/tooling owner
  should pick between a receipt converter and loosening the CLI to accept
  multiple `--aa` records scoped by reference-dump identity.
- To still get the frozen statistics onto the real data, I imported
  `ab_verdict`'s own `metric_specs`/`evaluate`/`binding` and `ab_stats.holm_levels`
  directly and ran them against the real `ab_score_compare.py` output (same
  code, no CLI gate). That is what the verdicts below are computed with.
- `tools/ab_suite_compare.py` also could not run: it reads a graded
  `<run>/responses/*.json` + `INTEGRITY.json` archive
  (`qualification/ds4_eval/compare_runs.py` grading), and no such archive
  exists anywhere under `~/sparkdata/qab-f*/runs/` on sparkf. This matches the
  frozen plan's own notes, not a bug: the lead-accepted COMPSEC fallback for
  W1 (`W1_COMPSEC_BATCHES` = L1's two T1 prompts, `W1_COMPSEC_GRADE=none`) never
  produced a graded archive, only a liveness/output-neutrality check (below).

## A/A and determinism gates (the noise floor)

- **F1 vs F1AA (the campaign A/A gate, permuted order, concurrent engines):**
  BIT-IDENTICAL. `merged_sha256` and `generated_token_ids_sha256` equal
  (`e4c21df3efb3048403a68f222799b87356b3c5bd12a6a3e6f1e3d188fe6bb35f` /
  `9860b309a3ca0eab0ac54bc84f0f675f6a73c9fdbeea96327f1553708b6f7504`). This is
  an exact 0 KL_b / 0 flips by construction (identical dumps), not a sampled
  noise floor.
- **F2 first run vs F2 repeat (fresh engine, concurrent with F3):**
  BIT-IDENTICAL (`merged_sha256 9f63b0c5e1a2a64a988e4824cddf3deeccdf1276920d63f5b3a3cda9e95e9db7`,
  tokens `8e94438eab5c5ddce70fcdbcef86db63ef0290931a5203a6a63622edb16efbd9`).
  Confirms the E-axis anchor's measurement is itself reproducible.
- **F2 COMPSEC dump on vs off:** served `tokens_sha256` identical
  (`8e6570943295a1e22050aa68dabc57703b318551fe85c45dd3bd672d82d7d1ed` on both
  `F2-f2-compsec-dump` and `F2-f2-compsec-nodump`), confirming the score dump
  is output-neutral. `events_sha256` differs, as expected (dump-mode events
  carry extra dump bookkeeping).

## Per-arm results

CT-short (200 docs, 1,280 tokens each, 255,800-256,000 scored rows;
`bde6ca4c...`). Position bins only populate 0-1024 and 1024-4096 (CT-short
documents are shorter than the next bin edge; CT-long is not scored in W1).
Exact KL is the Tier-2 sample (20,000 rows, seed 20260929); "bucket <= exact"
calibration had 0 violations on all four comparisons below (bucket KL_b runs
~8-13% under the exact full-vocabulary KL, as expected for a lower bound).
Bootstrap CIs are 10,000 replicates, by document, seed 20260929 (percentile
and BCa, more conservative bound kept), per the frozen plan.

### F1 - reference

`flash.S1.e-bf16.k-bf16/bf16/fp32.d-none`, arm_digest
`213c0afe9f57c0e3244d5733955a5018606e9ba29a5c3931fb955cb47d1a23b4`, publisher
BF16 spine (S1) with publisher BF16 experts. Role: reference; not scored
against itself beyond the A/A gate above.

### F2 - anchor (publisher FP8 experts, S1 spine)

`flash.S1.e-fp8.k-bf16/bf16/fp32.d-none`, arm_digest
`95d95cb3ec8c7184a014a68960c91663d942942938fdb5e4b75413c6b677fc69`.

- KL_b(F1,F2) mean 0.008672 [0.008096, 0.009338] (95% CI); exact-subset KL
  0.008922 (bucket 0.008209 on the same rows, 0/20000 calibration violations).
- Top-1 agreement 96.766% [96.644, 96.885]; near-ties excluded 91/255800;
  tail mass p99 0.258.
- dNLL relative to reference NLL: -0.0077% [-0.049, +0.035] (pt of NLL, not
  +/-0.5 pp - well inside any plausible margin).
- Per content class: code KL_b 0.01424 (n=40, top1 96.98%); prose-old 0.00652
  (n=40, top1 97.11%); prose-recent 0.00753 (n=120, top1 96.58%).
- Position bins: 0-1024 KL_b 0.00946; 1024-4096 KL_b 0.00554.
- Verdict: **ANCHOR (backstops: calibration)**. The design's absolute
  backstops (mean KL <= 0.005, top-1 >= 99.0%) are reported, not decisive, until
  the plan is marked `calibrated` against this exact F2-vs-F1 comparison
  (design section 10.16). Worth flagging to the owner: F2 itself - the intended
  calibration pair - misses both proposed absolute numbers (0.0087 > 0.005,
  96.77% < 99.0%), which is why calibration should not be rubber-stamped at
  the current proposed values.
- COMPSEC-17: not graded in W1 (fallback); dump on/off served tokens
  identical (above).

### F3 - E arm, NVIDIA NVFP4 experts, S1 spine

`flash.S1.e-nvfp4nv.k-bf16/bf16/fp32.d-none`, arm_digest
`0f44aefa7c7ad46eacea8b8a361953e96218ceeb53767983f4b457378a065581`, anchor F2.

- KL_b(F1,F3) mean 0.035212 (255800 rows); exact-subset KL
  0.035606 (bucket 0.033158 on the same rows, 0/20000 calibration violations)
  - about 4x the anchor's KL on the identical rows.
- Top-1 agreement 93.249% (near-ties excluded 91/255800).
- Per content class: code KL_b 0.03922 (top1 94.83%); prose-old 0.03397
  (top1 93.21%); prose-recent 0.03429 (top1 92.73%) - uniformly ~4x F2's
  per-class KL, no stratum escapes it.
- Position bins: 0-1024 KL_b 0.03842; 1024-4096 KL_b 0.02240.
- dNLL relative to reference NLL: +0.4505% (close to, but inside, the 0.5%
  backstop that is not yet decisive).
- **Margin test (design section 3.5 / PLAN margins, Holm family size 1 since F3 is
  the only statistical "arm" on axis E in flash-w1, so the Holm level is the
  plain one-sided 95% bound):**
  - `R = KL(F1,F3) / KL(F1,F2)`: estimate 4.061, CI [3.837, 4.291], margin
    upper <= 1.1 -> **outside** (bootstrap p ~ 1.0, not a borderline call).
  - top-1 agreement minus the anchor's: estimate -3.517 pt, CI [-3.663,
    -3.375], margin lower >= -0.3 pt -> **outside** (p ~ 1.0).
  - **Verdict: INFERIOR.** Both primary E-axis metrics fail the frozen margin
    by a wide, well-resolved margin (~3.5-4x beyond the CI bound, not a
    borderline call that more data would flip).
- COMPSEC-17: not graded in W1 (fallback only; F3's fallback run completed
  2/2 requests, no suite signal).

### F0 - spine bridge (production's S0 pack, FP8 experts)

`flash.S0.e-fp8.k-bf16/bf16/fp32.d-none`, arm_digest
`5a2149d328715e784b59b4912a8768920b7963e4915ebc92188036f42b1dacf7`, PLAN
`compare_to` = F2 (isolates the S0-vs-S1 spine difference at fixed FP8
experts).

- vs F2 (the declared spine-bridge comparison): KL_b mean 0.008870 [0.008293,
  0.009539]; exact-subset KL 0.009148 (bucket 0.008425, 0/20000 violations).
  Top-1 agreement 96.683% [96.556, 96.803]. dNLL relative -0.0027%.
  - Per content class: code 0.01404 (top1 96.92%); prose-old 0.00670 (top1
    97.01%); prose-recent 0.00787 (top1 96.49%) - closely tracks F2's own
    per-class numbers (previous section), i.e. the S0 spine adds only a small
    increment on top of the FP8-expert distance, no stratum-specific
    surprise.
  - Position bins: 0-1024 KL_b 0.00965; 1024-4096 KL_b 0.00575.
  - **Verdict: BRIDGE (backstops: calibration)** - descriptive only, no
    Holm decision, per design. Same absolute-backstop caveat as F2 applies
    (0.0089 KL, 96.68% top-1, both over the proposed but not-yet-decisive
    absolute numbers).
- vs F1 (extra descriptive point, not a PLAN comparison - mixes the spine and
  expert axes together): KL_b mean 0.009012, exact-subset KL 0.009238 (bucket
  0.008490), top-1 96.681%, dNLL -0.0142%. This matches the two independent
  `klpartial` results already recorded live in `window.log`
  (`kl_mean 0.00923836...` vs F1, `0.00914794...` vs F2), which the recomputed
  exact-subset means above reproduce exactly.
- Production's release check at window close confirmed root files and greedy
  smoke tokens identical to the pre-window baseline, so F0's run did not
  perturb what's actually serving.

## Verdict summary

| Arm | Codec (vs S1/BF16 reference) | Role | KL_b (mean) | Top-1 | R vs anchor | Verdict |
|---|---|---|---|---|---|---|
| F1 | BF16 experts (reference) | reference | - | - | - | - |
| F2 | FP8 experts | anchor | 0.008672 | 96.766% | 1.00 | ANCHOR (backstops: calibration) |
| F3 | NVIDIA NVFP4 experts | arm | 0.035212 | 93.249% | 4.06 [3.84, 4.29] | **INFERIOR** |
| F0 | FP8 experts, production S0 spine | bridge (vs F2) | 0.008870 | 96.683% | n/a | BRIDGE (backstops: calibration) |

## Production-eligible arm

**F2 (publisher FP8 experts) remains the only W1-qualified codec** - and it is
already what production effectively runs today (F0, production's actual S0
pack with FP8 experts, tracks F2's numbers closely per the spine bridge
above). W1 does not surface a new arm to promote.

**F3 (NVIDIA NVFP4 experts) is not production-eligible under the frozen W1
margins.** The failure is decisive, not marginal: ~4x the anchor's KL and a
top-1 gap of -3.5 pt against a -0.3 pt floor, uniform across every content
stratum and both scored position bins. This is consistent with the design
doc's own caveat (`lanes/quant-ab-design.md` section 4.2 item 5): NVIDIA calibrated
this checkpoint for W4A4 while SparkPipe serves W4A16, so the publisher's
accuracy claims do not transfer as-is. A different NVFP4 recalibration or one
of the community EXL3 arms (W5, separate campaign `flash-w5dq`) would need
its own W1-equivalent window before any 4-bit expert codec could be
considered.

## Caveats

- Absolute backstops (mean KL <= 0.005, top-1 >= 99.0%, |dNLL| <= 0.5%) are
  still `status: calibration`, not decisive for any arm here; see the F2 note
  above about the proposed numbers looking too tight once real calibration
  data exists.
- COMPSEC-17 carries no suite signal for W1 by design (lead-accepted
  fallback); breakage/regression detection returns in W2.
- CT-long (position bins beyond 4096, kv-I8 behavior, retrieval strata) and
  the on-policy corpus are out of scope for W1.
- No performance/roofline numbers are in this report; W1 is accuracy-only.
- `ab_verdict.py`/`ab_report.py` could not be run end-to-end on the real
  fleet receipts (see the tool-chain finding above); the verdicts above come
  from the same frozen statistics functions invoked directly, not from the
  documented CLI, pending a receipt-schema fix that I judged too large a
  design call to make unreviewed.
- The interaction check (design section 3.5, one 2x2 factorial) was not run: W1 has
  only one K codec (bf16) in play, so there is no second K point to pair with
  the two E arms.
