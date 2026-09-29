# Quantization A/B: arm identity, corpora and statistics

This directory holds the pre-registration material for the quantization A/B campaigns: the plan template, the receipt schema and the frozen corpus indexes. The design is `lanes/quant-ab-design.md` in the coordination repository. Its §2 covers arms, §3 metrics and verdicts, §5 the harness, and §10 the critic's corrections, which override the earlier sections.

Every arm differs from its reference on exactly one axis:

- **E**: the routed-expert codec.
- **K**: the KV codecs (latent, index and state).
- **D**: the drafter.

The spine is byte-identical between the two arms of a comparison. The default frame is S1, the publisher BF16 release. F0 (production's S0 spine) is the declared `axis: spine` bridge.

## Files

| File | What it is |
|---|---|
| `PLAN.template.json` | Starting point for a campaign `PLAN.json`. `tools/ab_plan.py freeze` validates it, stamps `plan_sha256` and writes the frozen plan. A frozen plan is never edited: a change is a new campaign. |
| `receipt.schema.json` | `sparkpipe-ab-receipt-v1`, one receipt per arm run (design §2.6). |
| `corpora/CT-short.index.json`, `corpora/CT-long.index.json` | Frozen corpus indexes (`sparkpipe-ab-corpus-v1`). The token files are not committed. |

## Arm descriptor and `arm_digest`

`arm.json` has the format `sparkpipe-quant-arm-v1`. Two implementations parse it and apply the same rules, and `tests/test_ab_arm.py` cross-checks them:

- C: `include/sparkpipe/spark_quant_arm.h` and `src/spark_quant_arm.c`.
- Python: `tools/ab_arm.py`.

It has exactly these members (all required):

```
format "sparkpipe-quant-arm-v1"
arm_id   <model>.<frame>.e-<expert label>.k-<latent>/<index>/<state>.d-<drafter label>
model    lowercase token, e.g. "flash"
revision the MODEL_REVISION the build uses
topology {tp, pp, kv_shard}
spine    {frame "S<n>", source "repo@rev", spine_digest[tp*pp]}
expert   {codec bf16|fp8|nvfp4|mxfp4|int8|int7|int6, label (starts with codec),
          producer publisher|community|experiment, source "repo@rev", recipe_sha256|null}
kv       {latent bf16|fp8|mxfp4, index bf16|fp8, state fp32|bf16, group 0|32|64|128, mode sim|store}
drafter  {kind none|lookup|mtp|dflash|oracle|adversary|random, label (starts with kind),
          codec, head, sidecar_sha256[]}
pack_sha256[tp*pp]
artifacts {module_archive_sha256, driver_sha256, adapter_sha256}
```

In `arm_id`, a quantized KV codec carries its group size, e.g. `k-fp8g128/bf16/fp32`.

**Refusals.** The parser refuses:

- unknown, missing or duplicate members;
- a non-canonical integer or any float;
- uppercase hex;
- a per-rank array of the wrong length;
- an `arm_id` that does not match the fields;
- a KV group that does not fit the codecs (all-bf16 needs group 0 and mode `store`; `mxfp4` needs group 32);
- an `experiment` expert without its `recipe_sha256`;
- a weightless drafter that carries a codec, head or sidecar, or a weighted drafter (`mtp`, `dflash`) without them.

**Digests.**

- `arm_digest` is the SHA-256 of the canonical JSON: keys sorted, no whitespace, ASCII only.
- `pack_set_sha256` is the SHA-256 of the canonical `pack_sha256` array.

**Where the digest travels.** `sparkpipe_model_batch` and `sparkpipe_model_api` accept an optional `--quant-arm ARM_JSON`. Without the flag their output is byte-identical to before. With it:

- the batch `ready` event adds `arm_id`, `arm_digest`, `arm_kv` (`latent/index/state/group/mode`), `pack_set_sha256` and the per-rank `pack_sha256`;
- every API `request_measurements` line adds `arm_id`, `arm_digest`, `arm_kv` and `pack_set_sha256`;
- an arm file that fails to parse stops the process before it connects.

`build/sparkpipe_quant_arm --digest|--canonical|--summary ARM_JSON` prints the same values for scripts on the nodes.

## Merged score dump (`sparkpipe-score-merged-v1`)

The analysis reads one merged `.npz` per (arm, corpus run). It is written by `tools/score_merge.py` (lane L1) and read by `tools/ab_dump.py`. Rows are in corpus order (document, then position), whatever order the documents were submitted in. That makes the permuted A/A run byte-comparable. The writer uses fixed zip timestamps, so the same content gives the same bytes.

| Array | Type | Meaning |
|---|---|---|
| `header` | 0-d unicode | canonical JSON: `format`, `arm_digest`, `corpus_sha256`, `tokenizer_sha256`, `probe_sha256` (null for the run that defines the probes), `probe_k`, `rows` |
| `doc`, `pos`, `target` | uint32[N] | corpus document, position of the predicted token, corpus token at that position |
| `lse` | float64[N] | merged log-sum-exp over the whole vocabulary |
| `target_logit` | float32[N] | logit of `target` |
| `top_ids`, `top_logits` | uint32/float32[N,K] | this run's global top-K; logit descending, lowest id first on ties |
| `probe_logits` | float32[N,K] | logits at the probe ids, which are the reference run's `top_ids`. For the reference run itself this equals `top_logits`. |
| `select_hash` (optional) | uint64[N,L] | hash of the sparse-attention selection per indexed layer (K arms) |

**Exact KL.** Exact full-vocabulary KL on the Tier-2 rows comes as `sparkpipe-score-exact-kl-v1` from `tools/score_kl_partial.py`:

- `header`: `reference_arm_digest`, `arm_digest`, `corpus_sha256`;
- `rows` uint32[M];
- `kl` float64[M].

## Metrics and verdicts

`tools/ab_score_compare.py` produces per-document tables. `tools/ab_verdict.py` decides.

**Metrics.**

- **KL_b(P‖Q).** The bucket partition is the reference's top-K ids plus one tail bucket. It is a lower bound on the exact KL (data-processing inequality), and it is exactly 0 for identical dumps.
- **Tail rule.** If the probe probabilities sum above 1 by more than 1e-6, the dump is refused as inconsistent. If they sum to between 1 and 1+1e-6, the tail is exactly 0 and the row is counted in `tail_zero_rows`.
- **Top-1 agreement.** Rows where the reference's top-2 gap is under `near_tie_nats` are excluded and counted separately. A decisive flip is a flip where the reference margin is above `decisive_nats`.
- **dNLL.** `NLL_arm − NLL_ref` of the corpus token (positive means the arm is worse). It is reported relative to the reference NLL as a ratio of document means.

**Unit and bootstrap.**

- The unit is the per-document mean.
- The bootstrap resamples documents, with indices from a SplitMix64 stream keyed by the plan seed, so every host gives the same interval.
- Each bound is the more conservative of the percentile and BCa bounds.

**Holm.**

- Each primary metric gets a bootstrap p-value against its margin. The arm's p-value is the largest of them (intersection-union).
- Holm step-down runs across the statistical arms of one axis. The j-th arm in p order uses the one-sided `1 − α/(m−j+1)` bound.
- Once an arm is not equivalent, no later arm can be declared equivalent.

**Verdicts.**

| Verdict | Condition |
|---|---|
| **EQUIVALENT** | Every primary bound is inside its margin at the arm's Holm level, and no suite has a McNemar regression with p < 0.05. |
| **INFERIOR** | A bound on the favourable side is already outside its margin, or a suite shows a regression or breakage. |
| **INCONCLUSIVE** | Anything else. Grow the corpus; never re-run a subset. |

**Margins per axis** (lead decision O3: the design's §3.5 defaults with the critic's corrections):

- **E arms:**
  - ratio `R = mean KL(ref‖arm) / mean KL(ref‖anchor)`, upper bound ≤ 1.1;
  - top-1 agreement minus the anchor's, lower bound ≥ −0.3 pt. The critic widened this from 0.2 pt (§10.14). A plan may keep 0.2 pt only with `ct_short_doubled`.
- **Absolute backstops** (mean KL ≤ 0.005, top-1 ≥ 99.0 %, |dNLL| ≤ 0.5 %):
  - While the plan says `backstops.status: calibration`, they are reported and never decide.
  - They start deciding only once the plan is `calibrated` and cites the SHA-256 of the anchor-versus-reference comparison (F2 vs F1) they were calibrated on (critic §10.16). The dry run shows why: a synthetic anchor fails the 99 % top-1 backstop.
- **K arms:** the absolute margins, applied in every position bin.
- **D arms:** zero token differences over the exactness set.
- **Anchor and bridge arms:** shown as `ANCHOR` and `BRIDGE` with their backstop numbers.

**Suites** (`tools/ab_suite_compare.py`) are breakage detectors only:

- They regrade every case with the ds4-eval rule and refuse any archive whose `INTEGRITY.json` does not match.
- They report the exact McNemar p, the Newcombe CI, the transitions, and the greedy divergence index with its Kaplan-Meier median.
- On the retained archives in `qualification/ds4_eval/runs/`:
  - every run passes 14/17, with no transitions;
  - dd3526b and 09fdad6 (both sequential) are identical on 12/17;
  - the 09fdad6 sequential repeat is identical on 17/17;
  - the three concurrent runs are identical to sequential on 11, 12 and 11/17.

## Receipts and the comparability contract

`tools/ab_receipt.py` validates a receipt against the schema and against itself:

- the arm digest;
- the ready-event digest and pack set;
- pack SHAs per rank;
- topology;
- `cached_prompt_tokens == 0` for every request;
- the KV snapshot directory strictly inside the arm's own root. `SparkKvSnapshotPrune` deletes other layouts' files, so a shared directory would delete another root's snapshots.

`compare` refuses the comparison if any of these fail:

- **Axes:** the arms must differ on exactly the declared axis.
- **Spine:** outside a declared spine comparison, every per-rank spine digest must be equal.
- **C1, build:** the source commit must be the same and on the plan's pinned commit. Build flags may differ only in the axis's own flags (`EXPERT_CODEC`, `KV_QUANT_SIM`, `DRAFT_EXPERT_CODECS`, or `MODEL_REVISION`/`CONTRACT_SHA256` for a spine comparison).
- **Packs:** K and D arms must use their base arm's pack bytes.
- **C2, topology:** tp, pp, kv_shard and the rank→node map must match.
- **C3, wave:** the sequential flag, prefill rows per submission and output budget must match.
- **C4, cache:** each run needs its own snapshot directory.
- **C5, execution:** graph/eager mode, drop-in hash and weightd residency must match.
- **C6, inputs:** corpus, tokenizer, probe file and plan SHA must match.

**Other subcommands.**

- `aa` is the A/A gate: merged dumps and generated token ids must be bit-identical, or the campaign stops.
- `dumpcheck` proves that served tokens are identical with the score dump on and off.
- Expert GEMM-path waves are flagged in the report as "prefill-path" (design §4.2 item 5).

## Corpora

`tools/ab_corpus_build.py` tokenizes with the pinned GLM-5.3 Flash `tokenizer.json` (sha `19e77364…`, see `qualification/ds4_eval/tokenizer/README.md`) through `build/sparkpipe_tokenize_prompt`. It cuts every document to an exact length and accepts documents in a seeded order.

**Block uniqueness.**

- A candidate that shares any aligned 64-token block with an already accepted document is skipped and listed in `skipped`.
- The finished corpus is re-checked, and refused if any block repeats across documents. Otherwise prefix reuse could serve cached KV and break C4.
- `tests/test_ab_corpus.py` covers a colliding pair and a shared framing prefix.
- `ab_corpus_build.py check INDEX` re-verifies a corpus against its token file.

**CT-short** (200 documents of 1,280 tokens, 256,000 tokens, `corpus_sha256 bde6ca4c…`):

| Stratum | Documents | Source | Contamination |
|---|---|---|---|
| `prose-recent` | 120 | arXiv 2607.x (July 2026, CC-BY/BY-SA/CC0), LaTeX after `\begin{document}` | These are the newest papers in `secemp9/arxiv-complete@cee89483`. GLM-5.3 was published around 2026-08, so whether they predate its training cutoff is unverified. |
| `prose-old` | 40 | arXiv astro-ph/0101 (January 2001) | Almost certainly in pre-training data. The contrast with `prose-recent` shows the contamination effect: memorized text understates KL and flips (design §3.4). |
| `code` | 40 | SparkPipe C/CUDA sources at `266b6adbc`, not public | Leading include and pragma lines are stripped. |

Report the strata separately.

**On-policy stratum.** It is a separate corpus (CT-onpolicy), because it has to be generated through the production API. It is built from token ids with `score_from` at the start of the generation. Its dNLL never enters a verdict (the plan's `exclusions`).

**CT-long** (32 documents of exactly 32,768 tokens, 1,048,576 tokens, `corpus_sha256 5afe80e7…`):

- 32 documents meet the critic's minimum (§10.15). The verdict basis is the document-level CI.
- It has no concatenated unrelated papers. Instead:

| Stratum | Documents | Contents |
|---|---|---|
| `prose-long` | 16 | single long arXiv 2606/2607 papers |
| `code-long` | 8 | one SparkPipe source directory per document, files in path order |
| `retrieval` | 8 | a long paper with a 32-entry access-code registry in front and 8 recall lines at the end. `probe_spans` marks the answer tokens, all past position 30,000, as a teacher-forced long-context KV probe. |

- kv-I8 is inert at context ≤ 2048 (design §4.3), so measure it only on positions above 2048.

**Where the token files live.** They are not committed. They are on the workstation at `/Users/mac/wf/qab-l2-data/corpora/` and on the hub at `rtx5090:~/qab-corpora/`. The indexes pin their SHA-256.

## Dry run

```
python3 tools/ab_dry_run.py OUT_DIR
```

This runs the whole chain on synthetic dumps with no GPU:

1. corpus, arms, dumps, exact-KL files, receipts and suites;
2. plan freeze;
3. score and suite comparisons, then verdicts;
4. the §5 report block for every comparison.

The synthetic set is:

- an anchor;
- three E arms, which come out EQUIVALENT, INCONCLUSIVE and INFERIOR (the INFERIOR one also has a suite regression);
- a spine bridge;
- two K arms, one of which degrades beyond 4k positions.

`tests/test_ab_dry_run.py` runs the dry run and checks it.
