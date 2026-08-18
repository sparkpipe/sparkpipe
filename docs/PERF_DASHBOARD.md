# Performance dashboard (HWM records only)

Updated on every new high-water-mark measurement or at ~15min intervals.
Values are the LATEST RECORDED per model - no benchmark reruns. Sources are
receipts/PERF docs in-tree; accuracy rows show the golden-hash state.

Legend: [HWM] = best recorded. spec = speculation ON. Accuracy bar:
a model's row is accuracy-VALIDATED only when its output reproduces the
pinned golden (no-spec O128 hashes, or the COMPSEC-17 set once accuracy is
reached - COMPSEC-17 scores get their own column then).

## Output speed (decode tok/s, HWM)

| Model | B1 no-spec | B1 spec | B8 | B16 | other B* |
| --- | --- | --- | --- | --- | --- |
| DSV4 Flash (TP4) | **40.46** [exact 3/3] | **8.48** (k=7, hash WRONG - not valid) | - | - | - |
| Qwen 3.8 27B (TP1) | **8.03** mixed/FP8 29.9GB pack (repro 8.028 of the 8.00 HWM; true-BF16 54.6GB pack = 4.45 - CORRECTED from stale 3.10) [validated] | 7.16 (MTP D=2, 2.05x - still < no-spec); DSpark k=7 blocked on adapter init, next | - | - | - |
| K3 | **none measured** (never run on fleet hardware) | - | - | - | - |
| GLM52 | **6.91** | - (draft weights untrained) | 2-3 | 2-3 | - |
| Qwen3.8-Max | 1.29 per-request | - | 0.61 | - | ~39 agg @B256 (TP4xPP4 replicated) |
| DSV4 Pro | ~12-13 (bandwidth-grounded est, not e2e) | - | - | - | - |

## Prefill speed (tok/s, HWM)

| Model | B8 | B1024 | notes |
| --- | --- | --- | --- |
| K3 | - | - | RETRACTED (not hardware-measured) |
| others | - | - | not recorded / not wired (Max) |

## Accuracy / correctness state

| Model | no-spec golden | spec golden | TP correctness |
| --- | --- | --- | --- |
| DSV4 Flash | O128 3/3 exact | CSA keep-old fix landed: 23 exact tokens (2c40465d); full-O128 not yet (golden pin moved to 211462f2 batch) | TP4 exact |
| Qwen 3.8 27B | bit-exact GPU validation + genuine prompt; O128 pins: BF16 7b58db3a..., mixed/FP8 80e8fb9d... (a9385d0b is the DSV4 golden, NOT qwen36) | DSpark draft parity BIT-EXACT vs fixed reference [220,16,92,198,12,328,82] (draft[1] = genuine 19.625 tie, first-max -> 16); spec speed not yet measured | TP1 |
| K3 | not measured | - | TP16 serial replay sum-vs-golden PASS (worst 0.0508) - correctness only |
| GLM52 | pre-fleet | - | - |
| Qwen3.8-Max | level-1 state | - | - |
| DSV4 Pro | - | - | transport-open validated |

## Active climbs (references being matched)

- 27B: DSpark rung-3 PARITY RESOLVED (bit-exact, landed 1cd258f); no-spec 8.02 mixed / 4.45 BF16 confirmed; spec blocker ROOT-CAUSED: adapter arms STAGE_MTP=1 on any speculation without a spec-method check (same defect in landed lines 428-432) -> MTP init rejects dspark; fix = gate the arm with spec-method != dspark (GO issued, ships in the port deliverable); then B1 spec k=7 vs 8.02; then 20-58 DSpark.
- Flash: keep-old CSA landed (23 exact); staging CLOSED by measurement (no double-count, no pads, -1u formula matches engine continuity; 7-token lag = symptom); BUG = layer-0 wq_a FP8 pair: batch-coupled activation scale, ch 0-255 clean / 256-1023 diverge at the 128-tile boundary, all 8 rows wrong; fix in progress = align both pair paths (row-count split: !=1 native-tiled vs ==1 flat-pair) to per-row staging; verify r1_wq_a == delta_ge_raw -> then k-sweep k=5/7/8/10.
- COMPSEC-17: added as a tracked stat once spec accuracy is reached.

Last update: 2026-08-18 23:50 UTC (CI green; Flash: staging falsified-closed, kernel fix aligned to row-count split; 27B: MTP-gate application check pending; no HWM change since 8.03)
