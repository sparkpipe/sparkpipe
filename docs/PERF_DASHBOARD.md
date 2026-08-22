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
| Qwen 3.8 27B (TP1) | **8.03** mixed/FP8 29.9GB pack (repro 8.028 of the 8.00 HWM; true-BF16 54.6GB pack = 4.45 - CORRECTED from stale 3.10) [validated] | DFlash2 block8: **5.93** @ L=0.55 first-pos 45% - FIRST VALID LOSSLESS spec measurement (spec stream == no-spec golden, 128/128 bit-exact, hash 84c1035f); LOSES to no-spec (0.74x); acceptance gap vs authors' L=3.39 -> tap-capture interrogation next. MTP D=2 7.16; DSpark k=7 5.068 @ L=0.735 (10.5%) | - | - | - |
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

- 27B: DSpark port LANDED (49aa3b3 + gate coverage); greedy 5.078 @ 10.5%; sampling lever empirically CLOSED (probabilistic worse vs greedy target). DFLASH2 ADOPTION GO (spec landed c5053e7): Qwen3.8-27B drafter exists (z-lab, 3.85GB BF16, 5 layers, block 8, taps [5,19,33,47,61]); acceptance 4.80 vs DSpark 3.62 (+32.6%), vs no-spec 3.11-3.43x @ conc1; conv+selector bolt-ons, Markov slots = codebooks VERBATIM; gates: LM-head-BF16 check (step zero) + upstream sm120 OOB warning; W1-W8 ~8-11 days.
- Flash: keep-old CSA landed (23 exact); staging FULLY EXONERATED (frame-149 = anchor 290@150 + 7 distinct drafts sequential - mixed-staging falsified; pad = first-decode only, all-identical per-row); BUG = layer-0 wq_a FP8 pair batch-coupled activation scale (ch 256+ at 128-tile boundary); FIX DRAFT EXISTS in fork clone (multi-row -> per-row-exact routing) - build + r1_wq_a==delta_ge_raw verification pending; then k-sweep k=5/7/8/10.
- COMPSEC-17: added as a tracked stat once spec accuracy is reached.

Last update: 2026-08-22 13:20 UTC (PEER SESSIONS DELIVERED R1 - 5 of 7 reports in docs/coord/reports-r1 (committed 1121ab2). FLEET: only 7/16 boxes reachable (spark1,8,9,a-f DARK - banner timeouts); GLM5.2 PP7 BOX CORRECTION per user: spark4-7 ARE RESERVED FOR DSV4 FLASH and are alive - the PP7 squad comes from the dark boxes (spark1/8/9/a-f) once recovered. GLM52 audit: prerequisite #1 = restore the GPU numerical validator (driver ships with no runnable in-tree GPU gate); ~1200-1500 lines collapsible via one shared stagepack reader across four drivers; specific compat shims + paranoid checks named for deletion. HWIFACE v0: grounded in measured facts (61 launch wrappers/93 call sites, graph islands already exist as 3xL+1, runtime primitive inventory complete) - v1 must fold in the advisor boundary table. K3: #667 divergence root-caused+fixed on unified (documented); TP16 pack production + e2e run remain. MAX: bring-up runbook delivered (driver exists, never hardware-tested). DSV4PRO+DSV4FLASH peers: relaunching (died in the web-UI contention - headless boots now launched solo). qwen36 DEPRECATED per user. Agents: fresh-spawn attrition ~80pct during saturation; legacy agents dead-on-turn; peer-session transport works (5/7 delivered). NO HWM MOVES.
