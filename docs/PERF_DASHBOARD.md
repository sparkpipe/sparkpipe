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
| Qwen 3.8 27B (TP1) | **8.00 FP8** (tiled decode; 3.10 BF16) [validated] | 7.16 (D=2, 2.05x - still < no-spec) | - | - | - |
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
| Qwen 3.8 27B | bit-exact GPU validation + genuine prompt | D=2 validated (no B1 gain) | TP1 |
| K3 | not measured | - | TP16 serial replay sum-vs-golden PASS (worst 0.0508) - correctness only |
| GLM52 | pre-fleet | - | - |
| Qwen3.8-Max | level-1 state | - | - |
| DSV4 Pro | - | - | transport-open validated |

## Active climbs (references being matched)

- 27B: DSpark rung-3 weights packed (1.36B); parity harness draft[0]=220 matches reference, draft[1] 16-vs-17 near-tie; HF oracle DEAD (drafter trust_remote_code import deadlocks after fla/flashinfer install) -> vLLM-engine oracle confirmed on spark3 (own qwen3_dspark.py classes); instrumenting speculator for 7-draft + 5-tap capture; then 20-58 DSpark.
- Flash: keep-old CSA landed (23 exact); skew TESTED: -1u staging correction moved the stream (hash 47cf1a47) but idx 23 = STILL 688, prefix STILL 23 exact -> staging exonerated for real, corruption is INSIDE the 8-row island math. Next: zero-reference per-row pad diff (8 duplicate rows must be bit-identical per island - any row diff = row-index bug); b1 reference unblock via uniform-tps adapter fix (fork's lane); direct-body island dump in parallel -> k-sweep k=5/7/8/10.
- COMPSEC-17: added as a tracked stat once spec accuracy is reached.

Last update: 2026-08-18 21:13 UTC (CI green; Flash: pool_full128 pos-83 dump + frame-149 8-row staged-tuple dump next (pad rows may be MIXED 149/150 - non-idempotent by construction); 27B: vLLM-engine tap capture on spark3; no new HWM since 06:45)
