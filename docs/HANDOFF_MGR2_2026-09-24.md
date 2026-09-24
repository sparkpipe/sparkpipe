# MGR2 HANDOFF ADDENDUM — 2026-09-24 (post-astra station, both PRs merged)

Read the 2026-09-23 handoff first (docs/HANDOFF_MGR_2026-09-23.md on lane/handoff-mgr) — the
process laws, the rig identification, and the lane histories all still apply. This addendum
covers astra's station deployment and the rebased lane map.

## WHAT ASTRA LANDED (both MERGED on main, now at 68e80c07)

**#1198 — Restore reproducible serving and protect residents.** The real defects fixed:
competing APIs disconnecting each other's residents (now BUSY), deployment drift (replaced
binaries, disabled graphs, wrong batching geometry, bad memory reservations), the API ignoring
its local runtime adapter path, and CUDA failures hidden by cleanup errors. Reproduced 13.33
tok/s aggregate from the original release; the restored server passed 18 requests / 576
matching tokens including 4 concurrent callers. All 32 fleet services managed and accounted.

**#1199 — The guarded development station.** APIs and tokenizers on RTX5090 (user spec);
Sparks run residents + shared weightd + queued builds. Seven families resident (Qwen Max
excluded — exceeds the live memory allowance by ~7 GiB). The station workflow pins
binaries/config, checks memory and ports, waits for every rank, rejects shared-core edits in
driver updates, documents rollback. Station runbook:
`/Users/mac/.sparkpipe/station-20260924/README.md` on mac-studio; also
docs/operations/managed-driver-station.md in the repo.

## THE PERFORMANCE CORRECTION (load-bearing)

The 13.5 tok/s figure was AGGREGATE across 8 instances (~1.7 each). The isolated API on the
new stack measures **5.70 decode tok/s median** — a regression vs. the campaign's 12.81
isolated, and the top performance item. PR1198 carries the exact reproduction commands.

## LIVE API

`http://100.123.97.61:8433/health` → `{"status":"ok","served":5,"tokenizer":true}` (verified).
B1 / 512-token context profile, qualified binaries.

## REBASED LANE MAP (the six families needing inference fixes = our lanes)

| Driver | Astra's verified status | Lane's named fix needed |
|---|---|---|
| GLM Flash | 4 exact 32-token reference passes; concurrent OK | WORKING — perf regression (5.70 vs 12.8) |
| Gemma 4 | 6 repeatable requests incl. restart; numerical oracle needed | Launch-16 proven; add the oracle |
| Qwen 27B | Initializes; MTP/prefix-publication contract fails | The MTP-skip (#1194) may need more; check against the contract |
| Qwen Max | Initializes separately; admission callback unimplemented | The admission callback — maps to our #1194-era work |
| Kimi K3 | Initializes; request/lease completion stalls | The width contract (#1190) may not cover this; the lease-stall is new |
| Laguna | Initializes; routed-expert CUDA execution fails | After #1191's frame fix; the routed-expert path is the next layer |
| Ling | Initializes; CUDA execution fails | After #1189's collective fix; the CUDA execution is the next layer |
| MiniMax | Initializes; inference hits CUDA illegal access | The decode cell (#1195) will hit this; the illegal access is the next layer |

## THE STATION WORKFLOW (new for all lanes)

1. Read the runbook FIRST (path above). The station manages placement, budgets, and rank
   readiness — do not hand-deploy around it.
2. Driver updates must descend from the registry's core_commit, change only recorded
   driver_paths + the two generated inventory files. Run check-driver before scheduling builds.
3. Core/daemon/transport changes require an operator release (the boundary law, now enforced
   by tooling).
4. Known limitation: a driver crashing during GPU work can require restarting the shared mesh.
   Six drivers still need inference fixes; crash isolation is unfinished.

## NEXT ACTIONS (priority order)

1. **Performance regression**: the 5.70 vs 12.8 gap on GLM is the program's top number.
   The hillclimb lane's rates branch + the station's reproduction commands are the entry.
2. **Six inference fixes** (the table above) — each maps to a lane with its full history.
3. **Qwen Max admission** (the 8th family, memory-excluded) — needs either a memory trim
   or the arena-basis pinning decision (#1155).
4. The rig (wdcore/sup.sh from 10.20.0.1) — astra is active; confirm they own it now before
   any kill orders.
5. The open items from the prior handoff (ceph clients, workflow token, #1142/#1155) stand.
