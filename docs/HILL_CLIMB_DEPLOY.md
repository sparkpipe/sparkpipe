# Deploy cycle hill climb: 10 minutes to 25 seconds

## The starting point

Deploys took 10+ minutes when they worked at all. Most of the time they didn't:
nodes would wedge, sessions would deadlock, the fleet would split across
configuration generations, and manual kill-drain-restart waves were the only
way to recover. One bad deploy killed spark6 (115GB leak per boot attempt).

## What was measured

`time(MANIFEST change on hub) -> 16/16 residentds reporting ready`

## The climb

| Change | Time | What was removed |
|--------|------|------------------|
| TCP rendezvous (baseline) | 600s+ or wedge | The entire failure surface: per-pair listeners, port tables, EADDRINUSE races, listener fd leaks, 2-minute connect timeouts, ghost pairs from stale QPNs |
| Two-phase UPDATE ledger | -60s | 16 nodes appending down:/up: lines via ssh to coordinate a synchronized stop; replaced by kill-9 + independent restart (nothing durable lives in a residentd) |
| TERM drain (30s wait) | -30s | Ceremonial graceful shutdown of a process that holds no state; kill-9 is instant and safe |
| 5s agent loop | -4s | 1s loop; manifest fetch is one curl |
| Serial module build | -120s | -j16 on all three build targets (host, adapter, module publish); 20 cores on sparkf |
| Force-delete artifacts | -60s | Incremental builds: only changed objects recompile; a transport-only change rebuilds in 2s |
| Warm-attach SHA-256 | -11s | Client re-hashed 24,192 manifest entries on EVERY attach (11 seconds of pure CPU); now skipped when the daemon reports the pack already resident |
| Serial route maintainer | -120s | All dead routes retry concurrently (was: one route at a time × 120s timeout each = 14 minutes worst case) |
| Agent record fetch in loop | -300s | The agent's sync_rendezvous blocked the entire 1-second loop on a wait for 15 parallel curl subshells; fire-and-forget upload only |
| **HTTP direct fetch** | **-180s** | **The transport itself fetches peer records from the hub over HTTP (~4ms per fetch). No agent intermediary. Record propagation went from 3-5 seconds to 4ms.** |
| Open timeout 120s -> 10s | -110s | Sessions waited 2 minutes before the await timeout could fire; the await timeout (10s) was gated on the open deadline (120s) |
| Rendezvous await 180s -> 10s | -170s | Per-session await timeout; the maintainer retries at 200ms so convergence = 10s worst case per cycle |

## Final numbers

| Metric | Before | After |
|--------|--------|-------|
| Build (transport change) | 5 min | **2s** |
| Build (module change) | 5 min | **70s** |
| Build (full clean) | 5 min | **~2 min** |
| Agent detect + fetch | 5s | **1s** |
| Kill + restart | 30s | **<1s** |
| Record propagation | 3-5s per hop | **4ms** |
| Session open timeout | 120s | **10s** |
| Route recovery | 120s serial per route | **10s parallel all routes** |
| **Total (manifest -> ready)** | **600s+** | **~25s** |

## What made the difference

1. **Delete the coordination, not add resilience to it.** The two-phase
   ledger, the TERM drains, the port tables — these existed to manage a
   fragile design. Kill-9 + independent restart + patient route recovery
   made them all unnecessary.

2. **Put the data where the reader already is.** Peer records went from
   NFS mount -> agent scp -> agent curl -> local file (3-5 seconds, 3
   network hops) to a direct HTTP GET from the transport (~4ms, 1 hop).

3. **Verify once, load forever.** The warm-attach SHA-256 over 24K entries
   was checking something the daemon had already checked. The daemon's
   `loaded_from_pack` flag tells the client when to skip.

4. **Time everything.** Every optimization above was invisible until we
   measured the individual stages (build, detect, fetch, kill, boot, wire,
   register). The 120-second open timeout was hiding inside "the fleet
   takes a long time to converge" until we looked.

5. **Remove, don't add.** Every line deleted was a bug that couldn't
   happen anymore: TCP listeners can't leak if they don't exist, ports
   can't collide if there are no port tables, coordination can't deadlock
   if there's no coordinator.

## Remaining gap to theoretical (~3s)

- The agent uploads records to the hub on a 3-second cadence (scp + ssh);
  inlining the upload into the transport's HTTP POST would drop this to
  ~4ms
- The weightd attach still does a 2s spine I/O load on warm attach (the
  map needs to carry spine offsets so the client can use the daemon's
  already-loaded copy)
- The module build (nvcc) takes 70s even with -j16; pre-compiled kernel
  objects cached by content hash would drop this to link-only (~5s)
