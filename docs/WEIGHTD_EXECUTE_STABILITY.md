# weightd module-execute stability — diagnosis and fix (2026-09-13, D-1 lane)

FAILURE CLASS ON RECORD (mgr2 dispatch): the 00:26 09-12 weightd deploy was
unstable under module-execute; spark3 showed lease/map IO_ERROR during module
execute; spark2 double-restarted; the phase-2 fixed-slot ring wedge was open.

## Verdict first

TWO independent causes, neither of them the 00:26 daemon build:

1. CLIENT BUG (code, fixed here): `SparkWeightdMapDestroy` freed the client
   VMM reservation with `span_bytes` while the reservation made by
   `SparkWeightdMapCreate` is `span_bytes + chunk_bytes` (the eviction-epoch
   control chunk, added in a08eb8e "weightd: eviction epoch"). CUDA requires
   the full reservation size in `cuMemAddressFree`; the partial-size call
   fails, so EVERY clean lazy-map teardown returned SPARK_STATUS_IO_ERROR.
   That is the "lease/map IO_ERROR under module execute" signature: every
   module run ends its wave with teardown, and every teardown failed. It is
   invisible in the stub-based unit tests because the stub accepts any free
   size; it reproduces 100% on real CUDA.

2. OPERATIONAL KILLER (not in this repo, evidence for mgr2): every weightd
   daemon in the fleet is SIGKILLed in synchronized waves every ~10-30
   minutes, continuously since at least 09-10, across all 16 nodes. Each wave
   fires ~12 seconds after a glm53flash.fp8.tp16 root-manifest republish on
   the release server. A live process trap on spark3 caught a `tar -C
   .../glm53flash.fp8.tp16 -xf -` stream-deploy 9 seconds before the kill.
   There is no kill path for weightd in the deployed fleet_node_agent.sh
   except install_core (which logs "deliberate restart"; absent in these
   waves), no kernel crash records (no segfault/trap/OOM in kmsg), no
   coredumps, and the daemon logs end mid-ship with no shutdown line. The
   killer therefore arrives via SSH as uid-1000/root automation outside this
   repository's tools. Every wave also restarts the serving residentds, so
   the operator-visible serving instability, the client-side socket-EOF
   errors, and likely the ring re-wedge cycles are downstream symptoms.

## Evidence chain (cause 1, code)

- `runtime/spark_weightd_map.c`: reservation at MapCreate is
  `span_bytes + chunk_bytes` (map_initialize_cuda); Destroy freed
  `cuMemAddressFree(map->base, span_bytes)` — partial size.
- Reproduced on spark0 (GB10, CUDA 13) before the fix: fixture probe ends
  `ERRSITE runtime/spark_weightd_map.c:221 status=4` (IO_ERROR) on every
  run, then `probe-18` (LazyPackDestroy) — 100% of runs.
- After the fix: 50/50 fixture execute cycles PASS and 50/50 real-pack
  (qwenmax.nvfp4.tp16 rank0, 98 GB, 2944-expert manifest) lease
  acquire/read/record/release waves PASS, daemon alive, zero daemon errors.
- `tests/test_weightd_working_set.c` could not catch this: it did not even
  compile since a08eb8e (3-arg SparkWeightdMapCreate vs the 4-arg epoch
  signature), and the cuda stub's `cudaErrorNotReady` was 34 while real
  CUDA's is 600, so even repaired it mis-derived release status on machines
  with a real CUDA toolchain. Both repaired here; the repaired test now
  passes its map-lifetime section and reaches a pre-existing final
  outstanding-allocs failure (present with HEAD's map.c too — separate
  test-infra follow-up).

## The fix (runtime/spark_weightd_map.c)

- `map_free_initial` returns the `cuMemAddressFree` status (full
  `span_bytes + chunk_bytes` size, epoch unmap first) instead of ignoring it.
- `SparkWeightdMapDestroy` drops its wrong-size partial free and propagates
  `map_free_initial`'s status; the create-failure path discards it.
- No wire/ABI change; daemon untouched; consumers need no rebuild beyond the
  library.

## The killer (evidence chain for mgr2)

- Timeline pairs (spark3 fleet-agent journal): every "weightd: starting"
  death lands 11-13 s after a "glm53flash.fp8.tp16: manifest changed;
  syncing" line — 20+ exact pairs on 09-14 00:00-04:40 KST alone; deaths
  continue under daemon builds d0288f4 (pre-00:26), 595487 (the 00:26
  deploy) and 49b8e01e (current), so no daemon build is implicated.
- Same-second deaths on spark0-spark5 checked (03:26:45/46, 03:52:59,
  04:00:43, 04:10:09-11, 04:19:37-38) — fleet-wide fan-out.
- bash reports the deaths as "Killed" (SIGKILL), never "Terminated" or
  segfault; `sudo journalctl -k` has zero weightd trap/segfault/OOM records;
  no coredumps.
- Deployed fleet_node_agent.sh has exactly one weightd kill path
  (install_core) and it logs "core: weightd ... deliberate restart" before
  killing; that line does not appear in the kill waves.
- Live /proc watcher trap on spark3: `tar -C /home/spark3/sparkdata/
  glm53flash.fp8.tp16 -xf -` (a stream-deploy via SSH) 9 s before the
  04:49:50 KST kill; the tar is not part of any repo tool.
- Consequence: every republish of the glm53flash root takes the entire
  fleet's weightd mesh and serving stack down for seconds and repeatedly
  re-wedges the phase-2 rings. Identify and fix the owning republish loop
  (it does not live in this repository; candidate owners are the
  workstation-side glm53flash release lanes or the hub's release server
  hooks).

## The execute-receipt rig

`tools/weightd_execute_receipt.py` (plus its engine
`tools/weightd_execute_probe.c`) proves daemon execute-stability per node:
it spawns a node-PRIVATE daemon (never the production socket), runs 50
synthetic full-execute lifecycles, then N real-pack lease
acquire/read/record/release waves against a placed pack, optionally soaks
(`--soak-seconds`), and fails on any probe failure, daemon death, or daemon
error line. It prints a JSON receipt. This is the serve-gate prerequisite
check: a node must present a PASS receipt for its target pack before qwen38max
serving is promoted onto it.

Rig run law on shared nodes: run it under
`sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M
--uid=1000`, purge the disk cache after batches
(`sudo -n sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`), and
never point it at `/tmp/spark_weightd.sock`.

## What it means for the qwen38max serve-gate

- The map-teardown IO_ERROR would have failed every module-execute wave on
  the real arm; it is fixed and proven on the placed qwenmax.nvfp4.tp16
  rank0 pack on spark0.
- The gate is NOT green until the republish killer is retired: any wave can
  kill the serving stack mid-execute regardless of code health. The rig
  receipt is per-node proof of code health; the killer is an operational
  blocker that must be traced to its owning loop (mgr2/operator action).
- PR #991 (v2 128B stagepack wire) remains open and is unaffected by this
  lane; the rig works at the daemon wire level and does not need it.
