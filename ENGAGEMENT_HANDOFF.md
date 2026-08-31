# glm5_next engagement redeploy — debug state handoff (2026-08-31 ~11:50 KRT)

## LANDED ON MAIN (all offline-gated):
- 15b412b: adapter private algorithm loader -> #760 semantics (latent bug, fixed)
- 6396c91: step-rail shape fix — THE config-layer root cause: generator emits
  16-entry d2a peer routes; both step-rail validators demanded 3. Fixed both
  (shared template + glm5_next). NEW GATE: tests/test_glm5_next_adapter_config_load.py
  (generate -> compile real adapter -> load; kills generator/adapter shape drift).
- Driver/loader now reach: LoadConfiguration rc=0, LoadTpCollective rc=0 ON HARDWARE.

## CURRENT FLEET STATE (16/16 staged, NOT serving):
- driver: clean-tree build fa4f0634 (spark2:~/g5v, from 6396c91+diag prints)
- adapter: d3f11a91 (clean tree) — plus G5DRVLOAD-diag variant on spark2
- residentd/api: g5v builds (main)
- configs: generator (1024 rows + d2a + split 2048) 16/16
- packs: lane's fixed2 (unchanged, headers verified: contract a40e9ec5@0xA1, config, recipe all present)
- Backups: each node ~/sparkdata/glm5_next.tp16.bak-*/ (pre-engagement full set)

## THE ONE OPEN FAILURE: adapter_initialize rc=1 (INVALID_ARGUMENT)
- ModuleConfigure SUCCEEDS (G5CFG print fired: abi=5 db=3304 stage 0/1 L45 codec5 tp2/16, all geometry checks pass by eye)
- The module validator (same source, same pack, same node) PASSES end-to-end
- => the delta is in what the RESIDENTD passes vs what the validator harness passes:
  host_services kv_logical/physical XOR check (spark_module_abi.h:70) is the prime
  suspect — never printed on the residentd path. Deployment sets both 0 (XOR passes
  on paper); setting them non-zero fails EARLIER at deployment_validation (reverted).
- Remaining un-instrumented: InitializeState's variable-returns + the glue. Attempts to
  instrument were DEFEATED BY THE MODULE_LIBRARY CACHE: publish reuses link_units and
  the compile binds the active record — edited source does NOT always reach the driver.
  WORKAROUND for next session: rm -rf build/module_library/link_units build/module_library/active
  before EVERY publish, and VERIFY with `strings driver.so | grep <tag>` BEFORE testing.

## KEY BUILD FACTS (hard-won):
- CONTRACT_SHA256 must be the FIRMWARE-JSON sha a40e9ec5... (NOT the authoritative contract sha — TARGET_MISMATCH otherwise)
- spark0's ~/g5m5-src tree is POLLUTED (stale objects produced wrong artifacts; clean
  builds on spark2:~/g5v produce different, correct hashes). Do not trust incremental builds there.
- The M5 stage script's elif staging fallback copies the OLD driver — the fresh one
  lands in $RR/stages/stage_000/model_driver.so. Use that path explicitly.
- sparke is FULL; spark0 is the M5 build host; ssh aliases route via ds4_spark_ssh_proxy
  (sparkf timed out transiently once; retry).
- pkill -f <pattern> in an ssh command kills YOUR OWN session if the pattern appears
  in your command line. Run pkills in a separate ssh.

## QUEUE (operator directive, pending): stale jobs lock the queue (19h-running task with
45min ttl qwen38flash-shamatch2; 2 note entries 19h old). Fix: ttl enforcement + expiry
at dispatch; flash dev will use the queue going forward.
