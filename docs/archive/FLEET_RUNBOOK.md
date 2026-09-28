# FLEET RUNBOOK — how to manage the sparks, updates, and everything around them

Maintainer: mgr2. Established 2026-09-15 from the 09-13/15 wave record. Every rule
below was paid for by an incident — the incident is cited so the rule is not
relitigated. This file is referenced from the coordinator memory index; if you are
about to improvise a fleet operation, STOP and find the section here first.

## 0. Channel isolation (read first — it decides who may touch what)

| Channel | Owner | Deploy path | mgr2 tooling may... |
|---|---|---|---|
| glm53flash (glm5_next) | coredev | his `tools_local/{p,b}` scripts (tar-stream + restart) | **NOTHING.** No deploy, no restart, no monitor, no kill |
| weightd (shared daemon, production) | coredev | announced `WEIGHTD_BIN` shas | sync deliberately after announce; never kill/rebuild from module publishes |
| weightsd (stable weights channel) | mgr2 | hub root `sparkf:/home/sparkf/release/weightsd/` + per-node installer | own fully; restarts only on announced main upgrades |
| driver roots (ling, gemma4, laguna, dsv5, muse, hy4, k3, qwen38max, qwen3flash, qwen38-27b) | mgr2 lanes | per-driver release roots, nodes subscribe per ROOTS | own fully |

Isolation is bidirectional (operator ruling 09-14): coredev's deploys are his;
mgr2's waves are ours. Incident that teaches it: the out-of-band `p`/`b` deploys
`kill -9` everything matching `*sparkpipe_model*` with cwd in a root OR exe under
`sparkdata/weightd` — every shared daemon dies on every republish. Keep mgr2 state
outside every such pattern (weightsd exe lives in `weightsd1/build/`, lane
residentds in lane roots — both survive).

## 1. Fleet map

- **16 sparks**: `spark0..spark9`, `sparka..sparkf`. 128 GB LPDDR5X unified each;
  measured GPU BW 242–247.5 GB/s; NVMe 3.7 TB (data staged HERE for inference;
  ceph is warm/source tier only — operator law).
- **Hub**: `sparkf` — CODE ONLY. Release roots served over HTTP :8802 from
  `/home/sparkf/release/` (MANIFEST per root = the version; trigger = MANIFEST byte
  diff; **there is no UPDATED file**). `fleet_view_serve.py` :8801 aggregates
  heartbeats. `coredev` runs test deployments from sparkf — never touch sparkf
  processes.
- **Warm storage**: ceph at `/mnt/model-warm` — pack SOURCE only, never a serve
  path, never staged state. DEGRADED: per-client/per-region stalls and hangs are
  chronic (see §5).
- **Node paths**: packs + configs `~/sparkdata/<root>/`; binaries per-node home
  (e.g. `~/weightsd1/build/sparkpipe_weightsd`); weightsd env
  `/etc/sparkpipe/weightsd.env` (`WEIGHTSD_BIN/SOCKET/EXTRA_ARGS`); unit
  `/etc/systemd/system/sparkpipe_weightsd.service` **+ required per-node drop-in
  `sparkpipe_weightsd.service.d/10-user.conf` (`User=<node>`) — installing only the
  service file fails**; node agent `~/sparkdata/core/bin/fleet_node_agent.sh`.
- **Ports**: per-driver uniform 1024 blocks 12000-30999 (PORT_LEDGER.md);
  weightsd 61000-61127; services in the 60xxx/64xxx planes; append-only, never
  renumber.

## 2. Standing laws (each paid for — do not relearn)

1. **One active ceph consumer fleet-wide.** Concurrent warm readers collapse to
   kB/s and hang at deterministic offsets. Claim
   `/Users/mac/sparkpipe-coord/CEPH_LEASE` before warm reads; migrate to an idle
   client when one crawls (dd-probe per candidate; spark4 crawled 64 KB/s while
   spark5 did 69 s vs 84 min for the same reads).
2. **Never read_bytes() a multi-GB pack inside a capped cgroup.** The 30.5 GB
   `read_bytes()` verifier livelocked in `mem_cgroup_handle_over_high` for hours —
   indistinguishable from a storage stall. Stream digests; MemoryMax ≥ 12 G for
   byte-trace verifiers; sparkcap 2900 M livelocks multi-GB-tensor packers (k3
   needed 64 G/56 G).
3. **Hub-first for every fleet binary.** Workstation-originated scp loops hit sshd
   rate limits (48 rapid connections → 100 % failure) and bypass the version
   channel. Stage on sparkf, MANIFEST last, nodes pull.
4. **Upgrade installs must be temp+mv.** Writing over a running daemon's binary is
   ETXTBSY ("dest open: Failure" on every upgrade — first installs never expose
   it). `install` to `<bin>.new`, `mv -f`, then restart with the guard.
5. **Deploy tools never touch shared daemons.** Kill only `cwd == own runtime
   root`; drain (TERM + poll + deadline) before any `-9`; `weightd` is never a
   `mgr2` kill target (weightsd owns its lifecycle via announced shas).
6. **No silent fallbacks.** `#ifndef X / #define X <guess>` and every
   env-or-default for required config doubles the state space (2^N). Required =
   fail loudly (missing `SET_ME_*`, `#error`, named fail-closed status).
   Optional = declared, single-sourced, asserted.
7. **Every claim graded MEASURED / DERIVED / ASSUMED** with an artifact trail
   (file/node/date/build sha). Ungraded = not a fact. T1 (accurate tokens vs
   reference) precedes any hillclimb.
8. **Purge after heavy I/O batches** (`sudo -n sync; echo 3 >
   /proc/sys/vm/drop_caches`); one real TP16 mesh at a time (MESH_LEASE);
   mini-mesh/smoke stacks are a lighter class (SMOKE marker, ≤ 2 extra CUDA
   processes per node).
9. **Placement receipt chain**: verify gate → place per replica law (TP8: rank r →
   spark r + r+8; TP4: rank = node % 4) → dest-sha → `.experts` regenerated →
   `.sha256` + receipt → `chattr +i` → census. Deletions: re-verify sha first,
   `lsof`/`lsattr` pre-check, receipt bytes.
10. **Asset identity**: pack content must match its declared source snapshot
    (router-gate u16-multiset fingerprint vs the source checkpoint). A pack whose
    declared revision misrepresents its content is a defect even when it decodes
    (glm53flash incident, 09-15).

## 3. Procedures

### 3.1 Code/binary rollout (hub-first)
1. Build on a node from a merged main sha (never from a lane).
2. Stage into the hub release root: `/home/sparkf/release/<root>/` — `mv` into
   place, **write MANIFEST last** (it is the commit bit).
3. Nodes poll (~1 s) and apply on MANIFEST diff. Verify: heartbeat JSON updates,
   per-root sha receipts.
4. Rollback = restore the previous root tree + MANIFEST (keep one generation).

### 3.2 Weightsd update (stable channel)
1. Land the change on main via PR (the daemon ships from main only — no lane
   builds of weightsd reach the fleet).
2. Build: `make build/sparkpipe_weightsd` on a node; record sha256.
3. Announce the sha (this IS the restart authorization).
4. Per node: install to the env's `WEIGHTSD_BIN` path (temp+mv), restart with
   `WEIGHTSD_DEPLOY_RESTART=1`, verify active + socket + sha16. The unit drop-in
   (`User=<node>`) must exist — see §1 node paths.
5. Receipt: `runs/weightsd-channel/` per node.

### 3.3 Pack placement
See law 9. Sources: warm storage (claim CEPH_LEASE) or an existing placed
verified pack (node-to-node). The verify gate must stream (no `read_bytes()`),
or use the parked-receipt pattern only when the receipt is trusted.

### 3.4 Node health triage (the NVRM/degradation class)
Symptoms seen: `NVRM NV_ERR_NO_MEMORY` at kernel context creation
(fleet-synchronized 01:22 incident; cudaSetDevice=2 for every process); a node
refusing NEW CUDA contexts while old ones run (sparka, spark9).
1. `nvidia-smi` (GB10 shows memory "Not Supported" — do not misread), `fuser -v
   /dev/nvidia*`, identify holders and ages.
2. A fresh boot allocating fine then failing in-kernel (`kgrctxA` OOM) = driver
   pool exhausted → the fix is a node **reboot**, authorized by the operator
   (sysadmin/coredev execute). Killing holders did NOT clear it.
3. After reboot verify: driver banner clean, weightsd auto-started (unit is
   enabled), production root boot via the node agent, zero new NVRM lines.
4. Escalate the pattern (three nodes in one day) to coredev/sysadmin — multi-
   process CUDA over days may leak the unified pool; root cause OPEN.

### 3.5 Multi-dev parallel work
- 8 lanes live in weightsd (post-#1013): each connecting residentd gets its own
  lane; eviction arbitration L < M succeeds, L ≥ M → `SPARK_STATUS_EVICT_DENIED`;
  lowest-priority weights evict first; evicted devs re-stream via lazy miss.
- Measured: N=1/2/4/8 concurrent meshes PASS (200/200 rounds each); per-dev
  CUDA co-residency PASS at 7 devs + live mesh on one node (~80 % aggregate
  bandwidth retained; MPS unsupported on GB10 and not needed).
- Working-set quantification: weightsd lease accounting gives bytes-per-prompt
  per dev (ling 1.01 GB, glm53full 11.13 GB, qwenmax 52 GB measured — spine does
  not shrink, dense models do not shrink). Size concurrency from measured
  working sets, not pack sizes.

### 3.6 Zombie / leftover sweep
Scope by owner patterns only: lane roots (`~/t1g53`, `~/t1qmax`, `~/acc1`, …),
private test daemons (distinct socket paths), workstation scopes older than the
task. NEVER: production roots, `coredev`'s processes, other lanes' cwd roots,
anything holding an ACTIVE lease. Every kill receipted (pid, cwd, age, reason).

## 4. Escalation and ownership

- **Operator**: node reboots, snapshot/canonical rulings, tolerance rulings,
  coredev coordination.
- **coredev**: engine core, weightd internals, glm53flash channel (his deploys,
  his wedge — the 1-request wedge is his known class).
- **sysadmin**: ceph health (degradation is chronic — do not trust "fixed"
  without a per-client probe), node hardware.
- **mgr2**: everything on the 12 non-glm53flash lanes, weightsd channel, pack
  placement, the release protocol, and this runbook.

## 5. Locations

- Coordination: `/Users/mac/sparkpipe-coord/` (SHARED_DECISIONS.md append-only,
  PORT_LEDGER.md, LANES.md, CEPH_LEASE, MESH_LEASE,
  DEV_MISSION_GOALS_TEMPLATE.md).
- Repo docs: `docs/DEPLOY_PROTOCOL_MAP.md`, `docs/DEPLOY_MULTI_DEV_PROPOSAL.md`,
  `docs/DEPLOY_ROLLOUT.md` (dormant — see PR #1011 comment),
  `docs/CONSTANT_AUDIT.md`, `docs/COMMON_MODULE_ARCHITECTURE.md`,
  `docs/WEIGHTSD.md`, `docs/T1_REFERENCE_COMPARE.md`,
  `docs/WEIGHTD_EXECUTE_STABILITY.md`.
- Instruments: `tools/weightd_execute_receipt.py` (+ `.c` probe),
  `tools/roofline_estimator.py`, `tools/t1_reference_decoder.py` +
  `t1_reference_compare.py`, `tools/k3_checkpoint_oracle.py`,
  `tools/acc_parity_oracle.py`, `tools/fleet_release_hygiene.sh`.
- Wave ledgers: `docs/AGENT_LANE_BRIEFS/reports/` on their `lane/*` branches
  (see SHARED_DECISIONS for the branch+SHA index).
