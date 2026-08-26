# Top-10 infra improvements — SYSADMIN

Metric: maximize **validated Solutions / (net production code size²)**.
Every item states the solution it buys. Infra items buy *operational* solutions
(availability, MTTR reduction, or DRY) rather than the performance-level ladder;
DRY wins are flagged where the delta is negative (deleting lines is a solution
at zero cost). Owner for every item is SYSADMIN unless noted. "First step" is
the smallest landed change that unblocks the rest.

---

## 1. Boot-time fsck fix: nofail + fs_passno=0 on every data mount (DESIGNED, LAND IT)

- **What:** Apply `/tmp/ds4_fastboot_fix.sh` to `/etc/fstab` on all 16 hosts:
  add `nofail` + `x-systemd.device-timeout=10s` and set `fs_passno=0` for
  every data mount (extnvme/sparkdata/kv/nvme/raid and /home), leaving
  root/boot/var/usr/tmp/var-log alone (`/tmp/ds4_fastboot_fix.sh:15,26-35`).
- **Why:** This is the exact failure that stalled the GLM52 band for 14 h: a
  dirty data NVMe held `systemd-fsck@` → boot never reached
  `systemd-user-sessions` → pam_nologin stayed up → no SSH. With
  `nofail`+`fs_passno=0`, a dirty data volume can never block boot or SSH.
- **Expected value:** Eliminates an entire failure class = one operational
  solution (band never dark from a dirty data NVMe again). Zero ongoing cost.
- **Code delta:** 0 new lines — the script already exists; the work is landing
  it (plus a one-time `systemctl daemon-reload`).
- **Owner:** SYSADMIN.
- **First step:** Land it on each host as it returns (playbook §5 step 2), and
  fold the patch into the base image/cloud-init so future hosts ship fixed.

## 2. Emergency sshd must bypass pam_nologin

- **What:** Give the :2222 emergency listener its own PAM service with
  `pam_nologin` removed (or `PermitRootLogin prohibit-password` + a
  dedicated recovery key), so it stays usable during a stuck boot.
- **Why:** Verified live this incident: :2222 answers with a banner but rejects
  `sparkemerg` with "System is booting up … pam_nologin(8)". The emergency
  channel is dead exactly when it is needed — this is why the watcher is stuck
  at stage 2 (banner) and never reaches stage 3 (login)
  (`/tmp/ds4_bootwatch.log:259,390`).
- **Expected value:** Turns a blind power-cycle into an in-band login + manual
  fsck path → MTTR drops from "smart-plug gamble" to "ssh in and fix". One
  operational solution (recoverability).
- **Code delta:** ~10 lines (a `/etc/pam.d/sshd-emergency` without
  `pam_nologin`, pointed to by the 2222 unit's `SSHD_OPTS`).
- **Owner:** SYSADMIN.
- **First step:** Patch the emergency-sshd unit on one host and verify
  `ssh -p2222 sparkemerg@…` works while `/run/nologin` is present.

## 3. Complete the failover profile for sparkd/e/f

- **What:** Add `sparkd/e/f` nodes to `spark_ssh_failover.json` (mgmt
  10.20.0.23/24/25, fabric 10.10.100.23/24/25, tailscale
  100.81.20.86/100.100.21.86/100.88.217.33) and add their `-emergency`/
  `-wifi`/`-200g` ssh-config entries + emergency known_hosts.
- **Why:** The profile ends at sparkc (`spark_ssh_failover.json:119-123`);
  `--node sparkd` fails "unknown node". ssh config gives them only direct mgmt
  + `ProxyJump spark0` for fabric (`~/.ssh/config:788-823,862-897`) — no
  emergency/wifi path. 3 of the 8 incident hosts ride a second-class path.
- **Expected value:** One DRY/consistency solution — all 16 hosts go through the
  same proxy + route ladder instead of two divergent code paths.
- **Code delta:** ~+30 lines of config (no new logic).
- **Owner:** SYSADMIN.
- **First step:** Add the three `NodeRoute` entries and regenerate the
  `# DS4 SPARKNETWORK` block of `~/.ssh/config`.

## 4. Fix tailscale fleet identity for sparkd/e/f

- **What:** Re-enroll sparkd/e/f under the fleet's `tagged-devices` identity.
- **Why:** `tailscale status` shows sparkd/e/f on `experiencenow-ai@` while
  the other 13 are `tagged-devices`. Split-tailnet means ACLs/tags/key-rotation
  do not apply uniformly — a security and consistency gap on 3/16 hosts.
- **Expected value:** One DRY/security solution (uniform fleet identity).
- **Code delta:** 0 (a tailscale re-auth operation, not code).
- **Owner:** SYSADMIN.
- **First step:** Re-issue the tagged auth key and re-login sparkd/e/f, then
  confirm all 16 show `tagged-devices`.

## 5. Stage a GRUB one-shot fsck.mode=skip entry on every host

- **What:** Pre-bake a fallback `ds4-fastboot` menuentry with
  `fsck.mode=skip fsck.repair=no` (via `/etc/grub.d/`) plus a grub-env flag
  so the smart plug can make it the next boot.
- **Why:** The recovery decision tree (playbook §4) depends on the one-shot skip
  being *pre-staged* — with no BMC/console there is otherwise no way to inject
  it, and a plain power-cycle reboots back into the same fsck stall.
- **Expected value:** Makes the primary remote recovery lever actually
  executable → MTTR for the next dirty-NVMe event drops to ~1 reboot.
- **Code delta:** ~+15 lines (a grub.d snippet + `grub-editenv` convention).
- **Owner:** SYSADMIN.
- **First step:** Land the menuentry on one host and prove a flagged reboot
  skips fsck on the data NVMe.

## 6. Out-of-band console (vPro/AMT/serial-over-LAN) on the mgmt VLAN

- **What:** Enable and document an OOB console path (Intel AMT/vPro SOL, or a
  serial concentrator) on 10.20.0.0/24, reachable from the Mac.
- **Why:** This incident has **no** console: every recovery is a blind
  power-cycle and root cause is inferred, not observed. An OOB console converts
  "reboot and hope" into "watch fsck and intervene".
- **Expected value:** One operational solution (observability) — permanently
  lowers MTTR and removes the blind-reseat risk.
- **Code delta:** 0 code (hardware/config enablement + a runbook).
- **Owner:** SYSADMIN (ops/procurement).
- **First step:** Inventory which hosts expose AMT/vPro and enable SOL on two,
  then verify console from the Mac.

## 7. OOM guardrails on the residentd (root-cause containment)

- **What:** Run `sparkpipe_model_residentd` under a systemd unit with
  `MemoryHigh`/`MemoryMax`, `OOMScoreAdjust`, and `Restart=on-failure`,
  instead of `setsid -f … ` + `pkill` (`tools/fleet_swap.sh:63-78`).
- **Why:** The GLM52 OOM (resident decode) is the root cause; a hard
  power-cycle on top of an OOM'd resident left the NVMe dirty. Bounded memory +
  clean restart prevents the OOM→kill→dirty-NVMe chain.
- **Expected value:** One operational solution (resilience) — removes the
  trigger for the entire incident class.
- **Code delta:** ~+20 lines (a unit template + start_model via `systemctl`).
- **Owner:** SYSADMIN (with coordinator sign-off on `tools/fleet_swap.sh`).
- **First step:** Convert `start_model` to `systemctl start` for one model
  and validate `fleet_status.sh` still reads it.

## 8. Single source of truth for the node inventory (DRY)

- **What:** Derive `spark_ssh_failover.json`, the `~/.ssh/config` spark block,
  and the telemetry topology from one canonical inventory (extend
  `tools/devcycle/fleet_registry.json` with mgmt/fabric/tailscale fields).
- **Why:** Node identity now lives in ≥3 places (registry, failover profile,
  telemetry `_load_spark_nodes` with its sparkd/e/f special-case at
  `spark_telemetry_common.py:16-43`) and has already drifted (items #3, #4).
- **Expected value:** DRY win — delete duplicated node lists (negative code
  delta) and make drift structurally impossible.
- **Code delta:** net **negative** (removes duplicated lists).
- **Owner:** SYSADMIN (coordinator lands shared `tools/` changes).
- **First step:** Add the three IPs per node to the registry and regenerate the
  failover profile from it.

## 9. Watcher fsck-progress signal (fix the wait-vs-reboot blind spot)

- **What:** Expose the boot's blocking-unit status (e.g. tail of
  `journalctl -b -u systemd-fsck*`) on a tiny :2223 endpoint served by the
  emergency sshd, and have the watcher log it.
- **Why:** The v4 watcher sat at stage 2 for 14 h with no way to distinguish
  "fsck progressing" from "fsck hung" (`/tmp/ds4_bootwatch.log:259-390`) — the
  wait-vs-reboot decision was a coin flip.
- **Expected value:** One operational solution (observability) — turns the
  decision tree into a data-driven branch.
- **Code delta:** ~+30 lines (endpoint + watcher read).
- **Owner:** SYSADMIN.
- **First step:** Add a read-only status endpoint to the emergency sshd and
  sample it from the watcher loop.

## 10. Telemetry as a managed, self-healing service

- **What:** Wrap `spark_telemetry_collect.py` in a launchd unit with
  `--loop-interval` and per-node staleness alerting (it already has
  `--stale-ok-seconds`, `spark_telemetry_collect.py:31`).
- **Why:** Collection is a manual Mac-side loop; after a band returns, telemetry
  must be remembered-and-restarted, and a dark node is only noticed by looking.
- **Expected value:** One operational solution (MTTR) — band-health gaps are
  detected and collection resumes without a human.
- **Code delta:** ~+20 lines (a launchd plist + a stale-node notifier).
- **Owner:** SYSADMIN.
- **First step:** Install a launchd plist for the collector and verify it
  auto-restarts across a node flap.
