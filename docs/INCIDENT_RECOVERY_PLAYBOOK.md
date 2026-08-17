# GLM52-band boot-stall incident — recovery playbook

Owner: SYSADMIN. Scope: physical/operational layer only (no model code).
Status: PROPOSAL/PLAYBOOK — no commits, no pushes. All live checks below were
run read-only from mac-studio on 2026-08-17 (~07:00 local).

---

## 1. Incident summary

The 8-host GLM52 band — spark8, spark9, sparka, sparkb, sparkc, sparkd, sparke,
sparkf — has been stuck mid-boot since an OOM in the GLM52 resident decode
followed by a hard power-cycle. The band is the "big" GLM52 tier, TP8 over the 8
hosts (`tools/devcycle/fleet_registry.json:44-66`, runtime_root
`/home/{host}/sparkdata/glm52.tp8.fp8`).

Watcher timeline (`/tmp/ds4_bootwatch.log`):

- :1 — v2 watcher started 2026-08-16 17:02:32, watching the 8 hosts.
- :8 — v3 "fastboot fix" watcher started 17:14:48.
- :259 — v4 watcher 2026-08-17 01:51:57, stage legend
  `0=nothing 1=tcp22 2=emer-banner 3=emer-login 4=full-login`.
- :390 (last line) — 2026-08-17 07:04:20, **all 8 still at stage 2**.

Total: ~14 h with zero stage movement. Stage 2 = the emergency sshd on 2222
serves its banner but login never succeeds.

---

## 2. What I verified from the Mac (read-only)

| Check | Result | Evidence |
|---|---|---|
| tailscale up | all 8 pong in 1–2 ms | `tailscale ping` on 100.96.224.2 … 100.88.217.33, live |
| mgmt :22 (main sshd) | **REFUSED** on all 8 (5/5 samples on spark8) | `nc -z -G1 10.20.0.18 22`, live |
| mgmt :2222 (emergency sshd) | **OPEN** on all 8 | `nc -z 10.20.0.1{8,9}…:2222`, live |
| :2222 banner | `SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13.18` | live banner grab |
| :2222 login | **blocked by pam_nologin** | `ssh -p2222 sparkemerg@10.20.0.18 true` → `"System is booting up. Unprivileged users are not permitted to log in yet. … pam_nologin(8)"` |
| proxy route probe | every route fails "endpoint did not provide an SSH banner" | `ds4_spark_ssh_proxy.py --probe` on spark8/spark9 |

Two findings that change the recovery plan:

1. **The emergency sshd is NOT a recovery channel right now.** It answers on
   2222 but sits behind the same pam_nologin gate as the main sshd, so
   `sparkemerg` cannot log in. That is exactly why the watcher is stuck at
   stage 2 (banner) and never reaches stage 3 (login). There is **no in-band
   login path** to any of the 8 hosts.

2. **sparkd/e/f are only partially wired into the failover tooling.**
   - The proxy profile ends at sparkc (`spark_ssh_failover.json:119-123`); it
     has no `sparkd/e/f` nodes, so `--node sparkd` fails "unknown node".
   - ssh config gives sparkd/e/f only direct mgmt + `ProxyJump spark0` for
     fabric (`~/.ssh/config:788-823` for sparkd, :862-897 for sparkf); no
     `-emergency` / `-wifi` entries, no `-200g` proxy route.
   - tailscale shows sparkd/e/f under `experiencenow-ai@`, not the fleet's
     `tagged-devices` (`tailscale status`, live).
   Despite this, their emergency sshd IS reachable on mgmt :2222 (verified OPEN),
   so recovery can still target them by mgmt IP + port 2222.

---

## 3. Root-cause model

Consistent with fsck-on-dirty-NVMe being the blocker:

- pam_nologin "System is booting up" means systemd has not finished boot —
  `/run/nologin` is removed only after `basic.target`; a unit stuck earlier
  (e.g. `systemd-fsck@<data-nvme>.service`) keeps it in place.
- tailscaled is up (network early), main sshd is down/refusing, emergency sshd
  is up but nologin-gated — all consistent with boot stalled at the data mount.
- The dirty NVMe is the GLM52 data volume `/home/{host}/sparkdata/glm52.tp8.fp8`
  (the OOM'd resident's runtime root, `fleet_registry.json:58`).
- 14 h with no stage change says fsck is **hung**, not "just slow".

Caveat: without console/BMC this is inference. The first post-recovery action is
to pull `journalctl -b` and confirm the exact blocking unit before declaring
root cause closed.

---

## 4. Recovery decision tree

Constraints: no in-band login (emergency sshd is nologin-gated, §2), no
BMC/IPMI, user 10,000 miles away with smart plugs. The one-shot GRUB
`fsck.mode=skip` entry **cannot be staged yet** — GRUB config lives on the
hosts' boot disks, unreachable while wedged — so it becomes available only
after first login. Recovery is therefore two phases.

**Phase 0 — break the wedge (now):**
1. **WAIT-FOR-FSCK** — exhausted (14 h, no stage movement, `bootwatch:1,390`);
   keep only as a background poll.
2. **PLAIN SMART-PLUG CYCLE on ONE host (spark8 first).** A plain cycle is a
   lottery until the one-shot exists: it either re-enters the fsck stall or the
   host completes boot (if the filesystem has since recovered, or plug timing
   changes the boot path). It costs nothing and might just work; a single host
   keeps blast radius to one node and yields a clean reference recovery.
   - spark8 returns (stage 3/4) → run §5 in full, then Phase 1.
   - spark8 does NOT return after ~2 cycles → do not blind-cycle all 8; hold
     and escalate (reseat / on-site is the only remaining lever).
3. **RESEAT / physical console** — last resort; not available at 10,000 miles.

**Phase 1 — make the next wedge cheap (immediately after ANY host returns):**
- Land the permanent fix (nofail + fs_passno=0, `tools/devcycle/ds4_fastboot_fix.sh`) on
  the returned host first — from then on every future power-cycle boots fast
  forever, so a dirty data NVMe can never re-wedge that host.
- Pre-stage the one-shot GRUB fallback
  (`tools/devcycle/stage_ds4_fastboot_grub.sh`: a `ds4-fastboot` menuentry +
  `grub-reboot` flag) on the returned host, then fan it out to all 8. After
  that a future wedge is breakable by smart plug alone: `sudo grub-reboot
  ds4-fastboot` while healthy, then a plug cycle.

**Mechanics:** the coordinator owns the persistent watcher (bash-4, log
`/tmp/ds4_bootwatch.log`) and will notify SYSADMIN when a host reaches stage
3/4; SYSADMIN then drives §5 for that host through the coordinator.

---

## 5. Per-host return checklist (mechanical, in order)

Run per host `H`; tick all before moving to the next host.

1. **Full login (stage 4).** `ssh -o BatchMode=yes -o ConnectTimeout=8 $H true`
   exits 0 (main sshd up, nologin cleared).
2. **Land the boot-time fsck fix.** `scp tools/devcycle/ds4_fastboot_fix.sh $H:/tmp/ &&
   ssh $H 'sudo /tmp/ds4_fastboot_fix.sh'` — requires passwordless sudo
   (`tools/devcycle/ds4_fastboot_fix.sh:6`). Verify: every data mount (extnvme/sparkdata/
   kv/nvme/raid, and /home) now has `nofail` + `x-systemd.device-timeout=10s`
   and `fs_passno=0` in `/etc/fstab` (`tools/devcycle/ds4_fastboot_fix.sh:26-35`);
   root/boot/var/usr/tmp/var-log untouched (:15). Confirm `systemctl daemon-reload`. Use the repo copy — the old `/tmp/ds4_fastboot_fix.sh` had a corrupted Python write line (literal newlines) and must not be run.
2b. **Pre-stage the GRUB one-shot fallback.** `scp tools/devcycle/stage_ds4_fastboot_grub.sh $H:/tmp/ && ssh $H 'sudo /tmp/stage_ds4_fastboot_grub.sh'` — installs a `ds4-fastboot` menuentry (fsck.mode=skip fsck.repair=no) + `update-grub`, then verifies it landed in `/boot/grub/grub.cfg`. Do this on every host once any host is up (Phase 1); until it exists, a wedged host can only be broken by plain power-cycle lottery.
3. **NVMe mounts.** `ssh $H 'df -h; mount | grep -iE "extnvme|sparkdata|nvme|raid"'`
   — every GLM52 data volume present and rw, and
   `ls -d /home/$H/sparkdata/glm52.tp8.fp8` exists (runtime_root,
   `fleet_registry.json:58`).
4. **If you used fsck.mode=skip:** `ssh $H 'sudo fsck -f /dev/<nvme-data>` (or
   `xfs_repair`/btrfs equivalent) NOW, while mounted-skip is still in effect
   for this boot, and record the result. Never leave a skipped fsck dirty.
5. **Networking.** mgmt `10.20.0.x` up; fabric `10.10.100.x` up
   (`ip -4 addr`); `tailscale status` shows the host `tagged-devices` and
   `direct`. Flag any host still on `experiencenow-ai@` (sparkd/e/f anomaly).
6. **Telemetry.** `ssh $H 'ls -la /tmp/ds4_telemetry'` shows a fresh
   `node_telemetry.csv` (`spark_telemetry_common.py:52`). On the Mac, restart
   the collector if it exited during the outage:
   `python3 ~/.local/share/ds4_telemetry/spark_telemetry_collect.py --loop-interval 60`
   and confirm the dashboard sees all 16 nodes.
7. **Model/band restore.** `tools/devcycle/fleet_status.sh` (or
   `tools/fleet_swap.sh status`) — fleet state lives at
   `/tmp/sparkpipe_fleet_state.json` (authoritative on spark0,
   `tools/fleet_swap.sh:27-28`). If GLM52 was current, restore with
   `tools/fleet_swap.sh glm52`; else confirm the band reports "free" and wait
   for the coordinator's reservation.
8. **Root-cause closeout.** `ssh $H 'journalctl -b -u systemd-fsck* -u local-fs.target'`
   — confirm the blocking unit matches the fsck hypothesis; attach to the
   incident record.

---

## 6. Recovery commands (copy-paste)

```bash
# stage probe, per host
ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" true            # expect exit 0 when recovered
nc -z -G2 -w2 10.20.0.X 2222                                   # emergency sshd reachable?
ssh -p2222 -o BatchMode=yes sparkemerg@10.20.0.X true         # nologin still blocking?

# land the permanent fix (post-recovery)
scp tools/devcycle/ds4_fastboot_fix.sh "$H":/tmp/
ssh "$H" 'sudo /tmp/ds4_fastboot_fix.sh'

# pre-stage the GRUB one-shot fallback (Phase 1, post-recovery)
scp tools/devcycle/stage_ds4_fastboot_grub.sh "$H":/tmp/
ssh "$H" 'sudo /tmp/stage_ds4_fastboot_grub.sh'

# verify mounts + networking + telemetry
ssh "$H" 'df -h; mount | grep -iE "extnvme|sparkdata|nvme|raid"; ip -4 addr; ls -la /tmp/ds4_telemetry'
```

Key file references (all read, no edits): `/tmp/ds4_bootwatch.log`,
`/Users/mac/.local/libexec/ds4-spark-management/spark_ssh_failover.json`,
`/Users/mac/.local/libexec/ds4-spark-management/ds4_spark_ssh_proxy.py`,
`~/.ssh/config`, `tools/devcycle/fleet_registry.json`,
`tools/fleet_swap.sh`, `tools/devcycle/ds4_fastboot_fix.sh`,
`tools/devcycle/stage_ds4_fastboot_grub.sh`.
