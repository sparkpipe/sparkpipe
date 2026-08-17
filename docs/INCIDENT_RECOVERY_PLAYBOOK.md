# GLM52-band + spark1 boot-stall incident — recovery playbook

Owner: SYSADMIN. Scope: physical/operational layer only (no model code).
Status: PLAYBOOK — DOCS ONLY, no host edits, no staging, no automation.
Last revised 2026-08-17 after the spark1 canary verdict.

---

## 1. Incident summary

**Nine** hosts are wedged mid-boot with the same signature:
- The GLM52 band — spark8, spark9, sparka, sparkb, sparkc, sparkd, sparke, sparkf
  — wedged since ~2026-08-16 16:25 (telemetry stop) / watcher start 17:02, after a
  GLM52 resident OOM + hard power-cycle.
- spark1 (always-on qwen27b band, spark0-3) — wedged after an OOM death
  ~2026-08-17 04:46 (mem_available 3691→0 MiB in ~11 s, mem_used 100%) + user power-cycle.

The wedge chain is now PROVEN, not inferred:
**OOM → hard power-cycle → dirty ROOT filesystem → root fsck at boot →
pam_nologin never clears → no login (stage 2).**

The fact that drove this revision: `sparkdata` (the model `runtime_root`, e.g.
`/home/{host}/sparkdata/glm52.tp8.fp8`) lives on the **root** filesystem
`/dev/nvme0n1p2` (fs_passno=1), NOT on a separate data mount. `df` on spark0 and
spark2 proves `/home/sparkN/sparkdata` resolves to root. So the dirty filesystem
that blocks boot is **root**, and any data-mount `nofail` fix can never prevent it.

Watcher timeline (`/tmp/ds4_bootwatch.log`): v2 start :1 (17:02:32), v3 :8
(17:14:48), v4 legend :259 (01:51:57; 0=nothing 1=tcp22 2=emer-banner 3=emer-login
4=full-login), all 8 stuck at stage 2 through :390 (07:04:20). spark1 shows the
same stage-2 signature independently.

---

## 2. What I verified from the Mac (read-only)

| Check | Result | Evidence |
|---|---|---|
| tailscale up | all 9 pong 1–2 ms | `tailscale ping`, live |
| mgmt :22 (main sshd) | REFUSED / accepts-no-banner | `nc` probes, live |
| mgmt :2222 (emergency sshd) | OPEN on all 9 | `nc` probes, live |
| :2222 banner | `SSH-2.0-OpenSSH_9.6p1 Ubuntu` | live banner grab |
| :2222 sparkemerg login | blocked by pam_nologin | `ssh -p2222 sparkemerg@…` → "System is booting up… pam_nologin(8)" |
| :2222 root login | **fails** — `Permission denied (publickey)` | `ssh -p2222 root@10.20.0.11`; tailscale :2222 times out |
| proxy route probe | every route fails "no SSH banner" | `ds4_spark_ssh_proxy.py --probe` |

Two findings that reset the doctrine:

1. **No in-band login for ANY user.** sparkemerg is pam_nologin-gated, and root is
   not a bypass: the emergency sshd is pubkey-only, no fleet root key exists on the
   Mac, so `ssh -p2222 root@10.20.0.11` → `Permission denied (publickey)` (and
   tailscale does not route 2222). pam_nologin's root exemption never gets a chance
   to apply because auth fails at the key stage.
2. **The 0-7 "permanent fix" never covered root.** Ground truth from spark0
   (authoritative) + spark2 (identical): the ONLY data mount is
   `/home/sparkN/extnvme` with `noauto,nofail,x-systemd.automount,…device-timeout=30,noatime`
   and **fs_passno=2** (not 0); root is `errors=remount-ro` **fs_passno=1**; sparkdata
   is not a mount at all (it is on root).

---

## 3. Root-cause model

**The blocker is root fsck on the dirty NVMe (nvme0n1p2).**

- A model process OOMs and exhausts ~124 GiB (spark1: mem_available → 0 in ~11 s).
- A hard power-cycle (smart plug / user) cuts power to a live, dirty root fs
  (the resident's writes to sparkdata land on root).
- At next boot systemd runs root fsck (fs_passno=1) on the dirty ext4; if it hangs
  or is slow, boot stalls before systemd-user-sessions, so `/run/nologin` stays.
- sshd (port 22 and the 2222 emergency listener) serves a banner but pam_nologin
  denies non-root logins and root has no key → stage 2 forever.

The data-mount `nofail` / `fs_passno=0` change cannot fix this: the dirty fs is root
(correctly never nofail'd), and `/home/sparkN/extnvme` is `noauto`+`automount` so it
is never fsck'd at boot and cannot block.

---

## 4. Recovery decision tree

Nine hosts wedged (8 GLM52 + spark1). Constraints: no in-band login (sparkemerg
nologin-gated, root keyless), no BMC/IPMI, user 10,000 miles away with smart plugs.

**Current wedge — the only remote lever is the plain plug-cycle lottery.** root fsck
must complete on its own (or the filesystem recover enough to boot), or the user
manually cycles power. No `fsck.mode=skip` is staged yet (GRUB config lives on the
wedged hosts' boot disks, unreachable), so a plug cycle either re-enters the
root-fsck stall or completes.

**User's order (unchanged, do not deviate):** (1) wait out the 22h timeout with NO
automated actions — the persistent watcher is stopped; (2) after the timeout the
user manually recovers **sparkf first**, then the rest in the order the user brings
them up. Each host is SYSADMIN's to drive through §5 as it appears.

**Once any host reaches full login (stage 4), in this order:**
1. **Stage the GRUB `fsck.mode=skip` entry AS DEFAULT — PRIMARY recovery lever (§5 step 2).**
   Default mode makes EVERY boot skip ALL fsck including root, so a dirty-root
   wedge can never block boot again — and it needs NO per-boot re-arming (the
   old one-shot grub-reboot scheme consumed itself after one boot). Manual fsck
   moves to maintenance windows. Do this on EVERY host at full login: spark0-7
   as they come up, spark1 + the 8 after manual recovery.
2. **Install the fleet root key (§6.1) — emergency access.** Root pubkey on both
   main sshd (22) and emergency sshd (2222) so a wedged host still admits root
   (pam_nologin exempts root; the key is what was missing).
3. **Land the OOM guardrails (§7) — preventive containment.** They do NOT stop the
   model OOMing (a 108G cap still kills the process); they stop the BOX from
   freezing so no hard power-cycle is ever needed. Verified live on spark2.
4. **Data-mount `fs_passno=0` hygiene (§5 step 3) — secondary.** Helps a dirty DATA
   disk only; it is NOT the wedge fix.

**Reseat / physical console** — last resort, not available at 10,000 miles.

---

## 5. Per-host return checklist (mechanical, in order)

Run per host `H`; tick all before moving to the next host.

1. **Full login (stage 4).** `ssh -o BatchMode=yes -o ConnectTimeout=8 $H true` exits 0.
2. **Stage the GRUB skip-fsck entry AS DEFAULT (PRIMARY recovery lever).**
   `scp tools/devcycle/stage_ds4_fastboot_grub.sh $H:/tmp/ && ssh $H 'sudo /tmp/stage_ds4_fastboot_grub.sh'`
   — default mode: installs the `ds4-fastboot` menuentry (`fsck.mode=skip
   fsck.repair=no`), sets `GRUB_DEFAULT=ds4-fastboot`, `update-grub`, and
   verifies `set default="ds4-fastboot"` in grub.cfg. EVERY boot then skips all
   fsck (including root) with NO re-arming needed. Verify with
   `ssh $H 'sudo /tmp/stage_ds4_fastboot_grub.sh --check'`. The old one-shot
   mode (`--one-shot` + grub-reboot) is still available but must be re-armed
   after every boot - do not use it for the permanent fix. Apply on every host,
   including spark0-7 as they come up.
3. **Data-mount hygiene (NOT the wedge fix).**
   `scp tools/devcycle/ds4_fastboot_fix.sh $H:/tmp/ && ssh $H 'sudo /tmp/ds4_fastboot_fix.sh'`
   — adds `nofail` + `fs_passno=0` to a dirty DATA disk. Marginal on this fleet
   (sparkdata is on root and extnvme is already nofail+automount), but apply for
   hygiene. Use the repo copy (the old /tmp copy had a corrupted Python write line).
4. **Root + data mounts.** `ssh $H 'df -h; mount | grep -iE "extnvme|sparkdata|nvme|raid"'`
   — root rw (NOT stuck ro from errors=remount-ro), sparkdata present (runtime_root,
   `fleet_registry.json:58`), extnvme present.
5. **If you used fsck.mode=skip:** `ssh $H 'sudo fsck -f /dev/nvme0n1p2'` (or
   `xfs_repair`/btrfs equivalent) NOW, while skip is in effect for this boot, and
   record the result. Never leave a skipped root dirty.
6. **Networking.** mgmt `10.20.0.x` up; fabric `10.10.100.x` up (`ip -4 addr`);
   `tailscale status` shows `tagged-devices` + `direct`. Flag sparkd/e/f still on
   `experiencenow-ai@`.
7. **Telemetry.** `ssh $H 'ls -la /tmp/ds4_telemetry'` shows a fresh
   `node_telemetry.csv` (`spark_telemetry_common.py:52`). Restart the collector on
   the Mac if it exited during the outage.
8. **Model/band restore.** `tools/devcycle/fleet_status.sh` / `tools/fleet_swap.sh status`
   — fleet state lives at `/tmp/sparkpipe_fleet_state.json` (authoritative on spark0,
   `tools/fleet_swap.sh:27-28`). Restore `glm52` if it was current; else confirm
   `free` and wait for the coordinator's reservation.
9. **Root-cause closeout.** `ssh $H 'journalctl -b -u systemd-fsck* -u local-fs.target'`
   — confirm the blocking unit is root fsck; attach to the incident record.

---

## 6. Recovery commands (copy-paste)

```bash
# stage probe, per host
ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" true            # expect exit 0 when recovered
nc -z -G2 -w2 10.20.0.X 2222                                   # emergency sshd reachable?
ssh -p2222 -o BatchMode=yes sparkemerg@10.20.0.X true         # nologin still blocking?

# PRIMARY recovery lever: pre-stage the GRUB one-shot (post-recovery)
scp tools/devcycle/stage_ds4_fastboot_grub.sh "$H":/tmp/
ssh "$H" 'sudo /tmp/stage_ds4_fastboot_grub.sh'

# hygiene: data-mount nofail+fs_passno=0 (NOT the wedge fix)
scp tools/devcycle/ds4_fastboot_fix.sh "$H":/tmp/
ssh "$H" 'sudo /tmp/ds4_fastboot_fix.sh'

# verify mounts + networking + telemetry
ssh "$H" 'df -h; mount | grep -iE "extnvme|sparkdata|nvme|raid"; ip -4 addr; ls -la /tmp/ds4_telemetry'
```

---

## 7. OOM guardrails — drop-in spec (PRIMARY preventive; DOCS ONLY, apply later with coordinator go)

Node RAM observed: **124610 MiB ≈ 121.7 GiB** (`mem_total_mib`). The wedge chain
starts with an OOM, so keep the box from OOMing and root never gets dirtied.

Sizing (per host):
```text
R = 124 GiB  (node RAM)
O =  16 GiB  (reserve: kernel + systemd + sshd + tailscaled + telemetry + page-cache floor)
B = R - O = 108 GiB  (budget for model services)
```
Per-service caps (sized from `tools/devcycle/fleet_registry.json` topology — TP8 vs
TP4 changes the per-host footprint; the residentd is the dominant consumer):
```text
sparkpipe_model_residentd   MemoryHigh=100G  MemoryMax=108G
vllm (if a separate unit)   MemoryHigh=8G    MemoryMax=16G   (subtract from residentd)
ds4_gateway                 MemoryHigh=1G    MemoryMax=2G
```

Drop-in unit (`/etc/systemd/system/sparkpipe_model_residentd.service.d/10-oom-guardrails.conf`):
```ini
[Service]
MemoryHigh=100G
MemoryMax=108G
OOMScoreAdjust=500
Restart=on-failure
RestartSec=5s
WatchdogSec=30s
StartLimitIntervalSec=60
StartLimitBurst=3
```
- `OOMScoreAdjust=500` makes the model services die FIRST under pressure; sshd and
  systemd stay at their (negative) systemd defaults so the management path survives.
- `MemoryHigh` throttles before `MemoryMax` hard-kills; `MemoryMax` is the backstop.
- `Restart=on-failure` + `WatchdogSec=30s` replaces the current
  `setsid -f bin/sparkpipe_model_residentd` / `pkill` pattern (`tools/fleet_swap.sh:63-78`)
  so a hung decode is killed + restarted, not a box freeze.
- Optional belt-and-suspenders: `earlyoom` (or systemd-oomd) to preemptively kill the
  biggest memory hog before the kernel OOM-killer can hard-hang the node.

---

## 8. Backlog (NOT part of the current recovery)

- **Relocate `sparkdata` off root onto the already-nofail `/home/sparkN/extnvme`**
  (or the cold RAID6). Structural change: move `runtime_root` + `pack_dir`, update
  `tools/devcycle/fleet_registry.json`, re-verify. This removes the dirty-root failure
  class entirely — model writes never touch root, so an OOM + hard power-cycle can
  no longer wedge boot. Larger change; do it after the 9 hosts recover.

---

Key file references (all read, no edits): `/tmp/ds4_bootwatch.log`,
`/Users/mac/.local/libexec/ds4-spark-management/spark_ssh_failover.json`,
`/Users/mac/.local/libexec/ds4-spark-management/ds4_spark_ssh_proxy.py`,
`~/.ssh/config`, `tools/devcycle/fleet_registry.json`, `tools/fleet_swap.sh`,
`tools/devcycle/ds4_fastboot_fix.sh`, `tools/devcycle/stage_ds4_fastboot_grub.sh`.
