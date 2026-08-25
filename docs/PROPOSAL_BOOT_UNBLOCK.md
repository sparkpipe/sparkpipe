# Superseded Boot-Unblock Proposal

Owner: SYSADMIN. Status: SUPERSEDED by the fleet brickproof policy in
`docs/INCIDENT_RECOVERY_PLAYBOOK.md`.

This document preserves the original incident diagnosis. Its proposed Ceph and
fabric timeout drop-ins must not be deployed. The landed policy instead launches
the bounded SparkPipe fabric services from timers outside the login-critical
path and fully quarantines Ceph startup by stopping and masking every service and
target. The remote-storage compatibility drop-ins remain managed centrally by
`tools/devcycle/ds4_spark_brickproof.py`.

## 1. Root cause (diagnosed live on spark2 via root@2222, 2026-08-17)

The second (non-fsck) boot wedge is the FABRIC config chain, not fsck:

1. `ds4-switched-fabric.service` (ExecStart=/usr/local/sbin/ds4-switched-fabric-apply)
   HANGS when the 200G fabric is down. Its interfaces are all DOWN when the
   cluster is down: enp1s0f0np0, enp1s0f1np1, enP2p1s0f0np0, enP2p1s0f1np1 (0x
   `ip -br addr`). It sits `Before=network-online.target`.
2. `ds4-direct-pair-fabric.service` is `After=ds4-switched-fabric.service` AND
   `Before=network-online.target`, so it also blocks network-online.
3. `network-online.target` has `After=ds4-switched-fabric.service
   ds4-direct-pair-fabric.service`, so it never gets reached.
4. The cascade: network-online -> remote-fs-pre.target -> remote-fs.target ->
   ceph.target + ceph mon/osd/crash + rbdmap -> systemd-user-sessions.service
   (`After=remote-fs.target`) -> /run/nologin stays.
5. sshd is ALSO indirectly blocked: ssh.service `After=pollinate.service`, and
   `pollinate.service` is `Before=pollinate` on network-online.target (i.e.
   pollinate waits for network-online). So port 22 accepts TCP but serves no
   banner (ssh.service 'start waiting').

ceph units hang because cluster quorum is impossible with 9 hosts wedged + the
fabric down. This is a PRE-EXISTING issue (unrelated to the fsck fix).

## 2. Waiting-units inventory (45 jobs, all 'start waiting' except ds4-switched-fabric='running')

fabric: ds4-switched-fabric(running/hung), ds4-direct-pair-fabric.
targets: multi-user, network-online, remote-fs-pre, remote-fs, timers, getty.
ceph: ceph.target, ceph-<fsid>.target, ceph-<fsid>@mon, @osd.4, @crash.
remote storage: rbdmap, nvmf-autoconnect, open-iscsi.
ssh/entropy: ssh, pollinate.
misc boot: systemd-pcrphase, setvtrgb, systemd-user-sessions, update-utmp-runlevel,
systemd-update-utmp, blk-availability, cups-browsed, docker, podman, samba-ad-dc,
ubuntu-advantage, apport, fwupd-refresh, cron, getty@tty1, serial-getty@ttyS0,
user@1000, nvidia-dgx-telemetry, nvidia-spark-run-apt-upgrade-once, centaur-sparkring-agent,
sparkpipe_model_residentd, sparkpipe-fsck-health, plymouth-quit, rpc-statd-notify,
update-notifier-motd/download timers.

## 3. Historical proposal (do not deploy)

Each is a drop-in under /etc/systemd/system/<unit>.d/10-boot-timeout.conf:

```ini
# ds4-switched-fabric.service.d/10-boot-timeout.conf + ds4-direct-pair-fabric.service.d/10-boot-timeout.conf
[Service]
TimeoutStartSec=120
```

```ini
# ceph-<fsid>@.service.d/10-boot-timeout.conf  (the cephadm template)
[Service]
TimeoutStartSec=180
```

```ini
# rbdmap.service.d, nvmf-autoconnect.service.d, open-iscsi.service.d (one each)
[Service]
TimeoutStartSec=120
```

```ini
# pollinate.service.d/10-boot-timeout.conf  (unblocks sshd indirectly)
[Service]
TimeoutStartSec=30
```

Additional (data-mount-hygiene mirror for remote/rbd mounts): any rbd/ceph/remote
fstab entries get `nofail,x-systemd.device-timeout=10s` so a missing remote device
can never hold remote-fs-pre.target open.

## 4. sshd note

sshd does NOT sit directly behind network-online.target (its After= is
`network.target` + `pollinate.service spark-firewall.service auditd.service`). The
indirect block is via pollinate -> network-online -> fabric. The pollinate
TimeoutStartSec=30 drop-in above removes that indirect block without touching sshd's
After= lines.

## 5. Result after the manual unblock (spark2)

Commands run (root@2222): `systemctl stop ds4-switched-fabric.service
ds4-direct-pair-fabric.service`; `systemctl stop ceph.target ceph-<fsid>.target`;
`systemctl stop ceph-<fsid>@mon.aitopatom-931a.service ceph-<fsid>@osd.4.service
ceph-<fsid>@crash.aitopatom-931a.service`; `systemctl stop rbdmap.service
nvmf-autoconnect.service open-iscsi.service`.
Outcome: /run/nologin removed, full login (root :22 exit 0), 'No jobs running',
`systemctl is-system-running`=degraded (ceph down, expected), sparkpipe-fsck-health
ran this boot and wrote last.json classification HEALTHY.
