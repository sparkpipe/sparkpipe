# boot-unblock drop-ins (fail-soft instead of hang-on-boot)

Deploy: copy each *.conf to /etc/systemd/system/<unit>.d/10-boot-timeout.conf,
then 'systemctl daemon-reload'. Affects NEXT boot only.

Unit -> TimeoutStartSec (all Type=oneshot except ceph = forking; TimeoutStartSec
applies to both; no Type=simple so no 'timeout'-wrapped ExecStart is needed):
  ds4-switched-fabric.service           120
  ds4-direct-pair-fabric.service        120
  ceph-b52b3459-...@.service (template) 180
  rbdmap.service                        120
  nvmf-autoconnect.service              120
  open-iscsi.service                    120
  pollinate.service                     30

Root cause (spark2 cycle test): ds4-switched-fabric hangs when the 200G fabric is
down, blocks network-online.target -> remote-fs -> ceph -> systemd-user-sessions
(nologin). These drop-ins make each hang fail soft so boot always completes.
