# Boot-Unblock Drop-Ins

These are bounded compatibility drop-ins for operating-system services that
SparkPipe does not own. The brickproof controller installs only the files named
by `BOOT_TIMEOUT_SOURCES`; do not copy this directory wholesale.

The SparkPipe-owned switched and direct-pair fabric services carry their own
60-second `TimeoutStartSec` and are launched by timers outside the login-critical
boot path. They have no duplicate drop-ins here.

Ceph is not made boot-safe with a timeout. The fleet policy stops and masks all
Ceph services and targets, removes their persistent startup links, and preserves
the data, configuration, devices, and packages for a future tested design.

Current compatibility bounds:

| Unit | TimeoutStartSec |
| --- | ---: |
| `rbdmap.service` | 120 |
| `nvmf-autoconnect.service` | 120 |
| `open-iscsi.service` | 120 |
| `pollinate.service` | 30 |
