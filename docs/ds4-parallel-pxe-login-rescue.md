# DS4 parallel PXE login rescue

This service is stage one of Spark recovery. It serves the same signed ARM64
UEFI boot chain to every PXE client and boots each Spark's installed root into
`multi-user.target`. Its generated initramfs prepends the deployment-supplied
recovery public key to the installed root account's `authorized_keys` before
switching root. It does not change firmware boot order or persistent systemd
policy.

The rescue kernel command line masks only:

- `ds4-switched-fabric.service`
- `ds4-direct-pair-fabric.service`

Management networking is acquired with DHCP on `enP7s7`. The installed root is
mounted from `/dev/nvme0n1p2`, matching the commissioned Spark layout. The PXE
server uses Microsoft's signed ARM64 shim followed by Canonical's signed GRUB,
so Secure Boot remains enabled.

## Deploy and start

Run from a clean, committed checkout:

```bash
python3 tools/devcycle/ds4_parallel_pxe_rescue.py deploy --server spark0
python3 tools/devcycle/ds4_parallel_pxe_rescue.py start --server spark0
python3 tools/devcycle/ds4_parallel_pxe_rescue.py status --server spark0 --require-active
python3 tools/devcycle/ds4_parallel_pxe_rescue.py probe --server spark0
```

The service is deliberately not enabled at server boot. Multiple Sparks may be
PXE-booted concurrently while it is active.

## Stage two

After rescued nodes reach login, repair and audit their persistent boot policy
with the fleet's normal configuration-management path. Keep fabric and other
nonessential distributed services out of the boot-critical path: they should
start only after the OS and emergency SSH are available.

Stop PXE service after recovery:

```bash
python3 tools/devcycle/ds4_parallel_pxe_rescue.py stop --server spark0
```

Every controller action writes a JSON receipt under the local temporary
directory. `probe` concurrently fetches and hashes `grub/grub.cfg`, the path
embedded in Canonical's network GRUB, through TFTP from
the surviving Sparks. `status --require-active` fails if the signed assets differ from the
manifest, DHCP/TFTP listeners are absent, the temporary firewall rule is absent,
or the service is unexpectedly enabled persistently.
