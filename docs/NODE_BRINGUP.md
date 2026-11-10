# New node bring-up

This runbook takes a Spark from its factory DGX OS image to the point where
[NODE_BASELINE.md](NODE_BASELINE.md) `apply` can take over. It exists because
spark10 (2026-11-10, the second fleet's node 0) lost about two hours and a power
cycle to its first `apt full-upgrade` (see [What went wrong on spark10](#what-went-wrong-on-spark10)).

`sparkN` is the node name and user, and `A` is its rail-2 host number (step 3).

## Fleets and staging

- A node is named `spark` followed by its node number in hex. The first fleet
  is `spark0`-`sparkf` (0-15); the second fleet is `spark10`-`spark1f`
  (16-31), so `spark10` is its node 0; the third fleet starts at `spark20`.
- Only the first fleet runs Ceph.
- Until the new office (new switches, more power), nodes of later fleets are
  staged: they go through this runbook and the network-independent part of the
  baseline, then wait. Their fabric addresses, ranks in service and switch
  ports are set at the new office.

## Rules

1. **Headless before the first upgrade.** Purge the browsers and boot
   `multi-user.target` (step 4) while dpkg is still clean. The Ubuntu archive
   replaces the firefox and thunderbird debs with snap transition packages.
   Their pre-install script downloads the snap, and snapd does not use apt's
   proxy.
2. **dpkg never runs inside an SSH session.** Package changes run as transient
   units (`systemd-run --unit=...`). A dropped session then cannot stop dpkg,
   and the unit's journal keeps the record.
3. **Open an SSH master before anything that can replace `openssh-server`.**
   While `openssh-server` is unpacked but not configured, sshd refuses new
   connections. Open sessions survive that, and so do new sessions over an open
   master.
4. **Only the owner types passwords.** The owner does the key install and sudo
   (step 2) and Wi-Fi (step 5).

## 1. Owner: first boot and cabling

- In the first-boot setup, create user `sparkN` (uid 1000) and set the hostname
  to `sparkN`. spark10 came up as `cube-4545` because the hostname was left at
  its default.
- Cable port 0 to `cisco0`. A node with no RJ45 on the LAN can still be brought
  up over rail 2.
- The first-boot Wi-Fi step may not save a profile: spark10 had none. Wi-Fi is
  step 5.

## 2. Find the node and get in

On `cisco0` (access: `sparkpipe-coord/network/cisco0.md`),
`show lldp neighbors` gives the new port, the factory hostname and the MAC of
port 0.

The node has no IPv4 address on the fabric, only IPv6 link-local addresses.
The hub's rail-2 port has no IPv6, so reach the node from a fleet Spark's
rail-2 netdev and pick its addresses out by MAC:

```bash
ssh spark0 'ping -6 -c 2 ff02::1%enP2p1s0f0np0 >/dev/null; ip -6 neigh show dev enP2p1s0f0np0 | grep -i <MAC prefix>'
```

- Both PCIe functions of port 0 answer on a factory node, each with its own
  address.
- NetworkManager derives each address from the active connection profile, so
  an address can change after a reboot. Look it up again by MAC.
- `ssh -J spark0` to a link-local address fails with `Stdio forwarding request
  failed`. Nest the command instead: `ssh spark0 'ssh sparkN@fe80::...%enP2p1s0f0np0 ...'`.

The owner then installs the fleet keys and passwordless sudo, typing the
first-boot password at each prompt. The first command copies spark0's
`authorized_keys` (the workstation, the hub and every peer). The second creates
the fleet's `codexadmin` group and checks the sudoers file.

```bash
ssh -t spark0 'ssh -o StrictHostKeyChecking=accept-new sparkN@<link-local>%enP2p1s0f0np0 "mkdir -p -m 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" < ~/.ssh/authorized_keys'
```

```bash
ssh -t spark0 'ssh -t sparkN@<link-local>%enP2p1s0f0np0 "sudo groupadd -f codexadmin && sudo usermod -aG codexadmin sparkN && echo \"%codexadmin ALL=(ALL) NOPASSWD: ALL\" | sudo tee /etc/sudoers.d/90-codexadmin >/dev/null && sudo chmod 440 /etc/sudoers.d/90-codexadmin && sudo visudo -c"'
```

## 3. Rail-2 address and the SSH master

Give port 0 a runtime rail-2 address so the hub reaches the node directly. The
address is lost at reboot. The bring-up address is the node number:
`spark0`-`sparkf` hold `10.10.201.0`-`.15`, `spark10` takes `10.10.201.16`,
`spark1f` takes `.31`.

```bash
ssh spark0 'ssh sparkN@<link-local>%enP2p1s0f0np0 "sudo nmcli dev set enP2p1s0f0np0 managed no && sudo ip addr add 10.10.201.A/24 dev enP2p1s0f0np0 && sudo ip link set enP2p1s0f0np0 mtu 9000"'
```

Open the master from the workstation, then send every later command over it:

```bash
ssh -o AddressFamily=inet -o ServerAliveInterval=20 -o ControlMaster=yes -o ControlPersist=yes -o ControlPath=~/.ssh/cm-sparkN-bringup -f -N -J rtx5090 sparkN@10.10.201.A
```

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'uname -r'
```

## 4. Headless

Check the removal plan first. It must only remove the browsers, their locale
packages and the desktop metapackages (`ubuntu-desktop`,
`ubuntu-desktop-minimal`, `nvidia-system-station*`), and install nothing:

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'apt-get -s purge "^firefox" "^thunderbird" ubuntu-desktop ubuntu-desktop-minimal "^nvidia-system-station" | grep -E "^(Inst|Remv|Purg)"'
```

Then purge, keep what the metapackages held, and switch the boot target:

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'sudo systemd-run --wait --collect --unit=node-headless --setenv=DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get -y purge "^firefox" "^thunderbird" ubuntu-desktop ubuntu-desktop-minimal "^nvidia-system-station"'
```

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A "sudo apt-mark manual \$(apt-get -s autoremove | awk '/^Remv/{print \$2}') >/dev/null; apt-get -s autoremove | grep -c ^Remv; sudo systemctl set-default multi-user.target; sudo systemctl mask plymouth-quit-wait.service nvidia-spark-run-apt-upgrade-once.service"
```

- The metapackages only list dependencies and ship no files, but `firefox` and
  `thunderbird` cannot go without them. Marking their dependencies manual keeps
  GNOME, NetworkManager and the rest installed, so a later `autoremove` removes
  nothing. The count printed must be 0.
- `multi-user.target` and the `plymouth-quit-wait` mask are baseline items: the
  factory splash wait never ends, so units ordered after `multi-user.target`
  never start.
- `nvidia-spark-run-apt-upgrade-once` runs apt at boot. It is a baseline mask,
  and the bring-up decides when apt runs.
- The fleet 1 nodes still carry the snap transition debs, both snaps and the
  metapackages. They got the snaps while they had internet.

## 5. Internet

- **Wi-Fi (owner).** The node joins the fleet's primary Wi-Fi network with the
  fleet profile settings. The command is in the coordination store
  (`sparkpipe-coord/node-files/WIFI_FIRST_CONNECT.md`), because it names the
  owner's networks. It asks for the network password. The full uplink failover
  set comes with the private node files ([NODE_BASELINE.md](NODE_BASELINE.md#replacing-or-adding-a-node), step 4).
- **No Wi-Fi or LAN yet.** Give apt a SOCKS proxy through the workstation with
  a reverse dynamic forward, and keep it running for the whole upgrade:

  ```bash
  ssh -o AddressFamily=inet -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -J rtx5090 -N -R 127.0.0.1:1080 sparkN@10.10.201.A
  ```

  `AddressFamily=inet` is required: without it the workstation tries the
  mirrors' IPv6 addresses first and `ports.ubuntu.com` times out. apt takes the
  proxy per command, and nothing is written to its configuration:
  `-o Acquire::http::Proxy=socks5h://127.0.0.1:1080 -o Acquire::https::Proxy=socks5h://127.0.0.1:1080`.

## 6. Upgrade

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'P="-o Acquire::http::Proxy=socks5h://127.0.0.1:1080 -o Acquire::https::Proxy=socks5h://127.0.0.1:1080"; sudo apt-get $P update >/dev/null && apt-get -s $P full-upgrade | grep -E "upgraded,|^Inst .*snap"'
```

Only `snapd` itself may match `snap`. A `1snap` version is a snap transition
package: purge it as in step 4 before upgrading. Then run the upgrade:

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'P="-o Acquire::http::Proxy=socks5h://127.0.0.1:1080 -o Acquire::https::Proxy=socks5h://127.0.0.1:1080"; sudo systemd-run --wait --unit=node-upgrade --setenv=DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get -y $P -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold full-upgrade; sudo dpkg --audit | wc -l'
```

- Drop the `P` options when the node has Wi-Fi.
- If the session drops, the unit keeps going. Wait for it with
  `while systemctl is-active -q node-upgrade; do sleep 20; done`.
- A factory image from 2026-11 takes about 650-690 packages, including the 7.0
  nvidia HWE kernel (7.0.0-1025 on 2026-11-10) and driver 580.178.04. That moves
  the node off the 6.17 nvidia kernel, which is end of life and will not get
  the CVE-2026-98432 fix.

If dpkg failed, read the unit's journal and finish the half-installed packages
before anything else, then run `apt-get -f install` and the upgrade again, each
as a transient unit:

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'sudo systemd-run --wait --unit=node-configure --setenv=DEBIAN_FRONTEND=noninteractive /usr/bin/dpkg --force-confdef --force-confold --configure -a'
```

`nvidia-persistenced` fails until the reboot, because the new driver libraries
do not match the loaded kernel module.

Reboot with `sudo systemctl reboot`. Then find the node again (step 2), re-add
the rail-2 address, reopen the master, and check:

```bash
ssh -o ControlPath=~/.ssh/cm-sparkN-bringup sparkN@10.10.201.A 'uname -r; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; systemctl get-default; systemctl --failed --no-legend; sudo dpkg --audit | wc -l'
```

## 7. Hand-off to the baseline

- **First fleet:** continue with
  [NODE_BASELINE.md](NODE_BASELINE.md#replacing-or-adding-a-node) from step 2
  (firmware).
- **Later fleets (staged):** firmware (baseline step 2) and the
  network-independent baseline items: boot masks, watchdog and recovery
  sysctls, earlyoom and the user slice, the dirty-page cap, emergency SSH and the
  recovery key, the firewall, the DMA guard, the tool packages and the node
  layout roots. No Ceph. `ds4_spark_brickproof.py` accepts only `spark0`-`sparkf`
  today, so staged nodes are not yet applied or audited by it.
- A new node has no predecessor, so baseline step 3 (identity) does not apply:
  the node keeps its own host keys.

## Vendor notes

| Vendor and board | Seen on | BIOS | Notes |
|---|---|---|---|
| H3C LinSeer MegaCube | spark10, 2026-11-10 | `5.36_CUBE_300` | Factory image: Ubuntu 24.04.4, kernel 6.17.0-1022, driver 580.159.03, CX-7 28.45.4028, Wi-Fi MediaTek MT7925 (`mt7925e`). firefox 148 and thunderbird 140 are Mozilla team (`~mt`) builds with no Mozilla source configured, so the archive's epoch-2 snap transition packages replace them. The vendor firmware path is not checked yet: do not stage FE or Gigabyte capsules on it. |

## What went wrong on spark10

1. The node had no Wi-Fi. apt ran through the SOCKS tunnel (step 5) as a
   transient unit, but before steps 3 and 4.
2. The downloads (3.3 GB) took 8.5 minutes. Then the firefox pre-install
   script waited 47 minutes on a snap download that could not leave the node
   and failed, and thunderbird did the same for another 47 minutes.
3. apt had handed dpkg one batch of about 690 packages. dpkg unpacked the rest
   of the batch, returned an error, and apt stopped before the configure pass.
   About 350 packages stayed unpacked and unconfigured, `openssh-server` among
   them.
4. sshd refused every new connection. The node still answered ping and LLDP,
   and the two open sessions were bound to fixed commands, so nobody could get
   in to run `dpkg --configure -a`.
5. The owner power-cycled the node. It booted the old kernel with sshd up, and
   the bring-up resumed with steps 3 and 4. The purge then needed the
   metapackages removed as well, which step 4 now does in one pass.
