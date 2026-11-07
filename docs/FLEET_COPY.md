# Copying files between machines

```sh
sparkcp SRC... DST
```

Each `SRC` and `DST` is a local path or `NODE:PATH`. `NODE` is `spark0`-`sparkf`,
`lenovo7`-`lenovo9` or `hub`. A relative remote path starts at that node's home.

```sh
sparkcp spark3:sparkdata/k3/packs lenovo7:models/
sparkcp hub:/mnt/cold-raid5/qwen/rank3.pack spark3:sparkdata/qwen/
sparkcp build/libstage.so spark3:bin/
```

Run it from the Mac, the hub or any node, and neither end has to be the machine
you are on. It is installed as `/usr/local/bin/sparkcp` on the hub and all 19
nodes and as `~/.local/bin/sparkcp` on the Mac; `tools/sparkcp.py` is the source.

Destinations follow `cp -r`: a directory that already exists receives the
source inside it. To bring a copy up to date, repeat the same command. Files with
the same size and mtime are skipped. A file that changed is sent whole.

## What it does for you

- **Path.** The receiver offers its addresses and the sender takes the first that
  connects: cisco (`10.10.201.x`), then the CRS804 (`10.10.100.x`), management,
  LAN and tailscale. The summary line names the path used, so a copy that fell back
  says so.
- **Integrity.** Every chunk carries a SHA-256 computed from what was read at the
  source and checked on arrival before it is written. A file reaches its final
  name only when all its chunks have arrived, so a failed or killed copy leaves
  no partial file behind and the previous version intact. The check covers the
  read and the network. It does not read back what the disk wrote.
- **Memory.** Files of 64 MiB or more use direct I/O and are preallocated, so a
  bulk copy does not fill the page cache on a Spark during a run.
- **Priority.** Both ends run at nice 10. On cisco0 TCP sits in the default queue,
  so RoCE keeps its share of the port and the priority classes stay ahead of it.

It keeps file modes, mtimes and inner symlinks. A symlink named on the command
line is followed. Owners are not kept: files belong to the user at the destination.

Data travels unencrypted between the two machines, authenticated by a one-time
token. Every path above is private to the office or the tailnet.

## Speed

Measured 2026-11-07 on an idle fleet, disk to disk, over cisco:

| Copy | Rate |
| --- | --- |
| hub NVMe to spark3, 23.6 GB | 5.0 GB/s |
| lenovo7 to lenovo8, 10.5 GB | 5.2 GB/s |
| spark3 to spark4, 23.6 GB | 2.3 GB/s |
| lenovo7 to lenovo8, 124 files, 255 GB | 1.8 GB/s |
| lenovo7 to lenovo8, 817,314 small files, 30 GB | 12,000 files/s |

The destination NVMe is the limit, not the network: 4 TCP streams reach 98 Gb/s,
and these drives sustain about 2 GB/s of writes once a transfer is large.
`rsync` over ssh measured 0.95 GB/s on the same pair, 1.5 GB/s with 4 in parallel.

## The Cisco path

The 16 Sparks are `10.10.201.<rank>`, the Lenovos `10.10.201.247`-`.249` and the
hub `10.10.201.250`, all on one subnet of cisco0.

On the Lenovos the address comes from `ds4-cisco-port.timer`, 30 s after boot. The
fleet's NetworkManager configuration leaves these ports unmanaged on purpose, so
this follows the rail 2 pattern: a timer outside the boot-critical path and a
bounded oneshot.

On the hub, `ssh`, `scp` and `rsync` by node name use cisco when the node's cisco
address answers, and otherwise the old path through sparkf at 10 GbE. The block
sits at the top of `~/.ssh/config` between `# BEGIN cisco primary path` markers.
Spark-to-Spark `ssh` by name stays on the CRS804, which is also 100G.

Install or update everything from the Mac:

```sh
tools/devcycle/fleet/install_sparkcp.sh
```
