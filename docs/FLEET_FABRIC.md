# Fleet fabric

Since 2026-11-06 the sixteen Sparks have no direct pair links. Each Spark has
two 100 Gb/s ports, each on its own switch and its own PCIe Gen5 x4 link:
two switched rails that every rank reaches directly. This document is the
layout, the boot configuration that keeps it, and the usage rules for anyone
writing collectives or other RDMA traffic against it.

## Layout

| | Rail 1 | Rail 2 |
|---|---|---|
| Switch | MikroTik CRS804-4DDQ, RouterOS 7.24.4 | Cisco Nexus N9K-C9336C-FX2 `cisco0`, NX-OS 9.3(1) |
| Spark port and PCIe function | port 1 via link A, `0000:01:00.1` | port 0 via link B, `0002:01:00.0` |
| Netdev / RDMA device | `enp1s0f1np1` / `rocep1s0f1` | `enP2p1s0f0np0` / `roceP2p1s0f0` |
| Address | `10.10.100.(10+rank)/24` | `10.10.201.rank/24` |
| RoCE GID | index 3, RoCE v2, IPv4 | index 3, RoCE v2, IPv4 |
| Bulk class | DSCP 26 (tclass 106), lossless TC3, no ECN | DSCP 26 (tclass 106), lossless cos 3, ECN marking |
| Fast class | none | DSCP 50 (tclass 202), lossless cos 6, strict priority |
| CNP | DSCP 48, strict | DSCP 48, strict |

- The other two PCIe functions (port 0 through link A, `enp1s0f0np0`, and
  port 1 through link B, `enP2p1s0f1np1`) stay down. Each port has exactly one
  active function, and each rail has its own PCIe link.
- MTU is 9000 for IP and 4096 for RoCE. The largest RoCE payload rate on one
  100G port is about 98 Gb/s.
- `cisco0` ports: odd ports `Eth1/1`-`Eth1/31` are spark0-sparkf in rank order;
  `Eth1/2`, `Eth1/4` and `Eth1/6` are lenovo7, lenovo8 and lenovo9
  (`10.40.0.x/24`, runtime only); `Eth1/8` is the hub. `mgmt0` is reached
  through lenovo9. Map ports with LLDP, not port comments.
- The hub `rtx5090` has `enp175s0f1np1` (ConnectX-5 Ex, PCIe Gen4 x4, about
  63 Gb/s raw ceiling) on `cisco0` at `10.10.201.250/24`. Drafter taps still use
  the 10 GbE `eno1` link to sparkf (`10.10.250.2`) until the drafters move.

## Boot configuration

| Item | Where |
|---|---|
| Rail 1 address and rings | `ds4-switched-fabric.timer` (`tools/devcycle/fleet/ds4_switched_fabric_apply.sh`) |
| Rail 1 QoS | `sparkpipe-roce-qos.service` (`tools/fabric/sparkpipe_roce_qos.sh`): trust DSCP, PFC 3, `prio2buffer 0,0,0,1` |
| Rail 2 everything | `ds4-rail2-fabric.timer`, 30 s after boot (`tools/devcycle/fleet/ds4_rail2_fabric_apply.sh`): idle function down, address, MTU 9000, rings 1024, trust DSCP, PFC 3 and 6, one shared receive buffer, 802.3x pause off, spark2's forced link |
| weightd rail 2 device | fleet-agent drop-in `~/.config/systemd/user/fleet-agent.service.d/40-rail2.conf`: `SPARK_MESH_PAIR_INTERFACE=roceP2p1s0f0` |
| Hub | `/etc/netplan/61-hub-100g.yaml` (address) and `sparkpipe-hub-rail2.service` (`tools/hub/system/`) |
| `cisco0` | startup config, policies `ROCE_CLASSIFY2`, `ROCE_NQ2`, `ROCE_OUT2` |

`ds4_spark_brickproof.py` installs and audits the Spark side. Its audit fails a
node whose rail 2 address, idle function, trust state, PFC priorities or
buffer map drift. The pair-fabric units are retired paths: apply moves them
to `/var/backups/ds4-fleet` and audit reports any that remain.

## Usage rules

1. **Mark traffic by class.** Bulk RDMA uses traffic class 106 (DSCP 26) on
   both rails. Latency-critical small messages may use traffic class 202
   (DSCP 50) on rail 2 only. The CRS804 has a single lossless traffic class,
   so DSCP 50 is lossy on rail 1. Never send bulk in the fast class: it is
   strict priority and starves everything else on the port.
2. **Never run unscheduled many-to-one bulk.** When every node writes to all
   fifteen peers at once, each receiver port is oversubscribed fifteen to one.
   The switch then pauses senders, a paused sender stalls all of its flows, and
   small messages behind it wait. Measured on rail 2: 88 Gb/s a node instead
   of 98, and a small-message p99 of 0.65-1.3 ms (p99.9 1-1.7 ms). No ECN,
   PFC-threshold or buffer setting removed this; only the traffic pattern does.
3. **Schedule bulk as permutations.** In step `k` (1-15) node `i` sends to
   `(i + k) mod 16` and receives from `(i - k) mod 16`; or use pairwise
   halving and doubling (`i XOR 2^s`), which is bandwidth-optimal for
   reduce-scatter and all-gather on sixteen ranks. Every step is one-to-one, so
   no port is oversubscribed. Measured: 98.0 Gb/s a node on all sixteen at
   once, no pauses, no retransmissions, and a small-message p99 of 30 us in the
   bulk class (4.9 us in the fast class) while it runs.
4. **Split bulk across both rails.** The rails have separate switches and
   separate PCIe links, so a step's chunk split in half across them doubles
   the per-node rate.
5. **Pick the small-message algorithm by size.** One hop costs about 2.5 us
   (8 B, idle) and the wire moves about 12.25 GB/s a rail. A one-shot
   broadcast all-reduce (each node writes its whole vector to all fifteen
   peers, then sums) costs one hop plus `15 x size` of wire time; a
   reduce-scatter plus all-gather costs two hops plus about `2 x size`. The
   broadcast is faster below about 3 KB a node on one rail, the two-hop form
   above it. Small bursts do not trigger pauses: fifteen 4 KB writes stay far
   under the switch's per-port pause threshold.
6. **Do not change addresses under live queue pairs.** Deleting an address
   from a rail netdev deletes its RoCE GID even when the same IP stays under
   another prefix. A queue pair holding the GID keeps the slot pending, and the
   re-added GID lands at a new index (index 4 on spark4 on 2026-11-06). The
   rail 2 apply script therefore only adds addresses while queue pairs are
   live, and removes stale ones at boot.
7. **Keep one receive buffer on rail 2.** On a port driven through the link-B
   function, any `prio2buffer` map that uses two buffers makes the NIC drop
   every incoming broadcast (`rx_prio0_buf_discard` counts them), so ARP
   fails while cached unicast still works. Rail 1 is not affected and keeps
   its split map.
8. **Do not change switch QoS during inference.** NX-OS cannot edit an
   attached network-qos policy, so a change means building a new policy and
   swapping it in, which briefly re-programs every port. Change it in a
   window, then `copy running-config startup-config`.

## Checks

- Host: `ethtool -S enP2p1s0f0np0` for `rx_prio3_pause`, `rx_prio6_pause`,
  `rx_prio0_buf_discard` and `rx_discards_phy`;
  `/sys/class/infiniband/roceP2p1s0f0/ports/1/hw_counters/` for
  `out_of_sequence`, `packet_seq_err`, `local_ack_timeout_err` and
  `roce_adp_retrans`.
- `cisco0`: `show queuing interface` (tail drops per QoS group, ingress pause
  drops), `show interface priority-flow-control`,
  `show interface counters errors`.

## Measurements, 2026-11-06

All on rail 2, perftest 6.20, 1 MiB writes, two queue pairs a flow, traffic
class 106 unless noted.

| Test | Result |
|---|---|
| One flow each way, a pair | 196 Gb/s total, 0 corrected bits |
| 3:1 incast | 98.0 Gb/s total, 0 loss, pauses absorbed |
| 3:1 incast plus 4-stream TCP into the same node | RoCE 59 + TCP 43 Gb/s; 0 RoCE drops, 982 TCP tail drops |
| All-to-all, 16 nodes, 240 flows | 88 Gb/s a node each way, 0 loss or retransmission, 955 M RoCE packets with 0 switch drops |
| Permutation, 16 nodes, one or three shifts | 98.0 Gb/s a node |
| Idle one-way latency, 8 B / 4 KB / 64 KB | 2.46 / 4.11 / 9.90 us typical, p99.9 3.4 / 5.1 / 10.8 us (rail 1: 2.46 / 4.29 / 10.22) |
| 8 B latency during all-to-all, bulk class | 2.4 us typical, p99 650-1,300 us |
| 8 B latency during all-to-all, fast class | p99 300 us |
| 8 B latency during permutation at line rate, bulk class | 24 us typical, p99 30 us |
| 8 B latency during permutation at line rate, fast class | 4.5 us typical, p99 4.9 us, p99.9 28 us |

ECN thresholds from 20 KB to 1 MB minimum and a larger PFC pause threshold
left all-to-all at 85-88 Gb/s. Earlier marking lowered it.
