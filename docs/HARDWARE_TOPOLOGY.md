# Hardware Topology

SparkPipe supports four-, eight-, and sixteen-Spark deployment profiles, and
is designed to support pools of one to eight Mac Studios on their own or
alongside Sparks. Every profile is generated from one hardware contract; Spark
profiles contain complete direct pairs.

## Spark node contract

Each Spark contributes:

- one 100 Gb/s port on the CRS804 switched all-to-all fabric;
- one nominal 200 Gb/s port connected directly to `rank XOR 1`;
- 128 GB unified memory;
- one 4 TB internal NVMe; and
- at least one 4 TB external NVMe.

The useful direct-link payload is capped near 110 Gb/s by the GB10 PCIe path and
normally lands near 100 Gb/s. The switched and direct rails are therefore
treated as equal-rate useful data paths.

## Supported Spark sizes

| Nodes | Ranks | Direct pairs |
| ---: | --- | --- |
| 4 | `0-3` | `0<->1`, `2<->3` |
| 8 | `0-7` | `0<->1`, `2<->3`, `4<->5`, `6<->7` |
| 16 | `0-F` | `0<->1`, `2<->3`, `4<->5`, `6<->7`, `8<->9`, `A<->B`, `C<->D`, `E<->F` |

Every node remains on the switched fabric. Direct pairing adds a second path;
it does not partition the switched topology.

## Route contract

The generated topology names, per rank:

- global rank and deployment size;
- switched and direct interfaces and addresses;
- direct partner;
- forward and reverse split-ring neighbors and required rails;
- recursive XOR partners and required rails;
- NIC, GID, queue, registered-arena, MTU, and credit profile; and
- immutable topology and hardware-profile hashes.

A direct destination must route only over the direct interface. Every other TP
peer uses the switched interface. Startup validates route choice with interface
counters and fails if either mandatory rail is absent, duplicated, misrouted,
or below the deployment gate.

Management and control networks never satisfy an inference route.

## Storage contract

Internal NVMe per Spark:

```text
2.5 TB  hot KV and resumable request state
1.0 TB  active rank-local model shards
0.5 TB  OS, runtime, receipts, bounded scratch
```

External NVMe per Spark:

```text
>= 1.0 TB  direct rank-local model access and staging
remainder   contribution to the striped RAID-like model-data pool
```

The pooled external tier targets at least 20 Gb/s useful model-shard reads.
Every tier stores content-addressed package bytes. Path names alone never prove
device placement; startup binds each mount to its expected physical device,
capacity, role, and package identity.

## RTX 5090 host

The RTX 5090 workstation (`rtx5090`, x86_64, one RTX 5090 with 32 GB) is the
fleet's release hub, runs the GLM API and is the draft-model host
([RTX5090_SPECULATION_NODE.md](RTX5090_SPECULATION_NODE.md)). It is not on the
inference fabric:

- its 10 GbE port `eno1` (`10.10.250.2/30`) is cabled to sparkf's former
  management port `enP7s7` (`10.10.250.1`);
- the other Sparks reach it through sparkf on the switched fabric
  (`10.10.250.0/30 via 10.10.100.25 dev enp1s0f1np1`), a route installed by
  `sparkpipe-hub-route.service`.

The busy-polled 64 B UDP round trip is 22.6 µs p50 and 25.2 µs p99 from
sparkf, and about 300 µs p50 from the other Sparks through sparkf (lead-dev
measurement, 2026-09-28). Treat this link as its own link class: a design that
puts the host in every decode step budgets these round trips. Operations on
the host are in [FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md).

## Mac Studio pool

Eight Mac Studios with M5 Ultra and 256 GB are on order. Per Apple's
specifications, each has 256 GB of unified memory at 1.2 TB/s, six
Thunderbolt 5 ports (two front, four back) and one 10GbE port. Eight give
2 TB at 9.6 TB/s: the memory of sixteen Sparks at 2.2 times the bandwidth,
with a slower interconnect. Per box, the published figures put an M5 Ultra's
matrix throughput near a Spark's: EXO rates a Spark at about 100 FP16 TFLOPS
against 26 for an M3 Ultra ([EXO][exo-spark]), and Apple claims up to 4 times
the M3 Ultra's prompt processing for the M5 Ultra ([Apple][m5-ultra]). Sixteen
Sparks therefore have about twice the matrix throughput of eight Studios. The
Studios run SparkPipe through the Metal backend, which does not exist yet.

**Thunderbolt 5 RDMA.** From macOS 26.2, Macs expose RDMA over Thunderbolt 5
through a verbs-compatible API ([TN3205][tn3205]). It is narrower than the
Sparks' RoCE:

- Mac to Mac only;
- send and receive, with no remote read or write;
- unreliable-connection queue pairs, at most ten;
- point-to-point links with no routing, so software forwards a message to a
  Mac without a direct cable.

Published rates are 3.5-3.8 GB/s per link on an M3 Ultra mesh, and up to
7.4 GB/s on a mesh of two M4 Pro minis and an M3 Ultra
([MLX discussion 3481][mlx3481]). The plot in [MLX PR 2808][mlx2808] shows
JACCL's four-way all-reduce under 50 µs on M3 Ultra. EXO's RDMA mode requires
a full mesh, and EXO reports that a Mac Studio's Thunderbolt 5 port next to
the Ethernet port cannot carry RDMA ([EXO][exo]). A Thunderbolt 5 cable is its
own link class in the topology model.

**Islands.** Six ports cannot fully mesh eight Macs; that takes seven, and
if EXO's port limit holds on M5 Ultra, five ports carry RDMA. The pool is two
islands of four:

- within an island, a full mesh: three ports per Studio;
- across islands, Studio `i` to Studio `i + 4`: one port;
- one RDMA port stays free, and so does the port next to the Ethernet port,
  which can still carry IP over Thunderbolt.

**Placement.**

- A model that fits one Studio runs one replica per Studio, with no
  inter-Studio traffic on the decode path.
- A model that needs two to four Studios runs TP inside one island.
- A larger model runs TP4 x PP2 across the islands. A stage boundary carries
  each token's residual stream.

GLM 5.3 Flash at the fleet's precision (FP8 routed experts, BF16 elsewhere) is
about 320 GB, from the per-rank shapes in
[`GLM5_NEXT_ROOFLINE.md`](GLM5_NEXT_ROOFLINE.md). It needs TP2, so the pool
holds four replicas. A B1 step reads about 25 GB; split over two Studios that
is a 10.4 ms floor, 96 tok/s per replica. A 4-bit build fits one Studio, but
to the provider network it is a different model: a separate offer, with its
own reference nodes.

## Mixed fleets

Sparks and Studios serve one catalog behind one router. No tensor-parallel
collective crosses hardware classes. A mixed plan crosses them only at a
request or stage boundary, over a named bridge.

**Prefill on Sparks, decode on Studios.** Prefill is compute-bound, and
sixteen Sparks have about twice the matrix throughput of eight Studios. Decode
is bandwidth-bound, and eight Studios have 2.2 times the bandwidth of sixteen
Sparks. The Sparks prefill a prompt and stream its KV cache
to a Studio group layer by layer, overlapping the transfer with the remaining
layers; the Studios decode. EXO showed this split with two Sparks and an
M3 Ultra over 10GbE ([EXO][exo-spark]). For GLM 5.3 Flash the cache is
16.5 KiB per token (DSA latent and indexer keys over 11 layers) plus 142 MiB
of KDA state and convolution windows per sequence: about 430 MB for a
16K-token prompt, a third of a second at 10 Gb/s. Whether the split beats
either pool alone is the mixed-fleet gate in
[`PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md#target-gates).

**Pipeline stages across classes.** A model too large for either pool runs
PP2, one stage on the Sparks and one on the Studios. For GLM 5.3 Flash a stage
boundary carries the four 4096-wide hyper-connection streams, 32 KiB per token
in BF16: 2 MiB per step at B64, 1.7 ms at 10 Gb/s. Stages run at the
pace of the slower one, so the planner balances layers by measured stage time.
Pipeline stages first need cancellation for routes waiting on transport
input: today a stage that has posted its receive waits forever when a peer
rank fails.

**The bridge.** Apple's Thunderbolt RDMA is Mac to Mac, so a Mac reaches Linux
over TCP. The candidates, all to be measured when the Studios arrive:

- Each Studio's 10GbE port into one switch, uplinked to the CRS804. This
  reaches all eight Studios with no host in the path, and needs a free CRS804
  port.
- IP over Thunderbolt from a Studio to the RTX 5090 host. Linux's
  `thunderbolt-net` driver speaks Apple's protocol ([kernel docs][tbnet]). It
  has measured about 17 Gb/s from an M1 Mac ([Bergeron][tb-m1]) and 13.7 Gb/s
  between Linux hosts over USB4 ([gist][tb-usb4]), and it receives on one
  core ([netdev][tb-netdev]). On AMD boards with the ASMedia ASM4242
  controller, host-to-host networking dropped all traffic until a kernel fix
  in July 2026. After the fix, two ASM4242 hosts measured 4.2-5.2 Gb/s, below
  10GbE ([LKML][asm4242-fix]), and distribution kernels without the fix still
  carry no traffic to a Mac ([report][asm4242]).
- A ConnectX-6 in the RTX 5090 host, which puts that host on the Sparks'
  fabric. Each Spark's two ports are already taken, one on the switch and one
  on its pair, so the card joins through the CRS804 and needs a free port
  there. No ConnectX-6 linked directly to a Spark's ConnectX-7 is published;
  the nearest evidence is a Spark linked at 200GBASE-CR4 to a MikroTik switch
  over a QSFP56 DAC ([forum][spark-cr4]).

Each bridge is a named link class with a measured rate in the deployment
JSON, like the Spark rails. Management networks never carry it.

[tn3205]: https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt
[mlx3481]: https://github.com/ml-explore/mlx/discussions/3481
[mlx2808]: https://github.com/ml-explore/mlx/pull/2808
[m5-ultra]: https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
[exo]: https://github.com/exo-explore/exo
[exo-spark]: https://blog.exolabs.net/nvidia-dgx-spark/
[tbnet]: https://docs.kernel.org/admin-guide/thunderbolt.html
[tb-m1]: https://chrisbergeron.com/2021/07/25/ultra-fast-thunderbolt-nas-with-apple-m1-and-linux/
[tb-usb4]: https://gist.github.com/geosp/80fbd39e617b7d1d9421683df4ea224a
[tb-netdev]: https://ratatoskr.run/netdev/2026/03/11349474
[asm4242-fix]: https://ratatoskr.run/lkml/2026/07/17323580/t
[asm4242]: https://discuss.cachyos.org/t/asm4242-thunderbolt-networking-unusable-on-linux-cachyos-7-2-0-1/34826
[spark-cr4]: https://forums.developer.nvidia.com/t/connectx-7-200gbe-via-mikrotik-crs812-qsfp-dd-400g-2xqsfp56-200g-breakout/357162

## Generation and validation

One deployment JSON is the source of truth. Generators emit C topology tables,
runtime configuration, release manifests, and qualification plans. Generated
files are never edited by hand.

Validation rejects duplicate ranks or addresses, incomplete direct pairs,
unknown hardware types, missing rails, cross-communicator pairs, inconsistent
MTU or datatype profiles, and stale generated output.
