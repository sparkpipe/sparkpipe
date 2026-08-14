# Hardware Topology

SparkPipe supports four-, eight-, and sixteen-Spark deployment profiles. Every
profile is generated from one hardware contract and contains complete direct
pairs.

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

## DGX Station profiles

DGX Station deployments use separate measured hardware profiles while
preserving the same API, scheduler, package, and evidence contracts. A Station
may be standalone, augment a Spark deployment, or participate in a four- or
eight-Station fabric.

Mixed Spark and Station plans explicitly name ownership and transport between
hardware classes. The runtime never infers a path from a marketing link rate or
silently substitutes a management network.

## Generation and validation

One deployment JSON is the source of truth. Generators emit C topology tables,
runtime configuration, release manifests, and qualification plans. Generated
files are never edited by hand.

Validation rejects duplicate ranks or addresses, incomplete direct pairs,
unknown hardware types, missing rails, cross-communicator pairs, inconsistent
MTU or datatype profiles, and stale generated output.
