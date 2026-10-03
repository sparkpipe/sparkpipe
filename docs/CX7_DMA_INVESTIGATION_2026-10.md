# ConnectX-7 / SMMU DMA investigation, 2026-10-01 to 2026-10-03

Dated record of two days of work on the Lenovo ConnectX-7 completion-timeout
crash: what happened, every experiment and its result, what is in force now,
what it means for performance work, and what is still unknown. Times are UTC.
Nodes log in KST (UTC+9). The lead workstation clock ran about 1.3 s fast on
10-02; hub and node times are NTP-based.

Labels used throughout:

- **FACT**: read directly from a log, counter, config dump or file.
- **DOCUMENTED**: stated by upstream source code or vendor documentation; not
  re-verified on the Ubuntu `6.17.0-1026-nvidia` tree.
- **INFERENCE**: a conclusion drawn from facts; it can be wrong.

Unlabelled statements in the summary are FACT unless they say otherwise.

## Contents

1. [Summary](#1-summary)
2. [Hardware and software under test](#2-hardware-and-software-under-test)
3. [Chronology](#3-chronology)
4. [Incident 1: spark8, 2026-10-01, superblock overwritten](#4-incident-1-spark8-2026-10-01-superblock-overwritten)
5. [Incident 2: spark8, 2026-10-02, group descriptors overwritten](#5-incident-2-spark8-2026-10-02-group-descriptors-overwritten)
6. [The failure chain](#6-the-failure-chain)
7. [The DMA guard](#7-the-dma-guard)
8. [The reproducer (surekill) and its harness](#8-the-reproducer-surekill-and-its-harness)
9. [Every run](#9-every-run)
10. [What each single change showed](#10-what-each-single-change-showed)
11. [NIC counter forensics](#11-nic-counter-forensics)
12. [Configuration and firmware comparison](#12-configuration-and-firmware-comparison)
13. [Side experiments](#13-side-experiments)
14. [External evidence](#14-external-evidence)
15. [Mitigations in force and what they cost](#15-mitigations-in-force-and-what-they-cost)
16. [Implications](#16-implications)
17. [The BIOS question](#17-the-bios-question)
18. [Open questions and next experiments](#18-open-questions-and-next-experiments)
19. [Corrections to earlier statements](#19-corrections-to-earlier-statements)
20. [Tools](#20-tools)
21. [Owner decisions](#21-owner-decisions)
22. [Evidence locations](#22-evidence-locations)
23. [Glossary](#23-glossary)

## 1. Summary

### What happened

- **Two unguarded failures corrupted spark8's root filesystem.**
  - 10-01: the superblock and three group-descriptor blocks were overwritten
    with pages that belonged to other processes.
  - 10-02: four group-descriptor blocks were overwritten.
  - Both happened during fleet tests that ran two TP collectives at once on
    one engine (a lane's second mesh band). Each needed a PXE rescue and an
    `e2fsck` from the backup superblock.
- **The DMA guard was then installed on all 16 nodes.** It caught 13 more
  events in about 4 hours of deliberate crash testing on 10-02. Every one was
  on a Lenovo ThinkStation PGX (spark7 9, spark9 2, spark8 2). Every guarded
  crash left the root filesystem clean.
- **Nothing triggered after 10-02 23:20Z.** That covers about 7.5 hours,
  including two TCP floods, a single-band gate and an 8-driver run.

### What is established

- **The crash is Lenovo-only.**
  - Events: 15 of 15 on the three Lenovo units.
  - The 13 non-Lenovo nodes had zero events under the same symmetric
    exposure: about 380K two-band prefill token-equivalents each.
- **Every event starts on SMMU0.**
  - 14 of 15 began with a ConnectX-7 PCIe AER uncorrectable completion
    timeout (CmpltTO, Requester). In the 10 guarded events with detail lines,
    and both unguarded ones, it is confirmed as CmpltTO.
  - The 15th began with an SMMU0 `CMD_SYNC timeout` and no AER.
- **The SMMU0 command queue then stalls.** The consumer index freezes and
  queued commands are never consumed. On the unguarded nodes, NVMe writes
  later landed at the correct block numbers with the wrong page contents.
- **Two collectives in flight trigger it.** With the cd9786e build:
  - two-band prefill crashed a Lenovo node in 14 of 14 valid runs;
  - the same build with 512-row waves (single band, no overlap) survived
    3 of 3 full gates (1,086,819 prefill tokens).
- **Raw NIC bandwidth alone does not trigger it.** No trigger in any of these
  (Lenovo units in passthrough):
  - 5 minutes of bidirectional TCP at about 110 Gb/s per Lenovo node, about
    4x the crashing RDMA rate;
  - the same TCP flood with 142 GB/s of synthetic GPU memory traffic;
  - eight concurrent GLM Flash engines at about 4.5 Gb/s.
- **Moving traffic to the second PCIe link does not avoid it.**
  - In one event the pair function on link A and the switch function on
    link B timed out within the same 30 ms.
  - Both links sit behind the same SMMU0.
- **IOMMU passthrough lowers the crash rate about 2x (pooled) but does not
  stop it.** Three crashes happened in identity mode.
- **SparkPipe did not write the bad data.** The foreign pages came from sshd
  processes, log files and the journal, through the kernel's own ext4/jbd2
  writeback.

### What is believed but unproven (INFERENCE)

- **Mechanism of the corruption.** After the SMMU0 stall, the kernel freed
  and reused IOVAs while stale IOTLB entries still pointed at the old pages.
  NVMe DMA then read the wrong physical pages.
- **Passthrough closes the corruption path.** Identity domains have no IOVAs
  to reuse.
- **The trigger is the access pattern, not the rate:**
  - about twice the outstanding NIC reads of host memory;
  - the GPU writing host memory that the NIC reads at the same time.
- **The Lenovo-specific cause sits in the platform layer:**
  - candidates: the Lenovo UEFI/DSDT power and DVFS block, the board, the
    power input, or unit variation;
  - not established;
  - a firmware inventory taken on 10-03 shows that EC and UEFI age alone
    does not separate crashers from survivors (section 12.4).

### What is in force now

| Mitigation | Status |
|---|---|
| One collective in flight per engine | Production runs single-band release 0b5371e. The two-band overlap is parked in draft PR #1427. |
| DMA guard | Armed on 16/16 nodes. sysrq-b reboot on the first CX-7 uncorrectable error or SMMU failure line. |
| IOMMU passthrough | spark7, spark8 and spark9 only. The other 13 nodes stay Translated (DMA-FQ). |
| No serialization of RDMA across drivers | Owner decision 10-02. |
| No BIOS/EC/SoC flashing | Owner decision 10-02. Reopened 10-03 06:58Z, undecided. |

### What it costs

- **Today: nothing measurable.** The two-band overlap gave no prefill
  speedup: 16K TTFT 20.8 s against 19.1 s for single-band production.
- **Later: likely a lot.**
  - Prefill is transport-bound and runs at about 26% of its one-port wire
    floor.
  - Closing that gap means 4x more NIC traffic and more concurrency, which
    is the region where the Lenovo units fail.
  - Several fast drivers on one node also put several collectives in flight
    per node. The 8-driver run tested this only at low rate (section 16.2).

### Most valuable next steps

1. **Surekill-gate the production configuration** (single band, 1024-row
   waves). It has never been tested; only 512-row single band was
   (section 10.1).
2. **Build a SparkPipe-free reproducer.**
   - What: line-rate multi-QP `ib_write_bw` incast on a Lenovo node, in
     Translated and in identity mode.
   - Why: either outcome localizes the trigger, and a vendor case needs one.
3. **Run the zero-risk BIOS checks** before any flash (section 17):
   - a cold AC drain;
   - a position swap and a power-brick swap with a GIGABYTE unit;
   - a Lenovo support case while the warranty runs.
4. **Test 2–4 fast drivers at once on a Translated Lenovo node.** That is
   the multi-driver case still uncovered.

## 2. Hardware and software under test

### 2.1 Fleet

16 GB10 nodes plus the rtx5090 hub. Live DMI read on 10-03. Section 12.4 has
the EC versions read on 10-03.

| Node | Vendor / model | BIOS (date) | Kernel | Root NVMe | IOMMU now (CX-7, NVMe) | Real guard triggers |
|---|---|---|---|---|---|---|
| spark0 | GIGABYTE AI TOP ATOM | 5.36_0ACUM05 (2025-11-07) | 1026 | Samsung | DMA-FQ | 0 |
| spark1 | MSI MS-C931 EdgeXpert | 5.36_1.6.0 (2026-03-16) | 1026 | Samsung | DMA-FQ | 0 |
| spark2 | GIGABYTE | 0ACUM05 (2025-11-07) | 1026 | Samsung | DMA-FQ | 0 |
| spark3 | GIGABYTE | 0ACUM06 (2026-03-22) | 1026 | Samsung | DMA-FQ | 0 |
| spark4 | GIGABYTE | 0ACUM04 (2025-10-23) | 1026 | Phison | DMA-FQ | 0 |
| spark5 | GIGABYTE | 0ACUM06 | 1026 | Samsung | DMA-FQ | 0 |
| spark6 | GIGABYTE | 0ACUM06 | 1026 | Phison | DMA-FQ | 0 |
| **spark7** | **LENOVO 30KLS02X00 (ThinkStation PGX)** | **S0QKT09A (2025-10-14)** | 1026 | Phison | identity since 10-02 23:26Z | **9** |
| **spark8** | **LENOVO 30KLS02X00** | **S0QKT09A** | 1026 | Phison | identity since 10-02 18:11Z | **2** (plus the 2 unguarded corruptions) |
| **spark9** | **LENOVO 30KLS02X00** | **S0QKT09A** | 1026 | Phison | identity since 10-02 23:24Z | **2** |
| sparka | GIGABYTE | 0ACUM06 | 1026 | Phison | DMA-FQ | 0 |
| sparkb | GIGABYTE | 0ACUM04 (2025-10-23) | 1026 | Phison | DMA-FQ | 0 |
| sparkc | Dell Pro Max with GB10 FCM1253 | 5.36_3.0.0 (2025-10-08) | 1026 | Phison | DMA-FQ | 0 |
| sparkd | GIGABYTE | 0ACUM08 (2026-07-02) | 1029 | Phison | DMA-FQ | 0 |
| sparke | GIGABYTE | 0ACUM08 | 1029 | Phison | DMA-FQ | 0 |
| sparkf | GIGABYTE | 0ACUM08 | 1029 | Phison | DMA-FQ | 0 |

**Common to all 16:**

- Kernel and driver: 1026 = `6.17.0-1026-nvidia` (DGX OS 7.2.3, driver
  580.159.03); 1029 = `6.17.0-1029-nvidia` (DGX OS 7.4.0, 580.173.02).
- CX-7: firmware 28.45.4028, PSID NVD0000000087.
- Secure Boot enabled, kernel lockdown `[integrity]`.
- Common kernel arguments: `init_on_alloc=0`, `pci=pcie_bus_safe`,
  `fsck.mode=skip fsck.repair=no`, `crashkernel=1G-:0M` (so no kdump), and
  `console=ttyS0,921600`.
- The NVMe model is not Lenovo-specific: the Phison E27T is in 11 nodes, all
  three Lenovos included.

### 2.2 ConnectX-7 and PCIe layout (identical on every node)

Each CX-7 exposes its two QSFP ports over two independent PCIe Gen5 x4
links. That gives four physical functions (PFs), all at 32 GT/s x4. A Gen5
x4 link carries about 13.6 GB/s per direction, shared by the two functions
on it.

| PCI function | netdev / RDMA device | Port | Role | PCIe link (root port) | IOMMU group |
|---|---|---|---|---|---|
| 0000:01:00.0 | enp1s0f0np0 / rocep1s0f0 | 0 (200G) | Pair link to rank^1 (10.10.200.x/31); weightd pair QPs, 1/15 of mesh bytes | A (0000:00:00.0) | 7 |
| 0000:01:00.1 | enp1s0f1np1 / rocep1s0f1 | 1 (100G) | MikroTik CRS804 switch (10.10.100.(10+rank)/24); weightd switch QPs, 14/15 of mesh bytes; Ceph TCP | A | 8 |
| 0002:01:00.0 | enP2p1s0f0np0 / roceP2p1s0f0 | 0 | Pair link on link B (10.10.201.x/31, runtime bring-up); no mesh traffic by default | B (0002:00:00.0) | 10 |
| 0002:01:00.1 | enP2p1s0f1np1 / roceP2p1s0f1 | 1 | Switch "door B" (10.10.101.(10+rank)/24); down by default | B | 11 |

**PCIe configuration**

- Byte-identical on all 16 nodes for the CX-7 functions and root ports:
  - MPS 512, MRRS 512, RlxdOrd+, NoSnoop+, ExtTag+;
  - DevCtl2 completion timeout 50 µs–50 ms (ranges A, B, C supported,
    TimeoutDis-); 10BitTagReq-;
  - ASPM disabled; CmpltTO is non-fatal; no ACS on the device;
  - root port ACSCtl SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+.
- Root ports report two board-level retimers (LnkCap2 Retimer+ 2Retimers+).
  No retimer firmware is visible to fwupd.

**NIC firmware settings**

- All 218 CX-7 NV parameters are identical across nodes and functions, and
  every one is at its default.
- `ADVANCED_PCI_SETTINGS=False` hides `MAX_ACC_OUT_READ`.
- `ATS_ENABLED=False`.

### 2.3 SMMU layout

| SMMU | Base | Translates |
|---|---|---|
| SMMU0 `arm-smmu-v3.0.auto` | 0x13800000 | All four CX-7 PFs, the root NVMe 0004:01:00.0, the Realtek management NIC 0007:01:00.0, the MT7925 WiFi, the PCIe root ports, and every xHCI controller (including the Ceph USB OSD disk) |
| SMMU1 `arm-smmu-v3.1.auto` | 0x13000000 | Only the GPU 000f:01:00.0, its root port and HDA audio |
| SMMU2 `arm-smmu-v3.2.auto` | 0x14900000 | Two platform devices |

**SMMU0**

- One command queue serves the NIC, the root disk, the USB/Ceph disk and the
  management NIC. A stall there affects all of them.
- Boot lines: 40-bit ias/oas, features 0x0396dfbf, a 65,536-entry command
  queue.
- The ACPI IORT marks every SMMU as model 0 (generic), so there is no NVIDIA
  CMDQV.
- The implementer (IIDR) is unknown.

**IOMMU groups and modes**

- Group numbers are identical on every node: CX-7 7/8/10/11, NVMe 13, GPU 20.
  Each group holds one device.
- On a Translated node every group is DMA-FQ (lazy invalidation), except GPU
  group 20, which is DMA (strict).
- ATS and PASID are enabled only on the GPU. No CX-7 function, the NVMe or
  any root port has ATS, PRI or PASID, so SMMU0 never issues ATC
  invalidations.

**How translation is configured**

- The kernel is built with `CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y`.
- Translation is on only because the DGX OS file
  `/etc/default/grub.d/iommu.cfg` appends `iommu.passthrough=0`.
- On the Lenovo units, `zz-spark-iommu-passthrough.cfg` sorts after it and
  appends `iommu.passthrough=1`; the last value wins.

### 2.4 How SparkPipe drives the NIC

**Queue pairs**

- weightd runs `--mesh-interface rocep1s0f1 --mesh-pair-interface
  rocep1s0f0`, so all mesh RDMA rides PCIe link A.
- Per node there are 15 send and 15 receive RC QPs: 2 to the pair partner,
  and 28 over the switch to the other 14 peers.
- QP settings: RDMA WRITE only, path MTU 4096, QP timeout 14, retry 7,
  `max_send_wr` 64, traffic class 106 (DSCP 26, PFC priority 3).

**Memory regions**

- The NIC reaches SparkPipe memory only through weightd's long-lived MRs on
  memfd host memory:
  - the mesh region (268,632,064 B) and staging (134,217,728 B);
  - each registered once on both protection domains and never deregistered
    while weightd runs.
- No GPU-memory MR exists in production. GPUDirect RDMA is unsupported on
  DGX Spark.
- Mesh RDMA therefore does no per-transfer IOMMU map or unmap. On Translated
  nodes the per-I/O SMMU0 map, unmap and TLB-invalidate traffic comes from
  NVMe, TCP (including Ceph) and USB (INFERENCE).

**Bands**

- A band is a slot region of the single weightd mesh buffer:
  band = 2 × lane + band index, so 32 bands.
- Band 1 uses the same per-peer QPs, the same NIC port and the same SMMU0.
- What a second concurrently active band changes, at about the same bytes
  per wave:
  - outstanding payload writes per node rise from 15 (up to 3.75 MiB of NIC
    reads of host memory) to 30 (up to 7.5 MiB, up to two per QP);
  - exchanges per 1024-row wave rise from 1,099 to 1,414;
  - a second staging slice and slot area become DMA targets.
- A 1024-row prefill wave sends and receives about 3.82 GiB (3,916.5 MiB)
  per node each way in both modes.

**The two-band workload (cd9786e)**

- Build: `drivers/glm52-prefill-overlap`, prefill compute/communication
  overlap.
- With `SPARK_GLM52_PREFILL_WAVE_ROWS=1024` (anything above 512), each
  1024-row wave runs as two 512-row halves on two CUDA streams. The second
  half's all-reduces use band 1, on lane 12, bands 24 (owned) and 25
  (borrowed).
- Waves of 512 rows or fewer, decode and t1 stay single-band.
- rank0 logs `GLM52-PREFILL-OVERLAP enabled=1 ... band=1` when overlap is
  on.

**Flash and K3 also use band 1, serially.** GLM Flash opens band 1 for its HC
all-reduce and K3 for its wide gate/up reduce, but each queues both
collectives in order on one stream. The precise rule is therefore "one
collective in flight per engine", not "one band".

## 3. Chronology

| UTC | Event |
|---|---|
| 08-27 | SMMU1 (GPU) CMD_SYNC timeouts on spark2 (1) and spark3 (3,514) after GPU Xid 119. Roots intact. Unrelated SMMU instance. |
| 09-30 04:54–07:35 | 3–5 concurrent GLM Full TP16 dev lanes on all 16 nodes, 8–17-row waves (about 60x less data per collective than 1024-row waves). No AER. |
| 10-01 ~11:11 | `ds4-direct-pair-fabric-p2.service` boot unit enabled fleet-wide (link-B pair up). Later blamed without evidence; disabled on the other 15 at 16:36. |
| 10-01 15:58 | Commit e8e973d8b, "WIP pipelined two-band all-reduce" (template forces `mesh_band_count=2`). |
| 10-01 16:00:49 | **Incident 1**: spark8 CX-7 AER CmpltTO, then SMMU0 stall, then superblock corruption, panic and reboot (section 4). |
| 10-02 02:17–03:39 | spark8 PXE rescue; `e2fsck` from the backup superblock. PRs #1418 (PXE masks the p2 unit) and #1419 (PXE `--hold bottom`). |
| 10-02 11:44–11:46 | Single-band 0b5371e overlap runs, including two concurrent 16K prefills. No AER. |
| 10-02 12:07 | 9dfd62e overlap: about 3 s of two-band, then a functional rejection. |
| 10-02 12:13:44 | **Incident 2**: spark8 CmpltTO in pk60k, SMMU0 stall, node dark; later found with group descriptors corrupted (section 5). |
| 10-02 16:46–17:04 | spark8 PXE rescue and `e2fsck`; local boot 17:04:19. |
| 10-02 17:48 | Owner: guard approved, `iommu.strict` rejected, passthrough favoured, pre-production authority. |
| 10-02 17:54–18:06 | Guard dry-run injection (17:54:50) and armed injected reboot test (18:06:26) on spark8. |
| 10-02 18:11:44 | spark8 boots `iommu.passthrough=1`. |
| 10-02 18:12 | Guard armed on all 16 nodes. |
| 10-02 19:07:20 | First spark7 event (link-A pair function). Guard reboot. |
| 10-02 19:14:55 | Guard reinstalled with 30 ms detail forwarding (commit d45211a7a). |
| 10-02 19:32 | Owner: no BIOS flash; startup must be order-independent. |
| 10-02 20:40–20:52 | Door campaign; no workload ran (script bug). |
| 10-02 20:50 | Owner: surekill method, one change at a time. |
| 10-02 21:26 | PR #1428 (guard) merged. |
| 10-02 21:42–21:58 | Surekill baseline: 3 of 3 spark7 crashes. |
| 10-02 21:58–22:24 | Single band: survived twice, 362,273 tokens each. |
| 10-02 22:27 | Baseline recheck on the same boot crashes. |
| 10-02 22:32–22:46 | spark7 in passthrough: crashes at 72,308 tokens; next run spark9 (Translated) crashes. |
| 10-02 22:50–23:03 | Completion-timeout change attempted, blocked by lockdown. Two baseline repeats crash spark7; one of them is SMMU-first. |
| 10-02 23:02:59 | Owner: one band per engine, guard on 16, Lenovo passthrough. |
| 10-02 23:06:50 | Link-B test: spark8 (identity) times out on link A and link B at once. |
| 10-02 23:14:06 | Owner: passthrough only on Lenovo; no RDMA serialization. |
| 10-02 23:14:15 | spark9 crash (baseline recheck 2). |
| 10-02 23:20:03 | spark8 (identity) crash during the TX-cap test (cap not enforced). Last trigger to date. |
| 10-02 23:20:52 | DB-mining workers on spark0-6 stopped, with owner OK. |
| 10-02 23:24–23:28 | spark9 and spark7 to passthrough; fleet 16/16 armed; production and API restored. |
| 10-02 23:50–10-03 00:12 | TCP floods (with and without GPU stress): no trigger. |
| 10-03 00:06 | Single-band gap recheck survives (Lenovos in passthrough, miners gone). |
| 10-03 00:12–00:44 | TCP vs RDMA benchmark. |
| 10-03 01:04–01:35 | 8-driver GLM Flash run: no trigger, identical outputs. |
| 10-03 06:37 | PRs #1431–#1435 (startup resilience) merged. |
| 10-03 06:58 | Owner reopens the BIOS question. |
| 10-03 ~08:00 | Firmware inventory of all 16 nodes (section 12.4). Ceph recovered: three USB OSDs re-attached; the test harness's leftover `noout`/`nobackfill`/`norebalance` flags cleared. |

## 4. Incident 1: spark8, 2026-10-01, superblock overwritten

### 4.1 Workload

- Commit e8e973d8b on branch `perf/pair-first-allreduce`, unpushed: a
  pipelined two-band all-reduce. The shared template forced
  `mesh_band_count=2`.
- Engine "news" on lane 12, bands 24+25, `SPARK_TP_WAIT_MODE=hardware`,
  graph chains. All nodes Translated. No guard existed yet.
- Sessions t1, b1 and b16 passed (5,134 prompt tokens).
  - b16 ran six prefill chains of 361–997 rows. By the code threshold they
    should already have taken the two-band path, but no log records which
    path ran.
- The 994-row ttft1k request was accepted at 16:00:49.162 (sparkf clock).

### 4.2 Fault sequence (FACT unless marked)

| Time | Event |
|---|---|
| 16:00:49.355 kernel (16:00:49.509 rsyslog) | AER Uncorrectable (Non-Fatal) from 0000:01:00.1: Transaction Layer, Requester ID, status 0x00004000 = `[14] CmpltTO (First)`. About 0.2–0.4 s into the 994-row request. It is the only AER in spark8's rasdaemon database. |
| +4 ms | `mlx5_pci_err_detected` on 0000:01:00.0 (AER recovery notifies both PFs of link A). |
| ~16:00:49.44 (INFERENCE) | The stuck CMD_SYNC was issued, about 85 ms after the AER, if the upstream 1 s poll applies. |
| 16:00:50.439 kernel (16:00:50.594 rsyslog) | First `arm-smmu-v3 ... CMD_SYNC timeout at 0x00017be3 [hwprod 0x00017bea, hwcons 0x00017be3]`, **1.085 s after the AER**. hwcons stayed 0x17be3 in every later line while hwprod rose to about 0x18288 (about 1,700 queued commands never consumed). |
| 16:00:50.646 | `/var/log/syslog` on disk stops mid-line, exactly on a 4 KiB page boundary. |
| 16:01:04.565 | Last intact ext4 journal transaction (2208818). |
| ~16:01:09.6 | First wrong-page write with a known time: the slot of transaction 2208819 holds a byte copy of an older descriptor block. |
| ~16:01:19.8 | Ranks 0 and 9 report chain io_error after 30.6 s. |
| 16:01:23–25 | spark9 weightd CQ error status 12, QP repair. |
| 16:01:59, 16:02:13, 16:02:43, 16:03:06 | More wrong-page writes: an `auth.log` page in page 0 of the user journal; rsyslog Xid text inside the system journal; syslog text in osd.8's log. |
| 16:03:06 | GPU Xid 119 (GSP RPC timeout after 45 s), then Xid 154 "GPU Reset Required". SMMU1 (GPU) logged no CMD_SYNC timeouts. |
| 16:03:46–16:04:05 | efi-pstore: a CPU spinning in `arm_smmu_cmdq_issue_cmdlist` ← `arm_smmu_tlb_inv_context` ← `arm_smmu_flush_iotlb_all` ← `fq_flush_iotlb` ← `fq_flush_timeout`. 92 CMD_SYNC lines and 26 NVMe `F_PERMISSION ... unpriv data write` faults (stream ID 0x50100). |
| 16:04:31.32 | Last ext4 journal commit. |
| 16:04:34.80 | `Kernel panic - not syncing: hung_task: blocked tasks` (systemd blocked over 122 s in `synchronize_rcu`). `kernel.panic=60`, `hung_task_panic=1`. |
| 16:05:36 | Automatic reboot; root unmountable. The owner power-cycled at 16:13:53. |

No memory pressure: earlyoom logged 52,765 MiB available of 122,502 MiB and
swap 100% free at 16:00:50.

### 4.3 Damage

**Block 0 (the superblock) of `nvme0n1p2`**

- It held one complete, page-aligned 4 KiB glibc heap page from a process
  using OpenSSL 3.0.13, most likely an sshd (identified by elimination).
- The page contained secret key material. The dump is private and is not
  reproduced here.
- The magic bytes read `c0 49` instead of `0xEF53`.
- No genuine superblock byte survived, and the ext4 stamping fields were heap
  bytes. So the page was written by DMA, not by `ext4_commit_super`.

**Other damage in the first 64 blocks**

- Block 10 (GDT for groups 576–639) was a byte-exact relocated page of
  `libkrb5support.so.0.1` from a second sshd process.
- Blocks 48 and 49 held checksum-valid descriptors that belong to GDT blocks
  398/399 (a constant 350-block offset: an earlier in-memory state).
- 192 bad descriptors in total, all in blocks 10, 48 and 49.

**The pattern: right destination, wrong source page.** The superblock and
GDT home locations are what jbd2 checkpointing writes.

**Ruled out**

- Swap: a raw scan of all 52 GiB of swap found no fingerprint of either
  source process.
- The p2 boot unit: a PXE boot with it masked still could not mount root.

### 4.4 Why SparkPipe is not the writer

- No SparkPipe binary links libcrypto or krb5.
- The bad writes went through the kernel's own ext4/jbd2 metadata writeback.

SparkPipe was not ruled out as the trigger. Evidence against it at the time:

- six pipelined-size chains ran in the 23 s before without incident;
- 15 identical nodes saw nothing;
- weeks of fleet RDMA had produced zero SMMU0 timeouts;
- weightd uses only standard pinned `ibv_reg_mr`.

**SparkPipe defects found by that review.** All are bounded to SparkPipe's
own registered memory:

- weightd does not range-check client offsets, lengths or lkeys in
  `MESH_WRITE`/`MESH_BROADCAST`;
- QP repair goes RESET → RTS with PSN 0;
- second-band "shipped" credits are lost at chain epochs;
- host accounting ignores band 1;
- the shared template forces `mesh_band_count=2`;
- the `cudaHostRegister` skip does not check the length;
- 8 remote-write MRs of 8 bytes leak;
- the per-lane `residentd.log` is overwritten by the next run;
- no log records which collective path ran.

### 4.5 Recovery

1. GRUB stopped at its prompt.
2. PXE rescue boots (02:45, 03:06, 03:08 on 10-02) with `fsck.mode=skip`
   waited for root without mounting it.
3. A dmesg line in the rescue boot, `BERT: [Hardware Error]: Skipped 1 error
   records`, has never been read.
4. The 256 KiB head of `nvme0n1p2` was dumped, then `e2fsck -n` and `-y`
   were run from the backup superblock.
5. First good boot 03:31:50; local boot 03:39:31.

The 10-01 incident boot is absent from spark8's journal.

## 5. Incident 2: spark8, 2026-10-02, group descriptors overwritten

### 5.1 Workload

- Run `ovl-20261002T1210Z`, release cd9786e, `WAVE_ROWS=1024`, all nodes
  Translated. No guard.
- Passed: t1, b1, ttft1k, ttft16k (TTFT 20.81 s), pk8k (9.0 s), pk16k
  (16.5 s) and pk32k (36.3 s).
- pk60k (59,983 tokens) was accepted at 12:13:29 and never produced a token.

### 5.2 Fault sequence (FACT)

| Time | Event |
|---|---|
| 12:13:44.079 | AER CmpltTO (Requester) from 0000:01:00.1, 14.6–15.1 s into pk60k, after 75,402 completed tokens (about 100 s of two-band prefill). |
| 12:13:44.083 | `mlx5_pci_err_detected` on 0000:01:00.0. |
| 12:13:45.139 | First SMMU0 CMD_SYNC timeout at 0xb1b7, **1.060 s after the AER**. 10 lines through 12:13:46.875, hwcons frozen. |
| 12:13:45.46 | A UAS read on the Ceph USB disk reported stuck in flight. |
| 12:13:45.455 | osd.8 BlueStore: 6 crc32c checksum errors while deep-scrubbing two EC PGs (one location returned data whose checksum belongs to another location). These are the only checksum errors osd.8 has ever logged. |
| 12:14:09 | The monitor marked osd.8 failed. |
| 12:14:35 | spark8's standby mgr still received monmap messages over 10.10.100.x, so TCP on 0000:01:00.1 partly worked 50 s after the AER. |
| 12:16:25 | The node was dark: no ping on either network, sshd silent, switch link up. Unlike 10-01, it did not panic or reboot despite `kernel.panic=60` and `hung_task_panic=1`. |
| after 12:17:01 | Foreign pages written into GDT blocks 41, 44, 47 and 445 (205 descriptors): syslog text whose last line is 12:17:01, the standby mgr.spark8 log ending 12:14:35, a page about 78% one byte value, and an essentially constant-byte page. The wrong-page writes therefore happened more than 3 minutes after the stall. |

### 5.3 Recovery

- The owner power-cycled spark8 at 16:46 into PXE rescue
  (`--hold bottom`). Root would not mount: `EXT4-fs: group descriptors
  corrupted`.
- The primary superblock was intact.
- Repair: `e2fsck -n -f -b 32768 -B 4096`, then `-y` (911,095 files; exit 1,
  errors corrected), then a verify. Local boot 17:04:19.

Caveats:

- The incident boot's journal is effectively lost (12 s of entries).
- One user journal file is reported corrupted on every read.
- The test gate's "guard after: 2 bad" line was two ssh timeouts for spark8,
  not two bad superblocks. That gate reads only the primary superblock magic
  and cannot see group-descriptor damage.

## 6. The failure chain

### 6.1 Step by step

| Step | Statement | Status |
|---|---|---|
| 1 | A CX-7 PF on a Lenovo node reports AER Uncorrectable (Non-Fatal) CmpltTO tagged Requester ID | FACT, 14 of 15 events |
| 1a | The missing completion was for a NIC DMA read of host memory (WQE fetch, or RDMA WRITE payload read from the staging memfd) | INFERENCE. AER semantics only establish that the reporting PF issued a non-posted request and got no completion within its 50 µs–50 ms window. Posted writes cannot produce CmpltTO. |
| 1b | Once, the first line was an SMMU0 CMD_SYNC timeout with no AER before it or within 30 ms (spark7, 10-02 22:55:51) | FACT |
| 2 | `mlx5_pci_err_detected` follows on 0000:01:00.0 within 3–4 ms | FACT |
| 3 | SMMU0 CMD_SYNC timeouts appear 1.06–1.085 s after the AER, with the queue consumer frozen while the producer advances. Translation is not wholly dead: NVMe and USB DMA continue, and ext4 commits continued for 3.5 minutes on 10-01. | FACT |
| 3a | The stuck CMD_SYNC was issued about 85 ms (10-01) and about 60 ms (10-02) after the AER | INFERENCE, assumes the upstream 1 s poll |
| 3b | Causal direction between CmpltTO and the SMMU0 stall | UNKNOWN. Either one wedged inbound transaction in the SoC fabric blocks both, or the mlx5 error-recovery path wedges the queue. The SMMU-first event and the 1.06 s-late AER detail at 23:01:31 favour "SMMU or fabric stall first, CmpltTO as a symptom" (INFERENCE). With translation on, the root-port AER MSI address is itself an IOVA through SMMU0, so a stalled SMMU0 can delay or swallow the AER report. |
| 3c | A common point above the root ports | INFERENCE from the 23:06:50 event: CmpltTO on two different root ports within 30 ms |
| 4 | Upstream v6.17 frees IOVAs and IO page-table pages after a failed CMD_SYNC, in both lazy and strict modes | DOCUMENTED. `arm_smmu_cmdq_issue_cmdlist` logs and returns `-ETIMEDOUT`; the flush and sync callers return void; `fq_flush_iotlb` counts the flush as finished regardless; `fq_ring_free` then frees. The strict path frees after a failed `iommu_iotlb_sync` too. The CMD_SYNC poll runs with local IRQs off for up to 1 s. A full queue makes issuers loop on "CMDQ timeout". |
| 5 | Stale IOTLB entries plus reused IOVAs make NVMe write commands DMA-read whatever physical page the stale entry points at, and store it at the correct LBA | INFERENCE, rated high. Supported by 26 NVMe `F_PERMISSION unpriv data write` faults (device writes hitting stale read-only translations), whole foreign pages at correct metadata LBAs, misplaced pages holding in-window RAM content, and zero corruption on nodes without a stall. |
| 6 | Wrong-page writes come seconds to minutes after the stall | FACT: from 16:01:09.6 on 10-01; after 12:17:01 on 10-02 |
| 7 | SparkPipe is not the writer | FACT (section 4.4) |
| 8 | Two concurrent collectives on a Lenovo node trigger it | FACT for the correlation: 15 of 15 under two-band, 0 under single-band gates. The mechanism is INFERENCE. |
| 9 | A Lenovo-specific factor (platform firmware, board, power, unit variation) | INFERENCE, unresolved |

### 6.2 Why passthrough helps, and why strict mode would not

- **Identity (passthrough)** installs a bypass STE for the device. There are
  no IOVAs, no IO page tables, no IOTLB entries and no unmap invalidations
  for it (DOCUMENTED). The recorded stale-IOVA corruption path cannot occur
  (INFERENCE). The NIC completion timeout still happens (FACT, 3 identity
  crashes).
  - Cost: no DMA isolation and no SMMU fault log for the CX-7, NVMe and USB.
    A stray DMA from device firmware or a driver silently lands in RAM.
  - RDMA keeps only the NIC's own lkey/rkey checks, which matters because
    weightd does not range-check client offsets.
- **`iommu.strict=1`** would not close the hole: a strict unmap after a
  timed-out CMD_SYNC still frees the IOVA (DOCUMENTED, upstream).
  - Strict only slows reuse (about one stale IOVA per CPU per 1 s timeout)
    and pushes the node toward a hang sooner.
  - Published costs elsewhere: Grace NVMe random read 16.4M to 2.8M IOPS;
    ConnectX-6 TCP down 29%.
  - Rejected by the owner ("too big a price").
- **A runtime alternative** exists: unbind, write the group type, rebind. It
  works for the CX-7 and USB groups, not for the root NVMe, and is lost at
  every reboot. Not used.

### 6.3 Hang mechanism (INFERENCE)

- CPUs spin in 1 s CMD_SYNC polls from flush-queue softirqs and starve RCU.
  systemd then blocks in `synchronize_rcu`, and the hung-task watchdog
  panics (10-01).
- On 10-02 the node hung hard without the panic firing; why is unknown.

## 7. The DMA guard

`tools/devcycle/spark_dma_guard.py`: commits 751b4419b (2026-10-02 18:14Z)
and d45211a7a (19:15Z), PR #1428 (merged 21:26Z). The procedure is in
[INCIDENT_RECOVERY_PLAYBOOK.md](INCIDENT_RECOVERY_PLAYBOOK.md#dma-guard).

### 7.1 Design

- **Input.** It reads `/dev/kmsg` from the start of the current boot's ring
  buffer, so lines logged before it started still count. Only
  kernel-facility records match, unless `--accept-user` is given.
- **Triggers** (first match wins):
  - SMMU `CMD_SYNC timeout`;
  - SMMU `CMDQ error`, `unexpected global error reported`, or `device has
    entered Service Failure Mode`;
  - `mlx5_pci_err_detected`, or an mlx5 `PCIe Bus Error:
    severity=Uncorrectable`;
  - a fatal firmware-first `[Hardware Error]`;
  - AER `Uncorrectable (Fatal)` from any device;
  - AER `Uncorrectable (Non-Fatal)` only from a Mellanox device (vendor
    0x15b3).
- **Deliberately not triggers** (checked by the self-test):
  - `AER: enabled`, corrected AER from the CX-7, and NVMe Non-Fatal AER;
  - the cmdq allocation line, SMMU2 events, GPU Xid 31, `Cable plugged`, and
    corrected `[Hardware Error]`;
  - user-space copies of crash lines.
- **Reaction.**
  1. Send the trigger line by UDP to the hub at 192.168.50.4:5514 (each
     datagram 3 times, so every event appears 3 times in the hub log).
  2. For 30 ms, forward up to 24 further kernel lines as `event=detail`.
  3. Sleep 0.1 s.
  4. Write `b` to `/proc/sysrq-trigger`: immediate reboot, no sync, no
     unmount. It fires once per process.
  - The reset lands about 130 ms after the trigger line on a responsive node
    (INFERENCE from the code). At 23:01:31 it was at least 1.06 s.
- **Startup.** It opens `/proc/sysrq-trigger` and calls `mlockall` up front,
  registers the boot in `/var/lib/spark-dma-guard/boots.json` (fsynced
  atomic replace; last 20 kept) and reports `event=start mode=...`.
- **Loop protection.**
  - It runs watch-only (reports, does not reboot) if the 3 most recent
    previous boots were all unclean and the oldest started less than 6 h
    ago, or if the state file cannot be written.
  - A clean service stop marks the boot clean. Trimming `boots.json` re-arms
    the next start.
- **Units.**
  - `spark-dma-guard.service`: After multi-user, ssh and network-online;
    `Restart=on-failure`, `RestartSec=10`, at most 5 starts in 600 s.
  - `spark-dma-guard.timer`: `OnBootSec=60`, so it is never in the
    boot-critical path.
  - `install` succeeds only if the service is active with `NRestarts=0`.
  - Run `scan` before installing.
- **Hub listener.**
  - `spark_dma_guard.py listen --port 5514 --log
    ~/spark-dma-guard/events.log`, running as the transient user unit
    `spark-dma-guard-listen` on rtx5090 since 10-02 17:54:47.
  - It stamps each datagram with the hub receive time and source IP.
  - Being transient, it needs a manual restart after a hub reboot (startup
    item H1).

### 7.2 Review fixes before deployment

An adversarial review (wf_dfe597c8-01a, 10-02 ~17:55Z) found 12 should-fix
issues, all fixed before the fleet install:

1. Seeking to the end of kmsg skipped earlier records. It now reads the
   whole boot.
2. No directory fsync after the state-file replace.
3. A hostname target meant blocking DNS before the reboot. It now resolves
   once at start.
4. The reboot fired before the UDP note left. It now sleeps 0.1 s first.
5. An unwritable state file failed open to armed. It now goes watch-only.
6. Unclean boots with no time window would disarm it. A 6 h window was
   added.
7. Non-Fatal AER matched any device. It now requires a Mellanox device.
8. `mlockall` and the sysrq open now happen at start.
9. The other SMMU failure lines were added.
10. The mlx5 `PCIe Bus Error` line was added.
11. `After=` ordering was added.
12. `install` now skips the self-copy and checks `NRestarts`.

### 7.3 Verification and catches

- **Tests on spark8.**
  - 17:54:50: dry-run injection.
  - 18:06:26: armed reboot from a line injected from user space into a test
    instance (`--accept-user`). The hub received it in 19 ms.
  - Trigger to the next boot's first journal entry: 94 s.
- **Real triggers.** 13 between 19:07:20 and 23:20:03 on 10-02:
  - spark7 9, spark9 2, spark8 2; zero on the non-Lenovo nodes.
  - By reason: `pcie-aer-nic-uncorrectable` 12, `smmu-cmd-sync-timeout` 1.
- **Disk after every guarded crash.**
  - Root `Filesystem state: clean`, 0 EXT4/I/O/nvme-timeout lines, and
    `fsck.fat -n` on the ESP made no changes.
  - Only the dirty bit and the journal tail were lost. The node's Ceph OSD
    went down temporarily.
  - These checks cannot prove file-content integrity. A successful root
    mount does validate group descriptors.
- **Timing, measured on every guarded crash.**
  - Trigger to the next boot's first journal entry: 89–95 s.
  - Trigger to guard re-armed: 144–155 s.
  - One crash attempt takes about 4–5 minutes end to end.
  - Hub latency, trigger to first detail line: 31–35 ms in 11 events;
    1.064 s at 23:01:31.
- **Journal loss at each sysrq-b.** spark7 lost 3–34 s of journal tail,
  spark9 2–23 s and spark8 4–26 s. Crashed boots leave the ESP marked
  not-cleanly-unmounted.

### 7.4 Limits

1. **Late arming.** The guard arms 60 s after boot, and engines can start
   first (spark7 19:09:40 engines vs 19:09:44.6 guard).
2. **It needs a log line.** A node that wedges without printing a trigger is
   not reset. A CMD_SYNC-first event is visible only about 1 s after the
   stall began.
3. **In-flight NVMe commands** are not covered: a short window remains.
4. **Watch-only is unsafe for crash work.** A crash is then not rebooted.
   `fleet_ready.sh` re-arms a watch-only guard by trimming and restarting it.
   `guard_rearm.sh` only trims, which affects the next start.
5. **Hub log coverage.** The hub log starts 10-02 17:54Z, so the two spark8
   incidents are only in node logs.

## 8. The reproducer (surekill) and its harness

### 8.1 Method (owner, 10-02 20:50)

1. Reproduce the crash reliably.
2. Change exactly one thing.
3. Re-run, then recheck the baseline.

A door campaign is worth running only if it measures per-door bandwidth and
latency, not just crash/no-crash. A change passes after 2 survivals followed
by a baseline re-crash.

### 8.2 Runner

`glmfull-stream-kit/scratch/surekill.sh LABEL [SW] [PR]`, with `WAVE_ROWS`
from the environment (default 1024).

**Steps**

1. `fleet_ready.sh`: abort unless `READY 16/16 armed`.
2. Snapshot RDMA statistics and weightd `WD-MESH` timing, CQ and stats on
   all 16 nodes.
3. Start 0.5 s `ethtool -S` collectors on spark6–9, plus a tail of spark7's
   weightd and residentd logs.
4. `ceph -s`; record the hub log line count.
5. `ab_arm.sh`:
   - set Ceph flags;
   - re-arm guards;
   - re-apply pair door B;
   - set the mesh doors;
   - start production engines (which creates the arena);
   - run `ovl_gate.sh t1,ttft16k,pk,pk128`.
6. Stop the collectors; collect hub trigger and detail lines.
7. On each crashed host: wait for ssh, then record:
   - `tune2fs` state and error count;
   - `fsck.fat -n` on the ESP;
   - a dmesg error count;
   - the previous boot's last kernel line;
   - Ceph down OSDs.

**Verdict**

- **CRASH** on any guard trigger.
- **SURVIVED** if no trigger and at least 100,000 completed prefill tokens
  (sum of `prompt_len` for requests with a TTFT).
- **INCONCLUSIVE** otherwise. It exits 2, which stops a series.

**Gate and crash depth**

- The full gate is 362,273 prefill tokens: t1 13 + ttft16k 16,308 + pk
  115,970 + pk128 229,982.
- Completed-token counts at a crash map to the request in flight: 13 =
  inside ttft16k; 16,321 = inside pk8k; 24,320 = inside pk16k; 72,308 =
  inside pk60k.

### 8.3 Harness pieces

| Script | Role |
|---|---|
| `fleet_ready.sh` | Hub listener active; current-boot guard journal says `mode=armed` on 16/16; re-arms watch-only nodes; 60 × 15 s. |
| `guard_rearm.sh` | Trims `boots.json` to the current boot on 16 nodes; checks the service is active. Does not restart. |
| `ab_arm.sh` | One arm: Ceph flags, re-arm, door B pair, doors, engines, gate. **Leaves Ceph `noout`, `nobackfill` and `norebalance` set**; they stayed set until 10-03 ~08:00Z. |
| `ovl_gate.sh` | Build fetch and checksum, root-superblock check (magic at two offsets, page cache and direct I/O), stage, stop production, memory wait, `perf.sh` lane 12, grade, restore production. |
| `mesh_doors.sh` | Switch RDMA doors through a `/run` drop-in; fresh weightd; door B down then up; rings 1024; PFC/DSCP check. |
| `surekill_hook.sh`, `surekill_series*.sh` | Pre-run hook variant; N-run series that stops at the first non-zero exit. |
| `chain_*.sh` | Unattended sequences: passthrough apply and revert with boot-id and domain-type verification, test series, the Lenovo rollout. |

### 8.4 Harness bugs found on the way

| Bug | Fix |
|---|---|
| macOS `wc -l` pads its count, so `[ "$n" = 16 ]` never matched. Every door switch in the 20:40 campaign "failed" at 16/16. | `wc -l \| tr -d ' '` |
| `snap()`'s bare `wait` also waited on the never-ending collectors. | Wait on its own job PIDs |
| Editing a running bash script is unsafe (bash reads incrementally). | Edit only past the affected point, or make a copy (`surekill_hook.sh`) |
| `pkill -f pattern` inside ssh killed its own shell. | `[x]pattern` or `pkill -x` |
| The passthrough grub check looked for an adjacent literal string. | Regex `iommu.passthrough=0.* iommu.passthrough=1` |
| perf preflight required 50 GiB free, and spark0/2 had 48–49 after stopping production. | `PERF_FLOOR_GIB=8`, same workload. Seven nodes have about 2 GiB less MemTotal (spark0, 2, 4, 7, 8, 9, b). |
| Door B rings of 8192 × 20 channels cost about 2 GiB per node and failed the 48 GiB preflight. | `mesh_doors.sh` sets door B rings to 1024 |
| `spark_fabric_door.sh up-switch-b` fails on a door that is already up (GID 3 empty). | `mesh_doors.sh` runs `down-switch-b` first. **The merged tool is still unfixed.** |
| The `ovl_gate.sh` error regex `iommu.*fault` matches `iommu: Default domain type`. | Open; its error count is meaningless |
| Void probes printed success (`tcpprobe-2344Z` "no crash" with zero load; `md8-0049Z` "crash=none" with every lane failed). | Fixed in the scripts; the runs are excluded |

## 9. Every run

Configurations:

- **2B** = cd9786e two-band (`WAVE_ROWS=1024`, overlap on, lane 12 bands
  24+25).
- **1B** = cd9786e with `WAVE_ROWS=512` (overlap off).
- **Doors:** C0 = switch and pair on link A (default); C2 = switch on link B
  (0002:01:00.1), pair on link A.
- **IOMMU s7/s8/s9:** T = Translated (DMA-FQ), PT = identity. The 13
  non-Lenovo nodes were T throughout.

"Into request" is seconds after that request was accepted, from sparkf
monotonic time (window 0.35–1.0 s).

### 9.1 Valid runs

| # | Run | Config | IOMMU s7/s8/s9 | Change | Outcome | Node, function | Crash time | Depth |
|---|---|---|---|---|---|---|---|---|
| R01 | pair2-20261001T1559Z | e8e973d two-band | T/T/T | — (no guard) | CRASH | spark8 01:00.1 CmpltTO | 10-01 16:00:49.355 | 0.2–0.4 s into ttft1k; 5,134 tokens before; superblock corrupted |
| R03 | ovl-20261002T1210Z | 2B C0 | T/T/T | — (no guard) | CRASH | spark8 01:00.1 CmpltTO | 12:13:44.079 | pk60k +14.6–15.1 s; 75,402 tokens; GDT corrupted |
| R05 | ab-pt1-1906Z | 2B C0 | T/PT/T | — | CRASH | spark7 **01:00.0** (type not recorded) | 19:07:20.629 | ttft16k +8.4–9.4 s (7 waves) |
| R08 | ab-c0-pt2-1919Z | 2B C0 | T/PT/T | — | CRASH | spark7 01:00.1 CmpltTO | 19:20:19.619 | pk8k +2.6–3.4 s; 16,321 tokens |
| R17 | sk-base1 | 2B C0 | T/PT/T | — | CRASH | spark7 01:00.1 | 21:45:08.204 | ttft16k wave 1 |
| R18 | sk-base2 | 2B C0 | T/PT/T | — | CRASH | spark7 01:00.1 | 21:50:26.845 | ttft16k wave 10 |
| R19 | sk-base3 | 2B C0 | T/PT/T | — | CRASH | spark7 01:00.1 | 21:55:42.168 | ttft16k wave 14 |
| R20 | sk-x1-single-band-1 | **1B** C0 | T/PT/T | single band | **SURVIVED** | — | — | 362,273 tokens; TTFT16k 22.92 s |
| R21 | sk-x1-single-band-2 | **1B** C0 | T/PT/T | single band | **SURVIVED** | — | — | 362,273 tokens; TTFT16k 23.01 s |
| R22 | sk-base-recheck1 | 2B C0 | T/PT/T | — (same spark7 boot as R20/R21) | CRASH | spark7 01:00.1 | 22:27:32.925 | ttft16k wave 2 |
| R23 | sk-x2-passthrough-spark7-1 | 2B C0 | **PT**/PT/T | spark7 passthrough | CRASH | spark7 (identity) 01:00.1 | 22:37:07.991 | pk60k; 72,308 tokens |
| R24 | sk-x2-passthrough-spark7-2 | 2B C0 | PT/PT/T | spark7 passthrough | CRASH (spark7 survived) | spark9 (T) 01:00.1, its first ever | 22:42:42.807 | pk8k; 16,321 tokens |
| R27 | sk-x5-cto-65-210ms-r1 | 2B C0 | T/PT/T | completion timeout (blocked → baseline) | CRASH | spark7 **SMMU0 CMD_SYNC, no AER** | 22:55:51.929 | ttft16k +4.8–5.8 s |
| R28 | sk-x5-cto-65-210ms-r2 | 2B C0 | T/PT/T | completion timeout (blocked → baseline) | CRASH | spark7 01:00.1 (detail +1.06 s) | 23:01:31.224 | pk8k; 16,321 tokens |
| R29 | sk-t1-switch-on-linkB-r1 | 2B **C2** | T/PT/T | switch traffic on link B | CRASH | spark8 (identity) **0000:01:00.0 and 0002:01:00.1** within 30 ms | 23:06:50.660 | ttft16k +0.5–1.5 s |
| R31 | sk-base-recheck2 | 2B C0 | T/PT/T | — | CRASH (spark7 survived) | spark9 (T) 01:00.1 | 23:14:15.096 | pk60k; 72,308 tokens |
| R32 | sk-t2-spark7-tx-cap-20g-1 | 2B C0 | T/PT/T | spark7 TX cap (not enforced → baseline) | CRASH (spark7 survived) | spark8 (identity) 01:00.1 | 23:20:03.231 | pk16k; 24,320 tokens |
| R36 | tcpprobe-2350Z | TCP flood | PT/PT/PT | about 110 Gb/s TX per node, 300 s | NO TRIGGER | — | — | — |
| R37 | sk-gap-recheck-single-band | **1B** C0 | PT/PT/PT | miners gone | **SURVIVED** | — | — | 362,273 tokens; TTFT16k 22.16 s |
| R38 | tcpprobe-0006Z | TCP flood + GPU stress | PT/PT/PT | 101.6–104.2 Gb/s TX + 142 GB/s GPU memory | NO TRIGGER | — | — | — |
| R42 | md8-20261003T0104Z | 8x GLM Flash TP16 | PT/PT/PT | multi-driver | NO TRIGGER | — | — | NIC p50 4.5 Gb/s |

R39 (TCP vs RDMA benchmark on non-Lenovo nodes) is not a crash test.

### 9.2 Runs not counted

| Run | Reason |
|---|---|
| R02 ovl-1207Z (9dfd62e) | Functional rejection after about 3 s of two-band; no crash |
| R04, R06, R07 | Empty directory; engines dead at start; engines did not start |
| R09–R12 door campaign | Padded `wc -l` broke every door switch; arm 4 abandoned |
| R13, R15 | Readiness only; stopped before the gate |
| R14, R16 | Preflight memory floor (50 GiB) |
| R25, R26 | Preflight 48 GiB, short because of door B rings |
| R30 | Door B bring-up failed on 15 nodes (idempotency bug) |
| R33 | Stopped at the door switch |
| R34, R35 | iperf3 servers absent; no load |
| R40, R41 | Lazy experts: batch path bug; then no batch finished in 10 minutes |

### 9.3 Tallies

**By configuration**

| Configuration | Valid runs | Crashes |
|---|---|---|
| e8e973d two-band, all T | 1 | 1 (spark8 T) |
| cd9786e 2B, any IOMMU, any doors | 14 | 14 |
| cd9786e 1B (512-row waves) | 3 | 0 (1,086,819 tokens) |
| TCP flood ± GPU stress, Lenovo PT | 2 | 0 |
| 8x Flash multi-driver, Lenovo PT | 1 | 0 |

**All 15 events**

- By node: spark7 9 (8 T + 1 PT), spark8 4 (2 T unguarded + 2 PT), spark9 2
  (both T), non-Lenovo 0.
- By IOMMU mode: Translated 12, identity 3.
- By PCI function:
  - 0000:01:00.1 (link-A switch): 12;
  - 0000:01:00.0 (link-A pair): 2;
  - 0002:01:00.1 (link-B switch): 1, inside the same event as one of the
    0000:01:00.0 ones;
  - SMMU-first with no AER: 1.

**Depth at crash** (14 cd9786e crashes)

- 7 inside ttft16k, under 16.3K tokens.
- 4 in pk8k or pk16k, at about 19–27K.
- 3 in pk60k, at about 83–88K.
- Per wave, the crash chance looks flat: crashes came in waves 1, 10, 14
  and 2 (4 of 27 waves), not after a build-up.

### 9.4 Exposure and rates (INFERENCE)

Exposure is completed tokens plus the partial request at about 820 tok/s.
A run ends at the first crash on any node, so every survival figure for the
other nodes is censored. Every cell is small (n ≤ 8), so all intervals are
wide.

| Node / mode | Crashes / two-band exposure | Rate |
|---|---|---|
| spark7 Translated | 8 / about 274K | about 1 per 34K |
| spark7 identity | 1 / about 106K | — |
| spark8 Translated | 1 / about 88K | — |
| spark8 identity | 2 / about 293K | about 1 per 146K |
| spark9 Translated | 2 / about 380K | about 1 per 190K |
| Each non-Lenovo node | 0 / about 380K | — |
| **Lenovo Translated, pooled** | 11 / about 741K | **about 1 per 67K** |
| **Lenovo identity, pooled** | 3 / about 399K | **about 1 per 133K** |

- **Passthrough lowers the rate about 2x pooled** (about 3x for spark7
  alone). The difference between Lenovo units (spark7 vs spark9, both
  Translated) is as large as the IOMMU effect.
- **Single-band exposure with zero crashes:** spark7 T 724K, spark9 T 724K,
  spark8 PT 1,087K, spark7 PT 362K, spark9 PT 362K. At the pooled two-band
  rates, about 22 crashes would be expected for the Translated Lenovo
  single-band exposure and about 14 for identity.
- **spark7's rate shifted for no known reason.**
  - It survived about 87K two-band tokens at 12:11 (R03), then crashed in 8
    straight Translated runs from 19:07 to 23:03, about 1 per 9.6K.
  - On boot cb2ad2f0 it then survived about 110K (R29, R31, R32). At its
    earlier rate that has about a 4% chance.
  - Candidate causes between 12:11 and 19:07, all untested: spark8 moved to
    identity; the guard was installed; Ceph flags were set; the gate
    dropped its warm-up sessions.
  - The DB miners do not explain it: they ran only on spark0–6, from 19:28
    to 23:21.

## 10. What each single change showed

### 10.1 Single band: survives, but confounded

- **Results.**
  - R20 and R21 (spark7 Translated) and R37 (Lenovos in identity) survived
    full gates.
  - R22 crashed 2.2 s into two-band on the same spark7 boot that had just
    survived about 1,233 s of single-band prefill. So boot state and run
    order are not the explanation.
- **Confound.** One knob (`WAVE_ROWS=512`) changed three things at once:
  - wave size;
  - the second band;
  - compute/communication overlap.

  The result cannot attribute the effect to band count alone.
- **Production was never gated.** Production runs single band with 1024-row
  waves (0b5371e, 16K TTFT 19.1 s). That configuration has never been
  through a surekill gate.
- **Two-band was not faster.** Two-band TTFT16k (19.41–20.81 s, mean
  19.89 s) is not faster than production. The 22–23 s single-band figures
  come from 512-row waves.

### 10.2 IOMMU passthrough: lower rate, not zero

- **Rate.** About 2x lower pooled (section 9.4).
- **Crashes still happened:** spark7 at 72K; spark8 at 13 tokens and at
  24K.
- **Corruption path.** It removes the stale-IOVA wrong-page path
  (INFERENCE). All three identity crashes left clean disks, as did every
  Translated crash under the guard.
- **Performance effect: unmeasured cleanly.**
  - One spark8 prompt took 26.2 s TTFT in passthrough against about 22.7 s
    expected, while Ceph was rebalancing.
  - The fleet already reaches 98 Gb/s (switch) and 109 Gb/s (pair) with
    `ib_write_bw` in Translated lazy mode, so little RDMA gain is expected.
- **Forum caveat.** On one ASUS GX10, translated mode capped the CX-7 near
  13 Gb/s per function. Our fleet shows no such cap.

### 10.3 Switch traffic on link B: both links failed together

- **Setup.** R29 put the switch RDMA on 0002:01:00.1 (link B) and left the
  pair on link A. spark8 (identity) crashed about 1.1 s into traffic.
- **What timed out.**
  - CmpltTO on 0000:01:00.0 (link-A pair function, about 2.1 Gb/s) and on
    0002:01:00.1 (link-B switch function, about 29.5 Gb/s), within the same
    30 ms detail window.
  - Followed by `mlx5_pci_err_detected` on both links.
- **Interpretation.** Simultaneous completion timeouts behind two different
  root ports point above the root ports: SMMU0, the SoC fabric or the
  memory path (INFERENCE). Moving traffic between PCIe links does not avoid
  the fault.

### 10.4 Longer completion timeout: blocked

- **Attempt.** Raise the CX-7 completion timeout from 50 µs–50 ms to range B
  (65–210 ms): `setpci CAP_EXP+0x28.w=0006:000f`.
- **Result.** Every write failed with `Operation not permitted`. The kernel
  logged `Lockdown: setpci: direct PCI access is restricted`.
- **Consequence.** R27 and R28 are baseline repeats despite their labels.
  The change needs Secure Boot and lockdown off.
- **Even if possible, it is not advisable.** The SMMU-first event shows a
  longer timeout would at most hide a symptom.

### 10.5 RoCE TX cap: not enforced

- **Attempt.** `mlnx_qos -i enp1s0f1np1 -r 0,0,0,20,0,0,0,0` set
  `tc: 3 ratelimit: 20.0 Gbps` on spark7.
- **Result.** spark7's switch-port RDMA TX still averaged 25.4 Gb/s
  (maximum 31.1). R32 is a baseline repeat, and spark8 crashed in it, not
  spark7.
- **Why the per-TC max rate does not apply to RDMA QPs** on this firmware
  and driver is unexplained.

### 10.6 Door campaign: void

- **Run.** An 11-arm A/B over the four door combinations (C0–C3) ran no
  workload, because of the padded `wc -l`.
- **Owner verdict.** Not useful unless it measures per-door bandwidth and
  latency. Replaced by single-change surekill runs.
- **Bandwidth columns.** Its "transport bw/latency" columns were roofline
  constants, not measurements.

### 10.7 The SMMU-first event

- **R27, spark7 Translated.** The guard fired on
  `CMD_SYNC timeout at 0x0000eaab [hwprod 0x0000eab0, hwcons 0x0000eaab]`,
  then on 0xeaad with hwcons still 0xeaab. No CX-7 AER line came first or
  within 30 ms.
- **Timing.**
  - spark7's last counter sample came 0.94 s before the trigger; the next,
    due about 0.5 s later, never arrived.
  - With a 1 s poll, the stall began around when counter polling stopped
    (INFERENCE).

## 11. NIC counter forensics

Collected by `surekill.sh` as a 0.5 s loop on spark6–9 (real interval
0.52–0.55 s).

**Captured:**

- `vport_rdma_unicast_bytes` and `vport_unicast_bytes`;
- `discards_phy` and `prio3_pause`;
- `outbound_pci_stalled_*` and `*_pci_signal_integrity`;
- NVMe diskstats.

**Not captured:**

- pause durations and `rx_out_of_buffer`;
- correctable AER counts;
- outstanding-read counts;
- SMMU activity.

| Measure (spark7 switch function) | Two-band crash runs | Single-band survival |
|---|---|---|
| Switch TX mean while active | 24.0–24.8 Gb/s | 21.6 Gb/s (ttft16k); 16.1 (whole gate) |
| Switch TX p99 / max | 28.3–29.5 / 28.8–29.7 | 23.9 / 25.7 |
| Link-A TX max | 30.8–31.8 | 27.5–28.1 |
| Pair TX mean | 1.72–1.77 | 1.54 |
| Rate variability (0.5 s std/mean) | 0.09–0.125 | 0.061–0.066 |
| s per 1024 rows | 1.265–1.284 | 1.417–1.427 |
| PFC pauses received per active s | 629–931 | 2,247–2,308 |
| PFC pauses per GB received | 209–410 | 850–853 |
| `outbound_pci_stalled` / signal integrity / discards | 0 / 0 / 0 | 0 / 0 / 0 |
| NVMe writes in the run window | 0.4–18 MB | 1,024–1,490 MB per gate |

Findings:

- **Rate.** Two-band moved about 11% more bytes per second: the same bytes,
  compressed in time.
  - A pure 0.5 s-rate threshold near 22 Gb/s is ruled out. The survival runs
    spent 125 intervals at 22 Gb/s or more; 7.4–9.8 crashes were expected;
    P(0) = 0.001.
  - A threshold at 23–24 Gb/s is not ruled out (P(0) = 0.03–0.19). Survival
    spent only about 21 s above 23 Gb/s and never exceeded 26.3.
  - Within crash runs the hazard looked flat from 23 to 30 Gb/s: 13
    intervals at 27 Gb/s or more produced 1 crash (INFERENCE).
- **Pause frames.** Crash runs saw 2.5–4x fewer PFC pauses per byte than
  survival runs, so pause storms do not track the crash.
- **No precursor in the stall counters.** They read 0 in every sample,
  including samples taken after the AER. `outbound_pci_stalled` counts
  posted-credit starvation, not completion latency, so it would not see
  CmpltTO anyway.
- **NVMe load does not explain it.** Crash runs had far less NVMe activity.
  Most of spark7's NVMe writes in survival runs were weightd trace logging.
- **Symmetric load.** spark6–9 moved the same RDMA bytes in each run (for
  example 294.0 GB each way in R23). Nodes that never crashed ran the same
  or higher rates.
- **CQ errors.** Only base1 shows `WD-MESH-CQERR` (2 × RDMA_WRITE
  `IBV_WC_LOC_PROT_ERR`, vendor 81), untimestamped.
- **Mid-chain stalls.** Lenovo nodes closed 101 of 201 node-windows whose
  worst gate was a stall of 40 ms or more. The 44–59 ms events closed by
  spark7, spark8 and spark9 each fall in a run where that node then
  crashed. They are probably the CX-7 stalling before the crash, not a
  steady population of slow transfers (INFERENCE).

## 12. Configuration and firmware comparison

### 12.1 Identical on all 16 nodes

- CX-7 firmware and all 218 NV parameters on all four functions.
- PCIe config space of the CX-7 functions and root ports (section 2.2).
- Port and QoS settings:
  - MTU 9000, 20 channels, rings 8192, adaptive coalescing 8 µs/128, RS FEC;
  - switch port: DSCP trust, PFC on priority 3 only, buffers
    262080/459360;
  - pair port: global pause, PFC off, PCP trust;
  - IRQ pinning, GID 3 RoCEv2, identical QoS service file.
- ACPI IORT, MCFG, PPTT, GTDT, FACP and SSDT1–5 (byte-identical across
  GIGABYTE, MSI and Lenovo).
- SMMU features and queue sizes.

### 12.2 What differs

| Item | Lenovo spark7/8/9 | Others | Tracks the crash? |
|---|---|---|---|
| UEFI build | S0QKT09A (Lenovo UEFI 1.09, AMI 0ACUM10 5.36.01.09, 2025-10-14) | AMI 5.36_* reference builds, several versions | Lenovo vs rest |
| DSDT | NVDA8800 "MTEL" CPU power/DVFS device differs (CPUEB, SSPM, SPBM, FASTDVFS, CPPC; different RBUF window; extra `_DSM` with PLID; added `CPUEB_BPT_LOG_OFFSET`; no HFRP); extra package-thermal `_DSM`; TPM flags; GNVS address | AMI reference | Lenovo vs rest |
| MADT | SPE overflow interrupt 0 | 21 | Lenovo vs rest (no DMA path) |
| CX-7 VPD board serial | 1584024600048 on all three (provisioning artifact) | unique | Lenovo vs rest (no DMA path) |
| TPM | present (7.2.4.0 / 7.2.4.1) | — | Lenovo vs rest |
| IOMMU mode at 10-02 22:48Z | spark8 identity; spark7/9 DMA-FQ | DMA-FQ | spark8 vs 7/9 only |

### 12.3 Rejected discriminators

None of these separates crashers from survivors:

- pair-port rx buffer size (auto-sized at link-up);
- IPv6 address generation mode;
- the pair cable (spark8↔spark9 use a 2 m copper cable, the others 0.5 m);
- the switch breakout leg;
- NVMe model or MRRS;
- PCIe parity bits and VSEC dwords;
- kernel 1029;
- sysctls;
- switch-side PFC pause counts.

The owner's "different port configuration" hypothesis is not supported.

### 12.4 Firmware inventory, 10-03 (A1 of the BIOS memo)

`fwupdmgr get-devices`, DMI type 0, ESRT and `lspci` were read on all 16
nodes.

| Nodes | Vendor | UEFI date | EC version |
|---|---|---|---|
| spark7, spark8, spark9 | Lenovo | 2025-10-14 | 2.66 (DMI firmware revision; fwupd shows a placeholder) |
| **sparkb** | GIGABYTE | **2025-10-23** | **0x02004203 = 2.66.3** |
| spark4 | GIGABYTE | 2025-10-23 | placeholder (0x00000001) |
| spark0, spark2 | GIGABYTE | 2025-11-07 | 0x02004b03 = 2.75.3 |
| spark3, spark5, spark6, sparka | GIGABYTE | 2026-03-22 | 0x02004e18 = 2.78.24 |
| sparkd, sparke, sparkf | GIGABYTE | 2026-07-02 | 0x03000508 = 3.5.8 |
| spark1 | MSI | 2026-03-16 | 10600 (vendor format) |
| sparkc | Dell | 2025-10-08 | 0x01000a00 (vendor format) |

Findings:

- **EC.** The EC versions are the NVIDIA reference EC line shared across
  vendors (Lenovo's own releases ship 2.66.3, 2.75.3, 2.78.24 and 3.5.8).
  sparkb runs **the same EC version (2.66.3) as the Lenovo units**, a UEFI
  from the same month, and the same kernel (1026). It had zero events under
  identical exposure. So the EC version and UEFI age are not the
  discriminator.
- **SoC firmware cannot be compared.** Every node's ESRT "UEFI Device
  Firmware" entries show placeholder versions (1 or 0x00000001), so the SoC
  firmware level that Lenovo 1.14 would replace is not visible.
- **ATS.** No CX-7 function has an ATS capability on any node (ATSCtl
  absent). The upstream ATC-invalidation quarantine series does not apply.
- **PCIe config.** The completion timeout (50 µs–50 ms) and MPS/MRRS 512
  are the same everywhere.

## 13. Side experiments

### 13.1 TCP crash probes (Lenovo units in identity)

**Setup**

- `tcp_crashprobe.sh`: 300 s of `iperf3 -P 8 -l 1M --bidir` from spark7,
  spark8 and spark9 on both ports at once, with the default DSCP.
- Optional `GPU_STRESS=1` runs `gpu_mem_stress` on each Lenovo node: a 4 GiB
  device-to-device copy plus D2H/H2D copies of 1 GiB through pinned host
  memory.

| Probe | GPU stress | NIC TX per node | Triggers |
|---|---|---|---|
| tcpprobe-2350Z | no | 110.2–110.6 Gb/s (pair about 51 + switch about 59) | none |
| tcpprobe-0006Z | 142.0–143.2 GB/s for 308 s | 101.6–104.2 Gb/s | none |

**Reading**

- TCP drove about 4x the crashing RDMA rate through the same NIC, PCIe
  links and SMMU0 without a trigger.
- GPU stress cut switch-port receive from about 43 to about 27 Gb/s and
  raised retransmits (host CPU and memory contention).

**Not tested**

- Translated Lenovo nodes.
- Real GPU prefill compute instead of synthetic copies.
- GPU and NIC touching the same host pages.

The note's "4–5 crashes expected in that exposure" is an extrapolation
across transport and IOMMU mode.

### 13.2 TCP vs RDMA benchmark

**Setup.** Pair path spark0 (GIGABYTE) ↔ spark1 (MSI); switch path spark0
↔ spark3 (GIGABYTE). Tools: `ib_write_lat -x 3 -F`, `ib_write_bw` (12 MB,
4 QPs), iperf 2 bounce-back and iperf3. All TCP ran with the default DSCP.

| Size | RDMA one-way, pair (typ / p99 µs) | RDMA one-way, switch (typ / p99 µs) | TCP |
|---|---|---|---|
| 12,288 B | 4.59 / 4.72 | 5.86 / 5.99 | Bounce-back round trip: pair average 353 µs (max 1,551); switch average 663 µs (max 1,992) |
| 196,608 B | 20.21 / 20.41 | not run | bounce-back failed |
| 786,432 B | 64.19 / 64.49 | 69.44 / 69.88 | bounce-back failed |
| 3,145,728 B | 236.67 / 237.05 | not run | — |
| 12,582,912 B | 930.12 / 931.23 | 1,032.40 / 1,033.61 | — |
| Throughput | 111.9 Gb/s (pair, 4 QPs) | — | pair 57.2 (1 stream) / 110.4 (4 streams) Gb/s; switch 80.4 / 98.9 Gb/s |

- **RDMA tails are tight:** p99 within 1–2% of typical.
- **Decode must stay on RDMA.** TCP small-message round trips are about
  38–57x the RDMA equivalent, with millisecond outliers (INFERENCE).
- **4-stream TCP reaches wire parity for large messages:** 12.6 MB in about
  0.92 ms (pair) and 1.02 ms (switch), at about 2 cores of system CPU per
  side.

### 13.3 All-reduce gate tails and the DB miners

**Before** (19:20–23:16 on 10-02, about 19.5M gates)

- 69% of busy 10 s windows had a gate p99 of 2 ms or more. The worst gate
  per window had a median of about 18 ms.
- RDMA transfer time (ship) p99 was under 1 ms in every window, with 0
  retransmits and 0 timeouts.
- The waits were late senders. spark0–6 (7 of 16 ranks) carried 70.7% of
  the excess wait and closed 87% of the window-worst gates.
- Those nodes were running owner DB-mining workers: 16 `sf17_arm` processes
  at 100% per node, about 5.5 GB RSS, heavy memory-compaction stalls, and
  hottest zones near 98 °C.

**After** (miners stopped at 23:20:52, single-band recheck 23:55–00:06)

- Windows with gate p99 of 2 ms or more fell to 4%.
- Worst gate per window: median 9.7 ms, maximum 48.7.
- spark0–6 share of the excess fell to 46.9% (fair share 43.75%).
- 16K TTFT went from 22.92/23.01 to 22.16 s.

**Reading**

- The millisecond gaps were host-side lateness, not slow RDMA, and not
  specific to non-Lenovo NICs.
- Passthrough did not change gate tails (spark8 identity mean gate 453 µs
  vs sparka 455 µs).
- Single-band 16K prefill spends 30.8% of wall time in peer gates.
- The weightd doorbell thread is an ordinary unpinned thread.

### 13.4 Eight drivers at once

**Setup** (01:04–01:35 on 10-03)

- 8 GLM-5.3 Flash fp8 TP16 engines on lanes 1, 2, 3, 4, 5, 7, 8 and 9;
  production lane 6 stopped and restored afterwards.
- Experts pinned in one shared 24 GiB arena per node (graph path), fresh
  weightd, Lenovo units in identity.
- Lazy experts (the owner's spec) completed no batch in 10 minutes and were
  replaced by pinned shared experts. K3 x8 does not fit at lane defaults.

**Phase A: decode**

- Batches of 8 requests × 64-token prompts × 256 output tokens.
- Every lane finished 2,048 tokens in 525.5–532.0 s, about 2.05 s per 8-row
  step. Start skew was 10 ms or less.

**Phase B: prefill**

- Batches of 16 × 1,280-token prompts.
- Every lane finished 32 prompts (40,960 tokens) at about 47.6 prompt
  tokens/s per lane; 327,680 prefill tokens in total.

**Results**

- Outputs were token-identical across all 8 lanes in both phases.
- NIC switch TX on spark6–9: phase A p50 3.2 Gb/s (max 12.4); phase B p50
  4.6 Gb/s (max 5.2); zero PFC pauses.
- Worst gate per window: median 30.5 ms, max 115.6 ms (GPU time-slicing
  skew).
- 0 guard triggers.

**Reading**

- Multi-driver over RDMA works as tested.
- The GPU was the bottleneck: eight engines time-slice one GPU, so the NIC
  carried less than a fifth of the crashing rate.

### 13.5 Telemetry before two crashes

- **Sampling.** GPU every 500 ms (SM clock, power, temperature, throttle
  reasons) and host every 0.5 s (thermal zones, CPU frequency), on spark0,
  spark6, spark7, spark8 and spark9 from 23:13:44.
- **Before spark9's 23:14:15 crash** (last samples 0.27 s before the
  trigger):
  - GPU `clocks_event_reasons` 0x0, SM 2457–2476 MHz, 45–55 W;
  - hottest zone 76–81 °C;
  - CPU frequency constant.
- **Before spark8's 23:20:03 crash:** hottest zone 60–69 °C, GPU 2502–2522
  MHz, no throttle reason.
- **Overall.** All 84,622 GPU samples through 01:36 show zero throttle
  reasons.
- **Caveat.** `scaling_cur_freq` never moves on these nodes, so CPU
  throttling is not observable this way.

### 13.6 Earlier concurrency precedent

On 09-30, 3–5 GLM Full TP16 dev lanes ran at once on all 16 nodes for about
2.5 hours with no AER. Their waves were 8–17 rows, about 60x less data per
collective than the 1024-row crash waves.

## 14. External evidence

- **"AER Storm on GX10"** (NVIDIA forum 384436, late September 2026). One
  ASUS GX10 and an HP ZGX:
  - logged fatal Surprise Down on both CX-7 root ports, then mlx5 recovery
    soft lockups; identical peers stayed clean;
  - NVIDIA staff blamed a PCIe power-rail disturbance or PERST# event on the
    unit and pointed to fieldiag and an OEM RMA;
  - `iommu.passthrough=1` did not stop it;
  - the same thread called passthrough "the recommended setting" and showed
    translated mode capping the CX-7 near 13 Gb/s per function.

  It is a different signature from ours: no SMMU stall and no corruption.
- **MiaAI-Lab DeepSeek-v4-Flash-DSpark issue #146** (ASUS GX10, TP=2, the
  same CX-7 firmware). Non-Fatal CmpltTO on one root port plus a fatal
  Surprise Down on the other, on the head node only. No fix.
- **"Total host freeze during multi-node TP=2 vLLM prefill"** (NVIDIA forum
  376882). Same kernel 1026 and driver; heavy uncached prefill over the
  NICs froze one node five times. Resolved by exchanging the units.
- **Not found anywhere:**
  - a GB10 report of an arm-smmu-v3 CMD_SYNC timeout;
  - filesystem corruption after a PCIe or SMMU error;
  - an NVIDIA statement linking CX-7 CmpltTO to the SMMU;
  - a GB10 SMMU erratum.
- **Upstream "iommu/arm-smmu-v3: Quarantine broken devices"** (Nicolin Chen,
  NVIDIA; v5 2026-07). It names stale translations after a skipped ATC
  invalidation as a direct path to silent corruption. It covers only ATS
  devices and is not in 6.17. Our CX-7 has no ATS.
- **Same class elsewhere.** The wrong-page-on-IOVA-reuse class is documented
  for VT-d (a 6.12 stable lazy-flush race), not SMMUv3.
- **Ubuntu defaults.** linux-nvidia 6.17 defaults to passthrough
  (LP #2129776), and the DGX OS image adds `iommu.passthrough=0`. No
  rationale for that file was found.
- **Other software ruled out:**
  - the `cx7-pcie-hotplug` driver (fails to probe on every spark8 boot);
  - CX-7 firmware release notes (nothing on completion timeout);
  - DOCA's firmware updater (never installed; it bricked one ASUS CX-7).

## 15. Mitigations in force and what they cost

| Mitigation | What it prevents | What it does not prevent | Cost today |
|---|---|---|---|
| One collective in flight per engine | The only known reproducer: two concurrent collectives in one engine | Several engines on one node each having a collective in flight at the same time | None measured: the overlap gave no speedup |
| DMA guard (16/16) | Disk corruption after a detected event. 13 catches, 0 damaged disks. | The crash itself; wedges without a log line; the first 60 s after boot | One node reboot per event (about 90 s plus engine restart); TP16 serving stops meanwhile |
| Lenovo identity mode | The stale-IOVA wrong-page path (INFERENCE); about half the crash rate | The CX-7 completion timeout | No DMA isolation or SMMU fault log for the CX-7, NVMe and USB on three nodes |
| No RDMA serialization across drivers | — (owner decision to keep performance) | — | — |
| TCP for large collectives | Not built. The existing TCP all-reduce (`ring/transport/tp_collective.c`) is K3-only and not graph-compatible. | — | — |

## 16. Implications

### 16.1 Is it "RDMA too fast"?

The data says it is not raw speed, but it is load-dependent:

- **Against pure rate:**
  - TCP at about 110 Gb/s per Lenovo node (4x the crash rate) and
    `ib_write_bw` at 98–109 Gb/s on other nodes never failed;
  - single band at 21.6 Gb/s mean (peaks 25.7) survived over 1M tokens;
  - a pure threshold near 22 Gb/s is statistically ruled out.
- **What does correlate:**
  - two concurrent collectives;
  - about twice the outstanding NIC reads of host memory;
  - higher burstiness (rate variability nearly doubled);
  - host memory that the GPU writes while the NIC reads it.

  The TCP probes had none of the last item: the NIC read socket buffers the
  CPU wrote.
- **The fault point is above the PCIe links.** Both links timed out at once.
- **Rate is not fully excluded.** A threshold at 23–24 Gb/s of this RDMA
  pattern cannot be ruled out.

The owner's concern still stands (INFERENCE):

- Every route to prefill roofline raises RDMA concurrency and rate on the
  same SMMU0 path.
- The guard and passthrough are containment, not a fix.
- "If it hurts, don't do it" will bind once concurrency is the lever.
- In TP16 every node is in every collective, so a limit on three Lenovo
  nodes is a limit on the whole fleet.

### 16.2 Multi-driver is the real exposure

The rule "one collective in flight per engine" does not bound collectives
in flight per node. N engines each running prefill on one node put N
collectives in flight at once, which is the same pattern as two-band.

- The 8-driver run passed only because the GPU was the bottleneck: each
  lane ran slowly and the NIC carried about 4.5 Gb/s.
- Two to four fast drivers on a Translated Lenovo node is untested. It is
  the case most likely to reproduce the crash in normal use.
- Until it is tested, treat simultaneous heavy prefill from several engines
  on spark7/8/9 as hazardous. The guard contains it.

### 16.3 Prefill and the roofline

GLM Full 16K prefill takes 19.1 s (about 850 tok/s) on single-band
production. Prefill is transport-bound:

- True bytes sent per rank are about 65.4 GB for 16K. The scorecard's
  88.67 GB overcounts by 1.36x.
- The one-port wire floor is about 5.0 s, so production runs at about 26%
  of that floor.
- Closing the gap needs about 4x today's mean NIC rate (21.6 to about
  95 Gb/s on the switch function).
- It also needs overlap of compute and communication, which means more
  concurrency.

Both levers push into the region where the Lenovo units fail.

### 16.4 The second PCIe link

- **Capacity.** Link A carries both ports at about 13.6 GB/s per direction.
  Today the switch traffic (14/15 of bytes) and the pair traffic share it.
  Moving the switch function to link B doubles host bandwidth.
- **On non-Lenovo nodes** this is a real lever, worth up to 2x NIC host
  bandwidth once rates approach link capacity.
- **On Lenovo nodes** it does not avoid the fault (R29) and probably raises
  the hazard by adding concurrency.
- **Door B** needs:
  - its idempotency fix;
  - rings of 1024, not 8192 (about 2 GiB per node);
  - port-wide PFC, which is shared with link A on the same physical port.

### 16.5 TCP as a fallback

- **Large collectives:** 4-stream TCP reaches wire parity at about 2 cores
  per side, and 110 Gb/s TCP did not crash identity Lenovo nodes.
- **Not available for GLM.** The device collective accepts only the weightd
  mesh, and graph and linear chain modes need stream-ordered hardware
  waits. A graph-compatible TCP path would have to be a TCP wire inside
  weightd's mesh relay, which does not exist.
- **Decode cannot use TCP:** round trips are 38–57x worse, with jitter.
- **Keep it as a design option** in case the Lenovo fault cannot be fixed.

### 16.6 Decode

- Decode collectives are small (12 KB at B1).
- The 8-driver decode phase ran at a p50 of 3.2 Gb/s without incident.
- Decode is not at risk at current batch sizes.
- Larger-batch decode (B64 and up) and speculation verify waves move toward
  prefill-like transfer sizes and should be included in any future
  Lenovo gate.

### 16.7 Data integrity

- With the guard and identity mode, 13 events caused no detected disk
  damage.
- Residual risks:
  - a node that wedges silently;
  - the first 60 s after boot;
  - the about 1 s delay seen once;
  - NVMe commands in flight.
- Writes on spark8 in both unguarded windows (root files, osd.8 BlueStore)
  may be silently wrong. No completed deep scrub of osd.8 is recorded.

### 16.8 Operations

- Every event costs a TP16 outage of about 90 s plus an engine restart.
- After a reboot:
  - pair door B must be re-applied;
  - peers re-wire through the agent;
  - the Ceph USB OSD may re-enumerate and need `lvchange --refresh`.

  The startup-resilience work (PRs #1429–#1435) targets exactly this.

## 17. The BIOS question

The owner reopened flashing at 10-03 06:58Z. A research memo (workflow
wf_1bfa61ea-65d) and the 10-03 inventory give the following.

### 17.1 What a flash would change

- Lenovo 1.14 (S0QKT0EA, 2026-07-15) replaces three components:
  - UEFI 1.09 → 1.14, which carries the DSDT;
  - EC 2.66.3 → 3.5.8;
  - SoC firmware → 2.0.14.
- Every changelog entry after 1.09 says only "Enhance system stability"
  plus setup, SMBIOS and RTC items. Nothing mentions PCIe, ConnectX-7, AER,
  completion timeout, SMMU or power.
- Lenovo rates it "Recommended", not "Critical".
- The plausible connection is the power and DVFS layer, where the Lenovo
  DSDT differs (INFERENCE).

### 17.2 What the inventory says

- **Against EC age.** sparkb (GIGABYTE) runs the same EC version, a
  same-month UEFI and the same kernel, and never failed. Three survivors
  run October 2025 platform firmware.
- **Still open.** The flash could fix a Lenovo-specific UEFI/DSDT or SoC
  firmware problem, but nothing points to it.
- **Odds** of fixing the fault: unknown, low to moderate.

### 17.3 Risk

- **Brick risk.** Low for an uninterrupted, supervised, vendor-signed
  capsule, but public data cannot bound it. The memo's estimate is about 1%
  or less, with low confidence.
- **How bricks happen.** Almost every public GB10 firmware brick followed an
  interrupted update. No Lenovo PGX brick was found.
- **If it bricks, there is no documented recovery:**
  - no BMC and no EC-reset pinhole;
  - no PGX jumper procedure;
  - NVIDIA's recovery USB is for the Founders Edition and needs a working
    UEFI.

  The outcome is a Lenovo system-board RMA.
- **Interruption** is the hazard we control:
  - power loss;
  - a forced reset;
  - automation power-cycling a node that "looks dead".
- **Remote updates hang** on PGX firmware older than 0DA (ours is 09A). The
  update must be run at the machine with an HDMI display and a keyboard;
  plan for 45 minutes or more of black screen.
- **One-way.** Treat the flash as irreversible: rollback prevention and SoC
  anti-rollback may block a return to 1.09.
- **Possible regressions:**
  - EC 3.5.8 fan and thermal reports on other vendors (none on Lenovo);
  - a different usable-memory figure, which would change SparkPipe sizing.
- **Warranty timing.** The base warranty is one year. Units shipped with an
  October 2025 UEFI may lose cover within weeks. A brick under warranty is
  a repair; out of warranty it is a roughly 5k loss of a scarce unit.

### 17.4 Recommendation

**1. Zero-risk checks first.** About a day; the physical steps are owner
actions.

- **A2: open a Lenovo support case** while the warranty runs. Attach the
  hub log, the failure chain and the inventory. Ask whether 1.10–1.14
  change PCIe, SMMU or DVFS behaviour.
- **A4: cold AC drain at 1.09.** Unplug the brick and peripherals for 60 s
  on one Lenovo, then rerun the reproducer. Every sysrq-b reboot so far was
  warm and kept EC and PD state, and the PD controller has a known
  low-power state that only a cold drain clears.
- **A5: position swap.** Swap a Lenovo and a GIGABYTE in rank, cables and
  switch ports. If the failure stays with the position, the "Lenovo-only"
  premise is wrong.
- **A6: power-input swap.** Swap the 240 W bricks between a Lenovo and a
  GIGABYTE, and run a Lenovo off the UPS.
- **A SparkPipe-free reproducer.** A line-rate multi-QP `ib_write_bw`
  incast on a Lenovo node, in Translated and in identity mode, with and
  without a GPU kernel writing the source buffers. A vendor case needs one,
  and it makes any later A/B cheap.

**2. If the fault follows the unit and stays unexplained, flash one
Lenovo,** after the owner's explicit go:

- Procedure:
  - LVFS capsules through `fwupdmgr`, one device per reboot (system
    firmware first, then the EC; skip PD and TPM);
  - `fwupd` 2.0.20 or later; no setup password;
  - cluster idle; every automation that could power-cycle the node
    disabled;
  - at the machine; 60 minutes hands-off.
- Choice of unit:
  - spark7 gives the most detection power (highest crash rate);
  - spark8 is the memo's choice because it is already quarantined;
  - spark7 hosts a Ceph monitor, which survives with 2 of 3.
- Then A/B in Translated mode against a 1.09 control:
  - pass = zero events over at least 500K two-band tokens;
  - RDMA throughput must match, so a slower NIC cannot pass for the wrong
    reason.

**3. Keep the guard permanently** whatever happens, and track the
arm-smmu-v3 quarantine series.

The 10-02 decision (no flashing) stands until the owner decides.

## 18. Open questions and next experiments

Ordered by decision value per hour of fleet time.

| # | Question | Experiment | Cost |
|---|---|---|---|
| 1 | Is the production configuration (single band, 1024-row waves) safe on Lenovo? | Surekill gate of 0b5371e at 1024 rows, Lenovo Translated and identity | About 15 min per run |
| 2 | Is it band count, wave size or overlap? | cd9786e 2B at 512 rows is impossible (overlap needs more than 512), so run 2985290fd single band at 1024 against cd9786e 2B at 1024 | 2–4 runs |
| 3 | Can a SparkPipe-free load reproduce it? | `ib_write_bw` multi-QP incast at line rate onto a Lenovo node; then with a GPU writer on the source buffers | 1–2 h |
| 4 | Do several fast drivers crash a Translated Lenovo node? | 2–4 GLM Flash or Full engines with real prefill on one node set, Lenovo Translated | 1 h |
| 5 | Does a cold drain, position or power input change it? | A4, A5, A6 (owner physical actions) | Half a day |
| 6 | Why did spark7's rate jump and then drop? | Repeat the baseline at a fixed state; log boot, guard, Ceph and warm-up state per run | Folded into 1–2 |
| 7 | Which collective, band and QP was in flight at the AER? | Timestamped per-wave and per-collective logging off-node (residentd logs are overwritten or NUL-tailed today) | Code change |
| 8 | Does the Ubuntu 1026 arm-smmu-v3 and dma-iommu code match upstream v6.17 (poll timeout, IOVA free after a failed sync)? | Read the linux-nvidia-6.17 source | 1 h, no fleet |
| 9 | Which command sat at the frozen hwcons, and what is the SMMU0 implementer? | Read IIDR; add GERROR/CMDQ_ERR capture | Short |
| 10 | What is in the 10-02 BERT record? | Read `/sys/firmware/acpi/tables/data/BERT` on spark8 | Minutes |
| 11 | Is osd.8 data intact? | `ceph osd deep-scrub 8` (owner asked 10-02 17:22; completion not recorded) | Background |
| 12 | Sub-0.5 s bursts, correctable AER, pause durations, outstanding reads | Faster counter sampling plus `aer_dev_correctable` in surekill | Script change |
| 13 | Does the guard always fire before the first IOVA reuse? | Not directly measurable; keep identity mode on Lenovo | — |

## 19. Corrections to earlier statements

Claims made during the work, in messages or in notes, that the full
evidence does not support.

| Earlier statement | Correct statement |
|---|---|
| Passthrough lowers the crash rate "about 6x" | About 2x pooled (about 3x for spark7). The 6x came before spark8's two identity crashes. Small n, censored. |
| Translated Lenovo crashes "within about 16K two-band tokens" | True only for spark7 from 19:07 to 23:03. spark9 crashed at 16K and 72K, and spark7 later survived about 110K. |
| The link-B test crashed spark8 "on 0000:01:00.0 at about 2 Gb/s, so not link-A bandwidth" | It also logged CmpltTO on 0002:01:00.1 (link B, about 29.5 Gb/s) within the same 30 ms. Both links at once. |
| The 23:20:03 crash was on 0000:01:00.0 | 0000:01:00.1. 01:00.0 appeared only in the mlx5 detail line. |
| "SMMU0 stalled about 85 ms after the AER" | The first CMD_SYNC timeout printed 1.085 s after the AER. The stuck command was issued about 85 ms after, if the upstream 1 s poll applies. |
| "Single band is safe" | Only 512-row single band was tested. Production's 1024-row single band has never been gated. |
| Two-band prefill is about 13% faster | A wave-size artifact. Two-band at 1024 rows (19.9 s) was not faster than production single band at 1024 rows (19.1 s). |
| TCP 12 KB round trip 223 µs (pair) / 542 µs (switch) | First 1 s interval only. Full 10 s averages: 353 / 663 µs, max 1.55 / 1.99 ms. |
| The TCP-vs-RDMA benchmark ran on GIGABYTE pairs | spark1 is an MSI MS-C931. |
| 8-driver decode "about 1.6 s per 8-row step" | About 2.05 s per step. |
| Miners killed at 23:18 | 23:20:52, after the owner's OK at 23:20:45. 16 processes per node, not 17 (pgrep self-match). |
| spark7's first event came "7 s into the first 16K" | 8.4–9.4 s, after Mac-clock correction. |
| Guard reset node "back in 52 s" (playbook) | About 90 s from trigger to the next boot's first journal entry, in every event including the test. |
| X5 "completion timeout 65–210 ms" and T2 "20 Gb/s cap" runs | Neither change took effect. Both are baseline repeats. |
| "Restarting fleet-agent no longer restarts weightd" | True only for unit-managed weightd. Legacy weightds still in the agent cgroup (seen on spark0 and spark7 on 10-03) die with the agent until their next restart. |
| Only spark0 and spark2 have smaller MemTotal | Seven nodes are about 2 GiB smaller: spark0, 2, 4, 7, 8, 9 and b. |
| "Only the Lenovo BIOS differs" | True only as Lenovo vs the rest. The GIGABYTE nodes themselves run four different UEFI builds. |
| The 10-02 GDT held "Ceph OSD log text" | It is the standby mgr.spark8 log. |

## 20. Tools

All scratch tools are in `glmfull-stream-kit/scratch/` on the lead
workstation. Only the guard and the door tool are in this repository.

| Tool | Purpose | Open issues |
|---|---|---|
| `tools/devcycle/spark_dma_guard.py` (#1428) | The guard: `run`, `listen`, `install`, `scan`, `self-test` | Hub listener is a transient unit (H1) |
| `tools/devcycle/spark_fabric_door.sh` (#1429) | Door B (0002:01:00.1) up or down | `up-switch-b` not idempotent; 8192 rings cost about 2 GiB; leaves PFC alone |
| `surekill.sh`, `surekill_hook.sh`, `surekill_series*.sh` | Reproducer runner, pre-run hook, series | Unused `WAVES` variable |
| `fleet_ready.sh`, `guard_rearm.sh` | Readiness gate (armed 16/16); boot-history trim | `guard_rearm.sh` does not restart a watch-only guard |
| `ab_arm.sh`, `ovl_gate.sh` | One arm; build, stage, gate, restore | `ab_arm.sh` leaves the Ceph flags set; `ovl_gate.sh` regex false positive; superblock check blind to GDT damage |
| `mesh_doors.sh`, `door_campaign.sh` | Door switching; abandoned campaign | — |
| `chain_*.sh` | Unattended sequences and the Lenovo passthrough rollout | — |
| `tcp_crashprobe.sh`, `gpustress/gpu_mem_stress.cu` | TCP flood; synthetic GPU memory load | — |
| `tcp_vs_rdma_bench.sh` | RDMA vs TCP latency and throughput | iperf 2 bounce-back hangs at 192 KB and up |
| `telemetry.sh` | GPU and host sampling | `STOP` checked only on reconnect |
| `gap_share.py` | Gate tails and per-sender excess share | Reads the whole weightd log since its last restart |
| `md/` (`md_stage.py`, `md_up.sh`, `md_fire.py`, `md_run.sh`, `md_down.sh`, `md_warm.sh`) | 8-driver test | `md_fire.py` timeout kills lane threads at the deadline; `md_warm.sh` attach fails; lazy mode too slow |

None of these scripts takes the `perf_window.py` lock; serialization relied
on running one chain at a time.

## 21. Owner decisions

| UTC | Decision | Status |
|---|---|---|
| 10-01 16:43 | Never add anything to the boot path | Standing |
| 10-02 17:11 / 17:22 | Get Ceph running; deep-scrub osd.8 | Ceph recovered 10-03; scrub completion not recorded |
| 10-02 17:48 | Reboot and update sparks without approval during pre-production; guard approved; `iommu.strict` rejected; passthrough favoured | Standing |
| 10-02 18:43 / 18:54 | Door A/B: reproduce, use the second path, test all viable combinations | Superseded by 20:50 |
| 10-02 19:32 | No BIOS flash; startup must be order-independent | Flash question reopened 10-03 06:58 |
| 10-02 20:50 | Surekill plus one change at a time; door campaigns only with bandwidth/latency measurement | Standing |
| 10-02 22:11 | Approved tests need no re-ask | Standing |
| 10-02 22:34 / 22:37 | Multi-driver (8 drivers) is the scenario that matters | Done 10-03 at low rate; fast-driver case open |
| 10-02 23:02 | One collective in flight per engine; guard on 16; Lenovo passthrough | Standing |
| 10-02 23:14 | Passthrough on Lenovo only; no RDMA serialization across drivers | Standing |
| 10-02 23:20 | Stop the DB miners (resumable) | Done |
| 10-02 23:22 | TCP for big prefill collectives as the workaround | Fallback only |
| 10-03 06:35 | Back to GLM Full performance work | Current |
| 10-03 06:58 | BIOS question reopened | Pending |

## 22. Evidence locations

Primary evidence lives on the lead workstation and the hub, not in this
repository:

- **Hub guard log:** `rtx5090:~/spark-dma-guard/events.log` (351 lines,
  every event 3 times).
- **Run directories:** `/Users/mac/wf/glmfull-stream-kit/perf-out/`:
  - `sk-*`, `ab-*`, `ovl-*`, `tcpprobe-*`, `tcp-vs-rdma-*`, `md8-*`,
    `telemetry/`;
  - `surekill-summary.txt`, one line per run.
- **Scripts:** `/Users/mac/wf/glmfull-stream-kit/scratch/`.
- **Archive:** `/Users/mac/wf/evidence/cx7-dma-2026-10/` (private,
  mode 0700). It holds:
  - the investigation workflow journals (fact sheets, BIOS memo, forensic
    analyses);
  - the 10-03 live collections and firmware inventory;
  - ACPI and PCIe dumps and kernel-source excerpts;
  - spark8 forensic material (kernel logs, pstore, jbd2 and GDT scans,
    `dumpe2fs` and `fsck` logs, and raw disk-head dumps).
  - The disk dumps contain secret key material and log text from spark8.
    They must stay private.
- **Node logs:** `journalctl --list-boots` on spark7, spark8 and spark9
  shows each guarded reset as a boot whose journal ends seconds before the
  hub trigger.

## 23. Glossary

- **AER**: PCIe Advanced Error Reporting.
- **CmpltTO**: completion timeout. A requester's non-posted request (for
  example a DMA read) got no completion within the DevCtl2 window (here
  50 µs–50 ms).
- **SMMU / SMMUv3**: Arm's IOMMU. It translates device DMA addresses (IOVAs)
  to physical memory.
- **CMD_SYNC**: an SMMU command-queue barrier. A timeout means the queue
  stopped consuming commands.
- **hwprod / hwcons**: the command queue's producer and consumer indices. A
  frozen hwcons with a rising hwprod is a stalled queue.
- **DMA-FQ**: lazy IOMMU mode. Unmapped IOVAs wait in a per-CPU flush queue
  and are invalidated in batches.
- **Identity / passthrough**: the IOMMU bypasses translation for the device.
- **Band**: a slot region of the weightd mesh buffer. Two bands active at
  once means two collectives in flight.
- **Surekill**: the crash reproducer runner and its verdict rule.
- **Door A/B**: PCIe link A (0000) or B (0002) functions used for mesh
  traffic.
- **TTFT16k, pk8k … pk60k, pk128**: gate sessions. A 16K-token prompt;
  passkey prompts of 8K–60K tokens; the 128K passkey session.
