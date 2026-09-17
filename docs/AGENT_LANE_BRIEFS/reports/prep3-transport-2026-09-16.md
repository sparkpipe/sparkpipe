# PREP-3 — TP transport publication-delivery: the 64 KiB base-shift is PROVEN, fix staged offline (2026-09-16)

Agent: PREP-3 (mgr2 dispatch; SOURCE-LEVEL ONLY — no GPU runs, no daemon
restarts, no module execution; coredev owns the sparks while debugging his
allreduce). Branch `lane/prep3-transport` off origin/main `776e979` (fresh
clone `/Users/mac/prep3`). Identity verified
`tools/sparkpipe_github_pat.sh gh api user --jq .login` = `sparkpipe`; every
GitHub command through the wrapper.

Mission: complete the TP>1 publication-delivery diagnosis offline. Symptom
(PRs #1026, #1016, #1025): every TP>1 lane stops at the first TP hidden
allreduce; all ranks publish, ZERO peer publications visible; qwen3flash
shows a SIGBUS variant.

## Verdict: the 64 KiB alignment-shift suspect is PROVEN

The client's mapping base was displaced from the daemon's fd-offset-0 view
by a per-attach random delta. All coordinates on the client side (doorbell
cell, payload slot, peer end-word) were shifted; the daemon scanned
unshifted. Exact expressions, current main `776e979`:

Constants (`include/sparkpipe/spark_weightd.h`):

- `P = SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES = 64 * 1024 = 65536`
- `SLOT = SPARK_WEIGHTD_MESH_SLOT_BYTES = 128 * 32768 = 4194304`
- `SLOTS_PER_BAND = 16 * 16 = 256`, `BANDS = 2 * 8 = 16`
- `BUFFER = SPARK_WEIGHTD_MESH_BUFFER_BYTES = 4194304 * 256 * 16 = 17179869184`
- `DOORBELL_OFFSET = BUFFER`
- `REGION = SPARK_WEIGHTD_MESH_REGION_BYTES = 17179869184 + 8192 = 17179877376`

Client side — pre-fix `SparkWeightdClientAttachLazy`
(`runtime/spark_weightd.c`, old lines 3168-3187):

- `slack = REGION + P - (REGION mod P) - REGION = 65536 - 8192 = 57344`
- `raw = mmap(0, REGION + 57344, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)`
- `aligned = (raw + 65535) & ~65535` — so
  `delta = aligned - raw = (65536 - (raw mod 65536)) mod 65536`
- `mesh_send_buffer_addr = aligned`; `result->mesh_mapping = aligned`
- every client access is `aligned + X` → **file offset `delta + X`**

Client publish/spin (`ring/transport/tp_device_collective.c:640-677,
725-732`; base set at :1109 from `mesh_send_buffer_addr`):

- doorbell write file offset =
  `delta + DOORBELL_OFFSET + (band*16 + rank)*24`
- payload write file offset =
  `delta + band*1073741824 + slot_index*4194304`
- peer end-word spin reads file offset =
  `delta + band*1073741824 + peer_slot*4194304 + bytes`

Daemon side (`node/weightd_mesh.c:414-426` maps, `:657-659` doorbell base,
`:686-688` scan, `:734-737` slot base — no displacement anywhere):

- `recv_buffer = mmap(0, REGION, PROT_READ|PROT_WRITE, MAP_SHARED, memfd, 0)`
- doorbell scan file offset = `DOORBELL_OFFSET + index*24`,
  `index = band*16 + rank`
- RDMA ship reads file offset =
  `band*1073741824 + slot*4194304` (via `recv_mr`, content = memfd bytes)

Equality of the two coordinate systems requires `delta = 0`. mmap returns
OS-page-aligned (`q`) bases with `q in {4096, 16384}` on the fleet, so
`P(delta = 0) = q/65536 = 1/16` per attach under ASLR. For the other
**15/16 of attaches**: the daemon reads `seq = 0` at its doorbell coordinate
forever (`weightd_mesh.c:704` `seq == 0 → continue`) → zero RDMA ships →
zero peer publications. The receive side is broken by the same delta: peer
ships land at file offset `slot_base`, the local client spins at file
offset `delta + slot_base` → never sees the end word → `MESH-SPIN-TIMEOUT`
(`tp_device_collective.c:707-719`). "All ranks publish, ZERO peer
publications visible", deterministically per rank per boot.

SIGBUS tail (PR #1025 variant): the client touches file offsets
`[delta, delta + REGION)` (top doorbell cell ends at `REGION - 1`) inside a
mapping of length `REGION + 57344`; `SIGBUS` iff
`delta + REGION > REGION + 57344` iff `delta > 57344`; with `q = 4096` the
deltas are multiples of 4096 in `[0, 61440]`, so exactly `delta = 61440`
(1/16 of attaches) faults at the top doorbell cells.

Coredev's "ALMOST working but not stable" is what 1/16-of-attaches-works
looks like across ranks and boots: any rank that happened to get a
64 KiB-aligned raw base functioned; the rest hung at the first allreduce.

## The fix (lane/prep3-transport)

`runtime/spark_weightd.c` `SparkWeightdClientAttachLazy` — map exactly
`mesh_send_buffer_bytes` at fd offset 0 and use the mapping start as the
coordinate base, deleting the slack computation and the 64 KiB round-up:

- `mesh_mapping = mesh_send_buffer_addr = raw` (was `raw + delta`)
- `delta = 0` by construction, on every host page size
- the fd-passed region is `ftruncate`d to `REGION`
  (`node/weightd_mesh.c:420`) — every client touch
  `[0, REGION)` stays in-file
- `cudaHostRegister` in `tp_device_collective.c:1096-1099` keeps working:
  `raw` is OS-page-aligned on every host (4 KiB or 64 KiB), and registration
  failure was already non-fatal (`MESH-REGISTER-FALLBACK`)
- side repair: the old code's `munmap(mesh_mapping=aligned, REGION)` left
  the head page `[raw, aligned)` mapped forever
  (`runtime/spark_weightd_lazy_pack.c:41-44`); with the fix the unmap is
  exact

`include/sparkpipe/spark_weightd.h` — `SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES`
deleted (repo-wide grep: only the deleted lines referenced it).

## Offline harness: two processes, shared memfd, no GPU, no NIC

`tests/test_weightd_mesh_doorbell.c` (registered as
`build/test_weightd_mesh_doorbell` in the Makefile `TEST_NAMES` + build
rule; runs under `make test`). Forks a daemon-role child that replicates
the `weightd_mesh.c` scan loop (stable-read of the 24-byte cell, `seq==0`
skip, slot/bytes validation, payload + end-word verification, ack cells in
the doorbell control area) and a client-role parent that replicates the
`tp_device_collective.c:659-677` publish order
(payload → end word → entry[2] slot → entry[1] bytes → entry[0] seq,
round-blocked like the real `RunRound`). Two legs per run:

- fixed leg — the new product math (`mmap` offset 0, length `REGION`, base
  = mapping start): must deliver all 64 ops
- legacy-align leg (negative control) — the client base displaced from
  fd offset 0 by the legacy round-up
  `aligned = (raw + 65535) & ~65535`; where the host mmap hands out
  already-aligned bases (this darwin workstation returns the same
  64 KiB-aligned base for every mmap — measured `raw mod 64K = 0` on 8/8
  tries), the displacement is forced to the legacy maximum `65536` so the
  control is deterministic; on the Linux fleet the verbatim round-up
  produces `delta in {1..61440}` multiples of 4096 in 15/16 of attaches.
  Must deliver 0 ops within the 2 s deadline.

Run output (this workstation, 5/5 identical, exit 0, ~2.3 s):

```
mesh-doorbell fixed-mode delivered=64/64 verify=pass
mesh-doorbell legacy-align delivered=0/64 negative-control=pass
```

The harness is self-contained (public header constants only) so it builds
and runs on any host; on the fleet it additionally exercises the real
`memfd_create` path (Linux branch of the backing-file helper).

## Accidental W72 reproduction (for coredev)

My first harness draft published all 64 ops back-to-back (unpaced) and the
daemon-role child saw only the latest seq: `delivered=2/64`. That is the
W72 latest-wins failure mode in miniature. The production client is
naturally paced — `RunRound` blocks on peer end-words before the next
publish, so the doorbell is consumed before it is overwritten — and the
existing `resync_mask` ring-ship (`weightd_mesh.c:745-810`) recovers gaps
up to `SPARK_WEIGHTD_MESH_SLOTS_PER_RANK = 16` missed ops per cell. Deeper
slips (capture-graph replay publishing many rounds without waiting) are
unrecoverable with a single-entry doorbell. Your lossless per-slot
bitmap/range-ship direction is the right hardening for that, and this
alignment fix is its prerequisite: with the base-shift present, 15/16 of
rank processes never deliver anything at all, which alone explains the
instability you are chasing. The two defects are independent layers of the
same delivery path; the mapping fix does not compete with your doorbell
work — it removes the deterministic zero-delivery floor so your lossless
doorbell changes become measurable.

## Verification

- `make build/test_weightd_mesh_doorbell` — clean under
  `-Wall -Wextra -Werror`; 5/5 runs exit 0 (output above)
- `cc -fsyntax-only -I. -Iinclude -Isrc -Itests/cuda_stub -std=c11 -Wall
  -Wextra -Werror -D_GNU_SOURCE -DPOLLRDHUP=0x2000
  runtime/spark_weightd.c` — clean
- NOT verifiable on this host: the `test_weightd*` family that compiles
  `runtime/spark_weightd.c` cannot build on darwin — pre-existing
  `POLLRDHUP` (Linux-only) at `runtime/spark_weightd.c:207/210`, confirmed
  failing identically on unpatched `776e979` via `git stash` build. The
  `make test` skip contract covers this; the weightd family should be run
  once on a Linux host before merge.

## Deliverables on lane/prep3-transport

- `runtime/spark_weightd.c` — attach-lazy mapping fix (the one hunk)
- `include/sparkpipe/spark_weightd.h` — dead `SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES` deleted
- `tests/test_weightd_mesh_doorbell.c` — the two-process memfd harness (fixed leg + legacy negative control)
- `Makefile` — `test_weightd_mesh_doorbell` registered (TEST_NAMES + build rule)
- this report
