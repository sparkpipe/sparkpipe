# Parallel resident qualification

The release gate is concurrent inference with correct tokens and bounded shared
residency, followed by verified cleanup. Starting processes or printing readiness
is insufficient. Different model families also need their own execution evidence;
a repeated GLM deployment does not establish their numerical correctness.
Later exact-source reruns and full logs are attached to the
[shared-serving release](https://github.com/sparkpipe/sparkpipe/releases/tag/shared-serving-20260922).

## First shared-daemon fleet campaign

Source `2b24873d89c0da9744391d104f9d61369fb750f0` passed a fresh 164-check host
campaign with zero failures, setup failures or timeouts, plus CUDA compilation,
GPU module publication, driver linking/loading and exact artifact verification.
CI `compile-sm121a` also passed. The campaign selected 114 of 115 registered C
targets; the Qwen GPU target and 165 unselected Python files remain separately
accounted for. Host fixtures do not establish every model's GPU behavior.

The authoritative controller is `mac@mac-studio`, using its existing
`/Users/mac/.sparkpipe/queue` ledger. Three stale dispatchers were stopped and
replaced by one dispatcher from the pinned source. Existing notes/invalid jobs
and a ledger backup were preserved. The old fleet-agent services were stopped
on all sixteen Sparks; orphaned Spark3 test residents were drained as well.
Every new participant ran inside its queue-owned systemd cgroup.

Each Spark ran one shared weightd plus independent GLM5.3 Flash FP8 TP16
resident processes. Every resident had B1, one in-flight submission, one active
sequence, context capacity 512, 128 logical/physical pages and a 2 GiB backing
limit. Ports, runtime/cache roots and explicitly reserved collective lanes were
separate. CUDA function/data loading were LAZY, connection count 32, hardware
waits enabled and graph/expert pinning explicitly enabled.

The pinned `.wset` contains 336 expert keys and was warmed once per daemon.
Graph execution subsequently pins every expert; this campaign accounts for the
whole 21.7 GB pack, not a small working set. The declared daemon budget was
28 GiB plus 512 MiB overhead; each resident had a 4 GiB device reservation and
4 GiB host reservation. Measured device allocations were 20,874 MiB per daemon
and 3,434 MiB per resident.

| Residents per Spark | Result | Complete outputs | Decode tok/s per resident | Aggregate common-window tok/s | Device MiB per Spark |
| --- | --- | ---: | --- | ---: | ---: |
| 2 | PASS | 64 | 6.545, 6.301 | 12.940 | 27,742 |
| 3 | PASS | 96 | 4.322, 4.326, 4.330 | 12.958 | 31,176 |
| 4 | FAIL | Three requests completed; one stalled | Not qualified | Not qualified | Recorded in retained snapshot |

All successful requests produced the exact same 32 tokens as the isolated
baseline. The common decode windows overlapped for 4.714 and 7.023 seconds.
All 48 and 64 owned processes respectively were absent after successful
shutdown; the queue independently confirmed stopped control groups.
Independent engines share device capacity but do not combine requests into one
batched GEMM. These results therefore establish concurrent developer execution,
not continuous-batch throughput scaling or independent model-quality validation.

## Four-resident failure and causal regression

The four-resident run exposed a completion boundary at rank zero, lane two,
submission 172. The graph completed successfully, then `EndChain` returned
`BUSY` and the module retained ownership without completing the request. The
other three residents finished their 32-token requests. Logs were retained and
the queue cancelled the failed run; all participant control groups stopped.

The common CUDA wait helper published callback completion before the callback
returned, allowing a successful wait while the stream still reported not-ready.
A direct GPU reproduction using the unchanged helper observed one such success
in 200,000 waits across four independent CUDA processes. The preceding 50,000
single-process waits observed none. The deterministic host regression controls
the notification/retirement gap instead of relying on this rare schedule.

Commit `3f888927` retains the completion receipt until the stream is terminal,
within the original deadline. It waits on the condition during GPU work and
uses bounded retirement checks after notification. Timeout preserves ownership
and reuses the pending receipt rather than appending callbacks. The retained real-CUDA regression then passed 200,000 waits across four
concurrent processes: zero false successes, all four exited zero and all PIDs
were absent. Its binary SHA256 is
`2139eff5fc2458dae58b6bd698c8ef7f7c3e70b60cc0e283ccea163941bd8076`,
built from `bd25d2c7`. This is a common completion-helper gate; the subsequent
fleet results are recorded below.

## Eight-resident fleet qualification

Source `44fe4af719daeb5ad3ecaacd78daed5247067fb4` completed the repaired four-
and eight-resident runs through the authoritative queue. All sixteen ranks
passed exact output, assigned-lane, GPU-budget and terminal-cleanup checks.
The unchanged isolated baseline supplied the expected tokens.

| Residents per Spark | Result | Output tokens | Per-resident decode tok/s | Aggregate common-window tok/s | Device MiB per Spark | Owned processes gone |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 4 | PASS | 128 | 3.100–3.133 | 12.388 | 34,610 | 80 |
| 8 | PASS | 256 | 1.658–1.708 | 13.300 | 48,346 | 144 |

The shared decode windows were 9.768 and 17.745 seconds. The eight residents
used lanes 0–7, separate ports 20000–27999 and separate cache/runtime roots.
Each had the same finite B1/context 512/KV plan described above. Both jobs
reserved 16 GiB host memory per Spark. The eight-resident device reservation was
61,952 MiB, total 78,336 MiB, with the queue retaining 8,192 MiB node headroom.
Measured device usage was 47.2 GiB per Spark. Reclaimable pack page cache filled
one host cgroup during startup without an OOM; these are declared reservations
plus observed allocations, not a hard CUDA allocator limit.

The corresponding [compact receipt](receipts/parallel-residents-44fe4af7.json)
contains every rank receipt hash and timing boundary. Raw receipts remain under
`/private/tmp/sparkpipe-pr1082-receipts/shared-44fe4af7/` on the review workstation.
Eight is the current explicit lane capacity. These are eight independent GLM
instances sharing one daemon per physical Spark; they do not establish eight
model families or continuous-batch kernel reuse.

The separate mixed-topology probe reached 34 ready CUDA children but failed
15 mesh registrations before any numerical round. The shared lazy-attach code
rounded a file mapping pointer without preserving file offset zero. Commit `498275a6` repairs that defect by reserving an aligned range and mapping
file offset zero into it. Nine focused offset/failure/schema cases pass; the old
implementation fails the forced-misalignment byte oracle. The mixed-topology
hardware result remains a failure until rerun.
All sixteen queue control groups stopped. The failed attempt is
`76dbd15ed4154c38a49cef183513a116`; its [failure receipt](receipts/mixed-mesh-44fe4af7-failure.json)
retains the exact initialization error and qualification boundary.

## Aligned-mapping repeat

Source `852e01a2c9ee8b7b273370a793761dc70db5b005` passed the complete eight-resident
campaign again after the shared-memory mapping repair. All sixteen queue groups
exited zero; 256 output tokens matched and all 144 owned processes were absent.
Aggregate decode was **13.010 tok/s**, per resident **1.623–1.669 tok/s**, with
**48,346 MiB** device memory on every Spark. The common window contained 236 tokens
over 18.140 seconds. Budgets and model geometry were unchanged.
The [repeat receipt](receipts/parallel-residents-852e01a2.json) pins every rank.

The separate mixed-layout probe registered all 55 children and performed numerical
work, then failed at an eager gather capability check before full qualification.
The legacy callback check prevented reaching the existing hardware gather path.
Commit `734c69ee` restricts the legacy callback requirement to the path that
actually uses it. Hardware and B2+ tree routes exercise their native kernels.
The focused suite passes 208 common checks, 2,046 fuzzer checks and 1,656 sanitizer
checks; the original guard fails 28 controlled assertions. The failed hardware
receipt remains partial evidence; see the release archive for subsequent reruns.

## Evidence

The compact [fleet receipt](receipts/parallel-residents-2b24873d.json) pins source,
bundle, attempts, output counts, timing boundaries and per-rank receipt hashes.
Full evidence is retained at
`/private/tmp/sparkpipe-pr1082-receipts/shared-2b24873d/` on the review workstation.
The unchanged-helper GPU reproduction is
`/private/tmp/sparkpipe-cuda-receipt-probe-2b24873d-four.log`.
The fresh host receipt is
`/private/tmp/sparkpipe-pr1082-receipts/host-2b24873d/reliability/results.json`.

Partial-pool eviction, common lane assignment across different model families,
and nonidentity physical rank maps require their own subsequent qualification.
The first two passing rows above do not imply those cases passed.
