# Managed driver station

This is a development station, not a declaration that eight model drivers pass
inference acceptance. Startup, token generation, numerical agreement and cache
correctness are separate gates. Keep each driver's evidence explicit.

## Ownership and placement

- RTX5090 runs every HTTP API and tokenizer as user `spec`.
- Sparks run one shared weightd per node plus independently named residents.
- ARM/CUDA firmware builds run through the controller queue on a Spark.
- `/Users/mac/.sparkpipe/station-20260924/station.json` is the installed station
  registry. It names exact releases, manifest hashes, lanes, ports and budgets.
- Core, daemon and transport changes require an operator release. A driver update
  must descend from the registry's `core_commit` and change only its recorded
  `driver_paths`. Run `check-driver` before scheduling its build.

| Family | Lane | Sparks | RTX5090 port |
|---|---:|---|---:|
| GLM Flash | 0 | 0–f | 8433 |
| Qwen 27B | 1 | 0–3 | 8401 |
| Qwen Max | 2 | 0–f | 8402 |
| Kimi K3 | 3 | 0–f, TP4 × PP4 | 8403 |
| Gemma 4 | 6 | 0–f | 8406 |
| Laguna | 8 | 0–f, TP8 × PP2 | 8408 |
| Ling | 9 | 0–f | 8409 |
| MiniMax text | 10 | 8–b | 8410 |

The initial profile explicitly reserves B1 and 512 context positions. Qwen 27B
allows nine internal input rows for MTP verification while keeping one active
sequence. Raising
these limits requires recalculating memory and qualifying the new shape. The
shared daemon's pool reservations are 60 GiB on Sparks 0–3 and 49 GiB elsewhere.
The complete declared station, including resident device and host memory,
reserves at most 112128 MiB per node against a 114688 MiB admission limit.
The live queue additionally checks current unified-memory use and an 8192 MiB
headroom. Seven families initialized concurrently. Adding Qwen Max alongside
K3 was rejected on Sparks 0–3: its 17408 MiB reservation plus existing owners'
remaining growth exceeded available memory minus headroom by about 7117 MiB on
Spark0. **Eight simultaneous families are not qualified.** Do not bypass this
check or exempt weightd from reservations. Stop K3 before testing Qwen Max and
re-run admission; even that swap remains subject to current memory use.

## Developer commands

Run lifecycle commands on `mac-studio`, using its default queue ledger. Do not
use a `spark_queue.py` from a developer checkout. The dispatcher is the pinned
`/Users/mac/.sparkpipe/station-20260924/spark_queue-913a2864.py`, supervised by
LaunchAgent `ai.sparkpipe.station-queue`; the station verifies its hash. The old
queue checkout excluded weightd from memory bounds, which made its admission
results unsafe for this station.

```sh
cd /Users/mac/.sparkpipe/station-20260924
/opt/homebrew/bin/python3 spark_station.py --station station.json plan
/opt/homebrew/bin/python3 spark_station.py --station station.json status all
/opt/homebrew/bin/python3 spark_station.py --station station.json start qwen27
/opt/homebrew/bin/python3 spark_station.py --station station.json smoke qwen27
/opt/homebrew/bin/python3 spark_station.py --station station.json stop qwen27
```

`start` verifies release hashes and the fresh mesh, admits the family through the
queue, starts and tracks its residents, waits for **every rank** to initialize,
and then starts the workstation API. Failed startup stops only that family and
untracks stopped units. A failed cleanup retains reservations and returns an
error. `status` reports service state; it does not qualify inference. `smoke`
requires two repeated two-token HTTP requests and matching tokens, with no
numerical oracle. Use the model's independent reference test for acceptance.

Do not start an extra API against the same residents. Do not run an old family
launcher, replace files inside an installed release, edit shared daemon units,
or clear the queue ledger. The API endpoint is the model selector for now.

Create the driver branch from `core_commit` in the installed registry. Commit
the driver change before syncing; uncommitted edits are not build inputs.
For a candidate driver branch:

```sh
/opt/homebrew/bin/python3 spark_station.py --station station.json check-driver \
  qwen27 --repo /path/to/checkout --ref candidate-commit
```

An accepted source diff is not an installed release. Build from that exact
commit in a fresh queue-synced checkout, preserve the family validation receipts,
and hand the bundle plus source commit to the station operator. The operator
installs a new version directory, records the binary/configuration/unit hashes,
and changes only that family's registry entry. Keep the previous version for
rollback. Do not rebuild or replace common binaries during a driver update.

## Mesh recovery

After a crashed resident or an orphaned GPU operation, preserve the first error
and run the operator command on `mac-studio`:

```sh
cd /Users/mac/.sparkpipe/station-20260924
/opt/homebrew/bin/python3 spark_station.py --station station.json recover-core
/opt/homebrew/bin/python3 spark_station.py --station station.json start glm
```

`recover-core` refuses to run over active queue jobs or unknown persistent
services. It verifies the core manifests, stops all station APIs/residents,
stops and untracks the daemons, starts and retracks the pinned daemons, and
exchanges all sixteen fresh mesh records. It leaves models stopped. This exact
command passed on the fleet; a `.ready` file alone is not sufficient.

Coordinate this command with the other developers: it interrupts every station
model. A driver that disappears with outstanding GPU work currently faults the
shared mesh until this coordinated restart. Per-lane crash isolation remains
unfinished. Do not clear the orphan guard or reuse a crashed lane by deleting
its lease. Never run `mesh-exchange` over live residents.

## Building a driver update

The source check and queue sync are separate steps. On the controller, a Qwen
27B example is:

```sh
cd /Users/mac/.sparkpipe/station-20260924
/opt/homebrew/bin/python3 spark_station.py --station station.json check-driver \
  qwen27 --repo /path/to/driver-checkout --ref EXACT_COMMIT
/opt/homebrew/bin/python3 spark_queue-913a2864.py sync \
  --repo /path/to/driver-checkout --id UNIQUE_BUILD_ID \
  --nodes spark0 --ref EXACT_COMMIT
```

Wait for sync to finish and use its returned `cwd` literally. Submit the GPU
publisher through the same pinned queue, reserving 32768 MiB total, 16384 MiB
device, and port 31901 for its private validation daemon:

```sh
/opt/homebrew/bin/python3 spark_queue-913a2864.py add \
  --id UNIQUE_BUILD_ID --nodes spark0 --resources gpu-shared \
  --memory-mib 32768 --device-memory-mib 16384 --ports 31901:31901 \
  --ttl-min 15 --cwd RETURNED_CWD \
  --cmd 'bash tools/qwen38_27b_lane_build_release.sh'
/opt/homebrew/bin/python3 spark_queue-913a2864.py status --id UNIQUE_BUILD_ID
```

The output is `build/qwen38_27b_lane1.tar.gz` with its validation receipts. The
build may wait for memory admission; coordinate stopping a family when needed.
The publisher builds common executables too: **do not install those during a
driver-only update**. Promotion remains an operator operation: stop the family,
install a new immutable version containing the new driver/adapter and existing
pinned common binaries, record `COMPONENT_SOURCES.json` and all manifest hashes,
update only its registry entry, then run `start` and `smoke`. Keep the previous
registry entry, configuration and unit for rollback. Never re-hash changed live
files to make a drift check pass.

## Recovery findings, 2026-09-24

The historical 13.521 tok/s result was **aggregate across eight GLM residents**,
about 1.7 tok/s each. It was not one model producing 13.5 tok/s. An unmanaged
Spark-hosted API also took over resident connections in the reproduced failure.
PR1198 restores API-local artifact roots and prevents a competing API from
stealing a live resident session.

Live multi-family staging found additional concrete defects:

- A present `TP_STANDALONE=0` disabled collectives in several families.
- Dense drivers created collectives without mapping the shared mesh.
- Partial-pool weightd attachment allocated a redundant unusable spine.
- Qwen Max had stale schema/stage-name and TP-versus-PP geometry contracts.
- Qwen 27B required another full pack of free memory despite weightd sharing,
  and left decode descriptor reserved fields uninitialized.
- Gemma's host KV-view constructor wrote through a device pointer; constructing
  another view also erased the previous KV error. GDB located the live SIGSEGV.
- Mesh RDMA completion IDs used eight bits for 512 band/rank combinations.
  Lanes 8–15 could post transfers but never consume their completion. Decode
  now preserves all nine bits; all 512 identities are exercised by the test.
- Qwen MTP scratch was sized for TP shards although MTP writes full hidden
  vectors. Gemma had the same error in four residual buffers; its shared attention output
  was also too small for the wider full-attention heads. Real allocation
  tests fail before each fix and pass afterward.
- Hybrid completion validation accepted token outputs only from the last rank,
  rejecting other ranks in the final TP group. Every rank of TP2/4/8/16 groups
  is now tested, including rejection of tokens from earlier PP groups.
- The shared dense adapter did not validate or complete zero-row release work.
  Gemma generated tokens then looped on cleanup rejection. Its test now covers
  release, repeated release and completion residency, not just decode/reset.
- Restarted collectives accepted a previous run's nonzero epoch cell. Live
  ranks split between sequences 1025 and 2049 and timed out. Mapping now records
  the previous epoch before the all-rank startup barrier; followers wait for a
  new root publication. The test covers both root-first and follower-first
  ordering and fails on the original code.
- The controller queue had an explicit weightd memory exemption. The pinned
  queue counts every owner and subtracts observed CUDA use when calculating
  remaining unified-memory growth.
- MiniMax produced an illegal CUDA memory access on its first inference. The
  batch engine treated connected-driver IO failures as retryable, resubmitting
  the failed request thousands of times. Connected driver failures now terminate;
  actual disconnected-session recovery remains covered separately.

GLM passed three exact 32-token reference requests with core `60ba45ce`:
81.336, 21.562 and 18.863 seconds while other driver tests were running. These
are correctness checks with contention and cold-start effects, not a new
throughput benchmark. The isolated single-resident performance regression is
not yet resolved. This PR is not evidence for 13.5 tok/s from one resident.

## Hardware status and remaining driver work

All eight families have reached their all-rank initialization barrier with the
station. Seven were resident concurrently; Qwen Max was tested separately.
Initialization is not inference acceptance. Preserve this distinction when
reporting progress.

| Family | Latest inference evidence | Remaining work |
|---|---|---|
| GLM Flash | Three exact 32-token reference requests pass | Isolated throughput regression; broader concurrent qualification |
| Gemma 4 | Repeated two-token HTTP requests pass after `74a996d3`; no numerical oracle | Independent numerical oracle and broader cache/batch cases |
| Qwen 27B | Prefill and MTP execute; HTTP ends with UNSUPPORTED | Speculation versus prefix publication contract (`model_batch_engine.c`); do not remove the guard |
| Qwen Max | All 16 ranks initialize; first request returns UNSUPPORTED | `SparkQwen38MaxModuleAdmit` is an unimplemented stub |
| Laguna | Collective passes; first routed-expert launch returns INTERNAL_ERROR | `SparkLagunaLazyExperts` / `SparkLagunaLaunchCudaLayerMlpExperts` |
| Ling | First inference returns INTERNAL_ERROR | Locate the first failing CUDA layer launch |
| Kimi K3 | First request stalls; lease release returns BUSY | Lease/GPU completion path; request was interrupted by coordinated recovery |
| MiniMax | First inference returns IO_ERROR; CUDA 700 | Illegal access preceding `seed-round-tag`; no automatic request retry |

The controller registry records installed component hashes and per-family
qualification. Source and component commits can differ intentionally: an
updated driver does not replace common resident/transport binaries. The
registry's `core_commit` is the source base for new driver branches;
`COMPONENT_SOURCES.json` records the binary source for each installed component.
Use the installed registry and receipts for provenance, not a branch name or
an old launch directory. The station is a reproducible development environment;
**eight-driver inference acceptance and crash isolation are still incomplete**.
