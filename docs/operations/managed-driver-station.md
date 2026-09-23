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

The initial profile explicitly reserves B1 and 512 context positions. Raising
these limits requires recalculating memory and qualifying the new shape. The
shared daemon's pool reservations are 60 GiB on Sparks 0–3 and 49 GiB elsewhere.
The complete declared station, including resident device and host memory,
reserves at most 112128 MiB per node against a 114688 MiB admission limit.
These are reservations, not proof of simultaneous peak execution.

## Developer commands

Run lifecycle commands on `mac-studio`, using its default queue ledger:

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

A daemon restart creates new mesh records. After stopping all station APIs and
residents, restart and retrack the shared daemons through the operator workflow,
then run `mesh-exchange`. It checks live daemon invocation IDs and exchanges all
sixteen current records. It refuses to exchange while a station model is active.
Only then start residents and APIs. A `.ready` file alone is insufficient.

An owner that disappears with outstanding GPU work currently faults the shared
mesh until a coordinated restart. This is an unresolved isolation limitation;
do not clear that guard or call a crashed driver's lane reusable just because
its HTTP request ended. Preserve the first failure and use the operator recovery
workflow. Experimental driver faults can still require a station restart.

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
- MiniMax produced an illegal CUDA memory access on its first inference. The
  batch engine treated connected-driver IO failures as retryable, resubmitting
  the failed request thousands of times. Connected driver failures now terminate;
  actual disconnected-session recovery remains covered separately.

GLM passed three exact 32-token reference requests after deploying the shared
mesh changes. This does not qualify the seven other drivers. The hardware test
matrix and installed registry must be refreshed after each candidate rollout.
