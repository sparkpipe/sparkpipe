# Phase 6 Change Manifest

Phase 6 is based on the Phase 5 working tree, which was not separately released
as a deterministic archive. The released Phase 6 package therefore contains
both the Phase 5 acknowledged-work foundation and the Phase 6 completion-
ownership integration.

## Added

```text
tests/test_distributed_work.c
```

## Modified implementation and ABI files

```text
Makefile
include/sparkpipe/spark_distributed_work.h
include/sparkpipe/spark_ring_runtime.h
include/sparkpipe/spark_ring_work_control.h
include/sparkpipe/spark_work_transaction.h
node/backend.c
node/rank_daemon.c
ring/distributed_work.c
runtime/net.h
runtime/work_transaction.c
scheduler/work_control.c
serving/spark_cuda_resident_ipc.c
```

## Modified tests and gates

```text
tests/test_code_size.py
tests/test_dry_law.py
tests/test_glm52_cuda_resident_ipc.c
tests/test_glm52_ring_rank_daemon.c
tests/test_glm52_ring_service_backend_internal.c
tests/test_ring_service_backend_transactions.c
tests/test_work_transaction.c
tools/gates.sh
```

## Added Phase 6 documentation

```text
docs/PHASE6_TRANSACTIONAL_COMPLETION_OWNERSHIP.md
docs/PHASE6_REMAINING_WORK.md
docs/PHASE6_CHANGE_MANIFEST.md
docs/PHASE6_VALIDATION_STATUS.md
docs/PHASE6_VALIDATION_STATUS.json
```

The deterministic package manifest records the SHA-256 and exact byte size of
every retained payload file.
