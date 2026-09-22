# Shared development cycle

Use the authoritative `spark_queue.py` controller and its existing ledger for
builds, smoke tests and persistent residents. The current workflow is documented
in [Parallel driver debugging](PARALLEL_DRIVER_DEBUG.md). Each developer receives
a clean checkout from `spark_queue.py sync`, explicit ports and finite host/device
memory reservations. Model packs can be shared; runtime roots, KV state and
collective lanes belong to each resident.

Push the reviewed source, sync its exact commit, then submit a GPU-owned build
job in that checkout. GLM5.3 Flash FP8 uses:

```sh
bash tools/glm5_next_build_release.sh
```

Families implementing the common archive/adapter/publish Makefile interface use:

```sh
bash tools/module_build_release.sh FAMILY CODEC OUTPUT_NAME MODEL_REVISION CONTRACT_JSON
```

The builder requires `SPARK_QUEUE_ID`, refuses a dirty tracked tree or an existing
output or preexisting compiled objects, and takes a nonblocking lock in this checkout. `FIRMWARE_JSON` explicitly
selects a firmware description when its filename differs from the family/codec
convention. An unsupported adapter/build interface fails visibly.

The output is a coherent `build/OUTPUT_NAME.tar.gz` containing the host services,
weight daemon, transport, adapter, validated driver, module publication records,
source identity and relative `SHA256SUMS`. Verify the archive checksum and then
`sha256sum -c SHA256SUMS` after extraction. GPU publication qualifies the module
validator; complete serving still needs the model's exact packs, configuration
and numerical reference.

The script builds only in the current checkout. It does not reset a shared tree,
stop residents, write a hub release or trigger fleet updates. Branch arguments
from the legacy publishing workflow are rejected. The `tools_local` entry point
forwards to the same implementation.

Run the resulting deployment inside admitted queue jobs. Reserve the same lane
and physical rank map on every participating daemon, unique ports and bounded
KV/workspace allocations. Keep each lane's topology fixed for that daemon's
lifetime. Coordinate `.wset` budgets across models and retain token, overlap,
memory and terminal-cleanup receipts. A ready process alone is not a passed
smoke test. See [parallel qualification](PARALLEL_RESIDENT_QUALIFICATION.md) for
what has actually run.
