# Development and Release Runbook

This is the release path for every model and deployment class.

## Source workflow

1. Start from a clean checkout of current `origin/main`.
2. Create a feature branch and make the scoped change.
3. Run host, source, schema, and package tests.
4. Open a PR and qualify the exact PR commit on target hardware when required.
5. Merge only after required checks and hardware gates pass.
6. Pull merged `main` into clean target checkouts.
7. Rebuild and assemble one immutable release from those checkouts.
8. Install that release on every rank and restart the owning services.
9. Validate source, package, process, topology, and ready identities on all
   ranks before measuring behavior.

No hotpatch, copied binary, dirty checkout, unmerged branch, or mixed release
generation can produce a deployment or performance qualification receipt.

## Runtime contract

- Every rank runs one generic resident process.
- Every client uses the generic request and streaming event contract.
- A package binds one model checkpoint, adapter, driver, stage pack, precision
  route, model placement, hardware profile, and collective profile.
- Package, adapter, driver, stage pack, model shards, and topology identity must
  agree at startup.
- Production packages contain no reference, serialized, compatibility, scalar,
  alternate-network, or alternate-kernel fallback.

## Host gates

```sh
make clean
make -j1 all
make test
sh tools/gates.sh
```

The default host suite does not prove CUDA, physical networking, numerical
correctness, or live model service behavior.

## Target CUDA gate

Run the exact package CUDA gate on a GB10 or Station target with the release's
declared architecture and toolchain. Retain compiler version, command, complete
output, generated artifacts, and SHA-256 identities.

## Deployment gate

For 4-, 8-, or 16-Spark deployments:

1. validate physical devices and storage roles;
2. validate switched and direct interfaces and pinned routes;
3. validate every complete direct pair concurrently in both directions;
4. validate package and rank-local shard identity;
5. start all resident processes from one release generation;
6. validate communicator and collective profile agreement; and
7. publish ready only after every required rank reports the same generation.

Station and mixed Station-plus-Spark deployments follow the same sequence with
their exact hardware profile.

## Numerical gate

Use the exact checkpoint, tokenizer, prompt, precision route, placement, and
workload. Compare complete emitted token IDs and required intermediate tensors
with the authoritative implementation under named tolerances. Transport
success, plausible text, or matching token count is not numerical evidence.

## Performance gate

Record:

- merged commit, release, driver, checkpoint, and topology identities;
- request count, prompt, context, output length, batch and scheduler policy;
- exact start and stop boundary;
- warm/cold residency and KV state;
- latency distribution, output and aggregate throughput;
- direct and switched interface counters; and
- full token parity with the accepted control.

Write accepted measurements to [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).
Write remaining work to [`../TECHDEBT.md`](../TECHDEBT.md). Do not add a phase
report, handoff diary, or second status ledger.
