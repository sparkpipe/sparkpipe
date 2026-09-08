# Spark queue runbook

The current contract and commands are in [Parallel driver debugging](PARALLEL_DRIVER_DEBUG.md).

The previous v1 guidance about releasing before cleanup, reachability-only unfencing, and guaranteed lazy residency was incorrect. Stop the v1 dispatcher before migrating to v2.
