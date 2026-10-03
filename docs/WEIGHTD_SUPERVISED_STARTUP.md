# Supervised weightd startup

A deployment containing `weightd.socket_path` requires that daemon. Residentd
fails before loading the adapter if the socket is unavailable, the runtime root
has no digest sidecar, a digest is malformed, or multiple sidecars are present.
`SPARK_WEIGHTD_ATTACH=0` conflicts with this deployment and is rejected.

The current identity convention is exactly one `packs/*.sha256` in each rank's
runtime root. It contains a lowercase SHA-256, optionally followed by the usual
`sha256sum` filename field. The startup check parses this identity; it does not
rehash or qualify the pack. Verified placement must supply the correct digest.
The subsequent shared attach handshake validates the daemon IPC and arena.

Start `sparkpipe_weightd --socket <deployment socket>` as a supervised process
before residentd. For a shared cache daemon, its service owner must provide
supervision independently of driver lifetimes. In production that owner is
`ensure_weightd` in `tools/fleet_node_agent.sh`, and the socket is
`/tmp/spark_weightd.sock` ([WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md#production-ownership)).
An isolated debug daemon, with its own socket inside the driver's queue-owned
cgroup, is only possible on a host without the fleet agent: on a serving Spark
any other `sparkpipe_weightd` process stops the agent from managing that node.
Residentd no longer forks, detaches, or attempts to repair that service. A
successful socket probe is not proof of model readiness.

This removes the configured startup fallback. It does not implement lazy expert
working sets or GPU completion leases, and it does not change deployments that
omit weightd. The complete shared lazy path remains tracked separately.

Host checks:

```sh
python3 tests/test_weightd_supervised.py
make -j4 build/test_model_resident_deadline
build/test_model_resident_deadline
```

## Startup check

`model_residentd` calls `SparkModelResidentdPrepareWeightd`
(`node/weightd_spawn.c`) when the deployment has a non-empty
`weightd.socket_path`, before `SparkModelResidentdInitialize`. It checks
`SPARK_WEIGHTD_ATTACH`, reads the digest sidecar, probes the socket and sets
two environment variables.

- `SPARK_WEIGHTD_ATTACH`, when set, must be exactly `1`. Any other value,
  the empty string included, fails.
- The sidecar must be a regular file. Its first 64 bytes must be lowercase
  hex, followed by end of file, a space, a tab or a newline. Nothing after
  that is read.
- The socket probe is one non-blocking `connect` on the Unix socket, closed
  at once. It sends no `HELLO`.
- On success it sets `SPARK_WEIGHTD_SOCKET` to the deployment socket and
  `SPARK_WEIGHTD_PACK_SHA256` to the digest, replacing any existing values.
  `SparkWeightdAttachRequested` reads the socket from there. The shared lazy
  open helper (`include/sparkpipe/family/module/spark_module_lazy_open.h`)
  reads the digest, and `SparkWeightdAttachPack` falls back to it when the
  slice carries none.
- On failure it returns a negative code. Residentd logs a line starting
  `model_residentd weightd-required status=<code> root=<root> socket=<socket>`
  and exits 1.
