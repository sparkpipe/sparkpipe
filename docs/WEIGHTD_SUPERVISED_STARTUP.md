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
before residentd. For an isolated debug run, give it a lane-specific socket and
keep it in the same queue-owned cgroup as the driver. For a shared cache daemon,
its service owner must provide supervision independently of driver lifetimes.
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
