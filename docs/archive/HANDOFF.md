# Handoff — Phase 10 Spark Hardware Qualification

The exact operational procedure is in:

```text
docs/SPARK_HARDWARE_HANDOFF.md
```

## Host validation

```sh
make clean
make -j2 all
make -j2 test
sh tools/gates.sh
make hardware_handoff
```

## Exact CUDA compile-only validation

```sh
export NVCC=/usr/local/cuda/bin/nvcc
export CUDA_ARCH=sm_121a
export SPARK_CUDA_GATE_SCOPE=complete
sh tools/cuda13_sm121a_compile_gate.sh
```

## Hardware order

```text
1. direct single-rail ring, physical PP order matching adjacency
2. one MikroTik single-switch 100 Gbit/s rail
3. dual rail only after both single-rail modes close
```

## Qualification chain

```text
source archive SHA-256
→ exact plan
→ per-node configuration SHA-256
→ preflight artifact inventory
→ exact cell receipts
→ aggregate
→ generated policy
→ question closure
```

The suite is fail-closed. Missing production providers, missing numerical references, artifact-identity mismatches, missing peers, timeouts, integrity failures, incomplete cells, and unanswered production questions prevent closure.
