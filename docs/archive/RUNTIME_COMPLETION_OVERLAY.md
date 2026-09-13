# Runtime-completion affected-files overlay

This overlay applies only to the Phase 10 hardware-handoff source package with
archive SHA-256:

```text
c602c8174902ba2138b68828fc21a9eb6d07977fa6fb21aaa10b2cb20bad5550
```

It adds a fail-closed host runtime for:

- all-participant prepare, staged execution, commit, and cancellation;
- transaction generations and terminal replay detection;
- bounded selective-acknowledgement transmission slots;
- acknowledged final-event ownership;
- exact model-provider artifact and precision validation;
- K3, GLM 5.2, Qwen 3.6 27B, DSV4 Flash, and DSV4 Pro operation contracts;
- an exact `sm_121a`, `--no-undefined` GLM final-artifact build receipt.

The model runtime deliberately does not claim CUDA or Blackwell qualification.
Each provider must supply the exact production operations and immutable artifact
identity before registration succeeds.

Apply from the extracted overlay root:

```sh
sh tools/runtime/apply_runtime_completion_overlay.sh \
    /path/to/overlay-root \
    /path/to/sparkpipe-phase10-source
```

Then run:

```sh
make -j2 runtime_completion_tests
make -j2 all
```
