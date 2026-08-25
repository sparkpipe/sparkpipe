GOAL: DSV4 Flash on spark4-7 from 40.67 tok/s no-spec to 100+ tok/s with DFlash2 speculation, exact-output-preserving.
Base: ~/sparkpipe @ unified. Runtime: /tmp/dsv4-base-runtime pattern (devcycle.sh). Serialize ring access: only one benchmark at a time; log every window in the state file.
Steps: reproduce 40.67 baseline (gate24+3xO128 receipts) -> read docs/QWEN38_DFLASH2_RUNBOOK.md + docs/DFLASH2_ADOPTION_SPEC.md -> port drafter to Flash geometry -> k-sweep behind token-parity gate -> retain receipts under qualification/dsv4/performance/.
