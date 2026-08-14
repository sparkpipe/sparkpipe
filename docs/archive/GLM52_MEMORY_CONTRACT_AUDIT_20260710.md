# GLM-5.2 Memory Contract Audit

Date: 2026-07-10

## Scope

The audit covers first-party C, CUDA, headers, and Python tooling. It excludes
generated build output and vendored third-party code.

## Closed Findings

- GLM-5.2 model geometry now has one source in `model_contracts/glm52.json`.
- `tools/glm52_model_contract.py` generates the public C contract and checks it
  for drift.
- C and CUDA allocations use element types and `sizeof(type)` instead of raw
  byte widths.
- Direct allocation, copy, zero, read, write, and hash-update calls reject raw
  numeric byte counts in `tests/test_memory_contracts.py`.
- FP8 and B12x pack headers use named fields and compile-time wire-size checks.
- The duplicate B12x public ABI header was removed.
- The PP13 final-event magic, wire type, descriptor size, and receive-buffer
  extent now have one definition in `spark_glm52_pp13_runtime.h`.
- Model-derived stage, MTP, FP8-scale, topology, hidden, sideband, vocabulary,
  DSA, and DSpark constants alias the generated model contract.
- Validation no longer reads 6144 FP8 scale values from a 2048-value host
  seed.
- Query latent and query RoPE allocations cover all attention heads.

## Closed P0: Expanded K/V Cache Addressing

The former PP13 tiled-attention builder allocated the expanded caches as:

```text
key_nope elements = max_active * selected_token_count * qk_nope_head_dimension
value elements    = max_active * selected_token_count * value_head_dimension
```

The CUDA prepare kernel addresses those allocations as:

```text
physical_token_slot * head_count * per_head_dimension
```

The implied safe physical-token capacity is therefore:

```text
max_active * selected_token_count / head_count
```

For B1024 this is 32,768 physical slots. The node advertises
`SPARK_GLM52_MODEL_KV_POOL_TOKENS`, or 4,194,304 physical slots. The writer can
therefore address 128 times beyond the allocated expanded cache.

Allocating the full expanded cache is not a valid fix. With the current model
geometry, full BF16 key-nope plus value storage is about 224 GiB per layer.
The tiled path would require a separate bounded expansion workspace:

1. Keep full-context resident storage in the 576-element MLA representation.
2. Expand only selected tokens into a bounded attention workspace.
3. Address that workspace by compact selected-row index, never physical KV
   slot.
4. Carry the physical-slot to compact-row mapping as an explicit typed view.
5. Give the MLA cache and expanded workspace separate capacity fields and
   validate both at the kernel boundary.

The PP13 production builder instead uses absorbed MLA directly, keeps only the
576-element compressed cache row, and leaves the expanded K/V pointers null.
Speculative rollback clears the compressed MLA row and DSA index row only. The
exact FP8 five-token six-layer gate on GB10 measured a worst final-layer
relative L2 of `0.026662` and minimum cosine of `0.999645` against the official
model.

## Permanent Gate

`tests/test_memory_contracts.py` runs under `make test` and rejects:

- stale generated model contracts;
- raw element-width multipliers in memory operations;
- direct numeric byte counts in standard memory and I/O calls;
- numeric protocol receive-buffer extents;
- anonymous indexed FP8 wire-header fields;
- numeric wire descriptor sizes instead of `sizeof(wire_type)`;
- duplicate PP13 final-event or B12x public wire definitions;
- model geometry literals outside the canonical contract;
- required model aliases that stop pointing at their canonical symbol.
