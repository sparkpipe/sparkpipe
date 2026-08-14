# GLM52 PP13 KV Slot Probe Receipt - 2026-07-09

This records the PP13 FP8 ring receipt from the `SPARKPIPE_MLA_SLOT_PROBE`
and KV-slot probe added in PR #242.

## Build And Release

Merged main commit:

```text
c1307c17da3d1734f0777565d1b8c5500dd9608c
```

Release:

```text
glm52-fp8-main-c1307c17-b1-kv-slot-probe
```

Builder shared object hash on all ranks:

```text
4aa3935f7b8f2c3178e26cc0c7b74d8e23f2d033533a068b15c3be7ce9a5bf41
```

Release validation:

```text
valid release_id=glm52-fp8-main-c1307c17-b1-kv-slot-probe generation=20260709214500 files=11 roles=4
```

Post-run gateway health:

```json
{"service":"sparkpipe-glm52","backend_ready":1,"pp13_ready":1,"max_context_tokens":1048576,"production_contract_flags":3903,"connected_clients":1,"live_requests":0,"queued_requests":0,"event_backlog":0,"dropped_events":0,"jit_prefetch_dispatches":0,"jit_prefetch_blocks":0,"async_jit_prefetch_starts":0,"async_jit_prefetch_completions":0,"first_blocker":""}
```

## Inference Runs

Prompt:

```text
Say OK. OK.
```

Request settings:

```text
stream=true
temperature=0
max_tokens=1
```

Results:

```text
run1 token_id=21974 text="woman" elapsed=1.55s
run2 token_id=29843 text="ilib"  elapsed=1.05s
run3 token_id=24    text="9"     elapsed=1.05s
```

Rank0 TX SHA256 samples:

```text
run1 token0 e8d769e1b93c74151e8d25061edfeeee40d075e47c108cde267b2b532d78c684
run1 token1 f7e410ee150921acc70363acc239288daa10b15d01ea259583da2c912c956a1f
run1 token5 7a6dc93b9d89932f9346a8e4d1a72683fceb1becb973dfd18ab8be558b8f556f

run2 token0 275fee963b21648191e3c142c8e172531784d859b1024d0712d60d2d4569d91a
run2 token1 067500a4ca43c8614401dbd7089ca0fdda12dc1fa6a1f3eb070b11ad96f41ae4
run2 token5 27fc9ccde7863aa0ee38d3e5e404251123e484b4ab0c0ccfa10cb286e5ec96fc

run3 token0 58aa094319aea1340e1cb1823bda826acf646cdda365a1d23e218d433c1c778e
run3 token1 55770ad54c4986092c7be2ba73a16858c4f13319010cc31ab03f46d7365d2eba
run3 token5 b49d69097b664a8bb7ab07d74ec4ce5ebec8b06d76366cb8c144f03c7920e2c0
```

## KV Slot Probe

All three requests had identical request-entry KV state and slot resolution:

```text
request=5000000000 kv_missing=0 kv_inflight=0 kv_resident=0 lane0_blocks=1 pos0_block=0 pos0_slot=0 pos1_block=0 pos1_slot=1
request=5000000001 kv_missing=0 kv_inflight=0 kv_resident=0 lane0_blocks=1 pos0_block=0 pos0_slot=0 pos1_block=0 pos1_slot=1
request=5000000002 kv_missing=0 kv_inflight=0 kv_resident=0 lane0_blocks=1 pos0_block=0 pos0_slot=0 pos1_block=0 pos1_slot=1
```

All three requests had the same per-token device slot mapping:

```text
token_offset=0 position=0 device_slot_mapping=0
token_offset=1 position=1 device_slot_mapping=1
token_offset=2 position=2 device_slot_mapping=2
token_offset=3 position=3 device_slot_mapping=3
token_offset=4 position=4 device_slot_mapping=4
```

The MLA slot probe was stable across all three requests:

```text
mla_slot0=afbb093b2c5ab30d mla_slot0_rope_pair0=2803f37783917bcd
mla_slot1=b089a6b8b7f7d05c mla_slot1_rope_pair0=a5c7457c73d5d410
```

## Layer-Local Fingerprint

For token0, layer 0 and layer 1 were stable across warm requests. Layer 2 was
the first stage0 layer whose hash changed across requests:

```text
run1 token0 layer0=a578f96fa2ec4958 layer1=8de5ca43d597ed62 layer2=6418e8f44997720a
run2 token0 layer0=a578f96fa2ec4958 layer1=8de5ca43d597ed62 layer2=e80bf2d0d7eec14d
run3 token0 layer0=a578f96fa2ec4958 layer1=8de5ca43d597ed62 layer2=c087ce910e3b72da
```

The final rank0 token0 output therefore differs after layer 2, not at ingress,
tokenization, layer 0, layer 1, hidden transport, or slot allocation.

## Artifact Paths

On spark0:

```text
/tmp/sparkpipe_agent_resident_c1307c17.log
/tmp/glm52_kvprobe_run1
/tmp/glm52_kvprobe_run2
/tmp/glm52_kvprobe_run3
/tmp/glm52_receipt_c1307c17_kvprobe_ring
```

## Current Conclusion

The KV allocator drift hypothesis is not supported by this run. Slot mapping,
request-entry KV state, and MLA slot hashes are stable while the generated token
and rank0 token0 output vary.

The most useful next target is stage0 layer 2. The run shows deterministic
layer0 and layer1 output for token0 across warm requests, followed by divergent
layer2 output. That points at layer2-local state or a layer2 workspace/input
dependency, not at request-level slot mapping.
