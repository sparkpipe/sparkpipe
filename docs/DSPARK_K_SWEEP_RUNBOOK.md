# DSV4 Flash DSpark k-sweep — runbook (re-pack → build → sweep)

Follow-up to the measurement hour. Baseline locked: **40.19 tok/s mean no-spec B1,
3/3 exact** (driver `3d962820`). This doc records the exact procedure to close
the k-sweep (k=5/7/8/10) now that downloads are authorized. No commits/pushes.

## Root cause recap (why the first hour blocked)

- The b8 spec driver (sha `95f09489`) fails at `adapter_initialize` with
  `dsv4_stage pack_geometry_mismatch field=format_version`.
- The deployed pack is **format_version 3** (header `44535634 03000000`); the
  current source is **format_version 4** (`spark_dsv4_stagepack_format.h:31`).
  The 3→4 bump landed in `84efd5b` ("DSpark P1 - pack the 3 checkpoint draft
  layers (stagepack v4)"). So the v3 pack has **no DSpark draft weights** —
  re-packing at v4 is mandatory, not a re-stamp. Every pack on spark4-7 is v3.

## Phase 1 — re-pack at format_version 4 (download in progress)

Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b`, 48 shards, ~143 GB.
Partial dir `/home/spark4/srcdata/dsv4_flash.fp8.pp13` had 3 shards; a background
downloader (`download_dsv4_flash.py` + `dsv4_flash_contract.json` in that dir) is
filling the rest, verifying sha256 against `model_contracts/dsv4_flash.json`
(`source_files` is a **dict** filename→{bytes,sha256}). ~31 s/shard ⇒ ~22 min.
Progress: `tail download.log`.

When `DONE fail=0`, re-pack (contract `runtime.packed_mtp_layer_count=3` ⇒ the
packer appends the 3 draft layers):

```sh
# on spark4
cd /tmp/sparkpipe-devcycle
python3 tools/dsv4_stagepack.py \
  --model-dir /home/spark4/srcdata/dsv4_flash.fp8.pp13 \
  --first-layer 0 --layer-count 43 \
  --output /home/spark4/srcdata/dsv4_flash.fp8.pp13/dsv4_flash_stage_v4.spstage
```

Confirm `xxd -l 8` shows `44535634 04000000`, then re-shard/deploy per rank
(the rank-local TP shard step the lean runtime already uses) into each runtime's
`packs/dsv4_flash_stage.spstage`.

## Phase 2 — buckets 6/9/11 + SPEC_STEP override (k-sweep builds)

k is compile-time: spec gate is `SPARK_BATCH_BUCKET == SPEC_STEP + 1`
(`module.c:1989,2531,5607`); `SPEC_STEP 7u` (`spark_dsv4_model.h:43`). Only
bucket 8 (k=7) is valid today; buckets 6/9/11 (k=5/8/10) are rejected by
`spark_dsv4_batch_tuning.h:30-35`.

Required edits (model dir, dsv4-flash lane):
1. `spark_dsv4_batch_tuning.h`: add `SPARK_BATCH_BUCKET != 6u`, `!= 9u`, `!= 11u`
   to the allowlist, plus b6/b9/b11 module-ID macros and the `#elif` entries.
2. Make `SPEC_STEP` overridable: guard `spark_dsv4_model.h:43` with
   `#ifndef SPARK_DSV4_MODEL_DSPARK_SPEC_STEP`.
3. Generate `dsv4_resident_decode_stage_firmware_b{6,9,11}.json` (the generator
   that produced b1/b8), since `build_remote.sh` copies `firmware_b<BUCKET>.json`
   and pins its sha into the driver.

Build one per k (bucket = k+1, spec step = k):

```sh
./tools/devcycle.sh build lean-dspark-k5  6    # + -DSPARK_DSV4_MODEL_DSPARK_SPEC_STEP=5u
./tools/devcycle.sh build lean-dspark-k7  8
./tools/devcycle.sh build lean-dspark-k8  9
./tools/devcycle.sh build lean-dspark-k10 11
```

Greedy-only is acceptable for the first sweep (probabilistic OFF — note it in
the receipt; the verify path is greedy Leviathan at `module.c:3582-3586`).

## Phase 3 — run the sweep end-to-end

For each k driver: `devcycle setup dspark-kN` → `devcycle deploy dspark-kN
tools/devcycle/drivers/lean-dspark-kN` → `devcycle ready` → `devcycle run
dspark-kN 3` (exact O128 hash `a9385d0b…` is the output-identity gate — greedy
DSpark must reproduce it). Record per run: tok/s, and from
`/tmp/devcycle-dspark-kN-rank0.log` the `dspark_accept … accepted=N` lines ⇒
tok/step = mean(1+N), per-position acceptance p[i] = P(N>i). Label every receipt
`spec=k7-greedy` etc.

## Infrastructure fixes (landed 6fd4dc8 — redeploy before relaunch)

- Unit: `Environment=LD_LIBRARY_PATH` removed; fleet_swap.sh writes
  `LD_LIBRARY_PATH` into `/etc/sparkpipe/residentd.env`. **Redeploy the unit on
  spark4-7** (copy `tools/devcycle/sparkpipe_model_residentd.service` to
  `/etc/systemd/system/` + `daemon-reload`; remove the temp `20-ldpath.conf`
  drop-in). Then `fleet_swap.sh dsv4-flash` starts the band under MemoryMax=108G.
- fleet_swap ssh now `ConnectTimeout=8` + skip-unreachable (no glm52 hang).

## Status

- ✅ baseline 40.19 tok/s (3/3 exact) · ✅ download running (~22 min) ·
  ⏳ re-pack → ⏳ buckets/rebuild → ⏳ k-sweep.