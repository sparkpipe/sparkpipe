# Datafile Naming: packs, recipes, and KV cache entries

Canonical grammar for every model-derived datafile the ring produces or
consumes. Implemented by `tools/generate_recipe.py`, held honest by
`tests/test_recipe_generation.py`. The grammar exists so that a file's name
is its contract: anything that changes the bytes changes the name, and a
name that matches is a proof the bytes fit.

## Model tags

Six lowercase, dot-free tags — one per served model descriptor. The tag is
the datafile's first field; contract file names are provenance, not names.

```text
tag      authoritative contract
k3       model_contracts/k3_authoritative.json
dsv4     model_contracts/dsv4_flash_authoritative.json   (DeepSeek-V4-Flash-0731 GA)
dsv4pro  model_contracts/dsv4_pro_authoritative.json     (DeepSeek-V4-Pro)
glm52    model_contracts/glm52.json
qwen36   model_contracts/qwen36_authoritative.json
mimo25   model_contracts/mimo25_authoritative.json
```

## Packs and recipes: `<model>.<strategy><n>.<content-hash>`

```text
k3.TP16.<content-hash>.json            recipe: TP over 16 nodes
k3.PP13.<content-hash>.json            recipe: PP over 13 stages
k3.TP16.<content-hash>.rank00.pack     rank 0's shard of the TP16 pack
```

- `<strategy>` is `TP` (tensor-parallel, every rank runs every layer on its
  shard) or `PP` (pipeline-parallel, every stage owns a contiguous layer
  range). `<n>` is the node/stage count.
- `<content-hash>` is the first 16 hex of SHA-256 over the canonical JSON
  (sorted keys, compact separators) of the recipe body — contract path and
  contract SHA, strategy, degree, topology, shard table or stage plan, KV
  geometry. **Any** input change — a contract edit, a different topology, a
  rebalanced stage plan — produces a different name, so a stale pack or
  recipe can never be picked up under a fresh recipe's identity.
- Per-rank pack files extend the grammar with `.rank<rr>` after the hash;
  the three-field prefix stays intact so a directory listing groups a
  strategy's whole pack set.

## KV cache entries: `<model>.<strategy><n>.<geometry-hash>/...`

```text
k3.TP16.<geometry-hash>/<sequence-id>/layer<ll>/page<pp>
```

The `<geometry-hash>` is the first 16 hex of SHA-256 over the canonical JSON
of the model's **KV geometry**: exactly the contract fields that change KV
*content* — layer counts by attention kind, head counts and head dims, KV
head counts, latent widths (`kv_lora_rank`, rope dims), KV element type,
rope convention and thetas, sliding-window extents, cached value scales,
recurrent-state shapes for the hybrid models. Nothing else enters: MoE
expert counts, intermediate widths and precision policy move the
content-hash (a new recipe, new packs) but leave KV bytes interpretable, so
they leave the geometry-hash standing.

This is what makes TP16↔PP16 switching with NVMe resume safe:

- the geometry-hash is strategy- and degree-free by construction —
  `k3.TP16` and `k3.PP16` carry the **same** geometry-hash;
- after a strategy switch, an NVMe-resident KV entry is served to the new
  recipe only when the geometry-hash matches — a model whose KV geometry
  drifted under the same tag (a contract edit that changed a latent width,
  a head dim, a rope convention) hashes differently and the old entries are
  unreachable, never misread;
- the strategy+degree field stays in the path because TP and PP *place* KV
  differently (per-rank shards vs per-stage ownership); placement is
  re-laid-out on adoption, content is not re-computed.

## What this is not

`SparkTpShardGeometryHash` (serving/spark_tp_shard.c) is the pack-to-kernel
*binding* hash: FNV-1a over one tensor's shard view so a kernel refuses a
shard image built under another degree or geometry. It binds bytes already
named by this grammar; it does not name anything. The two hashes compose —
naming gets the right files to the node, the binding hash refuses the wrong
ones at the kernel boundary.

## The runtime side: kv_namespace

The switch state machine (include/sparkpipe/spark_topology_switch.h) names
NVMe tier records by `SparkTopologySwitchKvKey(namespace, content_hash)`,
where `namespace` commits to the model identity and the neutral cache
geometry — layer count, slot bytes, block tokens, precision — and
deliberately excludes the strategy. The `<geometry-hash>` of this grammar is
the same commitment in datafile-name form: a recipe's geometry-hash and the
tier's namespace must move on exactly the same contract edits, or the
offline naming and the runtime keys would disagree about what KV content
is. One refinement on the runtime side: the tier stores ONE neutral record
per block (layer-major) that both strategies slice on read, so the
`<strategy><n>` component of the KV path above scopes placement and
planning; content adoption across a switch is decided by the
geometry-hash == namespace match, never by the strategy field.

## Generation and CI

`tools/generate_recipe.py` (no flags) regenerates
`examples/recipes/<tag>.<TP|PP><n>.<hash>.json` for every known contract at
n=16 (dual-switch production topology, read from
`examples/topologies/dual_switch_16node_production.json` when present) and
n=13 (current ring); `--check` verifies the set is current without writing,
and is wired into `tools/gates.sh`. A contract edit that lands without
re-running the generator fails the gate on the stale filename alone.
