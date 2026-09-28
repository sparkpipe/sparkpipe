# Recipe and artifact naming

Pack directories and runtime roots are named by
[STAGEPACK_NAMING.md](STAGEPACK_NAMING.md). This file defines the serving
recipe names that `tools/generate_recipe.py` emits and
`tests/test_recipe_generation.py` checks.

No pack in the tree uses the content-addressed
`<model-id>.<placement-id>.<precision-id>.rank<rank>.<kind>.<sha256>` scheme
this file used to describe. A pack's identity is its arm name plus its
`.sha256` sidecar digest.

## Recipe datafiles

    <tag>.<strategy><degree>.<content-hash>.json

- `tag`: a key of `MODELS` in the generator (k3, dsv4, dsv4pro, glm52,
  qwen38_27b, mimo25); lowercase letters, digits and underscores
  (`DATAFILE_RE`).
- `strategy`: `TP` or `PP`.
- `degree`: the rank or stage count; the defaults are 16 and 13
  (`DEFAULT_DEGREES`).
- `content-hash`: the first 16 hex digits of the SHA-256 of the recipe body
  serialized with sorted keys and no whitespace, taken before `content_hash`
  and `datafile` are added. It covers the contract path and SHA-256, the
  topology, the KV geometry, the TP shard table or PP stage plan, and any
  stage-capacity profile.

Recipes are written to `examples/recipes/` (for example
`dsv4.TP16.28dd8d71ef616cfc.json`); `--check` fails when the committed set
differs from a fresh generation.

## KV entry prefix and geometry hash

    kv_entry_prefix = <tag>.<strategy><degree>.<geometry-hash>/
    geometry-hash   = first 16 hex digits of SHA-256 of {"family", "kv_geometry"}

`kv_geometry` holds only the contract fields that change KV content (layer
counts, head dimensions, latent widths, KV dtype, rope conventions), chosen
by each family adapter in the generator. `test_geometry_hash_invalidation`
checks the properties:

- TP and PP recipes of one model share the geometry-hash at any degree.
- A latent-width change mints a new geometry-hash; an expert-count change
  keeps it and moves the content-hash.

The prefix also carries strategy and degree, so the TP16 and PP16 prefixes
of one model differ although their geometry-hash is equal. Reusing KV across
a strategy switch must key on `geometry_hash`, not on the prefix. No runtime
code reads `kv_entry_prefix` today.

## Rules for every model-derived artifact

- A consumer accepts a file only when its expected identity and dependent
  contract hashes match. Similar names, tensor geometry, rank count,
  precision labels or file length never authorize reuse.
- A checkpoint revision is an exact upstream revision or an immutable
  internal conversion revision, never a branch, alias or marketing name.
- Partial files use a temporary suffix and cannot be discovered as ready;
  publication is atomic after length and SHA-256 validation.
