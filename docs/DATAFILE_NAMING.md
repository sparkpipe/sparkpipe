# Model Artifact Naming

Every model-derived file is immutable and content-addressed. Human-readable
fields aid operations; SHA-256 identities establish compatibility.

## Canonical identity

```text
<publisher>.<model>.<checkpoint-revision>.<artifact-kind>.<content-sha256>
```

Names use lowercase ASCII, digits, and hyphens. The checkpoint revision is the
exact upstream revision or an immutable internal conversion revision. A mutable
branch, alias, or marketing name is not a revision.

## Rank-local shards

```text
<model-id>.<placement-id>.<precision-id>.rank<rank>.<kind>.<sha256>
```

Examples of `kind` are `weights`, `stagepack`, `kv-layout`, `driver`, and
`collective-profile`. The manifest binds full hashes for:

- model and checkpoint contracts;
- source implementation revision;
- tokenizer and prompt template;
- placement and hardware topology;
- precision and pack recipe;
- exact payload bytes; and
- every dependent artifact.

## Placement identity

Placement names describe the model plan, not a permanent cluster product:

```text
tp4
tp8
tp16
tp4-pp2
tp4-pp4
```

Node count, rank map, PP slices, TP communicators, direct pairs, and storage
roles remain inside the hashed placement manifest. Two placements with the same
short name but different rank ownership produce different content identities.

## Storage tiers

The same verified artifact may appear in the external pooled store, external
direct tier, internal active-shard tier, or a mounted release. Tier and path are
not part of artifact identity. Promotion verifies bytes after every transfer or
reflink and before binding.

Partial files use a noncanonical temporary suffix and cannot be discovered as
ready artifacts. Publication is atomic after length and SHA-256 validation.

## Compatibility rule

A consumer accepts a file only when its expected full identity and dependent
contract hashes match. Similar model names, tensor geometry, rank count,
precision labels, or file length never authorize reuse.
