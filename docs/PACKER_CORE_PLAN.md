# spark_pack_common.py — shared packer core plan (coordinator)

Covers DRY_CONSOLIDATION_PLAN.md items 3 (shared packer core) and 5 (one
replicated-draft rule). Inventory verified 2026-08-17 against the unified
tree; line numbers are from the current packers.

## Verified duplication inventory

| Primitive | Copies (file:line) |
| --- | --- |
| PackFailure | glm52_resident:90, dsv4:116, qwen38_27b:102, qwen38:111 |
| sha256 helpers | glm52_stagepack:508, glm52_resident:439, dsv4:181, qwen38_27b:106, qwen38:115 |
| align/align_up | glm52_stagepack:180, glm52_resident:125, dsv4:370, qwen38_27b:117, qwen38:126 |
| SafetensorsSource reader (check_config/shard_header/resolve) | dsv4:396, qwen38_27b:266, qwen38:285 |
| TensorRef + build_inventory + kind_shape + layer_tensor_name + is_gdn_layer | qwen38_27b (215/225/127/162/121) mirrored by qwen38 (233/261/135/173/130) |
| Record/directory/header serialization | dsv4:754-812 + glm52_resident pack writer + qwen pair |
| Replicated-draft rule | dsv4 sharder (MTP block, sentinel 0xFFFFFFFB, count 3, flag-gated) vs qwen38 (MTP_LAYER=0xFFFFFFFE, count 1, sentinel compare) — two incompatible encodings of the same rule |

The glm52_stagepack torch-based reader (TensorSource.load) is a third
reader style; it folds only once the torch dependency is contained
behind the shared core's shard-reader interface.

Corrections from the GLM52 agent's inventory (grep-verified):
- The GLM52 packers are DRAFT-FREE (0 mtp/draft/dspark tokens). They pack
  the 78 base layers only; the DSpark draft is a separate model via
  tools/glm52_dspark_manifest.py and the MTP layer is a separate weight
  file. The replicated-draft rule is therefore a DSV4+qwen consolidation,
  NOT a GLM52 consumer.
- tools/glm52_resident_pack_common.py (99 lines) is a DORMANT seed already
  holding PackFailure/SafetensorReader/align_up/tp_shard_range - imported
  by nothing and side-effect-loading the GLM52 contract at import. It is
  the natural starting point for the neutral module once the side-effect
  import is removed.
- PackFailure drift: RuntimeError in 6 files, Exception in qwen38_27b/qwen38
  (inconsistent), plain ValueError in glm52_stagepack. The shared core
  standardizes on RuntimeError.

## Proposed module layout (tools/spark_pack_common.py)

\`\`\`python
# errors + hashing + alignment
class PackFailure(RuntimeError)
def sha256_file(path: Path) -> str
def sha256_bytes(data: bytes) -> bytes
def align_up(value: int, alignment: int) -> int

# safetensors streaming reader (header-only, no torch)
class SafetensorsSource:
    def __init__(self, root: Path)
    def check_config(self) -> None
    def shard_header(self, shard: str) -> dict
    def resolve(self, name: str) -> tuple[str, dict, int]   # shard, meta, payload offset
    def check_shape(self, ref, meta) -> None

# draft (speculation) tensor placement — THE replicated-draft rule
DRAFT_SENTINELS = {"dsv4": (0xFFFFFFFB, 3), "qwen38": (0xFFFFFFFE, 1)}
def draft_layer_bounds(kind: str, family: str) -> tuple[int, int] | None
def draft_rows_replicated(kind: str, layer: int, family: str) -> bool

# pack header/directory serialization (bytes in, bytes out - no policy)
def pack_header(records, first_layer, layer_count, file_bytes, codecs) -> bytes
def pack_entry(entry) -> bytes
def make_directory(records) -> tuple[list, int]
\`\`\`

Everything above is pure data movement: no model constants, no topology
policy. The per-model config tables (geometry, kinds, sentinels) stay in
each packer, as they already do for the sharder's MODEL_GEOMETRY.

## The one replicated-draft rule

Today the DSV4 sharder and the qwen38 packer each encode "draft tensors
replicate full-width to every rank" with a private sentinel + count. The
shared rule takes the family's sentinel table entry and answers the
replication question; each packer calls it in exactly one place (the DSV4
sharder in plan_entry, the qwen38 packer in build_inventory) and deletes
its private branch. GLM52's DSpark backend has no in-pack draft tensors
today; when the 0813-style draft packing arrives for Pro/K3, they add a
table entry instead of a new branch.

## Migration order (byte-identical, pinned by the existing tests)

1. Land tools/spark_pack_common.py with the primitives above.
2. Migrate the DSV4 sharder first (test_dsv4_tp4_pp4_stagepack.py +
   test_dsv4_stagepack.py pin the bytes) - replace its PackFailure/sha/
   align/make_directory with the shared ones; delete the local copies.
3. Migrate the qwen pair next (test_qwen38_27b_stagepack.py +
   test_qwen38_stagepack.py pin the qwen38 case); qwen38_27b stays
   frozen-deprecated and keeps working through the shared core.
4. Migrate the glm52 resident packer (test_glm52_stage_pack.py pins the
   v3 wire format bytes).
5. Fold glm52_stagepack's torch reader behind the shared reader interface.
6. Last: the dsv4 full packer's geometry table (H2) - it is the biggest
   single win but touches the most qualified pack; do it after the shared
   core has four green adopters.
