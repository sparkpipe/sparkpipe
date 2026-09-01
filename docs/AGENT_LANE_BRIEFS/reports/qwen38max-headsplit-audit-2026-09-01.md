# headsplit-design branch — layout audit receipt (2026-09-01)

`tools/qwen38_stagepack_layout_audit.py --git-ref lane/qwen38max-headsplit-design`
(exit 0):

- packer HEADER_STRUCT `<26I2Q` → layout v1 (main's canonical 120-byte
  full-width wire, no per-rank tp fields — the post-pivot format).
- C family header vs python packer: format_version=1, 28 fields, every
  offset and size agrees.
- codec check: 5 C firmware codes checked, every python counterpart
  agrees (MXFP4_E2M1=3 lineage, post-#753).

This retires the divergence flagged on #765 for the head-split lineage:
the branch is wire-consistent with main end-to-end. The 128-byte
tail-order pair from the #753 audit remains only on the retired #754
lane-python lineage (lane-internal tooling).
