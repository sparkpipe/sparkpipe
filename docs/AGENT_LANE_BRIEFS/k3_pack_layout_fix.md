# k3-finish: delegated — repair test_k3_pack_layout.py fixture (offline-gates red)

> EXECUTED BY THE COORDINATOR 2026-08-29 (~14:2x): subagent spawns were down
> (provider misconfiguration after the app restart), so this brief was done
> directly on main. See the coordinator log. REMAINING for k3-finish: item 2
> of the brief below (the fleet wave) — item 0 (report) landed as 0be9505 on
> lane/k3-finish.
>
> 2026-08-30 session 2: the wave is STAGED and window-gated — stage-2 was
> found built against the wrong slice (48_23; the runner's stage table wants
> 47_23) and rebuilt; stage-0 deployed with receipts; the taker chain lives
> in reports/k3-finish-2026-08-30b.md section 7. The window went to
> lane-glm5attractor at 17:01 UTC; k3-finish holds fire until the next one.

Owner: k3-finish lane (worktree /tmp/lane-k3finish). Delegated by the
coordinator at merge b0daf75 (2026-08-29); this is the ONLY remaining red
gate in `make offline-gates`.

## The bug (diagnosed, do not re-derive)

`tests/test_k3_pack_layout.py::mini_checkpoint` builds a synthetic Kimi-K3
checkpoint that predates two packer changes:

1. `2d30fec` pinned the released-checkpoint `language_model.` tensor prefix
   and `text_config` nesting. The coordinator already fixed the prefix in
   b0daf75 (writer now emits `language_model.` + name, same pattern as the
   passing `tests/test_k3_pack.py`). Config top-level still works — the
   packer nests only if `text_config` exists.
2. `55cd2f9` full-rank gate reconciliation (read `docs/K3_GATE_RECONCILIATION.md`
   first — it is the spec). The fixture still emits the OLD scheme:
   - writes `self_attn.g_a_proj.weight` + `g_b_proj.weight` (low-rank pair);
     the packer now reads **full-rank `self_attn.g_proj.weight`**
     (shape `[kda_dim, hidden]`, shard_class output_dim_heads)
   - `self_attn.f_a_proj.weight` is now the standalone replicated
     `kda_decay_down_weight` `[kda_head, hidden]` — check the fixture's
     current shape for it
   - verify against `tools/k3_pack.py` lines ~640-700: conv1d trio
     (`q_conv1d`/`k_conv1d`/`v_conv1d` F32 `[kda_dim, 1, kernel]`),
     `dt_bias` F32 `[kda_dim]`, `A_log` F32 read at `A_LOG_SOURCE_HEADS`
     then sliced `[:kda_heads*4]` (fixture must supply the source length),
     `o_norm.weight` F32 `[kda_head]`, `o_proj.weight` `[hidden, kda_dim]`,
     and the full-attention (non-linear) layer's MLA spellings in the
     `else:` branch — the fixture's layer-1 names must match what that
     branch reads.

The packer's expectations are ground truth — the released checkpoint is
what ships. Fix the FIXTURE, not the packer. The failure mode to preserve:
`PACK FAILURE: missing tensor: <name>` stays loud for real drift.

## Acceptance

- `python3 tests/test_k3_pack_layout.py` green end-to-end (its checks —
  fused-section tiling, interleave grid closure, relay addressing,
  128-alignment — must still assert, not be weakened).
- `make offline-gates` fully green (ratchet re-run if code size moves;
  tests/ is excluded from the counter, so likely unmoved).
- Commit on the lane, push, report in docs/AGENT_LANE_BRIEFS/reports/,
  request integration. Do NOT merge main yourself.

## Fleet context

CPU-only work — no reservation, no nodes. Main is at b0daf75 (W1 loader
merged); rebase the lane on it first (`git fetch origin && git rebase
origin/main` — expect zero conflicts, your lane's last main-side base is
recent). spark5 GPU windows are probe-fix-gated; irrelevant for this task.
