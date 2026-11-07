# Anticipator plan for Customer Zero, 5 November 2026

The first plan of `tools/anticipator.py` for this repository ("Customer Zero") at `13bb07a47`. It is a planning receipt for issue P11: no pack was built or warmed, and nothing ran on a fleet.

## Inputs

| Input | Value |
| --- | --- |
| Commit | `13bb07a47` (main), 5,577 tracked files, 4,565 of them plannable |
| Excluded | 735 by pattern or `.gitignore`, 262 binary, 10 generated, 5 over 2 MiB |
| History | 6,957 commits in the 90 days before the run; 40 branches pushed in the last 7 days |
| Sessions | 14 Claude Code sessions of the last 7 days from this workstation (the main checkout and 8 worktrees): 10,759 evidence records, paths only |
| Hints | None |
| Model | `kimi-k3`: 460 tok/s cold prefill, 16,384-token resume boundary, 1.9 KB of cache per token per node (`tools/anticipator_models.json`) |
| Tokens | Kimi K3's own tokenizer (`tiktoken.model`, SHA-256 `b6c497a7469b33ce…`), with the split pattern of its `tokenization_kimi.py` |
| Settings | `settings.json`: defaults, with the model and the window 19:00–09:00 Asia/Seoul |
| Run | `--now 2026-11-05T10:00:00Z` (19:00 KST) |

The evidence file is not committed: it is the account's working state.

## Result (`plan.json`, plan `1bdf292ea6b8cc8b`)

- **Runtime:** the planner ran in 3.6–3.9 s, plus 5.3 s to read about 460 MB of session logs. P11 asks for under 5 minutes.
- **Determinism:** two runs on the same inputs gave identical plans, apart from the runtime field.
- **Budgets:** 50,400 s of night compute and 250 GB of cache per node, half of 500 GB. The plan uses 2,351 s and 2.05 GB.
- **Root:** 10 files, 29,922 tokens, padded to 32,768:
  - the project instructions (`AGENTS.md`, `README.md`, `CONTRIBUTING.md`);
  - the stable Markdown files they link to.

| Leaf | Files | Pack tokens | Expected uses | Build | Why |
| --- | ---: | ---: | ---: | ---: | --- |
| `core/work-tools+site` | 43 | 393,216 | 1.45 | 855 s | 8 recent sessions worked mostly in its files; 614 commits in 90 days |
| `core/scripts` | 34 | 360,448 | 0.26 | 712 s | 2 sessions; 1,412 commits |
| `core/model-families-hy4-device` | 31 | 393,216 | 0.24 | 784 s | 3 sessions; 449 commits; 27 include links inside it |

**Skipped candidates:** 289 of the 292.
- 279 because their expected uses do not pay for their build time and cache.
- 10 because their files do not fit within `max_leaf_tokens` once the root is in.

## What it says about the method

- Most of this week's sessions were broad: the lead session alone read 446 files across the repository. So one working-set leaf covers most of the focused work, and the rest of the night budget stays unused.
- Gate D2 measures whether that is the right call: at least 60% of sessions starting on an anticipated leaf over 10 working days. That needs the fleet, P10 and C12 (`docs/ANTICIPATOR.md`).
