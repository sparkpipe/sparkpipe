# Anticipator: planner, controls and report

`tools/anticipator.py` decides, each night, which leaf packs a fleet builds and keeps warm for the next working day. It covers issues P11 (planner) and P13 (controls and report) of `docs/FLEET_API_ISSUES.md`, against `docs/FLEET_API.md` §5.2–5.7. It needs only the standard library. Real token counts use the `tokenizers` package (a `tokenizer.json`) or `tiktoken` (a `tiktoken.model`, with the split pattern from the model's profile).

The planner reads a git checkout, session evidence, hints and settings, and writes a plan. Building and warming the packs, measuring them on a fleet, and Gate D2 belong to the fleet side; how to run them is at the end.

## What it reads

| Signal | Source | Switched off by |
| --- | --- | --- |
| Structure | Includes and imports parsed from the checkout: C, C++, CUDA and Metal `#include "…"`, Python `import` and `from`, JavaScript and TypeScript relative imports, HTML `src`/`href`. Test files are matched to the files whose name they contain | `signals.structure` |
| History | `git log` of the last 90 days: changes per file, decayed with a 7-day half-life; files changed together (pairs from commits of 2–50 files); the day's commits; files changed on branches pushed in the last 7 days | `signals.history` |
| Sessions | Session evidence records (below) from the last 7 days: reads, edits and searches, decayed with a 2-day half-life | `signals.sessions` |
| Hints | Focus paths, phrases, a task list and pinned leaves | `signals.hints` |

Packs are always rendered least-changed file first, whatever the signals say, as §5.1 requires.

### Session evidence

One JSON object per line, `"format": "sparkpipe-session-evidence-1"`:

```json
{"format": "sparkpipe-session-evidence-1", "session": "47c3609961083c77", "time": "2026-11-05T03:12:09Z", "kind": "read", "path": "tools/serving/chat_frontend.py"}
```

- `kind` is `read`, `edit` or `search`. `path` is relative to the repository and names a file in it.
- A record holds only a path. It never holds file contents or tool arguments.
- `session` is the first 16 hex digits of a SHA-256 of the source and the client's session id, so the raw id is never kept.

The chat layer is to write these records as working state (issue C12). Until then, `anticipator.py evidence` writes them from Claude Code logs (`~/.claude/projects/*/…jsonl`) and Codex logs (`~/.codex/sessions`):
- **Claude Code:** `Read`, `Edit`, `MultiEdit`, `Write`, `NotebookEdit`, `Grep` on a file, and the file paths in `Bash` commands (including `rev:path` arguments).
- **Codex:** `apply_patch` files and the file paths in shell commands.
- **Which logs count:** only those whose working directory is a checkout of the same repository (the same git common directory, or a path under the repository).

## How it plans

1. **Files.** Files enter a pack only if all of these hold:
   - tracked at the commit;
   - not ignored;
   - not excluded: the settings' `exclude` and the defaults (`.env*`, keys and certificates, `vendor/`, `third_party/`, `node_modules/`, minified and lock files, generated protobuf);
   - not binary, not marked generated, and at most 2 MiB.
2. **Heat.** Each file's heat is the sum of its signal weights: sessions × 1, history × 0.5, branches × 0.5, hints × 2.
3. **Candidates.**
   - Each directory with hot files.
   - The eight most strongly tied pairs of hot directories.
   - A working set per recent session: the files that make up 80% of its activity, at most 80. Sessions whose two main directories match share one working set.
   - Each candidate's files are its hot files and their neighbours (dependencies, dependents in the same directory, tests, files changed or used together), ranked by relevance.
4. **Tree.**
   - **Root:** the project instructions (`AGENTS.md`, `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`), the Markdown files they link to, and files that at least three directory candidates share. Each is changed at most 4 times in 90 days. The root is at most 15% of `max_leaf_tokens`.
   - **Trunk:** one per subsystem (the first two path components): stable files that at least two candidate files of that subsystem include. A trunk is at most 25% of `max_leaf_tokens`, and is dropped when it would not fill a quarter of the model's resume boundary.
   - **Leaf:** the rest, best first, until root, trunk and leaf fit in `max_leaf_tokens`.
   - **Padding:** each level is padded to the model's resume boundary (16,384 tokens for Kimi K3).
5. **Expected uses tomorrow.**
   - **Sessions:** a recent session counts as a use when at least 30% of its weighted activity falls in the leaf, decayed by age. The sum is divided by the decayed count of days the account worked, which gives uses per working day.
   - **Hints:** a leaf with focus-hinted files gains 2 uses.
   - **Repeated non-use:** a leaf unused for three nights running loses half its score, and half again for each further night.
6. **Score.** `uses × saved seconds − 0.1 × build seconds − 10 s per GB of cache per node`.
   - **Saved seconds per use:** the pack's tokens at the model's cold prefill rate, less 5 s for the warm start.
   - **Build seconds:** only what tonight must prefill. A level unchanged since the last plan is reused up to the resume boundary before its first changed file.
   - **Why build time counts a tenth:** prefill at night runs on an idle fleet, so a second of it costs much less than a second of an engineer waiting. Without that weight, a leaf would need more than one use a day before it paid for itself, and most of the night would go unused.
7. **Selection.**
   - **Pinned leaves** come first, if they fit.
   - **Then greedy:** the best score per GB of new cache, counting a shared root or trunk once, until the night's compute (`max_hours`, or the whole window) or the cache budget (`cache_share` of the per-node cache) runs out.
   - **No double counting:** a session counted for one leaf is not counted for another.
   - **Skipped candidates:** every one is listed with its reason.

Same inputs, same plan. Nothing depends on the clock except the `now` the run is given, and the plan's id is a digest of every input.

## Model profiles

`tools/anticipator_models.json` holds, per model:
- the cold prefill rate;
- the resume boundary;
- the cache bytes per token per node;
- the cache per node;
- the window;
- for tiktoken models, the split pattern.

Each value names its source. A model with a missing figure is refused, naming the figure: the cache bytes per token are measured only for Kimi K3 (`docs/FLEET_API.md` §5.3).

## Controls and the API

`Store` keeps an account's settings, focus, plans, approvals and reports as JSON files. `api(store, method, path, query, body)` answers the routes of §5.6 and returns `(status, body)`, so the chat layer can serve them directly:

| Route | Does |
| --- | --- |
| `GET /v1/anticipator`, `PUT /v1/anticipator` | Settings. Every field of §5.6 is checked; an unknown field or a bad value answers 400 |
| `GET /v1/anticipator/plan?date=` | The plan, with `build` (what will be built) and `approval_required` |
| `POST /v1/anticipator/plan/approve` | `{"leaves": "all" \| [names]}`. Only in `plan` mode (otherwise 409); names outside the plan answer 400 |
| `PUT /v1/anticipator/focus` | `{"paths", "phrases", "tasks", "pins", "date"}` |
| `GET /v1/anticipator/report?date=` | The morning report |

- **Modes:** `off` plans and builds nothing. `plan` builds only approved leaves. `auto` builds the whole plan.
- **Hints:** focus hints change the next plan.
- **Exclusions:** excluded paths never enter any level.

## The morning report

`report(plan, built, records, start, end, previous)` reads the telemetry records of the working day: the request records of `GET /v1/requests` (§3.6), with `time`, `session`, `pack`, `prompt_tokens`, `cached_tokens` and `cold_prefill_ms_estimate`. It returns:
- the leaves built and why;
- sessions started, and the share whose first request named an anticipated leaf (`pack`, with any `model+` prefix and `@version` stripped);
- for those sessions, the first-turn prompt tokens and the cached share;
- the prefill time avoided on anticipated leaves;
- the built leaves no request used, and their build hours;
- `unused_nights` for the next plan.

Every figure is a sum over the records, so it reconciles with them.

## Command line

```sh
python3 tools/anticipator.py evidence --repo . --claude-logs ~/.claude/projects/-Users-me-repo --out evidence.jsonl
python3 tools/anticipator.py plan --repo . --settings settings.json --evidence evidence.jsonl --tokenizer tiktoken.model --out plan.json
python3 tools/anticipator.py settings --store state --set mode='"plan"'
python3 tools/anticipator.py approve --store state --leaf core/work-tools+site
python3 tools/anticipator.py focus --store state --path 'site/**' --phrase "checkout"
python3 tools/anticipator.py report --store state --records requests.jsonl
```

`sparkpipe anticipate …` (§15) is to call the API routes once they are served.

## Customer Zero receipt

`qualification/anticipator/20261105/` holds the plan for this repository at `13bb07a47`, from 14 Claude Code sessions of the last 7 days (10,759 evidence records), with Kimi K3's tokenizer, the default settings and the window 19:00–09:00 Asia/Seoul. It ran in 3.6 s and gives the same plan on every run. See its `README.md`.

## Not built here

- **Serving it:** the chat layer does not yet serve the routes or write session evidence (C12), and the gateway does not forward `/v1/anticipator*`.
- **Building packs:** the pack builder (P4) and pack trees (P10) do not exist yet. The plan gives each level's files, blobs, tokens and padding, which is what they consume.
- **Structure:** it comes from regular expressions over the checkout until the repository index (P3) exists. Token counts are per file and summed.
- **Scheduling:** it runs in the night window after any Function that unloads the model, as the Function runner (F1) schedules it.

## For the CUDA dev: running it on the fleet

1. Push the repository to the fleet (P3), or use a checkout on the hub.
2. Write session evidence for the account from the chat layer (C12), or with `anticipator.py evidence` from the account's own logs.
3. Plan with the served model's tokenizer:
   ```sh
   python3 tools/anticipator.py plan --repo <checkout> --store <state> --model kimi-k3 --tokenizer <tiktoken.model>
   ```
4. For each leaf in `GET /v1/anticipator/plan`'s `build` list, render root, trunk and leaf in order, each padded to the boundary, and warm it with `POST /v1/cache/warm` at background priority. Stop at the window's end.
5. Run Gate D2 for 10 working days in `auto` mode with default budgets. Make each day's report from `GET /v1/requests` and check:
   - at least 60% of sessions start on an anticipated leaf;
   - at least 90% of those sessions' first-turn prompt tokens come from the cache;
   - the night's prefill stays within `budgets.compute_seconds`.
