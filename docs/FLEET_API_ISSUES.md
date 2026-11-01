# Fleet API issues

The build list for `docs/FLEET_API.md`. Everything here ships for 4 January 2027.

Each issue is one piece of work with a test that closes it. Sizes: **S** is up to a week, **M** one to two weeks, **L** three to four weeks. Issues of different components touch different files, so they can run as parallel sessions. "Needs" lists the issues that must land first.

## Engine and working state

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| E1 | Merge working state: held-hours retention, wipe at a change of user, pair mirror (#1691, #1692) | A customer's cache survives a 14-hour gap with the fleet held, and is gone within a minute of another account's first request | M | |
| E2 | Snapshot store at the full per-node budget (500 GB own, 500 GB partner) | Capacity, write and restore rates and crash recovery measured at the full budget and recorded in `PERFORMANCE_STATUS.md`. Today only 8 GiB per rank is proven | L | E1 |
| E3 | Prefix index survives a release and a restart | After a hub release and a coordinator restart, a 150K conversation's next turn is a cache hit | M | E1 |
| E4 | Bound K3's recurrent-state store | A 3,000-request K3 conversation runs to the end; today it fails after about 977 | M | |
| E5 | Commit-only requests (prefill, no generation) | `commit_only: true` commits the prompt and returns its token count and prefill time, with no token generated | S | |
| E6 | Pins | A pinned chain outlives an LRU sweep that clears every unpinned chain, and pins are refused beyond half the cache | M | E1 |
| E7 | Streamed prefill publishes its blocks | A 1M-token prompt prefilled once is answered from the cache by a follow-up, first token under 10 s (Gate C) | L | E2 |
| E8 | Priority classes end to end | A background request never delays an interactive request by more than one decode step, and two sequences stay free for interactive use while background work runs | M | |
| E9 | Cancellation | A client disconnect frees the sequence within one decode step | S | |
| E10 | Structured outputs | Grammar-constrained sampling from a JSON schema: 1,000 schema-bound requests all parse and validate | L | |
| E11 | Parallel samples (`n` up to 8) | Eight samples share one prefill of the prompt, and their aggregate decode speed is recorded | M | |
| E12 | Log probabilities and seeds | `logprobs`, `top_logprobs` up to 20; the same seed, prompt and build give the same tokens | S | |
| E13 | Speculation at batch 2 and more | Two concurrent whole-file rewrites both speculate; tok/s per stream recorded | L | |
| E14 | Pair-loss report | When both nodes of a pair lose their working state, the hub reports which contexts and storage paths were lost, within a minute | M | E1 |
| E15 | Bounded model switch with rollback | A switch finishes in under 90 s or rolls back to the previous model; no failed switch leaves the fleet without a model (`TECHDEBT.md:424-437`) | M | |
| E18 | Several resident models per fleet | A per-node memory budget for each model's weights and cache. A set that fits loads side by side (GLM 5.3 with GLM 5.3 Flash), and each model serves at once with its own cache pool. A request for one never evicts the other's cache, and a set that does not fit is refused with the numbers | L | E15 |
| E16 | Move working state between held fleets of the same build | `POST /v1/state/move` copies snapshots and storage, and the moved conversation's next turn is a hit on the new fleet | M | E2 |
| E17 | Images on vision-capable models | An image block in each of the three API shapes is answered by a vision model, and refused with `image_not_supported` by others | M | C1, C2 |

## Chat layer (hub)

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| C1 | `POST /v1/messages` (§3.3) | Claude Code runs a 300-turn session with tools, thinking and subagents through it (Gate A) | L | |
| C2 | `POST /v1/responses` (§3.4) | Codex runs a 100-turn session with tools through it, with and without `previous_response_id` | L | |
| C3 | Token-exact replay store (§3.2) | In C1's and C2's sessions, every turn after the first matches the cache up to the end of the previous reply | M | C1, C2 |
| C4 | Telemetry (§3.6) | Every response carries the `sparkpipe` object. `cold_prefill_ms_estimate` is within 10% of a measured cold prefill at 50K, 150K and 500K tokens | M | |
| C5 | `count_tokens`, `tokenize`, `detokenize`, `HEAD /api/hello`, model listing with packs and custom models | Counts equal the engine's prompt token counts for 100 recorded requests | S | |
| C6 | Alias table and residency rules | A `claude-haiku-*` request goes to the small resident model, or to the main one. A request for a model that fits loads it without unloading anything. One that would need another model unloaded gets 409 `model_not_loaded` from a key without `allow_swap` | S | E18 |
| C7 | Keep-alives and busy handling | A 10-minute prefill keeps a Claude Code and a Codex stream open. When all sequences are busy, requests wait up to 60 s, then get 429 with `retry-after` | S | |
| C8 | Fair share between members | With two members flooding requests, neither gets more than three quarters of the sequences while the other waits | M | G5 |
| C9 | Pack prefixing with version affinity (§5) | A session keeps its pack version through a refresh. New sessions get the new version only once it is warm | M | P4 |

## Gateway and account (VPS)

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| G1 | Pass the new paths through | `/v1/messages`, `/v1/responses`, `/mcp` and the rest reach the hub. `X-Api-Key` works, bodies go up to 32 MiB, and holder checks apply to every path | S | |
| G2 | Key options | Scopes, `fleet`, `allow_swap`, expiry, IP allowlist and rotation (the old key valid for an hour) are enforced and tested | M | |
| G3 | Usage records and `GET /v1/requests` | Records arrive from the hub per request, are kept a year, and sum into `stats` | M | C4 |
| G4 | Bearer sessions for the CLI on `/market/...` | The CLI books a week and joins the waitlist with a bearer session | S | |
| G5 | Teams | Invites, roles (owner, admin, member), per-member keys and usage; a member cannot book or pay | L | G2 |
| G6 | Audit log | Every event in §12 is logged and listed by `GET /v1/audit`, without prompts or replies | M | G5 |
| G7 | Webhooks and email notifications | Every event in §13 is delivered signed, with retries. `booking.deadline` fires 48 h and 6 h before the Sunday deadline | M | |
| G8 | Auto-extend | A fleet with auto-extend keeps its weeks booked ahead across a simulated year, and a short balance triggers an early warning | M | G7 |
| G9 | Several fleets: prefix-affinity routing | With two held fleets and `fleet: "any"`, a session's turns all land on the fleet where its prefix is warm | M | C4 |
| G10 | Automatic rebuild credit (Agreement 6.6) | A pair-loss report from E14 credits 24 hours of the fleet within an hour, once per event, and sends `working_state.lost` | S | E14, G7 |

## Storage, repositories and packs

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| P1 | Customer storage service on the nodes' scratch area | `PUT`, `GET`, `DELETE` and list with resumable chunks, a 4 TB quota and the working-state deletion rule; a 50 GB upload resumes after a cut | L | E1 |
| P2 | Fetch (§6.1) | A 1.5 TB Hugging Face model at a pinned revision is fetched across the fleet's nodes with parallel ranges, survives a cut connection, matches every published SHA-256, and lands in storage without passing through the VPS. `--then pack` puts it in the model list | L | P1, P8, P9 |
| P8 | Tokens stay with the customer (§6.2) | The token arrives with the request and is held only in the fetch worker's memory. A disk and log scan after a fetch finds no trace of it, and it is never sent to any host but the source's, even through a redirect. A worker restart pauses the fetch and `fetch resume` continues it. The CLI keeps tokens in the OS keychain | M | |
| P9 | Egress guard for fetch | Requests to private, loopback and link-local addresses, to our own hosts, and redirects to any of them are refused. DNS answers are pinned per connection. A test suite of rebinding and redirect attacks passes | M | |
| P3 | Repositories pushed to the fleet | `git push` to `/git/{repo}.git` with the SparkPipe key updates the fleet's copy, and each push triggers the scheduled refreshes. A tree-sitter symbol index and a semantic index are built on the sandbox host, and a 1M-line repository is indexed in under an hour | L | P1, F3 |
| P4 | Pack builder | Least-changed files first, deterministic rendering, budget and versions; the same commit always gives the same tokens | M | P3 |
| P5 | Pack refresh and warm-up | After a push, the new version is warm in the background, and only the part after the first changed file is prefilled again | M | P4, E5, E8 |
| P6 | Export and restore | An archive of storage, drafters and pack definitions restores on another fleet | M | P1 |
| P7 | Daily backup by pull | `GET /v1/storage/changes` and `sparkpipe storage sync` copy only the day's changes to a local directory or a bucket the customer's machine writes to; the morning report shows each path's last pull | M | P1, L1 |

## Functions and the sandbox host

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| F1 | Function runner (one per fleet) | Runs, schedules in the account's time zone, logs, results and the morning report, all at background priority | L | E8 |
| F2 | Functions that read code and call the model | Review the day's changes, second opinions, ask the whole codebase, keep docs current, morning report, benchmark your repository, resume sessions, prefill ahead, scheduled model, rebuild after a loss: each runs on the Customer Zero repository (Gate E) | L | F1, P4 |
| F3 | Sandbox host per fleet | Hardware chosen and installed per fleet. Firecracker microVMs with the repository copied in and no network by default, destroyed after each run, and the host wiped at a change of user | L | |
| F4 | Agent loop | Our own loop, published with the engine, with read, search, edit and run tools against the sandbox, finishing a 20-task list on the Customer Zero repository | L | F3 |
| F5 | Functions that run code | Tests and fuzzing, confirmed findings, security audit, overnight agent, codebase-wide change, evaluate models: each runs on the Customer Zero repository (Gate E) | L | F3, F4 |
| F6 | Batch jobs (§11) | A 10,000-request JSONL file runs overnight at background priority, ordered by shared prefix, with results in storage | M | F1 |

## Codebase tools over MCP

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| M1 | MCP server at `/mcp` (§10) | Claude Code, Codex and OpenCode each call every tool. `run` executes the test suite with the local diff applied on the sandbox host | L | P3, F3 |
| M2 | `sparkpipe mcp` stdio bridge | A client that only starts local servers uses every tool through it | S | M1, L1 |

## Draft models

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| D1 | Nightly drafter training from `drafter/nightly-retrain` | The Function swaps K3 out, trains, swaps back, and compares acceptance on the account's own recent requests; the better drafter stays active | L | F1, E15 |
| D2 | Per-fleet drafter selection | `PUT /v1/drafters/active` loads an account's drafter where that fleet's drafts run; the drafter is wiped at a change of user | M | E1 |

## Customer models

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| X1 | Tier A: a packer per offered family for fine-tunes | A fine-tune of Qwen 3.8 27B and one of Gemma 4 31B go from storage to the account's model list, run on the sandbox host. Pickle-based weights (`.bin`, `.pt`, `.pkl`) are refused | L | P1, F3 |
| X2 | Tiers B and C: the driver guide and review | A coding agent writes a tier B driver from the guide alone. Review and qualification pass, and the model serves through all three API shapes (Gate F) | M | X1 |
| X3 | Private model lists | A customer model appears only in its account's listing, and is removed with the working state at a change of user | M | X1 |

## CLI

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| L1 | `sparkpipe` core | `login`, `keys`, `status`, `book` and `waitlist` | M | G4 |
| L2 | The rest of §15 | Every command in §15, with tests against the sandbox service | M | grows with each component |
| L3 | `sparkpipe proxy` | A tool that cannot set headers reaches the right fleet through `localhost`, with 429s retried | S | L1 |
| L4 | Publishing | `pip install sparkpipe` installs the single-file CLI, built from this repository on each tagged release | S | L1 |

## The three harnesses (§3.8)

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| H1 | Claude Code: `sparkpipe setup claude-code`, the plugin (MCP server, slash commands, `SessionStart` warm-up hook, skill) and `sparkpipe statusline` | On a fresh machine, one command later Claude Code finishes a 50-turn task using the fleet's `run` and `search_code`. After a 3-hour pause and `--resume`, the hook has warmed the pack and the first turn is a cache hit | L | C1, M1, L1, E5 |
| H2 | Codex: `sparkpipe setup codex` (profile, provider, subagent model, MCP) and `GET /v1/codex/models` | `codex --profile sparkpipe` finishes a scripted session with tools and MCP, and knows the context window from the catalogue with no manual setting | M | C2, M1, L1 |
| H3 | Codex WebSocket transport on `/v1/responses` | Codex with `supports_websockets` finishes the same session over WebSocket; time per turn recorded against HTTP | M | C2 |
| H4 | OpenCode: `sparkpipe setup opencode`, the plugin, and the models.dev entries | OpenCode, through its `@ai-sdk/anthropic` provider, finishes a scripted session with tools, MCP and cache hits after the first turn. Sharing is disabled and no request leaves for opencode.ai | M | C1, M1, L1 |
| H5 | Hint headers and affinity keys | Claude Code's compaction and auxiliary requests run at background priority. Its session and agent ids, Codex's `prompt_cache_key` and OpenCode's session keep each conversation on its pack version and fleet | S | C1, C2, E8, C9 |
| H6 | Release checks for the three | Each new release of Claude Code, Codex or OpenCode runs its scripted session against staging within a day: cache share, tool calls, MCP, subagents. The result goes on an internal status page before customers are told to upgrade | M | H1, H2, H4 |

## Site

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| S1 | Account page: Functions, Your models, Team, Webhooks, Backups and Stats tabs | Each tab drives its API in the live service and the sandbox | L | F1, X3, G5, G7, P7, G3 |
| S2 | API reference on the site | A page built from `docs/FLEET_API.md`, in English and Korean, linked from the home page and the account page | M | |

## Acceptance tests

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| T1 | Gates A–F of `docs/FLEET_API.md` §18 | Each gate run on Fleet 1 and recorded in `PERFORMANCE_STATUS.md` before 4 January 2027 | M | all |
