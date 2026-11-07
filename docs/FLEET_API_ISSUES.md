# Fleet API issues

The build list for `docs/FLEET_API.md`. Everything here ships for 4 January 2027.

Each issue is one piece of work with a test that closes it. Sizes: **S** is up to a week, **M** one to two weeks, **L** three to four weeks. Issues of different components touch different files, so they can run as parallel sessions. "Needs" lists the issues that must land first.

State on 6 November 2026:

- **Merged to main:** G8 auto-extend (#1763), G11 uptime records and the downtime credit (#1767), G12 failover (#1769), and the CLI's L1, L3 and L4 (#1749).
- **Built on main, gate not yet run:** C1 `/v1/messages`, C2 `/v1/responses` and C5's endpoints.
- **Everything else is open.** An open PR names the IDs it closes, and its merge moves them to the first line.

## Engine and working state

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| E1 | Merge working state: held-hours retention, wipe at a change of user, pair mirror (#1691, #1692) | A customer's cache survives a 14-hour gap with the fleet held. It is gone within a minute of another account's first request, and within 24 hours of the holder's last booked hour when nobody else books the fleet | M | |
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
| E19 | Restore a fleet to our own image at a change of user | Before a new holder's first hour, every node of the fleet runs our released image again, checked against the release's hashes. A file the previous holder's driver or Function left anywhere outside its working state and storage is gone, and the restore is recorded with the image hash | M | E1 |

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
| C12 | Session evidence | For each session, the chat layer records the file paths in tool calls (reads, edits, searches) and the pack it started on, as working state, never file contents. A scripted session's evidence lists exactly the files it touched, and switching the signal off records nothing | S | C1, C2 |

## Gateway and account (VPS)

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| G1 | Pass the new paths through | `/v1/messages`, `/v1/responses`, `/mcp` and the rest reach the hub. `X-Api-Key` works, bodies go up to 32 MiB, and holder checks apply to every path | S | |
| G2 | Key options | Scopes, `fleet`, `allow_swap`, expiry, IP allowlist and rotation (the old key valid for an hour) are enforced and tested | M | |
| G3 | Usage records and `GET /v1/requests` | Records arrive from the hub per request, are kept a year, and sum into `stats` | M | C4 |
| G4 | Bearer sessions for the CLI on `/market/...` | The CLI books a week and joins the waitlist with a bearer session | S | |
| G5 | Teams | Invites, roles (owner, admin, member), per-member keys and usage; a member cannot book or pay; the gateway sends `X-SparkPipe-Member` for fair share per member (C8) | L | G2 |
| G6 | Audit log | Every event in §12 is logged and listed by `GET /v1/audit`, without prompts or replies; the hub reports its own events (model switches, wipes, storage access, fetches, runs) to `POST /hub/audit` | M | G5 |
| G7 | Webhooks and email notifications | Every event in §13 is delivered signed, with retries. `booking.deadline` fires 72 h, 48 h, 6 h and 2 h before the Sunday deadline, and when auto-extend waits or stops | M | |
| G8 | Auto-extend | On by default from the first booked or granted week, one week ahead, with a one-click off switch and the weeks-ahead setting on the account page. A fleet with auto-extend keeps its weeks booked ahead across a simulated year, a holder who does nothing keeps the fleet, and a short balance triggers an early warning | M | G7 |
| G9 | Several fleets: prefix-affinity routing | With two held fleets and `fleet: "any"`, a session's turns all land on the fleet where its prefix is warm | M | C4 |
| G11 | Uptime records and the automatic downtime credit (Agreement 4.4) | The VPS probes each Fleet's hub every 30 seconds and records every period without a good probe, gaps in its own probing included. Every booked hour a period of more than 15 minutes touches is credited once at the Fleet's Rate, as soon as it passes 15 minutes, daily, and at its end, never twice with maintenance. `status.html` and its Korean twin show status, uptime and periods from those records | M | |
| G12 | Failover to a free fleet (Agreement 15.7) | Each fleet has its own upstream. While a held fleet is down, its holder's requests go to the first free, serving, healthy fleet as that fleet, without working state, and come back when it serves again; a booking of the free fleet takes it back. The free fleet's holder schedule names the account, and `fleet.failover` says where the account is served | M | G11 |
| G13 | Delete an account's request records on request (DPA 9.3) | `DELETE /v1/requests` and `sparkpipe_pay.py erase-records` delete the VPS's usage records of the account at once, and records sent again for that time are refused. The hub's `request_record` lines, which hold the reply's token numbers, need a per-account store on the hub that can be deleted the same way | S | G3 |
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
| P10 | Pack trees (§5.2) | A tree of one root, three trunks and twelve leaves stores the root and trunks once (cache bytes measured), each level ends on the model's resume boundary, and moving a session from one leaf to another recomputes only the leaf's tail | M | P4, E6 |
| P11 | Anticipator planner (§5.3) | On the Customer Zero repository it turns structure, history, session evidence and hints into a plan within the compute and cache budgets, in under 5 minutes, with reasons for every leaf. The same inputs always give the same plan | L | P3, P10, C12 |
| P12 | Automatic pack selection, `+auto` (§5.4) | Claude Code, Codex and OpenCode sessions started in three different subdirectories each land on the expected leaf at their first request, keep it for the whole session, and report it in telemetry; an unmatched session starts without a pack | M | C9, P10 |
| P13 | Anticipator controls and report (§5.6, §5.7) | Every setting is enforced, and `plan` mode builds nothing until approved. Focus hints change the next plan, and excluded paths never enter a pack. The morning report's figures reconcile with the telemetry records | M | P11, G3, F1 |
| P6 | Export and restore | An archive of storage, drafters and pack definitions restores on another fleet | M | P1 |
| P7 | Daily backup by pull | `GET /v1/storage/changes` and `sparkpipe storage sync` copy only the day's changes to a local directory or a bucket the customer's machine writes to; the morning report shows each path's last pull | M | P1, L1 |

## Functions and the sandbox host

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| F1 | Function runner (one per fleet) | Runs, schedules in the account's time zone, logs, results and the morning report, all at background priority | L | E8 |
| F2 | Functions that read code and call the model | Review the day's changes, second opinions, ask the whole codebase, keep docs current, morning report, benchmark your repository, resume sessions, prefill ahead, scheduled model, rebuild after a loss: each runs on the Customer Zero repository (Gate E) | L | F1, P4 |
| F3 | A sandbox for each fleet | Firecracker microVMs with the repository copied in and no network by default, destroyed after each run. Each fleet's sandbox has its own reserved CPU, memory and disk, shares nothing with another fleet's, and is wiped at a change of user | L | |
| F4 | Agent loop | Our own loop, published with the engine, with read, search, edit and run tools against the sandbox, finishing a 20-task list on the Customer Zero repository | L | F3 |
| F5 | Functions that run code | Tests and fuzzing, confirmed findings, security audit, overnight agent, codebase-wide change, evaluate models: each runs on the Customer Zero repository (Gate E) | L | F3, F4 |
| F6 | Batch jobs (§11) | A 10,000-request JSONL file runs overnight at background priority, ordered by shared prefix, with results in storage | M | F1 |
| F7 | Train an adapter (§7.2) | An adapter for an offered model and one for a customer model train from a dataset in storage, land in storage, and serve on their base model through all three API shapes | L | F1, X1 |
| F8 | Make training data (§7.2) | A dataset lands in storage, generated only from models whose licence allows training another model on their outputs. A model without that licence is refused by name, and the licence list is published | M | F1 |

## Codebase tools over MCP

| ID | Issue | Done when | Size | Needs |
| --- | --- | --- | --- | --- |
| M1 | MCP server at `/mcp` (§10) | Claude Code, Codex and OpenCode each call every tool. `run` executes the test suite with the local diff applied on the sandbox host | L | P3, F3 |
| M2 | `sparkpipe mcp` stdio bridge | A client that only starts local servers uses every tool through it | S | M1, L1 |
| M3 | Language servers (§10 `definition`, `references`, `hover`, `diagnostics`, `rename_preview`) | On the Customer Zero repository, warm servers for C and C++, Python, Rust, Go and TypeScript answer every tool, and `diagnostics` matches the project's own build within 30 s of a push | L | P3, F3 |
| M4 | Warm builds and affected tests (§10 `run`) | On the Customer Zero repository, a second `run` of the test suite after a one-line change reuses the build caches and runs only the affected tests, at least five times faster than the first, with the same result as a full run | L | M1, F1 |
| M5 | Semantic search over code and history (§10 `search_code`, `search_history`) | For 50 written questions about the Customer Zero repository, the right file or commit is in the top five answers for at least 40 of them | M | P3 |
| M6 | Dependency mirror (§10 `deps`) | The pinned dependency versions of three sample repositories (pip, npm, cargo) are mirrored, hash-checked and searchable; turning the mirror off deletes it | M | P2, P9 |
| M7 | Headless browser (§10 `browse`) | A scripted check of a web app served by `run` returns screenshots and the console and network errors, with no network beyond the sandbox unless allowed | M | F3 |
| M8 | Fan-out reading (§10 `map`) | 1,000 files are summarized in under 10 minutes with no decode by the main model, and interactive latency on the main model does not change meanwhile | M | M1 |

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
| T1 | Gates A–F and D2 of `docs/FLEET_API.md` §18 | Each gate run on Fleet 1 and recorded in `PERFORMANCE_STATUS.md` before 4 January 2027 | M | all |
