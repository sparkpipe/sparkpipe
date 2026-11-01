# Working state

A customer's working state on a fleet is its KV cache (device, host, weightd
pool, NVMe snapshot files and the API's prefix index) and its scratch area.
This note says how the engine and the hub keep the agreed rules in the
background. The rules go into the Customer Agreement; this file describes
what is built to keep them.

## Rules

Fleets are rented whole: one account uses a fleet at a time, and there is no
shared pool (owner, 2026-11-01).

1. **Retention.** Working state stays on a fleet while its account holds the
   fleet, and across gaps in the account's rental if nobody else used the
   fleet. It is deleted when a different account starts using the fleet (at
   the start of its held hour, a lazy delete), or 24 hours after the later of
   its last use and the end of the account's last held hour on that fleet,
   whichever comes first.
2. **Isolation** is per account. With one account per fleet at a time, the
   wipe at every change of user is the isolation: no request ever runs while
   another account's working state is reachable.
3. **Mirror.** Each node's working state is mirrored to its pair partner
   (`rank ^ 1`) over the direct pair link, so a lost or swapped node is
   rebuilt from its partner.
4. **Exclusivity.** A held fleet (a reserved hour or a locked plan) serves
   only its holder during held hours.

A held hour is an hour an account holds on that fleet. Our own use (SparkPipe
Chat, the LAN door, operator tools) is one more account, `sparkpipe`, and
follows the same rules: it wipes a customer's state when it starts, and it
never runs during a held hour. There is no encryption at rest (owner,
2026-11-01): deletion is `unlink` and freeing pages, and a disk leaves the
facility only after a secure erase.

## The code today

Verified on `main` at `1807177be`:

- Cache identity is SHA-256 over the token ids alone, computed once on the
  coordinator, `sparkpipe_model_api` on the hub
  (`runtime/model_batch_engine.c:474-503`). Only the coordinator's prefix
  cache (`cache/prefix_cache.c`) decides how much of a prompt hits; ranks
  restore and reuse pages only for the prefix the coordinator matched
  (`SparkModelPipelineClientCacheHint`, `runtime/model_batch_engine.c:2318`).
  Nothing names an account, so any two callers share prefixes
  (`docs/LITELLM_FRONTEND.md`, "Prefix reuse itself is shared").
- Retention is `SPARK_KV_RETENTION_SECONDS` (86400) from last use
  (`include/sparkpipe/spark_kv_snapshot.h:24-28`), swept every 60 s in the
  page cache, the snapshot store (by mtime), the prefix index and the draft
  servers' contexts (`tools/context_retention.py`), and by weightd's purge
  when no model runs (`node/weightd.c:61-110`,
  `cache/kv_snapshot_purge.c:35-87`). Deletion is `unlink`. The invariant
  "Cached request state expires" says no flag lengthens the limit.
- KV is context-sharded: each node holds every 16th token's KV
  (`include/sparkpipe/spark_kv_shard.h:49-52`) and nothing is replicated, so
  one lost node invalidates every cached context.
- The market gateway forwards requests to the hub with one service key and
  no account (`tools/sparkpipe_pay.py:2269-2276`), and nothing keeps a held
  fleet to its holder (`docs/HOUR_MARKET.md`, "work for the fleet
  controller").
- There is no scratch area.

## Design

### The directive

The hub's working-state controller writes the fleet's retention facts to
`~/sparkdata/working-state/directive` on every node and on the hub, for the
user that runs the engines (the same home weightd's purge walks):

```text
sparkpipe-working-state 1
wiped 1764550800
held_until 1764554400
```

`wiped` is when the fleet's current user started using it; `held_until` is
the end of that user's last held hour (during a held hour, the end of that
hour; 0 when it has none). One function decides every deletion,
`SparkWorkingStateExpired` in `cache/working_state.c` and its twin in
`tools/serving/working_state.py`: state last used at `used` is deleted when

```text
used < wiped  or  now >= max(used, held_until) + 86400
```

Every sweep that applies retention today applies this instead: the
coordinator's prefix cache and index (and peer index imports), the page
cache and snapshot store on each rank, the detached weightd pool, weightd's
purge when no model runs, the scratch area and the draft servers' contexts.
A missing directive is `wiped 0` and `held_until 0`, which is today's rule.
A malformed one is logged on every sweep and treated as missing. The 86400
constant stays; held hours, not a flag, are what lengthen retention.

### Change of user

The controller runs one handover when a request or a held hour of an account
other than the fleet's current user arrives:

1. It stops admitting, and the chat layer cancels every in-flight request of
   the old user.
2. It waits for the engine's `live_requests` to reach 0 (the swap
   controller's drain).
3. It writes `wiped = now` and the new `held_until`, on the hub first and
   then on every node.
4. The engine API checks the directive every second. On a new `wiped` it
   expires its prefix cache at once, rewrites the index and reports the
   applied `wiped` on `/health`. The controller waits for that before it
   admits the new user. From then on nothing can match the old user's
   prefixes, so no node can restore or reuse them.
5. Nodes delete the old pages and files at their next sweep (at most 60 s).
   That is the lazy delete; it is safe because step 4 already made them
   unreachable.

A directive the controller fails to push to a node is retried every minute.
Until it lands, that node keeps old state a little longer, but unreachable.

### Scratch

Each node has a scratch area, `~/sparkdata/working-state/scratch/`, used by
the fleet's current user. Its last use is the mtime of its `.last-use`
stamp, which every access touches. weightd's purge applies the function to
the area as a whole and empties it when it is expired. The customer-facing
way to put and get files, and its quota, come with the autopilot; this change
builds the area, the rule and the mirror.

### Pair mirror and rebuild

`tools/working_state_mirror.py` runs on every node from the node agent. Every
30 s it pushes the node's snapshot stores (found as weightd's purge finds
them) and its scratch area to the partner's
`~/sparkdata/working-state/mirror/<rank>/` over the direct pair link
(`10.10.200.<rank ^ 1>`) with `rsync --archive --delete`, skipping
temporaries and store locks. Snapshot files are content-addressed, so they
are compared by size only and an unchanged file gets just its time; scratch
files are compared by size and time. Times are kept, so the partner deletes
a mirrored file by the same rule, and deletions reach it on the next push.
A replaced node runs `rebuild` before its engines start: it pulls its own
files back from the partner's mirror into stores whose lock nobody holds.
Files written in the 30 s before a node is lost are not in the mirror; a
restore that misses them recomputes, as on any cache miss.

Room: the 4 TB has no space for a second copy, so the 1 TB KV allocation is
500 GB of the node's own snapshot stores plus 500 GB of its partner's, and
the 500 GB scratch allocation is 250 GB own plus 250 GB of the partner's.

### Exclusivity on the hub

The controller lives in the swap controller (`tools/serving/working_state.py`,
loaded by `model_swap.py`, which already owns the node list, SSH and the
drain). It polls the market's holder schedule every 30 s and keeps the last
answer. A schedule older than one hour fails closed: every request gets 503
`holder_schedule_unknown`. It answers `POST /working-state/admit {"account"}`
from the chat layer:

- during a held hour, only the holder is admitted; anyone else gets 503
  `fleet_held` with `Retry-After` at the end of the holder's hours;
- an account other than the current user starts a handover, and the admit
  waits for it to finish;
- model swaps during a held hour are refused unless the holder's request
  asked for them.

The chat layer names the account of every request: the market account in
`X-SparkPipe-Account` on a request authenticated with a gateway key
(`"gateway": true` in `api_keys.json`), and `sparkpipe` for everything else.
It resets its prefix ledgers at a handover. The controller keeps the current
user, `wiped` and `held_until` on the hub (mode 600), so a restart keeps
them.

### Interfaces needed from the payment service and market

These belong to `tools/sparkpipe_pay.py` and `tools/sparkpipe_market.py`,
which are being rewritten elsewhere:

1. **Account on forwarded requests.** The gateway sends
   `X-SparkPipe-Account: <account id>` on every request it forwards to the
   hub, and strips the header from what the customer sends. The hub honours
   it only from a gateway key.
2. **Holder schedule.** `GET /market/fleets/<fleet>/holders?from=<slot>&to=<slot>`
   with a hub service token, answering
   `{"fleet": "...", "now": "...Z", "hours": [{"start": "2026-12-01T01:00Z", "account": "..."}]}`:
   every held hour of that fleet in the window, reservations and locked-plan
   hours alike. The hub asks for the last 25 hours through the next 25.
3. **No shared pool.** The gateway forwards only an account's own held
   hours.

### Contract and invariant changes

- `sparkpipe_invariants.md`, "Cached request state expires", states rule 1
  in place of "within 24 hours of last use".
- `docs/INTERFACE_CONTRACTS.md` retention paragraphs name the directive and
  the function.

## Delivery

1. This note and the engine: the directive, the function in every C sweep,
   the API's immediate expiry and `/health` report, weightd's purge of KV
   and scratch, invariant and contract text. Tests: the wipe at a change of
   user, the 24-hour rule from the later of last use and held hours, and no
   hit on the old user's prefixes after a handover.
2. The hub: the controller, the account header, admission, cancellation,
   swap refusal, the draft servers' contexts. Tests against a fake market,
   fake nodes and a fake engine.
3. The mirror and rebuild, and the node agent's calls. Tests: push, delete,
   rebuild with times kept, and the partner's purge of mirrored files.
