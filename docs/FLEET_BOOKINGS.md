# Fleet bookings: hour-credits, Hours, Weeks and the waitlist

This is the design of the payment and booking service for the elite-developer offer. The SparkPipe Customer Agreement (`site/terms/agreement-*.md`) governs it, and this document follows its clause numbers; where they differ, the Agreement wins. The published settings live in `site/pricing.js`. The code is `tools/sparkpipe_market.py` (bookings, the waitlist, auto-extend, keys, the gateway's checks, notices, the rebuild credit and maintenance) and `tools/sparkpipe_pay.py` (accounts, sessions, orders, payment processors, the ledger, the gateway to the hub, usage records, webhook delivery and the HTTP service). The customer-facing API is `docs/FLEET_API.md`; this document is how the VPS side of it works. Payments are described in `docs/PAYMENTS.md`.

## What a customer buys

- **Hour-credits (HC).** HC is the only currency. It is bought in any quantity from `hc.min`, to six decimal places (HC is counted to six decimals, clause 13.2), with no maximum.
  - **Price per HC** depends on the quantity in one purchase: the `hc.prices` entry with the largest `from` at most the quantity sets the price, in won (`krw`) for Toss Payments and in US dollars (`usd`) for stablecoins and USD bank transfers (SWIFT). Today 143 HC cost $2,860 or ₩3,718,000 and 144 HC cost $2,160 or ₩2,808,000, before VAT: 10% (`vatRate`), or 0% (`swiftVatRate`) on a USD bank transfer (SWIFT) under clause 3.3(b) (`docs/PAYMENTS.md`).
  - **Lots.** Each purchase, sandbox purchase and credit is a lot that lapses `hc.validityMonths` (12) months after it is issued, on the same day of the month and at the same time (a lot issued on 29 February lapses on 28 February).
  - **Spending.** HC is spent when a booking is made, from the lot that lapses first. Spent HC has left its lot, so it never lapses, however far ahead the booking runs (clause 13.3). Unspent HC in a lot lapses with it.
- **Hours** (clause 14). Whole hours on one Fleet, starting on the hour and not yet started, each costing `bookings.hourHC` × Rate.
  - **On the Demo Fleet** (`"demo": true`): from `demoFrom` (00:00 KST), ending at most `bookings.demoDaysAhead` days from now. Each account may hold at most `bookings.demoHoursPerWeek` Demo Fleet hours in each Week; an Hour booking that crosses midnight between Sunday and Monday counts in both Weeks.
  - **On any other serving Fleet** that is not pinned and that nobody holds in those hours, ending at most at the end of the current Week (Sunday 24:00 KST), so an Hour never blocks a Monday handover. A Fleet pinned in the current Week takes no Hours, from anyone, its holder included.
  - No Hour overlaps announced maintenance.
- **Weeks** (clause 15). Monday 00:00 to Sunday 24:00 KST (Sunday 15:00 UTC to Sunday 15:00 UTC), on a serving Fleet other than the Demo Fleet. A booked Week pins the Fleet: nobody else books any of its hours or Weeks.
  - **Price:** `bookings.weekHC` × Rate × (1 − discount) for each Week booked. The discount is the largest `bookings.discounts[].off` whose `weeks` is at most N, where N is the account's booked Weeks on that Fleet that have not ended, the current Week included, plus the Weeks being booked. It applies only to the Weeks in this booking. With today's settings: 3 Weeks cost 432 HC, 4 cost 547.2 (5% off), 13 cost 1,684.8 (10%), 26 cost 3,182.4 (15%) and 52 cost 5,990.4 (20%).
  - **First Week:** an account that holds no Week of the Fleet that has not ended books from the next Monday 00:00 KST, on a Fleet free for that Week, and only while no waitlist lock could pay for a Week of that Fleet (refusal `waitlist`): the waitlist comes first and gets free Fleets at the Sunday allocation. No Week starts before `weeksFrom` (outside the sandbox).
  - **Extension:** an account that holds Weeks of the Fleet books the Weeks right after its last booked Week; there is no other choice of start, so extensions are contiguous. The last Week may start at most `bookings.maxWeeksAhead` (52) Weeks after the current Week, and one booking holds 1 to 52 Weeks.
  - **Keeping the Fleet:** the holder's deadline is 00:00 KST on the Sunday of its last booked Week. Before it, nobody else may book the following Week. At the deadline the Sunday round (below) may allocate the Fleet to the waitlist; if it does not, the holder (as an extension) or anyone else (as a first Week, which is then the next Monday) may book that Week until it starts.
- **Waitlist** (clause 16).
  - **Joining:** lock at least `bookings.weekHC` HC of the unlocked balance; afterwards any amount may be added. Locked HC is taken out of its lots, recorded by lot, and cannot be spent on bookings. Leaving unlocks all of it back into its lots.
  - **Operator email:** each lock created or increased emails `SPARKPIPE_PAY_NOTIFY` (subject `Waitlist lock: <company>, <lock> HC (+<change>)`) with the account, the new lock, the change, the position and the number of locks. The mail is queued after the lock is written, so a mail that cannot be sent is only logged and never fails the lock.
  - **Order:** largest lock first, then earliest join (the ledger sequence of the first `lock` line). Adding keeps the join time. A customer sees only its own lock and position. The public state and `availability.txt` show only how many accounts hold a lock (`waitlist.locks`): no positions, amounts or identities.
  - **The Sunday round:** at each Sunday 00:00 KST, each serving non-demo Fleet, in Fleet List order, that has no booking in the following Week goes to the first customer in order whose lock pays for at least one Week on it; that customer leaves the order. Its lock books the largest number of consecutive Weeks from the next Monday, up to `maxWeeksAhead`, whose price under clause 15.2 is at most the lock, and the rest of the lock is unlocked. The price is not monotone in the count (25 Weeks at 10% off cost 3,240 HC, 26 at 15% cost 3,182.4), so a lock of 3,200 HC books 26 Weeks. The customer is emailed.
  - **Lapse while locked:** when a lot lapses, the HC it has locked is removed from the lock and recorded as a refund due (`lock_lapse`): for a purchased lot, its share of the order's total with VAT, in the order's currency, through the order's processor; credited lots carry no refund. `SPARKPIPE_PAY_NOTIFY` is emailed the list. A lock that falls below `bookings.weekHC` leaves the waitlist and the rest is unlocked.
- **Auto-extend** (`docs/FLEET_API.md` §16). `PUT /market/auto-extend {"fleet", "weeks_ahead"}`, with a session and the current terms, keeps `weeks_ahead` (1 to `maxWeeksAhead`) Weeks booked after the current Week on a Fleet the account holds Weeks of (409 `not_pinned` otherwise); `weeks_ahead: 0` turns it off.
  - **Booking:** the setting books at once what is missing, and then, at each Monday 00:00 KST when the count drops, one more Week as an extension, at the discount for the Weeks held, from the account's unlocked HC. The extension is due on the Monday of the last booked Week at the latest, so it always comes before the holder's Sunday deadline. Each booking is a `booking` line with `auto: true` and sends `booking.confirmed`.
  - **Short balance:** when the HC for a due extension is missing, it sends `booking.deadline` at once, early, with `early: true`, the `price`, the `balance` and `hc_needed`, once for that Week; it books as soon as the balance covers it. If the pinned Weeks run out unextended, or the Fleet is taken or stops serving, auto-extend stops (`auto_extend` with `weeks: 0` and a `reason`) and the account is emailed.
- **Maintenance** (clause 4.3): `sparkpipe_pay.py maintenance FLEET MONDAY [HOURS]` schedules up to `bookings.maintenanceHoursPerWeek` hours per Fleet per Week, starting at least 7 days ahead, on the consecutive hours with the fewest booked hours, then the KST start hour least used by that Fleet's earlier maintenance (so the times rotate), then the earliest. Holders it overlaps are emailed at once. When it ends, every booked hour it overlaps is credited as a lot of HC equal to the booking's Rate (clause 4.4) and a `maintenance_done` line is written.

## Events and their order

Time-driven events run whenever the service touches the market (every request and a 30-second tick), in time order, and at the same instant lapses first, then the round, then maintenance credits, then auto-extend, then deadline notices:

1. a lot that lapses while some of it is locked;
2. the Sunday round, for each Sunday 00:00 KST after the later of the last recorded round and the earliest join on the waitlist (there is no round while the waitlist is empty);
3. the end of a maintenance period;
4. an auto-extension that is due;
5. `booking.deadline` 48 hours and 6 hours before 00:00 KST on the Sunday of an account's last booked Week of a Fleet. Each is sent once (the `once` key of its `notice` line); after an outage only the latest mark that has passed is sent, and none once the deadline has passed. Extending moves the deadline, so the next notices follow the new last Week.

Each event is written with the instant it happened (`now`), so a service that was down, or a sandbox whose clock jumped, catches up with the same result.

## Ledger lines

The hash-chained ledger (`docs/PAYMENTS.md`) holds these market lines, each with `now` where it matters:

| Type | Data |
| --- | --- |
| `hc_lot` | `lot`, `account`, `hc`, `expires`, `source` (`purchase`, `credit`, `maintenance`, `rebuild`), `ref` (the order, credit, maintenance, or `<fleet>:<event>` of a pair loss), optional `note` |
| `hc_debit` | `account`, `hc`, `reason`, `ref`, optional `lot` (a refund's debit) |
| `booking` | `booking`, `account`, `fleet`, `kind` (`hour` or `week`), `start`, `end`, `hc`, `rate`, `hours` or `weeks` and `discount`, `source` (`hc`, or `waitlist` with `unlocked`), `auto` for an auto-extension |
| `lock`, `unlock` | `account`, `hc`; `unlock` gives `reason` (`leave` or `below`) |
| `lock_lapse` | `account`, `lot`, `hc`, `ref`, `refund` (`order`, `processor`, `currency`, `amount`, or null) |
| `round` | `sunday` |
| `api_key`, `api_key_revoke`, `api_key_rotate` | API keys (only the SHA-256 of a key is kept) with `note`, `scopes`, `fleet`, `allow_swap`, `expires`, `ips`, `priority` and `rotated_from`; a rotation's `new` key and the old key's `retires` |
| `notice` | `notice`, `account`, `type` (an event of `docs/FLEET_API.md` §13), `data`, `summary`, `text`, `targets` (webhook ids and `mail`), optional `once` |
| `webhook`, `webhook_delete` | `webhook`, `account`, `kind` (`url` or `email`), `target`, `events`, `slack`, `hint` (the secret's last four characters; the secret itself is in `webhook_secrets.json`) |
| `auto_extend` | `account`, `fleet`, `weeks` (0 turns it off), optional `reason` |
| `pair_loss` | `fleet`, `event`, `account`, `at`, `pair`, `contexts`, `storage`, `run`, `hc`, `lot` |
| `maintenance`, `maintenance_done` | `maintenance`, `fleet`, `start`, `hours`; `credited` |
| `clock` | the sandbox clock offset |

The live ledger's lines from before this design (`account`, `acceptance`, an `hourly` order and its invoice states) replay unchanged; that account is asked to accept the new Agreement.

## API

Everything is under `/pay/api`, and the sandbox is under `/pay/sandbox/api`. Times are UTC strings: `stamp` (`2027-01-06T03:00:00Z`) for instants and `slot_text` (`2027-01-10T15:00Z`) for hours. HC amounts are strings with six decimals. A session is the page's cookie or, for the CLI, `Authorization: Bearer sps-…` from the email-code sign-in (`docs/PAYMENTS.md`). Refusals are `{"error", "reason"}` with the HTTP status; the reasons a page acts on are `terms` (409: accept the current Agreement and send `accept: true`), `insufficient_hc` (402), `pinned`, `taken`, `too_far`, `not_open`, `started`, `demo_cap`, `not_weekly`, `maintenance`, `lock_too_small` (409).

| Method and path | Auth | Request | Answer |
| --- | --- | --- | --- |
| `GET /market/state` | optional session | | `now`, `sandbox`, `vatRate`, `demoFrom`, `weeksFrom`, `hc` (`prices`, `min`, `validityMonths`), `bookings` (the settings), `models`, `waitlist` (`locks`: the number of accounts that hold a lock), and `fleets`: each Fleet List entry with `serving`, `status` (`demo`, `pinned` or `free` now), `firstWeek` (the next Monday as a slot when a new customer can book it as a first Week now, else null), `booked` and `maintenance` (merged `[start, end]` spans from now on, with no identities). With a session, `account`: `email`, `company`, `termsCurrent`, `balance` (unlocked HC), `locked`, `position`, `joined`, `lots`, `bookings` (not ended more than a Week ago, each with `state` `upcoming`, `live` or `ended`), `pinned` (`fleet`, `weeks` not ended, `through`, `deadline`, `canExtend`), `refunds` (locked HC that lapsed), `keys` |
| `POST /market/quote-weeks` | session | `fleet`, `weeks` | `fleet`, `weeks`, `first`, `last` (start of the last Week), `end`, `count` (N), `discount`, `price`, `extends` |
| `POST /market/book-hours` | session, current terms | `fleet`, `start`, `hours` | the booking: `booking`, `fleet`, `kind`, `start`, `end`, `hours`, `hc`, `source`, `created`, `state` |
| `POST /market/book-weeks` | session, current terms | `fleet`, `weeks` | the quote with `booking` |
| `POST /market/waitlist` | session, current terms | `lock` | `locked`, `position` |
| `POST /market/waitlist/leave` | session | | `locked` (`0.000000`), `position` (null) |
| `POST /market/keys`, `POST /market/keys/revoke`, `POST /market/keys/rotate` | session | the key options below; `id`; `id` | the key with `key` (shown once); `id`, `revoked`; the new key with `replaces` and `retires` |
| `PUT /market/auto-extend` | session, current terms | `fleet`, `weeks_ahead` | `fleet`, `weeks_ahead`, `through` |
| `POST /market/sessions/revoke` | session | `id` | `id`, `revoked` |
| `POST /order` | session | `product: "credits"`, `hc` (from `hc.min`, to six decimals), `processor` (`toss` with optional `brn`, `coinpayments` with `coin`, or `swift` with `declare: true`), `accept` when the terms changed | `docs/PAYMENTS.md` |
| `POST /market/buy` | session | `hc` | sandbox: issues the HC at once (`lot`, `hc`, `expires`); with `processor: "toss"`, or on the live service, the same as `/order` |
| `GET /market/ledger` | session | | the account's own ledger lines with their hashes (clause 11.7) |
| every path of `docs/FLEET_API.md` §2 that the fleet serves | API key or session | passed through unchanged (below) | the hub's answer, unchanged |
| `GET /v1/requests` | session, or a key with `account:read` | `since`, `until`, `key`, `fleet`, `model`, `session`, `limit`, `after` | `data`, `totals`, `has_more`, `next` |
| `GET /v1/keys`; `POST /v1/keys`, `DELETE /v1/keys/<id>`, `POST /v1/keys/<id>/rotate` | session, or a key with `account:read` to list; session to change | the key options | as `/market/keys` |
| `GET /v1/webhooks`; `PUT /v1/webhooks`, `DELETE /v1/webhooks/<id>` | session, or a key with `account:read` to list; session to change | `url` or `email`, `events`, `secret`, `slack` | the webhooks, below |
| `GET /market/fleets/<fleet>/holders?from=<slot>&to=<slot>` | upstream key | | `fleet`, `now`, `hours`: `[{start, account}]` |
| `POST /hub/requests`, `POST /hub/events`, `POST /hub/pair-loss` | upstream key | below | below |
| `GET /availability.txt` | none | | text, below |
| `POST /market/sandbox/clock`, `POST /market/sandbox/reset` | session, sandbox only | `hours` | `now`; `reset` |

## The gateway

The gateway passes every path of `docs/FLEET_API.md` §2 that the fleet serves through to the hub's chat layer over the tunnel (`SPARKPIPE_POOL_UPSTREAM`, `http://127.0.0.1:18081/v1`; other paths go to the same host without `/v1`). Paths outside that list, such as the hub's own `/v1/sparkpipe/...` controls, and any path with a `.` or `..` segment, even percent-encoded, answer 404 and never reach the hub.

- **Credentials.** `Authorization: Bearer sk-hc-…`, `X-Api-Key: sk-hc-…` (what Claude Code sends with `ANTHROPIC_API_KEY`), or Basic authentication with the key as the password (what git sends; a `/git/` request without one gets 401 with `WWW-Authenticate: Basic realm="SparkPipe"`). The account session (cookie or bearer) works too, with every scope and `allow_swap`; a cookie request from another `Origin` is refused.
- **Scopes.** Each path needs one scope of the key: `inference` for `/v1/messages`, `/v1/responses`, `/v1/chat/completions`, `/v1/models`, `/v1/codex/models`, `/api/hello`, `/v1/tokenize`, `/v1/detokenize`, `DELETE /v1/requests/{id}`, `/v1/batches` and `/mcp`; `state` for `/v1/state`, `/v1/cache`, `/v1/packs` and reading `/v1/models/resident`, `/v1/models/aliases` and `/v1/drafters`; `models` for changing those and for `/v1/custom-models`; `storage` for `/v1/repos`, `/git/`, `/v1/storage`, `/v1/export`, `/v1/restore` and `/v1/fetch`; `functions` for `/v1/functions`, `/v1/runs`, `/v1/schedules` and `/v1/reports`. The hub gets the key's scopes too, so the MCP server can refuse tools by scope.
- **Holder check on every path.** The Fleet is the key's `fleet` if it has one; otherwise `X-SparkPipe-Fleet`, or `sparkpipe.fleet` in a chat completions body; otherwise the one Fleet the account holds now. An account that holds no Fleet, or not the one named, gets 403 `fleet_not_held`; one holding several and naming none gets 409 `fleet_required`. `POST /v1/state/move` needs both `from` and `to` held, and goes to `from`.
- **Headers to the hub.** The customer's headers pass unchanged, without hop-by-hop headers, `Host`, `Authorization`, `X-Api-Key`, `Cookie`, `X-Forwarded-*` and every `X-SparkPipe-*` header other than `X-SparkPipe-Priority` and `X-SparkPipe-Source-Token`. The gateway adds `Authorization: Bearer` the upstream key, `X-SparkPipe-Public: 1`, `X-SparkPipe-Account`, `X-SparkPipe-Fleet`, `X-SparkPipe-Key` (the key id, or `session`), `X-SparkPipe-Scopes` and `X-SparkPipe-Allow-Swap` (`1` or `0`), and `X-SparkPipe-Priority` from the key when the request sends none. A customer's own copies of these never reach the hub. The source token is passed on and never logged or stored.
- **Bodies** up to 32 MiB (413 above), streamed to the hub as they arrive. A chunked body, which git sends for a large push, is read whole up to 32 MiB and sent with `Content-Length`. A chat completions body loses its `sparkpipe` object; every other body is sent byte for byte.
- **Answers** come back unchanged: status, headers other than hop-by-hop ones, and the body, streamed as the hub sends it, so SSE streams, keep-alives and the hub's own errors (such as 409 `model_not_loaded`) pass through. A hub that does not answer gives 502 `upstream_error`. When the caller disconnects, the connection to the hub closes, so the hub can stop generation.
- **WebSocket.** `GET /v1/responses` with `Upgrade: websocket` is passed to the hub with the same headers; after the hub's 101 the gateway relays bytes both ways. Every `TUNNEL_CHECK` (30) seconds it checks that the account still holds the Fleet and closes the connection when it does not. Apache passes the upgrade with `upgrade=websocket` on the `/pay/api/` proxy (`deployment/vps/sparkpipe-pay.conf`).
- **Errors** of the gateway use the API shape's own body: Anthropic's `{"type": "error", "error": {"type", "message"}}` on `/v1/messages`, and `{"error": {"message", "type", "code"}}` elsewhere, with the codes of `docs/FLEET_API.md` §17 (`invalid_key`, `fleet_required`, `fleet_not_held`, `invalid_request`, `upstream_error`, and `busy` with `Retry-After` when an account sends more than 6,000 requests in ten minutes).
- **Sandbox.** The sandbox simulates `/v1/chat/completions` and `/v1/models` for a holder, and answers 503 `closed` on the other paths.

## Keys

`POST /v1/keys` or `POST /market/keys` creates a key with these options, all optional: `note` (one line, at most 100 characters), `scopes` (a list of `inference`, `state`, `storage`, `functions`, `models`, `account:read`; all of them by default), `fleet` (a Fleet id), `allow_swap` (false by default), `expires` (a UTC time in the future), `ips` (at most 32 addresses or networks, IPv4 or IPv6) and `priority` (`interactive` or `background`, the default for requests that send no `X-SparkPipe-Priority`). A bad option answers 400 `invalid_request`.

- **Checks.** A revoked or expired key, or a rotated one more than an hour after its rotation, answers 401 `invalid_key`. A key used from an address outside its `ips` (the address Apache adds last to `X-Forwarded-For`), or on a path outside its scopes, answers 403 `invalid_key`.
- **Rotation.** `POST /v1/keys/{id}/rotate` issues a new key with the same options and keeps the old one valid for one hour (`retires`). A key is rotated once.
- **Who manages keys.** Listing needs the session or a key with `account:read`; creating, revoking and rotating need the account session.

## Usage records

The hub posts one record per request to `POST /hub/requests` (`{"records": [...]}`, 1 to 10,000 a call, or one record), with the upstream key. A record keeps only these fields, and everything else, prompts and replies included, is dropped: `request`, `account`, `fleet` and `at` (a UTC time such as `2027-01-05T09:00:00Z`, from the last year), which are required, and `key`, `model`, `pack`, `api`, `priority`, `session`, `status`, `prompt_tokens`, `cached_tokens`, `prefill_tokens`, `queue_ms`, `prefill_ms`, `ttft_ms`, `output_tokens`, `duration_ms`, `decode_tps`, `draft_accept` and `cold_prefill_ms_estimate` (the telemetry of `docs/FLEET_API.md` §3.6). A record whose `request` is already stored for that day is a duplicate and is skipped, so the hub can resend. The answer is `accepted`, `duplicates` and `refused` (`index`, `error`).

Records are stored in `requests/<day>.jsonl` in the data directory and deleted 366 days after their day, so each is kept a year. `GET /v1/requests` lists the account's own records in `[since, until)` (default the last 7 days, at most a year apart), optionally by `key`, `fleet`, `model` or `session`, `limit` (1 to 10,000, default 1,000) at a time with `after` set to the last answer's `next`. `totals` sums the whole window, for `sparkpipe stats`: `requests`, the token and time fields, `prefill_avoided_ms` (the sum of `cold_prefill_ms_estimate`), `cached_share` and `decode_tps` (output tokens over decode time).

## Notifications

Every event of `docs/FLEET_API.md` §13 is a `notice` line written when it happens, then delivered to the account's webhooks and email.

- **Where the events come from.** The market sends `booking.confirmed` (every booking, auto-extensions included), `booking.deadline` (above, and early from auto-extend), `fleet.allocated` (the Sunday round), `waitlist.position` (whenever the account's own place changes because of someone else, a lapse or a round; never the other accounts' places) and `downtime.credited` (a maintenance credit). The hub posts `run.finished`, `run.failed`, `report.ready`, `fetch.finished`, `fetch.failed`, `fetch.needs_token`, `pack.warm`, `cache.restored` and `models.changed` to `POST /hub/events` (`{"events": [{"id", "account", "fleet", "type", "data", "summary"}]}`, with the upstream key); an `id` already seen for that Fleet is a duplicate. `working_state.lost` comes with the rebuild credit (below).
- **Webhooks.** `PUT /v1/webhooks {"url": "https://…", "events": [...] or ["*"], "secret", "slack"}`, with the session, adds a webhook or replaces the one with the same URL; at most 10 an account. Without `secret` we make one and show it once. The URL must be `https` with no user name, and must resolve only to public internet addresses other than our own hosts; the address is checked again and pinned at each delivery, and redirects are not followed. `DELETE /v1/webhooks/{id}` removes it with its secret.
- **Delivery.** A `POST` of `{"id", "type", "created", "account", "summary", "data"}`, or `{"text": summary}` when `slack` is true, with `X-SparkPipe-Event`, `X-SparkPipe-Delivery` (the notice id) and `X-SparkPipe-Signature: t=<unix time>,v1=<hex HMAC-SHA256 of "<t>." + body under the secret>`. A receiver checks the HMAC and that `t` is recent. Any answer other than 2xx is retried after 1 minute, 5 minutes, 30 minutes, 2, 6, 12 and 24 hours, and then given up. Each outcome is a line of `deliveries.jsonl`; on a restart, and every five minutes, deliveries of the last 3 days that have no outcome are sent again.
- **Email.** Unless the account sets its own email subscription, `booking.deadline`, `fleet.allocated`, `downtime.credited` and `working_state.lost` go to the account's email. `PUT /v1/webhooks {"email": "<the account's email>", "events": [...]}` chooses the events, and `events: []` turns email off. Email goes only to the account's own address until teams arrive.

## Rebuild credit (clause 6.6)

The hub posts each pair-loss report to `POST /hub/pair-loss` with the upstream key: `fleet`, `event` (the hub's id for the loss), `account`, optional `at` (default now), `pair` (the nodes), `contexts` and `storage` (what was lost, at most 1,000 each) and `run` (the rebuild run it started). If the account held the Fleet at `at`, the service credits at once a `rebuild` lot of 24 × the Rate of the booking that covered `at`, writes a `pair_loss` line, sends `working_state.lost` with what was lost, the credit and the run, and emails `SPARKPIPE_PAY_NOTIFY`. The same `fleet` and `event` again answers the first credit with `duplicate: true` and credits nothing. It answers 409 `not_held` when the account did not hold the Fleet then, 404 for an unknown Fleet and 400 for a bad report. Clause 10.4 does not refund HC credited this way.

## The holder schedule

`GET /market/fleets/<fleet>/holders?from=<slot>&to=<slot>` lets the hub see who holds each hour of a Fleet, so the engine can delete working state when another customer's booking starts or 24 hours after the last held hour (clause 6.2). It answers `{"fleet": "F3", "now": "<stamp>", "hours": [{"start": "<slot>", "account": "AC..."}]}`: every hour in `[from, to)` held by a booked Week or an HC-booked Hour, in order, without the hours of announced maintenance. The hub polls it every 30 seconds for the last 25 hours through the next 25; it reads only the bookings that overlap the window.

It is authenticated with `Authorization: Bearer` and the upstream key (`SPARKPIPE_POOL_KEY_FILE`), compared in constant time: the hub already holds that key to call the Fleets, and no other service credential exists. It answers 401 for a missing or wrong key (and whenever no upstream key is configured), 404 for an unknown Fleet, and 400 for a missing or malformed `from` or `to`, `to` not after `from`, or a window over 7 days. It is not rate limited.

## availability.txt

`https://sparkpipe.ai/availability.txt` (clause 4.2) had no server. The live service generates it on every request, and `deployment/vps/sparkpipe-pay.conf` maps the public path to `http://127.0.0.1:18090/availability.txt`:

```text
# SparkPipe availability, generated 2027-01-06T03:00:00Z, Customer Agreement clause 4.2
# fleet serving status (demo, pinned or available)
F1 serving demo
F3 serving pinned
waitlist locks 2
```

The last line is the number of accounts that hold a waitlist lock, the same number as `waitlist.locks` in the public state.

## Sandbox

`/pay/sandbox/api` runs the same service with `SPARKPIPE_PAY_SANDBOX=1`: every Fleet is serving, `demoFrom` and `weeksFrom` do not apply, `/market/buy` issues HC at once, `/market/sandbox/clock` moves the market clock forward by whole hours (rounds, lapses and maintenance credits catch up), `/market/sandbox/reset` starts an empty ledger, chat replies are simulated, Toss Payments runs in test mode, no mail is sent and the sign-in code is returned in the answer.

## Removed

- **The Hour Market:** releases, asks, bids, matching, the release fee and `market.txt`.
- **The shared pool:** per-seat billing, seat prices, settlement of released hours, model-switch deferral, `pool.txt` and `/pool/v1`.
- **Plans:** Plan Orders, quotes, Monthly and term plans, Tranches, gates, office hours and time zones, `record_signed`, and the `plan`, `end-plan` and `tranche` commands.
- **The HC cover rule** and the `cover` command.
- **The quote-request form** (`/lead`).
