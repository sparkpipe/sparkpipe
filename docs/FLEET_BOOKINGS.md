# Fleet bookings: hour-credits, Hours, Weeks and the waitlist

This is the design of the payment and booking service for the elite-developer offer. The SparkPipe Customer Agreement (`site/terms/agreement-*.md`) governs it, and this document follows its clause numbers; where they differ, the Agreement wins. The published settings live in `site/pricing.js`. The code is `tools/sparkpipe_market.py` (bookings, the waitlist, the gateway's checks and maintenance) and `tools/sparkpipe_pay.py` (accounts, orders, payment processors, the ledger and the HTTP service). Payments are described in `docs/PAYMENTS.md`.

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
- **Maintenance** (clause 4.3): `sparkpipe_pay.py maintenance FLEET MONDAY [HOURS]` schedules up to `bookings.maintenanceHoursPerWeek` hours per Fleet per Week, starting at least 7 days ahead, on the consecutive hours with the fewest booked hours, then the KST start hour least used by that Fleet's earlier maintenance (so the times rotate), then the earliest. Holders it overlaps are emailed at once. When it ends, every booked hour it overlaps is credited as a lot of HC equal to the booking's Rate (clause 4.4) and a `maintenance_done` line is written.

## Events and their order

Time-driven events run whenever the service touches the market (every request and a 30-second tick), in time order, and at the same instant lapses first, then the round, then maintenance credits:

1. a lot that lapses while some of it is locked;
2. the Sunday round, for each Sunday 00:00 KST after the later of the last recorded round and the earliest join on the waitlist (there is no round while the waitlist is empty);
3. the end of a maintenance period.

Each event is written with the instant it happened (`now`), so a service that was down, or a sandbox whose clock jumped, catches up with the same result.

## Ledger lines

The hash-chained ledger (`docs/PAYMENTS.md`) holds these market lines, each with `now` where it matters:

| Type | Data |
| --- | --- |
| `hc_lot` | `lot`, `account`, `hc`, `expires`, `source` (`purchase`, `credit`, `maintenance`), `ref` (the order, credit or maintenance), optional `note` |
| `hc_debit` | `account`, `hc`, `reason`, `ref`, optional `lot` (a refund's debit) |
| `booking` | `booking`, `account`, `fleet`, `kind` (`hour` or `week`), `start`, `end`, `hc`, `rate`, `hours` or `weeks` and `discount`, `source` (`hc`, or `waitlist` with `unlocked`) |
| `lock`, `unlock` | `account`, `hc`; `unlock` gives `reason` (`leave` or `below`) |
| `lock_lapse` | `account`, `lot`, `hc`, `ref`, `refund` (`order`, `processor`, `currency`, `amount`, or null) |
| `round` | `sunday` |
| `api_key`, `api_key_revoke` | API keys (only the SHA-256 of a key is kept) |
| `maintenance`, `maintenance_done` | `maintenance`, `fleet`, `start`, `hours`; `credited` |
| `clock` | the sandbox clock offset |

The live ledger's lines from before this design (`account`, `acceptance`, an `hourly` order and its invoice states) replay unchanged; that account is asked to accept the new Agreement.

## API

Everything is under `/pay/api`, and the sandbox is under `/pay/sandbox/api`. Times are UTC strings: `stamp` (`2027-01-06T03:00:00Z`) for instants and `slot_text` (`2027-01-10T15:00Z`) for hours. HC amounts are strings with six decimals. Refusals are `{"error", "reason"}` with the HTTP status; the reasons a page acts on are `terms` (409: accept the current Agreement and send `accept: true`), `insufficient_hc` (402), `pinned`, `taken`, `too_far`, `not_open`, `started`, `demo_cap`, `not_weekly`, `maintenance`, `lock_too_small` (409).

| Method and path | Auth | Request | Answer |
| --- | --- | --- | --- |
| `GET /market/state` | optional session | | `now`, `sandbox`, `vatRate`, `demoFrom`, `weeksFrom`, `hc` (`prices`, `min`, `validityMonths`), `bookings` (the settings), `models`, `waitlist` (`locks`: the number of accounts that hold a lock), and `fleets`: each Fleet List entry with `serving`, `status` (`demo`, `pinned` or `free` now), `firstWeek` (the next Monday as a slot when a new customer can book it as a first Week now, else null), `booked` and `maintenance` (merged `[start, end]` spans from now on, with no identities). With a session, `account`: `email`, `company`, `termsCurrent`, `balance` (unlocked HC), `locked`, `position`, `joined`, `lots`, `bookings` (not ended more than a Week ago, each with `state` `upcoming`, `live` or `ended`), `pinned` (`fleet`, `weeks` not ended, `through`, `deadline`, `canExtend`), `refunds` (locked HC that lapsed), `keys` |
| `POST /market/quote-weeks` | session | `fleet`, `weeks` | `fleet`, `weeks`, `first`, `last` (start of the last Week), `end`, `count` (N), `discount`, `price`, `extends` |
| `POST /market/book-hours` | session, current terms | `fleet`, `start`, `hours` | the booking: `booking`, `fleet`, `kind`, `start`, `end`, `hours`, `hc`, `source`, `created`, `state` |
| `POST /market/book-weeks` | session, current terms | `fleet`, `weeks` | the quote with `booking` |
| `POST /market/waitlist` | session, current terms | `lock` | `locked`, `position` |
| `POST /market/waitlist/leave` | session | | `locked` (`0.000000`), `position` (null) |
| `POST /market/keys`, `POST /market/keys/revoke` | session | `note`; `id` | `id`, `key` (shown once); `id`, `revoked` |
| `POST /order` | session | `product: "credits"`, `hc` (from `hc.min`, to six decimals), `processor` (`toss` with optional `brn`, `coinpayments` with `coin`, or `swift` with `declare: true`), `accept` when the terms changed | `docs/PAYMENTS.md` |
| `POST /market/buy` | session | `hc` | sandbox: issues the HC at once (`lot`, `hc`, `expires`); with `processor: "toss"`, or on the live service, the same as `/order` |
| `GET /market/ledger` | session | | the account's own ledger lines with their hashes (clause 11.7) |
| `GET /v1/models`, `POST /v1/chat/completions` | API key | OpenAI-compatible; optional `sparkpipe` object whose only field is `fleet` (400 `invalid_sparkpipe` for any other) | the upstream answer with `sparkpipe` (`fleet`) added |
| `GET /market/fleets/<fleet>/holders?from=<slot>&to=<slot>` | upstream key | | `fleet`, `now`, `hours`: `[{start, account}]` |
| `GET /availability.txt` | none | | text, below |
| `POST /market/sandbox/clock`, `POST /market/sandbox/reset` | session, sandbox only | `hours` | `now`; `reset` |

## The gateway

`/v1/chat/completions` serves only an account that holds a Fleet now: a booking (Hour or Week) of a serving Fleet covers this instant. Otherwise it answers 403 `no_fleet`. With more than one Fleet held, the request names one in `sparkpipe.fleet` (400 `fleet_required` without it, 403 `fleet_not_held` for a Fleet it does not hold). The model must be listed (404 `model_not_found`). The service forwards the body without `sparkpipe` to `SPARKPIPE_POOL_UPSTREAM/chat/completions` with `Authorization: Bearer` the upstream key, `X-SparkPipe-Public: 1`, `X-SparkPipe-Fleet: <fleet id>` and `X-SparkPipe-Account: <account id>`. The upstream request is built from scratch, so no header the customer sent, an `X-SparkPipe-Account` or `X-SparkPipe-Fleet` of its own included, reaches the Fleet. Streaming answers pass through. `/v1/models` needs a held Fleet too. In the sandbox the reply is simulated.

## The holder schedule

`GET /market/fleets/<fleet>/holders?from=<slot>&to=<slot>` lets the hub see who holds each hour of a Fleet, so the engine can delete working state when another customer's booking starts or 24 hours after the last held hour (clause 6.2). It answers `{"fleet": "F3", "now": "<stamp>", "hours": [{"start": "<slot>", "account": "AC..."}]}`: every hour in `[from, to)` held by a booked Week or an HC-booked Hour, in order, without the hours of announced maintenance. The hub polls it every 30 seconds for the last 25 hours through the next 25; it reads only the bookings that overlap the window.

It is authenticated with `Authorization: Bearer` and the upstream key (`SPARKPIPE_POOL_KEY_FILE`), compared in constant time: the hub already holds that key to call the Fleets, and no other service credential exists. It answers 401 for a missing or wrong key (and whenever no upstream key is configured), 404 for an unknown Fleet, and 400 for a missing or malformed `from` or `to`, `to` not after `from`, or a window over 7 days. It is not rate limited.

## availability.txt

`https://sparkpipe.ai/availability.txt` (clause 4.2) had no server. The live service generates it on every request, and `deployment/vps/sparkpipe-pay.conf` maps the public path to `http://127.0.0.1:18090/availability.txt`:

```text
# SparkPipe availability, generated 2027-01-06T03:00:00Z, Customer Agreement clause 4.2
# fleet serving status (demo, pinned or free)
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
