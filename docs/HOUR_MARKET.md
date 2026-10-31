# Hour Market

The Hour Market sells hour-credits (HC) and lets customers trade dated
fleet-hours with HYOR as the only counterparty. HC is the only hourly
product: `buy.html` sells it by amount, `market.html` is the market, and the
Hourly Terms and the Hour Market Rules (`site/terms/hourly-*.md`,
`site/terms/market-rules-*.md`) govern it. The design, the market view and
the contract changes are in the shared Hour Market document; this file
describes what is built, how it runs and how to test it.

## Units and rules

- **HC.** One HC buys one hour of a Rate-1 fleet. It is bought by amount
  (`market.hcPrice` in `site/pricing.js`, $20, plus VAT), from
  `market.minPurchase` ($10), to 6 decimal places. Each purchase or credit is
  a lot that lapses 12 months after it is issued, or later when the cover
  below needs it; spending takes the lot that lapses first. HC never moves
  between accounts and is never paid out.
- **Dated hour.** A reservation of one fleet for one hour, starting on the
  hour (UTC slots such as `2026-12-01T01:00Z`; the page shows KST). It is
  consumed when it occurs.
- **Our capacity.** Every unreserved fleet-hour within `market.horizonDays`
  (7) is for sale at the fleet's Rate, and from Monday to Saturday up to
  `market.advanceDays` (31) ahead at Rate × `market.advanceMultiplier`
  (1.25): the premium buys lead time. Sunday hours open only within 7 days,
  after maintenance is announced. Live, only fleets whose `state` is
  `serving` sell, and not before `hourlyEarliestStart`. Capacity kept for
  two-year customers and for earlier-lapsing HC (below) is not offered.
- **Release.** The holder of a reservation that has not started may release
  it at list (Rate) or a lower minimum, and cancel until it sells. A slot has
  at most one ask: ours, or the holder's release.
- **Bids.** A maximum price per hour for a window, with an optional fleet,
  a model, a number of hours and a consecutive flag. The bid holds
  `max × hours` HC until it fills, is cancelled or its window passes. A
  standing bid's window (at most 7 days) moves a week forward when it ends,
  and holds again; it pauses if the account cannot cover it. Reserving from
  the calendar is a one-off purchase of one slot.
- **Matching.** Cheapest first; at equal prices released hours fill before
  ours, earliest release first; older bids before newer. A trade against our
  capacity is at Rate; a trade with a release is at whichever order was
  resting first (the ledger sequence decides).
- **Settlement.** The buyer's HC is debited and the seller is credited the
  price less `market.releaseFee` (10%) as a new 12-month lot.
- **Shared pool.** In each hour, every serving fleet that nobody holds, and
  every released hour that reached its start unsold, serves per-request
  traffic. A request holds one seat; the price per seat-second is
  `1 / (seats × 3600)` HC times a congestion multiplier between
  `minMultiplier` and `maxMultiplier`, recomputed every minute from the last
  minute's load. Charges go to released hours first, earliest release first,
  up to a fleet's seat-seconds; after the hour each released hour's seller is
  credited what it earned less the fee. An account that holds the current
  hour on a fleet uses its keys there free.
- **Tranche hours.** A two-year customer's monthly hours reserve any free
  hour of their month, beyond the 7-day horizon, at no HC; released, they
  earn HC like any hour. Live, grant them with
  `sparkpipe_pay.py tranche EMAIL YYYY-MM HOURS`.

## No overselling, no bumping

Airlines sell more seats than a flight has and bump passengers when
everyone turns up. The market does neither: no reservation is ever taken
from its holder, and HC is sold only while it can be spent. Four rules in
`Cover` (`tools/sparkpipe_market.py`) carry this:

- **Set-aside instead of bumping.** Every active two-year agreement in
  `agreements.json` keeps capacity off sale for every Monday-to-Saturday
  hour of its term: one fleet for a Thirds election (8 hours in each of the
  Americas 00-08, Asia 08-16 and Europe 16-24 KST windows, `windows` in
  `site/pricing.js`), three fleets in the elected window for a 100%
  election (one fleet's worth of hours run at 3x during the customer's own
  working day; three regions tile three fleets with nothing to swap), plus
  one fleet in the other hours until Full Prepayment, when Thirds still
  applies. The holder's own Tranche reservations use up its set-aside.
  Capacity still unreserved `market.lastCallHours` (6) before an hour
  starts opens to HC, the point from which the Agreement's guarantee no
  longer applies. The two-year guarantee therefore never needs to take an
  hour anyone holds, and the Terms carry no displacement right.
- **Cover.** Free Monday-to-Saturday hours of serving fleets, after the
  set-aside, each at its fleet's Rate, must cover all unspent HC (lots and
  bid holds) that lapses by any date. Sundays and fleets still being built
  are not counted. A purchase is refused beyond the HC on sale: the cover
  12 months out minus all unspent HC and the HC in unpaid orders (reason
  `cover`). `onSale` in `/market/state`, the page and `market.txt` publish
  it. Sales and pool credits for released hours never weaken the cover, as
  the buyer's HC spent is more than the seller's credit and the hour was
  already taken.
- **Later lapse instead of overselling.** A payment that completes after
  others have bought the cover, and a credit we owe (`credit`, below), lapses
  on the first date the cover reaches it, never before 12 months.
- **Earlier lapse first.** When the free hours before a date only just cover
  the HC lapsing by then, a reservation or bid fill of our hour before that
  date that would leave that HC short is refused (reason `kept`) unless the
  HC it spends lapses by then. The protection ends `lastCallHours` before
  the hour starts, so nothing stays idle for HC whose holders never come.

Unspent HC can still lapse if its holder leaves free hours unused: that is
the HC's 12-month life, not an oversale.

Operations:

- `sparkpipe_pay.py cover [AGREEMENTS.json]` prints the serving fleets, the
  free hours in the next 12 months after set-asides, unspent HC, HC on sale,
  the tightest lapse date and the hours where a set-aside cannot be kept.
  It exits 1 when a set-aside cannot be kept or HC is left without cover.
  Before signing a two-year agreement, or approving a 100% election, run
  it with a copy of `agreements.json` that includes the new entry; sign only
  when it exits 0, or move the start date or wait for a fleet.
- `sparkpipe_pay.py credit EMAIL HC NOTE` issues HC we owe, such as an
  outage credit under clause 6 of the Hourly Terms, with the lapse date the
  cover allows.

## Code

`tools/sparkpipe_market.py` holds the engine: `State` rebuilds the market
from the ledger and `Market` validates actions, matches and settles.
`tools/sparkpipe_pay.py` imports it, shares its hash-chained ledger and
serves the routes. A background tick every 30 seconds, and every request,
moves released hours into the pool at their start, settles finished pool
hours, rolls standing bids and matches resting bids against capacity that
has come into the horizon.

Ledger lines it writes: `hc_lot`, `hc_debit`, `hc_hold`, `hc_unhold`,
`hc_spend_hold`, `reserve`, `ask`, `ask_cancel`, `bid`, `bid_cancel`,
`bid_close`, `bid_roll`, `pool_join`, `pool_settle`, `pool_charge`,
`api_key`, `api_key_revoke`, `tranche_grant` and, in the sandbox, `clock`.
Time-dependent lines carry the market time they used, so a replay gives the
same balances.

| Route | Who | What |
| --- | --- | --- |
| `GET /market/state` | anyone | Market time, settings, fleets, pool prices, HC on sale; the account's balance, lots, reservations, bids, Tranche hours, keys and charges when signed in |
| `GET /market/calendar?days=N` | anyone | Best ask per fleet-hour for up to 31 days |
| `GET /market/book` | anyone | Open releases and bids, without identities, and recent trades |
| `GET /market/market.txt`, `/market/pool.txt` | anyone | The same as plain-text feeds |
| `GET /market/ledger` | signed in | The account's own ledger lines with their hashes |
| `POST /market/buy` | signed in | `amount` in USD before VAT; live also `coin`, and returns the CoinPayments checkout |
| `POST /market/reserve` | signed in | `fleet`, `start`, `model`, optional `max`, `pay`: `hc` or `tranche` |
| `POST /market/release` | signed in | `reservation`, optional `min` |
| `POST /market/cancel` | signed in | `ask` or `bid` |
| `POST /market/bid` | signed in | `fleet` (or none), `model`, `from`, `to`, `hours`, `max`, `consecutive`, `standing` |
| `POST /market/keys`, `/market/keys/revoke` | signed in | Create a pool key (shown once) or revoke one |
| `POST /pool/v1/chat/completions`, `GET /pool/v1/models` | pool key | OpenAI format; replies carry `sparkpipe: {hc, seconds, dedicated}` |
| `POST /market/sandbox/clock`, `/tranche`, `/reset` | sandbox, signed in | Move the shared clock, take demo Tranche hours, clear the sandbox |

## The page, live and sandbox

`https://sparkpipe.ai/market.html` (`site/market.html`, `market.js`,
`market.css`) is the market against the live service at `/pay/api/`: real
accounts signed in by an emailed code, HC bought through `buy.html` and
CoinPayments, the live pool. `market.html?sandbox` is the same page against
a second instance of the service in sandbox mode, served at
`/pay/sandbox/api/`, with its own accounts and ledger; `market-test.html`
redirects there. In the sandbox:

- purchases settle at once and no CoinPayments call is made;
- `/code` returns the sign-in code instead of mailing it, and no mail is sent;
- a shared test clock moves forward, so hours start, released hours reach
  the pool and the pool settles;
- the pool answers with a built-in simulator, so the free sandbox never
  reaches the fleets;
- `/market/sandbox/reset` moves the ledger aside and starts empty.

Accepting the Hourly Terms records the SHA-256 of both languages of the
Hourly Terms and of the Hour Market Rules (`TERMS` in
`tools/sparkpipe_pay.py`); a change to any of the four texts asks the
account to accept again at its next order.

First install on the VPS, beside the live service:

```sh
sudo install -d -o sparkpipe-pay -g sparkpipe-pay -m 700 /var/lib/sparkpipe-pay-sandbox
sudo install -m 640 -g sparkpipe-pay deployment/vps/sparkpipe-pay-sandbox.env.example /etc/sparkpipe-pay/sandbox.env
sudo install -m 644 deployment/vps/sparkpipe-pay-sandbox.service /etc/systemd/system/
sudo install -m 644 deployment/vps/sparkpipe-pay.conf /etc/apache2/sparkpipe/
sudo systemctl daemon-reload
sudo systemctl enable --now sparkpipe-pay-sandbox
sudo apachectl configtest && sudo systemctl reload apache2
```

Later deploys follow docs/PAYMENTS.md and restart both services.

## Live pool

The live pool forwards to the fleets through the public API door on the
VPS (`127.0.0.1:18081`, docs/LITELLM_FRONTEND.md), marked public, with one
service key from the hub's `api_keys.json`. Put that key in
`/etc/sparkpipe-pay/pool_key` (mode 640, group `sparkpipe-pay`) and add to
`/etc/sparkpipe-pay/env`:

```sh
SPARKPIPE_POOL_UPSTREAM=http://127.0.0.1:18081/v1
SPARKPIPE_POOL_KEY_FILE=/etc/sparkpipe-pay/pool_key
```

Until both are set, the live pool answers 503 `pool_closed`. The key is
`k_dbfba343` in the hub's `~/sparkpipe-chat/api_keys.json` (note "sparkpipe.ai
Hour Market live pool", created 2026-10-31, expires 2036-10-28); revoke it
there with `api_keys.py revoke k_dbfba343`. The door allows four concurrent
requests per key, which caps the live pool at four requests at once until
the hub gives the pool its own limit. A public request for a model the
fleet has not loaded starts a swap, as any public request does, so a pool
request for the other pool model moves the whole fleet.


Dedicated hours are recorded, billed and released here; keeping a fleet
exclusive to its holder during the hour is work for the fleet controller.
The set-aside keeps one fleet per agreement-hour (three in a 100% window)
whatever fleet the holder later picks; a Thirds holder using several fleets
at once after Full Prepayment does so subject to availability.
