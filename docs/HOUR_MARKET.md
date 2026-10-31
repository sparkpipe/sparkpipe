# Hour Market

The Hour Market sells hour-credits (HC) and lets customers trade dated
fleet-hours with HYOR as the only counterparty. The design, the market view
and the contract changes are in the shared Hour Market document; this file
describes what is built, how it runs and how to test it.

## Units and rules

- **HC.** One HC buys one hour of a Rate-1 fleet. It is bought by amount
  (`market.hcPrice` in `site/pricing.js`, $20, plus VAT), from
  `market.minPurchase` ($10), to 6 decimal places. Each purchase or credit is
  a lot that lapses 12 months after it is issued; spending takes the lot
  that lapses first. HC never moves between accounts and is never paid out.
- **Dated hour.** A reservation of one fleet for one hour, starting on the
  hour (UTC slots such as `2026-12-01T01:00Z`; the page shows KST). It is
  consumed when it occurs.
- **Our capacity.** Every unreserved fleet-hour within `market.horizonDays`
  (7) is for sale at the fleet's Rate. Live, only fleets whose `state` is
  `serving` sell, and not before `hourlyEarliestStart`.
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
| `GET /market/state` | anyone | Market time, settings, fleets, pool prices; the account's balance, lots, reservations, bids, Tranche hours, keys and charges when signed in |
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

## Sandbox and test page

`https://sparkpipe.ai/market-test.html` (not linked, `noindex`) drives every
route against a second instance of the service in sandbox mode, served at
`/pay/sandbox/api/`:

- purchases settle at once and no CoinPayments call is made;
- `/code` returns the sign-in code instead of mailing it, and no mail is sent;
- a shared test clock moves forward, so hours start, released hours reach
  the pool and the pool settles;
- the pool answers with a built-in simulator, so the free sandbox never
  reaches the fleets;
- `/market/sandbox/reset` moves the ledger aside and starts empty.

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

Until both are set, the live pool answers 503 `pool_closed`. The door
allows four concurrent requests per key, which caps the live pool at four
requests at once until the hub gives the pool its own limit.

Dedicated hours are recorded, billed and released here; keeping a fleet
exclusive to its holder during the hour is work for the fleet controller.
