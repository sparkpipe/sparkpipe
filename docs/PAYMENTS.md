# Payments

Fleet-hours are paid in USDC or USDT through CoinPayments: USDC on Base,
Solana, Arbitrum, Polygon, BNB Chain or Ethereum, USDT on Tron, Solana, BNB
Chain, Arbitrum, Polygon or Ethereum. Only these two US-dollar stablecoins
are taken, so an invoice priced in dollars is paid in dollars and neither
side carries crypto price risk. Won bank transfers against a tax
invoice and USD wires under a two-year agreement are still arranged by hand.

## How a customer buys

1. `https://sparkpipe.ai/buy.html` asks three things, one step at a time:
   how many hours (4, 24, 72, 168 or any number) and from which day; the
   company's legal name and a work email, with a box accepting the Hourly
   Terms; and the coin and network the customer pays with. The order total, with
   VAT, stays in view.
2. `POST /pay/api/code` emails a 6-digit code. `POST /pay/api/verify`
   checks it, opens the account if the email is new, records the
   acceptance of the Hourly Terms and sets the session cookie. The email is
   the account: there is no password and no wallet.
3. `POST /pay/api/order` prices the order from `site/pricing.js` (never
   from the browser), checks the holding limit, records the order and
   creates a CoinPayments invoice in USD (currency `5057`) with the price,
   VAT at 10% as tax, a per-invoice webhook, and `payment.paymentCurrency`
   set to the chosen coin on the chosen network, so the checkout opens
   straight on the address, amount and QR code. The page sends the customer there.
4. CoinPayments posts signed webhooks to `/pay/api/coinpayments`. The
   service checks the HMAC and the timestamp, then fetches the invoice from
   the CoinPayments API and acts on that answer, not on the webhook body.
   Hours are issued only when the invoice is `completed` and its total and
   currency match the order: all the order's hours when the amount received
   covers the total, or, when CoinPayments completes it for less, hours in
   proportion to the amount received, to four decimals (Hourly Terms 3.5,
   agreement clause 5.5). The customer gets a receipt by email, and
   `SPARKPIPE_PAY_NOTIFY` gets a note of each new account, payment and
   mismatch.
5. CoinPayments sends the customer back to `account.html?order=…`, which
   follows the order (awaiting payment, payment seen, confirmed, hours
   issued) every 6 seconds and then shows the account: hours available,
   room under the holding limit, hour blocks, two-year agreements and
   orders.

Every five minutes the service also asks CoinPayments about each open
invoice from the last 14 days, so a lost webhook only delays the hours.

A two-year customer signs its agreement by hand. Once its entry is in
`agreements.json`, the account page lists the agreement for the account
with the same email, with what is paid, what is left and the next gate,
and takes prepayments in USDC or USDT. `two-year.html` plans a term and sends a
quote request to `POST /pay/api/lead`.

## Endpoints

| Route | Does |
| --- | --- |
| `POST /pay/api/code` | `email`: emails a 6-digit sign-in code, valid 10 minutes for 5 tries |
| `POST /pay/api/verify` | `email`, `code`, and for a new account `company` and `accept: true`: signs in, sets the cookie `sp_session` and returns the account |
| `POST /pay/api/signout` | Clears the cookie |
| `GET /pay/api/account` | The signed-in account: hours, limit, blocks, orders, agreements and whether it accepted the current Hourly Terms |
| `POST /pay/api/order` | Signed in. `product: "hourly"` with `hours`, `start`, `coin` (and `accept: true` when the terms changed since the account accepted them); `product: "rest"` with `parent` (a part-paid hourly order) and `coin`, for what is left of it at the same price; or `product: "two-year"` with `agreement`, `amount` before VAT and `coin`: returns the checkout link. `coin` is one of the twelve CoinPayments symbols in `COINS`, such as `USDC.BASE` or `USDT.TRC20` |
| `GET /pay/api/order/<order>` | The order's state, amounts and hours; the checkout link while unpaid. The order number is 80 random bits |
| `POST /pay/api/lead` | A two-year quote request: `company`, `email`, `name`, `fleets`, `start`, `message` |
| `POST /pay/api/coinpayments` | CoinPayments webhooks |
| `GET /pay/api/health` | `{"ok": true}` |

The session cookie is HttpOnly, SameSite=Lax, Secure and scoped to
`/pay/api`. It holds the account number, its expiry and an HMAC under
`session_key` in the data directory, made on first start. A POST whose
`Origin` is not `SPARKPIPE_PAY_PUBLIC` is refused. Each client IP address
may ask for 20 codes, 30 verifications, 10 orders and 5 quote requests in
ten minutes (quote requests: an hour), and each email for 5 codes an hour.

## Rules the service enforces

- An account opens only with the box ticked; the ledger keeps the
  statement accepted, the SHA-256 hashes of `terms/hourly-ko.md` and
  `terms/hourly-en.md`, the IP address and the browser (clause 12.4). When
  those files change, the next hourly order asks the customer to accept
  the new text.
- Hourly hours start no earlier than `hourlyEarliestStart` and within a
  year, and last 12 months (clause 4.2).
- The holding limit is 168 hours, or 720 for an email listed in
  `approved.json` (clause 4.1). It counts unexpired issued hours and orders
  still awaiting payment for 24 hours. Until reservations record
  consumption, every unexpired hour issued counts as held.
- A two-year prepayment is at least 1.00 USD and at most what is left of
  the term's price, counting unpaid orders. It is refused after Phase A
  with nothing paid (clause 5.2), after a missed gate (clause 6.4), and
  after the term ends. Each payment issues the amount before VAT divided by
  the hour price, to four decimals (clause 5.5).
- VAT is charged on every USDC or USDT payment (Hourly Terms 3.2, Schedule (c)).
  CoinPayments' own fee is ours; the customer pays the network fee.

## Data on the VPS

`/var/lib/sparkpipe-pay`, owned by the service user:

- `ledger.jsonl`: one JSON line per event (`account`, `acceptance`,
  `order`, `invoice`, `invoice_failed`, `invoice_state`, `payment`,
  `hours_issued`), each with the SHA-256 of the line before it. The service
  refuses to start if the chain is broken. An account or order line keeps
  the IP address and the browser, as the privacy policy's acceptance
  records say; a payment line keeps the coin CoinPayments reports.
- `session_key`: 32 random bytes that sign the session cookies. Deleting
  it signs everyone out.
- `leads.jsonl`: the two-year quote requests.
- `approved.json`: the emails that passed the business check, for example
  `["cto@acme.example"]`.
- `agreements.json`: one entry per signed two-year agreement:

```json
[{"id": "SP2Y-7K3QD9", "email": "billing@acme.example", "company": "Acme Co., Ltd.", "signed": "2026-12-15", "hourPrice": "15.00"}]
```

The service rereads `approved.json` and `agreements.json` on every
request, so edits need no restart.

## Mail

The service sends the sign-in codes, receipts and notes through
`SPARKPIPE_PAY_SMTP` (default `127.0.0.1:25`, the VPS's own mail server,
which the SPF record for sparkpipe.ai already names) as
`SPARKPIPE_PAY_MAIL_FROM` with replies to `SPARKPIPE_PAY_NOTIFY`. A code
that cannot be sent is an error on the page; a receipt that cannot be sent
is logged.

The VPS's rspamd signs that mail with DKIM for sparkpipe.ai, selector
`s202608` (`deployment/vps/rspamd-dkim_signing.conf`, installed as
`/etc/rspamd/local.d/dkim_signing.conf`; key
`/var/lib/rspamd/dkim/sparkpipe.ai.s202608.key`), and for hyor.io with
selector `s202610`. Its public key is the
TXT record `s202608._domainkey.sparkpipe.ai` at the domain's DNS host
(GoDaddy), and rspamd signs only while the two match (`check_pubkey`).
`sign_local = true` signs mail submitted from the VPS itself, as the
payment service's is; mailbox users are signed when they authenticate.
With SPF naming the VPS and DMARC at `p=quarantine`, receivers see SPF,
DKIM and DMARC pass.

## CoinPayments setup

1. In the CoinPayments dashboard, create an API integration and keep its
   client ID and client secret. Webhooks need no dashboard setting: every
   invoice names `https://sparkpipe.ai/pay/api/coinpayments` and the events
   `invoicePending`, `invoicePaid`, `invoiceCompleted`, `invoiceCancelled`
   and `invoiceTimedOut`. The webhook signature covers that exact URL, so
   it must not change behind the proxy.
2. Every invoice pre-selects the customer's coin and network, but the
   checkout still offers a currency menu. In Payment Settings, accept only
   `USDC.ERC20`, `USDC.BASE`, `USDC.ARB`, `USDC.POL`, `USDC.SOL`,
   `USDC.BEP20`, `USDT.TRC20`, `USDT.ERC20`, `USDT.BEP20`, `USDT.SOL`,
   `USDT.POL` and `USDT.ARB`, and turn every other coin and LTCT off, so no
   payment arrives in a coin whose price moves. Set the underpayment limit
   to a fixed $100,000 with no percentage (see Operating it). Payment Settings lists
   every currency; open each one that is not in that list and disable it.
   A disabled currency is no longer offered at checkout.
3. Choose the payout wallet or conversion there too; the service does not
   set payout overrides.

CoinPayments emails the payment instructions to the invoice's refund
email, which is the account email.

## Install on the VPS

```sh
sudo useradd --system --home /var/lib/sparkpipe-pay --shell /usr/sbin/nologin sparkpipe-pay
sudo install -d -o sparkpipe-pay -g sparkpipe-pay -m 700 /var/lib/sparkpipe-pay
sudo install -d -m 755 /opt/sparkpipe-pay /etc/sparkpipe-pay
sudo install -m 644 tools/sparkpipe_pay.py /opt/sparkpipe-pay/
sudo install -m 640 -g sparkpipe-pay deployment/vps/sparkpipe-pay.env.example /etc/sparkpipe-pay/env
sudo install -m 640 -g sparkpipe-pay /dev/null /etc/sparkpipe-pay/client_secret
sudo install -m 644 deployment/vps/sparkpipe-pay.service /etc/systemd/system/
sudo install -m 644 deployment/vps/sparkpipe-pay.conf /etc/apache2/sparkpipe/
sudo install -m 644 deployment/vps/sparkpipe-ai-ssl.conf /etc/apache2/sites-available/000-default-le-ssl.conf
sudo install -m 644 deployment/vps/rspamd-dkim_signing.conf /etc/rspamd/local.d/dkim_signing.conf
```

Fill `COINPAYMENTS_CLIENT_ID` in `/etc/sparkpipe-pay/env` and put the
client secret into `/etc/sparkpipe-pay/client_secret` by copying a file,
never on a command line. `sparkpipe-ai-ssl.conf` is sparkpipe.ai's port-443
virtual host and includes `sparkpipe-pay.conf` beside the other
`deployment/vps` includes; `mod_proxy_http` must be on. Copy `site/` to `/var/www/html` as usual, including
`site/terms/`: the acceptance names `https://sparkpipe.ai/terms/hourly-ko.md`
and `hourly-en.md`, and the service hashes the copies under
`/var/www/html/terms`. Then:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now sparkpipe-pay
sudo apachectl configtest && sudo systemctl reload apache2
sudo rspamadm configtest && sudo systemctl reload rspamd
curl -s https://sparkpipe.ai/pay/api/health
```

The service needs Python 3.8 or later and nothing outside its standard
library.

## Deploying a change

Only what is merged to `main` goes to the VPS: commit to a PR, run the
tests, merge, then deploy `main`. Pages, the service and the host
configuration in `deployment/vps` all go the same way; nothing is edited
on the host alone.

```sh
git fetch origin main
git archive --format=tar.gz -o pay-main.tgz origin/main site tools/sparkpipe_pay.py deployment/vps
```

On the VPS, back up `/var/www/html`, `/opt/sparkpipe-pay/sparkpipe_pay.py`
and any configuration file being replaced, copy `site/.` over
`/var/www/html`, install the service and changed configuration as in the
install steps, check the configuration, then restart `sparkpipe-pay` and
reload Apache or rspamd. The live pages, the service and the configuration
must then match `origin/main` byte for byte.

## Operating it

- `journalctl -u sparkpipe-pay` logs each account, invoice, payment and
  issue of hours, every refused webhook signature and every mail that
  failed, without payload contents.
- `sparkpipe_pay.py verify` checks the ledger chain and prints its head and
  counts. Run the admin commands as the service user with its environment:
  `sudo -u sparkpipe-pay sh -c 'set -a; . /etc/sparkpipe-pay/env; python3 /opt/sparkpipe-pay/sparkpipe_pay.py verify'`.
- Part payments settle without anyone. CoinPayments completes an
  underpaid invoice when the shortfall is within the underpayment limit in
  Payment Settings: a percentage of the invoice (at most 50%), a fixed
  amount, or both, in which case the smaller wins. Set only a fixed
  $100,000, with no percentage, so every part payment of an invoice under
  $100,000 completes with what arrived. The service then issues hours in
  proportion to what arrived, mails the customer a receipt that says so and
  links to paying the rest, and mails `SPARKPIPE_PAY_NOTIFY` "Paid in part".
  The order keeps what is left (`left`, `leftHours`): the account page
  offers "Pay the remaining" for it, which places a `rest` order at the
  same price, dates and coin, as often as the customer likes, until the
  hours issued across the orders equal the hours ordered. A part payment
  CoinPayments does not complete stays `pending` until the payment window
  closes, when CoinPayments refunds it and emails the customer a claim
  link. Overpayments and late payments are refunded the same way: once a
  part payment completes its invoice, money sent again to that address is
  an overpayment, which is why the rest goes through a new order.
- A completed invoice whose total or currency does not match the order
  (state `mismatch`) issues no hours and mails `SPARKPIPE_PAY_NOTIFY`. Check the invoice in the dashboard; if it is
  right, `sparkpipe_pay.py release <order>` fetches the invoice again and,
  once CoinPayments says `completed`, issues the hours the amount received
  covers: whole hours at the order's price for an hourly order, the amount
  before VAT for a two-year prepayment. It appends to the same ledger
  safely while the service runs.
- A customer whose payment fell short or arrived late is refunded through
  CoinPayments' own refund flow; the ledger records no hours for it.

## Tests

`python3 tests/test_pay.py` checks the Sunday-free tranches against
agreement clause 6.6, the session tokens, and the whole service against a
fake CoinPayments that verifies request signatures and a fake mail server:
codes, sign-in, terms acceptance and re-acceptance, cross-site refusal,
refusal of coins that are not USDC or USDT, invoice bodies on the chosen
coin and network, signed and tampered webhooks,
receipts, reconciliation, the holding limit, gates, mismatched totals,
failed invoices, quote requests and the ledger chain with two writers.
