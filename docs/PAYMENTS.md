# Payments

Fleet-hours are paid in USDC through CoinPayments, on any USDC network the
merchant account accepts. Won bank transfers against a tax invoice and USD
wires under a two-year agreement are still arranged by hand.

## How a payment works

1. On `https://sparkpipe.ai/buy.html` the customer connects an Ethereum
   wallet. The address is the account.
2. The page sends the order to `POST /pay/api/quote`. The service prices it
   from `site/pricing.js` (never from the browser), checks the holding limit
   or the agreement's gates, and returns the exact text to sign.
3. The wallet signs that text with `personal_sign` (EIP-191). For hourly
   orders the text states the SHA-256 hashes of `terms/hourly-ko.md` and
   `terms/hourly-en.md`, so the signature is the acceptance of the Hourly
   Terms (clause 12.4). Signing moves no funds.
4. `POST /pay/api/order` recovers the signer, records the order in the
   ledger and creates a CoinPayments invoice in USD (currency `5057`): the
   price, VAT at 10% as tax, and a per-invoice webhook. The page opens the
   CoinPayments checkout, where the customer picks a USDC network and pays.
5. CoinPayments posts signed webhooks to `/pay/api/coinpayments`. The
   service checks the HMAC and the timestamp, then fetches the invoice from
   the CoinPayments API and acts on that answer, not on the webhook body.
   Hours are issued only when the invoice is `completed` and its total and
   currency match the order.
6. CoinPayments sends the customer back to `buy.html?order=…`, which shows
   the order's state from `GET /pay/api/order/<order>`.

Every five minutes the service also asks CoinPayments about each open
invoice from the last 14 days, so a lost webhook only delays the hours.

## Endpoints

| Route | Does |
| --- | --- |
| `POST /pay/api/quote` | Price an hourly order (`company`, `email`, `hours`, `start`) or a two-year prepayment (`agreement`, `amount` before VAT) for an `account`; returns the order number and the text to sign. A quote lasts 15 minutes |
| `POST /pay/api/order` | `order` and `signature`: creates the invoice and returns the checkout link |
| `GET /pay/api/order/<order>` | The order's state, amounts and hours; the checkout link while unpaid |
| `POST /pay/api/agreement` | `agreement` and `account`: the agreement's hours, price, amount paid, amount left and next gate |
| `POST /pay/api/coinpayments` | CoinPayments webhooks |
| `GET /pay/api/health` | `{"ok": true}` |

Each client IP address may ask for 30 quotes, 10 orders and 30 agreement
lookups in ten minutes.

## Rules the service enforces

- Hourly hours start no earlier than `hourlyEarliestStart` and within a
  year, and last 12 months (clause 4.2).
- The holding limit is 168 hours, or 720 for an address listed in
  `approved.json` (clause 4.1). It counts unexpired issued hours and orders
  still awaiting payment for 24 hours. Until reservations record
  consumption, every unexpired hour issued counts as held.
- A two-year prepayment is at least 1.00 USD and at most what is left of
  the term's price, counting unpaid orders. It is refused after Phase A
  with nothing paid (clause 5.2), after a missed gate (clause 6.4), and
  after the term ends. Each payment issues the amount before VAT divided by
  the hour price, to four decimals (clause 5.5).
- VAT is charged on every USDC payment (Hourly Terms 3.2, Schedule (c)).
  CoinPayments' own fee is ours; the customer pays the network fee.

## Data on the VPS

`/var/lib/sparkpipe-pay`, owned by the service user:

- `ledger.jsonl`: one JSON line per event (`order`, `invoice`,
  `invoice_failed`, `invoice_state`, `payment`, `hours_issued`), each with
  the SHA-256 of the line before it. The service refuses to start if the
  chain is broken. An order line keeps the signed text, the signature, the
  IP address and the browser, as the privacy policy's acceptance records
  say.
- `approved.json`: the addresses that passed the business check, for
  example `["0xabc…"]`.
- `agreements.json`: one entry per signed two-year agreement:

```json
[{"id": "SP2Y-7K3QD9", "account": "0xabc…", "company": "Acme Co., Ltd.", "email": "billing@acme.example", "signed": "2026-12-15", "hourPrice": "15.00"}]
```

Give agreements numbers that cannot be guessed: the number and the
account together reveal the price. The service rereads both files on every
request, so edits need no restart.

## CoinPayments setup

1. In the CoinPayments dashboard, create an API integration and keep its
   client ID and client secret. Webhooks need no dashboard setting: every
   invoice names `https://sparkpipe.ai/pay/api/coinpayments` and the events
   `invoicePending`, `invoicePaid`, `invoiceCompleted`, `invoiceCancelled`
   and `invoiceTimedOut`. The webhook signature covers that exact URL, so
   it must not change behind the proxy.
2. In Payment Settings, accept only USDC: `USDC.ERC20`, `USDC.BASE`,
   `USDC.ARB`, `USDC.POL`, `USDC.SOL` and `USDC.BEP20`. Turn LTCT off.
3. Choose the payout wallet or conversion there too; the service does not
   set payout overrides.

## Install on the VPS

```sh
sudo useradd --system --home /var/lib/sparkpipe-pay --shell /usr/sbin/nologin sparkpipe-pay
sudo install -d -o sparkpipe-pay -g sparkpipe-pay -m 700 /var/lib/sparkpipe-pay
sudo install -d -m 755 /opt/sparkpipe-pay /etc/sparkpipe-pay
sudo install -m 644 tools/sparkpipe_pay.py tools/eth_signature.py /opt/sparkpipe-pay/
sudo install -m 640 -g sparkpipe-pay deployment/vps/sparkpipe-pay.env.example /etc/sparkpipe-pay/env
sudo install -m 640 -g sparkpipe-pay /dev/null /etc/sparkpipe-pay/client_secret
sudo install -m 644 deployment/vps/sparkpipe-pay.service /etc/systemd/system/
sudo install -m 644 deployment/vps/sparkpipe-pay.conf /etc/apache2/sparkpipe/
```

Fill `COINPAYMENTS_CLIENT_ID` in `/etc/sparkpipe-pay/env` and paste the
client secret into `/etc/sparkpipe-pay/client_secret` with an editor, not
on a command line. Include `sparkpipe-pay.conf` in the port-443 virtual
host beside the other `deployment/vps` includes; `mod_proxy_http` is already
on for the public API. Copy `site/` to `/var/www/html` as usual, including
`site/terms/`: the signed text points customers at
`https://sparkpipe.ai/terms/hourly-ko.md` and `hourly-en.md`, and the
service hashes the copies under `/var/www/html/terms`. Then:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now sparkpipe-pay
sudo apachectl configtest && sudo systemctl reload apache2
curl -s https://sparkpipe.ai/pay/api/health
```

The service needs Python 3.8 or later and nothing outside its standard
library.

## Operating it

- `journalctl -u sparkpipe-pay` logs each invoice, payment and issue of
  hours, and every refused webhook signature, without payload contents.
- `sparkpipe_pay.py verify` checks the ledger chain and prints its head. Run
  the admin commands as the service user with its environment:
  `sudo -u sparkpipe-pay sh -c 'set -a; . /etc/sparkpipe-pay/env; python3 /opt/sparkpipe-pay/sparkpipe_pay.py verify'`.
- An order whose completed invoice does not match (state `mismatch`) issues
  no hours. Check the invoice in the dashboard; if the customer paid in
  full, `sparkpipe_pay.py release <order>` fetches the invoice again and
  issues the hours when CoinPayments says `completed`. It appends to the
  same ledger safely while the service runs.
- A customer whose payment fell short or arrived late is refunded through
  CoinPayments' own refund flow; the ledger records no hours for it.

## Tests

`python3 tests/test_pay.py` checks Keccak-256 and signature recovery
against vectors from eth-account, the Sunday-free tranches against
agreement clause 6.6, and the whole service against a fake CoinPayments
that verifies request signatures: quotes, signed orders, invoice bodies,
signed and tampered webhooks, reconciliation, the holding limit, gates,
mismatched totals and the ledger chain with two writers.
