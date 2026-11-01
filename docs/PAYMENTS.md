# Payments

Customers buy hour-credits (HC) and spend them on Hours, Weeks, the waitlist and prefill accelerators (`docs/FLEET_BOOKINGS.md`). HC is paid in Korean won through Toss Payments (토스페이먼츠), or in US-dollar stablecoins through CoinPayments. HYOR AI Tech Ltd. stays the seller: both are payment processors that receive the money for us (clause 3.3 of the Customer Agreement). Toss Payments takes Korean and overseas cards, bank transfer, virtual accounts and Korean easy pay, in won at the `krw` prices of `hc.prices` in `site/pricing.js`, plus 10% VAT. The stablecoin checkout offers USDC on Base, Solana, Arbitrum, Polygon, BNB Chain or Ethereum, and USDT on Tron, Solana, BNB Chain, Arbitrum, Polygon or Ethereum, at the `usd` prices. CoinPayments' own currency menu also takes any other stablecoin enabled in Payment Settings. Only stablecoins are enabled, so an invoice priced in dollars is paid in dollars. HC follows the US-dollar value CoinPayments reports for a payment (`nativeActualAmount`), whatever coin it came in, and the receipt names the coin that was paid (`paymentCurrencySymbol`). USD bank transfers (clause 3.3(b)) are arranged by hand and their HC issued with `credit`.

The service is `tools/sparkpipe_pay.py` with `tools/sparkpipe_market.py`. It needs Python 3.8 or later and nothing outside the standard library.

## How a customer buys

1. `POST /pay/api/code` emails a 6-digit code. `POST /pay/api/verify` checks it, opens the account if the email is new (with the company's legal name and `accept: true`), records the acceptance of the Customer Agreement and sets the session cookie. The email is the account: there is no password.
2. `POST /pay/api/order` with `product: "credits"` and `hc`, any quantity from `hc.min` to six decimal places (clauses 13.1 and 13.2), prices the order from `site/pricing.js` (never from the browser): the price per HC is the `hc.prices` entry with the largest `from` at most `hc`, in won for Toss Payments and in dollars for stablecoins; the net is that price times `hc` and VAT is `vatRate` of the net, each rounded to the won or the cent. A card or won order goes on as under Toss Payments below. A stablecoin order (`coin`, one of the twelve symbols in `COINS` such as `USDC.BASE` or `USDT.TRC20`) creates a CoinPayments invoice in USD (currency `5057`) with the price, VAT as tax, a per-invoice webhook, and `payment.paymentCurrency` set to the chosen coin on the chosen network, and returns `{order, checkout}`.
3. CoinPayments posts signed webhooks to `/pay/api/coinpayments`. The service checks the HMAC and the timestamp, then fetches the invoice from the CoinPayments API and acts on that answer, not on the webhook body. HC is issued only when the invoice is `completed` and its total and currency match the order: the whole order when the amount received covers the total, or, when CoinPayments completes it for less, in proportion to the amount received, to six decimals at the order's price per HC. The HC is a lot that lapses 12 months after issue. The customer gets a receipt by email, and `SPARKPIPE_PAY_NOTIFY` gets a note of each new account, payment and mismatch.
4. The processor sends the customer back to `account.html?order=…`, which follows `GET /pay/api/order/<order>`.

Every five minutes the service also asks CoinPayments about each open invoice from the last 14 days, and Toss Payments about each open order, so a lost webhook only delays the HC.

## Endpoints

| Route | Does |
| --- | --- |
| `POST /pay/api/code` | `email`: emails a 6-digit sign-in code, valid 10 minutes for 5 tries |
| `POST /pay/api/verify` | `email`, `code`, and for a new account `company` and `accept: true`: signs in, sets the cookie `sp_session` and returns the account |
| `POST /pay/api/signout` | Clears the cookie |
| `GET /pay/api/account` | The signed-in account: `email`, `company`, `since`, `termsCurrent`, `hc` (the unlocked balance), `locked` and the last 50 `orders` |
| `POST /pay/api/order` | Signed in. `product: "credits"`, `hc` and `accept: true` when the Agreement changed since the account accepted it; then either `coin`, which returns `order` and `checkout`, or `processor: "toss"` with an optional `brn`, which returns `order` and the Toss Payments request for the page (`toss`: `clientKey`, `variantKey`, `test`, `orderId`, `orderName`, `amount` with VAT, `currency`, `customerEmail`, `customerName`, `successUrl`, `failUrl`). 503 `cards_closed` without Toss Payments keys |
| `GET /pay/api/order/<order>` | The order: `order`, `product`, `hc`, `price`, `net`, `vat`, `total`, `currency`, `coin`, `processor`, `state`, `issued`, `issuedHC`, `created`, `paid`, `received`, `method`, `deposit`, `refunded`, `reason`, `expires`, and `checkout` while unpaid. The order number is 80 random bits |
| `GET /pay/api/toss/config` | Whether cards are offered (`enabled`), the client key, the payment UI's `variantKey` and whether it is test mode |
| `GET /pay/api/toss/success/<order>` | Toss Payments' success redirect: checks the amount, confirms the payment and sends the browser to `account.html?order=…` |
| `GET /pay/api/toss/fail/<order>` | Toss Payments' fail redirect: closes the unpaid order |
| `POST /pay/api/toss/webhook` | Toss Payments webhooks (`PAYMENT_STATUS_CHANGED`, `DEPOSIT_CALLBACK`) |
| `POST /pay/api/coinpayments` | CoinPayments webhooks |
| `GET /pay/api/health` | `{"ok": true}` |

The booking routes, the gateway, the holder schedule and `availability.txt` are in `docs/FLEET_BOOKINGS.md`.

The session cookie is HttpOnly, SameSite=Lax, Secure and scoped to `/pay/api`. It holds the account number, its expiry and an HMAC under `session_key` in the data directory, made on first start. A POST whose `Origin` is not `SPARKPIPE_PAY_PUBLIC` is refused. Each client IP address may ask for 20 codes, 30 verifications and 10 orders in ten minutes, and each email for 5 codes an hour.

## Rules the service enforces

- An account opens only with the box ticked; the ledger keeps the statement accepted, the SHA-256 hashes of `terms/agreement-ko.md` and `terms/agreement-en.md`, the IP address and the browser. When either changes, the next order, Hour or Week booking or waitlist lock asks the customer to accept the new texts (409 `terms`, then `accept: true`), as clause 9.1 requires.
- HC is bought in any quantity from `hc.min`, to six decimal places, with no maximum beyond what the processors take, and each lot lapses `hc.validityMonths` (12) months after issue.
- No HC is issued before its payment counts (clause 3.5).

## Data on the VPS

`/var/lib/sparkpipe-pay`, owned by the service user:

- `ledger.jsonl`: one JSON line per event (`account`, `acceptance`, `order`, `invoice`, `invoice_failed`, `invoice_state`, `toss_payment`, `toss_state`, `payment`, `hc_issued`, `refund`, `tax_invoice`, and the market lines in `docs/FLEET_BOOKINGS.md`), each with the SHA-256 of the line before it. The service refuses to start if the chain is broken. An account or order line keeps the IP address and the browser, as the privacy policy's acceptance records say; a payment line keeps the coin CoinPayments reports, or, for Toss Payments, the payment key, the method, the card-slip link and `taxInvoice` (`needed`, `brn`, `supply`, `vat`). Lines of types the service no longer writes, such as the live ledger's `hourly` order, replay as they are.
- `session_key`: 32 random bytes that sign the session cookies. Deleting it signs everyone out.

## Mail

The service sends the sign-in codes, receipts and notes through `SPARKPIPE_PAY_SMTP` (default `127.0.0.1:25`, the VPS's own mail server in `docs/VPS_MAIL.md`) as `SPARKPIPE_PAY_MAIL_FROM` with replies to `SPARKPIPE_PAY_NOTIFY`. A code that cannot be sent is an error on the page; any other mail that cannot be sent is logged. The VPS's rspamd signs that mail with DKIM for sparkpipe.ai, selector `s202608` (`deployment/vps/rspamd-dkim_signing.conf`); with SPF naming the VPS and DMARC at `p=quarantine`, receivers see SPF, DKIM and DMARC pass.

## Toss Payments

The checkout uses Toss Payments' checkout-page payment (주문서형 결제, SDK v2). Only `buy.html` and `account.html` load the SDK, and only those two pages get a Content-Security-Policy that admits `*.tosspayments.com` and `*.toss.im` (`deployment/vps/sparkpipe-ai-ssl.conf`).

1. The page asks `GET /pay/api/toss/config`. Without keys (`enabled: false`) it offers only USDC or USDT.
2. On Pay, `POST /pay/api/order` with `processor: "toss"` prices the order in won on the server, records it with `processor`, `currency: "KRW"`, `brn` and an expiry 45 minutes ahead (Toss Payments' 30-minute payment window and its 10 minutes to confirm), and returns the request.
3. Toss Payments redirects to `/pay/api/toss/success/<order>?paymentKey&orderId&amount`. The service refuses an `amount` other than the order's (state `failed`, reason `amount`, nothing confirmed), records the payment key (`toss_payment`), and calls `POST /v1/payments/confirm` with the order's own amount and `Idempotency-Key: confirm-<order>`. A refused confirm is followed by `GET /v1/payments/{paymentKey}` before anything is recorded.
4. Every Toss Payments answer goes through one check: the payment must name this order and the stored payment key. `DONE` with the order's amount in KRW issues the HC once (`payment`, `hc_issued`, `hc_lot`); `DONE` or `WAITING_FOR_DEPOSIT` for another amount or currency is cancelled through the cancel API (state `cancelled`, reason `amount`, or `mismatch` if the cancel fails, with a note to `SPARKPIPE_PAY_NOTIFY`); `WAITING_FOR_DEPOSIT` records the virtual account (bank, number, deadline) as state `waiting` and emails it to the customer; `ABORTED`, `EXPIRED` and `CANCELED` close the order.
5. `POST /pay/api/toss/webhook` acknowledges every event and only queues a check of the order it names: the service fetches the payment from Toss Payments with the payment key it stored and applies that answer, never the webhook body.
6. A card order not completed within 45 minutes expires (`timedout`). A virtual account expires at the earlier of Toss Payments' deposit deadline and 24 hours after the order: the reconciliation cancels it at Toss Payments first, and issues the HC instead if the deposit arrived.

The receipt names the method; for a card it links the card slip, which is the VAT document. A payment made by bank transfer, into a virtual account or by easy pay from a bank account or balance is recorded with `taxInvoice.needed` and the business registration number the checkout took (`brn`). Issuing the electronic tax invoice stays a manual step:

- `sparkpipe_pay.py tax-invoices` lists every such payment without an invoice recorded: order, date (KST), company, email, registration number (or "missing: ask the customer"), supply amount, VAT, total, method, and any refund that needs an amended invoice.
- `sparkpipe_pay.py tax-invoice ORDER NUMBER` records the National Tax Service approval number of the invoice issued (`tax_invoice`).

Refunds happen only where the Customer Agreement allows them (clauses 3.2, 3.5, 10.4, 16.4 or where the law requires), and go back by the method of the payment (clause 3.6):

```sh
sparkpipe_pay.py refund ORDER CLAUSE KRW|all REASON [BANK ACCOUNT HOLDER]
```

It fetches the payment, refuses a clause outside that list, an amount above what is left at Toss Payments, and, for clause 10.4 or the law, an amount whose HC (the same share of the order's HC) is no longer unspent, unlocked and unlapsed in the order's lot. A refund of VAT not owed (clause 3.2) leaves the HC. A clause 16.4 refund pays back locked HC that lapsed (`docs/FLEET_BOOKINGS.md`): it is limited to the refunds due recorded for that order less the clause 16.4 refunds already made, and debits no HC, since the lapse already took it. It calls `POST /v1/payments/{paymentKey}/cancel` with `cancelAmount` for part of the balance, `refundableAmount` as a guard and `Idempotency-Key: refund-<order>-<balance>`; a virtual account payment needs the bank code, account number and holder for `refundReceiveAccount`. It checks the balance Toss Payments reports, then writes a `refund` line (clause, reason, amount, balance left, HC debited) and an `hc_debit` from the order's own lot, and emails the customer.

### Setup in the Toss Payments console

1. Sign the electronic payment contract and add the methods: domestic cards, overseas cards in KRW, bank transfer, virtual accounts and the easy-pay wallets. In the payment admin (결제 어드민), build the payment UI; keep its `variantKey` `DEFAULT` or put it in `TOSS_VARIANT_KEY`.
2. In the virtual account settings, set the deposit period to one day or less and turn on the refund-account form. Settle with the accountant whether to turn off automatic cash receipts (현금영수증) for bank transfers and virtual accounts: a payment that gets an electronic tax invoice must not also get a cash receipt.
3. Register the webhook `https://sparkpipe.ai/pay/api/toss/webhook` for `PAYMENT_STATUS_CHANGED` and `DEPOSIT_CALLBACK` on the live MID, and `https://sparkpipe.ai/pay/sandbox/api/toss/webhook` in the test environment.
4. Copy the live checkout-page keys (`live_gck_…`, `live_gsk_…`) into `/etc/sparkpipe-pay/env` as `TOSS_CLIENT_KEY` and `TOSS_SECRET_KEY`, by editing the file, never on a command line, and restart the service. The live service refuses test keys and the sandbox refuses live keys; a key without the other stops the service. Under API key access policy, allow only the VPS's address.

## CoinPayments setup

1. In the CoinPayments dashboard, create an API integration and keep its client ID and client secret. Webhooks need no dashboard setting: every invoice names `https://sparkpipe.ai/pay/api/coinpayments` and the events `invoicePending`, `invoicePaid`, `invoiceCompleted`, `invoiceCancelled` and `invoiceTimedOut`. The webhook signature covers that exact URL, so it must not change behind the proxy.
2. Keep `USDC.ERC20`, `USDC.BASE`, `USDC.ARB`, `USDC.POL`, `USDC.SOL`, `USDC.BEP20`, `USDT.TRC20`, `USDT.ERC20`, `USDT.BEP20`, `USDT.SOL`, `USDT.POL` and `USDT.ARB` on in Payment Settings, and every coin that is not a US-dollar stablecoin off, LTCT included. Set the underpayment limit to a fixed $100,000 with no percentage (see Operating it).
3. Choose the payout wallet or conversion there too; the service does not set payout overrides.

CoinPayments emails the payment instructions to the invoice's refund email, which is the account email.

## Install on the VPS

```sh
sudo useradd --system --home /var/lib/sparkpipe-pay --shell /usr/sbin/nologin sparkpipe-pay
sudo install -d -o sparkpipe-pay -g sparkpipe-pay -m 700 /var/lib/sparkpipe-pay /var/lib/sparkpipe-pay-sandbox
sudo install -d -m 755 /opt/sparkpipe-pay /etc/sparkpipe-pay
sudo install -m 644 tools/sparkpipe_pay.py tools/sparkpipe_market.py /opt/sparkpipe-pay/
sudo install -m 640 -g sparkpipe-pay deployment/vps/sparkpipe-pay.env.example /etc/sparkpipe-pay/env
sudo install -m 640 -g sparkpipe-pay deployment/vps/sparkpipe-pay-sandbox.env.example /etc/sparkpipe-pay/sandbox.env
sudo install -m 640 -g sparkpipe-pay /dev/null /etc/sparkpipe-pay/client_secret
sudo install -m 640 -g sparkpipe-pay /dev/null /etc/sparkpipe-pay/pool_key
sudo install -m 644 deployment/vps/sparkpipe-pay.service deployment/vps/sparkpipe-pay-sandbox.service /etc/systemd/system/
sudo install -m 644 deployment/vps/sparkpipe-pay.conf /etc/apache2/sparkpipe/
sudo install -m 644 deployment/vps/sparkpipe-ai-ssl.conf /etc/apache2/sites-available/000-default-le-ssl.conf
```

Fill `COINPAYMENTS_CLIENT_ID` in `/etc/sparkpipe-pay/env`, put the client secret into `/etc/sparkpipe-pay/client_secret` and the upstream key of the Fleets into `/etc/sparkpipe-pay/pool_key` by copying files, never on a command line. The upstream key authenticates both the gateway's requests to `SPARKPIPE_POOL_UPSTREAM` and the hub's calls to the holder schedule. `sparkpipe-pay.conf` proxies `/pay/api/`, `/pay/sandbox/api/` and `/availability.txt`; `mod_proxy_http` must be on. Copy `site/` to `/var/www/html` as usual, including `site/terms/`: the acceptance names `https://sparkpipe.ai/terms/agreement-ko.md` and `agreement-en.md`, and the service hashes the copies under `/var/www/html/terms`. Then:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now sparkpipe-pay sparkpipe-pay-sandbox
sudo apachectl configtest && sudo systemctl reload apache2
curl -s https://sparkpipe.ai/pay/api/health https://sparkpipe.ai/availability.txt
```

## Deploying a change

Only what is merged to `main` goes to the VPS: commit to a PR, run the tests, merge, then deploy `main`. Pages, the service and the host configuration in `deployment/vps` all go the same way; nothing is edited on the host alone except the mail address lists, passwords and keys that `docs/VPS_MAIL.md` keeps out of the public repository.

```sh
git fetch origin main
git archive --format=tar.gz -o pay-main.tgz origin/main site tools/sparkpipe_pay.py tools/sparkpipe_market.py deployment/vps
```

On the VPS, back up `/var/www/html`, `/opt/sparkpipe-pay/*.py`, the ledgers and any configuration file being replaced, copy `site/.` over `/var/www/html`, install the service and changed configuration as in the install steps, check the configuration, then restart `sparkpipe-pay` and `sparkpipe-pay-sandbox` and reload Apache. The live pages, the service and the configuration must then match `origin/main` byte for byte.

## Operating it

- `journalctl -u sparkpipe-pay` logs each account, invoice, payment and issue of HC, every refused webhook signature, every market tick error and every mail that failed, without payload contents.
- Run the admin commands as the service user with its environment: `sudo -u sparkpipe-pay sh -c 'set -a; . /etc/sparkpipe-pay/env; python3 /opt/sparkpipe-pay/sparkpipe_pay.py verify'`. They append to the ledger safely while the service runs.
  - `verify` checks the ledger chain and prints its head and counts.
  - `release ORDER` fetches a `mismatch` invoice again and, once CoinPayments says `completed`, issues the HC the amount received covers at the order's price per HC.
  - `credit EMAIL HC NOTE` adds a lot of HC under clause 4.4, or for a USD bank transfer.
  - `maintenance FLEET YYYY-MM-DD [HOURS]` schedules maintenance in the Week starting on that Monday (`docs/FLEET_BOOKINGS.md`).
  - `refund`, `tax-invoices` and `tax-invoice` are under Toss Payments above.
- Part payments settle without anyone. CoinPayments completes an underpaid invoice when the shortfall is within the underpayment limit; with a fixed $100,000 and no percentage, every part payment of an invoice under $100,000 completes with what arrived. The service then issues HC in proportion, mails the customer a receipt that says so, and mails `SPARKPIPE_PAY_NOTIFY` "Paid in part". A part payment CoinPayments does not complete stays `pending` until the payment window closes, when CoinPayments refunds it. Overpayments and late payments are refunded the same way.
- A completed invoice whose total or currency does not match the order (state `mismatch`) issues no HC and mails `SPARKPIPE_PAY_NOTIFY`; check it in the dashboard and use `release`.
- When locked HC lapses, `SPARKPIPE_PAY_NOTIFY` gets "Refund due: locked HC lapsed" with each order and amount: refund a Toss Payments order with `refund ORDER 16.4 AMOUNT REASON`, and a stablecoin order through CoinPayments' refund flow.

## Tests

`python3 tests/test_pay.py` runs the whole service against a fake CoinPayments that verifies request signatures, a fake Toss Payments that checks the secret key and records each call, and a fake mail server: session tokens, codes, sign-in, acceptance of the two Agreement texts and re-acceptance, cross-site refusal, the removed routes, HC orders by quantity (whole or not) and the price ladder at 143 and 144 HC in dollars and won, invoice bodies on the chosen coin and network, signed and tampered webhooks, receipts, part payments, reconciliation, mismatched totals, failed invoices, the ledger chain with two writers and the replay of the live ledger's nine lines; Toss Payments key gating, won amounts fixed on the server, a confirm sent once with its idempotency key, other amounts refused or cancelled, declined and cancelled payments, webhooks that only trigger a check, virtual account deposits and expiry, the refund command (clause 16.4 limited to lapsed locked HC) and the tax invoice list. `python3 tests/test_market.py` covers the bookings (`docs/FLEET_BOOKINGS.md`).
