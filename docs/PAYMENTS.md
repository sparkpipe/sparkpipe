# Payments

Customers buy hour-credits (HC) and spend them on Hours, Weeks and the waitlist (`docs/FLEET_BOOKINGS.md`). HC is paid in Korean won through Toss Payments (토스페이먼츠), in US-dollar stablecoins through CoinPayments, or in US dollars by USD bank transfer (SWIFT) straight to our US dollar account (clause 3.3 of the Customer Agreement). HYOR AI Tech Ltd. stays the seller: Toss Payments and CoinPayments are payment processors that receive the money for us. Toss Payments takes Korean and overseas cards, bank transfer, virtual accounts and Korean easy pay, in won at the `krw` prices of `hc.prices` in `site/pricing.js`, plus 10% VAT. The stablecoin checkout offers USDC on Base, Solana, Arbitrum, Polygon, BNB Chain or Ethereum, and USDT on Tron, Solana, BNB Chain, Arbitrum, Polygon or Ethereum, at the `usd` prices. CoinPayments' own currency menu also takes any other stablecoin enabled in Payment Settings. Only stablecoins are enabled, so an invoice priced in dollars is paid in dollars. HC follows the US-dollar value CoinPayments reports for a payment (`nativeActualAmount`), whatever coin it came in, and the receipt names the coin that was paid (`paymentCurrencySymbol`). A USD bank transfer (SWIFT, clause 3.3(b)) is priced at the `usd` prices with 0% VAT (`swiftVatRate`), only for a customer that declares it has no place of business in Korea and pays from its own account outside Korea; it is the only method without 10% VAT (SWIFT bank transfers, below).

The service is `tools/sparkpipe_pay.py` with `tools/sparkpipe_market.py`. It needs Python 3.8 or later and nothing outside the standard library.

## How a customer buys

1. `POST /pay/api/code` emails a 6-digit code. `POST /pay/api/verify` checks it, opens the account if the email is new (with the company's legal name and `accept: true`), records the acceptance of the Customer Agreement and sets the session cookie. The email is the account: there is no password.
2. `POST /pay/api/order` with `product: "credits"` and `hc`, any quantity from `hc.min` to six decimal places (clauses 13.1 and 13.2), with up to 9 digits before the point in one order (a larger quantity is refused with a message to place another order), prices the order from `site/pricing.js` (never from the browser): the price per HC is the `hc.prices` entry with the largest `from` at most `hc`, in won for Toss Payments and in dollars for stablecoins and SWIFT; the net is that price times `hc` and VAT is `vatRate` of the net (`swiftVatRate`, 0, for SWIFT), each rounded to the won or the cent. A card or won order goes on as under Toss Payments below, and a SWIFT order as under SWIFT bank transfers. A stablecoin order (`coin`, one of the twelve symbols in `COINS` such as `USDC.BASE` or `USDT.TRC20`) creates a CoinPayments invoice in USD (currency `5057`) with the price, VAT as tax, a per-invoice webhook, and `payment.paymentCurrency` set to the chosen coin on the chosen network, and returns `{order, checkout}`.
3. CoinPayments posts signed webhooks to `/pay/api/coinpayments`. The service checks the HMAC and the timestamp, then fetches the invoice from the CoinPayments API and acts on that answer, not on the webhook body. HC is issued only when the invoice is `completed` and its total and currency match the order: the whole order when the amount received covers the total, or, when CoinPayments completes it for less, in proportion to the amount received, to six decimals at the order's price per HC. The HC is a lot that lapses 12 months after issue. The customer gets a receipt by email, and `SPARKPIPE_PAY_NOTIFY` gets a note of each new account, payment and mismatch.
4. The processor sends the customer back to `account.html?order=…`, which follows `GET /pay/api/order/<order>`.

Every five minutes the service also asks CoinPayments about each open invoice from the last 14 days, and Toss Payments about each open order, so a lost webhook only delays the HC.

## Endpoints

| Route | Does |
| --- | --- |
| `POST /pay/api/code` | `email`: emails a 6-digit sign-in code, valid 10 minutes for 5 tries; while a code is live it emails that code again |
| `POST /pay/api/verify` | `email`, `code`, and for a new account `company` and `accept: true`: signs in, sets the cookie `sp_session` and returns the account. The code is used up only when a session is issued: a refusal for a missing `company` (404 `new`) or `accept` (409 `terms`) leaves it valid. An email with a pending team invitation joins that account as a member instead (`docs/FLEET_BOOKINGS.md`, Teams). With `bearer: true` and an optional `device` (the CLI), it sets no cookie and returns `session` (`sps-…`) and `sessionExpires` with the account |
| `POST /pay/api/signout` | Clears the cookie, or ends the bearer session sent with it |
| `GET /pay/api/account` | The signed-in account: `email` (the member's own), `company`, `member`, `role`, `since`, `termsCurrent`, `hc` (the unlocked balance), `locked` and, for the owner, the last 50 `orders`, each as `GET /pay/api/order/<order>` shows it |
| `POST /pay/api/order` | Signed in. `product: "credits"`, `hc` and `accept: true` when the Agreement changed since the account accepted it; then either `coin`, which returns `order` and `checkout`, or `processor: "toss"` with an optional `brn`, which returns `order` and the Toss Payments request for the page (`toss`: `clientKey`, `variantKey`, `test`, `orderId`, `orderName`, `amount` with VAT, `currency`, `customerEmail`, `customerName`, `successUrl`, `failUrl`). 503 `cards_closed` without Toss Payments keys. Or `processor: "swift"` with `declare: true`, which returns `order`, `processor`, `reference`, `hc`, `price`, `net`, `vatRate` (`"0"`), `vat` (`"0.00"`), `total`, `currency` (`USD`) and `bank` (`beneficiary`, `account`, `bank`, `address`, `bic`); 400 `declaration` without `declare: true`, 503 `swift_closed` without the bank settings, 404 in the sandbox |
| `GET /pay/api/order/<order>` | Signed in as the owner of the order's account (401 signed out, 403 for another role, 404 for another account's order). The order: `order`, `product`, `hc`, `price`, `net`, `vatRate`, `vat`, `total`, `currency`, `coin`, `processor`, `state`, `issued`, `issuedHC`, `created`, `paid`, `received`, `method`, `deposit`, `refunded`, `reason`, `expires`, `reference` and `bank` (SWIFT), and `checkout` while unpaid. The order number is 80 random bits |
| `GET /pay/api/swift/config` | Whether USD bank transfers (SWIFT) are offered (`enabled`): all five bank settings set, outside the sandbox |
| `GET /pay/api/toss/config` | Whether cards are offered (`enabled`), the client key, the payment UI's `variantKey` and whether it is test mode |
| `GET /pay/api/toss/success/<order>` | Toss Payments' success redirect: checks the amount, confirms the payment and sends the browser to `account.html?order=…` |
| `GET /pay/api/toss/fail/<order>` | Toss Payments' fail redirect: closes the unpaid order |
| `POST /pay/api/toss/webhook` | Toss Payments webhooks (`PAYMENT_STATUS_CHANGED`, `DEPOSIT_CALLBACK`) |
| `POST /pay/api/coinpayments` | CoinPayments webhooks |
| `GET /pay/api/health` | `{"ok": true}` |

The booking routes, the gateway, the holder schedule and `availability.txt` are in `docs/FLEET_BOOKINGS.md`.

The session cookie is HttpOnly, SameSite=Lax, Secure and scoped to `/pay/api`. It holds the account number, its expiry and an HMAC under `session_key` in the data directory, made on first start. A bearer session is for the CLI: `Authorization: Bearer sps-…` works wherever the cookie does, on `/account`, `/market/...` and the API. It lasts 90 days. `sessions.json` keeps only its SHA-256, with the session id, the account, the `device` name and `created` and `expires`; signing out or `POST /market/sessions/revoke` ends it and deletes that record, and an expired one is deleted at the next housekeeping (every five minutes). `GET /market/state` lists the account's live bearer sessions. A POST whose `Origin` is not `SPARKPIPE_PAY_PUBLIC` is refused. Each client IP address may ask for 20 codes, 30 verifications and 10 orders in ten minutes; each email may get 5 codes an hour from one IP address and 30 an hour in all. Asking again while a code is live sends that same code and restarts its 10 minutes without resetting its tries, so nobody else can invalidate the code a customer just received.

## Rules the service enforces

- An account opens only with the box ticked; the ledger keeps the statement accepted, the SHA-256 hashes of `terms/agreement-ko.md` and `terms/agreement-en.md`, the IP address and the browser. When either changes, the next order, Hour or Week booking or waitlist lock asks the customer to accept the new texts (409 `terms`, then `accept: true`), as clause 9.1 requires.
- HC is bought in any quantity from `hc.min`, to six decimal places, with no maximum beyond what the processors take, and each lot lapses `hc.validityMonths` (12) months after issue.
- No HC is issued before its payment counts (clause 3.5).
- VAT is 10% on every payment except a USD bank transfer (SWIFT), which is 0% and is taken only with the customer's clause 3.3(b) declaration.

## Data on the VPS

`/var/lib/sparkpipe-pay`, owned by the service user:

- `ledger.jsonl`: one JSON line per event (`account`, `acceptance`, `order`, `invoice`, `invoice_failed`, `invoice_state`, `toss_payment`, `toss_state`, `swift_state`, `payment`, `hc_issued`, `refund`, `tax_invoice`, and the market lines in `docs/FLEET_BOOKINGS.md`), each with the SHA-256 of the line before it. The service refuses to start if the chain is broken. An account or order line keeps the IP address and the browser, as the privacy policy's acceptance records say; a payment line keeps the coin CoinPayments reports, or, for Toss Payments, the payment key, the method, the card-slip link and `taxInvoice` (`needed`, `brn`, `supply`, `vat`). A SWIFT order line keeps its `reference`, the `bank` details its invoice gave and the customer's `declaration`; `swift_state` records the operator's confirmation (`received`, `note`) or cancellation (`reason`). Lines of types the service no longer writes, such as the live ledger's `hourly` order, replay as they are.
- `session_key`: 32 random bytes that sign the session cookies. Deleting it signs everyone out of the page; bearer sessions are in `sessions.json`.
- `sessions.json` (mode 600): the live bearer sessions, above, each with its team `member`.
- `audit/<account>/<day>.jsonl`: the audit log of each account (`docs/FLEET_BOOKINGS.md`, Teams), without prompts or replies, deleted a year after its day.
- `requests/<account>/<day>.jsonl`: the usage records the hub reports, one line per request, without prompts or replies, deleted a year after their day (`docs/FLEET_BOOKINGS.md`).
- `uptime/state.json` and `uptime/<YYYY-MM>.jsonl`: each Fleet's last good probe and its down periods, kept five years (`docs/FLEET_BOOKINGS.md`, Uptime and the downtime credit).
- `notices/<day>.jsonl` and `deliveries/<day>.jsonl`: the notices sent to accounts and the outcome of each webhook delivery, deleted 30 days after their day.
- `webhooks.json` (mode 600): each account's webhooks with their signing secrets, and its email subscription without the address, until the account deletes them.
- `notices/`, `deliveries/` and `requests/` are folders of mode 700 with files of mode 600, and both units set `UMask=0077`, so every file the service creates is its own. A line of these files that does not parse is skipped and logged, and a torn last line (from a full disk or a power cut) is cut back to the last newline before the next record is appended, so the service always starts. The ledger is different: a torn or broken line stops the service until an operator looks.

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

### Saved payment for automatic top-ups (not built)

An automatic HC top-up would charge a saved card through a Toss Payments billing key (자동결제, 빌링) when auto-extend waits for HC (`docs/FLEET_BOOKINGS.md`), so that a short balance never costs a holder its Fleet. It is not built, because three things are missing:

1. **The merchant contract.** Billing keys are a separate Toss Payments service that Toss reviews and adds to a MID by contract. Nothing in this repository or the live configuration records that our MID has it. The service today holds only the checkout-page keys (`live_gck_…`, `live_gsk_…`); billing uses the API individual-integration keys (`live_ck_…`, `live_sk_…`) of a MID with 자동결제 enabled, and takes cards only.
2. **The customer's standing authorization.** Clause 3.3(a) of the Customer Agreement has Customer pay through Toss Payments "as its payment window offers them for the order", and a billing charge has no payment window. Whether an Order placed by a standing authorization fits clause 3.3(a) is for the pending legal review; this service does not change the contract text.
3. **The operator's settings:** the billing keys in `/etc/sparkpipe-pay/env`, and the limits the account page would offer (the most HC per top-up and per month).

Once the contract and the legal review allow it, the build is: the account page calls the SDK's `requestBillingAuth({method: "CARD", …})`; the success redirect sends `authKey` and `customerKey` to `POST /v1/billing/authorizations/issue`; the billing key is kept in a mode-600 file like `webhooks.json`, never in the ledger, which records only that a card is saved (its masked number) and the customer's limits and acceptance; a top-up is an ordinary Toss Payments order, priced on the server at the `krw` ladder with VAT, charged with `POST /v1/billing/{billingKey}` and `Idempotency-Key: billing-<order>`, and settled through the same check as every Toss Payments payment (`payment`, `hc_issued`, `hc_lot`, receipt by email); it runs only when auto-extend writes `auto_extend_wait` with `insufficient_hc`, for the HC that Week needs (at least `hc.min`), within the customer's limits; and one click deletes the card, its billing key at Toss Payments and the file entry.

## SWIFT bank transfers

A USD bank transfer (SWIFT) is clause 3.3(b): US dollars from the customer's own account outside Korea to our US dollar account at a Korean foreign exchange bank. It is zero-rated for a customer with no place of business in Korea, so it is the only method at 0% VAT; Toss Payments and USDC/USDT stay at 10%.

1. `buy.html` offers "USD bank transfer (SWIFT)" when `GET /pay/api/swift/config` says `enabled`. The customer ticks the declaration that its company has no place of business in Korea and pays from its own account outside Korea; without it the service answers 400 `declaration` and takes no order.
2. `POST /pay/api/order` with `processor: "swift"` and `declare: true` prices the order on the `usd` ladder with `swiftVatRate` (0) and records it, state `unpaid`, with a payment reference of its own (`SW` and 8 random base32 characters, unique among orders, not derived from the order number), the bank details from the settings and the declaration text. The page then opens `account.html?order=…`, which shows the amount, the reference and the bank details, and the account's "USD bank transfers (SWIFT) to make" lists every unpaid SWIFT order the same way.
3. The customer is emailed the invoice: seller, buyer, the declaration, HC, price, net, VAT (0% on SWIFT payments), the total, the beneficiary, account number, bank, bank address, SWIFT/BIC and the payment reference, with charges OUR so that the whole amount arrives (clause 3.4). `SPARKPIPE_PAY_NOTIFY` is told of the order and its reference.
4. When the transfer reaches the account (clause 3.5), the operator runs `swift-confirm REFERENCE USD [NOTE]` (or the order number instead of the reference) with the whole amount received for the order. It refuses an amount below the invoice total and records nothing: the Agreement has the customer pay the full amount and bear every bank charge (clause 3.4), and its proportional rule covers only stablecoins (clause 3.5), so there is no part-payment rule for a bank transfer. Ask the customer for the rest, and once it arrives confirm the sum of the transfers. At or above the total it writes `swift_state` (`paid`, `received`, `note`), then `payment`, `hc_issued` and `hc_lot`, and emails the receipt. It refuses an order that is cancelled or already issued its HC.
5. An amount above the total issues the order's HC; the receipt says the excess goes back, and `SPARKPIPE_PAY_NOTIFY` gets "Overpaid by SWIFT". Send it back by SWIFT to the account it came from, then record it with `refund ORDER 3.5 USD REASON`: clause 3.5 on a SWIFT order returns only what arrived above the total and debits no HC.
6. `swift-cancel REFERENCE REASON` cancels an unpaid SWIFT order (state `cancelled`, the reason shown on the order) and emails the customer not to send a transfer. A transfer that still arrives for it goes back by SWIFT to the account it came from.

Refunds of a SWIFT order use `refund ORDER CLAUSE USD|all REASON` with the same rules as Toss Payments (below): the clause list, at most what is left of the amount received, clause 16.4 only up to what lapsed while locked, clause 3.5 only what arrived above the total, clauses 3.2, 3.5 and 16.4 debiting no HC, and clause 10.4 and the law only up to the HC that is unspent, unlocked and not lapsed in the order's lot. Nothing is sent automatically: the command records the `refund` line and the `hc_debit`, emails the customer that it goes back by SWIFT to the account it came from (clause 3.6), and prints the amount to send.

The bank details are five settings in `/etc/sparkpipe-pay/env`, empty in the example: `SPARKPIPE_PAY_SWIFT_BENEFICIARY`, `SPARKPIPE_PAY_SWIFT_BANK`, `SPARKPIPE_PAY_SWIFT_BANK_ADDRESS`, `SPARKPIPE_PAY_SWIFT_ACCOUNT` and `SPARKPIPE_PAY_SWIFT_BIC`, each on one line in double quotes. With all of them empty SWIFT is not offered; with some set and others empty, or a SWIFT/BIC that is not 8 or 11 characters, the service does not start. The sandbox never offers SWIFT.

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

Fill `COINPAYMENTS_CLIENT_ID` and the five `SPARKPIPE_PAY_SWIFT_*` bank settings in `/etc/sparkpipe-pay/env`, put the client secret into `/etc/sparkpipe-pay/client_secret` and the upstream key of the Fleets into `/etc/sparkpipe-pay/pool_key` by copying files, never on a command line. `SPARKPIPE_FLEET_UPSTREAMS` maps each Fleet to its hub's tunnel (`F1B=http://127.0.0.1:18081/v1`), and a Fleet it does not list uses `SPARKPIPE_POOL_UPSTREAM`; `SPARKPIPE_PAY_HEALTH_SECONDS` (30) is how often the service probes each hub for the uptime records. Make that upstream key on the hub as a gateway key, `python3 api_keys.py --file api_keys.json create --gateway --note "VPS gateway pool key"` in `~/sparkpipe-chat` (`docs/LITELLM_FRONTEND.md`, "Public API keys"): only a gateway key's `X-SparkPipe-Account` makes each customer its own account on the hub, and a pool key made without `--gateway` would put every customer's replay store, alias table and fair share in one. The upstream key authenticates both the gateway's requests to `SPARKPIPE_POOL_UPSTREAM` and the hub's calls to the holder schedule. `sparkpipe-pay.conf` proxies `/pay/api/`, `/pay/sandbox/api/` and `/availability.txt`; `mod_proxy_http` must be on. Copy `site/` to `/var/www/html` as usual, including `site/terms/`: the acceptance names `https://sparkpipe.ai/terms/agreement-ko.md` and `agreement-en.md`, and the service hashes the copies under `/var/www/html/terms`. Then:

Both services also need `SPARKPIPE_GATEWAY_CAPACITY`, the number of requests the gateway sustains in 10 minutes. There is no default, and a service without it does not start. It sets the gateway's limit for holders (`docs/FLEET_API.md` §17): the holders of the N Fleets rented at a moment share it, `SPARKPIPE_GATEWAY_CAPACITY / N` each per 10 minutes. Accounts without an active rental keep 6,000. Measure it on the VPS against the sandbox, never production:

1. Start the sandbox with a capacity far above anything it can serve, such as `SPARKPIPE_GATEWAY_CAPACITY=100000000` in `/etc/sparkpipe-pay/sandbox.env`, so that the cap itself does not cut the measurement.
2. In the sandbox, book an Hour on a Fleet that includes now and make a key for that account. Put the key in a file such as `~/sandbox_key` (mode 600), never on a command line.
3. Write a request body: `printf '%s' '{"model":"kimi-k3","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' > /tmp/capacity_body.json`.
4. Run `python3 tools/gateway_capacity_check.py http://127.0.0.1:18091/v1/chat/completions --key-file ~/sandbox_key --body-file /tmp/capacity_body.json --concurrency 64 --seconds 60 --out /tmp/gateway_capacity.json` on the VPS. It refuses any URL that is not loopback, sandbox or staging.
5. It writes the sustained 2xx requests per second, p50 and p99 latency, the statuses, and `capacity_per_window`, which is that rate times 600. It fails on any 429 or error, because a measurement that met a limit is not the gateway's capacity.
6. Raise `--concurrency` until the rate stops growing or p99 passes 1 s. Set `SPARKPIPE_GATEWAY_CAPACITY` in `/etc/sparkpipe-pay/env` to the `capacity_per_window` of the best run that kept p99 under 1 s. The service queues up to 1,024 connections that arrive while it is busy, and the kernel caps that queue at `net.core.somaxconn`: check that it is at least 1,024 on the VPS, or bursts of connections are reset before the service sees them.
7. Set the sandbox back to the same value, then restart both services.

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
  - `credit EMAIL HC NOTE` adds a lot of HC under clause 4.4 and tells the customer by `downtime.credited`, with the note, always by email. The service credits downtime it records by itself; this is for a customer's records that ours do not cover (clause 4.4, second sentence).
  - `uptime FLEET FROM TO` (UTC dates or times) lists a Fleet's down periods in a window from the uptime records.
  - `erase-records EMAIL NOTE` deletes the records of an account's requests on its written request (DPA clause 9.3: within 30 days), with the request as the note, and emails the owner (`docs/FLEET_BOOKINGS.md`, Usage records). It also emails `SPARKPIPE_PAY_NOTIFY` the chat layer's `--erase-records` command, to run on each hub within the same 30 days (`docs/LITELLM_FRONTEND.md`).
  - `swift-confirm REFERENCE USD [NOTE]` and `swift-cancel REFERENCE REASON` are under SWIFT bank transfers above; `refund` takes SWIFT orders too.
  - `maintenance FLEET YYYY-MM-DD [HOURS]` schedules maintenance in the Week starting on that Monday (`docs/FLEET_BOOKINGS.md`).
  - `refund`, `tax-invoices` and `tax-invoice` are under Toss Payments above.
- Part payments settle without anyone. CoinPayments completes an underpaid invoice when the shortfall is within the underpayment limit; with a fixed $100,000 and no percentage, every part payment of an invoice under $100,000 completes with what arrived. The service then issues HC in proportion, mails the customer a receipt that says so, and mails `SPARKPIPE_PAY_NOTIFY` "Paid in part". A part payment CoinPayments does not complete stays `pending` until the payment window closes, when CoinPayments refunds it. Overpayments and late payments are refunded the same way.
- A completed invoice whose total or currency does not match the order (state `mismatch`) issues no HC and mails `SPARKPIPE_PAY_NOTIFY`; check it in the dashboard and use `release`.
- When locked HC lapses, `SPARKPIPE_PAY_NOTIFY` gets "Refund due: locked HC lapsed" with each order and amount: refund a Toss Payments order with `refund ORDER 16.4 AMOUNT REASON`, record a SWIFT order's refund with the same command and send it by SWIFT to the account it came from, and refund a stablecoin order through CoinPayments' refund flow.

## Tests

`python3 tests/test_pay.py` runs the whole service against a fake CoinPayments that verifies request signatures, a fake Toss Payments that checks the secret key and records each call, and a fake mail server: session tokens, codes, sign-in, acceptance of the two Agreement texts and re-acceptance, cross-site refusal, the removed routes, HC orders by quantity (whole or not) and the price ladder at 143 and 144 HC in dollars and won, invoice bodies on the chosen coin and network, signed and tampered webhooks, receipts, part payments, reconciliation, mismatched totals, failed invoices, the ledger chain with two writers and the replay of the live ledger's nine lines; Toss Payments key gating, won amounts fixed on the server, a confirm sent once with its idempotency key, other amounts refused or cancelled, declined and cancelled payments, webhooks that only trigger a check, virtual account deposits and expiry, the refund command (clause 16.4 limited to lapsed locked HC) and the tax invoice list; and USD bank transfers (SWIFT): bank settings that are incomplete or malformed, SWIFT closed without them and in the sandbox, the declaration required, the US dollar ladder at 143 and 144 HC with 0% VAT only on SWIFT, the invoice email with the bank details and a unique reference, `swift-confirm` refusing amounts below the total and confirming once, an overpayment returned under clause 3.5 without debiting HC, a clause 10.4 refund, and `swift-cancel`. `python3 tests/test_market.py` covers the bookings, the uptime records and the downtime credit (`docs/FLEET_BOOKINGS.md`).
