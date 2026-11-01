# Payments

Fleet-hours are paid in Korean won through Toss Payments (토스페이먼츠), or in
US-dollar stablecoins through CoinPayments. HYOR AI Tech Ltd. stays the
seller: both are payment processors that receive the money for us (clause
3.3 of the Customer Agreement). Toss Payments takes Korean and overseas
cards (Visa, Mastercard, JCB), bank transfer, virtual accounts and Korean
easy pay, all in won at the won prices in `site/pricing.js`
(`market.hcPriceKRW`, `plans.terms[].hourPriceKRW`, `plans.lockPremiumKRW`)
plus 10% VAT; it is described under Toss Payments below. The stablecoin
checkout offers USDC on Base, Solana, Arbitrum, Polygon, BNB Chain or
Ethereum, and USDT on Tron, Solana, BNB Chain, Arbitrum, Polygon or
Ethereum. CoinPayments' own currency menu also takes any other stablecoin
enabled in Payment Settings (DAI and TUSD today). Only stablecoins are
enabled, so an invoice priced in dollars is paid in dollars and neither side
carries crypto price risk. Hours follow the US-dollar value CoinPayments
reports for a payment (`nativeActualAmount`), whatever coin it came in, and
the receipt names the coin that was paid (`paymentCurrencySymbol`). USD
wires under a Plan Order are still arranged by hand, and those Plan Orders
are recorded with `sparkpipe_pay.py plan` (below).

## How a customer buys

1. `https://sparkpipe.ai/buy.html` asks three things, one step at a time:
   how much HC (10, 100, 400 or 3,200 USD before VAT, or any amount from
   `market.minPurchase`; `?amount=` presets it); the company's legal name
   and a work email, with a box accepting the Customer Agreement and the
   Hour Market Rules; and how the customer pays: "Card or Korean bank
   transfer" (Toss Payments, in won) or "USDC or USDT" (the coin and
   network). The amount can be a preset or any amount in US dollars or in
   won; won amounts are the same HC count at the won price. The order
   total, with VAT, stays in view. HC is the only hourly product; the market
   that spends it is described in docs/HOUR_MARKET.md.
2. `POST /pay/api/code` emails a 6-digit code. `POST /pay/api/verify`
   checks it, opens the account if the email is new, records the
   acceptance of the Customer Agreement and the Hour Market Rules and sets the session cookie. The email is
   the account: there is no password and no wallet.
3. `POST /pay/api/order` prices the order from `site/pricing.js` (never
   from the browser) and records it. A card or won order goes on as under
   Toss Payments below. A stablecoin order
   creates a CoinPayments invoice in USD (currency `5057`) with the price,
   VAT at 10% as tax, a per-invoice webhook, and `payment.paymentCurrency`
   set to the chosen coin on the chosen network, so the checkout opens
   straight on the address, amount and QR code. The page sends the customer there.
4. CoinPayments posts signed webhooks to `/pay/api/coinpayments`. The
   service checks the HMAC and the timestamp, then fetches the invoice from
   the CoinPayments API and acts on that answer, not on the webhook body.
   HC (or plan hours) is issued only when the invoice is `completed` and
   its total and currency match the order: the whole order when the amount
   received covers the total, or, when CoinPayments completes it for less,
   in proportion to the amount received, HC to six decimals and plan hours
   to four. The customer gets a receipt by email, and
   `SPARKPIPE_PAY_NOTIFY` gets a note of each new account, payment and
   mismatch.
5. CoinPayments sends the customer back to `account.html?order=…`, which
   follows the order (awaiting payment, payment seen, confirmed, issued)
   every 6 seconds and then shows the account: the HC balance, plans and
   orders, with a link to the Hour Market.

Every five minutes the service also asks CoinPayments about each open
invoice from the last 14 days, so a lost webhook only delays the hours.

Every customer is under one contract, the Customer Agreement
(`site/terms/agreement-*.md`). A reserved plan is a Plan Order under it,
which the customer orders on `plans.html`:

1. The page builds the plan from the published choices (term, windows, days,
   fleets, fleet lock, Start Date) and, once the customer is signed in (the
   same email-code sign-in, which opens an account with the company's legal
   name or the person's own name and the acceptance box), asks what the
   Order Form needs: company or individual; for a company, where it is
   organised and its registration number; the address; a notice email; a
   Korean business registration number if any; and the Chosen Models, from
   `models` in `site/pricing.js`, which `tests/test_site.py` keeps equal to the
   models the Fleet List offers.
2. `POST /pay/api/plans/quote` prices it from `plans.terms` (plus
   `lockPremium` for a lock), works out the hours, Tranches and gates,
   checks that we can keep its capacity (below), picks the locked fleets,
   and returns the whole Plan Order: one text, Korean first, then English,
   following the Customer Agreement's Plan Order form, with its SHA-256.
3. The customer reads it and ticks the box. `POST /pay/api/plans/order`
   builds the text again; if anything moved (capacity, the locked fleets,
   a published text or setting) the hash differs and the order is refused
   with reason `changed`. Otherwise the service signs the hash with its
   Ed25519 key and records the Plan Order in the ledger with the text, the
   hash, the IP address, the browser and the signature: acceptance by both
   parties in the account under clause 12.4.
4. The customer is emailed the Plan Order with its hash and our signature,
   `SPARKPIPE_PAY_NOTIFY` gets a note, and `plan-order.html?plan=…` shows
   it, ready to print, with a download of the exact text.

The account page lists each plan with what is paid, what is left and the
next payment, links its Plan Order, and takes prepayments by card or
Korean bank transfer in won (when the Plan Order states a won Hour Price)
or in USDC or USDT. A plan's standing is counted in Plan hours: each
payment counts its amount before VAT divided by the Hour Price in its own
currency, exactly, so payments in won and in dollars add up, and the
amounts due are shown in both currencies, rounded up to the cent or the
won (`remaining`, `remainingKRW`, `nextGate`, `nextGateKRW`).
Each payment issues its hours to the earliest Tranches not yet paid as
Tranche hours the customer reserves in the Hour Market (`tranche_grant`
lines). Plan hours that occurred before they were paid for count as
consumed (clause 20.1), so a payment first covers the hours of its Tranche
that have already passed and issues the rest. An upfront plan's Tranche
hours are issued once it is paid in full. For more than `plans.maxFleets`
fleets, or a plan paid by USD wire, `plans.html` still sends a quote
request to `POST /pay/api/lead`.

## Endpoints

| Route | Does |
| --- | --- |
| `POST /pay/api/code` | `email`: emails a 6-digit sign-in code, valid 10 minutes for 5 tries |
| `POST /pay/api/verify` | `email`, `code`, and for a new account `company` and `accept: true`: signs in, sets the cookie `sp_session` and returns the account |
| `POST /pay/api/signout` | Clears the cookie |
| `GET /pay/api/account` | The signed-in account: HC balance, orders, plans and whether it accepted the current Customer Agreement and Hour Market Rules |
| `POST /pay/api/order` | Signed in. `product: "credits"` with `amount` before VAT (and `accept: true` when the terms changed since the account accepted them), or `product: "plan"` with `plan` and `amount` before VAT; then either `coin`, one of the twelve CoinPayments symbols in `COINS` such as `USDC.BASE` or `USDT.TRC20`, with `amount` in US dollars, which returns the checkout link; or `processor: "toss"` with `amount` in whole won and an optional `brn`, which returns the order and the Toss Payments request for the page (`toss`: client key, `orderId`, `orderName`, `amount` with VAT, `customerEmail`, `customerName`, `successUrl`, `failUrl`). 503 `cards_closed` without Toss Payments keys |
| `GET /pay/api/toss/config` | Whether cards are offered (`enabled`), the client key, the payment UI's `variantKey` and whether it is test mode |
| `GET /pay/api/toss/success/<order>` | Toss Payments' success redirect: checks the amount, confirms the payment and sends the browser to `account.html?order=…` |
| `GET /pay/api/toss/fail/<order>` | Toss Payments' fail redirect: closes the unpaid order and sends the browser to `account.html?order=…` |
| `POST /pay/api/toss/webhook` | Toss Payments webhooks (`PAYMENT_STATUS_CHANGED`, `DEPOSIT_CALLBACK`) |
| `GET /pay/api/order/<order>` | The order's state, amounts and hours; the checkout link while unpaid. The order number is 80 random bits |
| `POST /pay/api/lead` | A plan quote request: `company`, `email`, `name`, `fleets`, `start`, `message` |
| `POST /pay/api/plans/quote` | Signed in. `start`, `months`, `windows` (ids), `days`, `fleets`, `lock`, `models` (ids) and `customer` (`kind` `company` or `individual`, `address`, optional `notice` and `brn`; a company also `jurisdiction` and `registration`): returns the Plan Order's number, text, rows, SHA-256, hours, prices and schedule, or 409 `capacity` |
| `POST /pay/api/plans/order` | Signed in. The same fields with the quoted `plan` and `hash` and `agree: true` (and `accept: true` when the Customer Agreement changed since the account accepted it): records and signs the Plan Order. 409 `changed` when the text no longer matches, `unpaid` while another of the account's Plan Orders has no payment |
| `GET /pay/api/plan/<plan>` | Signed in, the account's own: the plan's standing, its Plan Order rows and text, the acceptance time, our signature and its Tranches |
| `GET /pay/api/plans/key` | Our Ed25519 public key for Plan Orders |
| `POST /pay/api/coinpayments` | CoinPayments webhooks |
| `GET /pay/api/health` | `{"ok": true}` |

The session cookie is HttpOnly, SameSite=Lax, Secure and scoped to
`/pay/api`. It holds the account number, its expiry and an HMAC under
`session_key` in the data directory, made on first start. A POST whose
`Origin` is not `SPARKPIPE_PAY_PUBLIC` is refused. Each client IP address
may ask for 20 codes, 30 verifications, 10 orders and 5 quote requests in
ten minutes (quote requests: an hour), and for 60 Plan Order quotes and
orders an hour, and each email for 5 codes an hour.

## Rules the service enforces

- An account opens only with the box ticked; the ledger keeps the
  statement accepted, the SHA-256 hashes of `terms/agreement-ko.md`,
  `terms/agreement-en.md`, `terms/market-rules-ko.md` and
  `terms/market-rules-en.md`, the IP address and the browser.
  When any of them changes, the next HC order asks the customer to accept
  the new texts.
- HC is bought from `market.minPurchase` to 100,000 USD at a time and each
  lot lasts `market.validityMonths` (12) months from issue; there is no
  holding limit.

- A Plan Order is taken online only when `SPARKPIPE_PAY_SUPPLIER_ADDRESS_KO`
  and `SPARKPIPE_PAY_SUPPLIER_ADDRESS_EN` hold our full notice address,
  which every Plan Order states, and never in the sandbox. Its Start Date
  is from `plans.earliestStart` (or today) to `plans.startWithinDays` (60)
  days ahead, its fleets at most `plans.maxFleets`, and a fleet lock only on
  a full day every day. The published window times, release fee,
  maintenance allowance and fleet Rates are copied into it and kept for its
  term.
- Clause 16.2: a Plan Order is made only if its set-aside fits on serving
  fleets in every hour of its term with every earlier plan and every
  reservation (`Cover.clash`), and only if the HC already sold, and the HC
  in unpaid orders, stays covered with it (docs/HOUR_MARKET.md). A fleet
  lock takes the first serving fleets that pass both checks.
- An account may hold one Plan Order with no payment at a time. An unpaid
  Plan Order keeps its capacity until it lapses: a gated plan at the end of
  Phase A, an upfront plan at the end of its Start Date. After a missed
  gate, a plan keeps capacity only to the end of its last paid Tranche.
- A plan prepayment is at least 1.00 USD and at most what is left of the
  term's price, counting unpaid orders. A gated plan's prepayment is
  refused after Phase A with nothing paid, after a missed gate, and after
  the term ends; an upfront plan's after its Start Date unless it is paid
  in full. Each payment issues the amount before VAT divided by the hour
  price, to four decimals.
- VAT is charged on every payment for HC or a plan. A card payment's card
  slip is its VAT document; a bank transfer, a virtual account payment or
  an easy-pay payment from a bank account or balance gets an electronic tax
  invoice (세금계산서) to the business registration number given (below).
  CoinPayments' and Toss Payments' own fees are ours; the customer pays the
  network fee, and a card issued abroad pays its own conversion fees.
- A won HC order is from `market.minPurchase` × `hcPriceKRW` / `hcPrice`
  (13,000 KRW) to 100,000 USD's worth (130,000,000 KRW), in whole won; its
  HC is the amount before VAT divided by `hcPriceKRW`, to six decimals. A
  won plan prepayment is at least 1,000 KRW and at most what is left in won.

## Data on the VPS

`/var/lib/sparkpipe-pay`, owned by the service user:

- `ledger.jsonl`: one JSON line per event (`account`, `acceptance`,
  `order`, `invoice`, `invoice_failed`, `invoice_state`, `toss_payment`,
  `toss_state`, `payment`, `hours_issued`, `refund`, `tax_invoice`,
  `plan_order`, `plan_end`, and the Hour Market's lines),
  each with the SHA-256 of the line before it. The service
  refuses to start if the chain is broken. An account or order line keeps
  the IP address and the browser, as the privacy policy's acceptance
  records say; a payment line keeps the coin CoinPayments reports, or, for
  Toss Payments, the payment key, the method, the card-slip link and
  `taxInvoice` (`needed`, `brn`, `supply`, `vat`).
- `session_key`: 32 random bytes that sign the session cookies. Deleting
  it signs everyone out.
- `signing_key`: the 32-byte Ed25519 private key that signs Plan Orders,
  made on first use. Back it up with the ledger: Plan Orders already
  signed stay verifiable with the public key they record, but a new key
  signs new ones.
- `leads.jsonl`: the plan quote requests.

A Plan Order line (`plan_order`) holds its number (`plan`), `account`,
`email`, `company`, `start`, `months`, `windows` (with their KST hours as
published when it was made), `days`, `fleets`, `lock` (fleet ids, one per
fleet, only on a full day every day), `hourPrice` (lock premium included), `hourPriceKRW` (the won Hour Price,
lock premium included; optional on a signed Plan Order, which without it
is paid in US dollars only), `payment` (`upfront` or `gates`), `releaseFee` and
`maintenanceHoursPerWeek` as published then, and, for one made in the
account, `customer`, `models`, `threshold`, `currency`, `rates`, `hours`,
`total`, `schedule`, `agreement` (the hashes and version of the texts it
is made under), `supplier`, `text`, `hash`, `accepted` (IP address and
browser) and `signature`. A plan's hours are each elected window's hours
times `fleets` on every plan day, by calendar-month Tranche.

A Plan Order signed on paper is recorded with
`sparkpipe_pay.py plan FILE.json`, a JSON object or list with `id`, the
account's `email` and the fields above from `start` to
`maintenanceHoursPerWeek`:

```json
[{"id": "SP-PLAN-7K3QD9", "email": "billing@acme.example", "start": "2026-12-15", "months": 24, "windows": [{"id": "asia", "from": 8, "to": 16, "localDay": 0}], "days": "every", "fleets": 1, "hourPrice": "15.00", "payment": "gates", "releaseFee": "0.10", "maintenanceHoursPerWeek": 3}]
```

It refuses an unknown email, a number already used and a plan whose
capacity we cannot keep (clause 16.2); `sparkpipe_pay.py cover FILE.json`
shows the cover with the file's plans added, without recording them.
`sparkpipe_pay.py end-plan PLAN YYYY-MM-DD REASON` records that a Plan
Order holds no capacity from that day, such as one we end under clause
10.2.

## Mail

The service sends the sign-in codes, receipts and notes through
`SPARKPIPE_PAY_SMTP` (default `127.0.0.1:25`, the VPS's own mail server
in `docs/VPS_MAIL.md`, which the SPF record for sparkpipe.ai already names) as
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

## Toss Payments

The checkout uses Toss Payments' checkout-page payment (주문서형 결제,
formerly the Payment Widget, SDK v2, `https://js.tosspayments.com/v2/standard`).
Only `buy.html` and `account.html` load the SDK, and only those two pages
get a Content-Security-Policy that admits `*.tosspayments.com` and
`*.toss.im` (`deployment/vps/sparkpipe-ai-ssl.conf`; `tests/test_site.py`
holds both to the payment pages).

1. The page asks `GET /pay/api/toss/config`. Without keys (`enabled:
   false`) it hides "Card or Korean bank transfer" and offers only USDC or
   USDT. Otherwise it renders Toss Payments' payment methods and terms with
   the won total, VAT included.
2. On Pay, `POST /pay/api/order` with `processor: "toss"` prices the order
   in won on the server, records it with `processor`, `currency: "KRW"`,
   `brn` and an expiry 45 minutes ahead (Toss Payments' 30-minute payment
   window and its 10 minutes to confirm), and returns the request. The
   page sets that amount and calls `requestPayment`.
3. Toss Payments redirects to `/pay/api/toss/success/<order>?paymentKey&orderId&amount`.
   The service refuses an `amount` other than the order's (state `failed`,
   reason `amount`, nothing confirmed), records the payment key
   (`toss_payment`), and calls `POST /v1/payments/confirm` with the order's
   own amount and `Idempotency-Key: confirm-<order>`. A refused confirm is
   followed by `GET /v1/payments/{paymentKey}` before anything is
   recorded.
4. Every Toss Payments answer goes through one check: the payment must name
   this order and the stored payment key. `DONE` with the order's amount in
   KRW issues the hours once (`payment`, `hours_issued`); `DONE` or
   `WAITING_FOR_DEPOSIT` for another amount or currency is cancelled
   through the cancel API (state `cancelled`, reason `amount`, or
   `mismatch` if the cancel fails, with a note to `SPARKPIPE_PAY_NOTIFY`);
   `WAITING_FOR_DEPOSIT` records the virtual account (bank, number,
   deadline) as state `waiting` and emails it to the customer;
   `ABORTED`, `EXPIRED` and `CANCELED` close the order.
5. `POST /pay/api/toss/webhook` acknowledges every event and only queues a
   check of the order it names: the service fetches the payment from Toss
   Payments with the payment key it stored and applies that answer, never
   the webhook body. The reconciliation every five minutes checks every
   open Toss Payments order the same way, so a lost webhook only delays
   the hours.
6. A card order not completed within 45 minutes expires (`timedout`). A
   virtual account expires at the earlier of Toss Payments' deposit
   deadline and 24 hours after the order: the reconciliation cancels it at
   Toss Payments first, and issues the hours instead if the deposit
   arrived. An open order holds its HC out of the HC on sale, and its plan
   amount out of what is left to pay, until it expires.

The receipt names the method; for a card it links the card slip, which is
the VAT document. A payment made by bank transfer, into a virtual account
or by easy pay from a bank account or balance is recorded with
`taxInvoice.needed`, with the business registration number the checkout
took (`brn`; for a plan, the Plan Order's when none is given). Issuing the
electronic tax invoice stays a manual step:

- `sparkpipe_pay.py tax-invoices` lists every such payment without an
  invoice recorded: order, date (KST), company, email, registration number
  (or "missing: ask the customer"), supply amount, VAT, total, method, and
  any refund that needs an amended invoice.
- `sparkpipe_pay.py tax-invoice ORDER NUMBER` records the National Tax
  Service approval number of the invoice issued (`tax_invoice`), which takes
  the payment off the list.

Refunds happen only where the Customer Agreement allows them (clauses 3.2,
3.5, 10.4, 20.2, 24.3 or where the law requires), and go back by the method
of the payment (clause 3.6):

```sh
sparkpipe_pay.py refund ORDER CLAUSE KRW|all REASON [BANK ACCOUNT HOLDER]
```

It fetches the payment, refuses a clause outside that list, an amount above
what is left at Toss Payments, and, for an HC order, an amount whose HC
(the same share of the order's HC) is no longer unspent in the order's
lot (a refund of VAT not owed, clause 3.2, leaves the HC). It calls
`POST /v1/payments/{paymentKey}/cancel` with `cancelAmount` for part of the
balance, `refundableAmount` as a guard and `Idempotency-Key:
refund-<order>-<balance>`; a virtual account payment needs the bank code,
account number and holder for `refundReceiveAccount`. It checks the
balance Toss Payments reports, then writes a `refund` line (clause, reason,
amount, balance left, HC debited) and an `hc_debit` from the order's own
lot, and emails the customer. A plan refund changes no Plan hours: end the
plan with `end-plan` as its clause requires.

### Setup in the Toss Payments console

1. Sign the electronic payment contract and add the methods: domestic
   cards, overseas cards in KRW (a separate contract and card-company review
   of about two weeks), bank transfer, virtual accounts and the easy-pay
   wallets. In the payment admin (결제 어드민), build the payment UI; keep
   its `variantKey` `DEFAULT` or put it in `TOSS_VARIANT_KEY`.
2. In the virtual account settings, set the deposit period to one day or
   less and turn on the refund-account form. Settle with the accountant
   whether to turn off automatic cash receipts (현금영수증) for bank
   transfers and virtual accounts: a payment that gets an electronic tax
   invoice must not also get a cash receipt.
3. Register the webhook `https://sparkpipe.ai/pay/api/toss/webhook` for
   `PAYMENT_STATUS_CHANGED` and `DEPOSIT_CALLBACK` on the live MID, and,
   once the merchant's own test keys replace the documentation keys in the
   sandbox, `https://sparkpipe.ai/pay/sandbox/api/toss/webhook` in the test
   environment.
4. Copy the live checkout-page keys (`live_gck_…`, `live_gsk_…`) into
   `/etc/sparkpipe-pay/env` as `TOSS_CLIENT_KEY` and `TOSS_SECRET_KEY`,
   by editing the file, never on a command line, and restart the service.
   The live service refuses test keys and the sandbox refuses live keys; a
   key without the other stops the service. Under API key access policy,
   allow only the VPS's address.

## CoinPayments setup

1. In the CoinPayments dashboard, create an API integration and keep its
   client ID and client secret. Webhooks need no dashboard setting: every
   invoice names `https://sparkpipe.ai/pay/api/coinpayments` and the events
   `invoicePending`, `invoicePaid`, `invoiceCompleted`, `invoiceCancelled`
   and `invoiceTimedOut`. The webhook signature covers that exact URL, so
   it must not change behind the proxy.
2. Every invoice pre-selects the customer's coin and network, but the
   checkout still offers a menu of every coin enabled in Payment Settings.
   Keep `USDC.ERC20`, `USDC.BASE`, `USDC.ARB`, `USDC.POL`, `USDC.SOL`,
   `USDC.BEP20`, `USDT.TRC20`, `USDT.ERC20`, `USDT.BEP20`, `USDT.SOL`,
   `USDT.POL` and `USDT.ARB` on. Other US-dollar stablecoins may stay on
   (the Customer Agreement takes any the processor accepts). Turn
   every other coin off, LTCT included, so no payment arrives in a coin whose
   price moves. Set the underpayment limit
   to a fixed $100,000 with no percentage (see Operating it). Payment Settings lists
   every currency; open each one that is not a stablecoin and disable it.
   A disabled currency is no longer offered at checkout.
   `GET /api/v1/merchant/currencies` lists every currency, enabled or not;
   one is on when its `switcherStatus` is `enabled`.
3. Choose the payout wallet or conversion there too; the service does not
   set payout overrides.

CoinPayments emails the payment instructions to the invoice's refund
email, which is the account email.

## Install on the VPS

```sh
sudo apt-get install -y python3-cryptography
sudo useradd --system --home /var/lib/sparkpipe-pay --shell /usr/sbin/nologin sparkpipe-pay
sudo install -d -o sparkpipe-pay -g sparkpipe-pay -m 700 /var/lib/sparkpipe-pay
sudo install -d -m 755 /opt/sparkpipe-pay /etc/sparkpipe-pay
sudo install -m 644 tools/sparkpipe_pay.py tools/sparkpipe_market.py /opt/sparkpipe-pay/
sudo install -m 640 -g sparkpipe-pay deployment/vps/sparkpipe-pay.env.example /etc/sparkpipe-pay/env
sudo install -m 640 -g sparkpipe-pay /dev/null /etc/sparkpipe-pay/client_secret
sudo install -m 644 deployment/vps/sparkpipe-pay.service /etc/systemd/system/
sudo install -m 644 deployment/vps/sparkpipe-pay.conf /etc/apache2/sparkpipe/
sudo install -m 644 deployment/vps/sparkpipe-ai-ssl.conf /etc/apache2/sites-available/000-default-le-ssl.conf
sudo install -m 644 deployment/vps/rspamd-dkim_signing.conf /etc/rspamd/local.d/dkim_signing.conf
```

Fill `COINPAYMENTS_CLIENT_ID` in `/etc/sparkpipe-pay/env` and put the
client secret into `/etc/sparkpipe-pay/client_secret` by copying a file,
never on a command line. Put our full registered address, in Korean and in
English, in `SPARKPIPE_PAY_SUPPLIER_ADDRESS_KO` and
`SPARKPIPE_PAY_SUPPLIER_ADDRESS_EN`: every Plan Order states it as our
notice address, and online Plan Orders stay closed until both are set. `sparkpipe-ai-ssl.conf` is sparkpipe.ai's port-443
virtual host and includes `sparkpipe-pay.conf` beside the other
`deployment/vps` includes; `mod_proxy_http` must be on. Copy `site/` to `/var/www/html` as usual, including
`site/terms/`: the acceptance names `https://sparkpipe.ai/terms/agreement-ko.md`,
`agreement-en.md` and the two `market-rules` texts, and the service hashes the copies under
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
on the host alone except the mail address lists, passwords and keys that
`docs/VPS_MAIL.md` keeps out of the public repository.

```sh
git fetch origin main
git archive --format=tar.gz -o pay-main.tgz origin/main site tools/sparkpipe_pay.py tools/sparkpipe_market.py deployment/vps
```

On the VPS, back up `/var/www/html`, `/opt/sparkpipe-pay/*.py`, the ledgers
and any configuration file being replaced, copy `site/.` over
`/var/www/html`, install the service and changed configuration as in the
install steps, check the configuration, then restart `sparkpipe-pay` and
`sparkpipe-pay-sandbox` (docs/HOUR_MARKET.md) and reload Apache or rspamd. The live pages, the service and the configuration
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
  $100,000 completes with what arrived. The service then issues HC (or
  plan hours) in proportion to what arrived, mails the customer a
  receipt that says so, and mails `SPARKPIPE_PAY_NOTIFY` "Paid in part".
  HC is bought by amount, so a customer who wants more simply buys again. A part payment
  CoinPayments does not complete stays `pending` until the payment window
  closes, when CoinPayments refunds it and emails the customer a claim
  link. Overpayments and late payments are refunded the same way: once a
  part payment completes its invoice, money sent again to that address is
  an overpayment, so more HC is always a new order.
- A completed invoice whose total or currency does not match the order
  (state `mismatch`) issues no hours and mails `SPARKPIPE_PAY_NOTIFY`. Check the invoice in the dashboard; if it is
  right, `sparkpipe_pay.py release <order>` fetches the invoice again and,
  once CoinPayments says `completed`, issues the hours the amount received
  covers: HC at the order's price for an HC order, the amount before VAT
  divided by the hour price for a plan prepayment. It appends to the same ledger
  safely while the service runs.
- A customer whose payment fell short or arrived late is refunded through
  CoinPayments' own refund flow; the ledger records no hours for it.

## Tests

`python3 tests/test_pay.py` checks plan hours and gates for every window,
day choice, term and fleet count against the Customer Agreement's examples, the session tokens, and the whole service against a
fake CoinPayments that verifies request signatures and a fake mail server:
codes, sign-in, terms acceptance and re-acceptance, cross-site refusal,
refusal of coins that are not USDC or USDT, invoice bodies on the chosen
coin and network, signed and tampered webhooks,
receipts, part payments, reconciliation, gates, mismatched totals,
failed invoices, quote requests and the ledger chain with two writers. A
fake Toss Payments server checks the secret key and records each call:
key gating on the live service and in the sandbox, won amounts fixed on
the server, a confirm sent once with its idempotency key, other amounts
refused or cancelled, declined and cancelled payments, webhooks that only
trigger a check (a forged `DONE` issues nothing), virtual account deposits
and expiry, plan payments in won, the refund command and the tax invoice
list.
