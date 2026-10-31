# SparkPipe Customer Agreement

## Parties

This Agreement is made between **HYOR AI Tech Ltd.** (효르에이아이테크 유한회사), a company organised under the laws of the Republic of Korea, business registration number 641-87-03809, of Yongsan-gu, Seoul, Republic of Korea, whose full address we give on request and in each Plan Order ("**we**"), and the company, professional or independent developer identified in the account that accepts it ("**Customer**"). A Plan Order identifies Customer by its legal name and address, and a company also by its jurisdiction of incorporation and registration number.

## Part 1. General

## 1. Definitions and documents

1.1. In this Agreement:

- **Fleet** means a group of computing machines that we operate as one unit and list on our **Fleet List**, published at sparkpipe.ai/fleets.html.
- **Fleet-hour** or **hour** means the use of one entire Fleet for sixty minutes.
- **Rate** means the number that the Fleet List gives each Fleet, 1 unless the Fleet List states otherwise: the HC that one hour of our capacity on that Fleet costs up to 7 days ahead (clause 14.1), and the Plan hours that one reserved hour on it consumes (clause 22.1).
- **Reservation** (or **dated hour**) means the right to use one named Fleet for the sixty minutes from a named start on the hour. To **reserve** is to obtain one.
- To **release** a Reservation is to offer it for resale in the **Hour Market** under the Hour Market Rules.
- **Hour-credit** (**HC**) means the prepaid unit in which Customer pays for hours and for the shared pool under Part 2.
- **Plan** means the reserved plan that a Plan Order creates under Part 3, and **Plan hours** means the hours it holds (clause 17.3).
- **Hour Market Rules** means our SparkPipe Hour Market Rules, published at sparkpipe.ai/market-rules.html.
- **KST** means Korea Standard Time (UTC+9). All times and days in this Agreement and in each Order are in KST.

1.2. **Orders.** Customer buys under this Agreement by Orders. An **Order** is either (a) a purchase of HC made through our website under Part 2, or (b) a **Plan Order**: a one-page order form in the form set out at the end of this Agreement, stating the fields listed in clause 16.1, signed by both parties or accepted by both in the Account (clause 12.4).

1.3. **Precedence.** A Plan Order prevails over this Agreement for its own fields, and the Hour Market Rules prevail over it on the matters listed in clause 1.4; otherwise this Agreement prevails. A Plan Order does not vary clause 5, 6, 8 or 12 unless it says so expressly.

1.4. **Hour Market Rules.** The Hour Market Rules form part of this Agreement. They govern the release of Reservations, bids, the matching and settlement of trades, the price of the shared pool and its use of released hours, and the release fee.

## 2. Professional use and account

2.1. **Professional use.** Customer confirms that it is purchasing and using hours, hour-credits and the shared pool for business, professional, commercial, or independent research and development purposes, and not as a retail household consumer, and that its account details are true and current. We may refuse or close an account that does not meet this.

2.2. Customer's **Account** is its account on our website, identified by the email address Customer verifies with us and signed in to with one-time codes we send to that address. Acceptance of this Agreement, purchases of HC, Reservations and changes to the Account are made while signed in to it, or through our API with an access token issued under it. Customer may issue access tokens and add users under the Account. Access is automated, with no notice to us, approval or waiting period. We set the technical mechanism and may change it without amending this Agreement, provided equivalent access continues.

2.3. Customer bears all risk of loss or compromise of its Account email and access tokens; any use under them is Customer's use. Customer may name a recovery email address, which can replace the Account email after a 72-hour delay with notice to the Account email.

## 3. Prices, VAT and payment

3.1. **Prices.** HC is priced at the HC Price (clause 13.1), and Plan hours at the Hour Price stated in the Plan Order (clause 16.4). Prices are exclusive of value-added tax (VAT).

3.2. **VAT.** We charge VAT at the statutory rate (10% today) on every payment and show it separately on our invoice, except a payment that is zero-rated under clause 3.3(b). Any amount charged as VAT that is not owed is an overpayment, and we refund it to Customer. If a payment made as zero-rated is held subject to VAT because Customer's information was incorrect or Customer did not pay in the way the Order states, Customer pays us that VAT, with any additional tax and penalty, on demand.

3.3. **Methods.** Customer pays in advance: no HC and no Plan hour is issued before it is paid for. Customer pays in one of the following ways, as our website offers it for a purchase of HC or as the Plan Order states:

- (a) in Korean won, where Customer has a Korean business registration number, by bank transfer from Customer's account at a Korean bank to our won account named on the electronic tax invoice (세금계산서) we issue;
- (b) in US dollars, by bank transfer from Customer's own account outside the Republic of Korea to our US dollar account at a Korean foreign exchange bank, whose details both parties confirm before payment; such a payment is zero-rated where Customer has no place of business in the Republic of Korea and the law allows; or
- (c) in a US dollar stablecoin, such as USDC or USDT, that our payment processor CoinPayments accepts, on any network it accepts for that coin, through the invoice it issues for us.

3.4. Customer pays every amount in full, without set-off, deduction or withholding. If a law requires Customer to withhold or deduct any tax from a payment, Customer pays an additional amount so that we receive what we would have received without the withholding or deduction. Customer bears its own and any intermediary bank, transfer, network and currency conversion charges, so that the full amount invoiced is paid. We bear our payment processor's own fees.

3.5. **Cleared payment.** A payment counts once we have verified it: a bank transfer when it reaches our account, and a stablecoin payment when our payment processor reports it completed. If the processor completes a stablecoin payment for less than the invoice total, the payment counts for the amount received, as the processor values it in US dollars, and the HC or Plan hours it issues are reduced in proportion. A payment the processor does not complete, an amount above the invoice total, and a payment that arrives after the processor's payment window are returned through the processor's refund procedure.

## 4. Fleets, Reservations and maintenance

4.1. We provide a service, not hardware. We keep ownership, possession and operation of every Fleet. Customer reserves on the Fleets we operate at the time, current or future, subject to availability, except as clause 19 provides for a locked Fleet.

4.2. **Reserving.** Customer reserves hours through our website while signed in to its Account, or through our API with an access token issued under it, naming the Fleet, the hours and the model and paying in HC under Part 2 or with Plan hours under Part 3, or by a bid under the Hour Market Rules. A Reservation exists once our system confirms it and debits its price, and is then final: it cannot be cancelled, moved or refunded, except under clause 24.3, but Customer may release it under the Hour Market Rules before it starts.

4.3. **Consumption.** A reserved hour is consumed when it occurs, whether or not it is used, and not before. Requests still running at its end have 30 seconds to finish. Loading the reserved model counts as part of the hour.

4.4. **No displacement.** A confirmed Reservation is never taken, moved or displaced, and is delivered on the Fleet it was made on, except on a suspension under clause 5.3, when a prearranged trade is voided under clause 8 of the Hour Market Rules, where maintenance overlaps it under clause 4.6, or where a locked Fleet cannot serve under clause 19.4.

4.5. **Availability.** We publish live capacity in machine-readable form at https://sparkpipe.ai/availability.txt, regenerated at least every 30 seconds: each Fleet we operate, whether it is serving, and the hours open for Reservation. Wherever this Agreement makes anything subject to availability, availability means what that publication shows when Customer reserves.

4.6. **Maintenance.** We may take each Fleet out of service for maintenance for up to the maintenance allowance we publish (3 hours in each calendar week, Monday to Sunday, today). We announce each maintenance period at least 7 days ahead, choose the times with the fewest Reservations, and rotate those times between the windows (clause 17.1). Hours under announced maintenance cannot be reserved. A Reservation that maintenance overlaps is credited under clause 4.7 as if its Fleet were unavailable throughout it.

4.7. **Automatic credit.** If a reserved hour is unavailable, or its Fleet is unreachable, for more than 15 consecutive minutes, every reserved hour in which any part of that period falls is credited back in full within 24 hours after the period ends, from our own records, to the account that holds the Reservation: as HC equal to the price it paid, for a Reservation paid in HC; and as Plan hours under clause 23.3, for a Reservation made with Plan hours. Where Customer's records show such a period and ours do not, Customer may send them within 7 days, and we credit the hours unless our records show the Fleet was available throughout.

4.8. Credits are made in HC or Plan hours, never in cash, and are Customer's only remedy for any failure of performance, subject to clause 8.5.

## 5. Use and responsibility

5.1. **We provide capacity only.** Customer has the exclusive right to use each hour it has reserved. Customer alone decides what runs on its hours and in its requests to the shared pool, and is solely responsible for its workloads, the data and content it processes, the outputs it obtains and every use made of them. We do not monitor, review or control them and are not responsible for them. Customer is responsible, as for its own acts, for everyone who uses hours or the shared pool under its Account, access tokens or API keys, including its end users and anyone it resells, transfers or gives hours or access to, and must bind those persons by terms no less protective of us than clauses 5 and 6. The Hour Market is the only resale of hours in which we take part. We keep no register of, and are not a party to, any other resale or transfer of hours: it happens outside our system, we do not match, price, clear or settle it, and we have no liability for its price or counterparties.

5.2. **Lawful use.** Customer must comply, and must ensure that every user of its hours and of its access to the shared pool complies, with every law that applies to the use, including laws on export control and sanctions, personal data, intellectual property and computer misuse, and with the licence of each model it uses. No hour, and no request to the shared pool, may be used to: (a) create, store or distribute unlawful content, including child sexual abuse material; (b) attack, probe or disrupt any system or network, or distribute malicious code; (c) infringe the rights of any person; (d) benefit any person, entity or country subject to sanctions of the United Nations, the Republic of Korea, the United States or the European Union, or serve any end use that export control laws prohibit, including weapons of mass destruction; or (e) interfere with a Fleet or with other customers. Customer represents that neither it nor any person that owns or controls it is subject to those sanctions. Where Customer uses its hours or the shared pool to provide generative AI products or services to others, Customer is their provider and bears the duties that apply to them, including the transparency and labelling duties of the Korean Framework Act on the Development of Artificial Intelligence and Establishment of Trust.

5.3. **Suspension.** We may suspend a Reservation or access under the Account, at once and without prior notice, where (a) a law, court or competent authority requires it, (b) we reasonably believe clause 5.2 is being breached, or (c) the use threatens the security or integrity of a Fleet or of others. We tell Customer as soon as practicable and lift the suspension when its cause ends. Hours lost to a suspension caused by the use of Customer's hours are consumed and are not credited.

## 6. Customer data

6.1. Customer keeps all rights in its data and outputs. We access them only as needed to operate and secure the service, or where the law requires. Customer is responsible for the lawfulness of any personal data it processes and for keeping its own copies: we keep no backups of Customer data. The Fleets and our servers are in Seoul, Republic of Korea.

6.2. We hold no Customer data beyond transient caches, which we delete within 24 hours, and the model state and caches that clause 19.3 keeps on a locked Fleet. Nothing is therefore returned when an Order or this Agreement ends.

6.3. We tell Customer without delay of any intrusion, any leak of its information and any unannounced interruption of the service. We handle account data and Customer data as our published privacy policy states.

## 7. Models

7.1. We offer only open-source models whose availability cannot be withdrawn by any third party, and list the models available on each Fleet on the Fleet List. A model chosen for a confirmed Reservation stays available for that Reservation. Other models are available while they are listed, except that the Chosen Models stay available as clause 7.2 provides.

7.2. **Chosen Models.** The models Customer chooses from the Fleet List and names in a Plan Order (the **Chosen Models**) remain available to Customer for the whole Term of that Plan, on at least one Fleet and, under a Fleet lock, on each locked Fleet.

7.3. **Model licences and outputs.** Each model is third-party open-source software provided under its own licence, which Customer must comply with. Outputs are generated automatically from Customer's inputs: we do not author, review or endorse them, and we give no warranty that any output is accurate, complete, lawful, safe or fit for any purpose. Customer is responsible for evaluating outputs before relying on, publishing or acting on them.

## 8. Liability

8.1. Our aggregate liability for all causes combined is capped: (a) for claims arising under or in connection with a Plan Order, at the amounts we received from Customer under that Plan Order in the 6 months before the event giving rise to the claim; and (b) for all other claims, at the amounts we received from Customer for HC in the 6 months before that event.

8.2. We are not liable for lost profit, lost data, lost opportunity, business interruption, or indirect or consequential loss, whether or not foreseeable.

8.3. Customer indemnifies us, our officers and our employees against any claim, loss, liability, fine, penalty or cost, including reasonable legal fees, arising from (a) the use of its hours, or of the shared pool under its Account, by anyone, including Customer, its end users and anyone it resells, transfers or gives hours or access to, (b) its workloads, data, content and outputs, (c) any breach of clause 5 or 6, or (d) any third-party claim, or investigation by an authority, connected with them. We give no reciprocal indemnity.

8.4. The service and the models are provided "as available". We give no warranty of merchantability, fitness for purpose, accuracy of outputs or uninterrupted service beyond clauses 4.7 and 23.

8.5. Nothing in this Agreement limits or excludes liability for intentional misconduct or gross negligence, or any other liability that cannot lawfully be limited or excluded.

8.6. Clauses 8.1 and 8.2 limit our liability only. They do not limit Customer's payment obligations or its liability under clauses 3.2, 5, 6 and 8.3.

8.7. **Force majeure.** Neither party is liable for a delay or failure caused by an event beyond its reasonable control, including natural disaster, fire, flood, war, terrorism, epidemic, action of a government or authority, or failure of public power or telecommunications networks. Clause 4.7 still applies to reserved hours lost to such an event and is Customer's only remedy for them.

## 9. Changes and published settings

9.1. We may change this Agreement by publishing a new version and emailing Customer. A new version applies to Orders made after Customer accepts it, and we accept no further Order from Customer until it has. HC already bought, and each Plan Order, stay under the version in force when they were made, unless Customer accepts a newer version for them.

9.2. We change the Hour Market Rules in the same way, and each version applies as clause 13.1 of the Hour Market Rules provides.

9.3. **Published settings.** We publish the following on our website as **published settings**: the HC Price, the advance multiplier, the release fee, the window times, the plan price list (the Terms, Hour Prices and payment schedules we offer for Plans, and the price of a Fleet lock) and the maintenance allowance. A change to a published setting applies only to Orders made after we publish it, never to an existing Order. HC, and each Reservation paid with it, keep the settings in force when that HC was bought; where a Reservation is paid with HC from more than one purchase, the settings of the HC spent first apply. A Plan Order keeps, for its whole Term, the settings in force when it was made, and states those that apply to it.

## 10. Ending, refunds and survival

10.1. Customer may close its Account at any time while no Plan Order under it is in force. Its HC and Reservations then lapse, and its orders in the Hour Market are cancelled.

10.2. We may close the Account and end this Agreement, or end a Plan Order, by written notice: (a) if Customer materially breaches this Agreement and does not remedy the breach within 30 days after our written notice; or (b) at once, where a breach of clause 5.2 cannot be remedied, where Customer or a person that owns or controls it becomes subject to the sanctions in clause 5.2(d), or where a law, court or competent authority requires us to stop serving Customer. We may not end a Plan Order because of a missed gate.

10.3. **Right of use only; no refunds.** HC, Reservations and Plan hours give the right to use the service only. They give no ownership of, or rights in, our company, hardware, software, models or intellectual property, are not a deposit, loan, security or investment, carry no expectation of profit from our efforts, and have no cash value. HC and Plan hours cannot be moved to another account or withdrawn. **Every amount paid is non-refundable and non-cancellable**, except under clause 10.4, 20.2 or 24.3 or where the law requires. Customer may not set off, deduct, counterclaim or withhold against any amount. We do not buy back HC or hours.

10.4. If we stop offering service for HC, or close an Account other than for Customer's breach, we refund the purchased HC not yet spent or lapsed at the price paid, within 60 days, counting HC as spent from the oldest lot first. HC credited for a released hour is not refunded.

10.5. Clauses 3.2, 3.4, 5, 6, 7.3, 8, 10, 11, 12, 22.3 and 24.3 survive the end of this Agreement and of each Order.

## 11. General

11.1. **Entire agreement.** This Agreement, with the Hour Market Rules and Customer's Orders, is the whole agreement between the parties. Neither party relies on any pre-contractual statement, projection, benchmark or representation, including anything said in diligence or during Phase A, except a Baseline recorded under clause 23.1.

11.2. **Amendment and waiver.** This Agreement changes only as clause 9 provides. A Plan Order is made, and changed, only in writing signed or accepted by both parties under clause 12.4. No conduct, delay or extension operates as a waiver.

11.3. **Third parties.** This Agreement gives no rights to anyone except the parties.

11.4. **Confidentiality.** Each party keeps confidential the other's non-public information that it receives in connection with a Plan Order, including the Plan Order itself, for 3 years after that Plan ends, and pricing and technical information indefinitely.

11.5. **Publicity.** Customer may not publicise a Plan Order or our relationship under it without our written consent. We may refer to Customer by name and logo once a Plan of Customer's reaches Full Prepayment.

11.6. **Assignment.** Customer may not assign this Agreement or an Order without our written consent. Letting others use reserved hours under clause 5.1 is not an assignment. We may assign this Agreement and any Order freely, including to an acquirer or financier.

11.7. **Notices.** We send notices to the email address verified on the Account, which Customer keeps current. Notices under a Plan Order, including any withdrawal or termination, must be in writing and sent to the notice address or email it states; our notices may also be sent to the Account email.

## 12. Law, disputes, language and acceptance

12.1. This Agreement and each Order are governed by the laws of the Republic of Korea.

12.2. Any dispute arising out of or in connection with this Agreement or an Order is finally settled by arbitration in Seoul under the Arbitration Rules of the Korean Commercial Arbitration Board, before a sole arbitrator, in the Korean language. Either party may apply to a competent court for interim measures.

12.3. This Agreement is published in English and Korean, and each Plan Order is made in both. The Korean text prevails.

12.4. **Acceptance and signature.** Customer accepts this Agreement, which includes the Hour Market Rules, by ticking the acceptance box while signed in to its Account. We record the SHA-256 hashes of the exact Korean and English texts of this Agreement and of the Hour Market Rules accepted, with the time, the IP address and the browser. A Plan Order is signed by both parties, by hand or electronically, or accepted by both in the Account: by Customer accepting the hash of its final text while signed in, and by us signing that hash with our signing key. The record of any negotiation of a Plan Order is kept with it. Each such acceptance or signature has the same effect as a handwritten signature.

## Part 2. Hour-credits

## 13. Hour-credits

13.1. **Buying HC.** The price of one HC is the **HC Price** shown on our website when Customer buys it: in Korean won for payment in won, otherwise in US dollars. It is fixed for the HC bought. Customer buys HC by amount, within the minimum and maximum per purchase shown on our website, and receives the amount paid, net of VAT, divided by the HC Price, rounded down to six decimal places. HC is issued when the payment counts under clause 3.5.

13.2. **Balance.** HC is counted to six decimal places. Each purchase, and each credit under this Agreement or the Hour Market Rules, is added to Customer's balance as a separate lot.

13.3. **Lapse.** Each lot lapses 12 months after it is issued, or on the later date clause 13.4 sets, with any HC left in it. HC is spent from the lot that lapses first. The same applies to HC credited for a released hour or under clause 4.7.

13.4. **No overselling.** We sell HC only while it can be spent. We accept a purchase only if, for the date its lot lapses and for every later date on which other HC lapses, the free hours of our capacity before that date cover all HC not yet spent or lapsed that lapses by it, including the new lot, HC held for bids, and HC in orders not yet paid as if it were issued at once. Free hours are the hours of serving Fleets, on every day of the week, that nobody holds, after the capacity kept for Plans under clauses 18 and 19 and after each Fleet's maintenance allowance (clause 4.6), each counted at its Fleet's Rate; Fleets not yet serving are not counted. We refuse a purchase beyond that cover, and show the HC still on sale on our website. If a payment counts, or we credit HC under clause 4.7, when that cover no longer reaches 12 months for the new lot, the lot lapses instead on the first date that the cover reaches with the lot included.

## 14. Reserving with HC

14.1. With HC, Customer may reserve (a) an hour of our capacity on a serving Fleet, from the first date we offer, up to 7 days ahead at its Rate in HC, and more than 7 and up to 31 days ahead at its Rate times the advance multiplier (1.25 today); or (b) an hour another customer has released, at the price the Hour Market Rules set, however far ahead it starts. Capacity kept for Plans is offered only as clauses 18.2 and 19.2 allow, capacity kept under clause 14.2 only as that clause allows, and hours under announced maintenance not at all (clause 4.6).

14.2. **Earlier lapse first.** Where reserving an hour of our capacity with HC that lapses after a date would leave the free hours before that date short of the HC that lapses by it (clause 13.4), that hour can be reserved, until 6 hours before it starts, only with HC that lapses by that date; from then on, with any HC.

## 15. Shared pool

15.1. The **shared pool** serves individual requests on Fleets that nobody holds in that hour and on released hours that reach their start unsold. Its meter is the **seat-second**, one second of one request slot on a pooled Fleet.

15.2. Customer may send requests to the models we list for the shared pool, through our OpenAI-compatible endpoint at https://sparkpipe.ai/pay/api/pool/v1, with API keys it issues under its Account. API keys are access tokens for the purposes of clause 2, and Customer may revoke one at any time. Each request holds one seat and is charged in HC for its seat-seconds, from when we accept it until its reply ends and whether or not it succeeds, at the price per seat-second posted when it starts (clause 7.2 of the Hour Market Rules), rounded down to six decimal places and never more than Customer's balance. We accept a request only while Customer holds at least 0.0001 HC. The shared pool has no guaranteed capacity: when no Fleet is in it, or its seats are taken, requests are refused. Requests made with Customer's API keys during an hour in which Customer holds a Reservation are not charged.

## Part 3. Reserved Plans

## 16. Plan Orders and the Term

16.1. **Fields.** A Plan Order states: its order number; Customer's legal details; the version of this Agreement under which it is made; the **Start Date**; the **Term**, in whole months; the **Windows** and the **Days** (clause 17); the number of **Fleets** (clause 18); whether the Plan has a **Fleet lock**, and which Fleets it locks (clause 19); the **Hour Price**; the **Payment schedule**, Upfront or Gated (clause 20); the Chosen Models (clause 7.2); the Performance Threshold (clause 23.2); the Account email; each party's notice address and email; Customer's VAT status; the payment currency and account; and the published settings that apply to it (clause 9.3). It also states the **Hours in the Term** and the **Price of the Term**, worked out from those fields under clauses 17.3 and 16.4. The fields are fixed for the Term.

16.2. **Making a Plan Order.** We make a Plan Order only where we can keep the capacity it needs under clauses 18 and 19 alongside every Plan Order made before it. A Plan Order is made on the date of the last signature or acceptance under clause 12.4, and its Start Date is on or after that date.

16.3. **Term.** The Term starts at 00:00:00 on the Start Date and ends at 23:59:59 on the day before the same date in the month that falls the Term's number of months after the Start Date; where that month has no such date, the Term ends at 23:59:59 on the last day of that month. **Day N** means the Nth calendar day counting the Start Date as Day 1.

16.4. **Price.** Each Plan hour is priced at the Hour Price, fixed for the Term; under a Fleet lock, the Hour Price includes the lock. The Hour Price is a single all-in price: it covers the Fleets and the interconnect that joins the machines of each Fleet, the models in clause 7, and operation and support, and no other fee, surcharge or charge applies during the Term, apart from the release fee that the Hour Market Rules deduct from what a released hour earns. The Price of the Term is the Hour Price multiplied by the Hours in the Term. All balances, refunds and claims under a Plan Order are counted in Plan hours, valued at the Hour Price. A Plan Order sets no price for any other capacity Customer buys.

16.5. The Rate of each Fleet on the Fleet List when a Plan Order is made is fixed for its Term.

## 17. Windows, Days and Plan hours

17.1. **Windows.** Each day has three **windows**, whose times we publish. They are today the **Americas Window**, 00:00–08:00 KST, which falls on the previous calendar day in the Americas; the **Asia Window**, 08:00–16:00 KST; and the **Europe Window**, 16:00–24:00 KST. The windows are periods of the day, not separate pools of capacity or separate machines. A Plan Order elects one or more windows; all three together are a **Full day**.

17.2. **Days.** A Plan Order elects **Every day** or **Weekdays**. Under Every day, each elected window counts on every day of the Term. Under Weekdays, it counts only on Monday to Friday in its own region: the Americas Window on each day from Tuesday to Saturday, and the Asia and Europe Windows on each day from Monday to Friday.

17.3. **Plan hours and Tranches.** For every day of the Term, each elected window that counts on that day holds the window's hours times the Fleets; these are the **Plan hours**. Each calendar month within the Term is one **Tranche**, holding the Plan hours of its days within the Term. Tranche 1 runs from the Start Date to the end of that month, and the final Tranche from the first day of its month to the end of the Term; there are as many Tranches as the Term has months where the Start Date is the first day of a month, and one more otherwise. The Hours in the Term are the Plan hours of all Tranches.

17.4. A Plan hour expires at 23:59:59 on the last day of its Tranche unless consumed. Expired hours do not carry forward, convert, accumulate or have any cash value.

17.5. Example: a Plan Order with a Start Date of 15 December 2026, a Term of 24 months, the Asia Window, Every day, 1 Fleet, no Fleet lock and the Gated schedule:

| Tranche | Period | Plan hours |
| --- | --- | --: |
| 1 | 15–31 Dec 2026 | 136 |
| 2 | Jan 2027 | 248 |
| 15 | Feb 2028 (leap) | 232 |
| 25 | 1–14 Dec 2028 | 112 |
| **All** | **15 Dec 2026 – 14 Dec 2028** | **5,848** |

The same Plan Order on Weekdays holds 4,184 Plan hours.

## 18. Guaranteed Limit and set-aside

18.1. A Plan's **Guaranteed Limit** is its Fleets at once, in every hour of its elected windows that counts under clause 17.2.

18.2. **Set-aside.** We keep the capacity of the Guaranteed Limit, on Fleets that serve the Chosen Models, off sale to anyone else until 6 hours before each hour starts. From then on, capacity Customer has not reserved may be reserved by others. Except under a Fleet lock, the guarantee is for capacity, not for a particular Fleet: where one Fleet is fully reserved, Customer reserves on another. Customer may make Reservations within its Guaranteed Limit for any time in the Term, subject to clauses 20 and 22.1. With Plan hours, Customer may reserve more Fleets at once, or other hours, only as Extra Hours after Full Prepayment (clause 21.2).

18.3. If Customer asks, more than 6 hours before it starts, for a Reservation within its Guaranteed Limit and we cannot confirm it on any Fleet that serves the Chosen Models, we credit Customer one Plan hour under clause 23.3 for each hour asked for, in addition to its balance.

18.4. The guarantee does not cover hours of maintenance announced under clause 4.6 before Customer asks for them. For each Fleet of its Guaranteed Limit that such maintenance prevents us from keeping in an hour, Customer is credited one Plan hour under clause 23.3.

18.5. **Priority over hourly use.** Customer's priority over hourly use comes from the capacity kept for it under this clause and clause 19, which nobody else can reserve until 6 hours before each hour starts or, on a locked Fleet, at all. Customer never takes, and never needs to take, an hour that anyone else holds, and no confirmed Reservation is ever displaced to make room for another (clause 4.4).

## 19. Fleet lock

19.1. A Plan whose Windows are a Full day and whose Days are Every day may have a **Fleet lock**. Its Plan Order then names as many Fleets as its Fleets (the **locked Fleets**).

19.2. Each locked Fleet is held for Customer in every hour of the Term and is Customer's alone: it is never offered to anyone else, not even 6 hours before an hour starts, is not counted in the free hours under clause 13.4, and does not serve the shared pool. Reservations within the Guaranteed Limit are made and delivered on the locked Fleets. Another customer uses a locked Fleet only in an hour that Customer itself has released in the Hour Market and that is sold or joins the shared pool.

19.3. Because nobody else uses a locked Fleet, Customer's model state and caches may stay on it between Customer's hours for the Term, encrypted and isolated from other customers. We delete them when Customer asks, when the Plan ends, and before the start of any hour that Customer has released and that is sold or joins the shared pool.

19.4. If a locked Fleet cannot serve, including under maintenance (clause 4.6), we deliver the affected Reservations on another Fleet, without the kept model state and caches, or credit the hours under clause 4.7.

## 20. Payment schedule

20.1. No Plan hour is issued or served until it is paid for. Each payment that counts under clause 3.5 issues Plan hours equal to the amount paid, net of VAT, divided by the Hour Price and rounded down to four decimal places, to the earliest Tranches not yet fully paid. Plan hours that occur before they are paid for are counted as consumed: the Price of the Term does not change.

20.2. **Upfront.** Under the Upfront schedule, Customer pays the Price of the Term before the Start Date, and reaches Full Prepayment when it is paid. If the Price of the Term is not paid in full by the end of the Start Date, the Plan ends then, with nothing further owed by either party, except that we refund any amount paid under it as clause 24.3(b) and (c) provide.

20.3. **Gated.** Under the Gated schedule, Day 1 to Day 30 inclusive is **Phase A**. No payment is due during Phase A. Customer may pay any amount at any time.

20.4. **Withdrawal.** Customer may withdraw from a Gated Plan during Phase A only, by making no payment under its Plan Order by the end of Day 30. If no payment has been made by then, the Plan ends at 23:59:59 on Day 30, with no payment or liability owed by either party, except under clause 11.4. Once Customer pays any amount under the Plan Order, this right ends. We may extend the Gate 1 deadline in writing.

20.5. **Gates.** Customer prepays in whole Tranches, in cleared funds, by these deadlines; a gate that names more Tranches than the Term has covers all of them:

| Gate | Deadline | Tranches paid in total | Effect |
| --- | --- | --- | --- |
| 1 | end of Day 30 | 1–6 | paid hours used at Customer's choice |
| 2 | end of Day 60 | 1–12 | paid hours used at Customer's choice |
| 3 | end of Day 90 | all | Full Prepayment (clause 21) |

20.6. Customer may pay any amount early, including the Price of the Term at signing. Extra Hours and spending ahead start only on Full Prepayment. The Term runs from the Start Date regardless (clause 20.1).

20.7. Paying a gate does not oblige Customer to pay the next gate. No unpaid amount is ever overdue: there is no interest, acceleration, suspension or default for a missed gate.

20.8. If Customer misses a gate after making any payment:

- Customer keeps every Plan hour already paid for and may reserve it until the end of its Tranche, and the Plan ends at the end of the last Tranche with paid hours;
- Customer may make no further prepayment and can never reach Full Prepayment, so clause 21.2 never applies; and
- no amount becomes due, and neither party may end the Plan because of the missed gate.

20.9. Example, for the Plan Order in clause 17.5. The amount due at each gate is the hours due multiplied by the Hour Price:

| Gate | Deadline | Hours paid in total | Hours due at the gate |
| --- | --- | --: | --: |
| 1 | 13 Jan 2027 | 1,344 | 1,344 |
| 2 | 12 Feb 2027 | 2,808 | 1,464 |
| 3 | 14 Mar 2027 | 5,848 | 3,040 |
| **Total** |  |  | **5,848** |

## 21. Full Prepayment

21.1. **Full Prepayment** means our receipt, in cleared funds, of the Price of the Term. Until then, Customer reserves its paid Plan hours at times of its choosing within the Tranche each belongs to and within its Guaranteed Limit.

21.2. From Full Prepayment, for all Plan hours remaining in the Term:

- **Extra Hours.** Customer may reserve Plan hours beyond its Guaranteed Limit: on more Fleets at the same time than its Fleets, or outside its elected windows or the days on which they count (the **Extra Hours**). Extra Hours are subject to availability, may be reserved at most 30 days before they start, and never take an hour that anyone holds or capacity kept for another Plan (clause 18.5).
- **Spending ahead.** Customer may reserve more Plan hours in a period than its current Tranche holds. As each reserved hour occurs it is drawn from the unexpired Tranche nearest to expiry, so the excess comes from future Tranches. Plan hours of a future Tranche consumed early do not reappear in their own month.

## 22. Using Plan hours

22.1. A Reservation made with Plan hours consumes as many Plan hours as its Fleet's Rate. Customer may make one only if its paid Plan hours can cover it together with every Reservation it already holds under the Plan, each from Plan hours that have not expired when it occurs.

22.2. **Records.** Our technical records of payments, Reservations and consumption are kept in an append-only, hash-chained ledger whose daily roots we publish, and Customer may obtain its own entries with proof that they were recorded and not altered. These records are conclusive, except for manifest error.

22.3. **Release.** Before a reserved hour starts, Customer may release it in the Hour Market under the Hour Market Rules. A released hour that is sold, or that reaches its start unsold, counts as consumed, and Customer is credited only the HC it earns under those Rules, less the release fee. In the Hour Market we buy no hour, guarantee no sale or price, and pay no cash. Except as the Hour Market Rules provide for a released hour, we do not buy back, credit or pay cash for any expiring or unused hour, at any time, for any reason, and we are never obliged to buy any hour.

22.4. Hours Customer reserves with HC are outside the Plan: they draw on no Tranche, and neither the Guaranteed Limit nor any other limit of this Part applies to them.

## 23. Delivery obligation

23.1. This is our only performance commitment for a Plan. Each hour reserved under the Plan delivers **materially the same performance as measured on Customer's own workload in the 30 days that begin with Customer's first hour reserved under the Plan** (the **Baseline**). We record the Baseline at the end of that period and give Customer a copy.

23.2. Clause 4.7 also applies to any period of more than 15 consecutive minutes in which Customer's workload on an hour reserved under the Plan runs below the **Performance Threshold** stated in the Plan Order. Where Customer sends its records of such a period, we credit the hours unless our records show the Fleet was available and at or above the Performance Threshold throughout.

23.3. **Credits in Plan hours.** A Reservation credited under clause 4.7 is credited back with the Plan hours it consumed, which are not counted as consumed. Clauses 18.3 and 18.4 credit one Plan hour for each hour or Fleet they state, within 24 hours after the hour concerned. Credited Plan hours go to the Tranche current when they are credited, or to the next Tranche where fewer than 7 days remain in the current one, and expire with the Tranche they are credited to.

## 24. Renewal and ending of a Plan

24.1. **Renewal.** A Plan ends at the end of its Term and does not renew automatically. Customer may continue by a new Plan Order, which may start on the day after the Term ends and may state different fields.

24.2. Customer has no right to end a Plan Order, for convenience or for any other reason, except the withdrawal right in clause 20.4. If a court or tribunal holds that Customer has a termination right that cannot lawfully be excluded, Customer may exercise it only if we fail to deliver reserved hours under the Plan for 60 consecutive days after Customer's written notice identifying the failure, and Customer's only financial remedy is the refund in clause 24.3.

24.3. Where a Plan ends under clause 10.2 or 24.2:

- (a) we waive every unpaid amount, do not pursue the remainder of the Term, and cancel Reservations under the Plan for hours that have not yet occurred, which count as unconsumed;
- (b) the refund basis is Plan hours paid for minus Plan hours consumed, at the Hour Price; and
- (c) we pay that refund within 60 days, without interest, net of any amount Customer owes us, except where a law prohibits the payment.

24.4. At the end of the Term, all unconsumed Plan hours lapse without refund.

## Plan Order form

| Item | Entry |
| --- | --- |
| Order number | \[ \] |
| Customer | \[legal name\], a company organised under the laws of \[jurisdiction\], registration number \[ \], registered address \[ \] |
| Customer Agreement (clause 9.1) | Version of \[date\]; SHA-256 of the Korean text \[ \] and of the English text \[ \] |
| Start Date (clause 16.3) | \[ \] |
| Term (clause 16.3) | \[ \] months, ending on \[ \] |
| Windows (clause 17.1) | \[Americas Window (00:00–08:00 KST) / Asia Window (08:00–16:00 KST) / Europe Window (16:00–24:00 KST) / Full day\] |
| Days (clause 17.2) | \[Every day / Weekdays\] |
| Fleets (clause 18.1) | \[1\] |
| Fleet lock (clause 19) | \[None / Fleets \[ \], one for each of the Fleets; only with Windows of a Full day and Days of Every day\] |
| Hour Price (clause 16.4) | KRW \[ \] or USD \[ \] per fleet-hour, in the payment currency below, exclusive of VAT, including any Fleet lock, fixed for the Term |
| Hours in the Term (clause 17.3) | \[ \] |
| Price of the Term (clause 16.4) | KRW \[ \] or USD \[ \], exclusive of VAT |
| Payment schedule (clause 20) | \[Upfront: the Price of the Term before the Start Date / Gated: Gate 1 by \[date\], Gate 2 by \[date\], Gate 3 by \[date\]\] |
| Chosen Models (clause 7.2) | \[models from the Fleet List\] |
| Performance Threshold (clause 23.2) | \[80\]% of the Baseline |
| Account email (clause 2.2) | \[ \] |
| Supplier notice address and email (clause 11.7) | \[ \] |
| Customer notice address and email (clause 11.7) | \[ \] |
| Customer VAT status (clause 3.2) | \[No place of business in the Republic of Korea / Korean business registration number \[ \]\] |
| Payment currency and account (clause 3.3) | \[KRW, clause 3.3(a) / USD, clause 3.3(b): \[bank, account holder, account number, SWIFT\] / US dollar stablecoin, clause 3.3(c)\] |
| Published settings (clause 9.3) | As published when this Plan Order is made: release fee \[10\]%; maintenance allowance \[3\] hours a week per Fleet; window times as stated under Windows |
| Fleet Rates (clause 16.5) | As published on the Fleet List when this Plan Order is made: \[Fleet 1: 1.00; Fleet 2: 1.00; Fleet 3: 1.00\] |

## Signatures

**Supplier: HYOR AI Tech Ltd.**

Name: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Title: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Customer**

Name: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Title: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
