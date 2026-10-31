# SparkPipe 2-Year Reserved Fleet Service Agreement

## Parties and definitions

This Agreement is made on the Signature Date between:

- **Supplier:** HYOR AI Tech, Ltd., a company organised under the laws of the Republic of Korea, business registration number 641-87-03809, with its registered office in Yongsan-gu, Seoul, Republic of Korea, whose full address we give on request and in the signed Agreement ("**we**", "**Supplier**"); and
- **Customer:** \[legal name\], a company organised under the laws of \[jurisdiction\], registration number \[number\], registered address \[address\] ("**Customer**").

In this Agreement:

- **Signature Date** means the date of the last signature below. The Term begins on it, and the Tranches, Phase A and every payment deadline run from it.
- **KST** means Korea Standard Time (UTC+9). All times and days in this Agreement are KST.
- **Day N** means the Nth calendar day counting the Signature Date as **Day 1**.
- **Fleet** means a group of computing machines that we operate as a single unit and list on our Fleet List, published at sparkpipe.ai/fleets.html.
- **Fleet-hour** or **hour** means the use of one entire Fleet for sixty minutes.
- **Rate** means the number of hours that one hour on a Fleet consumes, as published on our Fleet List; 1 unless the Fleet List states otherwise. The Rate of each Fleet listed at the Signature Date is fixed for the Term.
- **East Window** and **West Window** mean the two 12-hour periods of each day defined in the Schedule.
- **Tranche** means the hours of one calendar month within the Term, as set out in clause 3.
- **Reservation** means the booking of one or more hours on a selected Fleet through our system under clause 8. **Reserve** has a matching meaning.
- **Phase A** means Day 1 to Day 30 inclusive.
- **Full Prepayment** means our receipt, in cleared funds, of the price of every Tranche in the Term.
- **Account** means Customer's account with us under clause 8.5.
- **Hourly Terms** means our SparkPipe Hourly Terms, published at sparkpipe.ai/hourly-terms.html. **Hour-credit** (or **HC**) has the meaning given there.
- **Hour Market Rules** means our SparkPipe Hour Market Rules, published at sparkpipe.ai/market-rules.html, in the version in force when Customer releases an hour.

## 1. Subject

1.1. We provide Customer the SparkPipe reserved fleet service for the Term: reserved capacity on our Fleets, the guaranteed models in clause 10, the interconnect that joins the machines of each Fleet, and operation and support. The fleet-hour is the unit in which the service is measured, reserved and paid for. Customer selects the Fleet at each Reservation from the Fleets we operate at that time, current or future, subject to availability under clause 8.8.

1.2. We provide a service, not hardware. We keep ownership, possession and operation of every Fleet.

## 2. The hour and the price

2.1. The price of each fleet-hour is the **Hour Price** stated in the Schedule, fixed for the Term.

2.2. The Hour Price is a single all-in price. It covers the Fleet and its interconnect, the models in clause 10, and operation and support. No other fee, surcharge or charge applies during the Term, apart from the release fee that the Hour Market Rules deduct from what a released hour earns. The Hour Price is exclusive of value-added tax (VAT). We charge VAT at the statutory rate (10% at the Signature Date) on every payment and show it separately on our invoice, except a payment that the Schedule states is zero-rated and that Customer makes in the way the Schedule states. Any amount charged as VAT that is not owed is an overpayment, and we refund it to Customer. If a payment Customer made as zero-rated is held subject to VAT because Customer's information was incorrect or Customer did not pay in the way the Schedule states, Customer pays us that VAT, with any additional tax and penalty, on demand.

2.3. The Term contains 24 hours for every day in it other than a Sunday: **15,000 to 15,048 hours**, depending on how many Sundays fall within the Term and whether 29 February does. The Schedule states the number. The price of the Term is the Hour Price multiplied by those hours.

2.4. All balances, refunds and claims are counted in hours, valued at the Hour Price.

2.5. Customer pays every amount in full, in cleared funds, without set-off, deduction or withholding. If a law requires Customer to withhold or deduct any tax from a payment, Customer pays an additional amount so that we receive what we would have received without the withholding or deduction. Customer bears its own and any intermediary bank, transfer, network and currency conversion charges, so that the full amount invoiced is paid. We bear our payment processor's own fees.

## 3. Term and Tranches

3.1. The Term starts at 00:00:00 on the Signature Date and ends at 23:59:59 on the day before the date 24 calendar months later. Where the Signature Date is 29 February, the Term ends at 23:59:59 on 28 February two years later.

3.2. Each calendar month within the Term is one Tranche. Tranche 1 runs from the Signature Date to the end of that month. The final Tranche runs from the first day of its month to the end of the Term. There are 24 Tranches where the Signature Date is the first day of a month, and 25 otherwise.

3.3. Each Tranche holds 24 hours for each of its days within the Term other than a Sunday. A whole month holds 576 to 648 hours, depending on its length and how many Sundays it has. A partial first or final Tranche counts only its days within the Term.

3.4. Example, signed on 15 December 2026:

| Tranche | Period | Hours |
| --- | --- | --- |
| 1 | 15–31 Dec 2026 | 360 |
| 2 | Jan 2027 | 624 |
| 15 | Feb 2028 (leap) | 600 |
| 25 | 1–14 Dec 2028 | 288 |
| **All** | **15 Dec 2026 – 14 Dec 2028** | **15,048** |

3.5. An hour expires at 23:59:59 on the last day of its Tranche unless consumed. Expired hours do not carry forward, convert, accumulate or have any cash value. Before a reserved hour occurs, Customer may release it under the Hour Market Rules (clause 8.6). A released hour that is sold, or that reaches its start unsold, counts as consumed, and Customer is credited the hour-credits it earns under those Rules.

## 4. Timezone election

4.1. Each day consists of 12 East Window hours and 12 West Window hours. East and West are periods of the day. They are not separate pools of capacity or separate machines.

4.2. Customer makes one timezone election in the Schedule at signing: **50/50**, **100% East** or **100% West**. The election is fixed for the Term and may be changed only on renewal.

4.3. A 100% election needs our written approval. We give it in the order customers sign, and only where we can keep the capacity that clause 4.6 guarantees.

4.4. The election sets Customer's **Guaranteed Limit**: the hours Customer may reserve in each window on each day from Monday to Saturday:

| Election | East hours per day | West hours per day | Rate |
| --- | --: | --: | --- |
| 50/50 | 12 | 12 | 1× |
| 100% East | 24 | 0 | 2× |
| 100% West | 0 | 24 | 2× |

"2×" means twice as many hours per day in the elected window as a 50/50 election gives. It does not change the price per hour or the total hours in clause 2.3. Hours Customer reserves with hour-credits under the Hourly Terms are outside the election and the Guaranteed Limit and draw on no Tranche.

4.5. The election applies only from Full Prepayment (clause 7). Until then Customer's Guaranteed Limit is that of a 50/50 election.

4.6. **Guarantee.** We keep enough capacity, on Fleets that serve Customer's Chosen Models, for Customer to reserve hours up to its Guaranteed Limit on every day from Monday to Saturday in the Term. Customer may make such Reservations for any time in the Term, subject to clauses 5.4 and 8.2. The guarantee is for capacity, not for a particular Fleet: where one Fleet is fully reserved, Customer reserves on another. If Customer asks for a Reservation within its Guaranteed Limit more than 6 hours before it starts and we cannot confirm it on any such Fleet, we credit Customer one hour under clause 9.3 for each hour asked for, in addition to its balance.

4.7. **Sundays.** Sundays are kept for our own use, including scheduled maintenance. The Term holds no hours for them (clause 2.3), and the Guaranteed Limit does not apply on them. Customer may still reserve its hours on a Sunday where capacity is available, at most 30 days before they start: within the daily limits of a 50/50 election before Full Prepayment, and as Extra Hours (clause 7.1) after it. We carry out scheduled maintenance only on Sundays. Hours we set aside for it are not open for Reservation, and a Sunday Reservation already confirmed is never displaced (clause 8.4).

## 5. Phase A and withdrawal

5.1. No payment is due during Phase A. Customer may pay any amount at any time, and each cleared payment issues hours under clause 5.5.

5.2. Customer may withdraw from this Agreement during Phase A only, by making no payment under it by the end of Day 30. If no payment has been made by then, this Agreement ends at 23:59:59 on Day 30, with no payment or liability owed by either party, except under the confidentiality terms in clause 13. We may extend the Gate 1 deadline in writing.

5.3. Once Customer pays any amount under this Agreement, the withdrawal right in clause 5.2 ends.

5.4. There are no unpaid hours: until Customer pays, no hour is issued or served under this Agreement. Until Full Prepayment, Customer reserves its paid hours at times of its choosing under clause 8, within the Tranche each hour belongs to and its Guaranteed Limit (12 East Window and 12 West Window hours per day), and on Sundays under clause 4.7. Hours Customer reserves with hour-credits are outside these limits (clause 4.4).

5.5. Each cleared payment issues hours equal to the amount paid, net of VAT, divided by the Hour Price, to the earliest Tranches not yet fully paid. Hours that occur before they are paid for are counted as consumed: the price of the Term does not change.

## 6. Prepayment gates

6.1. Customer prepays in whole Tranches, in cleared funds, by these deadlines:

| Gate | Deadline | Tranches paid in total | Effect |
| --- | --- | --- | --- |
| 1 | end of Day 30 | 1–6 | paid hours used at Customer's choice |
| 2 | end of Day 60 | 1–12 | paid hours used at Customer's choice |
| 3 | end of Day 90 | all | Full Prepayment (clause 7) |

6.2. Customer may pay any amount early, including Full Prepayment at signing. Fungibility, the elected rate and Extra Hours start only on Full Prepayment. The Term runs from the Signature Date regardless, so hours that occur before they are paid for are consumed under clause 5.5 and are not restored.

6.3. Paying a gate does not oblige Customer to pay the next gate. No unpaid amount is ever overdue: there is no interest, acceleration, suspension or default for a missed gate.

6.4. If Customer misses a gate after making any payment:

- Customer keeps every hour already paid for and may reserve it until the end of its Tranche, and this Agreement expires at the end of the last Tranche with paid hours;
- Customer may make no further prepayment and can never reach Full Prepayment, so clause 7 never applies; and
- no amount becomes due, and neither party may terminate this Agreement because of the missed gate.

6.5. Hours are issued only on cleared payment, under clause 5.5. Hours not paid for are not issued.

6.6. Example, signed on 15 December 2026. The amount due at each gate is the hours due multiplied by the Hour Price:

| Gate | Deadline | Hours paid in total | Hours due at the gate |
| --- | --- | --: | --: |
| 1 | 13 Jan 2027 | 3,456 | 3,456 |
| 2 | 12 Feb 2027 | 7,224 | 3,768 |
| 3 | 14 Mar 2027 | 15,048 | 7,824 |
| **Total** |  |  | **15,048** |

## 7. Full Prepayment

7.1. On Full Prepayment, for all hours remaining in the Term:

- **Fungibility.** Hours may be reserved in either window up to Customer's Guaranteed Limit (clause 4.4).
- **Elected rate.** A 100% election takes effect, giving a Guaranteed Limit of 24 hours per day in the elected window.
- **Extra Hours.** Customer may reserve hours beyond its Guaranteed Limit from Monday to Saturday, and without a daily limit on Sundays (the **Extra Hours**). Extra Hours are subject to availability, may be reserved at most 30 days before they start, and are never displaced once confirmed (clause 8.4). They never take an hour held under the Hourly Terms (clause 8.13).
- **Spending ahead.** Customer may reserve more hours in a period than its current Tranche holds. As each reserved hour occurs it is drawn from the unexpired Tranche nearest to expiry, so the excess comes from future Tranches. Hours of a future Tranche consumed early do not reappear in their own month.
- **Concurrency.** Customer may hold Reservations on several Fleets at the same time.

Hours Customer reserves with hour-credits are not Extra Hours, and none of the limits of this clause applies to them (clause 4.4).

7.2. Customer's hours may be reserved on any Fleet we operate, current or future, subject to availability. This Agreement does not set the price of any further capacity Customer buys later.

## 8. Reservation, consumption, use and responsibility

8.1. Customer reserves hours through our website while signed in to its Account, or through our API with an access token issued under it, naming the Fleet, the start time and the number of hours. A Reservation exists once our system confirms it. A reserved hour on a Fleet consumes as many hours as that Fleet's Rate.

8.2. A Reservation is final once confirmed: it cannot be cancelled, moved or refunded, except under clause 11.3, but Customer may release it under the Hour Market Rules before it occurs (clause 8.6). Customer may not make a Reservation unless its paid hours can cover it together with every Reservation it already holds, each from hours that have not expired when it occurs. A reserved hour is consumed when that hour occurs, whether or not it is used, and not before. Our technical records of payments, Reservations and consumption are kept in an append-only, hash-chained ledger whose daily roots we publish, and Customer may obtain its own entries with proof that they were recorded and not altered. These records are conclusive, except for manifest error.

8.3. **Customer has the exclusive right to use each reserved hour.** Customer may use it, or let others use it by resale, transfer or gift through its Account and the access tokens it issues, and may run any lawful workload on it, subject to clauses 8.9 to 8.12. Any such arrangement is made outside our system and is out of our control (clause 8.6).

8.4. We do not move, reassign, displace or interfere with a reserved hour, except under clause 8.11. A Reservation is delivered on the Fleet it was made on.

8.5. Access is through Customer's Account: the account on our website identified by the email address Customer registers with us, signed in to with one-time codes sent to that address. It authorises Reservations and the use of each reserved hour, and Customer may issue access tokens and add users under it. Access is automated: no notice to us, approval or waiting period. We set the technical mechanism and may change it without amending this Agreement, provided equivalent access continues. Customer bears all risk of loss or compromise of its Account email and access tokens; any use under the Account is Customer's use.

8.6. **Hour Market.** We run the Hour Market under the Hour Market Rules, and Customer may release a reserved hour through it before the hour occurs. In the Hour Market we buy no hour, guarantee no sale or price, and pay no cash: a released hour earns only the hour-credits those Rules provide, less the release fee. Otherwise, we keep no register of, and are not a party to, any resale or transfer of hours. We do not match, price, clear or settle such transactions, have no liability for their price or counterparties, and are never obliged to buy any hour. Except as the Hour Market Rules provide for a released hour, we do not buy back, credit or pay cash for any expiring or unused hour, at any time, for any reason.

8.7. Hours give the right to use the service only. They give no ownership of, or rights in, our company, hardware, software, models or intellectual property. They are not a deposit, loan, security or investment, and carry no expectation of profit from our efforts.

8.8. We publish live capacity in machine-readable form at the address stated in the Schedule, regenerated at least every 30 seconds. It lists each Fleet we operate, whether it is serving, and the hours open for Reservation in each window. Wherever this Agreement makes anything subject to availability, availability means the capacity that publication shows when Customer makes the Reservation.

8.9. **We provide capacity only.** Customer alone decides what runs on its hours. Customer is solely responsible for its workloads, the data and content it processes, the outputs it obtains, and every use made of them. We do not monitor, review or control workloads, data or outputs, and we are not responsible for them. Customer is responsible for every person who uses hours under its Account or this Agreement, including its end users, as if their acts were its own. Customer must bind anyone it resells or transfers hours to by terms no less protective of us than clauses 8.9 to 8.12, and remains responsible to us for that person's compliance.

8.10. **Lawful use.** Customer must comply, and must ensure that every user of its hours complies, with every law that applies to the use, including laws on export control and sanctions, personal data, intellectual property and computer misuse, and with the licence terms of any model it uses. In particular, no hour may be used to: (a) create, store or distribute unlawful content, including child sexual abuse material; (b) attack, probe or disrupt any system or network, or distribute malicious code; (c) infringe the rights of any person; (d) benefit any person, entity or country subject to sanctions of the United Nations, the Republic of Korea, the United States or the European Union, or serve any end use that export control laws prohibit, including weapons of mass destruction; or (e) interfere with a Fleet or with other customers. Customer represents that neither it nor any person that owns or controls it is subject to those sanctions. Where Customer uses its hours to provide generative AI products or services to others, Customer is their provider and bears the duties that apply to them, including the transparency and labelling duties of the Korean Framework Act on the Development of Artificial Intelligence and Establishment of Trust.

8.11. **Suspension.** We may suspend a Reservation or access under the Account, at once and without prior notice, where (a) a law, court or competent authority requires it, (b) we reasonably believe clause 8.10 is being breached, or (c) the use threatens the security or integrity of a Fleet or of others. We tell Customer as soon as practicable and lift the suspension when its cause ends. Hours lost to a suspension caused by the use of Customer's hours are consumed and are not credited under clause 9.2.

8.12. **Customer data.** Customer keeps all rights in its data and outputs. We access them only as needed to operate and secure the service, or where the law requires. Customer is responsible for the lawfulness of any personal data it processes and for keeping its own copies: we keep no backups of Customer data. We handle Customer data as stated in our published privacy policy. The Fleets and our servers are located in Seoul, Republic of Korea. We hold no Customer data beyond transient caches, which we delete within 24 hours, so nothing is returned when this Agreement ends. We tell Customer without delay of any intrusion, any leak of its information and any unannounced interruption of the service.

8.13. **Priority over hourly use.** Within its Guaranteed Limit, Customer may reserve an hour that another customer reserved from our own capacity under the Hourly Terms, if that hour starts more than 6 hours later; the hour then becomes Customer's Reservation. An hour bought in the Hour Market as a released hour is never taken. Extra Hours and hours on a Sunday never take an hour held under the Hourly Terms. Customer's own Reservations are never displaced (clause 8.4).

## 9. Delivery obligation

9.1. This is our only performance commitment. Each reserved hour will deliver **materially the same performance as measured on Customer's own workload in the 30 days that begin with Customer's first reserved hour** (the **Baseline**). We record the Baseline at the end of that period and give Customer a copy.

9.2. **Automatic credit.** If a reserved hour is unavailable, or the Fleet it was reserved on is unreachable for more than 15 consecutive minutes, every reserved hour in which any part of that period falls is credited back to Customer in full and is not counted as consumed. The same applies to any period of more than 15 consecutive minutes in which Customer's workload runs below the Performance Threshold stated in the Schedule. No claim is needed: we credit from our own records. Where Customer's records show such a period and ours do not, Customer may send its records within 7 days, and we credit the hours unless our records show the Fleet was available and at or above the Performance Threshold throughout that period.

9.3. Credits are made in hours, never in cash: each affected hour is credited as one hour. We credit within 24 hours after the end of the period, to the Tranche current at that time, or to the next Tranche where fewer than 7 days remain in the current one. Credited hours expire with the Tranche they are credited to.

9.4. Credited hours are Customer's sole and exclusive remedy for any failure of performance, subject to clause 12.5.

## 10. Models

10.1. We offer only open-source models whose availability cannot be withdrawn by any third party. We publish the available models on the Fleet List.

10.2. The models Customer chooses from the Fleet List and names in the Schedule (the **Chosen Models**) remain available to Customer, on at least one Fleet, for the whole Term. Other models on the Fleet List are available while they are listed.

10.3. **Model licences and outputs.** Each model is third-party open-source software provided under its own licence, which Customer must comply with. Outputs are generated automatically from Customer's inputs: we do not author, review or endorse them, and we give no warranty that any output is accurate, complete, lawful, safe or fit for any purpose. Customer is responsible for evaluating outputs before relying on, publishing or acting on them.

## 11. No refunds and termination

11.1. **Every amount paid is non-refundable and non-cancellable**, except under clause 11.3. Customer may not set off, deduct, counterclaim or withhold against any amount.

11.2. Customer has no right to terminate or rescind this Agreement, for convenience or for any other reason, except the withdrawal right in clause 5.2. If a court or tribunal holds that Customer has a termination right that cannot lawfully be excluded, Customer may exercise it only if we fail to deliver reserved hours for 60 consecutive days after Customer's written notice identifying the failure, and Customer's only financial remedy is the refund in clause 11.3(c).

11.3. We may terminate this Agreement by written notice (a) if Customer materially breaches it and does not remedy the breach within 30 days after our written notice, or (b) at once, where a breach of clause 8.10 cannot be remedied, where Customer or a person that owns or controls it becomes subject to the sanctions in clause 8.10(d), or where a law, court or competent authority requires us to stop serving Customer. We may not terminate because of a missed gate.

Where this Agreement ends under clause 11.2 or 11.3:

- (a) we waive every unpaid amount, do not pursue the remainder of the Term, and cancel Reservations for hours that have not yet occurred, which count as unconsumed;
- (b) the refund basis is hours paid for minus hours consumed, at the Hour Price; and
- (c) we pay that refund within 60 days, without interest, net of any amount Customer owes us, except where a law prohibits the payment.

11.4. At the end of the Term, all unconsumed hours lapse without refund.

11.5. Clauses 2.2, 2.5, 8.6, 8.7, 8.9 to 8.12, 10.3, 11, 12, 13 and 14 survive the end of this Agreement.

## 12. Liability

12.1. Our aggregate liability for all causes combined is capped at the amounts we received from Customer under this Agreement in the 6 months before the event giving rise to the claim.

12.2. We are not liable for lost profit, lost data, lost opportunity, business interruption, or indirect or consequential loss, whether or not foreseeable.

12.3. Customer indemnifies us, our officers and our employees against any claim, loss, liability, fine, penalty or cost, including reasonable legal fees, arising from (a) the use of its hours by Customer, its end users or anyone it resells or transfers hours to, (b) its workloads, data, content and outputs, (c) any breach of clauses 8.9 to 8.12, or (d) any third-party claim, or investigation by an authority, connected with them. We give no reciprocal indemnity.

12.4. The service and the models are provided "as available". We give no warranty of merchantability, fitness for purpose, accuracy of outputs or uninterrupted service beyond clause 9.

12.5. Nothing in this Agreement limits or excludes liability for intentional misconduct or gross negligence, or any other liability that cannot lawfully be limited or excluded.

12.6. Clauses 12.1 and 12.2 limit our liability only. They do not limit Customer's payment obligations or its liability under clauses 2.2, 8.9 to 8.12 and 12.3.

12.7. **Force majeure.** Neither party is liable for a delay or failure caused by an event beyond its reasonable control, including natural disaster, fire, flood, war, terrorism, epidemic, action of a government or authority, or failure of public power or telecommunications networks. Clause 9.2 still applies to reserved hours lost to such an event and is Customer's only remedy for them.

## 13. General

13.1. **Entire agreement.** This Agreement is the whole agreement between the parties. Neither party relies on any pre-contractual statement, projection, benchmark or representation, including anything said in diligence or during Phase A, except the Baseline recorded under clause 9.1.

13.2. **Amendment and waiver.** Amendments must be in writing and signed by both parties. No conduct, delay or extension operates as a waiver.

13.3. **Third parties.** This Agreement gives no rights to anyone except the parties.

13.4. **Publicity.** Customer may not publicise this Agreement or our relationship without our written consent. We may refer to Customer by name and logo after Full Prepayment.

13.5. **Confidentiality.** Each party keeps the other's non-public information confidential for 3 years after this Agreement ends, and pricing and technical information indefinitely.

13.6. **Assignment.** Customer may not assign this Agreement without our written consent. Reselling or transferring hours under clause 8.3 is not an assignment. We may assign this Agreement freely, including to an acquirer or financier.

13.7. **Notices.** Notices under this Agreement, including any withdrawal, election or termination, must be in writing and sent to the notice address or email stated in the Schedule. Our notices may also be sent to the email address verified on Customer's account.

13.8. **Business use.** Customer enters into this Agreement in the course of its business and not as a consumer. Each party confirms that it has read every clause and had the opportunity to negotiate it.

13.9. **Electronic signature.** Either party may sign this Agreement electronically, including by Customer accepting the hash of the final text while signed in to its Account, or by a signature with our signing key over that hash. Such a signature has the same effect as a handwritten signature. The record of any negotiation of this Agreement is kept with the signed text.

## 14. Law, disputes and language

14.1. This Agreement is governed by the laws of the Republic of Korea.

14.2. Any dispute arising out of or in connection with this Agreement is finally settled by arbitration in Seoul under the Arbitration Rules of the Korean Commercial Arbitration Board, before a sole arbitrator, in the Korean language. Either party may apply to a competent court for interim measures.

14.3. This Agreement is executed in English and Korean. The Korean text prevails.

## Schedule

| Item | Entry |
| --- | --- |
| Customer legal name and jurisdiction | \[ \] |
| Signature Date | \[ \] |
| Hours in the Term (clause 2.3) | \[ \] |
| Hour Price (clause 2) | KRW \[ \] or USD \[ \] per fleet-hour, in the currency of payment below, exclusive of VAT, fixed for the Term |
| Customer VAT status (clause 2.2) | \[No place of business in the Republic of Korea / Korean business registration number \[ \]\] |
| Timezone election | \[50/50 / 100% East / 100% West\], subject to clause 4.3 |
| East Window | \[hh:mm–hh:mm KST\] |
| West Window | \[hh:mm–hh:mm KST\] |
| Performance Threshold (clause 9.2) | \[80\]% of the Baseline |
| Live capacity address (clause 8.8) | https://sparkpipe.ai/availability.txt |
| Chosen Models (clause 10.2) | \[models from the Fleet List\] |
| Customer Account email (clause 8.5) | \[ \] |
| Payment currency and account | Either (a) KRW, by bank transfer from Customer's account at a Korean bank to our won account named on our electronic tax invoice (세금계산서), with VAT charged under clause 2.2; or (b) USD, by bank transfer from Customer's own account outside the Republic of Korea to our USD account at a Korean foreign exchange bank, zero-rated where Customer has no place of business in the Republic of Korea: \[bank, account holder, account number, SWIFT\], confirmed by both parties before payment; or (c) a US dollar stablecoin, such as USDC or USDT, that our payment processor CoinPayments accepts, on any network it accepts for that coin, through the invoice it issues for us, with VAT charged under clause 2.2 |
| Supplier notice address and email | \[ \] |
| Customer notice address and email | \[ \] |
| Fleet Rates at the Signature Date | As published on the Fleet List at signing: \[Fleet 1 (16 Sparks): 1.00; Fleet 2 (M5U): 1.00; Fleet 3 (16 Sparks): 1.00\] |
| Release fee (clause 8.6) | As published under the Hour Market Rules |

## Signatures

**Supplier: HYOR AI Tech, Ltd.**

Name: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Title: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Customer**

Name: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Title: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Signature: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
