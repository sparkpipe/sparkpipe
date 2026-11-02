# SparkPipe Customer Agreement

## Parties

This Agreement is made between **HYOR AI Tech Ltd.** (효르에이아이테크 유한회사), a company organised under the laws of the Republic of Korea, business registration number 641-87-03809, of Building B, Unit 1522, 95 Hangang-daero (Raemian Yongsan The Central), Yongsan-gu, Seoul, Republic of Korea ("**we**"), and the company, professional or independent developer identified in the account that accepts it ("**Customer**").

## Part 1. General

## 1. Definitions and documents

1.1. In this Agreement:

- **Fleet** means a group of computing machines that we operate ourselves as one unit in Korea and list on our **Fleet List**, published at sparkpipe.ai/fleets.html, which states the machines in each Fleet. The **Demo Fleet** is the Fleet that the Fleet List names as such.
- **Rate** means the number that the Fleet List gives each Fleet, which we set and may change as clause 9.2 provides. A Fleet's Rate is 1 unless the Fleet List states otherwise. A booking is priced at the Rate published when it is made.
- **Hour-credit** (**HC**) means the prepaid unit in which Customer pays for bookings.
- **Booking** means the right to use one entire Fleet for an hour (an **Hour**, clause 14) or for a Week (clause 15).
- **Week** means the seven days from 00:00 on a Monday to 24:00 on the following Sunday.
- To **hold** a Fleet is to have a Booking of it that is running.
- **Working state** means Customer's cached request state and the files in its Customer storage (clause 6.2).
- **Function** means a function we offer under clause 7.4.
- **KST** means Korea Standard Time (UTC+9). All times and days in this Agreement are in KST.

1.2. **Orders.** Customer buys under this Agreement by purchasing HC through our website (an **Order**). We may also agree other terms with Customer in a written order signed by both parties, which prevails over this Agreement for the terms it states; it does not vary clause 5, 6, 8 or 12 unless it says so expressly.

## 2. Professional use and account

2.1. **Professional use.** Customer confirms that it is purchasing and using hour-credits and Bookings for business, professional, commercial, or independent research and development purposes, and not as a retail household consumer, and that its account details are true and current. We may refuse or close an account that does not meet this.

2.2. Customer's **Account** is its account on our website, identified by the email address Customer verifies with us and signed in to with one-time codes we send to that address. Acceptance of this Agreement, purchases of HC, Bookings and changes to the Account are made while signed in to it, or through our API with an access token issued under it. Customer may issue access tokens and add users under the Account. Access is automated, with no notice to us, approval or waiting period. We set the technical mechanism and may change it without amending this Agreement, provided equivalent access continues.

2.3. Customer bears all risk of loss or compromise of its Account email and access tokens; any use under them is Customer's use. Customer may name a recovery email address, which can replace the Account email after a 72-hour delay with notice to the Account email.

## 3. Prices, VAT and payment

3.1. **Prices.** HC is priced at the HC Price (clause 13.1), and Bookings in HC (clauses 14.2 and 15.2). We set our prices on our website and may change them at any time, as clause 9.2 provides. Prices are exclusive of value-added tax (VAT).

3.2. **VAT.** We charge VAT at the statutory rate on every payment and show it separately, except on a payment that is zero-rated under clause 3.3(b). VAT is shown, for a payment under clause 3.3(a) made by card, including a card used through an easy-pay service, on the card slip, which is the VAT document for that payment; for any other payment under clause 3.3(a), on the electronic tax invoice (세금계산서) we issue where Customer gives its Korean business registration number; and for any other payment, on our invoice. Any amount charged as VAT that is not owed is an overpayment, and we refund it to Customer. If a payment made as zero-rated is held subject to VAT because Customer's information was incorrect or Customer did not pay in the way the Order states, Customer pays us that VAT, with any additional tax and penalty, on demand.

3.3. **Methods.** Customer pays in advance: no HC is issued before it is paid for. Our payment processors receive payments for us, and we remain the seller. Customer pays in one of the following ways, as our website offers it for the Order or a written order states:

- (a) in Korean won, through our payment processor Toss Payments (토스페이먼츠 주식회사), by a card issued in or outside the Republic of Korea, bank transfer, virtual account or easy-pay service, as its payment window offers them for the order;
- (b) in US dollars, by bank transfer from Customer's own account outside the Republic of Korea to our US dollar account at a Korean foreign exchange bank, whose details both parties confirm before payment; such a payment is zero-rated where Customer has no place of business in the Republic of Korea and the law allows; or
- (c) in a US dollar stablecoin, such as USDC or USDT, that our payment processor CoinPayments accepts, on any network it accepts for that coin, through the invoice it issues for us.

3.4. Customer pays every amount in full, without set-off, deduction or withholding. If a law requires Customer to withhold or deduct any tax from a payment, Customer pays an additional amount so that we receive what we would have received without the withholding or deduction. Customer bears its own and any intermediary bank, transfer, network and currency conversion charges, so that the full amount invoiced is paid. We bear our payment processors' own fees.

3.5. **Cleared payment.** A payment counts once we have verified it with its processor or bank: a payment under clause 3.3(a) when Toss Payments reports it approved or, for a virtual account, reports the deposit; a bank transfer under clause 3.3(b) when it reaches our account; and a stablecoin payment when CoinPayments reports it completed. A payment under clause 3.3(a) counts only for the exact amount of its order: a payment of any other amount is cancelled and returned through Toss Payments, and a virtual account that is not paid by the deadline Toss Payments shows lapses with its order. If CoinPayments completes a stablecoin payment for less than the invoice total, the payment counts for the amount received, as CoinPayments values it in US dollars, and the HC it issues is reduced in proportion. A stablecoin payment that CoinPayments does not complete, an amount above the invoice total, and a payment that arrives after CoinPayments' payment window are returned through CoinPayments' refund procedure.

3.6. **Refunds.** A refund that this Agreement provides or the law requires is made by the method and through the processor of the payment it refunds, in the currency of that payment: for a payment under clause 3.3(a), through Toss Payments to the card or account it was paid from, or, for a virtual account, to an account at a Korean bank that Customer names; for a bank transfer under clause 3.3(b), to the account it came from; and for a stablecoin payment, through CoinPayments' refund procedure.

## 4. Fleets and maintenance

4.1. We provide a service, not hardware. We keep ownership, possession and operation of every Fleet.

4.2. **Availability.** We publish live capacity in machine-readable form at https://sparkpipe.ai/availability.txt, regenerated at least every 30 seconds: each Fleet we operate, whether it is serving, and whether it is the Demo Fleet, pinned (clause 15) or available. Wherever this Agreement makes anything subject to availability, availability means what that publication shows when Customer books.

4.3. **Maintenance.** We may take each Fleet out of service for maintenance for up to the maintenance allowance we publish in each Week. We announce each maintenance period at least 7 days ahead, choose the times of least use, and rotate those times across the hours of the day. A booked hour under maintenance is credited under clause 4.4 as if the Fleet were unavailable throughout it, unless we serve Customer on another Fleet in that hour under clause 15.7.

4.4. **Automatic credit.** If a Fleet that Customer holds is unavailable, or unreachable, for more than 15 consecutive minutes, every hour of its Booking in which any part of that period falls is credited within 24 hours after the period ends, from our own records, as HC equal to that Fleet's Rate for each such hour. Where Customer's records show such a period and ours do not, Customer may send them within 7 days, and we credit the hours unless our records show the Fleet was available throughout.

4.5. Credits under clauses 4.4 and 6.6 are made in HC, never in cash, and are Customer's only remedy for any failure of performance, subject to clause 8.5.

## 5. Use and responsibility

5.1. **We provide capacity only.** Customer has the exclusive right to use each Fleet it holds for the time of its Booking. Customer alone decides what runs on the Fleets it holds and in the Functions it runs, and is solely responsible for its workloads, the data and content it processes, the outputs it obtains and every use made of them. We do not monitor, review or control them and are not responsible for them. Customer is responsible, as for its own acts, for everyone who uses a Fleet under its Account, access tokens or API keys, including its end users and anyone it gives access to, and must bind those persons by terms no less protective of us than clauses 5 and 6. We take part in no resale or transfer of Bookings or access: we keep no register of it, are not a party to it, do not match, price, clear or settle it, and have no liability for its price or counterparties.

5.2. **Lawful use.** Customer must comply, and must ensure that every user of the Fleets it holds complies, with every law that applies to the use, including laws on export control and sanctions, personal data, intellectual property and computer misuse, and with the licence of each model it uses. No Booking or Function may be used to: (a) create, store or distribute unlawful content, including child sexual abuse material; (b) attack, probe or disrupt any system or network, or distribute malicious code; (c) infringe the rights of any person; (d) benefit any person, entity or country subject to sanctions of the United Nations, the Republic of Korea, the United States or the European Union, or serve any end use that export control laws prohibit, including weapons of mass destruction; or (e) interfere with a Fleet or other customers. Customer represents that neither it nor any person that owns or controls it is subject to those sanctions. Where Customer uses its Bookings to provide generative AI products or services to others, Customer is their provider and bears the duties that apply to them, including the transparency and labelling duties of the Korean Framework Act on the Development of Artificial Intelligence and Establishment of Trust.

5.3. **Suspension.** We may suspend a Booking or access under the Account, at once and without prior notice, where (a) a law, court or competent authority requires it, (b) we reasonably believe clause 5.2 is being breached, or (c) the use threatens the security or integrity of a Fleet or others. We tell Customer as soon as practicable and lift the suspension when its cause ends. Hours lost to a suspension caused by Customer's use are consumed and are not credited.

## 6. Customer data

6.1. Customer keeps all rights in its data and outputs, including models it trains with a Function. We access them only as needed to operate and secure the service, or where the law requires. Customer is responsible for the lawfulness of any personal data it processes and for keeping its own copies. The Fleets and our servers are in the Republic of Korea (clause 6.5).

6.2. **Working state.** Customer's working state is the cached state of its requests and the files that Customer, and the Functions it runs, keep in the file storage of a Fleet it holds (its **Customer storage**, of the size the Fleet List states). Working state stays on that Fleet, isolated from other customers, from one Booking of Customer's to the next, and is deleted when another customer's Booking of that Fleet starts, or 24 hours after the end of the last hour Customer holds on that Fleet, whichever comes first, or earlier when Customer asks.

6.3. **Working state is temporary.** Customer must treat its working state as temporary, also while it holds a Fleet for many Weeks ahead, and keep its own copies of anything it wants to keep. We recommend exporting Customer storage at least daily (clause 7.5). Working state may be lost to a failure of the Fleet, an event under clause 8.7, maintenance or any other cause, and nothing is returned when a Booking or this Agreement ends. We keep no copy of working state outside the Fleet that holds it, make no backup of it and use it for nothing but serving Customer. Before another customer's Booking of a Fleet starts, we restore that Fleet's software to our own image.

6.4. We tell Customer without delay of any intrusion, any leak of its information and any unannounced interruption of the service. We handle account data and Customer data as our published privacy policy states.

6.5. **In Korea only.** We process requests and Customer data only on Fleets and servers that we operate in Korea, including servers we rent there, and we send no request, prompt, output or other Customer data to any other computing or AI service, in Korea or abroad.

6.6. **Rebuild credit.** If an event on a Fleet that Customer holds destroys every copy we keep of part of Customer's working state on that Fleet, so that its cached state has to be rebuilt, we credit Customer, once for each such event, HC equal to 24 hours of that Fleet at its Rate, within 24 hours after we find the loss, and we tell Customer what was lost. This credit, with any credit under clause 4.4 for the same event, is Customer's only remedy for the loss of working state, whatever its cause, subject to clause 8.5.

## 7. Models and Functions

7.1. We offer only open-source models whose availability cannot be withdrawn by any third party, and list the models available on each Fleet on the Fleet List. A model listed when a Booking is made stays available for that Booking. Other models are available while they are listed. Customer uses the Fleets it holds through our API, as our website documents it, with API keys it issues under its Account, which are access tokens for the purposes of clause 2 and which Customer may revoke at any time. We may add to and change the API without amending this Agreement.

7.2. **Model licences and outputs.** Each model is third-party open-source software provided under its own licence, which Customer must comply with. Outputs are generated automatically from Customer's inputs: we do not author, review or endorse them, and we give no warranty that any output is accurate, complete, lawful, safe or fit for any purpose. Customer is responsible for evaluating outputs before relying on, publishing or acting on them.

7.3. Bookings cover the use of the Fleet, the models in this clause 7, the Functions, Customer storage, and operation and support; no other fee applies to them.

7.4. **Functions.** We offer, as our website documents them, software functions that Customer may run on Fleets it holds (**Functions**). A Function runs only when Customer starts it or schedules it, on a Fleet Customer holds, and processes only Customer's data; it uses that Booking like any other use of the Fleet. What a Function produces, including any model it trains, is Customer's output under clause 6.1, is kept as working state and is used only for Customer's own requests. Functions are provided as available, with no guarantee of any result or speed. We may add, change or withdraw Functions without amending this Agreement.

7.5. **Export and restore.** Customer may export the files in its Customer storage, including any model a Function trained, through our API while they are kept, and restore them to the Customer storage of any Fleet it holds.

7.6. **Customer's models.** Customer may add to its Customer storage a model it has the right to use, with a driver for it written to the interface our website documents. Such a model is not offered by us under clause 7.1: it is available only to Customer, on Fleets Customer holds, is Customer data kept as working state, and its licence and outputs are Customer's responsibility as clause 7.2 provides. We give no guarantee that it runs or runs at any speed, and we may refuse or stop a driver that threatens the security or integrity of a Fleet (clause 5.3).

## 8. Liability

8.1. Our aggregate liability for all causes combined is capped at the amounts we received from Customer for HC in the 6 months before the event giving rise to the claim.

8.2. We are not liable for lost profit, lost data, lost opportunity, business interruption, or indirect or consequential loss, whether or not foreseeable.

8.3. Customer indemnifies us, our officers and our employees against any claim, loss, liability, fine, penalty or cost, including reasonable legal fees, arising from (a) the use of the Fleets Customer holds, or of Customer's Account, access tokens or API keys, by anyone, including Customer, its end users and anyone it gives access to, (b) its workloads, data, content and outputs, (c) any breach of clause 5 or 6, or (d) any third-party claim, or investigation by an authority, connected with them. We give no reciprocal indemnity.

8.4. The service, the models and the Functions are provided "as available". We give no warranty of merchantability, fitness for purpose, accuracy of outputs or uninterrupted service beyond clause 4.4.

8.5. Nothing in this Agreement limits or excludes liability for intentional misconduct or gross negligence, or any other liability that cannot lawfully be limited or excluded.

8.6. Clauses 8.1 and 8.2 limit our liability only. They do not limit Customer's payment obligations or its liability under clauses 3.2, 5, 6 and 8.3.

8.7. **Force majeure.** Neither party is liable for a delay or failure caused by an event beyond its reasonable control, including natural disaster, fire, flood, war, terrorism, epidemic, action of a government or authority, or failure of public power or telecommunications networks. Clause 4.4 still applies to booked hours lost to such an event, and clause 6.6 to working state lost to it; they are Customer's only remedies for them.

## 9. Changes and published settings

9.1. We may change this Agreement by publishing a new version and emailing Customer. A new version applies to Orders and Bookings made after Customer accepts it, and we accept no further Order or Booking from Customer until it has. HC already bought, and Bookings already made, stay under the version in force when they were made, unless Customer accepts a newer version for them.

9.2. **Published settings.** We publish the following on our website as **published settings**: the HC Price for each quantity bought (clause 13.1); the HC for an Hour and for a Week, and the discounts for Weeks booked ahead (clauses 14.2 and 15.2); the Rates on the Fleet List; how far ahead Hours may be booked and how many Hours of the Demo Fleet each account may hold in a Week (clause 14.1); and the maintenance allowance. We may change any of them at any time by publishing the change, without amending this Agreement, and a change applies from when we publish it. A change never alters the price of HC already bought or of a Booking already made, nor the HC locked under clause 16.

## 10. Ending, refunds and survival

10.1. Customer may close its Account at any time while it has no Booking that has not ended. Its HC then lapses, and HC locked under clause 16 is unlocked first and lapses with it.

10.2. We may close the Account and end this Agreement by written notice: (a) if Customer materially breaches this Agreement and does not remedy the breach within 30 days after our written notice; or (b) at once, where a breach of clause 5.2 cannot be remedied, where Customer or a person that owns or controls it becomes subject to the sanctions in clause 5.2(d), or where a law, court or competent authority requires us to stop serving Customer.

10.3. **Right of use only; no refunds.** HC and Bookings give the right to use the service only. They give no ownership of, or rights in, our company, hardware, software, models or intellectual property, are not a deposit, loan, security or investment, carry no expectation of profit from our efforts, and have no cash value. HC and Bookings cannot be moved to another account or withdrawn. **Every amount paid is non-refundable and non-cancellable**, except under clause 10.4 or 16.4 or where the law requires. Customer may not set off, deduct, counterclaim or withhold against any amount. We do not buy back HC or Bookings.

10.4. If we stop offering service for HC, or close an Account other than for Customer's breach, we refund, at the price paid and within 60 days, the purchased HC not yet spent or lapsed and the HC spent on Bookings for hours that have not yet occurred, counting HC as spent from the oldest lot first. HC credited under clause 4.4 or 6.6 is not refunded.

10.5. Clauses 3.2, 3.4, 5, 6, 7.2, 8, 10, 11 and 12 survive the end of this Agreement.

## 11. General

11.1. **Entire agreement.** This Agreement, with Customer's Orders and any written order under clause 1.2, is the whole agreement between the parties. Neither party relies on any pre-contractual statement, projection, benchmark or representation.

11.2. **Amendment and waiver.** This Agreement changes only as clause 9 provides. A written order is made, and changed, only in writing signed by both parties. No conduct, delay or extension operates as a waiver.

11.3. **Third parties.** This Agreement gives no rights to anyone except the parties.

11.4. **Confidentiality.** Each party keeps confidential the other's non-public information that it receives in connection with a written order under clause 1.2, including the order itself, for 3 years after it ends, and pricing and technical information indefinitely.

11.5. **Assignment.** Customer may not assign this Agreement, its HC or a Booking without our written consent. Letting others use the Fleets it holds under clause 5.1 is not an assignment. We may assign this Agreement freely, including to an acquirer or financier.

11.6. **Notices.** We send notices to the email address verified on the Account, which Customer keeps current. Customer sends notices to hyor@sparkpipe.ai or to our address above.

11.7. **Records.** Our technical records of payments, HC, Bookings, the waitlist and credits are kept in an append-only, hash-chained ledger. Customer may download its own entries, with their hashes, while signed in to its Account, and obtain proof that they were recorded and not altered. These records are conclusive, except for manifest error.

## 12. Law, disputes, language and acceptance

12.1. This Agreement is governed by the laws of the Republic of Korea.

12.2. Any dispute arising out of or in connection with this Agreement is finally settled by arbitration in Seoul under the Arbitration Rules of the Korean Commercial Arbitration Board, before a sole arbitrator, in the Korean language. Either party may apply to a competent court for interim measures.

12.3. This Agreement is published in English and Korean. The Korean text prevails.

12.4. **Acceptance.** Customer accepts this Agreement by ticking the acceptance box while signed in to its Account. We record the SHA-256 hashes of the exact Korean and English texts accepted, with the time, the IP address and the browser. That acceptance has the same effect as a handwritten signature.

## Part 2. Hour-credits

## 13. Hour-credits

13.1. **Buying HC.** The price of one HC is the **HC Price** that our website shows, when Customer buys it, for the quantity bought in that Order: in Korean won for payment in won, otherwise in US dollars. It is fixed for the HC bought. Customer buys any quantity of HC, from the smallest we publish, with no maximum. HC is issued when the payment counts under clause 3.5.

13.2. **Balance.** HC is counted to six decimal places. Each purchase, and each credit under this Agreement, is added to Customer's balance as a separate lot.

13.3. **Lapse.** HC is spent when Customer books with it, and is spent from the lot that lapses first. Each lot lapses 12 months after it is issued with any HC left in it that is not locked under clause 16; HC spent on a Booking does not lapse, whenever the Booking's hours occur. The same applies to HC credited under clause 4.4.

13.4. HC does not guarantee that any Fleet will be free when Customer wishes to book it, except as clause 16 provides for the waitlist.

## Part 3. Bookings

## 14. Hours

14.1. **Booking Hours.** Customer may book whole hours, starting on the hour: on the Demo Fleet, up to as far ahead as we publish and up to the number of Hours we publish for each account in a Week; and on any other serving Fleet that is not pinned and that nobody else holds in those hours, up to the end of the current Week.

14.2. **Price.** Each Hour costs the HC we publish for an Hour, times the Fleet's Rate.

14.3. **Final and consumed.** A Booking exists once our system confirms it and debits its HC, and is then final: it cannot be cancelled, moved or refunded, except under clause 10.4. A booked hour is consumed when it occurs, whether or not it is used. Requests still running at the end of a Booking have 30 seconds to finish. Loading a model counts as part of the Booking.

14.4. **No displacement.** A confirmed Booking is never taken, moved or displaced, and is delivered on the Fleet it was made on, except on a suspension under clause 5.3, where maintenance overlaps it under clause 4.3, or where its Fleet cannot serve under clause 15.7.

## 15. Weeks and pinned Fleets

15.1. **Pinning.** A Week booked on a Fleet pins that Fleet to Customer for the Week. While a Fleet is pinned, it is Customer's alone and nobody else may book any of its hours. The Demo Fleet is never pinned.

15.2. **Price.** Each Week costs the HC we publish for a Week, times the Fleet's Rate, less the discount we publish for the number of Customer's booked Weeks on that Fleet that have not ended, counting the Weeks being booked. The discount applies to the Weeks being booked.

15.3. **First Week.** Customer may book a first Week, and any Weeks following it, starting on the next Monday, on a serving Fleet other than the Demo Fleet that is available for that Week, that clause 16 has not allocated, and for which no customer on the waitlist has locked enough HC to pay for a Week. The waitlist comes first: while such a customer waits, a free Fleet is allocated under clause 16.3.

15.4. **Extending.** Customer may book further Weeks on a Fleet pinned to it, each following a Week already booked, and no Week more than 52 Weeks after the current Week. Nobody else may book any Week of a Fleet pinned to Customer.

15.5. **Keeping a pinned Fleet.** A Fleet stays pinned to Customer from Week to Week while Customer has booked the next Week by 00:00 on the Sunday of its last booked Week. If it has not, the Fleet is allocated at that time under clause 16.3; if it is not allocated, Customer, or any other customer under clause 15.3, may book its next Week until that Week starts.

15.6. **Use.** A booked Week is consumed as it occurs, whether or not it is used, and clauses 14.3 and 14.4 apply to it. Customer needs no further order to use a Fleet pinned to it.

15.7. If a Fleet that Customer holds cannot serve, including under maintenance (clause 4.3), we serve Customer on another Fleet that is free, without its working state, or credit the hours under clause 4.4.

## 16. Waitlist

16.1. **Joining.** Customer may join the waitlist by locking at least the HC we publish for a Week from its balance. Locked HC cannot be spent while it is locked. Customer may add to its lock, or leave the waitlist and unlock all of it, at any time.

16.2. **Order.** The waitlist is ordered by the HC locked, largest first, and at equal locks by the time of joining. We show each customer only its own position.

16.3. **Allocation.** At 00:00 on each Sunday, each serving Fleet other than the Demo Fleet that is free for the following Week, including a Fleet whose holder has not booked that Week under clause 15.5, is allocated to the first customer on the waitlist, then the next Fleet to the next customer, and so on. The allocated customer's whole lock books consecutive Weeks on that Fleet from the following Monday, as many as it pays for under clause 15.2, and any HC left over is unlocked. We email the allocated customer at once.

16.4. **Lapse while waiting.** Where locked HC would lapse under clause 13.3, we unlock it and refund it at the price paid under clause 3.6, instead of letting it lapse.

16.5. We give no guarantee that, or when, any Fleet will become free for the waitlist.
