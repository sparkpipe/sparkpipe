# SparkPipe Hourly Terms

## Parties

These Terms are made between **HYOR AI Tech Ltd.** (효르에이아이테크 유한회사), business registration number 641-87-03809, of Yongsan-gu, Seoul, Republic of Korea, whose full address we give on request ("**we**"), and the business identified in the account that accepts them ("**Customer**").

## 1. The service

1.1. We sell hour-credits and provide, without commitment, the use of our Fleets paid for with them. A **Fleet** is a group of computing machines that we operate as one unit and list on our Fleet List, published at sparkpipe.ai/fleets.html. A **fleet-hour** (or **hour**) is the use of one entire Fleet for sixty minutes. An **hour-credit** (**HC**) is the prepaid unit in which Customer pays for the service. Each Fleet's **Rate**, on the Fleet List, is the number of HC that one hour of our capacity on it costs; it is 1 unless the Fleet List states otherwise. A **Reservation** (or **dated hour**) is the right to use one named Fleet for the sixty minutes from a named start on the hour. The **shared pool** serves individual requests on Fleets that nobody holds in that hour and on released hours that reach their start unsold; its meter is the **seat-second**, one second of one request slot on a pooled Fleet. To **release** a Reservation is to offer it for resale in the **Hour Market** under the Hour Market Rules (clause 13).

1.2. We provide a service, not hardware. We keep ownership, possession and operation of every Fleet.

1.3. **Businesses only.** Customer confirms that it acts in the course of its business and not as a consumer, and that its account details are true and current. We may refuse or close an account that does not meet this.

## 2. Account

2.1. Customer's account is identified by the email address Customer verifies with us, and Customer signs in to it with one-time codes we send to that address. Acceptance of these Terms, orders, Reservations and changes to the account are made while signed in to the account, or through our API with an access token issued under it. Customer may issue access tokens and add users under the account.

2.2. Customer bears all risk of loss or compromise of its account email and access tokens; any use under them is Customer's use. Customer may name a recovery email address, which can replace the account email after a 72-hour delay with notice to the account email.

2.3. We send notices to the email address verified on the account. Customer keeps it current.

## 3. Price, VAT and payment

3.1. The price of one HC is the **HC Price** shown on our website when Customer buys it: in Korean won for payment in won, otherwise in US dollars. It is fixed for the HC bought. Customer buys HC by amount, within the minimum and maximum per purchase shown on our website, and receives the amount paid, net of VAT, divided by the HC Price, rounded down to six decimal places.

3.2. The HC Price is exclusive of value-added tax (VAT). We charge VAT at the statutory rate (10% today) on every payment and show it separately on our invoice. Any amount charged as VAT that is not owed is an overpayment, and we refund it to Customer.

3.3. Customer pays in full and in advance, in one of two ways: (a) if Customer has a Korean business registration number, in Korean won, by bank transfer to the account named on the electronic tax invoice (세금계산서) we issue; or (b) in a US dollar stablecoin, such as USDC or USDT, that our payment processor CoinPayments accepts, on any network it accepts for that coin, through the invoice it issues for us. Customer bears network fees, bank charges and any tax a law requires it to withhold, so that the full amount invoiced is paid. We bear the payment processor's own fees.

3.4. HC is issued once we have verified the payment: for a won transfer when it reaches our account, and for a stablecoin payment when our payment processor reports it completed.

3.5. If our payment processor completes a stablecoin payment for less than the invoice total, we issue HC in proportion to the amount received, as the processor values it in US dollars, rounded down to six decimal places. A payment the processor does not complete, an amount above the invoice total, and a payment that arrives after the processor's payment window are returned through the processor's refund procedure.

## 4. Hour-credits

4.1. **Balance.** HC is counted to six decimal places. Each purchase, and each credit under these Terms or the Hour Market Rules, is added to Customer's balance as a separate lot.

4.2. **Expiry.** Each lot lapses 12 months after it is issued, with any HC left in it. HC is spent from the lot that lapses first. The same applies to HC credited for a released hour or under clauses 5.2 and 6.

4.3. HC and Reservations give the right to use the service only. They are not a deposit, loan, security or investment, carry no expectation of profit, and have no cash value. HC cannot be sent to another account or withdrawn. **Paid amounts are not refunded**, except under clause 11.3 or where the law requires. We do not buy back HC or hours.

## 5. Reservations and the shared pool

5.1. Customer reserves hours through our website while signed in to its account, or through our API with an access token issued under it, naming the Fleet, the start hour and the model, or by a bid under the Hour Market Rules. Customer may reserve (a) an hour of our capacity on a serving Fleet, at its Rate in HC, from the first date we offer and up to 7 days ahead; or (b) an hour another customer has released, at the price the Hour Market Rules set, however far ahead it starts. A Reservation exists once our system confirms it and debits its price from Customer's HC, and is then final: it cannot be cancelled, moved or refunded, but Customer may release it under the Hour Market Rules before it starts.

5.2. **Displacement.** A customer under our 2-Year Reserved Fleet Service Agreement may take an hour Customer has reserved from our capacity (clause 5.1(a)) from Monday to Saturday, within that customer's guaranteed daily limit, if that hour starts more than 6 hours later. An hour Customer bought as a released hour is never taken. The price Customer paid for the hour taken is credited back to Customer at once as HC, and we tell Customer by email. A Reservation is not displaced within 6 hours of its start, and a Reservation on a Sunday is never displaced. Days in these Terms are in Korea Standard Time (UTC+9).

5.3. A reserved hour is consumed when it occurs, whether or not it is used. Requests still running at its end have 30 seconds to finish. Loading the reserved model counts as part of the hour.

5.4. **Shared pool.** Customer may send requests to the models we list for the shared pool, through our OpenAI-compatible endpoint at https://sparkpipe.ai/pay/api/pool/v1, with API keys it issues under its account. API keys are access tokens for the purposes of clause 2, and Customer may revoke one at any time. Each request holds one seat and is charged in HC for its seat-seconds, from when we accept it until its reply ends and whether or not it succeeds, at the price per seat-second posted when it starts (clause 7.2 of the Hour Market Rules), rounded down to six decimal places and never more than Customer's balance. We accept a request only while Customer holds at least 0.0001 HC. The shared pool has no guaranteed capacity: when no Fleet is in it, or its seats are taken, requests are refused. Requests made with Customer's API keys during an hour in which Customer holds a Reservation are not charged.

## 6. Delivery and credits

6.1. **Automatic credit.** If a reserved hour is unavailable, or its Fleet is unreachable, for more than 15 consecutive minutes, every reserved hour in which any part of that period falls is credited back in full within 24 hours, from our own records, to the account that holds the Reservation, as HC equal to the price it paid. Where Customer's records show such a period and ours do not, Customer may send them within 7 days, and we credit the hours unless our records show the Fleet was available throughout.

6.2. Credits are made in HC, never in cash, and are Customer's only remedy for any failure of performance, subject to clause 9.5.

6.3. We publish live capacity at https://sparkpipe.ai/availability.txt, regenerated at least every 30 seconds; availability means what it shows when Customer reserves. Scheduled maintenance takes place only on Sundays and is announced at least 7 days ahead, and those hours cannot be reserved.

## 7. Use and responsibility

7.1. **We provide capacity only.** Customer alone decides what runs on its hours and in its requests to the shared pool, and is solely responsible for its workloads, the data and content it processes, the outputs it obtains and every use made of them. We do not monitor, review or control them. Customer is responsible for everyone who uses hours or the shared pool under its account, access tokens or API keys, including its end users and anyone it resells, transfers or gives hours or access to. The Hour Market is the only resale of hours in which we take part; any other such arrangement happens outside our system and is out of our control. Customer must bind those persons by terms no less protective of us than this clause 7.

7.2. **Lawful use.** Customer must comply, and ensure every user of its hours and of its pool access complies, with every law that applies, including laws on export control and sanctions, personal data, intellectual property and computer misuse, and with each model's licence. No hour, and no request to the shared pool, may be used to: (a) create, store or distribute unlawful content, including child sexual abuse material; (b) attack, probe or disrupt any system, or distribute malicious code; (c) infringe anyone's rights; (d) benefit any person, entity or country subject to sanctions of the United Nations, the Republic of Korea, the United States or the European Union, or serve an end use that export control law prohibits; or (e) interfere with a Fleet or other customers. Customer represents that neither it nor anyone who owns or controls it is subject to those sanctions. Where Customer uses its hours or the shared pool to provide generative AI products or services to others, Customer is their provider and bears the duties that apply to them, including the transparency and labelling duties of the Korean Framework Act on the Development of Artificial Intelligence and Establishment of Trust.

7.3. **Suspension.** We may suspend Reservations or access at once, without prior notice, where a law, court or authority requires it, where we reasonably believe clause 7.2 is being breached, or where use threatens the security of a Fleet or others. We tell Customer as soon as practicable and lift the suspension when its cause ends. Hours lost to a suspension caused by the use of Customer's hours are consumed and not credited.

7.4. **Customer data.** Customer keeps all rights in its data and outputs, and we access them only to operate and secure the service or where the law requires. We hold no Customer data beyond transient caches, deleted within 24 hours, and keep no backups of it. The Fleets and our servers are in Seoul, Republic of Korea. We tell Customer without delay of any intrusion, leak of its information or unannounced interruption. Our privacy policy applies to account data.

## 8. Models

8.1. Each model is third-party open-source software under its own licence, which Customer must follow. Outputs are generated automatically from Customer's inputs: we do not author, review or endorse them and give no warranty that any output is accurate, complete, lawful, safe or fit for a purpose.

8.2. The models available on each Fleet are listed on the Fleet List. A model chosen for a confirmed Reservation stays available for that Reservation.

## 9. Liability

9.1. Our aggregate liability for all causes combined is capped at the amounts we received from Customer in the 6 months before the event giving rise to the claim.

9.2. We are not liable for lost profit, lost data, lost opportunity, business interruption, or indirect or consequential loss, whether or not foreseeable.

9.3. Customer indemnifies us, our officers and our employees against any claim, loss, liability, fine, penalty or cost, including reasonable legal fees, arising from the use of its hours, or of the shared pool under its account, by anyone, its workloads, data, content and outputs, any breach of clause 7, or any third-party claim or investigation by an authority connected with them.

9.4. The service and the models are provided "as available". We give no warranty of merchantability, fitness for purpose, accuracy of outputs or uninterrupted service beyond clause 6.

9.5. Nothing in these Terms limits liability for intentional misconduct or gross negligence, or any liability that cannot lawfully be limited.

9.6. Clauses 9.1 and 9.2 limit our liability only.

9.7. **Force majeure.** Neither party is liable for a delay or failure caused by an event beyond its reasonable control, including natural disaster, fire, flood, war, terrorism, epidemic, action of an authority, or failure of public power or telecommunications networks. Clause 6.1 still applies to reserved hours lost to such an event and is Customer's only remedy for them.

## 10. Changes to these Terms

We may change these Terms by publishing a new version and emailing Customer. A new version applies to purchases made after Customer accepts it; HC already bought stays under the version accepted when it was bought.

## 11. Ending

11.1. Customer may close its account at any time. Its HC and Reservations then lapse, and its orders in the Hour Market are cancelled.

11.2. We may close an account by written notice if Customer materially breaches these Terms and does not remedy the breach within 30 days, or at once for an unremediable breach of clause 7.2, sanctions, or a legal order to stop serving Customer.

11.3. If we stop offering service under these Terms, or close an account other than for Customer's breach, we refund the purchased HC not yet spent or lapsed at the price paid, within 60 days, counting HC as spent from the oldest lot first. HC credited for a released hour is not refunded.

11.4. Clauses 3.2, 4.3, 7, 8.1, 9, 11 and 12 survive the end of these Terms.

## 12. Law, disputes and language

12.1. These Terms are governed by the laws of the Republic of Korea.

12.2. Any dispute arising out of or in connection with these Terms is finally settled by arbitration in Seoul under the Arbitration Rules of the Korean Commercial Arbitration Board, before a sole arbitrator, in the Korean language. Either party may apply to a competent court for interim measures.

12.3. These Terms are published in English and Korean. The Korean text prevails.

12.4. Customer accepts these Terms, which include the Hour Market Rules, by ticking the acceptance box while signed in to its account. We record the SHA-256 hashes of the exact Korean and English texts of these Terms and of the Hour Market Rules accepted, with the time, the IP address and the browser. That acceptance has the same effect as a handwritten signature.

## 13. Hour Market Rules

13.1. Our SparkPipe Hour Market Rules, published at sparkpipe.ai/market-rules.html, form part of these Terms. They govern the release of Reservations, bids, the matching and settlement of trades, the price of the shared pool and its use of released hours, and the release fee. On those matters they prevail over these Terms.
