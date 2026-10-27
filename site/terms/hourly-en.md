# SparkPipe Hourly Terms

## Parties

These Terms are made between **HYOR AI Tech Ltd.** (효르에이아이테크 유한회사), business registration number 641-87-03809, of Yongsan-gu, Seoul, Republic of Korea, whose full address we give on request ("**we**"), and the business identified in the account that accepts them ("**Customer**").

## 1. The service

1.1. We provide prepaid fleet-hours on our Fleets, without commitment. A **fleet-hour** (or **hour**) is the use of one entire Fleet for sixty minutes. A **Fleet** is a group of computing machines that we operate as one unit and list on our Fleet List, published at sparkpipe.ai/fleets.html. Each Fleet's **Rate**, on the Fleet List, is the number of hours one hour on it consumes; it is 1 unless the Fleet List states otherwise.

1.2. We provide a service, not hardware. We keep ownership, possession and operation of every Fleet.

1.3. **Businesses only.** Customer confirms that it acts in the course of its business and not as a consumer, and that its account details are true and current. We may refuse or close an account that does not meet this.

## 2. Account and Key

2.1. Customer's account is identified by an Ethereum address. The **Key** is that address's private key. Acceptance of these Terms, Reservations and changes to the account are messages signed with the Key. Customer may issue access tokens and add signers under it.

2.2. Customer bears all risk of loss or compromise of its Key and access tokens; any use under them is Customer's use. Customer may name a recovery address, which can replace the Key after a 72-hour delay with notice to the account email.

2.3. We send notices to the email address verified on the account. Customer keeps it current.

## 3. Price, VAT and payment

3.1. The price of an hour is the **Hourly Price** shown on our website when Customer buys it: in Korean won for payment in won, otherwise in US dollars. It is fixed for the hours bought.

3.2. The Hourly Price is exclusive of value-added tax (VAT). We charge VAT at the statutory rate (10% today) on every payment and show it separately on our invoice. Any amount charged as VAT that is not owed is an overpayment, and we refund it to Customer.

3.3. Customer pays in full and in advance, in one of two ways: (a) if Customer has a Korean business registration number, in Korean won, by bank transfer to the account named on the electronic tax invoice (세금계산서) we issue; or (b) in USDC, through the payment invoice our system issues. Customer bears the payment service's fees, network fees, bank charges and any tax a law requires it to withhold, so that we receive the full amount invoiced.

3.4. Hours are issued once we have verified the full payment: for a won transfer when it reaches our account, and for USDC when the payment service confirms it.

## 4. Hours

4.1. **Holding limit.** Customer may hold at most 168 hours (one week) at a time, counting unused hours and reserved hours that have not yet occurred. Once we approve Customer's business after our verification, the limit is 720 hours. We may change these limits for future purchases.

4.2. Hours may be used from the start date chosen at purchase, which may not be earlier than the first date we offer. They expire 12 months after that start date.

4.3. Hours give the right to use the service only. They are not a deposit, loan, security or investment, carry no expectation of profit, and have no cash value. **Paid amounts are not refunded**, except under clause 11.3 or where the law requires. We do not buy back hours.

## 5. Reservations

5.1. Customer reserves hours through our website or API by a message signed with its Key, naming the Fleet, the start hour, the number of hours and the model. Reservations may be made up to 7 days ahead. A Reservation exists once our system confirms it and is then final: it cannot be cancelled or moved.

5.2. **Displacement.** A customer under our 2-Year Reserved Fleet Service Agreement may take an hour Customer has reserved, if that hour starts more than 6 hours later. The hour returns to Customer's balance at once, outside the holding limit, and expires at the later of its original expiry or 30 days later; we tell Customer by email. A Reservation is not displaced within 6 hours of its start.

5.3. A reserved hour is consumed when it occurs, whether or not it is used. Requests still running at its end have 30 seconds to finish. Loading the reserved model counts as part of the hour.

## 6. Delivery and credits

6.1. **Automatic credit.** If a reserved hour is unavailable, or its Fleet is unreachable, for more than 15 consecutive minutes, every reserved hour in which any part of that period falls is credited back in full within 24 hours, from our own records. Where Customer's records show such a period and ours do not, Customer may send them within 7 days, and we credit the hours unless our records show the Fleet was available throughout.

6.2. Credits are made in hours, never in cash, and are Customer's only remedy for any failure of performance, subject to clause 9.5.

6.3. We publish live capacity at https://sparkpipe.ai/availability.txt, regenerated at least every 30 seconds; availability means what it shows when Customer reserves. Maintenance is announced at least 7 days ahead, and those hours cannot be reserved.

## 7. Use and responsibility

7.1. **We provide capacity only.** Customer alone decides what runs on its hours and is solely responsible for its workloads, the data and content it processes, the outputs it obtains and every use made of them. We do not monitor, review or control them. Customer is responsible for everyone who uses hours under its Key or access tokens, including its end users and anyone it resells, transfers or gives hours to. Such arrangements happen outside our system and are out of our control. Customer must bind those persons by terms no less protective of us than this clause 7.

7.2. **Lawful use.** Customer must comply, and ensure every user of its hours complies, with every law that applies, including laws on export control and sanctions, personal data, intellectual property and computer misuse, and with each model's licence. No hour may be used to: (a) create, store or distribute unlawful content, including child sexual abuse material; (b) attack, probe or disrupt any system, or distribute malicious code; (c) infringe anyone's rights; (d) benefit any person, entity or country subject to sanctions of the United Nations, the Republic of Korea, the United States or the European Union, or serve an end use that export control law prohibits; or (e) interfere with a Fleet or other customers. Customer represents that neither it nor anyone who owns or controls it is subject to those sanctions. Where Customer uses its hours to provide generative AI products or services to others, Customer is their provider and bears the duties that apply to them, including the transparency and labelling duties of the Korean Framework Act on the Development of Artificial Intelligence and Establishment of Trust.

7.3. **Suspension.** We may suspend Reservations or access at once, without prior notice, where a law, court or authority requires it, where we reasonably believe clause 7.2 is being breached, or where use threatens the security of a Fleet or others. We tell Customer as soon as practicable and lift the suspension when its cause ends. Hours lost to a suspension caused by the use of Customer's hours are consumed and not credited.

7.4. **Customer data.** Customer keeps all rights in its data and outputs, and we access them only to operate and secure the service or where the law requires. We hold no Customer data beyond transient caches, deleted within 24 hours, and keep no backups of it. The Fleets and our servers are in Seoul, Republic of Korea. We tell Customer without delay of any intrusion, leak of its information or unannounced interruption. Our privacy policy applies to account data.

## 8. Models

8.1. Each model is third-party open-source software under its own licence, which Customer must follow. Outputs are generated automatically from Customer's inputs: we do not author, review or endorse them and give no warranty that any output is accurate, complete, lawful, safe or fit for a purpose.

8.2. The models available on each Fleet are listed on the Fleet List. A model chosen for a confirmed Reservation stays available for that Reservation.

## 9. Liability

9.1. Our aggregate liability for all causes combined is capped at the amounts we received from Customer in the 6 months before the event giving rise to the claim.

9.2. We are not liable for lost profit, lost data, lost opportunity, business interruption, or indirect or consequential loss, whether or not foreseeable.

9.3. Customer indemnifies us, our officers and our employees against any claim, loss, liability, fine, penalty or cost, including reasonable legal fees, arising from the use of its hours by anyone, its workloads, data, content and outputs, any breach of clause 7, or any third-party claim or investigation by an authority connected with them.

9.4. The service and the models are provided "as available". We give no warranty of merchantability, fitness for purpose, accuracy of outputs or uninterrupted service beyond clause 6.

9.5. Nothing in these Terms limits liability for intentional misconduct or gross negligence, or any liability that cannot lawfully be limited.

9.6. Clauses 9.1 and 9.2 limit our liability only.

9.7. **Force majeure.** Neither party is liable for a delay or failure caused by an event beyond its reasonable control, including natural disaster, fire, flood, war, terrorism, epidemic, action of an authority, or failure of public power or telecommunications networks. Clause 6.1 still applies to reserved hours lost to such an event and is Customer's only remedy for them.

## 10. Changes to these Terms

We may change these Terms by publishing a new version and emailing Customer. A new version applies to purchases made after Customer accepts it; hours already bought stay under the version accepted when they were bought.

## 11. Ending

11.1. Customer may close its account at any time. Unused hours then lapse.

11.2. We may close an account by written notice if Customer materially breaches these Terms and does not remedy the breach within 30 days, or at once for an unremediable breach of clause 7.2, sanctions, or a legal order to stop serving Customer.

11.3. If we stop offering hourly service, or close an account other than for Customer's breach, we refund the paid hours not yet used at the price paid, within 60 days.

11.4. Clauses 3.2, 4.3, 7, 8.1, 9, 11 and 12 survive the end of these Terms.

## 12. Law, disputes and language

12.1. These Terms are governed by the laws of the Republic of Korea.

12.2. Any dispute arising out of or in connection with these Terms is finally settled by arbitration in Seoul under the Arbitration Rules of the Korean Commercial Arbitration Board, before a sole arbitrator, in the Korean language. Either party may apply to a competent court for interim measures.

12.3. These Terms are published in English and Korean. The Korean text prevails.

12.4. Customer accepts these Terms by signing the hash of their exact text with its Key. That signature has the same effect as a handwritten signature.
