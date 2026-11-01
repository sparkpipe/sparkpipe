# SparkPipe Hour Market Rules

## Parties

These Rules are made between **HYOR AI Tech Ltd.** (효르에이아이테크 유한회사), business registration number 641-87-03809, of Yongsan-gu, Seoul, Republic of Korea, whose full address we give on request ("**we**"), and each company, professional or independent developer whose account takes part in the Hour Market ("**Customer**").

## 1. Scope

1.1. These Rules govern the **Hour Market**, in which Reservations are released, bid for and bought with hour-credits, and the shared pool's use of released hours that remain unsold. They form part of our SparkPipe Customer Agreement (the "**Customer Agreement**") and are accepted with it. Words defined in the Customer Agreement have the same meaning in these Rules.

1.2. **Participants.** Only businesses with an account under the Customer Agreement take part, each acting for itself.

1.3. **One counterparty.** Customers do not trade with each other: every sale is settled with us under clause 6. HC never moves between accounts and is never paid out.

## 2. Release

2.1. The holder of a Reservation that has not started may release it, at **list**, which is the Fleet's Rate in HC per hour, or at a minimum price it sets of more than 0 and at most 1,000 HC per hour. This applies equally to a Reservation bought in the Hour Market and to one made with Plan hours under a Plan Order.

2.2. A release rests as an ask until it is sold or its hour starts. Until then Customer may cancel it, and the hour is again Customer's Reservation, which Customer may use or release again.

2.3. Customer cannot buy its own released hour.

## 3. Bids

3.1. A **bid** is an order to reserve hours within a window. It names the model, a window that starts and ends on the hour, the number of hours wanted, at least one and at most the hours in the window, and the most Customer will pay per hour, more than 0 and at most 1,000 HC; it may also name a Fleet. A **consecutive** bid fills only with that number of unbroken hours on one Fleet, all at once. A **standing bid** repeats every week.

3.2. A window must end after it starts and in the future, and may span at most 31 days, or 7 days for a standing bid. A bid fills only with hours that start after the current hour.

3.3. **Hold.** Placing a bid holds its maximum price times its number of hours from Customer's balance, which must cover that amount. As each hour fills, its price is paid from the hold and the rest of that hour's maximum returns to the balance. When the window ends, or Customer cancels the bid, the HC still held returns to the balance. HC is held from Customer's lots in the order of clause 13.3 of the Customer Agreement and returns to the lots it came from; HC returned to a lot that has lapsed lapses with it.

3.4. When a standing bid's window ends, the window moves forward by whole weeks to the first one that has not ended, and the bid holds again. If Customer's balance cannot cover that hold, the bid is paused and fills no more.

3.5. Customer may cancel a bid at any time. Hours already filled remain Customer's Reservations.

3.6. **Reserving.** Choosing an hour on the calendar, or reserving it through our API, buys the ask for that hour at its price. Each hour has at most one ask: ours, or its holder's release. Customer may state the most it will pay, and the purchase is refused if the price is higher.

## 4. Our capacity

4.1. Every hour of a serving Fleet that nobody holds rests as our ask, from the first date we offer and up to 31 days ahead, at the Fleet's Rate under clause 14.1 of the Customer Agreement. Hours further ahead are offered only as released hours. Capacity kept under clause 4.3 is offered only as that clause allows, hours under maintenance announced under clause 4.6 of the Customer Agreement are not offered, and an hour that clause 14.2 of the Customer Agreement keeps for HC that lapses sooner can be bought only with such HC until 6 hours before it starts.

4.2. We never bid, and we never use information about any Customer's orders to trade against it.

4.3. **Capacity kept for Plan Orders.** Until 6 hours before each hour starts, we keep off sale the capacity that Plan Orders guarantee their holders (clause 18 of the Customer Agreement). From then on, any of it they have not reserved rests as our ask under clause 4.1. A locked Fleet is never offered (clause 19 of the Customer Agreement).

4.4. **HC on sale.** We sell HC only as far as clause 13.4 of the Customer Agreement allows, and publish the HC still on sale on our website and under clause 11.

## 5. Matching

5.1. Matching is continuous. Each new order is matched at once against the orders resting in the Hour Market, and resting bids are matched again as new asks appear, as our capacity comes within the reach of clause 4.1, and as capacity kept under clause 4.3 is released 6 hours before it starts.

5.2. An order fills from the cheapest asks within its limits. At equal prices, released hours fill before our capacity, and earlier releases before later ones; otherwise earlier hours fill first. Older bids are matched before newer ones. A consecutive bid takes the run of hours with the lowest total price, and of equal runs the earliest.

5.3. A trade with our capacity is made at our ask: the Fleet's Rate (clause 4.1). A trade between a bid and a release is made at the price of the order that was resting first: the release's minimum price if the release came first, and the bid's maximum price if the bid came first.

## 6. Settlement

6.1. When an order matches, in one step: the buyer's HC is debited, or paid from its bid's hold; the hour becomes the buyer's Reservation under the Customer Agreement, for the model the buyer named; and, for a released hour, the seller is credited the price less the release fee (clause 9), rounded down to six decimal places, as a new lot of HC under clause 13.3 of the Customer Agreement.

6.2. In form, the seller surrenders the released hour to us, and we supply it to the buyer. We buy no released hour and pay no cash for one: the seller is credited only HC under clause 6.1.

6.3. A settled trade is final, except under clause 8.

## 7. Unsold hours and the shared pool

7.1. A released hour still unsold when it starts joins the shared pool for that hour, and its release ends.

7.2. **Pool price.** The price of a seat-second in the shared pool is 1 HC divided by the seat-seconds of one pooled Fleet in an hour, which are the seats per Fleet we publish times 3,600, multiplied by a factor between the minimum and maximum we publish that we set about once a minute from the share of the pool's seats used in the minute before. We post the current price for each model on our website and under clause 11.

7.3. The HC charged for the seat-seconds of pool requests in an hour is allocated first to the released hours in the pool, earliest release first, each up to its Fleet's seat-seconds for the hour, and then to our own Fleets.

7.4. After the hour ends, the seller of a released hour in the pool is credited the HC allocated to it, less the release fee, rounded down to six decimal places, as a new lot of HC under clause 13.3 of the Customer Agreement. The credit may be zero.

7.5. A pool request that uses an accelerator holds no seat, and pays no seat-seconds, while it waits for or is computed on the accelerator. It takes a seat when the accelerator hands its computed prompt back to the Fleet, or when its prompt is computed on its Fleet without an accelerator under clause 4.11 of the Customer Agreement. A pool request for a model that its Fleet must first load holds no seat, and pays no seat-seconds, while the Fleet loads that model, and takes a seat when the Fleet serves it.

## 8. Prearranged trades

8.1. Customers may not prearrange trades between accounts, such as a release and a bid timed or priced to meet each other, or trade with themselves through several accounts. We may void such a trade: the buyer's HC is returned, the hour returns to the seller if it has not started, and the HC credited to the seller is withdrawn from its balance, as far as the balance allows. We may also close the accounts involved under clause 10.2 of the Customer Agreement.

## 9. Release fee

9.1. We deduct a **release fee** from each sale of a released hour and from each pool credit for one: a percentage of the price or of the HC allocated, which we publish on our website. Each release bears the release fee published when Customer releases the hour. Releasing, bidding and cancelling are free.

9.2. The release fee is a published setting under clause 9.3 of the Customer Agreement. A change of it applies to every hour released after we publish it, including an hour reserved with Plan hours.

## 10. Records

10.1. Every order, cancellation, trade, hold, credit, pool charge and accelerator charge is recorded as a line in our append-only, hash-chained ledger. Customer may download its own ledger lines, with their hashes, while signed in to its account.

10.2. These records are conclusive, except for manifest error.

## 11. Market data

11.1. We publish, without any account number, name or other identity: for each hour of each Fleet up to 31 days ahead, the best ask, whether it is ours or a release, and the best bid; the HC still on sale (clause 4.4); the open releases and bids; recent trades, with their Fleet, hour, price and kind; the shared pool's prices, seats and Fleets; and the accelerators, each with its Rate, speed and whether it is serving, and the live queue, with each waiting prompt's queue bid and estimated accelerator-seconds. They are on our website and at https://sparkpipe.ai/pay/api/market/market.txt and https://sparkpipe.ai/pay/api/market/pool.txt.

## 12. Suspension

12.1. We may suspend the Hour Market, in whole or in part. Resting releases and bids are then cancelled, released hours return to their holders as Reservations, and HC held for bids returns to the balance. Reservations and settled trades are not affected.

## 13. Changes, law and language

13.1. We change these Rules as clause 9.1 of the Customer Agreement provides for that Agreement, and the release fee as clause 9.2 provides. For an hour reserved under a Plan Order, the version in force when Customer releases it applies.

13.2. These Rules are governed by the laws of the Republic of Korea. Disputes under them are settled as clause 12.2 of the Customer Agreement provides.

13.3. These Rules are published in English and Korean. The Korean text prevails.
