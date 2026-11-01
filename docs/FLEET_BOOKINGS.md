# Fleet bookings: hour-credits, Hours, Weeks and the waitlist

This is the design of the payment and booking service for the elite-developer offer. The SparkPipe Customer Agreement (`site/terms/agreement-*.md`) governs it, and this document follows its clause numbers. The published settings live in `site/pricing.js`.

## What a customer buys

- **Hour-credits (HC).** HC is bought in any quantity from `hc.min`, with no maximum.
  - **Price per HC** depends on the quantity in one purchase: `hc.prices` lists the price from each quantity up (today $20 or ₩26,000 below 144 HC, and $15 or ₩19,500 from 144 HC).
  - **Lots.** Each purchase is a lot that lapses `hc.validityMonths` (12) after it is issued.
  - **Spending.** HC is spent when a booking is made, from the lot that lapses first. HC spent on a booking never lapses, however far ahead the booking runs (clause 13.3).
- **Hours** (clause 14). These are whole hours on one Fleet, each costing `bookings.hourHC` × Rate.
  - **On the Demo Fleet** (the fleet with `"demo": true`), hours can be booked up to `bookings.demoDaysAhead` days ahead. Each account may hold at most `bookings.demoHoursPerWeek` Demo Fleet hours in one Week.
  - **On any other serving Fleet** that is not pinned and that nobody holds in those hours, hours can be booked only up to the end of the current Week (Sunday 24:00 KST). That way an Hour booking never blocks a Monday handover.
- **Weeks** (clause 15). A Week runs from Monday 00:00 to Sunday 24:00 KST, on one Fleet other than the Demo Fleet. A booked Week pins the Fleet: nobody else can book any of its hours or Weeks.
  - **Price per Week:** `bookings.weekHC` (144) × Rate × (1 − discount). The discount is the largest `bookings.discounts` entry whose `weeks` is at most N, where N is the account's booked Weeks on that Fleet that have not ended, counting the current Week and the Weeks being booked. The discount applies only to the Weeks in this booking.
  - **First Week:** starts on the next Monday, on a serving non-demo Fleet that is free for that Week and was not allocated to the waitlist. No Week starts before `weeksFrom`.
  - **Extension:** only the holder may extend, each Week following an already booked Week, with no gaps. The last Week may start at most `bookings.maxWeeksAhead` (52) Weeks after the current Week.
  - **Keeping the Fleet:** if the holder has not booked the next Week by 00:00 on the Sunday of its last booked Week, the Fleet goes to the waitlist allocation at that moment. If nobody is allocated, the holder or anyone else under the first-Week rule may still book that next Week until it starts.
- **Waitlist** (clause 16).
  - **Joining:** lock at least `bookings.weekHC` HC. Locked HC can't be spent. The customer may add to the lock or leave (unlock everything) at any time.
  - **Order:** largest lock first, then earliest join. Adding to a lock keeps the original join time. A customer sees only their own position.
  - **Allocation:** at 00:00 every Sunday KST, each serving non-demo Fleet that is free for the following Week goes to the next customer in order. That customer's whole lock books consecutive Weeks on the Fleet from the following Monday, as many as it pays for at the discount for that count, and any HC left over is unlocked. The customer is emailed.
  - **Lapse while locked:** locked HC that reaches its lapse date is unlocked and recorded as a refund due, and `SPARKPIPE_PAY_NOTIFY` is emailed with the orders to refund under clause 3.6. A lock that falls below `bookings.weekHC` leaves the waitlist.
- **Accelerators** (clauses 4.6–4.8): charged in HC only, as before.
- **Maintenance** (clause 4.3): up to `bookings.maintenanceHoursPerWeek` per Fleet per Week, announced at least 7 days ahead, at the times of least use. Every booked hour it overlaps is credited as HC equal to the Fleet's Rate (clause 4.4).

## Removed

- **The Hour Market:** releases, asks, bids, matching, the release fee and `market.txt`.
- **The shared pool:** per-seat billing, seat prices, settlement of released hours, `pool.txt` and `/pool/v1`.
- **Plans:** Plan Orders, quotes, Monthly and term plans, Tranches, gates, office hours and time zones.
- **The HC cover rule.**
- **The quote-request form.**

## API

Everything is under `/pay/api`, and the sandbox is under `/pay/sandbox/api`. Times are UTC strings in the existing `stamp` and `slot_text` formats.

| Method and path | Auth | Purpose |
| --- | --- | --- |
| `GET /market/state` | optional session | Public state (fleets with `status` `demo`, `pinned` or `free`, prices, booking settings, accelerators, models). With a session it also returns `account`: balance, lots, locked HC, waitlist position, bookings, the account's pinned Fleets with their booked-through date and renewal deadline, API keys and accelerator charges. |
| `POST /market/quote-weeks` | session | `{fleet, weeks}` returns the price, discount, first and last Week. |
| `POST /market/book-hours` | session | `{fleet, start, hours}` books Hours. |
| `POST /market/book-weeks` | session | `{fleet, weeks}` books a first Week (and following Weeks) or extends. |
| `POST /market/waitlist` | session | `{lock}` joins or adds to the lock; returns `{locked, position}`. |
| `POST /market/waitlist/leave` | session | Unlocks everything. |
| `POST /market/keys`, `POST /market/keys/revoke` | session | API keys. |
| `POST /order` | session | `{product: "credits", hc, processor}` buys HC; the price comes from the quantity. |
| `GET /v1/models`, `POST /v1/chat/completions` | API key | Served on the Fleet the key's account holds now (an Hour or a Week). With more than one held Fleet, the request names one in `sparkpipe.fleet`. The request goes to `SPARKPIPE_POOL_UPSTREAM` with the upstream key and `X-SparkPipe-Fleet: <id>`. Requests from an account that holds no Fleet are refused. |
| `GET /availability.txt` | none | Each Fleet: serving, and `demo`, `pinned` or `free` (clause 4.2). |
