# The mega-PR against the product: will an elite developer save time?

Written on 2026-11-10 at the head of pull request #1903, after the codebase
audit, the runner-owned schedule, tree verification, the Metal line and the
carried fixes were merged into it. This page does not grade the PR against
README or TECHDEBT items. It asks the only question the site asks: an
engineer who lives in a large codebase, working all day with coding agents on
a fleet of their own, does the stack after this PR give them their time back?

The site sells three things, in this order: the codebase stays warm all week
(first token in 0.3 to 5 s at 137,700 tokens, no re-prefill after a break or
a branch switch), generation at the fleet's own speed (126 tok/s on Kimi K3
file rewrites with speculation, identical output), and work done while the
engineer is away (Functions that ingest the repository, refresh the cache,
prefill ahead, review, test and run an overnight agent). Demo hours start on
1 December 2026 and weeks on 4 January 2027.

## What the PR changes for each of the three waits

**The warm turn.** Nothing in this PR makes a warm turn faster by itself, and
nothing makes it slower. The prefix cache, the 24-hour retention and the
token-exact replay that make a warm turn possible were on `main` before it.
What the PR adds is the ground under them: one configuration schema with no
defaults, so a lane never picks a wait regime or a context length on its own;
a decode chain that is one host turn, with sampling inside the graph and the
tokens read back once per chain instead of once per step; and a failed
streamed prefill that takes the lane down loudly instead of serving behind
queued collectives. For the engineer that is fewer afternoons where the fleet
is "slow today" for a reason nobody can name. The one-host-turn chain is
expected to move B1 decode latency (the November perf audit put the step at
bytes over bandwidth plus about 75 µs per collective round, with host turns
on top of that); it has not been measured on the fleet, on the owner's
budget call, so it is not a number yet.

**Generation.** Four changes bear on tok/s, none measured on the fleet:

- Verify frames of 2 to 16 rows read each dense weight once, so a
  speculative frame costs about a one-row launch where it cost up to four
  (bit-identical, device test on the build host).
- The BF16 all-reduce at 16 ranks takes the two-phase path from 10,240
  elements on dual rail, so 2-row frames stop paying the direct path.
- GLM 5.3 Flash's DSA index round gathers each rank's top pools instead of
  every score (10,000 to 4,096 bytes per rank row); at 100K tokens a DSA
  layer spent 106 of 125 ms in that round.
- Tree verification: a verify frame holds a trunk plus branches, so a draft
  that forks (two plausible continuations) is verified in one frame. On the
  Metal gate (Qwen3.8-27B at TP1) trees were token-identical and neutral on
  reasoning prompts (19.3 against 19.2 tok/s) and slower on an easy rewrite
  (154.6 against 181.8), because the tool added every available alternate
  where chains already accepted most drafts. The product value of trees is
  therefore unproven and depends on the planner adding branches only where a
  chain would fail. Trees must stay off by default on the fleet until the
  agent-turn gate shows a gain; the planner's calibration fix (#1873) is
  the first piece of that.

The gap that matters most for a Claude Code or Codex user is not in this PR:
batches of two or more never speculate (PERFORMANCE_STATUS, "Speculation").
A developer who fans out subagents, or a team sharing a fleet, falls from 126
to 15 to 35 tok/s per stream on K3. The closed PR #1765 had the engine side
of speculation at B2 to B8 (one draft connection per lane, shared frames) but
it was written against the chain engine and unmeasured; it has to be redone
on the tree-frame engine. Until then the site's generation number holds for
one stream at a time.

**Work done while away.** This PR does not touch it. The repository's own API
plan (FLEET_API.md, section 1) records Functions as "None. No sandbox." and
the working-state rules, cache warm-ups and pins as open pull requests (#1691
to #1693, #1821). For the site's promise this is the largest gap, and it is
outside the inference stack.

## The 1M prefill, which the owner calls the one weakness

Measured today (PERFORMANCE_STATUS): a cold 1M prefill takes 4,542 s on Kimi
K3, 3,534 s on GLM 5.3 Full and 814 s on DeepSeek V4.1 Flash; 8K prompts
run at 0.9 to 3.7% of the compute floor, long prompts at 460 to 736 tok/s.
The compute floor for K3 is 16,123 tok/s at 16 nodes, so a 1M prefill at
100% would take 65 s and the 3-minute goal is 35% of the floor: a tenfold
gain over today, not a rewrite of physics. The loss is in the structure of
the work, not in the kernels: every wave pays collective rounds per layer
with no compute overlapped (the 11-08 GLM 5.3 Full 16K profile showed 22 to
31% of the span as exposed collectives), and prefill shares the lane with
decode.

What this PR did for it: the runner owns the layer loop and both prefill
lanes run as frames on one thread, a model declares its minimum rows per
frame, prefill rows are built sequence by sequence, and the GLM index gather
shrank. What it did not do: the collective plan from the schedule, the second
lane's rounds on their own mesh band, the sequence-boundary split of a
multi-sequence wave, GLM 5.3 Full's wave chunking and DSV4.1's tail rows in
the runner, K3's dispatch loop, and a roofline meter that logs bytes moved
and rounds per step against the measured bandwidth so attainment is a
number the rank reports. Those are the prefill program, in the audit page's
order. The PR made them the next steps rather than a thread rewrite away.

For the product this means the 1M number stays a cold-start cost for the
launch: the daily loop lives on a cache that is refreshed overnight and
between turns, and on the separately licensed memory program bounding what
the model must read at all. The site's Functions (ingest, refresh, prefill
ahead) are the operational form of that sidestep and are not built.

## Correctness, trust, and what the engineer notices

Numerics now follow the publisher (residual sum, norm gain, output gate,
flash-form attention, rope rounded as the publisher's code rounds). The
site's "identical output" claim becomes checkable against the publisher's
own code rather than against our last build. Every slow path logs once per
shape, a decode graph that cannot be captured aborts the rank instead of
running eager, and a lane that cannot serve again exits for a restart. For
the engineer these are invisible when the fleet is healthy and decisive when
it is not: a rank that aborts at startup is a rank the fleet owner sees in
the morning, not a day of 6x slower decode nobody can explain. The price is
that a release now needs the slow-path survey (which `SLOW-PATH` lines fire
on each lane) before it is rolled, because what used to be a silent fallback
is now a fatal error.

Metal serving through the common stack puts the same engine on an M5 Ultra
for Qwen 3.8 27B, Max and the Qwen3.5 small models. The site does not sell
it. Its product value is a developer's local evaluation of the same engine,
the drafter development loop, and a hedge; it costs nothing on the fleet
and the merged kernels still need their compile on M5U0.

## Where the stack stands against the site, after this PR

| The site promises | After this PR | What moves it |
| --- | --- | --- |
| Warm first token 0.3 to 5 s at 137,700 tokens | Measured before the PR; mechanisms unchanged; one host turn per chain expected to help | Fleet serving gates at 1, 8 and 16 lanes |
| No re-prefill after a break or branch switch | Prefix cache and 24 h retention on `main`; pins and warm-ups open (#1821); working state open (#1691 to #1693) | Merge the launch PRs after this one |
| 126 tok/s file rewrites, identical output | Holds for one stream; frames cheaper; trees unproven | Agent-turn gate; speculation at B2 to B8 |
| Parallel agents and teams | 8 sequences on K3, no speculation in batches | Shared frames on the tree engine |
| Overnight Functions | None, no sandbox | A sandboxed runner and the first three Functions (ingest, refresh, prefill ahead) |
| A model switch in 20 to 80 s | Shared spine through the direct reader (#1871 folded) | Measure |
| 1M context | Cold 14 to 76 min; cache keeps it warm afterwards | The prefill program above |
| 18 models | All twenty adapters build; the Metal gates for Max and Qwen3.5 have not run | Release gates |

## What to do first, with the product in view

1. Roll this PR through the release kit with the serving gates at 1, 8 and 16
   lanes, greedy and sampled, and the slow-path survey. Until then every
   number above is the old build's.
2. Land the launch PRs behind it: cache warm-ups and pins, working state,
   the request records. These are what the site's first section promises.
3. Speculation in batches on the tree engine, gated by the agent-turn
   workload. This is the single largest generation-time gain for a Claude
   Code user with subagents.
4. The prefill program in the audit page's order, with the roofline meter
   first so the CUDA dev measures attainment instead of estimating it.
5. The first three Functions: ingest the repository, refresh the cache,
   prefill ahead. They are the sidestep of slow prefill in daily use.

The memory program is separately licensed and reached only through the chat
layer's protocol (`docs/MEMORY_PROTOCOL.md`). Its product role is to bound
what the model must read and to keep a history past the window. Its own
audit lives with its repository.

## Verdict

An elite developer saves time on this stack when the cache holds across their
day and their speculation holds at one stream; those two held before this PR
and still hold, on firmer ground. They do not yet save time when they fan
out agents, when the fleet starts cold on a million tokens, or overnight,
because those three are not built. This PR did not move the site's numbers.
It made the stack one path, fail-loud and measurable, deleted the work that
was not that path, and turned the prefill program from a rewrite into a list.
That is the right order: a fleet that cannot be debugged cannot be made fast,
and the CUDA dev's job after this PR is the prefill program, not the
archaeology.
