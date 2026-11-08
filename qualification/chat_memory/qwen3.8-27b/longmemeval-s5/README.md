# Context memory on LongMemEval_S: Qwen 3.8 27B

**Not state of the art yet.** Over 35 questions, memory recalled 20, the full
history 23 and compaction 9. The bar in `docs/CONTEXT_MEMORY.md` is recall
within 2 points of the full history, and memory is 8.6 points below it. It
answers from a prompt a quarter the size of the full history's.

## Recall

| arm | recall | single-session user | single-session assistant | single-session preference | multi-session | knowledge update | temporal reasoning | abstention |
|---|---|---|---|---|---|---|---|---|
| full | 23/35 | 5/5 | 5/5 | 2/5 | 2/5 | 5/5 | 2/5 | 2/5 |
| compaction | 9/35 | 3/5 | 0/5 | 0/5 | 0/5 | 2/5 | 2/5 | 2/5 |
| memory | 20/35 | 5/5 | 5/5 | 3/5 | 0/5 | 2/5 | 2/5 | 3/5 |

Memory missed 5 questions that the full history answered. Two of them come
from a defect, below: a curator reply cut short leaves its exchanges with only
automatic lines, their first words.
- c6853660 (knowledge update): the three exchanges where the user raised the
  morning coffee limit to two cups have only automatic lines, so the ledger
  held only the older "one cup", and memory answered "decreased";
- 55241a1f (multi-session): the exchange that gave the video's 21 comments has
  only an automatic line;
- b6019101 and 9ea5eabc (knowledge updates): both facts were in the ledger,
  and the answer took the older one. In b6019101 the user says on 20 May that
  they have watched 4 MCU films in the last 3 months, and on 29 May that it is
  5; memory answered 4;
- a346bb18 (multi-session): memory had both marathon times, 4h10m and 4h22m,
  and answered 28 minutes instead of 12.

The full history missed 2 that memory answered: 54026fce and 6aeb4375_abs.
`summary.json` lists every arm's misses.

35 questions are too few to measure a 2-point gap. 7 questions split the two
arms, 5 for full and 2 for memory, and a coin gives a split at least that
uneven 45% of the time. Four of memory's five extra misses have a cause in
the ledger, though, so they are faults to fix, not noise.

The compaction arm is the chat layer's `/compact full`. Its listing clips each
user message to 600 characters and each assistant message to 400, so a
111K-token history reaches the summarizer as about 37K tokens. That fits its 0
of 5 on single-session assistant questions, whose answers sit in the
assistant's replies. The docs' baseline is
provider-style compaction, one summary of the whole text; this arm is weaker
than that, so memory's lead over it says little.

## Two regimes

Cold: nothing is cached, and the whole prompt is new prefill. Warm: the
arm's prefix is cached.

In the main bench, full and compaction cannot be warm. Each question brings a
new history, so its first answer is the first time the engine sees that
prompt. Memory's import pre-warms its base, so its first answer is warm. Three
questions were in flight at once, four for the last eight imports, so these
times include waiting behind the other questions' prefill.

| arm | regime | mean prompt | new prefill per answer, mean / median | time to first token, median / mean |
|---|---|---|---|---|
| full | cold | 111,233 | 111,233 / 111,088 | 169 s / 198 s |
| compaction | cold | 14,311 | 14,311 / 14,298 | 13.3 s / 13.2 s |
| memory | warm | 26,960 | 430 / 63 | 1.8 s / 2.7 s |

The isolated pass, every arm both ways with nothing else running, follows in
the next commit.

## A continuing chat: the KV it keeps and the window it leaves

Qwen 3.8 27B keeps KV for its 16 full-attention layers: 4 KV heads of 256 in
BF16, 64 KiB a token. Its other 48 layers keep a recurrent state of fixed
size, the same for every arm.

| arm | mean prompt | KV for the chat | share of the 128K window | chats the 32 GiB pool holds |
|---|---|---|---|---|
| full | 111,233 | 6.8 GiB | 85% | 4 |
| compaction | 14,311 | 0.9 GiB | 11% | 36 |
| memory | 26,960 | 1.6 GiB | 21% | 19 |

A full history is warm from its second turn only while its 6.8 GiB of KV stays
cached, and this host holds 4 such chats at once. It also leaves 19,839 tokens
of the window. A chat that keeps going reaches the window and has to compact
or drop its oldest turns; here that is the compaction arm, at 9 of 35. Memory
keeps its prompt near 27K tokens however long the chat runs.

## What memory costs

Memory pays once to import a history and a little after every answer.

| | jobs | new prefill tokens | output tokens | job time |
|---|---|---|---|---|
| import of one history (about 111K tokens, 250 exchanges) | 73: 36 curator renders, 32 pre-warms, 5 stems | 315,432 | 18,400 | 58 min |
| upkeep after each live answer | 2: one curator call, one pre-warm | 2,545 | 90 | – |

- The import's time is mostly its curator renders' 18,400 output tokens. A
  third of all render output, 291,200 of 858,143 tokens, went to replies cut
  at their budget and dropped (the first defect below), so its fix should
  also shorten imports.
- Every pre-warm is reused in full by the next render, so skipping pre-warms
  would save no tokens.
- The import times above ran beside the other arms, three or four imports at
  once.
- A compaction costs one call: 38,714 new prefill tokens and 652 output
  tokens, 128 s.
- The upkeep's job time is left out: with other questions in flight, it
  measures the queue, not the job.

## How it ran

- Data: LongMemEval_S (cleaned), 5 questions of each of its 7 types drawn with
  seed 1107. Each question comes with its own history of about 50 sessions,
  about 111K tokens as the full arm renders it.
- Model: Qwen 3.8 27B in BF16 on one Mac Studio (M5 Ultra), served by
  `sparkpipe_model_residentd` and `sparkpipe_model_api` at TP 1 with 4
  sequences of 131,072 tokens (a 32 GiB KV pool).
- Chat layer: `tools/serving/chat_frontend.py` with memory on, 4 background
  builds at once, builds beside foreground requests.
- Arms, from `tools/chat_memory_bench.py`, all on the same model and the same
  question:
  - full: the whole history with memory off;
  - compaction: `/compact full` on the whole history, then the question on its
    summary;
  - memory: `POST /v1/sparkpipe/memory/import` with 8 exchanges per curator
    call, then the question.
- Reasoning effort minimal for every answer and for the judge. Every request
  sampled at the model's defaults, temperature 1.0 and top-p 0.95, once.
- The judge is Qwen 3.8 27B itself, with LongMemEval's own judge prompt for
  each question type. LongMemEval's paper judges with GPT-4o, so these recall
  figures compare the arms with each other, not with published LongMemEval
  scores.
- Code: the first 95 records ran on `metal/serve-memory` at d5400c149, whose
  memory code equals #1859 at 8056d464c. The last 10 ran on
  `memory/bench-receipts` at 775f8778f, whose chat layer adds the import fix
  below as 7c5001831.

## Defects the run found

- **A curator reply that also committed ran out of room** (fixed in
  #1875). The curator's output budget counted only the lines it asked for,
  160 tokens plus 80 a line, while the same reply also writes its commits.
  340 of the 419 renders that also committed stopped at the budget, against 24
  of the 1,272 that did not. A cut reply did not parse at all, so the batch was
  rendered again, and a second cut fell back to automatic lines: the first
  words of each exchange. 1,523 of the store's 11,890 lines are such
  fallbacks, about one in eight. Every recall figure above carries this
  defect.
- **An import that outlived its token failed** (fixed in #1859, 4512c6ce9).
  The import checked the user's token again before every batch. One import ran
  past the bench token's hour and failed at its next batch with "the
  conversation cannot open a memory session". It now checks the identity once,
  when it arrives.
- **An answer shown twice after a memory round** (open). One memory answer
  (c6853660) took 4 engine rounds and showed its paragraph twice. The chat
  layer streams each round's text. When a round writes text and then calls a
  memory tool, the next round opens a new assistant message, and here the
  model restated its answer. Proposed fix: start the next round's message with
  the text already shown, so the model continues it instead of restating it.

## Next

1. Run again on the curator fix (#1875), so no line is a fallback.
2. Knowledge updates: when a later exchange changes a fact, the older line
   should say so. In b6019101 and 9ea5eabc both facts were in the ledger and
   the answer took the older one.
3. A provider-style compaction arm, one summary of every word, before memory
   is compared with compaction again.
4. Import time: 58 minutes a history. Shorter lines, batches of 16, renders
   without the open tail, measured one at a time on an idle host.
5. More questions: the 2-point bar needs most of LongMemEval_S's 500, which
   the import time puts out of reach today.

## Files

- `summary.json`: per arm and question type, recall, the questions missed,
  prompt and prefill sizes, times to first token and the memory jobs, joined
  from the chat layer's log.
- `records.jsonl`: one line per answer: the question's id and type, the arm,
  the reply, the verdict, prompt and cached tokens, and the import's result.
