# Perf program 2 — kimi's end-to-end walk (2026-08-30)

THE PATTERN (three mistakes repeated at every stage): per-token work
on the serialized critical path; staged paths that exist but aren't
WIRED; latency-bound kernels filling too little of the machine.

## The ranked rocks (kimi's order, our execution)

R1 SCREENED HEAD AT B1 + DRAFTER (~8ms/token qwen38 no-spec, ~10ms/
   spec round; DISPATCH-ONLY): spark_lm_kernels.cuh:4894 short-
   circuits row_count==1 to the full 2.54GB BF16 rescore; the
   certified screen (~0.64GB) exists+validated. Same shape in the
   dflash2 drafter + dormant MTP. NEAR-ZERO RISK, biggest $/line.
R2 PREFILL WIDTH (TTFT): engine caps 16 rows/submission; the qwen38
   adapter chunks to max_active_sequence_count (1 row/frame in the
   shipped template = one 27GB re-stream per prompt TOKEN = the
   measured 21.7 tok/s); dsv4 prefill attention wave-serial
   (module.c:3051-3066) = the 8x 32K regression. Fixes are wiring:
   chunk to max_input_row_count, raise the budget, dsv4's bulk
   causal-prefill kernel (its own header prescribes it).
R3 FLASH-DECODE the shared attention family (30-60x at long context;
   the biggest kernel project): 2-byte scalar loads, block reduction
   per position, KV read twice, no split-K, 24-64 CTA grids. The DSA
   scorer shares the disease; its sibling can't launch past 64K
   positions.
R4 BATCH WEIGHT AMORTIZATION: WS/native default from rows>=2 +
   pipelining (the 125-205 vs 228 GB/s gap is sync-serialized k-tiles
   + 1 CTA/SM, not fundamental); stage activations once per tile.
R5 VALIDATION/ADMISSION FANOUT OFF THE PER-FRAME PATH: 6-10 full
   validations per submission, 3 admissions per dsv4 submit, ~35
   O(request) scans per Progress, SHA probes in the loop = pure GPU
   idle between completion N and launch N+1.
R6 DSV4 STRUCTURE: island chaining (130 host-callback graphs), kill
   the 129 RA-joins (dependency doesn't exist), event diet (~770/
   token), expert activation staging 16x redundant, routing grid.sync.
R7 BLOCK-TABLE UPLOAD: kill the per-frame full upload (8MB at
   ceiling) + per-frame memsets (dirty tracking).

## Correctness MUST-fixes (before/with the perf work)

C-EOS: stop_token_count=0 unwired — every request runs to FULL
   budget (correctness + throughput tax). Wire EOS (model default +
   API field).
C-CANCEL: disconnect→cancel exists but unwired (departed clients burn
   GPU). Wire it.
C-TRAP: expert kernels asm volatile("trap;") on corrupt route map
   kills the whole GPU context — fail the frame instead.
C-BOUNDS: sparse attention trusts top-k page indices unchecked (wild
   KV address on corruption).
C-UE8M0: LmFloatToUe8m0 rounds DOWN (systematic quant error; the
   certified path avoids it).
C-RANS: fixed 20KB smem window, no bound check (dormant landmine).
C-COVERAGE: hot kernels w/o direct validation (LmHeadCandidateKernel
   live in glm52/qwen, several MoE paths) get validator entries
   BEFORE optimization touches them. The certification culture is why
   these rocks are safe to take — do not weaken it for speed.

## API notes (mostly moot under liteLLM, but):
EOS + cancel survive the replacement (below the API layer). The
O(n^2) parse_token_array (260k-prompt = tens of seconds) dies with
the JSON path or gets the one-pass fix if it survives.
