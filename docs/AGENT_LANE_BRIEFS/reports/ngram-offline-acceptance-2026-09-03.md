# Offline acceptance: NGram speculator on glm5.3 reference generations

Date: 2026-09-03. Harness: /home/spec/draft_service/acceptance_offline.py on the RTX 5090.
Corpus: bench_corpus.json — 92 cases from qualification/ds4_eval quality-fixtures-glm5.3-flash,
20,738 anchor positions (min anchor 16), depth-7 chains.

Method: at each position A, draft a chain from committed ids only; acceptance = longest prefix
matching the recorded continuation. This is exact greedy-target acceptance only if the fixtures
are greedy decodes; if they were sampled, treat these numbers as a lower-fidelity proxy.
Tap-conditioned drafters (DFlash2) are excluded — their proposals without real taps are
meaningless; their acceptance waits on the tap-extraction seam.

## Results

- Coverage: 57% of positions yield an ngram proposal.
- Mean accepted per proposal: 0.874 tokens; per position overall: 0.498.
- P(accept >= d | proposal): d=1 0.387, d=2 0.202, d=3 0.116, d=4 0.073, d=5 0.047,
  d=6 0.031, d=7 0.020.
- Domain spread is wide: Engineering 1.29 accepted/position, Chemistry 0.63, Science 0.82;
  code/security writeups much lower (Firebird 0.085, Botan 0.15).

## Reading

NGram is free (no taps, ~0.02 ms at short history) and decorrelated from model-based drafters,
so its tree branches cost only their verify rows. At ~0.5 accepted tokens/position it earns a
speculator slot only where verify rows are cheap or the tree has slack — the branch-cost policy
decides per budget. The earlier bench_farm result (ngram proposes nothing at 16-token prefixes)
was a short-history artifact: acceptance grows with committed history, which serving has.
