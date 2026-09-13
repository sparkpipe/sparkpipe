# ds4-eval — Kimi K3

Run completed on 2026-07-28 against the 92 embedded `ds4_eval.c` cases.

## Result

| Family | Correct | Total | Accuracy |
|---|---:|---:|---:|
| GPQA Diamond | 23 | 25 | 92.0% |
| SuperGPQA | 21 | 25 | 84.0% |
| AIME2025 | 21 | 25 | 84.0% |
| COMPSEC | 16 | 17 | 94.1% |
| **Overall** | **81** | **92** | **88.0%** |

- API/model: Kimi coding Chat Completions, `k3`
- Temperature: `1.0`
- Maximum completion: `16,000` tokens
- Operational errors: `0`
- Returned model IDs: `92/92` exactly `k3`
- API attempts: `92/92` completed in one attempt
- Finish reasons: `88` `stop`, `4` `length`
- Usage: `32,676` prompt + `241,499` completion = `274,175` total tokens
- Missed case indices: `28, 41, 50, 51, 52, 63, 69, 71, 72, 74, 82`

## Provenance and validation

- Source repository: `https://github.com/antirez/ds4`
- Source commit: `54b36ed9ba42da31b24f2d1a5feb075c2475dbb1`
- `ds4_eval.c` SHA-256: `19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23`
- Extracted distribution: 25 GPQA Diamond, 25 SuperGPQA, 25 AIME2025, 17 COMPSEC
- Coverage: all indices `1..92`, 92 unique IDs, 92 response artifacts
- Strict cache revalidation: `92` cached, `0` pending
- API credential present in result artifacts: no
- `summary.json` SHA-256 after final revalidation: `3957de26be737427ee71cc6785232f8c92180791b1849dbb514565ed3e9820af`

The four `length` responses reached the evaluator's default 16,000-token limit;
one scored correct and three scored incorrect. This is the embedded ds4
capability-regression suite, not an official leaderboard result.
