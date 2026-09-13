# COMMENT STRIP + NAME CLEANUP — ALL PRODUCTION FILES

## Directive from coordinator (user-ordered):
1. Strip ALL comments from all production .c/.h/.cu/.cuh files
2. Fix model-specific names in general/shared code
3. Fix general names on model-specific code

## Rules:
- Only touch files under modules/, runtime/, node/, include/, src/, inference/, cache/, ring/, deployment/, model-families/
- DO NOT touch vendor/ directories (hip-headers are upstream, leave their comments)
- DO NOT touch tests/ or docs/ 
- Use a proper C-aware comment stripper that handles string literals and character constants
- After stripping, verify: make contract for each affected module still passes
- Commit in batches of ~20 files with clear messages

## Model-name contamination in shared code:
- Shared headers (runtime/paged_kv_common.h, runtime/adapter_common.*, etc.) are CLEAN after the dry-law fix wave
- Check every shared file before stripping: any remaining model tokens in symbols/types must be renamed to neutral names
- Comments mentioning models get stripped along with everything else — no special handling needed

## Verification per batch:
- cc -fsyntax-only on stripped files (with appropriate includes)
- Run the module's contract target if available
- git diff --stat to confirm only comments were removed