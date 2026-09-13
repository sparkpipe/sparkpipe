# generation-4 mechanics golden — minted 2026-08-23

`v4_mechanics.golden.txt` is the frozen verdict stream of
tests/test_stagepack_parity_v4.c against the reader state that introduced
the generation-4 half (runtime/spark_stagepack_reader.h: version dispatch,
ONE accounting table, normalized entry chain, inventory equality, dual-read
divergence proof). 5,828 verdicts.

The gate NEVER regenerates this file. Re-minting is a deliberate act that
belongs to a reviewed generation-4 semantics change:

    cc -std=c11 -O2 -Wall -Wextra -Werror -I. -Iinclude \
        tests/test_stagepack_parity_v4.c -o /tmp/v4 && /tmp/v4 \
        > tests/fixtures/stagepack_parity/_v4/v4_mechanics.golden.txt

Generation 3 stays pinned by the five per-family frozen-reference legs and
is not re-mintable at all (its behavior is frozen contract).
