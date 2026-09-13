# Rolling collective Program quarantine

This directory preserves the strongest current rolling-Program transport
experiment without adding it to the selected DSV4 runtime. The patch is based
on `da7f91090c0d40729352b4e4180ad231971c90a2` and belongs to the isolated
worktree `/private/tmp/dsv4-collective-owned-send-poc`.

The experiment gives each Program operation a dedicated stream, retains every
recursive-step reservation, and keeps allocation, graph capture, node patching,
and legacy submission out of the hot start path. Its focused host and source
contract tests pass.

It is quarantined for two concrete reasons:

1. The final recursive-stream source, transport SHA-256
   `4678dfcf1ba8afde38b8851b1be1a02c37389d9689978c21194c4e11ab40ee81`,
   has not run on the Spark hardware. The included live receipts identify the
   preceding `d297fc...` transport revision and are not provenance for the
   frozen patch.
2. The receipt named B1024 moved the correct 8 MiB byte count, but its actual
   runtime descriptor was `{active_sequence_count=1,
   hidden_dimension=4194304}`. It did not execute the required exact shape
   `{1024,4096}`. Unit preparation of that shape passes; live execution does
   not yet have a receipt.

Do not apply this patch to production merely because the byte-equivalent tests
pass. First run current-source recursive hardware validation and an exact
`{1024,4096}` start/progress/drain/rearm test, refresh all embedded source and
binary hashes, and repeat the DSV4 O24/O128 end-to-end gate.

See `summary.json` for the exact identity, tests, evidence, and acceptance
blockers.
