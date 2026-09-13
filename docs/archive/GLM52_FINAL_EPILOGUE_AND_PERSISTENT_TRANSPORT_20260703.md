# GLM52 PP13 Final Epilogue and Persistent Hidden Transport Quality Pass

This pass finishes two narrow performance paths as connected runtime features.

## Final-stage fused epilogue

The exact PP13 final stage can advertise `BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE` together with `FUSED_FINAL_TOKEN_TAIL`.
When this capability is present on the final `72:6` stage, the SM121 required CUDA path skips the old sequence of materializing restricted logits and MTP draft logits before a separate tail kernel. It launches a built-in candidate-reduction epilogue over the restricted vocabulary groups, then commits:

- restricted-token selection
- MTP draft-token selection
- MTP accept/reject mask
- committed MTP token ids
- MTP event counters

The exact-stage plan must provide workspace large enough for candidate scores and candidate token ids across the active batch capacity. Module validation rejects a final-stage built-in epilogue plan without this workspace.

## Persistent hidden transport integration

Persistent hidden transport is now an execution-frame dependency for production stages that require hidden input or output transport. The firmware package no longer links directly against hidden-transport global symbols. Instead, the frame context carries session-specific send/post-receive callbacks, which keeps the firmware package independent while making the transport dependency explicit at execution time.

For non-initial non-final PP13 stages:

- input hidden transport is posted before backend submission
- output hidden transport is prepared before submission and sent only after backend completion
- completion reports a failed status if the output transport send fails
- execution rejects missing frame context, missing sessions, missing callbacks, or packet/layout mismatches

The firmware tests cover deferred output transport: receive is posted before execution completes, send is issued at completion, and the persistent ring statistics confirm both events.
