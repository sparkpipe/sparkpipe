# TEAM TODO (coordinator-managed; agents strike items, coordinator verifies)

| Item | Owner | Status | Notes |
|---|---|---|---|
| VA_TRACE x70 debug fprintf sites uncommitted in runtime/model_serving_adapter.c | dsv4-flash | OPEN - owned | delete or keep per dev judgment; resolve this iteration |
| BF16 target + deep-accept crash (deterministic at base 142 accepted=6) | dsv4-flash | OPEN - repro done | crash evidence in .agents/coord/evidence_bf16_deepaccept/; fix needed |
| MERGE-NOTE: bonus-fold/multi-block vs all-7-drafts draft-count conventions | qwen27b-dev | OPEN - flagged in code | decide: port fold-aware accounting onto all-7-drafts path, or validate cache-path mode covers main's flow |
| Prefix-cache core -> qwen36 model port | qwen36prefix dev | IN PROGRESS | gated on pccore INTEGRATION.md |
| pccore interface header finalization | pccore dev | IN PROGRESS (4/5) | |
| GLM52 GPU numerical validator restore | glm52 | IN PROGRESS | prerequisite for validate/publish on glm52 |
| Stagepack 4-family collapse (~1200-1500 ln deletion) | glm52 | QUEUED | after validator restore; pure deletion round |
| GLM5.2 DFlash2 adoption (second model) | glm52 | QUEUED - user-directed | follow qwen27B proven pattern; target ~100 tok/s class |
| hwiface Q3/Q4 decisions | RESOLVED | CLOSED | F1 stays whole; C2/C3 split verdict recorded |
| Auditor: verify context-fix lossless claim independently | auditor | OPEN | re-run lossless check adversarially |
| Auditor: fact-check reports r1 (sysadmin/glm52 claims) | auditor | QUEUED | spot-check boxes + line counts |
| qwen36->qwen38 directory/symbol rename | coordinator + glm52 (timing) | QUEUED - window open (glm dev paused) | mechanical but touches active files |
| glm52-named general code -> shared layers | glm52 | QUEUED | same disease as stagepack families |
| GB10 hardcode sweep -> parameterization | hwiface definer | IN PROGRESS | part of v1/v2 |

| Residentd endpoint connection leak — CLOSE-WAIT sockets accumulate per client session; after ~N sessions new connections fail with IO_ERROR | dsv4-flash dev | OPEN | model_residentd.c endpoint accept/close path; repro: run model_batch twice without restarting daemon; second run gets status=4 |
