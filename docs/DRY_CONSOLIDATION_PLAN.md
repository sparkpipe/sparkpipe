# DRY consolidation plan (coordinator-verified)

Merges the GLM52 agent's and the DSV4 Pro session's DRY audits into one
plan. Every item below was re-verified against the tree; the agents' claims
that did not check out are corrected inline.

## Agreed, verified quick wins

1. **DSV4 packer family (H2)** - the Pro packer is a geometry-override
   wrapper over the Flash packer. Fold both into one parameterized packer
   with a MODEL_GEOMETRY table, the same pattern already landed for the
   sharder pair (tools/dsv4_tp16_stagepack.py). Owner: DSV4 Pro session
   (offered), coordinator gates. Tests: test_dsv4_stagepack.py.
2. **qwen36/qwen38 packer fork** - 379 identical lines (measured);
   qwen38 = qwen36 + codec delta. One shared qwen packer base with
   per-variant config. Owner: coordinator or qwen session. qwen36 stays
   frozen-deprecated; the shared base survives with qwen38.
3. **Shared packer core (H3)** - safetensors source reader, record/
   directory/header serialization, sha/align helpers re-implemented in
   all six packers; extract tools/spark_pack_common.py. Owner:
   coordinator, after 1-2 land (it is the substrate for them).

## Corrected (agent claim did not verify)

- **glm52_stagepack.py is NOT dead.** It has live consumers
  (tests/test_glm52_stage_pack.py, tests/test_glm52_quantized_cuda_contract.py,
  tests/test_production_selection_contract.py, Makefile). It is the strict
  PP13 packer; the resident packer is v3. Do not delete; fold both into
  the shared packer core instead.

## In flight

- **Dispatch-policy split (audit step 3)** - GLM52 agent's classification
  landed as a plan (767/877 lines neutral). Coordinator landed the
  prerequisites: SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT 13u /
  PP_STAGE_LAYER_COUNT 6u, SPARK_DSPARK_POLICY_REALTIME_PRIORITY_THRESHOLD,
  pinned. Agent is now producing the neutral header/source proposal
  (struct neutralization + pure-rename .c); coordinator reviews, lands
  under include/sparkpipe/ + src/, wires the Makefile, pins the GLM52 path
  byte-identical, and runs CI.

## Queued

4. **tp_collective config extraction** - ServingLoadTpAlgorithms /
   LoadTpRailHosts / LoadTpStepRails / ValidateTpCollectiveMembers +
   InitializeTpCollective re-pasted across adapters (~500 lines).
   Extract into ring/transport as one shared config module. Owner:
   coordinator (shared code), after step 3.
5. **One contract-generator library** - the JSON->C #define emitter shape
   shared by glm52/dsv4/k3 generators; per-model files become configs.
   Owner: coordinator.
6. **Identity generation (M1/M2)** - module-id/description shas generated
   from the contract everywhere (Makefile.pro, aliases header, description
   JSONs, manifests); bucket-pair description JSONs template-generated.
   Owner: DSV4 Pro session.
7. **Adapter capability chains (L2)** - shared base-chain macro for the
   four serving adapters. Owner: coordinator.
8. **Devcycle script family (M3)** - build/ship/validate script
   consolidation + de-hardcoding host paths. Owner: per-model sessions.

## Deliberately not consolidated

- Per-family model headers (identical shapes, different constants - that
  is the DRY law working).
- Per-model TP chain state machines (geometries genuinely differ; common
  parts already live in ring/transport + runtime).
- The validation harness's intentional control-vs-candidate independence.
