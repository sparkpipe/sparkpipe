# Suite notes — HEAD window 43aa497..bdf58b6 (2026-08-25)

## Landed this window (mine, via coordinator commits a9cc709/eef7255 wave)
- Makefile: dead PYTHON_TESTS entry tests/test_dsv4_accum_export_contracts.py removed (file deleted upstream; PY-FAIL rc=2 noise gone).
- tests/test_code_size.py: untracked json/ mirror tree excluded from authored-line counter (was +107,091 phantom lines). Residue now +177 = concurrent in-flight landings (MI350P gate etc.), owned by those waves per CEILING law.
- tests/test_glm52_dspark_pack_ingest.c + tests/test_glm52_pp7_configure.c: parent-dir creation (build/, build/tmp/) before leaf mkdir(2); both latent-clean-build failures, verified green at bdf58b6 (rc=0 each).

## Transient (verified passing standalone)
- tests/test_release_agent.py: failed mid-suite v4 (undefined SparkRelease* from spark_release.o) but spark_release.c defines them static :997; make build/sparkpipe_release_manager rc=0 and py-test rc=0 when tree settled. Cause: concurrent make clean/rebuild races in shared checkout. Coordinator's tmp/run_suite_v2.sh private-scratch design addresses exactly this.

## OPEN: driver_compiler + orchestrator x86_64 generated-link (deterministic)
Evidence trail (this session):
- Shim-captured argv of failing link is IDENTICAL to a manual cc invocation that succeeds producing arm64 (/tmp/manual.so).
- Failure: ld "ignoring link_units/*.o: found arm64, required x86_64" -> generated object is x86_64; undefined _SparkTest{AddOne,Double}*. Hash suffix on generated object changes every run (de1639/4f950c/af43c9).
- No "-arch"/triple strings anywhere in src/include/tools/data; SparkRunProcess is plain fork+execvp, inherited env; env has no ARCHFLAGS/SDKROOT.
Next steps for owning lane:
1. Rerun failing compile with cc -v wrapper INSIDE SparkRunProcess child to dump ld's exact -arch resolution (my /tmp/shimbin approach works; see /tmp/cc_args.log capture).
2. Diff generated C between tool-run and manual-run moments (hash differs per run - confirm content delta vs pure naming).
3. Suspect Xcode 16 clang driver behavior with -dynamiclib -Wl,-exported_symbol ordering under fork/execvp cwd=repo-root; try adding explicit -arch arm64 host-default via ARCH_TUNE_FLAGS passthrough in request.extra_compiler_arguments as a workaround gate.
Baseline refs: tmp/suite_results.tsv (139P/30F) vs tmp/suite_results_head_20260825.tsv (157P/18F at 9dded88); pair was PASS in baseline.
