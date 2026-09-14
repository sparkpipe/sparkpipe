# Wave-Finisher report — 2026-09-14

Agent: wave-finisher (mgr2 dispatch; SOLE ceph warm consumer for the wave).
Branch `lane/wave-finisher-0914` off origin/main. Identity verified
`sparkpipe` via tools/sparkpipe_github_pat.sh before any write op. Inputs:
wave-pr ledger `lane/wave-pr-stagepack-sp4r` @ b648ee6 and wave-p sweep
`lane/wave-p-stagepack-sp4` @ 6637980. Filesystem + packer only: zero daemon
contact, zero module execution; spark3/spark6 touched only by pack placement
inside their lawful slots, behind lsof pre-checks. Per-pack detail in
docs/AGENT_LANE_BRIEFS/ledgers/wave-finisher-2026-09-14.md (same tree).

## Before/after matrix (✅ = placed/verified/locked/purged; S = in flight; ❌ = flagged)

| arm | before (wave-p + wave-pr 09-13) | after (wave-finisher 09-14) |
|---|---|---|
| qwen3flash.fp8.tp8 | 8 rebuilt ranks staged, placement BLOCKED on the frozen verify gate | ✅ 16/16 slots: verify gate re-run per rank (PASS: header + 1215 entries + 8-sample byte-trace + streaming sha == receipt), placed per rank r -> spark r + r+8 under each node's lawful name, .experts v2 regenerated from placed bytes, .sha256 + receipt pair, chattr +i, purged. Old defective generation (30,425,355,496 / 30,518,614,272 B classes) retired in place with replaced_sha256 receipts |
| qwen3flash.fp8.tp4pp4 stage3 | ranks 12-15 rebuilt, placement gated | ✅ 4/4 placed on sC-F after the same gate (305 entries + byte-trace); superseded stage3 generation (byte-copy rank12 + two competing gens) + s6/s7 strays deleted with sha pre-verification |
| k3.mxfp4.tp4pp4 | stage0-2 scopes frozen mid-read, stage3 unlaunched; 0/16 rebuilt | S stage0 COMPLETE: rebuilt end-to-end on spark9 (verify --quick PASS x4, oracle --digest PASS x4 with 0 mismatches, abi=format V2/manifest reserve 1048560/payload base 1048576), placed 4/4 on s0-s3 with full receipt chain, locked, purged. Stages 1-3: autonomous watchdog rebuild loops running on spark9 (journal-resume makes every retried byte count); NOT placed, verdicts pending |
| qwen3flash.fp8.tp8 .tmp partials | 31 never-referenced .tmp rows (~119 GB) | ✅ 30/30 remaining .tmp deleted (105,660,516,864 B) with size+sha256+lsof pre-checks, zero mismatches; sF compact.tmp was already deleted by wave-pr |
| minimax-h3 tp16 | ranks 00-03 byte-exact + rank04 partial on ceph; sparke MDS corruption fixed per sysadmin | ❌ NOT STARTED this wave — queued behind the k3 warm stream (one-warm-consumer law); resume is one command on sparke: rerun /mnt/model-warm/staging/minimax-lane/13b410f/emit_tp16_chain.sh (skips receipted ranks 00-03, resumes rank04 from progress sig v2), then tools/minimax_h3_tp16_boundary_check.py per rank vs warm |

## Headline numbers

- Placed this wave: 24 pack slots (16 qwen tp8 + 4 stage3 + 4 k3 stage0),
  ~1.04 TB of new-generation bytes, every slot dest-sha verified against its
  build receipt, .experts regenerated from placed bytes, .sha256 + receipt
  pair, chattr +i, per-node purge logged in the ledger.
- Deleted this wave: 105,660,516,864 B of .tmp partials (30 files, 9 nodes)
  + the superseded stage3 generation (35,003,483,808 B in-place replacements
  with receipts + 17,871,277,568 B s6/s7 strays) + two wrong-slot copies
  (61,037,224,960 B) the finisher itself created and retracted same-day.
- Verifier gate fix (load-bearing for every future wave): the "frozen verify
  gate" was never ceph — qwen4_flash_pack_verify's receipt check does
  hashlib.sha256(pack.read_bytes()), pulling the whole 30.5 GB pack into RAM
  inside a capped cgroup: dirty-anonymous memory the kernel cannot reclaim,
  pinning the task at MemoryHigh forever (wchan mem_cgroup_handle_over_high,
  RSS == the cap). With the receipt parked beside the pack, the same verifier
  PASSES in <60 s. Same wchan signature on SP-4R's frozen k3 scopes (their
  5,590,697,728 / 900,703,104 / 577,957,888 B payloads) — those runs died in
  the throttle, not in the storage.

## Key findings this wave

1. VERIFY-GATE ROOT CAUSE (above) — fix is one line in
   tools/qwen4_flash_pack_verify.py (stream the digest in chunks; never
   read_bytes a 30 GB pack). Left uncommitted: docs-only wave.
2. CEPH WARM-PATH HEALTH (sysadmin intel, instrumented all wave): cold reads
   degrade intermittently and per-client — whole regions crawl at 1-3 MB/s
   while a second node reads the same object at 400-600 MB/s, and some
   client sessions wedge after ~100 GB streamed (folio_wait_bit_common with
   zero progress, never self-recovers within hours). Proven playbook: kill,
   drop caches, retry (k3_pack's resume journal banks completed tensors), and
   MIGRATE the work dir between nodes when one client is poisoned. ceph is
   NOT fully fixed; the data plane is, the read path is not.
3. SPARKCAP MEMORY CEILING: the 4096M/2900M cap recipe livelocks any packer
   or verifier that reads multi-GB tensors (k3 stage packs carry ~10 GB
   source tensors; 64G/56G runs them at 121 MB/s). Budget heavy warm builds
   accordingly.
4. SP-4R's shipped sp4r_k3_experts_gen.py never ran to completion: ctypes
   argtypes reject bytearray contexts. A patched copy (byref + c_ubyte
   arrays) generated all four stage0 sidecars (41,216 ranges each,
   self-validated). The fix should be upstreamed to the node trees.
5. abi_version=6/descriptor=72: ungroundable — see ledger item 3. The
   verifiable post-2b27e64 format law (ver 2, manifest_len 1048560, payload
   base 1048576, tp echo, v2 experts) is what the placed k3 packs carry.

## Purge log (sudo -n sync; drop_caches=3, UTC)

Item-1/2 placements: s0 08:54:47Z, s8 08:53:13Z, s9 08:59:20Z, sd
08:56:37Z + 08:59:50Z + 09:41:10Z, s1 09:04:36Z, s2 09:05:43Z, sf
09:00:36Z + 09:59:52Z + 10:01:09Z, s4 09:16:05Z, sc 09:17:09Z, s5
09:53:06Z, s6 09:56:25Z, s7 09:57:36Z, se 09:58:56Z, sa 10:08:57Z, sb
10:10:04Z (node-local timestamps for the 17:xx-18:1x entries; clocks are
UTC+8/9 skew, all purge events listed in order). k3 stage0 placements: s0
17:51:01Z, s1 17:57:47Z, s2 18:03:37Z, s3 18:10:38Z (node-local). Plus
purges at every watchdog retry and every verify-stall kill.

## Carry-forward

1. k3 stages 1-3: the watchdog loops run unattended on spark9
   (/tmp/fin_k3_stage.sh S FIRST COUNT 40; state under ~/sp4rbuild/k3; per
   stage: pack -> shard -> shas -> experts -> then verify --quick x4 +
   oracle --digest x4 + place on nodes 4S..4S+3 + purge). Everything is
   scripted; no human decisions left.
2. minimax-h3 ranks 04-15: emit + boundary-check commands staged (see
   matrix); needs a free warm window only.
3. tools/qwen4_flash_pack_verify.py digest streaming fix + the patched
   sp4r_k3_experts_gen.py should land as the next code commit (this wave is
   docs-only per orders).
