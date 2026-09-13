/*
 * Shared serving-adapter skeleton: the contract every model family's
 * spark_<family>_serving_adapter.c re-implemented independently (env
 * parsing, slot claiming, completion routing, residency bookkeeping),
 * extracted once so a new driver starts from the template instead of the
 * next copy.
 *
 * Division of labor after the extraction:
 *
 * - THIS layer owns everything that is byte-identical across the four
 *   existing adapters: the pending-slot table discipline (claim, count,
 *   lane maps), the orphan-completion/wake wiring handed to the driver,
 *   the driver-completion match predicate, the serving-completion header
 *   echo, configuration validation, the driver load/create sequence, the
 *   quiesce/snapshot merge, the destroy guards, and the strict env
 *   readers.
 *
 * - THE FAMILY owns what genuinely differs: its descriptor and geometry,
 *   its config JSON schema, row-order law, frame building, admission
 *   shape, and any model-specific completion payload (speculation credit,
 *   cache lanes, emit rows).
 *
 * A family embeds SparkAdapterCommonState (usually as the first member of
 * its adapter state) and points common.pending at ITS pending array whose
 * first member is SparkAdapterPendingCore; every helper below then works
 * on family records without knowing their tail.
 *
 * Behavior contract: this file is a MOVE, not a redesign. Each helper
 * reproduces the exact statement sequence of the four adapters it was
 * lifted from (the four migrated adapters at unified head), including error
 * codes and evaluation order, so migrating an adapter is behavior-neutral
 * by construction and stays provable against the family host gates.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "spark_filesystem.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_ADAPTER_COMMON_ABI_VERSION 1u

/*
 * The submission identity every adapter echoes verbatim into its serving
 * completions: nine u64 fields copied once at reservation (slot claiming)
 * and replayed at completion (routing). Order matches the wire structs.
 */
typedef struct SparkAdapterSubmissionIdentity
{
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
} SparkAdapterSubmissionIdentity;

/*
 * Head of every family's per-slot record. A family's pending array is an
 * array of its own struct whose FIRST member is this core, so table walks
 * here and payload accesses there share one allocation.
 */
typedef struct SparkAdapterPendingCore
{
	uint32_t active;
	uint32_t row_count;
	uint32_t lane_count;
	uint32_t active_sequence_count;
	uint32_t work_kind;
	SparkAdapterSubmissionIdentity identity;
} SparkAdapterPendingCore;

/* The upstream serving completion route captured at initialize. */
typedef struct SparkAdapterCompletionSink
{
	SparkModelServingCompletionFunction function;
	void *context;
} SparkAdapterCompletionSink;

/*
 * The adapter-state prologue every family duplicated: counters, routes,
 * the loaded driver handle, and the residency bookkeeping inputs. Embed
 * one; initialize it with SparkAdapterInitializePrologue.
 */
typedef struct SparkAdapterCommonState
{
	/* Orphan accounting first: the shared driver-completion callback
	 * receives this struct and bumps exactly this counter. */
	uint64_t orphan_completion_count;
	SparkAdapterCompletionSink sink;
	SparkModelServingWakeFunction wake_function;
	void *wake_context;
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	/* Points at the family pending array (cores as first members);
	 * pending_element_bytes is sizeof of ONE family record - the stride
	 * between consecutive cores. */
	SparkAdapterPendingCore *pending;
	size_t pending_element_bytes;
	uint32_t pipeline_slot_count;
	uint32_t quiescing;
	/* Residency bookkeeping inputs mirrored from the configuration. */
	SparkModelServingRuntimeLimits runtime_limits;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	void *execution_stream;
} SparkAdapterCommonState;

/*
 * The driver identity a family will accept, plus which capability check
 * to apply. check_kind selects between the two load-time disciplines the
 * four adapters shipped with:
 *  0 = SparkModelDriverProgramSupportsRuntimeLimits over required_flags
 *      and the configured limits (the runtime-limits school);
 *  1 = exact flag mask equality plus explicit profile floors
 *      (max_inflight / max_active_slots / max_new_tokens; the exact-mask school).
 * They are NOT equivalent (kind 1 additionally floors the program's own
 * max_inflight against the configured slot count, and the profile floors -
 * max_active_slots / max_new_tokens - have no kind-0 counterpart; neither
 * school floors max_resident_sequences at load time), so both stay
 * available; new drivers should pick kind 0 unless they inherit a published
 * contract that pins kind 1.
 */
typedef struct SparkAdapterDriverContract
{
	const char *model_id;
	const char *model_revision;
	const char *stage_name;
	/* NULL skips the target compare. Target-pinning is orthogonal to
	 * check_kind: the runtime-limits school usually leaves it NULL but need
	 * not (glm52 pins the target AND checks kind 0), and the exact-mask
	 * school pins it; every combination loads through this one contract. */
	const char *target;
	const char *description_sha256;
	uint32_t required_program_flags;
	uint32_t check_kind;
	void *node_context;
} SparkAdapterDriverContract;

/* One staged module-environment assignment (setenv overwrite, fail loud). */
typedef struct SparkAdapterEnvironmentSetting
{
	const char *name;
	const char *text;
} SparkAdapterEnvironmentSetting;

/* ---- slot claiming ---------------------------------------------------- */

/*
 * Copy the identity + shape half of a submission into a freshly zeroed
 * core. The caller zeroes the WHOLE family record, fills family payload,
 * and sets active itself (the four adapters disagree on when active is
 * stamped relative to payload fills; no observer can tell, but the move
 * preserves each one's choice).
 */
void SparkAdapterPendingCapture(
	SparkAdapterPendingCore *core,
	const SparkModelServingSubmission *submission);

/*
 * Index of the first inactive slot below slot_count, or -1 when every
 * slot is claimed (the BUSY signal). Scans [0, pipeline_slot_count) only
 * - slots beyond the configured count never activate, exactly as the
 * four adapters scan today.
 *
 * The table is the FAMILY pending array, so callers hand the element
 * STRIDE (sizeof of one family record); cores are read at
 * (char *)table + index * pending_element_bytes.
 */
int32_t SparkAdapterPendingClaim(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t slot_count);

/* Inactive-slot count (the adapters' AvailableSubmissionCount). */
uint32_t SparkAdapterAvailableSubmissionCount(
	const void *pending_table,
	size_t pending_element_bytes,
	uint32_t slot_count);

/* Row -> lane map every completion path rebuilds: last row index seen
 * per lane (identity echo for token scatter). */
void SparkAdapterCaptureLastRowByLane(
	const SparkModelServingSubmission *submission,
	uint32_t *last_row_by_lane);

/* Lane-indexed resident-slot capture (lane-major slot layout). */
void SparkAdapterCaptureResidentSlotsPerLane(
	const SparkModelServingSubmission *submission,
	uint32_t lane_count,
	uint32_t *resident_slots);

/* Row-indexed resident-slot capture (row-major slot layout: the slot of each
 * row's own lane, indexed BY ROW). */
void SparkAdapterCaptureResidentSlotsPerRow(
	const SparkModelServingSubmission *submission,
	uint32_t *slots_by_row);

/* ---- completion routing ------------------------------------------------ */

/*
 * Driver create-request completion callback for pre-route completions:
 * context is the owning SparkAdapterCommonState; bumps the orphan counter
 * and ignores the payload (the exact body all four adapters ship).
 */
void SparkAdapterOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion);

/*
 * Driver wake trampoline: context is the owning SparkAdapterCommonState;
 * forwards to the upstream wake route when present. Hand this to
 * request.wake_function so families stop re-declaring the static.
 */
void SparkAdapterDispatchWake(void *wake_context);

/*
 * The match predicate: request id, frame sequence identity, and program
 * id all agree. Single-frame adapters pass the reserved identity as the
 * expected frame ids; multi-frame adapters pass the frame currently
 * in flight.
 */
uint32_t SparkAdapterDriverCompletionMatches(
	const SparkModelDriverCompletion *driver_completion,
	uint64_t request_id,
	uint64_t sequence_id,
	uint64_t sequence_position,
	uint32_t program_id);

/* Accumulate one frame's latency/credit counters (multi-frame adapters). */
void SparkAdapterAccumulateFrameCounters(
	uint64_t *accepted_token_count,
	uint64_t *queue_delay_ns,
	uint64_t *service_time_ns,
	const SparkModelDriverCompletion *driver_completion);

/*
 * Zero a serving completion and echo ABI version, descriptor bytes, and
 * the reserved identity. Status, counters, tokens, and extensions stay
 * caller-owned - that is where the families legitimately diverge.
 *
 * RESIDENCY LAW (adoption-blocker class): this builder
 * leaves completion->residency ZEROED. The residentd validates every
 * published completion's residency token against the route submission's
 * (exact compare; a mismatch kills the batch), so EVERY family must set
 * completion->residency from its own anchor BEFORE publish. The four
 * migrated adapters do this family-side today - all three anchor schools
 * exist (driver-frame echo, driver-via-pending mirror, submission-anchored
 * pending). A new driver following the recipe must use
 * SparkAdapterBuildCompletionHeaderWithResidency or set the field itself;
 * publishing a zeroed token is a wire-visible failure, not a default.
 *
 * ABI LAW: abi_version is STAMPED here, not echoed from the submission.
 * Deliberate: the serving-runtime validator rejects any completion whose
 * abi_version differs from SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION, so
 * stamping always validates while echoing survives only while submissions
 * carry the same constant. An echo-school adopter validates before echo.
 */
void SparkAdapterBuildCompletionHeader(
	SparkModelServingCompletion *completion,
	const SparkAdapterPendingCore *core);

/*
 * SparkAdapterBuildCompletionHeader plus ONE residency-token copy - the
 * wire-enforced field the plain builder cannot own because each family
 * anchors it differently (driver frame, pending mirror, or submission).
 * Pass the family's anchor pointer; NULL keeps the zeroed default so the
 * call site stays explicit about which anchor it chose.
 */
void SparkAdapterBuildCompletionHeaderWithResidency(
	SparkModelServingCompletion *completion,
	const SparkAdapterPendingCore *core,
	const SparkModelDriverResidencyToken *residency);

/* Saturating u64 -> u32 narrow for accumulated accepted-token credits. */
uint32_t SparkAdapterClampAcceptedTokenCount(uint64_t accepted_token_count);

/* ---- lifecycle --------------------------------------------------------- */

/*
 * Shared ValidateConfiguration body: null/ABI checks, runtime-limits
 * validation against the family descriptor, then the required-pointer
 * and program-name checks bounded by stage_count.
 */
SparkStatus SparkAdapterValidateConfiguration(
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingAdapterConfiguration *configuration,
	const char *program_name,
	uint32_t stage_count);

/*
 * Fill the common prologue from the configuration (routes, limits, slot
 * count, stream). pending must be pointed at the family array afterwards
 * (the family owns its storage).
 */
SparkStatus SparkAdapterInitializePrologue(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterConfiguration *configuration);

/*
 * Shared LoadDriver body: load the shared object, match the descriptor
 * identity against the contract, find the program, apply the contract's
 * capability check, and create the instance wired to the shared orphan
 * completion and wake trampoline above.
 */
SparkStatus SparkAdapterLoadDriver(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterConfiguration *configuration,
	const SparkAdapterDriverContract *contract);

/*
 * Open of every validate_submission/submit path: state presence, the
 * quiescing gate, and descriptor/runtime-limits submission validation.
 * Family-specific boundary and row-order law follows this call.
 */
SparkStatus SparkAdapterValidateSubmissionOpen(
	SparkAdapterCommonState *common,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingSubmission *submission);

/*
 * The complete quiesce body all four adapters ship: presence/deadline
 * gate, the quiescing latch, the idle-slot demand, then the driver
 * active-submission poll.
 */
SparkStatus SparkAdapterQuiesce(
	SparkAdapterCommonState *common,
	uint64_t deadline_time_ns);

/* Snapshot merge: driver counters + slot availability + orphans. */
SparkStatus SparkAdapterSnapshot(
	SparkAdapterCommonState *common,
	SparkModelServingAdapterSnapshot *snapshot);

/*
 * Destroy guard: nonzero only when every slot is idle AND the driver
 * reports no active submissions (or has no snapshot to ask).
 */
uint32_t SparkAdapterDestroyReady(SparkAdapterCommonState *common);

/* Destroy + unload the driver instance (guarded, idempotent). */
void SparkAdapterTeardownDriver(SparkAdapterCommonState *common);

/* ---- env parsing ------------------------------------------------------- */

/* Raw getenv passthrough (NULL when unset). */
const char *SparkAdapterEnvText(const char *name);

/*
 * Default-ON switch: unset/empty reads enabled; exactly "0" reads
 * disabled; anything else enabled. The prefix-cache A/B toggles and the
 * speculation kill switch ("0" forces the config value off) all
 * use this semantics.
 */
uint32_t SparkAdapterEnvFlagDefaultOn(const char *name);

/*
 * Default-OFF audit switch: enabled iff set AND not exactly "0".
 * ("", "0" stay off; "01" is ON - strcmp semantics, not prefix.)
 */
uint32_t SparkAdapterEnvFlagDefaultOffExactZero(const char *name);

/*
 * Default-OFF truthy switch (KV-flat ledger selection): enabled
 * iff set, nonempty, and the FIRST BYTE is not '0'. Note this differs
 * from ExactZero above: "01" is OFF here, ON there. Both exist because
 * both behaviors are published env contracts.
 */
uint32_t SparkAdapterEnvFlagDefaultOffTruthy(const char *name);

/* ---- module environment staging ---------------------------------------- */

/*
 * setenv(name,text,1) mapped to SPARK_STATUS_INTERNAL_ERROR on failure -
 * the loud-fail discipline both env-staging adapters spell out in their
 * SET_TEXT macros.
 */
SparkStatus SparkAdapterSetEnvironmentText(const char *name, const char *text);

/* "%u" formatter for staged unsigned values (32-byte scratch typical). */
void SparkAdapterFormatEnvironmentUnsigned(
	char *buffer,
	size_t buffer_bytes,
	uint32_t value);

/* ---- tp_collective stanza parsing (shared dsv4/glm52; policy-parameterized) --- */

/*
 * The four knobs on which the two source adapters legitimately disagreed.
 * adaptive_algorithm_mask/count pin the required "algorithms" array
 * (dsv4: all three known; glm52: recursive_doubling alone).
 * require_zero_thresholds selects the threshold law (dsv4: both nonzero
 * AND ordered d2a < split_ring; glm52: both must be zero).
 */
typedef struct SparkAdapterTpCollectivePolicy
{
	uint32_t adaptive_algorithm_mask;
	uint32_t adaptive_algorithm_count;
	uint32_t require_zero_thresholds;
} SparkAdapterTpCollectivePolicy;

typedef struct SparkAdapterTpCollectiveParsed
{
	uint32_t backend_kind;
	uint64_t collective_identifier;
	uint16_t listen_port;
	uint16_t control_port_base;
	uint32_t connect_timeout_milli;
	uint32_t operation_timeout_milli;
	uint16_t peer_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	char backend_module_path[SPARK_INTERNAL_PATH_BYTES];
	SparkTpDeviceCollectiveTopology topology;
} SparkAdapterTpCollectiveParsed;

/*
 * Parse and validate one "tp_collective" stanza off the configuration root.
 * expected_rank_count pins the peer_hosts/peer_ports/rail-host counts
 * (dsv4: stage count; glm52: tp_degree). validate_port_span enforces the
 * consecutive-port contract (ports[i] == ports[0] + i, overflow-guarded)
 * and fills control_port_base. Statement order and error codes are lifted
 * verbatim from the two adapters; the ONLY deliberate tightening is the
 * overflow-guarded span compare (the unguarded form accepted wrapped port
 * aliases near UINT16_MAX).
 */
SparkStatus SparkAdapterLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	uint32_t expected_rank_count,
	uint32_t validate_port_span,
	const SparkAdapterTpCollectivePolicy *policy,
	SparkAdapterTpCollectiveParsed *parsed);

/* ---- shared admission glue ------------------------------------------------ */

/*
 * Build the admission request from the serving submission when non-NULL,
 * otherwise from the driver frame itself; evaluate-and-apply it against the
 * loaded driver; submit the frame when submit_on_apply is set and admission
 * succeeded. Exact statement sequence of the four adapters' admit sites.
 */
SparkStatus SparkAdapterAdmitFrame(
	const SparkLoadedModelDriver *driver,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame,
	uint32_t submit_on_apply);
#ifdef __cplusplus
}
#endif
