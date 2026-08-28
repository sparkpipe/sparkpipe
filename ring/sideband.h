#pragma once

// Sidebands: what crosses a rank boundary besides the hidden state.
//
// Harvested from three functions in the old decode stage that did the same thing
// three ways - MaybeImportStageSliceIndexShareSideband, MaybeCarryDsparkHiddenTaps
// and LaunchContextPrefixSparseIndices. Each attached a different payload to a
// hidden-state packet, each had its own import and carry path, and none of them
// named the shared idea.
//
// The idea is that a pipeline rank produces two kinds of output. The hidden
// state, which the next rank consumes and which every rank produces. And
// derived work - a selection, a tap, an index - which SOME later rank needs and
// which recomputing costs more than carrying.
//
// WHY CARRYING BEATS RECOMPUTING, per payload:
//
//   INDEX SHARE. GLM 5.2 selects DSA positions once per four layers. A rank
//   whose slice starts mid-group would recompute a selection the previous rank
//   already made - correct output, and at thirteen ranks that is thirteen index
//   passes where the model calls for nineteen in total. The selection is 2048
//   uint32 per row; the pass that produces it reads the whole cache.
//
//   HIDDEN TAP. A drafter needs hidden states from particular layers, and those
//   layers live on whichever rank owns them. Tapping in place and carrying is
//   the only option that does not run the model twice.
//
//   PREFIX INDICES. A shared prompt prefix has already had its sparse selection
//   computed by whoever first served it. Recomputing per request throws away
//   exactly the sharing the prefix cache exists to create.
//
// WHAT A SIDEBAND IS NOT. It is not a general message. It is a fixed set of
// payload kinds with fixed shapes, because a rank must be able to size its
// receive buffer before the sender exists - and a variable payload turns a
// pipeline stall into an allocation.

#include <stdint.h>

#define LM_SIDEBAND_OK 0
#define LM_SIDEBAND_ERR_SHAPE (-91)
#define LM_SIDEBAND_ERR_ABSENT (-92)
#define LM_SIDEBAND_ERR_STALE (-93)
#define LM_SIDEBAND_ERR_CAPACITY (-94)

typedef enum LmSidebandKind
{
	LM_SIDEBAND_NONE = 0,
	LM_SIDEBAND_INDEX_SHARE = 1,     /* selected positions, valid for a layer group */
	LM_SIDEBAND_HIDDEN_TAP = 2,      /* a layer's hidden state, for a drafter */
	LM_SIDEBAND_PREFIX_INDICES = 3   /* a cached prefix's selection */
}
LmSidebandKind;

// The header every payload carries.
//
// source_layer and generation together are what make a stale sideband
// detectable. A rank that receives a selection made for a different layer group,
// or for a previous step, must reject it rather than use it - using it attends
// to positions chosen for another context, which is fluent and wrong. The old
// code checked the layer; it did not check the generation.
typedef struct LmSidebandHeader
{
	uint32_t kind;
	uint32_t source_rank;
	uint32_t source_layer;
	uint32_t generation;             /* the step this was produced in */
	uint32_t row_count;
	uint32_t element_count;          /* per row */
	uint32_t element_bytes;
	uint32_t reserved;
}
LmSidebandHeader;

// Which layers a drafter taps, and which rank consumes them.
//
// THIS IS WHERE DRAFTING PLACEMENT LIVES. The old code assumed the last rank,
// because that is where the logits are. For GLM 5.2 that is the wrong choice:
// its first three layers are dense and fast, so the rank holding them has slack
// the last rank does not, and drafting there costs less wall time. A model whose
// early layers are expensive wants the opposite.
//
// So the consuming rank is a field, not an assumption. A model description sets
// it; nothing in the kernels or the transport cares which value it takes.
typedef struct LmTapPlan
{
	uint32_t tap_layer[8];
	uint32_t tap_count;
	uint32_t consumer_rank;          /* which rank runs the drafter */
}
LmTapPlan;

static int32_t LmSidebandPayloadBytes(const LmSidebandHeader *header, uint64_t *bytes_out)
{
	uint64_t bytes;
	if ( header == 0 || bytes_out == 0 )
		return(LM_SIDEBAND_ERR_SHAPE);
	if ( header->element_bytes == 0u || header->element_count == 0u )
		return(LM_SIDEBAND_ERR_SHAPE);
	bytes = (uint64_t)header->row_count * header->element_count * header->element_bytes;
	// A payload that would not fit a rank's receive buffer must fail here rather
	// than at the copy, because at the copy the sender has already committed and
	// the pipeline is mid-stage.
	if ( bytes > (uint64_t)1u << 32 )
		return(LM_SIDEBAND_ERR_CAPACITY);
	*bytes_out = bytes;
	return(LM_SIDEBAND_OK);
}

// Is this sideband usable by the rank that received it?
//
// Three questions, and the old code asked one. The kind must match what the
// consumer wants. The generation must be the current step - a selection from the
// previous step was made against a shorter context and omits the position that
// was just written. And for an index share, the layer must be in the same group,
// because a selection is only valid for the group it was computed for.
static int32_t LmSidebandAcceptable(const LmSidebandHeader *header, uint32_t want_kind, uint32_t current_generation, uint32_t consumer_layer, uint32_t share_group_layers)
{
	if ( header == 0 || header->kind == LM_SIDEBAND_NONE )
		return(LM_SIDEBAND_ERR_ABSENT);
	if ( header->kind != want_kind )
		return(LM_SIDEBAND_ERR_SHAPE);
	if ( header->generation != current_generation )
		return(LM_SIDEBAND_ERR_STALE);
	if ( header->kind == LM_SIDEBAND_INDEX_SHARE && share_group_layers != 0u )
	{
		if ( (header->source_layer / share_group_layers)
			!= (consumer_layer / share_group_layers) )
			return(LM_SIDEBAND_ERR_STALE);
	}
	return(LM_SIDEBAND_OK);
}

// Should this rank produce an index-share sideband for the next one?
//
// Only when it computed a selection its neighbour will still be inside the group
// for. A rank whose slice ends exactly on a group boundary produces nothing,
// because the next rank starts a new group and must select for itself.
static int32_t LmSidebandShouldExportIndexShare(uint32_t last_layer_on_rank, uint32_t share_group_layers)
{
	if ( share_group_layers == 0u )
		return(0);
	return(((last_layer_on_rank + 1u) % share_group_layers) != 0u);
}

// Which of a rank's layers need tapping, given the plan.
//
// Returns the count and fills the caller's array with layer indices local to
// this rank. A rank owning none of the tapped layers does no work and sends
// nothing, which is the common case: a plan taps three layers out of
// seventy-eight, so ten of thirteen ranks are uninvolved.
static uint32_t LmSidebandTapsOnRank(const LmTapPlan *plan, uint32_t first_layer, uint32_t layer_count, uint32_t *local_out, uint32_t capacity)
{
	uint32_t index,found = 0u;
	if ( plan == 0 || local_out == 0 )
		return(0u);
	for (index = 0u; index < plan->tap_count && index < 8u; ++index)
	{
		uint32_t layer = plan->tap_layer[index];
		if ( layer < first_layer || layer >= first_layer + layer_count )
			continue;
		if ( found >= capacity )
			break;
		local_out[found++] = layer - first_layer;
	}
	return(found);
}

// Does this rank run the drafter?
//
// A field rather than "is this the last rank", which is the whole point of the
// harvest. The scheduler decides placement from the model description and the
// transport does not care.
static int32_t LmSidebandRankDrafts(const LmTapPlan *plan, uint32_t rank)
{
	return(plan != 0 && plan->consumer_rank == rank);
}

static void LmSidebandInitialiseHeader(LmSidebandHeader *header, uint32_t kind, uint32_t rank, uint32_t layer, uint32_t generation, uint32_t rows, uint32_t elements, uint32_t element_bytes)
{
	header->kind = kind;
	header->source_rank = rank;
	header->source_layer = layer;
	header->generation = generation;
	header->row_count = rows;
	header->element_count = elements;
	header->element_bytes = element_bytes;
	header->reserved = 0u;
}
