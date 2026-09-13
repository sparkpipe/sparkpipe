#pragma once

// Which attention a layer runs.
//
// Four of six models alternate attention kinds by layer index and each said so
// differently: one family had an enum with no selector, two families had
// the same boolean predicate under two names, a fourth had nothing at all
// and exported one entry point for three kinds. A fifth is uniform and so never
// needed to say anything.
//
// Eight distinct kinds across the six collapse to these six, because CSA and
// HCA are one mechanism at two compression rates, and gated DeltaNet and Delta Attention
// Delta Attention are one recurrence with two gate parameterisations.
//
// A MODEL DECLARES A FUNCTION OF THE LAYER INDEX, NOT A TABLE. Three of the
// four patterns are periodic and one is not quite - the HCA family has two
// exceptional layers at the front and alternates after. A macro taking the
// index expresses both; a literal array expresses both too but has to be
// written out 43 or 61 times and kept in step with LAYERS by hand.

enum LmLayerKind
{
	// Softmax over the whole cache.
	LM_LAYER_FULL = 0,
	// Softmax over the last N positions. A window is a position list, which is
	// the same argument the sparse path takes.
	LM_LAYER_WINDOW = 1,
	// An indexer selects top-k positions and attention runs over those.
	LM_LAYER_SPARSE = 2,
	// The same selection at a higher compression rate. Separate from SPARSE
	// because the rate changes the rope theta, not because the kernel differs.
	LM_LAYER_COMPRESSED = 3,
	// Latent-absorbed: one shared KV row per slot regardless of head count.
	LM_LAYER_LATENT = 4,
	// A recurrent state. No cache, no growth with context.
	LM_LAYER_RECURRENT = 5,
	LM_LAYER_KIND_COUNT = 6
};

// Does this kind read a KV cache that grows with context? The scheduler needs
// exactly this and nothing else about the kind: a recurrent layer's resident
// cost is fixed per sequence, so it does not enter the pool arithmetic.
static inline int LmLayerKindGrows(enum LmLayerKind kind)
{
	return(kind != LM_LAYER_RECURRENT);
}

// Does this kind attend over a subset chosen at runtime? SPARSE and COMPRESSED
// pass selected_positions; WINDOW passes a position list built from the window.
// FULL and LATENT pass neither, and RECURRENT has no positions at all.
static inline int LmLayerKindSelects(enum LmLayerKind kind)
{
	return(kind == LM_LAYER_WINDOW || kind == LM_LAYER_SPARSE ||
		kind == LM_LAYER_COMPRESSED);
}
