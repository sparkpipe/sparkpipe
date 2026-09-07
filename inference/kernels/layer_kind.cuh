#pragma once


enum LmLayerKind
{
	LM_LAYER_FULL = 0,
	LM_LAYER_WINDOW = 1,
	LM_LAYER_SPARSE = 2,
	LM_LAYER_COMPRESSED = 3,
	LM_LAYER_LATENT = 4,
	LM_LAYER_RECURRENT = 5,
	LM_LAYER_KIND_COUNT = 6
};

static inline int LmLayerKindGrows(enum LmLayerKind kind)
{
	return(kind != LM_LAYER_RECURRENT);
}

static inline int LmLayerKindSelects(enum LmLayerKind kind)
{
	return(kind == LM_LAYER_WINDOW || kind == LM_LAYER_SPARSE ||
		kind == LM_LAYER_COMPRESSED);
}
