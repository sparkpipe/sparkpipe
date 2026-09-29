#include "sparkpipe/spark_graph_reloc.h"
#include "sparkpipe/spark_error_site.h"
#include <stdlib.h>
#include <string.h>

static const char *const SparkGraphRelocReasonNames[] =
{
	"GRAPH-RELOC-OK",
	"GRAPH-RELOC-UNKNOWN",
	"GRAPH-RELOC-AMBIGUOUS",
	"GRAPH-RELOC-NODE-KIND",
	"GRAPH-RELOC-NODE-QUERY",
	"GRAPH-RELOC-CAPACITY",
	"GRAPH-RELOC-REGION",
	"GRAPH-RELOC-SHAPE",
	"GRAPH-RELOC-CONSTANT",
	"GRAPH-RELOC-SITE",
	"GRAPH-RELOC-SLOT",
	"GRAPH-RELOC-TARGET",
	"GRAPH-RELOC-APPLY"
};

const char *SparkGraphRelocReasonName(uint32_t reason)
{
	if ( reason >= sizeof(SparkGraphRelocReasonNames) / sizeof(SparkGraphRelocReasonNames[0]) )
		return("GRAPH-RELOC-INVALID");
	return(SparkGraphRelocReasonNames[reason]);
}

static SparkStatus SparkGraphRelocRefuse(SparkGraphRelocFault *fault,SparkStatus status,uint32_t reason,uint32_t node,uint32_t byte_offset,uint64_t word)
{
	if ( fault != 0 )
	{
		fault->reason = reason;
		fault->node = node;
		fault->byte_offset = byte_offset;
		fault->word = word;
		fault->detail = 0u;
	}
	SPARK_FAIL(status);
}

void SparkGraphRelocRegistryInit(SparkGraphRelocRegistry *registry,SparkGraphRelocRegion *regions,SparkGraphRelocKey *keys,uint32_t capacity,uint64_t window_margin)
{
	memset(registry,0,sizeof(*registry));
	registry->regions = regions;
	registry->keys = keys;
	registry->capacity = capacity;
	registry->window_margin = window_margin;
}

void SparkGraphRelocRegistryReset(SparkGraphRelocRegistry *registry)
{
	registry->count = 0u;
	registry->sealed = 0u;
	registry->window_low = 0u;
	registry->window_high = 0u;
}

SparkStatus SparkGraphRelocRegister(SparkGraphRelocRegistry *registry,uint32_t id,const void *base,uint64_t bytes,uint32_t region_class,uint32_t slot)
{
	SparkGraphRelocRegion *region;
	uint64_t address = (uint64_t)(uintptr_t)base;
	if ( registry == 0 || address < SPARK_GRAPH_RELOC_WORD_FLOOR || bytes == 0u || address > UINT64_MAX - bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( registry->count >= registry->capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	region = &registry->regions[registry->count++];
	region->base = address;
	region->bytes = bytes;
	region->id = id;
	region->region_class = region_class;
	region->slot = slot;
	region->reserved = 0u;
	registry->sealed = 0u;
	return(SPARK_STATUS_OK);
}

static int SparkGraphRelocCompareBase(const void *left,const void *right)
{
	const SparkGraphRelocRegion *a = (const SparkGraphRelocRegion *)left,*b = (const SparkGraphRelocRegion *)right;
	return(a->base < b->base ? -1 : (a->base > b->base ? 1 : 0));
}

static int SparkGraphRelocCompareKey(const void *left,const void *right)
{
	const SparkGraphRelocKey *a = (const SparkGraphRelocKey *)left,*b = (const SparkGraphRelocKey *)right;
	if ( a->id != b->id )
		return(a->id < b->id ? -1 : 1);
	return(a->slot < b->slot ? -1 : (a->slot > b->slot ? 1 : 0));
}

static void SparkGraphRelocWindow(SparkGraphRelocRegistry *registry)
{
	const SparkGraphRelocRegion *last = &registry->regions[registry->count - 1u];
	uint64_t low = registry->regions[0].base,high = last->base + last->bytes,margin = registry->window_margin;
	registry->window_low = low > margin ? low - margin : 0u;
	registry->window_high = high > UINT64_MAX - margin ? UINT64_MAX : high + margin;
}

SparkStatus SparkGraphRelocSeal(SparkGraphRelocRegistry *registry,SparkGraphRelocFault *fault)
{
	SparkGraphRelocRegion *regions;
	SparkGraphRelocKey *keys;
	uint32_t index;
	if ( registry == 0 || registry->count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	regions = registry->regions;
	keys = registry->keys;
	qsort(regions,registry->count,sizeof(*regions),SparkGraphRelocCompareBase);
	for (index=0u; index<registry->count; index++)
	{
		keys[index].id = regions[index].id;
		keys[index].slot = regions[index].slot;
		keys[index].region = index;
	}
	qsort(keys,registry->count,sizeof(*keys),SparkGraphRelocCompareKey);
	for (index=1u; index<registry->count; index++)
	{
		if ( regions[index - 1u].base + regions[index - 1u].bytes > regions[index].base )
			return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_REGION,SPARK_GRAPH_RELOC_NONE,regions[index].id,regions[index].base));
		if ( keys[index - 1u].id == keys[index].id && keys[index - 1u].slot == keys[index].slot )
			return(SparkGraphRelocRefuse(fault,SPARK_STATUS_DUPLICATE,SPARK_GRAPH_RELOC_REASON_REGION,SPARK_GRAPH_RELOC_NONE,keys[index].id,keys[index].slot));
	}
	SparkGraphRelocWindow(registry);
	registry->sealed = 1u;
	return(SPARK_STATUS_OK);
}

void SparkGraphRelocClassify(const SparkGraphRelocRegistry *registry,uint64_t word,SparkGraphRelocMatch *match)
{
	const SparkGraphRelocRegion *regions = registry->regions;
	uint32_t low = 0u,high = registry->count,middle;
	match->offset = 0u;
	match->region = SPARK_GRAPH_RELOC_NONE;
	match->verdict = SPARK_GRAPH_RELOC_WORD_CONSTANT;
	if ( word < SPARK_GRAPH_RELOC_WORD_FLOOR )
		return;
	while ( low < high )
	{
		middle = low + (high - low) / 2u;
		if ( regions[middle].base <= word )
			low = middle + 1u;
		else
			high = middle;
	}
	if ( low != 0u && word - regions[low - 1u].base <= regions[low - 1u].bytes )
	{
		match->region = low - 1u;
		match->offset = word - regions[low - 1u].base;
		match->verdict = low >= 2u && regions[low - 2u].base + regions[low - 2u].bytes == word ? SPARK_GRAPH_RELOC_WORD_AMBIGUOUS : SPARK_GRAPH_RELOC_WORD_REGION;
		return;
	}
	if ( (word >= registry->window_low && word <= registry->window_high) || (registry->probe != 0 && registry->probe(registry->probe_context,word) != 0u) )
		match->verdict = SPARK_GRAPH_RELOC_WORD_UNKNOWN;
}

uint32_t SparkGraphRelocFind(const SparkGraphRelocRegistry *registry,uint32_t id,uint32_t slot)
{
	SparkGraphRelocKey key = {id,slot,0u};
	uint32_t low = 0u,high,middle;
	int order;
	if ( registry == 0 || registry->sealed == 0u )
		return(SPARK_GRAPH_RELOC_NONE);
	high = registry->count;
	while ( low < high )
	{
		middle = low + (high - low) / 2u;
		order = SparkGraphRelocCompareKey(&registry->keys[middle],&key);
		if ( order == 0 )
			return(registry->keys[middle].region);
		if ( order < 0 )
			low = middle + 1u;
		else
			high = middle;
	}
	return(SPARK_GRAPH_RELOC_NONE);
}

void SparkGraphRelocImageReset(SparkGraphRelocImage *image,uint32_t slot)
{
	image->blob_bytes = 0u;
	image->word_count = 0u;
	image->node_count = 0u;
	image->param_count = 0u;
	image->site_count = 0u;
	image->kernel_count = 0u;
	image->slot = slot;
}

static uint64_t SparkGraphRelocAlign(uint64_t bytes)
{
	return((bytes + SPARK_GRAPH_RELOC_BLOB_ALIGN - 1u) & ~(uint64_t)(SPARK_GRAPH_RELOC_BLOB_ALIGN - 1u));
}

uint8_t *SparkGraphRelocImageTail(SparkGraphRelocImage *image,uint32_t bytes)
{
	uint64_t offset = SparkGraphRelocAlign(image->blob_bytes);
	if ( offset > image->blob_capacity || bytes > image->blob_capacity - offset )
		return(0);
	memset(image->blob + offset,0,bytes);
	return(image->blob + offset);
}

static SparkStatus SparkGraphRelocScanWord(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,SparkGraphRelocNode *node,uint32_t byte_offset,SparkGraphRelocFault *fault)
{
	const SparkGraphRelocRegion *region;
	SparkGraphRelocSite *site;
	SparkGraphRelocMatch match;
	uint64_t value;
	memcpy(&value,image->blob + node->blob_offset + byte_offset,sizeof(value));
	image->word_count++;
	SparkGraphRelocClassify(registry,value,&match);
	if ( match.verdict == SPARK_GRAPH_RELOC_WORD_CONSTANT )
		return(SPARK_STATUS_OK);
	if ( match.verdict != SPARK_GRAPH_RELOC_WORD_REGION )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,match.verdict == SPARK_GRAPH_RELOC_WORD_UNKNOWN ? SPARK_GRAPH_RELOC_REASON_UNKNOWN : SPARK_GRAPH_RELOC_REASON_AMBIGUOUS,image->node_count,byte_offset,value));
	region = &registry->regions[match.region];
	if ( region->slot != SPARK_GRAPH_RELOC_SLOT_SHARED && region->slot != image->slot )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_SLOT,image->node_count,byte_offset,value));
	if ( image->site_count >= image->site_capacity )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_CAPACITY_EXCEEDED,SPARK_GRAPH_RELOC_REASON_CAPACITY,image->node_count,byte_offset,value));
	site = &image->sites[image->site_count++];
	site->offset = match.offset;
	site->node = image->node_count;
	site->byte_offset = byte_offset;
	site->region_id = region->id;
	site->region_slot = region->slot;
	node->site_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGraphRelocScanNode(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,SparkGraphRelocNode *node,SparkGraphRelocFault *fault)
{
	const SparkGraphRelocParam *param;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index,word;
	for (index=0u; status==SPARK_STATUS_OK && index<node->param_count; index++)
	{
		param = &image->params[node->param_first + index];
		if ( (param->offset % 8u) != 0u )
			continue;
		for (word=0u; status==SPARK_STATUS_OK && word + 8u <= param->bytes; word+=8u)
			status = SparkGraphRelocScanWord(image,registry,node,param->offset + word,fault);
	}
	return(status);
}

static SparkStatus SparkGraphRelocCheckInput(const SparkGraphRelocImage *image,const SparkGraphRelocNodeInput *input,SparkGraphRelocFault *fault)
{
	uint32_t index,end = 0u;
	uint64_t offset = SparkGraphRelocAlign(image->blob_bytes);
	if ( image->node_count >= image->node_capacity || input->param_count > image->param_capacity - image->param_count || offset > image->blob_capacity || input->blob_bytes > image->blob_capacity - offset )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_CAPACITY_EXCEEDED,SPARK_GRAPH_RELOC_REASON_CAPACITY,image->node_count,0u,0u));
	for (index=0u; index<input->param_count; index++)
	{
		const SparkGraphRelocParam *param = &input->params[index];
		if ( param->offset < end || param->offset > input->blob_bytes || param->bytes > input->blob_bytes - param->offset )
			return(SparkGraphRelocRefuse(fault,SPARK_STATUS_INVALID_ARGUMENT,SPARK_GRAPH_RELOC_REASON_SHAPE,image->node_count,param->offset,param->bytes));
		end = param->offset + param->bytes;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGraphRelocImageCommit(SparkGraphRelocImage *image,const SparkGraphRelocRegistry *registry,const SparkGraphRelocNodeInput *input,SparkGraphRelocFault *fault)
{
	SparkGraphRelocNode *node;
	SparkStatus status;
	if ( image == 0 || registry == 0 || input == 0 || registry->sealed == 0u || (input->param_count != 0u && input->params == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGraphRelocCheckInput(image,input,fault);
	if ( status != SPARK_STATUS_OK )
		return(status);
	node = &image->nodes[image->node_count];
	node->function = input->function;
	node->kind = input->kind;
	node->param_first = image->param_count;
	node->param_count = input->param_count;
	node->blob_offset = (uint32_t)SparkGraphRelocAlign(image->blob_bytes);
	node->blob_bytes = input->blob_bytes;
	node->site_first = image->site_count;
	node->site_count = 0u;
	node->reserved = 0u;
	image->handles[image->node_count] = input->handle;
	memcpy(&image->params[image->param_count],input->params,(size_t)input->param_count * sizeof(*input->params));
	status = SparkGraphRelocScanNode(image,registry,node,fault);
	if ( status != SPARK_STATUS_OK )
	{
		image->site_count = node->site_first;
		return(status);
	}
	image->param_count += input->param_count;
	image->blob_bytes = (uint64_t)node->blob_offset + node->blob_bytes;
	image->kernel_count += input->kind == SPARK_GRAPH_RELOC_NODE_KERNEL ? 1u : 0u;
	image->node_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGraphRelocCompareSite(const SparkGraphRelocImage *first,const SparkGraphRelocImage *second,const SparkGraphRelocSite *a,const SparkGraphRelocSite *b,SparkGraphRelocFault *fault)
{
	uint32_t shared_a = a->region_slot == SPARK_GRAPH_RELOC_SLOT_SHARED ? 1u : 0u,shared_b = b->region_slot == SPARK_GRAPH_RELOC_SLOT_SHARED ? 1u : 0u;
	if ( a->byte_offset != b->byte_offset || a->region_id != b->region_id || a->offset != b->offset )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_SITE,a->node,a->byte_offset,b->offset));
	if ( shared_a != shared_b || (shared_a == 0u && (a->region_slot != first->slot || b->region_slot != second->slot)) )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_SLOT,a->node,a->byte_offset,b->region_slot));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGraphRelocCompareBytes(const uint8_t *a,const uint8_t *b,uint32_t node,uint32_t begin,uint32_t end,SparkGraphRelocFault *fault)
{
	uint64_t word = 0u;
	uint32_t index,aligned;
	for (index=begin; index<end; index++)
		if ( a[index] != b[index] )
		{
			aligned = index & ~7u;
			if ( aligned + 8u <= end )
				memcpy(&word,b + aligned,sizeof(word));
			return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_CONSTANT,node,index,word));
		}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGraphRelocCompareNode(const SparkGraphRelocImage *first,const SparkGraphRelocImage *second,uint32_t index,SparkGraphRelocFault *fault)
{
	const SparkGraphRelocNode *a = &first->nodes[index],*b = &second->nodes[index];
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t site,cursor = 0u,next;
	if ( a->kind != b->kind || a->function != b->function || a->blob_bytes != b->blob_bytes || a->param_count != b->param_count || a->site_count != b->site_count || memcmp(&first->params[a->param_first],&second->params[b->param_first],(size_t)a->param_count * sizeof(SparkGraphRelocParam)) != 0 )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_SHAPE,index,0u,b->kind));
	for (site=0u; status==SPARK_STATUS_OK && site<=a->site_count; site++)
	{
		next = site < a->site_count ? first->sites[a->site_first + site].byte_offset : a->blob_bytes;
		status = SparkGraphRelocCompareBytes(first->blob + a->blob_offset,second->blob + b->blob_offset,index,cursor,next,fault);
		if ( status == SPARK_STATUS_OK && site < a->site_count )
			status = SparkGraphRelocCompareSite(first,second,&first->sites[a->site_first + site],&second->sites[b->site_first + site],fault);
		cursor = next + 8u;
	}
	return(status);
}

SparkStatus SparkGraphRelocValidatePair(const SparkGraphRelocImage *first,const SparkGraphRelocImage *second,SparkGraphRelocFault *fault)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	if ( first == 0 || second == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( first->node_count != second->node_count )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_SHAPE,first->node_count < second->node_count ? first->node_count : second->node_count,0u,second->node_count));
	for (index=0u; status==SPARK_STATUS_OK && index<first->node_count; index++)
		status = SparkGraphRelocCompareNode(first,second,index,fault);
	return(status);
}

static SparkStatus SparkGraphRelocRebaseSite(const SparkGraphRelocImage *image,const SparkGraphRelocRegistry *target,uint32_t target_slot,const SparkGraphRelocSite *site,uint8_t *blob,SparkGraphRelocFault *fault)
{
	uint32_t shared = site->region_slot == SPARK_GRAPH_RELOC_SLOT_SHARED ? 1u : 0u,region;
	uint64_t value;
	if ( shared == 0u && site->region_slot != image->slot )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_VALIDATION_FAILED,SPARK_GRAPH_RELOC_REASON_SLOT,site->node,site->byte_offset,site->region_slot));
	region = SparkGraphRelocFind(target,site->region_id,shared != 0u ? SPARK_GRAPH_RELOC_SLOT_SHARED : target_slot);
	if ( region == SPARK_GRAPH_RELOC_NONE || site->offset > target->regions[region].bytes )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_NOT_FOUND,SPARK_GRAPH_RELOC_REASON_TARGET,site->node,site->byte_offset,site->region_id));
	value = target->regions[region].base + site->offset;
	memcpy(blob + image->nodes[site->node].blob_offset + site->byte_offset,&value,sizeof(value));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGraphRelocRebase(const SparkGraphRelocImage *image,const SparkGraphRelocRegistry *target,uint32_t target_slot,uint8_t *blob,uint64_t blob_capacity,SparkGraphRelocFault *fault)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	if ( image == 0 || target == 0 || blob == 0 || target->sealed == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( image->blob_bytes > blob_capacity )
		return(SparkGraphRelocRefuse(fault,SPARK_STATUS_CAPACITY_EXCEEDED,SPARK_GRAPH_RELOC_REASON_CAPACITY,SPARK_GRAPH_RELOC_NONE,0u,image->blob_bytes));
	memcpy(blob,image->blob,(size_t)image->blob_bytes);
	for (index=0u; status==SPARK_STATUS_OK && index<image->site_count; index++)
		status = SparkGraphRelocRebaseSite(image,target,target_slot,&image->sites[index],blob,fault);
	return(status);
}

SparkStatus SparkGraphRelocHeldReset(const SparkGraphRelocImage *image,uint8_t *held,uint64_t held_capacity)
{
	if ( image == 0 || held == 0 || held == image->blob )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( image->blob_bytes > held_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(held,image->blob,(size_t)image->blob_bytes);
	return(SPARK_STATUS_OK);
}

static uint64_t SparkGraphRelocCarve(uint64_t *cursor,uint64_t bytes,uint8_t *base,void **out)
{
	uint64_t offset = SparkGraphRelocAlign(*cursor);
	*cursor = offset + bytes;
	if ( base != 0 )
		*out = base + offset;
	return(*cursor);
}

static void SparkGraphRelocBindImage(SparkGraphRelocImage *image,const SparkGraphRelocCapacity *capacity,void *const *parts)
{
	image->nodes = parts[0];
	image->handles = parts[1];
	image->params = parts[2];
	image->sites = parts[3];
	image->blob = parts[4];
	image->blob_capacity = capacity->blob_bytes;
	image->node_capacity = capacity->nodes;
	image->param_capacity = capacity->params;
	image->site_capacity = capacity->sites;
}

static uint64_t SparkGraphRelocLayout(const SparkGraphRelocCapacity *capacity,uint8_t *base,SparkGraphRelocWorkspace *workspace)
{
	uint64_t cursor = sizeof(SparkGraphRelocWorkspace),total;
	void *regions = 0,*keys = 0,*patch = 0,*held[2] = {0,0};
	uint32_t index;
	(void)SparkGraphRelocCarve(&cursor,(uint64_t)capacity->regions * sizeof(SparkGraphRelocRegion),base,&regions);
	(void)SparkGraphRelocCarve(&cursor,(uint64_t)capacity->regions * sizeof(SparkGraphRelocKey),base,&keys);
	for (index=0u; index<2u; index++)
	{
		void *parts[5] = {0,0,0,0,0};
		(void)SparkGraphRelocCarve(&cursor,(uint64_t)capacity->nodes * sizeof(SparkGraphRelocNode),base,&parts[0]);
		(void)SparkGraphRelocCarve(&cursor,(uint64_t)capacity->nodes * sizeof(void *),base,&parts[1]);
		(void)SparkGraphRelocCarve(&cursor,(uint64_t)capacity->params * sizeof(SparkGraphRelocParam),base,&parts[2]);
		(void)SparkGraphRelocCarve(&cursor,(uint64_t)capacity->sites * sizeof(SparkGraphRelocSite),base,&parts[3]);
		(void)SparkGraphRelocCarve(&cursor,capacity->blob_bytes,base,&parts[4]);
		(void)SparkGraphRelocCarve(&cursor,capacity->blob_bytes,base,&held[index]);
		if ( base != 0 )
		{
			SparkGraphRelocBindImage(&workspace->images[index],capacity,parts);
			workspace->held_blob[index] = held[index];
		}
	}
	total = SparkGraphRelocCarve(&cursor,capacity->blob_bytes,base,&patch);
	if ( base != 0 )
	{
		SparkGraphRelocRegistryInit(&workspace->registry,regions,keys,capacity->regions,0u);
		workspace->patch_blob = patch;
	}
	return(total);
}

SparkStatus SparkGraphRelocWorkspaceCreate(const SparkGraphRelocCapacity *capacity,uint64_t window_margin,SparkGraphRelocWorkspace **out)
{
	SparkGraphRelocWorkspace *workspace;
	uint64_t bytes;
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( capacity == 0 || capacity->regions == 0u || capacity->nodes == 0u || capacity->params == 0u || capacity->sites == 0u || capacity->blob_bytes == 0u || capacity->blob_bytes > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	bytes = SparkGraphRelocLayout(capacity,0,0);
	if ( bytes > (uint64_t)SIZE_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	workspace = (SparkGraphRelocWorkspace *)calloc(1u,(size_t)bytes);
	if ( workspace == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)SparkGraphRelocLayout(capacity,(uint8_t *)workspace,workspace);
	workspace->registry.window_margin = window_margin;
	*out = workspace;
	return(SPARK_STATUS_OK);
}

void SparkGraphRelocWorkspaceDestroy(SparkGraphRelocWorkspace *workspace)
{
	free(workspace);
}
