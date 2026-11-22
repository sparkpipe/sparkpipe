#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_cubin.h"
#include "runtime/device/nvrm/nvrm_manifest.h"
#include "runtime/device/nvrm/nvrm_names.h"
#include "runtime/device/nvrm/nvrm_qmd.h"
#include "tests/nvrm/fixture_kernels.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif
#ifndef SPARK_NVRM_FIXTURE_MANIFEST
#define SPARK_NVRM_FIXTURE_MANIFEST "tests/fixtures/nvrm/fixture_kernels.sm_121a.manifest"
#endif

#define RESERVED_SHARED_BYTES 1024u
#define PARAMETER_END(type, last) ((uint32_t)(offsetof(type,last) + sizeof(((type *)0)->last)))

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

typedef struct FirstLightParameters { uint32_t *output; uint32_t value; } FirstLightParameters;
typedef struct ParameterMixParameters { uint8_t small; const uint64_t *table; uint16_t lanes; SparkNvrmFixtureShape shape; float scale; int8_t sign; double *output; } ParameterMixParameters;
typedef struct StaticSharedParameters { float *output; uint32_t count; } StaticSharedParameters;
typedef struct LocalStackParameters { uint32_t *output; uint32_t pick; } LocalStackParameters;
typedef struct LargeParameterParameters { SparkNvrmFixtureBlock block; uint32_t *output; } LargeParameterParameters;
typedef struct HugeParameterParameters { uint32_t *output; SparkNvrmFixtureHuge block; } HugeParameterParameters;
typedef struct RmsNormParameters { const uint16_t *input; const uint16_t *weight; uint16_t *output; uint32_t dimension; uint32_t row_stride; float epsilon; } RmsNormParameters;
typedef struct StreamReadParameters { const void *source; uint64_t count; uint32_t *sink; } StreamReadParameters;
typedef struct StreamCopyParameters { const void *source; void *destination; uint64_t count; } StreamCopyParameters;
typedef struct StreamFillParameters { void *destination; uint64_t count; uint32_t value; } StreamFillParameters;
typedef struct FusedRmsNormParameters { const uint16_t *input; const uint16_t *residual; const uint16_t *weight; uint16_t *residual_out; uint16_t *output; uint32_t dimension; uint32_t row_stride; float epsilon; } FusedRmsNormParameters;

typedef struct ExpectedLayout
{
	const char *name;
	uint32_t parameter_bytes;
} ExpectedLayout;

static const ExpectedLayout expected_layouts[] = {
	{"SparkNvrmFirstLight", PARAMETER_END(FirstLightParameters,value)},
	{"SparkNvrmParameterMix", PARAMETER_END(ParameterMixParameters,output)},
	{"SparkNvrmStaticShared", PARAMETER_END(StaticSharedParameters,count)},
	{"SparkNvrmLocalStack", PARAMETER_END(LocalStackParameters,pick)},
	{"SparkNvrmLargeParameter", PARAMETER_END(LargeParameterParameters,output)},
	{"SparkNvrmHugeParameter", PARAMETER_END(HugeParameterParameters,block)},
	{SPARK_NVRM_FIXTURE_LONG_TEXT, PARAMETER_END(FirstLightParameters,value)},
	{"_Z19LmBf16RmsNormKernelILj256EEvPKtS1_Ptjjf", PARAMETER_END(RmsNormParameters,epsilon)},
	{"_Z28LmFusedResidualRmsNormKernelILj256EtLb0EEvPKtS1_PKT0_PtS5_jjf", PARAMETER_END(FusedRmsNormParameters,epsilon)},
	{"SparkNvrmStreamRead", PARAMETER_END(StreamReadParameters,sink)},
	{"SparkNvrmStreamCopy", PARAMETER_END(StreamCopyParameters,count)},
	{"SparkNvrmStreamFill", PARAMETER_END(StreamFillParameters,value)},
	{"SparkNvrmProbeDimensions", PARAMETER_END(FirstLightParameters,output)},
	{"SparkNvrmGlobalSwap", PARAMETER_END(FirstLightParameters,value)},
	{"SparkNvrmConstantTable", PARAMETER_END(FirstLightParameters,output)},
	{"SparkNvrmCallStack", PARAMETER_END(FirstLightParameters,value)},
	{"SparkNvrmDeepFrame", PARAMETER_END(FirstLightParameters,value)},
	{"SparkNvrmNoParameters", 0u},
};

static uint8_t *ReadFile(const char *path, uint64_t *bytes)
{
	FILE *file = fopen(path,"rb");
	uint8_t *data;
	long length;
	if ( file == 0 )
		return(0);
	fseek(file,0,SEEK_END);
	length = ftell(file);
	fseek(file,0,SEEK_SET);
	data = (uint8_t *)malloc((size_t)length + 1u);
	if ( data != 0 && fread(data,1u,(size_t)length,file) != (size_t)length )
	{
		free(data);
		data = 0;
	}
	fclose(file);
	if ( data != 0 )
	{
		data[length] = 0u;
		*bytes = (uint64_t)length;
	}
	return(data);
}

static uint64_t Le(const uint8_t *at, uint32_t width)
{
	uint64_t value = 0u;
	while ( width-- > 0u )
		value = (value << 8) | at[width];
	return(value);
}

static void PutLe(uint8_t *at, uint64_t value, uint32_t width)
{
	uint32_t index;
	for (index = 0u; index < width; index++)
		at[index] = (uint8_t)(value >> (8u * index));
}

static uint32_t ExpectedParameters(const char *name)
{
	char wanted[SPARK_NVRM_SHORT_NAME_BYTES],listed[SPARK_NVRM_SHORT_NAME_BYTES];
	uint32_t index;
	SparkNvrmShortName(name,wanted);
	for (index = 0u; index < sizeof(expected_layouts) / sizeof(expected_layouts[0]); index++)
	{
		SparkNvrmShortName(expected_layouts[index].name,listed);
		if ( strcmp(listed,wanted) == 0 )
			return(expected_layouts[index].parameter_bytes);
	}
	return(UINT32_MAX);
}

static void TestFixture(void)
{
	uint64_t bytes = 0u,manifest_bytes = 0u;
	uint8_t *blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes);
	char *manifest = (char *)ReadFile(SPARK_NVRM_FIXTURE_MANIFEST,&manifest_bytes);
	char *line;
	uint32_t listed = 0u;
	SparkNvrmCubin cubin;
	CHECK(blob != 0 && manifest != 0,"the fixture cubin and its compiler report are readable");
	if ( blob == 0 || manifest == 0 )
		return;
	CHECK(SparkNvrmCubinParse(blob,bytes,"fixture",&cubin) == SPARK_STATUS_OK,"the release-built fixture cubin parses");
	for (line = strtok(manifest,"\n"); line != 0; line = strtok(0,"\n"))
	{
		char what[512];
		char *name = line + strlen("kernel "),*space;
		unsigned registers,barriers,stack,stores,loads,smem,cumulative;
		const char *tail;
		const SparkNvrmCubinKernel *kernel;
		if ( strncmp(line,"kernel ",strlen("kernel ")) != 0 || (space = strchr(name,' ')) == 0 )
			continue;
		*space = '\0';
		if ( strncmp(space + 1,"short ",strlen("short ")) != 0 || (space = strchr(space + 1 + strlen("short "),' ')) == 0 )
			continue;
		if ( sscanf(space + 1,"registers %u barriers %u stack %u spill_stores %u spill_loads %u smem %u",&registers,&barriers,&stack,&stores,&loads,&smem) != 6 )
			continue;
		tail = strstr(space + 1," cumulative ");
		cumulative = tail != 0 ? (unsigned)strtoul(tail + strlen(" cumulative "),0,10) : stack;
		listed++;
		kernel = SparkNvrmCubinKernelNamed(&cubin,name);
		snprintf(what,sizeof(what),"the parser finds the compiler's kernel %.200s",name);
		CHECK(kernel != 0,what);
		if ( kernel == 0 )
			continue;
		snprintf(what,sizeof(what),"%.200s: registers %u, call-chain stack %u and shared %u match the compiler's report",name,kernel->registers,
			kernel->stack_bytes,kernel->shared_bytes);
		CHECK(kernel->registers == registers && kernel->stack_bytes == cumulative && kernel->shared_bytes == smem + RESERVED_SHARED_BYTES,what);
		snprintf(what,sizeof(what),"%.200s: parameters at %#x for %u bytes follow the driver words and the C layout",name,kernel->parameter_offset,kernel->parameter_bytes);
		CHECK(kernel->parameter_offset == SPARK_NVRM_BANK0_DRIVER_BYTES && kernel->parameter_bytes == ExpectedParameters(name),what);
		snprintf(what,sizeof(what),"%.200s: code and constant bank 0 are aligned and inside the image",name);
		CHECK(kernel->code_bytes != 0u && kernel->code_offset % SPARK_NVRM_CUBIN_SECTION_ALIGN == 0u &&
			kernel->code_offset + kernel->code_bytes <= cubin.image_bytes && kernel->bank[0].present != 0u &&
			kernel->bank[0].image_offset % SPARK_NVRM_CUBIN_SECTION_ALIGN == 0u &&
			kernel->bank[0].image_offset + kernel->bank[0].bytes <= cubin.image_bytes &&
			kernel->bank[0].bytes == (uint64_t)kernel->parameter_offset + kernel->parameter_bytes,what);
	}
	CHECK(listed == sizeof(expected_layouts) / sizeof(expected_layouts[0]) && cubin.kernel_count == listed,
		"the parser lists exactly the compiler's entry functions");
	CHECK(SparkNvrmCubinKernelNamed(&cubin,"SparkNvrmAbsent") == 0,"an absent kernel is not found");
	CHECK(cubin.relocation_count != 0u,"the module's device variable is reached through relocations");
	SparkNvrmCubinRelease(&cubin);
	CHECK(cubin.image == 0 && cubin.kernels == 0 && cubin.kernel_count == 0u,"release clears the parsed image");
	CHECK(SparkNvrmCubinParse(blob,63u,"fixture",&cubin) == SPARK_STATUS_PARSE_ERROR && cubin.image == 0,"a cubin cut inside its header is refused");
	PutLe(blob + 18u,62u,2u);
	CHECK(SparkNvrmCubinParse(blob,bytes,"fixture",&cubin) == SPARK_STATUS_TARGET_MISMATCH &&
		strcmp(SparkDeviceLastError()->operation,"library_load") == 0 && strcmp(SparkDeviceLastError()->owner,"fixture") == 0,
		"an ELF image for another machine is refused against library_load and its owner");
	free(blob);
	free(manifest);
}

static void TestManifest(void)
{
	uint64_t bytes = 0u,text_bytes = 0u;
	uint8_t *blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes);
	char *text = (char *)ReadFile(SPARK_NVRM_FIXTURE_MANIFEST,&text_bytes);
	SparkNvrmManifest manifest;
	SparkNvrmCubin cubin;
	uint32_t index,agree = 0u;
	CHECK(blob != 0 && text != 0,"the fixture cubin and its manifest are readable");
	if ( blob == 0 || text == 0 )
		return;
	CHECK(SparkNvrmCubinParse(blob,bytes,"manifest",&cubin) == SPARK_STATUS_OK,"the fixture parses");
	CHECK(SparkNvrmManifestParse(text,text_bytes,"manifest",&manifest) == SPARK_STATUS_OK && manifest.kernel_count == cubin.kernel_count,
		"the manifest parses and lists every kernel");
	CHECK(SparkNvrmManifestCheck(&manifest,blob,bytes,&cubin,"manifest") == SPARK_STATUS_OK,"the manifest describes the fixture cubin");
	for (index = 0u; index < cubin.kernel_count; index++)
	{
		const SparkNvrmManifestKernel *reported = SparkNvrmManifestKernelNamed(&manifest,cubin.kernels[index].name);
		agree += reported != 0 && SparkNvrmManifestParameterEnd(reported) == cubin.kernels[index].parameter_bytes &&
			SparkNvrmManifestParameterEnd(reported) == ExpectedParameters(cubin.kernels[index].name);
	}
	CHECK(agree == cubin.kernel_count,"every kernel's PTX parameter table ends where its constant bank and its C layout say");
	{
		const SparkNvrmManifestKernel *mix = SparkNvrmManifestKernelNamed(&manifest,"SparkNvrmParameterMix");
		const SparkNvrmManifestKernel *shared = SparkNvrmManifestKernelNamed(&manifest,"SparkNvrmStaticShared");
		CHECK(mix != 0 && mix->parameter_count == 7u && mix->parameters[3].align == 8u && mix->parameters[3].offset == 24u && mix->max_threads == 128u,
			"a mixed parameter list keeps its PTX types, alignment, offsets and launch bound");
		CHECK(shared != 0 && shared->barriers == 1u,"the compiler's barrier count is kept for the launch descriptor");
		{
			const SparkNvrmManifestKernel *call = SparkNvrmManifestKernelNamed(&manifest,"SparkNvrmCallStack");
			const SparkNvrmCubinKernel *placed = SparkNvrmCubinKernelNamed(&cubin,"SparkNvrmCallStack");
			CHECK(call != 0 && placed != 0 && call->stack_bytes == 0u && call->cumulative_reported != 0u &&
				call->cumulative_bytes >= SPARK_NVRM_FIXTURE_DEEP_WORDS * sizeof(uint32_t) && SparkNvrmManifestStackNeed(call) == placed->stack_bytes,
				"a kernel that calls a function with its own frame needs its call chain's stack, which is what the cubin records");
		}
		{
			const SparkNvrmManifestKernel *named = SparkNvrmManifestKernelNamed(&manifest,SPARK_NVRM_FIXTURE_LONG_TEXT);
			const SparkNvrmCubinKernel *placed = SparkNvrmCubinKernelNamed(&cubin,SPARK_NVRM_FIXTURE_LONG_TEXT);
			char expected[SPARK_NVRM_SHORT_NAME_BYTES];
			SparkNvrmShortName(SPARK_NVRM_FIXTURE_LONG_TEXT,expected);
			CHECK(strlen(SPARK_NVRM_FIXTURE_LONG_TEXT) == 152u && named != 0 && placed != 0 && strcmp(named->name,expected) == 0 &&
				strcmp(placed->name,expected) == 0 && strlen(expected) <= SPARK_DEVICE_KERNEL_NAME_LIMIT &&
				strncmp(expected,"SparkNvrmLongTemplate_",strlen("SparkNvrmLongTemplate_")) == 0,
				"a 152-character mangled kernel is stored under its short name, the base identifier and a digest, in the manifest and the cubin alike");
		}
	}
	blob[bytes - 1u] ^= 1u;
	CHECK(SparkNvrmManifestCheck(&manifest,blob,bytes,&cubin,"manifest") == SPARK_STATUS_TARGET_MISMATCH,"a manifest for another cubin is refused");
	blob[bytes - 1u] ^= 1u;
	manifest.kernels[0].parameters[manifest.kernels[0].parameter_count - 1u].bytes += 4u;
	CHECK(SparkNvrmManifestCheck(&manifest,blob,bytes,&cubin,"manifest") == SPARK_STATUS_TARGET_MISMATCH,
		"a parameter table that ends elsewhere than the kernel's constant bank is refused");
	SparkNvrmManifestRelease(&manifest);
	{
		static const char *const broken[] = {
			"cubin 00 12\nkernel K short K registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n",
			"kernel K short K registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nparam K 0 u32 align 4 bytes 4 offset 0\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nkernel K short K registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n"
			"param K 0 u32 align 3 bytes 4 offset 0\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nkernel K short K registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n"
			"param K 1 u32 align 4 bytes 4 offset 0\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nkernel K short K registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n"
			"bound K warpsize 32\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nsomething else\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nkernel K short K registers 1 barriers 0 stack 16 spill_stores 0 spill_loads 0 smem 0"
			" cumulative 8\n",
			"cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nkernel K short K registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0"
			" cumulative 64 extra\n",
		};
		uint32_t refused = 0u;
		for (index = 0u; index < sizeof(broken) / sizeof(broken[0]); index++)
			refused += SparkNvrmManifestParse(broken[index],strlen(broken[index]),"manifest",&manifest) == SPARK_STATUS_PARSE_ERROR &&
				manifest.kernels == 0;
		CHECK(refused == sizeof(broken) / sizeof(broken[0]),
			"a manifest without its cubin, a parameter before its kernel or out of order, a bad alignment, an unknown bound or line, a cumulative stack below the "
			"frame or trailing words are refused");
	}
	{
		static const char head[] = "cubin 333ac825ff31bb2f8e6e9f0e93e7768578f6cfbe05826078b5beb98fbb8b11f0 1\nkernel ";
		static const char tail[] = " registers 1 barriers 0 stack 0 spill_stores 0 spill_loads 0 smem 0\n";
		const size_t name = 4096u;
		char short_name[SPARK_NVRM_SHORT_NAME_BYTES];
		char *line = (char *)malloc(sizeof(head) + name + 8u + SPARK_NVRM_SHORT_NAME_BYTES + sizeof(tail));
		CHECK(line != 0,"allocate a long manifest line");
		if ( line != 0 )
		{
			char *symbol = (char *)malloc(name + 2u);
			CHECK(symbol != 0,"allocate a long symbol");
			if ( symbol != 0 )
			{
				memset(symbol,'K',name + 1u);
				symbol[name + 1u] = '\0';
				SparkNvrmShortName(symbol,short_name);
				snprintf(line,sizeof(head) + name + 8u + SPARK_NVRM_SHORT_NAME_BYTES + sizeof(tail),"%s%s short %s%s",head,symbol,short_name,tail);
				CHECK(SparkNvrmManifestParse(line,strlen(line),"manifest",&manifest) == SPARK_STATUS_PARSE_ERROR && manifest.kernels == 0,
					"a kernel symbol longer than a manifest takes is refused");
				symbol[name] = '\0';
				SparkNvrmShortName(symbol,short_name);
				snprintf(line,sizeof(head) + name + 8u + SPARK_NVRM_SHORT_NAME_BYTES + sizeof(tail),"%s%s short %s%s",head,symbol,short_name,tail);
				CHECK(SparkNvrmManifestParse(line,strlen(line),"manifest",&manifest) == SPARK_STATUS_OK && manifest.kernel_count == 1u &&
					strcmp(manifest.kernels[0].name,short_name) == 0 && strlen(short_name) <= SPARK_DEVICE_KERNEL_NAME_LIMIT,
					"a 4096-character symbol parses and is kept only as its short name");
				SparkNvrmManifestRelease(&manifest);
				snprintf(line,sizeof(head) + name + 8u + SPARK_NVRM_SHORT_NAME_BYTES + sizeof(tail),"%s%s short K_000000000000%s",head,symbol,tail);
				CHECK(SparkNvrmManifestParse(line,strlen(line),"manifest",&manifest) == SPARK_STATUS_PARSE_ERROR && manifest.kernels == 0,
					"a short name other than the runtime's for the symbol is refused");
				free(symbol);
			}
			free(line);
		}
	}
	SparkNvrmCubinRelease(&cubin);
	free(blob);
	free(text);
}

typedef struct Synthetic
{
	uint16_t machine;
	uint32_t relocation_type;
	uint32_t relocation_symbol;
	uint64_t relocation_offset;
	uint32_t register_records;
	uint32_t parameter_offset;
	const char *text_name;
	const char *bank_name;
	uint64_t text_address;
	uint32_t cut_info;
} Synthetic;

typedef struct Image
{
	uint8_t data[16384];
	uint32_t used;
} Image;

static uint32_t Append(Image *image, const void *data, uint32_t bytes, uint32_t align)
{
	uint32_t offset;
	image->used = (image->used + align - 1u) / align * align;
	offset = image->used;
	if ( data != 0 )
		memcpy(image->data + offset,data,bytes);
	image->used += bytes;
	return(offset);
}

static uint32_t AppendName(char *table, uint32_t *used, const char *name)
{
	const uint32_t offset = *used;
	memcpy(table + offset,name,strlen(name) + 1u);
	*used += (uint32_t)strlen(name) + 1u;
	return(offset);
}

typedef struct SectionSpec
{
	const char *name;
	uint32_t type;
	uint64_t flags;
	uint64_t address;
	uint32_t offset;
	uint64_t bytes;
	uint32_t link;
	uint32_t info;
	uint64_t align;
	uint64_t entry;
} SectionSpec;

static uint32_t Record(uint8_t *at, uint32_t attribute, uint32_t symbol, uint32_t value)
{
	at[0] = 4u;
	at[1] = (uint8_t)attribute;
	PutLe(at + 2u,8u,2u);
	PutLe(at + 4u,symbol,4u);
	PutLe(at + 8u,value,4u);
	return(12u);
}

static uint32_t BuildSynthetic(const Synthetic *knobs, Image *image)
{
	SectionSpec sections[11];
	char names[512] = {0},symbol_names[64] = {0};
	uint8_t symbols[4u * 24u],info[64],kernel_info[32],text[64],bank[SPARK_NVRM_BANK0_DRIVER_BYTES + 16u],relocations[3u * 24u],extra[64];
	uint32_t names_used = 1u,symbols_used = 1u,count = 10u,index,info_used = 0u,table;
	const uint32_t name_kernel = AppendName(symbol_names,&symbols_used,"k"),name_bank = AppendName(symbol_names,&symbols_used,".nv.constant0.k");
	const uint32_t name_missing = AppendName(symbol_names,&symbols_used,"missing");
	memset(image,0,sizeof(*image));
	memset(sections,0,sizeof(sections));
	memset(symbols,0,sizeof(symbols));
	memset(text,0xa5,sizeof(text));
	memset(bank,0,sizeof(bank));
	memset(extra,0,sizeof(extra));
	memset(info,0,sizeof(info));
	memset(kernel_info,0,sizeof(kernel_info));
	memset(relocations,0,sizeof(relocations));
	names[0] = '\0';
	symbol_names[0] = '\0';
	PutLe(symbols + 24u,name_kernel,4u);
	symbols[24u + 4u] = 0x12u;
	PutLe(symbols + 24u + 6u,6u,2u);
	PutLe(symbols + 48u,name_bank,4u);
	symbols[48u + 4u] = 0x03u;
	PutLe(symbols + 48u + 6u,7u,2u);
	PutLe(symbols + 72u,name_missing,4u);
	for (index = 0u; index < knobs->register_records; index++)
		info_used += Record(info + info_used,0x2fu,1u,24u);
	info_used += Record(info + info_used,0x12u,1u,32u);
	kernel_info[0] = 3u;
	kernel_info[1] = 0x1bu;
	PutLe(kernel_info + 2u,0xffu,2u);
	kernel_info[4] = 4u;
	kernel_info[5] = 0x0au;
	PutLe(kernel_info + 6u,8u,2u);
	PutLe(kernel_info + 8u,2u,4u);
	PutLe(kernel_info + 12u,knobs->parameter_offset,2u);
	PutLe(kernel_info + 14u,16u,2u);
	PutLe(relocations,knobs->relocation_offset,8u);
	PutLe(relocations + 8u,((uint64_t)knobs->relocation_symbol << 32) | knobs->relocation_type,8u);
	PutLe(relocations + 16u,0x10u,8u);
	PutLe(relocations + 24u,16u,8u);
	PutLe(relocations + 32u,((uint64_t)2u << 32) | 0x38u,8u);
	PutLe(relocations + 48u,32u,8u);
	PutLe(relocations + 56u,((uint64_t)2u << 32) | 0x39u,8u);
	Append(image,0,64u,1u);
	sections[1] = (SectionSpec){".shstrtab",3u,0u,0u,0u,0u,0u,0u,1u,0u};
	sections[2] = (SectionSpec){".strtab",3u,0u,0u,Append(image,symbol_names,symbols_used,1u),symbols_used,0u,0u,1u,0u};
	sections[3] = (SectionSpec){".symtab",2u,0u,0u,Append(image,symbols,sizeof(symbols),8u),sizeof(symbols),2u,1u,8u,24u};
	sections[4] = (SectionSpec){".nv.info",0x70000000u,0u,0u,Append(image,info,info_used,4u),info_used,3u,0u,4u,0u};
	sections[5] = (SectionSpec){".nv.info.k",0x70000000u,0u,0u,Append(image,kernel_info,16u,4u),16u - knobs->cut_info,3u,6u,4u,0u};
	sections[6] = (SectionSpec){knobs->text_name,1u,6u,knobs->text_address,Append(image,text,sizeof(text),128u),sizeof(text),3u,0u,128u,0u};
	sections[7] = (SectionSpec){".nv.constant0.k",1u,2u,0u,Append(image,bank,sizeof(bank),4u),sizeof(bank),0u,6u,4u,0u};
	sections[8] = (SectionSpec){".rela.text.k",4u,0x40u,0u,Append(image,relocations,sizeof(relocations),8u),sizeof(relocations),3u,6u,8u,24u};
	sections[9] = (SectionSpec){".nv.shared.k",8u,3u,0u,image->used,0x400u,0u,6u,16u,0u};
	if ( knobs->bank_name != 0 )
		sections[count++] = (SectionSpec){knobs->bank_name,1u,2u,0u,Append(image,extra,sizeof(extra),4u),sizeof(extra),0u,0u,4u,0u};
	{
		uint32_t offsets[11] = {0};
		for (index = 1u; index < count; index++)
			offsets[index] = AppendName(names,&names_used,sections[index].name);
		sections[1].offset = Append(image,names,names_used,1u);
		sections[1].bytes = names_used;
		table = Append(image,0,count * 64u,8u);
		for (index = 1u; index < count; index++)
		{
			uint8_t *at = image->data + table + index * 64u;
			PutLe(at,offsets[index],4u);
			PutLe(at + 4u,sections[index].type,4u);
			PutLe(at + 8u,sections[index].flags,8u);
			PutLe(at + 16u,sections[index].address,8u);
			PutLe(at + 24u,sections[index].offset,8u);
			PutLe(at + 32u,sections[index].bytes,8u);
			PutLe(at + 40u,sections[index].link,4u);
			PutLe(at + 44u,sections[index].info,4u);
			PutLe(at + 48u,sections[index].align,8u);
			PutLe(at + 56u,sections[index].entry,8u);
		}
	}
	memcpy(image->data,"\177ELF",4u);
	image->data[4] = 2u;
	image->data[5] = 1u;
	image->data[6] = 1u;
	PutLe(image->data + 16u,1u,2u);
	PutLe(image->data + 18u,knobs->machine,2u);
	PutLe(image->data + 40u,table,8u);
	PutLe(image->data + 52u,64u,2u);
	PutLe(image->data + 58u,64u,2u);
	PutLe(image->data + 60u,count,2u);
	PutLe(image->data + 62u,1u,2u);
	return(image->used);
}

static Synthetic SyntheticDefaults(void)
{
	Synthetic knobs;
	memset(&knobs,0,sizeof(knobs));
	knobs.machine = 190u;
	knobs.relocation_type = 2u;
	knobs.relocation_symbol = 2u;
	knobs.register_records = 1u;
	knobs.parameter_offset = SPARK_NVRM_BANK0_DRIVER_BYTES;
	knobs.text_name = ".text.k";
	return(knobs);
}

static SparkStatus ParseSynthetic(const Synthetic *knobs, SparkNvrmCubin *cubin)
{
	static Image image;
	const uint32_t bytes = BuildSynthetic(knobs,&image);
	return(SparkNvrmCubinParse(image.data,bytes,"synthetic",cubin));
}

static void TestRelocations(void)
{
	Synthetic knobs = SyntheticDefaults();
	SparkNvrmCubin cubin;
	const SparkNvrmCubinKernel *kernel;
	const uint64_t base = 0x123400000000ull;
	uint8_t *copy;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_OK,"a minimal kernel image parses");
	kernel = SparkNvrmCubinKernelNamed(&cubin,"k");
	CHECK(kernel != 0 && cubin.kernel_count == 1u,"the minimal image holds its one kernel");
	if ( kernel == 0 )
		return;
	CHECK(kernel->code_offset == 0u && kernel->code_bytes == 64u && kernel->bank[0].image_offset == 128u &&
		kernel->bank[0].bytes == SPARK_NVRM_BANK0_DRIVER_BYTES + 16u && cubin.image_bytes == 128u + SPARK_NVRM_BANK0_DRIVER_BYTES + 16u,
		"loaded sections are laid out in order at 128-byte alignment");
	CHECK(kernel->registers == 24u && kernel->stack_bytes == 32u && kernel->shared_bytes == 0x400u &&
		kernel->parameter_offset == SPARK_NVRM_BANK0_DRIVER_BYTES && kernel->parameter_bytes == 16u,
		"register, stack, shared and parameter records reach the kernel");
	CHECK(cubin.image[0] == 0xa5u && cubin.image[63] == 0xa5u,"the code bytes are copied into the image");
	CHECK(cubin.relocation_count == 3u && cubin.relocations[0].kind == SPARK_NVRM_RELOCATION_ADDRESS64 &&
		cubin.relocations[0].image_offset == 0u && cubin.relocations[0].target == 128u + 0x10u &&
		cubin.relocations[1].kind == SPARK_NVRM_RELOCATION_ADDRESS_LOW32 && cubin.relocations[1].image_offset == 20u &&
		cubin.relocations[2].kind == SPARK_NVRM_RELOCATION_ADDRESS_HIGH32 && cubin.relocations[2].image_offset == 36u &&
		cubin.relocations[2].target == 128u,"relocations name their image field and their target with the addend");
	copy = (uint8_t *)malloc((size_t)cubin.image_bytes);
	if ( copy != 0 )
	{
		memcpy(copy,cubin.image,(size_t)cubin.image_bytes);
		CHECK(SparkNvrmCubinRelocate(&cubin,base,copy,"synthetic") == SPARK_STATUS_OK,"relocation into a copy succeeds");
		CHECK(Le(copy,8u) == base + 0x90u && Le(copy + 20u,4u) == ((base + 128u) & 0xffffffffu) && Le(copy + 36u,4u) == (base + 128u) >> 32 &&
			copy[8] == 0xa5u && copy[19] == 0xa5u && copy[24] == 0xa5u && copy[40] == 0xa5u,
			"relocated fields hold the target's address and nothing else changes");
		free(copy);
	}
	SparkNvrmCubinRelease(&cubin);
}

static void TestRefusals(void)
{
	Synthetic knobs;
	SparkNvrmCubin cubin;
	knobs = SyntheticDefaults(); knobs.relocation_type = 0x77u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_UNSUPPORTED && strstr(SparkDeviceLastError()->text,"0x77") != 0 && cubin.image == 0,
		"an unknown relocation type is refused, named");
	knobs = SyntheticDefaults(); knobs.relocation_symbol = 3u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_UNSUPPORTED,"a relocation against an undefined symbol is refused");
	knobs = SyntheticDefaults(); knobs.relocation_offset = 60u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_UNSUPPORTED,"a relocation running past its section is refused");
	knobs = SyntheticDefaults(); knobs.register_records = 0u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_PARSE_ERROR,"a kernel without a register count is refused");
	knobs = SyntheticDefaults(); knobs.register_records = 2u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_PARSE_ERROR,"a register count recorded twice is refused");
	knobs = SyntheticDefaults(); knobs.cut_info = 2u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_PARSE_ERROR,"an attribute record cut short is refused");
	knobs = SyntheticDefaults(); knobs.parameter_offset = SPARK_NVRM_BANK0_DRIVER_BYTES + 17u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_PARSE_ERROR,"parameters outside constant bank 0 are refused");
	knobs = SyntheticDefaults(); knobs.text_name = ".text.other";
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_PARSE_ERROR,"a kernel without its code section is refused");
	knobs = SyntheticDefaults(); knobs.text_address = 0x1000u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_UNSUPPORTED,"a loaded section at a fixed address is refused");
	knobs = SyntheticDefaults(); knobs.bank_name = ".nv.constant9";
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_UNSUPPORTED,"a constant bank beyond the descriptor's eight is refused");
	knobs = SyntheticDefaults(); knobs.bank_name = ".nv.constant3";
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_OK && cubin.kernels[0].bank[3].present != 0u && cubin.kernels[0].bank[3].bytes == 64u &&
		cubin.kernels[0].bank[3].image_offset % SPARK_NVRM_CUBIN_SECTION_ALIGN == 0u,"a shared constant bank reaches every kernel");
	SparkNvrmCubinRelease(&cubin);
	knobs = SyntheticDefaults(); knobs.machine = 183u;
	CHECK(ParseSynthetic(&knobs,&cubin) == SPARK_STATUS_TARGET_MISMATCH,"an ELF image for another machine is refused");
	CHECK(SparkNvrmCubinParse(0,0u,"synthetic",&cubin) == SPARK_STATUS_PARSE_ERROR,"an empty image is refused");
	CHECK(SparkNvrmCubinParse((const uint8_t *)"\177ELF",4u,"synthetic",0) == SPARK_STATUS_INVALID_ARGUMENT,"a parse without an output is refused");
}

int main(void)
{
	TestFixture();
	TestManifest();
	TestRelocations();
	TestRefusals();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the nvrm cubin parser matches the compiler's report on release-built kernels and refuses malformed images\n");
	return(0);
}
