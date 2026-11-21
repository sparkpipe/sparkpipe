#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_cubin.h"

#define ELF_HEADER_BYTES 64u
#define ELF_SECTION_BYTES 64u
#define ELF_SYMBOL_BYTES 24u
#define ELF_REL_BYTES 16u
#define ELF_RELA_BYTES 24u
#define ELF_CLASS_64 2u
#define ELF_DATA_LITTLE 1u
#define ELF_MACHINE_CUDA 190u
#define ELF_SECTION_PROGBITS 1u
#define ELF_SECTION_SYMTAB 2u
#define ELF_SECTION_RELA 4u
#define ELF_SECTION_NOBITS 8u
#define CUBIN_SHARED_PREFIX ".nv.shared."
#define ELF_SECTION_REL 9u
#define ELF_FLAG_ALLOC 2u
#define INFO_FORMAT_VALUE 4u
#define INFO_PARAMETER_BANK 0x0au
#define INFO_MINIMUM_STACK 0x12u
#define INFO_REGISTER_COUNT 0x2fu
#define RELOCATION_ADDRESS64 2u
#define RELOCATION_LOW32 0x38u
#define RELOCATION_HIGH32 0x39u
#define RELOCATION_FIELD_OFFSET 4u

typedef struct CubinSection
{
	uint32_t type;
	uint64_t flags;
	uint64_t address;
	uint64_t offset;
	uint64_t bytes;
	uint32_t link;
	uint32_t info;
	uint64_t align;
	uint64_t entry_bytes;
	const char *label;
	uint64_t placed;
	uint32_t loaded;
} CubinSection;

typedef struct CubinReader
{
	const uint8_t *blob;
	uint64_t bytes;
	const char *owner;
	CubinSection *sections;
	uint32_t count;
	const CubinSection *symbols;
	const CubinSection *symbol_names;
	uint64_t symbol_count;
} CubinReader;

typedef struct CubinRecord
{
	uint32_t format;
	uint32_t attribute;
	uint32_t bytes;
	const uint8_t *data;
} CubinRecord;

#define CUBIN_REFUSE(code, reader, text) \
	do { \
		const SparkStatus cubin_status = SparkDeviceRecordError((code),"library_load",(reader)->owner,0,(text)); \
		SPARK_FAIL(cubin_status); \
	} while (0)

static uint64_t CubinRead(const uint8_t *at, uint32_t width)
{
	uint64_t value = 0u;
	uint32_t index;
	for (index = width; index > 0u; index--)
		value = (value << 8) | at[index - 1u];
	return(value);
}

static void CubinWrite(uint8_t *at, uint64_t value, uint32_t width)
{
	uint32_t index;
	for (index = 0u; index < width; index++)
		at[index] = (uint8_t)(value >> (8u * index));
}

static int CubinSpan(uint64_t offset, uint64_t bytes, uint64_t limit)
{
	return(offset <= limit && bytes <= limit - offset);
}

static const char *CubinString(const CubinReader *reader, const CubinSection *table, uint64_t offset)
{
	const char *start;
	if ( table == 0 || offset >= table->bytes )
		return(0);
	start = (const char *)reader->blob + table->offset + offset;
	return(memchr(start,'\0',(size_t)(table->bytes - offset)) != 0 ? start : 0);
}

static SparkStatus CubinHeader(CubinReader *reader, uint64_t *section_offset, uint32_t *names_index)
{
	const uint8_t *blob = reader->blob;
	if ( reader->bytes < ELF_HEADER_BYTES || blob[0] != 0x7fu || blob[1] != 'E' || blob[2] != 'L' || blob[3] != 'F' )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"not an ELF image");
	if ( blob[4] != ELF_CLASS_64 || blob[5] != ELF_DATA_LITTLE )
		CUBIN_REFUSE(SPARK_STATUS_TARGET_MISMATCH,reader,"not a 64-bit little-endian ELF image");
	if ( CubinRead(blob + 18u,2u) != ELF_MACHINE_CUDA )
		CUBIN_REFUSE(SPARK_STATUS_TARGET_MISMATCH,reader,"the ELF image is not a GPU kernel image");
	*section_offset = CubinRead(blob + 40u,8u);
	reader->count = (uint32_t)CubinRead(blob + 60u,2u);
	*names_index = (uint32_t)CubinRead(blob + 62u,2u);
	if ( CubinRead(blob + 58u,2u) != ELF_SECTION_BYTES || reader->count == 0u ||
		!CubinSpan(*section_offset,(uint64_t)reader->count * ELF_SECTION_BYTES,reader->bytes) || *names_index >= reader->count )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"the section header table runs past the image");
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinSections(CubinReader *reader)
{
	uint64_t table;
	uint32_t names,index;
	SparkStatus status = CubinHeader(reader,&table,&names);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	reader->sections = (CubinSection *)calloc(reader->count,sizeof(CubinSection));
	if ( reader->sections == 0 )
		CUBIN_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,reader,"out of host memory for the section table");
	for (index = 0u; index < reader->count; index++)
	{
		const uint8_t *at = reader->blob + table + (uint64_t)index * ELF_SECTION_BYTES;
		CubinSection *section = &reader->sections[index];
		section->type = (uint32_t)CubinRead(at + 4u,4u);
		section->flags = CubinRead(at + 8u,8u);
		section->address = CubinRead(at + 16u,8u);
		section->offset = CubinRead(at + 24u,8u);
		section->bytes = CubinRead(at + 32u,8u);
		section->link = (uint32_t)CubinRead(at + 40u,4u);
		section->info = (uint32_t)CubinRead(at + 44u,4u);
		section->align = CubinRead(at + 48u,8u);
		section->entry_bytes = CubinRead(at + 56u,8u);
		if ( section->type != ELF_SECTION_NOBITS && !CubinSpan(section->offset,section->bytes,reader->bytes) )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a section runs past the image");
	}
	for (index = 0u; index < reader->count; index++)
	{
		const uint8_t *at = reader->blob + table + (uint64_t)index * ELF_SECTION_BYTES;
		reader->sections[index].label = CubinString(reader,&reader->sections[names],CubinRead(at,4u));
		if ( reader->sections[index].label == 0 )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a section name is not a terminated string");
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinSymbols(CubinReader *reader)
{
	uint32_t index;
	for (index = 0u; index < reader->count; index++)
	{
		const CubinSection *section = &reader->sections[index];
		if ( section->type != ELF_SECTION_SYMTAB )
			continue;
		if ( reader->symbols != 0 )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"the image has two symbol tables");
		if ( section->entry_bytes != ELF_SYMBOL_BYTES || section->bytes % ELF_SYMBOL_BYTES != 0u || section->link >= reader->count )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"the symbol table has the wrong entry size or no string table");
		reader->symbols = section;
		reader->symbol_names = &reader->sections[section->link];
		reader->symbol_count = section->bytes / ELF_SYMBOL_BYTES;
	}
	if ( reader->symbols == 0 )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"the image has no symbol table");
	return(SPARK_STATUS_OK);
}

static uint32_t CubinPlaced(const CubinSection *section)
{
	if ( (section->flags & ELF_FLAG_ALLOC) == 0u )
		return(0u);
	if ( section->type == ELF_SECTION_PROGBITS )
		return(1u);
	return(section->type == ELF_SECTION_NOBITS && strncmp(section->label,CUBIN_SHARED_PREFIX,strlen(CUBIN_SHARED_PREFIX)) != 0 ? 1u : 0u);
}

static SparkStatus CubinLayout(CubinReader *reader, SparkNvrmCubin *cubin)
{
	uint64_t cursor = 0u;
	uint32_t index;
	for (index = 0u; index < reader->count; index++)
	{
		CubinSection *section = &reader->sections[index];
		uint64_t align = section->align > SPARK_NVRM_CUBIN_SECTION_ALIGN ? section->align : SPARK_NVRM_CUBIN_SECTION_ALIGN;
		if ( CubinPlaced(section) == 0u )
			continue;
		if ( section->address != 0u )
			CUBIN_REFUSE(SPARK_STATUS_UNSUPPORTED,reader,"a loaded section asks for a fixed address");
		if ( (align & (align - 1u)) != 0u )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a loaded section's alignment is not a power of two");
		section->placed = (cursor + align - 1u) & ~(align - 1u);
		section->loaded = 1u;
		cursor = section->placed + section->bytes;
	}
	cubin->image_bytes = cursor;
	cubin->image = (uint8_t *)calloc(1u,(size_t)(cursor != 0u ? cursor : 1u));
	if ( cubin->image == 0 )
		CUBIN_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,reader,"out of host memory for the kernel image");
	for (index = 0u; index < reader->count; index++)
		if ( reader->sections[index].loaded != 0u && reader->sections[index].type == ELF_SECTION_PROGBITS )
			memcpy(cubin->image + reader->sections[index].placed,reader->blob + reader->sections[index].offset,(size_t)reader->sections[index].bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinNextRecord(const CubinReader *reader, const CubinSection *section, uint64_t *cursor, CubinRecord *record)
{
	const uint8_t *at = reader->blob + section->offset + *cursor;
	if ( section->bytes - *cursor < 4u )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"an attribute record is cut short");
	record->format = at[0];
	record->attribute = at[1];
	record->bytes = (uint32_t)CubinRead(at + 2u,2u);
	record->data = at + 4u;
	if ( record->format != INFO_FORMAT_VALUE )
	{
		*cursor += 4u;
		return(SPARK_STATUS_OK);
	}
	if ( section->bytes - *cursor - 4u < record->bytes )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"an attribute value runs past its section");
	*cursor += 4u + record->bytes;
	return(SPARK_STATUS_OK);
}

static const CubinSection *CubinFind(const CubinReader *reader, const char *prefix, const char *name)
{
	const size_t length = strlen(prefix);
	uint32_t index;
	for (index = 0u; index < reader->count; index++)
	{
		const char *label = reader->sections[index].label;
		if ( strncmp(label,prefix,length) == 0 && strcmp(label + length,name) == 0 )
			return(&reader->sections[index]);
	}
	return(0);
}

static SparkStatus CubinParameterBank(const CubinReader *reader, const CubinSection *info, uint32_t *offset, uint32_t *found)
{
	uint64_t cursor = 0u;
	CubinRecord record = {0u, 0u, 0u, 0};
	*found = 0u;
	while ( cursor < info->bytes )
	{
		const SparkStatus status = CubinNextRecord(reader,info,&cursor,&record);
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
		if ( record.format != INFO_FORMAT_VALUE || record.attribute != INFO_PARAMETER_BANK )
			continue;
		if ( record.bytes < 6u || *found != 0u )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a kernel's parameter bank record is short or repeated");
		*offset = (uint32_t)CubinRead(record.data + 4u,2u);
		*found = 1u;
	}
	return(SPARK_STATUS_OK);
}

static uint32_t CubinBankIndex(const char *label, const char **suffix)
{
	const char *digits = label + strlen(".nv.constant");
	uint32_t value = 0u;
	if ( strncmp(label,".nv.constant",strlen(".nv.constant")) != 0 || *digits < '0' || *digits > '9' )
		return(UINT32_MAX);
	while ( *digits >= '0' && *digits <= '9' && value < 1000u )
		value = value * 10u + (uint32_t)(*digits++ - '0');
	if ( *digits != '\0' && *digits != '.' )
		return(UINT32_MAX);
	*suffix = *digits == '.' ? digits + 1 : 0;
	return(value);
}

static SparkStatus CubinBanks(const CubinReader *reader, SparkNvrmCubinKernel *kernel)
{
	uint32_t index;
	for (index = 0u; index < reader->count; index++)
	{
		const CubinSection *section = &reader->sections[index];
		const char *suffix = 0;
		const uint32_t bank = CubinBankIndex(section->label,&suffix);
		if ( bank == UINT32_MAX || (suffix != 0 && strcmp(suffix,kernel->name) != 0) )
			continue;
		if ( bank >= SPARK_NVRM_CUBIN_BANKS )
			CUBIN_REFUSE(SPARK_STATUS_UNSUPPORTED,reader,"a constant bank index is beyond the launch descriptor's banks");
		if ( section->loaded == 0u )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a constant bank is not a loaded section");
		if ( kernel->bank[bank].present != 0u && suffix == 0 )
			continue;
		kernel->bank[bank].image_offset = section->placed;
		kernel->bank[bank].bytes = section->bytes;
		kernel->bank[bank].present = 1u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinKernel(const CubinReader *reader, const char *name, uint32_t parameter_offset, SparkNvrmCubinKernel *kernel)
{
	const CubinSection *text = CubinFind(reader,".text.",name);
	const CubinSection *shared = CubinFind(reader,".nv.shared.",name);
	SparkStatus status;
	kernel->name = name;
	if ( text == 0 || text->loaded == 0u )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a kernel has no loaded code section");
	if ( shared != 0 && shared->type != ELF_SECTION_NOBITS )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a kernel's shared memory section carries data");
	kernel->code_offset = text->placed;
	kernel->code_bytes = text->bytes;
	kernel->shared_bytes = shared != 0 ? (uint32_t)shared->bytes : 0u;
	kernel->registers = UINT32_MAX;
	kernel->stack_bytes = UINT32_MAX;
	status = CubinBanks(reader,kernel);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( kernel->bank[0].present == 0u || parameter_offset > kernel->bank[0].bytes )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a kernel's parameters lie outside its constant bank 0");
	kernel->parameter_offset = parameter_offset;
	kernel->parameter_bytes = (uint32_t)(kernel->bank[0].bytes - parameter_offset);
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinKernels(const CubinReader *reader, SparkNvrmCubin *cubin)
{
	const size_t prefix = strlen(".nv.info.");
	uint32_t index,found,offset = 0u;
	cubin->kernels = (SparkNvrmCubinKernel *)calloc(reader->count,sizeof(SparkNvrmCubinKernel));
	if ( cubin->kernels == 0 )
		CUBIN_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,reader,"out of host memory for the kernel table");
	for (index = 0u; index < reader->count; index++)
	{
		const CubinSection *section = &reader->sections[index];
		SparkStatus status;
		if ( strncmp(section->label,".nv.info.",prefix) != 0 || section->label[prefix] == '\0' )
			continue;
		status = CubinParameterBank(reader,section,&offset,&found);
		if ( status == SPARK_STATUS_OK && found != 0u )
			status = CubinKernel(reader,section->label + prefix,offset,&cubin->kernels[cubin->kernel_count++]);
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
	}
	if ( cubin->kernel_count == 0u )
		CUBIN_REFUSE(SPARK_STATUS_NOT_FOUND,reader,"the image holds no kernel");
	return(SPARK_STATUS_OK);
}

static SparkNvrmCubinKernel *CubinKernelForSymbol(const CubinReader *reader, SparkNvrmCubin *cubin, uint64_t symbol)
{
	const char *name;
	uint32_t index;
	if ( symbol >= reader->symbol_count )
		return(0);
	name = CubinString(reader,reader->symbol_names,CubinRead(reader->blob + reader->symbols->offset + symbol * ELF_SYMBOL_BYTES,4u));
	for (index = 0u; name != 0 && index < cubin->kernel_count; index++)
		if ( strcmp(cubin->kernels[index].name,name) == 0 )
			return(&cubin->kernels[index]);
	return(0);
}

static SparkStatus CubinFunctionRecords(const CubinReader *reader, SparkNvrmCubin *cubin)
{
	const CubinSection *info = CubinFind(reader,".nv.info","");
	uint64_t cursor = 0u;
	CubinRecord record = {0u, 0u, 0u, 0};
	uint32_t index;
	if ( info == 0 )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"the image has no function attribute section");
	while ( cursor < info->bytes )
	{
		SparkNvrmCubinKernel *kernel;
		uint32_t *field;
		const SparkStatus status = CubinNextRecord(reader,info,&cursor,&record);
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
		if ( record.format != INFO_FORMAT_VALUE || (record.attribute != INFO_REGISTER_COUNT && record.attribute != INFO_MINIMUM_STACK) )
			continue;
		if ( record.bytes != 8u )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a function attribute record is not a symbol and a value");
		kernel = CubinKernelForSymbol(reader,cubin,CubinRead(record.data,4u));
		if ( kernel == 0 )
			continue;
		field = record.attribute == INFO_REGISTER_COUNT ? &kernel->registers : &kernel->stack_bytes;
		if ( *field != UINT32_MAX )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a kernel's register count or stack size is recorded twice");
		*field = (uint32_t)CubinRead(record.data + 4u,4u);
	}
	for (index = 0u; index < cubin->kernel_count; index++)
		if ( cubin->kernels[index].registers == UINT32_MAX || cubin->kernels[index].stack_bytes == UINT32_MAX )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a kernel has no register count or no stack size record");
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinRelocation(const CubinReader *reader, const CubinSection *table, const CubinSection *target, uint64_t entry,
	SparkNvrmCubinRelocation *relocation)
{
	const uint8_t *at = reader->blob + table->offset + entry * table->entry_bytes;
	const uint64_t offset = CubinRead(at,8u),information = CubinRead(at + 8u,8u),symbol = information >> 32;
	const uint32_t type = (uint32_t)(information & 0xffffffffu);
	const int64_t addend = table->type == ELF_SECTION_RELA ? (int64_t)CubinRead(at + 16u,8u) : 0;
	const uint8_t *symbol_at;
	uint32_t defined;
	if ( symbol >= reader->symbol_count )
		CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a relocation names a symbol beyond the symbol table");
	symbol_at = reader->blob + reader->symbols->offset + symbol * ELF_SYMBOL_BYTES;
	defined = (uint32_t)CubinRead(symbol_at + 6u,2u);
	if ( defined == 0u || defined >= reader->count || reader->sections[defined].loaded == 0u )
		CUBIN_REFUSE(SPARK_STATUS_UNSUPPORTED,reader,"a relocation names a symbol outside the loaded sections");
	relocation->target = (uint64_t)((int64_t)(reader->sections[defined].placed + CubinRead(symbol_at + 8u,8u)) + addend);
	if ( type == RELOCATION_ADDRESS64 && offset <= target->bytes && target->bytes - offset >= 8u )
	{
		relocation->kind = SPARK_NVRM_RELOCATION_ADDRESS64;
		relocation->image_offset = target->placed + offset;
	}
	else if ( (type == RELOCATION_LOW32 || type == RELOCATION_HIGH32) && offset <= target->bytes &&
		target->bytes - offset >= RELOCATION_FIELD_OFFSET + 4u )
	{
		relocation->kind = type == RELOCATION_LOW32 ? SPARK_NVRM_RELOCATION_ADDRESS_LOW32 : SPARK_NVRM_RELOCATION_ADDRESS_HIGH32;
		relocation->image_offset = target->placed + offset + RELOCATION_FIELD_OFFSET;
	}
	else
	{
		char text[96];
		snprintf(text,sizeof(text),"relocation type %#x at offset %#llx is unknown or out of its section",type,(unsigned long long)offset);
		CUBIN_REFUSE(SPARK_STATUS_UNSUPPORTED,reader,text);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinRelocations(const CubinReader *reader, SparkNvrmCubin *cubin)
{
	uint64_t total = 0u,entry;
	uint32_t index;
	for (index = 0u; index < reader->count; index++)
	{
		const CubinSection *table = &reader->sections[index];
		const uint64_t width = table->type == ELF_SECTION_RELA ? ELF_RELA_BYTES : ELF_REL_BYTES;
		if ( table->type != ELF_SECTION_REL && table->type != ELF_SECTION_RELA )
			continue;
		if ( table->entry_bytes != width || table->bytes % width != 0u || table->info >= reader->count )
			CUBIN_REFUSE(SPARK_STATUS_PARSE_ERROR,reader,"a relocation table has the wrong entry size or no target");
		if ( reader->sections[table->info].loaded != 0u )
			total += table->bytes / width;
	}
	cubin->relocations = (SparkNvrmCubinRelocation *)calloc((size_t)(total != 0u ? total : 1u),sizeof(SparkNvrmCubinRelocation));
	if ( cubin->relocations == 0 )
		CUBIN_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,reader,"out of host memory for the relocation table");
	for (index = 0u; index < reader->count; index++)
	{
		const CubinSection *table = &reader->sections[index];
		if ( (table->type != ELF_SECTION_REL && table->type != ELF_SECTION_RELA) || reader->sections[table->info].loaded == 0u )
			continue;
		for (entry = 0u; entry < table->bytes / table->entry_bytes; entry++)
		{
			const SparkStatus status = CubinRelocation(reader,table,&reader->sections[table->info],entry,&cubin->relocations[cubin->relocation_count]);
			if ( status != SPARK_STATUS_OK )
				SPARK_FAIL(status);
			cubin->relocation_count++;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus CubinOwnNames(const CubinReader *reader, SparkNvrmCubin *cubin)
{
	size_t total = 0u,used = 0u;
	uint32_t index;
	for (index = 0u; index < cubin->kernel_count; index++)
		total += strlen(cubin->kernels[index].name) + 1u;
	cubin->names = (char *)malloc(total);
	if ( cubin->names == 0 )
		CUBIN_REFUSE(SPARK_STATUS_CAPACITY_EXCEEDED,reader,"out of host memory for the kernel names");
	for (index = 0u; index < cubin->kernel_count; index++)
	{
		const size_t length = strlen(cubin->kernels[index].name) + 1u;
		memcpy(cubin->names + used,cubin->kernels[index].name,length);
		cubin->kernels[index].name = cubin->names + used;
		used += length;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvrmCubinParse(const uint8_t *blob, uint64_t bytes, const char *owner, SparkNvrmCubin *cubin)
{
	CubinReader reader;
	SparkStatus status;
	memset(&reader,0,sizeof(reader));
	reader.blob = blob;
	reader.bytes = bytes;
	reader.owner = owner;
	if ( cubin == 0 || (blob == 0 && bytes != 0u) )
		CUBIN_REFUSE(SPARK_STATUS_INVALID_ARGUMENT,&reader,"needs an image and an output");
	memset(cubin,0,sizeof(*cubin));
	status = CubinSections(&reader);
	if ( status == SPARK_STATUS_OK )
		status = CubinSymbols(&reader);
	if ( status == SPARK_STATUS_OK )
		status = CubinLayout(&reader,cubin);
	if ( status == SPARK_STATUS_OK )
		status = CubinKernels(&reader,cubin);
	if ( status == SPARK_STATUS_OK )
		status = CubinFunctionRecords(&reader,cubin);
	if ( status == SPARK_STATUS_OK )
		status = CubinRelocations(&reader,cubin);
	if ( status == SPARK_STATUS_OK )
		status = CubinOwnNames(&reader,cubin);
	free(reader.sections);
	if ( status != SPARK_STATUS_OK )
	{
		SparkNvrmCubinRelease(cubin);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

const SparkNvrmCubinKernel *SparkNvrmCubinKernelNamed(const SparkNvrmCubin *cubin, const char *name)
{
	uint32_t index;
	for (index = 0u; cubin != 0 && name != 0 && index < cubin->kernel_count; index++)
		if ( strcmp(cubin->kernels[index].name,name) == 0 )
			return(&cubin->kernels[index]);
	return(0);
}

SparkStatus SparkNvrmCubinRelocate(const SparkNvrmCubin *cubin, uint64_t base, uint8_t *image, const char *owner)
{
	uint32_t index;
	if ( cubin == 0 || image == 0 )
	{
		const SparkStatus status = SparkDeviceRecordError(SPARK_STATUS_INVALID_ARGUMENT,"library_load",owner,0,"needs a parsed image and a destination");
		SPARK_FAIL(status);
	}
	for (index = 0u; index < cubin->relocation_count; index++)
	{
		const SparkNvrmCubinRelocation *relocation = &cubin->relocations[index];
		const uint64_t address = base + relocation->target;
		if ( relocation->kind == SPARK_NVRM_RELOCATION_ADDRESS64 )
			CubinWrite(image + relocation->image_offset,address,8u);
		else
			CubinWrite(image + relocation->image_offset,relocation->kind == SPARK_NVRM_RELOCATION_ADDRESS_LOW32 ? address & 0xffffffffu : address >> 32,4u);
	}
	return(SPARK_STATUS_OK);
}

void SparkNvrmCubinRelease(SparkNvrmCubin *cubin)
{
	if ( cubin == 0 )
		return;
	free(cubin->image);
	free(cubin->kernels);
	free(cubin->relocations);
	free(cubin->names);
	memset(cubin,0,sizeof(*cubin));
}
