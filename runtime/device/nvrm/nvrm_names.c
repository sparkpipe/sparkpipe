#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_sha256.h"
#include "runtime/device/nvrm/nvrm_names.h"

#define NAMES_MANGLED_PREFIX "_Z"
#define NAMES_MANGLED_PREFIX_BYTES (sizeof(NAMES_MANGLED_PREFIX) - 1u)

static void NamesBase(const char *name, const char **base, size_t *length)
{
	const char *rest;
	*base = name;
	*length = strlen(name);
	if ( strncmp(name,NAMES_MANGLED_PREFIX,NAMES_MANGLED_PREFIX_BYTES) != 0 )
		return;
	rest = name + NAMES_MANGLED_PREFIX_BYTES;
	if ( *rest == 'N' )
		for (rest++; *rest == 'r' || *rest == 'V' || *rest == 'K' || *rest == 'R' || *rest == 'O'; rest++)
			;
	while ( *rest != '\0' )
	{
		char *after = 0;
		unsigned long count;
		if ( *rest == 'L' )
		{
			rest++;
			continue;
		}
		if ( !isdigit((unsigned char)*rest) )
			break;
		count = strtoul(rest,&after,10);
		if ( count == 0u || strlen(after) < count )
			break;
		*base = after;
		*length = count;
		rest = after + count;
	}
}

void SparkNvrmShortName(const char *name, char short_name[SPARK_NVRM_SHORT_NAME_BYTES])
{
	char digest[SPARK_SHA256_HEX_BYTES];
	const char *base;
	size_t length;
	if ( strlen(name) <= SPARK_DEVICE_KERNEL_NAME_LIMIT )
	{
		memcpy(short_name,name,strlen(name) + 1u);
		return;
	}
	NamesBase(name,&base,&length);
	if ( length > SPARK_DEVICE_KERNEL_NAME_LIMIT - 1u - SPARK_NVRM_SHORT_DIGEST_DIGITS )
		length = SPARK_DEVICE_KERNEL_NAME_LIMIT - 1u - SPARK_NVRM_SHORT_DIGEST_DIGITS;
	(void)SparkSha256Bytes((const uint8_t *)name,strlen(name),digest);
	snprintf(short_name,SPARK_NVRM_SHORT_NAME_BYTES,"%.*s_%.*s",(int)length,base,(int)SPARK_NVRM_SHORT_DIGEST_DIGITS,digest);
}
