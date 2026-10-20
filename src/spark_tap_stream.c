#include "sparkpipe/spark_tap_stream.h"

#include <stddef.h>

_Static_assert(sizeof(SparkTapStreamHello) == 624u, "the hello header is 624 bytes on every host");
_Static_assert(sizeof(SparkTapStreamFrame) == 96u, "the frame header is 96 bytes on every host");

static uint64_t SparkTapStreamMix(uint64_t hash, const void *bytes, size_t count)
{
	const uint8_t *cursor = (const uint8_t *)bytes;
	for ( size_t index = 0u; index < count; ++index )
	{
		hash ^= cursor[index];
		hash *= UINT64_C(0x100000001b3);
	}
	return hash;
}

uint64_t SparkTapStreamFingerprint(const SparkTapStreamHello *hello)
{
	uint64_t hash = UINT64_C(0xcbf29ce484222325);
	if ( hello == 0 )
		return 0u;
	hash = SparkTapStreamMix(hash, hello->model_tag, sizeof(hello->model_tag));
	hash = SparkTapStreamMix(hash, hello->pack_identity, sizeof(hello->pack_identity));
	hash = SparkTapStreamMix(hash, &hello->hidden, sizeof(hello->hidden));
	hash = SparkTapStreamMix(hash, &hello->total_layers, sizeof(hello->total_layers));
	hash = SparkTapStreamMix(hash, &hello->dtype, sizeof(hello->dtype));
	hash = SparkTapStreamMix(hash, &hello->entry_count, sizeof(hello->entry_count));
	return SparkTapStreamMix(hash, hello->entries, (size_t)hello->entry_count * sizeof(hello->entries[0]));
}
