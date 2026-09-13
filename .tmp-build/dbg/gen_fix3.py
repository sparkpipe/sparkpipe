src = open('modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu').read()

a = src

# fix 1: undefined macro -> literal geometry
src = src.replace('uint32_t slot_owner[SPARK_GLM52_EXPERTS + 1u];',
	'/* 256 = GLM52_EXPERTS; the validator pins the model geometry literally. */\n\tuint32_t slot_owner[256u + 1u];')
src = src.replace('for (owner_expert = SPARK_GLM52_EXPERTS; owner_expert > 0u',
	'for (owner_expert = 256u; owner_expert > 0u')

# fix 2: broken string literal from generation escape
broken = 'printf("glm52_validation check=routed_expert_owner slot=%u owner=%u' + chr(10) + '",'
fixed = 'printf("glm52_validation check=routed_expert_owner slot=%u owner=%u' + chr(92) + 'n",'
assert broken in src, 'broken printf not found'
src = src.replace(broken, fixed)

assert src != a
open('.tmp-build/dbg/new/spark_glm52_resident_decode_stage_cuda_validation.cu','w').write(src)
print('v2 ok', len(src))
