#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_directory="${BUILD_DIRECTORY:-${repository_root}/build/minimax_cpu_validate}"
mkdir -p "${build_directory}"

cc -std=c11 -Wall -Wextra -Werror -O3 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -fPIC \
	-I"${repository_root}" \
	-I"${repository_root}/include" \
	-I"${repository_root}/src" \
	-I"${repository_root}/tests/cuda_stub" \
	-I"${repository_root}/model-families/common/include" \
	-I"${repository_root}/model-families/minimax/include" \
	-I"${repository_root}/model-families/minimax/include/sparkpipe" \
	-I"${repository_root}/modules/minimax_resident_decode_stage/include" \
	-I"${repository_root}/modules/minimax_resident_decode_stage/include/sparkpipe" \
	-I"${repository_root}/modules/minimax_resident_decode_stage/source" \
	-DSPARK_MINIMAX_MODULE_BUILD=1 \
	-include "${repository_root}/modules/minimax_resident_decode_stage/include/sparkpipe/spark_minimax_model.h" \
	-c "${repository_root}/modules/minimax_resident_decode_stage/source/spark_minimax_resident_decode_stage_module.c" \
	-o "${build_directory}/module.o"

cc -std=c11 -Wall -Wextra -Werror -O3 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -fPIC \
	-I"${repository_root}" -I"${repository_root}/include" -I"${repository_root}/tests/cuda_stub" \
	-I"${repository_root}/modules/minimax_resident_decode_stage/include" \
	-I"${repository_root}/modules/minimax_resident_decode_stage/source" \
	-c "${repository_root}/modules/minimax_resident_decode_stage/validation/minimax_cpu_validate.c" \
	-o "${build_directory}/validate.o"

for source in \
	runtime/stage_module_lifecycle.c \
	runtime/stage_module_common.c \
	runtime/stagepack_format.c \
	ring/transport/tp_device_collective.c \
	src/spark_admission.c; do
	object="${build_directory}/$(basename "${source%.c}").o"
	cc -std=c11 -Wall -Wextra -Werror -O3 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -fPIC \
		-I"${repository_root}" -I"${repository_root}/include" -I"${repository_root}/tests/cuda_stub" \
		-I"${repository_root}/model-families/common/include" \
		-c "${repository_root}/${source}" -o "${object}"
done


cc -std=c11 -O2 -D_GNU_SOURCE -fPIC \
	-I"${repository_root}/tests/cuda_stub" -I"${repository_root}/include" -I"${repository_root}" -I"${repository_root}/model-families/common/include" \
	-c "${repository_root}/tests/cuda_stub/cuda_runtime_stub.c" \
	-o "${build_directory}/cuda_runtime_stub.o"

objects="${build_directory}/validate.o ${build_directory}/module.o"
for source in \
	stage_module_lifecycle stage_module_common stagepack_format \
	tp_device_collective spark_admission cuda_runtime_stub; do
	objects="${objects} ${build_directory}/${source}.o"
done

# The validation binary exercises the pack loader and the host reference math
# only; serving-only symbols pulled in through module.o (weightd, mesh, hidden
# transport) are linked as never-called placeholders so the tool stays CPU-only.
stubs="${build_directory}/link_stubs.c"
: > "${stubs}"
for attempt in 1 2 3 4 5 6; do
	if # shellcheck disable=SC2086
	cc ${objects} -lm -o "${build_directory}/minimax_cpu_validate" 2>"${build_directory}/link.log"; then
		break
	fi
	symbols="$(grep -o "undefined reference to \`[A-Za-z0-9_]*'" "${build_directory}/link.log" | sed "s/undefined reference to \`//;s/'//" | sort -u)"
	if [ -z "${symbols}" ]; then
		cat "${build_directory}/link.log" >&2
		exit 1
	fi
	for symbol in ${symbols}; do
		grep -q "${symbol}" "${stubs}" || printf 'void %s(void) {}\n' "${symbol}" >> "${stubs}"
	done
	cc -std=c11 -O2 -fPIC -c "${stubs}" -o "${build_directory}/link_stubs.o"
	objects="${objects} ${build_directory}/link_stubs.o"
done
echo "minimax_cpu_validate built: ${build_directory}/minimax_cpu_validate"


