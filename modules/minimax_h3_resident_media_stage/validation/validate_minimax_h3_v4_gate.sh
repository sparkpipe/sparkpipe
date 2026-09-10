#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
work="${MINIMAX_H3_V4_WORK:-/mnt/model-warm/staging/minimax-lane/v4work}"
warm="${MINIMAX_H3_V4_WARM:-/mnt/model-warm/minimax-h3}"
video_fixtures="${module_directory}/validation/fixtures/real/v4_video"
audio_fixtures="${module_directory}/validation/fixtures/real/v4_audio"
cc_bin="${MINIMAX_H3_V4_CC:-cc}"
omp_flag="${MINIMAX_H3_V4_OMP:--fopenmp}"

mkdir -p "${work}/video_weights" "${work}/audio_weights"

video_names="post_quant_conv.weight,post_quant_conv.bias,decoder.proj_in.weight,decoder.proj_in.bias,decoder.register_tokens,decoder.norm_out.weight,decoder.norm_out.bias,decoder.proj_out.weight,decoder.proj_out.bias"
audio_names="dec_in_proj.weight,dec_in_proj.bias,decoder.conv_pre.weight_g,decoder.conv_pre.weight_v,decoder.conv_pre.bias,decoder.conv_post.weight_g,decoder.conv_post.weight_v,decoder.activation_post.act.alpha,decoder.activation_post.act.beta,decoder.activation_post.upsample.filter,decoder.activation_post.downsample.lowpass.filter"
for index in $(seq 0 35); do
	for tensor in "attn.to_q.weight" "attn.to_q.bias" "attn.to_k.weight" \
		"attn.to_k.bias" "attn.to_v.weight" "attn.to_v.bias" "attn.to_out.0.weight" \
		"attn.to_out.0.bias" "ff.net.0.proj.weight" "ff.net.0.proj.bias" \
		"ff.net.2.weight" "ff.net.2.bias" "norm1.weight" "norm2.weight" "scale1" \
		"scale2"; do
		video_names="${video_names},decoder.transformer_blocks.${index}.${tensor}"
	done
done
for resblock in $(seq 0 20); do
	for conv in 0 1 2; do
		for part in weight_g weight_v bias; do
			audio_names="${audio_names},decoder.resblocks.${resblock}.convs1.${conv}.${part}"
			audio_names="${audio_names},decoder.resblocks.${resblock}.convs2.${conv}.${part}"
		done
	done
	for activation in 0 1 2 3 4 5; do
		for part in act.alpha act.beta upsample.filter downsample.lowpass.filter; do
			audio_names="${audio_names},decoder.resblocks.${resblock}.activations.${activation}.${part}"
		done
	done
done
for upsample in 0 1 2 3 4 5 6; do
	for part in weight_g weight_v bias; do
		audio_names="${audio_names},decoder.ups.${upsample}.0.${part}"
	done
done

if [ ! -f "${work}/video_weights/manifest.txt" ]; then
python3 "${repository_root}/tools/minimax_h3_extract_tensors.py" \
	--component-dir "${warm}/vae" --names "${video_names}" \
	--out "${work}/video_weights"
fi
if [ ! -f "${work}/audio_weights/manifest.txt" ]; then
python3 "${repository_root}/tools/minimax_h3_extract_tensors.py" \
	--component-dir "${warm}/audio_vae" --no-index --names "${audio_names}" \
	--out "${work}/audio_weights"
fi

"${cc_bin}" -O2 -march=native -Wall -Wextra ${omp_flag} \
	-o "${work}/minimax_h3_v4_gate" \
	"${script_directory}/spark_minimax_h3_v4_gate.c" -lm

video_status=0
"${work}/minimax_h3_v4_gate" video "${video_fixtures}" "${work}/video_weights" || video_status=$?
audio_status=0
"${work}/minimax_h3_v4_gate" audio "${audio_fixtures}" "${work}/audio_weights" || audio_status=$?
echo "V4_VIDEO_STATUS=$video_status V4_AUDIO_STATUS=$audio_status"
[ "$video_status" -eq 0 ] && [ "$audio_status" -eq 0 ]
