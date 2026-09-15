import argparse
import json
import os
import struct
import sys

import numpy as np

HIDDEN = 2560
VOCAB = 248320
EXPERTS = 512
EXPERT_DIM = 640
SHARED_DIM = 640
GDN_QK = 2048
GDN_VALUE = 6144
GDN_CONV = 10240
GDN_VALUE_HEADS = 48
GDN_KEY_HEADS = 16
GDN_HKD = 128
GDN_HVD = 128
ATTN_HEADS = 24
ATTN_KV_HEADS = 2
ATTN_HEAD_DIM = 256
HC_STREAMS = 4
HC_WIDTH = HC_STREAMS * HIDDEN
HC_LOW = 320
IDX_HEADS = 4
IDX_KV_HEADS = 1
IDX_HD = 128
PLE_ROWS = 320001536
PLE_HEADS = 16
PLE_HD = 160
LAYERS = 48
ATTN_PERIOD = 4
FULL_PHASE = 3
PLE_LAYER = 1

FMT_BF16 = 0
FMT_F32 = 1
FMT_U32 = 2
FMT_MXFP4 = 3
FMT_FP8_F32B128 = 4
FMT_FP8_E8M0B128 = 6
FMT_I64 = 7
FMT_NVFP4 = 8

T_GLOBAL = {0: "embedding", 1: "final_norm", 2: "lm_head", 41: "mixer_down", 42: "mixer_up"}
T_EVERY = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 32, 33, 34, 35, 36, 37]
T_GDN = [13, 14, 15, 16, 17, 18, 19, 20, 21]
T_ATTN = [22, 23, 24, 25, 26, 27, 38, 39, 40]
T_PLE = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]


def is_gdn(layer):
	return (layer % ATTN_PERIOD) != FULL_PHASE


def natural_shape(kind, layer, tp_degree):
	experts = EXPERTS // tp_degree
	def e(kind_id, rows, cols, fmt=FMT_BF16):
		return (rows, cols, fmt)
	if kind in (3, 4):
		return e(kind, 1, HC_WIDTH)
	if kind == 5:
		return e(kind, EXPERTS, HIDDEN)
	if kind in (6, 7):
		return e(kind, experts * EXPERT_DIM, HIDDEN, FMT_FP8_F32B128)
	if kind == 8:
		return e(kind, experts * HIDDEN, EXPERT_DIM, FMT_FP8_F32B128)
	if kind in (9, 10):
		return e(kind, EXPERT_DIM // tp_degree, HIDDEN)
	if kind == 11:
		return e(kind, HIDDEN, EXPERT_DIM // tp_degree)
	if kind == 12:
		return e(kind, 1, HIDDEN)
	if kind == 13:
		return e(kind, 2 * (GDN_KEY_HEADS // tp_degree) * GDN_HKD + (GDN_VALUE_HEADS // tp_degree) * GDN_HVD, HIDDEN)
	if kind == 14:
		return e(kind, GDN_VALUE // tp_degree, HIDDEN)
	if kind in (15, 16):
		return e(kind, GDN_VALUE_HEADS // tp_degree, HIDDEN)
	if kind == 17:
		return e(kind, HIDDEN, GDN_VALUE // tp_degree)
	if kind == 18:
		return e(kind, 2 * (GDN_KEY_HEADS // tp_degree) * GDN_HKD + (GDN_VALUE_HEADS // tp_degree) * GDN_HVD, 4)
	if kind in (19, 20):
		return e(kind, 1, GDN_VALUE_HEADS // tp_degree, FMT_F32)
	if kind == 21:
		return e(kind, 1, GDN_HVD)
	if kind == 22:
		return e(kind, (ATTN_HEADS // tp_degree) * 2 * ATTN_HEAD_DIM, HIDDEN)
	if kind in (23, 24):
		return e(kind, ATTN_KV_HEADS * ATTN_HEAD_DIM, HIDDEN)
	if kind == 25:
		return e(kind, HIDDEN, (ATTN_HEADS // tp_degree) * ATTN_HEAD_DIM)
	if kind in (26, 27):
		return e(kind, 1, ATTN_HEAD_DIM)
	if kind == 32 or kind == 35:
		return e(kind, HC_LOW, HC_WIDTH)
	if kind == 33 or kind == 36:
		return e(kind, HC_WIDTH, HC_LOW)
	if kind == 34 or kind == 37:
		return e(kind, HC_STREAMS, HC_WIDTH)
	if kind == 38:
		return e(kind, (IDX_HEADS + IDX_KV_HEADS) * IDX_HD, HIDDEN)
	if kind in (39, 40):
		return e(kind, 1, IDX_HD)
	if kind == 41:
		return e(kind, HC_LOW, HC_WIDTH)
	if kind == 42:
		return e(kind, HC_WIDTH, HC_LOW)
	if kind == 45:
		return e(kind, HC_WIDTH, HIDDEN)
	if kind == 46:
		return e(kind, HIDDEN, HIDDEN)
	if kind in (47, 48, 49):
		return e(kind, 1, HC_WIDTH)
	if kind == 50:
		return e(kind, HC_WIDTH, 4)
	if kind == 51:
		return e(kind, 1, 3, FMT_I64)
	if kind in (52, 53):
		return e(kind, 1, PLE_HEADS, FMT_I64)
	if kind == 54:
		return e(kind, PLE_ROWS // tp_degree, PLE_HD)
	if kind == 0 or kind == 2:
		return e(kind, VOCAB // tp_degree, HIDDEN)
	if kind == 1:
		return e(kind, 1, HC_WIDTH)
	raise ValueError(f"unknown tensor kind {kind}")


def payload_bytes(fmt, rows, cols):
	elements = rows * cols
	if fmt in (FMT_MXFP4, FMT_NVFP4):
		return elements // 2
	if fmt in (FMT_FP8_F32B128, FMT_FP8_E8M0B128):
		return elements
	if fmt in (FMT_F32, FMT_U32):
		return elements * 4
	if fmt == FMT_I64:
		return elements * 8
	return elements * 2


def scale_bytes(fmt, rows, cols):
	if fmt == FMT_MXFP4:
		return (rows * cols) // 32
	if fmt == FMT_NVFP4:
		plane = rows * (cols // 16)
		per_expert = EXPERT_DIM if cols == HIDDEN else (HIDDEN if cols == EXPERT_DIM else 0)
		if per_expert == 0 or rows % per_expert != 0:
			return 0
		return plane + (rows // per_expert) * 8
	if fmt == FMT_FP8_F32B128:
		return (rows // 128) * (cols // 128) * 4
	if fmt == FMT_FP8_E8M0B128:
		return rows * (cols // 128)
	return 0


E4M3 = np.array([0.0] * 256, dtype=np.float32)
for _b in range(256):
	_s = 1.0 if _b < 128 else -1.0
	_e = (_b >> 3) & 0xF
	_m = _b & 0x7
	if _e == 0:
		E4M3[_b] = _s * (_m / 8.0) * 2.0 ** -6
	elif _e == 15 and _m == 7:
		E4M3[_b] = np.nan
	else:
		E4M3[_b] = _s * (1.0 + _m / 8.0) * 2.0 ** (_e - 7)


def read_dir(path):
	with open(path, "rb") as fh:
		header = fh.read(120)
		fields = struct.unpack("<26I 2Q", header)
		(magic, fmt_ver, header_bytes, entry_bytes, tensor_count, hidden_dim,
		 layer_count, first_layer, total_layers, period, full_phase,
		 gkh, gvh, ghkd, ghvd, conv_k, qh, kvh, ahd, rope_d, re, ept, eid,
		 ovc, mxfp4_gs, mtp_count, directory_offset, file_bytes) = fields
		if magic != 0x50533451:
			raise SystemExit(f"magic {magic:#x} wrong")
		fh.seek(directory_offset)
		directory = fh.read(tensor_count * entry_bytes)
	entries = []
	for i in range(tensor_count):
		off = i * entry_bytes
		kind, layer, fmt, rows, cols, sgs, poff, pbytes, soff, sbytes = \
			struct.unpack_from("<6I Q Q Q Q", directory, off)
		entries.append(dict(kind=kind, layer=layer, fmt=fmt, rows=rows, cols=cols,
			scale_group_size=sgs, payload_offset=poff, payload_bytes=pbytes,
			scale_offset=soff, scale_bytes=sbytes))
	return dict(fmt_ver=fmt_ver, header_bytes=header_bytes, entry_bytes=entry_bytes,
		tensor_count=tensor_count, hidden=hidden_dim, layer_count=layer_count,
		first_layer=first_layer, total_layers=total_layers, period=period,
		full_phase=full_phase, mtp=mtp_count, directory_offset=directory_offset,
		file_bytes=file_bytes), entries


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pack", required=True)
	parser.add_argument("--tp-degree", type=int, default=8)
	parser.add_argument("--json-out", default=None)
	arguments = parser.parse_args()
	size = os.path.getsize(arguments.pack)
	header, entries = read_dir(arguments.pack)
	failures = []
	formats = {}
	covered = 120 + header["tensor_count"] * header["entry_bytes"]
	seen_ranges = []
	for entry in entries:
		kind = entry["kind"]
		layer = entry["layer"]
		fmt = entry["fmt"]
		formats[fmt] = formats.get(fmt, 0) + 1
		label = f"kind={kind} layer={layer} fmt={fmt}"
		if header["fmt_ver"] != 2:
			failures.append((label, f"format_version {header['fmt_ver']}"))
		if layer == 0xFFFFFFFF:
			expected = natural_shape(kind, 0, arguments.tp_degree)
		elif layer == 0xFFFFFFFE:
			failures.append((label, "mtp layer present in a --no-mtp pack"))
			continue
		else:
			expected = natural_shape(kind, layer, arguments.tp_degree)
		rows, cols, natural = expected
		if (rows, cols) != (entry["rows"], entry["cols"]):
			failures.append((label, f"geometry {entry['rows']}x{entry['cols']} != {rows}x{cols}"))
		if fmt != natural:
			failures.append((label, f"format stamp {fmt} != contract {natural}"))
		want_payload = payload_bytes(fmt, entry["rows"], entry["cols"])
		want_scale = scale_bytes(fmt, entry["rows"], entry["cols"])
		if entry["payload_bytes"] != want_payload:
			failures.append((label, f"payload_bytes {entry['payload_bytes']} != {want_payload}"))
		if entry["scale_bytes"] != want_scale:
			failures.append((label, f"scale_bytes {entry['scale_bytes']} != {want_scale}"))
		if entry["payload_offset"] % 256 != 0:
			failures.append((label, f"payload_offset {entry['payload_offset']} unaligned"))
		end = max(entry["payload_offset"] + entry["payload_bytes"],
			entry["scale_offset"] + entry["scale_bytes"])
		if end > size:
			failures.append((label, f"entry end {end} beyond file {size}"))
		seen_ranges.append((entry["payload_offset"], end, label))
		covered = max(covered, end)
	seen_ranges.sort()
	for i in range(1, len(seen_ranges)):
		if seen_ranges[i][0] < seen_ranges[i - 1][1]:
			failures.append((seen_ranges[i][2], f"overlaps {seen_ranges[i - 1][2]}"))
			break
	with open(arguments.pack, "rb") as fh:
		hand_checks = {}
		gdn0 = next(e for e in entries if e["kind"] == 13 and e["layer"] == 0)
		fh.seek(gdn0["payload_offset"])
		raw = np.frombuffer(fh.read(2 * 2560), dtype="<u2").astype(np.uint32)
		bf16_vals = np.frombuffer((raw << 16).astype("<u4").tobytes(), dtype="<f4")
		hand_checks["gdn0_qkv_bf16"] = dict(count=2560, finite=bool(np.isfinite(bf16_vals).all()),
			std=float(bf16_vals.std()), mean=float(bf16_vals.mean()))
		if not np.isfinite(bf16_vals).all() or bf16_vals.std() == 0.0:
			failures.append(("gdn0_qkv bf16 hand-check", "degenerate"))
		w1 = next(e for e in entries if e["kind"] == 6 and e["layer"] == 0)
		fh.seek(w1["payload_offset"])
		f8 = np.frombuffer(fh.read(128 * 128), dtype=np.uint8)
		fh.seek(w1["scale_offset"])
		scales = np.frombuffer(fh.read((w1["rows"] // 128) * (w1["cols"] // 128) * 4), dtype="<f4")
		block_scale = float(scales[0])
		decoded = E4M3[f8.reshape(128, 128)] * np.float32(block_scale)
		hand_checks["moe0_w1_fp8_block"] = dict(scale=float(block_scale), finite=bool(np.isfinite(decoded).all()),
			std=float(decoded.std()), nonzero=float((decoded != 0).mean()))
		if not np.isfinite(decoded).all() or decoded.std() == 0.0 or abs(block_scale) > 1e6:
			failures.append(("moe0_w1 fp8 hand-check", f"degenerate decode scale={block_scale:.4g}"))
		fp8_entries = [e for e in entries if e["fmt"] in (FMT_FP8_F32B128, FMT_FP8_E8M0B128)]
		sane, broken = 0, []
		for entry in fp8_entries:
			fh.seek(entry["scale_offset"])
			head = np.frombuffer(fh.read(min(64, entry["scale_bytes"])), dtype="<f4")
			if entry["fmt"] == FMT_FP8_E8M0B128:
				sane += 1
				continue
			if np.isfinite(head).all() and float(np.abs(head).max()) <= 1e6:
				sane += 1
			else:
				broken.append(f"kind={entry['kind']} layer={entry['layer']}")
		hand_checks["fp8_scale_planes"] = dict(total=len(fp8_entries), sane=sane,
			broken=broken, broken_sample=broken[:6])
		if broken:
			failures.append(("fp8 scale planes", f"{len(broken)} of {len(fp8_entries)} carry non-f32-scale content"))
		mult = next(e for e in entries if e["kind"] == 51)
		fh.seek(mult["payload_offset"])
		mvals = np.frombuffer(fh.read(24), dtype="<i8")
		hand_checks["ple_multipliers_i64"] = dict(values=[int(v) for v in mvals])
		if not np.all(mvals != 0):
			failures.append(("ple_multipliers", "zero multiplier"))
		voc = next(e for e in entries if e["kind"] == 52)
		off = next(e for e in entries if e["kind"] == 53)
		fh.seek(voc["payload_offset"])
		vvals = np.frombuffer(fh.read(128), dtype="<i8")
		fh.seek(off["payload_offset"])
		ovals = np.frombuffer(fh.read(128), dtype="<i8")
		hand_checks["ple_head_vocabs"] = dict(sum_rows=int(vvals.sum()), values=[int(v) for v in vvals])
		hand_checks["ple_head_offsets"] = dict(values=[int(v) for v in ovals])
		if int(vvals.sum()) > PLE_ROWS:
			failures.append(("ple_head_vocabs", f"sum {int(vvals.sum())} > {PLE_ROWS}"))
		if np.any(np.diff(ovals) <= 0):
			failures.append(("ple_head_offsets", "not ascending"))
		if int(ovals[-1] + vvals[-1]) > PLE_ROWS:
			failures.append(("ple_head_offsets", "last head exceeds table rows"))
	result = dict(pack=os.path.abspath(arguments.pack), file_bytes=size,
		header=header, format_histogram=formats, directory_sums=dict(
			payload=sum(e["payload_bytes"] for e in entries),
			scale=sum(e["scale_bytes"] for e in entries)),
		covered_bytes=int(covered), tail_unexplained=int(size - covered),
		hand_checks=hand_checks, failures=[list(f) for f in failures],
		entry_count=len(entries), tp_degree=arguments.tp_degree)
	print(json.dumps(result, indent=1))
	if arguments.json_out:
		with open(arguments.json_out, "w") as out:
			json.dump(result, out, indent=1)
	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
