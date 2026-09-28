import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import mimo26_stagepack as packer

MODEL_HEADER = ROOT / 'model-families/mimo26/include/sparkpipe/spark_mimo26_model.h'
TOOL = ROOT / 'modules/mimo26_resident_decode_stage/tools/mimo26_rank_pack_tool.c'


def table(name):
    text = MODEL_HEADER.read_text()
    body = re.search(name + r'\[SPARK_MIMO26_MODEL_LAYER_COUNT\] =\s*\{(.*?)\}', text, re.S).group(1)
    return [int(value) for value in body.replace('\n', '').split(',')]


def build_tool(directory):
    binary = directory / 'mimo26_rank_pack_tool'
    subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Werror',
                    '-I', str(ROOT / 'include'), '-I', str(ROOT / 'model-families/mimo26/include'),
                    '-I', str(ROOT / 'modules/mimo26_resident_decode_stage/source'),
                    str(TOOL), str(ROOT / 'src/spark_status.c'), str(ROOT / 'src/spark_ck128.c'),
                    '-o', str(binary)], check=True)
    return binary


def plan(rank, tp=4):
    config = {'hybrid_layer_pattern': table('SPARK_MIMO26_MODEL_LAYER_KIND'),
              'moe_layer_freq': table('SPARK_MIMO26_MODEL_LAYER_IS_MOE')}
    records = packer.build_plan('flash', config, tp, rank)
    layout, file_bytes = packer.plan_layout(records)
    header = list(packer.header_fields('flash', records, packer.HEADER_BYTES, file_bytes, tp, rank))
    entries = []
    for record, (payload_offset, scale_offset) in zip(records, layout):
        entries.append([record.kind, record.layer, record.weight_format, record.rows, record.columns, 0,
                        payload_offset, record.payload_bytes, scale_offset, record.scale_bytes])
    return header, entries, file_bytes


def write_pack(path, header, entries, file_bytes):
    with open(path, 'wb') as out:
        out.write(packer.HEADER_STRUCT.pack(*header))
        for entry in entries:
            out.write(packer.ENTRY_STRUCT.pack(*entry))
        out.truncate(file_bytes)


def check(binary, path, tp=4):
    return subprocess.run([str(binary), 'check', str(path), str(tp)], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        binary = build_tool(directory)
        expert_bytes = None
        for rank in range(4):
            header, entries, file_bytes = plan(rank)
            path = directory / f'rank{rank}.sp'
            write_pack(path, header, entries, file_bytes)
            result = check(binary, path)
            assert result.returncode == 0, result.stderr
            assert f'file_bytes={file_bytes}' in result.stdout, result.stdout
            assert file_bytes == 42513228288, file_bytes
            expert_bytes = re.search(r'expert_bytes=(\d+)', result.stdout).group(1)
        assert int(expert_bytes) == 47 * 64 * 3 * (2048 * 4096 // 2 + 2048 * 4096 // 32), expert_bytes
        header, entries, file_bytes = plan(0)
        qkv = next(i for i, e in enumerate(entries) if e[0] == packer.KIND_QKV and e[1] == 5)
        expert = next(i for i, e in enumerate(entries) if e[0] == packer.KIND_EXPERT_DOWN and e[1] == 7)
        mutations = {
            'qkv rows of the fused order': lambda h, e, f: e[qkv].__setitem__(3, 13568),
            'missing tensor': lambda h, e, f: (e.pop(expert), h.__setitem__(4, h[4] - 1)),
            'tensor count': lambda h, e, f: h.__setitem__(4, h[4] - 1),
            'duplicate kind': lambda h, e, f: e[qkv + 1].__setitem__(0, packer.KIND_QKV),
            'unaligned payload': lambda h, e, f: e[qkv].__setitem__(6, e[qkv][6] + 16),
            'expert codec': lambda h, e, f: e[expert].__setitem__(2, packer.WEIGHT_BF16),
            'scale plane': lambda h, e, f: e[expert].__setitem__(9, e[expert][9] // 2),
            'hidden': lambda h, e, f: h.__setitem__(5, 6144),
            'layer count': lambda h, e, f: h.__setitem__(6, 47),
        }
        for label, mutate in mutations.items():
            h, e = list(header), [list(entry) for entry in entries]
            mutate(h, e, file_bytes)
            path = directory / 'mutated.sp'
            write_pack(path, h, e, file_bytes)
            result = check(binary, path)
            assert result.returncode != 0 and 'FAIL' in result.stderr, label
        path = directory / 'short.sp'
        write_pack(path, header, entries, file_bytes - 256)
        assert check(binary, path).returncode != 0, 'short file'
        path = directory / 'rank0.sp'
        assert check(binary, directory / 'rank0.sp', 2).returncode != 0, 'tp2 against a tp4 pack'
        assert check(binary, directory / 'rank0.sp', 8).returncode != 0, 'tp8 against a tp4 pack'
    print('PASS mimo26 rank pack binder: 4 TP4 ranks from the packer plan bind; 11 corrupted packs fail closed')


if __name__ == '__main__':
    main()
