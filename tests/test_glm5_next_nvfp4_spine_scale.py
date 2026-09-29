import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import glm5_next_resident_stagepack as pack

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
E4M3_ONE = 0x38
E4M3_TWO = 0x40


def write_checkpoint(directory, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    with open(directory / 'model.safetensors', 'wb') as out:
        out.write(struct.pack('<Q', len(encoded)))
        out.write(encoded)
        for blob in blobs:
            out.write(blob)
    (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: 'model.safetensors' for name in tensors}}))
    (directory / 'config.json').write_text('{}')


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main():
    rows, real_cols = 2, 32
    codes = np.array([[c % 16 for c in range(real_cols)], [(15 - c) % 16 for c in range(real_cols)]], dtype=np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    blocks = np.array([[E4M3_ONE, E4M3_TWO], [E4M3_TWO, E4M3_ONE]], dtype=np.uint8)
    block_values = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    global_scale = np.array([0.375], dtype=np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write_checkpoint(directory, {
            'w': ('U8', packed.shape, packed.tobytes()),
            'w_scale': ('U8', blocks.shape, blocks.tobytes()),
            'w_scale_2': ('F32', (1,), global_scale.tobytes()),
        })
        got = bf16_to_f32(pack.SourceReader(directory).spine_bf16('w'))
    want = np.array(E2M1, dtype=np.float32)[codes] * np.repeat(block_values, 16, axis=1) * global_scale[0]
    assert got.shape == (rows, real_cols), got.shape
    np.testing.assert_array_equal(got, want)
    print('PASS nvfp4 dense spine: value = e2m1 * e4m3 block * global, every code, no extra factor')


if __name__ == '__main__':
    main()
