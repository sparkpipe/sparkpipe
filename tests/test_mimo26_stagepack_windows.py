import io
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import mimo26_stagepack as packer


class Source:
    def __init__(self, root, tensors):
        self.root = root
        self.tensors = tensors
        self.weight_map = {name: 'fixture.bin' for name in tensors}

    def resolve(self, name):
        base, array = self.tensors[name]
        return 'fixture.bin', {'shape': list(array.shape)}, base


def main():
    rng = np.random.default_rng(29)
    weight = rng.integers(0, 65535, size=(300, 96), dtype=np.uint16)
    grid = rng.random((3, 12), dtype=np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with open(root / 'fixture.bin', 'wb') as out:
            out.write(b'\0' * 40)
            out.write(weight.tobytes())
            out.write(grid.tobytes())
        source = Source(root, {'w': (40, weight), 'w_scale_inv': (40 + weight.nbytes, grid)})
        reader = packer.SourceReader(source)
        for chunk_bytes in (packer.CHUNK_BYTES, 4 * 96 * 2, 1):
            packer.CHUNK_BYTES = chunk_bytes
            for row0, rows, col_base, columns in ((0, 300, 0, 24), (0, 300, 72, 24), (17, 200, 24, 48), (5, 1, 8, 8)):
                out = io.BytesIO()
                span = packer.Span(packer.SPAN_RECT, 'w', 'BF16', row0, rows, 96, col_base, columns,
                                   scale_name='w_scale_inv', scale_row0=0, scale_rows=3,
                                   scale_col_base=col_base // 8, scale_columns=columns // 8)
                reader.copy_rect(span, out)
                want = weight[row0:row0 + rows, col_base:col_base + columns]
                assert out.getvalue() == want.tobytes(), (chunk_bytes, row0, rows, col_base, columns)
                out = io.BytesIO()
                reader.copy_scale(span, out)
                want = grid[:, col_base // 8:col_base // 8 + columns // 8]
                assert out.getvalue() == want.tobytes(), ('scale', chunk_bytes, col_base, columns)
    print('PASS mimo26 packer column windows: payload and scale windows equal the numpy slice for every chunk size')


if __name__ == '__main__':
    main()
