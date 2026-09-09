import io
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from glm5_next_kda_host_oracle import Safetensors

class Reads(io.BytesIO):
    def read(self,count=-1):
        self.requested = count
        return super().read(count)

def main():
    reader = object.__new__(Safetensors)
    reader.map = {'matrix':'file'}
    reader.headers = {'file':({'matrix':{'dtype':'BF16','shape':[5,4],'data_offsets':[6,46]}},8)}
    values = np.arange(20,dtype=np.uint16).reshape(5,4)
    stream = Reads(b'x'*14+values.tobytes()+b'tail')
    reader.fds = {'file':stream}
    np.testing.assert_array_equal(reader.raw_rows('matrix',2,2),values[2:4])
    assert stream.requested == 16 and stream.tell() == 46
    for first,count in ((-1,1),(0,0),(4,2)):
        try:
            reader.raw_rows('matrix',first,count)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid row range accepted')
    reader.fds['file'] = Reads(b'x'*15)
    try:
        reader.raw_rows('matrix',0,1)
    except ValueError:
        pass
    else:
        raise AssertionError('truncated rows accepted')
    print('PASS bounded checkpoint row reads, offsets, range rejection and truncation')

if __name__ == '__main__':
    main()
