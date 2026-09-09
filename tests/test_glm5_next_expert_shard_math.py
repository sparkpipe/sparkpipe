import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import glm5_next_resident_stagepack as pack

class Source:
    weight_map = {'model.language_model.layers.3.mlp.experts.0.up_proj.weight':'fixture'}

    def __init__(self):
        rng = np.random.default_rng(173)
        self.weights = {}
        for expert in range(2):
            for projection in ('up','gate','down'):
                shape = (128,2048) if projection == 'down' else (2048,128)
                values = rng.normal(0,0.1,shape).astype(np.float32)
                bits = values.view(np.uint32)
                self.weights[f'model.language_model.layers.3.mlp.experts.{expert}.{projection}_proj.weight'] = ((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')

    def meta(self,name):
        return 'BF16',self.weights[name].shape

    def expert_payload(self,name,r0,r1,c0,c1):
        return self.weights[name][r0:r1,c0:c1].tobytes()

def decode(blob,shape):
    return (np.frombuffer(blob,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape).astype(np.float64)

def activate(up,gate):
    up,gate = np.clip(up,-10,10),np.minimum(gate,10)
    return up*gate/(1+np.exp(-gate))

def main():
    source = Source()
    x = np.random.default_rng(419).normal(size=(3,128))
    with patch.multiple(pack,EXPERTS=2,HIDDEN=128,EXPERT_INTER=2048):
        expected = []
        for expert in range(2):
            matrices = [decode(source.weights[f'model.language_model.layers.3.mlp.experts.{expert}.{name}_proj.weight'].tobytes(),shape) for name,shape in (('up',(2048,128)),('gate',(2048,128)),('down',(128,2048)))]
            up,gate,down = matrices
            expected.append(activate(x@up.T,x@gate.T)@down.T)
        for degree in (1,4,16):
            actual = [np.zeros((3,128)) for _ in range(2)]
            for rank in range(degree):
                builder = pack.Packer(source,degree,rank,3,1,False,False,False)
                builder.add_experts(3)
                w1,w2 = [decode(b''.join(item.produce_payload()),(2,item.entry.rows,item.entry.columns)) for item in builder.plan]
                for expert in range(2):
                    projected = x@w1[expert].T
                    up,gate = np.split(projected,2,axis=1)
                    actual[expert] += activate(up,gate)@w2[expert].T
            for expert in range(2):
                np.testing.assert_allclose(actual[expert],expected[expert],rtol=1e-10,atol=1e-10,err_msg=f'TP{degree} expert {expert}')
    print('PASS packed expert SwiGLU: TP1/TP4/TP16, every rank, two experts, three inputs')

if __name__ == '__main__':
    main()
