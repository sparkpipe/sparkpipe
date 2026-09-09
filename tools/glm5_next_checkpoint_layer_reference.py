import argparse
import json
from pathlib import Path

import numpy as np

from glm5_next_kda_host_oracle import PREFIX, bf16_round_f32, bf16_to_f32
from glm5_next_dsa_host_oracle import DsaSafetensors, fp8_block_to_bf16

class Checkpoint(DsaSafetensors):
    def tensor(self, name):
        raw = self.raw(name)
        if raw.dtype == np.uint16:
            return bf16_to_f32(raw)
        if raw.dtype == np.uint8:
            if raw.ndim != 2:
                raise ValueError(f'unsupported quantized tensor shape: {name}')
            scale = self.raw(name + '_scale_inv')
            return bf16_to_f32(fp8_block_to_bf16(raw,scale,*raw.shape))
        return raw.astype(np.float32)

    def linear(self, x, name):
        return bf16_round_f32(self.tensor(name + '.weight') @ x)

def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0),-x))

def rms(x, weight, epsilon):
    normalized = bf16_round_f32(x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon))
    return bf16_round_f32(normalized*weight)

def hc_site(checkpoint, streams, prefix, config):
    hc = config['hc_mult']
    epsilon = config['hc_eps']
    flat = streams.reshape(-1)
    flat = flat / np.sqrt(np.mean(flat*flat)+config['rms_norm_eps'])
    mix = checkpoint.tensor(prefix+'_fn') @ flat
    base = checkpoint.tensor(prefix+'_base')
    scale = checkpoint.tensor(prefix+'_scale')
    pre = sigmoid(mix[:hc]*scale[0]+base[:hc])+epsilon
    post = 2*sigmoid(mix[hc:2*hc]*scale[1]+base[hc:2*hc])
    comb = (mix[2*hc:]*scale[2]+base[2*hc:]).reshape(hc,hc)
    comb = np.exp(comb-comb.max(axis=1,keepdims=True))
    comb = comb/comb.sum(axis=1,keepdims=True)+epsilon
    comb /= comb.sum(axis=0,keepdims=True)+epsilon
    for _ in range(config['hc_sinkhorn_iters']-1):
        comb /= comb.sum(axis=1,keepdims=True)+epsilon
        comb /= comb.sum(axis=0,keepdims=True)+epsilon
    collapsed = bf16_round_f32(np.sum(pre[:,None]*streams,axis=0))
    return collapsed,post,comb

def hc_post(streams, output, post, comb):
    contribution = bf16_round_f32(bf16_round_f32(post)[:,None]*output)
    residual = bf16_round_f32(bf16_round_f32(comb).T @ streams)
    return bf16_round_f32(contribution+residual)

def first_kda(checkpoint, x, prefix, config, receipt):
    linear = config['linear_attn_config']
    heads,dim = linear['num_heads'],linear['head_dim']
    projected = []
    for name in ('q','k','v'):
        raw = checkpoint.linear(x,prefix+name+'_proj')
        conv = checkpoint.tensor(prefix+name+'_conv1d.weight').reshape(heads*dim,-1)
        value = bf16_round_f32(raw*conv[:,-1])
        projected.append(bf16_round_f32(value*sigmoid(value)).reshape(heads,dim))
    q,k,v = projected
    q = q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)/np.sqrt(np.float32(dim))
    k = k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
    beta = bf16_round_f32(sigmoid(checkpoint.linear(x,prefix+'b_proj')))
    state = k[:,:,None]*(beta[:,None]*v)[:,None,:]
    core = bf16_round_f32(np.sum(state*q[:,:,None],axis=1))
    gate = checkpoint.linear(checkpoint.linear(x,prefix+'g_a_proj'),prefix+'g_b_proj').reshape(heads,dim)
    normalized = core/np.sqrt(np.mean(core*core,axis=1,keepdims=True)+config['rms_norm_eps'])
    gated = bf16_round_f32(normalized*checkpoint.tensor(prefix+'o_norm.weight')*sigmoid(gate))
    receipt.update(kda_core=core,kda_gated=gated,kda_state=state)
    return checkpoint.linear(gated.reshape(-1),prefix+'o_proj')

def dense_mlp(checkpoint, x, prefix, config):
    limit = config['swiglu_limit']
    gate = np.minimum(checkpoint.linear(x,prefix+'gate_proj'),limit)
    up = np.clip(checkpoint.linear(x,prefix+'up_proj'),-limit,limit)
    activated = bf16_round_f32(bf16_round_f32(gate*sigmoid(gate))*up)
    return checkpoint.linear(activated,prefix+'down_proj')

def dense_layer(checkpoint, streams, prefix, config):
    receipt = {}
    collapsed,post,comb = hc_site(checkpoint,streams,prefix+'hc_attn',config)
    x = rms(collapsed,checkpoint.tensor(prefix+'input_layernorm.weight'),config['rms_norm_eps'])
    attention = first_kda(checkpoint,x,prefix+'self_attn.',config,receipt)
    streams = hc_post(streams,attention,post,comb)
    receipt.update(attention_norm=x,attention_output=attention,after_attention=streams.copy())
    collapsed,post,comb = hc_site(checkpoint,streams,prefix+'hc_ffn',config)
    x = rms(collapsed,checkpoint.tensor(prefix+'post_attention_layernorm.weight'),config['rms_norm_eps'])
    mlp = dense_mlp(checkpoint,x,prefix+'mlp.',config)
    streams = hc_post(streams,mlp,post,comb)
    receipt.update(mlp_norm=x,mlp_output=mlp,layer_output=streams)
    return streams,receipt

def run(checkpoint_path, token, output, layers=1):
    config = json.loads((checkpoint_path/'config.json').read_text())['text_config']
    if token < 0 or token >= config['vocab_size']:
        raise ValueError('token outside checkpoint vocabulary')
    if layers < 1 or layers >= len(config['layer_types']):
        raise ValueError('reference requires a nonempty prefix and a following layer')
    for layer in range(layers):
        if config['layer_types'][layer] != 'linear_attention' or config['mlp_layer_types'][layer] != 'dense':
            raise ValueError(f'layer {layer} is not KDA/dense')
    checkpoint = Checkpoint(str(checkpoint_path))
    embedding = checkpoint.tensor('model.language_model.embed_tokens.weight')[token].copy()
    streams = np.tile(embedding,(config['hc_mult'],1))
    receipt = {'embedding':embedding}
    for layer in range(layers):
        streams,values = dense_layer(checkpoint,streams,PREFIX+str(layer)+'.',config)
        receipt.update({f'layer{layer}_{key}':value for key,value in values.items()})
    collapsed,_,_ = hc_site(checkpoint,streams,PREFIX+str(layers)+'.hc_attn',config)
    weight = checkpoint.tensor(PREFIX+str(layers)+'.input_layernorm.weight')
    receipt['next_attention_norm'] = rms(collapsed,weight,config['rms_norm_eps'])
    if any(not np.isfinite(value).all() for value in receipt.values()):
        raise ValueError('nonfinite checkpoint reference result')
    np.savez(output,**receipt)
    print(json.dumps({'token':token,'layers':layers,'position':0,'scope':'unsharded checkpoint KDA/dense prefix reference','output':str(output),'norms':{k:float(np.linalg.norm(v)) for k,v in receipt.items()}}),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--token',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layers',type=int,default=1)
    args = parser.parse_args()
    run(args.checkpoint,args.token,args.output,args.layers)
