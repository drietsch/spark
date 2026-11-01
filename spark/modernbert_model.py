"""Full ModernBERT MLM -> typed mathematical program and independent native CPU.

Inference imports neither torch nor transformers. Export uses the source model
only to read its parameters/config; numerical execution is C++ + CPU BLAS.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
import struct
import subprocess
from pathlib import Path

FORMAT='spark-modernbert-mathematical-model-v1'
HEADER=struct.Struct('<8s12I3f')
MAGIC=b'SPRKMB01'
ACTIVATIONS={'gelu':0,'gelu_pytorch_tanh':1,'gelu_new':1}


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
    return h.hexdigest()


def expected_program(layers,global_every=3,tied=True):
    program=[{'id':'embedding','op':'embedding_lookup_layer_norm','inputs':['input_ids'],'output':'state'}]
    for i in range(layers):
        program.extend([{'id':f'layer{i}.attention','op':'rotary_attention_residual','inputs':['state','attention_mask','position_ids'],'output':'state',
                         'normalize_input':i>0,'attention':'global' if i%global_every==0 else 'local'},
                        {'id':f'layer{i}.function','op':'normalized_gelu_gated_function_residual','inputs':['state'],'output':'state'}])
    program.extend([{'id':'final_norm','op':'layer_norm','inputs':['state'],'output':'encoded'},
                    {'id':'prediction','op':'dense_gelu_layer_norm','inputs':['encoded','query_positions'],'output':'prediction'},
                    {'id':'decoder','op':'dense_vocabulary_projection','inputs':['prediction'],'output':'logits','weight':'embedding.weight' if tied else 'decoder.weight'}])
    return program


def export_model(model,directory,*,source):
    """Export every parameter of a supported eval-mode ModernBERT masked LM."""
    import numpy as np
    c=model.config
    if model.training or c.model_type!='modernbert':raise ValueError('requires eval-mode ModernBERT')
    if c.hidden_activation not in ACTIVATIONS or c.classifier_activation not in ACTIVATIONS:raise ValueError('unsupported activation')
    if getattr(c,'rope_scaling',None) not in (None,{}):raise ValueError('dynamic/scaled RoPE is unsupported')
    h,i,l,heads,vocab=c.hidden_size,c.intermediate_size,c.num_hidden_layers,c.num_attention_heads,c.vocab_size
    if not 1<=h<=2048 or not 1<=i<=8192 or not 1<=l<=64 or not 1<=heads<=64 or h%heads or (h//heads)%2 or not 1<=vocab<=131072:
        raise ValueError('model geometry outside supported budgets')
    if not 1<=c.max_position_embeddings<=8192 or not 1<=c.local_attention<=8192 or not 1<=c.global_attn_every_n_layers<=64 or not 0<c.norm_eps<1:
        raise ValueError('invalid execution constants')
    tied=model.decoder.weight.data_ptr()==model.model.embeddings.tok_embeddings.weight.data_ptr()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);weights=[];seen=set()
    tmp=directory/'weights.sparkmb.tmp'
    def array(t):
        if str(t.dtype) not in ('torch.float32','torch.float16','torch.bfloat16'):raise ValueError('unsupported source coefficient dtype')
        a=t.detach().float().cpu().numpy()
        if not np.isfinite(a).all():raise ValueError('nonfinite source weights')
        return np.asarray(a,dtype='<f4',order='C')
    with tmp.open('wb') as f:
        f.write(HEADER.pack(MAGIC,1,h,i,l,heads,vocab,c.max_position_embeddings,c.local_attention,c.global_attn_every_n_layers,
                int(tied),ACTIVATIONS[c.hidden_activation],ACTIVATIONS[c.classifier_activation],c.norm_eps,c.global_rope_theta,c.local_rope_theta or c.global_rope_theta))
        def write(name,value,shape=None):
            a=array(value) if value is not None else np.zeros(shape,dtype='<f4')
            if shape is not None and tuple(a.shape)!=tuple(shape):raise ValueError('unexpected tensor geometry: '+name)
            if name in seen:raise ValueError('duplicate parameter name')
            seen.add(name);offset=f.tell();raw=a.tobytes(order='C');f.write(raw)
            weights.append({'name':name,'shape':list(a.shape),'offset':offset,'bytes':len(raw)})
        def norm(prefix,module):
            if not hasattr(module,'weight'):
                write(prefix+'.weight',None,(h,));write(prefix+'.bias',None,(h,))
            else:write(prefix+'.weight',module.weight,(h,));write(prefix+'.bias',module.bias,(h,))
        def dense(prefix,module,out,ins):
            write(prefix+'.weight',module.weight,(out,ins));write(prefix+'.bias',module.bias,(out,))
        write('embedding.weight',model.model.embeddings.tok_embeddings.weight,(vocab,h));norm('embedding.norm',model.model.embeddings.norm)
        for n,layer in enumerate(model.model.layers):
            norm(f'layer{n}.attention.norm',layer.attn_norm)
            dense(f'layer{n}.attention.qkv',layer.attn.Wqkv,3*h,h)
            dense(f'layer{n}.attention.output',layer.attn.Wo,h,h)
            norm(f'layer{n}.function.norm',layer.mlp_norm)
            dense(f'layer{n}.function.input',layer.mlp.Wi,2*i,h)
            dense(f'layer{n}.function.output',layer.mlp.Wo,h,i)
            write(f'layer{n}.attention.rotary_frequencies',layer.attn.rotary_emb.inv_freq,(h//heads//2,))
        norm('final.norm',model.model.final_norm);dense('prediction.dense',model.head.dense,h,h);norm('prediction.norm',model.head.norm)
        if not tied:write('decoder.weight',model.decoder.weight,(vocab,h))
        write('decoder.bias',model.decoder.bias,(vocab,))
    if tmp.stat().st_size>2*2**30:tmp.unlink();raise ValueError('export exceeds model payload budget')
    tmp.replace(directory/'weights.sparkmb')
    manifest={'format':FORMAT,'source':dict(source),'geometry':{'hidden':h,'intermediate':i,'layers':l,'heads':heads,'vocab':vocab,
        'max_sequence':c.max_position_embeddings,'local_window':c.local_attention,'global_every':c.global_attn_every_n_layers,'tied_decoder':tied},
        'constants':{'norm_epsilon':c.norm_eps,'global_rope_theta':c.global_rope_theta,'local_rope_theta':c.local_rope_theta or c.global_rope_theta,
            'mlp_activation':c.hidden_activation,'prediction_activation':c.classifier_activation},
        'program':expected_program(l,c.global_attn_every_n_layers,tied),'parameters':weights,'parameter_payload_bytes':(directory/'weights.sparkmb').stat().st_size,
        'weights_sha256':sha256_file(directory/'weights.sparkmb'),'inference':'masked-language-model evaluation; no training/autograd',
        'representation':'explicit typed function composition; original Transformer dimensions and arithmetic retained'}
    (directory/'model.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest


def compile_runtime(directory,*,force=False):
    directory=Path(directory);source=Path(__file__).resolve().parents[1]/'runtime/spark_modernbert.cpp'
    source_hash=sha256_file(source);cache=directory/'native';cache.mkdir(parents=True,exist_ok=True)
    library=cache/f'spark_modernbert_{source_hash[:12]}.so';receipt=library.with_suffix('.json')
    if not force and library.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        if r.get('source_sha256')==source_hash and r.get('library_sha256')==sha256_file(library):return library
    compiler=__import__('shutil').which('c++') or __import__('shutil').which('g++')
    if compiler is None:raise RuntimeError('C++17 compiler required')
    flags=['-std=c++17','-O2','-ffp-contract=off','-fPIC','-shared']
    subprocess.run([compiler,*flags,str(source),'-o',str(library),'-l:libblas.so.3'],check=True,capture_output=True,text=True)
    receipt.write_text(json.dumps({'source_sha256':source_hash,'library_sha256':sha256_file(library),'flags':flags},indent=2)+'\n')
    return library


def validate_manifest(directory,*,verify=True):
    directory=Path(directory);d=json.loads((directory/'model.json').read_text())
    if type(d) is not dict or d.get('format')!=FORMAT or type(d.get('geometry')) is not dict or type(d.get('parameters')) is not list:
        raise ValueError('invalid mathematical model manifest')
    g=d['geometry']
    if (type(g.get('layers')) is not int or not 1<=g['layers']<=64 or type(g.get('global_every')) is not int or
        not 1<=g['global_every']<=64 or type(g.get('tied_decoder')) is not bool or
        d.get('program')!=expected_program(g['layers'],g['global_every'],g['tied_decoder'])):
        raise ValueError('unsupported or modified function program')
    path=directory/'weights.sparkmb'
    if not HEADER.size<=path.stat().st_size<=2*2**30 or path.stat().st_size!=d['parameter_payload_bytes']:raise ValueError('invalid parameter payload size')
    with path.open('rb') as f:header=HEADER.unpack(f.read(HEADER.size))
    if header[:2]!=(MAGIC,1):raise ValueError('invalid parameter binary format')
    if header[2:11]!=(g['hidden'],g['intermediate'],g['layers'],g['heads'],g['vocab'],g['max_sequence'],g['local_window'],g['global_every'],int(g['tied_decoder'])):
        raise ValueError('parameter header and program geometry disagree')
    constants=d['constants']
    if header[11:13]!=(ACTIVATIONS.get(constants.get('mlp_activation')),ACTIVATIONS.get(constants.get('prediction_activation'))):
        raise ValueError('activation program differs from binary')
    if any(not math.isclose(header[13+i],constants.get(name,float('nan')),rel_tol=1e-6) for i,name in enumerate(('norm_epsilon','global_rope_theta','local_rope_theta'))):
        raise ValueError('execution constants differ from binary')
    offset=HEADER.size;names=set()
    for tensor in d['parameters']:
        if type(tensor) is not dict or set(tensor)!={'name','shape','offset','bytes'} or type(tensor['name']) is not str or tensor['name'] in names or type(tensor['shape']) is not list or not tensor['shape'] or any(type(v) is not int or not 1<=v<=131072 for v in tensor['shape']):raise ValueError('invalid tensor binding')
        if type(tensor['offset']) is not int or type(tensor['bytes']) is not int or tensor['offset']!=offset or tensor['bytes']!=math.prod(tensor['shape'])*4:raise ValueError('invalid tensor bounds')
        offset+=tensor['bytes'];names.add(tensor['name'])
    if offset!=path.stat().st_size:raise ValueError('parameter layout does not cover payload')
    if verify and sha256_file(path)!=d['weights_sha256']:raise ValueError('parameter checksum mismatch')
    return d


class NativeModernBert:
    def __init__(self,directory,*,verify=True):
        self.directory=Path(directory);self.manifest=validate_manifest(directory,verify=verify)
        path=compile_runtime(directory)
        try:self.lib=ctypes.CDLL(str(path.resolve()))
        except OSError:self.lib=ctypes.CDLL(str(compile_runtime(directory,force=True).resolve()))
        L=self.lib;P=ctypes.c_void_p;I=ctypes.POINTER(ctypes.c_int32);F=ctypes.POINTER(ctypes.c_float)
        L.spark_mb_load.argtypes=[ctypes.c_char_p];L.spark_mb_load.restype=P
        L.spark_mb_error.restype=ctypes.c_char_p;L.spark_mb_destroy.argtypes=[P]
        L.spark_mb_meta.argtypes=[P,I]
        L.spark_mb_run.argtypes=[P,I,I,I,ctypes.c_int,I,ctypes.c_int,F,F]
        L.spark_mb_set_width.argtypes=[P,ctypes.c_int,ctypes.c_int]
        L.spark_mb_set_coefficient.argtypes=[P,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_float]
        self.handle=L.spark_mb_load(os.fsencode(self.directory/'weights.sparkmb'))
        if not self.handle:raise ValueError(self._error())
        meta=(ctypes.c_int32*6)();L.spark_mb_meta(self.handle,meta)
        self.hidden,self.intermediate,self.layers,self.heads,self.vocab,self.max_sequence=list(meta)
        g=self.manifest['geometry']
        if (self.hidden,self.intermediate,self.layers,self.heads,self.vocab,self.max_sequence)!=(g['hidden'],g['intermediate'],g['layers'],g['heads'],g['vocab'],g['max_sequence']):
            self.close();raise ValueError('native geometry differs from manifest')

    def _error(self):return (self.lib.spark_mb_error() or b'native execution rejected').decode(errors='replace')

    def run(self,input_ids,*,attention_mask=None,position_ids=None,query_positions=None,return_hidden=False):
        n=len(input_ids)
        if not 1<=n<=self.max_sequence or any(type(v) is not int or not 0<=v<self.vocab for v in input_ids):raise ValueError('invalid input token IDs/sequence budget')
        mask=[1]*n if attention_mask is None else list(attention_mask);positions=list(range(n)) if position_ids is None else list(position_ids)
        query=list(range(n)) if query_positions is None else list(query_positions)
        if len(mask)!=n or any(type(v) is not int or v not in (0,1) for v in mask) or not any(mask):raise ValueError('invalid attention mask')
        if len(positions)!=n or any(type(v) is not int or not 0<=v<self.max_sequence for v in positions):raise ValueError('invalid rotary positions')
        if not query or any(type(v) is not int or not 0<=v<n or not mask[v] for v in query):raise ValueError('invalid prediction query positions')
        ids=(ctypes.c_int32*n)(*input_ids);m=(ctypes.c_int32*n)(*mask);p=(ctypes.c_int32*n)(*positions);q=(ctypes.c_int32*len(query))(*query)
        output=(ctypes.c_float*(len(query)*self.vocab))();hidden=(ctypes.c_float*(n*self.hidden))() if return_hidden else None
        if not self.lib.spark_mb_run(self.handle,ids,m,p,n,q,len(query),output,hidden):raise RuntimeError(self._error())
        result={'logits':output,'shape':(len(query),self.vocab),'query_positions':query}
        if return_hidden:result.update(hidden=hidden,hidden_shape=(n,self.hidden))
        return result

    def set_active_width(self,layer,width):
        if type(layer) is not int or type(width) is not int or not self.lib.spark_mb_set_width(self.handle,layer,width):raise ValueError(self._error())

    def set_function_coefficient(self,layer,tensor,index,value):
        names={'input_weight':0,'input_bias':1,'output_weight':2,'output_bias':3}
        if tensor not in names or type(layer) is not int or type(index) is not int or index<0 or type(value) not in (int,float) or not math.isfinite(value) or abs(value)>1e4:raise ValueError('invalid live coefficient update')
        if not self.lib.spark_mb_set_coefficient(self.handle,layer,names[tensor],index,value):raise ValueError(self._error())

    def close(self):
        if getattr(self,'handle',None):self.lib.spark_mb_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
