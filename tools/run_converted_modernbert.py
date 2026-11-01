#!/usr/bin/env python3
"""Run the converted mathematical MLM without importing torch/transformers."""
import argparse
import heapq
import json
import math
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spark.modernbert_model import NativeModernBert


def encode(directory,text,max_length):
    from tokenizers import Tokenizer
    directory=Path(directory);spec=json.loads((directory/'tokenizer/spark_tokenizer.json').read_text())
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer/tokenizer.json'))
    ids=tokenizer.encode(text,add_special_tokens=False).ids
    mask=spec['mask_id'];positions=[i for i,t in enumerate(ids) if t==mask]
    if not positions:raise ValueError('input must contain a mask token')
    capacity=max_length-2
    if len(ids)>capacity:
        if len(positions)>1:raise ValueError('long multiple-mask text needs explicit token windows')
        start=max(0,min(positions[0]-capacity//2,len(ids)-capacity));ids=ids[start:start+capacity]
    ids=[spec['cls_id'],*ids,spec['sep_id']]
    return ids,[i for i,t in enumerate(ids) if t==mask],tokenizer


def predict(model,ids,queries,*,tokenizer=None,topk=5,attention_mask=None,position_ids=None):
    result=model.run(ids,attention_mask=attention_mask,position_ids=position_ids,query_positions=queries)
    output=[]
    for row,position in enumerate(queries):
        values=result['logits'][row*model.vocab:(row+1)*model.vocab];maximum=max(values)
        denominator=sum(math.exp(float(v)-maximum) for v in values)
        indices=heapq.nlargest(topk,range(model.vocab),key=lambda i:values[i])
        output.append({'position':position,'topk':[{'token':i,'text':tokenizer.decode([i]) if tokenizer else None,
            'logit':float(values[i]),'prob':math.exp(values[i]-maximum)/denominator} for i in indices]})
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    inputs=p.add_mutually_exclusive_group(required=True);inputs.add_argument('--text');inputs.add_argument('--ids-json',type=Path)
    p.add_argument('--topk',type=int,default=5);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.topk<=256 or not 1<=a.threads<=8:p.error('topk must be 1..256 and threads 1..8')
    os.environ['OPENBLAS_NUM_THREADS']=str(a.threads);os.environ['OMP_NUM_THREADS']=str(a.threads)
    with NativeModernBert(a.model) as model:
        if a.text:
            ids,queries,tokenizer=encode(a.model,a.text,model.max_sequence);mask=positions=None
        else:
            d=json.loads(a.ids_json.read_text());ids=d['input_ids'];queries=d['query_positions'];tokenizer=None
            mask=d.get('attention_mask');positions=d.get('position_ids')
        predictions=predict(model,ids,queries,tokenizer=tokenizer,topk=a.topk,attention_mask=mask,position_ids=positions)
        print(json.dumps({'backend':'spark-native-mathematical-program','predictions':predictions,
            'framework_imports':{name:name in sys.modules for name in ('torch','transformers','numpy')}},indent=2))
