#!/usr/bin/env python3
"""Download the selected real checkpoint on Actions; return one data-only block.

The complete checkpoint is downloaded to the runner cache. Only the selected
MLP weights and reference samples are returned through the isolated result branch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path


def export(args,report):
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
    from modernbert_teacher import prepare_masked_inputs
    torch.set_num_threads(2)
    report['stage']='resolve_checkpoint'
    info=HfApi().model_info(args.model,revision=args.revision,token=False)
    revision=info.sha
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('checkpoint revision is not immutable')
    report['resolved_revision']=revision;report['stage']='download_checkpoint'
    config=AutoConfig.from_pretrained(args.model,revision=revision,trust_remote_code=False,token=False)
    if config.model_type!='modernbert':raise ValueError('expected ModernBERT architecture')
    config.reference_compile=False
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=revision,trust_remote_code=False,token=False)
    model=AutoModelForMaskedLM.from_pretrained(args.model,revision=revision,config=config,
        trust_remote_code=False,use_safetensors=True,token=False,torch_dtype=torch.float32,attn_implementation='eager').cpu().eval()
    if not 0<=args.layer<len(model.model.layers):raise ValueError('invalid selected layer')
    report['stage']='export_weights';mlp=model.model.layers[args.layer].mlp
    if config.hidden_activation not in ('gelu','gelu_pytorch_tanh','gelu_new'):raise ValueError('unsupported activation')
    wi=mlp.Wi.weight.detach().float().numpy();wo=mlp.Wo.weight.detach().float().numpy()
    bi=np.zeros(wi.shape[0],np.float32) if mlp.Wi.bias is None else mlp.Wi.bias.detach().float().numpy()
    bo=np.zeros(wo.shape[0],np.float32) if mlp.Wo.bias is None else mlp.Wo.bias.detach().float().numpy()
    block=args.output/'block';block.mkdir(parents=True,exist_ok=True)
    np.savez(block/'weights.npz',wi=wi,wo=wo,bi=bi,bo=bo)
    provenance={'checkpoint':args.model,'revision':revision,'layer':args.layer,'pretrained':True,
        'source':'https://huggingface.co/'+args.model,'torch':torch.__version__}
    manifest={'format':'spark-modernbert-gated-function-v1','hidden':config.hidden_size,
        'intermediate':config.intermediate_size,'activation':config.hidden_activation,
        'payload_sha256':hashlib.sha256((block/'weights.npz').read_bytes()).hexdigest(),'provenance':provenance}
    (block/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'config.json').write_text(config.to_json_string())
    report['stage']='reference_inference';inputs=[];outputs=[];traces=[];capture=[]
    def hook(_,values):capture.append(values[0].detach().cpu())
    handle=mlp.register_forward_pre_hook(hook)
    try:
        for line in args.inputs.read_text().splitlines():
            if not line.strip():continue
            text=json.loads(line)['text'];ids,pos,cropped=prepare_masked_inputs(tokenizer,text,args.max_length)
            capture.clear()
            with torch.inference_mode():
                result=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long))
                if len(capture)!=1 or capture[0].ndim!=3:raise ValueError('unexpected MLP hook geometry')
                x=capture[0][0,pos].clone();y=mlp(x[None])[0].clone()
                values,tokens=torch.topk(torch.softmax(result.logits[0,pos].float(),dim=-1),5)
            inputs.append(x.numpy());outputs.append(y.numpy())
            traces.append({'text':text,'input_ids':ids,'mask_index':pos,'cropped':cropped,
                'topk':[{'token':int(t),'prob':float(v),'text':tokenizer.decode([int(t)])} for t,v in zip(tokens,values)]})
    finally:handle.remove()
    if not inputs:raise ValueError('no reference inputs')
    xs=np.asarray(inputs);ys=np.asarray(outputs)
    with torch.inference_mode():double=copy.deepcopy(mlp).double()(torch.tensor(xs,dtype=torch.float64)).numpy()
    np.savez(args.output/'reference.npz',inputs=xs,torch_float32=ys,torch_float64=double)
    (args.output/'teacher-smoke.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
    report.update(status='completed',stage='complete',layer=args.layer,hidden=config.hidden_size,
        intermediate=config.intermediate_size,activation=config.hidden_activation,reference_samples=len(xs),
        full_checkpoint_downloaded=True,returned_scope='Selected feed-forward block and reference data; not the complete checkpoint or converted MLM.',
        parameter_count=sum(p.numel() for p in model.parameters()),
        selected_parameter_count=int(wi.size+wo.size+bi.size+bo.size),payload_sha256=manifest['payload_sha256'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='answerdotai/ModernBERT-base');ap.add_argument('--revision',default='main')
    ap.add_argument('--layer',type=int,default=0);ap.add_argument('--inputs',type=Path,default=Path('examples/modernbert_masks.jsonl'))
    ap.add_argument('--max-length',type=int,default=128);ap.add_argument('--output',type=Path,default=Path('artifacts/modernbert-transfer-runner'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    run_id=os.environ.get('GITHUB_RUN_ID');repository=os.environ.get('GITHUB_REPOSITORY')
    report={'status':'started','checkpoint':args.model,'requested_revision':args.revision,'layer':args.layer,
        'workflow_url':f'https://github.com/{repository}/actions/runs/{run_id}' if run_id and repository else None}
    try:export(args,report)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=re.sub(r'https?://\S+','<url>',str(exc))[:1200])
        raise SystemExit(1)
    finally:
        (args.output/'runner-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
