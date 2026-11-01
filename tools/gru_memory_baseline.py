#!/usr/bin/env python3
"""GRU comparison for M17 delayed-query memory; same training episodes.

The GRU receives exactly the event metadata available to the SPARK router and
is trained only on downstream query targets. It has more trainable parameters
than the learned SPARK write-gate, so this is not compute/parameter matched.
"""
from __future__ import annotations
import argparse,json,sys,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spark.learned_memory import episodes


def tensorize(sequences,torch,keys=4):
    length=max(map(len,sequences));batch=len(sequences)
    x=torch.zeros(batch,length,3+keys)
    targets=torch.zeros(batch,length)
    mask=torch.zeros(batch,length)
    lengths=[]
    for i,seq in enumerate(sequences):
        lengths.append(len(seq))
        for j,e in enumerate(seq):
            x[i,j,0]=e.signal;x[i,j,1]=e.value;x[i,j,2]=float(e.query)
            x[i,j,3+e.key]=1.
            if e.query:targets[i,j]=e.target;mask[i,j]=1.
    return x,targets,mask,lengths


def score(model,sequences,torch,*,batchsize=16):
    from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
    correct=0;se=0.;queries=0
    model.eval()
    with torch.no_grad():
        for start in range(0,len(sequences),batchsize):
            x,y,mask,lengths=tensorize(sequences[start:start+batchsize],torch)
            p=model(x,lengths)
            error=(p-y)*mask
            se+=float(error.square().sum())
            correct+=int(((error.abs()<1e-2)*mask).sum().item())
            queries+=int(mask.sum().item())
    return {'queries':queries,'absolute_error_under_0.01':correct,'mse':se/queries}


def run(*,epochs=25,width=12):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(41)
    train=episodes(11,450,distractors=(4,18))
    class GRU(nn.Module):
        def __init__(self):
            super().__init__();self.cell=nn.GRU(7,width,batch_first=True)
            self.readout=nn.Linear(width,1)
        def forward(self,x,lengths):
            from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
            packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
            h,_=self.cell(packed)
            padded,_=pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
            return self.readout(padded).squeeze(-1)
    net=GRU()
    optimizer=torch.optim.Adam(net.parameters(),lr=.008)
    rng=random.Random(42)
    for epoch in range(epochs):
        order=list(range(len(train)));rng.shuffle(order)
        net.train()
        for k in range(0,len(order),24):
            x,y,mask,lengths=tensorize([train[i] for i in order[k:k+24]],torch)
            pred=net(x,lengths)
            loss=((pred-y)**2*mask).sum()/mask.sum()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.0)
            optimizer.step()
    short=episodes(88,80,distractors=(4,18))
    # Long test on limited episodes: still 100k recurrent steps, high GRU cost.
    long=episodes(89,12,distractors=(1024,2048))
    return {'parameters':sum(p.numel() for p in net.parameters()),
            'training_episodes':len(train),'epochs':epochs,'short':score(net,short,torch),
            'unseen_delay_1024_2048':score(net,long,torch),
            'limitations':'Larger neural baseline; trained only on short sequences; not parameter- or compute-matched'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/m17/gru.json'))
    p.add_argument('--epochs',type=int,default=25)
    a=p.parse_args();result=run(epochs=a.epochs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
