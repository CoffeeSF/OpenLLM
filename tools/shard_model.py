#!/usr/bin/env python3
"""Create four OCTP tensor-parallel shards from a legacy stories260K bin.

Each shard retains only its Q/K/V rows, SwiGLU channels, reduction columns,
and classifier rows. Norms and embeddings are replicated because they are tiny.
"""
from __future__ import annotations
import argparse, hashlib, math, struct
from pathlib import Path

HEADER = 256

def floats(f, n):
    b=f.read(n*4)
    if len(b)!=n*4: raise ValueError("truncated legacy checkpoint")
    return list(struct.unpack("<%df"%n,b))

def rows(out, values, row_ids, width):
    for row in row_ids:
        src=values[row*width:(row+1)*width]
        peak=max(map(abs,src),default=0.0); scale=peak/127 if peak else 1.0
        q=[max(-127,min(127,math.floor(v/scale+.5) if v>=0 else math.ceil(v/scale-.5))) for v in src]
        out.write(struct.pack("<f",scale)); out.write(struct.pack("<%db"%width,*q))

def selected_columns(values, rows_count, source_width, starts, count):
    # A reduction matrix has all output rows, but only input columns owned here.
    return [values[row*source_width+col] for row in range(rows_count) for col in range(starts,starts+count)]

def main():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("output",type=Path); a=p.parse_args()
    with a.input.open("rb") as f:
        d,h,l,nh,nkv,rv,sl=struct.unpack("<7i",f.read(28)); v=abs(rv); kv=d*nkv//nh
        if (d,h,l,nh,nkv,v)!=(64,172,5,8,4,512) or rv<0: raise ValueError("expected shared-classifier stories260K legacy checkpoint")
        emb=floats(f,v*d); ra=floats(f,l*d); wq=floats(f,l*d*d); wk=floats(f,l*kv*d); wv=floats(f,l*kv*d); wo=floats(f,l*d*d)
        rf=floats(f,l*d); w1=floats(f,l*h*d); w2=floats(f,l*d*h); w3=floats(f,l*h*d); final=floats(f,d)
    a.output.mkdir(parents=True,exist_ok=True)
    # Integer partitions: [0,43), [43,86), [86,129), [129,172).
    for worker in range(4):
        fstart=(h*worker)//4; fend=(h*(worker+1))//4; fc=fend-fstart
        qstart=worker*16; kvstart=worker*8; vocabstart=worker*128
        path=a.output/("shard-%d.bin"%worker)
        with path.open("wb") as out:
            head=struct.pack("<4sI7I7I",b"OCTP",1,worker,fstart,fc,qstart,kvstart,vocabstart,128, d,h,l,nh,nkv,v,sl)
            out.write(head+b"\0"*(HEADER-len(head)))
            out.write(struct.pack("<%df"%len(ra),*ra)); out.write(struct.pack("<%df"%len(rf),*rf)); out.write(struct.pack("<%df"%len(final),*final))
            rows(out,emb,range(v),d)
            for layer in range(l): rows(out,wq[layer*d*d:(layer+1)*d*d],range(qstart,qstart+16),d)
            for layer in range(l): rows(out,wk[layer*kv*d:(layer+1)*kv*d],range(kvstart,kvstart+8),d)
            for layer in range(l): rows(out,wv[layer*kv*d:(layer+1)*kv*d],range(kvstart,kvstart+8),d)
            for layer in range(l):
                part=selected_columns(wo[layer*d*d:(layer+1)*d*d],d,d,qstart,16); rows(out,part,range(d),16)
            for layer in range(l): rows(out,w1[layer*h*d:(layer+1)*h*d],range(fstart,fend),d)
            for layer in range(l):
                part=selected_columns(w2[layer*d*h:(layer+1)*d*h],d,h,fstart,fc); rows(out,part,range(d),fc)
            for layer in range(l): rows(out,w3[layer*h*d:(layer+1)*h*d],range(fstart,fend),d)
            rows(out,emb,range(vocabstart,vocabstart+128),d)
        print(path, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=="__main__": main()
