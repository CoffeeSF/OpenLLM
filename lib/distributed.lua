-- Worker-local tensor-parallel operations. Coordinator performs reductions.
local tensor=require("lib.tensor"); local protocol=require("lib.protocol")
local D={}
function D.new(model, context)
  local p=model.p; return setmetatable({m=model,p=p,context=context,kc={},vc={},norm=tensor.zeros(64),xb=tensor.zeros(64),q=tensor.zeros(16),k=tensor.zeros(8),v=tensor.zeros(8),att=tensor.zeros(context),hb=tensor.zeros(p.ffn_count),hb2=tensor.zeros(p.ffn_count),out=tensor.zeros(64)}, {__index=D})
end
local function rope(vec,start,pos)
  for i=1,#vec,2 do local hd=(start+i-1)%8; local a=pos/(10000^(hd/8)); local c,s=math.cos(a),math.sin(a); local x,y=vec[i],vec[i+1]; vec[i],vec[i+1]=x*c-y*s end
end
function D:attention(x,layer,pos)
  local m,p=self.m,self.p; m:norm("att",layer,self.norm); tensor.rmsnorm(self.xb,x,self.norm,64); m:matvec(m.s.wq,layer*16,self.xb,self.q,16); m:matvec(m.s.wk,layer*8,self.xb,self.k,8); m:matvec(m.s.wv,layer*8,self.xb,self.v,8); rope(self.q,p.q_start,pos); rope(self.k,p.kv_start,pos)
  self.kc[pos+1]=protocol.pack_q12(self.k,8); self.vc[pos+1]=protocol.pack_q12(self.v,8); local attended={}
  for head=0,1 do
    for t=0,pos do local key=assert(protocol.unpack_q12(self.kc[t+1],8)); local score=0; for i=1,8 do score=score+self.q[head*8+i]*key[i] end; self.att[t+1]=score/math.sqrt(8) end
    tensor.softmax(self.att,pos+1); for i=1,8 do attended[head*8+i]=0 end
    for t=0,pos do local val=assert(protocol.unpack_q12(self.vc[t+1],8)); for i=1,8 do attended[head*8+i]=attended[head*8+i]+self.att[t+1]*val[i] end end
  end
  m:matvec(m.s.wo,layer*64,attended,self.out,64); return self.out
end
function D:ffn(x,layer)
  local m,p=self.m,self.p; m:norm("ffn",layer,self.norm); tensor.rmsnorm(self.xb,x,self.norm,64); m:matvec(m.s.w1,layer*p.ffn_count,self.xb,self.hb,p.ffn_count); m:matvec(m.s.w3,layer*p.ffn_count,self.xb,self.hb2,p.ffn_count)
  for i=1,p.ffn_count do local a=self.hb[i]; self.hb[i]=a/(1+math.exp(-a))*self.hb2[i] end; m:matvec(m.s.w2,layer*64,self.hb,self.out,64); return self.out
end
function D:logits(x)
  local m=self.m; m:matvec(m.s.cls,0,x,self.out,0); local o={}; m:matvec(m.s.cls,0,x,o,128); return o
end
return D
