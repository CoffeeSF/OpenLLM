local raw={...}; local root=raw[1] or "/openllm"; package.path=root.."/?.lua;"..package.path
local component=require("component"); local event=require("event"); local shell=require("shell"); local computer=require("computer")
local protocol=require("lib.protocol"); local Shard=require("lib.distributed_model"); local Engine=require("lib.distributed"); local tensor=require("lib.tensor"); local Tok=require("lib.tokenizer"); local Sampler=require("lib.sampler")
local args,opts=shell.parse(...); root=args[1] or root; local cfg=assert(loadfile(root.."/config/rack.lua"))(); local context=tonumber(opts.context) or cfg.context or 16
local modem=assert(component.modem,"network card required"); modem.open(protocol.port); local m=Shard.open(root.."/model/shard-0.bin"); local e=Engine.new(m,context); local tok=Tok.load(root.."/model/tokenizer.bin",512); local sampler=Sampler.new(tonumber(opts.temperature) or .7,math.floor(computer.uptime()*1000)); local norm=tensor.zeros(64); local session=tostring(math.floor(computer.uptime()*100000)); local seq=0
local function sum(parts) local out=tensor.zeros(64); for _,v in pairs(parts) do for i=1,64 do out[i]=out[i]+v[i] end end; return out end
local function remote(kind,layer,pos,x)
  seq=seq+1; local wanted={session=session,sequence=seq,kind=kind.."-result",layer=layer,position=pos}; local payload=protocol.pack_q12(x,64); for id=1,3 do assert(cfg.workers[id],"missing worker address "..id); protocol.send(modem,cfg.workers[id],session,seq,kind,layer,pos,payload) end
  local got={}; local deadline=computer.uptime()+cfg.timeout
  while #got<3 do local left=deadline-computer.uptime(); if left<=0 then error("worker timeout during "..kind.." layer "..layer) end; local _,_,from,port,_,tag,sq,number,k,l,p,data=event.pull(left,"modem_message"); if port==protocol.port and protocol.valid(tag,sq,number,k,l,p,data,wanted) then for id=1,3 do if from==cfg.workers[id] and not got[id] then local v,why=protocol.unpack_q12(data,kind=="logits" and 128 or 64); assert(v,why); got[id]=v end end end end
  return got
end
local function forward(token,pos)
  local x=tensor.zeros(64); m:embedding(token,x)
  for l=0,4 do local r=remote("att",l,pos,x); r[0]=e:attention(x,l,pos); local a=sum(r); for i=1,64 do x[i]=x[i]+a[i] end; r=remote("ffn",l,pos,x); r[0]=e:ffn(x,l); local f=sum(r); for i=1,64 do x[i]=x[i]+f[i] end end
  m:norm("final",0,norm); tensor.rmsnorm(x,x,norm,64); local r=remote("logits",0,pos,x); r[0]=e:logits(x); local logits={}; for id=0,3 do for i=1,128 do logits[id*128+i]=r[id][i] end end; return logits
end
print("OpenLLM rack coordinator; context="..context); while true do io.write("> "); local prompt=io.read("l"); if not prompt or prompt=="/quit" then break end; local tokens=tok:encode(prompt); if #tokens>context then print("prompt too long") else local prev=tokens[1]; for pos=0,#tokens-1 do local logits=forward(prev,pos); prev=tokens[pos+2] or sampler:sample(logits,512) end; local last=tokens[#tokens]; for n=1,context-#tokens do if prev==2 then break end; io.write(tok:decode(last,prev)); local logits=forward(prev,#tokens+n-1); last=prev; prev=sampler:sample(logits,512) end; print() end end
