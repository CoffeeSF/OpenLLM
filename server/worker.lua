local raw={...}; local root=raw[1] or "/openllm"; package.path=root.."/?.lua;"..package.path
local component=require("component"); local event=require("event"); local shell=require("shell")
local protocol=require("lib.protocol"); local Shard=require("lib.distributed_model"); local Engine=require("lib.distributed")
local args=shell.parse(...); root=args[1] or root; local id=tonumber(args[2]); if not id or id<1 or id>3 then error("usage: worker <root> <1|2|3>") end
local cfg=assert(loadfile(root.."/config/rack.lua"))(); local modem=assert(component.modem,"network card required"); modem.open(protocol.port)
local engine=Engine.new(Shard.open(root.."/model/shard-"..id..".bin"),cfg.context or 128)
while true do
  local _,_,from,port,_,tag,session,seq,kind,layer,pos,payload=event.pull("modem_message")
  if port==protocol.port and tag==protocol.tag and from==cfg.coordinator and type(payload)=="string" then
    local x=protocol.unpack_q12(payload,64); local result
    if kind=="att" then result=engine:attention(x,layer,pos); protocol.send(modem,from,session,seq,"att-result",layer,pos,protocol.pack_q12(result,64))
    elseif kind=="ffn" then result=engine:ffn(x,layer,pos); protocol.send(modem,from,session,seq,"ffn-result",layer,pos,protocol.pack_q12(result,64))
    elseif kind=="logits" then result=engine:logits(x); protocol.send(modem,from,session,seq,"logits-result",layer,pos,protocol.pack_q12(result,128))
    elseif kind=="stop" then break end
  end
end
