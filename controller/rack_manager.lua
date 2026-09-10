-- Internet-connected case controller. Requires Internet + Network Card.
local component=require("component"); local event=require("event"); local filesystem=require("filesystem"); local internet=require("internet")
local root=(... ) or "/openllm-controller"; package.path=root.."/?.lua;"..package.path
local cfg=assert(loadfile(root.."/config/gateway.lua"))(); local modem=assert(component.modem,"controller needs a Network Card"); local PORT=27184; local TAG="openllm-gateway-1"; modem.open(PORT)
local session=tostring(math.floor(require("computer").uptime()*100000)); local sequence=0
local function mkdir(path) if filesystem.exists(path) then return end; local p=filesystem.path(path); if p and p~=path then mkdir(p) end; assert(filesystem.makeDirectory(path)) end
local function fetch(relative)
  local h,why=internet.request(cfg.release.."/"..relative); if not h then error("download failed "..relative..": "..tostring(why)) end
  local chunks={}; for chunk in h do chunks[#chunks+1]=chunk end; local data=table.concat(chunks); if #data==0 then error("empty download "..relative) end; return data
end
local function wait(address,wanted)
  local deadline=require("computer").uptime()+cfg.timeout
  while true do local left=deadline-require("computer").uptime(); if left<=0 then error("timeout waiting for "..address.." ("..wanted..")") end
    local _,_,from,port,_,tag,s,k,detail=event.pull(left,"modem_message")
    if from==address and port==PORT and tag==TAG and s==session then if k=="error" then error(detail) end; if k==wanted then return detail end end
  end
end
local function send(address,kind,a,b,c) modem.send(address,PORT,TAG,session,kind,a,b,c) end
local function put(address,path,data)
  send(address,"begin",path); wait(address,"ready"); local size=4096
  for at=1,#data,size do local index=(at-1)/size; send(address,"chunk",path,index,data:sub(at,at+size-1)); wait(address,"ack") end
  send(address,"finish",path); wait(address,"done")
end
local common={"lib/storage.lua","lib/tensor.lua","lib/tokenizer.lua","lib/sampler.lua","lib/protocol.lua","lib/distributed_model.lua","lib/distributed.lua","server/worker.lua","server/coordinator.lua","model/tokenizer.bin"}
local function rack_config()
  return "return { coordinator = "..string.format("%q",cfg.coordinator)..", workers = { [1] = "..string.format("%q",cfg.workers[1])..", [2] = "..string.format("%q",cfg.workers[2])..", [3] = "..string.format("%q",cfg.workers[3]).." }, context = 128, timeout = "..cfg.timeout.." }\n"
end
local function install(address,id)
  print("Installing server "..id); for _,file in ipairs(common) do put(address,cfg.install_root.."/"..file,fetch(file)) end
  put(address,cfg.install_root.."/model/shard-"..id..".bin",fetch("model/shard-"..id..".bin")); put(address,cfg.install_root.."/config/rack.lua",rack_config()); print("Server "..id.." ready")
end
local command=(select(2,...) or "install")
if command=="install" then install(cfg.coordinator,0); for i=1,3 do install(cfg.workers[i],i) end; print("All rack servers installed. Run with: lua controller/rack_manager.lua "..root.." start")
elseif command=="start" then for i=1,3 do send(cfg.workers[i],"launch",cfg.install_root.."/server/worker.lua",cfg.install_root,i); wait(cfg.workers[i],"launched") end; send(cfg.coordinator,"launch",cfg.install_root.."/server/coordinator.lua",cfg.install_root,""); wait(cfg.coordinator,"launched"); print("Rack coordinator started")
elseif command=="status" then for i=1,3 do send(cfg.workers[i],"status"); print(i..": "..wait(cfg.workers[i],"status")) end
else error("usage: rack_manager.lua <root> [install|start|status]") end
