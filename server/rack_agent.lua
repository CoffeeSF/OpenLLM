-- One-time bootstrap agent for a rack server. It receives files only from the
-- configured controller and then can launch a worker or coordinator remotely.
local component=require("component"); local event=require("event"); local filesystem=require("filesystem")
local modem=assert(component.modem,"rack agent requires a Network Card"); local PORT=27184
local controller="REPLACE_WITH_CONTROLLER_MODEM_ADDRESS" -- set before first run
local files={}
local function mkdir(path)
  if filesystem.exists(path) then return end
  local parent=filesystem.path(path); if parent and parent~=path then mkdir(parent) end
  assert(filesystem.makeDirectory(path))
end
local function reply(address,session,kind,detail) modem.send(address,PORT,"openllm-gateway-1",session,kind,detail or "") end
modem.open(PORT); print("OpenLLM rack agent listening on "..modem.address())
while true do
  local _,_,from,port,_,tag,session,kind,a,b,c=event.pull("modem_message")
  if port==PORT and tag=="openllm-gateway-1" and from==controller then
    local ok,why=pcall(function()
      if kind=="ping" then reply(from,session,"pong",modem.address())
      elseif kind=="begin" then
        local path=a; if type(path)~="string" or path:sub(1,1)~="/" then error("invalid destination") end
        mkdir(filesystem.path(path)); local f=assert(io.open(path,"wb")); f:close(); files[path]={next=0}; reply(from,session,"ready",path)
      elseif kind=="chunk" then
        local path,index,data=a,b,c; local state=files[path]; if not state or type(index)~="number" or index~=state.next or type(data)~="string" then error("invalid file chunk") end
        local f=assert(io.open(path,"ab")); f:write(data); f:close(); state.next=index+1; reply(from,session,"ack",path..":"..index)
      elseif kind=="finish" then local state=files[a]; if not state then error("unknown file") end; files[a]=nil; reply(from,session,"done",a)
      elseif kind=="launch" then
        reply(from,session,"launched",a); local entry=assert(loadfile(a)); return entry(b,c)
      elseif kind=="status" then reply(from,session,"status","agent-ready")
      else error("unknown command") end
    end)
    if not ok then reply(from,session,"error",tostring(why)) end
  end
end
