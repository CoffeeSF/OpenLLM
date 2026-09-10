-- Autorun entry point for four identical cloned rack disks.
local root=(... ) or "/openllm"; package.path=root.."/?.lua;"..package.path
local filesystem=require("filesystem"); local cluster=require("lib.cluster")
print("OpenLLM rack: discovering four compatible servers...")
local state=cluster.join(30)
local function mkdir(path) if filesystem.exists(path) then return end; local p=filesystem.path(path); if p and p~=path then mkdir(p) end; assert(filesystem.makeDirectory(path)) end
mkdir(root.."/config")
local f=assert(io.open(root.."/config/rack.lua","w")); f:write("return { coordinator = ",string.format("%q",state.coordinator),", workers = { [1] = ",string.format("%q",state.workers[1]),", [2] = ",string.format("%q",state.workers[2]),", [3] = ",string.format("%q",state.workers[3])," }, context = 16, timeout = 30 }\n"); f:close()
print("OpenLLM rack ready: assigned shard "..state.id..(state.id==0 and " (coordinator)" or " (worker)"))
if state.id==0 then return assert(loadfile(root.."/server/coordinator.lua"))(root) end
return assert(loadfile(root.."/server/worker.lua"))(root,state.id)
