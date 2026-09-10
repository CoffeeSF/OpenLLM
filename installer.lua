-- OpenLLM installer for OpenComputers. Requires an Internet Card only here.
local component = require("component")
local computer = require("computer")
local filesystem = require("filesystem")
local shell = require("shell")
local internet = require("internet")

local REQUIRED_RAM = 2048 * 1024
local REQUIRED_STORAGE = 1024 * 1024 -- model is 277 KiB; leave room for OS and cache
local DEFAULT_BASE = "https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main"
local FILES = {
  "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "init.lua", "openllm.lua",
  "lib/storage.lua", "lib/model.lua", "lib/tokenizer.lua", "lib/tensor.lua", "lib/sampler.lua", "lib/llm.lua",
  "lib/protocol.lua", "lib/distributed_model.lua", "lib/distributed.lua",
  "server/coordinator.lua", "server/worker.lua", "config/rack.lua",
  "model/config.lua", "model/model.bin", "model/model.bin.sha256", "model/tokenizer.bin",
  "model/shard-0.bin", "model/shard-1.bin", "model/shard-2.bin", "model/shard-3.bin",
}

local function mkdir_p(path)
  if filesystem.exists(path) then return end
  local parent = filesystem.path(path)
  if parent and parent ~= path and parent ~= "" then mkdir_p(parent) end
  assert(filesystem.makeDirectory(path))
end

local function space_available(fs)
  -- Component filesystem proxies expose spaceTotal/spaceUsed. The OpenOS
  -- filesystem library offers spaceAvailable(path), but proxies usually do not.
  if fs.spaceAvailable then return fs.spaceAvailable() end
  return fs.spaceTotal() - fs.spaceUsed()
end

local function report_filesystems()
  print("Detected filesystems:")
  for address in component.list("filesystem") do
    local fs = component.proxy(address)
    local total = fs.spaceTotal and fs.spaceTotal() or 0
    local free = space_available(fs)
    print("  " .. address:sub(1, 8) .. "  " .. math.floor(total / 1024) .. " KiB total, " .. math.floor(free / 1024) .. " KiB free")
  end
end

local function download(url, destination)
  local response, reason = internet.request(url)
  if not response then error("download failed: " .. tostring(reason) .. "\n" .. url) end
  local out, out_reason = io.open(destination, "wb")
  if not out then error("cannot write " .. destination .. ": " .. tostring(out_reason)) end
  for chunk in response do out:write(chunk) end
  out:close()
  local file = io.open(destination, "rb")
  local size = file and file:seek("end") or 0
  if file then file:close() end
  if size == 0 then error("download was empty: " .. url) end
end

local arguments = shell.parse(...)
local destination = arguments[1] or "/openllm"
local base = (arguments[2] or DEFAULT_BASE):gsub("/$", "")

local ram = computer.totalMemory()
print("OpenLLM installer")
print("RAM: " .. math.floor(ram / 1024) .. " KiB (requires 2048 KiB)")
if ram < REQUIRED_RAM then error("not enough installed RAM") end
report_filesystems()
local target_fs = filesystem.get(destination)
if not target_fs then error("cannot find filesystem for " .. destination) end
local available = space_available(target_fs)
print("Destination: " .. destination .. " (" .. math.floor(available / 1024) .. " KiB free)")
if available < REQUIRED_STORAGE then error("destination needs at least " .. REQUIRED_STORAGE / 1024 .. " KiB free") end

mkdir_p(destination)
for _, relative in ipairs(FILES) do
  local path = destination .. "/" .. relative
  mkdir_p(filesystem.path(path))
  io.write("Downloading " .. relative .. " ... ")
  download(base .. "/" .. relative, path)
  print("ok")
end

local model = assert(io.open(destination .. "/model/model.bin", "rb"))
local model_size = model:seek("end"); model:close()
local tokenizer = assert(io.open(destination .. "/model/tokenizer.bin", "rb"))
local tokenizer_size = tokenizer:seek("end"); tokenizer:close()
if model_size ~= 276448 or tokenizer_size ~= 6227 then
  error("downloaded model asset has an unexpected size; do not run it")
end
mkdir_p(destination .. "/cache")

local launcher = "/bin/openllm.lua"
local replace_launcher = not filesystem.exists(launcher)
if not replace_launcher then
  local prior = io.open(launcher, "r")
  local prior_text = prior and prior:read("*a") or ""
  if prior then prior:close() end
  -- Safely upgrade a launcher made by an earlier OpenLLM installer, but never
  -- overwrite an unrelated user command with the same name.
  replace_launcher = prior_text:find("loadfile", 1, true) ~= nil and prior_text:find("/init.lua", 1, true) ~= nil
end
if replace_launcher then
  local out = assert(io.open(launcher, "w"))
  out:write("-- OpenLLM launcher\nlocal root = " .. string.format("%q", destination) .. "; package.path = root .. '/?.lua;' .. package.path; return assert(loadfile(root .. '/init.lua'))(root, ...)\n")
  out:close()
  print("Installed command: openllm")
else
  print("Did not replace existing " .. launcher .. "; start with: " .. destination .. "/openllm.lua " .. destination)
end
print("Installation complete. Remove the Internet Card if desired; inference is fully local.")
print("Start with: openllm  (or: " .. destination .. "/openllm.lua " .. destination .. ")")
