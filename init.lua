-- Interactive entry point. Run with: openllm [model-root]
local shell = require("shell")
local llm = require("lib.llm")

local arguments, options = shell.parse(...)
local root = arguments[1] or "/openllm"
local context = tonumber(options.context)
local temperature = tonumber(options.temperature) or 0

local function open_runtime()
  return llm.new(root, { context = context, temperature = temperature })
end

local runtime = open_runtime()
context = runtime.context
io.write("OpenLLM stories260K (local INT8)\n")
io.write("context=" .. context .. "  temperature=" .. temperature .. "\n")
io.write("model storage=" .. runtime.model.storage_mode .. "  KV cache=" .. runtime.cache_mode .. "\n")
io.write("Enter a short story beginning. Commands: /quit, /temp <number>, /context <2-512> (default temperature 0)\n\n")
while true do
  io.write("> ")
  local prompt = io.read("l")
  if not prompt then break end
  if prompt == "/quit" or prompt == "/exit" then break end
  local temp = prompt:match("^/temp%s+([%d%.]+)$")
  local next_context = prompt:match("^/context%s+(%d+)$")
  if temp then
    temperature = tonumber(temp)
    runtime:close(); runtime = open_runtime()
    print("temperature=" .. temperature)
  elseif next_context then
    context = math.max(2, math.min(512, tonumber(next_context)))
    runtime:close(); runtime = open_runtime()
    print("context=" .. context)
  elseif #prompt > 0 then
    local ok, reason = pcall(function()
      runtime:generate(prompt, context, function(piece) io.write(piece) end)
    end)
    io.write("\n\n")
    if not ok then print("Cannot generate: " .. tostring(reason)) end
  end
end
runtime:close()
