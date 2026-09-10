-- Convenience launcher installed beside init.lua.
local shell = require("shell")
local arguments = shell.parse(...)
local root = arguments[1] or "/openllm"
package.path = root .. "/?.lua;" .. package.path
local entry, reason = loadfile(root .. "/init.lua")
if not entry then error(reason) end
return entry(root, ...)
