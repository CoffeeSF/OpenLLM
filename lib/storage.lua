-- Small binary and filesystem helpers for the OpenComputers runtime.
local storage = {}

local function byte(s, i) return string.byte(s, i) or 0 end

function storage.u32le(s, i)
  i = i or 1
  return byte(s, i) + byte(s, i + 1) * 256 + byte(s, i + 2) * 65536 + byte(s, i + 3) * 16777216
end

function storage.i16be(s, i)
  local n = byte(s, i) * 256 + byte(s, i + 1)
  return n >= 32768 and n - 65536 or n
end

-- IEEE-754 float32 decode without string.unpack (OpenComputers uses Lua 5.2).
function storage.f32le(s, i)
  local bits = storage.u32le(s, i)
  local sign = bits >= 2147483648 and -1 or 1
  if bits >= 2147483648 then bits = bits - 2147483648 end
  local exponent = math.floor(bits / 8388608)
  local fraction = bits - exponent * 8388608
  if exponent == 255 then return sign * math.huge end
  if exponent == 0 then return sign * fraction * 2^-149 end
  return sign * (1 + fraction / 8388608) * 2^(exponent - 127)
end

function storage.signed_byte(s, i)
  local n = byte(s, i)
  return n >= 128 and n - 256 or n
end

function storage.read_exact(file, bytes)
  local value = file:read(bytes)
  if not value or #value ~= bytes then error("unexpected end of binary file") end
  return value
end

function storage.read_f32(file)
  return storage.f32le(storage.read_exact(file, 4))
end

function storage.mkdir_p(filesystem, path)
  if filesystem.exists(path) then return end
  local parent = filesystem.path(path)
  if parent and parent ~= path and parent ~= "" then storage.mkdir_p(filesystem, parent) end
  local ok, reason = filesystem.makeDirectory(path)
  if not ok and not filesystem.exists(path) then error("cannot create " .. path .. ": " .. tostring(reason)) end
end

function storage.write_i16be(file, number)
  number = math.max(-32768, math.min(32767, math.floor(number + (number >= 0 and 0.5 or -0.5))))
  if number < 0 then number = number + 65536 end
  file:write(string.char(math.floor(number / 256), number % 256))
end

return storage
