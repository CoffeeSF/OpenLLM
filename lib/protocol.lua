-- Fixed-port, binary-vector protocol for four-server tensor parallelism.
local storage = require("lib.storage")
local protocol = { port = 27183, tag = "openllm-tp1" }

function protocol.pack_q12(values, count)
  local out = {}
  for i = 1, count do
    local n = math.max(-32768, math.min(32767, math.floor(values[i] * 4096 + (values[i] >= 0 and .5 or -.5))))
    if n < 0 then n = n + 65536 end
    out[i] = string.char(math.floor(n / 256), n % 256)
  end
  return table.concat(out)
end

function protocol.unpack_q12(bytes, count)
  if #bytes ~= count * 2 then return nil, "invalid packed vector length" end
  local out = {}
  for i = 1, count do out[i] = storage.i16be(bytes, (i - 1) * 2 + 1) / 4096 end
  return out
end

function protocol.send(modem, address, session, sequence, kind, layer, position, payload)
  modem.send(address, protocol.port, protocol.tag, session, sequence, kind, layer, position, payload)
end

function protocol.valid(tag, session, sequence, kind, layer, position, payload, wanted)
  return tag == protocol.tag and session == wanted.session and sequence == wanted.sequence
    and kind == wanted.kind and layer == wanted.layer and position == wanted.position
    and type(payload) == "string"
end

return protocol
