local tensor = require("lib.tensor")
local sampler = {}

function sampler.new(temperature, seed)
  return setmetatable({ temperature = temperature or 0, seed = seed or 123456789 }, { __index = sampler })
end

function sampler:random()
  self.seed = (1664525 * self.seed + 1013904223) % 4294967296
  return self.seed / 4294967296
end

function sampler:sample(logits, count)
  if self.temperature <= 0 then
    local best = 1
    for i = 2, count do if logits[i] > logits[best] then best = i end end
    return best - 1
  end
  for i = 1, count do logits[i] = logits[i] / self.temperature end
  tensor.softmax(logits, count)
  local coin, cumulative = self:random(), 0
  for i = 1, count do
    cumulative = cumulative + logits[i]
    if coin < cumulative then return i - 1 end
  end
  return count - 1
end

return sampler
