-- Fixed-size numeric working buffers. Only activations live in Lua tables.
local tensor = {}

function tensor.zeros(size)
  local out = {}
  for i = 1, size do out[i] = 0 end
  return out
end

function tensor.copy(out, input, size)
  for i = 1, size do out[i] = input[i] end
end

function tensor.rmsnorm(out, input, weight, size)
  local sum = 0
  for i = 1, size do sum = sum + input[i] * input[i] end
  local scale = 1 / math.sqrt(sum / size + 1e-5)
  for i = 1, size do out[i] = weight[i] * input[i] * scale end
end

function tensor.softmax(values, count)
  local maximum = values[1]
  for i = 2, count do if values[i] > maximum then maximum = values[i] end end
  local sum = 0
  for i = 1, count do values[i] = math.exp(values[i] - maximum); sum = sum + values[i] end
  for i = 1, count do values[i] = values[i] / sum end
end

return tensor
