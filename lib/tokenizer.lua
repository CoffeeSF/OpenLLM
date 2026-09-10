-- Tiny BPE tokenizer compatible with the tokenizer.bin exported by llama2.c.
local storage = require("lib.storage")
local tokenizer = {}

function tokenizer.load(path, vocab_size)
  local file, reason = io.open(path, "rb")
  if not file then error("cannot open tokenizer: " .. tostring(reason)) end
  local maximum = storage.u32le(storage.read_exact(file, 4))
  local self = { vocab = {}, score = {}, lookup = {}, maximum = maximum, vocab_size = vocab_size }
  for i = 1, vocab_size do
    self.score[i] = storage.read_f32(file)
    local length = storage.u32le(storage.read_exact(file, 4))
    self.vocab[i] = storage.read_exact(file, length)
    self.lookup[self.vocab[i]] = i
  end
  file:close()
  return setmetatable(self, { __index = tokenizer })
end

function tokenizer:encode(text)
  local tokens = { 2 } -- Lua index 2 is model token id 1: BOS.
  local space = self.lookup[" "]
  if space then tokens[#tokens + 1] = space end
  for i = 1, #text do
    local character = text:sub(i, i)
    local id = self.lookup[character]
    if not id then id = string.byte(character) + 4 end -- model token id byte+3, Lua index +1
    if id < 1 or id > self.vocab_size then error("prompt contains an unsupported byte") end
    tokens[#tokens + 1] = id
  end
  while true do
    local best_score, best_at, best_id = -math.huge, nil, nil
    for i = 1, #tokens - 1 do
      local piece = self.vocab[tokens[i]] .. self.vocab[tokens[i + 1]]
      local id = self.lookup[piece]
      if id and self.score[id] > best_score then best_score, best_at, best_id = self.score[id], i, id end
    end
    if not best_at then break end
    tokens[best_at] = best_id
    table.remove(tokens, best_at + 1)
  end
  for i = 1, #tokens do tokens[i] = tokens[i] - 1 end -- switch to model token ids
  return tokens
end

function tokenizer:decode(previous, token)
  local piece = self.vocab[token + 1] or ""
  if previous == 1 and piece:sub(1, 1) == " " then piece = piece:sub(2) end
  local hex = piece:match("^<0x(%x%x)>$")
  return hex and string.char(tonumber(hex, 16)) or piece
end

return tokenizer
