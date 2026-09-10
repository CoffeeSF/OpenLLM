-- Local, disk-streamed INT8 stories260K forward pass for OpenComputers.
local filesystem = require("filesystem")
local storage = require("lib.storage")
local model_reader = require("lib.model")
local tensor = require("lib.tensor")
local sampler = require("lib.sampler")
local tokenizer = require("lib.tokenizer")
local computer = require("computer")

local llm = {}

local function cache_path(root, layer) return root .. "/kv-" .. layer .. ".bin" end

function llm.new(root, options)
  options = options or {}
  local self = setmetatable({}, { __index = llm })
  self.root = root
  self.model = model_reader.open(root .. "/model/model.bin")
  self.p = self.model.p
  self.context = math.min(options.context or 16, 64, self.p.max_seq_len)
  if self.context < 2 then error("context must be at least 2 tokens") end
  self.tok = tokenizer.load(root .. "/model/tokenizer.bin", self.p.vocab_size)
  self.sample = sampler.new(options.temperature or 0.7, options.seed or math.floor(computer.uptime() * 1000) + 1)
  self.cache_root = root .. "/cache"
  storage.mkdir_p(filesystem, self.cache_root)
  local p = self.p
  self.x, self.xb, self.xb2 = tensor.zeros(p.dim), tensor.zeros(p.dim), tensor.zeros(p.dim)
  self.q, self.k, self.v = tensor.zeros(p.dim), tensor.zeros(p.dim * p.n_kv_heads / p.n_heads), tensor.zeros(p.dim * p.n_kv_heads / p.n_heads)
  self.hb, self.hb2 = tensor.zeros(p.hidden_dim), tensor.zeros(p.hidden_dim)
  self.norm = tensor.zeros(p.dim)
  self.att = tensor.zeros(self.context)
  self.logits = tensor.zeros(p.vocab_size)
  self.row = tensor.zeros(p.hidden_dim)
  return self
end

function llm:close() self.model:close() end

function llm:reset()
  for layer = 0, self.p.n_layers - 1 do
    local file = io.open(cache_path(self.cache_root, layer), "wb")
    if not file then error("cannot reset KV cache; model directory must be writable") end
    file:close()
  end
end

local function cache_vector(file, vector, count)
  for i = 1, count do storage.write_i16be(file, vector[i] * 4096) end
end

local function read_cache_vector(bytes, offset, target, target_offset, count)
  for i = 1, count do target[target_offset + i] = storage.i16be(bytes, offset + (i - 1) * 2) / 4096 end
end

function llm:forward(token, position)
  local p, m = self.p, self.model
  local dim, kv_dim, head_size = p.dim, #self.k, p.dim / p.n_heads
  m:row(m.sections.emb, token, self.x)
  for layer = 0, p.n_layers - 1 do
    m:norm("att", layer, self.norm)
    tensor.rmsnorm(self.xb, self.x, self.norm, dim)
    m:matvec(m.sections.wq, layer * dim, self.xb, self.q, dim)
    m:matvec(m.sections.wk, layer * kv_dim, self.xb, self.k, kv_dim)
    m:matvec(m.sections.wv, layer * kv_dim, self.xb, self.v, kv_dim)

    -- RoPE, computed rather than retaining the source checkpoint's lookup table.
    for i = 1, dim, 2 do
      local head_dim = (i - 1) % head_size
      local angle = position / (10000 ^ (head_dim / head_size))
      local cosine, sine = math.cos(angle), math.sin(angle)
      local a, b = self.q[i], self.q[i + 1]
      self.q[i], self.q[i + 1] = a * cosine - b * sine, a * sine + b * cosine
      if i <= kv_dim then
        a, b = self.k[i], self.k[i + 1]
        self.k[i], self.k[i + 1] = a * cosine - b * sine, a * sine + b * cosine
      end
    end

    local path = cache_path(self.cache_root, layer)
    local append = assert(io.open(path, "ab"))
    cache_vector(append, self.k, kv_dim); cache_vector(append, self.v, kv_dim)
    append:close()
    local cache = assert(io.open(path, "rb"))
    local cache_bytes = storage.read_exact(cache, (position + 1) * kv_dim * 4)
    cache:close()

    for head = 0, p.n_heads - 1 do
      local q_offset = head * head_size
      local kv_offset = math.floor(head / (p.n_heads / p.n_kv_heads)) * head_size
      for t = 0, position do
        local base = t * kv_dim * 4 + kv_offset * 2 + 1
        local score = 0
        for i = 1, head_size do score = score + self.q[q_offset + i] * storage.i16be(cache_bytes, base + (i - 1) * 2) / 4096 end
        self.att[t + 1] = score / math.sqrt(head_size)
      end
      tensor.softmax(self.att, position + 1)
      for i = 1, head_size do self.xb[q_offset + i] = 0 end
      for t = 0, position do
        local base = t * kv_dim * 4 + kv_dim * 2 + kv_offset * 2 + 1
        local probability = self.att[t + 1]
        for i = 1, head_size do self.xb[q_offset + i] = self.xb[q_offset + i] + probability * storage.i16be(cache_bytes, base + (i - 1) * 2) / 4096 end
      end
      computer.pullSignal(0)
    end
    m:matvec(m.sections.wo, layer * dim, self.xb, self.xb2, dim)
    for i = 1, dim do self.x[i] = self.x[i] + self.xb2[i] end

    m:norm("ffn", layer, self.norm)
    tensor.rmsnorm(self.xb, self.x, self.norm, dim)
    m:matvec(m.sections.w1, layer * p.hidden_dim, self.xb, self.hb, p.hidden_dim)
    m:matvec(m.sections.w3, layer * p.hidden_dim, self.xb, self.hb2, p.hidden_dim)
    for i = 1, p.hidden_dim do
      local value = self.hb[i]
      self.hb[i] = value / (1 + math.exp(-value)) * self.hb2[i]
    end
    m:matvec(m.sections.w2, layer * dim, self.hb, self.xb, dim)
    for i = 1, dim do self.x[i] = self.x[i] + self.xb[i] end
    computer.pullSignal(0)
  end
  m:norm("final", 0, self.norm)
  tensor.rmsnorm(self.x, self.x, self.norm, dim)
  m:matvec(m.sections.cls, 0, self.x, self.logits, p.vocab_size)
  return self.logits
end

function llm:generate(prompt, maximum, emit)
  local tokens = self.tok:encode(prompt)
  if #tokens > self.context then
    error("prompt is " .. #tokens .. " tokens; this build supports at most " .. self.context .. " including BOS")
  end
  maximum = math.min(maximum or 32, self.context - #tokens)
  self:reset()
  local previous, logits = tokens[1], nil
  for position = 0, #tokens - 1 do
    logits = self:forward(previous, position)
    previous = tokens[position + 2] or self.sample:sample(logits, self.p.vocab_size)
  end
  local generated = {}
  local last_token = tokens[#tokens]
  for _ = 1, maximum do
    if previous == 2 then break end -- EOS
    generated[#generated + 1] = previous
    if emit then emit(self.tok:decode(last_token, previous)) end
    logits = self:forward(previous, #tokens + #generated - 1)
    last_token = previous
    previous = self.sample:sample(logits, self.p.vocab_size)
    computer.pullSignal(0)
  end
  return generated
end

return llm
