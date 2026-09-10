-- OCQ8 reader. 2 MiB machines stream rows from disk; 4 MiB machines retain
-- the same compact model bytes in one string to avoid repeated filesystem reads.
local storage = require("lib.storage")
local computer = require("computer")
local model = {}

local HEADER_SIZE = 256
local IN_MEMORY_MIN_RAM = 4096 * 1024

local function checked_header(header)
  if header:sub(1, 4) ~= "OCQ8" then error("model.bin is not an OCQ8 model") end
  local version = storage.u32le(header, 5)
  if version ~= 1 then error("unsupported OCQ8 model version " .. version) end
  local p = {
    dim = storage.u32le(header, 9), hidden_dim = storage.u32le(header, 13),
    n_layers = storage.u32le(header, 17), n_heads = storage.u32le(header, 21),
    n_kv_heads = storage.u32le(header, 25), vocab_size = storage.u32le(header, 29),
    max_seq_len = storage.u32le(header, 33),
  }
  p.shared_classifier = (string.byte(header, 37) or 0) ~= 0
  if p.dim ~= 64 or p.hidden_dim ~= 172 or p.n_layers ~= 5 or p.n_heads ~= 8 or p.n_kv_heads ~= 4 or p.vocab_size ~= 512 then
    error("this runtime is intentionally locked to stories260K OCQ8 dimensions")
  end
  return p
end

local function section_offsets(p)
  local kv_dim = p.dim * p.n_kv_heads / p.n_heads
  local offset = HEADER_SIZE
  local norm_att = offset; offset = offset + p.n_layers * p.dim * 4
  local norm_ffn = offset; offset = offset + p.n_layers * p.dim * 4
  local norm_final = offset; offset = offset + p.dim * 4
  local function matrix(rows, columns)
    local here = offset
    offset = offset + rows * (columns + 4)
    return { offset = here, rows = rows, columns = columns }
  end
  local sections = {
    norm_att = norm_att, norm_ffn = norm_ffn, norm_final = norm_final,
    emb = matrix(p.vocab_size, p.dim),
    wq = matrix(p.n_layers * p.dim, p.dim),
    wk = matrix(p.n_layers * kv_dim, p.dim),
    wv = matrix(p.n_layers * kv_dim, p.dim),
    wo = matrix(p.n_layers * p.dim, p.dim),
    w1 = matrix(p.n_layers * p.hidden_dim, p.dim),
    w2 = matrix(p.n_layers * p.dim, p.hidden_dim),
    w3 = matrix(p.n_layers * p.hidden_dim, p.dim),
  }
  if not p.shared_classifier then sections.cls = matrix(p.vocab_size, p.dim) else sections.cls = sections.emb end
  return sections, offset
end

local function yield_if_needed(index)
  if index % 16 == 0 then computer.pullSignal(0) end
end

local function cached_norms(bytes, p, sections)
  local function vector(offset)
    local out = {}
    for i = 1, p.dim do out[i] = storage.f32le(bytes, offset + (i - 1) * 4 + 1) end
    return out
  end
  local norms = { att = {}, ffn = {}, final = vector(sections.norm_final) }
  for layer = 0, p.n_layers - 1 do
    norms.att[layer] = vector(sections.norm_att + layer * p.dim * 4)
    norms.ffn[layer] = vector(sections.norm_ffn + layer * p.dim * 4)
  end
  return norms
end

local function cached_scales(bytes, sections)
  local scales = {}
  for _, name in ipairs({ "emb", "wq", "wk", "wv", "wo", "w1", "w2", "w3", "cls" }) do
    local section = sections[name]
    if not scales[section] then
      local values = {}
      for row = 0, section.rows - 1 do
        values[row + 1] = storage.f32le(bytes, section.offset + row * (section.columns + 4) + 1)
      end
      scales[section] = values
    end
  end
  return scales
end

function model.open(path)
  local file, reason = io.open(path, "rb")
  if not file then error("cannot open model: " .. tostring(reason)) end
  local header = storage.read_exact(file, HEADER_SIZE)
  local ok, p = pcall(checked_header, header)
  if not ok then file:close(); error(p) end
  local sections, expected = section_offsets(p)
  local actual = file:seek("end")
  if actual < expected then file:close(); error("model.bin is truncated") end
  if computer.totalMemory() >= IN_MEMORY_MIN_RAM then
    file:seek("set", 0)
    local bytes = storage.read_exact(file, actual)
    file:close()
    return setmetatable({ bytes = bytes, p = p, sections = sections, norms = cached_norms(bytes, p, sections), scales = cached_scales(bytes, sections), storage_mode = "in-memory" }, { __index = model })
  end
  return setmetatable({ file = file, p = p, sections = sections, storage_mode = "disk-streamed" }, { __index = model })
end

function model:close()
  if self.file then self.file:close(); self.file = nil end
  self.bytes = nil
  self.norms = nil
  self.scales = nil
end

function model:read(offset, count)
  if self.bytes then return self.bytes:sub(offset + 1, offset + count) end
  self.file:seek("set", offset)
  return storage.read_exact(self.file, count)
end

function model:norm(kind, layer, out)
  if self.norms then
    local values = kind == "att" and self.norms.att[layer] or kind == "ffn" and self.norms.ffn[layer] or self.norms.final
    for i = 1, self.p.dim do out[i] = values[i] end
    return
  end
  local offset
  if kind == "att" then offset = self.sections.norm_att + layer * self.p.dim * 4
  elseif kind == "ffn" then offset = self.sections.norm_ffn + layer * self.p.dim * 4
  else offset = self.sections.norm_final end
  local bytes = self:read(offset, self.p.dim * 4)
  for i = 1, self.p.dim do out[i] = storage.f32le(bytes, (i - 1) * 4 + 1) end
end

function model:row(section, row, out)
  local width = section.columns
  if row < 0 or row >= section.rows then error("model row outside section") end
  if self.bytes then
    local offset = section.offset + row * (width + 4)
    local scale = self.scales[section][row + 1]
    for j = 1, width do out[j] = storage.signed_byte(self.bytes, offset + 4 + j) * scale end
    return
  end
  local bytes = self:read(section.offset + row * (width + 4), width + 4)
  local scale = storage.f32le(bytes, 1)
  local values = bytes:sub(5)
  for j = 1, width do out[j] = storage.signed_byte(values, j) * scale end
end

function model:matvec(section, row_base, input, output, rows)
  local width = section.columns
  if self.bytes then
    for row = 0, rows - 1 do
      local offset = section.offset + (row_base + row) * (width + 4)
      local scale = self.scales[section][row_base + row + 1]
      local sum = 0
      for col = 1, width do sum = sum + storage.signed_byte(self.bytes, offset + 4 + col) * input[col] end
      output[row + 1] = sum * scale
      yield_if_needed(row + 1)
    end
    return
  end
  for row = 0, rows - 1 do
    local bytes = self:read(section.offset + (row_base + row) * (width + 4), width + 4)
    local scale = storage.f32le(bytes, 1)
    local weights = bytes:sub(5)
    local sum = 0
    for col = 1, width do sum = sum + storage.signed_byte(weights, col) * input[col] end
    output[row + 1] = sum * scale
    yield_if_needed(row + 1)
  end
end

return model
