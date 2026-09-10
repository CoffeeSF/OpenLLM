-- Disk-backed OCQ8 model reader. It never materializes model weights in Lua.
local storage = require("lib.storage")
local computer = require("computer")
local model = {}

local HEADER_SIZE = 256

local function checked_header(file)
  local header = storage.read_exact(file, HEADER_SIZE)
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

function model.open(path)
  local file, reason = io.open(path, "rb")
  if not file then error("cannot open model: " .. tostring(reason)) end
  local ok, p = pcall(checked_header, file)
  if not ok then file:close(); error(p) end
  local sections, expected = section_offsets(p)
  local actual = file:seek("end")
  if actual < expected then file:close(); error("model.bin is truncated") end
  return setmetatable({ file = file, p = p, sections = sections }, { __index = model })
end

function model:close() if self.file then self.file:close(); self.file = nil end end

function model:norm(kind, layer, out)
  local offset
  if kind == "att" then offset = self.sections.norm_att + layer * self.p.dim * 4
  elseif kind == "ffn" then offset = self.sections.norm_ffn + layer * self.p.dim * 4
  else offset = self.sections.norm_final end
  self.file:seek("set", offset)
  local bytes = storage.read_exact(self.file, self.p.dim * 4)
  for i = 1, self.p.dim do out[i] = storage.f32le(bytes, (i - 1) * 4 + 1) end
end

function model:row(section, row, out)
  local width = section.columns
  if row < 0 or row >= section.rows then error("model row outside section") end
  self.file:seek("set", section.offset + row * (width + 4))
  local scale = storage.read_f32(self.file)
  local values = storage.read_exact(self.file, width)
  for j = 1, width do out[j] = storage.signed_byte(values, j) * scale end
end

function model:matvec(section, row_base, input, output, rows)
  local width = section.columns
  for row = 0, rows - 1 do
    self.file:seek("set", section.offset + (row_base + row) * (width + 4))
    local scale = storage.read_f32(self.file)
    local weights = storage.read_exact(self.file, width)
    local sum = 0
    for col = 1, width do sum = sum + storage.signed_byte(weights, col) * input[col] end
    output[row + 1] = sum * scale
    yield_if_needed(row + 1)
  end
end

return model
