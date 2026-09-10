-- Reader for one OCTP tensor-parallel shard. Weights remain packed strings.
local storage = require("lib.storage")
local M = {}; local HEADER = 256
local function u(s,i) return storage.u32le(s,i) end
local function section(offset, rows, columns) return {offset=offset, rows=rows, columns=columns}, offset+rows*(columns+4) end

function M.open(path)
  local f, why = io.open(path, "rb"); if not f then error("cannot open shard "..path..": "..tostring(why)) end
  local data=f:read("*a"); f:close(); local h=data:sub(1,HEADER); if h:sub(1,4)~="OCTP" or u(h,5)~=1 then error("invalid OCTP shard") end
  local p={id=u(h,9), ffn_start=u(h,13), ffn_count=u(h,17), q_start=u(h,21), kv_start=u(h,25), vocab_start=u(h,29), vocab_count=u(h,33), dim=u(h,37), hidden_dim=u(h,41), n_layers=u(h,45), n_heads=u(h,49), n_kv_heads=u(h,53), vocab_size=u(h,57), max_seq_len=u(h,61)}
  if p.dim~=64 or p.hidden_dim~=172 or p.n_layers~=5 or p.vocab_count~=128 then error("unsupported shard dimensions") end
  local o=HEADER; local att=o; o=o+p.n_layers*p.dim*4; local ffn=o; o=o+p.n_layers*p.dim*4; local final=o; o=o+p.dim*4
  local s={norm_att=att,norm_ffn=ffn,norm_final=final}; s.emb,o=section(o,512,64); s.wq,o=section(o,80,64); s.wk,o=section(o,40,64); s.wv,o=section(o,40,64); s.wo,o=section(o,320,16); s.w1,o=section(o,p.n_layers*p.ffn_count,64); s.w2,o=section(o,320,p.ffn_count); s.w3,o=section(o,p.n_layers*p.ffn_count,64); s.cls,o=section(o,128,64)
  if #data<o then error("truncated shard") end
  return setmetatable({data=data,p=p,s=s},{__index=M})
end
function M:close() end
function M:norm(kind, layer, out)
  local o=kind=="att" and self.s.norm_att+layer*256 or kind=="ffn" and self.s.norm_ffn+layer*256 or self.s.norm_final
  for i=1,64 do out[i]=storage.f32le(self.data,o+(i-1)*4+1) end
end
function M:matvec(sec, row_base, input, output, count)
  local w=sec.columns
  for r=0,count-1 do
    local o=sec.offset+(row_base+r)*(w+4); local scale=storage.f32le(self.data,o+1); local sum=0
    for c=1,w do sum=sum+storage.signed_byte(self.data,o+4+c)*input[c] end
    output[r+1]=sum*scale
  end
end
function M:embedding(token,out) local o=self.s.emb.offset+token*68; local sc=storage.f32le(self.data,o+1); for i=1,64 do out[i]=storage.signed_byte(self.data,o+4+i)*sc end end
return M
