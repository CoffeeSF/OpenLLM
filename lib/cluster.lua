-- Clone-safe OpenLLM rack discovery. Identity is the immutable computer address,
-- while modem addresses are used only for transport.
local component=require("component"); local computer=require("computer"); local event=require("event")
local C={tag="openllm-cluster",version=1,port=27185,model="stories260K-OCTP-v1"}
local function sort_keys(t) local a={}; for k in pairs(t) do a[#a+1]=k end; table.sort(a); return a end
function C.join(timeout)
  local modem=assert(component.modem,"OpenLLM rack mode needs a Network Card"); modem.open(C.port)
  local mine=computer.address(); local members={[mine]={computer=mine,modem=modem.address,model=C.model}}; local deadline=computer.uptime()+(timeout or 30); local next_hello=0
  while computer.uptime()<deadline do
    if computer.uptime()>=next_hello then modem.broadcast(C.port,C.tag,C.version,"hello",mine,modem.address,C.model); next_hello=computer.uptime()+2 end
    local _,_,from,port,_,tag,version,kind,caddr,maddr,model=event.pull(math.min(1,deadline-computer.uptime()),"modem_message")
    if port==C.port and tag==C.tag and version==C.version and kind=="hello" and type(caddr)=="string" and type(maddr)=="string" and model==C.model and from==maddr then
      members[caddr]={computer=caddr,modem=maddr,model=model}
    end
    local keys=sort_keys(members)
    if #keys==4 then
      -- One more hello period rejects an accidental fifth compatible node.
      local settle=computer.uptime()+2
      while computer.uptime()<settle do
        local _,_,from2,p2,_,t2,v2,k2,c2,m2,mo2=event.pull(settle-computer.uptime(),"modem_message")
        if p2==C.port and t2==C.tag and v2==C.version and k2=="hello" and mo2==C.model and from2==m2 then members[c2]={computer=c2,modem=m2,model=mo2} end
      end
      keys=sort_keys(members); if #keys==4 then
        local index; for i=1,4 do if keys[i]==mine then index=i-1 end end
        return {id=index, coordinator=members[keys[1]].modem, workers={[1]=members[keys[2]].modem,[2]=members[keys[3]].modem,[3]=members[keys[4]].modem}, members=members}
      end
    end
  end
  local count=#sort_keys(members); error("OpenLLM cluster unavailable: expected exactly 4 compatible servers, found "..count)
end
return C
