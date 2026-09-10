# Internet gateway controller

`controller/rack_manager.lua` lets one ordinary OpenComputers case machine
with an Internet Card and Network Card download a release once and install it
onto four networked rack servers. The rack servers do not need Internet Cards.

OpenComputers machines are independent: a server cannot accept remote files or
run remote code until it has a receiving program. Therefore copy the small
`server/rack_agent.lua` bootstrap file to each rack server once, replace its
`REPLACE_WITH_CONTROLLER_MODEM_ADDRESS` value with the controller modem address,
and run it. This is the only per-server bootstrap step.

On the controller case, copy `controller/rack_manager.lua` and
`config/gateway.lua` under one root, edit `config/gateway.lua` with the four
rack-server modem addresses, then run:

```sh
lua /openllm-controller/controller/rack_manager.lua /openllm-controller install
lua /openllm-controller/controller/rack_manager.lua /openllm-controller start
```

The controller downloads files from GitHub, sends 4 KiB binary chunks over the
modem, waits for an acknowledgement for every chunk, writes each server's
role-specific shard/configuration, then starts workers 1–3 followed by the
coordinator. `status` checks whether the bootstrap agents are reachable before
they are launched. A transfer failure or missing acknowledgement stops with a
clear timeout instead of silently continuing.
