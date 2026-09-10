# Four-server rack mode

This optional mode keeps the original single-machine runtime unchanged. It
uses four independent Tier-3 OpenComputers servers connected through the rack
network with a network card/modem installed in every server.

## What is parallelized

Every transformer layer is processed by all four workers. Each one owns two Q
heads, one KV head, approximately one quarter of the 172 SwiGLU channels, and
128 vocabulary rows. Each keeps only its own 108,152-byte OCTP shard in memory
as packed binary data plus its compact local KV cache.

For attention and FFN, workers return a packed 64-value partial output vector.
The coordinator sums the four partial vectors and broadcasts the residual state
for the next phase. The 172-channel FFN activation is never sent over the
network. The classifier is split into four 128-logit responses.

This is tensor parallelism, not a pipeline: every generated token still waits
for all layers and workers. It may reduce weight-read work, but modem latency
can outweigh the benefit on some 1.7.10 worlds.

## Clone-safe disk image install

For four identical disks that configure themselves on boot, use a separate
master OpenOS computer only while preparing one blank destination disk:

```sh
wget -f https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main/installer.lua installer.lua
lua installer.lua rack-image /mnt/314/openllm
```

This writes all four model shards and a small `/autorun.lua` boot service to
the destination disk. Shut down the master, duplicate that disk four times,
put one clone in each Tier-3 rack server, install a Network Card/modem in each,
and turn on all four servers. The master is not used again.

Every clone broadcasts its computer and modem addresses, accepts only the
`openllm-cluster` protocol version and matching model ID, waits for exactly four
compatible members, then sorts the four **computer** addresses. The lowest
address receives shard 0 and becomes coordinator; the other addresses receive
shards 1–3 in sorted order. Each clone writes its own generated
`config/rack.lua` and starts its role automatically. A reboot repeats discovery
and receives the same deterministic role when the same machines are present.

If fewer or more than four compatible servers are found, no inference starts.
An unrelated modem is ignored. A worker timeout during inference aborts the
current generation instead of using stale results.

## Manual install

Install the same release on all four servers; this downloads all shards so each
machine can be assigned a role:

```sh
wget -f https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main/installer.lua installer.lua
lua installer.lua /mnt/314/openllm
```

On each server, run `component.modem.address` in Lua and record the four
addresses. Edit `/mnt/314/openllm/config/rack.lua` on every server so
`coordinator` is server 0's modem address and `workers[1]` through `workers[3]`
are the other three addresses. Keep the same file on all four machines.

Start servers 1–3 first:

```sh
lua /mnt/314/openllm/server/worker.lua /mnt/314/openllm 1
lua /mnt/314/openllm/server/worker.lua /mnt/314/openllm 2
lua /mnt/314/openllm/server/worker.lua /mnt/314/openllm 3
```

Then start the coordinator on server 0:

```sh
lua /mnt/314/openllm/server/coordinator.lua /mnt/314/openllm
```

Workers reject messages not addressed from the configured coordinator and each
request carries a session ID, sequence number, layer, token position, and
message type. A missing or invalid worker response times out clearly rather
than silently mixing a stale vector into inference.

Use `/quit` in the coordinator to stop its terminal loop. Stop workers with
`Ctrl+C` or reboot their server.
