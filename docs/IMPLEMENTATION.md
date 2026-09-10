# Implementation guide

## What runs where

After installation, everything required for inference runs inside one
OpenComputers computer. The Internet Card is used only by `installer.lua` to
download files. The running model does not contact the network, a host process,
or an external inference service.

The implementation targets the `stories260K` TinyStories checkpoint:

| Property | Value |
| --- | --- |
| Architecture | Llama 2-style decoder-only transformer |
| Dimensions | 64 model width, 172 feed-forward width |
| Transformer layers | 5 |
| Attention | 8 query heads, 4 key/value heads |
| Vocabulary | 512 BPE pieces |
| Model maximum sequence length | 512 tokens |
| Runtime context limit | 512 tokens; default 128 |
| Installed INT8 model | 276,448 bytes |

The runtime supports the model's full 512-token sequence length and defaults
to 128 tokens. Larger contexts consume more OpenComputers RAM and make each
generated token slower. It is not a rolling window: prompt tokens plus
generated tokens must fit inside the chosen context.

## Installed layout

```text
<install-root>/
  init.lua                 interactive terminal application
  openllm.lua              launcher
  lib/
    llm.lua                generation loop and transformer forward pass
    model.lua              row-addressable OCQ8 reader
    tokenizer.lua          512-piece BPE encoder/decoder
    sampler.lua            greedy/temperature sampling
    tensor.lua             fixed-size activation helpers
    storage.lua            binary decoding and filesystem helpers
  model/
    model.bin              row-quantized INT8 checkpoint
    tokenizer.bin          llama2.c-compatible tokenizer data
    config.lua             generated model metadata
  cache/
    kv-0.bin ... kv-4.bin  temporary per-layer key/value caches
```

`cache/` is created at first run. It is safe to delete while the model is not
running; it only contains the current prompt's transient attention state.

## OCQ8 model format

`model/model.bin` starts with a fixed 256-byte header containing the `OCQ8`
magic, version, model dimensions, vocabulary size, maximum sequence length,
and classifier-sharing flag. The runtime rejects files with the wrong format or
the wrong `stories260K` dimensions.

The header is followed by float32 RMS-normalization vectors, then quantized
matrices. Every matrix row is stored independently:

```text
float32 scale | int8 weight[columns]
```

For a matrix-vector multiply, `lib/model.lua` reads one row's scale and signed
INT8 bytes, computes the dot product with the current activation, then releases
those temporary bytes. At 2048 KiB it seeks each row from disk. At 4096 KiB or
more it automatically retains the complete unchanged OCQ8 file as one compact
Lua string, avoiding repeated filesystem reads without turning weights into
large Lua number tables.

The token embedding table is also row-addressable. Because stories260K ties its
classifier to the embedding table, the final vocabulary projection reuses that
same on-disk section.

## One token of inference

For each input or generated token, `lib/llm.lua` performs this sequence:

1. Read the token's embedding row into the 64-value activation buffer.
2. For each of the five layers, read its normalization values and stream the
   query, key, value, attention-output, and feed-forward matrix rows.
3. Apply RoPE position rotation in Lua; source RoPE lookup tables are not kept
   in the installed model.
4. Append the layer's key and value vectors to `cache/kv-<layer>.bin`.
5. Read only the context-so-far cache to calculate causal attention.
6. Apply the feed-forward SwiGLU block and residual connection.
7. Apply final RMS normalization and stream the tied embedding rows to produce
   512 logits.
8. Select the next token, print its decoded text, and repeat until the context
   budget or EOS is reached.

`computer.pullSignal(0)` is called during matrix and attention work so one long
forward pass yields to OpenComputers instead of tripping its execution
watchdog.

## Memory and cache strategy

Only the working activations are Lua numeric tables: several 64-value vectors,
two 172-value feed-forward buffers, 512 logits, and an attention-score vector
the size of the selected context. Weights remain compact bytes on disk.

The KV cache stays on disk rather than as a large Lua table. Key/value elements
are Q12 fixed-point signed 16-bit values (a step of 1/4096). This is a small,
intentional approximation in addition to the INT8 weights and is what lets the
runtime fit comfortably within the 2048 KiB target.

## Tokenizer and sampling

`tokenizer.bin` contains 512 ordered BPE pieces and their scores in the format
used by llama2.c. At startup it is small enough to load into a lookup table.
Prompt encoding adds the model's BOS token and a leading-space token, applies
byte fallback for unknown bytes, then repeatedly merges the highest-scoring
pair.

Sampling is greedy when temperature is zero. At a positive temperature, the
runtime applies softmax and samples from the complete 512-token distribution
using a local linear-congruential pseudo-random generator. There is no external
randomness or service dependency.

## Installation and launch

The installer checks total RAM, detects all filesystem components, calculates
free space as `spaceTotal - spaceUsed`, and requires at least 1 MiB free at the
destination. It downloads the Lua files and model assets, verifies expected
model/tokenizer byte sizes, and creates an `openllm` launcher when `/bin` is
writable.

For an OpenOS filesystem mounted at `/mnt/314`:

```sh
wget -f https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main/installer.lua installer.lua
lua installer.lua /mnt/314/openllm
openllm
```

If the `openllm` command was created by an earlier installer revision, it can
still be used by explicitly providing the root:

```sh
openllm /mnt/314/openllm
```

In the terminal interface, use a short story beginning. `/temp 0` chooses the
most likely next token; `/temp 0.7` is the default stochastic mode;
`/context 128` changes the context for the next prompt; and `/quit` exits.

## Development verification

`tools/convert_model.py` converts the upstream float32 llama2.c checkpoint to
OCQ8. `tools/convert_tokenizer.py` converts the upstream SentencePiece model
and reproduces the official `tok512.bin` layout. `tools/verify_model.py`
checks the binary geometry and tokenizer records. `tools/verify_numerics.py`
compares a first-token float checkpoint forward pass with OCQ8 plus the Q12 KV
cache behavior; the included artifacts retain the same first-token argmax.

In-game performance and behavioral tests are intentionally left to the person
running the Minecraft world.
