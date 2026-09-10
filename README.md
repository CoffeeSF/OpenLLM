# OpenLLM for OpenComputers

This is a complete local inference runtime for Karpathy's **TinyStories
stories260K** model. It runs in one OpenComputers computer: there is no server,
API, host-side process, or Internet Card requirement after installation.

The installed model is `OCQ8`, a 276,448-byte, row-quantized INT8 conversion
of `stories260K`. Lua seeks each matrix row from the RAID-backed file, uses it
once, and discards it. Only small activation buffers, tokenizer data, logits,
and a 16-token fixed-point key/value cache are resident in Lua.

## Hardware target

Use a Tier-3-class OpenComputers machine with **2048 KiB total RAM** and a
filesystem on the RAID with at least 1 MiB free (3 MiB drives in RAID are more
than sufficient). The default context is **16 tokens**. It can be changed up
to 64, but 16 is the safest initial setting for memory and responsiveness.

The runtime yields during matrix and attention work with
`computer.pullSignal(0)` to avoid the execution watchdog. It does not include
benchmarks or in-game performance tests.

## Install

Publish this repository under the `CoffeeSF/OpenLLM` name. From
OpenComputers, with an Internet Card installed, run:

```sh
wget -f https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main/installer.lua installer.lua
installer
```

To keep the OS and model on separate filesystems, pass both a destination on
the RAID and the exact release URL:

```sh
installer /mnt/raid/openllm https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main
```

The installer checks RAM and destination free space, lists every detected
filesystem, downloads the runtime and binary assets, validates model/tokenizer
sizes, and adds an `openllm` command when `/bin` is writable. The Internet Card
may then be removed.

## Use

```sh
openllm
```

Enter a short story beginning, e.g. `Once upon a time`. The interface supports
`/temp 0.7`, `/context 16`, and `/quit`. A prompt plus its continuation must
fit the selected context. This is a TinyStories continuation model, not an
instruction-following chat assistant.

## Files and conversion

- `model/model.bin` — row-addressable INT8 weights (committed artifact)
- `model/tokenizer.bin` — 512-piece BPE tokenizer (committed artifact)
- `tools/convert_model.py` — standard-library conversion from the upstream
  legacy llama2.c checkpoint to OCQ8
- `tools/convert_tokenizer.py` — standard-library conversion from the upstream
  SentencePiece tokenizer to llama2.c tokenizer format

To reproduce the artifacts on a development computer:

```sh
curl -L -o stories260K.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
curl -L -o tok512.model https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model
python tools/convert_model.py stories260K.bin model/model.bin --config model/config.lua
python tools/convert_tokenizer.py tok512.model model/tokenizer.bin
```

`convert_model.py` reports reconstruction error and writes the SHA-256 sidecar.
The runtime validates the OCQ8 magic, model version, architecture, and minimum
file length before inference.

## Numerical checks

The converter reads the official legacy llama2.c layout in its exact tensor
order and performs symmetric per-row INT8 quantization. Its reported maximum
weight reconstruction error for the included conversion is `0.00704962` and
the largest per-matrix mean error is `0.00155485`. `tools/verify_model.py`
validates the header, section geometry, row bounds, tokenizer vocabulary, and
reconstructs every quantized row to check the stored scales and integers.

The intentional additional approximation is Q12 fixed-point storage for the
on-disk KV cache (1/4096 steps); this keeps context state out of Lua tables.
Use short prompts and the default 16-token context when checking results
against a float reference.

## Model source and license

The source model is [`karpathy/tinyllamas`, stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K), trained with [Karpathy's llama2.c](https://github.com/karpathy/llama2.c).
The upstream model card identifies the model as a 260K-parameter TinyStories
model and labels the repository **MIT**. See `THIRD_PARTY_NOTICES.md` when
redistributing the converted weights. The OpenLLM Lua and Python code in this
repository is released under the MIT License in `LICENSE`.
