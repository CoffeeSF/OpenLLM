# Third-party notices

## stories260K model weights and tokenizer

`model/model.bin` and `model/tokenizer.bin` are converted from the
[`stories260K` files in karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K).
The TinyLlamas model repository is labeled **MIT** on its Hugging Face model
card. The conversion preserves the model architecture and tokenizer ordering;
OpenLLM adds only the OCQ8 row-quantized storage layout.

## llama2.c

The source checkpoint layout and tokenizer file format follow
[karpathy/llama2.c](https://github.com/karpathy/llama2.c), licensed under the
MIT License. No llama2.c code is copied into the OpenComputers runtime.
