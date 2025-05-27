# 🧶 yarn\_convert\_hf\_to\_gguf.py

A **modified script** based on [`convert_hf_to_gguf.py`](https://github.com/ggerganov/llama.cpp) from the `llama.cpp` project (around the `b500` release). This version injects **YaRN metadata** into the output GGUF model file to enable extended context support for local LLM inference.

## Purpose

This script is designed to convert HuggingFace models to GGUF format **with YaRN metadata included**, allowing models to utilize extended context lengths such as `131072` tokens.

---

## Key Modifications

Modifications were made specifically to the `ModelBase.prepare_metadata()` function:

```python
# Inject YaRN metadata
print("KV DATA STRUCTURE:", repr(self.gguf_writer.kv_data))
meta_dict = self.gguf_writer.kv_data[0]
meta_dict["rope_scaling.type"] = GGUFValue(value="yarn", type=GGUFValueType.STRING)
meta_dict["rope_scaling.factor"] = GGUFValue(value=1.0, type=GGUFValueType.FLOAT32)
meta_dict["context_length"] = GGUFValue(value=131072, type=GGUFValueType.UINT32)
meta_dict["max_position_embeddings"] = GGUFValue(value=131072, type=GGUFValueType.UINT32)
```

* This injects the necessary key/value pairs to inform compatible runtimes (like `llama.cpp`) that the model uses **YaRN-style RoPE scaling**.
* **Note:** The `131072` context length is hardcoded. You should change it to reflect your model’s actual max token support if different.

---

## Limitations

* The script has been **stripped of vision-related logic** that caused issues during conversion.
* Only tested on LLMs (specifically for a modified `QwQ 32B` model).
* **Not recommended** for vision or other non-LLM conversions in its current form.
* This script may break in future versions of `llama.cpp` due to upstream changes. Use cautiously and consider this experimental for now.

---

## Use Case

Use this script **only** if you are converting a HuggingFace-format LLM to GGUF **and want to enable YaRN support** during the conversion step.

---

## File

* `yarn_convert_hf_to_gguf.py`: The full script, modified from the `b500` version of llama.cpp.

---

## Credits

* Original script by the [llama.cpp](https://github.com/ggerganov/llama.cpp) team
