---
language:
- el
tags:
- greek
- stt
- speech-to-text
- asr
- gemma
- gemma-4
- gguf
- llama-cpp
- multimodal
base_model: unsloth/gemma-4-E4B-it
pipeline_tag: automatic-speech-recognition
license: cc-by-nc-4.0
---

# Gemma4GR E4B Greek STT

Greek speech-to-text fine-tune of `gemma-4-E4B-it`.

This release is the **STT-only** version of the project. It is intended for **Greek audio transcription**, not spoken question answering.

## What This Model Is

This model is:

- base `gemma-4-E4B-it`
- plus a Greek STT-only fine-tune
- exported as GGUF for local inference

This is **not** the earlier mixed merged version that combined STT and QA adapters.

## Intended Use

Use this model for:

- Greek speech transcription
- local Greek STT experiments
- llama.cpp multimodal audio inference

Do not assume this release improves:

- spoken question answering
- general assistant behavior
- multilingual dialogue

## Final Training Setup

The final STT run used:

- **3,216 real human-voice Greek WAV/transcript pairs**
- exact transcription targets only
- a strict STT prompt
- no spoken-QA audio in the final run
- no QA adapter merged into the exported model

Dataset split:

- total pairs: `3,216`
- train: `2,895`
- validation: `321`

## Evaluation

The final model was evaluated on `321` held-out Greek human-voice WAVs.

### Against base 4-bit E4B

Base E4B Q4:

- average token F1: `0.5846`
- exact normalized matches: `55 / 321`
- syntax-valid outputs: `264 / 321`

This release:

- average token F1: `0.5974`
- exact normalized matches: `65 / 321`
- syntax-valid outputs: `320 / 321`

### Against the previous merged project version

Previous merged version (`base + STT + QA`):

- average token F1: `0.5435`
- exact normalized matches: `44 / 321`
- syntax-valid outputs: `258 / 321`

This release:

- average token F1: `0.5974`
- exact normalized matches: `65 / 321`
- syntax-valid outputs: `320 / 321`

## Main Takeaway

This STT-only release performs better on held-out Greek human-voice transcription than:

- the base 4-bit E4B model
- the previous merged STT+QA release

The improvement is moderate, not dramatic, but consistent enough to make this the preferred STT release.

## Prompt

Prompt used for evaluation and intended inference:

`The spoken language is most likely Greek (el-GR). Transcribe exactly what is spoken. Keep the original language and script exactly as spoken. Reply with transcription only. Never translate. Never transliterate. Do not mix languages. If the speech is Greek, return only Greek script. If a short foreign word is clearly spoken, keep that word exactly as spoken. Output only the transcription text, with no newlines. Write numbers as digits.`

## Files

- `gemma4gr-e4b-stt-q4_k_m.gguf` — main GGUF (5.1 GB, Git LFS)
- `gemma4gr-e4b-stt-mmproj.gguf` — multimodal projector (944 MB, Git LFS)
- `chat_template.jinja` — Gemma 4 `<|turn>` / `<turn|>` template
- `Modelfile` — Ollama integration (temperature 0, no system prompt)
- `LICENSE` — CC BY-NC 4.0

## Sample Audio

Short listening examples from the held-out validation set are in `samples/`. Each `.wav` file has a matching `.txt` transcript.

| File | Transcript |
|---|---|
| `samples/sample_001.wav` | see `samples/sample_001.txt` |
| `samples/sample_002.wav` | see `samples/sample_002.txt` |
| `samples/sample_003.wav` | see `samples/sample_003.txt` |

## Limitations

- improvement over base is modest
- the model can still hallucinate plausible Greek sentences instead of exact transcription
- this release should not be treated as a spoken-QA model
- results come from the project held-out validation set, not a public benchmark
