# Gemma4GR

Fine-tuning Google Gemma 4 for Greek speech understanding and language generation — and building **JOY**, the first high-quality open-source Greek Piper TTS voice.

> **Hackathon submission:** [Google Gemma 4 Good Hackathon (Kaggle, May 2026)](https://www.kaggle.com/competitions/google-gemma-4-good-hackathon)

---

## Model

**[Efso/gemma-4-E4B-it-GR-v2](https://huggingface.co/Efso/gemma-4-E4B-it-GR-v2)** — Gemma 4 E4B fine-tuned for:
1. **Greek STT** — understands spoken Modern Greek across 17 voice categories (3,217 human recordings + JOY synthetic speech)
2. **Greek Text Q&A** — fluent, culturally accurate Greek prose across 10 topic categories (2,476 curated pairs)

Both adapters merged into a single GGUF, ready for Ollama or llama.cpp.

| File | Size | Use |
|------|------|-----|
| `gemma4gr-e4b-v2-q4_k_m.gguf` | 5.0 GB | Primary inference |
| `gemma4gr-e4b-v2-q8_0.gguf` | 7.5 GB | Higher precision |
| `gemma4gr-e4b-v2-mmproj.gguf` | 945 MB | Audio/vision projection |

---

## What This Project Builds

| Output | Description |
|--------|-------------|
| **Gemma4GR GGUF** | Gemma 4 E4B fine-tuned for Greek STT + Q&A. Two QLoRA adapters merged via Unsloth. |
| **JOY voice** (`el_GR-joy-medium.onnx`) | First high-quality open-source Greek Piper TTS voice, recorded by a native speaker. CC BY-NC 4.0. |

JOY is named after **Χαρά** (Chara — Joy in Greek), who records all voice sessions.

---

## Pipeline

```
Phase A — JOY Greek voice (Piper TTS)
  generate_voice_sentence_list.py
  → native speaker records 3,217 WAVs across 17 categories
  → train_piper.py (local)
  → el_GR-joy-medium.onnx  [HuggingFace: Efso/joy-greek-tts]

Phase 1 — STT adapter
  JOY synthetic WAVs + human voice recordings (3,217 WAVs)
  → prepare_stt_final_dataset.py
  → train_stt_final_local.py  (FastVisionModel, QLoRA r=32, 2 epochs)
  → output/e4b_stt_final/lora_adapter

Phase 2 — Q&A adapter
  generate_qa_pipeline.py (Ollama qwen3.5, 2,476 pairs, 10 categories)
  → prepare_qa_dataset.py
  → train_qa_local.py  (FastModel, QLoRA r=32, 2 epochs)
  → output/e4b_greek_qa/lora_adapter

Phase 3 — Merge + release
  merge_adapters.py → Unsloth save_pretrained_gguf
  → gemma4gr-e4b-v2-q4_k_m.gguf  [HuggingFace: Efso/gemma-4-E4B-it-GR-v2]
```

Full pipeline table: [CLAUDE.md](CLAUDE.md)

---

## JOY Voice — Technical Notes

Getting Piper to train locally on Windows required significant engineering work. The original plan was to use Vast.ai (Linux cloud GPU) but we got it running locally with Docker Desktop + NVIDIA Container Toolkit.

The main challenges solved:

**1. Building piper-phonemize from source inside Docker**
Piper's phonemizer is a C++ library (CMake) that downloads ONNX Runtime and builds espeak-ng as an ExternalProject. We pin exact git SHAs (`setup_piper_sources.py`) for reproducibility and drive the full build inside `Dockerfile.piper`.

**2. Upstream import path bug**
Piper's `monotonic_align` Cython extension has a broken relative import (`from .monotonic_align.core import` → `from .core import`). We patch it at build time in the Dockerfile.

**3. Runtime library survival across bind-mounts**
`train_piper.py` bind-mounts the host source tree over `/app/piper-phonemize-src` at runtime, which would hide the ONNX Runtime and espeak-ng `.so` files built into the image. We copy them to `/opt/piper-phonem-deps/lib` before the mount and set `LD_LIBRARY_PATH` accordingly.

**4. PyTorch 2.6 checkpoint compatibility**
PyTorch 2.6 defaults `torch.load(weights_only=True)`, which breaks Piper's `.ckpt` resume logic. Fixed with `ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the Dockerfile.

Scripts: [`Dockerfile.piper`](Dockerfile.piper) · [`training/setup_piper_sources.py`](training/setup_piper_sources.py) · [`training/train_piper.py`](training/train_piper.py)

---

## Evaluation

Evaluated on 54 curated Greek cases (20 text Q&A + 34 spoken audio):

| Metric | Base E4B | Gemma4GR v2 | Δ |
|--------|----------|-------------|---|
| Overall pass rate | 35% | 45% | **+29%** |
| Audio spoken_qa passes | 22% | 37% | **+67%** |
| Text Q&A passes | 57% | 60% | +4% |
| Avg token F1 | 0.222 | 0.284 | +28% |

Full results: [tests/benchmark_results/](tests/benchmark_results/)

---

## Quick Start

```bash
cp .env.example .env        # fill HF_TOKEN
python menu.py              # main TUI

# Standalone
python training/run_final_local_sequence.py   # full E4B training sequence
python training/merge_adapters.py             # merge + GGUF export
python tests/compare_4bit_models_gui.py       # compare base vs fine-tuned
```

---

## Environment

- Python 3.11 · Windows 10
- NVIDIA RTX 5060 Ti — 16 GB VRAM
- E4B (4B params) fits locally with QLoRA 4-bit
- Shared HuggingFace cache: `N:\.cache\huggingface\hub`
- Teacher model: `qwen3.5:397b-cloud` via Ollama (`localhost:11434`)

---

## License

| Component | License |
|-----------|---------|
| Code (this repo) | MIT |
| Gemma4GR GGUF model | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| JOY voice (`el_GR-joy-medium.onnx` + recordings) | CC BY-NC 4.0 — Chara Kaltsou / Gemma4GR project |

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — QLoRA fine-tuning
- [Google DeepMind](https://deepmind.google) — Gemma 4 base model
- [Piper TTS](https://github.com/rhasspy/piper) — Greek voice synthesis
- Chara Kaltsou — JOY Greek voice recordings
