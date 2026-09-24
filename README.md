# Gemma4GR

Fine-tuning Google Gemma 4 for Greek speech understanding and language generation — and building **JOY**, the first high-quality open-source Greek Piper TTS voice.

> **Hackathon submission:** [Google Gemma 4 Good Hackathon (Kaggle, May 2026)](https://www.kaggle.com/competitions/google-gemma-4-good-hackathon)

> **⚠️ Correction (2026-09-24).** The v2 GGUFs released in May 2026 (Hugging Face and Ollama) were exported incorrectly and contained the **unmodified base model** `google/gemma-4-E4B-it`. The LoRA adapters were fine; the export step silently fell back to the base weights. The v2 evaluation numbers (+29% overall, +67% audio) compared the base model with itself and are **withdrawn**. **v3** is the first build with the fine-tune actually merged, verified tensor by tensor. The measured results are below.

---

## Model

**[Efso/gemma-4-E4B-it-GR-v2](https://huggingface.co/Efso/gemma-4-E4B-it-GR-v2)** (v3 files; the repo name is kept from v2): Gemma 4 E4B with a Greek Q&A LoRA adapter merged in, trained on ~2,200 curated Greek question–answer pairs, for cleaner and more natural Modern Greek.

The v3 release contains **no speech or vision fine-tune**: audio and image input work at stock Gemma 4 E4B level.

| File | Size | Use |
|------|------|-----|
| `gemma-4-e4b-it-gr-v3-Q4_K_M.gguf` | 5.34 GB | Lightest |
| `gemma-4-e4b-it-gr-v3-Q6_K.gguf` | 6.22 GB | Middle ground |
| `gemma-4-e4b-it-gr-v3-Q8_0.gguf` | 8.03 GB | Closest to full precision |
| `gemma-4-e4b-it-gr-v3-mmproj.gguf` | 0.99 GB | Audio/image input (stock projector) |

Also on Ollama: `ollama run efso/gemma-4-e4b-it-gr-v2` (tags `latest` = `q4_k_m`, `q6_k`, `q8_0`; text only). If you pulled before 2026-09-24, pull again.

**Not a source of facts.** v3 writes fluent, confident Greek but doesn't know more than the base model. Verify anything factual.

---

## What This Project Builds

| Output | Description |
|--------|-------------|
| **Gemma4GR GGUF** | Gemma 4 E4B with a Greek Q&A QLoRA adapter merged in (v3). |
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

Phase 1 — STT adapter  (v2 experiment; not included in the v3 release)
  JOY synthetic WAVs + human voice recordings (3,217 WAVs)
  → prepare_stt_final_dataset.py
  → train_stt_final_local.py  (FastVisionModel, QLoRA r=32, 2 epochs)
  → output/e4b_stt_final/lora_adapter

Phase 2 — Q&A adapter
  generate_qa_pipeline.py (Ollama qwen3.5, 10 categories)
  → scripts/build_v3_dataset.py  (dedup + filtering → 2,219 train / 96 val, frozen eval sets)
  → training/run_v3.py → train_qa_local.py  (QLoRA r=32, response-only loss, 2 epochs)

Phase 3 — Merge + release
  merge_lora_tensors.py  (CPU tensor-level merge: W + (α/r)·B·A)
  → llama.cpp convert_hf_to_gguf + llama-quantize  (Q4_K_M, Q6_K, Q8_0)
  → gguf_not_base_guard.py  (refuses any GGUF that is still the base model)
  → gemma-4-e4b-it-gr-v3-*.gguf  [HuggingFace: Efso/gemma-4-E4B-it-GR-v2]
```

**What went wrong in v2:** `merge_adapters.py` relied on Unsloth's `save_pretrained_gguf`, which silently exported the base snapshot instead of the merged weights. v3 merges at tensor level, converts with llama.cpp directly, and checks every GGUF against the base before release.

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

## Evaluation (v3)

Frozen held-out sets, llama.cpp `llama-server`, identical settings for every model (temperature 0, seed 42, same Greek system prompt). Stock = `google/gemma-4-E4B-it` converted and quantized with the same pipeline.

| Model | Text chrF ↑ | Answers with foreign-script garbage ↓ | Didn't stop (of 150) ↓ | Speech CER ↓ |
|---|---|---|---|---|
| stock E4B Q4_K_M | 0.298 | 38.7% | 131 | 0.256 |
| stock E4B Q8_0 | 0.307 | 40.0% | 119 | 0.135 |
| **v3 Q4_K_M** | **0.391** | **1.3%** | **0** | 0.138 |
| **v3 Q6_K** | **0.392** | **0.7%** | 1 | 0.123 |
| **v3 Q8_0** | **0.391** | **0.7%** | **0** | 0.158 |

- **Text:** 150 held-out Greek questions with reference answers (chrF = character-level similarity).
- **Speech:** 40 held-out Greek clips; v3 is within the noise of stock, as expected with no speech fine-tune.
- **Factual spot check** (25 answers read by hand): v3 still makes confident factual errors at about the same rate as stock. It fixes the *form* of answers, not their *knowledge*.

Full method and per-quant details: [model card](https://huggingface.co/Efso/gemma-4-E4B-it-GR-v2).

---

## Quick Start

```bash
cp .env.example .env        # fill HF_TOKEN
python menu.py              # main TUI

# Standalone
python training/run_final_local_sequence.py   # full E4B training sequence
python training/merge_adapters.py             # merge + GGUF export (tensor-level, guarded)
python tests/compare_4bit_models_gui.py       # compare base vs fine-tuned
```

---

## Repository Layout

| Path | Contents |
| --- | --- |
| `training/` | Training, merge, GGUF export and the not-base guard |
| `scripts/` | v3 dataset build and eval harness (`scripts/legacy/`: v2-era helpers, kept for reference) |
| `tests/` | Unit tests (`test_*.py`) and the older benchmark tools used by `menu.py` |
| `docs/` | HF model card, JOY voice card, Piper recording guide (`docs/archive/`: v2-era guides, superseded) |

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
