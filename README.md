# Gemma4GR

Fine-tuning Google Gemma 4 for Greek STT and creating **JOY** — the first high-quality open-source Greek Piper TTS voice.

---

## What This Project Builds

| Output | Description |
|--------|-------------|
| **JOY voice** (`el_GR-joy-medium.onnx`) | A new high-quality Greek Piper TTS voice trained on native human recordings. Released CC-BY-NC. |
| **Gemma4GR GGUF** | Gemma 4 E4B fine-tuned for Greek STT. Trained on 3,216 human voice recordings (2,895 training, 321 validation). STT LoRA only — QA excluded from final release. Available at [Efso/gemma4gr-e4b-greek-stt](https://huggingface.co/Efso/gemma4gr-e4b-greek-stt) |

### Why JOY exists

Every freely available Greek TTS voice today was trained on synthetic or low-quality data — the audio is robotic, mispronounced, and unusable for educational software. This project fills that gap. JOY is recorded by a native Greek speaker in full sessions and trained using the official Piper pipeline. The resulting `.onnx` will be the best freely available Greek voice when it ships.

The voice is named after **Χαρά** (Chara — Joy in Greek), who records all sessions.

---

## License

| Component | License |
|-----------|---------|
| Code (this repo) | MIT |
| **JOY voice** (`el_GR-joy-medium.onnx` + recordings) | **CC BY-NC 4.0** — free to use, share, and adapt for non-commercial purposes with attribution. Commercial use requires separate permission. |

See [VOICE_CARD.md](VOICE_CARD.md) for full voice metadata and attribution requirements.

---

## Pipeline Overview

```
Phase A — JOY voice (human recording → Piper ONNX)
  generate_voice_sentence_list.py → record → train_piper.py
  Compute: Vast.ai Linux GPU (Docker-native, 6-12h)

Phase B — Gemma Greek Q&A (text LoRA) — built and tested; excluded from final merged model
  (QA LoRA trained, validated; intentionally not merged into the STT-only release)

Phase C — Gemma Greek STT (audio LoRA) — requires JOY ONNX from Phase A
  synthesize_qa_audio.py → prepare_stt_qa_dataset.py → train_stt_qa_local.py
  or synthesize_stt_audio_piper.py → prepare_stt_dataset.py → train_e2b_local.py
  Compute: local RTX 4060 Ti

Phase D — Ship
  merge_adapters.py → merged GGUF → Gemma4Kids runtime
  JOY ONNX speaks Gemma's Greek text output
```

Post-training note for the final JOY rebuild: after the full high-quality Piper training run finishes and the new `el_GR-joy-medium.onnx` is exported, do a short inference-tuning pass before treating that voice as final. Re-check `length_scale`, `noise_scale`, and `noise_w` in the matching `.onnx.json` or via Piper CLI overrides, because the best runtime settings for the small interim JOY voice may not be the best settings for the full-trained voice.

Full pipeline table: [CLAUDE.md](CLAUDE.md)  
Compute decisions: [COMPUTE_STRATEGY.md](COMPUTE_STRATEGY.md)  
Piper migration plan: [PIPER_REPLACES_MOIRA_PLAN.md](PIPER_REPLACES_MOIRA_PLAN.md)

---

## Compute Strategy (summary)

| Workload | Platform | Reason |
|----------|----------|--------|
| Piper voice training | **Vast.ai first** | Docker-native Linux GPU; 6-12h vs 24-48h local |
| Gemma E2B LoRAs | **Local** | Fits in 16 GB VRAM |
| Gemma E4B (if needed) | **Vast.ai A100 or Colab Pro** | 17 GB minimum; local does not fit |
| Merge → GGUF | **Local** | CPU task |

**Before spending money on Vast.ai:** always run the local smoke tests first.
- Piper parity test: Docker, 20 samples
- Gemma E2B smoke: native Windows, 200 Q/A pairs, bounded training
See [COMPUTE_STRATEGY.md](COMPUTE_STRATEGY.md) §6.

---

## Quick Start

```bash
# 1. Copy env and fill HF_TOKEN
cp .env.example .env

# 2. Piper Docker training (clone rhasspy sources + build image once)
python training/setup_piper_sources.py
docker build -f Dockerfile.piper -t piper-training:local .

# 3. Main menu
python menu.py

# 4. Standalone scripts
python training/generate_voice_sentence_list.py   # Phase A step 1
python training/generate_qa_pipeline.py           # Phase B step 1
python training/synthesize_stt_audio_piper.py     # Phase C step 1
python training/merge_adapters.py                 # Phase D
```

To compare `base E4B` and `fine-tuned E4B` side by side with Greek text and WAV prompts while playing answers through Piper JOY:

```bash
python tests/compare_4bit_models_gui.py
```

---

## Environment

- Python 3.11 · Windows 10 / WSL2
- RTX 4060 Ti or 5060 Ti — 16 GB VRAM
- Docker Desktop + NVIDIA Container Toolkit (for Piper training)
- Vast.ai account (for Piper + E4B cloud runs)
- HuggingFace token in `.env`
- Shared Hugging Face cache can live on `N:\.cache\huggingface\hub`

## Smoke Notes

- Gemma smoke tests run natively on Windows, not in Docker.
- Piper training is the path that needs Docker parity before any paid remote run.
- Local Gemma smoke now validates:
  - model resolution from `N:\.cache`
  - 200-pair text Q/A training
  - 200-pair audio-Q/A training with JOY-generated WAVs
  - trainer metrics export to JSON and CSV under each output directory

---

## Smoke Status

- Piper parity testing is a Docker job locally because the paid remote Piper path is also Docker/Linux.
- Gemma E2B smoke and local finetunes are native Windows GPU jobs, not Docker jobs.
- Shared Hugging Face model snapshots are expected under `N:\.cache\huggingface\hub\`.
- The validated local smoke corpus is `200` Q&A pairs:
- text smoke: `data/train_qa_smoke_200.jsonl` / `data/val_qa_smoke_200.jsonl`
- audio smoke: `data/train_stt_qa_smoke_200.jsonl` / `data/val_stt_qa_smoke_200.jsonl`
- The audio-Q&A trainer uses a script-level workaround for a TRL multimodal metrics bug. No package versions were changed to make the smoke run pass.

---

## Community Contribution

The JOY voice will be published on the Piper voice repository and HuggingFace under CC-BY-NC once training is complete and quality is validated. If you use it, please credit: **JOY Greek voice — Gemma4GR project, CC BY-NC 4.0**.
