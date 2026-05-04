# Gemma4GR

Fine-tuning Google Gemma 4 for Greek STT and creating **JOY** — the first high-quality open-source Greek Piper TTS voice.

---

## What This Project Builds

| Output | Description |
|--------|-------------|
| **JOY voice** (`el_GR-joy-medium.onnx`) | A new high-quality Greek Piper TTS voice trained on native human recordings. Released CC-BY-NC. |
| **Gemma4GR GGUF** | Gemma 4 E2B fine-tuned for Greek STT (hear Greek → transcribe) and Greek Q&A (write fluent Greek answers). Merged into a single GGUF for Gemma4Kids. |

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

Phase B — Gemma Greek Q&A (text LoRA) — runs in parallel with A
  generate_qa_pipeline.py → prepare_qa_dataset.py → train_qa_local.py
  Compute: local RTX 4060 Ti (fits in 16 GB VRAM)

Phase C — Gemma Greek STT (audio LoRA) — requires JOY ONNX from Phase A
  synthesize_stt_audio_piper.py → prepare_stt_dataset.py → train_e2b_local.py
  Compute: local RTX 4060 Ti

Phase D — Ship
  merge_adapters.py → merged GGUF → Gemma4Kids runtime
  JOY ONNX speaks Gemma's Greek text output
```

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

**Before spending money on Vast.ai:** always run the local smoke tests first (20-sample Piper test, 50-sample Gemma E2B test). See [COMPUTE_STRATEGY.md](COMPUTE_STRATEGY.md) §6.

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

---

## Environment

- Python 3.11 · Windows 10 / WSL2
- RTX 4060 Ti or 5060 Ti — 16 GB VRAM
- Docker Desktop + NVIDIA Container Toolkit (for Piper training)
- Vast.ai account (for Piper + E4B cloud runs)
- HuggingFace token in `.env`

---

## Community Contribution

The JOY voice will be published on the Piper voice repository and HuggingFace under CC-BY-NC once training is complete and quality is validated. If you use it, please credit: **JOY Greek voice — Gemma4GR project, CC BY-NC 4.0**.
