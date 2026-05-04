# Voice Card — JOY (el_GR-joy-medium)

## Overview

| Field | Value |
|-------|-------|
| Voice name | JOY |
| Greek name | Χαρά (Chara) |
| Language | Greek (el_GR) |
| Quality | medium |
| Architecture | Piper VITS |
| Sample rate | 22050 Hz |
| Speaker | Native Greek female speaker |
| License | **CC BY-NC 4.0** |
| File | `el_GR-joy-medium.onnx` + `el_GR-joy-medium.onnx.json` |
| Project | Gemma4GR — https://github.com/Efs-O/Gemma4GR |

---

## About the Voice

JOY is a high-quality Greek Piper TTS voice trained entirely on human recordings by a native Greek speaker. It was created to fill the gap in open-source Greek TTS — all existing community voices were trained on synthetic or low-quality data, resulting in robotic, mispronounced output unsuitable for educational or assistive applications.

JOY is designed for:
- Educational software (Gemma4Kids)
- Greek-language assistants and chatbots
- Accessibility tools
- Any application requiring natural-sounding Greek TTS

---

## Training

| Parameter | Value |
|-----------|-------|
| Training framework | piper-train (official) |
| Base checkpoint | Pre-trained Greek Piper checkpoint |
| Dataset size | ~3000 utterances (human recordings) |
| Dataset format | LJSpeech (22050 Hz WAV + metadata.csv) |
| Epochs | 20 |
| Batch size | 32 |
| Quality | medium (VITS) |
| Recording environment | Quiet room, USB condenser microphone |

---

## Dataset

The training dataset consists of human voice recordings covering:
- Children's speech and storytelling
- Everyday conversation
- Family and home language
- School and learning vocabulary
- Numbers, dates, prices (spelled in full)
- Greek cultural references and traditions
- Polite service interactions
- Healthcare and wellbeing phrasing

All recordings were made in a single-speaker controlled environment. No synthetic audio was used in Piper training.

---

## License: CC BY-NC 4.0

**Creative Commons Attribution-NonCommercial 4.0 International**

You are free to:
- **Share** — copy and redistribute the voice in any medium or format
- **Adapt** — remix, transform, and build upon the voice

Under the following terms:
- **Attribution** — You must give appropriate credit: *"JOY Greek voice — Gemma4GR project, CC BY-NC 4.0"* and link to https://github.com/Efs-O/Gemma4GR
- **NonCommercial** — You may not use this voice for commercial purposes without explicit written permission from the project author

For commercial licensing enquiries: amandoulou@yahoo.gr

Full license text: https://creativecommons.org/licenses/by-nc/4.0/legalcode

---

## Attribution (required when using this voice)

```
JOY Greek voice (el_GR-joy-medium)
Gemma4GR project — https://github.com/Efs-O/Gemma4GR
License: CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/
```

---

## Known Limitations

- Trained on a single speaker — accent and prosody reflect one native speaker's style
- Pronunciation of very rare proper nouns or foreign words may vary
- Optimized for standard Modern Greek; regional dialects not covered

---

## Citation

If you use this voice in a research or academic context:

```
@misc{gemma4gr-joy-voice-2026,
  title  = {JOY: A High-Quality Greek Piper TTS Voice},
  author = {Gemma4GR Project},
  year   = {2026},
  url    = {https://github.com/Efs-O/Gemma4GR},
  note   = {CC BY-NC 4.0}
}
```
