# Gemma4GR — HuggingFace Upload Guide

## Model Identity

- **Base:** Google Gemma 4 E4B (Unsloth Apr 11 update, `<|turn>` / `<turn|>` template)
- **Fine-tune:** STT LoRA only — QA LoRA intentionally excluded
- **Language:** Greek (el-GR)
- **Capabilities:** Greek speech-to-text transcription
- **Artifact:** `N:\.cache\huggingface\hub\gemma-4-E4B-it-GR-stt\gemma4gr-e4b-stt-q4_k_m.gguf`

---

## Exact Inference Prompts

These must be reproduced exactly. Any deviation breaks the LoRA signal.

### Q&A (text)

```python
messages = [
    {
        "role": "system",
        "content": (
            "Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. "
            "Μην χρησιμοποιείς Αγγλικά, Ρωσικά ή άλλη γλώσσα, εκτός αν το ζητά ρητά "
            "η ερώτηση. Δώσε άμεση, ουσιαστική και πλήρη απάντηση."
        ),
    },
    {
        "role": "user",
        "content": "{your question here}",
    },
]
```

- No extra instructions appended to the user message
- Temperature 0 for deterministic output
- The model was trained to give full, complete answers ("πλήρη απάντηση") — expect 2-4 sentences, not one-liners

### STT (speech-to-text)

```python
messages = [
    # NO system prompt — none was present during training
    {
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": "Transcribe the following speech segment in Greek into Greek text."},
        ],
    },
]
```

- Audio must be base64-encoded WAV, **16 kHz, mono** (training used resampled 16kHz mono WAVs — other sample rates may degrade accuracy)
- **No system prompt** — sending one will degrade transcription quality
- The instruction string must be verbatim: `"Transcribe the following speech segment in Greek into Greek text."`
- Expected output: raw Greek transcript, no punctuation beyond what was spoken, numbers as digits
- Recommended pre-processing: resample to 16kHz mono with `ffmpeg -ar 16000 -ac 1`

---

## Chat Template (turn format)

The model uses the Gemma 4 native template from the Unsloth Apr 11 update:

```
<bos><|turn>system
{system_prompt}<turn|>
<|turn>user
{question}<turn|>
<|turn>model
{answer}<turn|>
```

**Not** the old Gemma format (`<start_of_turn>` / `<end_of_turn>`). The GGUF has this baked in via llama.cpp's `--chat-template-file`.

---

## What to Upload to HuggingFace

### HF Repo Structure

```
Efso/gemma4gr-e4b-greek-stt/
├── README.md                        ← model card (from HF_README_greek_stt_v2.md)
├── gemma4gr-e4b-stt-q4_k_m.gguf   ← main artifact (Git LFS, 5.1 GB)
├── gemma4gr-e4b-stt-mmproj.gguf   ← multimodal projector (Git LFS, 944 MB)
├── chat_template.jinja
├── Modelfile
├── LICENSE                          ← CC BY-NC 4.0
└── samples/                         ← 3-5 example WAVs with transcripts
    ├── sample_001.wav
    ├── sample_001.txt
    └── ...
```

> **Git LFS required** for the GGUFs. Run `git lfs track "*.gguf"` before pushing.

### Files

| File | Notes |
|---|---|
| `gemma4gr-e4b-stt-q4_k_m.gguf` | Main artifact — llama.cpp / Ollama compatible, upload via Git LFS |
| `gemma4gr-e4b-stt-mmproj.gguf` | Multimodal projector — required for audio input, upload via Git LFS |
| `chat_template.jinja` | Copy from `tests/templates/gemma4gr_shared_chat_template.jinja` |
| `Modelfile` | Ollama integration — temperature 0, no system prompt |
| `README.md` | Model card (from `HF_README_greek_stt_v2.md`) |
| `LICENSE` | CC BY-NC 4.0 full text |
| `samples/` | 3-5 WAV + transcript pairs from validation set |

### `generation_config.json`

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "max_new_tokens": 256,
  "do_sample": false
}
```

### Ollama Modelfile

```
FROM ./gemma4gr-e4b-stt-q4_k_m.gguf
ADAPTER ./gemma4gr-e4b-stt-mmproj.gguf

PARAMETER temperature 0.0
PARAMETER num_ctx 4096
```

No SYSTEM prompt — STT training had no system prompt. The STT prompt belongs in the user message (see inference prompt above).

---

## Model Card README Template

```markdown
# Gemma4GR — Greek Q&A and STT

Fine-tuned from Google Gemma 4 E4B for Greek language Q&A and speech-to-text transcription.
Built for the Google Gemma 4 Good Hackathon (Kaggle, May 2026).

## Capabilities
- Greek text Q&A (history, culture, mythology, geography, language, food, science, religion, everyday)
- Greek speech-to-text transcription (17 categories, human voice recordings)

## Usage — Q&A

```python
import anthropic  # or any OpenAI-compatible client pointed at llama-server

messages = [
    {
        "role": "system",
        "content": (
            "Απάντησε μόνο στα Ελληνικά, με φυσική, σωστή και καθαρή γλώσσα. "
            "Μην χρησιμοποιείς Αγγλικά, Ρωσικά ή άλλη γλώσσα, εκτός αν το ζητά ρητά "
            "η ερώτηση. Δώσε άμεση, ουσιαστική και πλήρη απάντηση."
        ),
    },
    {"role": "user", "content": "Ποια ήταν η σημασία της μάχης της Σαλαμίνας;"},
]
```

## Usage — STT

```python
import base64

audio_b64 = base64.b64encode(open("speech.wav", "rb").read()).decode("ascii")

messages = [
    # NO system prompt
    {
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": "Transcribe the following speech segment in Greek into Greek text."},
        ],
    },
]
# Audio must be 16kHz mono WAV
# ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Training
- Base model: unsloth/gemma-4-E4B-it (Apr 11 2026 update, `<|turn>` / `<turn|>` template)
- STT LoRA: r=64, alpha=128, 3 epochs, lr=2e-4 — 3,056 human Greek voice recordings (16kHz mono WAV)
- QA LoRA: r=32, alpha=64, 3 epochs, lr=1e-4 — 2,227 training examples (from 2,476 pairs, 10% held for val)
- STT-QA LoRA: audio Q&A extension — 5,115 examples combining audio input with Greek Q&A answers
- Merge order: base → STT LoRA → QA LoRA → GGUF q4_k_m
- Training hardware: RTX 4060 Ti 16GB (E2B local) + Google Colab L4/A100 (E4B)

## Limitations
- STT works best on adult Greek speech; child-register audio has lower accuracy
- Q&A answers are full paragraphs by design — not suited for one-word factual lookups
- Not instruction-tuned for safety — intended for educational use with Gemma4Kids

## License
CC-BY-NC — see LICENSE
```

---

## Pre-upload Checklist

- [ ] Verify GGUFs exist at `N:\.cache\huggingface\hub\gemma-4-E4B-it-GR-stt\`
- [ ] Verify Modelfile is fixed: no SYSTEM prompt, `temperature 0.0` (see 2.3 in RELEASE_PLAN)
- [ ] Select 3-5 sample WAVs from `data/human_voice_dataset/wavs/` for `samples/`
- [ ] Copy matching transcripts from `data/human_voice_dataset/metadata.csv` into `samples/*.txt`
- [ ] Test STT inference on those samples with the exact prompt (spot-check)
- [ ] Run `git lfs track "*.gguf"` in the HF upload dir before the first push
- [ ] Set HF repo metadata: `language: el`, `license: cc-by-nc-4.0`, tags as in README frontmatter
- [ ] Create repo private → review → set public
- [ ] Submit `https://huggingface.co/Efso/gemma4gr-e4b-greek-stt` to Kaggle hackathon entry
