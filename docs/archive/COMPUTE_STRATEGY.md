# Compute Strategy — Gemma4GR Training Decisions

**Date:** 2026-05-04  
**Priority order: Vast.ai → Unsloth/Colab → Local**  
**Rule:** Always run local smoke tests (§6) before spending money on any remote machine.

---

## 1. Hardware Reality Check

### Local machine
| Resource | Spec | Implication |
|----------|------|-------------|
| GPU | RTX 4060 Ti / 5060 Ti — 16 GB VRAM | E2B QLoRA fits (~8-10 GB); E4B does **not** fit (~17 GB needed) |
| Piper training | Docker + NVIDIA Container Toolkit | Works locally; `train_piper.py` estimates **24-48 hours** for a full run |
| System RAM | Windows 10 | E4B model loading + dataset buffering under Unsloth will likely OOM even before VRAM is exhausted |
| OS | Windows 10 + Docker Desktop | Docker volume mounts work; NVIDIA Container Toolkit overhead adds ~10-15 min startup per Piper job |

### Vast.ai — FIRST CHOICE for cloud work
| What it gives you | Relevance |
|-------------------|-----------|
| Bare Linux VM with NVIDIA GPU | Piper Docker workflow runs **unchanged** — no adaptation needed |
| Choice of GPU tier (3090, A100, H100) | Pick to match workload and budget |
| Hourly billing — stop when done | Better than Colab time limits for 6-12h Piper jobs |
| SSH + `vastai copy` | Upload dataset, pull back `.onnx` after — I can run all of this from the terminal here |
| Docker pre-installed on most images | `train_piper.py` runs without modification |
| Full CLI automation | `vastai search`, `create`, `copy`, `destroy` — all scriptable |

### Unsloth / Google Colab — SECOND CHOICE (fallback)
| Tier | GPU | VRAM | Practical use |
|------|-----|------|---------------|
| Free T4 | T4 | 16 GB | E2B possible; E4B too slow / OOM |
| Colab Pro L4 | L4 | 24 GB | E4B fine-tune: feasible |
| Colab Pro A100 | A100 | 40 GB | E4B + large batch: comfortable |
| **Piper on Colab** | any | — | **Hard no** — Piper requires Docker; Colab cannot run Docker containers. This is not a workaround situation. |

### Local — THIRD CHOICE (smoke tests + E2B only)
Used for: smoke tests, Gemma E2B LoRAs (fits VRAM), merge/GGUF export, Piper synthesis (CPU task).  
Not used for: E4B training, full Piper training run.

---

## 2. Workload Assignment

| Workload | Platform | Why |
|----------|----------|-----|
| **Piper JOY voice training** (3000+ samples, 20 epochs) | **Vast.ai** (Linux GPU, Docker) | 24-48h local is too slow; Colab cannot run Docker |
| **Gemma E2B QA LoRA** | **Local** first → Vast.ai if stalled | Fits in 8-10 GB VRAM; ~2-4h |
| **Gemma E2B STT LoRA** | **Local** first → Vast.ai if stalled | Same size |
| **Gemma E4B** (if needed) | **Vast.ai A100** first → Colab Pro second | 17 GB minimum; local does not fit |
| **Merge adapters → GGUF** | **Local** | CPU task, no GPU needed |
| **Piper synthesis** (batch WAV generation) | **Local** | Subprocess only, no GPU |

---

## 3. Why Vast.ai Over Colab

- **No session timeout.** Colab free disconnects after 12h; Pro after ~24h. A Piper run is 6-12h on a good GPU — Colab free is risky, Pro is possible but tight. Vast.ai runs until you stop it.
- **Docker support.** Piper training requires `nvcr.io/nvidia/pytorch` Docker container. Colab cannot run Docker. This rules Colab out for Piper entirely.
- **Full CLI control.** I can operate the entire Vast.ai lifecycle (search, rent, upload, start training, download, destroy) from the terminal in this session — you don't need to do anything manually.
- **Cost.** RTX 3090 on Vast.ai ~$0.35-0.50/hr. A 10-hour Piper run costs ~$4. A100 costs more but finishes in 3-5h for a similar total.

---

## 4. Recommended Run Plan

```
Step 1 — Local smoke tests (free, ~1h total)
  a. 20-sample Piper test: verify Docker + NVIDIA + dataset format + measure steps/sec
  b. 200-pair Gemma E2B test: verify cache resolution, VRAM ceiling, dataset format, and one bounded train step

Step 2 — Vast.ai Piper (after smoke test passes)
  Gate: $0.10 dry-run (20 samples, 1 epoch) on cheapest available GPU
  Full run: RTX 3090 or A100, 3000 samples, 20 epochs → ~6-12h → ~$4-6

Step 3 — Local Gemma E2B LoRAs
  QA LoRA: train_qa_local.py (~2-4h)
  STT LoRA: train_e2b_local.py (~2-4h, after Piper ONNX exists)

Step 4 — Local merge
  merge_adapters.py → GGUF
```

---

## 5. What I Can Do Remotely

Once you have a Vast.ai API key set, I can execute all of this from the terminal here:

```bash
# Search for machines
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 verified=true rentable=true' -o 'dph_usd+'

# Rent + start
vastai create instance OFFER_ID --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime --disk 30 --ssh --direct

# Upload dataset
vastai copy local:./data/human_voice_dataset/ INSTANCE_ID:/workspace/data/human_voice_dataset/

# SSH in and start training (under tmux so it survives disconnects)
ssh root@HOST -p PORT "tmux new -d -s piper 'python training/train_piper.py'"

# Check progress
ssh root@HOST -p PORT "tail -50 logs/piper_training.log"

# Download result
vastai copy INSTANCE_ID:/workspace/output/piper_voice/ local:./output/piper_voice/

# Destroy immediately after download
vastai destroy instance INSTANCE_ID
```

You call me back at any point and I check the log, adjust, or pull results.

---

## 6. Pre-Flight Smoke Tests (run these before any remote spend)

### 6.1 Local Piper test — 20 samples (required before Vast.ai)

**Goal:** Confirm Docker health, NVIDIA Container Toolkit, dataset format. Get steps/sec to project full run time.

```bash
# 1. Set env for test run
# In .env: PIPER_DATASET_DIR=data/piper_test_20  PIPER_MAX_EPOCHS=2

# 2. Record (or copy) 20 WAVs into data/piper_test_20/wavs/
# metadata.csv already prepared — see data/piper_test_20/metadata.csv

# 3. Run
python training/train_piper.py

# 4. Note wall-clock time. Project: (3000/20) × (20/2) × measured_time
# If projected > 12h → Vast.ai. If < 8h → local is viable.
```

**Go/no-go:** If this fails, fix it here before spending anything on Vast.ai. The same failure will happen there.

### 6.2 Local Gemma E2B test — 200 pairs (required before full E2B run)

**Goal:** Confirm Unsloth loads E2B from shared cache, no OOM, dataset format correct, and both text and audio-QA trainers can execute.

```bash
# Build a 200-pair smoke corpus and prepare train/val JSONL
# Then run bounded 1-step smoke tests for:
# - training/train_qa_local.py
# - training/train_stt_qa_local.py
# Watch VRAM and keep max_steps=1 for the first pass
python training/train_qa_local.py
python training/train_stt_qa_local.py
```

Notes:
- Gemma smoke runs are native Windows runs, not Docker runs.
- Docker parity matters for Piper only.
- Exported metrics should be kept from `metrics_history.json` / `metrics_history.csv` before scaling the run up.

### 6.3 Vast.ai dry-run — $0.10 gate

Before the full Piper run, rent the cheapest GPU available, upload 20 samples, run 1 epoch, confirm training starts. Destroy. Total cost: negligible. Catches instance/network/volume issues before the paid run.

### 6.4 Do NOT test E4B locally

16 GB < 17 GB minimum. Do not attempt even a 1-step E4B run locally — go straight to Vast.ai A100 or Colab Pro L4.

---

## 7. Vast.ai Full Checklist

When ready for the Piper full run:

- [ ] Install Vast.ai CLI: `pip install vastai`
- [ ] Set API key: `vastai set api-key YOUR_KEY`
- [ ] Register SSH key: `vastai create ssh-key ~/.ssh/id_ed25519.pub`
- [ ] Run smoke test 6.1 locally first
- [ ] Run $0.10 dry-run (6.3) on Vast.ai
- [ ] Rent full instance: Ubuntu 22.04 + CUDA 11.8+, Docker pre-installed, RTX 3090 or A100
- [ ] Upload: `vastai copy local:./data/human_voice_dataset/ ID:/workspace/data/human_voice_dataset/`
- [ ] Upload scripts + `.env` (with `PIPER_BASE_CKPT` set)
- [ ] Confirm: `docker run --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi`
- [ ] Start under tmux: `tmux new -d -s piper 'python training/train_piper.py'`
- [ ] Monitor via log: `ssh ... "tail -f logs/piper_training.log"`
- [ ] After ONNX export: `vastai copy ID:/workspace/output/piper_voice/ local:./output/piper_voice/`
- [ ] Destroy instance immediately after download
- [ ] Verify `output/piper_voice/el_GR-joy-medium.onnx` exists and is non-zero

---

## 8. Decision Tree

```
Need to train Piper?
  YES → Run 6.1 local smoke test first
        → Then Vast.ai (Docker required; 24-48h local is too slow)

Need to train Gemma E2B?
  YES → Run 6.2 local smoke test first
        → Local (fits 16 GB VRAM); fall back to Vast.ai if issues

Need to train Gemma E4B?
  YES → Do NOT attempt local
        → Vast.ai A100 first
        → Colab Pro L4/A100 as fallback

Need to merge adapters or run Piper synthesis?
  → Local always
```

---

## 9. Current Smoke Notes

- The stronger validated Gemma smoke path is `200` pairs, not only the earlier `50`-sample text check.
- Text smoke: `prepare_qa_dataset.py` → `train_qa_local.py`
- Audio-Q&A smoke: `synthesize_qa_audio.py` → `prepare_stt_qa_dataset.py` → `train_stt_qa_local.py`
- The local smoke run now validates:
- base model resolution from `N:\.cache`
- JOY question-audio synthesis
- multimodal JSONL formatting
- one-step E2B audio-Q&A training
- Gemma smoke fixes should prefer repo-script patches over package upgrades or downgrades so the already-working Piper path is not destabilized.

*Canonical pipeline sequence: `PIPER_REPLACES_MOIRA_PLAN.md`. Update this document when hardware or budget decisions change.*
