"""
Generate Greek WAVs with legacy moiralabs/GreekTTS (CSM-1B + LoRA), not Orpheus.

`generate_pairs.py` uses Orpheus-3B + GreekTTS-1.5. This script uses the **Hugging Face**
`CsmForConditionalGeneration` stack only — **do not `import unsloth` here**: Unsloth patches CSM
(`cache_position` vs `codebook_indices`) and breaks `depth_decoder`/`codebooks_head` during `generate()`.

Output:
  data/moira_original_greektts_sample/raw_audio/pair_NNNN.wav
  data/moira_original_greektts_sample/transcripts/pair_NNNN.txt

Env:
  ORIGINAL_MOIRA_OUT, ORIGINAL_MOIRA_DIR, NUM_PAIRS, CSM_MAX_NEW_TOKENS, CSM_SPEAKER_ID
  CSM_MERGE_LORA       — 1 = merge LoRA into base (default 0; merge often garbles CSM+audio)
  CSM_TORCH_DTYPE      — float32 | bfloat16 (default float32, closer to README dtype=None full precision)
  CSM_REPO_ID          — HF repo for base CSM (default unsloth/csm-1b)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from env_bootstrap import ensure_unsloth_runtime

ensure_unsloth_runtime(BASE)

from peft import PeftModel
from transformers import AutoProcessor, CsmForConditionalGeneration
from transformers.generation.utils import GenerationMixin


def _patch_generation_validate_for_csm() -> None:
    """Depth decoder generate(..., backbone_last_hidden_state=...) trips TF 5.5 kwargs validation."""

    _orig = GenerationMixin._validate_model_kwargs

    def _wrapped(self, model_kwargs):
        mk = dict(model_kwargs)
        mk.pop("backbone_last_hidden_state", None)
        return _orig(self, mk)

    GenerationMixin._validate_model_kwargs = _wrapped  # type: ignore[assignment]


_patch_generation_validate_for_csm()


def _patch_depth_decoder_prepare(model) -> None:
    """Avoid KeyError when position_ids omitted from model_inputs (CSM depth decoder)."""

    import types

    def _fixed(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        **kwargs,
    ):
        kwargs = dict(kwargs)
        kwargs.setdefault("is_first_iteration", is_first_iteration)
        fi = kwargs["is_first_iteration"]
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            next_sequence_length,
            past_key_values,
            attention_mask,
            inputs_embeds,
            **kwargs,
        )
        if not fi:
            model_inputs.pop("backbone_last_hidden_state", None)
        model_inputs.pop("position_ids", None)
        return model_inputs

    model.depth_decoder.prepare_inputs_for_generation = types.MethodType(_fixed, model.depth_decoder)  # type: ignore[method-assign]


def _audio_numpy(out) -> torch.Tensor:
    if hasattr(out, "audio") and out.audio is not None and len(out.audio) > 0:
        t = out.audio[0]
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        t = out[0]
    else:
        raise TypeError(f"Unexpected generate() return type: {type(out)!r}")
    if t.dim() > 1:
        t = t.squeeze(0)
    return t


def _load_gp_sentences(n: int) -> list[tuple[int, str]]:
    spec = importlib.util.spec_from_file_location("_gp", BASE / "training" / "generate_pairs.py")
    assert spec and spec.loader
    _gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_gp)
    lines = _gp.get_sentences(n)
    return [(i, lines[i]) for i in range(len(lines))]


def _csm_snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    repo = os.getenv("CSM_REPO_ID", "unsloth/csm-1b").strip() or "unsloth/csm-1b"
    try:
        return Path(snapshot_download(repo, local_files_only=True, token=token))
    except Exception:
        return Path(snapshot_download(repo, local_files_only=False, token=token))


def main() -> None:
    out_rel = os.getenv("ORIGINAL_MOIRA_OUT", "data/moira_original_greektts_sample").strip()
    adapter_rel = os.getenv("ORIGINAL_MOIRA_DIR", "original moira").strip()
    n = int(os.getenv("NUM_PAIRS", "4"))
    max_new = int(os.getenv("CSM_MAX_NEW_TOKENS", "250"))
    speaker_id = int(os.getenv("CSM_SPEAKER_ID", "0"))
    merge_lora = os.getenv("CSM_MERGE_LORA", "0").strip().lower() in ("1", "true", "yes")
    dtype_s = os.getenv("CSM_TORCH_DTYPE", "float32").strip().lower()
    torch_dtype = torch.bfloat16 if dtype_s == "bfloat16" else torch.float32
    token = os.getenv("HF_TOKEN") or None

    out_root = BASE / out_rel if not Path(out_rel).is_absolute() else Path(out_rel)
    raw_dir = out_root / "raw_audio"
    txt_dir = out_root / "transcripts"
    adapter_dir = BASE / adapter_rel if not Path(adapter_rel).is_absolute() else Path(adapter_rel)

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"[ERROR] Missing adapter at {adapter_dir}")
        sys.exit(1)

    for d in (raw_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Gemma4GR — Original GreekTTS (CSM-1B + LoRA, HF only)")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Out:     {out_root}")
    print(f"  Pairs:   {n} | speaker_id={speaker_id} | max_new_tokens={max_new}")
    print(f"  LoRA merge: {merge_lora} | dtype: {torch_dtype}")
    print("=" * 55)

    csm_dir = _csm_snapshot_dir()
    print(f"\n  Snapshot: {csm_dir}")
    print("  Loading CsmForConditionalGeneration (no Unsloth FastModel) ...")
    model = CsmForConditionalGeneration.from_pretrained(
        str(csm_dir),
        torch_dtype=torch_dtype,
        device_map=None,
        local_files_only=True,
        token=token,
    )
    model = model.to("cuda")
    model.tie_weights()

    print(f"  Loading LoRA from {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    if merge_lora:
        model = model.merge_and_unload()
    model.eval()
    _patch_depth_decoder_prepare(model)

    print("  Loading processor from snapshot ...")
    processor = AutoProcessor.from_pretrained(str(csm_dir), local_files_only=True, token=token)

    pairs = _load_gp_sentences(n)
    failed = 0
    for i, text in tqdm(pairs, desc="CSM+Moira"):
        name = f"pair_{i:04d}"
        prompt = f"[{speaker_id}]{text}"
        try:
            inputs = processor(prompt, add_special_tokens=True).to("cuda")
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    output_audio=True,
                )
            audio = _audio_numpy(audio_values).to(torch.float32).cpu().numpy()
            audio = np.asarray(audio).reshape(-1)
            peak = np.max(np.abs(audio)) if audio.size else 0.0
            if peak > 1.0:
                audio = audio / peak
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            sf.write(str(raw_dir / f"{name}.wav"), audio, 24000)
            (txt_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"\n  [WARN] {name}: {e}")
            failed += 1

    print(f"\nDone. OK: {n - failed} | Failed: {failed}")
    print(f"  WAV: {raw_dir}")
    print(f"  TXT: {txt_dir}")


if __name__ == "__main__":
    main()
