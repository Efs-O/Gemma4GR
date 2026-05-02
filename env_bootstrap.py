"""
Helpers for selecting a CUDA-capable Unsloth Python interpreter on Windows.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


UNSLOTH_PYTHON_ENV = "GEMMA4GR_UNSLOTH_PYTHON"
ACTIVE_UNSLOTH_PYTHON_ENV = "GEMMA4GR_ACTIVE_UNSLOTH_PYTHON"


def _probe_current_python() -> dict[str, object]:
    torch_version = None
    cuda_available = False
    cuda_version = None
    unsloth_available = importlib.util.find_spec("unsloth") is not None

    try:
        import torch  # type: ignore[import]

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        pass

    return {
        "python": sys.executable,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "unsloth_available": unsloth_available,
    }


def _probe_python(python_exe: Path) -> dict[str, object] | None:
    if not python_exe.exists():
        return None

    probe = (
        "import importlib.util, json, sys\n"
        "data = {\n"
        "  'python': sys.executable,\n"
        "  'torch_version': None,\n"
        "  'cuda_available': False,\n"
        "  'cuda_version': None,\n"
        "  'unsloth_available': importlib.util.find_spec('unsloth') is not None,\n"
        "}\n"
        "try:\n"
        "  import torch\n"
        "  data['torch_version'] = getattr(torch, '__version__', None)\n"
        "  data['cuda_available'] = bool(torch.cuda.is_available())\n"
        "  data['cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', None)\n"
        "except Exception:\n"
        "  pass\n"
        "print(json.dumps(data))\n"
    )

    try:
        result = subprocess.run(
            [str(python_exe), "-c", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    stdout = result.stdout.strip()
    if not stdout:
        return None

    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return None


def _candidate_pythons(base_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    env_python = os.getenv(UNSLOTH_PYTHON_ENV)
    if env_python:
        candidates.append(Path(env_python))

    candidates.extend(
        [
            base_dir / ".venv" / "Scripts" / "python.exe",
            base_dir.parent / "ADVANCOPY230425" / ".venv" / "Scripts" / "python.exe",
            base_dir.parent / "ADVANLLM" / ".venv" / "Scripts" / "python.exe",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_unsloth_interpreter(base_dir: Path) -> tuple[Path | None, dict[str, object] | None]:
    for candidate in _candidate_pythons(base_dir):
        info = _probe_python(candidate)
        if not info:
            continue
        if info.get("unsloth_available") and info.get("cuda_available"):
            return candidate, info
    return None, None


def ensure_unsloth_runtime(base_dir: Path) -> None:
    current = _probe_current_python()
    if current["unsloth_available"] and current["cuda_available"]:
        return

    active_python = os.getenv(ACTIVE_UNSLOTH_PYTHON_ENV)
    if active_python and Path(active_python).resolve() == Path(sys.executable).resolve():
        details = (
            f"python={sys.executable} torch={current['torch_version']} "
            f"cuda={current['cuda_available']} unsloth={current['unsloth_available']}"
        )
        raise RuntimeError(f"Active Unsloth interpreter is still not usable: {details}")

    candidate, info = find_unsloth_interpreter(base_dir)
    if candidate is None or info is None:
        details = (
            f"current python={sys.executable}, torch={current['torch_version']}, "
            f"cuda={current['cuda_available']}, unsloth={current['unsloth_available']}"
        )
        raise RuntimeError(
            "No CUDA-capable Unsloth interpreter found. "
            f"Checked {len(_candidate_pythons(base_dir))} candidate locations. {details}"
        )

    env = os.environ.copy()
    env[ACTIVE_UNSLOTH_PYTHON_ENV] = str(candidate)
    print(
        "[INFO] Switching to CUDA Unsloth interpreter:",
        info["python"],
        f"(torch={info['torch_version']}, cuda={info['cuda_version']})",
    )
    result = subprocess.run([str(candidate), *sys.argv], env=env, check=False)
    raise SystemExit(result.returncode)


def select_python_for_script(base_dir: Path, script_path: Path) -> tuple[str, str | None]:
    unsloth_scripts = {
        "train_e2b_local.py",
        "train_qa_local.py",
        "merge_adapters.py",
    }
    if script_path.name not in unsloth_scripts:
        return sys.executable, None

    current = _probe_current_python()
    if current["unsloth_available"] and current["cuda_available"]:
        return sys.executable, None

    candidate, info = find_unsloth_interpreter(base_dir)
    if candidate and info:
        reason = (
            f"using CUDA Unsloth env at {info['python']} "
            f"(torch={info['torch_version']}, cuda={info['cuda_version']})"
        )
        return str(candidate), reason

    reason = (
        "no CUDA-capable Unsloth interpreter found; "
        f"staying on {sys.executable} (torch={current['torch_version']}, cuda={current['cuda_available']})"
    )
    return sys.executable, reason
