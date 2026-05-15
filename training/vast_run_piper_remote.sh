#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs output/piper_voice

echo "Gemma4GR Vast Piper remote run"
echo "Working directory: $ROOT_DIR"
echo "Python: $(python --version 2>&1)"
echo "Start time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python training/train_piper_vast.py
