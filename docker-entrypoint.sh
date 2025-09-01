#!/usr/bin/env bash
set -euo pipefail

# Optional: auto-download pretrained model folders into ./Models if missing
if [[ "${MODEL_AUTO_DOWNLOAD:-1}" == "1" ]]; then
  python3 scripts/download_models.py --models "${MODELS:-LJSpeech,LibriTTS}" || echo "[warn] model auto download failed (ok if mounted)"
fi

# Run API
exec python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --no-access-log