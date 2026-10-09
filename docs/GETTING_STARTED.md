# TTSapi — getting started



The GPU-oriented Compose file expects an NVIDIA-compatible Docker runtime. It is not a turnkey CPU/macOS deployment. You need model configurations/checkpoints, upstream prerequisites (including espeak-ng, libsndfile and FFmpeg) and a compatible PyTorch installation.

```bash
git clone https://github.com/yMScorpion/TTSapi.git
cd TTSapi
# Lightweight verification: no model, GPU, API key or inference needed
python3 scripts/verify_api_surface.py

# Full runtime setup (read prerequisites first)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Supply Models/LJSpeech or Models/LibriTTS config.yml and checkpoints.
# See scripts/download_models.py and the archived upstream instructions.
export JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export AUTH_USERNAME="local-reviewer"
read -s -p 'Local API password: ' AUTH_PASSWORD; export AUTH_PASSWORD
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Open `/docs` on the local service for generated OpenAPI. [Getting started](GETTING_STARTED.md) documents GPU Compose, permissions and model paths; [API reference](API.md) describes endpoints and limitations. The default credentials in existing code/config are development placeholders and **must be replaced**.


## Reproduction record

Use [VERIFICATION.md](VERIFICATION.md) to compare the code revision, toolchain and command. Never store real credentials in tracked files.
