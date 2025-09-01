#!/usr/bin/env python3
import os
import sys
import argparse
from huggingface_hub import snapshot_download

HF_REPOS = {
    "LJSpeech": "yl4579/StyleTTS2-LJSpeech",
    "LibriTTS": "yl4579/StyleTTS2-LibriTTS",
}

# We only pull the essential artifacts into Models/<name>/
ESSENTIALS = [
    "config.yml",
    "epochs_2nd_00020.pth",
]


def ensure_models_dir():
    os.makedirs("Models", exist_ok=True)


def download_model(name: str):
    if name not in HF_REPOS:
        print(f"Unknown model: {name}", file=sys.stderr)
        return 1
    target = os.path.join("Models", name)
    os.makedirs(target, exist_ok=True)

    # If essentials exist, skip
    if all(os.path.exists(os.path.join(target, f)) for f in ESSENTIALS):
        print(f"[ok] {name} already present, skipping")
        return 0

    repo = HF_REPOS[name]
    print(f"[dl] fetching from {repo} ...")
    local_dir = snapshot_download(repo_id=repo, allow_patterns=["*.yml", "*.pth", "*.pt", "*.bin", "*.json", "reference_audio.zip"], local_dir_use_symlinks=False)

    # Move essentials
    for f in ESSENTIALS:
        src = os.path.join(local_dir, f)
        if os.path.exists(src):
            dst = os.path.join(target, f)
            if not os.path.exists(dst):
                os.replace(src, dst)
    print(f"[ok] {name} ready at {target}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="LJSpeech,LibriTTS")
    args = ap.parse_args()
    ensure_models_dir()
    ret = 0
    for name in [s.strip() for s in args.models.split(',') if s.strip()]:
        ret |= download_model(name)
    sys.exit(ret)

if __name__ == "__main__":
    main()