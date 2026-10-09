# Licensing and attribution

The upstream StyleTTS2 implementation retains its original MIT license in `../LICENSE`, including Copyright (c) 2023 Aaron (Yinghao) Li. That permission is not revoked or replaced. Model weights, datasets and reference audio have their own terms; review their source pages before use.

The PolyForm Noncommercial 1.0.0 grant in `../LICENSE-API` applies only to Isaac Valeriano’s original API/infrastructure additions: `api_server.py`, `Dockerfile`, `compose.yml`, `docker-entrypoint.sh`, `prometheus.yml`, `grafana/`, `scripts/download_models.py`, and the new project-specific documentation/artwork under `docs/` (except the archived upstream README). Any pre-existing or third-party material in these files retains its own license.

Upstream code under `Modules/`, `Utils/`, the original model/training utilities, upstream README text, the vendored `vast.py` CLI, model weights and third-party dependencies are **not** claimed as Isaac’s original work or relicensed by `LICENSE-API`. Preserve their original notices and comply with their respective terms.

The original upstream README is archived at [STYLE_TTS2_UPSTREAM_README.md](STYLE_TTS2_UPSTREAM_README.md).

The repository is source available, not wholly under a single noncommercial license. Rights already granted under MIT remain available under MIT; commercial users cannot assume those permissions also cover Isaac’s new API layer.
