# TTSapi — verification boundary

Recorded 2026-10-09 UTC (2026-10-08 in Brazil) against baseline `342e52652c43cc6a5c060de6b12ace295b8392f1`, before documentation-only additions. Command: `python3 scripts/verify_api_surface.py`; Python 3.14 on macOS x86_64.

The verifier parsed the actual Python AST and dashboard JSON without importing PyTorch or downloading model weights. Result: **9 explicit HTTP method/path pairs, 4 output formats, 7 visualization panels** (3 additional section rows). [Raw machine-readable output](api-surface.json).

This is static configuration/contract validation, **not an HTTP integration test, generated audio sample, inference-quality test or benchmark**. No CUDA/GPU environment or working Docker daemon was available for a full runtime run. No latency, throughput, traffic or voice-quality result is published.

The dashboard image is generated from the versioned panel layout and titles, labeled as a configuration preview. It shows no synthetic telemetry masquerading as a real result. A future runtime report should include model/checkpoint provenance, device, warm-up, text length, diffusion settings, repeats, audio length and timing distribution. Review API security boundaries before exposing runtime benchmarks.
