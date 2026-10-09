# TTSapi delivery roadmap



- [x] HTTP synthesis, JWT bearer dependency and four-format output.
- [x] Device/model/style caching and process-local inference serialization.
- [x] Voice preset CRUD and ephemeral job submission.
- [x] Prometheus/GPU metric definitions and Grafana provisioning files.
- [x] Documented API surface and a reproducible, model-free static verifier.
- [ ] Add bounded workers, durable job state, result/status endpoints and cancellation.
- [ ] Enforce token expiry/claims, upload/text/parameter limits and safe voice paths.
- [ ] Disable/protect public benchmark compute; validate rate limits in multi-worker mode.
- [ ] Add HTTP integration tests and restart/concurrency scenarios with a fake inference backend.
- [ ] Pin runtime dependencies and validate CPU/GPU installation matrices.
- [ ] Run reproducible inference/quality benchmarks with hardware, audio samples and methodology.

[Detailed roadmap](ROADMAP.md).

## Attribution and licensing

StyleTTS2 was developed by **Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler and Nima Mesgarani**. [Original repository](https://github.com/yl4579/StyleTTS2) · [Research paper](https://arxiv.org/abs/2306.07691) · [Archived upstream README](STYLE_TTS2_UPSTREAM_README.md).

The original upstream code remains under **[MIT](../LICENSE)**. Isaac’s original API/infrastructure additions and new project-specific documentation/artwork use **[PolyForm Noncommercial 1.0.0](../LICENSE-API)**. See the exact [scope and third-party exclusions](LICENSING.md). Model weights, datasets, reference voices and dependencies have separate terms. Earlier MIT grants remain valid; this repository must not be presented as wholly relicensed.

Built by [Isaac Valeriano](https://github.com/yMScorpion). Behind every line of code, there is a builder.

## Release gate

A production-ready claim requires reproducible deployment, recovery tests, security boundaries and measured runtime behavior; source inspection alone is not sufficient.
