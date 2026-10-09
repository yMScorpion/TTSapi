# TTSapi HTTP contract

The application uses multipart form fields; `/docs` supplies runtime OpenAPI after dependencies are installed. This table is source-derived, not a claim that an inference server was exercised here.

| Method | Path | Auth | Contract / limitation |
|---|---|---|---|
| POST | `/auth/token` | Username/password form | Returns JWT; existing token generation has no expiry claim |
| POST | `/v1/tts` | Bearer | Text, model, diffusion settings, format; optional voice/reference audio; audio response |
| POST | `/v1/jobs` | Bearer | Same input; returns ID and queued status; state in memory |
| GET | `/v1/voices` | Bearer | Lists local presets |
| POST | `/v1/voices` | Bearer | Name, model and reference audio; saves style preset |
| DELETE | `/v1/voices/{name}` | Bearer | Removes local preset; path/authorization hardening remains |
| POST | `/v1/bench` | Bearer | Runs inference timings; consumes model compute |
| GET | `/v1/bench/public` | None | Also runs inference; do not expose unrestricted |
| GET | `/healthz` | None | Status/device detection; not proof that all model weights are loaded |

Instrumentator may add `/metrics` at startup. Output formats: WAV, MP3, FLAC, OGG. Models: upstream LJSpeech and LibriTTS. Sample-rate encoding is 24 kHz in the current service.

## Example local call

After configuring credentials and obtaining a token, set `TOKEN` locally:

```bash
curl --fail http://127.0.0.1:8000/v1/tts \
  -H "Authorization: Bearer $TOKEN" \
  -F 'text=Behind every line of code, there is a builder.' \
  -F 'model=LJSpeech' -F 'output_format=wav' --output speech.wav
```

No status/result retrieval endpoint exists for `/v1/jobs`. Request/voice ownership, parameter bounds, upload sizes, cancellation and durable state require further implementation. StreamingResponse transports completed encoded bytes; it is not incremental neural audio generation.
