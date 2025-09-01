# API server for StyleTTS2
# Notes: keep minimal, reuse project utilities, and follow existing code style.

import os
import io
import uuid
import time
import logging
from typing import Optional, Literal

import torch
import torchaudio
# removed unused librosa import
from pydub import AudioSegment

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError

import yaml
import soundfile as sf

from Utils.PLBERT.util import load_plbert
from models import load_ASR_models, load_F0_models, build_model
from utils import recursive_munch
# Observability and metrics
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import pynvml
from contextlib import nullcontext
import threading

# Security settings (simple JWT bearer). In production, integrate with real OAuth2 provider.
OAUTH2_SCHEME = OAuth2PasswordBearer(tokenUrl="/auth/token")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin")

# Light runtime optimizations
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
except Exception:
    pass

# Simple in-memory job store
JOBS = {}

# Rate limiting (very simple token bucket per IP/user). For production, move to Redis.
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "10"))
RATE_STATE = {}

# Supported output formats
AudioFormat = Literal["wav", "mp3", "flac", "ogg"]

# Metrics toggle and GPU lock
PROM_ENABLED = os.getenv("PROMETHEUS", "1").lower() in ("1", "true", "yes")
NVML_OK = True
GPU_LOCK = threading.Semaphore(1)

app = FastAPI(title="StyleTTS2 API", version="1.2")

# Metrics setup
GPU_MEM_TOTAL = Gauge("gpu_memory_total_bytes", "Total GPU memory in bytes") if PROM_ENABLED else None
GPU_MEM_USED = Gauge("gpu_memory_used_bytes", "Used GPU memory in bytes") if PROM_ENABLED else None
GPU_UTIL = Gauge("gpu_utilization_percent", "GPU utilization percent") if PROM_ENABLED else None
GPU_TEMP = Gauge("gpu_temperature_celsius", "GPU temperature in Celsius") if PROM_ENABLED else None

if PROM_ENABLED and NVML_OK:
    try:
        pynvml.nvmlInit()
    except Exception:
        NVML_OK = False

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def now_ms():
    return int(time.time() * 1000)


class TTSService:
    def __init__(self):
        self.device = get_device()
        self.models_cache = {}
        self.global_phonemizer = None
        self.style_cache = {}
        self.enable_amp = os.getenv("ENABLE_AMP", "true").lower() in ("1", "true", "yes") and self.device == "cuda"
        self.voices_dir = os.getenv("VOICES_DIR", os.path.join("Voices"))
        os.makedirs(self.voices_dir, exist_ok=True)

    def _ensure_phonemizer(self):
        if self.global_phonemizer is None:
            import phonemizer
            self.global_phonemizer = phonemizer.backend.EspeakBackend(
                language='en-us', preserve_punctuation=True, with_stress=True
            )

    def load_model(self, model_name: Literal["LJSpeech", "LibriTTS"], base_dir: str = "Models"):
        key = (model_name, self.device)
        if key in self.models_cache:
            return self.models_cache[key]

        model_dir = os.path.join(base_dir, model_name)
        config_path = os.path.join(model_dir, "config.yml")
        ckpt_path = None
        # Heuristic default checkpoints
        if model_name == "LibriTTS":
            ckpt_path = os.path.join(model_dir, "epochs_2nd_00020.pth")
        else:
            # LJSpeech usually has epochs_2nd_00020.pth too
            ckpt_path = os.path.join(model_dir, "epochs_2nd_00020.pth")
        if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
            raise RuntimeError(f"Missing model files at {model_dir}")

        config = yaml.safe_load(open(config_path, "r"))
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

        model_params = recursive_munch(config['model_params'])
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        # Load weights
        params_whole = torch.load(ckpt_path, map_location='cpu')
        params = params_whole['net']
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except Exception:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    model[key].load_state_dict(new_state_dict, strict=False)
        _ = [model[key].eval() for key in model]

        # Sampler
        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        self.models_cache[key] = {
            "config": config,
            "model": model,
            "sampler": sampler,
        }
        return self.models_cache[key]

    def compute_style(self, wav: torch.Tensor, sr: int, model_obj):
        # Expect mono waveform tensor (T,) in float32
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4
        mel = to_mel(wav)
        mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std
        with torch.no_grad():
            ref_s = model_obj.style_encoder(mel.unsqueeze(1).to(self.device))
            ref_p = model_obj.predictor_encoder(mel.unsqueeze(1).to(self.device))
        return torch.cat([ref_s, ref_p], dim=1)

    def load_voice(self, name: str):
        if name in self.style_cache:
            return self.style_cache[name]
        path = os.path.join(self.voices_dir, f"{name}.pt")
        if not os.path.exists(path):
            raise HTTPException(404, detail="Voice not found")
        emb = torch.load(path, map_location="cpu")
        self.style_cache[name] = emb
        return emb

    def save_voice(self, name: str, wav: torch.Tensor, sr: int, model_obj):
        emb = self.compute_style(wav, sr, model_obj)
        path = os.path.join(self.voices_dir, f"{name}.pt")
        torch.save(emb.cpu(), path)
        self.style_cache[name] = emb.cpu()
        return path

    def list_voices(self):
        voices = []
        for fn in os.listdir(self.voices_dir):
            if fn.endswith('.pt'):
                voices.append(os.path.splitext(fn)[0])
        return sorted(voices)

    def synthesize(self,
                   text: str,
                   model_name: Literal["LJSpeech", "LibriTTS"],
                   diffusion_steps: int = 5,
                   embedding_scale: float = 1.0,
                   alpha: float = 0.3,
                   beta: float = 0.7,
                   reference_audio: Optional[torch.Tensor] = None,
                   reference_sr: Optional[int] = None,
                   output_format: AudioFormat = "wav",
                   voice_name: Optional[str] = None,
                   ) -> bytes:
        bundle = self.load_model(model_name)
        model = bundle["model"]
        sampler = bundle["sampler"]
        self._ensure_phonemizer()

        # Phonemize
        from nltk import word_tokenize
        text = text.strip().replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        from text_utils import TextCleaner
        textclenaer = TextCleaner()
        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        amp_ctx = torch.cuda.amp.autocast(enabled=self.enable_amp, dtype=torch.float16) if self.device == "cuda" else nullcontext()
        # fallback context when CUDA not available
        try:
            from contextlib import nullcontext
        except Exception:
            class nullcontext:
                def __enter__(self): return None
                def __exit__(self, *args): return False

        with torch.inference_mode():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            from utils import length_to_mask
            text_mask = length_to_mask(input_lengths).to(self.device)
            with amp_ctx:
                t_en = model.text_encoder(tokens, input_lengths, text_mask)
                bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                if model_name == "LibriTTS":
                    if voice_name is not None:
                        ref_s = self.load_voice(voice_name).to(self.device)
                    else:
                        if reference_audio is None:
                            raise HTTPException(400, detail="LibriTTS requires reference_audio or voice name")
                        ref_s = self.compute_style(reference_audio.to(self.device), reference_sr, model)
                    s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                     embedding=bert_dur,
                                     embedding_scale=embedding_scale,
                                     features=ref_s,
                                     num_steps=diffusion_steps).squeeze(1)
                    s = s_pred[:, 128:]
                    ref = s_pred[:, :128]
                    ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
                    s = beta * s + (1 - beta) * ref_s[:, 128:]
                else:
                    noise = torch.randn(1, 1, 256).to(self.device)
                    s_pred = sampler(noise, embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                                     embedding_scale=embedding_scale).squeeze(0)
                    s = s_pred[:, 128:]
                    ref = s_pred[:, :128]

                d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
                x, _ = model.predictor.lstm(d)
                duration = model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze()).clamp(min=1)
                if model_name == "LJSpeech":
                    pred_dur[-1] += 5

                pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
                c_frame = 0
                for i in range(pred_aln_trg.size(0)):
                    pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                    c_frame += int(pred_dur[i].data)

                en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
                if bundle["config"]["model_params"]["decoder"]["type"] == "hifigan":
                    asr_new = torch.zeros_like(en)
                    asr_new[:, :, 0] = en[:, :, 0]
                    asr_new[:, :, 1:] = en[:, :, 0:-1]
                    en = asr_new

                F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

                asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
                if bundle["config"]["model_params"]["decoder"]["type"] == "hifigan":
                    asr_new = torch.zeros_like(asr)
                    asr_new[:, :, 0] = asr[:, :, 0]
                    asr_new[:, :, 1:] = asr[:, :, 0:-1]
                    asr = asr_new

                out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
                wav = out.squeeze().detach().cpu().numpy()
                wav = wav[..., : -50] if model_name == "LibriTTS" else wav

        # Encode to requested format at 24k default
        target_sr = 24000
        if output_format in ("wav", "flac"):
            buf = io.BytesIO()
            fmt = "WAV" if output_format == "wav" else "FLAC"
            subtype = "PCM_16" if output_format == "wav" else None
            sf.write(buf, wav, target_sr, format=fmt, subtype=subtype)
            buf.seek(0)
            return buf.getvalue()
        elif output_format in ("mp3", "ogg"):
            import numpy as np
            from pydub import AudioSegment
            audio_i16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("int16")
            seg = AudioSegment(
                audio_i16.tobytes(), frame_rate=target_sr, sample_width=2, channels=1
            )
            buf = io.BytesIO()
            seg.export(buf, format=output_format if output_format != "ogg" else "ogg")
            return buf.getvalue()
        else:
            raise HTTPException(400, detail="Unsupported format")


SERVICE = TTSService()


def require_auth(token: str = Depends(OAUTH2_SCHEME)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def check_rate_limit(identity: str):
    # identity can be user id from JWT or client ip
    now = time.time()
    bucket = RATE_STATE.setdefault(identity, [])
    # drop entries older than 60s
    RATE_STATE[identity] = [t for t in bucket if now - t < 60]
    if len(RATE_STATE[identity]) >= RATE_LIMIT_RPM:
        raise HTTPException(429, detail="Rate limit exceeded: 10 req/min")
    RATE_STATE[identity].append(now)


@app.post("/v1/tts")
async def tts(
    text: str = Form(...),
    model: Literal["LJSpeech", "LibriTTS"] = Form("LibriTTS"),
    diffusion_steps: int = Form(5),
    embedding_scale: float = Form(1.0),
    alpha: float = Form(0.3),
    beta: float = Form(0.7),
    output_format: AudioFormat = Form("wav"),
    voice: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    token: dict = Depends(require_auth),
):
    check_rate_limit(identity=str(token.get("sub", "anon")))

    ref_wav = None
    ref_sr = None
    if reference_audio is not None:
        data = await reference_audio.read()
        wav, sr = torchaudio.load(io.BytesIO(data))
        wav = wav.mean(dim=0)  # mono
        ref_wav = wav
        ref_sr = sr

    # serialize GPU work
    with GPU_LOCK:
        audio_bytes = SERVICE.synthesize(
            text=text,
            model_name=model,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            alpha=alpha,
            beta=beta,
            reference_audio=ref_wav,
            reference_sr=int(ref_sr) if ref_sr is not None else None,
            output_format=output_format,
            voice_name=voice,
        )

    media_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }[output_format]
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)


@app.post("/v1/jobs")
async def create_job(
    text: str = Form(...),
    model: Literal["LJSpeech", "LibriTTS"] = Form("LibriTTS"),
    diffusion_steps: int = Form(5),
    embedding_scale: float = Form(1.0),
    alpha: float = Form(0.3),
    beta: float = Form(0.7),
    output_format: AudioFormat = Form("wav"),
    voice: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    token: dict = Depends(require_auth),
):
    check_rate_limit(identity=str(token.get("sub", "anon")))

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None, "format": output_format}

    import threading
    def worker():
        try:
            JOBS[job_id]["status"] = "running"
            ref_wav = None
            ref_sr = None
            if reference_audio is not None:
                data = reference_audio.file.read()
                wav, sr = torchaudio.load(io.BytesIO(data))
                wav = wav.mean(dim=0)
                ref_wav = wav
                ref_sr = sr
            with GPU_LOCK:
                audio_bytes = SERVICE.synthesize(
                    text=text,
                    model_name=model,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                    alpha=alpha,
                    beta=beta,
                    reference_audio=ref_wav,
                    reference_sr=int(ref_sr) if ref_sr is not None else None,
                    output_format=output_format,
                    voice_name=voice,
                )
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = audio_bytes
        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
    threading.Thread(target=worker, daemon=True).start()

    return {"job_id": job_id, "status": "queued"}

# Voice presets management
@app.get("/v1/voices")
async def list_voices(token: dict = Depends(require_auth)):
    return {"voices": SERVICE.list_voices()}

@app.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    model: Literal["LJSpeech", "LibriTTS"] = Form("LibriTTS"),
    reference_audio: UploadFile = File(...),
    token: dict = Depends(require_auth),
):
    data = await reference_audio.read()
    wav, sr = torchaudio.load(io.BytesIO(data))
    wav = wav.mean(dim=0)
    bundle = SERVICE.load_model(model)
    with GPU_LOCK:
        path = SERVICE.save_voice(name, wav, int(sr), bundle["model"])
    return {"name": name, "path": path}

@app.delete("/v1/voices/{name}")
async def delete_voice(name: str, token: dict = Depends(require_auth)):
    path = os.path.join(SERVICE.voices_dir, f"{name}.pt")
    if os.path.exists(path):
        os.remove(path)
        SERVICE.style_cache.pop(name, None)
        return {"deleted": name}
    raise HTTPException(404, detail="Voice not found")

# Simple benchmarking endpoint
@app.post("/v1/bench")
async def bench(
    text: str = Form("Benchmarking one short sentence."),
    model: Literal["LJSpeech", "LibriTTS"] = Form("LJSpeech"),
    steps_list: str = Form("3,5,8"),
    repeats: int = Form(2),
    token: dict = Depends(require_auth),
):
    import statistics
    steps = [int(s.strip()) for s in steps_list.split(',') if s.strip()]
    timings = {}
    for steps_i in steps:
        durs = []
        for _ in range(repeats):
            start = time.time()
            with GPU_LOCK:
                _ = SERVICE.synthesize(
                    text=text,
                    model_name=model,
                    diffusion_steps=steps_i,
                    embedding_scale=1.0,
                    output_format="wav",
                )
            durs.append(time.time() - start)
        timings[str(steps_i)] = {
            "avg_sec": statistics.mean(durs),
            "min_sec": min(durs),
            "max_sec": max(durs),
        }
    return {"model": model, "results": timings}

@app.get("/v1/bench/public")
async def bench_public(
    text: str = "Benchmarking one short sentence.",
    model: Literal["LJSpeech", "LibriTTS"] = "LJSpeech",
    steps_list: str = "3,5,8",
    repeats: int = 1,
):
    import statistics
    steps = [int(s.strip()) for s in steps_list.split(',') if s.strip()]
    rows = []
    for steps_i in steps:
        durs = []
        for _ in range(repeats):
            start = time.time()
            with GPU_LOCK:
                _ = SERVICE.synthesize(
                    text=text,
                    model_name=model,
                    diffusion_steps=steps_i,
                    embedding_scale=1.0,
                    output_format="wav",
                )
            durs.append(time.time() - start)
        rows.append({
            "steps": steps_i,
            "avg_sec": float(sum(durs)/len(durs)),
            "min_sec": float(min(durs)),
            "max_sec": float(max(durs)),
        })
    return {"model": model, "steps": rows}


@app.get("/healthz")
async def healthz():
    try:
        device = get_device()
        return {"status": "ok", "device": device}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# Entry for uvicorn: uvicorn api_server:app --host 0.0.0.0 --port 8000

@app.post("/auth/token")
async def issue_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != AUTH_USERNAME or form_data.password != AUTH_PASSWORD:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    payload = {"sub": form_data.username, "iat": int(time.time())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"access_token": token, "token_type": "bearer"}


# Metrics instrumentation
if PROM_ENABLED:
    Instrumentator().instrument(app).expose(app)

# Periodic GPU metrics collector (best-effort)
if PROM_ENABLED and NVML_OK:
    import threading
    def _update_gpu_metrics():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            GPU_MEM_TOTAL.set(mem.total)
            GPU_MEM_USED.set(mem.used)
            GPU_UTIL.set(util.gpu)
            GPU_TEMP.set(temp)
        except Exception:
            pass
        finally:
            threading.Timer(5.0, _update_gpu_metrics).start()
    _update_gpu_metrics()