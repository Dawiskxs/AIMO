"""
tts.py
----
Multi-backend TTS integration (Kokoro + Piper) with word-level timestamp extraction.

When the orchestrator gets a turn from the AI, it passes the speech text
through synthesize().  The function returns:
  - a path to the generated audio file (WAV)
  - a list of word-level timing objects
  - a list of viseme (mouth shape) events for avatar lip-sync

The UI uses the timing data to highlight each word exactly when it is
spoken, keeping text display and audio perfectly in sync.  The viseme
track drives the 3D avatar's mouth.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# numpy / soundfile / kokoro / piper are imported lazily inside synthesize() so the
# module (and its cache/config helpers) import cleanly in builds that ship
# without the optional TTS dependencies.  See requirements-tts.txt.

log = logging.getLogger(__name__)

# -- Configuration -----------------------------------------------------------
# Kokoro defaults
KOKORO_SAMPLE_RATE = 24_000

# Defaults come from config so that direct synthesize() calls and the
# settings-driven path agree.  config already applies the *_* env
# overrides; duplicating os.getenv() here would silently diverge from it.
try:
    import config as _config
    KOKORO_VOICE = _config.KOKORO_VOICE
    KOKORO_SPEED = _config.KOKORO_SPEED
    KOKORO_DEVICE = _config.KOKORO_DEVICE
    PIPER_VOICE = _config.PIPER_VOICE
    PIPER_SPEED = _config.PIPER_SPEED
    PIPER_NOISE_SCALE = _config.PIPER_NOISE_SCALE
    PIPER_NOISE_W = _config.PIPER_NOISE_W
    PIPER_SENTENCE_SILENCE = _config.PIPER_SENTENCE_SILENCE
    PIPER_MODELS_DIR = _config.PIPER_MODELS_DIR
except Exception:  # noqa: BLE001 - keep tts.py importable standalone
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
    KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
    KOKORO_DEVICE = os.getenv("KOKORO_DEVICE", "auto")  # auto | cpu | cuda
    PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
    PIPER_SPEED = float(os.getenv("PIPER_SPEED", "1.0"))
    PIPER_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
    PIPER_NOISE_W = float(os.getenv("PIPER_NOISE_W", "0.8"))
    PIPER_SENTENCE_SILENCE = float(os.getenv("PIPER_SENTENCE_SILENCE", "0.0"))
    PIPER_MODELS_DIR = Path(os.getenv("AIMEE_DATA_DIR", str(Path.home() / ".config" / "aimee"))) / "models" / "piper"

AUDIO_CACHE_DIR = Path(tempfile.gettempdir()) / "aimee_tts"

# Backend selection
TTS_BACKEND = "kokoro"  # "kokoro" | "piper"

# Language-specific voice mappings
KOKORO_VOICES = {
    "en": "af_heart",
    "cs": "cf_czech",  # Czech voice for Kokoro (if available)
}

PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "cs": "cs_CZ-jirka-medium",  # Czech voice for Piper
}


def _detect_language(text: str) -> str:
    """Simple language detection based on character patterns."""
    # Czech-specific characters
    czech_chars = set('ěščřžýáíéúůťďňóĚŠČŘŽÝÁÍÉÚŮŤĎŇÓ')
    if any(c in czech_chars for c in text):
        return "cs"
    # Common Czech words
    czech_words = {'je', 'to', 'a', 'v', 'z', 'na', 'pro', 's', 'o', 'jako', 'tak', 'že', 'by', 'si', 'mi', 'ti', 'ho', 'mu', 'jeho', 'její', 'jejich', 'my', 'vy', 'oni', 'jsem', 'jsi', 'jsme', 'jste', 'jsou', 'byl', 'byla', 'bylo', 'byli', 'mám', 'máš', 'má', 'máme', 'máte', 'mají', 'dělá', 'děláš', 'děláme', 'dělají', 'půjde', 'půjdu', 'půjdeš', 'chci', 'chceš', 'chce', 'chceme', 'chcete', 'chtějí'}
    words = set(text.lower().split())
    if words & czech_words:
        return "cs"
    return "en"


def _select_voice_for_language(language: str) -> str:
    """Select appropriate voice for the given language and current backend."""
    if TTS_BACKEND == "piper":
        return PIPER_VOICES.get(language, PIPER_VOICES["en"])
    else:
        return KOKORO_VOICES.get(language, KOKORO_VOICES["en"])

# Lazy-import pipelines so the app can start even if packages are not installed.
_kokoro_pipeline = None
_kokoro_pipeline_device: str | None = None
_piper_voice = None
_piper_voice_name: str | None = None


def configure(
    voice: str | None = None,
    speed: float | None = None,
    device: str | None = None,
    backend: str | None = None,
    # Piper-specific
    piper_voice: str | None = None,
    piper_speed: float | None = None,
    piper_noise_scale: float | None = None,
    piper_noise_w: float | None = None,
    piper_sentence_silence: float | None = None,
) -> None:
    """Update default voice/speed/device from settings (live)."""
    global KOKORO_VOICE, KOKORO_SPEED, KOKORO_DEVICE, _kokoro_pipeline, _kokoro_pipeline_device
    global TTS_BACKEND, PIPER_VOICE, PIPER_SPEED, PIPER_NOISE_SCALE, PIPER_NOISE_W, PIPER_SENTENCE_SILENCE
    global _piper_voice, _piper_voice_name

    # Backend switch
    if backend:
        new_backend = str(backend).lower()
        if new_backend in ("kokoro", "piper") and new_backend != TTS_BACKEND:
            TTS_BACKEND = new_backend

    # Kokoro settings
    if voice:
        KOKORO_VOICE = str(voice)
    if speed is not None:
        try:
            KOKORO_SPEED = float(speed)
        except (TypeError, ValueError):
            pass
    if device:
        new_device = str(device).lower()
        if new_device != KOKORO_DEVICE:
            KOKORO_DEVICE = new_device
            if _kokoro_pipeline_device is not None and _resolve_kokoro_device() != _kokoro_pipeline_device:
                _kokoro_pipeline = None
                _kokoro_pipeline_device = None

    # Piper settings
    if piper_voice:
        new_voice = str(piper_voice)
        if new_voice != PIPER_VOICE:
            PIPER_VOICE = new_voice
            _piper_voice = None
            _piper_voice_name = None
    if piper_speed is not None:
        try:
            PIPER_SPEED = float(piper_speed)
        except (TypeError, ValueError):
            pass
    if piper_noise_scale is not None:
        try:
            PIPER_NOISE_SCALE = float(piper_noise_scale)
        except (TypeError, ValueError):
            pass
    if piper_noise_w is not None:
        try:
            PIPER_NOISE_W = float(piper_noise_w)
        except (TypeError, ValueError):
            pass
    if piper_sentence_silence is not None:
        try:
            PIPER_SENTENCE_SILENCE = float(piper_sentence_silence)
        except (TypeError, ValueError):
            pass


def _resolve_kokoro_device() -> str:
    if KOKORO_DEVICE == "cpu":
        return "cpu"
    if KOKORO_DEVICE == "cuda":
        return "cuda"
    return "cuda" if _cuda_available() else "cpu"


def _get_kokoro_pipeline(language: str = "en"):
    """Lazy initialiser for the Kokoro KPipeline."""
    global _kokoro_pipeline, _kokoro_pipeline_device
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Kokoro TTS is not installed. Install TTS extras with: "
            "pip install -r requirements-tts.txt"
        ) from exc

    device = _resolve_kokoro_device()
    # Map language to Kokoro lang_code
    lang_code_map = {
        "en": "a",      # American English
        "cs": "c",      # Czech (if supported)
    }
    lang_code = lang_code_map.get(language, "a")
    _kokoro_pipeline = KPipeline(lang_code=lang_code, device=device)
    _kokoro_pipeline_device = device
    log.info("Kokoro pipeline initialised on %s (voice=%s, lang_code=%s)", device, KOKORO_VOICE, lang_code)
    return _kokoro_pipeline


def _get_piper_voice():
    """Lazy initialiser for Piper voice."""
    global _piper_voice, _piper_voice_name
    if _piper_voice is not None and _piper_voice_name == PIPER_VOICE:
        return _piper_voice

    try:
        from piper.voice import PiperVoice
    except ImportError as exc:
        raise RuntimeError(
            "Piper TTS is not installed. Install TTS extras with: "
            "pip install -r requirements-tts.txt"
        ) from exc

    model_path = PIPER_MODELS_DIR / f"{PIPER_VOICE}.onnx"
    config_path = PIPER_MODELS_DIR / f"{PIPER_VOICE}.onnx.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper model not found: {model_path}. "
            f"Download voice models to {PIPER_MODELS_DIR}/"
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Piper config not found: {config_path}"
        )

    _piper_voice = PiperVoice.load(str(model_path), str(config_path))
    _piper_voice_name = PIPER_VOICE
    log.info("Piper voice loaded: %s (sample_rate=%d)", PIPER_VOICE, _piper_voice.config.sample_rate)
    return _piper_voice


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# -- Sentence chunking -------------------------------------------------------
#
# Kokoro is StyleTTS2: it predicts ONE style vector and ONE duration track per
# chunk it is handed.  Feed it a whole paragraph and the terminal contours --
# the fall on ".", the rise on "?" -- get averaged across every sentence in it,
# which is what makes long turns read flat and rushed.  Splitting per sentence
# gives each one its own contour, at the cost of one extra forward pass each.
#
# The lookbehind keeps the punctuation attached to the sentence it ends; only
# the whitespace after it is consumed by re.split.
_SPLIT_PATTERN = r"(?<=[.!?…])\s+|\n+"

# Extra trailing silence per terminal punctuation, in ms at speed 1.0.  These
# are ON TOP of the lead-in/tail-out padding Kokoro puts around every chunk,
# measured at ~525ms between sentences on its own -- so keep them small.  Much
# past this and short fragments ("Deepening. Slowly.") start reading stilted
# rather than deliberate.  Ordering follows the measured pause ranking: a
# question holds longest, a comma barely at all.
_GAP_MS = {"?": 220, "…": 180, "!": 120, ".": 90, ",": 0}
_DEFAULT_GAP_MS = 40


def _sentence_gaps(results, speed: float) -> list[int]:
    """
    Silence to append after each result chunk, in samples, aligned 1:1 with
    *results*.  Scaled by 1/speed so the pauses stay in proportion when the
    whole delivery is slowed down.  The final chunk gets none -- trailing dead
    air just delays the next turn.
    """
    gaps: list[int] = []
    for result in results:
        if getattr(result, "audio", None) is None:
            gaps.append(0)
            continue
        graphemes = (getattr(result, "graphemes", "") or "").rstrip()
        # An ASCII "..." is the same held pause as a real ellipsis.
        if graphemes.endswith("..."):
            ms = _GAP_MS["…"]
        elif graphemes:
            ms = _GAP_MS.get(graphemes[-1], _DEFAULT_GAP_MS)
        else:
            ms = _DEFAULT_GAP_MS
        gaps.append(round(KOKORO_SAMPLE_RATE * ms / 1000 / max(speed, 0.1)))

    if gaps:
        gaps[-1] = 0
    return gaps


# -- Public API --------------------------------------------------------------

def synthesize(text: str, voice: str | None = None, speed: float | None = None, language: str | None = None) -> dict[str, Any]:
    """
    Convert *text* to speech using selected backend (Kokoro or Piper).

    Returns a dict with keys:
        audio_url   -- relative URL the frontend can fetch (/api/tts/audio/<id>)
        audio_path  -- absolute filesystem path to the WAV file
        words       -- list of {"word", "phonemes", "start_ms", "end_ms"}
        visemes     -- list of {"t_ms", "dur_ms", "viseme", "weight"} for lip-sync
        duration_ms -- total audio duration in milliseconds

    The returned *words* list is already ordered and non-overlapping.
    """
    if not text or not text.strip():
        return {"audio_url": None, "audio_path": None, "words": [],
                "visemes": [], "duration_ms": 0}

    # Auto-detect language if not provided
    if language is None:
        language = _detect_language(text)

    # Auto-select voice based on language if not explicitly provided
    if voice is None:
        voice = _select_voice_for_language(language)

    # Dispatch to backend
    if TTS_BACKEND == "piper":
        return _synthesize_piper(text, voice, speed)
    else:
        return _synthesize_kokoro(text, voice, speed)


def _synthesize_kokoro(text: str, voice: str | None = None, speed: float | None = None, language: str = "en") -> dict[str, Any]:
    """Synthesize using Kokoro backend."""
    if not text or not text.strip():
        return {"audio_url": None, "audio_path": None, "words": [],
                "visemes": [], "duration_ms": 0}

    voice = voice or KOKORO_VOICE
    speed = speed if speed is not None else KOKORO_SPEED

    # Build a deterministic cache key so repeated identical sentences
    # do not re-synthesise. Include backend in key.
    cache_key = _make_cache_key(text, voice, speed, "kokoro")
    cache_path = AUDIO_CACHE_DIR / f"{cache_key}.wav"
    cache_meta = AUDIO_CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists() and cache_meta.exists():
        log.debug("TTS cache hit for key %s", cache_key)
        with cache_meta.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["audio_path"] = str(cache_path)
        meta["audio_url"] = f"/api/tts/audio/{cache_key}"
        return meta

    # -- Synthesis --------------------------------------------------------
    import numpy as np

    pipeline = _get_kokoro_pipeline(language)
    results = list(pipeline(text, voice=voice, speed=speed,
                            split_pattern=_SPLIT_PATTERN))

    # Concatenate audio chunks, padding each sentence with the pause its
    # terminal punctuation calls for.
    gaps = _sentence_gaps(results, speed)
    audio_chunks = []
    got_audio = False
    for result, gap in zip(results, gaps):
        if result.audio is None:
            continue
        got_audio = True
        audio_chunks.append(np.asarray(result.audio, dtype=np.float32))
        if gap:
            audio_chunks.append(np.zeros(gap, dtype=np.float32))

    if not got_audio:
        log.warning("Kokoro produced no audio for: %s", text[:80])
        return {"audio_url": None, "audio_path": None, "words": [],
                "visemes": [], "duration_ms": 0}

    audio = np.concatenate(audio_chunks)

    # Write WAV
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for TTS output.  pip install soundfile") from exc

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(cache_path), audio, KOKORO_SAMPLE_RATE)

    # -- Extract word timings from Kokoro tokens --------------------------
    words = _extract_word_timings(results, gaps)
    visemes = _build_visemes(words)
    duration_ms = round(len(audio) / KOKORO_SAMPLE_RATE * 1000)

    # Per-phoneme durations have done their job building the track above; the
    # frontend only highlights whole words, so keep them out of the payload.
    for w in words:
        w.pop("phoneme_ms", None)

    # -- Optional RVC voice conversion ------------------------------------
    # Re-timbres the Kokoro audio into the target voice.  RVC is
    # frame-synchronous, so duration is preserved to within a couple of ~10ms
    # frames and the word/viseme tracks above stay valid as-is.  Any failure
    # leaves the plain Kokoro audio in place.
    rvc_applied = False
    if _rvc_enabled():
        import rvc_client as rvc

        raw_path = cache_path.with_name(cache_path.stem + ".kokoro.wav")
        try:
            cache_path.replace(raw_path)
            result = rvc.convert(str(raw_path), str(cache_path))
            if result and cache_path.exists():
                rvc_applied = True
                # Trust the converted file's real length rather than assuming
                # it matches; the frame grid can shave a few ms.
                duration_ms = result.get("duration_ms", duration_ms)
            else:
                raw_path.replace(cache_path)  # restore Kokoro audio
        except Exception as exc:  # noqa: BLE001 - never lose speech over this
            log.warning("RVC conversion errored (%s); using Kokoro audio", exc)
            if raw_path.exists() and not cache_path.exists():
                raw_path.replace(cache_path)
        finally:
            if raw_path.exists():
                try:
                    raw_path.unlink()
                except OSError:
                    pass

    meta = {
        "audio_url": f"/api/tts/audio/{cache_key}",
        "audio_path": str(cache_path),
        "words": words,
        "visemes": visemes,
        "duration_ms": duration_ms,
        "rvc": rvc_applied,
    }

    with cache_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    log.info("Kokoro TTS generated %d words, %d ms audio", len(words), duration_ms)
    return meta


def _synthesize_piper(text: str, voice: str | None = None, speed: float | None = None, language: str = "en") -> dict[str, Any]:
    """Synthesize using Piper backend."""
    if not text or not text.strip():
        return {"audio_url": None, "audio_path": None, "words": [],
                "visemes": [], "duration_ms": 0}

    # For Piper, voice parameter is the model name, speed is length_scale
    piper_voice_name = voice or PIPER_VOICE
    length_scale = speed if speed is not None else PIPER_SPEED

    # Build a deterministic cache key including backend and Piper-specific params
    import hashlib
    raw = f"{text.strip()}|{piper_voice_name}|{length_scale:.3f}|{PIPER_NOISE_SCALE:.3f}|{PIPER_NOISE_W:.3f}|{PIPER_SENTENCE_SILENCE:.3f}|piper|v{_META_VERSION}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    safe_voice = re.sub(r"[^\w\-]+", "_", Path(piper_voice_name).stem)[:40]
    cache_key = f"{safe_voice}_{digest}"
    cache_path = AUDIO_CACHE_DIR / f"{cache_key}.wav"
    cache_meta = AUDIO_CACHE_DIR / f"{cache_key}.json"

    if cache_path.exists() and cache_meta.exists():
        log.debug("TTS cache hit for key %s", cache_key)
        with cache_meta.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["audio_path"] = str(cache_path)
        meta["audio_url"] = f"/api/tts/audio/{cache_key}"
        return meta

    # -- Synthesis --------------------------------------------------------
    import numpy as np
    import soundfile as sf

    piper_voice = _get_piper_voice()
    sample_rate = piper_voice.config.sample_rate

    # Piper synthesizes using SynthesisConfig and returns AudioChunk objects
    from piper.config import SynthesisConfig

    syn_config = SynthesisConfig(
        speaker_id=None,
        length_scale=length_scale,
        noise_scale=PIPER_NOISE_SCALE,
        noise_w_scale=PIPER_NOISE_W,
        normalize_audio=True,
        volume=1.0,
    )

    chunks = list(piper_voice.synthesize(text, syn_config=syn_config))

    # Combine audio from all chunks
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])

    if len(audio) == 0:
        log.warning("Piper produced no audio for: %s", text[:80])
        return {"audio_url": None, "audio_path": None, "words": [],
                "visemes": [], "duration_ms": 0}

    # Write WAV
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(str(cache_path), audio, sample_rate)

    # -- Generate approximate word timings for Piper -----------------------
    # Piper doesn't provide word-level timestamps, so we approximate
    # using the phoneme durations from the model output
    words = _approximate_word_timings_piper(text, audio, sample_rate, length_scale)
    visemes = _build_visemes(words)
    duration_ms = round(len(audio) / sample_rate * 1000)

    meta = {
        "audio_url": f"/api/tts/audio/{cache_key}",
        "audio_path": str(cache_path),
        "words": words,
        "visemes": visemes,
        "duration_ms": duration_ms,
        "rvc": False,  # RVC not applied to Piper (different sample rate)
    }

    with cache_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    log.info("Piper TTS generated %d words, %d ms audio", len(words), duration_ms)
    return meta


def _approximate_word_timings_piper(text: str, audio: "np.ndarray", sample_rate: int, length_scale: float) -> list[dict[str, Any]]:
    """
    Approximate word timings for Piper output.
    Since Piper doesn't provide word-level timestamps, we estimate based on
    text length and audio duration.
    """
    import re
    words_list = re.findall(r'\S+', text)
    if not words_list:
        return []

    total_duration_ms = len(audio) / sample_rate * 1000
    # Rough estimate: distribute time proportionally by word length
    total_chars = sum(len(w) for w in words_list)
    words = []
    current_ms = 0.0

    for word in words_list:
        word_duration_ms = (len(word) / max(total_chars, 1)) * total_duration_ms
        words.append({
            "word": word,
            "phonemes": "",
            "start_ms": round(current_ms),
            "end_ms": round(current_ms + word_duration_ms),
        })
        current_ms += word_duration_ms

    return words


def prewarm() -> bool:
    """
    Load and exercise every model the speech path needs.  Returns True if the
    voice is ready to synthesise at full speed.

    Both stages are lazy by nature -- Kokoro builds its pipeline on the first
    synthesise, the RVC worker only spawns on the first conversion -- which
    stacks ~30s of model loading in front of the first spoken line of a
    session.  Doing it at boot moves that cost to where nothing is waiting.

    The throwaway utterance is not optional padding: the first pass through
    either model also compiles its CUDA kernels, measured at ~4.5s on the RVC
    side alone.  Warming the pipeline without running audio through it leaves
    that cost exactly where it was.
    """
    import numpy as np

    kokoro_ok = False
    piper_ok = False

    # Warm up Kokoro for both languages if it's the active backend or if we want to be ready for switching
    try:
        for lang in ["en", "cs"]:
            pipeline = _get_kokoro_pipeline(lang)
            results = list(pipeline("Warming up.", voice=KOKORO_VOICE, speed=KOKORO_SPEED))
            chunks = [np.asarray(r.audio, dtype=np.float32)
                      for r in results if getattr(r, "audio", None) is not None]
            if chunks:
                kokoro_ok = True
    except Exception as exc:  # noqa: BLE001 - TTS extras may not be installed
        log.info("Kokoro prewarm skipped: %s", exc)

    # Warm up Piper if it's the active backend
    if TTS_BACKEND == "piper":
        try:
            _get_piper_voice()
            # Just loading the model is enough for prewarm
            piper_ok = True
        except Exception as exc:
            log.warning("Piper prewarm failed: %s", exc)
            piper_ok = False

    if TTS_BACKEND == "kokoro":
        if not kokoro_ok:
            log.warning("Kokoro prewarm failed")
            return False

        # Run the warm audio through RVC so its kernels compile here too.
        if kokoro_ok and _rvc_enabled():
            import rvc_client as rvc
            try:
                import soundfile as sf
                AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                src = AUDIO_CACHE_DIR / "_prewarm_in.wav"
                dst = AUDIO_CACHE_DIR / "_prewarm_out.wav"
                sf.write(str(src), np.concatenate(chunks), KOKORO_SAMPLE_RATE)
                ok = rvc.convert(str(src), str(dst)) is not None
                for f in (src, dst):
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass
                log.info("TTS prewarmed (Kokoro%s)", " + RVC" if ok else ", RVC unavailable")
            except Exception as exc:  # noqa: BLE001
                log.warning("RVC prewarm failed: %s", exc)
                log.info("TTS prewarmed (Kokoro only)")
        else:
            log.info("TTS prewarmed (Kokoro only)")
        return kokoro_ok
    else:
        # Piper backend
        if piper_ok:
            log.info("TTS prewarmed (Piper)")
            return True
        else:
            log.warning("Piper prewarm failed")
            return False


def get_audio_path(cache_key: str) -> Path | None:
    """Return the filesystem path for a cached audio file, or None."""
    path = AUDIO_CACHE_DIR / f"{cache_key}.wav"
    return path if path.exists() else None


def list_cache() -> list[dict[str, Any]]:
    """Return metadata for every cached utterance (for admin/debug)."""
    items = []
    for meta_file in sorted(AUDIO_CACHE_DIR.glob("*.json")):
        try:
            with meta_file.open("r", encoding="utf-8") as fh:
                items.append(json.load(fh))
        except Exception:
            continue
    return items


def clear_cache() -> int:
    """Delete all cached audio and metadata.  Returns number of files removed."""
    count = 0
    for ext in ("*.wav", "*.json"):
        for f in AUDIO_CACHE_DIR.glob(ext):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
    log.info("TTS cache cleared (%d files)", count)
    return count


# -- Internals ---------------------------------------------------------------

# Bumped whenever the shape of the cached metadata changes, so stale sidecar
# JSON (e.g. pre-viseme entries) is never served back.  v2: added phonemes +
# viseme track.  v3: audio may be RVC-converted, so v2 entries sound wrong.
# v5: viseme track is aligned to the model's per-phoneme durations.
_META_VERSION = 5


def _rvc_enabled() -> bool:
    """True when RVC conversion should be attempted for new synthesis."""
    try:
        import config
        import rvc_client as rvc
        return bool(config.RVC_ENABLED) and rvc.available()
    except Exception:  # noqa: BLE001 - RVC is strictly optional
        return False


def _rvc_signature() -> str:
    """Identifies the conversion settings, so changing them busts the cache."""
    if not _rvc_enabled():
        return "off"
    import config
    model = os.path.basename(config.RVC_MODEL)
    return f"{model}:{config.RVC_PITCH}:{config.RVC_INDEX_RATE:.2f}:{config.RVC_PROTECT:.2f}"


def _make_cache_key(text: str, voice: str, speed: float, backend: str = "kokoro") -> str:
    """Deterministic, filesystem-safe cache key."""
    import hashlib
    if backend == "kokoro":
        raw = f"{text.strip()}|{voice}|{speed:.3f}|{_rvc_signature()}|v{_META_VERSION}"
    else:
        raw = f"{text.strip()}|{voice}|{speed:.3f}|{PIPER_NOISE_SCALE:.3f}|{PIPER_NOISE_W:.3f}|{PIPER_SENTENCE_SILENCE:.3f}|{backend}|v{_META_VERSION}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    # A voice may be a bare name ("af_heart") or a path to a blended .pt pack;
    # use just the stem so cache filenames stay short and readable.
    label = Path(voice).stem if voice.endswith(".pt") else voice
    safe_voice = re.sub(r"[^\w\-]+", "_", label)[:40]
    return f"{safe_voice}_{digest}"


def _extract_word_timings(results, gaps=None) -> list[dict[str, Any]]:
    """
    Walk through every Kokoro result and its tokens, grouping tokens into
    words using the whitespace flag.  Produces word-level start/end times
    in milliseconds.

    *gaps* is the per-chunk trailing silence in samples that synthesize()
    spliced into the audio; it has to be walked in here too or every word
    after the first pause drifts early and the visemes desync.

    Each word also carries "phoneme_ms" -- the model's own per-phoneme
    durations, when available -- which is what _build_visemes aligns to.
    """
    words: list[dict[str, Any]] = []
    chunk_offset_seconds = 0.0
    gaps = gaps or [0] * len(results)

    for result, gap in zip(results, gaps):
        tokens = getattr(result, "tokens", None) or []
        token_durs = _phoneme_durations(result)
        if not tokens:
            # No token data -- fall back to chunk-duration approximation
            audio = getattr(result, "audio", None)
            if audio is not None:
                chunk_dur = (len(audio) + gap) / KOKORO_SAMPLE_RATE
                chunk_offset_seconds += chunk_dur
            continue

        current_text: list[str] = []
        current_phonemes: list[str] = []
        current_durs: list[float] = []
        current_start: float | None = None
        current_end: float | None = None
        last_end = chunk_offset_seconds

        for index, token in enumerate(tokens):
            text = getattr(token, "text", "")
            if not text:
                continue

            start_ts = getattr(token, "start_ts", None)
            end_ts = getattr(token, "end_ts", None)

            if current_start is None and start_ts is not None:
                current_start = chunk_offset_seconds + float(start_ts)
            if end_ts is not None:
                current_end = chunk_offset_seconds + float(end_ts)

            phonemes = getattr(token, "phonemes", "") or ""
            current_text.append(text)
            current_phonemes.append(phonemes)
            # Durations stay 1:1 with the phoneme string, so a token the model
            # gave us no timing for contributes zeros rather than a shift.
            durs = token_durs[index] if index < len(token_durs) else None
            current_durs.extend(durs if durs is not None else [0.0] * len(phonemes))

            # whitespace flag means this token completes a word
            if getattr(token, "whitespace", ""):
                word_text = "".join(current_text).strip()
                if word_text:
                    start = current_start if current_start is not None else last_end
                    end = current_end if current_end is not None else start
                    words.append({
                        "word": word_text,
                        "phonemes": "".join(current_phonemes),
                        "phoneme_ms": current_durs,
                        "start_ms": max(0, round(start * 1000)),
                        "end_ms": max(0, round(end * 1000)),
                    })
                    last_end = end

                current_text = []
                current_phonemes = []
                current_durs = []
                current_start = None
                current_end = None

        if current_text:
            word_text = "".join(current_text).strip()
            if word_text:
                start = current_start if current_start is not None else last_end
                end = current_end if current_end is not None else start
                words.append({
                    "word": word_text,
                    "phonemes": "".join(current_phonemes),
                    "phoneme_ms": current_durs,
                    "start_ms": max(0, round(start * 1000)),
                    "end_ms": max(0, round(end * 1000)),
                })

        # Advance offset by the audio length of this result chunk, plus the
        # silence spliced in after it.
        audio = getattr(result, "audio", None)
        if audio is not None:
                    chunk_offset_seconds += (len(audio) + gap) / KOKORO_SAMPLE_RATE

    # Post-process: ensure monotonic, non-overlapping, and clamp
    words = _sanitize_timings(words)
    return words


# Kokoro's duration predictor works on a 40 Hz frame grid; kokoro's own
# join_timestamps() counts these in half-frames over a divisor of 80.
_FRAME_MS = 1000.0 / 40.0


def _phoneme_durations(result) -> list[list[float] | None]:
    """
    Per-phoneme durations in ms for each token of *result*, or None per token
    where the model gave us nothing.

    The model predicts a duration for every phoneme, but kokoro only surfaces
    the per-token sums as start_ts/end_ts.  The raw tensor is on the result as
    `pred_dur`, indexed over the padded phoneme sequence: [<bos>, ...phonemes
    of each token in order, with one slot per whitespace..., <eos>].  This walk
    mirrors KPipeline.join_timestamps exactly -- keep the two in step.

    Using these instead of dividing a word's span evenly is what makes the
    mouth land on the same phoneme the voice is on.
    """
    tokens = getattr(result, "tokens", None) or []
    out: list[list[float] | None] = [None] * len(tokens)

    pred_dur = getattr(result, "pred_dur", None)
    if pred_dur is None or len(pred_dur) < 3:
        return out
    frames = [int(x) for x in pred_dur]

    i = 1
    for index, token in enumerate(tokens):
        phonemes = getattr(token, "phonemes", "") or ""
        if not phonemes:
            # A whitespace-only token occupies one slot, counted twice by
            # join_timestamps so it can be split across the two words.
            if getattr(token, "whitespace", ""):
                i += 2
            continue
        j = i + len(phonemes)
        if j >= len(frames):
            break
        out[index] = [f * _FRAME_MS for f in frames[i:j]]
        i = j + (1 if getattr(token, "whitespace", "") else 0)

    return out


# -- Viseme (lip-sync) generation --------------------------------------------
#
# Kokoro/misaki emit a *compressed* IPA where several diphthongs collapse to a
# single uppercase char (A=eɪ, I=aɪ, O=oʊ, W=aʊ, Y=ɔɪ).  The full US symbol set
# is misaki.en.US_VOCAB.  We fold that onto the five mouth shapes a VRM model
# exposes as standard expressions -- aa/ih/ou/ee/oh -- plus "sil" (closed).
#
# Note the VRM names follow Japanese vowels (a/i/u/e/o), so IPA "i" as in
# *see* maps to "ih", and IPA "ɛ" as in *bed* maps to "ee".  They are not the
# English letter names they look like.
_VISEME_MAP = {
    # diphthongs (Kokoro's single-char forms) -- mapped to their opening vowel
    "A": "ee", "I": "aa", "O": "oh", "W": "aa", "Y": "oh",
    # monophthongs
    "i": "ih", "u": "ou", "æ": "aa", "ɑ": "aa", "ɔ": "oh",
    "ə": "aa", "ɛ": "ee", "ɜ": "ee", "ɪ": "ih", "ʊ": "ou",
    "ʌ": "aa", "ᵊ": "aa", "ᵻ": "ih",
    # bilabials -- the mouth must actually close, this is what sells lip-sync
    "b": "sil", "p": "sil", "m": "sil",
    # labiodental / rounded consonants
    "f": "ou", "v": "ou", "w": "ou", "ɹ": "ou",
    "ʃ": "ou", "ʒ": "ou", "ʧ": "ou", "ʤ": "ou",
    # alveolar / dental -- narrow mouth
    "l": "ih", "n": "ih", "t": "ih", "d": "ih", "s": "ih", "z": "ih",
    "θ": "ih", "ð": "ih", "j": "ih", "ɾ": "ih", "ʔ": "ih",
    # misaki rewrites the US flap ɾ to "T" on its way out (en.py), so the
    # letter turns up in anything like "waiting" or "better".  Unmapped it
    # would read as a closure and shut the mouth mid-word.
    "T": "ih",
    # velars / glottal -- slightly open
    "k": "aa", "ɡ": "aa", "ŋ": "aa", "h": "aa",
}

_VOWELS = frozenset("AIOWYiuæɑɔəɛɜɪʊʌᵊᵻ")

# How wide the mouth opens for each phoneme class.  Consonants at full weight
# make the avatar look like it is chewing.  Stressed vowels open widest, which
# is what gives the delivery its visible rhythm.
_STRESS_WEIGHT = {"ˈ": 1.0, "ˌ": 0.8}
_UNSTRESSED_VOWEL_WEIGHT = 0.62
_VOWEL_WEIGHT = 1.0          # flat vowel weight, fallback path only
_CONSONANT_WEIGHT = 0.45

# Marks that carry no mouth shape of their own.
_LENGTH_MARK = "ː"

# Shorter than this, the mouth cannot reach the shape before the next one
# replaces it -- attempting it reads as flutter, so fold it into its neighbour.
MIN_VISEME_MS = 40

# Kokoro folds the pause before punctuation, and each chunk's tail padding,
# into the duration of the last phoneme before it -- so a sentence-final vowel
# can measure over half a second.  Holding a shape that long looks like a gape;
# cap the hold and let the rest of the span play as silence.
MAX_VISEME_MS = 200


def _fold_marks(phonemes: str, durs: list[float]) -> list[list[Any]]:
    """
    Pair each phoneme with its duration and the stress it carries, dropping
    the marks themselves: [symbol, duration_ms, stress_weight].

    Stress marks precede the syllable they apply to and a length mark follows
    the phoneme it lengthens, so their frames are given to that neighbour
    rather than discarded -- the timeline has to stay continuous.
    """
    folded: list[list[Any]] = []
    carry = 0.0
    stress = 0.0

    for symbol, dur in zip(phonemes, durs):
        if symbol in _STRESS_WEIGHT:
            stress = _STRESS_WEIGHT[symbol]
            carry += dur
        elif symbol == _LENGTH_MARK:
            if folded:
                folded[-1][1] += dur
            else:
                carry += dur
        else:
            folded.append([symbol, dur + carry, stress])
            carry = 0.0
            if symbol in _VOWELS:
                stress = 0.0    # the mark applies to its syllable's vowel only

    if carry and folded:
        folded[-1][1] += carry
    return folded


def _place_visemes(folded: list[list[Any]], start_ms: float) -> list[dict[str, Any]]:
    """Lay a word's phonemes out end to end from *start_ms*."""
    out: list[dict[str, Any]] = []
    cursor = float(start_ms)

    for symbol, dur, stress in folded:
        if dur <= 0:
            continue
        shape = _VISEME_MAP.get(symbol, "sil")   # punctuation and unknown G2P
        if shape == "sil":
            out.append({"t_ms": round(cursor), "dur_ms": round(dur),
                        "viseme": "sil", "weight": 0.0})
        else:
            if symbol in _VOWELS:
                weight = stress or _UNSTRESSED_VOWEL_WEIGHT
            else:
                weight = _CONSONANT_WEIGHT
            hold = min(dur, MAX_VISEME_MS)
            out.append({"t_ms": round(cursor), "dur_ms": round(hold),
                        "viseme": shape, "weight": round(weight, 3)})
            if dur > hold:
                out.append({"t_ms": round(cursor + hold), "dur_ms": round(dur - hold),
                            "viseme": "sil", "weight": 0.0})
        cursor += dur

    return out


def _merge_runs(visemes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive events sharing a shape into one continuous hold."""
    out: list[dict[str, Any]] = []
    for v in visemes:
        if v["dur_ms"] <= 0:
            continue
        if out and out[-1]["viseme"] == v["viseme"]:
            prev = out[-1]
            prev["dur_ms"] = max(prev["dur_ms"], v["t_ms"] + v["dur_ms"] - prev["t_ms"])
            prev["weight"] = max(prev["weight"], v["weight"])
            continue
        out.append(dict(v))
    return out


def _build_visemes(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Turn word-level phoneme strings + timings into a viseme track.

    Where the model gave us per-phoneme durations ("phoneme_ms") the track is
    aligned to them directly, so the mouth is on the same phoneme as the voice
    rather than on a proportional guess at it.  Words without them -- a chunk
    the model returned no `pred_dur` for -- fall back to dividing the word's
    span across its phonemes, vowels weighted heavier than consonants.

    Returns a list of {"t_ms", "dur_ms", "viseme", "weight"} ordered by time.
    """
    visemes: list[dict[str, Any]] = []

    for w in words:
        phonemes = w.get("phonemes") or ""
        durs = w.get("phoneme_ms") or []
        start_ms = w["start_ms"]
        span = max(0, w["end_ms"] - start_ms)

        timed = len(durs) == len(phonemes) and sum(durs) > 0
        folded = _fold_marks(phonemes, durs if timed else [0.0] * len(phonemes))

        if not folded or (not timed and span <= 0):
            # Punctuation, silence, or a word with no usable G2P -- close up.
            visemes.append({"t_ms": start_ms, "dur_ms": span,
                            "viseme": "sil", "weight": 0.0})
            continue

        if not timed:
            shares = [_VOWEL_WEIGHT if s in _VOWELS else _CONSONANT_WEIGHT
                      for s, _, _ in folded]
            total = sum(shares)
            for entry, share in zip(folded, shares):
                entry[1] = span * share / total

        visemes.extend(_place_visemes(folded, start_ms))

    # Close the mouth after the final phoneme so it does not hang open.
    if visemes:
        last = visemes[-1]
        visemes.append({
            "t_ms": last["t_ms"] + last["dur_ms"],
            "dur_ms": 120,
            "viseme": "sil",
            "weight": 0.0,
        })

    # Drop shapes the mouth has no time to reach, extending whatever precedes
    # them.  Silences are kept however brief: a bilabial closure is short by
    # nature and it is the cue that sells the whole track.
    visemes = _merge_runs(visemes)
    kept: list[dict[str, Any]] = []
    for v in visemes:
        if v["viseme"] != "sil" and v["dur_ms"] < MIN_VISEME_MS and kept:
            prev = kept[-1]
            prev["dur_ms"] = v["t_ms"] + v["dur_ms"] - prev["t_ms"]
            continue
        kept.append(v)

    return _merge_runs(kept)


def _sanitize_timings(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Clean up edge cases:
      - negative or zero-duration words get a small default duration
      - overlapping words are clipped so end <= next start
      - gaps are left as-is (they represent pauses)
    """
    if not words:
        return words

    MIN_WORD_MS = 50  # minimum visible duration for a word

    out = []
    for i, w in enumerate(words):
        start = w["start_ms"]
        end = w["end_ms"]

        if end < start:
            end = start + MIN_WORD_MS
        if end - start < MIN_WORD_MS:
            end = start + MIN_WORD_MS

        # If this word would overlap the next one, clip it
        if i + 1 < len(words):
            next_start = words[i + 1]["start_ms"]
            end = min(end, next_start)

        out.append({
            "word": w["word"],
            "phonemes": w.get("phonemes", ""),
            "phoneme_ms": w.get("phoneme_ms", []),
            "start_ms": start,
            "end_ms": end,
        })

    return out
