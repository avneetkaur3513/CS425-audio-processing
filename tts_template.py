"""
tts_template.py
===============
Text-to-Speech (TTS) template for CS425 Assignment 2.

Generates speech from text using pyttsx3 (or a synthetic fallback when
pyttsx3 is unavailable), applies signal processing transformations, and
visualises side-by-side before/after spectrograms.

Synthesis / Voice Parameters
-----------------------------
TEXT          – text to synthesise
RATE          – speech rate in words per minute (typical: 120–220)
VOICE_INDEX   – voice selection index (0 = first system voice, 1 = second)
VOLUME        – output volume (0.0 – 1.0)

Signal Processing Parameters
-----------------------------
PRE_EMPHASIS  – high-frequency boost coefficient (0.0 = none, 0.97 = standard)
NOISE_LEVEL   – Gaussian noise amplitude (0.0 = clean)
PITCH_STEPS   – pitch shift in semitones (0 = none, positive = higher)
SPEED_FACTOR  – time-stretch factor (1.0 = none, >1 = faster, <1 = slower)
LOW_CUT       – high-pass filter cutoff in Hz (0 = disabled)
HIGH_CUT      – low-pass filter cutoff in Hz (0 = disabled)
GAIN          – amplitude scaling factor (1.0 = no change)

References
----------
* Klatt, D. H. (1987) Review of text-to-speech conversion for English.
  JASA, 82(3), 737-793.
* McFee et al. (2015) librosa: Audio and Music Signal Analysis in Python.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Parameters – modify these for each experiment
# ---------------------------------------------------------------------------
TEXT         = "The quick brown fox jumps over the lazy dog."
RATE         = 150       # words per minute (typical: 120–220)
VOICE_INDEX  = 0         # 0 = first system voice, 1 = second
VOLUME       = 1.0       # 0.0 – 1.0

PRE_EMPHASIS = 0.0       # 0.0 = none | 0.0–0.97 = standard range
NOISE_LEVEL  = 0.0       # 0.0 = clean | 0.001–0.03 = typical range
PITCH_STEPS  = 0         # semitones: 0 = none | -3 to +3 = typical
SPEED_FACTOR = 1.0       # 1.0 = none | 0.7–1.3 = typical range
LOW_CUT      = 0         # Hz (0 = disabled)
HIGH_CUT     = 0         # Hz (0 = disabled)
GAIN         = 1.0       # 0.5–2.0 typical

OUTPUT_DIR = "outputs/tts"

# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------

def synthesize_tts(
    text: str,
    rate: int = 150,
    voice_index: int = 0,
    volume: float = 1.0,
    output_path: str | None = None,
) -> str | None:
    """Synthesise *text* to a WAV file using pyttsx3.

    Parameters
    ----------
    text : str
        Input text.
    rate : int
        Speech rate in words per minute.
    voice_index : int
        Voice selection index.
    volume : float
        Volume (0.0 – 1.0).
    output_path : str or None
        Destination WAV path; a temp file is created when None.

    Returns
    -------
    str or None
        Path to the saved WAV file, or None on failure.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", float(volume))
        voices = engine.getProperty("voices")
        if voices and voice_index < len(voices):
            engine.setProperty("voice", voices[voice_index].id)
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return output_path if os.path.isfile(output_path) else None
    except Exception as exc:
        print(f"[TTS] pyttsx3 unavailable ({exc}); using synthetic fallback.")
        return None


def generate_tts_audio(
    text: str,
    rate: int = 150,
    voice_index: int = 0,
    volume: float = 1.0,
    sr: int = 22050,
) -> tuple[np.ndarray, int]:
    """Generate TTS audio as a NumPy array.

    Tries pyttsx3 first; falls back to a synthetic speech-like signal when
    pyttsx3 or a system TTS engine is not available.

    Parameters
    ----------
    text : str
        Text to synthesise.
    rate : int
        Speech rate (words per minute).
    voice_index : int
        Voice index.
    volume : float
        Volume (0.0 – 1.0).
    sr : int
        Target sample rate in Hz.

    Returns
    -------
    tuple(np.ndarray, int)
        (audio signal, sample rate)
    """
    tmp_path = tempfile.mktemp(suffix=".wav")
    wav_path = synthesize_tts(text, rate, voice_index, volume, tmp_path)

    if wav_path and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 100:
        try:
            signal, actual_sr = librosa.load(wav_path, sr=sr, mono=True)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            return signal, actual_sr
        except Exception:
            pass

    # Clean up temp file if it exists
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    # Synthetic fallback
    from audio_io import generate_synthetic_audio
    print("[TTS] Using synthetic speech-like audio (pyttsx3 unavailable).")
    return generate_synthetic_audio(duration=3.0, sr=sr)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def apply_pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply first-order pre-emphasis filter (boosts high frequencies)."""
    if coeff == 0.0:
        return signal.copy()
    return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(signal.dtype)


def add_noise(signal: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add Gaussian white noise to the signal."""
    rng = np.random.default_rng(42)
    return (signal + rng.normal(0.0, noise_level, len(signal))).astype(signal.dtype)


def shift_pitch(signal: np.ndarray, sr: int, n_steps: float = 0) -> np.ndarray:
    """Shift pitch by *n_steps* semitones (duration unchanged)."""
    if n_steps == 0:
        return signal.copy()
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)


def time_stretch(signal: np.ndarray, speed_factor: float = 1.0) -> np.ndarray:
    """Change duration without altering pitch."""
    if speed_factor == 1.0:
        return signal.copy()
    return librosa.effects.time_stretch(signal, rate=speed_factor)


def bandpass_filter(
    signal: np.ndarray,
    sr: int,
    low_cut: float = 0,
    high_cut: float = 0,
) -> np.ndarray:
    """Apply high-pass and/or low-pass Butterworth filter.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sr : int
        Sample rate in Hz.
    low_cut : float
        High-pass cutoff frequency in Hz (0 = disabled).
    high_cut : float
        Low-pass cutoff frequency in Hz (0 = disabled).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = sr / 2.0
    if low_cut <= 0 and high_cut <= 0:
        return signal.copy()
    if low_cut > 0 and high_cut > 0 and low_cut < high_cut < nyq:
        sos = butter(4, [low_cut / nyq, high_cut / nyq], btype="band", output="sos")
    elif low_cut > 0 and low_cut < nyq:
        sos = butter(4, low_cut / nyq, btype="high", output="sos")
    elif high_cut > 0 and high_cut < nyq:
        sos = butter(4, high_cut / nyq, btype="low", output="sos")
    else:
        return signal.copy()
    return sosfilt(sos, signal).astype(signal.dtype)


def apply_gain(signal: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Scale signal amplitude; clips to [-1, 1] to prevent overflow."""
    return np.clip(signal * gain, -1.0, 1.0).astype(signal.dtype)


def process_tts_signal(
    signal: np.ndarray,
    sr: int,
    pre_emphasis: float = 0.0,
    noise_level: float = 0.0,
    pitch_steps: float = 0,
    speed_factor: float = 1.0,
    low_cut: float = 0,
    high_cut: float = 0,
    gain: float = 1.0,
) -> np.ndarray:
    """Apply the full TTS post-processing pipeline.

    Order: pre-emphasis → noise → pitch shift → time-stretch
           → bandpass filter → gain.

    Parameters
    ----------
    signal : np.ndarray
        Raw TTS signal.
    sr : int
        Sample rate in Hz.
    pre_emphasis : float
        Pre-emphasis coefficient.
    noise_level : float
        Gaussian noise amplitude.
    pitch_steps : float
        Pitch shift in semitones.
    speed_factor : float
        Time-stretch factor.
    low_cut : float
        High-pass cutoff in Hz.
    high_cut : float
        Low-pass cutoff in Hz.
    gain : float
        Amplitude scaling factor.

    Returns
    -------
    np.ndarray
        Fully processed signal.
    """
    out = apply_pre_emphasis(signal, pre_emphasis)
    if noise_level > 0.0:
        out = add_noise(out, noise_level)
    out = shift_pitch(out, sr, pitch_steps)
    out = time_stretch(out, speed_factor)
    out = bandpass_filter(out, sr, low_cut, high_cut)
    out = apply_gain(out, gain)
    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_spectrogram_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    title_original: str = "Original",
    title_processed: str = "Processed",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot original and processed log-power spectrograms side by side.

    Parameters
    ----------
    original, processed : np.ndarray
        Audio signals to compare.
    sr : int
        Sample rate in Hz.
    title_original, title_processed : str
        Subplot titles.
    save_path : str or None
        If provided, the figure is saved here and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, sig, ttl in [
        (axes[0], original, title_original),
        (axes[1], processed, title_processed),
    ]:
        D = librosa.stft(sig)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
        )
        ax.set_title(ttl)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Plot] Saved: {save_path}")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[TTS] Synthesising: '{TEXT}'")
    original, sr = generate_tts_audio(
        TEXT, rate=RATE, voice_index=VOICE_INDEX, volume=VOLUME
    )
    print(f"[TTS] Generated {len(original)} samples @ {sr} Hz")

    processed = process_tts_signal(
        original, sr,
        pre_emphasis=PRE_EMPHASIS,
        noise_level=NOISE_LEVEL,
        pitch_steps=PITCH_STEPS,
        speed_factor=SPEED_FACTOR,
        low_cut=LOW_CUT,
        high_cut=HIGH_CUT,
        gain=GAIN,
    )

    tag = (
        f"pre={PRE_EMPHASIS}_noise={NOISE_LEVEL}_pitch={PITCH_STEPS}"
        f"_speed={SPEED_FACTOR}_lc={LOW_CUT}_hc={HIGH_CUT}_gain={GAIN}"
    )
    plot_spectrogram_comparison(
        original, processed, sr,
        title_original="Original TTS",
        title_processed=f"Processed: {tag}",
        save_path=os.path.join(OUTPUT_DIR, f"{tag}.png"),
    )

    wav_path = os.path.join(OUTPUT_DIR, f"{tag}.wav")
    sf.write(wav_path, processed.astype(np.float32), sr)
    print(f"[TTS] Saved processed audio → {wav_path}")
