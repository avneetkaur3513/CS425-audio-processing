"""
stt_template.py
===============
Speech-to-Text (STT) template for CS425 Assignment 2.

Loads a speech audio file, applies signal processing transformations
(pre-emphasis, noise, speed change, pitch shift), performs a transcription
attempt, and visualises the waveform and spectrogram.

Modify the parameters in the **Parameters** section below, then run:
    python stt_template.py

References
----------
* McFee et al. (2015) librosa: Audio and Music Signal Analysis in Python.
  Proceedings of the 14th Python in Science Conference.
* Boll, S. F. (1979) Suppression of acoustic noise in speech using spectral
  subtraction. IEEE Transactions on Acoustics, Speech, and Signal Processing.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf

# ---------------------------------------------------------------------------
# Parameters – modify these for each experiment
# ---------------------------------------------------------------------------
DEFAULT_AUDIO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Audio files",
    "Speaking_Female.wav",
)
AUDIO_FILE   = DEFAULT_AUDIO_FILE
PRE_EMPHASIS = 0.97    # 0.0 = no boost | 0.5–0.8 = weak | 0.9–0.99 = standard
NOISE_LEVEL  = 0.0     # 0.0 = clean    | 0.001–0.003 = very slight | 0.01 = moderate
SPEED_FACTOR = 1.0     # 1.0 = normal   | >1 = faster | <1 = slower
PITCH_STEPS  = 0       # semitones: 0 = no change | +2 = higher | -2 = lower

OUTPUT_DIR = "outputs/stt"

# ---------------------------------------------------------------------------
# Signal processing functions
# ---------------------------------------------------------------------------

def apply_pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply a first-order pre-emphasis filter to boost high frequencies.

    Pre-emphasis compensates for the natural roll-off of the vocal tract
    spectral envelope and makes high-frequency consonants (e.g. /s/, /t/)
    more prominent.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    coeff : float
        Pre-emphasis coefficient. 0.0 disables the filter; 0.97 is the
        standard value used in speech processing.

    Returns
    -------
    np.ndarray
        Pre-emphasised signal.
    """
    if coeff == 0.0:
        return signal.copy()
    return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(signal.dtype)


def add_noise(signal: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add Gaussian white noise to the signal.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    noise_level : float
        Standard deviation (amplitude) of the added noise.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_level, len(signal)).astype(signal.dtype)
    return signal + noise


def change_speed(signal: np.ndarray, sr: int, speed_factor: float = 1.0) -> np.ndarray:
    """Change playback speed via time-stretching (pitch remains unchanged).

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    sr : int
        Sample rate in Hz (unused here but kept for API consistency).
    speed_factor : float
        Multiplier: 1.0 = normal, >1 = faster, <1 = slower.

    Returns
    -------
    np.ndarray
        Time-stretched signal.
    """
    if speed_factor == 1.0:
        return signal.copy()
    return librosa.effects.time_stretch(signal, rate=speed_factor)


def shift_pitch(signal: np.ndarray, sr: int, n_steps: float = 0) -> np.ndarray:
    """Shift the pitch by *n_steps* semitones without changing duration.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    sr : int
        Sample rate in Hz.
    n_steps : float
        Semitones to shift. Positive = higher pitch; negative = lower.

    Returns
    -------
    np.ndarray
        Pitch-shifted signal.
    """
    if n_steps == 0:
        return signal.copy()
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)


def process_audio(
    signal: np.ndarray,
    sr: int,
    pre_emphasis: float = 0.0,
    noise_level: float = 0.0,
    speed_factor: float = 1.0,
    pitch_steps: float = 0,
) -> np.ndarray:
    """Apply the full STT processing pipeline in order.

    Applies: pre-emphasis → noise → speed change → pitch shift.

    Parameters
    ----------
    signal : np.ndarray
        Raw input signal.
    sr : int
        Sample rate in Hz.
    pre_emphasis : float
        Pre-emphasis coefficient.
    noise_level : float
        Gaussian noise amplitude.
    speed_factor : float
        Playback speed multiplier.
    pitch_steps : float
        Pitch shift in semitones.

    Returns
    -------
    np.ndarray
        Fully processed signal.
    """
    out = apply_pre_emphasis(signal, pre_emphasis)
    if noise_level > 0.0:
        out = add_noise(out, noise_level)
    out = change_speed(out, sr, speed_factor)
    out = shift_pitch(out, sr, pitch_steps)
    return out


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(signal: np.ndarray, sr: int) -> str:
    """Attempt to transcribe *signal* to text.

    Tries the ``SpeechRecognition`` library (Google Web Speech API) first.
    If the library is absent or the network is unavailable, a labelled
    placeholder is returned so all other experiment outputs are still saved.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal to transcribe.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    str
        Transcribed text or a descriptive placeholder string.
    """
    try:
        import io
        import speech_recognition as sr_lib

        recogniser = sr_lib.Recognizer()
        buf = io.BytesIO()
        sf.write(buf, signal.astype(np.float32), sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        with sr_lib.AudioFile(buf) as source:
            audio_data = recogniser.record(source)
        try:
            return recogniser.recognize_google(audio_data)
        except sr_lib.UnknownValueError:
            return "[Google STT: speech not understood]"
        except sr_lib.RequestError as exc:
            return f"[Google STT: API error – {exc}]"
    except ImportError:
        return (
            "[SpeechRecognition not installed – install with: "
            "pip install SpeechRecognition, then re-run]"
        )
    except (IOError, OSError, RuntimeError, ValueError) as exc:
        return f"[Transcription error: {exc}]"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_waveform_spectrogram(
    signal: np.ndarray,
    sr: int,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Create a two-panel figure: waveform (left) + log-power spectrogram (right).

    Parameters
    ----------
    signal : np.ndarray
        Audio signal.
    sr : int
        Sample rate in Hz.
    title : str
        Figure suptitle.
    save_path : str or None
        If provided, the figure is saved to this path and closed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Waveform ────────────────────────────────────────────────────────────
    times = np.arange(len(signal)) / sr
    axes[0].plot(times, signal, linewidth=0.5, color="steelblue")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].grid(True, alpha=0.3)

    # ── Spectrogram ─────────────────────────────────────────────────────────
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="hz", ax=axes[1], cmap="magma"
    )
    axes[1].set_title("Spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    fig.suptitle(title, fontsize=11, y=1.01)
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
    # ── Load audio ───────────────────────────────────────────────────────────
    print(f"[STT] Using input audio: {AUDIO_FILE}")
    if os.path.isfile(AUDIO_FILE):
        signal, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
        print(f"[STT] Loaded '{AUDIO_FILE}' ({len(signal)} samples @ {sr} Hz)")
    else:
        from audio_io import generate_synthetic_audio
        print(
            f"[STT] '{AUDIO_FILE}' not found – using synthetic speech signal.\n"
            "      Place 'Speaking_Female.wav' at 'Audio files/Speaking_Female.wav' for real-audio results."
        )
        signal, sr = generate_synthetic_audio(duration=5.0, sr=22050)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Apply processing ─────────────────────────────────────────────────────
    processed = process_audio(
        signal, sr,
        pre_emphasis=PRE_EMPHASIS,
        noise_level=NOISE_LEVEL,
        speed_factor=SPEED_FACTOR,
        pitch_steps=PITCH_STEPS,
    )

    tag = (
        f"pre={PRE_EMPHASIS}_noise={NOISE_LEVEL}"
        f"_speed={SPEED_FACTOR}_pitch={int(PITCH_STEPS)}"
    )
    plot_waveform_spectrogram(
        processed, sr,
        title=f"STT Processed – {tag}",
        save_path=os.path.join(OUTPUT_DIR, f"{tag}.png"),
    )

    # ── Save processed audio ─────────────────────────────────────────────────
    wav_path = os.path.join(OUTPUT_DIR, f"{tag}.wav")
    sf.write(wav_path, processed.astype(np.float32), sr)
    print(f"[STT] Saved processed audio → {wav_path}")

    # ── Transcribe ───────────────────────────────────────────────────────────
    transcription = transcribe_audio(processed, sr)
    txt_path = os.path.join(OUTPUT_DIR, f"{tag}_transcription.txt")
    with open(txt_path, "w") as fh:
        fh.write(f"Parameters: {tag}\n")
        fh.write(f"Transcription: {transcription}\n")
    print(f"[STT] Transcription: {transcription}")
    print(f"[STT] Saved transcription → {txt_path}")
