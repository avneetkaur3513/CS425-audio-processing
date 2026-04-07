"""
audio_io.py - Audio I/O, quantization, and SNR utilities for CS425 Assignment 1.
"""

import numpy as np
import soundfile as sf
import librosa


def load_audio(filepath, target_sr=None):
    """Load an audio file and optionally resample it.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    target_sr : int or None
        If provided, resample the audio to this sample rate.

    Returns
    -------
    signal : np.ndarray
        Mono audio signal normalised to [-1, 1].
    sr : int
        Sample rate of the returned signal.
    """
    signal, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return signal, sr


def resample_audio(signal, orig_sr, target_sr):
    """Resample *signal* from *orig_sr* to *target_sr*.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    orig_sr : int
        Original sample rate.
    target_sr : int
        Desired sample rate.

    Returns
    -------
    np.ndarray
        Resampled signal.
    """
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)


def quantize(signal, bit_depth):
    """Quantize a floating-point signal to *bit_depth* bits.

    The input is expected to be in [-1, 1].  The output is the quantised
    signal re-normalised to [-1, 1] as a float64 array.

    Parameters
    ----------
    signal : np.ndarray
        Floating-point audio signal in the range [-1, 1].
    bit_depth : int
        Number of quantisation bits (e.g. 4, 8, 16).

    Returns
    -------
    np.ndarray
        Quantised signal (float64, range [-1, 1]).
    """
    levels = 2 ** bit_depth
    # Scale to [0, levels-1], round to integer, scale back
    quantised = np.round((signal + 1.0) / 2.0 * (levels - 1))
    quantised = np.clip(quantised, 0, levels - 1)
    return quantised / (levels - 1) * 2.0 - 1.0


def calculate_snr(original, processed):
    """Calculate the Signal-to-Noise Ratio (SNR) in dB.

    SNR = 10 * log10( power(signal) / power(noise) )

    Parameters
    ----------
    original : np.ndarray
        Original (reference) signal.
    processed : np.ndarray
        Processed (potentially noisy) signal of the same length.

    Returns
    -------
    float
        SNR in decibels.  Returns np.inf when noise power is zero.
    """
    # Truncate to the shorter of the two arrays
    n = min(len(original), len(processed))
    signal = original[:n]
    noise = processed[:n] - signal

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


def dynamic_range_db(bit_depth):
    """Return the theoretical dynamic range for a given bit depth.

    Dynamic Range ≈ 6.02 * N  dB  (for N-bit PCM)

    Parameters
    ----------
    bit_depth : int
        Number of quantisation bits.

    Returns
    -------
    float
        Dynamic range in dB.
    """
    return 6.02 * bit_depth


def save_audio(filepath, signal, sr):
    """Save a floating-point signal as a WAV file.

    Parameters
    ----------
    filepath : str
        Destination file path (should end in .wav).
    signal : np.ndarray
        Audio signal (float32 or float64).
    sr : int
        Sample rate in Hz.
    """
    # soundfile writes float32 or float64 natively
    sf.write(filepath, signal.astype(np.float32), sr)


def generate_synthetic_audio(duration=5.0, sr=44100):
    """Generate a synthetic speech-like audio signal for testing.

    The signal is a superposition of harmonics at typical speech fundamental
    frequencies, amplitude-modulated to mimic voiced speech patterns.  It is
    intended as a stand-in when a real audio file is unavailable.

    Parameters
    ----------
    duration : float
        Signal duration in seconds (default: 5.0).
    sr : int
        Sample rate in Hz (default: 44100).

    Returns
    -------
    signal : np.ndarray
        Mono audio signal normalised to [-1, 1].
    sr : int
        Sample rate (same as the input *sr*).
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    # Fundamental and harmonics for a speech-like tone (~150 Hz male voice)
    f0 = 150.0
    harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
    amplitudes = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08]

    signal = np.zeros_like(t)
    for k, amp in zip(harmonics, amplitudes):
        signal += amp * np.sin(2 * np.pi * f0 * k * t)

    # Add a second formant region (~900 Hz) to simulate a vowel
    signal += 0.3 * np.sin(2 * np.pi * 900.0 * t)

    # Amplitude envelope: slow AM at 4 Hz to mimic syllable rhythm
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    signal *= envelope

    # Low-level noise floor
    signal += 0.01 * rng.standard_normal(len(t))

    # Normalise to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    return signal.astype(np.float32), sr
