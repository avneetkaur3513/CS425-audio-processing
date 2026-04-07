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
