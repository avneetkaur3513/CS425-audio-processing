"""
effects.py - Clipping and downsampling effects for CS425 Assignment 1.
"""

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def hard_clip(signal, threshold):
    """Apply hard (flat-top) clipping to *signal*.

    Samples whose absolute value exceeds *threshold* are clipped to
    ±*threshold*.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal in [-1, 1].
    threshold : float
        Clipping threshold in (0, 1].

    Returns
    -------
    np.ndarray
        Clipped signal.
    """
    return np.clip(signal, -threshold, threshold)


def soft_clip(signal, threshold):
    """Apply soft (tanh-based) clipping to *signal*.

    Samples within [-threshold, threshold] pass through linearly; beyond
    that the function saturates smoothly using a tanh curve.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal in [-1, 1].
    threshold : float
        Clipping threshold in (0, 1].

    Returns
    -------
    np.ndarray
        Soft-clipped signal.
    """
    # Scale so that the threshold maps to the linear region of tanh
    scale = 1.0 / threshold
    return threshold * np.tanh(signal * scale)


def clipping_distortion_db(original, clipped):
    """Measure the clipping distortion as THD-like power ratio in dB.

    Distortion = 10 * log10( power(noise) / power(signal) )

    Parameters
    ----------
    original : np.ndarray
        Original un-clipped signal.
    clipped : np.ndarray
        Clipped signal.

    Returns
    -------
    float
        Distortion level in dB (more negative = less distortion).
    """
    n = min(len(original), len(clipped))
    noise = clipped[:n] - original[:n]
    signal_power = np.mean(original[:n] ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return -np.inf
    return 10.0 * np.log10(noise_power / signal_power)


def clipped_sample_fraction(original, threshold):
    """Return the fraction of samples that would be clipped.

    Parameters
    ----------
    original : np.ndarray
        Input signal.
    threshold : float
        Clipping threshold.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    return float(np.mean(np.abs(original) > threshold))


# ---------------------------------------------------------------------------
# Downsampling / Aliasing
# ---------------------------------------------------------------------------

def downsample(signal, sr, factor):
    """Downsample *signal* by an integer *factor* without anti-aliasing.

    This intentionally skips the low-pass filter to demonstrate aliasing
    artefacts.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    sr : int
        Original sample rate.
    factor : int
        Downsampling factor (e.g. 2 halves the sample rate).

    Returns
    -------
    downsampled : np.ndarray
        Downsampled signal.
    new_sr : int
        New sample rate.
    """
    downsampled = signal[::factor]
    new_sr = sr // factor
    return downsampled, new_sr


def downsample_with_filter(signal, sr, factor):
    """Downsample *signal* by *factor* with anti-aliasing (proper resampling).

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    sr : int
        Original sample rate.
    factor : int
        Downsampling factor.

    Returns
    -------
    downsampled : np.ndarray
        Properly resampled signal.
    new_sr : int
        New sample rate.
    """
    new_sr = sr // factor
    downsampled = librosa.resample(signal, orig_sr=sr, target_sr=new_sr)
    return downsampled, new_sr


def nyquist_frequency(sr):
    """Return the Nyquist frequency for *sr*.

    Parameters
    ----------
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Nyquist frequency in Hz.
    """
    return sr / 2.0
