"""
fourier_analysis.py - FFT, DFT, and time/phase-shift utilities for CS425 Assignment 1.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Frequency-domain helpers
# ---------------------------------------------------------------------------

def compute_fft(signal, sr):
    """Compute the one-sided magnitude spectrum using numpy FFT.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued audio signal.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (Hz), length N//2 + 1.
    magnitude : np.ndarray
        Magnitude spectrum (linear), same length as *freqs*.
    """
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return freqs, magnitude


def compute_dft_naive(signal):
    """Compute the DFT via the O(N²) definition (for small N only).

    Parameters
    ----------
    signal : np.ndarray
        Real or complex-valued signal of length N.

    Returns
    -------
    np.ndarray
        Complex DFT coefficients, length N.
    """
    n = len(signal)
    k = np.arange(n)
    # Vandermonde matrix approach: X[k] = sum_n x[n] * exp(-j2pi*k*n/N)
    dft_matrix = np.exp(-2j * np.pi * k[:, None] * k[None, :] / n)
    return dft_matrix @ signal.astype(complex)


def benchmark_dft_vs_fft(signal_length, repetitions=3):
    """Time naive DFT and numpy FFT for a random signal of *signal_length*.

    Parameters
    ----------
    signal_length : int
        Number of samples in the test signal.
    repetitions : int
        Number of timed repetitions (best time is reported).

    Returns
    -------
    dict
        Keys: 'length', 'dft_time_s', 'fft_time_s', 'speedup'
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(signal_length)

    # Time DFT
    dft_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        compute_dft_naive(x)
        dft_times.append(time.perf_counter() - t0)

    # Time FFT
    fft_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        np.fft.fft(x)
        fft_times.append(time.perf_counter() - t0)

    dft_best = min(dft_times)
    fft_best = min(fft_times)
    speedup = dft_best / fft_best if fft_best > 0 else np.inf

    return {
        "length": signal_length,
        "dft_time_s": dft_best,
        "fft_time_s": fft_best,
        "speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Time and phase shifts
# ---------------------------------------------------------------------------

def apply_time_shift(signal, sr, shift_ms):
    """Shift *signal* in time by *shift_ms* milliseconds (zero-pad left).

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    sr : int
        Sample rate in Hz.
    shift_ms : float
        Time shift in milliseconds (non-negative).

    Returns
    -------
    np.ndarray
        Shifted signal, same length as *signal*.
    """
    shift_samples = int(sr * shift_ms / 1000.0)
    if shift_samples == 0:
        return signal.copy()
    shifted = np.zeros_like(signal)
    shifted[shift_samples:] = signal[: len(signal) - shift_samples]
    return shifted


def apply_phase_shift(signal, phase_degrees):
    """Apply a constant phase rotation to every frequency component.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued input signal.
    phase_degrees : float
        Phase shift in degrees.

    Returns
    -------
    np.ndarray
        Phase-shifted signal (real part, same length as *signal*).
    """
    spectrum = np.fft.rfft(signal)
    phase_rad = np.deg2rad(phase_degrees)
    shifted_spectrum = spectrum * np.exp(1j * phase_rad)
    return np.fft.irfft(shifted_spectrum, n=len(signal))


def phase_shift_from_time_shift(shift_ms, frequency_hz):
    """Compute the phase shift (degrees) for a pure-tone at *frequency_hz*.

    phi = 2 * pi * f * delta_t  (converted to degrees)

    Parameters
    ----------
    shift_ms : float
        Time shift in milliseconds.
    frequency_hz : float
        Frequency of the tone in Hz.

    Returns
    -------
    float
        Phase shift in degrees (modulo 360).
    """
    delta_t = shift_ms / 1000.0
    phase_rad = 2.0 * np.pi * frequency_hz * delta_t
    return np.rad2deg(phase_rad) % 360.0
