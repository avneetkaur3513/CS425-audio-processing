"""
stft_analysis.py - Short-Time Fourier Transform utilities for CS425 Assignment 1.
"""

import numpy as np
import librosa


def compute_stft(signal, sr, frame_size=2048, hop_length=512, window="hann"):
    """Compute the Short-Time Fourier Transform of *signal*.

    Parameters
    ----------
    signal : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate in Hz.
    frame_size : int
        FFT window size (number of samples per frame).
    hop_length : int
        Number of samples between successive frames.
    window : str
        Window function name understood by librosa/scipy ('hann', 'hamming', …).

    Returns
    -------
    stft_matrix : np.ndarray (complex)
        STFT matrix, shape (1 + frame_size//2, n_frames).
    time_axis : np.ndarray
        Centre time of each frame in seconds.
    freq_axis : np.ndarray
        Frequency of each bin in Hz.
    """
    stft_matrix = librosa.stft(
        signal,
        n_fft=frame_size,
        hop_length=hop_length,
        window=window,
    )
    n_frames = stft_matrix.shape[1]
    time_axis = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    return stft_matrix, time_axis, freq_axis


def stft_to_db(stft_matrix, ref=np.max):
    """Convert a complex STFT matrix to a dB-scaled magnitude spectrogram.

    Parameters
    ----------
    stft_matrix : np.ndarray (complex)
        Output of :func:`compute_stft`.
    ref : callable or float
        Reference value for dB conversion (default: max of magnitude).

    Returns
    -------
    np.ndarray
        Power spectrogram in dB.
    """
    magnitude = np.abs(stft_matrix)
    return librosa.amplitude_to_db(magnitude, ref=ref)


def time_resolution(hop_length, sr):
    """Time resolution of the STFT in milliseconds.

    Parameters
    ----------
    hop_length : int
        Hop length in samples.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Time resolution in milliseconds.
    """
    return hop_length / sr * 1000.0


def frequency_resolution(frame_size, sr):
    """Frequency resolution of the STFT in Hz.

    Parameters
    ----------
    frame_size : int
        FFT window size.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Frequency resolution (bin width) in Hz.
    """
    return sr / frame_size


def stft_parameter_summary(frame_size, hop_length, window, sr):
    """Return a dictionary summarising STFT resolution metrics.

    Parameters
    ----------
    frame_size : int
    hop_length : int
    window : str
    sr : int

    Returns
    -------
    dict
        Keys: 'frame_size', 'hop_length', 'window', 'time_res_ms',
              'freq_res_hz', 'overlap_pct'.
    """
    overlap_samples = frame_size - hop_length
    overlap_pct = 100.0 * overlap_samples / frame_size
    return {
        "frame_size": frame_size,
        "hop_length": hop_length,
        "window": window,
        "time_res_ms": time_resolution(hop_length, sr),
        "freq_res_hz": frequency_resolution(frame_size, sr),
        "overlap_pct": overlap_pct,
    }
