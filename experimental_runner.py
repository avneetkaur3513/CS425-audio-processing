"""
experimental_runner.py
======================
Main orchestrator for CS425 Assignment 1 – Time and Frequency Domain Audio Analysis.

Runs all 7 experiment suites automatically and saves:
  * Publication-quality PNG plots  (outputs/plots/)
  * Processed WAV files            (outputs/audio/)
  * CSV result tables              (outputs/report_data/)
  * JSON summary archive           (outputs/report_data/all_experiments_results.json)

Usage
-----
    python experimental_runner.py [audio_file] [--output-dir DIR]

If *audio_file* is omitted the script looks for ``speech.wav`` in the
current directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_io import (
    calculate_snr,
    dynamic_range_db,
    load_audio,
    quantize,
    resample_audio,
    save_audio,
)
from effects import (
    clipped_sample_fraction,
    clipping_distortion_db,
    downsample,
    hard_clip,
    nyquist_frequency,
    soft_clip,
)
from fourier_analysis import (
    apply_time_shift,
    benchmark_dft_vs_fft,
    compute_fft,
    phase_shift_from_time_shift,
)
from main import ensure_output_dirs, plot_waveform, save_figure
from stft_analysis import compute_stft, stft_parameter_summary, stft_to_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_row(label: str, value: str) -> None:
    print(f"  {label:<35} {value}")


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"  [CSV] {path}")


# ---------------------------------------------------------------------------
# Experiment 1 – Sampling Rate
# ---------------------------------------------------------------------------

def experiment_sampling_rate(signal: np.ndarray, sr: int, dirs: dict) -> dict:
    """Resample signal to multiple rates; measure SNR and Nyquist frequency."""
    _print_header("Experiment 1: Sampling Rate Analysis")

    target_rates = [44100, 22050, 8000]
    results = []

    fig, axes = plt.subplots(len(target_rates), 2, figsize=(14, 10))
    fig.suptitle("Experiment 1: Sampling Rate Comparison", fontsize=14)

    for idx, target_sr in enumerate(target_rates):
        # Resample
        if target_sr == sr:
            resampled = signal.copy()
        else:
            resampled = resample_audio(signal, sr, target_sr)

        # Compute SNR by resampling back to original rate
        back = resample_audio(resampled, target_sr, sr)
        snr = calculate_snr(signal, back)
        nyq = nyquist_frequency(target_sr)

        _print_row(f"{target_sr} Hz – SNR", f"{snr:.2f} dB")
        _print_row(f"{target_sr} Hz – Nyquist", f"{nyq:.0f} Hz")

        results.append({
            "sample_rate_hz": target_sr,
            "nyquist_hz": nyq,
            "snr_db": round(snr, 2),
            "n_samples": len(resampled),
        })

        # Save audio
        audio_path = os.path.join(dirs["audio"], f"01_sr_{target_sr}hz.wav")
        save_audio(audio_path, resampled, target_sr)

        # Plot waveform
        t = np.arange(len(resampled)) / target_sr
        axes[idx, 0].plot(t, resampled, linewidth=0.5)
        axes[idx, 0].set_title(f"Waveform @ {target_sr} Hz")
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot spectrum
        freqs, mag = compute_fft(resampled, target_sr)
        axes[idx, 1].plot(freqs, mag, linewidth=0.7)
        axes[idx, 1].axvline(nyq, color="red", linestyle="--", linewidth=1,
                             label=f"Nyquist = {nyq:.0f} Hz")
        axes[idx, 1].set_title(f"Spectrum @ {target_sr} Hz")
        axes[idx, 1].set_xlabel("Frequency (Hz)")
        axes[idx, 1].set_ylabel("Magnitude")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "01_sampling_rate.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "01_sampling_rate_results.csv"))
    return {"experiment": "sampling_rate", "results": results}


# ---------------------------------------------------------------------------
# Experiment 2 – Quantization
# ---------------------------------------------------------------------------

def experiment_quantization(signal: np.ndarray, sr: int, dirs: dict) -> dict:
    """Quantise signal to multiple bit depths; compute SNR and dynamic range."""
    _print_header("Experiment 2: Quantization Analysis")

    bit_depths = [16, 8, 4]
    results = []

    fig, axes = plt.subplots(len(bit_depths), 2, figsize=(14, 10))
    fig.suptitle("Experiment 2: Quantization Comparison", fontsize=14)

    for idx, bits in enumerate(bit_depths):
        q_signal = quantize(signal, bits)
        snr = calculate_snr(signal, q_signal)
        dyn_range = dynamic_range_db(bits)

        _print_row(f"{bits}-bit SNR", f"{snr:.2f} dB")
        _print_row(f"{bits}-bit dynamic range", f"{dyn_range:.1f} dB")

        results.append({
            "bit_depth": bits,
            "snr_db": round(snr, 2),
            "dynamic_range_db": round(dyn_range, 1),
            "quantization_levels": 2 ** bits,
        })

        audio_path = os.path.join(dirs["audio"], f"02_quantized_{bits}bit.wav")
        save_audio(audio_path, q_signal, sr)

        # Waveform comparison
        t = np.arange(len(signal)) / sr
        axes[idx, 0].plot(t, signal, alpha=0.5, label="Original", linewidth=0.5)
        axes[idx, 0].plot(t, q_signal, alpha=0.7, label=f"{bits}-bit", linewidth=0.5)
        axes[idx, 0].set_title(f"Waveform: {bits}-bit quantization")
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)

        # Quantisation noise
        noise = q_signal - signal
        axes[idx, 1].plot(t, noise, linewidth=0.5, color="red")
        axes[idx, 1].set_title(f"Quantization Noise: {bits}-bit (SNR={snr:.1f} dB)")
        axes[idx, 1].set_xlabel("Time (s)")
        axes[idx, 1].set_ylabel("Error")
        axes[idx, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "02_quantization.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "02_quantization_results.csv"))
    return {"experiment": "quantization", "results": results}


# ---------------------------------------------------------------------------
# Experiment 3 – Time / Phase Shift
# ---------------------------------------------------------------------------

def experiment_time_phase_shift(signal: np.ndarray, sr: int, dirs: dict) -> dict:
    """Apply time shifts and compare in time vs frequency domain."""
    _print_header("Experiment 3: Time and Phase Shift Analysis")

    shift_ms_values = [0, 10, 50]
    # Use the dominant frequency (peak-magnitude FFT bin) for phase calculation
    freqs, mag = compute_fft(signal, sr)
    dominant_freq = freqs[np.argmax(mag)]
    results = []

    fig, axes = plt.subplots(len(shift_ms_values), 2, figsize=(14, 10))
    fig.suptitle("Experiment 3: Time / Phase Shift Comparison", fontsize=14)

    for idx, shift_ms in enumerate(shift_ms_values):
        shifted = apply_time_shift(signal, sr, shift_ms)
        phase_deg = phase_shift_from_time_shift(shift_ms, dominant_freq)

        snr = calculate_snr(signal, shifted)
        _print_row(f"Shift {shift_ms} ms – phase @ {dominant_freq:.0f} Hz",
                   f"{phase_deg:.1f}°")
        _print_row(f"Shift {shift_ms} ms – SNR", f"{snr:.2f} dB")

        results.append({
            "shift_ms": shift_ms,
            "shift_samples": int(sr * shift_ms / 1000),
            "dominant_freq_hz": round(dominant_freq, 1),
            "phase_shift_deg": round(phase_deg, 2),
            "snr_db": round(snr, 2),
        })

        audio_path = os.path.join(dirs["audio"], f"03_shift_{shift_ms}ms.wav")
        save_audio(audio_path, shifted, sr)

        t = np.arange(len(signal)) / sr
        axes[idx, 0].plot(t, signal, alpha=0.5, label="Original", linewidth=0.5)
        axes[idx, 0].plot(t, shifted, alpha=0.7, label=f"Shift {shift_ms} ms",
                          linewidth=0.5)
        axes[idx, 0].set_title(f"Waveform: {shift_ms} ms shift")
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)

        orig_freqs, orig_mag = compute_fft(signal, sr)
        shift_freqs, shift_mag = compute_fft(shifted, sr)
        axes[idx, 1].plot(orig_freqs, orig_mag, alpha=0.5, label="Original",
                          linewidth=0.7)
        axes[idx, 1].plot(shift_freqs, shift_mag, alpha=0.7,
                          label=f"Shifted {shift_ms} ms", linewidth=0.7)
        axes[idx, 1].set_title(f"Spectrum: {shift_ms} ms shift")
        axes[idx, 1].set_xlabel("Frequency (Hz)")
        axes[idx, 1].set_ylabel("Magnitude")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "03_time_phase_shift.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "03_time_phase_shift_results.csv"))
    return {"experiment": "time_phase_shift", "results": results}


# ---------------------------------------------------------------------------
# Experiment 4 – STFT Parameter Study
# ---------------------------------------------------------------------------

def experiment_stft(signal: np.ndarray, sr: int, dirs: dict) -> dict:
    """Compare 8 STFT configurations (2 frame sizes × 2 hop lengths × 2 windows)."""
    _print_header("Experiment 4: STFT Parameter Study")

    frame_sizes = [512, 2048]
    hop_lengths = [256, 128]
    windows = ["hann", "hamming"]

    configs = [
        (fs, hl, win)
        for fs in frame_sizes
        for hl in hop_lengths
        for win in windows
    ]

    results = []
    n_configs = len(configs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Experiment 4: STFT Configurations (dB Spectrograms)", fontsize=13)
    axes_flat = axes.flatten()

    for plot_idx, (frame_size, hop_length, window) in enumerate(configs):
        stft_mat, t_axis, f_axis = compute_stft(
            signal, sr,
            frame_size=frame_size,
            hop_length=hop_length,
            window=window,
        )
        db_spec = stft_to_db(stft_mat)
        summary = stft_parameter_summary(frame_size, hop_length, window, sr)

        label = (f"N={frame_size}, H={hop_length}, {window}\n"
                 f"Δt={summary['time_res_ms']:.1f}ms, "
                 f"Δf={summary['freq_res_hz']:.1f}Hz")
        _print_row(f"Config {plot_idx + 1}", label.replace("\n", " | "))

        results.append({
            "config_id": plot_idx + 1,
            "frame_size": frame_size,
            "hop_length": hop_length,
            "window": window,
            "time_res_ms": round(summary["time_res_ms"], 3),
            "freq_res_hz": round(summary["freq_res_hz"], 3),
            "overlap_pct": round(summary["overlap_pct"], 1),
            "n_frames": stft_mat.shape[1],
            "n_freq_bins": stft_mat.shape[0],
        })

        ax = axes_flat[plot_idx]
        img = ax.imshow(
            db_spec,
            aspect="auto",
            origin="lower",
            extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]],
            cmap="magma",
        )
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Freq (Hz)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "04_stft_comparison.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "04_stft_results.csv"))
    return {"experiment": "stft", "results": results}


# ---------------------------------------------------------------------------
# Experiment 5 – Clipping
# ---------------------------------------------------------------------------

def experiment_clipping(signal: np.ndarray, sr: int, dirs: dict) -> dict:
    """Apply hard and soft clipping at three thresholds."""
    _print_header("Experiment 5: Clipping Analysis")

    thresholds = [0.95, 0.6, 0.3]
    results = []

    fig, axes = plt.subplots(len(thresholds), 2, figsize=(14, 10))
    fig.suptitle("Experiment 5: Clipping Effects", fontsize=14)

    for idx, threshold in enumerate(thresholds):
        hard = hard_clip(signal, threshold)
        soft = soft_clip(signal, threshold)

        hard_dist = clipping_distortion_db(signal, hard)
        soft_dist = clipping_distortion_db(signal, soft)
        clipped_frac = clipped_sample_fraction(signal, threshold)
        hard_snr = calculate_snr(signal, hard)
        soft_snr = calculate_snr(signal, soft)

        _print_row(f"Threshold {threshold} – hard distortion",
                   f"{hard_dist:.2f} dB  (SNR {hard_snr:.1f} dB)")
        _print_row(f"Threshold {threshold} – soft distortion",
                   f"{soft_dist:.2f} dB  (SNR {soft_snr:.1f} dB)")
        _print_row(f"Threshold {threshold} – clipped fraction",
                   f"{clipped_frac * 100:.1f}%")

        results.append({
            "threshold": threshold,
            "clipped_fraction_pct": round(clipped_frac * 100, 1),
            "hard_clip_snr_db": round(hard_snr, 2),
            "hard_clip_distortion_db": round(hard_dist, 2),
            "soft_clip_snr_db": round(soft_snr, 2),
            "soft_clip_distortion_db": round(soft_dist, 2),
        })

        save_audio(
            os.path.join(dirs["audio"], f"05_hard_clip_{threshold}.wav"),
            hard, sr,
        )
        save_audio(
            os.path.join(dirs["audio"], f"05_soft_clip_{threshold}.wav"),
            soft, sr,
        )

        t = np.arange(len(signal)) / sr
        axes[idx, 0].plot(t, signal, alpha=0.4, label="Original", linewidth=0.5)
        axes[idx, 0].plot(t, hard, alpha=0.7, label="Hard clip", linewidth=0.5)
        axes[idx, 0].plot(t, soft, alpha=0.7, label="Soft clip", linewidth=0.5,
                          linestyle="--")
        axes[idx, 0].axhline(threshold, color="red", linestyle=":", linewidth=0.8)
        axes[idx, 0].axhline(-threshold, color="red", linestyle=":", linewidth=0.8)
        axes[idx, 0].set_title(f"Clipping @ threshold={threshold}")
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
        axes[idx, 0].legend(fontsize=7)
        axes[idx, 0].grid(True, alpha=0.3)

        _, orig_mag = compute_fft(signal, sr)
        freqs, hard_mag = compute_fft(hard, sr)
        _, soft_mag = compute_fft(soft, sr)
        axes[idx, 1].plot(freqs, orig_mag, alpha=0.4, label="Original", linewidth=0.7)
        axes[idx, 1].plot(freqs, hard_mag, alpha=0.7, label="Hard clip", linewidth=0.7)
        axes[idx, 1].plot(freqs, soft_mag, alpha=0.7, label="Soft clip",
                          linewidth=0.7, linestyle="--")
        axes[idx, 1].set_title(f"Spectrum @ threshold={threshold}")
        axes[idx, 1].set_xlabel("Frequency (Hz)")
        axes[idx, 1].set_ylabel("Magnitude")
        axes[idx, 1].legend(fontsize=7)
        axes[idx, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "05_clipping.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "05_clipping_results.csv"))
    return {"experiment": "clipping", "results": results}


# ---------------------------------------------------------------------------
# Experiment 6 – Aliasing (Downsampling)
# ---------------------------------------------------------------------------

def experiment_aliasing(signal: np.ndarray, sr: int, dirs: dict) -> dict:
    """Demonstrate aliasing by downsampling without anti-aliasing filter."""
    _print_header("Experiment 6: Aliasing / Downsampling Analysis")

    factors = [1, 2, 4]
    results = []

    fig, axes = plt.subplots(len(factors), 2, figsize=(14, 10))
    fig.suptitle("Experiment 6: Aliasing (no anti-alias filter)", fontsize=14)

    for idx, factor in enumerate(factors):
        ds_signal, new_sr = downsample(signal, sr, factor)
        nyq = nyquist_frequency(new_sr)

        # Resample back for SNR calculation
        back = resample_audio(ds_signal, new_sr, sr)
        snr = calculate_snr(signal, back)

        _print_row(f"Factor {factor}× – new SR", f"{new_sr} Hz")
        _print_row(f"Factor {factor}× – Nyquist", f"{nyq:.0f} Hz")
        _print_row(f"Factor {factor}× – SNR (back)", f"{snr:.2f} dB")

        results.append({
            "downsample_factor": factor,
            "original_sr_hz": sr,
            "new_sr_hz": new_sr,
            "nyquist_hz": nyq,
            "snr_after_resample_db": round(snr, 2),
            "n_samples_after": len(ds_signal),
        })

        save_audio(
            os.path.join(dirs["audio"], f"06_downsample_{factor}x.wav"),
            ds_signal, new_sr,
        )

        t_orig = np.arange(len(signal)) / sr
        t_ds = np.arange(len(ds_signal)) / new_sr
        axes[idx, 0].plot(t_orig, signal, alpha=0.5, label="Original", linewidth=0.5)
        axes[idx, 0].plot(t_ds, ds_signal, alpha=0.7, label=f"{factor}× DS",
                          linewidth=0.5)
        axes[idx, 0].set_title(f"Waveform: {factor}× downsampling → {new_sr} Hz")
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)

        orig_freqs, orig_mag = compute_fft(signal, sr)
        ds_freqs, ds_mag = compute_fft(ds_signal, new_sr)
        axes[idx, 1].plot(orig_freqs, orig_mag, alpha=0.5, label="Original",
                          linewidth=0.7)
        axes[idx, 1].plot(ds_freqs, ds_mag, alpha=0.7,
                          label=f"{factor}× DS ({new_sr} Hz)", linewidth=0.7)
        axes[idx, 1].axvline(nyq, color="red", linestyle="--", linewidth=1,
                             label=f"Nyquist = {nyq:.0f} Hz")
        axes[idx, 1].set_title(f"Spectrum: {factor}× downsampling")
        axes[idx, 1].set_xlabel("Frequency (Hz)")
        axes[idx, 1].set_ylabel("Magnitude")
        axes[idx, 1].legend(fontsize=7)
        axes[idx, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "06_aliasing.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "06_aliasing_results.csv"))
    return {"experiment": "aliasing", "results": results}


# ---------------------------------------------------------------------------
# Experiment 7 – DFT vs FFT
# ---------------------------------------------------------------------------

def experiment_dft_vs_fft(dirs: dict) -> dict:
    """Compare computational complexity of naive DFT versus numpy FFT."""
    _print_header("Experiment 7: DFT vs FFT Computational Complexity")

    lengths = [512, 1024, 2048]
    results = []

    timings_dft = []
    timings_fft = []
    speedups = []

    for n in lengths:
        bench = benchmark_dft_vs_fft(n)
        _print_row(f"N={n} DFT time", f"{bench['dft_time_s'] * 1000:.3f} ms")
        _print_row(f"N={n} FFT time", f"{bench['fft_time_s'] * 1000:.3f} ms")
        _print_row(f"N={n} speedup", f"{bench['speedup']:.1f}×")

        results.append({
            "signal_length": n,
            "dft_time_ms": round(bench["dft_time_s"] * 1000, 4),
            "fft_time_ms": round(bench["fft_time_s"] * 1000, 4),
            "speedup": round(bench["speedup"], 1),
            "dft_complexity": f"O(N²) = {n**2}",
            "fft_complexity": f"O(N log N) = {int(n * np.log2(n))}",
        })
        timings_dft.append(bench["dft_time_s"] * 1000)
        timings_fft.append(bench["fft_time_s"] * 1000)
        speedups.append(bench["speedup"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 7: DFT vs FFT Timing", fontsize=14)

    x = np.arange(len(lengths))
    width = 0.35
    axes[0].bar(x - width / 2, timings_dft, width, label="DFT (naive)")
    axes[0].bar(x + width / 2, timings_fft, width, label="FFT (numpy)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(n) for n in lengths])
    axes[0].set_xlabel("Signal Length (samples)")
    axes[0].set_ylabel("Execution Time (ms)")
    axes[0].set_title("Execution Time Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].plot(lengths, speedups, "o-", color="green", linewidth=2)
    axes[1].set_xlabel("Signal Length (samples)")
    axes[1].set_ylabel("Speedup (DFT / FFT)")
    axes[1].set_title("FFT Speedup over Naïve DFT")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, os.path.join(dirs["plots"], "07_dft_vs_fft.png"))

    df = pd.DataFrame(results)
    _save_csv(df, os.path.join(dirs["report_data"], "07_dft_vs_fft_results.csv"))
    return {"experiment": "dft_vs_fft", "results": results}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _generate_synthetic_audio(filepath: str, sr: int = 44100, duration: float = 5.0) -> None:
    """Write a multi-tone synthetic speech-like WAV file to *filepath*.

    The signal is a sum of sinusoids at typical speech fundamental and harmonic
    frequencies (100 Hz, 200 Hz, 300 Hz) with added broadband noise, providing
    a realistic test signal when no real recording is available.

    Parameters
    ----------
    filepath : str
        Destination WAV path (created automatically).
    sr : int
        Sample rate in Hz (default 44100).
    duration : float
        Duration in seconds (default 5.0).
    """
    import soundfile as sf

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech-like: fundamental + harmonics + gentle noise
    signal = (
        0.40 * np.sin(2 * np.pi * 100 * t)
        + 0.25 * np.sin(2 * np.pi * 200 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.10 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * np.random.default_rng(42).standard_normal(len(t))
    )
    # Normalise to [-1, 1]
    signal = signal / np.max(np.abs(signal))
    sf.write(filepath, signal.astype(np.float32), sr)


def run_all_experiments(audio_file: str, output_dir: str = "outputs") -> None:
    """Load *audio_file* and run all 7 experiments.

    If *audio_file* does not exist **or** cannot be decoded, a synthetic
    speech-like WAV is generated automatically so the experiments can always
    complete without manual setup.
    """
    # ------------------------------------------------------------------
    # Resolve audio source
    # ------------------------------------------------------------------
    audio_source_note = ""

    if not os.path.isfile(audio_file):
        if audio_file == "speech.wav":
            # Default file missing – silently create a synthetic one
            print(
                f"[INFO] '{audio_file}' not found – generating a synthetic test signal …"
            )
            _generate_synthetic_audio(audio_file)
            audio_source_note = "  (synthetic test signal – replace with a real recording for best results)"
        else:
            # User explicitly named a file that doesn't exist → hard error
            print(
                f"[ERROR] Audio file not found: {audio_file}\n"
                "        Please provide a valid WAV/MP3/FLAC file as the first argument.\n"
                "        Example: python experimental_runner.py speech.wav",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\n{'#' * 70}")
    print("#  CS425 Assignment 1 – Experimental Runner")
    print(f"#  Audio file : {audio_file}{audio_source_note}")
    print(f"#  Output dir : {output_dir}")
    print(f"{'#' * 70}\n")

    dirs = ensure_output_dirs(output_dir)

    print(f"[Loading] {audio_file} …")
    try:
        signal, sr = load_audio(audio_file)
    except Exception as exc:
        # File exists but is unreadable (e.g. downloaded HTML page instead of
        # a real audio file).  Fall back to a synthetic signal automatically.
        warnings.warn(
            f"Could not load '{audio_file}' ({exc}). "
            "Falling back to a synthetic test signal.",
            stacklevel=2,
        )
        synthetic_path = os.path.splitext(audio_file)[0] + "_synthetic.wav"
        print(f"[INFO] Writing fallback synthetic audio → {synthetic_path}")
        _generate_synthetic_audio(synthetic_path)
        signal, sr = load_audio(synthetic_path)
        audio_file = synthetic_path
    duration = len(signal) / sr
    print(f"[Loaded]  {len(signal)} samples @ {sr} Hz  ({duration:.2f} s)\n")

    all_results: list[dict] = []
    t_start = _time.perf_counter()

    all_results.append(experiment_sampling_rate(signal, sr, dirs))
    all_results.append(experiment_quantization(signal, sr, dirs))
    all_results.append(experiment_time_phase_shift(signal, sr, dirs))
    all_results.append(experiment_stft(signal, sr, dirs))
    all_results.append(experiment_clipping(signal, sr, dirs))
    all_results.append(experiment_aliasing(signal, sr, dirs))
    all_results.append(experiment_dft_vs_fft(dirs))

    elapsed = _time.perf_counter() - t_start

    # Save JSON summary
    json_path = os.path.join(dirs["report_data"], "all_experiments_results.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n  [JSON] {json_path}")

    print(f"\n{'=' * 70}")
    print(f"  All 7 experiments complete in {elapsed:.1f} s")
    print(f"  Plots      → {dirs['plots']}/")
    print(f"  Audio      → {dirs['audio']}/")
    print(f"  CSV tables → {dirs['report_data']}/")
    print(f"{'=' * 70}\n")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CS425 Assignment 1 – run all 7 audio experiments"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="speech.wav",
        help="Input audio file (default: speech.wav)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for outputs (default: outputs)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_all_experiments(args.audio_file, args.output_dir)
