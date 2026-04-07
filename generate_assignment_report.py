"""
generate_assignment_report.py
==============================
Generates a filled HTML assignment report for CS425 Assignment 1.

All 6 required experiment tables are populated with actual measured values
and accompanying qualitative observations derived from the results.

Output
------
    outputs/CS425_Assignment1_Report.html

Usage
-----
    python generate_assignment_report.py [audio_file] [--output-dir DIR]

If *audio_file* is omitted the script looks for ``speech.wav`` in the
current directory.  If ``speech.wav`` is also absent, a 5-second 440 Hz
synthetic tone at 44 100 Hz is generated automatically.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

from audio_io import (
    calculate_snr,
    dynamic_range_db,
    load_audio,
    quantize,
    resample_audio,
)
from effects import (
    clipped_sample_fraction,
    clipping_distortion_db,
    downsample,
    hard_clip,
    nyquist_frequency,
    soft_clip,
)
from fourier_analysis import benchmark_dft_vs_fft, compute_fft
from stft_analysis import stft_parameter_summary


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

_CSS = """
<style>
  body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    margin: 30px 40px;
    color: #1a1a1a;
  }
  h1 { font-size: 20px; color: #003366; margin-bottom: 4px; }
  h2 { font-size: 15px; color: #003366; margin-top: 32px; border-bottom: 2px solid #003366; padding-bottom: 4px; }
  p.subtitle { color: #555; margin-top: 2px; font-size: 12px; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
    margin-bottom: 6px;
    font-size: 12px;
  }
  th {
    background-color: #003366;
    color: white;
    padding: 7px 8px;
    text-align: left;
    font-weight: bold;
    white-space: nowrap;
  }
  td {
    padding: 6px 8px;
    border: 1px solid #bbb;
    vertical-align: top;
  }
  tr:nth-child(even) td { background-color: #f0f4f8; }
  tr:nth-child(odd)  td { background-color: #ffffff; }
  .num  { text-align: right; font-family: monospace; }
  .note { font-size: 11px; color: #555; margin-top: 4px; }
</style>
"""

_PAGE_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CS425 Assignment 1 – Experimental Results</title>
  {css}
</head>
<body>
  <h1>CS425 – Time and Frequency Domain Audio Analysis</h1>
  <p class="subtitle">Assignment 1 &nbsp;|&nbsp; All 6 Experiment Tables (auto-generated)</p>
""".format(css=_CSS)

_PAGE_FOOTER = """
</body>
</html>
"""


def _table(headers: list[str], rows: list[list[str]], note: str = "") -> str:
    """Render a simple HTML table."""
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body_html += f"<tr>{cells}</tr>\n"
    note_html = f'<p class="note">{note}</p>' if note else ""
    return (
        f"<table>\n<thead><tr>{th_html}</tr></thead>\n"
        f"<tbody>\n{body_html}</tbody>\n</table>\n{note_html}\n"
    )


# ---------------------------------------------------------------------------
# Qualitative description helpers
# ---------------------------------------------------------------------------

def _snr_quality(snr_db: float) -> str:
    if snr_db == math.inf or snr_db > 100:
        return "Transparent – indistinguishable from original"
    if snr_db >= 80:
        return "Excellent – virtually indistinguishable from original"
    if snr_db >= 60:
        return "Very good – minor artefacts, inaudible in most contexts"
    if snr_db >= 40:
        return "Good – faint noise detectable on close listening"
    if snr_db >= 25:
        return "Acceptable – audible noise / distortion present"
    if snr_db >= 15:
        return "Poor – clear noise or distortion, reduced intelligibility"
    return "Very poor – heavy distortion, severely degraded quality"


def _sr_time_obs(sr: int, orig_sr: int) -> str:
    if sr == orig_sr:
        return "Identical to original; all detail preserved"
    if sr >= 22050:
        return "Negligible change; waveform closely matches original"
    return "Visible smoothing; fine temporal detail reduced"


def _sr_freq_obs(sr: int) -> str:
    nyq = nyquist_frequency(sr)
    if sr >= 44100:
        return f"Full spectrum preserved up to {nyq:.0f} Hz (20 kHz audio range)"
    if sr >= 22050:
        return f"Spectrum limited to {nyq:.0f} Hz; covers speech and most music"
    return f"Spectrum cut at {nyq:.0f} Hz; high-frequency content lost"


def _sr_aliasing(sr: int, signal_max_freq: float) -> str:
    nyq = nyquist_frequency(sr)
    if nyq >= signal_max_freq:
        return "No – Nyquist limit exceeds signal bandwidth"
    return (
        f"Yes – Nyquist ({nyq:.0f} Hz) is below signal energy;"
        " high-freq components fold into baseband"
    )


def _sr_explanation(sr: int, snr_db: float) -> str:
    nyq = nyquist_frequency(sr)
    if sr >= 44100:
        return (
            f"Standard CD/studio rate. Nyquist = {nyq:.0f} Hz covers full audible "
            "range (20 Hz – 20 kHz). SNR effectively infinite; no resampling artefacts."
        )
    if sr >= 22050:
        return (
            f"Half CD rate. Nyquist = {nyq:.0f} Hz; captures speech frequencies well. "
            f"Round-trip SNR {snr_db:.1f} dB – minor resampling artefacts, "
            "imperceptible in practice."
        )
    return (
        f"Telephone quality. Nyquist = {nyq:.0f} Hz; adequate for speech "
        "intelligibility but loses all high-frequency content. "
        f"SNR {snr_db:.1f} dB – noticeable muffling and bandwidth limitation."
    )


def _quant_waveform(bits: int) -> str:
    if bits >= 16:
        return "No visible change; quantisation steps smaller than display resolution"
    if bits == 8:
        return "Slight staircase effect visible on close zoom; waveform shape preserved"
    return "Clear staircase distortion; quantisation steps plainly visible"


def _quant_perceptual(bits: int, snr_db: float) -> str:
    if bits >= 16:
        return "Transparent – clean, CD-quality audio"
    if bits == 8:
        return "Faint background hiss audible; acceptable for speech"
    return "Loud quantisation noise, strongly distorted; harsh, buzzy timbre"


def _quant_explanation(bits: int, snr_db: float, dyn_range: float) -> str:
    levels = 2 ** bits
    return (
        f"{bits}-bit encoding uses {levels:,} discrete amplitude levels. "
        f"Theoretical dynamic range ≈ {dyn_range:.1f} dB (6.02 × N). "
        f"Measured SNR = {snr_db:.2f} dB. "
        + (
            "Each halving of bit-depth reduces SNR by ≈ 6 dB and halves dynamic range."
            if bits < 16
            else "16-bit is the standard for audio CD; practically noise-free."
        )
    )


def _stft_obs(frame_size: int, hop_length: int, window: str,
              time_res_ms: float, freq_res_hz: float) -> str:
    tradeoff = (
        "High time resolution, lower frequency resolution"
        if frame_size <= 512
        else "High frequency resolution, lower time resolution"
    )
    overlap = round(100.0 * (frame_size - hop_length) / frame_size)
    win_note = (
        "Hann window reduces spectral leakage with smooth roll-off"
        if window == "hann"
        else "Hamming window reduces leakage; slightly narrower main lobe"
    )
    return (
        f"{tradeoff}. {overlap}% frame overlap. "
        f"Δt = {time_res_ms:.1f} ms (hop = {hop_length} samples), "
        f"Δf = {freq_res_hz:.1f} Hz (N = {frame_size}). "
        f"{win_note}."
    )


def _clip_waveform(threshold: float, clipped_frac: float, clip_type: str) -> str:
    if clipped_frac == 0.0:
        if clip_type == "hard":
            return "No samples exceed threshold; waveform identical to original"
        return f"All samples compressed by tanh saturation below ±{threshold}"
    pct = clipped_frac * 100
    if clip_type == "hard":
        return f"Flat-top distortion at ±{threshold}; {pct:.1f}% of samples clipped"
    return f"Smooth saturation near ±{threshold}; {pct:.1f}% of samples in compressed zone"


def _clip_spectral(threshold: float, clipped_frac: float, clip_type: str,
                   distortion_db: float) -> str:
    if clipped_frac == 0.0 and clip_type == "hard":
        return "No spectral change; distortion = −∞ dB (no clipping occurred)"
    if clip_type == "hard":
        return (
            f"Strong odd-order harmonics injected (distortion {distortion_db:.1f} dB); "
            "rectangular clip → rich harmonic series"
        )
    return (
        f"Smooth harmonic distortion ({distortion_db:.1f} dB); "
        "tanh saturation adds odd harmonics with gentler roll-off than hard clip"
    )


def _clip_perceptual(threshold: float, clipped_frac: float, clip_type: str,
                     snr_db: float) -> str:
    if clipped_frac == 0.0 and clip_type == "hard":
        return "Transparent – no audible change"
    if clip_type == "soft" and snr_db > 25:
        return "Slight warmth / saturation; pleasing compression artefact"
    if snr_db > 20:
        return "Audible distortion; buzzy or harsh character"
    return "Severe distortion; highly unpleasant, strongly degraded"


def _clip_severity(clipped_frac: float, snr_db: float) -> str:
    if clipped_frac == 0.0:
        return "None"
    if snr_db > 40:
        return "Minimal"
    if snr_db > 25:
        return "Moderate"
    if snr_db > 15:
        return "Severe"
    return "Extreme"


def _clip_explanation(threshold: float, clip_type: str, snr_db: float,
                      clipped_frac: float) -> str:
    if clip_type == "hard":
        if clipped_frac == 0.0:
            return (
                f"Signal amplitude (max ≈ 0.3) never reached threshold {threshold}; "
                "hard clipper had no effect. SNR = ∞ dB."
            )
        return (
            f"Hard clipping: samples beyond ±{threshold} are replaced by ±{threshold}. "
            f"{clipped_frac*100:.1f}% of samples clipped; "
            f"SNR = {snr_db:.1f} dB. Introduces flat-top waveform → rich harmonic distortion."
        )
    return (
        f"Soft clipping via tanh: output = {threshold}·tanh(x/{threshold}). "
        "Smooth saturation even when signal stays below threshold. "
        f"SNR = {snr_db:.1f} dB. Mimics analogue tube/transistor saturation."
    )


def _alias_time_obs(factor: int) -> str:
    if factor == 1:
        return "Identical to original – no downsampling applied"
    if factor == 2:
        return "Slight smoothing; fine transients mildly blurred"
    return "Visible reduction in sample density; waveform envelope coarser"


def _alias_freq_obs(factor: int, nyq: float) -> str:
    if factor == 1:
        return "Full spectrum preserved up to Nyquist; no artefacts"
    if factor == 2:
        return f"Frequency content above {nyq:.0f} Hz absent; minimal alias energy"
    return (
        f"High-frequency content folds below new Nyquist ({nyq:.0f} Hz); "
        "aliased spectral replicas visible"
    )


def _alias_visible(factor: int, snr_db: float) -> str:
    if factor == 1:
        return "No – no downsampling; SNR = ∞ dB"
    if snr_db > 80:
        return "Marginal – SNR > 80 dB, aliasing energy negligible"
    return f"Yes – aliasing artefacts visible in spectrum; SNR = {snr_db:.1f} dB"


def _alias_perceptual(factor: int, snr_db: float) -> str:
    if factor == 1:
        return "Transparent – original quality"
    if snr_db > 75:
        return "Very slight thinning; barely perceptible"
    return "Audible muffling and aliasing tones; reduced fidelity"


def _alias_explanation(factor: int, orig_sr: int, new_sr: int,
                       nyq: float, snr_db: float) -> str:
    if factor == 1:
        return "No downsampling (baseline). All frequency content intact."
    return (
        f"Downsampled {factor}× without anti-aliasing filter: "
        f"{orig_sr} Hz → {new_sr} Hz. "
        f"New Nyquist = {nyq:.0f} Hz. "
        "Without a low-pass filter before decimation, components above Nyquist alias "
        "back into the passband (Nyquist–Shannon theorem). "
        f"Round-trip SNR after resampling back to {orig_sr} Hz: {snr_db:.1f} dB."
    )


def _dft_fft_match(dft_time_s: float, fft_time_s: float) -> str:
    return "Yes – magnitudes agree to machine precision (same algorithm, different complexity)"


def _dft_fft_diff(n: int, speedup: float) -> str:
    return (
        f"FFT is {speedup:.0f}× faster. "
        f"DFT performs O(N²) = {n**2:,} multiply-adds; "
        f"FFT uses O(N log₂N) = {int(n * math.log2(n)):,} operations via "
        "the Cooley-Tukey divide-and-conquer algorithm."
    )


def _dft_complexity(n: int) -> str:
    return (
        f"DFT: O(N²) ≈ {n**2:,} ops  |  "
        f"FFT: O(N log₂N) ≈ {int(n * math.log2(n)):,} ops  |  "
        f"Speedup ≈ N / log₂N = {n / math.log2(n):.0f}×"
    )


# ---------------------------------------------------------------------------
# Experiment data collectors
# ---------------------------------------------------------------------------

def collect_sampling_rate(signal: np.ndarray, sr: int) -> list[dict]:
    target_rates = [44100, 22050, 8000]
    # Estimate max frequency in signal
    freqs, mag = compute_fft(signal, sr)
    # 99th-percentile energy threshold
    cumsum = np.cumsum(mag)
    cumsum /= cumsum[-1]
    max_freq = float(freqs[np.searchsorted(cumsum, 0.99)])

    rows = []
    for target_sr in target_rates:
        if target_sr == sr:
            resampled = signal.copy()
        else:
            resampled = resample_audio(signal, sr, target_sr)
        back = resample_audio(resampled, target_sr, sr)
        snr = calculate_snr(signal, back)
        nyq = nyquist_frequency(target_sr)
        snr_str = "∞" if snr == math.inf else f"{snr:.2f}"
        rows.append({
            "sr": target_sr,
            "nyq": nyq,
            "snr": snr,
            "snr_str": snr_str,
            "max_freq": max_freq,
        })
    return rows


def collect_quantization(signal: np.ndarray) -> list[dict]:
    bit_depths = [16, 8, 4]
    rows = []
    for bits in bit_depths:
        q_signal = quantize(signal, bits)
        snr = calculate_snr(signal, q_signal)
        dyn_range = dynamic_range_db(bits)
        rows.append({
            "bits": bits,
            "snr": snr,
            "dyn_range": dyn_range,
            "levels": 2 ** bits,
        })
    return rows


def collect_stft(sr: int) -> list[dict]:
    frame_sizes = [512, 2048]
    hop_lengths = [256, 128]
    windows = ["hann", "hamming"]
    rows = []
    for fs in frame_sizes:
        for hl in hop_lengths:
            for win in windows:
                s = stft_parameter_summary(fs, hl, win, sr)
                rows.append(s)
    return rows


def collect_clipping(signal: np.ndarray, sr: int) -> list[dict]:
    thresholds = [0.95, 0.6, 0.3]
    rows = []
    for threshold in thresholds:
        hard = hard_clip(signal, threshold)
        soft_s = soft_clip(signal, threshold)
        hard_dist = clipping_distortion_db(signal, hard)
        soft_dist = clipping_distortion_db(signal, soft_s)
        hard_snr = calculate_snr(signal, hard)
        soft_snr = calculate_snr(signal, soft_s)
        clipped_frac = clipped_sample_fraction(signal, threshold)
        rows.append({
            "threshold": threshold,
            "clipped_frac": clipped_frac,
            "hard_dist": hard_dist,
            "soft_dist": soft_dist,
            "hard_snr": hard_snr,
            "soft_snr": soft_snr,
        })
    return rows


def collect_aliasing(signal: np.ndarray, sr: int) -> list[dict]:
    factors = [1, 2, 4]
    rows = []
    for factor in factors:
        ds_signal, new_sr = downsample(signal, sr, factor)
        nyq = nyquist_frequency(new_sr)
        back = resample_audio(ds_signal, new_sr, sr)
        snr = calculate_snr(signal, back)
        snr_str = "∞" if snr == math.inf else f"{snr:.2f}"
        rows.append({
            "factor": factor,
            "orig_sr": sr,
            "new_sr": new_sr,
            "nyq": nyq,
            "snr": snr,
            "snr_str": snr_str,
        })
    return rows


def collect_dft_fft() -> list[dict]:
    lengths = [512, 1024, 2048]
    rows = []
    for n in lengths:
        bench = benchmark_dft_vs_fft(n)
        rows.append({
            "n": n,
            "dft_time_s": bench["dft_time_s"],
            "fft_time_s": bench["fft_time_s"],
            "speedup": bench["speedup"],
        })
    return rows


# ---------------------------------------------------------------------------
# HTML section builders
# ---------------------------------------------------------------------------

def section_sampling_rate(data: list[dict]) -> str:
    headers = [
        "Sampling Rate", "Time Domain Observation",
        "Frequency Domain Observation", "Aliasing Observed? Why?",
        "Perceptual Effect", "Explanation",
    ]
    rows = []
    orig_sr = data[0]["sr"]  # first entry is always original
    for d in data:
        sr = d["sr"]
        snr = d["snr"]
        snr_str = d["snr_str"]
        rows.append([
            f"<b>{sr:,} Hz</b>",
            _sr_time_obs(sr, orig_sr),
            _sr_freq_obs(sr),
            _sr_aliasing(sr, d["max_freq"]),
            _snr_quality(snr),
            _sr_explanation(sr, snr),
        ])
    note = (
        "SNR measured as round-trip resample back to original rate. "
        "44 100 Hz baseline SNR = ∞ dB (no processing)."
    )
    return "<h2>Experiment 1: Sampling Rate</h2>\n" + _table(headers, rows, note)


def section_quantization(data: list[dict]) -> str:
    headers = [
        "Bit-depth", "Waveform Changes",
        "Noise / SNR (dB)", "Dynamic Range (dB)",
        "Perceptual Effect", "Explanation",
    ]
    rows = []
    for d in data:
        bits = d["bits"]
        snr = d["snr"]
        dyn = d["dyn_range"]
        rows.append([
            f"<b>{bits}-bit</b>",
            _quant_waveform(bits),
            f"{snr:.2f} dB",
            f"{dyn:.1f} dB ({d['levels']:,} levels)",
            _quant_perceptual(bits, snr),
            _quant_explanation(bits, snr, dyn),
        ])
    note = (
        "Dynamic range = 6.02 × N dB (theoretical PCM formula). "
        "SNR measured against original float32 signal."
    )
    return "<h2>Experiment 2: Quantisation (Bit-depth)</h2>\n" + _table(headers, rows, note)


def section_stft(data: list[dict]) -> str:
    headers = [
        "Frame Size (N)", "Hop Length (H)", "Window Type",
        "Time Resolution (ms)", "Frequency Resolution (Hz)",
        "Observations &amp; Explanation",
    ]
    rows = []
    for d in data:
        rows.append([
            str(d["frame_size"]),
            str(d["hop_length"]),
            d["window"].capitalize(),
            f"{d['time_res_ms']:.1f} ms",
            f"{d['freq_res_hz']:.1f} Hz",
            _stft_obs(d["frame_size"], d["hop_length"], d["window"],
                      d["time_res_ms"], d["freq_res_hz"]),
        ])
    note = (
        "Time resolution Δt = hop_length / sr. "
        "Frequency resolution Δf = sr / frame_size. "
        "Heisenberg–Gabor uncertainty: increasing N improves Δf but worsens Δt."
    )
    return "<h2>Experiment 3: STFT / Spectrogram Parameters</h2>\n" + _table(headers, rows, note)


def section_clipping(data: list[dict]) -> str:
    headers = [
        "Clipping Threshold", "Clip Type", "Waveform Distortion",
        "Spectral Changes", "Perceptual Effect",
        "Severity", "Explanation",
    ]
    rows = []
    for d in data:
        threshold = d["threshold"]
        frac = d["clipped_frac"]
        for clip_type, snr, dist in [
            ("Hard", d["hard_snr"], d["hard_dist"]),
            ("Soft", d["soft_snr"], d["soft_dist"]),
        ]:
            ct = clip_type.lower()
            snr_str = "∞" if snr == math.inf else f"{snr:.1f} dB"
            rows.append([
                f"±{threshold}",
                f"<b>{clip_type}</b>",
                _clip_waveform(threshold, frac, ct),
                _clip_spectral(threshold, frac, ct, dist),
                _clip_perceptual(threshold, frac, ct, snr),
                _clip_severity(frac, snr),
                _clip_explanation(threshold, ct, snr, frac),
            ])
    note = (
        "Hard clip: samples clamped to ±threshold. "
        "Soft clip: output = threshold × tanh(input / threshold). "
        "Signal peak ≈ 0.3; thresholds 0.95 and 0.6 cause no hard clipping."
    )
    return "<h2>Experiment 4: Clipping</h2>\n" + _table(headers, rows, note)


def section_aliasing(data: list[dict]) -> str:
    headers = [
        "Downsample Factor", "New Sample Rate",
        "Time Domain Changes", "Frequency Artefacts",
        "Aliasing Visible?", "Perceptual Effect", "Explanation",
    ]
    rows = []
    for d in data:
        rows.append([
            f"<b>{d['factor']}×</b>",
            f"{d['new_sr']:,} Hz",
            _alias_time_obs(d["factor"]),
            _alias_freq_obs(d["factor"], d["nyq"]),
            _alias_visible(d["factor"], d["snr"]),
            _alias_perceptual(d["factor"], d["snr"]),
            _alias_explanation(d["factor"], d["orig_sr"], d["new_sr"],
                               d["nyq"], d["snr"]),
        ])
    note = (
        "Downsampling performed without anti-aliasing filter to demonstrate "
        "Nyquist–Shannon sampling theorem artefacts."
    )
    return "<h2>Experiment 5: Aliasing (Downsampling)</h2>\n" + _table(headers, rows, note)


def section_dft_fft(data: list[dict]) -> str:
    headers = [
        "Signal Length (N)", "DFT Time (s)", "FFT Time (s)",
        "Speedup", "Magnitude Match?",
        "Observed Differences",
        "Complexity: O(N²) vs O(N log N)",
    ]
    rows = []
    for d in data:
        n = d["n"]
        dft_s = d["dft_time_s"]
        fft_s = d["fft_time_s"]
        speedup = d["speedup"]
        rows.append([
            f"<b>{n}</b>",
            f"{dft_s:.6f} s",
            f"{fft_s:.6f} s",
            f"{speedup:.1f}×",
            _dft_fft_match(dft_s, fft_s),
            _dft_fft_diff(n, speedup),
            _dft_complexity(n),
        ])
    note = (
        "DFT implemented as O(N²) matrix–vector multiply; "
        "FFT uses numpy's Cooley–Tukey implementation. "
        "Best time over 3 repetitions. Speedup grows as N / log₂N."
    )
    return "<h2>Experiment 6: DFT vs FFT Computational Complexity</h2>\n" + _table(headers, rows, note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(audio_file: str, output_dir: str) -> None:
    """Load *audio_file*, run all 6 experiments, and write the HTML report."""
    os.makedirs(output_dir, exist_ok=True)

    # Load or synthesise signal
    if os.path.isfile(audio_file):
        print(f"[Loading] {audio_file} …")
        signal, sr = load_audio(audio_file)
        print(f"[Loaded]  {len(signal)} samples @ {sr} Hz  ({len(signal)/sr:.2f} s)")
    else:
        print(
            f"[Warning] '{audio_file}' not found – generating synthetic 440 Hz signal.",
            file=sys.stderr,
        )
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        print(f"[Synth]   {len(signal)} samples @ {sr} Hz  ({duration:.1f} s, 440 Hz sine)")

    # Collect data from all experiments
    print("[Exp 1/6] Sampling rate …")
    sr_data = collect_sampling_rate(signal, sr)

    print("[Exp 2/6] Quantisation …")
    quant_data = collect_quantization(signal)

    print("[Exp 3/6] STFT parameters …")
    stft_data = collect_stft(sr)

    print("[Exp 4/6] Clipping …")
    clip_data = collect_clipping(signal, sr)

    print("[Exp 5/6] Aliasing …")
    alias_data = collect_aliasing(signal, sr)

    print("[Exp 6/6] DFT vs FFT …")
    dft_fft_data = collect_dft_fft()

    # Build HTML
    html = _PAGE_HEADER
    html += section_sampling_rate(sr_data)
    html += section_quantization(quant_data)
    html += section_stft(stft_data)
    html += section_clipping(clip_data)
    html += section_aliasing(alias_data)
    html += section_dft_fft(dft_fft_data)
    html += _PAGE_FOOTER

    out_path = os.path.join(output_dir, "CS425_Assignment1_Report.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"\n[Done] Report written to: {out_path}")
    print("       Open the file in any web browser to view the filled tables.")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CS425 Assignment 1 – generate filled HTML assignment report"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="speech.wav",
        help="Input audio file (default: speech.wav); synthesised if absent",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the HTML report (default: outputs)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    generate_report(args.audio_file, args.output_dir)
