"""
run_tts_experiments.py
======================
Automated TTS parameter sweep – CS425 Assignment 2, Part 2 (Table 3).

For each parameter (one at a time):
  1. Generate a baseline TTS audio signal.
  2. Apply the modified parameter value.
  3. Save a side-by-side before/after spectrogram PNG.
  4. Save the processed audio WAV.
  5. Print a figure-summary block for pasting into the report.

Parameters covered
------------------
  Synthesis/Voice   : speech_rate, voice_index, volume
  Signal processing : pre_emphasis, noise_level, pitch_steps, speed_factor,
                      low_cut, high_cut, gain

Output directory: outputs/tts/

Usage
-----
    python run_tts_experiments.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import soundfile as sf

from tts_template import (
    generate_tts_audio,
    process_tts_signal,
    plot_spectrogram_comparison,
)

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------

TEXT = "The quick brown fox jumps over the lazy dog."

# Baseline synthesis parameters
BASELINE_TTS = dict(rate=150, voice_index=0, volume=1.0, sr=22050)

OUTPUT_DIR = "outputs/tts"

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each dict describes one parameter sweep (two values – A and B).
#
# tts_key   – key in BASELINE_TTS to vary for synthesis/voice experiments
#             (None for signal-processing experiments)
# signal_key – key in process_tts_signal() kwargs to vary
#             (None for synthesis/voice experiments)
# fig_nos   – Figure numbers (A, B) for the report (Table 3 figures start at 12)

TTS_EXPERIMENTS = [
    # ── Synthesis / Voice parameters ─────────────────────────────────────────
    {
        "param":      "speech_rate",
        "values":     [120, 220],
        "labels":     ["Speech Rate 120 (slow/clear)",
                       "Speech Rate 220 (fast/rushed)"],
        "tts_key":    "rate",
        "signal_key": None,
        "fig_nos":    [12, 13],
    },
    {
        "param":      "voice_index",
        "values":     [0, 1],
        "labels":     ["Voice Index 0 (default voice)",
                       "Voice Index 1 (alternate voice)"],
        "tts_key":    "voice_index",
        "signal_key": None,
        "fig_nos":    [14, 15],
    },
    {
        "param":      "volume",
        "values":     [0.5, 1.0],
        "labels":     ["Volume 0.5 (quiet)",
                       "Volume 1.0 (full)"],
        "tts_key":    "volume",
        "signal_key": None,
        "fig_nos":    [16, 17],
    },
    # ── Signal processing parameters ─────────────────────────────────────────
    {
        "param":      "pre_emphasis",
        "values":     [0.0, 0.95],
        "labels":     ["Pre-emphasis 0.0 (none)",
                       "Pre-emphasis 0.95 (boosted high-freq)"],
        "tts_key":    None,
        "signal_key": "pre_emphasis",
        "fig_nos":    [18, 19],
    },
    {
        "param":      "noise_level",
        "values":     [0.0, 0.02],
        "labels":     ["Noise Level 0.0 (clean)",
                       "Noise Level 0.02 (moderate noise)"],
        "tts_key":    None,
        "signal_key": "noise_level",
        "fig_nos":    [20, 21],
    },
    {
        "param":      "pitch_steps",
        "values":     [-2, 2],
        "labels":     ["Pitch Shift -2 semitones (lower voice)",
                       "Pitch Shift +2 semitones (higher voice)"],
        "tts_key":    None,
        "signal_key": "pitch_steps",
        "fig_nos":    [22, 23],
    },
    {
        "param":      "speed_factor",
        "values":     [0.8, 1.3],
        "labels":     ["Time-stretch 0.8 (slower speech)",
                       "Time-stretch 1.3 (faster speech)"],
        "tts_key":    None,
        "signal_key": "speed_factor",
        "fig_nos":    [24, 25],
    },
    {
        "param":      "low_cut",
        "values":     [200, 500],
        "labels":     ["Low-cut 200 Hz (slight bass removal)",
                       "Low-cut 500 Hz (telephone-like)"],
        "tts_key":    None,
        "signal_key": "low_cut",
        "fig_nos":    [26, 27],
    },
    {
        "param":      "high_cut",
        "values":     [3000, 4000],
        "labels":     ["High-cut 3000 Hz (narrowband / muffled)",
                       "High-cut 4000 Hz (slight treble roll-off)"],
        "tts_key":    None,
        "signal_key": "high_cut",
        "fig_nos":    [28, 29],
    },
    {
        "param":      "gain",
        "values":     [0.7, 2.0],
        "labels":     ["Gain 0.7 (reduced amplitude)",
                       "Gain 2.0 (amplified / may clip)"],
        "tts_key":    None,
        "signal_key": "gain",
        "fig_nos":    [30, 31],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _default_sp_kwargs() -> dict:
    """Return signal-processing kwargs with all parameters at their neutral values."""
    return dict(
        pre_emphasis=0.0,
        noise_level=0.0,
        pitch_steps=0,
        speed_factor=1.0,
        low_cut=0,
        high_cut=0,
        gain=1.0,
    )


def run_tts_experiment(exp: dict, output_dir: str) -> list[dict]:
    """Run one TTS parameter experiment (two values) and save all outputs.

    Returns a list of summary dicts (one per value).
    """
    os.makedirs(output_dir, exist_ok=True)
    summaries: list[dict] = []

    tts_key = exp["tts_key"]
    signal_key = exp["signal_key"]

    # Generate baseline TTS once for this experiment
    baseline_signal, sr = generate_tts_audio(TEXT, **BASELINE_TTS)

    for value, label, fig_no in zip(exp["values"], exp["labels"], exp["fig_nos"]):

        if tts_key is not None:
            # Synthesis/voice param: re-synthesise with the new value
            tts_kwargs = dict(BASELINE_TTS)
            tts_kwargs[tts_key] = value
            modified_signal, sr = generate_tts_audio(TEXT, **tts_kwargs)
            original = baseline_signal
            processed = modified_signal
            title_original = "Baseline TTS (rate=150, voice=0, vol=1.0)"
            title_processed = label
        else:
            # Signal-processing param: process the baseline signal
            sp_kwargs = _default_sp_kwargs()
            sp_kwargs[signal_key] = value
            original = baseline_signal
            processed = process_tts_signal(baseline_signal, sr, **sp_kwargs)
            title_original = "Original (unprocessed)"
            title_processed = label

        # ── Plot ──────────────────────────────────────────────────────────────
        plot_filename = f"tts_{exp['param']}_{value}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plot_spectrogram_comparison(
            original, processed, sr,
            title_original=title_original,
            title_processed=f"Figure {fig_no}: {label}",
            save_path=plot_path,
        )

        # ── Audio ─────────────────────────────────────────────────────────────
        wav_path = os.path.join(output_dir, f"tts_{exp['param']}_{value}.wav")
        sf.write(wav_path, processed.astype("float32"), sr)

        print(f"  Figure {fig_no}: {label}")
        print(f"    Plot  → {plot_path}")
        print(f"    Audio → {wav_path}")

        summaries.append({
            "param":       exp["param"],
            "value":       value,
            "label":       label,
            "fig_no":      fig_no,
            "plot_file":   plot_filename,
            "fig_caption": f"Figure {fig_no}: Spectrogram – {label}",
        })

    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  CS425 – Part 2: TTS Parameter Sweep  (Table 3)")
    print("=" * 70)

    all_summaries: list[dict] = []

    for exp in TTS_EXPERIMENTS:
        print(f"\n── Parameter: {exp['param']} ──────────────────────────────────────")
        summaries = run_tts_experiment(exp, output_dir=OUTPUT_DIR)
        all_summaries.extend(summaries)

    # ── Figure summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Figure Summary – paste into report immediately after Table 3")
    print("=" * 70)
    for s in all_summaries:
        print(f"  {s['fig_caption']}")
        print(f"    File: {OUTPUT_DIR}/{s['plot_file']}")
    print()


if __name__ == "__main__":
    main()
