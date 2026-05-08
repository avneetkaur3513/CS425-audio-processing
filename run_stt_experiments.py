"""
run_stt_experiments.py
======================
Automated A/B experiments for ALL STT parameters – CS425 Assignment 2, Part 1.

For each parameter (pre_emphasis, noise_level, speed_factor, pitch_steps):
  1. Load the audio file (falls back to synthetic speech if absent).
  2. Process with Value A and Value B independently.
  3. Save a combined waveform + spectrogram PNG.
  4. Save the processed WAV.
  5. Save the transcription to a text file.
  6. Print a figure-summary block for pasting into the report.

Output directory: outputs/stt/

Usage
-----
    python run_stt_experiments.py [audio_file]

    audio_file defaults to "Audio files/Speaking_Female.wav".
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import librosa
import soundfile as sf

from stt_template import (
    DEFAULT_AUDIO_FILE,
    process_audio,
    transcribe_audio,
    plot_waveform_spectrogram,
)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each dict defines one A/B parameter experiment.
# fig_no_A / fig_no_B are Figure numbers used in the report (Table 1).

EXPERIMENTS = [
    {
        "param":    "pre_emphasis",
        "value_A":  0.0,
        "value_B":  0.97,
        "label_A":  "Pre-emphasis 0.0",
        "label_B":  "Pre-emphasis 0.97",
        "fig_no_A": 1,
        "fig_no_B": 2,
    },
    {
        "param":    "noise_level",
        "value_A":  0.0,
        "value_B":  0.01,
        "label_A":  "Noise Level 0.0",
        "label_B":  "Noise Level 0.01",
        "fig_no_A": 3,
        "fig_no_B": 4,
    },
    {
        "param":    "speed_factor",
        "value_A":  1.0,
        "value_B":  1.25,
        "label_A":  "Speed Factor 1.0",
        "label_B":  "Speed Factor 1.25",
        "fig_no_A": 5,
        "fig_no_B": 6,
    },
    {
        "param":    "pitch_steps",
        "value_A":  0,
        "value_B":  2,
        "label_A":  "Pitch Shift 0",
        "label_B":  "Pitch Shift +2",
        "fig_no_A": 7,
        "fig_no_B": 8,
    },
]

OUTPUT_DIR = "outputs/stt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_file: str) -> tuple[np.ndarray, int]:
    """Load audio file, or generate a synthetic fallback."""
    print(f"[Load] Using input audio: {audio_file}")
    if os.path.isfile(audio_file):
        signal, sr = librosa.load(audio_file, sr=None, mono=True)
        print(f"[Load] '{audio_file}' ({len(signal)} samples @ {sr} Hz)")
        return signal, sr
    from audio_io import generate_synthetic_audio
    print(
        f"[Load] '{audio_file}' not found – using synthetic speech signal.\n"
        "       Place 'Speaking_Female.wav' at 'Audio files/Speaking_Female.wav' for real-audio results."
    )
    return generate_synthetic_audio(duration=5.0, sr=22050)


def _run_single(
    signal: np.ndarray,
    sr: int,
    param: str,
    value,
    label: str,
    fig_no: int,
    output_dir: str,
) -> dict:
    """Process, plot, save audio and transcription for one experiment value."""
    # Build processing kwargs – only the target param differs from default
    kwargs = dict(pre_emphasis=0.0, noise_level=0.0, speed_factor=1.0, pitch_steps=0)
    kwargs[param] = value

    processed = process_audio(signal, sr, **kwargs)

    # ── Plot ────────────────────────────────────────────────────────────────
    plot_filename = f"{param}_{value}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plot_waveform_spectrogram(
        processed, sr,
        title=f"Figure {fig_no}: Spectrogram – {label}",
        save_path=plot_path,
    )

    # ── Audio ────────────────────────────────────────────────────────────────
    wav_path = os.path.join(output_dir, f"{param}_{value}.wav")
    sf.write(wav_path, processed.astype(np.float32), sr)

    # ── Transcription ────────────────────────────────────────────────────────
    transcription = transcribe_audio(processed, sr)
    txt_path = os.path.join(output_dir, f"{param}_{value}_transcription.txt")
    with open(txt_path, "w") as fh:
        fh.write(f"Experiment: {label}\n")
        fh.write(f"Parameters: {param}={value}\n")
        fh.write(f"Transcription: {transcription}\n")

    print(f"  Figure {fig_no}: {label}")
    print(f"    Plot          → {plot_path}")
    print(f"    Audio         → {wav_path}")
    print(f"    Transcription → {txt_path}")
    snippet = transcription[:100] + ("…" if len(transcription) > 100 else "")
    print(f"    Text snippet  : {snippet}")

    return {
        "label":        label,
        "value":        value,
        "transcription": transcription,
        "plot_file":    plot_filename,
        "fig_no":       fig_no,
        "fig_caption":  f"Figure {fig_no}: Spectrogram – {label}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(audio_file: str = DEFAULT_AUDIO_FILE) -> None:
    signal, sr = _load_audio(audio_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("  CS425 – Part 1: STT Parameter A/B Experiments  (Table 1)")
    print("=" * 70)

    for exp in EXPERIMENTS:
        param = exp["param"]
        print(f"\n── Parameter: {param} ─────────────────────────────────────────────")
        results: dict[str, dict] = {}
        for suffix, val_key, lbl_key, fig_key in [
            ("A", "value_A", "label_A", "fig_no_A"),
            ("B", "value_B", "label_B", "fig_no_B"),
        ]:
            results[suffix] = _run_single(
                signal, sr,
                param=param,
                value=exp[val_key],
                label=exp[lbl_key],
                fig_no=exp[fig_key],
                output_dir=OUTPUT_DIR,
            )
        all_results[param] = results

    # ── Figure summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Figure Summary – paste into report immediately after Table 1")
    print("=" * 70)
    for param, results in all_results.items():
        for suffix in ("A", "B"):
            r = results[suffix]
            print(f"  {r['fig_caption']}")
            print(f"    File: {OUTPUT_DIR}/{r['plot_file']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all STT A/B experiments for CS425 Assignment 2, Part 1"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=DEFAULT_AUDIO_FILE,
        help="Input audio file (default: Audio files/Speaking_Female.wav)",
    )
    args = parser.parse_args()
    main(args.audio_file)
