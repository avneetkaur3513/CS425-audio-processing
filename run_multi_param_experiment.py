"""
run_multi_param_experiment.py
==============================
Multi-parameter experiment for CS425 Assignment 2, Part 1 – Table 2.

Compares three system conditions:
  • Original Audio        – no modifications applied
  • Configuration A (best)  – strong pre-emphasis, no noise, normal speed/pitch
  • Configuration B (worst) – no pre-emphasis, high noise, faster speed, pitch shift

For each condition the script saves:
  • Combined waveform + spectrogram PNG
  • Processed WAV file
  • Transcription TXT

Output directory: outputs/stt/multi_param/

Usage
-----
    python run_multi_param_experiment.py [audio_file]

    audio_file defaults to "Speaking_Female.wav".
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
    process_audio,
    transcribe_audio,
    plot_waveform_spectrogram,
)

# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

CONFIGURATIONS = [
    {
        "name":         "Original",
        "description":  "No modifications",
        "fig_no":       9,
        "pre_emphasis": 0.0,
        "noise_level":  0.0,
        "speed_factor": 1.0,
        "pitch_steps":  0,
    },
    {
        "name":         "Config_A_Best",
        "description":  "Best: pre_emphasis=0.97, noise=0.0, speed=1.0, pitch=0",
        "fig_no":       10,
        "pre_emphasis": 0.97,
        "noise_level":  0.0,
        "speed_factor": 1.0,
        "pitch_steps":  0,
    },
    {
        "name":         "Config_B_Worst",
        "description":  "Worst: pre_emphasis=0.0, noise=0.02, speed=1.3, pitch=+3",
        "fig_no":       11,
        "pre_emphasis": 0.0,
        "noise_level":  0.02,
        "speed_factor": 1.3,
        "pitch_steps":  3,
    },
]

OUTPUT_DIR = "outputs/stt/multi_param"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_file: str) -> tuple[np.ndarray, int]:
    """Load audio file, or generate a synthetic fallback."""
    if os.path.isfile(audio_file):
        signal, sr = librosa.load(audio_file, sr=None, mono=True)
        print(f"[Load] '{audio_file}' ({len(signal)} samples @ {sr} Hz)")
        return signal, sr
    from audio_io import generate_synthetic_audio
    print(
        f"[Load] '{audio_file}' not found – using synthetic speech signal.\n"
        "       Place 'Speaking_Female.wav' here for real-audio results."
    )
    return generate_synthetic_audio(duration=5.0, sr=22050)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(audio_file: str = "Speaking_Female.wav") -> None:
    signal, sr = _load_audio(audio_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  CS425 – Part 1: Multi-Parameter Experiment  (Table 2)")
    print("=" * 70)

    summaries: list[dict] = []

    for cfg in CONFIGURATIONS:
        name = cfg["name"]
        print(f"\n── {name}: {cfg['description']} ──────────────────────────────")

        processed = process_audio(
            signal, sr,
            pre_emphasis=cfg["pre_emphasis"],
            noise_level=cfg["noise_level"],
            speed_factor=cfg["speed_factor"],
            pitch_steps=cfg["pitch_steps"],
        )

        # ── Plot ─────────────────────────────────────────────────────────────
        plot_file = f"multi_param_{name}.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_file)
        plot_waveform_spectrogram(
            processed, sr,
            title=f"Figure {cfg['fig_no']}: {name} – {cfg['description']}",
            save_path=plot_path,
        )

        # ── Audio ─────────────────────────────────────────────────────────────
        wav_path = os.path.join(OUTPUT_DIR, f"multi_param_{name}.wav")
        sf.write(wav_path, processed.astype(np.float32), sr)

        # ── Transcription ─────────────────────────────────────────────────────
        transcription = transcribe_audio(processed, sr)
        txt_path = os.path.join(OUTPUT_DIR, f"multi_param_{name}_transcription.txt")
        with open(txt_path, "w") as fh:
            fh.write(f"Configuration: {name}\n")
            fh.write(f"Settings: {cfg['description']}\n")
            fh.write(f"Transcription: {transcription}\n")

        print(f"  Plot          → {plot_path}")
        print(f"  Audio         → {wav_path}")
        print(f"  Transcription → {txt_path}")
        snippet = transcription[:100] + ("…" if len(transcription) > 100 else "")
        print(f"  Text snippet  : {snippet}")

        summaries.append({
            "name":          name,
            "config":        cfg["description"],
            "transcription": transcription,
            "plot_file":     plot_file,
            "fig_no":        cfg["fig_no"],
        })

    # ── Figure summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Figure Summary – paste into report immediately after Table 2")
    print("=" * 70)
    for s in summaries:
        print(f"  Figure {s['fig_no']}: Spectrogram – {s['name']}")
        print(f"    File   : {OUTPUT_DIR}/{s['plot_file']}")
        print(f"    Config : {s['config']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-parameter experiment for CS425 Assignment 2"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="Speaking_Female.wav",
        help="Input audio file (default: Speaking_Female.wav)",
    )
    args = parser.parse_args()
    main(args.audio_file)
