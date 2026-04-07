"""
main.py - Core utilities and entry-point helpers for CS425 Assignment 1.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from audio_io import load_audio, save_audio


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs(base="outputs"):
    """Create the standard output directory tree if it does not exist.

    Parameters
    ----------
    base : str
        Root output directory (default: ``outputs``).

    Returns
    -------
    dict
        Mapping of logical name → absolute path for each sub-directory.
    """
    dirs = {
        "plots": os.path.join(base, "plots"),
        "audio": os.path.join(base, "audio"),
        "report_data": os.path.join(base, "report_data"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_figure(fig, filepath, dpi=300):
    """Save *fig* to *filepath* at *dpi* dots per inch.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str
        Destination path (PNG recommended).
    dpi : int
        Resolution in dots per inch.
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_waveform(signal, sr, title="Waveform", ax=None):
    """Plot a waveform on *ax* (or create a new figure).

    Parameters
    ----------
    signal : np.ndarray
        Audio signal.
    sr : int
        Sample rate in Hz.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; a new figure/axes is created when None.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()

    t = np.arange(len(signal)) / sr
    ax.plot(t, signal, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_spectrum(freqs, magnitude, title="Magnitude Spectrum", ax=None):
    """Plot a magnitude spectrum on *ax*.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis in Hz.
    magnitude : np.ndarray
        Magnitude values.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(freqs, magnitude, linewidth=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.set_xlim(0, freqs[-1])
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CS425 Audio Processing – utility runner"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="speech.wav",
        help="Path to input audio file (default: speech.wav)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for all outputs (default: outputs)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point: load audio and create basic output plots."""
    args = _parse_args(argv)

    if not os.path.isfile(args.audio_file):
        print(
            f"[main] Audio file not found: {args.audio_file}\n"
            "       Download a speech file and save it as 'speech.wav', then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[main] Loading {args.audio_file} …")
    signal, sr = load_audio(args.audio_file)
    print(f"[main] Loaded {len(signal)} samples at {sr} Hz "
          f"({len(signal)/sr:.2f} s)")

    dirs = ensure_output_dirs(args.output_dir)

    # Basic waveform plot
    fig, _ = plot_waveform(signal, sr, title="Input Waveform")
    out_path = os.path.join(dirs["plots"], "waveform_original.png")
    save_figure(fig, out_path)
    print(f"[main] Saved waveform plot → {out_path}")

    # Save a copy of the original audio
    audio_path = os.path.join(dirs["audio"], "original.wav")
    save_audio(audio_path, signal, sr)
    print(f"[main] Saved audio → {audio_path}")

    print("[main] Done.")


if __name__ == "__main__":
    main()
