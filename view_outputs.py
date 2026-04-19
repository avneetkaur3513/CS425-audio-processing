"""
view_outputs.py
===============
Inspect and summarise every file produced by ``experimental_runner.py``.

Run after ``python experimental_runner.py`` to verify the outputs:

    python view_outputs.py                   # default: outputs/
    python view_outputs.py --output-dir results

What this script does
---------------------
* Prints the complete directory tree under the output folder.
* Shows every CSV table (all rows) so you can read the data directly in the
  terminal and copy values into your assignment report.
* Pretty-prints the JSON summary produced by the experimental runner.
* Lists every PNG plot with its file size.  In GitHub Codespaces you can
  open any PNG by right-clicking the file in the Explorer panel and choosing
  "Open Preview".
* Lists every generated WAV file with its duration and sample rate so you
  can confirm the audio was written correctly.  In Codespaces click the file
  in the Explorer panel to play it directly in the browser.

GitHub Codespaces quick-open tips
----------------------------------
Plots (PNG)
  * Explorer panel → outputs/plots/ → right-click any .png → "Open Preview"
  * Or run:  code outputs/plots/01_sampling_rate.png

CSV tables
  * Explorer panel → outputs/report_data/ → click any .csv to open a table
  * Or run:  code outputs/report_data/01_sampling_rate_results.csv

Audio files
  * Explorer panel → outputs/audio/ → click any .wav to play in browser

JSON summary
  * Explorer panel → outputs/report_data/all_experiments_results.json
  * Or run:  code outputs/report_data/all_experiments_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# 1. Directory tree
# ---------------------------------------------------------------------------

def print_tree(output_dir: str) -> None:
    """Print the full file tree under *output_dir*."""
    _header("Output Directory Structure")

    if not os.path.isdir(output_dir):
        print(f"  [NOT FOUND] {output_dir!r} does not exist yet.")
        print("  Run  python experimental_runner.py  first.")
        return

    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        dirs.sort()
        files.sort()
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " + "│   " * level + "├── "
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " + "│   " * (level + 1) + "├── "
        for fname in files:
            fpath = os.path.join(root, fname)
            size = _human_bytes(os.path.getsize(fpath))
            print(f"{sub_indent}{fname}  ({size})")
            total_files += 1

    print(f"\n  Total: {total_files} file(s) in {output_dir}/")


# ---------------------------------------------------------------------------
# 2. CSV tables
# ---------------------------------------------------------------------------

def print_csv_tables(report_dir: str) -> None:
    """Print every CSV file found in *report_dir* as a formatted table."""
    _header("CSV Data Tables")

    if not os.path.isdir(report_dir):
        print(f"  [NOT FOUND] {report_dir!r} – no CSV data yet.")
        return

    csv_files = sorted(f for f in os.listdir(report_dir) if f.endswith(".csv"))
    if not csv_files:
        print("  No CSV files found.")
        return

    for fname in csv_files:
        fpath = os.path.join(report_dir, fname)
        _subheader(fname)
        try:
            with open(fpath, encoding="utf-8") as fh:
                lines = fh.readlines()
            if not lines:
                print("  (empty file)")
                continue
            # Simple column-aligned print
            rows = [line.rstrip("\n").split(",") for line in lines]
            col_widths = [max(len(row[c]) for row in rows if c < len(row))
                          for c in range(max(len(r) for r in rows))]
            for i, row in enumerate(rows):
                formatted = "  | " + " | ".join(
                    (row[c] if c < len(row) else "").ljust(col_widths[c])
                    for c in range(len(col_widths))
                ) + " |"
                print(formatted)
                if i == 0:  # separator after header
                    sep = "  |-" + "-|-".join("-" * w for w in col_widths) + "-|"
                    print(sep)
        except OSError as exc:
            print(f"  [ERROR] Could not read {fpath}: {exc}")


# ---------------------------------------------------------------------------
# 3. JSON summary
# ---------------------------------------------------------------------------

def print_json_summary(report_dir: str) -> None:
    """Pretty-print the JSON summary archive."""
    _header("JSON Experiment Summary")

    json_path = os.path.join(report_dir, "all_experiments_results.json")
    if not os.path.isfile(json_path):
        print(f"  [NOT FOUND] {json_path}")
        return

    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        print(json.dumps(data, indent=2, default=str))
    except OSError as exc:
        print(f"  [ERROR] {exc}")


# ---------------------------------------------------------------------------
# 4. PNG plots
# ---------------------------------------------------------------------------

_PLOT_DESCRIPTIONS = {
    "01_sampling_rate.png":   "Waveforms and spectra at 44100 / 22050 / 8000 Hz",
    "02_quantization.png":    "Quantisation noise at 16 / 8 / 4 bits",
    "03_time_phase_shift.png": "Time-domain delay and phase-shift effects",
    "04_stft_comparison.png": "STFT spectrograms for different window sizes/hops",
    "05_clipping.png":        "Hard and soft clipping distortion",
    "06_aliasing.png":        "Aliasing artefacts from downsampling",
    "07_dft_vs_fft.png":      "DFT vs FFT execution-time comparison",
}


def print_plots(plots_dir: str) -> None:
    """List all PNG files and print Codespaces viewing instructions."""
    _header("PNG Plot Files  (300 dpi)")

    if not os.path.isdir(plots_dir):
        print(f"  [NOT FOUND] {plots_dir!r} – no plots yet.")
        return

    png_files = sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
    if not png_files:
        print("  No PNG files found.")
        return

    for fname in png_files:
        fpath = os.path.join(plots_dir, fname)
        size = _human_bytes(os.path.getsize(fpath))
        desc = _PLOT_DESCRIPTIONS.get(fname, "")
        desc_str = f" – {desc}" if desc else ""
        print(f"  {fname}  ({size}){desc_str}")

    print()
    print("  ── How to view in GitHub Codespaces ──────────────────────────")
    print("  Option A (Explorer panel):")
    print("    1. Click the Explorer icon (📁) in the left sidebar")
    print(f"    2. Navigate to  {plots_dir}/")
    print("    3. Right-click any .png file → 'Open Preview'")
    print()
    print("  Option B (command line – opens in VS Code tab):")
    for fname in png_files[:3]:
        print(f"    code {os.path.join(plots_dir, fname)}")
    if len(png_files) > 3:
        print(f"    … and {len(png_files) - 3} more plots")


# ---------------------------------------------------------------------------
# 5. Audio files
# ---------------------------------------------------------------------------

def print_audio_files(audio_dir: str) -> None:
    """List WAV files and print their duration/sample-rate when soundfile is available."""
    _header("Generated Audio Files (WAV)")

    if not os.path.isdir(audio_dir):
        print(f"  [NOT FOUND] {audio_dir!r} – no audio files yet.")
        return

    wav_files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))
    if not wav_files:
        print("  No WAV files found.")
        return

    try:
        import soundfile as sf
        has_sf = True
    except ImportError:
        has_sf = False

    for fname in wav_files:
        fpath = os.path.join(audio_dir, fname)
        size = _human_bytes(os.path.getsize(fpath))
        detail = ""
        if has_sf:
            try:
                info = sf.info(fpath)
                detail = f"  {info.samplerate} Hz, {info.channels}ch, {info.duration:.2f}s"
            except Exception:
                pass
        print(f"  {fname}  ({size}){detail}")

    print()
    print("  ── How to play in GitHub Codespaces ──────────────────────────")
    print("  1. Click the Explorer icon (📁) in the left sidebar")
    print(f"  2. Navigate to  {audio_dir}/")
    print("  3. Click any .wav file – it opens the built-in audio player")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="View and verify all outputs from experimental_runner.py"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root output directory to inspect (default: outputs)",
    )
    parser.add_argument(
        "--section",
        choices=["tree", "csv", "json", "plots", "audio"],
        default=None,
        help=(
            "Show only one section. "
            "Choices: tree | csv | json | plots | audio "
            "(default: show all)"
        ),
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "plots")
    audio_dir = os.path.join(output_dir, "audio")
    report_dir = os.path.join(output_dir, "report_data")

    section = args.section

    if section is None or section == "tree":
        print_tree(output_dir)
    if section is None or section == "plots":
        print_plots(plots_dir)
    if section is None or section == "audio":
        print_audio_files(audio_dir)
    if section is None or section == "csv":
        print_csv_tables(report_dir)
    if section is None or section == "json":
        print_json_summary(report_dir)

    if section is None:
        _header("Codespaces Quick-Reference")
        print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  FILE TYPE   │  HOW TO VIEW IN CODESPACES                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  PNG plots   │  Explorer → outputs/plots/ → right-click → Open │
  │              │  Preview  (or: code outputs/plots/<name>.png)    │
  │  CSV tables  │  Explorer → outputs/report_data/ → click file   │
  │              │  (renders as a spreadsheet in VS Code)           │
  │  JSON        │  Explorer → outputs/report_data/ →              │
  │              │  all_experiments_results.json → click to open   │
  │  WAV audio   │  Explorer → outputs/audio/ → click file to play │
  └─────────────────────────────────────────────────────────────────┘

  Run specific sections only:
    python view_outputs.py --section tree    # directory tree
    python view_outputs.py --section plots   # plot file list
    python view_outputs.py --section audio   # audio file list
    python view_outputs.py --section csv     # all CSV tables
    python view_outputs.py --section json    # JSON summary
""")


if __name__ == "__main__":
    main()
