# CS425 Audio Processing – Assignment 1

Automated experimental framework for **CS425: Time and Frequency Domain Audio
Analysis**.  Run a single command and get publication-quality plots, processed
WAV files, and CSV data tables for all 7 experiments.

---

## Experiments

| # | Title | Key Metrics |
|---|-------|-------------|
| 1 | Sampling Rate | SNR, Nyquist frequency |
| 2 | Quantization | SNR, Dynamic range |
| 3 | Time / Phase Shift | Phase angle, time-domain delay |
| 4 | STFT Parameter Study | Time & frequency resolution |
| 5 | Clipping (hard & soft) | Distortion dB, clipped fraction |
| 6 | Aliasing (downsampling) | Nyquist violations, SNR |
| 7 | DFT vs FFT Complexity | Execution time, speedup |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/avneetkaur3513/CS425-audio-processing.git
cd CS425-audio-processing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Get an audio file

Download any mono or stereo speech recording (WAV/MP3/FLAC) and save it as
`speech.wav` in the project folder.  A good source is
[freesound.org](https://freesound.org/search/?q=speech).

> **Don't have a recording?**  No problem — if `speech.wav` is missing the
> script automatically generates a synthetic speech-like test signal and saves
> it as `speech.wav` before running.

### 4. Run all experiments

```bash
python experimental_runner.py
```

Pass a custom file or output directory if needed:

```bash
python experimental_runner.py my_audio.wav --output-dir results
```

---

## Output Structure

```
outputs/
├── plots/
│   ├── 01_sampling_rate.png
│   ├── 02_quantization.png
│   ├── 03_time_phase_shift.png
│   ├── 04_stft_comparison.png
│   ├── 05_clipping.png
│   ├── 06_aliasing.png
│   └── 07_dft_vs_fft.png
├── audio/
│   ├── 01_sr_44100hz.wav
│   ├── 02_quantized_16bit.wav
│   └── … (18+ processed WAV files)
└── report_data/
    ├── 01_sampling_rate_results.csv
    ├── 02_quantization_results.csv
    ├── 03_time_phase_shift_results.csv
    ├── 04_stft_results.csv
    ├── 05_clipping_results.csv
    ├── 06_aliasing_results.csv
    ├── 07_dft_vs_fft_results.csv
    └── all_experiments_results.json
```

---

## Using Your Results

### Filling Assignment Tables from CSV Files

Each CSV file maps directly to one experiment table in your report.

| CSV file | Columns you need |
|----------|-----------------|
| `01_sampling_rate_results.csv` | `sample_rate_hz`, `nyquist_hz`, `snr_db` |
| `02_quantization_results.csv` | `bit_depth`, `snr_db`, `dynamic_range_db` |
| `03_time_phase_shift_results.csv` | `shift_ms`, `phase_deg`, `snr_db` |
| `04_stft_results.csv` | `window_size`, `hop_size`, `time_res_ms`, `freq_res_hz` |
| `05_clipping_results.csv` | `clip_type`, `threshold`, `distortion_db`, `clipped_fraction` |
| `06_aliasing_results.csv` | `downsample_factor`, `effective_nyquist_hz`, `snr_db` |
| `07_dft_vs_fft_results.csv` | `signal_length`, `dft_time_s`, `fft_time_s`, `speedup` |

**Steps:**

1. Open the CSV in any spreadsheet program (Excel, Google Sheets, LibreOffice Calc):

   ```
   outputs/report_data/01_sampling_rate_results.csv
   ```

2. Copy the column values into the corresponding rows of your assignment table.

3. Repeat for each of the 7 CSV files.

Alternatively, view the data directly in the terminal:

```bash
python - << 'EOF'
import pandas as pd, glob, os

for csv in sorted(glob.glob("outputs/report_data/*.csv")):
    print(f"\n=== {os.path.basename(csv)} ===")
    print(pd.read_csv(csv).to_string(index=False))
EOF
```

---

### Viewing and Embedding Plots in Your PDF Report

All plots are saved as high-resolution PNG files (300 dpi) in `outputs/plots/`.

**Option A – View in VS Code / Codespaces:**

Click any `.png` file in the Explorer panel to preview it directly.

**Option B – Open from the terminal:**

```bash
# List all plots with their sizes
ls -lh outputs/plots/
```

**Option C – Embed in a LaTeX report:**

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{outputs/plots/01_sampling_rate.png}
  \caption{Sampling rate comparison — waveforms and spectra at 44.1 kHz,
           22.05 kHz, and 8 kHz.}
  \label{fig:sampling-rate}
\end{figure}
```

**Option D – Embed in a Word / Google Docs report:**

1. In VS Code / Codespaces right-click the PNG → **Download**.
2. In Word: **Insert → Pictures → This Device** and select the downloaded file.
3. In Google Docs: **Insert → Image → Upload from computer**.

**Option E – Download all outputs at once (Codespaces):**

```bash
# Zip everything for easy download
zip -r cs425_outputs.zip outputs/
```

Then right-click `cs425_outputs.zip` in the Explorer panel → **Download**.

---

### Listening to Processed Audio Files

The `outputs/audio/` folder contains the processed WAV files for each
experiment condition.  You can listen to them directly in VS Code by
clicking any `.wav` file, or download them and play locally.

---

## Module Overview

| File | Purpose |
|------|---------|
| `experimental_runner.py` | Main orchestrator – runs all 7 experiments |
| `audio_io.py` | Audio loading, resampling, quantization, SNR |
| `fourier_analysis.py` | FFT, naïve DFT, time/phase shift helpers |
| `stft_analysis.py` | STFT computation and resolution metrics |
| `effects.py` | Hard/soft clipping, downsampling, aliasing |
| `main.py` | CLI entry-point, plotting & directory utilities |

---

## Dependencies

See [`requirements.txt`](requirements.txt):

* `numpy` – numerical computing
* `scipy` – signal processing
* `matplotlib` – plotting
* `librosa` – audio analysis
* `soundfile` – WAV I/O
* `pandas` – CSV output
